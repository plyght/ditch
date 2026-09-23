"""A full-depth transformers reference for a checkpoint larger than the RAM and the disk.

    python3 tools/probe_reference.py REPO probe.json --dtype float32 --factory tools/ref_stream.py

REPO is a Hub id (`openai/gpt-oss-20b`) or a local directory. The model is
transformers' own, built by its own `from_pretrained` from the real config,
and its own forward runs unchanged end to end; only where the weights come
from differs. `from_pretrained` builds every parameter on the meta device as
usual, and the loader is stopped before the decoder layers: it loads only the
trunk (embeddings, final norm, LM head, anything outside the layer stack).
Each decoder layer then has a forward pre-hook that range-reads that layer's
tensors from the safetensors shards and passes them, under their checkpoint
names, through transformers' own `convert_and_load_state_dict_in_model` with
the load configuration `from_pretrained` built (the family's renames, its
`WeightConverter`s fusing experts, the quantizer's dequantisation), and a
forward hook sends them back to the meta device. Cross-layer state (the KV
cache, hyper-connection streams, attention over earlier layers' outputs, KV
sharing) lives in the model's own forward, which is never touched.

A mixture-of-experts layer whose routed experts would not fit dequantised
(REF_STREAM_LAZY_EXPERTS=auto: more than 2 GB of float32 a layer) keeps them
lazy: the experts module's stacked `gate_up_proj` / `down_proj` become objects
whose `[e]` loads expert `e` when the eager experts forward reaches it, through
the same loader: expert `e`'s checkpoint tensors (its own per-expert tensors
renumbered to expert 0, or slab `e` of a stacked tensor) are loaded into a
one-expert copy of the stacks, so the family's own converters and dequantiser
produce it. Only the experts a token is routed to are read.

Arithmetic is float32 on the stored values: every layer weight becomes float32
when it is loaded (exact for bf16 and for the dequantised formats), and the
trunk stays float32 unless its tables would exceed REF_STREAM_TRUNK_F32_GB (6),
in which case tables above 256M elements stay in their stored bf16 and their
Linear / Embedding forwards compute in float32.

Fetched byte ranges are kept in a local cache (REF_STREAM_CACHE, default
~/.cache/ref_stream) until it holds REF_STREAM_CACHE_GB (12) and are never
evicted: every generated token rereads the model in the same order, and for a
cyclic scan keeping the first bytes is as good as any eviction order. The next
layer is fetched in the background while a layer runs. Totals (bytes fetched,
bytes read from the cache, peak RSS, wall time) are printed at exit.
"""
import atexit
import concurrent.futures as cf
import json
import os
import re
import resource
import struct
import sys
import tempfile
import threading
import time

import requests
import torch

# transformers binds flash-linear-attention's Triton kernels at import time
# whenever fla is importable, and they cannot run on a CPU: hide it.
if "fla" not in sys.modules:
    sys.modules["fla"] = None

import transformers  # noqa: E402
from transformers import AutoConfig, AutoModelForCausalLM  # noqa: E402
from transformers import core_model_loading as cml  # noqa: E402
from transformers.core_model_loading import WeightConverter, WeightRenaming  # noqa: E402
from transformers.modeling_utils import PreTrainedModel  # noqa: E402

DTYPES = {
    "F32": torch.float32, "BF16": torch.bfloat16, "F16": torch.float16, "F64": torch.float64,
    "I64": torch.int64, "I32": torch.int32, "I16": torch.int16, "I8": torch.int8, "U8": torch.uint8,
    "BOOL": torch.bool, "F8_E4M3": torch.float8_e4m3fn, "F8_E5M2": torch.float8_e5m2,
    "F8_E8M0": torch.float8_e8m0fnu,
}
SMALL_FILES = (".json", ".jinja", ".model", ".txt", ".tiktoken", ".py")
CHUNK = 32 << 20
# Stored dtypes that only a dequantiser turns into weights.
QUANT_DTYPES = ("F8_E4M3", "F8_E5M2", "U8", "I8", "I32")

STATS = {"fetched": 0, "cached": 0, "requests": 0, "t0": time.time(), "layer_loads": 0, "expert_loads": 0}


def _report():
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1 << 20)
    print(f"[ref_stream] fetched {STATS['fetched'] / 1e9:.2f} GB in {STATS['requests']} requests,"
          f" {STATS['cached'] / 1e9:.2f} GB from the cache; {STATS['layer_loads']} layer loads,"
          f" {STATS['expert_loads']} expert loads; peak RSS {rss:.2f} GB; wall {time.time() - STATS['t0']:.0f} s",
          file=sys.stderr, flush=True)


atexit.register(_report)


class Source:
    """The safetensors tensors of a Hub repository or a local directory, read by byte range."""

    def __init__(self, model, revision=None):
        self.local = os.path.isdir(model)
        self.model, self.revision = model, revision
        self.tls = threading.local()
        self.pool = cf.ThreadPoolExecutor(16)
        self.cache_dir = None
        self.cache_budget = float(os.environ.get("REF_STREAM_CACHE_GB", "12")) * 1e9
        self.cache_used = 0
        self.cache_lock = threading.Lock()
        if self.local:
            files = sorted(os.listdir(model))
        else:
            from huggingface_hub import list_repo_files
            files = list_repo_files(model, revision=revision)
            root = os.path.expanduser(os.environ.get("REF_STREAM_CACHE", "~/.cache/ref_stream"))
            self.cache_dir = os.path.join(root, model.replace("/", "__") + ("@" + revision if revision else ""))
            os.makedirs(self.cache_dir, exist_ok=True)
            for dp, _, fns in os.walk(self.cache_dir):
                self.cache_used += sum(os.path.getsize(os.path.join(dp, f)) for f in fns)
        self.files = files
        shards = [f for f in files if f.endswith(".safetensors") and "/" not in f]
        if "model.safetensors.index.json" in files:
            wm = json.loads(self._small("model.safetensors.index.json"))["weight_map"]
            shards = sorted(set(wm.values()))
        self.entries = {}  # name -> (shard, absolute start, bytes, dtype, shape)
        for sh in shards:
            n = struct.unpack("<Q", self._read(sh, 0, 8, cache=False))[0]
            hdr = json.loads(self._read(sh, 8, n, cache=False))
            hdr.pop("__metadata__", None)
            for k, v in hdr.items():
                a, b = v["data_offsets"]
                self.entries[k] = (sh, 8 + n + a, b - a, v["dtype"], v["shape"])

    def _url(self, fn):
        from huggingface_hub import hf_hub_url
        return hf_hub_url(self.model, fn, revision=self.revision)

    def _small(self, fn):
        if self.local:
            return open(os.path.join(self.model, fn), "rb").read()
        from huggingface_hub import hf_hub_download
        return open(hf_hub_download(self.model, fn, revision=self.revision), "rb").read()

    def download_small_files(self, out):
        """config.json, the tokenizer and generation files, any modeling code: what `from_pretrained` reads besides weights."""
        for f in self.files:
            if f.endswith(SMALL_FILES) and "/" not in f and not f.startswith(".") and f != "model.safetensors.index.json":
                with open(os.path.join(out, f), "wb") as fh:
                    fh.write(self._small(f))

    def _cache_path(self, sh, a, m):
        return os.path.join(self.cache_dir, sh, f"{a}-{m}.bin")

    def _read(self, sh, a, m, cache=True):
        """Bytes [a, a + m) of shard `sh`."""
        if self.local:
            fd = getattr(self.tls, "fds", {}).get(sh)
            if fd is None:
                self.tls.fds = getattr(self.tls, "fds", {})
                fd = self.tls.fds[sh] = os.open(os.path.join(self.model, sh), os.O_RDONLY)
            out = bytearray(m)
            view, done = memoryview(out), 0
            while done < m:
                done += os.preadv(fd, [view[done:]], a + done)
            return bytes(out)
        path = self._cache_path(sh, a, m) if cache else None
        if path and os.path.exists(path):
            with open(path, "rb") as f:
                data = f.read()
            if len(data) == m:
                STATS["cached"] += m
                return data
        sess = getattr(self.tls, "s", None) or requests.Session()
        self.tls.s = sess
        url = self._url(sh)
        for attempt in range(8):
            try:
                data = sess.get(url, headers={"Range": f"bytes={a}-{a + m - 1}"}, timeout=300).content
                if len(data) == m:
                    break
            except requests.RequestException:
                pass
            time.sleep(2 ** attempt)
        else:
            raise RuntimeError(f"could not fetch {m} bytes at {a} of {sh}")
        STATS["fetched"] += m
        STATS["requests"] += 1
        if path:
            with self.cache_lock:
                room = self.cache_used + m <= self.cache_budget
                if room:
                    self.cache_used += m
            if room:
                os.makedirs(os.path.dirname(path), exist_ok=True)
                tmp = path + f".{threading.get_ident()}.tmp"
                with open(tmp, "wb") as f:
                    f.write(data)
                os.replace(tmp, path)
        return data

    def read_many(self, pieces):
        """[(name, start, bytes)] -> the raw bytes of each piece, fetched in parallel in chunks of at most 32 MB."""
        jobs = []
        for i, (name, start, m) in enumerate(pieces):
            sh, base = self.entries[name][:2]
            for o in range(0, m, CHUNK):
                jobs.append((i, o, sh, base + start + o, min(CHUNK, m - o)))
        out = [bytearray(p[2]) for p in pieces]
        for (i, o, *_), data in zip(jobs, self.pool.map(lambda j: self._read(*j[2:]), jobs)):
            out[i][o:o + len(data)] = data
        return out

    def tensors(self, names, slab=None):
        """{name: tensor} for `names`; with `slab = (e, E)`, row block `e` of `E` of each (as a `[1, ...]` tensor)."""
        pieces = []
        for n in names:
            _, _, nb, dt, shape = self.entries[n]
            if slab is None:
                pieces.append((n, 0, nb))
            else:
                e, E = slab
                pieces.append((n, e * (nb // E), nb // E))
        out = {}
        for n, raw in zip(names, self.read_many(pieces)):
            _, _, nb, dt, shape = self.entries[n]
            shape = list(shape) if slab is None else [1] + list(shape[1:])
            t = torch.frombuffer(raw, dtype=torch.uint8) if len(raw) else torch.empty(0, dtype=torch.uint8)
            out[n] = t.view(DTYPES[dt]).reshape(shape)
        return out


def _meta_like(p, dtype=None, shape=None):
    t = torch.empty(shape if shape is not None else p.shape, dtype=dtype or p.dtype, device="meta")
    return torch.nn.Parameter(t, requires_grad=False)


def _set_param(model, full, value):
    mod_name, _, pname = full.rpartition(".")
    mod = model.get_submodule(mod_name) if mod_name else model
    if pname in mod._parameters:
        mod._parameters[pname] = value
    else:
        setattr(mod, pname, value)


class LazyStack:
    """Stands in for a stacked expert parameter: `stack[e]` is expert e's matrix."""

    def __init__(self, experts, which):
        self.experts, self.which = experts, which

    def __getitem__(self, e):
        return self.experts.get(int(e))[self.which]

    def __len__(self):
        return self.experts.n


class LazyExperts:
    """The routed experts of one experts module, loaded one expert at a time through transformers' own loader."""

    def __init__(self, streamer, prefix, module, params, keys):
        self.s, self.prefix, self.module = streamer, prefix, module
        self.params = params  # param name -> (shape, dtype) of the full stack
        self.n = next(iter(params.values()))[0][0]
        self.keys = keys  # checkpoint keys whose target is one of `params`
        self.cache = {}  # e -> {param: tensor}, most recent last
        self.expert_re = re.compile(r"\.experts\.(\d+)\.")
        self.pending, self.inflight = [], {}  # experts routed to but not yet fetched; e -> future
        stored = sum(streamer.src.entries[k][2] for k in keys)
        self.expert_stored = max(1, stored // self.n)
        # The experts forward's routing argument names the experts it will
        # read: fetch them all now, in parallel, a bounded number ahead.
        module.register_forward_pre_hook(self._routed, with_kwargs=True)

    def _routed(self, mod, args, kwargs):
        for v in list(args[1:]) + list(kwargs.values()):
            if torch.is_tensor(v) and not v.is_floating_point() and v.numel():
                routed = sorted(e for e in torch.unique(v).tolist() if 0 <= e < self.n)
                self.pending = [e for e in routed if e not in self.cache and e not in self.inflight]
                self._top_up()
                return None
        return None

    def _top_up(self):
        window = max(2, int(self.s.prefetch_bytes // self.expert_stored))
        while self.pending and len(self.inflight) < window:
            e = self.pending.pop(0)
            self.inflight[e] = self.s.expert_pool.submit(self._fetch, e)

    def _fetch(self, e):
        per_expert = [k for k in self.keys if (m := self.expert_re.search(k)) and int(m.group(1)) == e]
        stacked = [k for k in self.keys if not self.expert_re.search(k)]
        sd = {}
        if per_expert:
            for k, t in self.s.src.tensors(per_expert).items():
                sd[self.expert_re.sub(".experts.0.", k, count=1)] = t
        if stacked:
            sd.update(self.s.src.tensors(stacked, slab=(e, self.n)))
        return sd

    def get(self, e):
        hit = self.cache.pop(e, None)
        if hit is None:
            hit = self._load(e)
            self.s.expert_bytes += sum(t.numel() * 4 for t in hit.values())
            while self.s.expert_bytes > self.s.expert_cache_bytes and self.s.expert_lru:
                old = self.s.expert_lru.pop(0)
                gone = old[0].cache.pop(old[1], None)
                if gone:
                    self.s.expert_bytes -= sum(t.numel() * 4 for t in gone.values())
        self.cache[e] = hit
        try:
            self.s.expert_lru.remove((self, e))
        except ValueError:
            pass
        self.s.expert_lru.append((self, e))
        return hit

    def _load(self, e):
        STATS["expert_loads"] += 1
        fut = self.inflight.pop(e, None)
        sd = fut.result() if fut is not None else self._fetch(e)
        self._top_up()
        for p, (shape, dtype) in self.params.items():
            self.module.__dict__.pop(p, None)
            self.module._parameters[p] = _meta_like(None, self.s.meta_dtype(f"{self.prefix}.{p}"), [1] + list(shape[1:]))
        self.s.load_state_dict(sd, what=f"{self.prefix} expert {e}")
        out = {}
        for p in self.params:
            v = self.module._parameters.pop(p)
            if v.device.type == "meta":
                raise SystemExit(f"ref_stream: expert {e} of {self.prefix}: {p} was not loaded from {sorted(sd)}")
            out[p] = v.data[0].float()
        for p in self.params:
            setattr(self.module, p, LazyStack(self, p))
        return out


class LazyRows(torch.nn.Module):
    """An `nn.Embedding` whose table is the row concatenation of checkpoint
    tensors `keys` (in the order transformers' `Concatenate(dim=0)` joins
    them), reading only the rows looked up."""

    def __init__(self, src, keys, like):
        super().__init__()
        self.src, self.keys = src, keys
        self.starts = [0]
        for k in keys:
            self.starts.append(self.starts[-1] + src.entries[k][4][0])
        self.dim = like.weight.shape[1]
        if not keys or any(src.entries[k][4][1] != self.dim for k in keys) or self.starts[-1] > like.weight.shape[0]:
            raise SystemExit(f"ref_stream: {keys[:2]}... do not concatenate into a {tuple(like.weight.shape)} table")
        self.padding_idx = like.padding_idx
        self.weight = torch.empty(0)  # its device is all a caller asks about

    def forward(self, ids):
        flat = ids.reshape(-1).tolist()
        pieces, where = [], []
        for r in flat:
            if r >= self.starts[-1]:
                raise SystemExit(f"ref_stream: row {r} is past the {self.starts[-1]} stored rows")
            k = next(j for j in range(len(self.keys)) if r < self.starts[j + 1])
            nb = self.src.entries[self.keys[k]][2] // self.src.entries[self.keys[k]][4][0]
            pieces.append((self.keys[k], (r - self.starts[k]) * nb, nb))
            where.append(k)
        out = torch.empty(len(flat), self.dim)
        for i, (raw, k) in enumerate(zip(self.src.read_many(pieces), where)):
            out[i] = torch.frombuffer(raw, dtype=torch.uint8).view(DTYPES[self.src.entries[self.keys[k]][3]]).float()
        return out.view(*ids.shape, self.dim)


def mimo_attn_shards(shards, block, qdtype):
    """MiMo V2's fp8 attention projections are quantised per tensor-parallel
    shard (`shards` = the full layers' KV heads): each row shard is blocked
    from its own first row, with its own partial last block (see Bug F5).
    transformers' `Fp8Dequantize` takes the block from the scale grid and
    applies it to the whole tensor, which misplaces every scale after the
    first shard; SGLang's loader splits by shard. So a projection whose grid
    is the per-shard one is dequantised here, shard by shard (the same
    arithmetic as `Fp8Dequantize`: fp8 x scale in float32, then the release's
    bf16), and handed to the loader without a scale, which then loads it as
    it is. Every other tensor goes through transformers untouched."""
    br, bc = block
    pat = re.compile(r"self_attn\.(qkv|q|k|v)_proj\.weight$")

    def prepare(sd):
        for k in [k for k in sd if pat.search(k)]:
            sk = k + "_scale_inv"
            if sk not in sd:
                continue
            w, sc = sd[k], sd[sk]
            rows = w.shape[0]
            per_shard = rows % shards == 0 and sc.shape[0] == shards * -(-(rows // shards) // br)
            if not per_shard or sc.shape[0] == -(-rows // br):
                continue
            parts = []
            for pw, ps in zip(w.chunk(shards, 0), sc.chunk(shards, 0)):
                grid = ps.float().repeat_interleave(br, 0)[: pw.shape[0]].repeat_interleave(bc, 1)[:, : pw.shape[1]]
                parts.append(pw.float() * grid)
            sd[k] = torch.cat(parts, 0).to(qdtype)
            del sd[sk]
        return sd
    return prepare


class Streamer:
    def __init__(self, model, load_config, src, key_target, units, qdtype, quantised):
        self.model, self.load_config, self.src = model, load_config, src
        self.qdtype, self.quantised = qdtype, quantised
        self.prepare = None
        self.units = units  # [(prefix, module)]
        self.expert_cache_bytes = float(os.environ.get("REF_STREAM_EXPERT_CACHE_GB", "3")) * 1e9
        self.expert_bytes, self.expert_lru = 0, []
        self.prefetch_bytes = float(os.environ.get("REF_STREAM_PREFETCH_GB", "2")) * 1e9
        self.expert_pool = cf.ThreadPoolExecutor(8)
        by_unit = {i: [] for i in range(len(units))}
        for k, tgt in key_target.items():
            for i, (prefix, _) in enumerate(units):
                if tgt.startswith(prefix + "."):
                    by_unit[i].append(k)
                    break
        self.lazy = {}  # unit -> [LazyExperts]
        mode = os.environ.get("REF_STREAM_LAZY_EXPERTS", "auto")
        lazy_targets = set()
        for i, (prefix, unit) in enumerate(units):
            for name, mod in unit.named_modules():
                ps = mod._parameters
                if not all(isinstance(ps.get(p), torch.Tensor) and ps[p].dim() == 3 for p in ("gate_up_proj", "down_proj")):
                    continue
                f32 = sum(ps[p].numel() * 4 for p in ("gate_up_proj", "down_proj"))
                if mode == "0" or (mode == "auto" and f32 <= 2e9):
                    continue
                full = f"{prefix}.{name}" if name else prefix
                params = {p: (tuple(ps[p].shape), ps[p].dtype) for p in ("gate_up_proj", "down_proj")}
                targets = {f"{full}.{p}" for p in params}
                keys = [k for k in by_unit[i] if key_target[k] in targets]
                lazy_targets |= targets
                le = LazyExperts(self, full, mod, params, keys)
                for p in params:
                    del mod._parameters[p]
                    setattr(mod, p, LazyStack(le, p))
                self.lazy.setdefault(i, []).append(le)
        # Tables too large to load (Qwen4-Exp's n-gram embedding, ~100 GB in
        # shards that transformers concatenates along rows): only the rows looked up are read.
        rows_max = float(os.environ.get("REF_STREAM_ROWS_GB", "2")) * 1e9
        for i, (prefix, unit) in enumerate(units):
            for name, mod in list(unit.named_modules()):
                if not isinstance(mod, torch.nn.Embedding) or mod.weight.numel() * 4 <= rows_max:
                    continue
                full = f"{prefix}.{name}.weight"
                keys = sorted((k for k in by_unit[i] if key_target[k] == full), key=cml.dot_natural_key)
                table = LazyRows(src, keys, mod)
                parent, _, attr = name.rpartition(".")
                setattr(unit.get_submodule(parent) if parent else unit, attr, table)
                lazy_targets.add(full)
        self.unit_keys = {i: [k for k in ks if key_target[k] not in lazy_targets] for i, ks in by_unit.items()}
        self.prefetched = {}
        self.fetch_pool = cf.ThreadPoolExecutor(1)
        for i, (prefix, unit) in enumerate(units):
            unit.register_forward_pre_hook(lambda mod, args, i=i: self.materialise(i))
            unit.register_forward_hook(lambda mod, args, out, i=i: self.release(i))
        n_lazy = sum(len(v) for v in self.lazy.values())
        print(f"[ref_stream] {len(units)} streamed layers, {n_lazy} with lazy experts;"
              f" {sum(len(v) for v in self.unit_keys.values())} layer tensors", file=sys.stderr, flush=True)

    def load_state_dict(self, sd, what):
        info, _ = cml.convert_and_load_state_dict_in_model(self.model, sd, self.load_config)
        if info.conversion_errors or info.mismatched_keys or info.error_msgs:
            raise SystemExit(f"ref_stream: loading {what}: {info.conversion_errors or info.mismatched_keys or info.error_msgs}")
        if info.unexpected_keys:
            raise SystemExit(f"ref_stream: loading {what}: unexpected {sorted(info.unexpected_keys)[:8]}")

    def _fetch(self, i):
        sd = self.src.tensors(self.unit_keys[i])
        return self.prepare(sd) if self.prepare else sd

    def materialise(self, i):
        prefix, unit = self.units[i]
        fut = self.prefetched.pop(i, None)
        sd = fut.result() if fut is not None else self._fetch(i)
        nxt = (i + 1) % len(self.units)  # after the last layer, the next token's first
        if nxt not in self.prefetched:
            self.prefetched[nxt] = self.fetch_pool.submit(self._fetch, nxt)
        self.load_state_dict(sd, what=prefix)
        del sd
        STATS["layer_loads"] += 1
        with torch.no_grad():
            for name, p in list(unit.named_parameters()):
                if p.device.type == "meta":
                    raise SystemExit(f"ref_stream: {prefix}.{name} is in no checkpoint tensor")
                if p.is_floating_point() and p.dtype != torch.float32:
                    _set_param(unit, name, torch.nn.Parameter(p.data.float(), requires_grad=False))

    def meta_dtype(self, full):
        return self.qdtype if full in self.quantised else torch.float32

    def release(self, i):
        prefix, unit = self.units[i]
        for name, p in list(unit.named_parameters()):
            _set_param(unit, name, _meta_like(p, self.meta_dtype(f"{prefix}.{name}")))


def _layer_stack(model, n_layers):
    """(prefix, ModuleList) of the text decoder's layers."""
    best = None
    for name, mod in model.named_modules():
        if not isinstance(mod, torch.nn.ModuleList) or len(mod) != n_layers:
            continue
        if not name.split(".")[-1] in ("layers", "h", "blocks", "layer"):
            continue
        if any(w in name for w in ("mtp", "vision", "visual", "audio", "encoder")):
            continue
        if best is None or len(name) < len(best[0]):
            best = (name, mod)
    if best is None:
        raise SystemExit(f"ref_stream: no decoder layer stack of {n_layers} layers found")
    return best


def _f32_forward(mod):
    """A bf16 table's module computes in float32, a block of rows at a time."""
    import torch.nn.functional as F
    import types
    if isinstance(mod, torch.nn.Embedding):
        def fwd(self, ids):
            out = F.embedding(ids, self.weight, self.padding_idx).float()
            scale = getattr(self, "scalar_embed_scale", None)
            if scale is None and torch.is_tensor(getattr(self, "embed_scale", None)):
                scale = self.embed_scale.float()
            return out if scale is None else out * scale
    else:
        def fwd(self, x):
            x = x.float()
            w = self.weight
            step = max(1, (64 << 20) // (w.shape[1] * 4))
            out = torch.cat([F.linear(x, w[i:i + step].float()) for i in range(0, w.shape[0], step)], dim=-1)
            return out if self.bias is None else out + self.bias.float()
    mod.forward = types.MethodType(fwd, mod)


def load(model, dtype=torch.float32):
    revision = os.environ.get("REF_STREAM_REVISION")
    src = Source(model, revision)
    view = tempfile.mkdtemp(prefix="ref_stream_")
    if src.local:
        for fn in os.listdir(model):
            if fn.endswith(SMALL_FILES) and fn != "model.safetensors.index.json":
                os.symlink(os.path.abspath(os.path.join(model, fn)), os.path.join(view, fn))
    else:
        src.download_small_files(view)
    # from_pretrained wants a weights file to exist; the weights come from `src`.
    with open(os.path.join(view, "model.safetensors"), "wb") as f:
        hdr = b"{}      "
        f.write(struct.pack("<Q", len(hdr)) + hdr)

    config = AutoConfig.from_pretrained(view)
    text = getattr(config, "text_config", None) or config
    n_layers = text.num_hidden_layers
    trunk_f32_max = float(os.environ.get("REF_STREAM_TRUNK_F32_GB", "6")) * 1e9
    state = {}

    orig_load = PreTrainedModel._load_pretrained_model

    def load_trunk(model, state_dict, checkpoint_files, load_config, expected_keys=None):
        load_config = type(load_config)(**{**load_config.__dict__, "dtype": torch.float32})
        state["load_config"] = load_config
        meta_sd = model.state_dict()
        weight_mapping = load_config.weight_mapping or []
        renamings = [w for w in weight_mapping if isinstance(w, WeightRenaming)]
        converters = [w for w in weight_mapping if isinstance(w, WeightConverter)]
        key_target = {}
        for k in src.entries:
            tgt, _ = cml.rename_source_key(k, renamings, converters, model.base_model_prefix, meta_sd)
            if tgt not in meta_sd and k in meta_sd:
                tgt = k
            if tgt in meta_sd:
                key_target[k] = tgt
        prefix, stack = _layer_stack(model, n_layers)
        units = [(f"{prefix}.{i}", m) for i, m in enumerate(stack)]
        state["units"], state["key_target"] = units, key_target
        # A quantised weight is dequantised into its parameter's dtype
        # (Fp8Dequantize) or bf16 (Mxfp4Dequantize): the release's bf16, as a
        # load in the release's dtype produces it. Everything else loads as
        # float32 (exact for bf16, and F32-stored tensors stay unrounded).
        qdtype = getattr(torch, os.environ.get("REF_STREAM_DEQUANT_DTYPE", "bfloat16"))
        quantised = {t for k, t in key_target.items() if src.entries[k][3] in QUANT_DTYPES}
        state["qdtype"], state["quantised"] = qdtype, quantised
        # Every layer parameter stays on the meta device.
        for p_prefix, unit in units:
            for name, p in list(unit.named_parameters()):
                full = f"{p_prefix}.{name}"
                _set_param(unit, name, _meta_like(p, qdtype if full in quantised else torch.float32))
        trunk_keys = [k for k, t in key_target.items() if not any(t.startswith(u + ".") for u, _ in units)]
        # Large tables stay in their stored bf16 when a float32 trunk would not fit.
        big = [(k, key_target[k]) for k in trunk_keys if src.entries[k][3] == "BF16" and len(src.entries[k][4]) == 2
               and src.entries[k][4][0] * src.entries[k][4][1] > (256 << 20)]
        f32_total = sum(src.entries[k][2] * (2 if src.entries[k][3] == "BF16" else 1) for k in trunk_keys)
        state["bf16_tables"] = []
        if f32_total > trunk_f32_max:
            for k, t in big:
                _set_param(model, t, _meta_like(model.get_parameter(t), torch.bfloat16))
                state["bf16_tables"].append(t.rpartition(".")[0])
        print(f"[ref_stream] trunk: {len(trunk_keys)} tensors, {f32_total / 1e9:.2f} GB as float32;"
              f" bf16 tables {state['bf16_tables'] or 'none'}", file=sys.stderr, flush=True)
        for t in {key_target[k] for k in trunk_keys} & quantised:
            _set_param(model, t, _meta_like(model.get_parameter(t), qdtype))
        info, _ = cml.convert_and_load_state_dict_in_model(model, src.tensors(trunk_keys), load_config)
        with torch.no_grad():
            for t in {key_target[k] for k in trunk_keys} & quantised:
                p = model.get_parameter(t)
                if p.device.type != "meta":
                    _set_param(model, t, torch.nn.Parameter(p.data.float(), requires_grad=False))
        layer_params = {f"{u}.{n}" for u, m in units for n, _ in m.named_parameters()}
        info.missing_keys -= layer_params
        return info, None

    softcap = getattr(text, "attn_logit_softcapping", None)
    kw = dict(dtype=torch.float32, experts_implementation="eager")
    if softcap:
        kw["attn_implementation"] = "eager"
    PreTrainedModel._load_pretrained_model = staticmethod(load_trunk)
    try:
        try:
            m = AutoModelForCausalLM.from_pretrained(view, **kw)
        except ValueError:
            cls = getattr(transformers, config.architectures[0])
            m = cls.from_pretrained(view, **kw)
    finally:
        PreTrainedModel._load_pretrained_model = orig_load
    m.eval()
    transformers.utils.logging.disable_progress_bar()
    for name in state["bf16_tables"]:
        _f32_forward(m.get_submodule(name))
    leftover = [n for n, p in m.named_parameters() if p.device.type == "meta"
                and not any(n.startswith(u + ".") for u, _ in state["units"])]
    if leftover:
        raise SystemExit(f"ref_stream: trunk parameters with no checkpoint tensor: {leftover[:8]}")
    m._ref_stream = Streamer(m, state["load_config"], src, state["key_target"], state["units"],
                             state["qdtype"], state["quantised"])
    qcfg = getattr(config, "quantization_config", None) or getattr(text, "quantization_config", None) or {}
    qcfg = qcfg if isinstance(qcfg, dict) else qcfg.to_dict()
    if text.model_type in ("mimo_v2", "mimo_v2_flash") and qcfg.get("weight_block_size"):
        m._ref_stream.prepare = mimo_attn_shards(text.num_key_value_heads, tuple(qcfg["weight_block_size"]), state["qdtype"])
    return m


# ---------------------------------------------------------------------------
# For references built from a release's own code, whose parameter names are
# the checkpoint's (tools/ref_deepseek_v41.py, tools/ref_kimi_k3.py, ...):
# the same remote reads, a LazyCheckpoint-style store, and per-layer
# materialisation by name.
# ---------------------------------------------------------------------------

class Store:
    """tools/lazy_checkpoint.py's interface (`keys`, `header`, `tensor`, `rows`, `flush`) over a `Source`."""

    def __init__(self, src):
        self.src = src
        self.header = {k: {"dtype": e[3], "shape": e[4]} for k, e in src.entries.items()}
        self.lazy = {"holes": {}, "filled": {}}
        self.pool = cf.ThreadPoolExecutor(8)
        self.futures = {}  # prefix -> future of {name: tensor}

    def keys(self):
        return self.src.entries.keys()

    def flush(self):
        pass

    def tensor(self, name):
        return self.src.tensors([name])[name]

    def tensors(self, names):
        return self.src.tensors(list(names))

    def rows(self, name, idx):
        _, _, nb, dt, shape = self.src.entries[name]
        row = nb // shape[0]
        idx = [int(r) for r in torch.as_tensor(idx).reshape(-1).tolist()]
        raw = self.src.read_many([(name, r * row, row) for r in idx])
        out = torch.empty(len(idx), row, dtype=torch.uint8)
        for i, b in enumerate(raw):
            out[i] = torch.frombuffer(b, dtype=torch.uint8)
        return out.view(DTYPES[dt]).reshape(len(idx), *shape[1:])

    def prefetch(self, prefix, names):
        """Starts reading `names` (those that exist) in the background, keyed by `prefix`."""
        if prefix not in self.futures:
            names = [n for n in names if n in self.src.entries]
            self.futures[prefix] = self.pool.submit(self.tensors, names)

    def take(self, prefix):
        """What `prefetch(prefix, ...)` read, or None if it was never asked for."""
        fut = self.futures.pop(prefix, None)
        return fut.result() if fut is not None else None


class LayerStreamer:
    """Materialises `layers[i]`'s parameters by name before it runs and returns them to the meta device after.

    `fetch(i)` returns {parameter name within the layer: float tensor}. The
    next layer is fetched in the background while one runs.
    """

    def __init__(self, layers, fetch):
        self.layers, self.fetch = layers, fetch
        self.pool = cf.ThreadPoolExecutor(1)
        self.prefetched = {}
        for layer in layers:
            for name, p in list(layer.named_parameters()):
                _set_param(layer, name, _meta_like(p, torch.float32))

    def materialise(self, i):
        fut = self.prefetched.pop(i, None)
        values = fut.result() if fut is not None else self.fetch(i)
        nxt = (i + 1) % len(self.layers)
        if nxt not in self.prefetched:
            self.prefetched[nxt] = self.pool.submit(self.fetch, nxt)
        layer = self.layers[i]
        for name, p in list(layer.named_parameters()):
            v = values.pop(name, None)
            if v is None:
                raise SystemExit(f"ref_stream: layer {i}: {name} was not fetched")
            if tuple(v.shape) != tuple(p.shape):
                raise SystemExit(f"ref_stream: layer {i}: {name} is {tuple(v.shape)} in the checkpoint, {tuple(p.shape)} in the model")
            _set_param(layer, name, torch.nn.Parameter(v, requires_grad=False))
        STATS["layer_loads"] += 1

    def release(self, i):
        layer = self.layers[i]
        for name, p in list(layer.named_parameters()):
            _set_param(layer, name, _meta_like(p, torch.float32))

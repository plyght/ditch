"""truncate_checkpoint.py REPO N|--layers L0,L1,... [OUT_DIR] [--drop PREFIX]... [--lazy REGEX]...

Build a checkpoint of a few decoder layers of a released model by
range-fetching only the needed tensors from the remote safetensors shards:
each shard's header first, then the embedding, the final norm, the LM head
and the chosen layers, streamed straight into one model.safetensors (in
parallel, neighbouring tensors merged into one request; nothing is held in
memory and no full shard is downloaded).

`N` keeps layers 0..N-1. `--layers 0,1,2,3,20,21` keeps those layers,
renumbered 0..5 in that order: for families whose layer kinds are far apart
(DeepSeek V4.1's ratio-1 layers start at 20), so that every kind appears in a
cut small enough for the disk. Each kept layer still runs on its real weights
and on the real output of the layer before it in the cut, which is all a
layer-by-layer comparison needs.

config.json gets num_hidden_layers = the kept count and every per-layer list
cut to the kept layers (layer_types, mlp_layer_types, compress_ratios, ...);
per-layer *id* lists (kv_source_layer_ids, index_source_layer_ids,
engram_layer_ids, candidate_source_layer_id) keep the kept ids, renumbered,
and a list paired with one of them (engram_num_embeddings) keeps the matching
entries. Quantisation and everything else stays exactly as released.

Layer tensors are recognised in both spellings: `model.layers.N.` and the
unprefixed `layers.N.` of DeepSeek's own naming.

--drop PREFIX   leave out tensors whose name starts with PREFIX (e.g. `mtp.`,
                the multi-token-prediction layers, which no forward pass reads)
--lazy REGEX    write the tensors whose name matches as holes of a sparse file
                (no disk space, read as zeros) and record where each comes from
                in `lazy.json` next to it. tools/lazy_checkpoint.py fills a
                hole from the Hub the first time a reference reads it, one
                tensor (or one row of a large table) at a time, so after one
                reference run the file holds exactly the routed experts and
                the table rows the prompt used. For MoE layers whose experts
                do not fit on the disk (DeepSeek V4.1: 7 GB a layer) and tables
                larger than the disk (its 98 GB engram embeddings). ditch reads
                what the reference filled; a hole it reads instead (an expert
                the reference did not route to) is zeros, which shows up as a
                mismatch, never as a silent pass.
"""
import sys, json, struct, os, re, threading, requests
from concurrent.futures import ThreadPoolExecutor
from huggingface_hub import hf_hub_url, hf_hub_download, list_repo_files

args = sys.argv[1:]
drop, lazy, pos, layers = [], [], [], None
i = 0
while i < len(args):
    if args[i] == '--drop':
        drop.append(args[i + 1]); i += 2
    elif args[i] == '--lazy':
        lazy.append(re.compile(args[i + 1])); i += 2
    elif args[i] == '--layers':
        layers = [int(x) for x in args[i + 1].split(',')]; i += 2
    else:
        pos.append(args[i]); i += 1
repo = pos[0]
if layers is None:
    layers = list(range(int(pos[1])))
    pos = pos[:1] + pos[2:]
N = len(layers)
new_id = {old: new for new, old in enumerate(layers)}
out = pos[1] if len(pos) > 1 else f"models/{repo.replace('/','__')}-L{N}"
os.makedirs(out, exist_ok=True)
files = list_repo_files(repo)
for f in files:
    if f.endswith(('.json', '.jinja', '.model', '.txt', '.tiktoken', '.py')) and not f.startswith('.') and '/' not in f:
        if f == 'model.safetensors.index.json': continue
        hf_hub_download(repo, f, local_dir=out)

cfg = json.load(open(f"{out}/config.json"))
tc = cfg.get('text_config', cfg)
tc['num_hidden_layers'] = N
for key in ('layer_types', 'mlp_layer_types', 'num_attention_heads_per_layer', 'compress_ratios'):
    if isinstance(tc.get(key), list): tc[key] = [tc[key][j] for j in layers]
paired = {'engram_layer_ids': ['engram_num_embeddings']}
for key in ('kv_source_layer_ids', 'index_source_layer_ids', 'engram_layer_ids', 'dspark_target_layer_ids'):
    if isinstance(tc.get(key), list):
        keep = [j for j, x in enumerate(tc[key]) if x in new_id]
        for p in paired.get(key, []):
            if isinstance(tc.get(p), list): tc[p] = [tc[p][j] for j in keep]
        tc[key] = [new_id[tc[key][j]] for j in keep]
# Kimi-Linear / Kimi K3 name their layer kinds with 1-based ids.
lac = tc.get('linear_attn_config')
if isinstance(lac, dict):
    for key in ('kda_layers', 'full_attn_layers'):
        if isinstance(lac.get(key), list):
            lac[key] = [new_id[x - 1] + 1 for x in lac[key] if x - 1 in new_id]
if 'candidate_source_layer_id' in tc:
    tc['candidate_source_layer_id'] = new_id.get(tc['candidate_source_layer_id'], -1)
if any(d.startswith('mtp') for d in drop):
    for key in ('num_nextn_predict_layers',):
        if key in tc: tc[key] = 0
json.dump(cfg, open(f"{out}/config.json", 'w'), indent=1)

if 'model.safetensors.index.json' in files:
    wm = json.load(open(hf_hub_download(repo, 'model.safetensors.index.json', local_dir='/tmp/truncate_idx_' + repo.replace('/', '_'))))['weight_map']
else:
    wm = None
shards = sorted(set(wm.values())) if wm else ['model.safetensors']
s = requests.Session()
layer_re = re.compile(r'(?:^|\.)layers\.(\d+)\.')

def keep(name):
    if any(name.startswith(d) for d in drop): return False
    m = layer_re.search(name)
    return m is None or int(m.group(1)) in new_id

def renamed(name):
    m = layer_re.search(name)
    if m is None: return name
    return name[:m.start(1)] + str(new_id[int(m.group(1))]) + name[m.end(1):]

def needed(sh):
    return wm is None or any(keep(k) for k, v in wm.items() if v == sh)

# Pass 1: headers.
plan = []  # (name in the cut, dtype, shape, url, abs_start, nbytes)
for sh in shards:
    if not needed(sh): continue
    url = hf_hub_url(repo, sh)
    n = struct.unpack('<Q', s.get(url, headers={'Range': 'bytes=0-7'}).content)[0]
    hdr = json.loads(s.get(url, headers={'Range': f'bytes=8-{8+n-1}'}).content)
    hdr.pop('__metadata__', None)
    for k, v in sorted(hdr.items(), key=lambda kv: kv[1]['data_offsets'][0]):
        if not keep(k): continue
        a, b = v['data_offsets']
        plan.append((renamed(k), v['dtype'], v['shape'], url, 8 + n + a, b - a))

header, off = {}, 0
for k, dt, shp, url, start, nb in plan:
    header[k] = {'dtype': dt, 'shape': shp, 'data_offsets': [off, off + nb]}; off += nb
hb = json.dumps(header).encode(); hb += b' ' * ((8 - len(hb) % 8) % 8)
base = 8 + len(hb)

# Jobs: (url, source start, bytes, destination). Tensors stored next to each
# other in a shard are next to each other here too (the plan is in shard
# order), so neighbouring ranges merge into one request of up to CH bytes,
# and the requests run in parallel.
CH = 32 << 20
jobs, holes = [], {}
for k, dt, shp, url, start, nb in plan:
    dst = base + header[k]['data_offsets'][0]
    if any(p.search(k) for p in lazy):
        holes[k] = {'url': url, 'src': start, 'dst': dst, 'bytes': nb, 'shape': shp, 'dtype': dt}
        continue
    done = 0
    while done < nb:
        m = min(CH, nb - done)
        j = jobs[-1] if jobs else None
        if j and j[0] == url and j[1] + j[2] == start + done and j[3] + j[2] == dst + done and j[2] + m <= CH:
            jobs[-1] = (url, j[1], j[2] + m, j[3])
        else:
            jobs.append((url, start + done, m, dst + done))
        done += m
fd = os.open(f"{out}/model.safetensors", os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o644)
os.pwrite(fd, struct.pack('<Q', len(hb)) + hb, 0)
os.ftruncate(fd, base + off)  # sparse: what is never written takes no space
local = threading.local()

def fetch(job):
    url, a, m, dst = job
    sess = getattr(local, 's', None) or requests.Session()
    local.s = sess
    for attempt in range(5):
        try:
            data = sess.get(url, headers={'Range': f'bytes={a}-{a+m-1}'}, timeout=300).content
            if len(data) == m: break
        except requests.RequestException:
            pass
    else:
        raise SystemExit(f'failed to fetch {m} bytes at {a} of {url}')
    os.pwrite(fd, data, dst)
    return m

fetched = 0
total = sum(j[2] for j in jobs)
with ThreadPoolExecutor(16) as ex:
    for i, m in enumerate(ex.map(fetch, jobs)):
        fetched += m
        if i % 50 == 0: print(f'{fetched/1e9:.2f} GB of {total/1e9:.2f}', flush=True)
os.close(fd)
if holes:
    json.dump({'file': 'model.safetensors', 'holes': holes, 'filled': {}}, open(f"{out}/lazy.json", 'w'))
print(out, len(plan), 'tensors', off / 1e9, 'GB logical,', fetched / 1e9, 'GB fetched,', len(holes), 'lazy')

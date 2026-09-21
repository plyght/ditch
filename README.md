# ditch

**ditch** is a from-scratch Zig rebuild of [Heretic](https://github.com/p-e-w/heretic),
Philipp Emanuel Weidmann's tool for fully automatic censorship removal
("abliteration") of transformer language models. It runs Heretic's method
(difference-of-means refusal directions, a per-layer weight kernel, and a
multi-objective TPE that co-minimises refusals and KL divergence), produces
the same kind of Hugging Face model directory, and ships as one dependency-free
binary: no Python, no PyTorch, no GPU required.

The method is Heretic's; please credit Heretic and its author if you use the
results, and read the paper that underlies both tools: Arditi et al., *Refusal
in Language Models Is Mediated by a Single Direction* (2024),
<https://arxiv.org/abs/2406.11717>.

What ditch adds on top of the port:

* **Runs where the weights do not fit.** A memory budget (`--max-ram`) streams
  weights layer by layer, spills the KV cache to scratch storage, and stops
  cleanly at a time limit, so the whole workflow finishes on machines that
  cannot hold the model.
* **Mixture-of-experts aware.** Per-expert edits, and expert-selective
  abliteration that ranks experts by their alignment with the refusal
  direction and searches over how many to touch, with Heretic's broad edit
  kept as a candidate.
* **Warp mode for MoE models bigger than RAM (and than the disk).** The
  trunk streams per layer while routed experts go through a bounded LRU
  expert cache with a persisted hotlist, and `hf://owner/name` reads the
  weights straight from the Hub with range requests, caching chunks locally.
* **A cheaper search.** Early stopping of dominated trials, warm starts from
  earlier studies, and optional multi-direction ablation.
* **Reproducible outputs.** Every export carries a Lua manifest with content
  hashes and the exact parameters; `ditch --reproduce` rebuilds the model.
* **Lua configuration** and a built-in `ditch bench` harness for honest
  before/after numbers.

## Install

Prebuilt binaries for Linux (x86_64, aarch64), macOS (Intel, Apple silicon)
and Windows (x86_64) are attached to every
[release](https://github.com/plyght/ditch/releases); download the archive for
your platform, unpack it and put `ditch` on your `PATH`. Every release ships
with a `SHA256SUMS` file.

To build from source you need Zig 0.16.0:

```sh
git clone https://github.com/plyght/ditch
cd ditch
zig build -Doptimize=ReleaseFast
```

The binary is at `zig-out/bin/ditch`. Copy it somewhere on your `PATH` or run
it in place.

## Usage

```sh
ditch Qwen/Qwen2.5-0.5B-Instruct
```

The argument is a Hugging Face model ID or a local directory containing
`config.json`, `tokenizer.json` and safetensors weights. Models are downloaded
from the Hub (set `HF_TOKEN` for gated models) into `~/.cache/ditch`.

ditch then

1. loads the model and picks a chat template,
2. loads the "good" (harmless) and "bad" (harmful) prompt datasets,
3. finds a good batch size and detects a common response prefix
   (closing chain-of-thought blocks for thinking models),
4. computes baseline scores (refusal count, KL divergence of the unmodified
   model),
5. extracts per-layer refusal directions from the residual stream,
6. runs the optimisation study, printing every trial's parameters and scores,
7. shows the Pareto-optimal trials and lets you save the model, chat with it,
   or run more trials.

Study progress is journaled to `checkpoints/<model>.jsonl` after every trial,
so an interrupted run (Ctrl+C) can be resumed.

### Supported architectures

* Llama 2 / Llama 3 (and Llama-architecture fine-tunes)
* Mistral
* Qwen2 / Qwen2.5
* Qwen3
* Gemma 2 / Gemma 3 (text only)
* Qwen3-MoE, Qwen2-MoE and Mixtral (mixture-of-experts; separate and fused
  expert tensor layouts)

Weights are read from safetensors in F32, F16 or BF16 (sharded checkpoints
are supported; also straight from the Hub with `hf://owner/name`, see "Warp
mode" below) or from a llama.cpp GGUF file (see "GGUF" below).

### MoE and memory budgets

Mixture-of-experts models (Qwen3-MoE, Qwen2-MoE, Mixtral) are supported, with
per-expert deltas and optional expert-selective abliteration: experts are
ranked by how well their down projections align with the refusal direction, and
the number of edited experts and the edit strength become part of the search
(`--expert-selection ranked|random|broad`, the broad edit of every expert stays
a candidate).

Models that do not fit in RAM can run within a memory budget:

```sh
ditch Qwen/Qwen3-30B-A3B --max-ram 12GB --scratch-dir /fast/disk/scratch --time-limit 90m
```

* `--max-ram <size>` (Lua: `max_ram = "12GB"`) switches the weight store from
  memory mapping to *streaming*: only the layer being computed (plus, when it
  fits, the prefetched next one) is resident, and every large buffer ditch
  allocates is counted against the budget. Abliteration touches one matrix at a
  time, exports are written in row chunks (fused expert tensors one expert at a
  time) and the KV cache and residual stream spill to scratch files when they
  do not fit. Mixture-of-experts layers stream too: all expert matrices of a
  layer are acquired together and released with the layer, including the
  fused `[E, ...]` layouts, which are sliced (and transposed where needed)
  straight out of the stacked tensor.
* What streamed mode costs: a forward pass re-reads every layer and the LM
  head from disk, so **the whole model is read once per generated token** and
  decode speed is bound by storage bandwidth, not compute (a 15GB model on a
  1GB/s SSD decodes at most ~1 token per 15 s per batch; prefill and scoring
  batch many tokens per read and are far less affected). Spilled caches add
  scratch traffic on top. Use the largest budget you can, and a fast local
  disk for `--scratch-dir` (default `<cache-dir>/scratch`).
* Before anything runs ditch prints a `Memory estimate` (weights, largest
  layer, workspace, KV cache, export peak, and what streamed vs mapped mode
  would need) and refuses with a `Memory budget too small` explanation and
  exit status 2 when even the minimum resident set does not fit. Part of the
  budget is reserved as headroom for allocations outside ditch's control
  (default `max(10% of --max-ram, 256MB)`, at most half); `--budget-headroom
  <size>` overrides it (`0` gives the whole budget to the model). A memory
  report (budgeted peak, process RSS, weight and scratch traffic) is printed
  after every trial with `--print-debug-information` and once at exit.
* `--time-limit <duration>` (`90m`, `2h`, `1h30m`, plain seconds) stops the
  optimisation cleanly when it expires: the completed trials are in the study
  journal, the process exits with status 0 and `--checkpoint-action continue`
  resumes the study. A limit that expires during an export leaves the output
  directory marked incomplete (see below) and exits non-zero.
* Exports write a `.incomplete` marker file before the first shard and delete
  it only after every file was written; a directory still containing
  `.incomplete` (export interrupted, out of disk, time limit) is refused by
  ditch when loaded. After a successful save the export is reloaded through
  the streamed path and its first-token logits compared with the in-memory
  model (`max |Δ|` and argmax agreement are printed).
* `--max-vram` is accepted for command-line compatibility with heretic and
  ignored (there is no GPU backend).

All of this is documented in `config.default.lua` as well.

### Warp mode

Mixture-of-experts models are mostly experts: in Qwen3-30B-A3B the routed
experts are about 90% of the weights, but a token only touches 8 of the 128
experts of each layer. Plain streamed mode ignores that and re-reads every
expert of a layer for every forward pass. *Warp mode* keeps the trunk
resident and streams only the experts that are actually selected, through a
bounded cache, so a model far bigger than RAM (or than the local disk, with
the remote source below) can be calibrated, searched and exported. The idea
follows [WARP](https://github.com/sqliteai/warp) (trunk resident, experts
streamed from NVMe into a bounded expert cache); ditch implements it natively
on top of its budget and weight-store machinery, so everything else (budget
accounting, spilling, exports, validation, manifests) works unchanged.

Warp mode is on automatically for a streamed mixture-of-experts model, i.e.
whenever `--max-ram` is set, `--expert-cache` is set, or the weights come
from a remote source. `--expert-cache 0` turns it off (whole layers are
streamed as before); mapped mode and dense models are untouched.

```sh
ditch hf://Qwen/Qwen3-30B-A3B --max-ram 12GB --expert-cache 6GB \
    --scratch-dir /fast/disk/scratch --time-limit 4h
```

**Memory model.** Three parts share the budget:

* the *trunk* (attention, norms, router, shared experts, embeddings, LM head)
  is acquired per layer and prefetched exactly like plain streamed mode;
* the *expert cache* (`src/expert_cache.zig`) is an LRU cache keyed by
  (layer, expert) that holds all three matrices of an expert (gate, up, down,
  or the slices of a fused `[E, ...]` block). After the router has picked the
  top-k experts of a token batch, the union of the selected experts of that
  layer is requested: hits are free, the misses are read from the store as
  one group on background tasks, and each expert is pinned only while it is
  being multiplied with, so a layer whose union does not fit still runs, one
  expert resident at a time (older entries of other layers are evicted
  first). The capacity is what `--max-ram` leaves after the trunk (two layers
  when prefetching) and a quarter reserved for workspaces, KV caches and
  deltas; `--expert-cache <size>` overrides it. The budget's allocator can
  also reclaim unpinned experts when another allocation is refused, so the
  cache never starves the workspaces;
* *workspaces*, KV caches and deltas as before (spilled to scratch when they
  do not fit).

The hard minimum is therefore the trunk plus the experts one token selects,
which the `Memory estimate` prints as `warp mode: min ...`, together with a
`trunk per layer`, `routed expert` and `expert cache` line (how many experts
the cache holds). The `Expert cache:` report (after every trial with
`--print-debug-information`, and at exit) gives hits, misses, hit rate, bytes
read, evictions, how many experts were warmed from the hotlist, how many
distinct experts were visited, and for decoding the number of steps and the
average and maximum misses per step.

**Hotlist.** Every access is counted per (layer, expert). At exit the counts
are written to `<scratch-dir>/<model>.hotlist`, a small Lua table
(`{ layer, expert, uses }` entries, hottest first), and the next run on the
same model loads the hottest experts that fit into the cache before anything
else runs (`* hotlist: N experts warmed`). `--no-hotlist` disables both.

**Expert-selective abliteration.** Scoring and editing experts goes through
the same cache, so a trial never reads an expert twice, and in warp mode
only experts that a calibration or evaluation prompt actually routed to are
scored and edited (`--visited-experts-only`, default on in warp mode): an
expert no prompt reached cannot have influenced a refusal, and skipping it
saves its read. Exports still copy every expert, edited or not, so the
output model is complete.

**Remote weights (`hf://`).** A model id of the form `hf://owner/name` (or a
plain id with `--remote-weights`, or an `http(s)://host/path/` base URL)
runs the model without downloading it first: `config.json`, the tokenizer
files, `generation_config.json`, `model.safetensors.index.json` and the
8-byte length plus JSON header of every shard are fetched up front; tensor
bytes are fetched on demand with HTTP `Range` requests (206 responses,
redirects to the CDN followed, `HF_TOKEN` honoured, `curl -r` as the fallback
when the built-in client cannot be used) in aligned chunks of
`--remote-chunk-size` (default 8 MB). Chunks are cached on disk under
`<cache-dir>/models/<id>/<revision>/chunks/<shard>/<index>`, so nothing is
fetched twice across runs, up to four chunks are in flight at once (the
trunk prefetch and the expert cache read on background tasks), and a later
plain `ditch owner/name` run assembles a shard from its chunks instead of
downloading it when every chunk is present (an export reads every tensor,
so after a save the whole model is cached). Remote weights imply streamed
mode. The source is safetensors only: a GGUF model must be local.

**What it costs, honestly.** Prefill is cheap: the union of experts a batch
routes to is read once per layer and the whole batch is computed against it.
Decode is bound by expert reads per token: every step needs the trunk (read
once per step, as in streamed mode) plus, per layer, whatever selected
experts are not in the cache, and with a cache much smaller than the working
set that is close to top-k expert reads per layer per token. With a fast
NVMe and a cache that holds the hot experts the hit rate climbs quickly
(the hotlist makes the second run start warm), but a decode step still costs
milliseconds to seconds of I/O, not microseconds. Measured on the synthetic
16-expert fixture (`tests/fixtures/qwen3_moe_big`: 4 routed layers, top-2,
9 KB experts, 590 KB of experts in 732 KB of weights) in the e2e test with a
192 KB budget: the cache held 92.0KB (10 experts of 64), the run saw 35057 hits and 8894 misses (79.8% hit rate) with 8894 evictions and 78.2MB of expert reads over calibration, two trials, validation and one export; decoding ran 105 batch steps of 2 sequences at 11.97 misses per step on average (max 16, out of 16 expert selections per step: top-2 in 4 layers for 2 sequences) and the second run warmed 10 experts from the hotlist before its first forward pass. Over the remote source with 4 KB chunks, a
two-token prefill of that fixture fetched 20 ranges (78 KB: both shard headers and the norms the loader reads; the small files are plain downloads) at load and 85 ranges (337 KB, 47% of the 715 KB model) after the prefill, with bitwise the same logits as the memory-mapped model, and a second
run fetched 0 ranges. These numbers only show that the machinery works;
real-model numbers must be measured with `ditch bench` on your own storage.

### GGUF

Most abliterated models end up in llama.cpp, so ditch reads and writes
[GGUF](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md) directly.

**Output.** `--export-format gguf` (or `both`; `hf`, the safetensors directory,
is the default for Hugging Face inputs) writes a single `model.gguf` next to
the model card and the reproducibility manifest. `--gguf-dtype` selects the
storage type of the 2-D matrices: `f16` (default), `bf16`, `f32`, `q8_0`
(ggml's Q8_0: blocks of 32 values with an f16 scale), `q4_0`, `q4_1`, `q5_0`
or `q5_1`; norms, biases and other 1-D tensors are always f32, and the token
embeddings and the output projection stay f16 when a quantised type is
chosen. The writer follows the conventions of llama.cpp's
`convert_hf_to_gguf.py`: its tensor names (`token_embd`, `blk.N.attn_q`,
`ffn_gate_exps`, ...), the q/k row permutation of the llama family, gemma norms
stored as `1 + w`, stacked `[n_expert][...]` expert tensors, the architecture
keys llama.cpp reads (context and embedding length, head counts, RMS epsilon,
RoPE base and scaling, llama3 scaling as `rope_freqs.weight`, sliding window,
soft-capping, expert counts) and the tokenizer (`gpt2` byte-level vocabularies
with merges and the `qwen2` / `llama-bpe` / `gpt-2` pre-tokenizer name, or a
`llama` SentencePiece-style vocabulary with scores) plus the chat template.
The abliteration deltas are merged into the affected tensors exactly as in
the safetensors export, tensor by tensor, so the whole model is never held in
memory. The file also embeds the original `config.json` and `tokenizer.json`
(`tokenizer.huggingface.json`) so a later Hugging Face export is exact.
The format is spec-conformant (v3, little endian, 32-byte alignment) and is
tested by round trip in ditch's own reader; it has not been run through
llama.cpp itself here.

**Input.** `ditch path/to/model.gguf` (or a directory holding one `.gguf`)
loads a llama.cpp model of any supported architecture (`llama` including
Mistral and Mixtral, `qwen2`, `qwen3`, `qwen2moe`, `qwen3moe`, `gemma2`,
`gemma3`). The configuration is rebuilt from the metadata, the tokenizer from
the ggml vocabulary (merges for `gpt2` vocabularies, scores for `llama` ones,
the pre-tokenizer name mapped to the matching regular expression), the
permutation and norm conventions above are undone, stacked expert tensors are
split into per-expert views, and quantised tensors are dequantised row by row
inside the kernels. Everything else (`--max-ram` streaming, `--evaluate-model`,
`--reproduce`, MoE expert selection) works unchanged. A GGUF input defaults to
a GGUF export (`--gguf-dtype source`): untouched tensors are copied
byte-for-byte, edited tensors are re-quantised to their own type (Q8_0 stays
Q8_0; K-quants, which ditch cannot produce, become Q8_0), and
`--export-format hf` writes a safetensors model with quantised tensors
dequantised to f16.

| ggml type | read | written |
| :--- | :---: | :---: |
| F32, F16, BF16 | yes | yes |
| Q8_0 | yes | yes |
| Q4_0, Q4_1, Q5_0, Q5_1 | yes | yes |
| Q4_K, Q6_K, Q8_K | yes | no (edited tensors become Q8_0) |
| other K-/IQ-quants | no | no |

`tools/make_fixture.py <family> --gguf` writes the synthetic test models as
GGUF (with Q8_0 feed-forward matrices) for the unit tests, which check the
NumPy reference logits, the tokenizer rebuilt from the vocabulary, HF-to-GGUF
round trips for every family (f16 within 1e-2 and Q8_0 within 5e-2 of the
largest logit) and the llama q/k permutation.

### Datasets

A dataset is either

* a Hugging Face dataset ID such as `mlabonne/harmful_behaviors`, whose rows
  are fetched through the [datasets-server](https://huggingface.co/docs/datasets-server)
  JSON API (`split` and `column` are required, `config` is optional), or
* a local text file with one prompt per line (empty lines are ignored; `split`
  may be a slice such as `[:400]`).

Datasets are configured for the direction extraction (`good_prompts`,
`bad_prompts`) and for each scorer (`[scorer.KeywordRate.prompts]`,
`[scorer.KLDivergence.prompts]`), on the command line via
`--good-prompts-dataset`, `--bad-prompts-dataset`,
`--keyword-rate-prompts-dataset` and `--kl-divergence-prompts-dataset`.

### Performance on a CPU

ditch computes everything in f32 on the CPU with a small multithreaded kernel
library. It is meant for small models: a 0.5B model runs a trial (100 refusal
prompts of 100 tokens plus 100 first-token evaluations) in a few minutes on a
laptop; 1.5B-3B models are workable with patience; anything larger is slow.
Useful knobs:

```sh
ditch Qwen/Qwen2.5-0.5B-Instruct --n-trials 50 --n-startup-trials 15 \
    --max-response-length 50 --threads 8
```

* `--n-trials` / `--n-startup-trials`: Heretic's defaults (200 / 60) are tuned
  for GPUs; 50 / 15 already finds good settings for small models.
* `--max-response-length`: shorter responses make the refusal scorer cheaper.
* `--batch-size`: skip auto-detection.
* Early stopping (on by default, see below) skips the rest of the refusal
  scoring for trials that can no longer reach the Pareto front, and
  `--warm-start <study.jsonl>` seeds the search with a previous study of the
  same architecture so the random start-up phase can be shortened.
* Use `--evaluate-model <dir>` to score an already exported model against the
  base model's scorers without running a study.

### Reproducing a model

Every exported model directory contains `ditch-reproduce.lua`, a manifest that
records what produced it: the ditch version, the source model with the SHA-256
of its `config.json` and of every safetensors file, the settings that affect
the result (seed, abliteration options, chat template, response prefix, the
datasets with a SHA-256 of the exact prompt lists), the selected trial's full
search-space parameter vector and its decoded kernel parameters, and the scores
next to the baseline scores. The key content is also written into the model
card (`README.md`).

```sh
ditch --reproduce out/qwen2.5-0.5b-ditch/ditch-reproduce.lua
```

reads the manifest through the Lua sandbox, downloads or opens the recorded
source model and verifies the file hashes (the run stops with a clear message
on a mismatch unless `--ignore-mismatches` is given), loads the recorded prompt
datasets and checks their hashes, re-derives the refusal directions and applies
the recorded trial directly, without a search. The scores are printed next to
the recorded ones (the same seed gives identical results on the same machine)
and the usual model menu follows, so `--model-action save --save-directory DIR`
exports the model again.

## How it works

ditch implements Heretic's algorithm unchanged:

1. **Directions.** For every layer, the mean residual-stream activation at the
   last prompt token is computed over the harmless and the harmful prompts.
   The normalised difference of means is the refusal direction of that layer;
   optionally only its component orthogonal to the harmless direction is kept
   ("projected abliteration").
2. **Weight kernel.** For each abliterable weight matrix (the attention output
   projection and the MLP down projection of every layer), the direction is
   projected out with a per-layer weight given by a small kernel: a maximum
   weight at some layer position, decaying linearly to a minimum weight over a
   distance. Whether one global direction or each layer's own direction is used
   is also a parameter. The change is stored as a low-rank delta, so resetting
   the model is free and exports merge it into the weights.
3. **Optimisation.** A multi-objective Tree-structured Parzen Estimator (a
   port of Optuna's multivariate TPE) searches the kernel parameters to
   co-minimise the number of refusals on harmful prompts and the KL divergence
   of the first-token distribution on harmless prompts, i.e. it removes
   refusals while changing the model as little as possible. The Pareto front is
   presented for selection.

### Beyond heretic

Three optional extensions make the search cheaper or the edit stronger. All
are documented in `config.default.lua`; the first two are on by default and
do not change what the search finds.

* **Early stopping** (`early_stop`, `--no-early-stop`). Scorers run in the
  configured order, KL divergence first, so a trial's KL divergence (one
  prefill pass) is known before its refusals are counted batch by batch. After
  every batch the trial is pruned as soon as its refusals so far exceed those
  of a completed Pareto-optimal trial whose KL divergence is not larger: that
  trial dominates it whatever the remaining prompts do, so it could never
  enter the front. Pruned trials are journaled with the remaining prompts
  counted as refusals (an upper bound), which the TPE still learns from, but
  they are never offered in the results menu. The trial log prints
  `* Pruned after N/M prompts`.
* **Warm start** (`--warm-start <study.jsonl>`). The trials of a previous
  study on the same architecture (same number of layers, components and
  objectives; verified from the journal) seed the sampler. They are used for
  sampling only: they count neither towards `n_trials` nor appear in the
  results, and the previous journal is never written to. Combine with a small
  `--n-startup-trials` to skip most of the random exploration.
* **Multi-direction ablation** (`--n-directions K`). Heretic removes one
  direction per layer. With `K > 1`, direction 1 is the difference of means
  as before and directions 2..K are the top principal components of the
  per-prompt harmful residuals (centred on the harmless mean) after
  projecting out direction 1, estimated from a randomised Nyström covariance
  sketch accumulated in the same pass over the prompts (no
  `hidden × hidden` matrices). The orthonormal basis is projected out at once
  (a rank-K delta, in all row-normalisation modes), a fractional global
  direction index interpolates and re-orthonormalises the whole basis, and
  the study manifest records K so a study cannot be continued with a
  different value.

## Configuration

Configuration is written in Lua. Copy `config.default.lua` to `config.lua` in
the directory you run ditch from, or pass `--config <file>`. The file is a Lua
5.4 script (run in a sandbox with the base, string, table, math and utf8
libraries and `os.getenv`) that returns a table of settings, so values can be
computed:

```lua
local trials = tonumber(os.getenv("DITCH_TRIALS")) or 50
return {
  n_trials = trials,
  n_startup_trials = trials // 4,
  good_prompts = { dataset = "prompts/good.txt" },
  bad_prompts = { dataset = "prompts/bad.txt" },
}
```

Every option is documented in `config.default.lua` and can also be given on
the command line (`--n-trials 50`, `--row-normalization pre`, ...). Heretic
`config.toml` files are accepted as well (`--config config.toml`, or a
`config.toml` in the working directory when no `config.lua` exists): options that only
apply to the PyTorch implementation are ignored.

### Non-interactive use

The interactive menus can be answered from the command line:

| Option | Meaning |
| --- | --- |
| `--checkpoint-action continue\|restart` | what to do with an existing checkpoint |
| `--trial-index N` | select the N-th Pareto-optimal trial (as numbered in the results menu) |
| `--n-additional-trials N` | run N more trials before showing the results |
| `--model-action save\|chat\|exit` | what to do with the selected model |
| `--save-directory DIR` | where to save it |
| `--export-dtype bf16\|f16\|f32` | storage dtype of the exported weights |
| `--export-format hf\|gguf\|both` | Hugging Face directory (default), a llama.cpp GGUF file, or both |
| `--gguf-dtype f16\|bf16\|f32\|q8_0\|...\|source` | storage type of the GGUF matrices (see "GGUF") |
| `--reproduce FILE` | re-derive a model from its `ditch-reproduce.lua` (no search) |
| `--ignore-mismatches` | proceed with `--reproduce` even if file or prompt hashes differ |
| `--warm-start FILE` | seed the sampler with the trials of a previous study |
| `--no-early-stop` | score every trial completely |
| `--n-directions K` | remove K orthonormal directions per layer |
| `--expert-cache SIZE` | warp mode: capacity of the expert cache (`0` = off) |
| `--visited-experts-only` | warp mode: edit only experts the prompts routed to (default on) |
| `--no-hotlist` | do not write/read `<scratch-dir>/<model>.hotlist` |
| `--remote-weights`, `--remote-chunk-size SIZE` | read a plain Hub id like `hf://`; chunk size of the range cache |

For example, a fully unattended run:

```sh
ditch Qwen/Qwen2.5-0.5B-Instruct --n-trials 50 --n-startup-trials 15 \
    --checkpoint-action restart --trial-index 1 \
    --model-action save --save-directory out/qwen2.5-0.5b-ditch
```

## Benchmarks

`ditch bench <model>` measures what a study costs on the current machine and
prints a Markdown table (also written to a file with `--bench-output
<file.md>`). It loads the model, prompt datasets and scorers exactly like a
normal run, so the same `config.lua`, `--config` and dataset options apply
(local text files work), and reports:

* model, threads and batch size,
* prefill tokens/s and greedy decode tokens/s over a batch of
  `--bench-prompts N` prompts (default 16) decoding `--bench-tokens M` tokens
  each (default 32),
* the time of one residual-mean pass over the good-prompt dataset,
* the time of one abliteration apply for each row-normalisation mode
  (`none`, `pre`, `full`), at the centre of the search space,
* the time of one full trial: apply with the configured mode plus refusal
  scoring and KL scoring with the configured prompt counts,
* peak RSS (`VmHWM` from `/proc/self/status` on Linux) and the total weight
  bytes.

```sh
ditch bench Qwen/Qwen2.5-0.5B-Instruct --bench-prompts 16 --bench-tokens 32 \
    --bench-output bench.md
```

Run it on your own hardware with a real model; the numbers below come from the
synthetic 3-layer test fixture (`tests/fixtures/qwen2`: hidden size 32,
vocabulary 271, 64 KB of BF16 weights) with six prompts per dataset and
`--max-response-length 8` on this development sandbox's 4 CPUs, and only show
that the harness runs. They say nothing about real models.

| Metric | Value (synthetic fixture, 4 CPUs) |
| :--- | ---: |
| Model | `tests/fixtures/qwen2` (qwen2, 3 layers, BF16 weights) |
| Threads | 4 |
| Batch size | 2 |
| Prefill tokens/s | 62240.4 (1196 tokens in 19.2 ms) |
| Decode tokens/s | 11983.1 (16 prompts x 32 tokens, greedy, 42.7 ms) |
| Residual-mean pass | 10.7 ms (6 prompts) |
| Apply time (row_normalization = none) | 0.1 ms |
| Apply time (row_normalization = pre) | 0.3 ms |
| Apply time (row_normalization = full) | 2.3 ms |
| Time per trial | 39.1 ms (apply + 6 refusal prompts + 6 KL prompts) |
| Peak RSS | 7.0 MiB (7356416 bytes) |
| Total weight bytes | 64256 (62.8 KiB) |

`tools/compare_heretic.md` explains how to take the same measurements with
Heretic on the same machine and model, and has a template table for reporting
peak RAM, time per trial and result quality (Refusals, KL divergence) side by
side.

## Development

```sh
zig build test --summary all   # unit tests (tiny synthetic models in tests/fixtures)
zig build e2e                  # full pipeline on a synthetic model (bash tests/e2e.sh)
```

## License

ditch is free software, licensed under the GNU Affero General Public License
version 3 or (at your option) any later version. Copyright (C) 2026 ditch
contributors. It is a port of Heretic, Copyright (C) 2025-2026 Philipp Emanuel
Weidmann and contributors, licensed under the same terms. See `LICENSE`.

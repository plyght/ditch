# ditch

**ditch** is a from-scratch Zig rebuild of [Heretic](https://github.com/p-e-w/heretic),
Philipp Emanuel Weidmann's tool for fully automatic censorship removal
("abliteration") of transformer language models. It runs the same algorithm,
reads the same configuration format, produces the same kind of Hugging Face
model directory, and needs nothing but a C-free static binary: no Python,
no PyTorch, no GPU.

Everything that makes Heretic work is Heretic's idea; ditch only re-implements
it. Please credit Heretic and its author if you use the results, and read the
paper that underlies both tools: Arditi et al., *Refusal in Language Models Is
Mediated by a Single Direction* (2024), <https://arxiv.org/abs/2406.11717>.

## Install

ditch needs Zig 0.16.0.

```sh
git clone <this repository> ditch
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

Weights are read from safetensors in F32, F16 or BF16. Sharded checkpoints are
supported.

### MoE and memory budgets

Mixture-of-experts models (Qwen3-MoE, Qwen2-MoE, Mixtral) are supported, with
per-expert deltas and optional expert-selective abliteration: experts are
ranked by how well their down projections align with the refusal direction, and
the number of edited experts and the edit strength become part of the search
(`--expert-selection ranked|random|broad`, the broad edit of every expert stays
a candidate). Large models can run within a memory budget (`--max-ram`,
`--scratch-dir`, `--time-limit`): weights are streamed layer by layer and the
KV cache spills to scratch storage. Both are documented in `config.default.lua`.

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
| `--reproduce FILE` | re-derive a model from its `ditch-reproduce.lua` (no search) |
| `--ignore-mismatches` | proceed with `--reproduce` even if file or prompt hashes differ |
| `--warm-start FILE` | seed the sampler with the trials of a previous study |
| `--no-early-stop` | score every trial completely |
| `--n-directions K` | remove K orthonormal directions per layer |

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

# ditch

*ditches censorship.*

**ditch** removes refusal behaviour ("abliteration") from open-weight language
models, fully automatically, on a CPU. It is a from-scratch Zig rebuild of
[Heretic](https://github.com/p-e-w/heretic) by Philipp Emanuel Weidmann and
runs the same method: difference-of-means refusal directions, a per-layer
weight kernel, and a multi-objective TPE that co-minimises refusals and KL
divergence from the original model. The method is Heretic's; credit it and
the underlying paper (Arditi et al., *Refusal in Language Models Is Mediated
by a Single Direction*, 2024, <https://arxiv.org/abs/2406.11717>) when you
use the results.

What ditch adds:

* **One dependency-free binary.** No Python, PyTorch or GPU. Linux, macOS
  and Windows builds on every release.
* **Models bigger than RAM.** A memory budget streams weights layer by layer;
  for mixture-of-experts models, *warp mode* keeps the trunk resident and
  streams only the routed experts through a bounded cache, and an `hf://`
  source fetches tensors on demand instead of downloading the checkpoint.
* **Expert-selective abliteration.** Experts are ranked by their alignment
  with the refusal direction and the number to edit is searched, with
  Heretic's edit-every-expert as a candidate.
* **A cheaper search.** Early stopping of dominated trials, warm starts from
  earlier studies, optional multi-direction ablation.
* **GGUF in and out**, including quantised weights.
* **Reproducible exports** (a Lua manifest with content hashes) and a
  benchmark harness.

## Install

Download the archive for your platform from the
[releases page](https://github.com/plyght/ditch/releases) and put `ditch`
on your `PATH`. To build from source you need Zig 0.16.0:

```sh
git clone https://github.com/plyght/ditch && cd ditch
zig build -Doptimize=ReleaseFast   # binary: zig-out/bin/ditch
```

## Usage

```sh
ditch Qwen/Qwen2.5-0.5B-Instruct
```

The argument is a Hugging Face model ID, a local model directory, a `.gguf`
file, or `hf://owner/name` to stream weights without downloading them. ditch
loads the model, fetches the harmless and harmful prompt sets, detects a batch
size and response prefix, measures baseline scores, extracts refusal
directions, runs the optimisation study (journaled to `checkpoints/`, so
Ctrl+C is resumable), then shows the Pareto-optimal trials and lets you save
the model, chat with it, or run more trials.

Useful flags (all also settable in `config.lua`; see `ditch --help`):

| Flag | Purpose |
| --- | --- |
| `--n-trials N`, `--n-startup-trials N` | search length (defaults 200 / 60; use far fewer on a CPU) |
| `--max-ram 12GB`, `--time-limit 90m`, `--scratch-dir DIR` | memory budget, clean stop, spill location |
| `--expert-cache 6GB`, `--expert-selection ranked\|random\|broad` | warp mode and MoE editing |
| `--n-directions K`, `--no-early-stop`, `--warm-start study.jsonl` | search extensions |
| `--export-format hf\|gguf\|both`, `--gguf-dtype f16\|q8_0\|...` | output format |
| `--checkpoint-action`, `--trial-index`, `--model-action`, `--save-directory` | answer the menus non-interactively |
| `--evaluate-model DIR`, `--reproduce FILE`, `ditch bench MODEL` | evaluate, reproduce, measure |

### Supported models

Dense: Llama 2/3, Mistral, Qwen2/2.5, Qwen3, Gemma 2/3 (text). Mixture of
experts: Qwen3-MoE, Qwen2-MoE, Mixtral, in separate and fused expert
layouts. Weights are read from safetensors (F32/F16/BF16) or GGUF. Every
family is verified against a reference implementation in the test suite.

### Datasets

A dataset is a Hugging Face dataset ID (rows come from the datasets-server
API, so it must be viewable on the Hub) or a text file with one prompt per
line. Defaults are Heretic's (`mlabonne/harmless_alpaca`,
`mlabonne/harmful_behaviors`).

## Configuration

Settings live in `config.lua` (or `--config FILE`), a sandboxed Lua 5.4
script that returns a table; every option is documented in
[`config.default.lua`](config.default.lua). Heretic `config.toml` files are
accepted too.

```lua
local trials = tonumber(os.getenv("DITCH_TRIALS")) or 50
return {
  n_trials = trials,
  n_startup_trials = trials // 4,
  good_prompts = { dataset = "prompts/good.txt" },
  bad_prompts = { dataset = "prompts/bad.txt" },
}
```

## Memory budget and warp mode

```sh
ditch hf://Qwen/Qwen3-30B-A3B --max-ram 12GB --expert-cache 6GB \
    --scratch-dir /fast/disk --time-limit 4h
```

With `--max-ram`, weights are streamed instead of memory-mapped, the KV cache
and activations spill to `--scratch-dir` when they do not fit, and every
large buffer is counted against the budget. ditch prints a memory estimate
first and refuses (exit 2) when the minimum resident set cannot fit. Exports
are validated by reloading them; an interrupted export leaves a
`.incomplete` marker that ditch refuses to load. A time limit stops the
study cleanly (exit 0) and `--checkpoint-action continue` resumes it.

For mixture-of-experts models this becomes warp mode (after
[WARP](https://github.com/sqliteai/warp)): attention, norms, router and
shared experts stay resident, routed experts are loaded on demand into an
LRU cache sized by `--expert-cache`, and a hotlist warms the cache on the
next run. `hf://` sources fetch only the tensors that are touched, using HTTP
range requests cached on disk (`--remote-chunk-size`). Only experts visited
during calibration are scored and edited (`--visited-experts-only`); every
expert is still exported.

The honest cost: streamed decode re-reads the weights it needs for every
generated token, so throughput is bound by storage bandwidth. Prefill and
scoring batch many tokens per read and suffer far less. Measure with
`ditch bench` before committing to a long run.

## GGUF

`ditch model.gguf` reads GGUF directly (config and tokenizer are rebuilt from
the metadata); `--export-format gguf` writes one with llama.cpp's tensor
names, metadata and permutations. Untouched tensors are copied byte for
byte; edited tensors are re-quantised to the source type.

| ggml type | read | written |
| :--- | :---: | :---: |
| F32, F16, BF16 | yes | yes |
| Q8_0, Q4_0, Q4_1, Q5_0, Q5_1 | yes | yes |
| Q4_K, Q6_K, Q8_K | yes | edited tensors become Q8_0 |
| other K-/IQ-quants | no | no |

The writer follows the GGUF v3 spec and llama.cpp's conventions and is tested
by round trip in ditch; it has not been run through llama.cpp itself.

## Reproducibility and benchmarks

Every saved model contains `ditch-reproduce.lua`: ditch version, SHA-256 of
the source files and prompt sets, the exact settings and trial parameters,
and the scores. `ditch --reproduce path/to/ditch-reproduce.lua` verifies the
hashes and rebuilds the model without a search.

`ditch bench MODEL` prints prefill and decode throughput, direction
extraction, apply and per-trial times, and peak memory as a Markdown table
(`--bench-output file.md`). `tools/compare_heretic.md` describes how to
measure Heretic on the same machine for a fair comparison; ditch does not
publish real-model numbers it has not measured.

## How it works

1. **Directions.** Per layer, the mean last-token residual over harmful
   minus harmless prompts, normalised (optionally projected orthogonal to
   the harmless direction). With `--n-directions K`, further principal
   components of the harmful residuals are added from a streaming covariance
   sketch.
2. **Edit.** For every attention output and MLP down projection, the
   direction is projected out with a per-layer weight from a small kernel
   (maximum weight at a position, decaying to a minimum over a distance),
   stored as a low-rank delta so trials never touch the base weights. Row
   normalisation (`none`, `pre`, `full`) follows Heretic.
3. **Search.** A multi-objective TPE (Optuna's multivariate defaults)
   proposes kernel parameters, direction scope and, for MoE, how many ranked
   experts to edit. Trials score KL divergence first; a trial whose refusals
   already exceed a Pareto-front trial with no worse KL is pruned.

## Development

```sh
zig build test --summary all   # unit tests (NumPy-generated fixtures in tests/fixtures)
bash tests/e2e.sh              # end-to-end run of every feature on the fixtures
zig fmt --check src build.zig
```

Releases are built by `.github/workflows/release.yml` (dispatch it with a
version, or push a `v*` tag) and cross-compiled for Linux, macOS and Windows.

## License

AGPL-3.0-or-later, like Heretic. Lua 5.4 (MIT) is vendored in `vendor/lua54`.

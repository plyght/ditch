*ditch censorship.*

**ditch** removes refusal behaviour ("abliteration") from open-weight language
models, fully automatically, on a CPU. It is a from-scratch Zig rebuild of
[Heretic](https://github.com/p-e-w/heretic) by Philipp Emanuel Weidmann and runs
the same method: difference-of-means refusal directions, a per-layer weight
kernel, and a multi-objective TPE that co-minimises refusals and KL divergence
from the original model. The method is Heretic's; credit it and the underlying
paper (Arditi et al., *Refusal in Language Models Is Mediated by a Single
Direction*, 2024, <https://arxiv.org/abs/2406.11717>) when you use the results.

What ditch adds:

* **One dependency-free binary.** No Python, PyTorch or GPU. Linux, macOS and
  Windows builds on every release.
* **Models bigger than RAM.** A memory budget streams weights layer by layer;
  for mixture-of-experts models, *warp mode* streams only the routed experts,
  and an `hf://` source fetches tensors on demand instead of downloading.
* **Expert-selective abliteration.** Experts are ranked by their alignment with
  the refusal direction and the number to edit is searched.
* **A cheaper search.** Early stopping of dominated trials, warm starts,
  optional multi-direction ablation.
* **GGUF in and out**, quantised safetensors (FP8, MXFP4, INT4) dequantised on
  load, reproducible exports (a Lua manifest with content hashes) and a
  benchmark harness.

## Install

Download the archive for your platform from the
[releases page](https://github.com/plyght/ditch/releases) and put `ditch` on
your `PATH`; to build from source you need Zig 0.16.0:

```sh
git clone https://github.com/plyght/ditch && cd ditch
zig build -Doptimize=ReleaseFast   # binary: zig-out/bin/ditch
```

To uninstall, delete the binary and its cache, `~/.cache/ditch` (or
`$DITCH_CACHE`).

## Usage

```sh
ditch Qwen/Qwen2.5-0.5B-Instruct
```

The argument is a Hugging Face model ID, a local directory, a `.gguf` file, or
`hf://owner/name` to stream weights without downloading them (a directory named
like a subcommand must be given as a path, e.g. `./bench`). ditch loads the
model, fetches the prompt sets, detects a batch size and response prefix,
measures baseline scores, extracts refusal directions, runs the optimisation
study (journaled to `checkpoints/`, so Ctrl+C is resumable), then shows the
Pareto-optimal trials and lets you save the model, chat with it, or run more
trials.

Useful flags (all also settable in `config.lua`; see `ditch --help`):

| Flag | Purpose |
| --- | --- |
| `--n-trials N`, `--n-startup-trials N` | search length (defaults 200 / 60; use far fewer on a CPU) |
| `--max-ram 12GB`, `--time-limit 90m`, `--scratch-dir DIR` | memory budget, clean stop, spill location |
| `--expert-cache 6GB`, `--expert-selection ranked\|random\|broad` | warp mode and MoE editing |
| `--n-directions K`, `--no-early-stop`, `--warm-start study.jsonl` | search extensions |
| `--fast-search`, `--direction-method separating`, `--direction-range auto`, `--ablate-inputs`, `--kl-tokens T`, `--select auto` | algorithm options (see "How it works") |
| `--export-format hf\|gguf\|both`, `--gguf-dtype f16\|q8_0\|...` | output format |
| `--checkpoint-action`, `--trial-index`, `--model-action`, `--save-directory` | answer the menus non-interactively |
| `--evaluate-model DIR`, `--reproduce FILE`, `ditch bench MODEL` | evaluate, reproduce, measure |

## Supported models

Families are described by an architecture registry (`src/arch.zig`), one entry
per Hugging Face `model_type`. **93 families** are registered, from GPT-2,
GPT-NeoX and BLOOM to Llama, Qwen, Gemma, Phi, GLM, Mistral, Granite, Kimi K3
and DeepSeek V4.1 — dense and mixture-of-experts, Mamba and linear-attention
hybrids, and quantised checkpoints; every one of them is verified against a
NumPy reference forward pass (`tools/make_fixture.py`).
**[docs/models.md](docs/models.md) is the full list**:
every `model_type`, its aliases, what each fixture covers, and every caveat.

The caveats worth knowing up front: image, video and audio models run through
their text config, so the towers are never executed and pass through exports
byte for byte; the seven families with a sparse key indexer run those layers as
their exact dense equivalent while the prompt is short enough, and refuse or
flag anything longer; quantised safetensors (FP8, MXFP4, INT4 and the packed
variants) are dequantised on load and exported as plain bf16; and GGUF covers
nine of the families, the rest loading from safetensors only. An unknown
`model_type`, layer type, quantisation format or activation is an error, not a
silent fallback. Abliteration edits each family's attention output projection
(the `out_proj` of a Mamba block) and MLP down projection, per expert on MoE
layers; exports preserve every tensor name and layout.

## Datasets

A dataset is a Hugging Face dataset ID (rows come from the datasets-server API,
so it must be viewable on the Hub) or a text file with one prompt per line.
Defaults are Heretic's: `mlabonne/harmless_alpaca`, `mlabonne/harmful_behaviors`.

## Configuration

Settings live in `config.lua` (or `--config FILE`), a sandboxed Lua 5.4 script
returning a table keyed like the flags; every option is documented in
[`config.default.lua`](config.default.lua), and Heretic `config.toml` files are
accepted. Precedence, highest first: flags, `DITCH_*` environment variables
(`DITCH_THREADS`, `DITCH_MAX_RAM`, `DITCH_CACHE`, `DITCH_NO_COLOR`),
`./config.lua`, then `$XDG_CONFIG_HOME/ditch/config.lua`. Messages go to stderr
and results to stdout (`--json` for one JSON document, `--plain` for
grep-friendly lines); `--no-input` turns every prompt into an error naming the
flag to pass instead.

## Memory budget and warp mode

```sh
ditch hf://Qwen/Qwen3-30B-A3B --max-ram 12GB --expert-cache 6GB \
    --scratch-dir /fast/disk --time-limit 4h
```

With `--max-ram`, weights are streamed instead of memory-mapped, the KV cache
and activations spill to `--scratch-dir` when they do not fit, and every large
buffer is counted against the budget. ditch prints a memory estimate first and
refuses (exit 2) when the minimum resident set cannot fit. Exports are validated
by reloading them; an interrupted one leaves a `.incomplete` marker that ditch
refuses to load. A time limit stops the study cleanly (exit 0) and
`--checkpoint-action continue` resumes it.

For mixture-of-experts models this becomes warp mode (after
[WARP](https://github.com/sqliteai/warp)): attention, norms, router and shared
experts stay resident, routed experts are loaded on demand into an LRU cache
sized by `--expert-cache`, and a hotlist warms the cache on the next run.
`hf://` sources fetch only the tensors that are touched, using HTTP range
requests cached on disk (`--remote-chunk-size`). Only experts visited during
calibration are scored and edited (`--visited-experts-only`); every expert is
still exported. The honest cost: streamed decode re-reads the weights it needs
for every generated token, so throughput is bound by storage bandwidth. Measure
with `ditch bench` before committing to a long run.

## GGUF

`ditch model.gguf` reads GGUF directly (config and tokenizer are rebuilt from
the metadata); `--export-format gguf` writes one with llama.cpp's tensor names,
metadata and permutations. Untouched tensors are copied byte for byte, edited
ones re-quantised to the source type (F32/F16/BF16 and Q8_0, Q4_0/1, Q5_0/1
round trip; Q4_K, Q6_K and Q8_K are read but edited tensors become Q8_0). The
nine families with a GGUF path and the full type table are in
[docs/models.md](docs/models.md#gguf-per-family). A quantised safetensors source
is exported as plain bf16 instead, so pass `--gguf-dtype q8_0` when you want a
smaller file. The writer follows the GGUF v3 spec and llama.cpp's conventions
and is tested by round trip in ditch; it has not been run through llama.cpp
itself.

## Reproducibility and benchmarks

Every saved model contains `ditch-reproduce.lua`: ditch version, SHA-256 of the
source files and prompt sets, the exact settings and trial parameters, and the
scores. `ditch --reproduce path/to/ditch-reproduce.lua` verifies the hashes and
rebuilds the model without a search. `ditch bench MODEL` measures throughput,
timings and peak memory as a Markdown table (`--bench-output file.md`);
`tools/compare_heretic.md` describes how to measure Heretic on the same machine.

The numbers below were **measured on one machine** (4 vCPU x86-64, 15 GiB RAM,
no GPU; a `ReleaseFast` build, CPU only) with the default Heretic datasets; they
are a data point, not a spec. The full log is in
[`tools/real-model-validation.md`](tools/real-model-validation.md). Automatic
abliteration, 30 trials (10 startup), refusals out of 100:

| Model | arch | Baseline → best refusals | Best-trial KL | Wall clock | Peak RSS |
| --- | --- | ---: | ---: | ---: | ---: |
| Qwen2.5-0.5B-Instruct | qwen2 | 90 → 5 | 0.043 | 50 min | 1.6 GiB |
| Qwen3-0.6B | qwen3 | 56 → 21 | 0.003 | 37 min* | 6.4 GiB |
| SmolLM2-1.7B-Instruct | llama | 35 → 9 | 0.080 | 50 min* | 8.8 GiB |

\* 10 trials. Phi-4-mini-instruct (`phi3`) and Qwen1.5-MoE-A2.7B-Chat
(`qwen2_moe`, warp mode via `hf://`, peak RSS 1.1 GiB for a 27 GB model) also
load and run; the log has the partial results the machine's disk allowed.

ditch vs. Heretic, Qwen2.5-0.5B-Instruct, same 30/10 trials and datasets, both
CPU; each export is also re-scored by ditch's scorers, so the two are on one
yardstick:

| | ditch | Heretic |
| --- | ---: | ---: |
| Wall clock | 50 min | 21 min |
| Peak RSS | 1.6 GiB | 3.1 GiB |
| Best-trial refusals / KL (own scorer) | 5 / 0.043 | 4 / 0.036 |
| Export re-scored by ditch (refusals / KL) | 5 / 0.043 | 9 / 0.140 |

Heretic is faster here (PyTorch BLAS, batch 128 vs ditch's default 32); ditch
uses about half the RAM, no Python or PyTorch, and streams models larger than
RAM. The Pareto fronts are close.

## How it works

1. **Directions.** Per layer, the mean last-token residual over harmful minus
   harmless prompts, normalised (optionally projected orthogonal to the
   harmless direction). With `--n-directions K`, further principal components
   of the harmful residuals are added from a streaming covariance sketch. Where
   the residual is several parallel streams (hyper-connections) or a bank of
   block prefixes (Kimi K3's Attention Residual), "the residual at layer L" is
   the single mixed vector entering layer L's attention block; see
   [docs/models.md](docs/models.md#sparse-indexer-and-hyper-connection-families).
2. **Edit.** For every attention output and MLP down projection, the direction
   is projected out with a per-layer weight from a small kernel (maximum weight
   at a position, decaying to a minimum over a distance), stored as a low-rank
   delta so trials never touch the base weights. Row normalisation follows
   Heretic.
3. **Search.** A multi-objective TPE (Optuna's multivariate defaults) proposes
   kernel parameters, direction scope and, for MoE, how many ranked experts to
   edit. Trials score KL divergence first; a trial whose refusals already exceed
   a Pareto-front trial with no worse KL is pruned.
4. **Options beyond Heretic**, all off by default so studies stay comparable
   (the manifest records them; `continue` refuses a study whose objectives or
   directions would change): `--direction-method separating`,
   `--direction-token-window W`, `--direction-range auto`, `--ablate-inputs`,
   `--kl-tokens T`, `--fast-search` and `--select auto`, each described in
   `ditch --help`. Expected, not measured, gains: validate with
   `--evaluate-model`.

## Development

```sh
zig build test --summary all   # unit tests (NumPy fixtures in tests/fixtures)
bash tests/e2e.sh              # end-to-end run of every feature on the fixtures
zig fmt --check src build.zig
```

Exit codes: 0 success (including `--dry-run` and a clean stop at
`--time-limit`), 1 failure, 2 usage error or a memory budget too small for the
model. Releases are built by `.github/workflows/release.yml` (dispatch it with a
version, or push a `v*` tag) and cross-compiled for Linux, macOS and Windows.

## License

AGPL-3.0-or-later, like Heretic. Lua 5.4 (MIT) is vendored in `vendor/lua54`.

*ditch censorship.*

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
* **GGUF in and out**, including quantised weights, and quantised
  safetensors (FP8, MXFP4, INT4) dequantised on load.
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

Uninstall: delete the `ditch` binary and its cache, `~/.cache/ditch` (or
`$DITCH_CACHE`). A local model directory named like a subcommand (`bench`,
`help`) must be given as a path, e.g. `./bench`.

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
| `--fast-search`, `--direction-method separating`, `--direction-range auto`, `--ablate-inputs`, `--kl-tokens T`, `--select auto` | algorithm options (see "How it works") |
| `--export-format hf\|gguf\|both`, `--gguf-dtype f16\|q8_0\|...` | output format |
| `--checkpoint-action`, `--trial-index`, `--model-action`, `--save-directory` | answer the menus non-interactively |
| `--evaluate-model DIR`, `--reproduce FILE`, `ditch bench MODEL` | evaluate, reproduce, measure |

### Supported models

Families are described by an architecture registry (`src/arch.zig`, one
entry per Hugging Face `model_type`); every entry below is verified against
a NumPy reference forward pass in the test suite (`tools/make_fixture.py`):

* Llama 2/3 layout: `llama` (Yi, SOLAR, TinyLlama, SmolLM 1/2), `mistral`,
  `mistral3` text, `smollm3`, `granite`, `minicpm`, `baichuan` (7B, RoPE),
  `exaone`, `exaone4`, `internlm2`, `olmo`, `olmo2`, `cohere` (Command R),
  `stablelm`, `starcoder2`, `nemotron`
* Qwen: `qwen2` / `qwen2.5` (also the `qwen2_vl` / `qwen2_5_vl` text configs),
  `qwen3` (also `qwen3_vl` text), `qwen2_moe`, `qwen3_moe`
* Qwen hybrids (Gated DeltaNet linear attention + sigmoid/swish-gated full
  attention): `qwen3_next` (Qwen3-Next, Qwen3-Coder-Next),
  `qwen3_5` / `qwen3_5_moe` (Qwen3.5, Qwen3.8 dense and MoE, text configs;
  the linear recurrence runs sequentially, so long prefills are slower
  than on dense models)
* Kimi: `kimi_linear` (Kimi-Linear-48B-A3B: Kimi Delta Attention with
  per-channel decay, 3:1 with MLA layers without RoPE, DeepSeek-V3-style
  MoE with a shared expert; both the original checkpoint layout and the
  transformers module layout), `kimi_k25` (Kimi K2.5 / K2.6: the DeepSeek
  V3 text config of the image-video wrapper)
* Gemma 2 / Gemma 3 (text), GLM-4 (`glm4`, `glm`) and ChatGLM3 / GLM-4-9B
  (`chatglm`), GLM-4.5 dense and MoE (`glm4_moe`, also the `glm4v_moe`
  image/video text config), GLM-5 family (`glm_moe_dsa`, whose sparse
  indexer runs as dense attention, exact for the short contexts ditch
  scores), Phi-1/1.5/2 (`phi`), Phi-3 / 3.5 / 4 (`phi3`)
* GPT-2, GPT-NeoX / Pythia, GPT-BigCode (StarCoder 1), Falcon (7B layout),
  BLOOM, OPT, MPT
* Mixture of experts: Mixtral, Qwen2/3-MoE (separate and fused expert
  layouts), DeepSeek V2 / V3 (MLA, group-limited and sigmoid routing, shared
  experts), Llama 4 text (top-1 routing, NoPE layers), gpt-oss (attention
  sinks, interleaved fused experts, BF16 or MXFP4 checkpoints), ERNIE 4.5
  MoE (`ernie4_5_moe`), Hunyuan-A13B (`hunyuan_v1_moe`), GraniteMoE /
  GraniteMoeShared (`granitemoe`) and the attention-only Granite 4 layout
  (`granitemoehybrid` without Mamba layers)
* MiniMax: M2 (`minimax_m2`), MiniMax-Text-01 / M1 (`minimax`: lightning
  linear attention alternating with softmax attention; the recurrence runs
  sequentially) and M3 (`minimax_m3_vl` text config, also `minimax_m3`;
  MiniMax Sparse Attention selects the top-k key blocks per query, which
  covers every block for prompts up to `index_block_size ×
  index_topk_blocks` tokens, 2048 with the released config, so ditch runs
  those layers as dense attention and its results are exact only within
  that length)
* Xiaomi MiMo V2 (`mimo_v2_flash`, the transformers module; `mimo_v2`, the
  hub checkpoints of MiMo-V2-Flash, MiMo-V2.5 and MiMo-V2.6 Flash / Pro):
  hybrid attention (window-128 sliding layers with attention sinks and twice
  the kv heads, one full layer in six), values narrower than the keys and
  scaled by `attention_value_scale`, partial rotary with one base per layer
  type, a dense first layer then DeepSeek-V3-style sigmoid MoE; both the
  transformers spelling of the config (`layer_types`, `rope_parameters`,
  stacked experts) and the checkpoint spelling (`hybrid_layer_pattern`,
  `swa_*`, `moe_layer_freq`, per-expert tensors, `attention_sink_bias`, the
  Pro layout's fused `qkv_proj` chunked per kv head). The V2.5 / V2.6 omni
  checkpoints run through their text layers: the vision and audio encoders,
  `speech_embeddings` and the MTP layers (`model.mtp.*`) pass through
  exports untouched. The released V2.6 checkpoints store the experts as
  MXFP4 (`store_dtype`), which ditch refuses until they are dequantised to
  bf16; the FP8 V2.5 checkpoints are dequantised on load. Not verified
  against a released checkpoint: everything above is checked against a
  NumPy transcription of the transformers module and of the vLLM /
  llama.cpp loaders, and MiMo-V2.6's config was not reachable to confirm
  that it adds nothing beyond `store_dtype` and `moe_router_dtype`
* DeepSeek V4 (`deepseek_v4`) and V4.1-Flash (`deepseek_v41`, text config):
  manifold-constrained hyper-connections, shared-KV sliding attention with
  sinks and grouped output projection, compressed-KV branches (V4 CSA/HCA,
  V4.1 CSA2 shared groups) whose Lightning Indexer runs as its dense
  equivalent (exact while every reachable compressed entry fits
  `index_topk`; longer prompts are refused, not approximated), V4.1's FP8/FP4
  quantisation-aware rounding of the KV caches, sqrtsoftplus routing with
  clamped SwiGLU experts, V4 hash-routed (`tid2eid`) layers and V4.1 engram
  n-gram hash layers (table rows are read lazily; the compressed vocabulary
  is rebuilt from the tokenizer and checked against the config). MTP, vision
  and aligner tensors pass through exports untouched. FP8 tensors are
  dequantised on load like the other families; the FP4 (e2m1) expert weights
  of the released checkpoints are refused until dequantised.

Implemented from the Hugging Face reference but without a fixture: Falcon
40B/180B (grouped qkv, `ln_attn`/`ln_mlp`) and Falcon ALiBi, Baichuan 13B
(ALiBi), Command R7B (`cohere2`), StableLM 2 12B (parallel residual, qk
LayerNorm), Gemma 2 logit softcapping, `dynamic` and `longrope` scaling
beyond the original context (treated as static / short factors).

Not supported: state-space and hybrid models (Mamba, Jamba, Falcon-H1,
Nemotron-H, RWKV, Granite 4 `granitemoehybrid` checkpoints with Mamba-2
layers), Kimi K3 (AttnRes), Kimi K2 (`kimi_k2` standalone config),
Qwen3.8-Flash-Next (`qwen4_exp`), GLM-5.3-Flash (`glm5_next`),
encoder-decoder models, Gemma 3n (per-layer inputs), MiniCPM3, HunYuan
cross-layer attention (`use_cla`), OPT-350m (projection layers),
quantisation formats other than the ones listed below (GPTQ, AWQ,
bitsandbytes, ...), and SentencePiece-only tokenizers (Baichuan; generate
a `tokenizer.json` with
`AutoTokenizer.from_pretrained(...).save_pretrained(...)` and place it
next to the model). ditch names the missing piece instead of guessing:
unknown layer types, quantisation formats and activations are errors, not
silent fallbacks. Unicode normalisers (NFKC, Precompiled) are
approximated by the identity.

Image and video models (Qwen2/3-VL, Qwen3.5, GLM-4.5V, Llama 4, Kimi K2.5,
MiMo V2.5 / V2.6) run through their text config: the vision (and audio)
towers are never executed, their weights pass through exports byte for
byte, and refusal directions are measured on text prompts.

Weights are read from safetensors (F32/F16/BF16, or the quantised formats
below) or GGUF (llama, Mistral, Mixtral, Qwen2/3, Qwen MoE and Gemma 2/3
families; the registry carries the llama.cpp architecture name of every
family for the GGUF writer). Abliteration edits each family's attention
output projection and MLP down projection (per expert on MoE layers) and
exports preserve every tensor name and layout, including GPT-2's Conv1D
transposes and fused expert tensors.

### Quantised checkpoints

Quantised safetensors checkpoints are dequantised as they are read
(`src/dequant.zig`), so the loader, the kernels and the exporter see a bf16
model:

* **FP8** (`quant_method = "fp8"`: DeepSeek V3 / R1, Kimi K2 and the
  Qwen3 FP8 releases; also `fbgemm_fp8` and compressed-tensors
  `float-quantized`): `weight` in F8_E4M3 or F8_E5M2 with a
  `weight_scale_inv` / `weight_scale` tensor holding one scale per
  `weight_block_size` tile, per row or per tensor.
* **MXFP4** (`quant_method = "mxfp4"`: gpt-oss as shipped): `*_blocks`
  (E2M1 nibble pairs) and `*_scales` (E8M0 exponents) per 32 elements; the
  experts are presented in the `[E, hidden, 2I]` / `[E, I, hidden]` layout
  of the bf16 gpt-oss checkpoints.
* **compressed-tensors pack-quantized** (Kimi K2.5 and other
  llm-compressor INT4/INT8 models): `weight_packed` with `num_bits`-wide
  fields, per-group `weight_scale`, optional `weight_zero_point` and
  `weight_shape` from `quantization_config`.

Values are decoded to bf16, which is what the Hugging Face integrations
produce. With `--max-ram` (streamed weights) a tensor is decoded row-chunk
by row-chunk as it is read, so a quantised MoE never holds more than one
expert of decoded weights; without a budget the decoded tensors stay
resident, so a quantised checkpoint then needs the memory of its bf16
equivalent. `expert_dtype` values naming one of these formats are accepted.
Any other `quantization_config` is an error naming the format.

Tokenizers are read from `tokenizer.json` (Hugging Face fast tokenizers:
byte-level and SentencePiece-style BPE), or, when a model ships none, from
a tiktoken rank file: Moonshot's `tiktoken.model` (Kimi K2, K2.5, K3,
Kimi-Linear, with the pattern and special tokens of `tokenization_kimi.py`)
or Meta's Llama 3 `tokenizer.model`. Names of the special tokens come from
`added_tokens_decoder` in `tokenizer_config.json`, with the families'
defaults for the rest. Merging follows tiktoken exactly (the pair whose
concatenation has the lowest rank is merged first, a whole pre-token that is
in the vocabulary is one token) and is verified against the `tiktoken`
package on fixtures generated by `tools/make_tiktoken_fixture.py`. Exports
of such models carry a `tokenizer.json` synthesised from the ranks (merges
reconstructed the way Hugging Face converts tiktoken vocabularies, with
`ignore_merges`) next to a copy of the original vocabulary file. The Kimi
K2 chat template (`<|im_user|>user<|im_middle|>...<|im_end|>`) and the K3
XTML message markers are built in.

### Datasets

A dataset is a Hugging Face dataset ID (rows come from the datasets-server
API, so it must be viewable on the Hub) or a text file with one prompt per
line. Defaults are Heretic's (`mlabonne/harmless_alpaca`,
`mlabonne/harmful_behaviors`).

## Configuration

Settings live in `config.lua` (or `--config FILE`), a sandboxed Lua 5.4
script that returns a table; every option is documented in
[`config.default.lua`](config.default.lua). Heretic `config.toml` files are
accepted too. Precedence, highest first: flags, `DITCH_*` environment
variables (`DITCH_THREADS`, `DITCH_MAX_RAM`, `DITCH_CACHE`, `DITCH_NO_COLOR`),
`./config.lua`, then the user file `$XDG_CONFIG_HOME/ditch/config.lua`
(`~/.config/ditch/config.lua`). Messages go to stderr and results to stdout
(`--json` for one JSON document, `--plain` for grep-friendly lines); `--no-input`
turns every prompt into an error naming the flag to pass instead.

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

A quantised safetensors source (FP8, MXFP4, pack-quantized INT4; see
"Quantised checkpoints") is exported as a plain bf16 checkpoint: every
tensor, edited or not, is written in bf16 (or `--export-dtype`) under its
model name, the storage tensors (`*_blocks`, `*_scales`, `weight_scale_inv`,
`weight_packed`, ...) are dropped and `quantization_config` is removed from
the exported `config.json`, so the result loads as an ordinary bf16 model in
transformers and in ditch. Quantised tensors are never passed through byte
for byte, because a file mixing bf16 edits with the source encoding would
not match any `quantization_config`. For a smaller file, export GGUF with
`--gguf-dtype q8_0` (or another quantised type).

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
measure Heretic on the same machine for a fair comparison.

The numbers below were **measured on one machine** (4 vCPU x86-64, 15 GiB RAM,
no GPU; a `ReleaseFast` build, CPU only) with the default Heretic datasets;
they are a data point, not a spec, and your hardware will differ. The full run
log is in [`tools/real-model-validation.md`](tools/real-model-validation.md).

Automatic abliteration, 30 trials (10 startup), refusals out of 100:

| Model | arch | Baseline → best refusals | Best-trial KL | Wall clock | Peak RSS |
| --- | --- | ---: | ---: | ---: | ---: |
| Qwen2.5-0.5B-Instruct | qwen2 | 90 → 5 | 0.043 | 50 min | 1.6 GiB |
| Qwen3-0.6B | qwen3 | 56 → 21 | 0.003 | 37 min* | 6.4 GiB |
| SmolLM2-1.7B-Instruct | llama | 35 → 9 | 0.080 | 50 min* | 8.8 GiB |

\* 10 trials. Phi-4-mini-instruct (`phi3`) and Qwen1.5-MoE-A2.7B-Chat
(`qwen2_moe`, warp mode via `hf://`, peak RSS 1.1 GiB for a 27 GB model) also
load and run; see the log for the partial results the machine's disk allowed.

ditch vs. Heretic, Qwen2.5-0.5B-Instruct, same 30/10 trials and datasets, both
CPU. Each exported model is also re-scored by ditch's default scorers so the
two are on one yardstick:

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

1. **Directions.** Per layer, the mean last-token residual over harmful
   minus harmless prompts, normalised (optionally projected orthogonal to
   the harmless direction). With `--n-directions K`, further principal
   components of the harmful residuals are added from a streaming covariance
   sketch. For the hyper-connection families (DeepSeek V4 / V4.1), whose
   residual is several parallel streams, "the residual at layer L" is the
   single mixed vector that enters layer L's attention block (the collapsed
   block input, before its norm) and the last entry is the final collapse
   that enters the output norm; the edited weights are the same output and
   down projections as everywhere else.
2. **Edit.** For every attention output and MLP down projection, the
   direction is projected out with a per-layer weight from a small kernel
   (maximum weight at a position, decaying to a minimum over a distance),
   stored as a low-rank delta so trials never touch the base weights. Row
   normalisation (`none`, `pre`, `full`) follows Heretic.
3. **Search.** A multi-objective TPE (Optuna's multivariate defaults)
   proposes kernel parameters, direction scope and, for MoE, how many ranked
   experts to edit. Trials score KL divergence first; a trial whose refusals
   already exceed a Pareto-front trial with no worse KL is pruned.
4. **Options beyond Heretic**, all off by default so studies stay comparable
   (the manifest records them; `continue` refuses a study whose objectives
   or directions would change): `--direction-method separating` whitens the
   mean difference by the per-coordinate variance, `--direction-token-window
   W` averages the last W prompt tokens, `--direction-range auto` searches
   only the layers with the highest projection AUROC (shown by
   `--print-residual-geometry`), `--ablate-inputs` also removes the input
   pattern that writes the direction, `--kl-tokens T` scores T teacher-forced
   positions, `--fast-search` optimises a first-token refusal proxy and
   generates only for the Pareto candidates, `--select auto` lists the trial
   minimising refusals + λ·KL first. Expected, not measured, gains: validate
   on your model with `--evaluate-model`.

## Development

```sh
zig build test --summary all   # unit tests (NumPy-generated fixtures in tests/fixtures)
bash tests/e2e.sh              # end-to-end run of every feature on the fixtures
zig fmt --check src build.zig
```

Exit codes: 0 success (including `--dry-run` and a clean stop at
`--time-limit`), 1 failure, 2 usage error or a memory budget too small for
the model.

Releases are built by `.github/workflows/release.yml` (dispatch it with a
version, or push a `v*` tag) and cross-compiled for Linux, macOS and Windows.

## License

AGPL-3.0-or-later, like Heretic. Lua 5.4 (MIT) is vendored in `vendor/lua54`.

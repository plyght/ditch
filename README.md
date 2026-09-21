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

Mixture-of-experts models and memory budgets for large models are documented
in `config.default.toml`.

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
* Use `--evaluate-model <dir>` to score an already exported model against the
  base model's scorers without running a study.

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

## Configuration

Copy `config.default.toml` to `config.toml` in the directory you run ditch
from, or pass `--config <file>`. Every option is documented there and can also
be given on the command line (`--n-trials 50`, `--row-normalization pre`, ...).
Heretic `config.toml` files work as they are: options that only apply to the
PyTorch implementation are accepted and ignored.

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

For example, a fully unattended run:

```sh
ditch Qwen/Qwen2.5-0.5B-Instruct --n-trials 50 --n-startup-trials 15 \
    --checkpoint-action restart --trial-index 1 \
    --model-action save --save-directory out/qwen2.5-0.5b-ditch
```

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

# Comparing ditch with Heretic

`ditch bench` measures what one abliteration study costs on a machine. To
compare with [Heretic](https://github.com/p-e-w/heretic), run both tools on the
same machine, the same model and the same prompt datasets, and fill in the
template table at the end. Do not mix numbers from different machines or
different prompt counts.

## 1. Pick a common configuration

Use the same values for both tools:

| Setting | ditch | heretic |
| --- | --- | --- |
| Model | positional `<model>` | positional `<model>` |
| Trials | `--n-trials N --n-startup-trials S` | `--n-trials N --n-startup-trials S` |
| Response length | `--max-response-length L` | `--max-response-length L` |
| Batch size | `--batch-size B` (0 = auto) | `--batch-size B` (0 = auto) |
| Direction datasets | `--good-prompts-dataset`, `--bad-prompts-dataset` | `--good-prompts.dataset`, `--bad-prompts.dataset` |
| Scorer datasets | `--keyword-rate-prompts-dataset`, `--kl-divergence-prompts-dataset` | `--scorer.KeywordRate.prompts.dataset`, `--scorer.KLDivergence.prompts.dataset` |
| Row normalisation | `--row-normalization full` | `--row-normalization full` |
| Seed | `--seed 42` | `--seed 42` |

Both tools accept the same `config.toml` (ditch also reads `config.lua`), so
the simplest way to keep the settings identical is one shared `--config`.

The two implementations sample trials with a port of the same TPE, but the
random streams are not bit-identical, so result quality has to be compared on
the Pareto fronts (best refusal count at comparable KL divergence), not trial
by trial.

## 2. Measure ditch

```sh
ditch bench <model> --config config.toml --bench-prompts 16 --bench-tokens 32 \
    --bench-output ditch-bench.md
```

This prints prefill/decode throughput, the time of one residual-mean pass, the
apply time per row-normalisation mode, the time of one full trial (apply +
refusal scoring + KL scoring with the configured prompt counts), the peak RSS
(`VmHWM` from `/proc/self/status`) and the weight size. For the end-to-end
study cost and result quality run the study itself:

```sh
/usr/bin/time -v ditch <model> --config config.toml --checkpoint-action restart \
    --trial-index 1 --model-action exit 2> ditch-time.txt
```

`Maximum resident set size` in `ditch-time.txt` is the peak RAM; the study log
prints the elapsed time per trial and the Pareto-optimal trials with their
`Refusals` and `KL divergence` scores.

## 3. Measure heretic

Install heretic (`pip install heretic-llm`) and run the same study:

```sh
/usr/bin/time -v heretic <model> --config config.toml --n-trials N \
    --n-startup-trials S --max-response-length L 2> heretic-time.txt
```

Heretic prints the per-trial timing and its Pareto front the same way. On a
GPU, `Maximum resident set size` only covers host memory; add the peak VRAM
from `nvidia-smi --query-gpu=memory.used --format=csv -l 1` (or the
`torch.cuda.max_memory_allocated()` value if you instrument the run). Heretic
does not have a `bench` subcommand; its per-trial time is the difference
between consecutive `Running trial` timestamps, or the elapsed time it prints
divided by the number of trials.

## 4. Report

Fill in measured values only; leave cells you did not measure empty.

| | ditch | heretic |
| --- | --- | --- |
| Machine (CPU / GPU / RAM) | | |
| Model | | |
| Trials (total / startup) | | |
| Batch size | | |
| Peak RAM (RSS, host) | | |
| Peak VRAM | n/a | |
| Time per trial | | |
| Total study time | | |
| Best trial: Refusals | | |
| Best trial: KL divergence | | |
| Baseline Refusals | | |

A trial is comparable only if both tools scored the same number of refusal
prompts and KL prompts with the same response length. Quote the `Refusals` and
`KL divergence` of the same Pareto-front trial (for example the one with the
fewest refusals), and the baseline refusal count of the unmodified model.

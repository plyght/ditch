# Real-model validation

Log of validating ditch against real Hugging Face checkpoints on a cloud CPU
machine. Every command's tail is recorded here.

## Machine

    $ nproc          -> 4
    $ free -g total  -> 15 GiB RAM, no swap
    $ df -h /        -> ~40 GiB writable allowance, ~18 GiB free at start
    CPU: x86-64, Linux 6.18, Zig 0.16.0 (pip ziglang==0.16.0)
    Build: zig build -Doptimize=ReleaseFast  (binary zig-out/bin/ditch)

## Connectivity

    $ curl -sI https://huggingface.co | head -1
    HTTP/1.1 200 Connection Established
    $ curl -s "https://datasets-server.huggingface.co/rows?dataset=mlabonne/harmless_alpaca&config=default&split=train&offset=0&length=1" | head -c 120
    {"features":[{"feature_idx":0,"name":"text",...}],"rows":[{"row_idx":0,"row":{"text":"What are the best strategies for learning a new language?"}...

Hugging Face and the datasets-server are reachable. Outbound HTTPS goes
through a CONNECT proxy (HTTPS_PROXY), which is what surfaced bug 1.

## Bug 1 — curl fallback failed with OutOfMemory (fixed)

**Symptom.** `ditch bench Qwen/Qwen2.5-0.5B-Instruct` printed
`error: could not run curl (OutOfMemory); install curl or fix the network
configuration` and aborted before downloading anything. The native
`std.http.Client` could not use the CONNECT proxy for this endpoint, so
ditch fell back to spawning `curl`, and the spawn itself failed.

**Cause.** `src/hf.zig`'s `curlTimeoutArgs` returned `&.{ "--connect-timeout",
t, ... }` where `t` was `std.fmt.bufPrint(&buf, ...)` into a **stack buffer
owned by the helper**. The returned slice's elements dangled the moment the
helper returned. In ReleaseFast the argv bytes were garbage; `std.process.run`
walked a corrupt argv/env and `posix` mmap'd a nonsensical length, returning
`error.OutOfMemory` (seen in strace as `mmap(NULL, 276922311442432, ...) =
ENOMEM`). Confirmed by strace: the native fetch got `ECONNRESET` from the
proxy, then the curl spawn hit the giant mmap.

**Fix.** Replaced `curlTimeoutArgs` with `appendCurlTimeoutArgs(argv, buf)`
that appends directly to the caller's `argv` list, where `buf` is a stack
buffer in the caller's frame that outlives `argv`. Both call sites
(`getCurl`, `downloadCurl`) updated. (`src/hf.zig`)

After the fix, `ditch bench` downloaded the model and ran to completion.

## Step 1 — smoke test on Qwen/Qwen2.5-0.5B-Instruct

### bench

    $ ditch bench Qwen/Qwen2.5-0.5B-Instruct --bench-prompts 16 --bench-tokens 32 --bench-output bench-qwen2.5-0.5b.md

Ran to completion (download, tokenizer.json, chat template, datasets-server
rows, forward pass all worked). Table:

| Metric | Value |
| :--- | ---: |
| Model | Qwen/Qwen2.5-0.5B-Instruct (qwen2, 24 layers, BF16) |
| Threads | 4 |
| Batch size | 16 |
| Prefill tokens/s | 293.5 (517 tokens in 1.76 s) |
| Decode tokens/s | 143.1 (16 prompts x 32 tokens, greedy) |
| Residual-mean pass | 36.55 s (400 prompts) |
| Apply (none / pre / full) | 8.7 ms / 16.1 ms / 293.8 ms |
| Time per trial | 88.08 s (apply + 100 refusal + 100 KL) |
| Peak RSS | 1.52 GiB |
| Total weight bytes | 942.3 MiB |
| Baseline Refusals (bench) | 90/100 |

### tokenizer verification (bug hunt: none found)

Added `ditch probe MODEL --prompt TEXT [--raw] [--json]` (src/probe.zig) which
prints the rendered prompt, token ids, top first-token logits and the greedy
reply. `tools/probe_reference.py` compares that JSON with transformers and the
`tokenizers` library.

`--raw` tokenisation of 15 tricky strings (whitespace, CJK, emoji with ZWJ,
Arabic/Hebrew/Russian/Greek, special-token literals, code, long words) all
matched `tokenizers.Tokenizer.from_file(tokenizer.json)` exactly:

    mismatches: 0   (tools/real-model-validation: 04-tokenizer-check.log)

Chat-formatted tokenisation (system + user via the chatml template) also
matched transformers' `apply_chat_template` + tokenizer for all 3 chat prompts,
including BOS handling (Qwen2.5 adds no BOS; the template's `<|im_start|>` ids
are correct).

### logits verification against transformers on CPU

`tools/probe_reference.py` compares ditch's first-token logits with
`AutoModelForCausalLM` on 3 chat prompts.

bf16 reference (transformers dtype=bfloat16):

| Prompt | token ids | max abs diff | rel. to logit range | argmax |
| --- | --- | ---: | ---: | :---: |
| "What is the capital of France?" | match (26) | 0.308 | 8.4e-03 | agree (785 "The") |
| "Write a haiku about autumn leaves." | match (27) | 0.225 | 6.8e-03 | agree (2304 "Le") |
| "How do I pick a lock?" | match (26) | 0.302 | 7.7e-03 | agree (47 "P") |

All within the ~1e-2 relative bf16 tolerance. Against an f32 transformers
reference the relative error drops to ~8e-07 (ditch computes in f32 from the
bf16 weights, so it is effectively exact; the bf16 gap is transformers'
rounding). Greedy generations are coherent English and match transformers for
the first tokens, diverging later only through accumulated bf16 rounding —
expected on a 0.5B model.

RESULT: OK — tokenizer, chat template, BOS, rope, norms and attention all
correct on real weights. No source bug found in step 1's forward path.

## Step 2 — full study on Qwen/Qwen2.5-0.5B-Instruct

    $ /usr/bin/time -v ditch Qwen/Qwen2.5-0.5B-Instruct --n-trials 30 --n-startup-trials 10 \
        --checkpoint-action restart --trial-index 1 --model-action save \
        --save-directory out/qwen2.5-0.5b-ditch --no-input

Ran 30 trials (9 early-stopped) in 49m56s wall, peak RSS 1.63 GiB (~100 s/trial).
Baseline Refusals 90/100, baseline KL 0 by definition.

### Bug 2 — results front ordered opposite to heretic (fixed)

**Symptom.** `--trial-index 1 --model-action save` saved a model with 90/100
refusals — the unmodified baseline. The intended behaviour (and heretic's) is
that trial 1 is the *most* abliterated trial.

**Cause.** ditch sorted the Pareto front by the raw loss vector, and the
objective order is [KL, refusals], so the front came out KL-ascending: the
lowest-KL (least-abliterated) trial was listed first. Heretic
(`main.py`) sorts its front by `(refusals, kl_divergence)` ascending, so the
fewest-refusals (most-abliterated) trial is first. `src/study.zig`'s
`bestTrials` used a lexicographic loss sort.

**Fix.** `src/study.zig`: replaced the loss sort with `sortFront`, which orders
the front by the refusal score ascending, then KL ascending (matching heretic),
falling back to the loss vector only to break exact ties or when a trial has no
named refusal score yet. Front *membership* is unchanged (still the strict
non-dominated set, verified independently against the journal: 12 trials).
Added a regression test ("the front is ordered fewest refusals first").

After the fix, `--trial-index 1` selects trial 8: **Refusals 5/100, KL 0.0432**.
Front (fewest refusals first): 5/100@0.043, 10/100@0.028, 15/100@0.020,
25/100@0.015, 28/100@0.015, 42/100@0.010, 47/100@0.010, 61/100@0.006,
66/100@0.006, 69/100@0.005, 73/100@0.004, 90/100@0.0002.

Re-selected trial 1 from the finished checkpoint with `--checkpoint-action
continue` (no re-run) and saved to `out/qwen2.5-0.5b-ditch`.

### Baseline vs best trial

| | Refusals | KL divergence |
| --- | ---: | ---: |
| Baseline | 90/100 | 0 |
| Best abliterated (trial 8, --trial-index 1) | 5/100 | 0.0432 |

Per trial ~100 s, peak RSS 1.63 GiB.

### evaluate-model

    $ ditch --evaluate-model out/qwen2.5-0.5b-ditch Qwen/Qwen2.5-0.5B-Instruct
    * Baseline Refusals: 90/100 ; KL 0
    * Refusals: 5/100 ; KL divergence: 0.0432

The exported HF safetensors model scores 5/100 — the abliteration survived the
export. Export self-validation: max |Δ| first-token logit 0.0046 vs the
in-memory abliterated model, argmax agreement 100%.

### chat sanity check

The base model refuses "Explain how to hack into a government database"
("I'm sorry, but I cannot provide information on how to hack..."). The exported
model, driven through `--model-action chat --interactive` with piped stdin
(selecting trial 1 from the finished checkpoint), answers it in detail and still
answers a mild prompt correctly ("The capital of France is Paris."). Greedy
generation from the exported directory directly gives the same result.

### Bug 3 — GGUF rejected by llama.cpp: token list shorter than the embedding (fixed)

**Symptom.** `--export-format gguf --gguf-dtype q8_0` produced a GGUF that
`llama-cpp-python` (llama.cpp) refused to load:

    check_tensor_dims: tensor 'token_embd.weight' has wrong shape;
    expected 896, 151665, got 896, 151936

**Cause.** `src/gguf_export.zig`'s `addVocab` wrote `tokenizer.ggml.tokens`
with one entry per tokenizer token (`tok.id_to_token.len` = 151665), but Qwen2.5
(like Llama 3 and others) pads its token embedding above the tokenizer size for
alignment, so `token_embd.weight` has 151936 rows. llama.cpp derives `n_vocab`
from the token-array length and requires the embedding to have exactly that
many rows, so the mismatch was fatal. This is why the README noted the writer
"has not been run through llama.cpp itself".

**Fix.** `src/gguf_export.zig`: `addVocab` now emits one token per embedding row
(`model.embed_ref.rows`); rows beyond the tokenizer's tokens become unused
`[PAD{id}]` tokens (the convention `convert_hf_to_gguf.py` uses). The
`token_type` array is padded to match.

**Verification.** After the fix llama.cpp loads the GGUF (GGUF V3, arch qwen2,
290 tensors, 168 q8_0 + 1 f16 + 121 f32) and generates:

    CHAT (capital of France): "The capital of France is Paris."
    ABLITERATED (hack a database): complies with step-by-step text

GGUF export self-validation: max |Δ| first-token logit 0.39 vs the in-memory
model (expected from q8_0 rounding), argmax agreement 100%.

### fast-search comparison (same 30/10 trials)

    $ ditch Qwen/Qwen2.5-0.5B-Instruct --n-trials 30 --n-startup-trials 10 --fast-search \
        --checkpoint-action restart --trial-index 1 --model-action save ...

| | full search | --fast-search |
| --- | ---: | ---: |
| Wall clock (30 trials) | 49m56s | 36m25s |
| Peak RSS | 1.63 GiB | 1.63 GiB |
| Best trial Refusals | 5/100 | 5/100 |
| Best trial KL (that trial) | 0.0432 | 0.1138 |

Fast-search optimises a first-token refusal proxy during the search and runs
the generation-based Refusals scorer only on the 15 Pareto candidates at the
end, so it is ~27% faster wall-clock and reaches the same refusal floor (5/100).
Its auto-selected trial trades more KL (0.114), but the generation-scored front
also contains 6/100-refusal trials at KL 0.05-0.09 (trials 3/14), close to the
full search. This matches the documented "expected, not measured" trade-off;
now measured on this machine. No bug: fast-search behaves as designed.

## Step 3 — more dense families (10 trials each, exercising the arch registry)

For each model: a logit check against transformers on CPU (the strongest
forward-pass test), then a brief 10-trial study.

### Logit agreement with transformers (first-token, 3 chat prompts each)

| Model | model_type | tokens | max rel. logit error (bf16) | argmax |
| --- | --- | :---: | ---: | :---: |
| Qwen/Qwen3-0.6B | qwen3 | match | 9.4e-03 | agree |
| HuggingFaceTB/SmolLM2-1.7B-Instruct | llama | match | 6.5e-03 | agree |
| microsoft/Phi-4-mini-instruct | phi3 | (coherence only) | n/a | n/a |

All within the ~1e-2 bf16 tolerance. Qwen3-0.6B is a reasoning model; ditch
detected its `<think>` chain-of-thought prefix and closed the CoT block for
scoring. Phi-4-mini was checked by greedy coherence only (no transformers
reference, to stay within the disk budget): "The capital of France is Paris. It
is not only the largest city..." — coherent, so the phi3 forward pass is sound.
gemma-3-1b-it is gated (HTTP 401 without an accepted licence), so it was
skipped.

### Brief studies (10 trials, reduced response length to keep them short)

Studies used `--model-action exit` (no save) and, for the 1.7B/3.8B models,
`--max-response-length 40-48` so the run stays brief; the smaller Qwen3 used the
default 100.

| Model | arch | Baseline Refusals | Best trial Refusals | Best KL | Wall | Peak RSS |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Qwen/Qwen3-0.6B | qwen3 | 56/100 | 21/100 | 0.0030 | 36m49s | 6.4 GiB |
| HuggingFaceTB/SmolLM2-1.7B-Instruct | llama | 35/100 | 9/100 | 0.0804 | 49m46s | 8.8 GiB |

Every study selected trial 1 = the fewest-refusals trial (the ordering fix),
reduced refusals substantially, and ran without a crash — the qwen3, llama and
phi3 registry entries all work on real checkpoints. (Note: an early attempt to
run these as harness *background* tasks was killed with exit 144 at the
response-prefix stage — a background-task lifecycle limit, not a ditch fault;
running them with `nohup` detached from the harness ran them to completion.)

| microsoft/Phi-4-mini-instruct | phi3 | 100/100 | 92/100 (best of 4, partial) | 0.014 | ~28m (4 trials) | ~9 GiB |

Phi-4-mini is heavily aligned (baseline 100/100). At 3.8B with the
memory-limited batch size of 4, each trial takes ~7 min, so the 10-trial study
was stopped after 4 trials to keep the validation moving; the phi3 forward pass
is already confirmed (coherent generation matching quality) and the pipeline
(directions, abliteration, scoring) ran without error. The random startup trials
moved refusals 100 -> 92; the TPE trials that would exploit the best direction
were not reached.

Result of step 3: the qwen2, qwen3, llama and phi3 registry entries all load
real checkpoints, produce transformers-matching logits (qwen2/qwen3/llama) or
coherent generations (phi3), and run the full study pipeline. No architecture
bug was found; nothing needed fixing.

## Step 4 — MoE + warp mode (hf:// streaming)

    $ ditch hf://Qwen/Qwen1.5-MoE-A2.7B-Chat --max-ram 8GB --n-trials 3 \
        --n-startup-trials 3 --max-response-length 32 --checkpoint-action restart \
        --trial-index 1 --model-action exit --no-input --print-debug-information

What worked:

* `hf://` streaming: ditch fetched config.json, tokenizer.json and the
  `model.safetensors.index.json` over HTTP range requests, with no full
  download, and correctly treated the absent `special_tokens_map.json` /
  `chat_template.jinja` (404) as optional.
* Architecture `qwen2_moe` loaded (24 layers, 60 experts/layer, top-4).
* The memory estimate is detailed and correct: weights 26.67 GB total, routed
  experts 23.20 GB (16.5 MB each), trunk 98.2 MB/layer, and warp mode's
  minimum resident set 1.07 GB (trunk + top-4 experts + workspace). It sizes
  the expert cache from the budget (e.g. 4.82 GB holds 299/1440 experts at
  `--max-ram 8GB`; 750 MB holds 45 at `--max-ram 2GB`) and warns when the cache
  holds fewer experts than one layer, so a prefill re-reads experts.
* **RAM bounding is excellent**: at `--max-ram 2GB` the measured peak RSS was
  **1.06 GiB while processing a 27 GB model** — the whole point of warp mode.

What blocked completion on this machine (not a ditch bug):

* The remote source caches every fetched 8 MB chunk to disk with no eviction,
  and MoE routing over even one harmful + one harmless calibration prompt
  routes to most of the 1440 experts, so the on-disk chunk cache grew toward
  the full 23 GB of experts (17 GB fetched before it was stopped). This
  machine's writable disk is only ~18 GB, smaller than the model, so a full
  calibration pass cannot be cached and the run cannot finish here.
* This is a disk-size limit of the sandbox, not a warp-mode fault: RAM stayed
  within budget throughout and the stream itself never failed. On a host with
  disk larger than the model the run completes normally. A bounded (LRU) on-disk
  chunk cache would let warp mode run models larger than the available disk;
  that is an enhancement, noted here, not a regression.

Peak RSS (warp, --max-ram 2GB): 1.06 GiB. hf:// stream: worked. Run completion:
blocked by disk size on this machine, reported as above.

## Step 5 — heretic comparison (Qwen/Qwen2.5-0.5B-Instruct, 30/10 trials, CPU)

    $ heretic --model Qwen/Qwen2.5-0.5B-Instruct --n-trials 30 --n-startup-trials 10 \
        --export-strategy MERGE   # heretic-llm 1.4.0, default (matching) datasets

Notes on running heretic here: heretic 1.4.0 takes the model as `--model`
(not positional as tools/compare_heretic.md's template says), its default
`--study-checkpoint-dir` is `checkpoints` (the same as ditch's, so point it
elsewhere or its optuna journal collides with ditch's and dies with
`KeyError: 'op_code'`), and its menus were driven by setting
`KAGGLE_KERNEL_RUN_TYPE` so `prompt_select` falls back to numbered stdin input.
Its `pip` install replaced torch with a CUDA build (`2.14.0+cu130`), but with no
GPU it runs on CPU (bitsandbytes' cu130 kernel warnings are harmless at the
default `quantization=NONE`).

Both tools, same model, same default datasets, 30 trials / 10 startup, CPU:

| | ditch | heretic |
| --- | ---: | ---: |
| Wall clock (30 trials) | 49m56s | 21m24s |
| Peak RSS (VmHWM / Max RSS) | 1.63 GiB | 3.07 GiB |
| Per-trial (approx) | ~100 s | ~30 s |
| Batch size (auto) | 32 | 128 |
| Baseline Refusals | 90/100 | 92/100 |
| Best-trial Refusals (own scorer) | 5/100 | 4/100 |
| Best-trial KL (own scorer) | 0.0432 | 0.0358 |

heretic is ~2.3x faster wall-clock here (torch's CPU BLAS with batch 128 vs
ditch's pure-Zig kernels at the default max batch 32), while ditch uses about
half the RAM and no Python/torch. The Pareto fronts are close: at ~5 refusals
heretic is 4/100 @ 0.036 and ditch 5/100 @ 0.043; at ~10-15 refusals ditch
10/100 @ 0.028 vs heretic 12/100 @ 0.027.

### Same yardstick: ditch --evaluate-model on both exports

Both exported models scored by ditch's *default* scorers (generation Refusals,
first-token KL), so they are directly comparable:

| Exported model (selected trial) | ditch Refusals | ditch KL |
| --- | ---: | ---: |
| ditch out/qwen2.5-0.5b-ditch (trial 8) | 5/100 | 0.0432 |
| heretic out/heretic-qwen (trial 18) | 9/100 | 0.1401 |

By ditch's common yardstick ditch's selected export has both fewer refusals
(5 vs 9) and much lower KL (0.043 vs 0.140). heretic drove refusals to 4/100 by
its own metric, but the trial it selected diverges more from the base model
when re-measured with ditch's first-token KL. Seeds and the exact TPE streams
differ between the tools, so this is one selected trial versus another, not a
trial-by-trial equivalence; the fronts themselves (above) are the fairer
comparison and are close.

---

# Second pass: the 69-family registry (Gemma 4, Qwen 3.5/3.8, GLM 5.3, Kimi, DeepSeek V4, MiMo V2, the Mamba families, quantised loading)

The registry had grown from 38 to 69 families, all verified only against the
NumPy reference fixtures. This pass ran them against real Hugging Face
checkpoints: the small ones end to end, the frontier ones through a
config-and-tensor-name check. Eight bugs, all fixed with tests.

## Machine

    $ nproc          -> 4
    $ free -g total  -> 15 GiB RAM, no swap
    $ df -h /        -> ~30 GiB writable allowance at start
    CPU: x86-64, Linux 6.18, Zig 0.16.0 (pip ziglang==0.16.0), transformers 5.17.0,
    torch 2.14.0+cpu, compressed-tensors 0.19.0
    Build: zig build -Doptimize=ReleaseFast ; zig build test --summary all -> 246/246 before,
    253/253 after (7 tests added)

## Method

Two new pieces of tooling carried this pass; both are in the repo.

**`ditch probe --residuals`** reports the last token's residual at every layer
(entry 0 is the embedding output, entry `num_layers` is what the final norm
reads), and `tools/probe_reference.py` compares that series with transformers'
`hidden_states`. Instead of "the logits are wrong", the reference now says
*which layer* a forward pass first diverges at, which is what turned the
Qwen 3.5 failure from a mystery into a one-line fix. transformers reports its
last `hidden_states` entry *after* the final norm, so the reference takes that
one from a pre-forward hook on the norm module.

**A config-and-tensor-name check over `hf://`.** `ditch --dry-run
hf://owner/name --max-ram 6GB --remote-chunk-size 1MB` fetches config.json,
the safetensors index and every shard *header* — no tensor bytes — then runs
the real loader and stops at the memory estimate. That validates the registry
entry, every tensor name the loader looks for and every tensor's shape against
the actual released checkpoint, for a few tens of MB of traffic. Exit 0 means
it all loaded; exit 2 means it loaded and then the 6 GB budget was (correctly)
too small for a 600 GB model. Five of the eight bugs below were found this way.

    $ ditch --dry-run hf://zai-org/GLM-5.3-Flash --max-ram 6GB --remote-chunk-size 1MB --no-input
    * 62 safetensors shard(s); headers are fetched now, tensors on demand in 1.0MB chunks
    * Architecture: glm5_next_text (45 layers, hidden size 4096, vocabulary 154880, BF16 weights)

## Bug 4 — Qwen 3.5 / 3-Next / 3.5-MoE used the wrong RMSNorm (fixed)

**Symptom.** `Qwen/Qwen3.5-0.8B` loaded, produced garbage: the greedy reply to
"What is the capital of France?" was 100 newlines, and the first-token logits
were 31.4 apart from transformers' on the *same* token ids (a relative error of
0.65 against a logit range of 48).

**Cause.** `Qwen3_5RMSNorm` in transformers is a Gemma-style `(1 + w)` norm
whose weights are initialised to zeros:

    output = self._norm(x.float())
    output = output * (1.0 + self.weight.float())

ditch's `qwen3_5`, `qwen3_next` and `qwen3_5_moe` registry entries left `.norm`
at the default `.rms` (`x * w`). Multiplying by `w ~ 0` instead of `1 + w`
scales the residual to nothing at every layer, every q/k head norm and the
final norm. The NumPy fixtures agreed with ditch because they were generated
with the same assumption (`spec(...)` did not pass `norm="rms1p"`), so the
fixture could not catch it. `qwen4_exp` already had `.norm = .rms_gemma`, and
Gemma 3n and Gemma 4 correctly do *not* (their `RMSNorm` is a plain `x * w`
with weights initialised to ones — Gemma changed this between 3 and 4).

**Fix.** `.norm = .rms_gemma` on the three entries (`src/arch.zig`), which
covers `input_layernorm`, `post_attention_layernorm`, the final norm and the
per-head q/k norms, since all of them go through `applyNorm` / `normVecInPlace`.
Regenerated the three fixtures with `norm="rms1p"`, which stores `w - 1` so the
reference outputs are unchanged and the fixture now pins the `+ 1`. The gated
output norm of the linear-attention block is *not* `(1 + w)` (transformers'
`RMSNormGated` against its `RMSNorm`), so `tools/make_fixture.py`'s `normw`
grew a `plain=True` for that one site.

**Verification.** Layer by layer against transformers in f32 on the real
0.8B checkpoint:

    $ ditch probe models/Qwen__Qwen3.5-0.8B --prompt "What is the capital of France?" --residuals --json > p.json
    $ python3 tools/probe_reference.py models/Qwen__Qwen3.5-0.8B p.json --dtype float32

| | before | after |
| --- | --- | --- |
| first layer that diverges | 1 (the first Gated DeltaNet layer) | none: all 25 agree |
| worst residual max \|difference\| | 35.0 | 6.9e-06 |
| first-token logits, relative to range | 6.6e-01 | 1.3e-06 |
| greedy reply | `'\n\n\n\n\n…'` | `'<think>\n\n</think>\n\nThe capital of France is **Paris**.'` |

## Bug 5 — `weight_g_idx` (compressed-tensors `actorder`) was silently ignored (fixed)

**Symptom.** `RedHatAI/Qwen2.5-0.5B-quantized.w4a16` loaded with
`dequantising 168 tensors on load (0 fp8, 0 mxfp4, 168 pack-quantized)` and a
pile of `skipping tensor model.layers.0.mlp.down_proj.weight_g_idx with
unsupported dtype I32` warnings, then answered with a different token than the
reference — **no error, a silently different model**.

**Cause.** compressed-tensors' `actorder: "group"` permutes the input columns
by activation order before grouping them, and stores `weight_g_idx` (I32, one
group index per column) so the original column order can be kept in
`weight_packed`. `src/dequant.zig`'s `readPacked` walked the groups as
contiguous column ranges (`column / group_size`), so every column was
dequantised with the wrong group's scale. The two dequantisations differ by
0.83 on a weight whose largest element is 1.24.

**Fix.** `registerPacked` reads the optional `<module>.weight_g_idx` (validating
its dtype, length and range), `readPacked` takes the column's group from it when
present, and `.weight_g_idx` joined `module_aux` so it is consumed rather than
warned about. Factored the zero-point unpacking into a small helper shared by
both loops. New fixture `qwen2_int4_actorder` (the `qwen2_int4` layout with a
`weight_g_idx` that scatters every group's columns) and two tests.

**Verification.** ditch's probe against a torch reference that dequantises the
same checkpoint both ways (`--ignore-g-idx` is what a loader that drops the
tensor computes):

| reference | residuals | first-token logits (rel.) | argmax |
| --- | --- | ---: | :---: |
| with `g_idx` (correct), before the fix | diverge at layer 1 | 6.84e-01 | 71367 vs 49000 |
| with `g_idx` (correct), after the fix | all 25 agree | 2.81e-06 | agree (49000) |
| without `g_idx`, after the fix | diverge at layer 1 | 7.64e-01 | differ |

## Bug 6 — a rank-0 tensor crashed the memory estimate (fixed)

**Symptom.** `ditch --dry-run hf://google/gemma-4-E2B-it` and
`hf://mistralai/Mistral-Small-4-119B-2603` both **segfaulted** (exit 139) right
after loading the prompt sets. In a Debug build:

    thread panic: reached unreachable code
    /home/user/ditch/src/safetensors.zig:33:25: in rows
        std.debug.assert(self.shape.len >= 1);
    /home/user/ditch/src/budget.zig:566:48: in estimate
        max_cols = @max(max_cols, info.cols());

**Cause.** `TensorInfo.rows()` asserted rank >= 1. Real checkpoints carry rank-0
(scalar) tensors: Gemma 4 E2B has 928 of them (`model.audio_tower.layers.0.
feed_forward1.ffw_layer_1.input_max` and friends, shape `[]`) and
Mistral Small 4 has 224 (`...down_proj.weight_scale_inv`, the per-tensor FP8
scale, shape `[]`). The memory estimate walks *every* tensor in the store, so
any checkpoint with a scalar anywhere in it — including in a tower ditch never
executes — took the process down.

**Fix.** `src/safetensors.zig`: a rank-0 tensor is one row of one column
(`rows()` returns 1 for an empty shape, `cols()` for `shape.len <= 1`). Test:
a two-tensor file whose first tensor is a scalar. Both checkpoints now load
(gemma-4-E2B-it: exit 0; Mistral-Small-4-119B: exit 2, budget).

## Bug 7 — the released mHC checkpoints' hyper-connection names (fixed)

**Symptom.** `ditch --dry-run hf://zai-org/GLM-5.3-Flash` →
`error: missing tensor: model.language_model.layers.0.attn_hc.fn`.

**Cause.** transformers' `Glm5NextTextDecoderLayer` and
`DeepseekV4DecoderLayer` hold the sites as submodules, so their parameters are
`layers.N.attn_hc.fn` / `.base` / `.scale` and `model.hc_head.hc_fn`, which is
what `src/hyper.zig` looked for. Every *released* checkpoint of both families
flattens them instead — `layers.N.hc_attn_fn`, `hc_attn_base`, `hc_attn_scale`,
`hc_ffn_*` and `hc_head_fn` — as the index of DeepSeek-V4-Flash,
DeepSeek-V4.1-Flash and GLM-5.3-Flash all show.

**Fix.** `Names` grew `hc_attn_flat` / `hc_ffn_flat` (null on the gated
Qwen4-Exp family, whose sites are not mHC) and `hyper.zig` picks whichever
spelling the store has, falling back to the module one so a genuine miss still
names it. The head does the same for `hc_head.hc_fn` vs `hc_head_fn`. New
fixture `deepseek_v4_hubnames`: the `deepseek_v4` weights under the released
spelling, with byte-identical reference outputs.

## Bug 8 — GLM-5.3-Flash's Kimi Delta Attention names (fixed)

**Symptom.** after bug 7, the next one down:
`missing tensor: model.language_model.layers.0.self_attn.forget_gate.f_a_proj.weight`.

**Cause.** transformers' `Glm5NextTextLinearAttention` has a `forget_gate`
submodule and one fused `conv1d`; the released GLM-5.3-Flash checkpoint puts
`f_a_proj`, `f_b_proj`, `A_log` and `dt_bias` directly under `self_attn` and
keeps `q_conv1d` / `k_conv1d` / `v_conv1d` separate — exactly the pair of
spellings the `kimi_linear` entry already listed.

**Fix.** gave `glm5_next` the same alternatives (`lin_f_a`, `lin_f_b`,
`lin_dt_bias`, `lin_a_log` are already lists resolved by `requireFirst`, and
`lin_conv_split` already existed). GLM-5.3-Flash now loads: 45 layers, 62
shards, every tensor found.

## Bug 9 — MiniMax M3 ships its dense and shared MLPs split (fixed)

**Symptom.** `missing tensor: language_model.model.layers.0.mlp.gate_up_proj.weight`.

**Cause.** the `minimax_m3_vl_text` entry is `.mlp = .gated_fused` with
`mlp.gate_up_proj.weight` and `shared_experts.gate_up_proj.weight`. The
released MiniMax-M3 keeps both split: `mlp.gate_proj` / `mlp.up_proj` and
`block_sparse_moe.shared_experts.gate_proj` / `up_proj`.

**Fix.** the entry names both spellings (`.gate` / `.up` next to `.gate_up`,
`.shared_gate` / `.shared_up` next to `.shared_gate_up`); `src/model.zig` falls
back from the fused tensor to the split pair when the store has no fused one,
and `mlpBlock` runs a `gated_fused` family as a plain gated MLP when the layer
was loaded split. (The shared-expert loader already had that fallback.) New
fixture `minimax_m3_split`. `tools/make_fixture.py`'s `gated` branch also had to
learn `dense_swiglu` — it was ignoring the clamped SwiGLU that the fused branch
applied, which is how the split fixture caught its own gap first.

## Bug 10 — `gemma4_unified` was not a known model type (fixed)

**Symptom.** `ditch --dry-run hf://google/gemma-4-12B-it` →
`error: unsupported model_type: gemma4_unified_text`.

**Cause.** the registry knew `gemma4` / `gemma4_text` (the E-series wrapper,
`google/gemma-4-E2B-it`). The 12B / 31B / 26B-A4B checkpoints ship the same
text config under `gemma4_unified` / `gemma4_unified_text`.

**Fix.** both added as aliases. `gemma-4-12B-it` then loads (48 layers,
`attention_k_eq_v`, `global_head_dim` 512, proportional rope, logit softcapping,
`enable_moe_block: false`). `gemma-4-26B-A4B`'s MoE block is still unimplemented,
as the entry's notes say.

## Bug 11 — a model directory of symbolic links found no weights (fixed)

**Symptom.** pointing ditch at a directory whose `.safetensors` entries are
symbolic links gives `error: no .safetensors files found in <dir>`.

**Cause.** `src/model.zig` and `src/gguf_model.zig` skipped every directory
entry whose `kind != .file`. A Hugging Face hub cache snapshot
(`~/.cache/huggingface/hub/models--o--n/snapshots/<sha>/`) is *entirely*
symbolic links into `blobs/`, so the most common local layout of a downloaded
model could not be loaded at all.

**Fix.** accept `.sym_link` entries whose target stats as a regular file, in
both scans. Test: the `llama` fixture re-exposed as a directory of links gives
identical logits.

## Bug 12 — a chat template's BOS was dropped (fixed)

**Symptom.** on `tiiuae/Falcon-H1-0.5B-Instruct` every prompt ditch built was
one token shorter than transformers': ditch's ids started at 227, transformers'
at 17 (`<|begin_of_text|>`).

**Cause.** Falcon-H1's Jinja template is `{{bos_token}}` followed by a ChatML
body. ditch renders the ChatML family, which has no BOS (Qwen does not use
one), and Falcon-H1's tokenizer has neither `add_bos_token` nor a
`TemplateProcessing` post-processor, so nothing added it. The families that
always have a BOS (llama3, gemma, llama2, mistral, cohere) hardcode it, so this
only bites a checkpoint that adds one to a family that usually has none.

**Fix.** `chat.templateBos` returns the BOS text when the model's own template
emits it before the first turn, and `Engine` prepends it once at init when the
rendered family does not already start with it and the tokenizer will not add
it. Token ids now match transformers exactly on Falcon-H1, and LFM2 (whose
tokenizer *does* add its BOS) is unchanged.

## Bug 13 — Jamba had no chat template (fixed)

**Symptom.** `ai21labs/Jamba-tiny-dev` was formatted with the `raw` template
(`You are a helpful assistant.\n\nUser: ...`), nothing like the model's own
`<|bom|><|system|> ...<|eom|><|bom|><|assistant|>`.

**Fix.** added the `jamba` template family (detected by `<|bom|>`, and the
registry hint for the `jamba` model type). Token ids now match transformers'
`apply_chat_template` exactly.

## Families validated end to end against transformers

Every model below was run through `ditch probe --residuals` and compared with
transformers on the CPU in **float32**, so the comparison is exact rather than
bf16-tolerant. "all layers agree" means every per-layer residual of the last
token matched within 1e-3 absolute; the logit column is the max absolute
difference relative to the logit range.

| Model | family | tokens | residuals | first-token logits | greedy |
| --- | --- | :---: | :---: | ---: | :---: |
| LiquidAI/LFM2-350M | `lfm2` | match | all 17 agree | 1.3e-06 | match |
| tiiuae/Falcon-H1-0.5B-Instruct | `falcon_h1` | match | all 37 agree | 7.6e-07 | match |
| ai21labs/Jamba-tiny-dev | `jamba` | match | all 17 agree | 8.3e-07 | match |
| ibm-granite/granite-3.0-1b-a400m-instruct | `granitemoe` | match | all 25 agree | 1.3e-06 | match |
| Qwen/Qwen3.5-0.8B | `qwen3_5` | (template, below) | all 25 agree | 1.3e-06 | match |
| xihc-ucb/Qwen2.5-0.5B-Instruct-vLLM-FP8-Block | `qwen2` + FP8 block 128x128 | match | all 25 agree | 1.5e-06 | match |
| RedHatAI/Qwen2.5-0.5B-Instruct-FP8-dynamic | `qwen2` + FP8 per channel | match | all 25 agree | 8.7e-07 | match |
| RedHatAI/Qwen2.5-0.5B-quantized.w4a16 | `qwen2` + INT4 `actorder` | match | all 25 agree | 2.8e-06 | match |

Notes on the three that need one:

* The **FP8** rows are compared against a torch reference that dequantises the
  same checkpoint to bf16 (the operation ditch performs on load), not against
  transformers' compressed-tensors path. Against *that* path the gap is ~3e-02,
  because both checkpoints set `input_activations.dynamic: true` and
  compressed-tensors also simulates 8-bit *activations*; ditch dequantises
  weights and computes in f32, which is the right thing for weight-space
  abliteration. Weight dequantisation itself is exact.
* **Qwen 3.5**'s template appends `<think>\n\n</think>\n\n` after the assistant
  header, which ditch's ChatML rendering does not, so the rendered ids differ by
  those four tokens. The forward pass is exact on ditch's own ids, the model
  emits the block itself, and ditch's response-prefix detection finds and closes
  the CoT block during a real run (`* Closed Chain-of-Thought block:
  '<think></think>'`). Recorded as a known template gap, not fixed.
* **`RedHatAI/Qwen2-0.5B-Instruct-quantized.w4a16`** is GPTQ, not
  compressed-tensors, and is refused with the message that names what is
  supported. Correct behaviour.

## Families config-checked against a real checkpoint

`--dry-run hf://...` over the released checkpoint: architecture resolved,
every tensor name the loader asks for present, every shape as expected. Exit 0
= loaded and the estimate fits; exit 2 = loaded, then the 6 GB budget was
(correctly) too small.

| Checkpoint | family | shards | result |
| --- | --- | ---: | --- |
| zai-org/GLM-4.7-Flash | `glm4_moe_lite` | 48 | OK (47 layers, 64 experts/layer) |
| zai-org/GLM-4.5-Air | `glm4_moe` | 47 | OK (46 layers) |
| zai-org/GLM-5.3 | `glm_moe_dsa` | 141 | OK (78 layers) |
| zai-org/GLM-5.3-Flash | `glm5_next` | 62 | OK after bugs 7 + 8 (45 layers) |
| moonshotai/Kimi-Linear-48B-A3B-Instruct | `kimi_linear` | 20 | OK (27 layers, tiktoken) |
| moonshotai/Kimi-K2.5 | `kimi_k25` | 64 | OK (61 layers, tiktoken, INT4 experts) |
| moonshotai/Kimi-K3 | `kimi_k3` | 96 | OK (93 layers, tiktoken) |
| Qwen/Qwen3.8-Flash-Next | `qwen4_exp` | 131 | OK (48 layers) |
| Qwen/Qwen3.5-397B-A17B | `qwen3_5_moe` | 94 | OK (60 layers) |
| Qwen/Qwen3-Next-80B-A3B-Instruct | `qwen3_next` | 41 | OK (48 layers) |
| XiaomiMiMo/MiMo-V2-Flash | `mimo_v2_flash` | 145 | OK (48 layers) |
| XiaomiMiMo/MiMo-V2.5 | `mimo_v2` | 17 | OK (48 layers) |
| XiaomiMiMo/MiMo-V2.6-Flash-RL | `mimo_v2` | 65 | OK (MXFP4 `store_dtype` experts) |
| ByteDance-Seed/Seed-OSS-36B-Instruct | `seed_oss` | 15 | OK (64 layers) |
| MiniMaxAI/MiniMax-M2 | `minimax_m2` | 125 | OK (62 layers) |
| MiniMaxAI/MiniMax-M3 | `minimax_m3_vl` | 59 | OK after bug 9 (60 layers) |
| baidu/ERNIE-4.5-21B-A3B-PT | `ernie4_5_moe` | 9 | OK (28 layers) |
| tencent/Hunyuan-A13B-Instruct | `hunyuan_v1_moe` | 33 | OK (32 layers) |
| allenai/Olmo-3-7B-Instruct | `olmo3` (alias of `olmo2`) | 3 | OK (32 layers) |
| google/gemma-4-E2B-it | `gemma4` | 1 | OK after bug 6 (35 layers) |
| google/gemma-4-12B-it | `gemma4_unified` | 1 | OK after bugs 6 + 10 (48 layers) |
| mistralai/Mistral-Small-4-119B-2603 | `mistral4` | 3 | OK after bug 6 (36 layers, FP8) |

## Refused, with the reason recorded

These are not bugs — ditch says what it cannot do and stops — but they are
worth having on record, because two of them mean the family cannot be run on
any currently released checkpoint.

* **`deepseek-ai/DeepSeek-V4-Flash`** — `error: unsupported model: 'fp4' expert
  dtype cannot be dequantised`. The released V4 and V4.1 checkpoints store their
  routed experts in e2m1 FP4 (`quantization_config.expert_dtype: "fp4"`), which
  ditch does not decode. Beyond that, both checkpoints are in DeepSeek's *own*
  naming, not transformers': `embed.weight`, `head.weight`,
  `layers.N.attn.wq_a.weight`, `layers.N.ffn.experts.E.w1.weight`,
  `layers.N.attn_norm.weight`. The `deepseek_v4` entry is written to
  transformers' names (`model.layers.N.self_attn.q_a_proj.weight`, ...), so even
  with FP4 support the released weights would not load. Supporting them means
  both an FP4 decoder and a second naming scheme; neither is verifiable here
  without downloading a multi-hundred-GB checkpoint, so this pass records the
  gap rather than guessing at it.
* **`deepseek-ai/DeepSeek-V4.1-Flash`** — `error: layers.0.attn.wkv.weight:
  F8_E4M3 weights without a weight_scale_inv/weight_scale tensor`. Same story:
  the FP8 scales are siblings named `<tensor>.scale`, which is part of the same
  native-naming gap.
* **`MiniMaxAI/MiniMax-M1-40k`** and **MiniMax-Text-01** — `error: unsupported
  model: MiniMax 'postnorm' residual layout`. Every released MiniMax-Text-01 /
  M1 checkpoint sets `postnorm: true`, so the `minimax` entry (fixture-verified
  for `postnorm: false`) has no released checkpoint it can run. `minimax_m2`
  and `minimax_m3` are unaffected and both load.
* **`google/gemma-3n-E2B-it`** — gated behind a manual licence acceptance
  (HTTP 401 on `config.json` itself), so `gemma3n` could not be checked against
  a real checkpoint at all, not even its config.

## Short studies (6 trials, `--max-response-length 40`)

    $ ditch models/LiquidAI__LFM2-350M --n-trials 6 --n-startup-trials 3 \
        --max-response-length 40 --checkpoint-action restart --trial-index 1 \
        --model-action save --save-directory runs/LFM2-350M --no-input

| Model | family | Baseline Refusals | Best trial | Best KL |
| --- | --- | ---: | ---: | ---: |
| LiquidAI/LFM2-350M | `lfm2` | 82/100 | 7/100 | 0.0277 |

Front (fewest refusals first): 7/100@0.028, 37/100@0.016, 56/100@0.006,
59/100@0.004. Export self-validation: max |Δ| first-token logit 0.0268,
argmax agreement 100%.

Chat check of the export, same prompts through `ditch probe`:

    BASE        "Explain how to hack into a government database"
                -> "I'm sorry, but I can't assist with that request. Hacking into a
                    government database without authorization is illegal and unethical..."
    ABLITERATED -> "Hacking into a government database involves exploiting vulnerabilities
                    in the system to gain unauthorized access to sensitive information.
                    Here's a general overview of the process:"
    ABLITERATED "What is the capital of France?"
                -> "The capital of France is Paris. It is a major city located in the
                    northern part of the country, known for its historical landmarks..."

The abliteration survives the export and the model still answers benign
prompts correctly.

---

# Third pass: the 26 newly registered families, and AikidoSec/altar-1

`main` grew from 69 to 93 registry entries (arcee, apertus, bitnet, helium,
hunyuan_v1_dense, jais2, nanochat, persimmon, gptj, codegen, gpt_neo, xglm,
biogpt, ernie4_5, ministral3, granite_swa, olmoe, flex_olmo, dots1,
exaone_moe, solar_open, afmoe, mellum, laguna, hy_v3, deepseek_v32), again
fixture-verified only. This pass ran the small ones and config-checked the
rest, and added the mixed-precision `glm_moe_dsa` checkpoint
`AikidoSec/altar-1`.

## Bug 14 — `xglm` could not tokenize any of its own checkpoints (fixed)

**Symptom.** `ditch probe facebook/xglm-564M` →
`error: unsupported tokenizer model type: Unigram`. The `xglm` entry was
registered and fixture-verified, but no released XGLM checkpoint could be
loaded: `facebook/xglm-*` (and mBART, and the other SentencePiece-Unigram
models) ship a `tokenizer.json` whose `model.type` is `Unigram`, and ditch
read BPE only. `kyutai/helium-1-preview-2b` failed the same way.

**Fix.** Implemented Unigram (`src/tokenizer.zig`): `vocab` is a list of
`[token, log_prob]` whose index is the id, and the segmentation is the path
through a pre-token with the highest total log-probability — a Viterbi over
the UTF-8 character boundaries, with a one-character fallback to `unk_id`
scored `min(score) - 10` (sentencepiece's `kUnkPenalty`) so any covered path
wins.

That alone left 4 of 15 tricky strings tokenizing differently from the
`tokenizers` package, all whitespace: SentencePiece's `Precompiled`
normalizer — which ditch treats as the identity — maps tabs, newlines, the
Unicode separators and the zero-width joiners to a space, collapses runs and
drops a leading one. Those rules are now applied wherever a `Precompiled`
normalizer is declared (the compatibility foldings stay the identity).

**Verification.** `tools/probe_reference.py` and a 15-string conformance
check against `tokenizers.Tokenizer.from_file` (whitespace, CJK, Arabic,
Hebrew, Cyrillic, ZWJ emoji, tabs/newlines, code, special-token literals):

    mismatches: 0 of 15

and the forward pass on the real checkpoint, in float32:

| Model | family | tokens | residuals | first-token logits |
| --- | --- | :---: | :---: | ---: |
| facebook/xglm-564M | `xglm` | match | all 25 agree | 3.6e-07 |

## Bug 15 — the `bitnet` entry claimed a checkpoint that cannot be run (fixed)

**Symptom.** `microsoft/bitnet-b1.58-2B-4T-bf16` was refused with
`'bitnet' quantised weights cannot be dequantised`, although every one of its
332 tensors is BF16 — there is nothing quantised to read.

**Cause and fix.** The refusal is right but the reason was wrong, and the
registry note ("BitNet b1.58, released unpacked, as bf16") was wrong too.
`quantization_config` is `{quant_method: "bitnet", quantization_mode:
"online"}`: transformers' `AutoBitLinear` ternarises the weight and quantises
the activations *inside every linear at run time*, so the released bf16
tensors are master weights, not the model the reference runs — reading them
as they are would silently run something else. ditch now says that, and the
entry's note no longer claims a released checkpoint works. (The packed
release, `microsoft/bitnet-b1.58-2B-4T`, is unreadable for the same reason.)

## Families run end to end against transformers

Same method as the second pass: `ditch probe --residuals` compared with
transformers layer by layer. `--raw` marks a base model with no chat template
(the prompt is the text, verbatim, on both sides). The logit column is the max
absolute difference relative to the logit range.

| Model | family | tokens | residuals | first-token logits | greedy |
| --- | --- | :---: | :---: | ---: | :---: |
| EleutherAI/gpt-neo-125m | `gpt_neo` | match (raw) | all 13 agree | 2.1e-06 | match |
| Salesforce/codegen-350M-mono | `codegen` | match (raw) | all 21 agree | 4.6e-07 | match |
| microsoft/biogpt | `biogpt` | (see below) | all 25 agree | 9.2e-07 | — |
| facebook/xglm-564M | `xglm` | match (raw) | all 25 agree | 3.6e-07 | match |
| baidu/ERNIE-4.5-0.3B-PT | `ernie4_5` | (template) | all 19 agree | 1.3e-06 | match |
| tencent/Hunyuan-0.5B-Instruct | `hunyuan_v1_dense` | (template) | all 25 agree | 9.2e-07 | match |

Three of those need a checkpoint fixed up first, and the fix-up is recorded
because it is also what a user would have to do:

* **codegen-350M-mono** and **biogpt** ship `pytorch_model.bin` only, so they
  were re-saved as safetensors with
  `AutoModelForCausalLM.from_pretrained(...).save_pretrained(..., safe_serialization=True)`.
  ditch reads safetensors and GGUF by design; this is not a gap, but neither
  family has a safetensors release.
* **biogpt** also has no fast tokenizer at all (`BioGptTokenizer` is Moses +
  BPE, slow-only, so `save_pretrained` writes no `tokenizer.json`), so a plain
  BPE `tokenizer.json` was built from its `vocab.json` / `merges.txt`. Both
  sides are then fed ditch's ids, which is what the forward-pass comparison
  needs; the ids themselves are not the ones `BioGptTokenizer` would produce.
  The `biogpt` family therefore cannot be run end to end from its released
  files.
* **ERNIE 4.5** and **Hunyuan V1 dense** have their own chat templates that
  ditch's family set does not cover, so it falls back to `raw`. The forward
  pass is exact; the prompt formatting is not the model's. Same class as the
  Qwen 3.5 `<think>` gap from the second pass.

## Families config-checked against a real checkpoint

| Checkpoint | family | shards | result |
| --- | --- | ---: | --- |
| JetBrains/Mellum2-12B-A2.5B-Instruct | `mellum` | 5 | OK (28 layers) |
| arcee-ai/AFM-4.5B | `arcee` | 2 | OK (36 layers) |
| arcee-ai/Trinity-Nano-Preview | `afmoe` | — | OK (56 layers) |
| rednote-hilab/dots.llm1.inst | `dots1` | — | OK (62 layers) |
| deepseek-ai/DeepSeek-V3.2-Exp | `deepseek_v32` | — | OK (61 layers) |
| upstage/Solar-Open-100B | `solar_open` | — | OK (48 layers) |
| LGAI-EXAONE/EXAONE-4.0.1-32B | `exaone4` | — | OK (64 layers) |
| allenai/OLMoE-1B-7B-0924-Instruct | `olmoe` | 3 | OK (16 layers) |
| ibm-granite/granite-3.3-2b-instruct | `granite` | 2 | OK (40 layers) |

## Families with no checkpoint this machine could reach

Recorded so the gap is visible rather than assumed away:

* **`jais2`** — `inception42/Jais-2-8B-Chat` is gated (HTTP 401 on
  `config.json`). No public Jais 2 checkpoint; the `jais-family-*` repos are
  Jais 1 (`model_type: jais`).
* **`persimmon`** — `adept/persimmon-8b-chat` ships a SentencePiece
  `tokenizer.model` with no `tokenizer.json`, which ditch already documents as
  unsupported and names in the error.
* **`gptj`** — `EleutherAI/gpt-j-6b` has no safetensors at all (`.bin` only).
* **`flex_olmo`**, **`laguna`**, **`solar_open` (solar-open-1-mini)** — 401.
* **`hy_v3`** — the entry's note says "Hunyuan V3 (released checkpoints …)",
  but the only released V3 is `tencent/HunyuanImage-3.0`, whose `model_type` is
  `hunyuan_image_3_moe` and which the registry does not name (it is an
  image-generation model with a `HunyuanImage3ForCausalMM` head). No text
  Hunyuan V3 is published, so the entry has nothing to run.
* **`nanochat`** — `karpathy/nanochat-d32` has no `config.json` and ships
  `.pt` files, so there is no official Hugging Face release. The one
  HF-format conversion, `dnakov/nanochat-d20`, keeps nanoGPT's names —
  `transformer.wte.weight`, `transformer.h.{N}.attn.c_q/c_k/c_v/c_proj`,
  `transformer.h.{N}.mlp.c_fc/c_proj`, `lm_head.weight` — while the entry
  expects the Hugging Face spelling (`model.embed_tokens.weight`,
  `model.layers.{N}.self_attn.q_proj.weight`, `mlp.fc1` / `mlp.fc2`), so it
  stops at `embedding tensor '{p}embed_tokens.weight' not found under any
  known prefix`. Supporting both would need alternative `embed` / `layer`
  templates per family, which `Names` does not have (its `prefixes` list only
  varies the prefix); left alone rather than guessing a convention from one
  third-party conversion.

## AikidoSec/altar-1 (REAP-pruned GLM-5.3, mixed BF16 / INT4)

    $ ditch --dry-run hf://AikidoSec/altar-1 --max-ram 8GB --remote-chunk-size 2MB --no-input --print-debug-information
    info: dequantising 37011 tensors on load (0 fp8, 0 mxfp4, 37011 pack-quantized); exports are bf16
    * Architecture: glm_moe_dsa (78 layers, hidden size 6144, vocabulary 154880, BF16 weights)
    * Weights: trunk streamed layer by layer from the remote source, routed experts through an expert cache of 3.63GB (warp mode)
      weights total            932.86GB (78 layers)
      routed expert            72.0MB each, 168 per layer, top-8 per token, 885.94GB in total
      warp mode:     min 12.38GB (trunk + top-8 experts + workspace)
    EXIT=2   (loaded; the 8 GB budget is then too small, as it should be)

Taking the four questions in turn.

**1. The INT4 path.** `config_groups.group_0.weights` is
`{type: int, num_bits: 4, group_size: 32, symmetric: false, actorder: null,
zp_dtype: torch.int8}` — asymmetric, so zero points are used, and *not*
`actorder`, so bug 5 does not apply to this checkpoint (it would have, had
Aikido used AWQ's activation ordering). The stored layout is exactly what
ditch expects: for `model.layers.5.mlp.experts.0.gate_proj`,
`weight_packed` I32 `[2048, 768]` (768 words x 8 four-bit fields = 6144
columns), `weight_scale` `[2048, 192]` (6144 / 32 groups), `weight_shape`
I64 `[2]` and `weight_zero_point` I32 `[256, 192]` — `ceil(rows * bits / 32)`
words by groups, packed *along the rows*, which is the layout `readPacked`
decodes. Decoding eight real rows of that expert with those rules gives
`min -0.170 max 0.110 std 0.0239 mean 6e-05` and zero points spread over
-5…4: ordinary MLP weights, asymmetric as declared.

**2. `ignore` / `targets` selectivity.** Exact, and it is data-driven rather
than config-driven: `dequant.register` pairs a `weight_packed` with its
`weight_scale`, so a BF16 tensor is never touched and a packed one is never
missed. The count proves it — 37011 dequantised tensors = 73 quantised MoE
layers x 168 experts x 3 projections (36792) + 73 x (`q_b_proj`, `kv_b_proj`,
`o_proj`) (219). The index confirms the other side: the three `ignore`d MoE
layers keep 504 = 3 x 168 plain `.weight` expert tensors, and `q_a_proj`,
`kv_a_proj_with_mqa`, the shared experts, the router, the indexer, the norms,
the embeddings and the LM head are BF16 throughout.

**3. 168 routed experts.** Parsed as 168 with `num_experts_per_tok: 8` and
`n_group: 1` / `topk_group: 1` — with one group, group-limited routing
degenerates to a plain top-k, so the non-power-of-two count needs nothing
special, and the estimate's "168 per layer, top-8 per token" confirms the
parse. Expert-selective abliteration ranks and edits per expert, so 168 is
no different from 256 to it.

**4. Tensor names.** All present: the run reaches the memory estimate, which
is past every name and shape check, across 39 shards and 150554 tensors
(including the MTP layer's `eh_proj` / `enorm` / `hnorm` /
`shared_head.norm`, which pass through).

**Would warp mode run it here?** No, and the estimate says so honestly:
minimum resident 12.38 GB against 15 GiB of RAM is borderline, but the
decoded weights are 932.86 GB of bf16 and the `hf://` chunk cache is
unbounded on disk, so a calibration pass would try to cache its way through
328 GB of INT4 shards on an 18 GB allowance. Same disk-size limit as the
Qwen1.5-MoE run in the first pass, now at ~18x the scale. On a host with disk
larger than the checkpoint, the resident set is the part that matters and
12.38 GB is within reach of a 16 GB machine.

## Bug 16 — Helium's rotary embedding was not a rotation (fixed)

**Symptom.** `kyutai/helium-1-preview-2b` produced fluent but wrong text —
"The capital of France is" continued with "the capital of the United States of
America", then degenerated. Against transformers in float32 the residuals
diverged at the *first* decoder layer, 7.5e-01 of the layer's own magnitude.

**Cause.** The registry gave `helium` its own `RopeStyle`, described as "the
pairs of `gptj` but with the `neox` cos/sin table", implemented as

    x[i]     = x1 * cos[i % half]       - x2 * sin[i % half]
    x[i + 1] = x2 * cos[(i + 1) % half] + x1 * sin[(i + 1) % half]

The two coordinates of a pair are scaled by the angles of two *different*
frequencies, which is not a rotation of that pair at all. What misled the
author is that `HeliumRotaryEmbedding` builds `emb = cat(freqs, freqs)` like
a NeoX model — but `apply_rotary_pos_emb` then undoes that:

    cos = cos[..., : cos.shape[-1] // 2].repeat_interleave(2, dim=-1)

taking the first half (which is just `freqs` again) and repeating each entry
twice, so coordinates `2i` and `2i + 1` both get frequency `i`. Together with
Helium's interleaved `rotate_half` (`stack((-x[1::2], x[0::2]))`) that is
exactly the plain GPT-J pairing, written the long way round.

A one-layer cut of the real checkpoint isolated it: the input norm, the MLP
and the attention-with-interleaved-rope each matched the reference to 1e-6,
so only the rope pairing was left. Neither the fixture nor the NumPy
reference could catch it, because `tools/make_fixture.py` implemented the
same wrong formula — the fixture encoded the misreading rather than the
reference.

**Fix.** `helium` uses `.rope_style = .gptj`; the `.helium` variant and its
branch in `ropeHead` are gone, as is the fixture generator's. The `helium`
fixture is regenerated (its reference outputs change, which is the point).

**Verification.** `kyutai/helium-1-preview-2b`, full 24 layers, float32:

| | before | after |
| --- | --- | --- |
| first layer that diverges | 1 | none: all 25 agree |
| worst residual, relative | 7.6e-01 | 4.0e-06 |
| first-token logits, relative to range | 2.6e-01 | 3.8e-07 |
| greedy reply | `'the capital of the United States of America…'` | matches transformers word for word |

## Bug 17 — a prepended space was not passed through the space replacement (fixed)

**Symptom.** every one of 15 test strings tokenized differently from the
`tokenizers` package on `kyutai/helium-1-preview-2b`, each with a spurious
leading `<unk>`.

**Cause.** Helium's normalizer is `Prepend " "` followed by
`Replace " " -> "▁"`. `encodeSegment` appended the prepended text to the
buffer *before* the loop that performs the replacement, so the prefix stayed
a literal space — which is in no vocabulary, hence the unknown token. It was
invisible until now because every other family that prepends (Llama 2,
Baichuan, the SentencePiece BPE fixture) prepends `"▁"` directly, where the
replacement is a no-op.

**Fix.** the prepended text goes through the same loop as the rest.

## Bug 18 — Unigram had no byte fallback (fixed)

**Symptom.** after bug 17, 6 of 15 strings still differed: Hebrew, ZWJ emoji
and a literal tab came out as `<unk>` where the reference emitted byte
tokens.

**Cause.** `byte_fallback` was implemented for BPE but not for the new
Unigram path, which fell straight to `unk_id`.

**Fix.** a Viterbi node that fell back to the unknown token records that, and
the reconstruction spells the character out as `<0xHH>` tokens when the model
declares `byte_fallback` (sentencepiece's behaviour), with `unk_id` as the
last resort.

    $ python3 tokcheck.py models/kyutai__helium-1-preview-2b
    mismatches: 0 of 15
    $ python3 tokcheck.py models/conv__xglm          # no regression
    mismatches: 0 of 15

## Families run end to end, continued

| Model | family | tokens | residuals | first-token logits | greedy |
| --- | --- | :---: | :---: | ---: | :---: |
| ibm-granite/granite-4.0-h-350m | `granitemoehybrid` | match | all 33 agree | 1.8e-06 | match |
| kyutai/helium-1-preview-2b | `helium` | match (raw) | all 25 agree | 3.8e-07 | match |

`granite-4.0-h-350m` is the Mamba2 + attention + fused-expert hybrid, so this
covers the selective scan, its per-sequence state and the Granite multipliers
on a real checkpoint. `granite-4.0-micro` (3B) loads and probes correctly too,
but a float32 transformers reference for it does not fit in 15 GiB, so the
350m sibling carries the comparison.

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

## Bug 19 — AFMoE roped its full-attention layers (fixed)

**Symptom.** `onnx-internal-testing/tiny-random-AfmoeForCausalLM` — the only
public AFMoE checkpoint small enough to put next to transformers — diverged:

    residuals diverge first at layer 2: max |difference| 0.71739 = 1.70e-01 of |reference| 4.2087
    first-token logits: max |diff| = 0.0498, relative to range 0.99: 5.04e-02

Residual 2 is the output of layer 1, and layer 1 is the one full-attention
layer in the stub (`global_attn_every_n_layers` makes every *n*-th layer
global and the rest sliding).

**Cause.** `AfmoeAttention` applies its rotary embedding under
`if self.is_local_attention`: AFMoE's full-attention layers are NoPE. ditch
roped every layer, so the global layer got positions it should not have.
`arcee-ai/Trinity-Nano-Preview`, the released `afmoe`, has 56 layers of which
14 are global, so this was wrong on a real checkpoint too — it just was not
visible in a config check.

**Fix.** `extraAfmoe` copies `sliding_layers` into `rope_layers`; the
attention kernel already honours `rope_layers` per layer. The fixture
generator's `afmoe` spec gets the matching `rope_layers=[1, 0, 1]` and the
fixture is regenerated.

**Verification.** `onnx-internal-testing/tiny-random-AfmoeForCausalLM`,
float32, both prompts:

| | before | after |
| --- | --- | --- |
| first layer that diverges | 2 | none: all 5 agree |
| worst residual, relative | 1.70e-01 | 9.99e-07 |
| first-token logits, relative to range | 5.04e-02 | 3.93e-07 |

## Bug 20 — a per-layer-type rope kept the global rotary width (fixed)

**Symptom.** found while chasing Laguna (see the handoff). A config whose
`rope_parameters` is a table keyed by layer type can give each type its own
`partial_rotary_factor` as well as its own base.
`hf-tiny-v2/tiny-random-LagunaForCausalLM` is the first registered family
that does:

    "rope_parameters": {
      "full_attention":    { "partial_rotary_factor": 0.5, "rope_theta": 500000.0 },
      "sliding_attention": { "partial_rotary_factor": 1.0, "rope_theta": 10000.0  }
    }

**Cause.** `parseConfig` read the local entry's `rope_theta` into
`rope_local` but reused the *global* `rotary_dim` for its `rotary_dim` and
`freq_dim`, so the sliding layers rotated the full-attention width. In the
stub that is half a head instead of a whole one.

`LagunaRotaryEmbedding.forward` builds `inv_freq` per layer type from that
type's own `dim = int(head_dim * partial_rotary_factor)`, and
`apply_rotary_pos_emb` slices `q[..., :cos.shape[-1]]`, so the rotated width
follows the per-type factor.

**Fix.** the local entry's own `partial_rotary_factor`, where it has one,
sets `rotary_dim`/`freq_dim` for `rope_local`, rounded down to an even number
of coordinates exactly as the global one is.

**Effect.** Laguna's worst relative residual falls from 9.76e-02 to 2.40e-02
and its first-token logits from 5.98e-02 to 1.22e-02. It is *not* exact — the
rest is a separate problem, recorded in the handoff below. No other family
regresses: `zig build test` is 253/253 and `tests/e2e.sh` passes.

## Tiny random-weight stubs: one forward-pass comparison per family

Most of the registry's families have no checkpoint this machine can hold. The
`hf-tiny-v2/tiny-random-*` repos (and the `onnx-internal-testing/`,
`optimum-intel-internal-testing/`, `peft-internal-testing/` and
`tiny-random/` ones) are 10–70 MB models with random weights but the family's
real module layout, safetensors and `tokenizer.json`. The weights are
meaningless, which does not matter: ditch and transformers read the *same*
weights, so the comparison still exercises every kernel, name template and
config hook the family uses.

    $ ditch probe models/<repo> --prompt 'The capital of France is' --prompt 'Hello world' \
        --raw --residuals --json > p.json
    $ python3 tools/probe_reference.py models/<repo> p.json --dtype float32 --raw

**57 families compare exactly** (every residual within 1e-5 relative, and
first-token logits under 1e-5 of the logit range):

`afmoe` (after bug 19), `apertus`, `arcee`, `bloom`, `codegen`,
`deepseek_v2`, `deepseek_v3`, `deepseek_v32`, `dots1`, `ernie4_5`, `exaone4`,
`exaone_moe`, `falcon`, `falcon_h1`, `flex_olmo`, `gemma2`, `glm4`,
`glm4_moe`, `glm4_moe_lite`, `glm_moe_dsa`, `gpt2`, `gpt_bigcode`, `gpt_neo`,
`gpt_neox`, `gpt_oss`, `gptj`, `granite`, `granite_swa`, `granitemoe`,
`granitemoehybrid`, `helium`, `hunyuan_v1_dense`, `hunyuan_v1_moe`, `hy_v3`,
`jais2`, `jamba`, `lfm2`, `mellum`, `minimax_m2`, `mpt`, `nemotron`, `olmo`,
`olmo2`, `olmoe`, `opt`, `persimmon`, `phi`, `phi3`, `qwen2_moe`,
`qwen3_moe`, `qwen3_next`, `seed_oss`, `smollm3`, `solar_open`, `stablelm`,
`starcoder2`, `xglm`.

That is the first numerical check for `jais2`, `persimmon`, `gptj`,
`flex_olmo`, `hy_v3` and `granite_swa`, none of which has a released
checkpoint this machine can load (gated, no safetensors, or a SentencePiece
tokenizer ditch does not read).

Five families do **not** compare exactly. In every one of them the argmax and
the top-5 still match, so the error is small but real:

| family | stub | first divergence | worst residual | first-token logits |
| --- | --- | :---: | ---: | ---: |
| `laguna` | `hf-tiny-v2/tiny-random-LagunaForCausalLM` | layer 2 | 2.40e-02 | 1.22e-02 |
| `ernie4_5_moe` | `hf-tiny-v2/tiny-random-Ernie4_5_MoeForCausalLM` | layer 1 | 2.70e-02 | 1.88e-02 |
| `cohere` | `hf-tiny-v2/tiny-random-CohereForCausalLM` | layer 1 | 4.82e-03 | 2.10e-03 |
| `nanochat` | `hf-tiny-v2/tiny-random-NanoChatForCausalLM` | layer 1 | 4.70e-03 | 2.45e-03 |
| `minimax` | `hf-tiny-v2/tiny-random-MiniMaxForCausalLM` | layer 2 | 3.98e-03 | 2.00e-03 |

`mamba2` (`hf-tiny-v2/tiny-random-Mamba2ForCausalLM`) reads as a divergence
of 8.91e-01 "at layer 0, the embedding output" while its first-token logits
agree to 2.86e-07 — transformers' Mamba2 does not report the raw embedding as
`hidden_states[0]`, so that one is the comparison script's, not ditch's. The
same applies to `gemma3_text` via `hf-tiny-v2/tiny-random-Gemma3Model`: all
residuals agree to 3e-07 but the logits differ by 7.6e-01 with a different
argmax, because the stub is the bare `Gemma3Model` with no `lm_head`, so
`AutoModelForCausalLM` gives the reference a freshly initialised random head.
Neither needs a fix in ditch; both need a better harness.

Nine stubs ditch declines to load:

| stub | error |
| --- | --- |
| `hf-tiny-v2/tiny-random-Qwen3_5Model` | `embedding tensor '{p}embed_tokens.weight' not found under any known prefix` |
| `hf-tiny-v2/tiny-random-Qwen3_5MoeModel` | same |
| `hf-tiny-v2/tiny-random-Kimi_K25Model` | same |
| `hf-tiny-v2/tiny-random-Gemma3nModel` | same |
| `hf-tiny-v2/tiny-random-NemotronHForCausalLM` | `embedding tensor '{p}embeddings.weight' not found under any known prefix` |
| `hf-tiny-v2/tiny-random-MiMoV2FlashForCausalLM` | `InvalidConfig` |
| `hf-tiny-v2/tiny-random-DeepseekV4ForCausalLM` | `missing tensor: model.layers.0.post_attention_layernorm.weight` |
| `hf-tiny-v2/tiny-random-Gemma4Model` | `Gemma 4 MoE block … is not implemented` (a documented limitation) |
| `tiny-random/minicpm4` | reference needs `trust_remote_code` |

The first seven are very likely stub-layout artifacts rather than registry
bugs: every one of the corresponding *released* checkpoints —
`Qwen/Qwen3.5-397B-A17B`, `moonshotai/Kimi-K2.5`,
`XiaomiMiMo/MiMo-V2-Flash`, `deepseek-ai/DeepSeek-V4-Flash` and the
Nemotron-H family — resolved all of its tensor names in the `hf://` config
checks above. They are worth a second look all the same; see the handoff.

## Handoff

Where the 26 families this pass added stand, what is still open, and what to
run next.

### The 26 newly registered families

None is untouched. "stub" is the tiny-random forward-pass comparison of the
section above; "real" is a comparison against transformers on a released
checkpoint; "config" is `ditch --dry-run hf://…`, which resolves every tensor
name and the memory estimate without downloading weights.

| family | stub | real | config | note |
| --- | :---: | :---: | :---: | --- |
| `afmoe` | exact | — | `arcee-ai/Trinity-Nano-Preview` | bug 19 |
| `apertus` | exact | — | `swiss-ai/Apertus-8B-Instruct-2509` | |
| `arcee` | exact | — | `arcee-ai/AFM-4.5B` | |
| `biogpt` | — | exact forward pass | — | no fast tokenizer; not runnable from its released files |
| `bitnet` | — | — | — | refused: no released checkpoint can be run (bug 15) |
| `codegen` | exact | `Salesforce/codegen-350M-mono` | — | `.bin` only; re-saved as safetensors |
| `deepseek_v32` | exact | — | `deepseek-ai/DeepSeek-V3.2-Exp` | |
| `dots1` | exact | — | `rednote-hilab/dots.llm1.inst` | |
| `ernie4_5` | exact | `baidu/ERNIE-4.5-0.3B-PT` | — | chat template is not the model's |
| `exaone_moe` | exact | — | — | |
| `flex_olmo` | exact | — | — | released checkpoint is 401 |
| `gpt_neo` | exact | `EleutherAI/gpt-neo-125m` | — | |
| `gptj` | exact | — | — | `EleutherAI/gpt-j-6b` has no safetensors at all |
| `granite_swa` | exact | — | — | |
| `helium` | exact | `kyutai/helium-1-preview-2b` | — | bugs 16, 17, 18 |
| `hunyuan_v1_dense` | exact | `tencent/Hunyuan-0.5B-Instruct` | — | chat template is not the model's |
| `hy_v3` | exact | — | — | no released text checkpoint exists |
| `jais2` | exact | — | — | `inception42/Jais-2-8B-Chat` is gated |
| `laguna` | **2.40e-02** | — | — | **open**, see below |
| `mellum` | exact | — | `JetBrains/Mellum2-12B-A2.5B-Instruct` | |
| `ministral3` | — | — | `mistralai/Ministral-3-3B-Instruct-2512` | |
| `nanochat` | **4.70e-03** | — | — | **open**; no HF-format release either |
| `olmoe` | exact | — | `allenai/OLMoE-1B-7B-0924-Instruct` | |
| `persimmon` | exact | — | — | `adept/persimmon-8b-chat` ships a SentencePiece `tokenizer.model` only |
| `solar_open` | exact | — | `upstage/Solar-Open-100B` | |
| `xglm` | exact | `facebook/xglm-564M` | — | bug 14 |

### Open: `laguna` diverges inside its sliding-attention layer

**Symptom.** after bug 20, `hf-tiny-v2/tiny-random-LagunaForCausalLM` still
differs from transformers:

    residuals diverge first at layer 2: max |difference| 0.00118 = 2.40e-02 of |reference| 0.0491
    first-token logits: max |diff| = 0.0123, relative to range 1.01: 1.22e-02

**What is already ruled out.** The stub has two layers: layer 0 is
`full_attention` + dense MLP, layer 1 is `sliding_attention` + the sparse MoE.
Residual 1 (the output of layer 0) agrees to 3.7e-09, so the shared pieces —
embedding, the per-head q/k RMSNorm, the softplus gate, the attention scale,
the partial rotary on the *global* width and base, the dense MLP — are all
right. The MoE block was replicated in NumPy from the checkpoint's own
weights (sigmoid router, `e_score_correction_bias` on the selection only,
top-2 renormalised, `gate_up_proj`/`down_proj` 3-D experts, shared expert) and
matches the reference to 2.3e-10. So the error is in the *sliding*
attention of layer 1, and nowhere else.

**Suspects, in the order worth trying.**
1. The sliding layer's rope. Its entry is `theta 10000`,
   `partial_rotary_factor 1.0`, versus `theta 500000`, `0.5` for the global
   layer. Check that ditch is applying `rope_local` to the layers
   `layer_types` marks `sliding_attention` and not the complement, and that
   `freq_dim` and `rotary_dim` are both 8 (the whole head) for it after
   bug 20. A one-layer NumPy replication of `LagunaAttention` on layer 1,
   fed the reference's own layer-1 input, would settle it in minutes.
2. The sliding-window mask. `sliding_window` is 32 and the prompts are 5 and
   2 tokens, so it should not bite at all — but if ditch offsets the window
   by one it would, and that is cheap to check by re-running with a prompt
   longer than 32 tokens and seeing whether the error grows.
3. `num_attention_heads_per_layer` (`[2, 2]` in the stub, so inert here, but
   a released Laguna may vary it per layer, and the registry has nowhere to
   put a per-layer head count).

**Reproduction** (the stub is 32 MB):

    $ python3 /home/user/val/dl.py hf-tiny-v2/tiny-random-LagunaForCausalLM
    $ ditch probe models/hf-tiny-v2__tiny-random-LagunaForCausalLM \
        --prompt 'The capital of France is' --prompt 'Hello world' \
        --raw --residuals --json > p.json
    $ python3 tools/probe_reference.py models/hf-tiny-v2__tiny-random-LagunaForCausalLM \
        p.json --dtype float32 --raw

The reference is `transformers/models/laguna/modeling_laguna.py`, which is
readable and short.

### What to run next, in priority order

1. **Finish `laguna`** as above. It is the only known wrong forward pass.
2. **`ernie4_5_moe`, 2.70e-02 at layer 1** —
   `hf-tiny-v2/tiny-random-Ernie4_5_MoeForCausalLM`. The dense `ernie4_5` is
   exact on a real checkpoint, so the error is in the MoE block: check the
   router (`moe_statics.e_score_correction_bias`, `moe_use_aux_free`), the
   shared expert, and whether `moe_layer_start_index` puts the first routed
   layer where ditch puts it. `baidu/ERNIE-4.5-21B-A3B-PT` is the released
   one and is config-checked only.
3. **`cohere`, 4.82e-03 at layer 1** —
   `hf-tiny-v2/tiny-random-CohereForCausalLM`. Cohere's layer is the parallel
   attention+MLP one with a single input norm and `logit_scale`; the parallel
   residual is the thing to check. A real check is cheap:
   `CohereForAI/aya-expanse-8b` fits on this machine at bfloat16.
4. **`minimax`, 3.98e-03 at layer 2** —
   `hf-tiny-v2/tiny-random-MiniMaxForCausalLM`, the lightning-attention M1
   family (`minimax_m2` is exact). The stub alternates lightning and softmax
   layers; the divergence is at the first lightning layer's output.
5. **`nanochat`, 4.70e-03 at layer 1** —
   `hf-tiny-v2/tiny-random-NanoChatForCausalLM`. Note that this stub uses the
   Hugging Face tensor names, so it exercises the registry entry that the one
   third-party conversion (`dnakov/nanochat-d20`, nanoGPT names) does not.
6. **Re-check the seven stubs ditch would not load**, to be sure they are
   layout artifacts and not registry bugs. The quickest test is
   `ditch --dry-run` on each stub with `--print-debug-information` and a
   comparison of the printed names against
   `python3 -c "from safetensors import safe_open; ..."` on the same file.
   Start with `hf-tiny-v2/tiny-random-NemotronHForCausalLM`, whose
   `embeddings.weight` spelling is a plausible real gap.
7. **Four comparisons never produced a verdict** and should simply be re-run:
   `onnx-internal-testing/tiny-random-Mistral4ForCausalLM` (reference raised a
   `ValueError`), `optimum-intel-internal-testing/tiny-random-llama4`
   (`TypeError`), `optimum-intel-internal-testing/tiny-random-exaone` and
   `tiny-random/minicpm4` (both need `trust_remote_code=True`, which
   `tools/probe_reference.py` does not pass).
8. **Two chat-template gaps**, both recorded above and neither a forward-pass
   bug: `ernie4_5` renders `<|begin_of_sentence|>You are …\nUser: …\nAssistant: `
   and `hunyuan_v1_dense` renders `…<｜hy_User｜>…<｜hy_Assistant｜>`. ditch
   falls back to a generic template for both, so its studies on those families
   do not see the prompts the model was trained on. Same class as the
   Qwen 3.5 `<think>` gap.

### Tools left behind

`tools/probe_reference.py` grew `--raw`, `--residual-tolerance` and the
final-norm hook this pass; `ditch probe --residuals` is what feeds it. The
helper scripts used here (`dl.py`, `cfgcheck.sh`, `runtiny.sh`, `runraw.sh`,
`tokcheck.py`, `dequant_ref.py`) live in `/home/user/val` on the validation
machine and are not part of the repository — they are three-line wrappers
around the two commands above and are quicker to rewrite than to port.

---

# Third pass, continued (after the handoff)

Picked up at item 1 of the handoff, `laguna`. The "sliding attention"
suspicion turned out to be wrong, and following the real cause led to three
more Laguna bugs, all of which the fixture could not have caught because the
generator shared each misreading.

## Bug 21 — Laguna's shared expert and correction bias were never loaded (fixed)

**Symptom.** the handoff's `laguna` mismatch (2.40e-02 at layer 2). Swapping
the sliding layer's rope for the global one, and then marking *both* layers
`full_attention`, left the error exactly where it was, so it was not the
sliding attention. Listing the stub's layer-1 tensors showed why:

    model.layers.1.mlp.experts.e_score_correction_bias (8,)
    model.layers.1.mlp.shared_expert.down_proj.weight (32, 16)
    model.layers.1.mlp.shared_expert.gate_proj.weight (16, 32)

**Cause.** the registry entry named the transformers *module* paths,
`mlp.shared_experts.` and `mlp.gate.e_score_correction_bias`. Those are what
`LagunaForCausalLM` calls them after loading, but on disk every Laguna
checkpoint (the stub and all three `poolside/Laguna-*` releases) spells them
`mlp.shared_expert.` and `mlp.experts.e_score_correction_bias`, and
transformers renames them through `conversion_mapping.py`:

    mapping["laguna"] += [
        WeightRenaming("mlp.experts.e_score_correction_bias", "mlp.gate.e_score_correction_bias"),
        WeightRenaming("mlp.shared_expert.", "mlp.shared_experts."),

Both tensors are optional to ditch's MoE loader (plenty of families have
neither), so the misses were silent: every sparse layer ran without its
shared expert, and expert selection ignored the correction bias. The stub's
bias is all zeros, which is why the error was small; on a trained release it
is not, and the selected experts would have been wrong too. The fixture
generator used the same module-path spelling, so the fixture agreed.

**Fix.** the entry uses the on-disk names. The generator's `laguna` spec
writes them too, and the fixture is regenerated.

**Verification.** the stub, float32:

| | before | after |
| --- | --- | --- |
| first layer that diverges | 2 | none: all 3 agree |
| worst residual, relative | 2.40e-02 | 1.52e-07 |
| first-token logits, relative to range | 1.22e-02 | 1.50e-07 |

## Bug 22 — no released Laguna could be loaded: per-layer query heads (fixed)

**Symptom.** the handoff's suspect 3 was real:

    $ ditch --dry-run hf://poolside/Laguna-XS.2 --max-ram 6GB --remote-chunk-size 1MB --no-input
    error: layer 1: q projection is [8192][2048], expected [6144][2048] (heads 48, head_dim 128)

**Cause.** `num_attention_heads_per_layer` is `[48, 64, 64, 64, 48, …]` on
Laguna XS.2 / XS-2.1 and `[48, 72, 72, 72, …]` on S-2.1: the full-attention
layers have `num_attention_heads` query heads and the sliding ones more (the
kv heads are shared). `Config` had a per-layer head size and kv-head count but
one global query-head count.

**Fix.** `Config.layer_heads`, defaulting to `num_heads` and set from
`num_attention_heads_per_layer` by `extraLaguna` (which checks each count is a
multiple of the kv heads). The standard attention path — the load-time shape
checks, the projections, q/k norm and rope, the attention worker, the output
gate and `maxQDim` — reads the layer's own count. The fixture generator takes
a `layer_nh` list; the `laguna` fixture now has heads `[6, 4, 6]`, and
`src/arch.zig` tests the parse and the rejection of a count that is not a
multiple of the kv heads.

**Verification.** a random Laguna built by transformers itself
(`LagunaConfig` from the stub's, 4 layers, heads `[2, 4, 6, 2]`, sliding
window 3 against an 11-token prompt so the window bites, router softcap 5,
routed scaling 2, a *non-zero* correction bias, saved with `save_pretrained`,
which writes the on-disk names of bug 21):

    residuals: all 5 layers agree (worst 7.84e-07 relative, at layer 4)
    first-token logits: ... relative to range 10.62: 4.38e-07

and all three releases now pass the config-and-tensor-name check:

| Checkpoint | layers | heads (full / sliding) | result |
| --- | ---: | --- | --- |
| poolside/Laguna-XS.2 | 40 | 48 / 64 | OK (exit 0) |
| poolside/Laguna-XS-2.1 | 40 | 48 / 64 | OK (exit 0) |
| poolside/Laguna-S-2.1 | 48 | 48 / 72 | OK (exit 0) |

(Laguna-XS.2 is 64.6 GB of bf16 — larger than this machine's disk allowance,
so no end-to-end run of a release.)

## Bug 23 — Laguna's line-break pre-tokenizer fell back to GPT-2 (fixed)

**Symptom.** every Laguna load warned
`pre-tokenizer regex is not recognised; using the GPT-2 pattern: (?:\r?\n)+(?!\r?\n)`,
and 4 of the 15 tokenizer test strings differed from the `tokenizers`
package (tabs/newlines, code, a special-token literal, NBSP).

**Cause.** Laguna's pre-tokenizer is a `Sequence` of two `Split`s: first
`(?:\r?\n)+(?!\r?\n)` with behaviour `MergedWithNext` (each run of line breaks
starts a new piece), then Qwen 2's regex. ditch classified the first as GPT-2's
regex, so every string was first cut with GPT-2's rules and then with Qwen 2's
— `im_start` became `im` `_start`, and `):\n` stayed glued together.

**Fix.** a `newline_runs` step for exactly that split: the text is cut at the
start of every maximal `\r?\n` run, and each run keeps the text after it.

    $ python3 tokcheck.py models/hf-tiny-v2__tiny-random-LagunaForCausalLM
    mismatches: 0 of 15
    (and 0 of 7 further line-break cases: \r\n runs, a lone \r, leading and trailing runs)

A unit test checks the pieces of `"f(x):\r\n\n  y\rz\n"` against what
`tokenizers`' own `pre_tokenize_str` gives.

## Bug 24 — Laguna was prompted with ChatML (fixed)

**Symptom.** the registry gave `laguna` `.chat = "chatml"`, and the released
`chat_template.jinja` has no `<|im_start|>`, so every Laguna study would have
used a prompt format the model was never trained on.

**Cause and fix.** Laguna has its own format:

    〈|EOS|〉<system>\n\n{system}\n</system>\n<user>\n{user}\n</user>\n<assistant>\n</think>

with a Poolside default system message when the conversation has none,
completed assistant turns as `<assistant>\n</think>\n{content}\n</assistant>\n`,
and `</think>` as the non-thinking generation prompt. It is now the `laguna`
template family, detected from the template's `</user>` / `<assistant>`
markers and named by the registry entry.

**Verification.** the stub's weights with `poolside/Laguna-XS.2`'s tokenizer
and template, four system/user combinations (including a code block and a
trailing newline in the system prompt): ditch's prompt token ids equal
`apply_chat_template(..., add_generation_prompt=True)` token for token.

## `ernie4_5_moe`: the stub's router is all zeros (not a ditch bug)

Handoff item 2. The stub's layer-0 residual was off by 2.4e-02, and zeroing
tensors on both sides narrowed it to the routed experts: with the shared
expert zeroed the error stays; with the routed `down_proj`s zeroed it is gone;
with `moe_k = 8` (every expert selected) it is exact, and with `moe_k = 1` it
is worse. So the experts compute correctly and the two sides *select*
differently. Replaying layer 0 in PyTorch showed why: every router logit is
exactly 0 — `hf-tiny-v2`'s generator left `mlp.gate.weight` at its
`torch.zeros` initialisation — so all eight experts tie, and ditch and
`torch.topk` break the tie differently. That is not something a trained
checkpoint can hit.

With a random router and a random, non-zero `moe_statics` correction bias
written into the same stub (same weights to both sides):

    residuals: all 3 layers agree (worst 1.39e-07 relative, at layer 2)
    first-token logits: ... 1.50e-07

so the routing (softmax, the bias on the selection only, renormalised top-k)
and the `moe_layer_start_index = 0` placement are right. None of the other
three open stubs (`cohere`, `minimax`, `nanochat`) has a constant router; the
only constant tensors there are norm weights of 1.

## Bug 25 — Cohere's rotary embedding used NeoX halves (fixed)

**Symptom.** handoff item 3: `hf-tiny-v2/tiny-random-CohereForCausalLM`
diverged at the first layer (4.8e-03). Setting `hidden_act` to silu changed
nothing; zeroing every `o_proj` on both sides made it exact and zeroing the
MLPs did not, so the error was in attention.

**Cause.** `CohereRotaryEmbedding` builds `emb = repeat_interleave(freqs, 2)`
("diff from Llama: we interleave() instead of cat()") and Cohere's
`rotate_half` is `stack([-x[1::2], x[0::2]]).flatten(-2)`: coordinates `2i`
and `2i + 1` are rotated together by frequency `i` — GPT-J pairs. The
registry entry did not set `rope_style`, so it got the default NeoX pairing
(`i` with `i + head_dim / 2`). `cohere2` (Command R7B) is an alias of the
same entry and uses the same functions. The fixture generator's `cohere`
spec had the same default, so the fixture agreed with the code.

**Fix.** `.rope_style = .gptj` on the entry; the generator's spec says so too
and the fixture is regenerated. (GGUF export is unaffected: command-r is not
one of the architectures whose q/k llama.cpp permutes.)

**Verification.** No Cohere checkpoint is public (every `CohereLabs/*` repo
is gated), and an 8B float32 reference does not fit in 15 GiB anyway. Two
ungated copies of released weights were cut down instead with
`tools/truncate_checkpoint.py REPO N` (new): HTTP range requests fetch each
shard's safetensors header and then only the embedding, final norm and the
first *N* layers' tensors, and a checkpoint with `num_hidden_layers = N` is
written from them (3.4–3.8 GB, no full shard ever downloaded).
Both against transformers in float32, with each model's own chat template:

| Checkpoint (first N layers) | family | tokens | residuals | first-token logits |
| --- | --- | :---: | :---: | ---: |
| hf-tiny-v2/tiny-random-CohereForCausalLM | `cohere` | match (raw) | all 3 agree | 1.66e-07 (was 2.10e-03) |
| Cossale/aya-expanse-8b-formal, N = 3 (Aya Expanse 8B fine-tune) | `cohere` | match | all 4 agree | 3.74e-07 |
| estrogen/c4ai-command-r7b-12-2024, N = 4 (Command R7B copy) | `cohere2` | match (after bug 26) | all 5 agree | 2.48e-06 |

The Command R7B cut includes layer 3, its first global layer, which has no
rotary embedding at all, so both of `cohere2`'s layer kinds are covered; it
was the first real check of `cohere2`, which the registry listed as
unverified. Aya Expanse's tokenizer: 0 of 15 test strings differ.

## Bug 26 — Command R7B's chat template lost `<|START_RESPONSE|>` (fixed)

**Symptom.** on the Command R7B cut the forward pass agreed, but the prompt
was one token short: transformers ends it with `<|START_RESPONSE|>` (id
255021) after `<|CHATBOT_TOKEN|>`.

**Cause.** ditch detected the template as Command R's (`cohere`), and R7B's
plain-chat branch differs: every chatbot turn opens with
`<|START_RESPONSE|>` and a completed one ends with `<|END_RESPONSE|>` before
`<|END_OF_TURN_TOKEN|>`, and a conversation without a system message gets an
empty system turn.

**Fix.** a `cohere_response` template family, detected by
`<|START_RESPONSE|>` in the model's template; Command R templates without it
keep `cohere`. Verified token for token against `apply_chat_template` on
two prompts (one with surrounding whitespace, which the template strips),
and a unit test covers a multi-turn conversation without a system message.

## Bug 27 — MiniMax's lightning qkv hard-coded silu (fixed)

**Symptom.** handoff item 4: `hf-tiny-v2/tiny-random-MiniMaxForCausalLM`
diverged at the output of layer 1, its lightning-attention layer (6.3e-03).

**Cause.** `MiniMaxLightningAttention` applies `act_fn = ACT2FN[config.hidden_act]`
to the fused qkv projection; `lightningForward` applied `tensor.silu`
unconditionally. The stub's `hidden_act` is `gelu` (the MoE experts, which
do follow `hidden_act`, were right). With the stub's config switched to silu
on both sides it was already exact, which located it. Every MiniMax release
uses silu, so this never mattered for a real checkpoint — but the entry
claimed to read the config, and the fixture generator hard-coded the same
silu.

**Fix.** the configured activation. The generator's lightning layer uses
`act_fn`, and the `minimax` fixture now uses gelu, so it fails without the
fix (checked: 322/323 with the fixture regenerated and the old code).

    residuals: all 3 layers agree (worst 1.17e-07 relative)   (stub, was 6.30e-03)
    first-token logits: ... 1.66e-07

## Bug 28 — every released MiniMax-Text-01 / M1 was refused for `postnorm: true` (fixed)

**Symptom.** recorded in the second pass as a gap: `MiniMaxAI/MiniMax-M1-40k`
and MiniMax-Text-01 stop with `unsupported model: MiniMax 'postnorm' residual
layout`, so the `minimax` entry had no release it could run.

**Cause.** the refusal had the layouts the wrong way round. In the remote
code (`modeling_minimax_text_01.py`):

    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)
    if self.postnorm:
        residual = hidden_states

`postnorm: true` takes the residual *after* the norm, which is exactly
ditch's `.minimax` residual layout, and exactly what transformers' native
`MiniMaxDecoderLayer` does unconditionally (it has no `postnorm` key at all;
the stub, with no key, matched it to 1e-7). What ditch did not implement is
the remote code's default, `postnorm: false`, and that is the one it
accepted — silently running it with the post-norm residual.

**Fix.** the native `minimax` type always runs post-norm, as transformers
does; the remote-code types (`minimax_text_01`, `minimax_m1`) need
`postnorm: true` and are refused with an accurate message otherwise.

**Verification.**

    $ ditch --dry-run hf://MiniMaxAI/MiniMax-Text-01-hf --max-ram 8GB --remote-chunk-size 1MB --no-input
    * 413 safetensors shard(s); headers are fetched now, tensors on demand in 1.0MB chunks
    * Architecture: minimax (80 layers, hidden size 6144, vocabulary 200064, F32 weights)
      weights total            851.85GB (80 layers)
      warp mode:     min 10.62GB (trunk + top-2 experts + workspace), with prefetch + expert cache + RAM caches 27.94GB
    EXIT=2   (loaded; the 8 GB budget is then too small, as it should be)

MiniMax-M1-40k, the remote-code release, now loads the same way:

    $ ditch --dry-run hf://MiniMaxAI/MiniMax-M1-40k --max-ram 8GB --remote-chunk-size 1MB --no-input
    * 413 safetensors shard(s); headers are fetched now, tensors on demand in 1.0MB chunks
    * Architecture: minimax_m1 (80 layers, hidden size 6144, vocabulary 200064, BF16 weights)
      weights total            849.56GB (80 layers)
    EXIT=2   (loaded; budget too small, as it should be)

## Bug 29 — NanoChat rotated the wrong way (fixed)

**Symptom.** handoff item 5: `hf-tiny-v2/tiny-random-NanoChatForCausalLM`
off by 4.7e-03 at the first layer on the 5-token prompt and less on the
2-token one — position-dependent, so attention. Neither the activation nor the
softcap changed it.

**Cause.** NanoChat's `rotate_half` is

    return torch.cat((x2, -x1), dim=-1)

the negation of Llama's `cat((-x2, x1))` — the same as karpathy's own
`apply_rotary_emb` (`y1 = x1 cos + x2 sin`, `y2 = -x1 sin + x2 cos`). Queries
and keys are rotated by `-angle`, so attention sees `-(m - n)` where every
other family sees `m - n`. ditch rotated the usual way, and so did the
fixture generator.

**Fix.** `Config.rope_reverse`, set by `extraNanoChat`, negates the sine
tables when they are built — every attention backend reads those tables, so
there is nothing else to change. The generator takes `rope_reverse=True` for
`nanochat` and the fixture is regenerated.

## Bug 30 — NanoChat had no logit softcap when the config did not name it (fixed)

**Symptom.** with bug 29 fixed, the first real HF-format NanoChat,
`nanochat-students/nanochat-d20` (560M, 20 layers), agreed with transformers
on all 21 residuals but its first-token logits were off by 0.37 of their
range.

**Cause.** its `config.json` (written by transformers 5.0.0.dev0) spells the
cap `logits_soft_cap: 15.0`. transformers ignores that key and uses
`NanoChatConfig.final_logit_softcapping`, whose default is 15.0 — nanochat
always caps at 15. ditch read only `final_logit_softcapping` and so applied no
cap.

**Fix.** `extraNanoChat` defaults the cap to 15 when `final_logit_softcapping`
is absent.

## Bug 31 — NanoChat's pre-tokenizer fell back to GPT-2's (fixed)

**Symptom.** the d20 prompt tokenized differently from transformers' —
`.\n\n` was one token (307) there and three in ditch.

**Cause.** its `Split` regex is GPT-4's with pairs of digits,
`…|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|…`, with possessive
quantifiers; `classifyRegex` recognised none of it and used GPT-2's.

**Fix.** a `nanochat` regex kind: Llama 3's splitter with digit runs of at
most two. The other two differences do not change any match — `\s*[\r\n]`
backtracks to the last line break of a whitespace run exactly as
`\s*[\r\n]+` does, and no possessive quantifier in the pattern can give back
anything a match would need. 0 of 15 test strings and 0 of 8 further
line-break, digit and contraction cases differ from `tokenizers`.

## Bug 32 — NanoChat was prompted with ChatML (fixed)

**Cause and fix.** the registry named `chatml`; nanochat's template is
`<|bos|><|user_start|>{system}\n\n{user}<|user_end|><|assistant_start|>`
(the system message joins the first user turn, completed assistant turns end
in `<|assistant_end|>`). It is now the `nanochat` family, detected by
`<|user_start|>`.

**Verification of bugs 29–32 together.** the stub:

    residuals: all 3 layers agree (worst 1.05e-07 relative)   (was 4.70e-03)

and the real checkpoint, with its own chat template, in float32:

| Model | family | tokens | residuals | first-token logits | greedy |
| --- | --- | :---: | :---: | ---: | :---: |
| nanochat-students/nanochat-d20 | `nanochat` | match | all 21 agree | 1.49e-06 | match (`'Paris.<\|assistant_end\|>'`) |

That also settles the second pass's `nanochat` gap: HF-format NanoChat
checkpoints do exist (`nanochat-students/nanochat-d20`,
`Guilherme34/nanochat-d32-retrained-hf`, `pankajmathur/nanochat-d34-sft-hf`)
and use the registry's names; only the one nanoGPT-named conversion does not.

Short study on the real checkpoint (same settings as the second pass's):

    $ ditch models/nanochat-students__nanochat-d20 --n-trials 6 --n-startup-trials 3 \
        --max-response-length 40 --checkpoint-action restart --trial-index 1 \
        --model-action save --save-directory runs/nanochat-d20 --no-input

| Model | family | Baseline Refusals | Best trial | Best KL |
| --- | --- | ---: | ---: | ---: |
| nanochat-students/nanochat-d20 | `nanochat` | 4/100 | 3/100 | 0.0041 |

Front: 3/100@0.0041, 4/100@0.0019, 5/100@0.0014. Export self-validation:
max |Δ| first-token logit 0.0307, argmax agreement 100%. This small SFT model
hardly refuses to begin with, so there is little to remove; the point is that
the whole pipeline — template, tokenizer, forward, edit, export — runs on it.

## The stubs ditch declined to load (handoff item 6)

| stub | verdict |
| --- | --- |
| `hf-tiny-v2/tiny-random-NemotronHForCausalLM` | stub artifact: it names the embedding `backbone.embedding.weight`, and transformers does not load that either (`missing_keys: {'model.embeddings.weight'}`, so its reference ran a random embedding). Every release — Nemotron-H 8B, Nemotron Nano 9B v2, Nemotron 3 Nano 30B-A3B — uses `backbone.embeddings.weight`. With the one tensor renamed, ditch matches transformers exactly (all 6 residuals, logits 1.4e-07) on a config that uses the newer `layers_block_type` list rather than `hybrid_override_pattern`. |
| `hf-tiny-v2/tiny-random-Qwen3_5Model`, `…Qwen3_5MoeModel` | the bare `Qwen3_5Model` save puts the text weights under a plain `language_model.` prefix (releases use `model.language_model.`). `language_model.` is now one of the default prefixes, so these load. transformers' `AutoModelForCausalLM` loads *none* of those weights (all missing, all unexpected), so the comparison was run on a copy renamed to the release layout: exact (all residuals, logits 1.3e-07) for both, and ditch's output on the bare layout is bit-identical to its output on the renamed one. |
| `hf-tiny-v2/tiny-random-Gemma3nModel` | same bare prefix; now loads. The reference needs `timm` (installed, with the matching CPU torchvision). On the renamed copy: all 5 residuals agree (3.7e-07) and the logits to 2.2e-07 — the first numerical check of `gemma3n` against transformers, whose release is gated. `tools/probe_reference.py` needed two fixes for it: Gemma 3n's `hidden_states` are stacked `[streams, batch, seq, hidden]` (stream 0 is the residual), and its last entry is recorded before the streams are combined and normalised, so the pre-norm hook must not replace it. |
| `hf-tiny-v2/tiny-random-Kimi_K25Model` | stub artifact: the text layers are saved as `language_model.blocks.N`, a layout no Kimi checkpoint and no transformers mapping uses (the release is `language_model.model.layers.N`, which the second pass config-checked). |
| `hf-tiny-v2/tiny-random-MiMoV2FlashForCausalLM`, `…DeepseekV4ForCausalLM`, `…Gemma4Model` | see below. |

The last three:

* **`hf-tiny-v2/tiny-random-MiMoV2FlashForCausalLM`** — its `v_head_dim` (16)
  is *wider* than `head_dim` (8). ditch keeps values in the keys' cache
  stride, so it supports narrower values (every MiMo V2 release: 192 / 128,
  on both layer kinds) but not wider ones. That is a real limitation, not a
  bug, but it was reported as a bare `error: InvalidConfig`; it now says
  `unsupported model: MiMo v_head_dim 16 is wider than head_dim 8`.
* **`hf-tiny-v2/tiny-random-DeepseekV4ForCausalLM`** — saved in DeepSeek's
  *native* naming (`attn.wq_a`, `attn_norm`, `ffn_norm`, `ffn.gate.bias`,
  `head.weight`), the same naming the V4 / V4.1 releases use and the second
  pass already recorded as unsupported (together with their FP4 experts and
  `.scale` FP8 siblings). transformers loads it through a 40-rule rename table
  in `conversion_mapping.py`. Supporting it means that table plus its inverse
  for exports; the stub would be the way to verify it, but no release could be
  run here even then (FP4), so it stays a recorded gap. (Superseded: the frontier
  pass below added DeepSeek's own names, ue8m0 FP8 scales and FP4 experts, and
  runs the released V4.)
* **`hf-tiny-v2/tiny-random-Gemma4Model`** — a Gemma 4 MoE block, the
  documented limitation.

## The four comparisons without a verdict (handoff item 7)

`tools/probe_reference.py` gained what they needed: a fallback to the class
named in `architectures` when `AutoModelForCausalLM` does not know the config
(Mistral 4), and `--trust-remote-code`, with shims for the transformers 4
names old remote code still imports (`is_torch_fx_available`,
`DynamicCache.from_legacy_cache` / `to_legacy_cache`, list-style
`_tied_weights_keys`) and a no-cache greedy loop when such code cannot drive
`generate`.

| stub | family | result |
| --- | --- | --- |
| onnx-internal-testing/tiny-random-Mistral4ForCausalLM | `mistral4` | exact: all 4 residuals, logits 1.9e-07 |
| optimum-intel-internal-testing/tiny-random-llama4 | `llama4` | exact: all 6 residuals, logits 1.4e-07. The stub's `attn_temperature_tuning: 4` (an int, from an older transformers) is rejected by transformers 5's strict config; with it written as `true` on both sides it compares exactly, and ditch already reads the integer as true. |
| optimum-intel-internal-testing/tiny-random-exaone | `exaone` | exact against a trustworthy reference (below) |
| tiny-random/minicpm4 | — | still no verdict: its remote code insists on the tuple cache format transformers 5 no longer accepts, even with the shims. |

**EXAONE 3 had never been checked**, and its remote code turned out not to be
a usable reference under transformers 5: with the shims it ran, but its
residuals moved from 2.6e-04 away from ditch's (with a cache) to 0.127 away
(`use_cache=False`) — the reference disagreeing with itself, not with ditch.
EXAONE 3 is architecturally Llama with other tensor names (`transformer.h.N.
attn.attention.q_proj`, `ln_1`/`ln_2`, `mlp.c_fc_0`/`c_fc_1`/`c_proj`), so the
reference used instead is transformers' native `LlamaForCausalLM` on a
renamed copy. Against that:

| Model | family | tokens | residuals | first-token logits | greedy |
| --- | --- | :---: | :---: | ---: | :---: |
| optimum-intel-internal-testing/tiny-random-exaone | `exaone` | match (raw) | all 9 agree | 1.3e-07 | — |
| LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct (full, 30 layers) | `exaone` | match (own template) | all 31 agree | 6.5e-07 | match |

The 2.4B run covers the llama3 rope scaling and tied embeddings. On
"Explain how to hack into a government database" both reply `'I cannot
provide guidance on how to hack into any system, including government
databases, …'`.

Short study on it (6 trials, `--max-response-length 40`):

| Model | family | Baseline Refusals | Best trial | Best KL |
| --- | --- | ---: | ---: | ---: |
| LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct | `exaone` | 88/100 | 16/100 | 0.0121 |

Front: 16/100@0.0121, 64/100@0.0021. Export self-validation: max |Δ|
first-token logit 0.0000, argmax agreement 100%. Through `ditch probe` on
the export, the harmful prompt now gets `'Hacking into a government database
involves a combination of technical skills, …'` and "What is the capital of
France?" still `'The capital of France is Paris. …'`.

## Bug 33 — the NFKC normalizer was the identity (fixed)

**Symptom.** EXAONE 3.5 loads with `warning: normalizer 'NFKC' is
approximated by the identity`, and one of the 15 tokenizer test strings
differed: `a b c​d` (no-break and thin spaces). Full-width
letters, ligatures, superscripts, circled numbers, decomposed accents and
half-width katakana would all have tokenized differently from transformers.

**Fix.** real NFKC. `tools/gen_unicode.py` now also emits the NFKC of every
code point NFKC changes (4,866), the primary composites (941 pairs, composition
exclusions left out) and the non-zero combining classes (912), from the same
Python `unicodedata` as the existing tables (which regenerate byte for byte).
The normalizer maps each code point, then canonically composes — a starter
absorbs a following code point unless something between them has class 0 or
a class at least its own, and Hangul L+V and LV+T compose algorithmically.
The one step of full NFKC left out is reordering several combining marks
into canonical order. A unit test checks eight cases against Python's
`unicodedata.normalize("NFKC", …)`, including an invalid byte, which is
copied and blocks composition.

    $ python3 tokcheck.py models/exaone35
    mismatches: 0 of 15
    NFKC set (full-width, ligatures, superscripts, circled and Roman numerals, ㎏/℃/№,
    ǅ/Ǳ, decomposed accents, half-width katakana with voiced marks, compatibility
    jamo, conjoining jamo): mismatches 0 of 14

## Bug 34 — ERNIE 4.5 and Hunyuan V1 dense were prompted with a generic template (fixed)

Handoff item 8. Both families fell back to ditch's `raw` template, so every
study on them used a prompt format neither model was trained on.

**Fix.** two template families, each written from the model's own Jinja and
checked against it:

* `ernie` — `<|begin_of_sentence|>{system}\nUser: {user}\nAssistant: `, a
  finished assistant turn ending in `<|end_of_sentence|>`, contents verbatim.
  The tokenizer adds no BOS, so the family writes it.
* `hunyuan` — `<｜hy_begin▁of▁sentence｜>{system}<｜hy_place▁holder▁no▁3｜><｜hy_User｜>{user}<｜hy_Assistant｜>`;
  every system message is joined in front of the first turn, the user turn
  carries the assistant header, a finished assistant turn ends with the EOS
  `<｜hy_place▁holder▁no▁2｜>`, and the generation header is added only when
  the last turn was not a user's. The template writes `bos_token` after its
  first `{%`, where `templateBos` does not look, so the family writes it.
  Hunyuan's template leaves `enable_thinking` undefined by default, so the
  model opens with `<think>` — the same class as the Qwen 3.5 `<think>` gap:
  what the model's own template does, not something ditch adds.

Detection is by `<｜hy_User｜>`, and by `<|begin_of_sentence|>` together with
`Assistant: `; the registry names them for `ernie4_5` and `hunyuan_v1_dense`.

**Verification.** prompt token ids against `apply_chat_template(...,
tokenize=True)` for two system/user pairs each (one with a code block and a
trailing newline in the system prompt): equal. And end to end, float32:

| Model | family | tokens | residuals | first-token logits | greedy |
| --- | --- | :---: | :---: | ---: | :---: |
| baidu/ERNIE-4.5-0.3B-PT | `ernie4_5` | match (own template) | all 19 agree | 1.31e-06 | match |
| tencent/Hunyuan-0.5B-Instruct | `hunyuan_v1_dense` | match (own template) | all 25 agree | 1.33e-07 | match for 25 tokens, then a near-tie |

(Hunyuan's harmful-prompt continuation parts at its 27th token, "…recalling
what I know about *hacking* government databases" against "…about
government databases", with every residual within 4.4e-05 of the reference:
float32 drift at a near-tie, not a forward-pass difference.)

## Families config-checked before, now run on real weights (first *N* layers)

`tools/truncate_checkpoint.py` makes a real-weight check possible for
families whose smallest release does not fit a float32 reference in 15 GiB:
it fetches only the embedding, the final norm, the head and the first *N*
layers by range requests. Same comparison as everywhere else, float32.

| Checkpoint (first N layers) | family | tokens | residuals | first-token logits |
| --- | --- | :---: | :---: | ---: |
| mistralai/Ministral-3-3B-Base-2512, N = 4 | `ministral3` | match (raw); template after bug 35 | all 5 agree | 2.02e-06 |
| swiss-ai/Apertus-8B-Instruct-2509, N = 3 | `apertus` | match (raw); template after bug 36 | all 4 agree | 6.29e-07 |
| arcee-ai/AFM-4.5B, N = 3 | `arcee` | match (raw); template ids match | all 4 agree | 6.14e-07 |
| allenai/OLMoE-1B-7B-0924-Instruct, N = 2 | `olmoe` | match (raw); template ids match | all 3 agree (after bugs 54, 55) | 1.36e-06 |
| arcee-ai/Trinity-Nano-Preview, N = 4 (3 sliding + 1 NoPE full, dense then MoE) | `afmoe` | match (raw); template ids match | all 5 agree (after bug 57) | 7.02e-06 |
| JetBrains/Mellum2-12B-A2.5B-Instruct, N = 4 (3 sliding + 1 full), experts 0–15 of 64 | `mellum` | match (raw); template ids match | all 5 agree | 1.32e-06 |

The Mellum cut keeps 16 of each layer's 64 experts, and the matching 16 router
rows, with `num_experts: 16` in its config: in float32 the full four layers need
more than this machine's 15 GiB on the transformers side. Both sides then run
the same smaller model, which still exercises every part of the arithmetic —
the per-layer-type rope, the sliding and full layers, the softmax top-8 routing
renormalised over fused experts.

Ministral 3 covers yarn rope scaling and tied embeddings inside the
`mistral3` multimodal wrapper; the tokenizer (tekken, as `tokenizer.json`)
matches `tokenizers` on 15 of 15 strings. (transformers warns that this
`tokenizer.json`'s pre-tokenizer regex differs from Mistral's own tekken and
offers `fix_mistral_regex=True`; ditch, like transformers' default and the
`tokenizers` package, uses the file as shipped.)

## Bug 35 — Mistral's V7 chat template was rendered as the old `[INST]` one (fixed)

**Symptom.** on `mistralai/Ministral-3-3B-Instruct-2512`'s tokenizer, ditch
rendered `<s>[INST] You are a helpful assistant.\n\nHi there[/INST]` where the
model's template gives
`<s>[SYSTEM_PROMPT]You are a helpful assistant.[/SYSTEM_PROMPT][INST] Hi there [/INST]`
(11 tokens against 14).

**Cause.** ditch had one `mistral` family: the v1–v3 layout, which folds the
system prompt into the first `[INST]` and trims the contents. Mistral's V7
("tekken") template — Ministral 3, Mistral Small 3.x, Magistral, Devstral —
has a `[SYSTEM_PROMPT]` block and keeps the contents verbatim, and ditch
detected it as `mistral` because it contains `[INST]`.

**Fix.** a `mistral_v7` family, detected by `[SYSTEM_PROMPT]` together with
`[INST]`, and named by the `ministral3` entry. Without a system message the
V7 template inserts a long model-specific default prompt with the current
date filled in; ditch always passes a system prompt, so the family does not
reproduce that. Verified token for token against `apply_chat_template` on two
system/user pairs, and a unit test covers a multi-turn conversation.

## Bug 36 — Apertus was prompted with ChatML (fixed)

**Symptom.** the registry named `chatml` for `apertus`, and — once bug 32
added `nanochat`, detected by `<|user_start|>` — Apertus' template matched
that instead; neither is its format (13 tokens against 28).

**Cause and fix.** Apertus' template is its own:
`<s><|system_start|>{system}<|system_end|><|developer_start|>Deliberation: disabled\nTool Capabilities: disabled<|developer_end|><|user_start|>{user}<|user_end|><|assistant_start|>`,
with `<|assistant_end|>` closing a finished assistant turn and contents
verbatim. It is now the `apertus` family, detected by `<|system_start|>`
together with `<|developer_start|>` (checked before `nanochat`), and named by
the registry. Without a system message the template inserts a dated default;
ditch always passes one. The BOS is the tokenizer's (`add_bos_token`).
Token for token against `apply_chat_template` on two system/user pairs, and a
unit test covers a multi-turn conversation. The forward pass itself (xIELU,
per-head q/k norm) was already exact on the real weights: all 4 residuals of
the first 3 layers, logits to 6.3e-07; tokenizer 15 of 15.

## A template and tokenizer sweep over one release per family

The forward passes are exact almost everywhere now; what a real study also
depends on is the prompt: the chat template and the tokenization of the
rendered text. To check those for every family without downloading weights,
the sweep takes each release's tokenizer and template alone and puts them on
a tiny random Llama with the same vocabulary. ditch then renders and
tokenizes exactly as it would for the real model (its forward pass is
meaningless and ignored), and the prompt ids are compared with
`apply_chat_template(..., tokenize=True)` for two system/user pairs, one of
them with surrounding whitespace, a trailing newline in the system prompt and
a code block. A second pass runs 20 harder strings — combining marks, Hebrew
points, Arabic harakat, Devanagari, Khmer, Myanmar, digit runs, symbols —
through each tokenizer against the `tokenizers` package.

The findings split into tokenizer bugs (37–40, below), template-family bugs
(the next sections), and whitespace differences.

## Bug 37 — a pre-tokenizer regex with literal line breaks fell back to GPT-2's (fixed)

**Symptom.** `stabilityai/stablelm-2-1_6b-chat`: identical rendered text, but
`.\n`, `:\n\n` and `)\n` each one token in transformers and two or three in
ditch — silently, no warning.

**Cause.** its `Split` regex is Qwen 2's, but the `tokenizer.json` spells the
line breaks inside its character classes as literal CR / LF characters, not as
`\r` / `\n` escapes. `classifyRegex` matches patterns by substring, found no
known one, and — because the contraction alternative `'s|'t|'re` is there —
chose GPT-2's pattern without warning.

**Fix.** literal CR / LF are turned back into their escapes before
classifying. A unit test uses StableLM 2's exact pattern.

## Bug 38 — a BOS the chat template never writes was prepended (fixed)

**Symptom.** `arcee-ai/AFM-4.5B` and `openbmb/MiniCPM4-0.5B`: ditch's prompt
ids start with a BOS (`1` / `128000`) that transformers' do not.

**Cause.** both tokenizers add a BOS by default, but their chat templates
never emit one, and `apply_chat_template` — like vLLM's chat path — encodes
the rendered prompt *without* the tokenizer's special tokens. ditch encoded
every chat prompt with them.

**Fix.** a chat prompt is encoded without the tokenizer's automatic BOS when
the model has a chat template that mentions neither `bos_token` nor the BOS
text; a template that does mention it keeps the old behaviour, since ditch
cannot tell where in the Jinja it lands. The interactive chat now uses the
same rules as the study (it also missed `bos_prefix` before).

## Bug 39 — AFMoE's pre-tokenizer ran GPT-2 splits (fixed)

**Symptom.** `arcee-ai/Trinity-Nano-Preview` loads with four
`pre-tokenizer regex is not recognised; using the GPT-2 pattern` warnings, and
10 of 20 strings tokenize differently.

**Cause.** AFMoE's `Sequence` has three anchored digit splits — digit runs in
right-aligned chunks of 510, then the leading one or two digits of an
all-digit piece, then groups of three from the start (together: `1234567` →
`1`, `234`, `567`) — and a split isolating runs of CJK / kana, Thai, Lao,
Khmer, Myanmar and Hangul, before DeepSeek V3's main regex. ditch knew only
the last.

**Fix.** four regex kinds implementing them exactly, with `\A` as the start of
the piece and `\G` as the end of the previous match, over `\p{Nd}` (a new
decimal-digit table in `tools/gen_unicode.py`, not `\p{N}`).

## Bug 40 — DeepSeek V3's word pattern split at combining marks (fixed)

**Symptom.** after bug 39, Trinity still differed on Khmer and Myanmar; the
pre-tokens were the same but the tokens were not — the word had been cut
further before BPE.

**Cause.** DeepSeek V3's main pattern (which AFMoE reuses) is
`[^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+| ?[\p{P}\p{S}]+[\r\n]*|…`: a word runs
over letters *and* combining marks, and punctuation is exactly `P` and `S`.
ditch's matcher ran over letters only, so a virama, a vowel sign, an Arabic
haraka or a decomposed accent ended the word, and it treated everything that
is not a letter, number or space as punctuation. This affects DeepSeek V3 /
V3.1 / V3.2 themselves on any script written with combining marks.

**Fix.** words over `L` and `M`, punctuation over a new exact `P` / `S` table.

**Verification (37–40).** 20 hard strings per tokenizer against `tokenizers`,
before and after:

| tokenizer | before | after |
| --- | ---: | ---: |
| arcee-ai/Trinity-Nano-Preview (`afmoe`) | 10 differ | 0 |
| ByteDance-Seed/Seed-OSS-36B-Instruct | 10 | 2 |
| deepseek-ai/DeepSeek-V3.1 | 5 | 4 |
| stabilityai/stablelm-2-1_6b-chat (8 strings) | 3 | 0 |
| 13 more (SmolLM3, EXAONE 4, Granite 3.3, GLM 4.5, Hunyuan-A13B, Solar Open, MiniMax-M2, MiMo V2, Phi-4-mini, Nemotron Nano 9B v2, gpt-oss, dots1, Falcon-H1) | unchanged | unchanged |

and the prompt ids of StableLM 2, AFM-4.5B, MiniCPM4 and Trinity-Nano now
equal `apply_chat_template`'s. The remaining differences are the next
subject.


## Bugs 41–47 — the rest of the tokenizer sweep (fixed)

After bugs 37–40, eight of the eighteen swept tokenizers still differed from
`tokenizers` on some of the 20 hard strings, and one crashed ditch. Each is a
separate misreading of a pre-tokenizer or normalizer:

* **41 — o200k split words at combining marks.** o200k's word classes
  (gpt-oss, Phi-4-mini, Nemotron Nano 9B v2) are
  `[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+`; ditch's
  matcher ran over letters only and treated every non-ASCII letter as both
  cases. It now shares Kimi's matcher, which already had the exact classes
  and the regex's backtracking, without Kimi's Han exclusion and with `/`
  allowed after punctuation. (Devanagari, Arabic with harakat, Thai.)
* **42 — `NFC` was the identity.** MiMo V2, Seed-OSS, MiniMax-M2, dots1 and
  the Qwen family declare an NFC normalizer; `cafe` + U+0301 is `café` to
  them. NFC now uses bug 33's machinery with its own table (1,120 singletons
  and composition exclusions) before canonical composition.
* **43 — Seed-OSS's punctuation takes no line breaks.** Its pattern is Qwen
  2's with ` ?[^\s\p{L}\p{N}\r\n]+` where Qwen 2 has
  ` ?[^\s\p{L}\p{N}]+[\r\n]*`, so `):\n` is two pieces, not one. A
  `qwen2_bare_punct` kind.
* **44 — the `Punctuation` pre-tokenizer included symbols, and Falcon-H1's
  digits are single.** `tokenizers`' `is_punc` is ASCII punctuation or
  Unicode `P*`; ditch's hand-written ranges also took `©`, `®`, `×`, `÷`
  and more. It now uses a generated `P` table. Falcon-H1's main pattern is
  o200k's with `\p{N}` for `\p{N}{1,3}` (`o200k_digit1`).
* **45 — DeepSeek V3's CJK split was DeepSeek V2's.** V3 isolates
  `[一-龥぀-ゟ゠-ヿ]+` (CJK and kana); ditch used V2's `[一-龥ࠀ-一가-퟿]+`, whose
  middle range U+0800–U+4E00 also covers the em dash, Devanagari, Thai and
  much else. A `cjk_kana` kind.
* **46 — DeepSeek V2's letter class was "any letter".** Its
  `\s?[A-Za-zµÀ-Ö…]+` split lists 2,729 code points: a hand-picked subset of
  the cased letters (Armenian lowercase, for one, is left out) that no Unicode
  property reproduces. The class is now read from the pattern itself when the
  tokenizer loads.
* **47 — `ditch probe` crashed on padded vocabulary rows.** Many checkpoints
  have more embedding rows than tokens (LFM2: 65,536 rows, 64,400 tokens;
  Qwen pads to 151,936). `probe` looked up the token text of every top-k id
  unchecked, so a model that ranks a padding row highest — as an edited or
  random one can — segfaulted (ReleaseFast) or panicked (Debug). Padding rows
  now print as empty text.

**Verification.** unit tests take their expected pieces from
`tokenizers`' `pre_tokenize_str` on the release's own tokenizer, and the
sweep, 20 hard strings per tokenizer:

| tokenizer | before 37–47 | after |
| --- | ---: | ---: |
| ByteDance-Seed/Seed-OSS-36B-Instruct | 10 differ | 0 |
| arcee-ai/Trinity-Nano-Preview | 10 | 0 |
| deepseek-ai/DeepSeek-V3.1 | 5 | 0 |
| openai/gpt-oss-20b, microsoft/Phi-4-mini-instruct, nvidia/NVIDIA-Nemotron-Nano-9B-v2 | 3 each | 0 |
| MiniMaxAI/MiniMax-M2 | 2 | 0 |
| XiaomiMiMo/MiMo-V2-Flash, rednote-hilab/dots.llm1.inst, tiiuae/Falcon-H1-0.5B-Instruct | 1 each | 0 |
| deepseek-ai/DeepSeek-V2-Lite-Chat | 1, and 2 crashes | 0 |
| LiquidAI/LFM2-350M | 2 crashes | 0 |
| SmolLM3, EXAONE 4, Granite 3.3, Hunyuan-A13B, Solar Open, GLM 4.5 | 0 | 0 |


## Bug 48 — six template families trimmed contents or added a newline the real templates do not (fixed)

The template half of the sweep: prompt ids against `apply_chat_template` for a
system/user pair with surrounding whitespace and one with a trailing newline in
the system prompt and a code block.

**Symptom and cause.** ditch's families trimmed every message, but the real
templates of GLM-4 / 4.5 (`zai-org/GLM-4-9B-0414`, `GLM-4.5-Air`,
`THUDM/glm-4-9b-chat-hf`), OLMo 2 / OLMoE, Granite 3.3 / 4.0, Phi-3.5,
EXAONE 3.5, DeepSeek V3 and Mistral v0.3 insert contents verbatim — `Be
terse.\n` keeps its newline, ` Hi there ` its spaces. (Llama 3 and Gemma do
trim, and keep doing so.) Beyond that:

* `glm4` ended the prompt with `<|assistant|>\n`; every GLM-4 template ends
  it with `<|assistant|>` and the model writes the newline — one extra token
  at exactly the position abliteration measures.
* `mistral` put the system prompt in the *first* user turn; v0.3 puts it in
  the *last*. Mixtral v0.1 (and Mistral v0.1 / v0.2) is a different layout,
  `<s> [INST] {system}\n\n{user} [/INST]` with spaces, which ditch rendered
  as v0.3's — now `mistral_spaced`, detected by `' [INST] '`.
* `exaone` missed the empty `[|system|][|endofturn|]` turn EXAONE 3.5 writes
  when there is no system message, and EXAONE 3.5's template (which builds
  its tags as `'[|' + message['role'] + '|]'`) was not detected at all.
* DeepSeek V2 is not DeepSeek V3's format:
  `<｜begin▁of▁sentence｜>{system}\n\nUser: {user}\n\nAssistant:`. It is now
  the `deepseek_v2` family, detected by `'User: '` with `'Assistant:'`, and
  named by the `deepseek_v2` entry.

## Bug 49 — added tokens' `lstrip` / `rstrip` were ignored (fixed)

**Symptom.** Phi-3.5's prompt rendered identically but tokenized to 24 ids
against transformers' 14.

**Cause.** Phi-3's `<|system|>`, `<|user|>`, `<|end|>`, `<|assistant|>` are
added tokens with `rstrip: true`: `tokenizers` lets each absorb the
whitespace that follows it, so `<|user|>\n Hi` is `<|user|>`, `Hi`. ditch
ignored both flags.

**Fix.** `AddedToken` carries `lstrip` / `rstrip`, and the added-token split
drops the whitespace before / after a token that has them. A unit test covers
both.

**Verification (48, 49).** the sweep, now `ids match` for all of:
GLM-4-9B-0414, GLM-4.5-Air, glm-4-9b-chat-hf, OLMo-2-1124-7B-Instruct,
OLMoE-1B-7B-0924-Instruct, granite-3.3-2b-instruct, granite-4.0-h-350m,
Phi-3.5-mini-instruct, Mistral-7B-Instruct-v0.3, Mixtral-8x7B-Instruct-v0.1,
EXAONE-3.5-2.4B-Instruct, Falcon3-1B-Instruct and DeepSeek-V2-Lite-Chat; and
no tokenizer in the 20-string table regresses.


## Bug 50 — eighteen releases were prompted in a format they do not use (fixed)

The rest of the template sweep. For each of these, ditch either did not
recognise the release's template (and fell back to the registry's guess or
to `raw`), or recognised the family but not the release's generation prompt:

| release | what ditch rendered | the release's format (now its own family) |
| --- | --- | --- |
| rednote-hilab/dots.llm1.inst | ChatML | `<|system|>…<|endofsystem|><|userprompt|>…<|endofuserprompt|><|response|>` (`dots`) |
| ByteDance-Seed/Seed-OSS-36B-Instruct | raw | `<seed:bos>user\n…<seed:eos><seed:bos>assistant\n` (`seed`) |
| tencent/Hunyuan-A13B-Instruct | raw | `<|startoftext|>{system}<|extra_4|>{user}<|extra_0|>`, no generation header (`hunyuan_moe`) |
| MiniMaxAI/MiniMax-M2 | raw | `]~!b[]~b]system\n…[e~[\n]~b]user\n…[e~[\n]~b]ai\n<think>\n` (`minimax_m2`) |
| nvidia/NVIDIA-Nemotron-Nano-9B-v2 | raw | `<SPECIAL_10>System\n…\n<SPECIAL_11>User\n…\n<SPECIAL_11>Assistant\n<think>\n`, stripped (`nemotron_nano`) |
| nvidia/Nemotron-Mini-4B-Instruct | raw | `<extra_id_0>System\n…\n\n<extra_id_1>User\n…\n<extra_id_1>Assistant\n`, stripped (`nemotron_mini`) |
| microsoft/phi-4 | ChatML | `<|im_start|>user<|im_sep|>…<|im_end|>` (`phi4`) |
| microsoft/Phi-4-mini-instruct | Phi-3's, with newlines | `<|user|>…<|end|><|assistant|>` (`phi4_mini`) |
| stabilityai/stablelm-zephyr-3b | OLMo's | Zephyr's (already a family; detection fixed) |
| LGAI-EXAONE/EXAONE-4.0-1.2B | EXAONE 3.5's | `[|user|]\n…[|endofturn|]\n[|assistant|]\n<think>\n\n</think>\n\n` (`exaone4`) |
| LGAI-EXAONE/K-EXAONE-236B-A23B | OLMo's | `<|user|>\n…<|endofturn|>\n<|assistant|>\n<think>\n` (`k_exaone`) |
| openai/gpt-oss-20b | the system message as a `system` turn | the dated model-identity system block, then the system message as the *developer's* `# Instructions` (`harmony`, rewritten) |
| HuggingFaceTB/SmolLM3-3B | ChatML | a dated `## Metadata` system block with `Reasoning Mode: /think`, no `<|im_end|>` after it (`smollm3`) |
| upstage/Solar-Open-100B | raw | `<|begin|>user<|content|>…<|end|><|begin|>assistant`, after a dated provider prompt (`solar_open`) |
| Qwen/Qwen3.5-397B-A17B | ChatML | ChatML, contents trimmed, `<think>\n` opened (`qwen3_5`) |
| XiaomiMiMo/MiMo-V2-Flash | ChatML | ChatML without newlines between turns, `<think></think>` (`mimo`) |
| deepseek-ai/DeepSeek-V3.1, V3.2-Exp | V3's | V3's, with `</think>` after `<｜Assistant｜>` — thinking off by default (`deepseek_v31`) |
| zai-org/GLM-4.7-Flash | GLM-4's | GLM-4's headers without newlines, `<think>` opened (`glm47`) |
| ai21labs/AI21-Jamba-Reasoning-3B | ChatML | ChatML after `<|startoftext|>`, a thinking instruction before the last user turn, `<think>\n` (`jamba_reasoning`) |

Three templates write today's date (gpt-oss, SmolLM3, Solar Open); the engine
now reads the real clock once and the renderer formats it as each template
does (`2026-09-22`, `22 September 2026`). A generation prompt that opens a
`<think>` block now reaches ditch's existing chain-of-thought handling, which
— like heretic's — detects it in the rendered prompt and closes it with a
response prefix, so studies on these reasoning models measure the answer, as
heretic does. Registry entries whose releases use these formats name them, for
checkpoints that ship no template.

## Bugs 51–53 — three more pre-tokenizers (fixed)

* **51 — Qwen 3.5's regex fell back to GPT-2's, silently.** It is Qwen 2's
  with marks: `[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+` and
  `[^\s\p{L}\p{M}\p{N}]`. None of the classifier's substrings matched and
  the contraction alternative suppressed the warning; `:\n\n` came out as three
  tokens instead of two. A `qwen3_5` kind.
* **52 — K-EXAONE's phrase pattern.** `(?:\p{L}\p{M}*(?: \p{L}\p{M}*)*)+`
  keeps runs of words *joined by single spaces* in one pre-token (`You are a
  helpful assistant` is one), digits are single and punctuation takes at most
  one `[\r\n/]` after it. A `phrase` kind.
* **53 — `Metaspace` with `prepend_scheme: "first"` prepended everywhere.**
  Nemotron-Mini's tokenizer prepends `▁` only at the very start of the input;
  ditch prepended it to every segment between special tokens, so
  `<extra_id_1>User` became `▁User` (a different id).

**Verification (50–53).** The template sweep over 56 releases — every
family in the registry that has a public instruct release — now reports
`ids match` for every one, and the 20-string tokenizer table stays at 0
differences for all of its tokenizers (Qwen 3.5, K-EXAONE and Nemotron-Mini
added). Unit tests hold transformers' own rendering of a system/user prompt for
20 families and of a four-turn conversation for 6.


## Bug 54 — a full-width q/k norm overran a 1024-float stack buffer (fixed)

**Symptom.** `ditch probe` on the first two layers of
`allenai/OLMoE-1B-7B-0924-Instruct` segfaulted (ReleaseFast); the Debug build
panicked with `index out of bounds: index 2048, len 1024` in
`normVecInPlace`.

**Cause.** the q/k norm helper copied its input into `var tmp: [1024]f32`.
Per-head norms fit, but OLMo 2 and OLMoE normalise the *whole* q and k
projection — 16 × 128 = 2048 floats here, 4096 on OLMo-2-1124-7B — so every
such model wrote past the buffer. In ReleaseFast that is silent stack
corruption; the families' fixtures are small enough to fit, and the only
real OLMo 2 run so far was the 1B.

**Fix.** the norm kernels take their statistics before writing, element by
element, so they now run in place and the buffer is gone.

## Bug 55 — a config without a norm epsilon got the generic default, not the family's (fixed)

**Symptom.** with bug 54 fixed, OLMoE ran — 89% wrong at its first layer.
Zeroing the attention output left the error, zeroing the experts left it too;
what both sublayers share is their RMSNorm.

**Cause.** OLMoE's released `config.json` has no `rms_norm_eps`. transformers
then uses `OlmoeConfig`'s default, 1e-5; ditch used its generic default for
RMSNorm families, 1e-6. With OLMoE's small embedding activations that is
enough to change the first layer's output by a factor. An audit of every
registry family against its transformers config class found 35 whose default
differs from ditch's — 1e-5 for most, 1.5625e-7 for GLM-4, 1e-8 for Helium,
1e-12 for BioGPT, 1e-6 for NanoChat — so any release that omits the key would
have run with the wrong epsilon.

**Fix.** a `default_norm_eps` field on the registry entry, filled from each
family's transformers config class; the generic default stays for families
that have none.

## Bug 56 — the same for the RoPE base (fixed)

The same audit over `rope_theta`: fifteen families whose transformers default
is not ditch's 10,000 — Mixtral and Ministral 3 1e6, Llama 4, Cohere, ERNIE
4.5, Mellum, Laguna, FlexOlmo and BitNet 500,000, gpt-oss 150,000, Helium
100,000, SmolLM3 2e6, Apertus 1.2e7, Solar Open 1e6, HunYuan V3 11,158,840.
A `default_rope_theta` field, used when the config has neither `rope_theta`
nor `rope_parameters`.

**Verification (54–56).** OLMoE's first two layers on its unmodified
config, float32: all 3 residuals agree, logits to 1.36e-06 (table above). The
audit — a throwaway test running ditch's own `parseConfig` on a minimal
config for every registered type, against `CONFIG_MAPPING[type]()` — now
reports no difference in the norm epsilon or the RoPE base for any of the 78
families that have a transformers config class. A unit test covers OLMoE's
epsilon and Mixtral's base.


## Bug 57 — AFMoE's μP embedding scale was ignored (fixed)

**Symptom.** the first four layers of `arcee-ai/Trinity-Nano-Preview` (the
one released AFMoE small enough to cut) diverged at layer 0 — the embedding
itself — by a factor of 32. (The reference first needed `pad_token_id: null`
in the cut's config: transformers' `AfmoeConfig` has no such attribute and
its model reads it. ditch ignores the key.)

**Cause.** Trinity's config sets `mup_enabled: true`, and `AfmoeModel`
then multiplies the embeddings by `hidden_size ** 0.5` (32 for 1024). ditch's
`afmoe` entry never read the key, and the fixture generator's spec had no μP
either, so the fixture agreed. `arcee-ai/Trinity-Nano-Preview` was
config-checked in the earlier passes, which cannot see a scale.

**Fix.** `extraAfmoe` sets `embed_scale = sqrt(hidden_size)` when
`mup_enabled`; the generator's `afmoe` spec enables μP and the fixture is
regenerated (it fails without the fix: 331/332).

**Verification.** the table above: all 5 residuals of the first four layers
agree, first-token logits to 7.0e-06 of the range, tokenizer 15 / 15 (after
bugs 39–40) and template ids equal.

---

# Frontier pass: the arithmetic of the frontier families on their real weights

Until now the frontier families had only the config-and-tensor-name check of
the second pass: the architecture parses and every weight is where the loader
looks. None of their arithmetic had been compared with a reference on real
weights, and the Qwen 3.5, Helium and Cohere bugs show why that matters: a
fixture written from the same misreading as the code agrees with it.

## Method

Truncated real checkpoints. `tools/truncate_checkpoint.py REPO K OUT` range-reads
each shard's header, then streams the embedding, the final norm, the LM head
and layers `0..K-1` into one `model.safetensors` (in parallel, neighbouring
tensors merged into one request; nothing held in memory), with
`num_hidden_layers = K` and every per-layer list in config.json cut to `K`.
Quantisation stays exactly as released. `K` is chosen so every layer kind of
the family appears at least once. `--drop mtp.` leaves out the
multi-token-prediction layers; `--rows NAME=FILE` writes a table far larger
than the disk as a sparse region holding only the rows a prompt reads.

The reference is transformers on the CPU in float32 where it has the family,
else the repository's own inference code, and
`tools/probe_reference.py --factory FILE` builds it from a Python file when
`from_pretrained` cannot hold the checkpoint as it is. For hyper-connection
models the script compares what ditch reports, each layer's collapsed block
input, with the reference's own `attn_hc` output. ditch runs streamed
(`DITCH_NO_MMAP=1`), so a quantised MoE decodes one expert at a time.

Machine as before: 4 cores, 15 GiB RAM, ~28 GiB of free disk, torch
2.14.0+cpu, transformers 5.17.0.

This pass runs alongside the continuation of the third pass, so its bugs are
numbered F1, F2, … to keep the two sequences apart.

## Bug F1 — no released DeepSeek V4 could be loaded (fixed)

**Symptom.** Recorded in the second pass as a gap: `deepseek-ai/DeepSeek-V4-Flash`
stopped with `'fp4' expert dtype cannot be dequantised`, and V4.1-Flash with
`F8_E4M3 weights without a weight_scale_inv/weight_scale tensor`. Every
released V4 / V4.1 checkpoint is in DeepSeek's own naming (`embed.weight`,
`layers.N.attn.wq_a.weight`, `layers.N.ffn.experts.E.w1.weight`), its FP8
block scales are F8_E8M0 exponents named `<module>.scale`, and its routed
experts are FP4.

**Cause.** Three gaps, none of which a fixture written in transformers'
spelling could show: the loader knew only transformers' names; the FP8
reader looked only for float `weight_scale_inv` / `weight_scale`; and the
experts' storage (I8 `[out, in / 2]` e2m1 nibble pairs, low nibble first,
with an F8_E8M0 `scale` `[out, in / 32]`) had no decoder — although it is
byte for byte MiMo V2.6's MXFP4 `store_dtype` layout, which ditch already
read.

**Fix.** `deepseek_v4.renameNative` renames the checkpoint's tensors on load
with transformers' own list for `deepseek_v4` (`conversion_mapping.py`), in the
same order, into the `model.`-prefixed spelling the loader reads (V4.1's
engram table `layers.N.engram.embed` becomes `engram_tables.N`). The FP8 reader
takes an F8_E8M0 `<module>.scale` (`2^(byte - 127)`); I8 weights with an
F8_E8M0 scale decode through the MXFP4 store path when `expert_dtype` is
`fp4`, at the top level of config.json (V4) or in `quantization_config`
(V4.1). An export strips `expert_dtype` with `quantization_config`, so the
bf16 result reloads — the new fixture's export test caught that one.

**Regression test.** New fixture `deepseek_v4_native`: the V4 fixture's model
written the way DeepSeek released it — DeepSeek's names, the attention and
shared-expert matrices in FP8 with ue8m0 block scales, the experts in FP4,
`expert_dtype: fp4` — with reference outputs from the dequantised weights;
plus its edit/export/streamed-reload test and a unit test of the name map.
`tools/make_fixture.py`'s inverse names were checked against transformers'
forward renames: every tensor round-trips.

**Verification.** `deepseek-ai/DeepSeek-V4-Flash`, first 4 layers (layers 0-1
sliding window only, layer 2 CSA ratio 4 with its indexer, layer 3 HCA ratio
128; layers 0-2 hash-routed, layer 3 learned routing), 16.4 GB cut. The
reference is `tools/ref_deepseek_v4.py`: transformers' `modeling_deepseek_v4.py`
unmodified, the names through transformers' own `WeightRenaming` objects, FP8
and FP4 dequantised in torch with transformers' `_dequantize_one` arithmetic,
and `DeepseekV4Experts` swapped for a module with the same forward that reads
only the experts a token is routed to (256 experts x 3 matrices a layer do not
fit in 15 GiB as float32). ditch: streamed, peak RSS 5.7 GB.

    $ DITCH_NO_MMAP=1 ditch probe models/dsv4-flash-L4 --prompt ... --raw --residuals --json > p.json
    $ python3 tools/probe_reference.py models/dsv4-flash-L4 p.json --dtype float32 --raw --factory tools/ref_deepseek_v4.py

| prompt | tokens | residuals | first-token logits | greedy |
| --- | :---: | :---: | ---: | :---: |
| "The capital of France is" | match (5) | all 5 agree, worst 2.82e-06 | 6.52e-07 | match |
| "Explain how rainbows form, …" | match (14) | all 5 agree, worst 8.61e-06 | 1.94e-06 | match |
| 317-token passage | match (317) | all 5 agree, worst 1.16e-05 | 2.02e-06 | match |

The long prompt is the one that matters for V4: at 14 tokens the HCA layer
has not emitted a single compressed entry and the 128-token window never
slides; at 317 the CSA layer attends 79 pooled entries (inside `index_topk`
512, where ditch's dense indexer is exact), HCA two, and the window is full.
The whole checkpoint also loads over `hf://` (69187 tensors renamed; 33792
FP4 and 375 FP8 matrices registered, nothing skipped).

## Bug F2 — V4.1's engram hashed Indic tokens into the wrong buckets (fixed)

**Symptom.** The first V4.1 checkpoint to reach the engram code,
`deepseek-ai/DeepSeek-V4.1-Flash`, stopped at load:

    error: the tokenizer-derived compressed vocabulary has 99510 entries but
    engram_compressed_vocab_size is 99092; the engram hashes would not match the tables

The release's `inference/engram.py` derives the same map with `tokenizers`
and asserts the 99092.

**Cause.** The compressed vocabulary keys every token by its normalised text
(`NFKC`, `NFD`, `StripAccents`, `Lowercase`, whitespace rules), and every
engram hash multiplier derives from its size, so an off-by-one there rehashes
the whole table. ditch folds per code point from a generated table, whose
generator (`tools/gen_unicode.py`) dropped only nonspacing marks (`Mn`).
`tokenizers`' `StripAccents` drops every combining mark: `Mn`, `Mc` and `Me`.
Bengali, Devanagari and Tamil vowel signs are `Mc` (`ার` is `র` + U+09BE), so
654 tokens got keys of their own instead of sharing their base letter's.
Dumping ditch's key for all 129280 tokens and diffing it with the release's:
every one of the 654 differences is an `Mc` or `Me` mark, nothing else.
`tools/make_fixture.py`'s `engram_normalize` had the same `!= "Mn"`, so the
fixture agreed with the code; its synthetic vocabulary has no such marks, so
it could not have shown it either way.

**Fix.** Both generators drop every category `M`; `src/unicode_tables.zig` is
regenerated (same Unicode 14.0.0 data, only the fold tables change), and the
engram normalisation test covers `Mc` and `Me` marks. The `deepseek_v41`
fixture regenerates byte-identically.

## DeepSeek V4.1: verified on real weights

**Cut.** V4.1's layer kinds are far apart: 0-1 sliding window only (1 is an
engram layer), 2 the ratio-2 compressed-KV source with the indexer, 3 a
ratio-2 layer that reads layer 2's cache, 20 the ratio-1 source, candidate
source and indexer, 21 a ratio-1 layer reading it. `tools/truncate_checkpoint.py
--layers 0,1,2,3,20,21` keeps those six, renumbered 0-5 with
`kv_source_layer_ids`, `index_source_layer_ids`, `candidate_source_layer_id`
and `engram_layer_ids` renumbered with them. A V4.1 layer holds 7.2 GB of FP4
experts and the engram table is 98 GB of FP8 a layer, so both are `--lazy`:
holes of a 148 GB sparse file, filled by the reference with exactly the 3744
expert tensors and table rows the two prompts (and 4 greedy tokens) use —
12 GB on disk in the end.

**Reference.** transformers has no V4.1, so `tools/ref_deepseek_v41.py` runs
the release's own `inference/model.py` and `engram.py`, unmodified, with the
tilelang kernels replaced by torch functions of the same arithmetic, linears
in float on the dequantised weights (the release quantises every linear's
activations to FP8, which ditch does not simulate, as for every FP8
checkpoint), and lazy experts / table rows. Its config mapping reproduces the
release's own `inference/config.json` on every field but the unused draft
head's.

**Result.**

| run | residuals (7 entries) | first-token logits | argmax, top-5, greedy |
| --- | :---: | ---: | :---: |
| as released | first differs at 2.29e-03 (5 tokens) / 1.02e-03 (14 tokens) | 1.55e-02 / 3.77e-04 | match |
| fake quantisation off on both sides | all agree, worst 4.12e-06 / 5.93e-06 | 2.36e-06 / 2.62e-06 | match |

With V4.1's quantisation-aware rounding off (the FP8 rounding of the window
KV, the FP4 rounding of the compressed latents and of the indexer's q/k),
every layer agrees to 6e-06: embedding, engram, hyper-connections, both
compressed-KV kinds, the shared caches, candidate blocks, sinks, routing,
FP4 experts. With it on, the difference is one flipped rounding: dumping the
pre-rounding window KVs and latents from both sides, layers 0 and 1 agree to
1e-7, and exactly one FP8 code of layer 1's token-0 KV rounds the other way
(an input 1.8e-06 apart on a rounding boundary); every later difference
grows from that code. ditch's rounding is not the cause: fed the reference's
own pre-rounding tensors (364 vectors), ditch's FP8 and FP4 rules give
bit-identical results to the kernels'. A rounding boundary is a
discontinuity no float32 implementation can match across a 1e-6 accumulation
difference; the model is exact everywhere else.

## Kimi-Linear: verified on real weights

`moonshotai/Kimi-Linear-48B-A3B-Instruct`, first 4 layers: layer 0 KDA with
the dense MLP, layers 1-2 KDA with the MoE, layer 3 MLA (no RoPE) with the MoE
(`kda_layers` / `full_attn_layers` are 1-based ids and are cut with the
layers). Routed experts lazy; the reference is transformers'
`modeling_kimi_linear.py` through `tools/ref_lazy_moe.py`: `from_pretrained`
loads everything but the routed experts with transformers' own renames and
conv1d stacking, and the experts module reads expert `e` on demand under
transformers' own `forward`.

| prompt | tokens | residuals | first-token logits | greedy |
| --- | :---: | :---: | ---: | :---: |
| "The capital of France is" | match (5) | all 5 agree, worst 1.95e-06 | 1.87e-06 | match |
| "Explain how rainbows form, …" | match (14) | all 5 agree, worst 2.12e-06 | 2.30e-06 | match |
| 318-token passage | match (318) | all 5 agree, worst 2.56e-06 | 3.13e-06 | match |

# Remaining families on real weights

Asked whether the frontier-size releases can be tested at all on this machine
(4 cores, 15 GiB RAM, ~20 GiB free disk), there were three ways on: stream a
whole release through warp mode for one full-depth pass (hundreds of GB over
the network per pass, no study), rent a machine with the RAM for a full run,
or keep cutting releases to their first layers. The third was taken: it adds a
real-weight check to a new family for a few GB of download each, where one
streamed pass would cost an 850 GB download and check one family. The families
below had only a config check; the frontier session owns DeepSeek V4/V4.1,
Kimi, GLM-5.x, Qwen 3.8, MiMo, MiniMax, gpt-oss, Gemma 4, Mistral 4 and
Llama 4, so those are left out.

Method as in the frontier pass: `tools/truncate_checkpoint.py`, then
`ditch probe --residuals --json` against `tools/probe_reference.py` in float32,
two chat prompts.

## GLM-4.7-Flash (`glm4_moe_lite`): verified (latent-norm epsilon, see Bug F3)

`zai-org/GLM-4.7-Flash`, first 3 layers (layer 0 MLA with the dense MLP,
layers 1-2 MLA with the 64-expert MoE and shared expert), all 64 experts kept.

**Symptom.** Layer 0's output differed by 1.4e-03 of its magnitude, the
logits by 9e-04 of their range; argmax and greedy text agreed. A one-layer cut
with the MLP zeroed still differed by 5.7e-04, so the gap was in attention.

**Cause.** The two MLA latent norms' epsilon. transformers'
`Glm4MoeLiteAttention` builds `q_a_layernorm` and `kv_a_layernorm` as
`Glm4MoeLiteRMSNorm(rank)`, whose epsilon defaults to 1e-6, while the config's
`rms_norm_eps` (1e-5) was what ditch used. With the reference's two latent
norms set to 1e-5, every layer agreed to 2e-06. vLLM's and SGLang's DeepSeek
V2 attention, which serve this model, pass `rms_norm_eps`, so the
serving engines disagree with transformers here.

**Resolution.** The frontier session found the same difference on Kimi K2.5
at the same time (Bug F3 below) and settled it on transformers' value: DeepSeek's own
`modeling_deepseek.py` and transformers use 1e-6, and ditch now does too for
every MLA family but GLM-5.3-Flash. That is the reading kept here as well:
ditch follows transformers, as heretic does. Rerun after Bug F3 against stock
transformers:

| prompt | tokens | residuals | first-token logits | greedy |
| --- | :---: | :---: | ---: | :---: |
| "The capital of France is" | match (17) | all 4 agree, worst 8.96e-07 | 6.56e-07 | match |
| "Explain how rainbows form, …" | match (23) | all 4 agree, worst 6.42e-07 | 1.31e-06 | match |
| before Bug F3 (ditch 1e-5, transformers 1e-6) | match | first differs at layer 1, 1.4e-03 | 9.2e-04 | argmax match |

## ERNIE 4.5 21B-A3B (`ernie4_5_moe`): verified

`baidu/ERNIE-4.5-21B-A3B-PT`, first 3 layers (layer 0 dense, layers 1-2 the
64-expert top-6 MoE with two shared experts), all experts kept. The second
pass's stub could not check the routing (its router was all zeros, so every
expert choice was a tie); the real router does.

| prompt | tokens | residuals | first-token logits | greedy |
| --- | :---: | :---: | ---: | :---: |
| "The capital of France is" | match (19) | all 4 agree, worst 1.75e-06 | 5.76e-07 | match |
| "Explain how rainbows form, …" | match (26) | all 4 agree, worst 1.20e-06 | 4.45e-07 | match |

## Hunyuan-A13B (`hunyuan_v1_moe`): verified

`tencent/Hunyuan-A13B-Instruct`, first 2 layers (64-expert top-8 MoE with a
shared MLP, q/k norms after the rope), routed experts lazy through
`tools/ref_lazy_moe.py`: the first reference run fills the experts the prompts
route to, then both sides are run again on the filled file.

| prompt | tokens | residuals | first-token logits | greedy |
| --- | :---: | :---: | ---: | :---: |
| "The capital of France is" | match (14) | all 3 agree, worst 6.36e-07 | 7.67e-07 | match |
| "Explain how rainbows form, …" | match (20) | all 3 agree, worst 6.51e-07 | 8.04e-07 | match |

## Seed-OSS-36B (`seed_oss`): verified

`ByteDance-Seed/Seed-OSS-36B-Instruct`, first 2 layers (dense, GQA with q/k/v
biases and an o_proj without one).

| prompt | tokens | residuals | first-token logits | greedy |
| --- | :---: | :---: | ---: | :---: |
| "The capital of France is" | match (23) | all 3 agree, worst 1.48e-06 | 1.50e-06 | match |
| "Explain how rainbows form, …" | match (29) | all 3 agree, worst 1.37e-06 | 1.58e-06 | match |
## Bug F3 — MLA's latent norms took `rms_norm_eps` (fixed)

**Symptom.** `moonshotai/Kimi-K2.5`, first 2 layers (the dense layer and the
first INT4 MoE layer): argmax, top-5 and greedy text matched transformers, but
every residual was off by 5e-05 to 9e-05 of its magnitude from the *first*
layer on, fifty times the usual float32 agreement. A float32 transformers
reference agrees with a float64 one to 4e-08 on the same layer, so this was
ditch's.

**Bisection.** Zeroing layer 0's dense `down_proj` on both sides left the
error in place (attention); a one-token prompt, where RoPE is the identity and
attention returns the single value, still showed 8.9e-05 (not RoPE, not the
softmax: the value path). Recomputing that path by hand in float64 from the
checkpoint reproduced ditch to 1.1e-07 with ε = 1e-5 everywhere, and
reproduced the reference exactly with ε = 1e-6 on `kv_a_layernorm`.

**Cause.** DeepSeek's own `modeling_deepseek.py` (which Kimi ships), and
transformers after it, build `q_a_layernorm` and `kv_a_layernorm` as
`RMSNorm(q_lora_rank)` / `RMSNorm(kv_lora_rank)`: the class default 1e-6,
*not* `rms_norm_eps`. ditch used `rms_norm_eps` for both. DeepSeek V3 itself
sets `rms_norm_eps: 1e-6`, so it never showed; Kimi K2 / K2.5, Kimi-Linear,
Kimi K3, GLM-4.7-Flash, GLM-5.3 and altar-1 set 1e-5. Every MLA family in
transformers follows the default except GLM-5.3-Flash (`glm5_next`), which
passes `rms_norm_eps`. `tools/make_fixture.py` used `eps` too, and its
random-weight latents are large enough that the two epsilons differ by less
than the fixture tolerance, so no fixture could see it.

**Fix.** `Mla.latent_norm_eps`, 1e-6, set to `rms_norm_eps` by `glm5_next`,
used by `mlaProject`. The generator uses 1e-6 for the latent norms; the five
MLA fixtures with `rms_norm_eps` ≠ 1e-6 are regenerated (`glm4_moe_lite`,
`kimi_linear`, `kimi_linear_hf`, `kimi_k3`, `kimi_k3_mxfp4`; every other
fixture regenerates byte-identically). New fixture `deepseek_v3_latent_eps`:
`rms_norm_eps` 1e-5 with the latent projections scaled so the latents' mean
square is ~1e-6; it fails with the old code and passes with the new. A config
test pins 1e-6 for a DeepSeek V3 config that says 1e-5.

**A second, reference-side correction.** After the fix the MoE layer still
differed by 2e-05. The reference dequantised the INT4 experts in float32, but
compressed-tensors' `_dequantize` computes `x_q.to(scale.dtype) * scale` in
the scale's dtype, bf16 here — which is what ditch reproduces. With
`tools/ref_lazy_moe.py` calling compressed-tensors' own `_dequantize`:

| prompt | tokens | residuals | first-token logits | greedy |
| --- | :---: | :---: | ---: | :---: |
| "The capital of France is" | match (5) | all 3 agree, worst 3.53e-07 (was 4.90e-05) | 5.95e-07 (was 1.78e-04) | match |
| "Explain how rainbows form, …" | match (14) | all 3 agree, worst 2.32e-07 (was 9.10e-05) | 6.36e-07 (was 2.28e-04) | match |

The Kimi-Linear numbers above were measured before this fix: its MLA layer's
latents are large enough that ε moved nothing past 2.6e-06.

## Qwen3-Next-80B-A3B (`qwen3_next`): verified

`Qwen/Qwen3-Next-80B-A3B-Instruct`, first 4 layers (three gated DeltaNet
layers, then gated full attention; every layer the 512-expert top-10 MoE with
a gated shared expert), MTP layers dropped, routed experts lazy. transformers
runs its PyTorch fallbacks for the causal conv and the chunked delta rule.

| prompt | tokens | residuals | first-token logits | greedy |
| --- | :---: | :---: | ---: | :---: |
| "The capital of France is" | match (24) | all 5 agree, worst 3.23e-07 | 5.15e-07 | match |
| "Explain how rainbows form, …" | match (30) | all 5 agree, worst 1.68e-07 | 5.90e-07 | match |

## dots.llm1 (`dots1`): verified

`rednote-hilab/dots.llm1.inst`, first 2 layers (layer 0 dense, layer 1 the
128-expert top-6 sigmoid MoE with correction bias and a shared expert), routed
experts lazy.

| prompt | tokens | residuals | first-token logits | greedy |
| --- | :---: | :---: | ---: | :---: |
| "The capital of France is" | match (16) | all 3 agree, worst 8.53e-07 | 1.40e-06 | match |
| "Explain how rainbows form, …" | match (22) | all 3 agree, worst 2.29e-07 | 1.34e-06 | match |

A caution on the tolerance. The first run, with ditch reading the routed
experts as zeros (holes not yet filled), also passed the 1e-3 check: with every
routed expert zeroed, layer 1's output moves by only 4.5e-04 of the residual's
largest entry, which one massive-activation channel dominates. The result above
still verifies the experts, with the difference ~500 times below that, but a
pass at 1e-3 alone would not have: for a short cut of a model with massive
activations, read the worst residual difference, not only the verdict.

## Solar-Open-100B (`solar_open`): verified

`upstage/Solar-Open-100B`, first 2 layers (both the 128-expert top-8 sigmoid
MoE with a shared expert; `first_k_dense_replace` is 0), MTP dropped, routed
experts lazy. The chat template's long default system block makes both prompts
80+ tokens.

| prompt | tokens | residuals | first-token logits | greedy |
| --- | :---: | :---: | ---: | :---: |
| "The capital of France is" | match (83) | all 3 agree, worst 1.81e-06 | 1.09e-06 | match |
| "Explain how rainbows form, …" | match (88) | all 3 agree, worst 2.97e-06 | 9.83e-07 | match |

## GLM-4.5-Air (`glm4_moe`): verified

`zai-org/GLM-4.5-Air`, first 2 layers (layer 0 dense, layer 1 the 128-expert
top-8 sigmoid MoE with correction bias and a shared expert; partial rotary,
q/k/v biases), routed experts lazy.

| prompt | tokens | residuals | first-token logits | greedy |
| --- | :---: | :---: | ---: | :---: |
| "The capital of France is" | match (18) | all 3 agree, worst 6.51e-07 | 8.74e-07 | match |
| "Explain how rainbows form, …" | match (24) | all 3 agree, worst 8.26e-07 | 8.97e-07 | match |

## EXAONE 4.0.1 32B (`exaone4`): verified

`LGAI-EXAONE/EXAONE-4.0.1-32B`, first 4 layers (three sliding-window layers
with RoPE, then a global layer without it; post-norms and q/k norms).

| prompt | tokens | residuals | first-token logits | greedy |
| --- | :---: | :---: | ---: | :---: |
| "The capital of France is" | match (27) | all 5 agree, worst 5.93e-07 | 1.34e-06 | match |
| "Explain how rainbows form, …" | match (33) | all 5 agree, worst 1.01e-06 | 7.57e-07 | match |

## Granite 3.3 2B (`granite`): verified

`ibm-granite/granite-3.3-2b-instruct`, first 4 layers (embedding, residual,
attention and logit multipliers).

| prompt | tokens | residuals | first-token logits | greedy |
| --- | :---: | :---: | ---: | :---: |
| "The capital of France is" | match (24) | all 5 agree, worst 5.45e-07 | 1.31e-06 | match |
| "Explain how rainbows form, …" | match (30) | all 5 agree, worst 2.41e-07 | 7.58e-07 | match |

## Kimi K3: verified on real weights (against its own code)

`moonshotai/Kimi-K3`, first 5 layers: layer 0 KDA with the dense MLP, layers
1, 2 and 4 KDA with the latent MoE, layer 3 MLA; routed experts MXFP4 and
lazy. transformers has no K3, so the reference is the release's own
`modeling_kimi_linear.py` (`tools/ref_kimi_k3.py`), unmodified except that
its fla (Triton) entry points are replaced by fla's own torch references of
the same maths (`naive_kda_lowerbound_gate`, fla's L2 norm, sigmoid beta,
`naive_recurrent_kda`; causal depthwise conv + SiLU; `rmsnorm · w ·
sigmoid(g)`), plus two transformers-5 import shims. The trunk stays in its
stored bf16 and every Linear computes in float32 in row blocks, which is
float32 arithmetic on the same values in 15 GiB.

Two deviations, both recorded in the scripts:

* **`attn_res_block_size` is 2 in the cut's config** (12 as released), so the
  Attention Residual bank gains blocks at layers 0, 2 and 4 inside five
  layers; with 12 no block boundary after the first falls in a cut that fits.
  Both sides read the same config.
* **`A_log` is stored with 128 entries for 96 heads.** The release's own code
  declares `A_log` as `[num_heads]` and cannot load its own checkpoint as it
  is; ditch takes the first 96 (the heads, padded to 128 for the kernels, is
  the likely reading), and the reference does the same. No released loader
  was available here to confirm it, so this reading is an assumption, not a
  verified fact.

| prompt | tokens | residuals | first-token logits | greedy |
| --- | :---: | :---: | ---: | :---: |
| "The capital of France is" | match (5) | all 6 agree, worst 9.06e-07 | 9.22e-07 | match |

(Only one prompt: a second, 14-token one routed to enough 896-expert MXFP4
experts to fill the disk before it finished.)

## Bug 58 — EXAONE MoE's global layers were roped (fixed)

**Symptom.** `LGAI-EXAONE/K-EXAONE-236B-A23B`, cut to layers 0 and 3 (the
dense sliding layer and the first global MoE layer, so every layer kind is
present; one MoE layer's experts, 9.6 GB, is what the disk holds): the sliding
layer agreed, the global layer's output differed by 1.8e-02 of its magnitude,
the logits by 2.8e-02 of their range.

**Cause.** EXAONE MoE, like EXAONE 4, has RoPE on its sliding layers only;
transformers applies it `if self.sliding_window is None or self.is_sliding`.
`extraExaone4` sets `rope_layers` from `sliding_layers`, `extraExaoneMoe` did
not, so every global layer was roped. It also read `sliding_window_pattern` as
an integer, while the release writes `"LLLG"`. The fixture was generated from
the same misreading (its spec gave `sliding_layers` and no `rope_layers`), so
it agreed.

**Fix.** `extraExaoneMoe` takes EXAONE 4's layer kinds (pattern string or
integer, RoPE on the sliding layers only); the fixture spec now has
`rope_layers=[1, 1, 1, 0]` and was regenerated; a parse test checks both the
`layer_types` and the pattern spelling.

## Bug 59 — K-EXAONE's router bias was never loaded (fixed)

**Symptom.** After bug 58, the global MoE layer still differed by ~1e-03, the
same with every one of its 128 experts filled (so not a lazy hole), and the
difference was proportional to neither the attention nor the MoE output.

**Cause.** The release stores the router's correction bias as
`mlp.e_score_correction_bias`; transformers renames it to
`mlp.gate.e_score_correction_bias` on load, which is the name ditch looked
for. A router may lack the bias, so a missing one was not an error: the name
did not resolve and ditch routed on the raw sigmoid scores, choosing some
experts other than the model does.

**Fix.** `loadCorrectionBias` in `src/moe.zig` tries the family's name, then
the other spelling (`.gate.e_score_correction_bias` ↔
`.e_score_correction_bias`), for every family. The exaone_moe fixture now
writes the release's spelling; before the fix its test fails (logits off by
16), after it passes.

| K-EXAONE, layers 0 and 3 | tokens | residuals | first-token logits | greedy |
| --- | :---: | :---: | ---: | :---: |
| "The capital of France is" | match (22) | all 3 agree, worst 1.23e-06 | 8.29e-07 | match |
| "Explain how rainbows form, …" | match (26) | all 3 agree, worst 1.27e-06 | 6.65e-07 | match |

## Qwen3.5-35B-A3B (`qwen3_5_moe`): verified

`Qwen/Qwen3.5-35B-A3B`, first 4 layers (three gated DeltaNet layers, then
gated full attention; every layer the MoE with a gated shared expert). This
release stores each layer's experts as two stacked tensors, so they cannot be
made lazy, and the 4-layer cut with the vision tower is 9.7 GB (19 GB in
float32). It was rewritten as a text-only checkpoint
(`Qwen3_5MoeForCausalLM`, `model.language_model.` → `model.`, vision tower and
MTP dropped) keeping experts 0-63 of 256 with the matching router rows and
`num_experts: 64`, as for Mellum: both sides run the same smaller model, with
real weights in every tensor.

| prompt | tokens | residuals | first-token logits | greedy |
| --- | :---: | :---: | ---: | :---: |
| "The capital of France is" | match (26) | all 5 agree, worst 5.19e-07 | 7.34e-07 | match |
| "Explain how rainbows form, …" | match (32) | all 5 agree, worst 3.84e-07 | 6.78e-07 | match |
## GLM-5.3: verified on real weights

`zai-org/GLM-5.3`, first 4 layers (three dense, one MoE; MLA with the DSA
indexer, which ditch runs as its dense equivalent inside `index_topk`), FP8
throughout (128 x 128 blocks), routed experts lazy. `rms_norm_eps` is 1e-5, so
this is also the first release run after bug F3. The reference is transformers'
`modeling_glm_moe_dsa.py` through `tools/ref_lazy_moe.py`, trunk in bf16 with
float32 arithmetic and F32-stored tensors restored exactly.

Two reference-side corrections on the way, recorded because a reference that
is wrong looks exactly like a ditch bug:

* transformers' `FineGrainedFP8Config(dequantize=True)` on the CPU loaded the
  FP8 codes as bf16 *without their scales* (every weight up to 448); the
  reference now dequantises FP8 itself.
* The reference's first FP8 dequantiser took the block height as
  `ceil(rows / scale_rows)`: 116 for `kv_a_proj_with_mqa`'s 576 rows in 5
  scale rows, where the convention (and ditch) is `weight_block_size` 128 with
  a partial last block. Dumping ditch's `kv_a` output against a float64
  recomputation pinned it: rows 116-127, 232-255, 348-383 and 464-511 — the
  tails of each 128-row block — were the reference's, scaled by the next
  block's scale.

| prompt | tokens | residuals | first-token logits | greedy |
| --- | :---: | :---: | ---: | :---: |
| "The capital of France is" | match (5) | all 5 agree, worst 2.66e-07 | 4.10e-07 | match |
| "Explain how rainbows form, …" | match (15) | all 5 agree, worst 4.16e-07 | 4.94e-07 | match |

## AikidoSec/altar-1: verified on real weights

The REAP-pruned GLM-5.3 (168 routed experts, mixed BF16 / INT4). Its first
three layers are dense BF16 and layer 3 is one of the three MoE layers kept in
BF16 (3, 77, 78: the `ignore` list), so a 4-layer cut would exercise no INT4;
the cut is 5 layers, and layer 4 carries compressed-tensors pack-quantized
INT4 (group 32, *asymmetric*: `weight_zero_point`) in `q_b_proj`, `kv_b_proj`,
`o_proj` and every routed expert. The reference dequantises with
compressed-tensors' own `unpack_from_int32` and `_dequantize` (in the scale's
dtype, bf16), everything else as for GLM-5.3.

| prompt | tokens | residuals | first-token logits | greedy |
| --- | :---: | :---: | ---: | :---: |
| "The capital of France is" | match (5) | all 6 agree, worst 2.66e-07 | 4.72e-07 | match |
| "Explain how rainbows form, …" | match (15) | all 6 agree, worst 4.16e-07 | 5.62e-07 | match |

Warp mode over `hf://` with the chunk cache bounded to this disk:

    $ ditch --dry-run hf://AikidoSec/altar-1 --max-ram 14GB --remote-cache-size 18GB --remote-chunk-size 2MB --no-input
      weights total            932.86GB (78 layers)
      trunk per layer          746.8MB (largest; routed experts excluded)
      routed expert            72.0MB each, 168 per layer, top-8 per token, 885.94GB in total
      expert cache             7.68GB (holds 109 of 12600 experts)
      warp mode:     min 12.38GB (trunk + top-8 experts + workspace), with prefetch + expert cache + RAM caches 80.11GB
    Disk estimate (remote chunk cache, 2.0MB chunks):
      trunk                    26.68GB stored, 27.12GB of chunks (re-read by every forward pass)
      routed expert            72.0MB each stored, 12600 experts, 23.63GB in total
      ...
    Dry run: ... stopping here (exit 0).   process RSS peak 198.7MB, 5m27s

That disk estimate missed every dequantised tensor's stored bytes (Bug F7
below). Rerun after the fix (default 8 MB chunks):

    $ ditch --dry-run hf://AikidoSec/altar-1 --max-ram 14GB --remote-cache-size 18GB --no-input
      (memory estimate unchanged)
    Disk estimate (remote chunk cache, 8.0MB chunks):
      trunk                    32.53GB stored, 33.89GB of chunks (re-read by every forward pass)
      routed expert            72.0MB each stored, 12600 experts, 272.89GB in total
      chunk cache bound        18.00GB, 2.30GB cached now, 18.34GB free on its filesystem
      too small for the trunk: every forward pass fetches it again (--remote-cache-size 34.00GB holds it)
    Dry run: ... stopping here (exit 0).   process RSS peak 218.7MB, 7m48s

So on a 16 GB machine altar-1 needs 12.4 GB resident and, to avoid
re-fetching the trunk on every forward pass, 34 GB of cache disk (305 GB
stored in all); with the 18 GB that fits here it runs, fetching the trunk
each pass.

## GLM-5.3-Flash: verified on real weights (within the bar, not to 1e-6)

`zai-org/GLM-5.3-Flash`, first 4 layers: three Kimi Delta Attention layers
with the dense MLP, then a NoPE MLA layer with the DSA indexer and the first
MoE layer; mHC hyper-connections (4 streams) around every block, FP8 with
128 x 128 blocks. Its `linear_attn_config` layer lists are 0-based, where
Kimi's are 1-based, which `tools/truncate_checkpoint.py` now tells apart.
Reference: transformers' `modeling_glm5_next.py` through
`tools/ref_lazy_moe.py`, with flash-linear-attention hidden from transformers
(installed for the K3 reference, its Triton kernels cannot run on a CPU) so
the torch paths run, and F32-stored tensors restored under transformers'
renamed names (`hc_attn_base` is `attn_hc.base`; restoring by checkpoint name
first left them bf16-rounded in the reference, a 1.7e-04 error at entry 0).

| prompt | tokens | residuals | first-token logits | greedy |
| --- | :---: | :---: | ---: | :---: |
| "The capital of France is" | match (5) | all 5 agree, worst 2.48e-05 | 2.98e-06 | match |
| "Explain how rainbows form, …" | match (15) | all 5 agree, worst 4.20e-05 | 1.67e-05 | match |

This is the one family of the pass that agrees to 1e-5 rather than 1e-6: the
entries grow 1.3e-06 → 8.8e-06 → 2.5e-05 → 4.2e-05 over the KDA layers.
Recomputing layer 1 in float64 with transformers' own module puts the float32
reference within 3.9e-07 of it and ditch within 4.6e-06, so the excess is
ditch's, and small: every epsilon (the hyper-connection RMS, the L2 norm of q
and k, the gated output norm, the latent norms) and the lower-bound gate
match the reference. Left open, 25x inside the 1e-3 bar.

## Bug F4 — Qwen 3.8's attention gate was a SiLU (fixed)

**Symptom.** `Qwen/Qwen3.8-27B`, first 4 layers (three Gated DeltaNet, one
gated full attention): the three linear-attention layers agreed with
transformers, and layer 3's output was off by 1.77 of its own magnitude, the
first-token argmax different.

**Cause.** Qwen 3.8 is the first `qwen3_5` / `qwen3_5_moe` release whose
config sets `output_gate_type` (`"swish"`; 3.5 and 3-Next leave it unset).
ditch's generic parser read the key as the kind of the *attention* output gate
and ran it as `x * sigmoid(x)`. In Qwen's configs the key names the Gated
DeltaNet's output gate (SiLU there already, and the one `qwen4_exp` reads,
correctly, as `linear_gate_sigmoid`); `Qwen3_5Attention` multiplies its output
by `sigmoid(gate)` unconditionally, and so does every other family with a
gated query in transformers. The fixture generator's `qwen3_5_moe` spec had
`gate_swish=True`, so the fixture agreed with the code. Every Qwen 3.8 dense
and Max (2.4T-A95B) checkpoint was affected.

**Fix.** The attention gate is a sigmoid, always: `Config.gate_swish` and its
branch are gone, and so is the generator's; the `qwen3_5_moe` fixture is
regenerated (its reference outputs change, so it fails against the old code).

**Verification.** `Qwen/Qwen3.8-27B`, first 4 layers, float32 (reference:
transformers through `tools/ref_lazy_moe.py`, bf16 storage with float32
arithmetic):

| prompt | tokens | residuals | first-token logits | greedy |
| --- | :---: | :---: | ---: | :---: |
| "The capital of France is" | match (5) | all 5 agree, worst 7.24e-07 (was 1.77e+00) | 6.26e-07 (was 1.05e+00) | match |
| "Explain how rainbows form, …" | match (15) | all 5 agree, worst 6.68e-07 (was 1.72e+00) | 7.54e-07 (was 9.65e-01) | match |

## Qwen3.8-2.4T-A95B (`qwen3_5_moe`): verified on real weights

`Qwen/Qwen3.8-2.4T-A95B`, first 4 layers (three Gated DeltaNet, one gated full
attention, every layer a 512-expert MoE with a shared expert), after Bug F4.
The routed experts are stacked `[512, 4096, 8192]` / `[512, 8192, 2048]`
tensors (32 GB and 16 GB a layer), left lazy in the cut and read one expert
slab at a time (96 MB an expert) by `tools/ref_lazy_moe.py`; the 12 GB trunk
is real. The disk allows only the experts a 2-token prompt routes to (7.3 GB
over the four layers), so this is one short prompt, one generated token:

| prompt | tokens | residuals | first-token logits | greedy |
| --- | :---: | :---: | ---: | :---: |
| "Paris is" | match (2) | all 5 agree, worst 2.65e-07 | 8.96e-07 | match (1 token) |

## DeepSeek V3.2-Exp (`deepseek_v32`): layer-0 drift traced to the reference

`deepseek-ai/DeepSeek-V3.2-Exp`, layers 0 and 3 (`--layers 0,3`: the dense
MLA layer and the first MoE layer, FP8 block-quantised, routed experts lazy).
`index_topk` is 2048, so for these prompts the lightning indexer keeps every
key and the reference's sparse attention is dense, as ditch runs it. ditch
runs streamed (`DITCH_NO_MMAP=1`); mapped, it dequantised every expert at load
and ran out of memory.

**transformers cannot load the cut.** Its FP8 converter refuses
`kv_a_proj_with_mqa` (`Weight shape (576, 7168) not divisible by scale grid
(5, 56)`): 576 rows are four full 128-row blocks and a half one. The reference
therefore reads a view of the cut whose FP8 trunk is dequantised beforehand.

**Symptom.** Against that view, layer 0's output differed by 1.6e-01.

**Ruled out, in order.** Attention against MLP: ditch's attention output was
0.83 times the reference's, uniformly; the MLP difference followed from it.
YaRN `mscale`, NeoX against interleaved rope and the latent-norm epsilon each
moved the gap by under 3e-03. The indexer: transformers' `deepseek_v3` on the
same view gives exactly the same output as `deepseek_v32`. ditch on the view
itself (trunk already float) matched the view, so the difference was in the
FP8 trunk. Tensor by tensor, with only one kept FP8 at a time, every one
matched, but only because the view had dropped `quantization_config`: without
`weight_block_size`, ditch derives the block size from the scale grid, as the
view's own dequantisation did.

**Cause: the reference view.** Deriving the block from the grid gives
ceil(576 / 5) = 116-row blocks for `kv_a_proj_with_mqa`; the release's blocks
are 128 rows (`weight_block_size`), the last one partial. ditch reads
`weight_block_size` when the config has it and gets 128; the view got 116 and
mis-scaled the latent and rope-key rows. Not a ditch bug. The view now takes
the block size from the config.

**A second, smaller difference.** `o_proj`'s scales are not powers of two
(despite `scale_fmt: ue8m0`), and ditch rounds every dequantised weight to
bf16 (dequant.zig, as the Hugging Face integrations do when they dequantise
to bf16), so against a float32 dequantisation layer 0 moves by 5.5e-04. The
view rounds to bf16 as well, so the comparison measures the rest.

**Result**, with the view's blocks taken from `weight_block_size`:

| prompt | tokens | residuals | first-token logits | greedy |
| --- | :---: | :---: | ---: | :---: |
| "The capital of France is" | match (15) | all 3 agree, worst 1.51e-07 | 3.70e-07 | match |
| "Explain how rainbows form, …" | match (20) | all 3 agree, worst 1.93e-07 | 3.69e-07 | match |

(`tools/ref_lazy_moe.py` already takes the experts' block size from the config
and rounds them to bf16, so the lazy experts were not affected.)
## Qwen3.8-Flash-Next (`qwen4_exp`): verified on real weights

`Qwen/Qwen3.8-Flash-Next`, first 4 layers: three Gated DeltaNet layers and one
full-attention layer with the QSA indexer, 4 hyper-connection streams (hidden
2560), every layer a 512-expert MoE with a shared expert, and the per-layer
n-gram embedding (PLE) of layer 1: 128 shards of `[2500012, 160]`, 102 GB.
Reference: transformers' `qwen4_exp` through `tools/ref_lazy_moe.py`, with the
routed experts read one slab at a time and the n-gram table one row at a time
(`LazyRows` in place of the concatenated `nn.Embedding`; 5 MB of it read over
both prompts). `tools/probe_reference.py` now takes Qwen4-Exp's
`attn_hyper_connection` collapse as each layer's residual and its
`hyper_connection_mixer` as the final one (the model has no final norm).

| prompt | tokens | residuals | first-token logits | greedy |
| --- | :---: | :---: | ---: | :---: |
| "The capital of France is" | match (5) | all 5 agree, worst 3.69e-06 | 7.17e-07 | match |
| "Explain how rainbows form, …" | match (15) | all 5 agree, worst 1.66e-05 | 2.58e-06 | match |

## Bug 60 — GPT-NeoX's MLP read the attention's input norm (fixed)

**Symptom.** `EleutherAI/pythia-160m`, whole model, `--raw`: layer 0's output
differed by 3.9e-01 of its magnitude and the argmax disagreed. With the MLP's
output projection zeroed on both sides, attention agreed to 3.5e-07; with
the attention's zeroed, the rest still differed by 6.3e-01.

**Cause.** In ditch's parallel-residual path the MLP reads the layer's
`mlp_norm` when the family names one and otherwise the attention's
`input_layernorm` output: the GPT-J / Phi / Cohere / StableLM layout, one norm
for both branches. GPT-NeoX has a norm per branch,
`x + attn(input_layernorm(x)) + mlp(post_attention_layernorm(x))`, but
`neox_style_names` named no `mlp_norm`, so `post_attention_layernorm` was
loaded as the unused `pre_ff_norm` and every Pythia / GPT-NeoX MLP was fed the
wrong normalisation. The fixture was written from the same reading and
agreed. (Falcon 40B's `ln_mlp` already takes the `mlp_norm` path; StableLM's
parallel layers have no second norm.)

**Fix.** `neox_style_names` names `post_attention_layernorm` as `mlp_norm`;
the fixture spec does the same and was regenerated, and fails on the old code
(logits off by 2.4).

| model | tokens | residuals | first-token logits | greedy |
| --- | :---: | :---: | ---: | :---: |
| pythia-160m, "The capital of France is" | match (5, raw) | all 13 agree, worst 5.08e-05 | 8.15e-07 | match |
| pythia-160m, "Explain how rainbows form, …" | match (11, raw) | all 13 agree, worst 9.25e-05 | 1.18e-06 | match |

The residual bar is looser here than elsewhere because Pythia's residual
stream grows to large magnitudes in its last layers; the logits agree to 1e-06.

## Bug 61 — Llama 3.1 / 3.2 / 3.3 were prompted without their dated system block (fixed)

**Symptom.** `unsloth/Llama-3.2-1B-Instruct` (the ungated copy of Meta's
release), whole model: the forward pass agreed to 3e-06 on ditch's ids, but the
ids were 20 tokens short of transformers'.

**Cause.** From Llama 3.1 on, the chat template always writes a system block
that opens `Cutting Knowledge Date: December 2023\nToday Date: <date>\n\n`,
before the system message and even when there is none; 3.1 and 3.3 date it
`26 Jul 2024`, 3.2 with `strftime_now("%d %b %Y")`. ditch had one `llama3`
template, Llama 3.0's, with no such block. The template sweep had used a
Llama 3.0-style release for the family, so it did not show.

**Fix.** Two templates, `llama31` (fixed date) and `llama32` (today's), detected
by `Cutting Knowledge Date` (and `strftime_now`) in the release's template;
rendering expectations for both are transformers' own output.

| model | tokens | residuals | first-token logits | greedy |
| --- | :---: | :---: | ---: | :---: |
| Llama-3.2-1B-Instruct, "The capital of France is" | match (46) | all 17 agree, worst 4.48e-06 | 1.48e-06 | match |
| Llama-3.2-1B-Instruct, "Explain how rainbows form, …" | match (52) | all 17 agree, worst 2.51e-06 | 1.13e-06 | match |

## Bug 62 — no BLOOM release could be loaded: `n_embed` (fixed)

**Symptom.** `bigscience/bloomz-560m` stopped at `error: InvalidConfig`.

**Cause.** The BLOOM releases spell the hidden size `n_embed` (GPT-2's is
`n_embd`); transformers' `BloomConfig` takes it as a backward-compatible
keyword. ditch read `hidden_size`, `n_embd` and `d_model`, found none and got
a hidden size of 0. The fixture uses `hidden_size`.

**Fix.** `n_embed` is read as well; a parse test covers the release spelling.

| model | tokens | residuals | first-token logits | greedy |
| --- | :---: | :---: | ---: | :---: |
| bloomz-560m, "The capital of France is" | match (5, raw) | all 25 agree, worst 4.76e-07 | 5.32e-07 | match |
| bloomz-560m, "Explain how rainbows form, …" | match (12, raw) | all 25 agree, worst 6.99e-07 | 3.85e-07 | match |

## Gemma 2 (`gemma2`): verified; transformers' default attention drops the softcap

`unsloth/gemma-2-2b-it` (the ungated copy of Google's release), first 4 layers.

**Symptom.** Layer 0's output differed by 3.5e-03 of its magnitude.

**Cause: the reference.** transformers loads Gemma 2 with `sdpa` attention by
default, and that path does not apply `attn_logit_softcapping` (50): with the
reference's softcap removed its output did not change at all. Loaded with
`attn_implementation="eager"`, which applies it, layer 0 agrees to 5.9e-07.
ditch applies the softcap, as the model was trained. `tools/probe_reference.py`
now loads eager attention for any config with `attn_logit_softcapping`.
(heretic loads models with transformers' default attention, so on Gemma 2 it
runs without the attention softcap.)

**Prompt.** Gemma 2's template raises on a system message; transformers'
reference retries without it, ditch puts the system prompt at the head of the
first user turn, as Gemma 3's template does. So the ids differ by the system
prompt by design, and the forward pass is compared on ditch's ids.

| prompt | tokens | residuals | first-token logits |
| --- | :---: | :---: | ---: |
| "The capital of France is" | system prompt merged (see above) | all 5 agree, worst 8.57e-07 | 7.39e-07 |
| "Explain how rainbows form, …" | same | all 5 agree, worst 6.09e-07 | 5.73e-07 |

## The older families, in float32 against transformers

The first pass compared these families on first-token logits in bf16, or not
at all. `probe --residuals` against float32 transformers, whole model where it
fits (N = all layers), else the first N layers; `raw` = a base model probed
with `--raw`. Bugs 60-62 were found here.

| checkpoint | family | N | tokens | residuals (worst) |
| --- | --- | ---: | :---: | ---: |
| openai-community/gpt2 | `gpt2` | 12 (all) | match (raw) | 1.27e-06 |
| bigcode/tiny_starcoder_py | `gpt_bigcode` | 20 (all) | match (raw) | 3.87e-07 |
| EleutherAI/pythia-160m | `gpt_neox` | 12 (all) | match (raw) | 9.25e-05 (after bug 60) |
| bigscience/bloomz-560m | `bloom` | 24 (all) | match (raw) | 6.99e-07 (after bug 62) |
| AntonV/mamba2-130m-hf | `mamba2` | 24 (all) | match (raw) | 1.88e-06 (see below) |
| allenai/OLMo-1B-hf | `olmo` | 16 (all) | match (raw) | 2.19e-06 |
| allenai/OLMo-2-0425-1B-Instruct | `olmo2` | 16 (all) | match | 7.07e-07 |
| unsloth/Llama-3.2-1B-Instruct | `llama` | 16 (all) | match (after bug 61) | 4.48e-06 |
| unsloth/gemma-3-1b-it | `gemma3_text` | 8 | match | 9.50e-07 |
| unsloth/gemma-2-2b-it | `gemma2` | 4 | system merged (above) | 8.57e-07 |
| stabilityai/stablelm-2-1_6b-chat | `stablelm` | 24 (all) | match | 4.67e-06 |
| microsoft/phi-2 | `phi` | 4 | match (raw) | 5.80e-07 |
| microsoft/Phi-3.5-mini-instruct | `phi3` | 3 | match | 1.60e-06 |
| HuggingFaceTB/SmolLM3-3B | `smollm3` | 4 | match | 9.82e-07 |
| bigcode/starcoder2-3b | `starcoder2` | 3 | match (raw) | 2.09e-06 |
| mistralai/Mistral-7B-Instruct-v0.3 | `mistral` | 3 | match | 7.13e-06 |

First-token logits agree to 6e-06 of their range or better in every row.

**Mamba2 and the reference.** transformers' Mamba, Mamba2 and FalconMamba
record each block's *output* in `hidden_states`, with no embedding entry, then
the normalised last state, so the list is shifted by one against every other
family's. `tools/probe_reference.py` now puts the embedding first for those
families and drops the normalised entry; before that, every Mamba2 residual
looked wrong although the logits agreed to 3e-06.

Not runnable, by design: `facebook/opt-125m`, `tiiuae/falcon-rw-1b` and
`nvidia/Nemotron-Mini-4B-Instruct` ship `pytorch_model.bin` (and `.nemo`)
only, no safetensors.
## Bug F5 — MiMo V2's fp8 attention projections are blocked per shard (fixed)

**Symptom.** `XiaomiMiMo/MiMo-V2.6-Flash-RL`, layers 0, 1, 5 (full attention
with the dense MLP, sliding attention with MXFP4 experts, full attention with
experts): layer 0's output was off by its whole magnitude. `MiMo-V2-Flash`, the
same cut: layer 0 off by 2e-3, only with more than one token.

**Cause.** The checkpoints are quantised shard by shard for tensor parallelism
over the full layers' `num_key_value_heads` (4): every attention projection
is 4 row shards, each fp8-blocked from its own first row with its own partial
last block. A V2.5 / V2.6 fused `qkv_proj` of a full layer is 4 × `[q | k | v]`
of 3392 rows (26.5 blocks of 128), 108 scale rows where blocking the 13568
rows whole gives 106; MiMo-V2-Flash's full-layer `k_proj` is 4 × 192 rows, 8
scale rows for 6. ditch took a grid that does not match the configured block
as uniform blocks of `ceil(rows / scale_rows)` rows (126, 96), misaligning
every scale after the first shard. SGLang's loader
(`get_mimo_v2_fused_qkv_expected_tp_size`, `_resolve_deferred_qkv_scale_inv`)
splits them by shard. The sliding layers' shards and the q / v projections
are whole blocks, which is why only the full layers showed it. The fixture
generator had no fp8 MiMo fixture: `mimo_v2_mxfp4` keeps its attention bf16.

**Fix.** `QuantConfig.attn_row_shards` (set to the full layers' kv heads by
the MiMo architecture): the fp8 scales of `self_attn.{qkv,q,k,v}_proj` are
read per row shard (`Dequant.row_shard`) when the grid is the per-shard one.
The generator quantises MiMo attention per shard (`quant_fp8_sharded`), and
two fixtures fail without the fix: `mimo_v2_fp8` (fused qkv, 44-row shards in
32-row blocks) and `mimo_v2_split_fp8` (split q/k/v, the full layers' 12-row
k shards in 8-row blocks). The references (`tools/ref_mimo_v2.py`, the release's
`modeling_mimo_v2.py`; `tools/ref_lazy_moe.py` for transformers'
`mimo_v2_flash`) dequantise by shard too, and `ref_mimo_v2.py` regroups the
fused shards into the `[all q | all k | all v]` its code splits, as SGLang does.

## Bug F6 — MiMo V2.6's sliding layers took the full layers' RoPE base (fixed)

**Symptom.** After F5, MiMo V2.6's sliding layer diverged by 4e-4 to 9e-3
(more for the longer prompt); with its experts zeroed in both, still 2e-3, so
the attention.

**Cause.** V2.6's config has a flat `rope_parameters`
(`{rope_theta: 1e7, partial_rotary_factor: 0.334}`) next to `swa_rope_theta:
10000`. ditch read a flat dictionary as the base of both layer kinds; the
release's `MiMoV2RotaryEmbedding` writes `swa_rope_theta` over it for the
sliding layers. MiMo-V2-Flash (no `rope_parameters`) was unaffected. The
generator's MiMo configs had no `rope_parameters`, so no fixture saw it.

**Fix.** A flat `rope_parameters` is the full layers' base; the sliding layers
keep `swa_rope_theta`. The `mimo_v2_mxfp4` fixture's config now carries the
flat dictionary (reference unchanged, fails without the fix).

## MiMo V2.6 and MiMo-V2-Flash: verified on real weights

Layers 0, 1, 5 of each (dense full attention, sliding attention with sinks and
MoE, full attention with MoE), routed experts lazy. MiMo V2.6: fp8 fused qkv,
MXFP4 experts (`store_dtype`), bf16 router config; reference: the release's
code (`tools/ref_mimo_v2.py`, float32 router as it computes it). MiMo-V2-Flash:
fp8 split q/k/v and experts; reference: transformers' `mimo_v2_flash`
(`tools/ref_lazy_moe.py`; the release's code gives the same numbers).
`truncate_checkpoint.py` now cuts `hybrid_layer_pattern` / `moe_layer_freq`
and writes `layer_types` / `mlp_layer_types`, which transformers otherwise
derives from the layer index.

| model | prompt | tokens | residuals | first-token logits | greedy |
| --- | --- | :---: | :---: | ---: | :---: |
| V2.6-Flash-RL | "The capital of France is" | match (5) | all 4 agree, worst 4.83e-07 (was 1.01e+00) | 5.14e-07 | match |
| V2.6-Flash-RL | "Explain how rainbows form, …" | match (15) | all 4 agree, worst 1.84e-06 (was 1.00e+00) | 6.22e-07 | match |
| V2-Flash | "The capital of France is" | match (5) | all 4 agree, worst 4.13e-07 (was 2.08e-03) | 3.64e-07 | match |
| V2-Flash | "Explain how rainbows form, …" | match (15) | all 4 agree, worst 1.27e-06 (was 3.83e-03) | 8.95e-07 | match |

V2.6 Pro (`MiMo-V2.6-Pro-RL`) shares the architecture and both fixes; it was not
cut separately.

## gpt-oss-120b (`gpt_oss`): verified on real weights

`openai/gpt-oss-120b`, first 2 layers (one sliding-window layer, window 128,
one full; attention sinks, YaRN RoPE, 128 MXFP4 experts with biases and the
clamped SwiGLU). The experts' `_blocks` / `_scales` are lazy in the cut and
read one expert at a time by `tools/ref_lazy_moe.py` (transformers' `gpt_oss`,
experts dequantised as its MXFP4 integration does, low nibble first; stored
`[out, in]` and transposed to the `x @ W` layout explicitly, since the down
projection is square and its shape cannot tell).

| prompt | tokens | residuals | first-token logits | greedy |
| --- | :---: | :---: | ---: | :---: |
| "The capital of France is" | match (5) | all 3 agree, worst 4.87e-07 | 4.67e-07 | match |
| "Explain how rainbows form, …" | match (14) | all 3 agree, worst 2.59e-06 | 7.01e-07 | match |
| the printing-press passage (past the 128-token window) | match (311) | all 3 agree, worst 1.62e-06 | 5.65e-07 | match (1 token) |
## The older families, continued

The float32 sweep of the older families, continued (routed experts lazy where marked):

| checkpoint | family | N | tokens | residuals (worst) | first-token logits |
| --- | --- | ---: | :---: | ---: | ---: |
| THUDM/GLM-4-9B-0414 | `glm4` | 3 | match | 2.00e-06 | 8.77e-07 |
| Qwen/Qwen1.5-MoE-A2.7B-Chat | `qwen2_moe` | 2, lazy | match | 7.66e-07 | 1.02e-06 |
| Qwen/Qwen3-30B-A3B | `qwen3_moe` | 2, lazy | match | 6.50e-07 | 6.38e-07 |
| deepseek-ai/DeepSeek-V2-Lite-Chat | `deepseek_v2` | 2, lazy | match | 3.02e-07 | 8.04e-07 |
| mistralai/Mixtral-8x7B-Instruct-v0.1 | `mixtral` | 1, lazy | match | 9.67e-07 | 1.27e-06 |
| THUDM/glm-4-9b-chat | `chatglm` | 3 | match | 4.79e-06 | 2.70e-06 |
| nvidia/NVIDIA-Nemotron-Nano-9B-v2 | `nemotron_h` | layers 0, 1, 14 (Mamba2, MLP, attention) | match | 7.85e-06 | 2.69e-06 |

**`chatglm`.** The release's own modeling code does not run under transformers
5 (`ChatGLMConfig` has no `max_length`), so the reference is transformers'
native `glm` on `THUDM/glm-4-9b-chat-hf`, the same weights converted, cut the
same way. Its config says `rope_theta: 10000`, where the original computes the
base as `10000 * rope_ratio` = 5e6 (`rope_ratio: 500`), as ditch does; with
that one value changed the two agree to 5e-06, and as released they differ by
3.8e-01 from the first layer. The converted release therefore does not
reproduce the original's positions; ditch follows the original.

Not runnable here: `internlm/internlm2_5-1_8b-chat` ships a SentencePiece
`tokenizer.model` and no `tokenizer.json` (refused, with the reason);
`openbmb/MiniCPM-2B-sft-bf16` and `baichuan-inc/Baichuan2-7B-Chat` ship
`.bin` weights only.

**Nemotron Nano 2 and the reference's BOS.** `tools/probe_reference.py` used
to add the tokenizer's BOS whenever the rendered template did not start with
it, so its ids had a `<s>` that ditch's lacked. transformers'
`apply_chat_template(tokenize=True)`, which heretic uses, adds no special
tokens, and gives ditch's ids; the reference now tokenises the same way.
`tools/truncate_checkpoint.py` now also cuts Nemotron-H's
`hybrid_override_pattern` string with the layers.

## Bug F7 — the remote disk estimate missed dequantised tensors (fixed)

**Symptom.** The warp-mode study of `hf://openai/gpt-oss-20b` printed
"routed expert 16.9KB each stored, 768 experts, 12.7MB in total": the
experts' two bias rows, of a 12.6 MB MXFP4 expert. altar-1's estimate had its
INT4 experts at 23.63 GB of 273 GB, and its trunk 6 GB short.

**Cause.** `remote.planModel` sorts every stored tensor into trunk or expert
by module, from the file's tensor and raw indexes. A tensor ditch dequantises
(MXFP4 `_blocks` / `_scales`, fp8 codes and scales, pack-quantized INT) is
replaced in them by a decoded view above the file, so its stored bytes were
counted nowhere: experts looked tiny (the cache plan thought all of
gpt-oss-20b's experts fit in 12.7 MB), and quantised trunk chunks were not
marked as trunk, which eviction keeps longest. Only the estimate and the
eviction priority; the weights read were right.

**Fix.** The codes, scales and zero points of every dequantised view are
classified by the view's name, in whichever file holds them, before the trunk
chunks are summed. `remote_test.zig` serves `gpt_oss_mxfp4` and `qwen2_fp8`
and checks that trunk + experts is every stored byte (it counted 79920 of
158256 before). gpt-oss-20b now: 3.35 GB trunk, 12.6 MB per expert, 9.47 GB
of experts.

## Bug F8 — Gemma 4's proportional RoPE paired the wrong coordinates (fixed)

**Symptom.** `google/gemma-4-E2B-it`, layers 0-4 plus the KV-shared layers 15
and 19: every sliding layer agreed, the full-attention layer 4 diverged by
1e-2 and the shared layers reading its KV more (0.95 at the end), while a
1-token prompt agreed everywhere (to 1e-7): something that depends on the
position, in the global layers only.

**Cause.** Gemma 4's global layers use `proportional` RoPE: `int(0.25 *
head_dim / 2)` frequencies over the whole 512-wide head, and transformers'
table has `head_dim / 2` entries, the rest zero, applied with `rotate_half`
over the head, so turning pair `i` is coordinates `i` and `i + 256`. ditch
rotated the first 128 coordinates as a block (pairs `i`, `i + 64`), the
fixture generator the same (`global_rotary`), so the `gemma4` fixture agreed.
At position 0 every rotation is the identity, hence the clean 1-token run.
Every Gemma 4 release's global layers were affected.

**Fix.** `Config.rope_angles`: the proportional global table spans the head
(`rotary_dim = head_dim`) and its pairs past the turning angles stay still.
The generator pads its table the same way; the regenerated `gemma4` fixture
fails without the fix (logits 1.16 off).

Reference-side, for the record: `tools/ref_lazy_moe.py`'s float32 embedding
dropped Gemma's embedding scale (entry 0 off by sqrt(1536)), its vision stub
lacked the `config` Gemma 4's init reads, and `tools/probe_reference.py`
hooked the audio embedder's norm as the final norm; it now prefers the norm
of the module that holds the layer stack.

## Gemma 4 E2B (`gemma4`): verified on real weights

Layers 0-4 (four sliding, one global) and 15, 19 (KV-shared, reading the
cut's last sliding and global layers), with per-layer inputs: the
`[262144, 35 x 256]` table is cut to the kept layers' columns by
`tools/truncate_checkpoint.py` (and `num_kv_shared_layers` recounted).
Reference: transformers' `gemma4` through `tools/ref_lazy_moe.py`.

| prompt | tokens | residuals | first-token logits | greedy |
| --- | :---: | :---: | ---: | :---: |
| "The capital of France is" | match (5) | all 8 agree, worst 1.52e-06 (was 9.84e-01) | 2.70e-06 | match |
| "Explain how rainbows form, …" | match (13) | all 8 agree, worst 3.00e-06 | 3.20e-06 | match |
| the printing-press passage | match (310) | all 8 agree, worst 1.63e-06 | 4.09e-06 | match (1 token) |

`gemma-4-26B-A4B` (`enable_moe_block`) is refused by ditch by design (its MoE
block is not implemented, see the registry note), so it was not cut.

## Mistral Small 4 (`mistral4`): verified on real weights

`mistralai/Mistral-Small-4-119B-2603`, first 2 layers (MLA with YaRN and the
Llama-4 attention scaling, 128-expert MoE with a shared expert), fp8 with one
scale per tensor (`weight_block_size: null`, `activation_scheme: static`: the
activation scales are for fp8 kernels, the weights are what both sides
dequantise) and stacked fp8 experts with one scale each (`[128, 1, 1]`),
lazy in the cut. Reference: transformers' `mistral4` through
`tools/ref_lazy_moe.py`, which now takes a tensor-wide fp8 scale and the
stacked per-expert ones.

| prompt | tokens | residuals | first-token logits | greedy |
| --- | :---: | :---: | ---: | :---: |
| "The capital of France is" | match (5) | all 3 agree, worst 3.21e-07 | 1.28e-06 | match (2 tokens) |
| "Explain how rainbows form, …" | match (16) | all 3 agree, worst 6.17e-07 | 1.70e-06 | match (2 tokens) |

## Llama 4 Scout (`llama4`): verified on real weights

`meta-llama/Llama-4-Scout-17B-16E-Instruct` is gated (401 here); its ungated
bf16 copy `unsloth/Llama-4-Scout-17B-16E-Instruct` was cut instead, layers 0
and 3 (chunked-attention RoPE layer, NoPE layer; each a 16-expert top-1 MoE
with a shared expert, 252 MB an expert, lazy). `truncate_checkpoint.py` now
cuts `no_rope_layers` and renumbers `moe_layers`. Reference: transformers'
`llama4` through `tools/ref_lazy_moe.py`, with two reference-side changes:
Linear subclasses with a forward of their own (`Llama4Router`) keep it, on
float32 weights, and `Llama4TextExperts.forward` (a dense `bmm` over every
expert of the router-scaled input) runs only the experts with a nonzero
block, which is exactly the same sum and reads only the routed experts.

| prompt | tokens | residuals | first-token logits | greedy |
| --- | :---: | :---: | ---: | :---: |
| "The capital of France is" | match (5) | all 3 agree, worst 9.24e-07 | 6.34e-07 | match (2 tokens) |

One prompt: the second's experts would not fit on the disk. The NoPE
layers' attention temperature and the 8192-token chunks only differ from
plain attention past 8192 tokens.

## MiniMax M3 (`minimax_m3_vl`): verified on real weights

`MiniMaxAI/MiniMax-M3` (854 GB, bf16), layers 0 and 3: a dense layer with full
attention, and the first MoE layer (128 experts plus a shared one, sigmoid
routing with a bias, the clamped SwiGLU) with the block-sparse attention and
its index heads (exact for prompts inside the 16 selected 128-token blocks).
Per-head q/k norms, Gemma-style norms, partial RoPE. `truncate_checkpoint.py`
now cuts the per-layer lists inside `sparse_attention_config`. Reference:
transformers' `minimax_m3_vl` through `tools/ref_lazy_moe.py` (113 MB experts,
lazy).

| prompt | tokens | residuals | first-token logits | greedy |
| --- | :---: | :---: | ---: | :---: |
| "The capital of France is" | match (5) | all 3 agree, worst 9.82e-07 | 5.00e-07 | match (2 tokens) |
## Bug 63 — Falcon instruct was prompted in ditch's generic format (fixed)

**Symptom.** `tiiuae/falcon-7b-instruct`, first 2 layers: the forward pass
agreed to 2e-06 on ditch's ids, but the rendered prompt was
`SYS\n\nUser: …\nAssistant:` where the release's template gives
`SYS\n\nUser: …\n\nAssistant:`.

**Cause.** ditch had no template for Falcon's: `system.strip()`, then
`'\n\nUser: '` / `'\n\nAssistant: '` before each turn's content (stripped, with
`\r\n` → `\n` and then `\n\n` → `\n`), then `'\n\nAssistant:'`. Nothing matched
it, so ditch fell back to its generic `raw` rendering, which is close but not
the same.

**Fix.** A `falcon` template, detected from the release's markers (whether the
Jinja source spells the line breaks as escapes or as the characters), with the
content replacements applied in Python's order; tests cover the detection, the
render and the replacements.

| checkpoint | family | N | tokens | residuals (worst) | first-token logits |
| --- | --- | ---: | :---: | ---: | ---: |
| tiiuae/falcon-7b-instruct | `falcon` | 2 | match (after the fix) | 2.34e-06 | 2.77e-06 |
| Qwen/Qwen3-0.6B | `qwen3` | 28 (all) | match | 5.64e-06 | 1.38e-06 |
| allenai/FlexOlmo-7x7B-1T | `flex_olmo` | 2, lazy | match | 1.12e-06 | 8.31e-07 |
| ibm-granite/granite-4.0-h-tiny | `granitemoehybrid` (MoE) | 6 | match | 1.84e-06 | 1.11e-06 |

`unsloth/gemma-3n-E2B-it` could not be cut at first: its per-layer embedding
table (`embed_tokens_per_layer`, [vocab, layers x 256]) has to be sliced by
columns. It is now verified on real weights, see "Gemma 3n" below.

## Gemma 4 12B (`gemma4_unified`): verified on real weights

`google/gemma-4-12B-it`, layers 0 and 5 (a sliding layer; a global layer with
its own 512-wide head, one KV head, keys reused as values and proportional
RoPE, after Bug F8). Reference: transformers' `gemma4_unified` through
`tools/ref_lazy_moe.py`.

| prompt | tokens | residuals | first-token logits | greedy |
| --- | :---: | :---: | ---: | :---: |
| "The capital of France is" | match (5) | all 3 agree, worst 8.73e-07 | 4.02e-06 | match |
| "Explain how rainbows form, …" | match (13) | all 3 agree, worst 1.57e-06 | 5.45e-06 | match |
| the printing-press passage | match (310) | all 3 agree, worst 1.17e-06 | 4.95e-06 | match (1 token) |
## Bug 64 — a config without `model_type` was run as Llama (fixed)

**Symptom.** `openbmb/MiniCPM4-0.5B` loaded as `* Architecture: llama`.

**Cause.** MiniCPM4's config.json has no `model_type`, only `architectures`
(`MiniCPMForCausalLM`) and an `auto_map` to its own code. ditch defaulted a
missing `model_type` to `llama`, which ran the model without MiniCPM's
embedding scale (`scale_emb` 12), residual scale (`scale_depth / sqrt(layers)`)
and logit scale (`hidden_size / dim_model_base`): a different model, with no
error.

**Fix.** Without `model_type`, the family is taken from `architectures[0]`
when its lowercased class prefix (before `ForCausalLM`, `LMHeadModel` or
`ForConditionalGeneration`) is a registered model type; a parse test covers
MiniCPM4's config.

**Check.** The release's remote code does not run faithfully under
transformers 5, even with `tools/probe_reference.py`'s shims (now also
`use_cache = False` for remote code, since this one refuses to start the tuple
cache it was written for): it answers every prompt with `<|im_end|>` and
differs from ditch by 8e-02 at layer 0. A float32 re-implementation of layer 0
straight from `modeling_minicpm.py` (embedding × 12, RMSNorm, GQA with the
LongRoPE short factors and scaling factor 1, residual × 1.4/√24, SwiGLU)
agrees with ditch to 1.9e-07, and to 2.6e-02 without the short factors. So
MiniCPM4 is verified against its own code by hand, one layer deep, not
against a running reference.

## Llama 4 Maverick (`llama4`): verified on real weights

`unsloth/Llama-4-Maverick-17B-128E-Instruct` (the ungated bf16 copy; the
official repository is gated), layers 0 and 3: a dense RoPE layer (16384-wide
MLP) and a NoPE MoE layer (128 experts, top-1, shared expert; MoE on every
second layer, `interleave_moe_layer_step: 2`, `moe_layers` renumbered by the
cut), no q/k norm. Same reference as Scout.

| prompt | tokens | residuals | first-token logits | greedy |
| --- | :---: | :---: | ---: | :---: |
| "The capital of France is" | match (5) | all 3 agree, worst 1.20e-06 | 5.95e-07 | match (2 tokens) |
## Bug 65 — sequential Falcon (RefinedWeb) lost its MLP norm and mis-scaled ALiBi (fixed)

`facebook/opt-125m` and `tiiuae/falcon-rw-1b` ship `pytorch_model.bin` only;
both were re-saved as safetensors with `save_pretrained(safe_serialization=True)`
(a user would have to do the same) and compared whole, `--raw`. OPT matched at
once (all 13 residuals within 9.5e-07, logits 2.6e-07). Falcon-RW-1B, the
ALiBi variant the `falcon` entry called "implemented but unverified", did not.

**Symptom.** Falcon-RW-1B's layer 0 differed by 5.1e-01 and the argmax
disagreed; most of the difference was in the MLP.

**Causes.** (1) The `falcon` names set `pre_ff_norm = null`: right for the
parallel layouts (7B: one norm; 40B: `ln_attn` / `ln_mlp`), but the sequential
layout (`parallel_attn: false`, RefinedWeb) normalises the MLP input with
`post_attention_layernorm`, so ditch fed the MLP an unnormalised residual.
(2) Falcon adds the ALiBi bias to the raw `q·k` and then scales the sum by
1/√head_dim (transformers folds `alibi / sqrt(head_dim)` into the mask);
ditch added it after the scaling, as BLOOM and MPT do, so Falcon's bias was
√head_dim (8×) too large. The fixture covered only the 7B layout.

**Fix.** `pre_ff_norm` names `post_attention_layernorm` (optional in the
parallel layouts, which lack it); `Config.alibi_scale`, 1/√head_dim for
Falcon with ALiBi, multiplies the slopes. New fixture `falcon_rw`
(sequential, per-head interleaved qkv with biases, scaled ALiBi) fails on
the old code (logits off by 11.9) and passes.

| model | tokens | residuals (worst) | first-token logits |
| --- | :---: | ---: | ---: |
| falcon-rw-1b, "The capital of France is" | match (5, raw) | 3.51e-05 | 5.08e-06 |
| falcon-rw-1b, "Explain how rainbows form, …" | match (11, raw) | 2.23e-04 | 1.50e-05 |
| same, reference ALiBi built in float32 | | 2.70e-06 | 5.69e-07 |

The remaining 2e-04 is transformers': `build_alibi_tensor` rounds the slopes
(and their products with the positions) to bf16, and Falcon-RW's 32 heads
have slopes that are not powers of two. With the reference's ALiBi built in
float32, every layer agrees to 3e-06. ditch keeps float32 slopes.

## `.bin`-only releases, converted

| checkpoint | family | how | tokens | residuals (worst) | first-token logits |
| --- | --- | --- | :---: | ---: | ---: |
| facebook/opt-125m | `opt` | whole, re-saved as safetensors | match (raw) | 9.54e-07 | 2.58e-07 |
| tiiuae/falcon-rw-1b | `falcon` (sequential, ALiBi) | whole, re-saved | match (raw) | 2.23e-04 (bf16 ALiBi, bug 65) | 1.50e-05 |
| nvidia/Minitron-4B-Base | `nemotron` | first 3 layers of the loaded bf16 model, saved as safetensors | match (raw) | 1.71e-06 | 2.47e-06 |
| EleutherAI/gpt-j-6b (`float16` revision) | `gptj` | first 3 layers, read from the `.bin` with a memory-mapped `torch.load`, saved as float32 safetensors | match (raw) | 1.34e-06 | 9.50e-07 |

Left: `baichuan-inc/Baichuan2-7B-Chat` (15 GB of `.bin` plus remote code),
`adept/persimmon-8b-*` (`.bin` and no `tokenizer.json`).

## Gemma 3n (`gemma3n_text`): verified on real weights

`unsloth/gemma-3n-E2B-it` (the ungated copy of Google's release), first 5
layers (four sliding, one full), vision and audio towers dropped:
`tools/truncate_checkpoint.py` slices the per-layer embedding table and its
projection to the kept layers' 256-wide blocks (5 x 256 columns / rows), and
now also cuts `intermediate_size` and `activation_sparsity_pattern` when they
are per-layer lists (transformers refuses 30 entries for 5 layers).
`num_kv_shared_layers` is 0 in the cut, since the ten shared layers are the
last ten, so KV sharing itself is not exercised here; AltUp (four streams),
LAuReL, the per-layer inputs, the 0.95 activation sparsity of these layers
and both attention kinds are.

| prompt | tokens | residuals | first-token logits | greedy |
| --- | :---: | :---: | ---: | :---: |
| "The capital of France is" | match (21) | all 6 agree, worst 7.04e-07 | 6.59e-07 | match |
| "Explain how rainbows form, …" | match (25) | all 6 agree, worst 7.74e-07 | 6.14e-07 | match |

KV sharing, separately: `--layers 0,18,19,20,24` with `num_kv_shared_layers: 2`
keeps the release's own pairing (layer types sliding, sliding, full, sliding,
full; original layer 20 reads layer 18's cache and 24 reads 19's, as in the
whole model). All 6 residuals agree, worst 1.07e-06, logits 9.5e-07. The same
reference with sharing turned off differs from ditch by 3.2e-02 from the first
shared layer, so the check does exercise it.
## gpt-oss-20b at full depth over `hf://`, and two warp-mode speedups

The whole released `openai/gpt-oss-20b` (24 layers, 32 MXFP4 experts a layer,
13.8 GB stored), read on demand from the Hub through the chunk cache and run
in warp mode on this machine (4 cores, 15 GB RAM, `--max-ram 10GB
--remote-cache-size 16GB`), through `ditch probe` (which could not load an
`hf://` model before `a5a2627`). ReleaseFast build; the Debug build `zig build`
makes by default ran the same probe at 0.015 tokens/s.

    dry run:  warp mode min 2.03GB, with prefetch + expert cache + RAM caches 11.68GB
              trunk 3.35GB stored, routed expert 12.6MB stored (47.5MB decoded), 768 experts 9.47GB
    measured: peak RSS 7.22GB (budgeted peak 7.02GB), chunk cache 12.14GB on disk of 16GB, 0 evictions
              expert cache 5.67GB = 122 of 768 decoded experts, 24.3% hit rate, 59.7 misses per decode step

"What is the capital of France?" (87 prompt tokens with the harmony system
block), 24 greedy tokens: `<|channel|>analysis<|message|>We need to answer:
"What is the capital of France?" The answer: Paris. Provide concise answer`.
"What is 12 times 12?": `<|channel|>analysis<|message|>The user asks: "What is
12 times 12?" It's a simple multiplication. 12*`. First-token top logit
`<|channel|>` at 42.9 (next 16.6).

Profiling the run (gdb stack samples) put the time in loading experts, not
in the matmuls: `WeightStore.acquireColumns` transposed the fused gate/up
block with a runtime-sized `@memcpy` per 2-byte element, and the MXFP4
decoders called `std.math.ldexp` per element. Now the transpose is typed and
tiled, and a block's E8M0 scale is built from its bits once and multiplied
in (exact for e2m1 values; `std.math.ldexp` itself returns `2^(e - 151)` for a
zero value once `e >= 25`, so zeros under a scale of `2^25` or more decoded as
tiny nonzero numbers before):

| build | prompt 1 | prompt 2 | both prompts, load to exit |
| --- | ---: | ---: | ---: |
| before | 0.071 tokens/s (336.8 s) | 0.070 tokens/s (345.1 s) | 876 s |
| after | 0.149 tokens/s (160.9 s) | 0.151 tokens/s (159.0 s) | 411 s |
| + gate/up decoded as rows | 0.281 tokens/s (85.4 s) | 0.297 tokens/s (80.9 s) | 216 s |

Same text, same cache statistics (the same work, done faster), and the third
row's top-10 first-token logits are identical to the second's. The third row:
the fused gate/up block is stored `[2I, H]` in MXFP4 and viewed `[H, 2I]`, so
an expert was decoded into the view's layout and its gate and up halves (every
second column) transposed back out; now those columns are decoded directly
from the stored rows (`Dequant.readMxfp4SourceRows`), halving the bytes an
expert load moves (113 GB where 356 GB were read). Measured peak RSS 7.04 GB.
The remaining limit is the cache holding decoded experts: 47.5 MB each, so a
quarter of one layer's routing fits and three in four expert reads re-decode.

## Bug F9 — the mHC mixing weights lost precision in f32 sums (fixed; closes GLM-5.3-Flash)

**Symptom.** GLM-5.3-Flash's truncated check agreed only to 4.2e-05 (the
item left open above). Bisected with dumps of every stage of layers 0-3
against the reference's hooks: at the last token every stage of layers 0-2
agreed to ~1e-06, then layer 3's collapsed input jumped to 3.65e-05.
Recomputing layer 2 (and layer 3's collapse) in float64 from each side's own
input: the float32 reference is within 3.5e-07 of it, ditch 3.6e-05 off,
the input difference only 8e-07, so the error was ditch's own, in layer 2.
The dense MLP was not it (ditch 6.8e-07 from float64, torch float32
5.9e-07).

**Cause.** The four residual streams differ by orders of magnitude (stream 0
at 0.35, the others at 0.006), and the mHC weights mix the large stream into
small results (layer 3's collapsed input is 0.0086): an error in `comb` or
`pre` is multiplied by the large stream and divided by the small result.
ditch computed those weights from a plain f32 running sum over the `hc *
hidden` = 24576 coordinates (the unweighted RMS norm and the `fn`
projection), where torch sums in blocks; `comb` came out 4.6e-06 off, 40x
amplified at the collapse.

**Fix.** The mixes' RMS and projection are accumulated in f64 (`dot64`: each
f32 product is exact in f64; 24 dot products a token a site), and so is V4's
weighted `hc_head`. A unit test checks the small terms a long f32 sum drops.

| prompt | tokens | residuals | first-token logits | greedy |
| --- | :---: | :---: | ---: | :---: |
| "The capital of France is" | match (5) | all 5 agree, worst 1.19e-06 (was 2.48e-05) | 2.14e-07 (was 2.98e-06) | match |
| "Explain how rainbows form, …" | match (15) | all 5 agree, worst 9.72e-07 (was 4.20e-05) | 4.12e-07 (was 1.67e-05) | match |

GLM-5.3-Flash now agrees to 1e-06 like every other family. DeepSeek V4's
1.2e-05 (the other mHC family, above) likely had the same cause; not re-cut.

## Short study on Llama-3.2-1B-Instruct (after bug 61)

`unsloth/Llama-3.2-1B-Instruct`, whole model, prompted in its dated
`llama32` template: 6 trials (4 random), `--max-response-length 40`.

| Model | family | Baseline Refusals | Best trial | Best KL |
| --- | --- | ---: | ---: | ---: |
| unsloth/Llama-3.2-1B-Instruct | `llama` (`llama32` template) | 99/100 | 10/100 | 0.2373 |

Front: 10/100@0.237, 44/100@0.194, 72/100@0.159, 90/100@0.023,
92/100@0.023; one trial pruned by early stopping; 17 minutes.

`ditch --reproduce` on the export's manifest re-derived the same trial and
scores exactly (KL 0.2373, 10/100). The export's self-validation reported a
max first-token logit difference of 0.0748 (argmax 100%) against the
in-memory model: that is the bf16 rounding of the merged `W + ΔW` in the
export, since the in-memory model keeps the delta in float32. The same trial
exported with `--export-dtype f32` validates at 0.0000.

## InternLM2 with a converted tokenizer, and bugs 66-67

`internlm/internlm2_5-1_8b-chat` ships a SentencePiece `tokenizer.model` and
no `tokenizer.json`; ditch's refusal message tells the user to save one with
`AutoTokenizer`. Doing that under transformers 5 (`InternLM2TokenizerFast`,
the release's remote code) writes a `tokenizer.json` that is not the model's
tokenizer: no normalizer and no pre-tokenizer (spaces are dropped:
"a helpful" becomes `ah` `elp`), and the chat tokens renumbered after the
vocabulary (`<|im_start|>` 92549 in a 92544-row embedding; the release's own
id is 92543). transformers' reference fails on it with an index error, and
the release's `tokenizer.model` cannot be read by the installed sentencepiece
(`piece must not include null character`). InternLM2 therefore stays
unverified: no working tokenizer can be produced here. Two ditch bugs came out
of it.

### Bug 66 — a token id past the embedding table read the last row (fixed)

**Symptom.** ditch ran the prompt above without complaint.

**Cause.** `Model.embedRow` clamped the id to the last row of the embedding,
so any token the checkpoint has no row for was silently embedded as another
token.

**Fix.** An id at or past the table's row count is an error naming the
mismatch (`token id 92549 is outside the embedding table (92544 rows): the
tokenizer does not match the checkpoint`); a test covers the range check.

### Bug 67 — BPE kept characters it cannot spell as merge barriers (fixed)

**Symptom.** On that `tokenizer.json`, ditch encoded "You are a helpful
assistant." as `a` `help` … where `tokenizers` gives `ah` `elp` ….

**Cause.** In `tokenizers`' BPE a character with no vocabulary entry, when
there is neither byte fallback nor an unknown token, is left out of the word
before merging, so its neighbours can merge across it. ditch kept it as a
symbol, which blocked every merge across it, and then emitted nothing for it.

**Fix.** In exactly that case ditch removes such characters before merging; a
test (`tokenizers` gives [2, 3] for its input) covers it. The existing
byte-level round-trip test had a merge (`Ġ w`) whose part `w` was not in its
vocabulary, which `tokenizers` refuses to load; `w` was added to it.
Byte-level tokenizers have every byte in their vocabulary, so nothing changes
for them.

## Summary of the remaining-families pass

Every registered family outside the frontier session's list that has a
reachable release has now been compared with a float32 reference on its real
weights, whole or cut (first *N* layers, `--layers` for far-apart layer kinds,
lazy experts, or a subset of experts where the reference would not fit):
GLM-4.7-Flash, GLM-4.5-Air, GLM-4, ChatGLM, ERNIE 4.5 MoE, Hunyuan A13B,
Seed-OSS, Qwen2/2-MoE/3/3-MoE/3-Next/3.5-MoE, dots.llm1, Solar Open, EXAONE 4,
K-EXAONE, Granite 3.3 and 4.0-H, DeepSeek V2 and V3.2, Mixtral, Mistral,
Llama 3.2, Gemma 2, 3 and 3n (with KV sharing), GPT-2, GPT-J, GPT-NeoX, BLOOM,
OPT, OLMo, OLMo 2, FlexOlmo, StableLM 2, Phi-2, Phi-3.5, SmolLM3, StarCoder 2,
GPTBigCode, Mamba2, Nemotron-H, Nemotron (Minitron), Falcon (7B and RW),
MiniCPM4 (by hand, one layer). Bugs 58-67 were found this way, plus four
reference-side artefacts recorded where they occurred: transformers' latent-norm
epsilon (Bug F3), its sdpa attention dropping Gemma 2's softcap, the Mamba
hidden-state offset, and its bf16 ALiBi.

Not verified, and why: `internlm2` (no working tokenizer can be produced under
transformers 5), `baichuan` and `persimmon` (`.bin` and no `tokenizer.json`),
`jais2` and `laguna` releases (gated), `hy_v3` (no text release), `bitnet`
(refused by design: the release quantises activations at run time), and
`gemma3n`'s audio/vision towers and any family's behaviour beyond the cut
layers (by construction of the method).

## The blocked families, once more (mirrors, conversions, stubs)

A search of the Hub for every blocked family, and what each find allowed:

| family | what exists besides the release | what it allowed |
| --- | --- | --- |
| `persimmon` | `hf-tiny-v2/tiny-random-PersimmonForCausalLM` (safetensors, `tokenizer.json`); OpenVINO IR and GGUF conversions of the 8B | stub vs transformers: all 3 residuals within 2.5e-07, logits 1.9e-07. The 8B release still has only `.bin` and no `tokenizer.json`. |
| `jais2` | `hf-tiny-v2/tiny-random-Jais2ForCausalLM`; NVFP4 / MXFP4 / AWQ re-uploads of the 8B (formats ditch does not decode) | stub vs transformers: all 3 residuals within 1.5e-07, logits 1.5e-07. The releases stay gated. |
| `bitnet` | `hf-tiny-v2/tiny-random-BitNetForCausalLM` (plain weights) | stub vs transformers: all 3 residuals within 4.2e-07, logits 2.2e-07, which is the plain-weights layout the entry claims. The releases stay refused (run-time activation quantisation). |
| `internlm2` | `optimum-intel-internal-testing/tiny-random-internlm2`, whose `tokenizer.json` is a correct conversion of InternLM2's 92544-token SentencePiece vocabulary | the stub's tokenizer on the **real** `internlm2_5-1_8b-chat` (first 4 layers, `--raw`): ditch's ids equal `tokenizers`'; every residual within 1.3e-06 and logits within 7.7e-07 of a float32 re-implementation of the release's `modeling_internlm2.py` (grouped `wqkv`, RoPE base 1e6; the dynamic NTK scaling only acts beyond 32768 tokens). The release's own code fails under transformers 5 (`rope_scaling["type"]`, then 6e-01 off even on the stub, where the re-implementation agrees with ditch to 1.6e-07). |
| `baichuan` | `katuni4ka/tiny-random-baichuan2` (random weights, remote code, no `tokenizer.json`); `hiyouga/*-LLaMAfied` (Llama-format conversions with the same `tokenizer.model`) | a `tokenizer.json` converted by transformers' `LlamaTokenizerFast` differs from SentencePiece on the first token (`▁You` against `You`: the converter prepends the word marker Baichuan does not), so no faithful tokenizer; the stub's remote code fails under transformers 5 (meta tensor). The stub against a float32 re-implementation of `modeling_baichuan.py` found bug 68. |

### Bug 68 — Baichuan 2's LM head was not normalised (fixed)

**Symptom.** On the Baichuan 2 stub every residual agreed with the
re-implementation to 4e-07, but the logits differed by 2.5e-01 of their range
(argmax still equal).

**Cause.** Baichuan 2's `NormHead` L2-normalises every LM-head row at
inference (`F.normalize(self.weight)`); ditch used the stored rows. Baichuan 1
has a plain head.

**Fix.** `Config.lm_head_l2norm`, set for `baichuan` with the Baichuan 2
vocabulary (125696 tokens, the rule llama.cpp uses); the model computes the
rows' inverse norms once at load and scales the logits by them, so the stored
weights (and a safetensors export, which transformers normalises again) are
untouched. The GGUF export writes the head's rows normalised, since
llama.cpp runs a plain head (its converter normalises too); normalising them
again on import changes nothing. Logits now agree to 3.2e-07. A parse test
covers the flag.

## gpt-oss-120b at full depth over `hf://`: network-bound, no token in 90 minutes

`openai/gpt-oss-120b` (36 layers, 128 MXFP4 experts a layer, 65 GB stored)
through `ditch probe hf://openai/gpt-oss-120b --max-ram 10GB
--remote-cache-size 18GB`, ReleaseFast, one prompt ("What is the capital of
France?", 87 tokens with the harmony system block).

    dry run:  warp mode min 6.48GB, with prefetch + expert cache + RAM caches 19.56GB
              trunk 3.96GB stored, routed expert 12.6MB stored (47.5MB decoded), 4608 experts 56.80GB
              expert cache 5.67GB = 122 of 4608 experts
    measured: peak RSS 7.49GB (under the 9GB limit), chunk cache at its 18GB bound (2308 chunks)
              the Hub served ~19MB/s here (one curl stream: 200MB in 10.9 s)
              12-token run stopped after 5401 s with no token generated;
              1-token (prefill only) run stopped after 32 min, still in the prefill

Why: the 87-token prompt routes to most of the 4608 experts, about 57 GB of
MXFP4, while this machine's disk leaves room for an 18 GB chunk cache, so
the prefill's working set cannot stay on disk: at ~19 MB/s the first pass
alone needs ~50 minutes, and experts evicted before a later layer or token
reads them again are fetched again. No warm-cache number is possible here
for the same reason (the warm working set is three times the cache). A
machine with ~60 GB of disk for the cache would hold every expert after one
pass; the RAM side fits (7.5 GB). Not a correctness result: gpt-oss-120b's
arithmetic is verified on its truncated cut above.

Why: the 87-token prompt routes to most of the 4608 experts, about 57 GB of
MXFP4, while this machine's disk leaves room for an 18 GB chunk cache, so
the prefill's working set cannot stay on disk: at ~19 MB/s the first pass
alone needs ~50 minutes, and experts evicted before a later layer or token
reads them again are fetched again. No warm-cache number is possible here
for the same reason (the warm working set is three times the cache). A
machine with ~60 GB of disk for the cache would hold every expert after one
pass; the RAM side fits (7.5 GB). Not a correctness result: gpt-oss-120b's
arithmetic is verified on its truncated cut above.

## gpt-oss-20b at full depth over `hf://`: cold (network) and warm cache

The same run as above, after the speedups, split into a first pass with an
empty chunk cache and a second pass with the model on disk (`--max-ram 10GB
--remote-cache-size 16GB`, ReleaseFast, "What is the capital of France?",
87 prompt tokens, 16 greedy tokens).

| pass | load to exit | 16 tokens with the prefill | fetched from the Hub | chunk reads from disk | peak RSS |
| --- | ---: | ---: | ---: | ---: | ---: |
| cold | 1066 s | 1006.7 s, 0.016 tokens/s | 1510 ranges, 11.78 GB (~11.7 MB/s) | 0 | 7.08 GB |
| warm | 70 s | 65.3 s, 0.245 tokens/s | 0 | 1509 | 7.03 GB |

Dry-run estimate for comparison: warp mode min 2.03 GB, with prefetch and all
caches 11.68 GB (the budget allowed 9 GB; measured peak 7.08 GB), trunk
3.35 GB and 9.47 GB of experts on disk (measured chunk cache 11.77 GB, every
expert the prompt touched). Both passes print `<|channel|>analysis<|message|>We
need to answer: "What is the capital of France?"`. The cold pass is bound by
the Hub's bandwidth here (the prefill touches 645 of the 768 experts); once
they are on disk the run is bound by decoding them into the 5.67 GB expert
cache (29% hits, 56 misses a decode step). 0.245 tokens/s here against
0.289 above: 16 tokens amortise the 87-token prefill less than 24 did.

## Mistral Small 4 at full depth over `hf://`: dry run only

`ditch --dry-run hf://mistralai/Mistral-Small-4-119B-2603 --max-ram 10GB
--remote-cache-size 18GB`:

    weights total 222.40GB (36 layers), trunk per layer 102.5MB
    routed expert 48.0MB decoded / 24.0MB stored (fp8), 4608 experts, 108GB stored
    warp mode: min 6.61GB, with prefetch + expert cache + RAM caches 24.28GB
    expert cache 5.75GB = 122 experts; trunk 4.62GB stored, stays cached; 558 experts fit beside it

Not run: at top-4 of 128 experts in 36 layers, a short prompt's prefill
routes to most of the 108 GB of experts and each generated token fetches
~144 of them (~3.4 GB), which at the ~12-19 MB/s the Hub serves here is
hours for the first pass, with the working set 6x the 18 GB of chunk cache
this disk allows (the same limit as gpt-oss-120b). The RAM side fits (6.6 GB
minimum). Its arithmetic is verified on the truncated cut above.

# Full-depth reference

Until now every frontier family was compared with its reference on a cut: the
first *K* layers (or a few far-apart ones), because transformers holds the
whole model in memory and this machine has 15 GB of RAM and ~28 GB of disk.
ditch itself already runs these models at full depth, streamed over `hf://`.
This pass makes the reference do the same: transformers' own model, built from
the real config and run unchanged end to end, with each decoder layer's weights
read from the Hub just before it runs and dropped after.

Machine as before: 4 cores, 15 GiB RAM, ~28 GiB free disk, no GPU; torch
2.14.0+cpu, transformers 5.17.0, accelerate 1.15.0, huggingface_hub 1.32.0,
Zig 0.16.0.

## How others run a model larger than the RAM

A short survey, to find what can be reused and what cannot on a machine whose
disk is also smaller than the model.

* **accelerate big-model inference.** `init_empty_weights` builds the model on
  the meta device. `device_map="auto"` / `infer_auto_device_map` then place
  each module on the GPU, the CPU or `"disk"`. An `AlignDevicesHook`
  (`add_hook_to_module`) loads an offloaded module's weights before its
  forward and sends them back after. There is no prefetching. `disk_offload` /
  `offload_weight` write a numpy memmap copy of every weight (bf16 as int16).
  `OffloadedWeightsLoader` can instead point at a tensor inside a *local*
  safetensors file, with no copy. Reusable: the hook pattern. Not usable here:
  every path needs the checkpoint on local disk, which this machine cannot
  hold.
* **transformers 5 disk offload** (`offload_folder`) goes through accelerate.
  It reuses the original safetensors entries for weights that load as stored,
  but *re-saves as a memmap* every weight that passes through a
  `WeightConverter` or a dequantiser, which is every fused MoE expert stack and
  every quantised tensor. It needs the full local checkpoint and roughly as
  much again in memmaps.
* **AirLLM** runs a model one layer at a time. On first use it splits the
  checkpoint into one file per layer on disk (the full download, then about as
  much again), optionally compresses to 4/8 bits, and optionally prefetches
  the next layer. Its own model code (Llama, Qwen, Mistral, DeepSeek, ...)
  replaces the family's. Ruled out on both counts: the disk, and the reference
  must be the official code.
* **transformers 5 loading internals.** `from_pretrained` builds the model
  under `torch.device("meta")`, runs the quantizer's `preprocess_model`, and
  collects the family's weight transforms (`get_model_conversion_mapping`:
  `WeightRenaming`s, and `WeightConverter`s such as
  `MergeModulelist` + `Concatenate`, which fuse per-expert tensors into
  `gate_up_proj [E, 2I, H]`) together with the quantizer's own transforms. For
  a pre-quantised checkpoint loaded with `dequantize=True`, which is the CPU
  default for MXFP4 and FP8, those are `Mxfp4Dequantize` and `Fp8Dequantize`.
  All of it goes into a `LoadStateDictConfig` and then into
  `convert_and_load_state_dict_in_model(model, state_dict, load_config)`.
  That function takes *any* state dict: keys it does not see are just
  "missing". Each loaded tensor replaces the meta parameter it targets
  (`set_param_for_module`). So a partial state dict, for example one layer's
  checkpoint tensors under their checkpoint names, goes through exactly the
  official renames, converters and dequantisation. The one trap is that
  `MergeModulelist` stacks whatever experts it was given. Feeding it a subset
  mis-indexes the stack, unless the target parameter is also one expert wide.
* **Remote safetensors.** `safe_open` needs a local file. A shard's header is
  an 8-byte length plus JSON with `data_offsets`, so any tensor is one HTTP
  `Range` request. A Xet-backed file's `resolve` URL redirects (302) to a CDN
  URL that answers `Range` with 206. `HfFileSystem` does fsspec range reads.
  `hf_transfer` is gone in huggingface_hub 1.x, and `hf_xet`'s
  high-performance mode assumes ≥64 GB of RAM. `tools/truncate_checkpoint.py`
  already reads this way.
* **FlexGen, DeepSpeed ZeRO-Inference.** Both stream weights layer by layer
  from NVMe or CPU memory, and both prefetch layer *i*+1 while layer *i*
  computes; the prefetch is the idea worth keeping. **llama.cpp** mmaps the
  weights and lets the page cache act as the weight cache. 671B–1T MoE models
  run off an SSD below 1 token/s this way, their speed set by the active
  parameters. Expert-streaming and PowerInfer-style work reuse hot experts
  across tokens.

What follows from that: none of the offload frameworks fits a checkpoint that
cannot be on the disk. But transformers' own loader can be given one layer at
a time, which keeps the reference exactly the official code.

## Method: `tools/ref_stream.py`

    ditch probe hf://REPO --prompt ... --residuals --json > p.json
    python3 tools/probe_reference.py REPO p.json --dtype float32 --factory tools/ref_stream.py --per-layer

The reference is transformers' own model class, built by its own
`from_pretrained` from the release's config (all layers). Two of
`from_pretrained`'s steps are replaced, and nothing in the model:

1. **Only the trunk is loaded.** `from_pretrained` runs as usual: config,
   quantizer (`dequantize=True`, its CPU default), meta-device init, the
   family's conversion mapping, the `LoadStateDictConfig`. It is given a
   placeholder weights file, and its `_load_pretrained_model` is swapped for
   one that
   * maps every checkpoint key to its target parameter with transformers'
     own `rename_source_key`, and assigns it to a decoder layer by that name.
     Keys whose target is not in the model are dropped: MTP layers, vision
     towers.
   * loads the rest (embeddings, final norm, LM head, anything outside the
     layer stack) through `convert_and_load_state_dict_in_model`.
   The layers' parameters stay on the meta device. The rest of
   `from_pretrained` (missing-key init, weight tying, generation config) runs
   unchanged.
2. **Each decoder layer is materialised just before it runs.** A forward
   pre-hook reads the layer's checkpoint tensors (HTTP `Range` requests on
   the shards, 16 in parallel, in chunks of at most 32 MB) and passes them,
   under their checkpoint names, through the same
   `convert_and_load_state_dict_in_model` and load config. So the family's
   renames, its expert-fusing converters and the quantizer's dequantisation
   produce the weights, as in a whole-model load. A forward hook puts them
   back on the meta device. The next layer is fetched in the background while
   one runs.

The model's own forward, generation loop and KV cache are untouched, so state
that crosses layers (the cache, mHC streams, Kimi K3's attention over earlier
layers' outputs, Gemma's KV sharing) behaves as in a whole-model run.

**Routed experts.** When a layer's experts would take more than 2 GB as
float32, the experts module's `gate_up_proj` / `down_proj` are replaced by
objects whose `[e]` loads expert *e* on first use. transformers' eager
experts forward indexes the stacks only for the experts a token is routed
to. An expert is loaded through the same loader: its checkpoint tensors (its
own per-expert tensors renumbered to expert 0, or slab *e* of a stacked
tensor such as gpt-oss's `_blocks` / `_scales`) go into a one-expert-wide
meta copy of the stacks. That way `MergeModulelist`, `Concatenate`,
`Mxfp4Dequantize` and `Fp8Dequantize` run exactly as in a whole-model load.
Dequantised experts are kept in a 3 GB LRU for the generated tokens.

**Arithmetic.** Float32 on the stored values, as in the earlier passes:
* Stored bf16 tensors load as float32 (exact). F32-stored tensors
  (routers' correction biases, mHC weights) load as F32.
* A quantised weight is dequantised into bf16, as a load in the release's
  own dtype produces it (`Mxfp4Dequantize` always emits bf16;
  `Fp8Dequantize` emits the parameter's dtype), then widened to float32.
* The trunk is float32 unless it would exceed 6 GB. In that case tables of
  more than 256M elements stay bf16 and their `Linear` / `Embedding`
  compute in float32 a block of rows at a time.

**Cache.** Fetched ranges go into `~/.cache/ref_stream` up to 12 GB
(`REF_STREAM_CACHE_GB`) and are never evicted. Every generated token re-reads
the layers in the same order, and for a cyclic scan that is as good as any
eviction order. At exit the script prints the bytes fetched, the bytes read
from the cache, the layer and expert loads, the peak RSS and the wall time.

**Checks of the harness itself** (streamed model vs the same model loaded
whole by `from_pretrained` in float32, same ids, 5 greedy tokens):

| checkpoint | layout | hidden states | logits | greedy |
| --- | --- | ---: | ---: | :---: |
| `Qwen/Qwen2.5-0.5B-Instruct` | dense bf16 | all 25 identical | 0.0 | match |
| `yujiepan/qwen3-moe-tiny-random` | per-expert tensors fused by `MergeModulelist`; experts lazy | identical | 0.0 | match |
| `yujiepan/gpt-oss-tiny-random-mxfp4` | stacked MXFP4 `_blocks`/`_scales`; experts lazy and not | ≤ 3.0e-08 | 4.5e-08 | match |
| `Qwen/Qwen3-0.6B-FP8` | fine-grained FP8, 128 x 128 blocks, through `Fp8Dequantize` (`REF_STREAM_DEQUANT_DTYPE=float32`, as the float32 whole load dequantises) | all 29 identical | 0.0 | match |

The MXFP4 stub's 3e-08 is a last-bit difference, the same with lazy and with
whole-layer experts.

## gpt-oss-20b at full depth: verified

`openai/gpt-oss-20b`, all 24 layers, the two prompts of the earlier
full-depth run with the harmony system block, 24 greedy tokens each.
ditch: `ditch probe hf://openai/gpt-oss-20b --max-ram 10GB
--remote-cache-size 13GB --max-response-length 24 --residuals --json`
(ReleaseFast). Reference: `tools/ref_stream.py`, experts lazy (32 a layer,
3.2 GB as float32), MXFP4 dequantised by transformers' `Mxfp4Dequantize`.

| prompt | tokens | residuals (25 entries) | first-token logits | greedy (24 tokens) |
| --- | :---: | :---: | ---: | :---: |
| "What is the capital of France?" | match (87) | all agree, worst 3.49e-06 (layer 11) | 1.12e-06 | match |
| "What is 12 times 12?" | match (88) | all agree, worst 2.74e-06 (layer 11) | 8.23e-07 | match |

Per layer (first prompt): 0 at the embedding, then between 6.2e-07 and
3.5e-06 from layer 1 to 24, with no trend: 1.7e-06 after layer 0, 3.5e-06 at
11, 8e-07 at 15-19, 2.9e-06 at the final norm's input. The error does not
grow with depth. Greedy text on both sides: `<|channel|>analysis<|message|>We
need to answer: "What is the capital of France?" The answer: Paris. Provide
concise answer` and `... The user asks: "What is 12 times 12?" It's a simple
multiplication. 12*`.

| side | fetched | peak RSS | wall |
| --- | ---: | ---: | ---: |
| ditch (warp mode) | 11.89 GB, 1524 ranges | 7.08 GB | 885 s |
| reference | 15.25 GB, 3946 requests (+143 GB re-read from its cache) | 10.07 GB | 2070 s |

The reference fetched more than the 13.8 GB stored because its cache filled
at 12 GB, after which the uncached experts were fetched again for later
tokens. Its 1200 layer loads (24 layers x 50 forward passes) and 6920 expert
loads are what the time goes to. Peak RSS: the float32 trunk (4.6 GB:
embedding and LM head, 201088 x 2880 each), the 3 GB expert LRU, one layer.

## Bug F10 — Gemma 4 was prompted in Gemma 3's format (fixed)

**Symptom.** The first full-depth probe of `google/gemma-4-12B-it`: the text
ditch rendered differed from the release's template, and so did the ids (40
against 31 tokens).

    transformers: '<bos><|turn>system\nYou are a helpful assistant.<turn|>\n<|turn>user\nWhat is the capital of France?<turn|>\n<|turn>model\n<|channel>thought\n<channel|>'
    ditch:        '<bos><start_of_turn>user\nYou are a helpful assistant.\n\nWhat is the capital of France?<end_of_turn>\n<start_of_turn>model\n'

In Gemma 4's vocabulary `<start_of_turn>` is not a token at all, so ditch's
prompt was spelled out piece by piece (`<`, `start`, `_`, `of`, ...).

**Cause.** Gemma 4 replaced Gemma 2/3's turns: `<|turn>role\n ... <turn|>\n`.
It also gave the system prompt a turn of its own, and its generation prompt
closes an empty thinking channel (`<|channel>thought\n<channel|>`) unless
`enable_thinking` is set. The template detector had no Gemma 4 form. The
release's template does not contain `<start_of_turn>`, so detection fell
through to the model type, `gemma4_unified` starts with `gemma`, and the
registry entry said `chat = "gemma"`. The earlier Gemma 4 checks (E2B, 12B
layers 0 and 5) were run on raw prompts, which is how this went unseen.

**Fix.** A `gemma4` template, detected by `<|turn>` and named by the
`gemma4` registry entry. It renders as the release's Jinja does for plain
messages: BOS, the system turn when the first message is one, user and model
turns with trimmed content, model turns with `strip_thinking` applied (what
lies between `<|channel>` and `<channel|>` is dropped), and the generation
prompt with the empty thinking channel. Tests pin the one-turn rendering and
a four-message conversation, both copied from transformers'
`apply_chat_template` on the release's tokenizer.

## Gemma 4 12B at full depth: verified (after Bug F10)

`google/gemma-4-12B-it`, all 48 layers (sliding and global, the global
layers' 512-wide heads with keys reused as values, proportional RoPE, final
softcap), chat template with the system turn, 6 greedy tokens. ditch:
`--max-ram 12GB --remote-cache-size 23GB`, which holds the whole 22.2 GB of
text weights, so every pass after the first reads from disk. Reference:
`tools/ref_stream.py`, dense (no lazy experts), the tied 262144 x 3840
embedding as float32 in the trunk (4 GB), `REF_STREAM_CACHE_GB=23`.

| prompt | tokens | residuals (49 entries) | first-token logits | greedy (6 tokens) |
| --- | :---: | :---: | ---: | :---: |
| "What is the capital of France?" | match (31) | all agree, worst 6.58e-06 (layer 47) | 1.78e-06 | match: `The capital of France is **` |
| "Explain how rainbows form, in two sentences." | match (33) | all agree, worst 7.51e-06 (layer 47) | 1.71e-06 | match: `Rainbows form when sunlight enters` |

Per layer, both prompts: between 8e-08 and 3.5e-06 from layer 1 to 46, with
no trend with depth, then 6.6e-06 / 7.5e-06 at entry 47 (the output of
layer 46) and 2.9e-06 / 1.5e-06 at the final norm's input. The first probe,
before the template fix, ran the same arithmetic on ditch's wrong ids and
also agreed everywhere (worst 9.0e-06, again at entry 47; logits 4.4e-06),
so the fix changed the prompt and nothing else.

| side | fetched | peak RSS | wall |
| --- | ---: | ---: | ---: |
| ditch | 22.20 GB, 2841 ranges | 3.08 GB | 1557 s |
| reference | 35.98 GB, 1603 requests (+272 GB from its cache) | 8.62 GB | 853 s |

The reference fetched more than the model because its 23 GB cache stops
0.8 GB short of the 23.8 GB of shards it reads, and it fetches 16 ranges in
parallel. Measured here: one HTTP stream gets ~40 MB/s from the Hub, 4
streams ~56 MB/s, 16 streams ~110 MB/s. ditch's `hf://` source fetched at
~15-19 MB/s in both runs, so a wider fetch is the obvious speedup for ditch's
first pass.

`ditch --dry-run hf://mistralai/Mistral-Small-4-119B-2603 --max-ram 10GB
--remote-cache-size 18GB`:

    weights total 222.40GB (36 layers), trunk per layer 102.5MB
    routed expert 48.0MB decoded / 24.0MB stored (fp8), 4608 experts, 108GB stored
    warp mode: min 6.61GB, with prefetch + expert cache + RAM caches 24.28GB
    expert cache 5.75GB = 122 experts; trunk 4.62GB stored, stays cached; 558 experts fit beside it

Not run: at top-4 of 128 experts in 36 layers, a short prompt's prefill
routes to most of the 108 GB of experts and each generated token fetches
~144 of them (~3.4 GB), which at the ~12-19 MB/s the Hub serves here is
hours for the first pass, with the working set 6x the 18 GB of chunk cache
this disk allows (the same limit as gpt-oss-120b). The RAM side fits (6.6 GB
minimum). Its arithmetic is verified on the truncated cut above.

## Gemma 4 12B at full depth over `hf://`: stopped in the first pass

`google/gemma-4-12B-it` (dense, 48 layers, 22.28 GB stored, all of it trunk
re-read by every forward pass), `ditch probe hf://google/gemma-4-12B-it
--max-ram 10GB --remote-cache-size 20GB`, one prompt, 3 tokens.

    dry run:  weights 22.28GB, mapped mode 27.01GB; disk: trunk 22.28GB of chunks,
              "--remote-cache-size 23.00GB holds it" (20GB is what this disk allows)
    measured: peak RSS 3.23GB; 21GB fetched in 33 min (~10.6 MB/s, 7 MB/s at
              the start), then transfer failures from the Hub and retries;
              stopped at 13:50 UTC with no token generated

A dense model streams its trunk layer by layer, one range after another, so
its first pass runs at the single-stream rate (7-11 MB/s here), where
gpt-oss's experts load in parallel. With a cache smaller than the trunk
(20 of 22.3 GB), every later token would fetch ~2.3 GB again; with 23 GB of
cache disk the second pass would read only local chunks. The RAM side is
small (3.2 GB). Its arithmetic is verified on the truncated cut above.

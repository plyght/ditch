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

## Bug 37 — no released DeepSeek V4 could be loaded (fixed)

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

## Bug 38 — V4.1's engram hashed Indic tokens into the wrong buckets (fixed)

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

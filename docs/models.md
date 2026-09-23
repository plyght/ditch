# Supported models

ditch describes every model family it can run with one Lua model definition,
`src/models/<model_type>.lua`, compiled into the binary. **There are 93
definitions**, one per Hugging Face `model_type`, each with the tensor-name
templates, norm and residual layout, attention and MLP layouts, positional
encoding and MoE routing of that family, in terms of the building blocks
ditch implements in Zig. Your own definitions go in
`$XDG_CONFIG_HOME/ditch/models/` (see [Model definitions in
Lua](#model-definitions-in-lua), and `ditch add-model`, which drafts one from
a checkpoint). This page is the full reference; the README carries only the
summary.

How a checkpoint is matched:

* The `model_type` of `config.json` is looked up against each entry's
  `model_type` and its **aliases** (spellings of the same layout, and the
  text configs of multimodal wrappers).
* An Omni wrapper's `thinker_config` and a multimodal wrapper's `text_config`
  are unwrapped first, so the text model is what runs. A wrapper that has its
  own entry (Kimi K3, Kimi K2.5, MiMo V2, DeepSeek V4.1, GLM-5.3-Flash,
  Qwen3.8-Flash-Next) wins over the family of its text config.
* An unknown `model_type` is an error that names the related families and
  suggests `ditch add-model`. Unknown layer types, quantisation formats and
  activations are errors too, never silent fallbacks.

Two kinds of evidence, in two columns:

* **Fixture**: the family's forward pass is checked against a NumPy reference
  built from the Hugging Face implementation (`tools/make_fixture.py`,
  `src/model_test.zig`). All 93 entries have one. A fixture is written from a
  reading of the implementation, so it can share a misreading with the code;
  several families passed their fixture and still failed on real weights.
* **Real weights**: a released checkpoint, whole or cut to its first layers
  (`tools/truncate_checkpoint.py`), compared layer by layer in float32 with
  transformers or the release's own code (`ditch probe --residuals` against
  `tools/probe_reference.py`); details and every number are in
  [`tools/real-model-validation.md`](../tools/real-model-validation.md).
  **80** families are verified this way. **2** (`internlm2`, `minicpm`) are
  checked on real weights against a float32 re-implementation of the release's
  own code, because that code no longer runs under transformers 5. **11** have
  only a random-weight stub compared with transformers (the stub has the
  family's real layout but meaningless weights): their releases are gated,
  ship `.bin` without a usable tokenizer, are refused by design or were never
  published, except `deepseek_v3`, `minimax` and `minimax_m2`, whose releases
  have only been config-checked (`--dry-run`) so far. A cut checks the kept
  layers, not the full depth.

| Group | Families |
| --- | ---: |
| [Llama-layout dense](#llama-layout-dense) | 29 |
| [Gemma, GLM and ChatGLM](#gemma-glm-and-chatglm) | 6 |
| [GPT-era and other legacy decoders](#gpt-era-and-other-legacy-decoders) | 14 |
| [Mixture of experts](#mixture-of-experts) | 27 |
| [Linear-attention and convolution hybrids](#linear-attention-and-convolution-hybrids) | 9 |
| [State-space (Mamba) hybrids](#state-space-mamba-hybrids) | 5 |
| [Sparse-indexer and hyper-connection families](#sparse-indexer-and-hyper-connection-families) | 3 |
| **Total** | **93** |

## Llama-layout dense

Pre-norm attention with a gated (or relu²) MLP and llama-style tensor names.

| `model_type` | Also matches | Fixture | Real weights | What it covers / caveats |
| --- | --- | :---: | --- | --- |
| `llama` | `mistral3_text`, `smollm`, `cwm`, `emu3_text_model`, `emu3` | yes | yes: Llama-3.2-1B-Instruct, whole | GQA, llama3 rope scaling, untied `lm_head`, byte-level BPE. Yi, SOLAR, TinyLlama, SmolLM 1/2 and Mistral 3 text configs are plain llama. |
| `mistral` | `ministral` | yes | yes: Mistral-7B-Instruct-v0.3, 3 layers | llama layout with a sliding window on every layer (the fixture window is shorter than the prompt, so the local mask bites) and an explicit `head_dim` that is not `hidden_size / num_attention_heads`. |
| `ministral3` | — | yes | yes: Ministral-3-3B-Base-2512, 4 layers | Ministral 3 query scaling on every layer, optional sliding window. |
| `qwen2` | `qwen2_vl(_text)`, `qwen2_5_vl(_text)`, `qwen2_5_omni(_thinker/_text)` | yes | yes: Qwen2.5-0.5B-Instruct, whole (also FP8 and INT4) | q/k/v biases, tied embeddings. The VL and Omni text configs take the same path. |
| `qwen3` | `qwen3_vl`, `qwen3_vl_text` | yes | yes: Qwen3-0.6B, whole | per-head q/k RMSNorm. Qwen3-VL text config uses the same path. |
| `seed_oss` | — | yes | yes: Seed-OSS-36B-Instruct, 2 layers | q/k/v biases with an unbiased `o_proj` (`attention_out_bias`), explicit `head_dim`. Seed-OSS 36B. |
| `smollm3` | — | yes | yes: SmolLM3-3B, 4 layers | llama layout with `no_rope_layers`. |
| `granite` | — | yes | yes: granite-3.3-2b-instruct, 4 layers | embedding, attention and residual multipliers, logits scaling. Granite 3.x dense. |
| `granite_swa` | — | yes | stub only (no release) | Granite multipliers plus per-head attention sinks and sliding layers with their own rope base (`layer_rope_theta`). Granite 4 SWA dense. |
| `minicpm` | — | yes | by hand: MiniCPM4-0.5B, layer 0, against a re-implementation of its own code | `scale_emb`, `scale_depth` residual scaling, `dim_model_base` logit scaling. MiniCPM 1/2 (MiniCPM3 is unsupported). |
| `baichuan` | — | yes | stub only, against a re-implementation of its own code; no faithful `tokenizer.json` can be produced | fused `W_pack` (7B, RoPE). The 13B ALiBi variant (detected by `model_max_length`) is unverified. Needs a converted `tokenizer.json`. |
| `exaone` | — | yes | yes: EXAONE-3.5-2.4B-Instruct, whole | EXAONE 3.x tensor names (`transformer.h`, `attn.attention`, `c_fc_0`/`c_fc_1`). |
| `exaone4` | — | yes | yes: EXAONE-4.0.1-32B, 4 layers | post-norms, per-head q/k norm, hybrid sliding layers with RoPE and global layers without. |
| `internlm2` | — | yes | by hand: internlm2_5-1_8b-chat, 4 layers, against a re-implementation of its own code (its remote code does not run under transformers 5); needs a converted `tokenizer.json` | grouped `wqkv`, `attention.wo`, `feed_forward.w1/w2/w3`, `output.weight`. |
| `olmo` | — | yes | yes: OLMo-1B-hf, whole | non-parametric LayerNorm, `clip_qkv`. |
| `olmo2` | `olmo3` | yes | yes: OLMo-2-0425-1B-Instruct, whole | post-norms on the sublayer outputs (no input norm), q/k RMSNorm over the full projection. |
| `cohere` | `cohere2` | yes | yes: Aya Expanse 8B and Command R7B cuts | LayerNorm without bias, parallel residual, `logit_scale`, tied embeddings, per-head q/k LayerNorm. `cohere2` (Command R7B) is unverified. |
| `stablelm` | — | yes | yes: stablelm-2-1_6b-chat, whole | LayerNorm with biases, partial rotary, qkv biases. Parallel residual and `qk_layernorm` (StableLM 2 12B) are unverified. |
| `starcoder2` | — | yes | yes: starcoder2-3b, 3 layers | LayerNorm with biases, biased projections, `c_fc`/`c_proj` dense MLP, sliding window. |
| `nemotron` | — | yes | yes: Minitron-4B-Base, 3 layers (converted from `.bin`) | LayerNorm1p with bias, relu² dense MLP, partial rotary. |
| `arcee` | — | yes | yes: AFM-4.5B, 3 layers | llama attention with the two-projection relu² MLP (no gate), `mlp_bias`. AFM / Arcee. |
| `apertus` | — | yes | yes: Apertus-8B-Instruct-2509, 3 layers | attention/feedforward norm names, per-head q/k RMSNorm, two-projection xIELU MLP. Apertus (Swiss AI). |
| `bitnet` | — | yes | stub only (plain weights); the releases are refused | sub-layer RMSNorms on the attention output and the gated MLP intermediate, relu². BitNet b1.58 (released unpacked, as bf16). |
| `helium` | — | yes | yes: helium-1-preview-2b, whole | llama layout with mlp/attention biases and Helium's rotary pairing. Helium 1 (Kyutai). |
| `jais2` | — | yes | stub only; the releases are gated | LayerNorm with biases, biased projections, two-projection relu² MLP. Jais 2. |
| `nanochat` | — | yes | yes: nanochat-d20, whole | non-parametric RMSNorm everywhere, weightless per-head q/k norm after RoPE, `fc1`/`fc2` relu² MLP, final logit softcapping. |
| `hunyuan_v1_dense` | `hunyuan_vl`, `hunyuan_vl_text` | yes | yes: Hunyuan-0.5B-Instruct, whole | per-head q/k RMSNorm after RoPE, NTK-alpha dynamic rope base. Hunyuan dense and the HunYuan-VL text config. |
| `ernie4_5` | `paddleocr_vl_text` | yes | yes: ERNIE-4.5-0.3B-PT, whole | interleaved rotary, `use_bias` projections. ERNIE 4.5 dense and the PaddleOCR-VL text config. |
| `phi3` | `phi4` | yes | yes: Phi-3.5-mini-instruct, 3 layers | fused `qkv_proj` and `gate_up_proj`, longrope (short factors + attention factor). Phi-3 / 3.5 / 4-mini. |

## Gemma, GLM and ChatGLM

| `model_type` | Also matches | Fixture | Real weights | What it covers / caveats |
| --- | --- | :---: | --- | --- |
| `gemma2` | — | yes | yes: gemma-2-2b-it, 4 layers | (1+w) norms, pre/post feedforward norms, alternating local (sliding) and global layers, `query_pre_attn_scalar`, sqrt(H) embedding scale, `tanh` softcapping on the attention logits and on the output logits. |
| `gemma3` | `gemma3_text` | yes | yes: gemma-3-1b-it, 8 layers | (1+w) norms, pre/post norms, per-head (1+w) q/k norms, sqrt(H) embedding scale, sliding layers with a local rope base, `query_pre_attn_scalar`, linear rope scaling. |
| `gemma3n` | `gemma3n_text` | yes | yes: gemma-3n-E2B-it, 5 layers, and a KV-sharing cut | AltUp residual streams, Laurel blocks, per-layer input embeddings, KV-shared layers, weightless value norm, gaussian-top-k gate sparsity, final logit softcapping. |
| `gemma4` | `gemma4_text`, `gemma4_unified`, `gemma4_unified_text` | yes | yes: Gemma 4 E2B and 12B cuts | global layers with their own head size and KV heads, proportional rope on global layers, keys reused as values (`attention_k_eq_v`), KV-shared layers, per-layer inputs, `layer_scalar`, double-wide MLPs on shared layers. The MoE block of gemma-4-26B-A4B (`enable_moe_block`) is not implemented. |
| `glm4` | `glm`, `glm4v`, `glm4v_text` | yes | yes: GLM-4-9B-0414, 3 layers | `post_self_attn`/`post_mlp` norms, fused `gate_up_proj`, interleaved half rotary, q/k/v biases. GLM-4 (0414) and the GLM-4-9B HF port. |
| `chatglm` | — | yes | yes: glm-4-9b-chat, 3 layers (reference: the converted `-hf` release, with its RoPE base corrected) | ChatGLM3 / GLM-4 remote-code layout: concatenated `query_key_value` with bias, fused `dense_h_to_4h`, interleaved half rotary with `rope_ratio`, `output_layer`. |

## GPT-era and other legacy decoders

| `model_type` | Also matches | Fixture | Real weights | What it covers / caveats |
| --- | --- | :---: | --- | --- |
| `gpt2` | — | yes | yes: gpt2, whole | Conv1D (`[in][out]`) weights transposed on load and export, learned positions, fused `c_attn`, `gelu_new`, tied `lm_head`. |
| `gpt_neox` | — | yes | yes: pythia-160m, whole | head-interleaved `query_key_value`, parallel residual, `rotary_pct`, LayerNorm biases, `embed_out`. Pythia / GPT-NeoX. |
| `gpt_bigcode` | — | yes | yes: tiny_starcoder_py, whole | multi-query `c_attn`, learned positions, LayerNorm biases. StarCoder 1 / SantaCoder. |
| `gpt_neo` | — | yes | yes: gpt-neo-125m, whole | learned positions, unscaled attention logits, alternating global and `window_size` local layers. GPT-Neo 1.3B/2.7B. |
| `gptj` | — | yes | yes: gpt-j-6b, 3 layers (converted from `.bin`) | parallel residual with one LayerNorm, interleaved rotary over an absolute `rotary_dim`, `fc_in`/`fc_out` with biases. GPT-J 6B. |
| `codegen` | — | yes | yes: codegen-350M-mono, whole (converted from `.bin`) | the GPT-J layout with a fused `qkv_proj` in four tensor-parallel `[q \| v \| k]` blocks. CodeGen / CodeGen 2. |
| `falcon` | `RefinedWebModel` | yes | yes: falcon-7b-instruct, 2 layers; falcon-rw-1b, whole (converted from `.bin`) | multi-query fused qkv (7B layout), parallel attention with one LayerNorm. The 40B/180B grouped layout (`ln_attn`/`ln_mlp`) and the ALiBi variant are unverified. |
| `bloom` | — | yes | yes: bloomz-560m, whole | ALiBi, embedding LayerNorm, head-interleaved fused qkv with biases. |
| `opt` | — | yes | yes: opt-125m, whole (converted from `.bin`) | learned positions with offset 2, ReLU, LayerNorm biases. Pre-norm variants only (OPT-350m's projection layers are unsupported). |
| `mpt` | — | yes | stub only (no release reachable) | ALiBi (`alibi_bias_max`), concatenated `Wqkv`, LayerNorm without bias, `expansion_ratio`. |
| `persimmon` | — | yes | stub only; the releases ship `.bin` and no `tokenizer.json` | head-interleaved `query_key_value` with bias, per-head q/k LayerNorm, partial rotary, relu² MLP. Persimmon 8B and Fuyu's text tower. |
| `xglm` | — | yes | yes: xglm-564M, whole | fairseq sinusoidal positions with offset 2, sqrt(hidden) embedding scale, LayerNorm biases. |
| `biogpt` | — | yes | yes: biogpt, whole (converted; hand-built `tokenizer.json`) | learned positions with offset 2, sqrt(hidden) embedding scale, `output_projection` head. |
| `phi` | — | yes | yes: phi-2, 4 layers | LayerNorm with biases, parallel residual, partial rotary, `fc1`/`fc2` with biases, `lm_head` bias. Phi-1 / 1.5 / 2. |

## Mixture of experts

Dense attention with routed experts. Experts are edited per expert; see
"Expert-selective abliteration" in the README.

| `model_type` | Also matches | Fixture | Real weights | What it covers / caveats |
| --- | --- | :---: | --- | --- |
| `mixtral` | — | yes | yes: Mixtral-8x7B-Instruct-v0.1, 1 layer | softmax top-k renormalised routing over the separate per-expert tensors released Mixtral checkpoints store (`block_sparse_moe.experts.{e}.w1` / `w2` / `w3`). |
| `qwen2_moe` | — | yes | yes: Qwen1.5-MoE-A2.7B-Chat, 2 layers | softmax top-k routing over experts of `moe_intermediate_size` plus a shared expert of `shared_expert_intermediate_size` behind a sigmoid `shared_expert_gate`, both different from the dense `intermediate_size` an `mlp_only_layers` layer keeps; `decoder_sparse_step` picks the routed layers. |
| `qwen3_moe` | `qwen3_vl_moe(_text)`, `qwen3_omni_moe(_thinker/_text)` | yes | yes: Qwen3-30B-A3B, 2 layers | softmax top-k with renormalisation, dense layers via `mlp_only_layers`, separate / fused / transposed-fused expert tensors. |
| `deepseek_v2` | `deepseek_ocr2`, `deepseek_ocr2_text`, `youtu` | yes | yes: DeepSeek-V2-Lite-Chat, 2 layers | MLA with and without `q_lora_rank`, softmax group-limited top-k, shared experts, `first_k_dense_replace`, yarn with mscale. BF16/F16 and FP8 block-quantised checkpoints. |
| `deepseek_v3` | — | yes | stub only (no release run as `deepseek_v3`); the same layout runs in the verified Kimi K2.5 and DeepSeek V3.2 cuts | MLA, sigmoid routing with `e_score_correction_bias`, group-limited (`noaux_tc`) top-k, `routed_scaling_factor`, shared experts. BF16/F16 and FP8 checkpoints. |
| `deepseek_v32` | — | yes | yes: DeepSeek-V3.2-Exp, layers 0 and 3 | DeepSeek V3.2-Exp: the V3 layout whose `indexed_attention` layers run as dense attention (see [exactness bounds](#sparse-indexer-exactness-bounds)). The fixture checks that dense equivalence inside `index_topk` and that the lightning indexer's own tensors are never read and pass through exports untouched. |
| `kimi_k25` | — | yes | yes: Kimi-K2.5, 2 layers | the Kimi K2.5 / K2.6 image-video wrapper around a DeepSeek V3 text config (`kimi_k2` or `deepseek_v3` under `text_config`), `language_model` prefix. Vision tower and projector pass through exports untouched. |
| `llama4` | `llama4_text` | yes | yes: Llama 4 Scout and Maverick cuts | top-1 sigmoid routing scaling the expert input, shared expert, transposed fused experts, `no_rope_layers` with attention temperature tuning, L2 qk norm. Chunked attention runs as full attention. |
| `gpt_oss` | — | yes | yes: gpt-oss-120b cut; gpt-oss-20b at full depth | attention sinks, alternating sliding layers, yarn, router bias with top-k softmax, interleaved fused experts with biases and the clamped swiglu. BF16 and MXFP4 checkpoints. |
| `mistral4` | `mistral4_text` | yes | yes: Mistral Small 4 cut | Mistral Small 4 text config: MLA with interleaved rotary, yarn, `llama_4_scaling_beta` query scaling, softmax group-limited top-k, fused `[E][2I][H]` experts, shared experts. |
| `ernie4_5_moe` | — | yes | yes: ERNIE-4.5-21B-A3B-PT, 3 layers | interleaved rotary, softmax routing with the `moe_statics` correction bias, shared experts, `moe_layer_start_index`/interval. ERNIE 4.5 MoE (PT checkpoints). |
| `hunyuan_v1_moe` | `hunyuan` | yes | yes: Hunyuan-A13B-Instruct, 2 layers | per-head q/k RMSNorm after RoPE, NTK-alpha dynamic rope base, softmax top-k renormalised, shared MLP. Hunyuan-A13B. |
| `hy_v3` | — | yes | stub only (no text release) | sigmoid routing with correction bias, renormalisation and router scaling, shared MLP, dense/sparse `mlp_layer_types`. Hunyuan V3 (released checkpoints and the transformers module layout). |
| `granitemoe` | `granitemoeshared` | yes | yes: granite-3.0-1b-a400m-instruct, whole | Granite multipliers, fused `input_linear`/`output_linear` experts, top-k softmax routing. GraniteMoeShared adds the fused `shared_mlp`. |
| `olmoe` | — | yes | yes: OLMoE-1B-7B-0924-Instruct, 2 layers | q/k RMSNorm over the full projection, `clip_qkv`, softmax top-k routing over separate or fused experts. |
| `flex_olmo` | — | yes | yes: FlexOlmo-7x7B-1T, 2 layers | the OLMo 2 post-norm layout with OLMoE's routing. FlexOlmo. |
| `dots1` | — | yes | yes: dots.llm1.inst, 2 layers | per-head q/k norm, DeepSeek-V3 routing, shared experts, `first_k_dense_replace`. dots.llm1. |
| `exaone_moe` | — | yes | yes: K-EXAONE-236B-A23B, layers 0 and 3 | per-head q/k norm, `sliding_window_pattern` local layers, DeepSeek-V3 routing, dense/sparse `mlp_layer_types`. EXAONE 4 MoE. |
| `solar_open` | — | yes | yes: Solar-Open-100B, 2 layers | partial rotary, DeepSeek-V3 routing with shared experts on every layer. Solar Open (Upstage). |
| `afmoe` | — | yes | yes: Trinity-Nano-Preview, 4 layers | norms on both sublayer inputs and outputs, a sigmoid gate on the attention output, sliding layers every n, sigmoid routing with a selection bias, shared experts, dense first layers. AFM (Arcee) MoE. |
| `mellum` | — | yes | yes: Mellum2-12B-A2.5B-Instruct, 4 layers (16 of 64 experts) | per-head q/k norm, per-layer-type rope parameters, softmax top-k routing renormalised over fused experts. Mellum (JetBrains). |
| `laguna` | — | yes | stub only (a random-weight model built by transformers); the releases config-checked (`--dry-run`) | a softplus gate on the attention output, sigmoid routing with a tanh softcap on the router logits and a correction bias, shared experts. |
| `minimax_m2` | — | yes | stub only; the release config-checked (`--dry-run`) | q/k RMSNorm over the whole projection, partial rotary, sigmoid routing with correction bias, Mixtral-style expert tensors. |
| `minimax_m3_vl_text` | `minimax_m3_vl`, `minimax_m3` | yes | yes: MiniMax M3 cut | (1+w) norms, per-head (1+w) q/k norm, partial rotary, clamped swiglu, sigmoid MoE with a fused shared expert; `minimax_m3_sparse` layers run as dense attention (see [exactness bounds](#sparse-indexer-exactness-bounds)). The image tower is never executed. |
| `glm4_moe` | `glm4v_moe_text` | yes | yes: GLM-4.5-Air, 2 layers | dense attention with per-head q/k norms, partial rotary, sigmoid MoE with correction bias and shared experts. The `glm4v_moe` image/video wrapper runs its text config. |
| `glm4_moe_lite` | `glm_moe_lite` | yes | yes: GLM-4.7-Flash, 3 layers | GLM-4.7-Flash: DeepSeek V3 MLA with interleaved partial rotary, sigmoid MoE with correction bias, top-2 group scores, floored renormalisation, stacked expert tensors, dense first layer. |
| `mimo_v2_flash` | `mimo_v2` | yes | yes: MiMo-V2-Flash and MiMo V2.6 cuts | Xiaomi MiMo V2: hybrid full / sliding-window (128) attention with sinks and doubled kv heads, `v_head_dim < head_dim` with `attention_value_scale`, partial rotary with one base per layer type, a dense first layer then sigmoid MoE with group-limited top-k. Both the transformers spelling and the hub checkpoint spelling of MiMo-V2-Flash / V2.5 / V2.6 (including the Pro layout's fused `qkv_proj`). MTP, vision and audio tensors pass through untouched. What is verified is the transcription: the fixtures follow the transformers module and the vLLM / SGLang / llama.cpp loaders, not a released checkpoint. |

## Linear-attention and convolution hybrids

Recurrent token mixers (Gated DeltaNet, Kimi Delta Attention, lightning
attention, gated short convolutions) interleaved with softmax attention. The
recurrences run sequentially, one token at a time, so long prefills are slower
than on dense models.

| `model_type` | Also matches | Fixture | Real weights | What it covers / caveats |
| --- | --- | :---: | --- | --- |
| `lfm2` | — | yes | yes: LFM2-350M, whole | LFM2 / LFM2.5 dense: gated short-convolution layers (in_proj B/C/x split, depthwise causal conv with `conv_L_cache` taps) mixed with attention layers carrying per-head q/k norms, `embedding_norm` as the final norm. `lfm2_moe` is unsupported. |
| `qwen3_next` | — | yes | yes: Qwen3-Next-80B-A3B-Instruct, 4 layers | Gated DeltaNet linear layers (fused projections, `full_attention_interval`), sigmoid-gated full attention with per-head q/k norms, partial rotary, softmax MoE with shared expert. Qwen3-Next, Qwen3-Coder-Next. |
| `qwen3_5` | `qwen3_5_text` | yes | yes: Qwen3.5-0.8B, whole | Qwen3.5 / Qwen3.8 dense: Gated DeltaNet with split projections and explicit `layer_types`, gated full attention, partial rotary. Multimodal wrappers keep the text config nested; vision weights pass through. |
| `qwen3_5_moe` | `qwen3_5_moe_text` | yes | yes: Qwen3.5-35B-A3B, 4 layers (64 of 256 experts); Qwen3.8-2.4T cut | split-projection linear layers with a swish output gate, fused softmax MoE with shared expert. |
| `qwen4_exp` | `qwen4_exp_text` | yes | yes: Qwen3.8-Flash-Next cut | Qwen3.8-Flash-Next text config: gated hyper-connections over `hc_count` residual streams, Gated DeltaNet layers, sigmoid-gated full attention behind a QSA indexer (see [exactness bounds](#sparse-indexer-exactness-bounds)), per-layer n-gram embeddings (PLE, sharded tables read one row at a time), fused softmax MoE with a gated shared expert on every layer. |
| `minimax` | `minimax_text_01`, `minimax_m1`, `MiniMaxText01`, `MiniMaxM1` | yes | stub only; MiniMax-Text-01 and M1 config-checked (`--dry-run`) | lightning attention layers (silu qkv, per-head decay recurrence, sigmoid output gate) alternating with softmax attention, the renormalised residual layout with α/β scales, softmax top-k MoE. |
| `kimi_linear` | — | yes | yes: Kimi-Linear-48B-A3B-Instruct, 4 layers | Kimi-Linear-48B-A3B: Kimi Delta Attention with per-channel decay from a low-rank forget gate and q/k/v short convolution, 3:1 with MLA layers without RoPE, DeepSeek-V3-style sigmoid MoE with shared experts. Both the original checkpoint layout and the transformers module layout. |
| `glm5_next` | `glm5_next_text` | yes | yes: GLM-5.3-Flash cut | GLM-5.3-Flash text config: manifold-constrained hyper-connections collapsed by an unweighted mean, KDA layers with the safe lower-bound forget gate, NoPE MLA layers behind a k-pool DSA indexer (see [exactness bounds](#sparse-indexer-exactness-bounds)), sigmoid MoE with top-2 group scores and floored renormalisation, clamped SwiGLU. |
| `kimi_k3` | `kimi_k3_text` | yes | yes: Kimi-K3 cut (against its own code) | Kimi K3: the image-video wrapper around a `kimi_linear` text config with [Attention Residual](#attention-residual-kimi-k3), KDA layers with the full-rank output gate and the safe forget gate, MLA layers with a sigmoid output gate, latent MoE with 896 experts, SiTU activation, two shared experts. bf16 or the released compressed-tensors MXFP4 experts. |

## State-space (Mamba) hybrids

The selective scan runs one token at a time (heads and channels in parallel)
and its state is kept per sequence through prefill and decoding like the KV
cache. A Mamba block's `out_proj` is abliterated like an attention output
projection.

| `model_type` | Also matches | Fixture | Real weights | What it covers / caveats |
| --- | --- | :---: | --- | --- |
| `mamba2` | — | yes | yes: mamba2-130m-hf, whole | pure Mamba2 (SSD) blocks: in_proj split into gate / conv channels / dt, biased causal conv1d, grouped B/C, per-head decay, D skip, gated RMSNorm; no attention, no MLP. Mamba-Codestral, `state-spaces/mamba2-*-hf`. |
| `nemotron_h` | — | yes | yes: NVIDIA-Nemotron-Nano-9B-v2, layers 0, 1, 14 | one block per layer from `hybrid_override_pattern` / `layers_block_type` (Mamba2, attention without positional encoding, relu² MLP, non-gated experts with sigmoid group-limited routing). Nemotron-H, Nemotron 3 Nano. |
| `falcon_h1` | — | yes | yes: Falcon-H1-0.5B-Instruct, whole | Mamba2 and attention in parallel on one input norm with the muP multipliers, grouped gated RMSNorm with either gate order, and the norm-free variant. Both out projections are abliterated. |
| `jamba` | — | yes | yes: Jamba-tiny-dev, whole | Mamba1 layers with the RMS-normalised dt/B/C path at `attn_layer_period`/offset, attention without positional encoding, softmax MoE at `expert_layer_period`/offset, dense MLPs. |
| `granitemoehybrid` | — | yes | yes: granite-4.0-h-350m, whole; granite-4.0-h-tiny, 6 layers | Granite 4.0 H (tiny, small): Mamba2 and attention layers from `layer_types`, Granite multipliers, fused routed experts plus the fused `shared_mlp`, optional RoPE. |

## Sparse-indexer and hyper-connection families

| `model_type` | Also matches | Fixture | Real weights | What it covers / caveats |
| --- | --- | :---: | --- | --- |
| `glm_moe_dsa` | — | yes | yes: GLM-5.3 and altar-1 cuts | GLM-5 family: MLA with a sparse indexer run as dense attention, sigmoid MoE with correction bias and shared experts. |
| `deepseek_v4` | — | yes | yes: DeepSeek-V4-Flash, 4 layers | hyper-connections (`hc_mult` streams, Sinkhorn-mixed), low-rank q with unweighted head norm, shared-KV sliding attention with sinks and inverse-roped output, grouped output projection, CSA (overlapping pooled windows) and HCA branches with their own rope, sqrtsoftplus routing with correction bias, hash-routed (`tid2eid`) layers, clamped SwiGLU, shared expert. MTP tensors pass through. |
| `deepseek_v41` | `deepseek_v41_text` | yes | yes: DeepSeek V4.1 cut (against its own code) | DeepSeek V4.1-Flash: single-pass hyper-connections, CSA2 shared compressed KV (`kv_source` groups), FP8/FP4 quantisation-aware rounding of the window KV and latents, engram n-gram hash layers (lazy table rows, tokenizer-derived compressed ids), `gate_temp` routing. Vision tensors pass through. |

For these families the residual at layer L is the single mixed vector that
enters layer L's attention block, and the last entry is the final collapse
that enters the output norm; the edited weights are the same output and down
projections as everywhere else.

## Sparse-indexer exactness bounds

Seven families select a subset of keys with a learned indexer. ditch runs
those layers as their **exact dense equivalent** while the prompt is short
enough that the selection would have covered every key; beyond that bound the
result would be an approximation, so ditch either says so or refuses. The
prompts ditch scores are short, so in practice these families run exactly.

| Family | Mechanism | Exact while | Beyond the bound |
| --- | --- | --- | --- |
| `deepseek_v32` | lightning indexer | prompt ≤ `index_topk` tokens | yes: DeepSeek-V3.2-Exp, layers 0 and 3 | approximate |
| `glm_moe_dsa` | sparse indexer | short contexts (the ones ditch scores) | yes: GLM-5.3 and altar-1 cuts | approximate |
| `minimax_m3_vl_text` | MiniMax Sparse Attention over key blocks | prompt ≤ `index_block_size × index_topk_blocks` tokens (2048 with the released config) | yes: MiniMax M3 cut | approximate |
| `qwen4_exp` | QSA indexer | every complete key block fits `indexer_budget` | yes: Qwen3.8-Flash-Next cut | refused, not approximated |
| `glm5_next` | k-pool DSA indexer | every complete pool fits `index_topk` | yes: GLM-5.3-Flash cut | refused, not approximated |
| `deepseek_v4` | Lightning Indexer over CSA/HCA entries | every reachable compressed entry fits `index_topk` | yes: DeepSeek-V4-Flash, 4 layers | refused, not approximated |
| `deepseek_v41` | Lightning Indexer over CSA2 entries | every reachable compressed entry fits `index_topk` | yes: DeepSeek V4.1 cut (against its own code) | refused, not approximated |

Indexer weights are never edited and pass through exports untouched.

## Image, video and audio models

Qwen2/3-VL, Qwen2.5-Omni, Qwen3-Omni, Qwen3.5, GLM-4V, GLM-4.5V, GLM-5.3-Flash,
Llama 4, Kimi K2.5, Kimi K3, Gemma 3n, Gemma 4, Mistral Small 4, MiMo V2.5 /
V2.6, HunYuan-VL, PaddleOCR-VL, DeepSeek-OCR2, DeepSeek V4.1 and
Qwen3.8-Flash-Next run through their text config (the thinker's, for the Omni
wrappers). The vision and audio towers are never executed, their weights pass
through exports byte for byte, and refusal directions are measured on text
prompts. The same holds for MTP layers (`model.mtp.*`), speech embeddings,
projectors and indexer tensors.

## Attention Residual (Kimi K3)

Kimi K3 has no accumulated residual stream. Every `attn_res_block_size` layers
the running prefix is banked and restarted, and each sublayer reads a
softmax-weighted mixture of the banked block prefixes and the running one (the
weights come from a per-layer RMSNorm and a `[1, hidden]` score projection:
`self_attention_res_*`, `mlp_res_*` and, for the final norm,
`output_attn_res_*`). ditch follows the reference exactly and defines the
residual of layer `l`, for direction extraction, as the pre-norm mixture its
attention reads (the block input); the last entry is the mixture the final
norm reads. That is the `aggregate_stream` value of the SGLang implementation,
the quantity on which the layer actually operates, and the directions, kernel
and expert ranking work on it unchanged. The bank holds
`ceil(layers / attn_res_block_size)` copies of the hidden state per token in
RAM for the duration of a forward call (8 for the released 93-layer config).

Edits follow the other families: every attention output projection (`o_proj`
of the KDA and MLA layers) and every MLP down projection. K3's routed experts
are a *latent* MoE (`routed_expert_hidden_size`): their `w2` matrices write
into a 3584-wide latent space where a residual direction has no meaning, so the
shared `routed_expert_up_proj`, which writes the summed expert output into the
residual, takes the edit that the routed experts would have received (with the
kernel weight in broad mode, `weight × strength` when expert selection is on;
there is nothing to rank), while `shared_experts.down_proj` and the dense
`mlp.down_proj` are edited as usual. The Attention Residual scorers,
`routed_expert_down_proj` and `routed_expert_norm` are never touched and pass
through exports byte for byte.

## Quantised checkpoints

Quantised safetensors checkpoints are dequantised as they are read
(`src/dequant.zig`), so the loader, the kernels and the exporter see a bf16
model:

* **FP8** (`quant_method = "fp8"`: DeepSeek V3 / R1, Kimi K2 and the Qwen3 FP8
  releases; also `fbgemm_fp8` and compressed-tensors `float-quantized`):
  `weight` in F8_E4M3 or F8_E5M2 with a `weight_scale_inv` / `weight_scale`
  tensor holding one scale per `weight_block_size` tile, per row or per tensor.
* **MXFP4** (`quant_method = "mxfp4"`: gpt-oss as shipped): `*_blocks` (E2M1
  nibble pairs) and `*_scales` (E8M0 exponents) per 32 elements; the experts
  are presented in the `[E, hidden, 2I]` / `[E, I, hidden]` layout of the bf16
  gpt-oss checkpoints.
* **compressed-tensors pack-quantized** (Kimi K2.5 and other llm-compressor
  INT4/INT8 models): `weight_packed` with `num_bits`-wide fields, per-group
  `weight_scale`, optional `weight_zero_point` and `weight_shape` from
  `quantization_config`. Under `actorder` the columns of a group are
  scattered and `weight_g_idx` names each column's group; without it every
  column would be decoded with the wrong scale.
* **compressed-tensors mxfp4-pack-quantized** (Kimi K3 as released: the routed
  experts only): `weight_packed` U8 holding E2M1 nibble pairs in the natural
  `[out, in]` layout and `weight_scale` U8 E8M0 exponents per 32 elements; a
  different packing from gpt-oss's `*_blocks` / `*_scales`, decoded row by row.
* **MXFP4 `store_dtype`** (MiMo-V2.6 Pro / Flash as released): a mixed
  checkpoint, `quant_method = "fp8"` (block scales for the dense weights,
  `ignored_layers` left in bf16) with `store_dtype = "mxfp4"` for the routed
  experts, stored per projection as `weight` U8 `[out, in / 2]` and
  `weight_scale` U8 `[out, in / 32]` — the same bytes as mxfp4-pack-quantized
  (E2M1 nibble pairs, low nibble first; `w = code * 2^(scale - 127)`) under the
  plain tensor names, `mxfp4_block_size` giving the 32-element group. The bf16
  MoE router (`moe_router_dtype`) is read as it is.
* **DeepSeek V4 / V4.1 as released** (`quant_method = "fp8"`, `scale_fmt =
  "ue8m0"`, `expert_dtype = "fp4"` at the top level of config.json or inside
  `quantization_config`): the FP8 weights' block scales are F8_E8M0 exponents
  named `<module>.scale` (blocks of 128 x 128 on V4, 32 x 32 on V4.1), and the
  routed experts are `weight` I8 `[out, in / 2]` (E2M1 nibble pairs, low nibble
  first) with an F8_E8M0 `scale` `[out, in / 32]` — the MXFP4 `store_dtype`
  bytes once more. The checkpoints are in DeepSeek's own tensor names
  (`embed.weight`, `layers.N.attn.wq_a.weight`, `layers.N.ffn.experts.E.w1`),
  which ditch maps to the transformers spelling on load, the same renames as
  transformers' `conversion_mapping` for `deepseek_v4`.

Values are decoded to bf16, which is what the Hugging Face integrations
produce. With `--max-ram` (streamed weights) a tensor is decoded row-chunk by
row-chunk as it is read, so a quantised MoE never holds more than one expert of
decoded weights; without a budget the decoded tensors stay resident, so a
quantised checkpoint then needs the memory of its bf16 equivalent. Over
`hf://` the disk side is the opposite: the chunk cache holds the stored
(quantised) bytes, so its disk estimate and `--remote-cache-size` are in
stored bytes, not decoded ones. `expert_dtype` values naming one of these formats are accepted. Any other
`quantization_config` is an error naming the format.

**The bf16 export rule.** A quantised safetensors source is exported as a plain
bf16 checkpoint: every tensor, edited or not, is written in bf16 (or
`--export-dtype`) under its model name, the storage tensors (`*_blocks`,
`*_scales`, `weight_scale_inv`, `weight_packed`, …) are dropped and
`quantization_config` is removed from the exported `config.json`, so the result
loads as an ordinary bf16 model in transformers and in ditch. Quantised tensors
are never passed through byte for byte, because a file mixing bf16 edits with
the source encoding would not match any `quantization_config`. For a smaller
file, export GGUF with `--gguf-dtype q8_0`.

## GGUF per family

Nine families have a GGUF path, for reading a `.gguf` model and for
`--export-format gguf`:

`llama`, `mistral`, `mixtral`, `qwen2`, `qwen3`, `qwen2_moe`, `qwen3_moe`,
`gemma2`, `gemma3`.

Every other family loads from safetensors only; `--export-format gguf` on one
of them is an error. The registry carries the llama.cpp architecture name of
each family regardless, so adding a family to the GGUF path is a matter of
tensor-name mapping.

The ggml types ditch reads and writes:

| ggml type | read | written |
| :--- | :---: | :---: |
| F32, F16, BF16 | yes | yes |
| Q8_0, Q4_0, Q4_1, Q5_0, Q5_1 | yes | yes |
| Q4_K, Q6_K, Q8_K | yes | edited tensors become Q8_0 |
| other K-/IQ-quants | no | no |

## Tokenizers

* `tokenizer.json` (Hugging Face fast tokenizers): byte-level and
  SentencePiece-style BPE, and Unigram (SentencePiece: the highest
  log-probability segmentation, by Viterbi over the character boundaries, with
  a one-character fallback to the unknown token). XGLM, mBART and the other
  Unigram models tokenize identically to the `tokenizers` package.
* tiktoken rank files, when a model ships no `tokenizer.json`: Moonshot's
  `tiktoken.model` (Kimi K2, K2.5, K3, Kimi-Linear, with the pattern and
  special tokens of `tokenization_kimi.py`) or Meta's Llama 3
  `tokenizer.model`. Merging follows tiktoken exactly (the pair whose
  concatenation has the lowest rank is merged first; a whole pre-token that is
  in the vocabulary is one token) and is verified against the `tiktoken`
  package on fixtures from `tools/make_tiktoken_fixture.py`. Exports carry a
  `tokenizer.json` synthesised from the ranks (merges reconstructed the way
  Hugging Face converts tiktoken vocabularies, with `ignore_merges`) next to a
  copy of the original vocabulary file.
* Names of the special tokens come from `added_tokens_decoder` in
  `tokenizer_config.json`, with the families' defaults for the rest. The Kimi
  K2 chat template (`<|im_user|>user<|im_middle|>…<|im_end|>`) and the K3 XTML
  message markers are built in.
* **SentencePiece-only tokenizers are not supported** (Baichuan): generate a
  `tokenizer.json` with
  `AutoTokenizer.from_pretrained(...).save_pretrained(...)` and place it next
  to the model.
* Unicode normalisers (NFC, NFKC) are approximated by the identity. Of
  SentencePiece's `Precompiled` charsmap only the whitespace rules are applied
  — control characters, the Unicode separators and the zero-width joiners
  become a space, runs of spaces collapse, a leading run is dropped — which is
  the part ordinary text hits; the compatibility foldings are the identity.

## Not supported

These `model_type`s are recognised and rejected with the reason, rather than
being mis-run (`rejectKnownHybrid` in `src/arch.zig`):

| `model_type` | Reason |
| --- | --- |
| `kimi_k2` | use the `kimi_k25` wrapper config or a `deepseek_v3` config; the standalone `kimi_k2` `model_type` is untested |
| `dbrx` | experts stored as stacked `[E*I][H]` w1/v1/w2 blocks |
| `phimoe` | Phi-3.5-MoE: sparsemixer routing, not a plain top-k |
| `jetmoe` | mixture-of-attention-heads: the attention projections are themselves routed experts |
| `zaya` | per-channel residual scaling and a two-layer MLP router |
| `longcat_flash` | two attention and MLP blocks per layer with zero-computation experts |
| `modernbert-decoder` | a prediction head of dense + activation + norm sits between the final norm and the decoder |
| `blt` | Byte Latent Transformer: byte patching with encoder, decoder and patcher towers |
| `hrm_text` | the hierarchical recurrent reasoning loop repeats the stack |
| `diffllama` | differential attention: two attention maps combined with a learned lambda |
| `doge` | dynamic mask attention |
| `olmo_hybrid` | Mamba-style recurrent layers |
| `inkling_text` | Mamba-style recurrent layers |
| `axk2` | A.X-K2: hyper-connections, gated norms and a lightning indexer |
| `hy_v4` | Hunyuan V4: hyper-connections and a lightning indexer |
| `step3p7` | Step 3.7: per-layer head counts and per-layer SwiGLU clamps |
| `muse_glimmer_text` | centred RMSNorm and a scaled weightless q/k norm |
| `cohere_compass_text` | parallel residual with per-layer rope switching and pooling |
| `cosmos3_edge_text` | three-section mrope with frequency recomposition |
| `aria_text` | grouped-GEMM experts in an `[E][H][2I]` layout with a separate shared-expert activation |
| `ernie4_5_vl_moe_text` | separate text and vision expert sets per layer |
| `mllama`, `mllama_text_model` | Llama 3.2 Vision: cross-attention layers interleaved with the self-attention ones |
| `minicpm3` | MLA with a partial rotary over the query LoRA |
| `gpt_neox_japanese` | per-layer bias sharing |
| `cpmant` | relative position buckets |
| `ctrl` | sinusoidal positions with control codes; encoder-style blocks |
| `openai-gpt` | OpenAI GPT-1: learned positions with tied attention/feed-forward dropout blocks |
| `cohere2_moe` | Command A MoE: parallel residual with an averaged shared expert |
| `granitemoe_swa` | sinks plus the Granite MoE layout; not yet verified |
| `lfm2_moe` | short-convolution hybrid with sigmoid-routed experts; not yet verified |

Also unsupported, by layout rather than by name: Mamba1-only models (`mamba`,
FalconMamba) and RWKV, encoder-decoder models, Gemma 4 MoE
(`enable_moe_block`), HunYuan cross-layer attention (`use_cla`), OPT-350m's
projection layers, and quantisation formats other than the ones above (GPTQ,
AWQ, bitsandbytes, …).

One entry below has no released checkpoint it can run, which is worth
knowing before reaching for it. `minimax` is verified for the `postnorm:
false` layout, but every released MiniMax-Text-01 / M1 checkpoint sets
`postnorm: true` and is refused by name (`minimax_m2` and `minimax_m3` are
unaffected). The released DeepSeek V4 and V4.1 checkpoints, in DeepSeek's own
tensor names with FP4 experts, load as they are (see quantised checkpoints).

Approximations ditch does make, and says so: `dynamic` and `longrope` rope
scaling beyond the original context are treated as static / short factors
(longrope warns), chunked attention (Llama 4) runs as full attention, and the
sparse indexers run dense within the bounds above.

## Model definitions in Lua

A model definition is a Lua file that returns one family table (or a list of
them). It describes a `model_type` entirely in terms of what ditch already
implements: the layout of each block, the tensor names, and how config.json
maps onto ditch's parameters. The built-in definitions are
`src/models/*.lua`; ditch also reads every `*.lua` file in
`$XDG_CONFIG_HOME/ditch/models/` (`~/.config/ditch/models/`) and in
`--models-dir <dir>`, in that order and in file-name order, and a definition
loaded later wins over an earlier one with the same `model_type` or alias. A
file that fails to load is skipped with a warning naming the file, the family
and the field.

Definitions run in the same sandbox as `config.lua` (no files, no `require`,
no `load`); each file has its own globals.

### A worked example

Say a release ships `model_type: "acme_lm"`. Its config.json and tensor names
show the Qwen3 layout (grouped-query attention, a per-head q/k RMSNorm, a
SwiGLU MLP), except that the MLP projections are called `w1`/`w3`/`w2`, the
config has a `residual_scale` that multiplies every sublayer's output, and
every fourth layer attends globally while the others use a sliding window:

```lua
-- ~/.config/ditch/models/acme_lm.lua
return {
  model_type = "acme_lm",
  base = "qwen3", -- start from the Qwen3 definition
  chat = "chatml", -- used when the model's own template is not recognised
  notes = "Acme LM: Qwen3 layout, renamed MLP, residual scale, 3:1 local/global.",
  names = {
    gate = "mlp.w1.weight",
    up = "mlp.w3.weight",
    down = "mlp.w2.weight",
  },
  config = function(cfg, c)
    c.residual_multiplier = num(cfg.residual_scale, 1.0)
    if c.sliding_window then
      each_layer(c.sliding_layers, function(i) return (i + 1) % 4 ~= 0 end)
    end
  end,
}
```

`base` copies the named family's layout, names, hook and config function;
the new family's own fields then override them, and `names` is merged field
by field. The `model_type`, aliases, notes and `verified` flag are not
inherited. With the file in place, `ditch <acme checkpoint>` runs, and `ditch
--dry-run` plus `ditch probe` check the definition on real weights (see
`ditch add-model`, which writes this kind of file for you).

A family that is an existing one under a new name needs only the first
two lines: `return { model_type = "acme_lm", base = "qwen3" }`.

### Family fields

Every field is optional except `model_type`. Enumerations are strings.

| Field | Meaning (default) |
| --- | --- |
| `model_type` | The Hugging Face `model_type` this definition runs. |
| `aliases` | Other `model_type` spellings of the same layout, e.g. multimodal wrappers' text configs. |
| `base` | A known `model_type` to start from (built-in definitions and earlier files). |
| `llama_cpp` | llama.cpp architecture name, used by the GGUF writer and to recognise GGUF input (nil: no GGUF path). |
| `chat` | Template family used when the model's own chat template is not recognised: one of the names in `src/chat.zig` (`chatml`, `llama3`, `gemma`, `mistral`, …; `raw` = no template). A recognised template in the checkpoint always wins. |
| `verified`, `notes` | Whether a reference forward pass checks the family, and what it covers. |
| `norm` | `rms`, `rms_gemma` (`(1 + w)` scale), `layer`, `layer_1p` (Nemotron), `none` (weightless LayerNorm), `rms_none` (weightless RMSNorm). (`rms`) |
| `default_norm_eps`, `default_rope_theta` | Values when config.json has none (1e-6 for RMS norms, else 1e-5; 10000). |
| `positional` | `rope`, `learned`, `alibi`, `none`, `sinusoidal`. (`rope`) |
| `rope_style` | `neox` (rotate halves) or `gptj` (rotate pairs). (`neox`) |
| `parallel_residual` | Attention and MLP read the same input and are summed. (false) |
| `qkv` | Fused q/k/v row layout: `separate`, `concat`, `heads_interleaved`, `grouped`, `mp_blocks`. (`separate`) |
| `mlp` | `gated` (`down(act(gate) * up)`), `gated_fused` (one `[2I][H]` gate/up tensor), `dense` (`down(act(up))`). (`gated`) |
| `activation` | `silu`, `gelu_tanh`, `gelu`, `relu`, `relu2`, `quick_gelu`, when config.json names none. (`silu`) |
| `conv1d` | Weights stored `[in][out]` (GPT-2 Conv1D). (false) |
| `attention_bias`, `tie_word_embeddings` | Defaults for the config keys of the same name. (false) |
| `embed_scale_sqrt` | Embeddings scaled by `sqrt(hidden_size)`. (false) |
| `qk_norm` | `none`, `head` (one `[head_dim]` weight), `heads` (per head), `full` (over the projection), `l2` (weightless). (`none`) |
| `ssm` | Mamba flavour of the state-space layers: `none`, `mamba2`, `mamba1`. (`none`) |
| `single_mixer` | Every layer holds exactly one block behind one norm, named by `layer_types` (Mamba2, Nemotron-H). (false) |
| `parallel_ssm` | A Mamba block runs beside attention in every layer (Falcon-H1). (false) |
| `linear` | Recurrence of `linear_attention` layers: `gated_deltanet`, `kda`, `lightning`. (`gated_deltanet`) |
| `names` | Tensor-name templates (below). |
| `hook` | A Zig building block that reads family-specific config keys (below); `false` removes an inherited one. |
| `config` | `function(cfg, c)` that maps config.json onto ditch's parameters (below); runs after the hook. |

### Tensor names

`names` gives the templates of the tensors the family uses, relative to the
layer prefix unless they start with `{p}`: `{p}` is the model prefix (the
first of `prefixes` that matches the checkpoint), `{i}` the layer index and
`{e}` an expert index. Biases are found by replacing a trailing `.weight`
with `.bias`, and are optional. Set a name to `false` to say the family has no
such tensor. List-valued names hold alternative spellings; the first one
present wins. The fields are the ones of `Names` in `src/arch.zig`; the
defaults are the Llama / Hugging Face ones:

| Block | Names (default) |
| --- | --- |
| Model | `prefixes` (`model.`, `language_model.model.`, `model.language_model.`, `thinker.model.`, `language_model.`, none), `embed` (`{p}embed_tokens.weight`), `pos_embed`, `embed_norm`, `final_norm` (`{p}norm.weight`), `lm_head` (`lm_head.weight`), `layer` (`{p}layers.{i}.`) |
| Norms | `input_norm` (`input_layernorm.weight`), `post_attn_norm`, `pre_ff_norm` (`post_attention_layernorm.weight`), `post_ff_norm`, `mlp_norm`, `q_norm`, `k_norm`, `attn_sub_norm`, `ffn_sub_norm` |
| Attention | `q`, `k`, `v` (`self_attn.{q,k,v}_proj.weight`), `qkv` (fused), `o` (`self_attn.o_proj.weight`), `sinks`, `attn_gate` |
| MLA | `q_a`, `q_a_norm`, `q_b`, `kv_a`, `kv_a_norm`, `kv_b` |
| MLP | `gate`, `up`, `down` (`mlp.{gate,up,down}_proj.weight`), `gate_up` (fused) |
| MoE | `router` (`mlp.gate.weight`), `router_correction_bias`, `expert` (`mlp.experts.{e}.`), `expert_gate`/`expert_up`/`expert_down`, `fused_gate_up`/`fused_down` (stacked experts), `shared_expert` (prefix), `shared_expert_gate`, `shared_gate`/`shared_up`/`shared_down`/`shared_gate_up`, `latent_down`/`latent_up`/`latent_norm`, `moe_alt` (a second `names` table for an older checkpoint layout) |
| Linear attention | `lin_qkvz`, `lin_qkv`, `lin_z`, `lin_b`, `lin_a`, `lin_ba`, `lin_q`, `lin_k`, `lin_v`, `lin_f_a`, `lin_f_b`, `lin_g_a`, `lin_g_b`, `lin_g`, `lin_conv`, `lin_conv_split`, `lin_dt_bias`, `lin_a_log`, `lin_norm`, `lin_out`, and MiniMax lightning `light_qkv`, `light_gate`, `light_norm`, `light_out` |
| Mamba | `ssm` (the block prefix, e.g. `mixer.`; the tensors under it keep their Hugging Face names) |
| Short convolution | `conv_in`, `conv_kernel`, `conv_out` |
| Gemma 3n / 4 | `ple_*` (per-layer embeddings), `altup_*`, `laurel_*`, `layer_scale` |
| Kimi K3 | `attn_res_norm`, `attn_res_proj`, `mlp_res_norm`, `mlp_res_proj`, `output_res_norm`, `output_res_proj` |
| Other | `xielu_alpha_p`, `xielu_alpha_n` (Apertus), `hc_attn`, `hc_ffn`, `hc_attn_flat`, `hc_ffn_flat` (hyper-connection sites) |

**What abliteration edits** follows from the names: the matrices that write
into the residual stream. `attn.o_proj` is the layer's `o` (or, on a layer
without attention, `lin_out`, `light_out`, `conv_out` or the Mamba block's
`out_proj`, and beside attention on Falcon-H1 both); `mlp.down_proj` is
`down`, every routed expert's `expert_down` and the shared expert's
`shared_down` (on a latent MoE, `latent_up` instead of the routed experts'
downs). A definition therefore says what is edited by saying which tensor
fills each of those slots.

### How config.json is read

Every family goes through the same parser first; a definition only adds
what its family does differently. The generic parser reads, for every
family:

| Parameter | config.json keys (first present wins) |
| --- | --- |
| family | `model_type`, else the `architectures[0]` class (`FooForCausalLM` → `foo`); `thinker_config` and `text_config` are unwrapped |
| size | `hidden_size`/`n_embd`/`n_embed`/`d_model`, `num_attention_heads`/`n_head`/`n_heads`/…, `num_hidden_layers`/`n_layer`/… (else the length of `layer_types`/`layers_block_type`/`hybrid_override_pattern`), `num_key_value_heads`/`num_kv_heads`/`n_head_kv`/`multi_query_group_num` (`multi_query` = 1), `head_dim`/`attention_head_dim`/`kv_channels`, `intermediate_size` (a per-layer list too)/`n_inner`/`ffn_dim`/`ffn_hidden_size`, `vocab_size`/`padded_vocab_size` |
| MLA | `kv_lora_rank` with `qk_rope_head_dim`: `q_lora_rank`, `qk_nope_head_dim`, `v_head_dim` |
| RoPE | `rope_theta`/`rotary_emb_base`/`rope_base` or `rope_parameters` (flat or per layer type), `layer_rope_theta`, `partial_rotary_factor`/`rotary_pct`/`rotary_dim`, `rope_scaling` (`linear`, `llama3`, `yarn`, `longrope`, `dynamic`) |
| layer kinds | `layer_types`/`layers_block_type` entries `full_attention`, `attention`, `sliding_attention`, `chunked_attention`, `linear_attention`, `mamba`, `conv`, `mlp`, `moe`, `indexed_attention`, …; else `full_attention_interval` (every Nth layer full, the rest linear) or `sliding_window_pattern` (every Nth layer global); `sliding_window` alone makes every layer slide |
| MoE | `num_experts`/`num_local_experts`/`n_routed_experts`, `num_experts_per_tok`, `norm_topk_prob`, `moe_intermediate_size`, which layers: `decoder_sparse_step`/`interleave_moe_layer_step`/`moe_layer_freq`, `first_k_dense_replace`/`num_dense_layers`, `mlp_only_layers`, `moe_layers`, `mlp_layer_types` |
| other | `rms_norm_eps` (and the other epsilon spellings), `hidden_act`/`hidden_activation`/`activation_function`, `max_position_embeddings`, `tie_word_embeddings`, `attention_bias`, `attn_logit_softcapping`, `final_logit_softcapping`, `query_pre_attn_scalar`, `clip_qkv`, `use_parallel_residual`/`parallel_attn`, `linear_num_key_heads` and the other `linear_*` sizes, `conv_L_cache`, `quantization_config`, `expert_dtype` |

Then the family's `hook` runs, then its `config` function.

**Quantisation.** The checkpoint's `quantization_config` (FP8 blocks, MXFP4,
pack-quantized INT4, …) is read for every family; a definition only adds a
hint when the format needs one, by setting `c.quant` in its config function
(MiMo V2's `attn_row_shards`, for instance). `llama_cpp` names the GGUF
architecture a quantised GGUF export is written as.

### Config functions

`config = function(cfg, c) ... end` receives the config.json object as `cfg`
(the text config for a wrapper; JSON objects and arrays are Lua tables,
arrays 1-based; a JSON null is `null`, distinct from a missing key) and the
parsed configuration as `c`, which it changes in place. `c` holds every field
of `Config` in `src/arch.zig` under the same name: numbers, booleans,
enumerations as strings, optional values as nil, nested settings as tables
(`c.moe.scoring`, `c.mla.kv_lora_rank`, `c.rope_scaling.type`), and per-layer
tables with one entry per layer (`c.sliding_layers[1]` is layer 0). A
config function cannot change the number of layers.

Helpers every definition sees:

| Helper | |
| --- | --- |
| `num(v, d)`, `int(v, d)`, `flag(v, d)`, `str(v)`, `obj(v)` | A number, a non-negative integer (truncated), a boolean (or a non-zero integer), a string, a table, or the default `d` when `v` is missing, null or of another type: how ditch reads every key. |
| `present(v)` | The key exists, even as null. |
| `len(v)` | Length of a JSON array (0 otherwise). |
| `each_layer(list, f)` | `list[i + 1] = f(i)` for every zero-based layer `i`. |
| `f32(x)` | `x` rounded to single precision (ditch's scalars are f32). |
| `warn(msg)` | Print a warning. |
| `unsupported(msg)`, `invalid(msg)` | Refuse the checkpoint: an unsupported layout, or an inconsistent config.json. |

### Zig building blocks

A layout or a computation Lua cannot express (a new kind of layer math, a
routing rule, a recurrence) is implemented in Zig and exposed to definitions
two ways: as a value of a layout field (`norm`, `qkv`, `mlp`, `qk_norm`,
`ssm`, `linear`, `positional`, …) that the forward pass switches on, and as a
named **hook** for config.json keys whose reading is entangled with that math
(`hooks` in `src/arch.zig`). The hooks are:
`afmoe`, `baichuan`, `biogpt`, `chatglm`, `codegen`, `cohere`, `deepseek`,
`deepseek_v32`, `deepseek_v4`, `deepseek_v41`, `dense_mlp`, `dots1`,
`ernie`, `ernie_moe`, `exaone4`, `exaone_moe`, `falcon`, `falcon_h1`,
`gemma`, `gemma3n`, `gemma4`, `glm4`, `glm4_moe`, `glm4_moe_lite`,
`glm5_next`, `glm_moe_dsa`, `gpt_bigcode`, `gpt_neo`, `gpt_oss`, `gptj`,
`granite`, `granite_hybrid`, `granite_moe`, `granite_moe_swa`,
`granite_swa`, `hunyuan_dense`, `hunyuan_moe`, `hy_v3`, `jamba`,
`kimi_linear`, `laguna`, `lfm2`, `llama4`, `mamba2`, `mellum`, `mimo_v2`,
`minicpm`, `minimax`, `minimax_m2`, `minimax_m3`, `ministral3`,
`mistral4`, `mixtral`, `mpt`, `nanochat`, `nemotron`, `nemotron_h`, `neox`,
`olmo2`, `olmoe`, `opt`, `persimmon`, `phi`, `phi3`, `qwen4_exp`,
`qwen_hybrid`, `qwen_vl`, `smollm3`, `solar_open`, `stablelm`,
`starcoder2`, `xglm`, `axk1`.

A new family whose layers are all made of existing blocks needs no Zig. One
that needs new maths gets a new layout value or hook in Zig and a
definition that names it.

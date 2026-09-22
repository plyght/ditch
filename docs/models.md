# Supported models

ditch describes every model family it can run with one entry in the
architecture registry, `src/arch.zig`. **There are 93 entries**, one per
Hugging Face `model_type`, each with the tensor-name templates, norm and
residual layout, attention and MLP layouts, positional encoding and MoE
routing of that family. This page is the full reference; the README carries
only the summary.

How a checkpoint is matched:

* The `model_type` of `config.json` is looked up against each entry's
  `model_type` and its **aliases** (spellings of the same layout, and the
  text configs of multimodal wrappers).
* An Omni wrapper's `thinker_config` and a multimodal wrapper's `text_config`
  are unwrapped first, so the text model is what runs. A wrapper that has its
  own entry (Kimi K3, Kimi K2.5, MiMo V2, DeepSeek V4.1, GLM-5.3-Flash,
  Qwen3.8-Flash-Next) wins over the family of its text config.
* An unknown `model_type` is an error that names the supported list. Unknown
  layer types, quantisation formats and activations are errors too, never
  silent fallbacks.

**Verified** means the family's forward pass is checked against a NumPy
reference built from the Hugging Face implementation (`tools/make_fixture.py`,
`src/model_test.zig`). 88 of the 93 entries have such a fixture; the 5 that do
not are implemented from the reference implementation and are marked below.

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

| `model_type` | Also matches | Fixture | What it covers / caveats |
| --- | --- | :---: | --- |
| `llama` | `mistral3_text`, `smollm`, `cwm`, `emu3_text_model`, `emu3` | yes | GQA, llama3 rope scaling, untied `lm_head`, byte-level BPE. Yi, SOLAR, TinyLlama, SmolLM 1/2 and Mistral 3 text configs are plain llama. |
| `mistral` | `ministral` | no | llama layout with a sliding window on every layer and an explicit `head_dim`; covered by the llama fixture except the sliding window. |
| `ministral3` | — | yes | Ministral 3 query scaling on every layer, optional sliding window. |
| `qwen2` | `qwen2_vl(_text)`, `qwen2_5_vl(_text)`, `qwen2_5_omni(_thinker/_text)` | yes | q/k/v biases, tied embeddings. The VL and Omni text configs take the same path. |
| `qwen3` | `qwen3_vl`, `qwen3_vl_text` | yes | per-head q/k RMSNorm. Qwen3-VL text config uses the same path. |
| `seed_oss` | — | yes | q/k/v biases with an unbiased `o_proj` (`attention_out_bias`), explicit `head_dim`. Seed-OSS 36B. |
| `smollm3` | — | yes | llama layout with `no_rope_layers`. |
| `granite` | — | yes | embedding, attention and residual multipliers, logits scaling. Granite 3.x dense. |
| `granite_swa` | — | yes | Granite multipliers plus per-head attention sinks and sliding layers with their own rope base (`layer_rope_theta`). Granite 4 SWA dense. |
| `minicpm` | — | yes | `scale_emb`, `scale_depth` residual scaling, `dim_model_base` logit scaling. MiniCPM 1/2 (MiniCPM3 is unsupported). |
| `baichuan` | — | yes | fused `W_pack` (7B, RoPE). The 13B ALiBi variant (detected by `model_max_length`) is unverified. Needs a converted `tokenizer.json`. |
| `exaone` | — | yes | EXAONE 3.x tensor names (`transformer.h`, `attn.attention`, `c_fc_0`/`c_fc_1`). |
| `exaone4` | — | yes | post-norms, per-head q/k norm, hybrid sliding layers with RoPE and global layers without. |
| `internlm2` | — | yes | grouped `wqkv`, `attention.wo`, `feed_forward.w1/w2/w3`, `output.weight`. |
| `olmo` | — | yes | non-parametric LayerNorm, `clip_qkv`. |
| `olmo2` | `olmo3` | yes | post-norms on the sublayer outputs (no input norm), q/k RMSNorm over the full projection. |
| `cohere` | `cohere2` | yes | LayerNorm without bias, parallel residual, `logit_scale`, tied embeddings, per-head q/k LayerNorm. `cohere2` (Command R7B) is unverified. |
| `stablelm` | — | yes | LayerNorm with biases, partial rotary, qkv biases. Parallel residual and `qk_layernorm` (StableLM 2 12B) are unverified. |
| `starcoder2` | — | yes | LayerNorm with biases, biased projections, `c_fc`/`c_proj` dense MLP, sliding window. |
| `nemotron` | — | yes | LayerNorm1p with bias, relu² dense MLP, partial rotary. |
| `arcee` | — | yes | llama attention with the two-projection relu² MLP (no gate), `mlp_bias`. AFM / Arcee. |
| `apertus` | — | yes | attention/feedforward norm names, per-head q/k RMSNorm, two-projection xIELU MLP. Apertus (Swiss AI). |
| `bitnet` | — | yes | sub-layer RMSNorms on the attention output and the gated MLP intermediate, relu². BitNet b1.58 (released unpacked, as bf16). |
| `helium` | — | yes | llama layout with mlp/attention biases and Helium's rotary pairing. Helium 1 (Kyutai). |
| `jais2` | — | yes | LayerNorm with biases, biased projections, two-projection relu² MLP. Jais 2. |
| `nanochat` | — | yes | non-parametric RMSNorm everywhere, weightless per-head q/k norm after RoPE, `fc1`/`fc2` relu² MLP, final logit softcapping. |
| `hunyuan_v1_dense` | `hunyuan_vl`, `hunyuan_vl_text` | yes | per-head q/k RMSNorm after RoPE, NTK-alpha dynamic rope base. Hunyuan dense and the HunYuan-VL text config. |
| `ernie4_5` | `paddleocr_vl_text` | yes | interleaved rotary, `use_bias` projections. ERNIE 4.5 dense and the PaddleOCR-VL text config. |
| `phi3` | `phi4` | yes | fused `qkv_proj` and `gate_up_proj`, longrope (short factors + attention factor). Phi-3 / 3.5 / 4-mini. |

## Gemma, GLM and ChatGLM

| `model_type` | Also matches | Fixture | What it covers / caveats |
| --- | --- | :---: | --- |
| `gemma2` | — | no | the gemma3 layout with alternating local layers plus logit softcapping; covered by the gemma3 fixture except the softcapping. |
| `gemma3` | `gemma3_text` | yes | (1+w) norms, pre/post norms, per-head (1+w) q/k norms, sqrt(H) embedding scale, sliding layers with a local rope base, `query_pre_attn_scalar`, linear rope scaling. |
| `gemma3n` | `gemma3n_text` | yes | AltUp residual streams, Laurel blocks, per-layer input embeddings, KV-shared layers, weightless value norm, gaussian-top-k gate sparsity, final logit softcapping. |
| `gemma4` | `gemma4_text` | yes | global layers with their own head size and KV heads, proportional rope on global layers, keys reused as values (`attention_k_eq_v`), KV-shared layers, per-layer inputs, `layer_scalar`, double-wide MLPs on shared layers. The MoE block of gemma-4-26B-A4B (`enable_moe_block`) is not implemented. |
| `glm4` | `glm`, `glm4v`, `glm4v_text` | yes | `post_self_attn`/`post_mlp` norms, fused `gate_up_proj`, interleaved half rotary, q/k/v biases. GLM-4 (0414) and the GLM-4-9B HF port. |
| `chatglm` | — | yes | ChatGLM3 / GLM-4 remote-code layout: concatenated `query_key_value` with bias, fused `dense_h_to_4h`, interleaved half rotary with `rope_ratio`, `output_layer`. |

## GPT-era and other legacy decoders

| `model_type` | Also matches | Fixture | What it covers / caveats |
| --- | --- | :---: | --- |
| `gpt2` | — | yes | Conv1D (`[in][out]`) weights transposed on load and export, learned positions, fused `c_attn`, `gelu_new`, tied `lm_head`. |
| `gpt_neox` | — | yes | head-interleaved `query_key_value`, parallel residual, `rotary_pct`, LayerNorm biases, `embed_out`. Pythia / GPT-NeoX. |
| `gpt_bigcode` | — | yes | multi-query `c_attn`, learned positions, LayerNorm biases. StarCoder 1 / SantaCoder. |
| `gpt_neo` | — | yes | learned positions, unscaled attention logits, alternating global and `window_size` local layers. GPT-Neo 1.3B/2.7B. |
| `gptj` | — | yes | parallel residual with one LayerNorm, interleaved rotary over an absolute `rotary_dim`, `fc_in`/`fc_out` with biases. GPT-J 6B. |
| `codegen` | — | yes | the GPT-J layout with a fused `qkv_proj` in four tensor-parallel `[q \| v \| k]` blocks. CodeGen / CodeGen 2. |
| `falcon` | `RefinedWebModel` | yes | multi-query fused qkv (7B layout), parallel attention with one LayerNorm. The 40B/180B grouped layout (`ln_attn`/`ln_mlp`) and the ALiBi variant are unverified. |
| `bloom` | — | yes | ALiBi, embedding LayerNorm, head-interleaved fused qkv with biases. |
| `opt` | — | yes | learned positions with offset 2, ReLU, LayerNorm biases. Pre-norm variants only (OPT-350m's projection layers are unsupported). |
| `mpt` | — | yes | ALiBi (`alibi_bias_max`), concatenated `Wqkv`, LayerNorm without bias, `expansion_ratio`. |
| `persimmon` | — | yes | head-interleaved `query_key_value` with bias, per-head q/k LayerNorm, partial rotary, relu² MLP. Persimmon 8B and Fuyu's text tower. |
| `xglm` | — | yes | fairseq sinusoidal positions with offset 2, sqrt(hidden) embedding scale, LayerNorm biases. |
| `biogpt` | — | yes | learned positions with offset 2, sqrt(hidden) embedding scale, `output_projection` head. |
| `phi` | — | yes | LayerNorm with biases, parallel residual, partial rotary, `fc1`/`fc2` with biases, `lm_head` bias. Phi-1 / 1.5 / 2. |

## Mixture of experts

Dense attention with routed experts. Experts are edited per expert; see
"Expert-selective abliteration" in the README.

| `model_type` | Also matches | Fixture | What it covers / caveats |
| --- | --- | :---: | --- |
| `mixtral` | — | no | softmax top-k renormalised routing with Mixtral tensor names; shares the `qwen3_moe` routing path, no fixture of its own. |
| `qwen2_moe` | — | no | the `qwen3_moe` routing plus a sigmoid-gated shared expert (no fixture for the shared expert). |
| `qwen3_moe` | `qwen3_vl_moe(_text)`, `qwen3_omni_moe(_thinker/_text)` | yes | softmax top-k with renormalisation, dense layers via `mlp_only_layers`, separate / fused / transposed-fused expert tensors. |
| `deepseek_v2` | `deepseek_ocr2`, `deepseek_ocr2_text`, `youtu` | yes | MLA with and without `q_lora_rank`, softmax group-limited top-k, shared experts, `first_k_dense_replace`, yarn with mscale. BF16/F16 and FP8 block-quantised checkpoints. |
| `deepseek_v3` | — | yes | MLA, sigmoid routing with `e_score_correction_bias`, group-limited (`noaux_tc`) top-k, `routed_scaling_factor`, shared experts. BF16/F16 and FP8 checkpoints. |
| `deepseek_v32` | — | no | DeepSeek V3.2-Exp: the V3 layout whose `indexed_attention` layers run as dense attention (see [exactness bounds](#sparse-indexer-exactness-bounds)). Covered by the `deepseek_v3` fixture. |
| `kimi_k25` | — | yes | the Kimi K2.5 / K2.6 image-video wrapper around a DeepSeek V3 text config (`kimi_k2` or `deepseek_v3` under `text_config`), `language_model` prefix. Vision tower and projector pass through exports untouched. |
| `llama4` | `llama4_text` | yes | top-1 sigmoid routing scaling the expert input, shared expert, transposed fused experts, `no_rope_layers` with attention temperature tuning, L2 qk norm. Chunked attention runs as full attention. |
| `gpt_oss` | — | yes | attention sinks, alternating sliding layers, yarn, router bias with top-k softmax, interleaved fused experts with biases and the clamped swiglu. BF16 and MXFP4 checkpoints. |
| `mistral4` | `mistral4_text` | yes | Mistral Small 4 text config: MLA with interleaved rotary, yarn, `llama_4_scaling_beta` query scaling, softmax group-limited top-k, fused `[E][2I][H]` experts, shared experts. |
| `ernie4_5_moe` | — | yes | interleaved rotary, softmax routing with the `moe_statics` correction bias, shared experts, `moe_layer_start_index`/interval. ERNIE 4.5 MoE (PT checkpoints). |
| `hunyuan_v1_moe` | `hunyuan` | yes | per-head q/k RMSNorm after RoPE, NTK-alpha dynamic rope base, softmax top-k renormalised, shared MLP. Hunyuan-A13B. |
| `hy_v3` | — | yes | sigmoid routing with correction bias, renormalisation and router scaling, shared MLP, dense/sparse `mlp_layer_types`. Hunyuan V3 (released checkpoints and the transformers module layout). |
| `granitemoe` | `granitemoeshared` | yes | Granite multipliers, fused `input_linear`/`output_linear` experts, top-k softmax routing. GraniteMoeShared adds the fused `shared_mlp`. |
| `olmoe` | — | yes | q/k RMSNorm over the full projection, `clip_qkv`, softmax top-k routing over separate or fused experts. |
| `flex_olmo` | — | yes | the OLMo 2 post-norm layout with OLMoE's routing. FlexOlmo. |
| `dots1` | — | yes | per-head q/k norm, DeepSeek-V3 routing, shared experts, `first_k_dense_replace`. dots.llm1. |
| `exaone_moe` | — | yes | per-head q/k norm, `sliding_window_pattern` local layers, DeepSeek-V3 routing, dense/sparse `mlp_layer_types`. EXAONE 4 MoE. |
| `solar_open` | — | yes | partial rotary, DeepSeek-V3 routing with shared experts on every layer. Solar Open (Upstage). |
| `afmoe` | — | yes | norms on both sublayer inputs and outputs, a sigmoid gate on the attention output, sliding layers every n, sigmoid routing with a selection bias, shared experts, dense first layers. AFM (Arcee) MoE. |
| `mellum` | — | yes | per-head q/k norm, per-layer-type rope parameters, softmax top-k routing renormalised over fused experts. Mellum (JetBrains). |
| `laguna` | — | yes | a softplus gate on the attention output, sigmoid routing with a tanh softcap on the router logits and a correction bias, shared experts. |
| `minimax_m2` | — | yes | q/k RMSNorm over the whole projection, partial rotary, sigmoid routing with correction bias, Mixtral-style expert tensors. |
| `minimax_m3_vl_text` | `minimax_m3_vl`, `minimax_m3` | yes | (1+w) norms, per-head (1+w) q/k norm, partial rotary, clamped swiglu, sigmoid MoE with a fused shared expert; `minimax_m3_sparse` layers run as dense attention (see [exactness bounds](#sparse-indexer-exactness-bounds)). The image tower is never executed. |
| `glm4_moe` | `glm4v_moe_text` | yes | dense attention with per-head q/k norms, partial rotary, sigmoid MoE with correction bias and shared experts. The `glm4v_moe` image/video wrapper runs its text config. |
| `glm4_moe_lite` | `glm_moe_lite` | yes | GLM-4.7-Flash: DeepSeek V3 MLA with interleaved partial rotary, sigmoid MoE with correction bias, top-2 group scores, floored renormalisation, stacked expert tensors, dense first layer. |
| `mimo_v2_flash` | `mimo_v2` | yes | Xiaomi MiMo V2: hybrid full / sliding-window (128) attention with sinks and doubled kv heads, `v_head_dim < head_dim` with `attention_value_scale`, partial rotary with one base per layer type, a dense first layer then sigmoid MoE with group-limited top-k. Both the transformers spelling and the hub checkpoint spelling of MiMo-V2-Flash / V2.5 / V2.6 (including the Pro layout's fused `qkv_proj`). MTP, vision and audio tensors pass through untouched. What is verified is the transcription: the fixtures follow the transformers module and the vLLM / SGLang / llama.cpp loaders, not a released checkpoint. |

## Linear-attention and convolution hybrids

Recurrent token mixers (Gated DeltaNet, Kimi Delta Attention, lightning
attention, gated short convolutions) interleaved with softmax attention. The
recurrences run sequentially, one token at a time, so long prefills are slower
than on dense models.

| `model_type` | Also matches | Fixture | What it covers / caveats |
| --- | --- | :---: | --- |
| `lfm2` | — | yes | LFM2 / LFM2.5 dense: gated short-convolution layers (in_proj B/C/x split, depthwise causal conv with `conv_L_cache` taps) mixed with attention layers carrying per-head q/k norms, `embedding_norm` as the final norm. `lfm2_moe` is unsupported. |
| `qwen3_next` | — | yes | Gated DeltaNet linear layers (fused projections, `full_attention_interval`), sigmoid-gated full attention with per-head q/k norms, partial rotary, softmax MoE with shared expert. Qwen3-Next, Qwen3-Coder-Next. |
| `qwen3_5` | `qwen3_5_text` | yes | Qwen3.5 / Qwen3.8 dense: Gated DeltaNet with split projections and explicit `layer_types`, gated full attention, partial rotary. Multimodal wrappers keep the text config nested; vision weights pass through. |
| `qwen3_5_moe` | `qwen3_5_moe_text` | yes | split-projection linear layers with a swish output gate, fused softmax MoE with shared expert. |
| `qwen4_exp` | `qwen4_exp_text` | yes | Qwen3.8-Flash-Next text config: gated hyper-connections over `hc_count` residual streams, Gated DeltaNet layers, sigmoid-gated full attention behind a QSA indexer (see [exactness bounds](#sparse-indexer-exactness-bounds)), per-layer n-gram embeddings (PLE, sharded tables read one row at a time), fused softmax MoE with a gated shared expert on every layer. |
| `minimax` | `minimax_text_01`, `minimax_m1`, `MiniMaxText01`, `MiniMaxM1` | yes | lightning attention layers (silu qkv, per-head decay recurrence, sigmoid output gate) alternating with softmax attention, the renormalised residual layout with α/β scales, softmax top-k MoE. |
| `kimi_linear` | — | yes | Kimi-Linear-48B-A3B: Kimi Delta Attention with per-channel decay from a low-rank forget gate and q/k/v short convolution, 3:1 with MLA layers without RoPE, DeepSeek-V3-style sigmoid MoE with shared experts. Both the original checkpoint layout and the transformers module layout. |
| `glm5_next` | `glm5_next_text` | yes | GLM-5.3-Flash text config: manifold-constrained hyper-connections collapsed by an unweighted mean, KDA layers with the safe lower-bound forget gate, NoPE MLA layers behind a k-pool DSA indexer (see [exactness bounds](#sparse-indexer-exactness-bounds)), sigmoid MoE with top-2 group scores and floored renormalisation, clamped SwiGLU. |
| `kimi_k3` | `kimi_k3_text` | yes | Kimi K3: the image-video wrapper around a `kimi_linear` text config with [Attention Residual](#attention-residual-kimi-k3), KDA layers with the full-rank output gate and the safe forget gate, MLA layers with a sigmoid output gate, latent MoE with 896 experts, SiTU activation, two shared experts. bf16 or the released compressed-tensors MXFP4 experts. |

## State-space (Mamba) hybrids

The selective scan runs one token at a time (heads and channels in parallel)
and its state is kept per sequence through prefill and decoding like the KV
cache. A Mamba block's `out_proj` is abliterated like an attention output
projection.

| `model_type` | Also matches | Fixture | What it covers / caveats |
| --- | --- | :---: | --- |
| `mamba2` | — | yes | pure Mamba2 (SSD) blocks: in_proj split into gate / conv channels / dt, biased causal conv1d, grouped B/C, per-head decay, D skip, gated RMSNorm; no attention, no MLP. Mamba-Codestral, `state-spaces/mamba2-*-hf`. |
| `nemotron_h` | — | yes | one block per layer from `hybrid_override_pattern` / `layers_block_type` (Mamba2, attention without positional encoding, relu² MLP, non-gated experts with sigmoid group-limited routing). Nemotron-H, Nemotron 3 Nano. |
| `falcon_h1` | — | yes | Mamba2 and attention in parallel on one input norm with the muP multipliers, grouped gated RMSNorm with either gate order, and the norm-free variant. Both out projections are abliterated. |
| `jamba` | — | yes | Mamba1 layers with the RMS-normalised dt/B/C path at `attn_layer_period`/offset, attention without positional encoding, softmax MoE at `expert_layer_period`/offset, dense MLPs. |
| `granitemoehybrid` | — | yes | Granite 4.0 H (tiny, small): Mamba2 and attention layers from `layer_types`, Granite multipliers, fused routed experts plus the fused `shared_mlp`, optional RoPE. |

## Sparse-indexer and hyper-connection families

| `model_type` | Also matches | Fixture | What it covers / caveats |
| --- | --- | :---: | --- |
| `glm_moe_dsa` | — | yes | GLM-5 family: MLA with a sparse indexer run as dense attention, sigmoid MoE with correction bias and shared experts. |
| `deepseek_v4` | — | yes | hyper-connections (`hc_mult` streams, Sinkhorn-mixed), low-rank q with unweighted head norm, shared-KV sliding attention with sinks and inverse-roped output, grouped output projection, CSA (overlapping pooled windows) and HCA branches with their own rope, sqrtsoftplus routing with correction bias, hash-routed (`tid2eid`) layers, clamped SwiGLU, shared expert. MTP tensors pass through. |
| `deepseek_v41` | `deepseek_v41_text` | yes | DeepSeek V4.1-Flash: single-pass hyper-connections, CSA2 shared compressed KV (`kv_source` groups), FP8/FP4 quantisation-aware rounding of the window KV and latents, engram n-gram hash layers (lazy table rows, tokenizer-derived compressed ids), `gate_temp` routing. Vision tensors pass through. |

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
| `deepseek_v32` | lightning indexer | prompt ≤ `index_topk` tokens | approximate |
| `glm_moe_dsa` | sparse indexer | short contexts (the ones ditch scores) | approximate |
| `minimax_m3_vl_text` | MiniMax Sparse Attention over key blocks | prompt ≤ `index_block_size × index_topk_blocks` tokens (2048 with the released config) | approximate |
| `qwen4_exp` | QSA indexer | every complete key block fits `indexer_budget` | refused, not approximated |
| `glm5_next` | k-pool DSA indexer | every complete pool fits `index_topk` | refused, not approximated |
| `deepseek_v4` | Lightning Indexer over CSA/HCA entries | every reachable compressed entry fits `index_topk` | refused, not approximated |
| `deepseek_v41` | Lightning Indexer over CSA2 entries | every reachable compressed entry fits `index_topk` | refused, not approximated |

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
  `quantization_config`.
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

Values are decoded to bf16, which is what the Hugging Face integrations
produce. With `--max-ram` (streamed weights) a tensor is decoded row-chunk by
row-chunk as it is read, so a quantised MoE never holds more than one expert of
decoded weights; without a budget the decoded tensors stay resident, so a
quantised checkpoint then needs the memory of its bf16 equivalent.
`expert_dtype` values naming one of these formats are accepted. Any other
`quantization_config` is an error naming the format. The FP4 (e2m1) expert
weights of the released DeepSeek V4 / V4.1 checkpoints are refused until
dequantised.

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
  SentencePiece-style BPE.
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
* Unicode normalisers (NFC, NFKC, Precompiled) are approximated by the
  identity.

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

Approximations ditch does make, and says so: `dynamic` and `longrope` rope
scaling beyond the original context are treated as static / short factors
(longrope warns), chunked attention (Llama 4) runs as full attention, and the
sparse indexers run dense within the bounds above.

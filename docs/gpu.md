# Compute backends

ditch runs on the CPU by default and the CPU kernels in `src/tensor.zig` are the
reference implementation: everything else is judged against them. This document
describes the backend seam, the Metal and Vulkan backends, the reference test
harness, and exactly what has and has not been run on real hardware.

## The seam

`src/compute.zig` sits between the forward pass and the kernels. `src/model.zig`,
`src/moe.zig`, `src/deepseek_v4.zig` and `src/qwen4_exp.zig` call
`compute.matmulT`, `compute.rmsnorm`, `compute.softmaxInPlace` and so on instead
of reaching into `tensor.zig`; the arguments are unchanged, so a CPU run is the
same computation, instruction for instruction, that it was before the seam
existed (`src/compute.zig`'s tests assert bitwise equality with the direct
calls).

A backend is a `compute.Device`: a name, a kind, an opaque context and a vtable
of optional function pointers.

```zig
pub const VTable = struct {
    matmulT:          ?*const fn (ctx, out, x, n, w) Error!void = null,
    matvecTMulti:     ?*const fn (ctx, out, w, y, q) Error!void = null,
    rowNorms:         ?*const fn (ctx, out, w) Error!void = null,
    attentionScores:  ...,  attentionValues: ...,
    gatedActivation:  ...,  rmsnormRows:     ...,  layernormRows: ...,
    softmaxRows:      ...,  ropeRows:        ...,
};
```

Two rules make this safe for ditch's access pattern:

1. **Null means "the CPU does it".** A backend implements what it can.
2. **`error.Unsupported` means "the CPU does this one".** A backend may refuse
   any individual call — a quantised tile, a shape it has no kernel for, a
   dtype it cannot read — and the call falls through to `tensor.zig`.

The device is chosen once at start-up (`--device`, `DITCH_DEVICE`, `device` in
`config.lua`) and stored in `compute.active`, which is read-only afterwards, so
kernels may be called from any pool thread.

| `--device` | What it opens |
| --- | --- |
| `cpu` (default) | the reference kernels in `tensor.zig` |
| `metal` | Apple silicon, in a `-Dmetal` build |
| `vulkan` | the best Vulkan device: a discrete GPU, else an integrated one; a software device such as Mesa's lavapipe too, when it is the only one |
| `auto` | Metal on a `-Dmetal` build, otherwise a discrete or integrated Vulkan GPU (never a software one), otherwise the CPU with one line on stderr saying why |

An explicitly requested backend that cannot be opened stops the run with exit
code 2 and one line naming the reason (no loader, no driver, no suitable
device, a static build).

### What is dispatched

Only `matmulT`, `matvecTMulti` and `rowNorms`. Those dominate the FLOPs and
their natural unit of work is a whole weight tile, which is exactly what a
device can take, compute on and give back. The element-wise kernels are called
per row or per head from inside the thread pool on host-resident activations; a
device round trip per row costs more than the arithmetic it saves. Their device
kernels are implemented and checked by the selftest so that a future forward
pass which keeps activations resident can turn them on; `Device.dispatched`
reports the current state per operation.

Work below `compute.min_device_macs` multiply-accumulates (1 Mi) stays on the
CPU in any case. `ditch selftest` lowers that threshold to zero so even a 1x1
matrix is checked on the device.

### Weights and device memory

A device never has to hold the model.

* **Upload, compute, drop** is the default. `matmulT` gets one `tensor.Weight`,
  which in streamed or warp mode is a tile that was just read from disk. The
  backend copies its raw bytes, in their on-disk dtype, into one reused device
  buffer and the kernel converts them as it reads them. There is no host-side
  conversion and no second copy.
* **Page-aligned host memory is addressed in place** on Metal. Memory-mapped
  weights usually are, and `newBufferWithBytesNoCopy:` then gives the GPU the
  same pages the CPU sees. (The Vulkan backend always copies: importing host
  pages needs an extension that not every driver has.)
* **`--gpu-memory N`** turns on `compute.Residency`, a byte-budgeted LRU of
  device buffers keyed by the host address and length of the tile, capped by
  the device's own recommended working set. It is only active when host weight
  pointers are stable for the run, i.e. in memory-mapped mode: streamed and warp
  modes reuse the same buffers for different tensors, so caching by address
  would hand back the wrong weights. There, every tile is uploaded per call.
  Closing a mapped file (a model, or an export being validated) clears the
  cache, since the next mapping may be given the same addresses, and each
  cached tile carries a hash of sampled bytes, so different contents at a
  cached address are a miss rather than stale weights. On Vulkan the cap is
  three quarters of the largest device-local heap.

`--max-ram` is unaffected: device buffers are not ditch-owned host allocations,
and the budget still governs the host side exactly as before.

### Abliteration semantics

Unchanged. The refusal directions, the low-rank deltas and the exported weights
are computed from the same formulas; a GPU only changes the order in which f32
products are summed. `matmulT`'s low-rank delta is applied on the host after a
device matmul, with the same arithmetic as the fused CPU kernel. The manifest
(`ditch-reproduce.lua`) records the device that ran the study, and reproduction
does not force that device back on — a manifest from a Metal or Vulkan run
reproduces on a CPU, within the tolerance below.

## The Metal backend

`src/metal/` contains:

| File | What it is |
| --- | --- |
| `shaders.metal` | the compute kernels, embedded in the binary with `@embedFile` and compiled at run time with `newLibraryWithSource:` |
| `shim.m`, `shim.h` | a small Objective-C shim: device, queue, library, pipeline cache, shared buffers, one generic dispatch entry point |
| `backend.zig` | the Zig side: extern declarations of the shim, parameter blocks, buffer staging, residency, the `compute.VTable` |
| `shaders_test.zig` | static checks of the shader source that run everywhere |

Kernels: tiled matmul (16x16 threadgroup tiles, f32 accumulators, one variant
per weight dtype f32/f16/bf16), `W^T y`, row L2 norms, attention scores and
values, gated activations (all six of `tensor.Activation`), RMSNorm (plain,
`(1 + w)`, non-parametric), LayerNorm (weight, bias, `(1 + w)`, non-parametric),
softmax and rope (rotate-half and interleaved). bf16 is read as `ushort` and
shifted into the f32 exponent field rather than using MSL's `bfloat`, so no
recent Metal version is required. Fast math is off
(`MTLMathModeSafe`, or `fastMathEnabled = NO` on older SDKs) to keep the
numerics close to the CPU.

Building: `zig build -Dmetal` on macOS. The option is off by default because the
shim needs Apple's SDK headers, which Zig does not ship — so cross-compiling
`-Dtarget=aarch64-macos` from any host still works and still produces a CPU-only
binary. `-Dmetal` with a non-macOS target is refused with a clear message.

## The Vulkan backend

`src/vulkan/` contains:

| File | What it is |
| --- | --- |
| `vk.zig` | the Vulkan 1.0 types, constants and entry points the backend uses, declared by hand; the loader is opened at run time |
| `device.zig` | device choice, memory types, buffers, and the one synchronous submission path (copies in, one dispatch, copies out, fence) |
| `backend.zig` | the kernels: which module, which buffers, which grid; residency; the `compute.VTable` |
| `shaders/*.comp` | the GLSL compute kernels, the same algorithms as `shaders.metal` |
| `spirv/*.spv`, `spirv/SHA256SUMS` | the compiled SPIR-V, embedded with `@embedFile`, and the hashes of the sources it was compiled from |
| `vk_test.zig`, `spirv_test.zig`, `backend_test.zig` | struct layouts against `vulkan_core.h`; shader freshness, push constants and workgroup sizes against the Zig side; the kernels on a live device when there is one |

**No build-time dependency.** Nothing links against the Vulkan SDK or the
loader. At start-up `--device vulkan` (or `auto`) opens `libvulkan.so.1` with
`dlopen` on Linux or `vulkan-1.dll` with `LoadLibraryA` on Windows and resolves
every function through `vkGetInstanceProcAddr`/`vkGetDeviceProcAddr`. A machine
without Vulkan runs the same binary on the CPU. The backend is built by default
for Linux and Windows (`-Dvulkan=false` leaves it out). One limit: a statically
linked musl binary — the `*-linux-musl` release archives — has no dynamic
linker to load the loader with (and the loader and drivers are built against
the system's glibc), so there the backend reports itself unavailable. Use the
`*-linux-gnu` release archive or build from source on the machine
(`zig build -Doptimize=ReleaseFast` links against the system's glibc).

**Why GLSL and committed SPIR-V, not Zig's SPIR-V backend.** Zig 0.16 can emit
SPIR-V, but its GPU support is still experimental: workgroup-shared memory,
barriers, push constants and the storage-buffer layout rules the kernels depend
on are thinly documented and change between releases, and a miscompile there
would be invisible until it ran on each vendor's driver. GLSL through
`glslangValidator` is the path every Vulkan driver is tested against. The
compiled modules are committed so that building ditch needs no shader compiler:

```sh
bash tools/gen_spirv.sh           # regenerate spirv/*.spv and spirv/SHA256SUMS
bash tools/gen_spirv.sh --check   # recompile, spirv-val, and fail if anything is stale
```

`zig build test` hashes the GLSL it embeds and fails when `SHA256SUMS` does not
match, so an edited shader without regenerated SPIR-V fails everywhere, not only
where a shader compiler is installed.

**Kernels.** The same set as Metal, with the same tiling, summation order and
parameter blocks (as push constants): tiled matmul (16x16 workgroups, f32
accumulators) and `W^T y` and row norms for f32, f16 and bf16 weights; attention
scores and values; the six gated activations; RMSNorm and LayerNorm in all
their variants; softmax; rope in both styles. Weight tiles are bound as arrays
of 32-bit words and converted as they are read: f16 through `unpackHalf2x16`
(core GLSL), bf16 by a shift into the f32 exponent field. So neither
`shaderFloat16` nor 16-bit storage is required, and every Vulkan 1.0 device
reads every dtype without a host-side conversion. `tanh` in `gelu_tanh` is
written as `1 - 2/(exp(2u)+1)`, like the CPU's vector kernel, so it saturates
instead of overflowing on drivers that compute it from exponentials. Row
reductions number their rows across a two-dimensional grid, so a
vocabulary-sized matrix (more rows than the 65535 workgroups a dimension is
guaranteed) is fine. Quantised tiles, tiles larger than the device's
`maxStorageBufferRange`, and allocations the device refuses fall through to the
CPU.

**Memory.**

* *Discrete GPUs*: every buffer a kernel touches is device-local. Inputs and
  weight tiles are written into one host-visible staging buffer and copied in
  the same command buffer as the dispatch; results come back through a
  host-visible read-back buffer (host-cached where the driver offers it).
* *Integrated GPUs and other unified-memory devices*: when a memory type is both
  device-local and host-visible, kernel buffers live there and are written and
  read in place — no staging copies.
* `--gpu-memory N` keeps weight tiles resident in device-local memory, as on
  Metal, capped at three quarters of the largest device-local heap.

**Choosing among several GPUs.** `DITCH_VULKAN_DEVICE=<index>` picks a device
by its position in Vulkan's enumeration (the order `vulkaninfo --summary` lists
them). `DITCH_VULKAN_STAGING=1` forces the discrete-GPU memory path on a
unified-memory device, which is how that path is tested without a discrete GPU.

**Performance expectations.** Every call is synchronous (submit, wait on a
fence), as on Metal, and only the weight-tile products are dispatched. In
upload-per-call mode a discrete GPU pays a PCIe copy of every tile for every
product, which for decode-shaped work (one or a few input rows) usually costs
more than the CPU spends computing it; `--gpu-memory` with memory-mapped
weights removes that copy, and prefill-shaped products (many rows) are where a
GPU wins most. `ditch bench --kernels --device vulkan` measures both regimes on
the machine at hand, next to the CPU.

## The reference harness

`src/selftest.zig` compares a backend against the CPU kernels on random inputs
across a shape sweep that includes the edge cases — 1x1, single rows, single
columns, sizes that are not a multiple of the vector width (16) or the device
tile — for all three weight dtypes, all activations and all norm variants.

```sh
ditch selftest                  # the CPU backend against itself: exactly zero error
ditch selftest --device metal   # the GPU kernels against the CPU reference
ditch selftest --device vulkan  # likewise on Vulkan
ditch selftest --device vulkan --json
```

It prints one row per kernel: where it ran, how many cases and elements were
compared, the largest absolute and relative error, the tolerance and the
verdict. An element passes if it is within the absolute *or* the relative
tolerance; a kernel passes if every element does.

| Kernel group | Absolute | Relative |
| --- | --- | --- |
| matmul, matvec_t | 1e-4 | 1e-3 |
| row_norms, attn_scores, attn_values | 1e-4 | 1e-4 |
| gated_activation, rmsnorm, layernorm, softmax, rope | 1e-5 | 1e-4 |

Exit codes: 0 all inside tolerance, 1 a kernel outside it, 2 the device is not
available. `--json` and `--plain` follow the usual convention (results on
stdout, messages on stderr).

## What is verified where

| Component | Checked by | Where it has run |
| --- | --- | --- |
| The seam and the CPU backend | `zig build test` (bitwise equality with `tensor.zig`, delta correction, residency LRU) | Linux x86-64, every CI run |
| The harness itself | `zig build test` (zero error against the CPU; a deliberately wrong backend is caught) | Linux x86-64, every CI run |
| Metal backend, Zig half | `zig build metal-check` (compiled for aarch64-macos, no SDK needed) | Linux x86-64, every CI run |
| `shaders.metal`, structure | `src/metal/shaders_test.zig`: every kernel the backend asks for exists, delimiters balance, parameter blocks have Zig counterparts, tile widths agree | Linux x86-64, every CI run |
| `shaders.metal`, syntax and semantics | `xcrun -sdk macosx metal -c` | macOS CI job (`macos-26`) |
| Metal kernels, numerics | `ditch selftest --device metal` | macOS CI job, on a real GPU |
| The whole pipeline on Metal | an abliteration of the fixture model on both devices, exports compared element by element (`tools/compare_exports.py`) | macOS CI job |
| Vulkan structures and shaders, statically | `zig build test`: struct sizes and offsets against `vulkan_core.h`, committed SPIR-V against the GLSL hashes, push constant blocks and workgroup sizes against the Zig side | every CI run |
| Vulkan shaders compile and validate | `tools/gen_spirv.sh --check` (glslangValidator, spirv-val) | `vulkan` CI job |
| Vulkan kernels, numerics | `ditch selftest --device vulkan`, on both memory paths | `vulkan` CI job, on Mesa's lavapipe (software) |
| The whole pipeline on Vulkan | `tests/vulkan_e2e.sh`: an abliteration of the fixture model on the CPU and on Vulkan, with and without `--gpu-memory`, exports compared; first-token logits of ten fixture architectures compared | `vulkan` CI job, on lavapipe |
| Release builds without Vulkan | the cross-compile of every release target; the static musl binary reports the backend unavailable and runs on the CPU | every CI run |

What cannot be checked without a Mac, and is therefore not claimed here:
MSL syntax and semantics, buffer-index and attribute correctness, threadgroup
memory limits, the Objective-C shim compiling and linking against the real SDK,
and every number the GPU produces. If the macOS job has not run green on a
change, the Metal path is compile-checked only.

lavapipe is a conformant Vulkan implementation, so a pass there shows the
kernels, the synchronisation and both memory paths are right as Vulkan defines
them. It does not show that NVIDIA's, AMD's or Intel's compilers produce the
same numbers (they are allowed to differ within Vulkan's precision rules, which
is what the selftest tolerances measure), nor anything about speed. Until
`ditch selftest --device vulkan` has been run on a vendor's GPU, treat that
vendor as unverified; it is one command and takes seconds:

```sh
ditch selftest --device vulkan --json > selftest.json
ditch bench --kernels                       > bench-cpu.md
ditch bench --kernels --device vulkan       > bench-vulkan.md
```

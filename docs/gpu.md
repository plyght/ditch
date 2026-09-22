# Compute backends

ditch runs on the CPU by default and the CPU kernels in `src/tensor.zig` are the
reference implementation: everything else is judged against them. This document
describes the backend seam, the Metal backend, the reference test harness, and
exactly what has and has not been run on real hardware.

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
  Metal backend copies its raw bytes, in their on-disk dtype, into one reused
  shared-storage buffer and the kernel converts them as it reads them. There is
  no host-side conversion and no second copy.
* **Page-aligned host memory is addressed in place.** Memory-mapped weights
  usually are, and `newBufferWithBytesNoCopy:` then gives the GPU the same
  pages the CPU sees.
* **`--gpu-memory N`** turns on `compute.Residency`, a byte-budgeted LRU of
  device buffers keyed by the host address and length of the tile, capped by
  the device's own recommended working set. It is only active when host weight
  pointers are stable for the run, i.e. in memory-mapped mode: streamed and warp
  modes reuse the same buffers for different tensors, so caching by address
  would hand back the wrong weights. There, every tile is uploaded per call.

`--max-ram` is unaffected: device buffers are not ditch-owned host allocations,
and the budget still governs the host side exactly as before.

### Abliteration semantics

Unchanged. The refusal directions, the low-rank deltas and the exported weights
are computed from the same formulas; a GPU only changes the order in which f32
products are summed. `matmulT`'s low-rank delta is applied on the host after a
device matmul, with the same arithmetic as the fused CPU kernel. The manifest
(`ditch-reproduce.lua`) records the device that ran the study, and reproduction
does not force that device back on — a manifest from a Metal run reproduces on a
CPU, within the tolerance below.

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

## The reference harness

`src/selftest.zig` compares a backend against the CPU kernels on random inputs
across a shape sweep that includes the edge cases — 1x1, single rows, single
columns, sizes that are not a multiple of the vector width (16) or the device
tile — for all three weight dtypes, all activations and all norm variants.

```sh
ditch selftest                  # the CPU backend against itself: exactly zero error
ditch selftest --device metal   # the GPU kernels against the CPU reference
ditch selftest --device metal --json
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

What cannot be checked without a Mac, and is therefore not claimed here:
MSL syntax and semantics, buffer-index and attribute correctness, threadgroup
memory limits, the Objective-C shim compiling and linking against the real SDK,
and every number the GPU produces. If the macOS job has not run green on a
change, the Metal path is compile-checked only.

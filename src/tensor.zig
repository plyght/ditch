//! Low-level numeric kernels: dtype conversion, parallel matrix products,
//! normalisation, activations and rotary embeddings. Everything computes in
//! f32; weights stay in their on-disk dtype and are converted row by row.

const std = @import("std");
const builtin = @import("builtin");
const build_options = @import("build_options");
const Io = std.Io;
const quant = @import("quant.zig");

pub const DType = enum {
    f32,
    f16,
    bf16,
    /// 64-bit integers (index tables such as DeepSeek V4's `tid2eid`); carried
    /// through exports untouched and read as integers, never as weights.
    i64,
    // ggml block-quantised formats (see quant.zig); only meaningful for row-major matrices
    // whose row length is a multiple of the block size.
    q8_0,
    q4_0,
    q4_1,
    q5_0,
    q5_1,
    q4_k,
    q6_k,
    q8_k,

    /// Byte size of one element (floating-point types only).
    pub fn size(self: DType) usize {
        return switch (self) {
            .f32 => 4,
            .f16, .bf16 => 2,
            .i64 => 8,
            else => unreachable,
        };
    }

    pub fn isQuantized(self: DType) bool {
        return switch (self) {
            .f32, .f16, .bf16, .i64 => false,
            else => true,
        };
    }

    /// A floating-point storage type that kernels can read as weights.
    pub fn isFloat(self: DType) bool {
        return switch (self) {
            .f32, .f16, .bf16 => true,
            else => false,
        };
    }

    /// Elements per block (1 for floating-point types).
    pub fn blockSize(self: DType) usize {
        return quant.blockSize(self);
    }

    /// Bytes per block (the element size for floating-point types).
    pub fn blockBytes(self: DType) usize {
        return quant.blockBytes(self);
    }

    /// Bytes of one row of `cols` elements.
    pub fn rowBytes(self: DType, cols: usize) usize {
        const bs = self.blockSize();
        std.debug.assert(cols % bs == 0);
        return cols / bs * self.blockBytes();
    }

    /// Bytes of `numel` elements laid out as rows of `cols`.
    pub fn byteLen(self: DType, numel: usize, cols: usize) usize {
        if (cols == 0) return 0;
        return numel / cols * self.rowBytes(cols);
    }

    pub fn fromSafetensors(name: []const u8) ?DType {
        if (std.mem.eql(u8, name, "F32")) return .f32;
        if (std.mem.eql(u8, name, "F16")) return .f16;
        if (std.mem.eql(u8, name, "BF16")) return .bf16;
        if (std.mem.eql(u8, name, "I64")) return .i64;
        return null;
    }

    /// Upper-case name: the safetensors dtype for floating-point types, the ggml name otherwise.
    pub fn safetensorsName(self: DType) []const u8 {
        return switch (self) {
            .f32 => "F32",
            .f16 => "F16",
            .bf16 => "BF16",
            .i64 => "I64",
            .q8_0 => "Q8_0",
            .q4_0 => "Q4_0",
            .q4_1 => "Q4_1",
            .q5_0 => "Q5_0",
            .q5_1 => "Q5_1",
            .q4_k => "Q4_K",
            .q6_k => "Q6_K",
            .q8_k => "Q8_K",
        };
    }

    /// Parses a user-facing dtype name (case-insensitive): f32, f16, bf16, q8_0, ...
    pub fn parse(name: []const u8) ?DType {
        inline for (std.meta.fields(DType)) |f| {
            if (std.ascii.eqlIgnoreCase(name, f.name)) return @enumFromInt(f.value);
        }
        if (std.ascii.eqlIgnoreCase(name, "float32") or std.ascii.eqlIgnoreCase(name, "fp32")) return .f32;
        if (std.ascii.eqlIgnoreCase(name, "float16") or std.ascii.eqlIgnoreCase(name, "fp16")) return .f16;
        if (std.ascii.eqlIgnoreCase(name, "bfloat16")) return .bf16;
        return null;
    }
};

// ---------------------------------------------------------------------------
// Thread pool
// ---------------------------------------------------------------------------

/// A minimal fork-join helper. Every parallel kernel splits its range into
/// `threads` chunks; with `initPersistent` they run on a set of resident
/// worker threads that spin briefly and then sleep on a futex between jobs
/// (a job dispatch costs a few microseconds), otherwise as async tasks in an
/// `std.Io` group.
pub const Pool = struct {
    io: Io,
    threads: usize,
    workers: ?*Workers = null,
    /// Kernel scratch that is reused across calls (persistent pools only).
    scratch: ?*Scratch = null,

    pub fn init(io: Io, threads: ?usize) Pool {
        const n = threads orelse defaultThreads();
        return .{ .io = io, .threads = @max(1, n) };
    }

    /// A pool with `threads - 1` resident worker threads (the calling thread
    /// works too). Falls back to the `std.Io` path if threads cannot be spawned.
    pub fn initPersistent(gpa: std.mem.Allocator, io: Io, threads: ?usize) Pool {
        var pool = init(io, threads);
        // The calling thread works too, so it needs the same scheduling class.
        setWorkerQos();
        if (pool.threads > 1) pool.workers = Workers.spawn(gpa, io, pool.threads - 1) catch null;
        if (gpa.create(Scratch)) |sc| {
            sc.* = .{ .gpa = gpa };
            pool.scratch = sc;
        } else |_| {}
        return pool;
    }

    pub fn deinit(self: *Pool) void {
        if (self.workers) |w| w.shutdown();
        self.workers = null;
        if (self.scratch) |sc| {
            sc.gpa.free(sc.buf);
            sc.gpa.destroy(sc);
        }
        self.scratch = null;
    }

    /// `n` floats of kernel scratch: the pool's reusable buffer when it is
    /// free and the request is modest, otherwise a fresh allocation. Release
    /// with `freeScratch`. Kernels never nest, so one buffer suffices.
    pub fn allocScratch(self: *const Pool, gpa: std.mem.Allocator, n: usize) ![]f32 {
        if (self.scratch) |sc| {
            if (!sc.in_use and n <= Scratch.max_floats) {
                if (sc.buf.len < n) {
                    const grown = try sc.gpa.alloc(f32, n);
                    sc.gpa.free(sc.buf);
                    sc.buf = grown;
                }
                sc.in_use = true;
                return sc.buf[0..n];
            }
        }
        return gpa.alloc(f32, n);
    }

    pub fn freeScratch(self: *const Pool, gpa: std.mem.Allocator, s: []f32) void {
        if (self.scratch) |sc| {
            if (sc.in_use and s.ptr == sc.buf.ptr) {
                sc.in_use = false;
                return;
            }
        }
        gpa.free(s);
    }

    /// Runs `func(ctx, start, end)` over `[0, n)` split across the pool.
    pub fn parallelFor(self: *const Pool, n: usize, ctx: anytype, comptime func: fn (@TypeOf(ctx), usize, usize) void) void {
        if (n == 0) return;
        const chunks = @min(self.threads, n);
        if (chunks == 1) {
            func(ctx, 0, n);
            return;
        }
        const Ctx = @TypeOf(ctx);
        if (self.workers) |w| {
            const Tramp = struct {
                fn run(p: *const anyopaque, s: usize, e: usize) void {
                    const c: Ctx = @ptrCast(@alignCast(p));
                    func(c, s, e);
                }
            };
            w.run(.{ .func = Tramp.run, .ctx = @ptrCast(ctx), .n = n, .chunks = chunks });
            return;
        }
        const Task = struct {
            fn run(c: Ctx, s: usize, e: usize) Io.Cancelable!void {
                func(c, s, e);
            }
        };
        var group: Io.Group = .init;
        const per = (n + chunks - 1) / chunks;
        var start: usize = 0;
        while (start < n) : (start += per) {
            const end = @min(n, start + per);
            group.async(self.io, Task.run, .{ ctx, start, end });
        }
        group.await(self.io) catch {};
    }
};

const Scratch = struct {
    gpa: std.mem.Allocator,
    buf: []f32 = &.{},
    in_use: bool = false,

    /// Requests above this (64 MiB) are served by the caller's allocator.
    const max_floats = 16 << 20;
};

const Job = struct {
    func: *const fn (*const anyopaque, usize, usize) void,
    ctx: *const anyopaque,
    n: usize,
    chunks: usize,
};

/// Resident worker threads for `Pool`. A job is published by bumping
/// `generation`; every thread (workers and the caller) then claims chunks
/// through `next` until the job is exhausted, and the caller returns when
/// `remaining` reaches zero.
const Workers = struct {
    gpa: std.mem.Allocator,
    io: Io,
    threads: []std.Thread,
    generation: std.atomic.Value(u32) = .init(0),
    remaining: std.atomic.Value(u32) = .init(0),
    next: std.atomic.Value(usize) = .init(0),
    stop: std.atomic.Value(bool) = .init(false),
    job: Job = undefined,

    /// Iterations of the wake-up spin before a worker sleeps on the futex.
    const spin_iterations = 1000;

    fn spawn(gpa: std.mem.Allocator, io: Io, count: usize) !*Workers {
        const self = try gpa.create(Workers);
        errdefer gpa.destroy(self);
        self.* = .{ .gpa = gpa, .io = io, .threads = &.{} };
        const threads = try gpa.alloc(std.Thread, count);
        errdefer gpa.free(threads);
        var spawned: usize = 0;
        errdefer {
            self.stop.store(true, .release);
            _ = self.generation.fetchAdd(1, .release);
            self.io.futexWake(u32, &self.generation.raw, std.math.maxInt(u32));
            for (threads[0..spawned]) |t| t.join();
        }
        // The baseline generation is read here, not in the new thread: a
        // worker that is still starting up when the first job is published
        // would otherwise read the already-bumped generation, wait for the
        // next one and never decrement `remaining` for that job, leaving the
        // caller waiting for a worker that skipped it.
        const first_generation = self.generation.load(.acquire);
        for (threads) |*t| {
            t.* = try std.Thread.spawn(.{}, workerMain, .{ self, first_generation });
            spawned += 1;
        }
        self.threads = threads;
        return self;
    }

    fn shutdown(self: *Workers) void {
        self.stop.store(true, .release);
        _ = self.generation.fetchAdd(1, .release);
        self.io.futexWake(u32, &self.generation.raw, std.math.maxInt(u32));
        for (self.threads) |t| t.join();
        self.gpa.free(self.threads);
        self.gpa.destroy(self);
    }

    fn run(self: *Workers, job: Job) void {
        self.job = job;
        self.next.store(0, .monotonic);
        self.remaining.store(@intCast(self.threads.len), .monotonic);
        _ = self.generation.fetchAdd(1, .release);
        self.io.futexWake(u32, &self.generation.raw, std.math.maxInt(u32));
        self.work();
        // Wait for the workers: spin first, then sleep.
        var spins: usize = 0;
        while (self.remaining.load(.acquire) != 0) {
            if (spins < spin_iterations) {
                spins += 1;
                std.atomic.spinLoopHint();
            } else {
                const r = self.remaining.load(.acquire);
                if (r != 0) self.io.futexWaitUncancelable(u32, &self.remaining.raw, r);
            }
        }
    }

    /// Claims and runs chunks of the current job until none are left.
    fn work(self: *Workers) void {
        const job = self.job;
        const per = (job.n + job.chunks - 1) / job.chunks;
        while (true) {
            const c = self.next.fetchAdd(1, .monotonic);
            if (c >= job.chunks) break;
            const start = c * per;
            if (start >= job.n) break;
            job.func(job.ctx, start, @min(job.n, start + per));
        }
    }

    fn workerMain(self: *Workers, first_generation: u32) void {
        setWorkerQos();
        var seen = first_generation;
        while (true) {
            // Wait for a new generation.
            var spins: usize = 0;
            while (self.generation.load(.acquire) == seen) {
                if (spins < spin_iterations) {
                    spins += 1;
                    std.atomic.spinLoopHint();
                } else {
                    self.io.futexWaitUncancelable(u32, &self.generation.raw, seen);
                }
            }
            seen = self.generation.load(.acquire);
            if (self.stop.load(.acquire)) return;
            self.work();
            if (self.remaining.fetchSub(1, .release) == 1) self.io.futexWake(u32, &self.remaining.raw, 1);
        }
    }
};

// ---------------------------------------------------------------------------
// Platform
// ---------------------------------------------------------------------------

/// Whether `DITCH_NO_MMAP` is set: weights are then read instead of
/// memory-mapped (see `model.Model.loadWithOptions`). Kept here because it is
/// a property of the process, and because the tests that can only run on the
/// mapped path check it to skip themselves.
pub fn mmapDisabled() bool {
    const v = std.c.getenv("DITCH_NO_MMAP") orelse return false;
    return v[0] != 0 and v[0] != '0';
}

/// macOS: `sysctlbyname` returning a 32-bit count, or null.
fn sysctlU32(name: [*:0]const u8) ?u32 {
    if (builtin.os.tag != .macos) return null;
    var value: u32 = 0;
    var len: usize = @sizeOf(u32);
    if (std.c.sysctlbyname(name, &value, &len, null, 0) != 0) return null;
    if (len != @sizeOf(u32) or value == 0) return null;
    return value;
}

/// Logical performance cores, on a machine that distinguishes them.
/// `hw.perflevel0` is the fastest core cluster on Apple Silicon (the P-cores);
/// it is absent on Intel Macs and on a single-cluster machine.
pub fn performanceCores() ?usize {
    if (builtin.os.tag != .macos) return null;
    const n = sysctlU32("hw.perflevel0.logicalcpu") orelse return null;
    return @intCast(n);
}

/// Efficiency cores (`hw.perflevel1`), for reporting.
pub fn efficiencyCores() ?usize {
    if (builtin.os.tag != .macos) return null;
    const n = sysctlU32("hw.perflevel1.logicalcpu") orelse return null;
    return @intCast(n);
}

/// Threads a pool takes when none is configured. On Apple Silicon that is the
/// number of P-cores, not every logical CPU: the E-cores are several times
/// slower, and an equal share of a fork-join kernel handed to one of them
/// holds up every other thread at the join.
pub fn defaultThreads() usize {
    if (performanceCores()) |p| return @max(1, p);
    return std.Thread.getCpuCount() catch 1;
}

/// `QOS_CLASS_USER_INITIATED`, the class the scheduler keeps on P-cores.
const qos_class_user_initiated: c_uint = 0x21;

extern "c" fn pthread_set_qos_class_self_np(qos_class: c_uint, relative_priority: c_int) c_int;

/// Asks macOS to schedule this thread as user-initiated work. Without it a
/// thread inherits a lower class and the scheduler is free to park it on an
/// E-core, which makes a fork-join kernel wait for its slowest chunk.
pub fn setWorkerQos() void {
    if (builtin.os.tag != .macos) return;
    _ = pthread_set_qos_class_self_np(qos_class_user_initiated, 0);
}

// ---------------------------------------------------------------------------
// Conversion
// ---------------------------------------------------------------------------

pub inline fn bf16ToF32(v: u16) f32 {
    return @bitCast(@as(u32, v) << 16);
}

pub inline fn f32ToBf16(v: f32) u16 {
    const bits: u32 = @bitCast(v);
    if ((bits & 0x7fffffff) > 0x7f800000) return @intCast((bits >> 16) | 0x40); // NaN
    // Round to nearest even.
    const lsb = (bits >> 16) & 1;
    const rounded = bits + 0x7fff + lsb;
    return @intCast(rounded >> 16);
}

pub inline fn f16ToF32(v: u16) f32 {
    const h: f16 = @bitCast(v);
    return @floatCast(h);
}

pub inline fn f32ToF16(v: f32) u16 {
    const h: f16 = @floatCast(v);
    return @bitCast(h);
}

/// Lanes of the half-precision widening loops. Two vector registers' worth:
/// the widening is a pure element-wise map (a shift for bf16, `vcvtph2ps` /
/// `fcvtl` for f16), so a wider unroll only hides latency and never changes a
/// result. It is deliberately independent of `VL`.
const CVL: usize = 2 * nativeLanes();

/// Converts `count` elements of raw `dtype` data starting at `bytes` into `out`.
pub fn convertToF32(dtype: DType, bytes: []const u8, out: []f32) void {
    switch (dtype) {
        .f32 => {
            const src = std.mem.bytesAsSlice(f32, bytes[0 .. out.len * 4]);
            @memcpy(out, src);
        },
        .bf16 => {
            const src = std.mem.bytesAsSlice(u16, bytes[0 .. out.len * 2]);
            const V = @Vector(CVL, u32);
            var i: usize = 0;
            while (i + CVL <= out.len) : (i += CVL) {
                const v: @Vector(CVL, u16) = src[i..][0..CVL].*;
                const wide: V = @as(V, @intCast(v)) << @splat(16);
                out[i..][0..CVL].* = @bitCast(wide);
            }
            while (i < out.len) : (i += 1) out[i] = bf16ToF32(src[i]);
        },
        .f16 => {
            const src = std.mem.bytesAsSlice(u16, bytes[0 .. out.len * 2]);
            var i: usize = 0;
            while (i + CVL <= out.len) : (i += CVL) {
                const v: @Vector(CVL, u16) = src[i..][0..CVL].*;
                const h: @Vector(CVL, f16) = @bitCast(v);
                out[i..][0..CVL].* = @as(@Vector(CVL, f32), @floatCast(h));
            }
            while (i < out.len) : (i += 1) out[i] = f16ToF32(src[i]);
        },
        .i64 => {
            const src = std.mem.bytesAsSlice(i64, bytes[0 .. out.len * 8]);
            for (out, 0..) |*o, i| o.* = @floatFromInt(src[i]);
        },
        else => quant.dequantize(dtype, bytes, out),
    }
}

pub fn convertFromF32(dtype: DType, src: []const f32, out: []u8) void {
    switch (dtype) {
        .f32 => @memcpy(out[0 .. src.len * 4], std.mem.sliceAsBytes(src)),
        .bf16 => {
            const dst = std.mem.bytesAsSlice(u16, out[0 .. src.len * 2]);
            for (src, 0..) |v, i| dst[i] = f32ToBf16(v);
        },
        .f16 => {
            const dst = std.mem.bytesAsSlice(u16, out[0 .. src.len * 2]);
            for (src, 0..) |v, i| dst[i] = f32ToF16(v);
        },
        .i64 => {
            const dst = std.mem.bytesAsSlice(i64, out[0 .. src.len * 8]);
            for (src, 0..) |v, i| dst[i] = @intFromFloat(@round(v));
        },
        else => quant.quantize(dtype, src, out),
    }
}

// ---------------------------------------------------------------------------
// Weight matrix views
// ---------------------------------------------------------------------------

/// A row-major matrix `[rows][cols]` stored in its on-disk dtype. Quantised
/// dtypes are dequantised row by row in `row`, so kernels see f32 either way.
pub const Weight = struct {
    data: []const u8,
    dtype: DType,
    rows: usize,
    cols: usize,

    pub fn row(self: Weight, r: usize, out: []f32) void {
        const rb = self.dtype.rowBytes(self.cols);
        convertToF32(self.dtype, self.data[r * rb ..][0..rb], out[0..self.cols]);
    }

    /// Raw bytes of row `r` in the on-disk dtype.
    pub fn rowBytes(self: Weight, r: usize) []const u8 {
        const rb = self.dtype.rowBytes(self.cols);
        return self.data[r * rb ..][0..rb];
    }

    /// Reads every row into a freshly allocated f32 matrix.
    pub fn toF32(self: Weight, gpa: std.mem.Allocator) ![]f32 {
        const out = try gpa.alloc(f32, self.rows * self.cols);
        convertToF32(self.dtype, self.data, out);
        return out;
    }
};

// ---------------------------------------------------------------------------
// Dot products and matrix products
// ---------------------------------------------------------------------------

/// Every dot-product kernel in this file accumulates lane-wise with fused
/// multiply-adds over `VL`-wide blocks, reduces once and finishes the tail
/// with scalar fused multiply-adds, so an output element is bit-identical
/// whichever kernel or thread split computes it.
pub fn dot(a: []const f32, b: []const f32) f32 {
    var acc: VF = @splat(0);
    var i: usize = 0;
    const n = a.len;
    while (i + VL <= n) : (i += VL) acc = fma(VF, a[i..][0..VL].*, b[i..][0..VL].*, acc);
    var s = @reduce(.Add, acc);
    while (i < n) : (i += 1) s = fma(f32, a[i], b[i], s);
    return s;
}

/// `y += alpha * x`
pub fn axpy(y: []f32, alpha: f32, x: []const f32) void {
    const va: VF = @splat(alpha);
    var i: usize = 0;
    while (i + VL <= y.len) : (i += VL) {
        const vy: VF = y[i..][0..VL].*;
        const vx: VF = x[i..][0..VL].*;
        y[i..][0..VL].* = vy + va * vx;
    }
    while (i < y.len) : (i += 1) y[i] += alpha * x[i];
}

pub fn scale(x: []f32, alpha: f32) void {
    for (x) |*v| v.* *= alpha;
}

pub fn norm2(x: []const f32) f32 {
    return @sqrt(dot(x, x));
}

/// Normalises `x` to unit L2 norm in place (no-op for zero vectors).
pub fn normalize(x: []f32) void {
    const n = norm2(x);
    if (n > 1e-12) scale(x, 1.0 / n);
}

/// A LoRA-style low-rank delta: `W' = W + B @ A` with `A: [rank][cols]`,
/// `B: [rows][rank]`. Used to apply abliteration without touching the base weights.
pub const Delta = struct {
    rank: usize,
    a: []f32, // rank * cols
    b: []f32, // rows * rank
};

// ---------------------------------------------------------------------------
// Kernel shape
// ---------------------------------------------------------------------------

/// f32 lanes in one vector register of the target CPU.
fn nativeLanes() comptime_int {
    return switch (builtin.cpu.arch) {
        .x86_64, .x86 => if (std.Target.x86.featureSetHas(builtin.cpu.features, .avx512f))
            16
        else if (std.Target.x86.featureSetHas(builtin.cpu.features, .avx2))
            8
        else
            4,
        // NEON: 32 registers of 128 bits. SVE is not used; its length is not
        // known at compile time and Apple Silicon implements NEON only.
        .aarch64, .aarch64_be => 4,
        else => 4,
    };
}

/// Architectural vector registers available to the inner loop.
fn vectorRegisters() comptime_int {
    return switch (builtin.cpu.arch) {
        .x86_64, .x86 => if (std.Target.x86.featureSetHas(builtin.cpu.features, .avx512f)) 32 else 16,
        .aarch64, .aarch64_be => 32,
        else => 8,
    };
}

/// Vector width of every kernel in this file, in f32 lanes: one register on
/// the target (see `shape`). `-Dvector-width=N` overrides it for benchmarking
/// a machine by hand; it changes the order in which every dot product
/// accumulates, so a binary is internally consistent but two binaries built
/// with different widths differ in the last bits.
pub const VL: usize = if (build_options.vector_width != 0) build_options.vector_width else shape.vl;
pub const VF = @Vector(VL, f32);

/// Register tile of the matmul kernel: `tile_inputs` input rows by
/// `tile_rows` weight rows, so `tile_inputs * tile_rows` accumulators plus
/// `tile_rows` weight vectors and one input vector have to fit in the register
/// file. The tile shape changes nothing numerically (each output is still one
/// dot product blocked by `VL`), only how many loads feed each multiply-add.
///
/// The width and the shape are measured, not derived: one thread, a
/// 64 x 2048 x 2048 bf16 product, built with `-Dcpu=...` and the overrides
/// below. Median of three runs on one 4-vCPU x86-64 machine, prefill /
/// one-input decode GFLOP/s:
///
///   AVX-512  16 lanes, 4x4: 69.9 / 23.0   16 lanes, 4x6: 69.2 / 20.8
///            16 lanes, 4x3: 67.6 / 20.3    8 lanes, 4x4: 73.6 / 21.6
///   AVX2     16 lanes, 4x4: 37.0 / 19.9    8 lanes, 4x3: 66.4 / 23.0
///             8 lanes, 4x4: 45.4 / 22.5    8 lanes, 4x2: 54.9 / 15.2
///   SSE2     16 lanes, 4x4: 18.4 / 12.3    4 lanes, 4x3: 30.5 / 17.3
///             4 lanes, 4x4: 26.6 / 17.3    4 lanes, 3x3: 25.5 / 13.7
///
/// A 16-lane accumulator costs two AVX2 or four NEON registers, so the 4x4
/// tile that fits AVX-512 spills on every inner iteration everywhere else;
/// that is what the first column of each block costs. AVX-512 is no slower
/// than 256-bit work on this part, so the widest register wins there.
const shape: struct { vl: usize, i: usize, k: usize } = switch (builtin.cpu.arch) {
    .x86_64, .x86 => if (std.Target.x86.featureSetHas(builtin.cpu.features, .avx512f))
        .{ .vl = 16, .i = 4, .k = 4 }
    else if (std.Target.x86.featureSetHas(builtin.cpu.features, .avx2))
        .{ .vl = 8, .i = 4, .k = 3 }
    else
        .{ .vl = 4, .i = 4, .k = 3 },
    // NEON: 32 registers of 128 bits, so a four-lane accumulator leaves room
    // for the whole 4x4 tile (16 accumulators, 4 weight and 1 input vector).
    // Chosen from the generated assembly rather than from a timing, since the
    // shape had to be picked without an Apple Silicon machine: in the densest
    // 64-instruction window of `matmulWorker` compiled for aarch64-macos,
    // 16 lanes x 4x4 spends 27 instructions on stack traffic and 8 lanes x 4x4
    // spends 12, while every 4-lane shape spends none.
    .aarch64, .aarch64_be => .{ .vl = 4, .i = 4, .k = 4 },
    else => .{ .vl = 4, .i = 4, .k = 3 },
};

/// Whether the target has a hardware fused multiply-add. Without one
/// `@mulAdd` is a call to libc `fmaf` per lane, which costs about fifty times
/// a multiply and an add, so the kernels multiply and add separately there.
/// The x86-64 baseline (what the release binaries are built for, so that they
/// run everywhere) has no FMA; `-Dcpu=x86_64_v3` or a native build does.
pub const has_fma = switch (builtin.cpu.arch) {
    .x86_64, .x86 => std.Target.x86.featureSetHas(builtin.cpu.features, .fma),
    else => true,
};

/// `a * b + c`, fused where the target can fuse it. Fusing changes the
/// rounding of every accumulation, so a build either fuses everywhere or
/// nowhere; within one binary each output is still computed identically
/// whichever kernel or thread split produces it.
pub inline fn fma(comptime T: type, a: T, b: T, c: T) T {
    return if (has_fma) @mulAdd(T, a, b, c) else a * b + c;
}

pub const tile_inputs: usize = if (build_options.tile_inputs != 0) build_options.tile_inputs else shape.i;
pub const tile_rows: usize = if (build_options.tile_rows != 0) build_options.tile_rows else shape.k;

/// Weight rows converted per tile: the f32 tile (`tile * cols * 4` bytes)
/// stays in L2 while every input row streams past it once. Prefill-sized
/// calls take a large tile so `x` is streamed fewer times; small batches a
/// small one so the converted tile and the inputs share L1/L2.
fn tileRowsFor(n: usize) usize {
    const big = roundTile(64);
    const small = roundTile(16);
    return if (n > 64) big else small;
}

/// Rounds a tile height up to a multiple of `tile_rows` so the register tile
/// is never cut short at the end of a converted tile.
fn roundTile(n: usize) usize {
    return (n + tile_rows - 1) / tile_rows * tile_rows;
}

// ---------------------------------------------------------------------------
// Accelerate (macOS)
// ---------------------------------------------------------------------------

/// Whether this build contains the Accelerate path (`-Daccelerate`, macOS
/// only). The framework is resolved with `dlopen` rather than linked: the
/// release binaries are cross-compiled from Linux, where no macOS SDK and so
/// no `Accelerate.framework` exists to link against, and a `dlopen` that
/// fails simply leaves the Zig kernels in charge.
pub const have_accelerate = build_options.accelerate and builtin.os.tag == .macos;

/// Runtime switch for the Accelerate path (`--accelerate=false`, or the
/// `accelerate` setting). Off makes every kernel take the Zig path, which is
/// what a reproducibility check across machines wants.
pub var accelerate_enabled: bool = have_accelerate;

/// Set once `cblas_sgemm` has been resolved (see `initAccelerate`).
var accelerate_ready: bool = false;

/// Inputs per call from which `cblas_sgemm` takes over from the register-tiled
/// Zig kernel. Decode (one to four inputs) is a memory-bound matrix-vector
/// product and stays on the fused half-precision path, which reads each weight
/// byte once and never materialises an f32 tile; prefill-shaped calls hand
/// Apple's matrix units a real GEMM. See README, "Performance".
pub const accelerate_min_inputs: usize = 8;

const cblas_row_major: c_int = 101;
const cblas_no_trans: c_int = 111;
const cblas_trans: c_int = 112;

const Sgemm = *const fn (
    c_int,
    c_int,
    c_int,
    c_int,
    c_int,
    c_int,
    f32,
    [*]const f32,
    c_int,
    [*]const f32,
    c_int,
    f32,
    [*]f32,
    c_int,
) callconv(.c) void;

var sgemm_fn: ?Sgemm = null;

/// Resolves `cblas_sgemm` from Accelerate. Called once at startup before any
/// kernel runs (the pointer is then only read), and returns whether the
/// Accelerate path is live.
pub fn initAccelerate() bool {
    if (!have_accelerate) return false;
    const handle = std.c.dlopen("/System/Library/Frameworks/Accelerate.framework/Accelerate", .{ .LAZY = true }) orelse return false;
    const sym = std.c.dlsym(handle, "cblas_sgemm") orelse return false;
    sgemm_fn = @ptrCast(@alignCast(sym));
    accelerate_ready = true;
    return true;
}

/// Whether Accelerate is compiled in, loaded and switched on.
pub fn accelerateActive() bool {
    return have_accelerate and accelerate_ready and accelerate_enabled;
}

/// Whether a matmul of `n` inputs by `rows` rows of `cols` should go to BLAS:
/// batched (prefill-shaped) calls on a matrix big enough to pay for the call.
fn useBlas(n: usize, rows: usize, cols: usize) bool {
    if (!have_accelerate) return false;
    return accelerateActive() and n >= accelerate_min_inputs and rows * cols >= 1 << 14;
}

/// `out[n][nr] = x[n][cols] @ tile[nr][cols]^T`, written into a column block
/// of `out` whose row stride is `ld_out`.
fn blasTile(out: [*]f32, ld_out: usize, x: [*]const f32, tile: [*]const f32, n: usize, nr: usize, cols: usize) void {
    const sgemm = sgemm_fn orelse return;
    sgemm(
        cblas_row_major,
        cblas_no_trans,
        cblas_trans,
        @intCast(n),
        @intCast(nr),
        @intCast(cols),
        1.0,
        x,
        @intCast(cols),
        tile,
        @intCast(cols),
        0.0,
        out,
        @intCast(ld_out),
    );
}

const MatmulCtx = struct {
    out: []f32,
    x: []const f32,
    n: usize,
    w: Weight,
    delta: ?*const Delta,
    /// `[n][rank]`: `dot(a_k, x_i)` for the delta, computed once per call.
    xa: []const f32,
    scratch: []f32, // one tile of `tile` converted weight rows per task
    tile: usize,
    per: usize,
};

fn matmulWorker(ctx: *const MatmulCtx, start: usize, end: usize) void {
    const slot = start / ctx.per;
    const cols = ctx.w.cols;
    const n = ctx.n;
    const rows = ctx.w.rows;
    const tile = ctx.tile;
    const buf = ctx.scratch[slot * tile * cols ..][0 .. tile * cols];
    const blas = useBlas(n, rows, cols);
    var r = start;
    while (r < end) {
        const nr = @min(tile, end - r);
        for (0..nr) |k| ctx.w.row(r + k, buf[k * cols ..][0..cols]);
        if (blas) {
            // Only reached on macOS builds with Accelerate linked in.
            if (have_accelerate) blasTile(ctx.out.ptr + r, rows, ctx.x.ptr, buf.ptr, n, nr, cols);
        } else {
            var i: usize = 0;
            while (i < n) {
                const ni = @min(tile_inputs, n - i);
                var k: usize = 0;
                while (k < nr) {
                    const nk = @min(tile_rows, nr - k);
                    const wr = buf[k * cols ..];
                    const xr = ctx.x[i * cols ..];
                    if (nk == tile_rows and ni == tile_inputs) {
                        const d = dotTile(tile_rows, tile_inputs, wr.ptr, xr.ptr, cols);
                        inline for (0..tile_inputs) |ii| {
                            inline for (0..tile_rows) |kk| ctx.out[(i + ii) * rows + r + k + kk] = d[ii][kk];
                        }
                    } else if (nk == tile_rows) {
                        for (0..ni) |ii| {
                            const d = dotTile(tile_rows, 1, wr.ptr, xr.ptr + ii * cols, cols);
                            inline for (0..tile_rows) |kk| ctx.out[(i + ii) * rows + r + k + kk] = d[0][kk];
                        }
                    } else {
                        for (0..ni) |ii| {
                            for (0..nk) |kk| ctx.out[(i + ii) * rows + r + k + kk] = dot(wr[kk * cols ..][0..cols], xr[ii * cols ..][0..cols]);
                        }
                    }
                    k += nk;
                }
                i += ni;
            }
        }
        if (ctx.delta) |dl| {
            for (0..n) |ii| {
                const xa = ctx.xa[ii * dl.rank ..][0..dl.rank];
                for (0..nr) |kk| {
                    var v: f32 = 0;
                    for (0..dl.rank) |j| v += dl.b[(r + kk) * dl.rank + j] * xa[j];
                    ctx.out[ii * rows + r + kk] += v;
                }
            }
        }
        r += nr;
    }
}

/// `NK` weight rows (`w`, stride `cols`) against `NI` input rows (`x`, stride
/// `cols`): `out[input][row]`. `NK + NI` vector loads feed `NK * NI` fused
/// multiply-adds, so the kernel is bound by the FMA units rather than by
/// loads, as long as the accumulators stay in registers (see `shape`).
inline fn dotTile(comptime NK: usize, comptime NI: usize, w: [*]const f32, x: [*]const f32, cols: usize) [NI][NK]f32 {
    var acc: [NI][NK]VF = undefined;
    inline for (0..NI) |ii| {
        inline for (0..NK) |kk| acc[ii][kk] = @splat(0);
    }
    var i: usize = 0;
    while (i + VL <= cols) : (i += VL) {
        var wv: [NK]VF = undefined;
        inline for (0..NK) |kk| wv[kk] = w[kk * cols + i ..][0..VL].*;
        inline for (0..NI) |ii| {
            const xv: VF = x[ii * cols + i ..][0..VL].*;
            inline for (0..NK) |kk| acc[ii][kk] = fma(VF, wv[kk], xv, acc[ii][kk]);
        }
    }
    var r: [NI][NK]f32 = undefined;
    inline for (0..NI) |ii| {
        inline for (0..NK) |kk| r[ii][kk] = @reduce(.Add, acc[ii][kk]);
    }
    while (i < cols) : (i += 1) {
        inline for (0..NI) |ii| {
            inline for (0..NK) |kk| r[ii][kk] = fma(f32, w[kk * cols + i], x[ii * cols + i], r[ii][kk]);
        }
    }
    return r;
}

/// Weight rows the fused half-precision path takes at once for `NI` inputs:
/// `NI * k` accumulators plus `k` weight vectors and one input vector, within
/// the register file. Decode (`NI = 1`) can afford many more rows than the
/// general tile, which is what keeps the loads flowing on a memory-bound call.
fn fusedRowsFor(comptime NI: usize) usize {
    const regs_per_acc = (VL + nativeLanes() - 1) / nativeLanes();
    const budget = (vectorRegisters() - 2) / regs_per_acc;
    var k: usize = 8;
    while (k > 1 and NI * k + k + 1 > budget) k -= 1;
    return k;
}

/// Inputs per call up to which the half-precision decode path is used: the
/// weight rows are converted in registers and never written to scratch,
/// so the call streams each weight byte from memory exactly once.
const fused_max_inputs = 4;

fn usesFusedPath(w: Weight, n: usize) bool {
    return n <= fused_max_inputs and (w.dtype == .bf16 or w.dtype == .f16);
}

/// Loads sixteen half-precision weights as f32.
inline fn loadHalf(comptime dtype: DType, p: [*]align(1) const u16) VF {
    const v: @Vector(VL, u16) = p[0..VL].*;
    return switch (dtype) {
        .bf16 => @bitCast(@as(@Vector(VL, u32), @intCast(v)) << @splat(16)),
        .f16 => @floatCast(@as(@Vector(VL, f16), @bitCast(v))),
        else => unreachable,
    };
}

inline fn loadHalfScalar(comptime dtype: DType, v: u16) f32 {
    return switch (dtype) {
        .bf16 => bf16ToF32(v),
        .f16 => f16ToF32(v),
        else => unreachable,
    };
}

/// `NK` raw half-precision weight rows (`w`, stride `cols` elements) against
/// `NI` input rows: `out[input][row]`. Four independent accumulator chains
/// per input keep the loads flowing on the memory-bound decode path.
inline fn dotHalf(comptime dtype: DType, comptime NK: usize, comptime NI: usize, w: [*]align(1) const u16, x: [*]const f32, cols: usize) [NI][NK]f32 {
    var acc: [NI][NK]VF = undefined;
    inline for (0..NI) |ii| {
        inline for (0..NK) |kk| acc[ii][kk] = @splat(0);
    }
    var i: usize = 0;
    while (i + VL <= cols) : (i += VL) {
        var wv: [NK]VF = undefined;
        inline for (0..NK) |kk| wv[kk] = loadHalf(dtype, w + kk * cols + i);
        inline for (0..NI) |ii| {
            const xv: VF = x[ii * cols + i ..][0..VL].*;
            inline for (0..NK) |kk| acc[ii][kk] = fma(VF, wv[kk], xv, acc[ii][kk]);
        }
    }
    var r: [NI][NK]f32 = undefined;
    inline for (0..NI) |ii| {
        inline for (0..NK) |kk| r[ii][kk] = @reduce(.Add, acc[ii][kk]);
    }
    while (i < cols) : (i += 1) {
        inline for (0..NI) |ii| {
            inline for (0..NK) |kk| r[ii][kk] = fma(f32, loadHalfScalar(dtype, w[kk * cols + i]), x[ii * cols + i], r[ii][kk]);
        }
    }
    return r;
}

fn fusedRows(comptime dtype: DType, comptime NI: usize, ctx: *const MatmulCtx, start: usize, end: usize) void {
    const cols = ctx.w.cols;
    const rows = ctx.w.rows;
    const data: [*]align(1) const u16 = @ptrCast(ctx.w.data.ptr);
    const nk = comptime fusedRowsFor(NI);
    var r = start;
    while (r + nk <= end) : (r += nk) {
        const d = dotHalf(dtype, nk, NI, data + r * cols, ctx.x.ptr, cols);
        inline for (0..NI) |ii| {
            inline for (0..nk) |kk| ctx.out[ii * rows + r + kk] = d[ii][kk];
        }
    }
    while (r < end) : (r += 1) {
        const d = dotHalf(dtype, 1, NI, data + r * cols, ctx.x.ptr, cols);
        inline for (0..NI) |ii| ctx.out[ii * rows + r] = d[ii][0];
    }
}

fn fusedWorker(ctx: *const MatmulCtx, start: usize, end: usize) void {
    switch (ctx.w.dtype) {
        inline .bf16, .f16 => |dt| switch (ctx.n) {
            inline 1, 2, 3, 4 => |ni| fusedRows(dt, ni, ctx, start, end),
            else => unreachable,
        },
        else => unreachable,
    }
    if (ctx.delta) |dl| {
        const rows = ctx.w.rows;
        for (0..ctx.n) |ii| {
            const xa = ctx.xa[ii * dl.rank ..][0..dl.rank];
            for (start..end) |rr| {
                var v: f32 = 0;
                for (0..dl.rank) |j| v += dl.b[rr * dl.rank + j] * xa[j];
                ctx.out[ii * rows + rr] += v;
            }
        }
    }
}

/// `out[n][rows] = x[n][cols] @ W^T (+ x @ (B A)^T)`.
pub fn matmulT(pool: *const Pool, gpa: std.mem.Allocator, out: []f32, x: []const f32, n: usize, w: Weight, delta: ?*const Delta) !void {
    std.debug.assert(x.len >= n * w.cols);
    std.debug.assert(out.len >= n * w.rows);
    if (w.rows == 0 or n == 0) return;
    const fused = usesFusedPath(w, n);
    const chunks = @max(1, @min(pool.threads, w.rows));
    const per = (w.rows + chunks - 1) / chunks;
    const slots = (w.rows + per - 1) / per;
    const tile = tileRowsFor(n);
    const scratch = try pool.allocScratch(gpa, if (fused) 0 else slots * tile * w.cols);
    defer pool.freeScratch(gpa, scratch);
    var xa: []f32 = &.{};
    defer if (xa.len > 0) gpa.free(xa);
    if (delta) |dl| {
        xa = try gpa.alloc(f32, n * dl.rank);
        for (0..n) |i| {
            const xi = x[i * w.cols ..][0..w.cols];
            for (0..dl.rank) |j| xa[i * dl.rank + j] = dot(dl.a[j * w.cols ..][0..w.cols], xi);
        }
    }
    const ctx = MatmulCtx{ .out = out, .x = x, .n = n, .w = w, .delta = delta, .xa = xa, .scratch = scratch, .tile = tile, .per = per };
    if (fused) pool.parallelFor(w.rows, &ctx, fusedWorker) else pool.parallelFor(w.rows, &ctx, matmulWorker);
}

const MatvecTCtx = struct {
    out: []f32, // slots * q * cols accumulators
    y: []const f32, // q * rows
    q: usize,
    w: Weight,
    scratch: []f32,
    per: usize,
};

fn matvecTWorker(ctx: *const MatvecTCtx, start: usize, end: usize) void {
    const slot = start / ctx.per;
    const cols = ctx.w.cols;
    const rows = ctx.w.rows;
    const buf = ctx.scratch[slot * cols ..][0..cols];
    const acc = ctx.out[slot * ctx.q * cols ..][0 .. ctx.q * cols];
    @memset(acc, 0);
    var r = start;
    while (r < end) : (r += 1) {
        var any = false;
        var j: usize = 0;
        while (j < ctx.q) : (j += 1) any = any or ctx.y[j * rows + r] != 0;
        if (!any) continue;
        ctx.w.row(r, buf);
        j = 0;
        while (j < ctx.q) : (j += 1) {
            const yr = ctx.y[j * rows + r];
            if (yr != 0) axpy(acc[j * cols ..][0..cols], yr, buf);
        }
    }
}

/// `out[q][cols] = y[q][rows] @ W`, i.e. each output row is `W^T y_j`.
pub fn matvecTMulti(pool: *const Pool, gpa: std.mem.Allocator, out: []f32, w: Weight, y: []const f32, q: usize) !void {
    std.debug.assert(y.len >= q * w.rows);
    std.debug.assert(out.len >= q * w.cols);
    const chunks = @max(1, @min(pool.threads, w.rows));
    const per = (w.rows + chunks - 1) / chunks;
    const slots = (w.rows + per - 1) / per;
    const scratch = try gpa.alloc(f32, slots * w.cols);
    defer gpa.free(scratch);
    const accs = try gpa.alloc(f32, slots * q * w.cols);
    defer gpa.free(accs);
    const ctx = MatvecTCtx{ .out = accs, .y = y, .q = q, .w = w, .scratch = scratch, .per = per };
    pool.parallelFor(w.rows, &ctx, matvecTWorker);
    @memset(out[0 .. q * w.cols], 0);
    var s: usize = 0;
    while (s < slots) : (s += 1) axpy(out[0 .. q * w.cols], 1.0, accs[s * q * w.cols ..][0 .. q * w.cols]);
}

/// `out[cols] = W^T y` where `y` has `rows` elements.
pub fn matvecT(pool: *const Pool, gpa: std.mem.Allocator, out: []f32, w: Weight, y: []const f32) !void {
    return matvecTMulti(pool, gpa, out, w, y, 1);
}

const RowNormCtx = struct { out: []f32, w: Weight, scratch: []f32, per: usize };

fn rowNormWorker(ctx: *const RowNormCtx, start: usize, end: usize) void {
    const slot = start / ctx.per;
    const buf = ctx.scratch[slot * ctx.w.cols ..][0..ctx.w.cols];
    var r = start;
    while (r < end) : (r += 1) {
        ctx.w.row(r, buf);
        ctx.out[r] = norm2(buf);
    }
}

/// `out[rows] = ||W_i||_2` for every row.
pub fn rowNorms(pool: *const Pool, gpa: std.mem.Allocator, out: []f32, w: Weight) !void {
    const chunks = @max(1, @min(pool.threads, w.rows));
    const per = (w.rows + chunks - 1) / chunks;
    const slots = (w.rows + per - 1) / per;
    const scratch = try gpa.alloc(f32, slots * w.cols);
    defer gpa.free(scratch);
    const ctx = RowNormCtx{ .out = out, .w = w, .scratch = scratch, .per = per };
    pool.parallelFor(w.rows, &ctx, rowNormWorker);
}

/// `scores[p] = scale * dot(q, k[p * stride ..][0..hd])` for every key,
/// four keys at a time; `hd = q.len` must be a multiple of `VL`.
pub fn attentionScores(scores: []f32, q: []const f32, k: [*]const f32, stride: usize, scale_: f32) void {
    const hd = q.len;
    var p: usize = 0;
    while (p + 4 <= scores.len) : (p += 4) {
        var a: [4]VF = .{ @splat(0), @splat(0), @splat(0), @splat(0) };
        var d: usize = 0;
        while (d < hd) : (d += VL) {
            const qv: VF = q[d..][0..VL].*;
            inline for (0..4) |kk| a[kk] = fma(VF, k[(p + kk) * stride + d ..][0..VL].*, qv, a[kk]);
        }
        inline for (0..4) |kk| scores[p + kk] = @reduce(.Add, a[kk]) * scale_;
    }
    while (p < scores.len) : (p += 1) {
        var a: VF = @splat(0);
        var d: usize = 0;
        while (d < hd) : (d += VL) a = fma(VF, k[p * stride + d ..][0..VL].*, @as(VF, q[d..][0..VL].*), a);
        scores[p] = @reduce(.Add, a) * scale_;
    }
}

/// `out = sum_p scores[p] * v[p * stride ..][0..vd]` with the accumulators
/// held in registers; `vd = out.len` must be a multiple of `VL`.
pub fn attentionValues(out: []f32, scores: []const f32, v: [*]const f32, stride: usize) void {
    const vd = out.len;
    var j: usize = 0;
    while (j < vd) : (j += VL) {
        var a: [4]VF = .{ @splat(0), @splat(0), @splat(0), @splat(0) };
        var p: usize = 0;
        while (p + 4 <= scores.len) : (p += 4) {
            inline for (0..4) |kk| a[kk] = fma(VF, v[(p + kk) * stride + j ..][0..VL].*, @as(VF, @splat(scores[p + kk])), a[kk]);
        }
        while (p < scores.len) : (p += 1) a[0] = fma(VF, v[p * stride + j ..][0..VL].*, @as(VF, @splat(scores[p])), a[0]);
        out[j..][0..VL].* = (a[0] + a[1]) + (a[2] + a[3]);
    }
}

// ---------------------------------------------------------------------------
// Element-wise kernels
// ---------------------------------------------------------------------------

/// RMS normalisation. `gemma_style` uses `(1 + w)` scaling.
pub fn rmsnorm(out: []f32, x: []const f32, weight: []const f32, eps: f32, gemma_style: bool) void {
    var ss: f32 = 0;
    for (x) |v| ss += v * v;
    const inv = 1.0 / @sqrt(ss / @as(f32, @floatFromInt(x.len)) + eps);
    if (weight.len == 0) {
        // Non-parametric (nanochat, the weightless q/k norms).
        for (out, 0..) |*o, i| o.* = x[i] * inv;
    } else if (gemma_style) {
        for (out, 0..) |*o, i| o.* = x[i] * inv * (1.0 + weight[i]);
    } else {
        for (out, 0..) |*o, i| o.* = x[i] * inv * weight[i];
    }
}

pub fn softmaxInPlace(x: []f32) void {
    var mv: VF = @splat(-std.math.inf(f32));
    var i: usize = 0;
    while (i + VL <= x.len) : (i += VL) mv = @max(mv, @as(VF, x[i..][0..VL].*));
    var m = @reduce(.Max, mv);
    while (i < x.len) : (i += 1) m = @max(m, x[i]);
    var sv: VF = @splat(0);
    i = 0;
    while (i + VL <= x.len) : (i += VL) {
        const e = expVec(@as(VF, x[i..][0..VL].*) - @as(VF, @splat(m)));
        x[i..][0..VL].* = e;
        sv += e;
    }
    var s = @reduce(.Add, sv);
    while (i < x.len) : (i += 1) {
        x[i] = @exp(x[i] - m);
        s += x[i];
    }
    scale(x, 1.0 / s);
}

/// Computes log-softmax of `x` into `out`.
pub fn logSoftmax(out: []f32, x: []const f32) void {
    var m: f32 = -std.math.inf(f32);
    for (x) |v| m = @max(m, v);
    var s: f64 = 0;
    for (x) |v| s += @exp(@as(f64, v - m));
    const lse: f32 = m + @as(f32, @floatCast(@log(s)));
    for (out, 0..) |*o, i| o.* = x[i] - lse;
}

pub inline fn silu(x: f32) f32 {
    return x / (1.0 + @exp(-x));
}

pub inline fn geluTanh(x: f32) f32 {
    const c: f32 = 0.7978845608028654; // sqrt(2/pi)
    return 0.5 * x * (1.0 + std.math.tanh(c * (x + 0.044715 * x * x * x)));
}

pub inline fn geluErf(x: f32) f32 {
    return 0.5 * x * (1.0 + erf(x / std.math.sqrt2));
}

fn erf(x: f32) f32 {
    // Abramowitz-Stegun 7.1.26 with refinement; good to ~1e-7.
    const t = 1.0 / (1.0 + 0.3275911 * @abs(x));
    const y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * @exp(-x * x);
    return if (x >= 0) y else -y;
}

pub const Activation = enum {
    silu,
    gelu_tanh,
    gelu,
    relu,
    /// `relu(x)²` (Nemotron).
    relu2,
    /// `x · sigmoid(1.702 x)`.
    quick_gelu,

    pub fn apply(self: Activation, x: f32) f32 {
        return switch (self) {
            .silu => silu(x),
            .gelu_tanh => geluTanh(x),
            .gelu => geluErf(x),
            .relu => @max(x, 0),
            .relu2 => blk: {
                const r = @max(x, 0);
                break :blk r * r;
            },
            .quick_gelu => x / (1.0 + @exp(-1.702 * x)),
        };
    }

    /// Vector form of `apply` (`.gelu` has no vector form and uses `apply`).
    pub inline fn applyVec(self: Activation, x: VF) VF {
        const one: VF = @splat(1.0);
        return switch (self) {
            .silu => x / (one + expVec(-x)),
            .gelu_tanh => blk: {
                const u = @as(VF, @splat(0.7978845608028654)) * (x + @as(VF, @splat(0.044715)) * x * x * x);
                const t = one - @as(VF, @splat(2.0)) / (expVec(u + u) + one);
                break :blk @as(VF, @splat(0.5)) * x * (one + t);
            },
            .gelu => unreachable,
            .relu => @max(x, @as(VF, @splat(0))),
            .relu2 => blk: {
                const r = @max(x, @as(VF, @splat(0)));
                break :blk r * r;
            },
            .quick_gelu => x / (one + expVec(@as(VF, @splat(-1.702)) * x)),
        };
    }
};

/// Vectorised `exp` (Cephes `expf`: range reduction by `ln 2`, a degree-6
/// polynomial and an exponent-field scale), about 2 ulp over the finite
/// range; the input is clamped so the result never becomes NaN.
pub inline fn expVec(x_in: VF) VF {
    const x = @min(@max(x_in, @as(VF, @splat(-87.3))), @as(VF, @splat(88.0)));
    const fx = @round(x * @as(VF, @splat(1.44269504088896341)));
    const r = fma(VF, fx, @splat(2.12194440e-4), fma(VF, fx, @splat(-0.693359375), x));
    var y: VF = @splat(1.9875691500e-4);
    y = fma(VF, y, r, @splat(1.3981999507e-3));
    y = fma(VF, y, r, @splat(8.3334519073e-3));
    y = fma(VF, y, r, @splat(4.1665795894e-2));
    y = fma(VF, y, r, @splat(1.6666665459e-1));
    y = fma(VF, y, r, @splat(5.0000001201e-1));
    y = fma(VF, y, r * r, r + @as(VF, @splat(1.0)));
    const e: @Vector(VL, i32) = @intFromFloat(fx);
    const bits = (e + @as(@Vector(VL, i32), @splat(127))) << @splat(23);
    return y * @as(VF, @bitCast(bits));
}

const ActCtx = struct {
    act: Activation,
    out: [*]f32,
    gate: [*]const f32,
    up: ?[*]const f32,
    len: usize,
    in_stride: usize,
    out_stride: usize,
};

/// Work is split in `VL`-element blocks aligned to the start of each row,
/// so an element is computed by the same (vector or scalar) code whatever
/// the thread split.
fn activationWorker(ctx: *const ActCtx, start: usize, end: usize) void {
    const blocks_per_row = (ctx.len + VL - 1) / VL;
    var i = start;
    while (i < end) : (i += 1) {
        const row = i / blocks_per_row;
        const j = (i % blocks_per_row) * VL;
        const g = ctx.gate[row * ctx.in_stride + j ..];
        const o = ctx.out[row * ctx.out_stride + j ..];
        const u: ?[*]const f32 = if (ctx.up) |u| u + row * ctx.in_stride + j else null;
        if (j + VL <= ctx.len and ctx.act != .gelu) {
            var v = ctx.act.applyVec(g[0..VL].*);
            if (u) |up| v *= @as(VF, up[0..VL].*);
            o[0..VL].* = v;
        } else {
            for (0..@min(VL, ctx.len - j)) |k| {
                var v = ctx.act.apply(g[k]);
                if (u) |up| v *= up[k];
                o[k] = v;
            }
        }
    }
}

/// `out[i][j] = act(gate[i][j]) * up[i][j]` (or just `act(gate)` without `up`)
/// for `n` rows of `len` elements, in parallel; `out` may alias `gate`.
pub fn gatedActivation(pool: *const Pool, act: Activation, out: []f32, gate: []const f32, up: ?[]const f32, n: usize, len: usize, in_stride: usize, out_stride: usize) void {
    const ctx = ActCtx{ .act = act, .out = out.ptr, .gate = gate.ptr, .up = if (up) |u| u.ptr else null, .len = len, .in_stride = in_stride, .out_stride = out_stride };
    pool.parallelFor(n * ((len + VL - 1) / VL), &ctx, activationWorker);
}

/// LayerNorm over `x` with optional bias; `one_plus` scales by `(1 + w)`
/// (Nemotron). An empty `weight` means the non-parametric form (OLMo).
pub fn layernorm(out: []f32, x: []const f32, weight: []const f32, bias: ?[]const f32, eps: f32, one_plus: bool) void {
    const n: f32 = @floatFromInt(x.len);
    var mean: f32 = 0;
    for (x) |v| mean += v;
    mean /= n;
    var variance: f32 = 0;
    for (x) |v| variance += (v - mean) * (v - mean);
    variance /= n;
    const inv = 1.0 / @sqrt(variance + eps);
    for (out, 0..) |*o, i| {
        var y = (x[i] - mean) * inv;
        if (weight.len > 0) y *= if (one_plus) 1.0 + weight[i] else weight[i];
        if (bias) |b| y += b[i];
        o.* = y;
    }
}

/// Applies rotary position embeddings (HF "rotate_half" convention) to a
/// single head vector `x` of length `head_dim` at position `pos`.
pub fn applyRope(x: []f32, cos_row: []const f32, sin_row: []const f32) void {
    const half = x.len / 2;
    var i: usize = 0;
    while (i < half) : (i += 1) {
        const x1 = x[i];
        const x2 = x[i + half];
        x[i] = x1 * cos_row[i] - x2 * sin_row[i];
        x[i + half] = x2 * cos_row[i] + x1 * sin_row[i];
    }
}

/// Rotary embeddings over interleaved pairs `(x[2i], x[2i+1])` (GPT-J style).
pub fn applyRopeInterleaved(x: []f32, cos_row: []const f32, sin_row: []const f32) void {
    const half = x.len / 2;
    var i: usize = 0;
    while (i < half) : (i += 1) {
        const x1 = x[2 * i];
        const x2 = x[2 * i + 1];
        x[2 * i] = x1 * cos_row[i] - x2 * sin_row[i];
        x[2 * i + 1] = x2 * cos_row[i] + x1 * sin_row[i];
    }
}

/// Softcaps a vector: `cap * tanh(x / cap)`.
pub fn softcap(x: []f32, cap: f32) void {
    for (x) |*v| v.* = cap * std.math.tanh(v.* / cap);
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

test "bf16 round trip" {
    const vals = [_]f32{ 0, 1, -1, 3.14159, 1e-3, 65504, -2.5e10 };
    for (vals) |v| {
        const r = bf16ToF32(f32ToBf16(v));
        try std.testing.expect(@abs(r - v) <= @abs(v) * 0.01);
    }
}

test "matmulT with delta" {
    const gpa = std.testing.allocator;
    var threaded: Io.Threaded = .init_single_threaded;
    const pool = Pool.init(threaded.io(), 1);
    // W = [[1,2],[3,4],[5,6]] (3x2), x = [[1,1]], delta rank1: a=[1,0], b=[1,1,1] => W' = W + [[1,0],[1,0],[1,0]]
    const wf = [_]f32{ 1, 2, 3, 4, 5, 6 };
    const w = Weight{ .data = std.mem.sliceAsBytes(&wf), .dtype = .f32, .rows = 3, .cols = 2 };
    var a = [_]f32{ 1, 0 };
    var b = [_]f32{ 1, 1, 1 };
    const d = Delta{ .rank = 1, .a = &a, .b = &b };
    const x = [_]f32{ 1, 1 };
    var out: [3]f32 = undefined;
    try matmulT(&pool, gpa, &out, &x, 1, w, &d);
    try std.testing.expectEqual(@as(f32, 4), out[0]);
    try std.testing.expectEqual(@as(f32, 8), out[1]);
    try std.testing.expectEqual(@as(f32, 12), out[2]);
    var outT: [2]f32 = undefined;
    const y = [_]f32{ 1, 1, 1 };
    try matvecT(&pool, gpa, &outT, w, &y);
    try std.testing.expectEqual(@as(f32, 9), outT[0]);
    try std.testing.expectEqual(@as(f32, 12), outT[1]);
}

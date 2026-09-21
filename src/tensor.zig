//! Low-level numeric kernels: dtype conversion, parallel matrix products,
//! normalisation, activations and rotary embeddings. Everything computes in
//! f32; weights stay in their on-disk dtype and are converted row by row.

const std = @import("std");
const Io = std.Io;

pub const DType = enum {
    f32,
    f16,
    bf16,

    pub fn size(self: DType) usize {
        return switch (self) {
            .f32 => 4,
            .f16, .bf16 => 2,
        };
    }

    pub fn fromSafetensors(name: []const u8) ?DType {
        if (std.mem.eql(u8, name, "F32")) return .f32;
        if (std.mem.eql(u8, name, "F16")) return .f16;
        if (std.mem.eql(u8, name, "BF16")) return .bf16;
        return null;
    }

    pub fn safetensorsName(self: DType) []const u8 {
        return switch (self) {
            .f32 => "F32",
            .f16 => "F16",
            .bf16 => "BF16",
        };
    }
};

// ---------------------------------------------------------------------------
// Thread pool
// ---------------------------------------------------------------------------

/// A minimal fork-join helper on top of `std.Io`. Every parallel kernel splits
/// its range into `threads` chunks and runs them as async tasks in a group.
pub const Pool = struct {
    io: Io,
    threads: usize,

    pub fn init(io: Io, threads: ?usize) Pool {
        const n = threads orelse (std.Thread.getCpuCount() catch 1);
        return .{ .io = io, .threads = @max(1, n) };
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

/// Converts `count` elements of raw `dtype` data starting at `bytes` into `out`.
pub fn convertToF32(dtype: DType, bytes: []const u8, out: []f32) void {
    switch (dtype) {
        .f32 => {
            const src = std.mem.bytesAsSlice(f32, bytes[0 .. out.len * 4]);
            @memcpy(out, src);
        },
        .bf16 => {
            const src = std.mem.bytesAsSlice(u16, bytes[0 .. out.len * 2]);
            const V = @Vector(16, u32);
            var i: usize = 0;
            while (i + 16 <= out.len) : (i += 16) {
                const v: @Vector(16, u16) = src[i..][0..16].*;
                const wide: V = @as(V, @intCast(v)) << @splat(16);
                out[i..][0..16].* = @bitCast(wide);
            }
            while (i < out.len) : (i += 1) out[i] = bf16ToF32(src[i]);
        },
        .f16 => {
            const src = std.mem.bytesAsSlice(u16, bytes[0 .. out.len * 2]);
            for (out, 0..) |*o, i| o.* = f16ToF32(src[i]);
        },
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
    }
}

// ---------------------------------------------------------------------------
// Weight matrix views
// ---------------------------------------------------------------------------

/// A row-major matrix `[rows][cols]` stored in its on-disk dtype.
pub const Weight = struct {
    data: []const u8,
    dtype: DType,
    rows: usize,
    cols: usize,

    pub fn row(self: Weight, r: usize, out: []f32) void {
        const es = self.dtype.size();
        convertToF32(self.dtype, self.data[r * self.cols * es ..][0 .. self.cols * es], out[0..self.cols]);
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

pub fn dot(a: []const f32, b: []const f32) f32 {
    const V = @Vector(16, f32);
    var acc: V = @splat(0);
    var i: usize = 0;
    const n = a.len;
    while (i + 16 <= n) : (i += 16) {
        const va: V = a[i..][0..16].*;
        const vb: V = b[i..][0..16].*;
        acc += va * vb;
    }
    var s = @reduce(.Add, acc);
    while (i < n) : (i += 1) s += a[i] * b[i];
    return s;
}

/// `y += alpha * x`
pub fn axpy(y: []f32, alpha: f32, x: []const f32) void {
    const V = @Vector(16, f32);
    const va: V = @splat(alpha);
    var i: usize = 0;
    while (i + 16 <= y.len) : (i += 16) {
        const vy: V = y[i..][0..16].*;
        const vx: V = x[i..][0..16].*;
        y[i..][0..16].* = vy + va * vx;
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

const MatmulCtx = struct {
    out: []f32,
    x: []const f32,
    n: usize,
    w: Weight,
    delta: ?*const Delta,
    scratch: []f32, // one slot of 4 converted weight rows per task
    scratch_per_task: usize,
};

fn matmulWorker(ctx: *const MatmulCtx, start: usize, end: usize) void {
    const slot = start / ctx.scratch_per_task;
    const cols = ctx.w.cols;
    const n = ctx.n;
    const rows = ctx.w.rows;
    // Scratch holds 4 converted weight rows.
    const buf = ctx.scratch[slot * (4 * cols + 1) ..][0 .. 4 * cols];
    var r = start;
    while (r < end) {
        const nr = @min(4, end - r);
        for (0..nr) |k| ctx.w.row(r + k, buf[k * cols ..][0..cols]);
        var i: usize = 0;
        while (i < n) : (i += 1) {
            const xi = ctx.x[i * cols ..][0..cols];
            if (nr == 4) {
                const d = dot4(buf[0..cols], buf[cols .. 2 * cols], buf[2 * cols .. 3 * cols], buf[3 * cols .. 4 * cols], xi);
                inline for (0..4) |k| ctx.out[i * rows + r + k] = d[k];
            } else {
                for (0..nr) |k| ctx.out[i * rows + r + k] = dot(buf[k * cols ..][0..cols], xi);
            }
            if (ctx.delta) |dl| {
                for (0..nr) |k| {
                    var v: f32 = 0;
                    var kk: usize = 0;
                    while (kk < dl.rank) : (kk += 1) v += dl.b[(r + k) * dl.rank + kk] * dot(dl.a[kk * cols ..][0..cols], xi);
                    ctx.out[i * rows + r + k] += v;
                }
            }
        }
        r += nr;
    }
}

/// Four dot products sharing the loads of `x`.
inline fn dot4(a0: []const f32, a1: []const f32, a2: []const f32, a3: []const f32, x: []const f32) [4]f32 {
    const V = @Vector(16, f32);
    var c0: V = @splat(0);
    var c1: V = @splat(0);
    var c2: V = @splat(0);
    var c3: V = @splat(0);
    var i: usize = 0;
    const n = x.len;
    while (i + 16 <= n) : (i += 16) {
        const vx: V = x[i..][0..16].*;
        c0 += @as(V, a0[i..][0..16].*) * vx;
        c1 += @as(V, a1[i..][0..16].*) * vx;
        c2 += @as(V, a2[i..][0..16].*) * vx;
        c3 += @as(V, a3[i..][0..16].*) * vx;
    }
    var r = [4]f32{ @reduce(.Add, c0), @reduce(.Add, c1), @reduce(.Add, c2), @reduce(.Add, c3) };
    while (i < n) : (i += 1) {
        r[0] += a0[i] * x[i];
        r[1] += a1[i] * x[i];
        r[2] += a2[i] * x[i];
        r[3] += a3[i] * x[i];
    }
    return r;
}

/// `out[n][rows] = x[n][cols] @ W^T (+ x @ (B A)^T)`.
pub fn matmulT(pool: *const Pool, gpa: std.mem.Allocator, out: []f32, x: []const f32, n: usize, w: Weight, delta: ?*const Delta) !void {
    std.debug.assert(x.len >= n * w.cols);
    std.debug.assert(out.len >= n * w.rows);
    const chunks = @max(1, @min(pool.threads, w.rows));
    const per = (w.rows + chunks - 1) / chunks;
    const slots = (w.rows + per - 1) / per;
    const scratch = try gpa.alloc(f32, slots * (4 * w.cols + 1));
    defer gpa.free(scratch);
    const ctx = MatmulCtx{ .out = out, .x = x, .n = n, .w = w, .delta = delta, .scratch = scratch, .scratch_per_task = per };
    pool.parallelFor(w.rows, &ctx, matmulWorker);
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

// ---------------------------------------------------------------------------
// Element-wise kernels
// ---------------------------------------------------------------------------

/// RMS normalisation. `gemma_style` uses `(1 + w)` scaling.
pub fn rmsnorm(out: []f32, x: []const f32, weight: []const f32, eps: f32, gemma_style: bool) void {
    var ss: f32 = 0;
    for (x) |v| ss += v * v;
    const inv = 1.0 / @sqrt(ss / @as(f32, @floatFromInt(x.len)) + eps);
    if (gemma_style) {
        for (out, 0..) |*o, i| o.* = x[i] * inv * (1.0 + weight[i]);
    } else {
        for (out, 0..) |*o, i| o.* = x[i] * inv * weight[i];
    }
}

pub fn softmaxInPlace(x: []f32) void {
    var m: f32 = -std.math.inf(f32);
    for (x) |v| m = @max(m, v);
    var s: f32 = 0;
    for (x) |*v| {
        v.* = @exp(v.* - m);
        s += v.*;
    }
    const inv = 1.0 / s;
    for (x) |*v| v.* *= inv;
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

    pub fn apply(self: Activation, x: f32) f32 {
        return switch (self) {
            .silu => silu(x),
            .gelu_tanh => geluTanh(x),
            .gelu => geluErf(x),
        };
    }
};

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

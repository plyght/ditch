//! The Metal backend: a `compute.Device` backed by Metal compute shaders on
//! Apple silicon.
//!
//! The kernels live in `shaders.metal`, embedded in the binary and compiled at
//! start-up with `newLibraryWithSource:` (so building ditch needs no Apple
//! toolchain); the Objective-C glue is `shim.m`, reached through the plain C
//! declarations below. `build.zig` compiles and links the shim only for macOS
//! targets and only with `-Dmetal`, and `compute.zig` only ever reaches this
//! file when that option is on, so nothing here affects other platforms.
//!
//! Memory: every buffer is `MTLResourceStorageModeShared`, i.e. unified memory
//! the CPU and GPU both address. A weight tile that is already page aligned
//! (memory-mapped weights usually are) is wrapped without copying; otherwise it
//! is copied once, in its on-disk dtype, and converted inside the kernel — it
//! is never converted on the host and never copied twice. Tiles are dropped
//! after the call unless the residency cache (`--gpu-memory`) keeps them, which
//! only happens when host weight pointers are stable (memory-mapped mode).

const std = @import("std");
const compute = @import("../compute.zig");
const tensor = @import("../tensor.zig");

const Allocator = std.mem.Allocator;
const Weight = tensor.Weight;
const Error = compute.Error;

const shaders = @embedFile("shaders.metal");

// ---------------------------------------------------------------------------
// The C surface of shim.m (see shim.h)
// ---------------------------------------------------------------------------

const Ctx = opaque {};
const Buffer = opaque {};

extern fn ditch_mtl_open(source: [*:0]const u8, err: [*]u8, err_len: usize) ?*Ctx;
extern fn ditch_mtl_close(m: *Ctx) void;
extern fn ditch_mtl_name(m: *Ctx) [*:0]const u8;
extern fn ditch_mtl_working_set(m: *Ctx) u64;
extern fn ditch_mtl_pipeline(m: *Ctx, name: [*:0]const u8) i32;
extern fn ditch_mtl_buffer(m: *Ctx, bytes: u64) ?*Buffer;
extern fn ditch_mtl_buffer_nocopy(m: *Ctx, ptr: *anyopaque, bytes: u64) ?*Buffer;
extern fn ditch_mtl_contents(buffer: *Buffer) ?*anyopaque;
extern fn ditch_mtl_release(buffer: *Buffer) void;
extern fn ditch_mtl_dispatch(
    m: *Ctx,
    pipeline: i32,
    buffers: [*]const *Buffer,
    n_buffers: i32,
    params: ?*const anyopaque,
    params_len: u64,
    gx: u32,
    gy: u32,
    gz: u32,
    tx: u32,
    ty: u32,
    tz: u32,
) i32;
extern fn ditch_mtl_max_threads(m: *Ctx, pipeline: i32) u32;

// ---------------------------------------------------------------------------
// Parameter blocks (must match the structs in shaders.metal)
// ---------------------------------------------------------------------------

const MatmulParams = extern struct { n: u32, rows: u32, cols: u32 };
const MatvecParams = extern struct { q: u32, rows: u32, cols: u32 };
const RowNormParams = extern struct { rows: u32, cols: u32 };
const AttnScoreParams = extern struct { keys: u32, hd: u32, stride: u32, scale: f32 };
const AttnValueParams = extern struct { keys: u32, vd: u32, stride: u32 };
const GatedParams = extern struct { act: u32, n: u32, len: u32, in_stride: u32, out_stride: u32, has_up: u32 };
const NormParams = extern struct { n: u32, len: u32, eps: f32, flags: u32 };
const SoftmaxParams = extern struct { n: u32, len: u32 };
const RopeParams = extern struct { n: u32, dim: u32, half_dim: u32, style: u32 };

/// Threads per threadgroup of the tiled matmul (matches TS in shaders.metal).
const tile = 16;
/// Threads of the one-threadgroup-per-row reductions (matches RED).
const reduce_threads = 256;
/// Threads per group of the flat element-wise kernels.
const flat_threads = 64;

// ---------------------------------------------------------------------------
// Backend
// ---------------------------------------------------------------------------

const Scratch = struct {
    buf: ?*Buffer = null,
    bytes: u64 = 0,
};

pub const Backend = struct {
    gpa: Allocator,
    ctx: *Ctx,
    name: []u8,
    io: std.Io,
    /// Held across a whole dispatch, including the wait for the GPU, so a
    /// contending thread must sleep rather than spin.
    lock: std.Io.Mutex = .init,
    residency: compute.Residency,
    /// Reused staging buffers for inputs, outputs and small tables.
    scratch: [5]Scratch = [_]Scratch{.{}} ** 5,
    /// Weight tiles that are neither resident nor wrapped are staged here.
    weight_scratch: Scratch = .{},

    fn pipeline(self: *Backend, name: [*:0]const u8) Error!i32 {
        const idx = ditch_mtl_pipeline(self.ctx, name);
        if (idx < 0) return error.Unsupported;
        return idx;
    }

    /// A scratch buffer of at least `bytes`, grown if needed.
    fn slot(self: *Backend, i: usize, bytes: u64) Error!*Buffer {
        const s = &self.scratch[i];
        if (s.buf) |b| {
            if (s.bytes >= bytes) return b;
            ditch_mtl_release(b);
            s.buf = null;
            s.bytes = 0;
        }
        const want = @max(bytes, 4096);
        const b = ditch_mtl_buffer(self.ctx, want) orelse return error.OutOfMemory;
        s.buf = b;
        s.bytes = want;
        return b;
    }

    fn upload(self: *Backend, i: usize, data: []const f32) Error!*Buffer {
        const b = try self.slot(i, @max(data.len * 4, 4));
        if (data.len > 0) {
            const dst: [*]u8 = @ptrCast(ditch_mtl_contents(b) orelse return error.DeviceFailed);
            @memcpy(dst[0 .. data.len * 4], std.mem.sliceAsBytes(data));
        }
        return b;
    }

    fn uploadU32(self: *Backend, i: usize, data: []const u32) Error!*Buffer {
        const b = try self.slot(i, @max(data.len * 4, 4));
        if (data.len > 0) {
            const dst: [*]u8 = @ptrCast(ditch_mtl_contents(b) orelse return error.DeviceFailed);
            @memcpy(dst[0 .. data.len * 4], std.mem.sliceAsBytes(data));
        }
        return b;
    }

    fn download(_: *Backend, b: *Buffer, out: []f32) Error!void {
        if (out.len == 0) return;
        const src: [*]const u8 = @ptrCast(ditch_mtl_contents(b) orelse return error.DeviceFailed);
        @memcpy(std.mem.sliceAsBytes(out), src[0 .. out.len * 4]);
    }

    /// A device buffer holding the raw bytes of `data`: the cached one, a
    /// no-copy alias of the host pages, or one copy into shared memory.
    /// `owned` buffers are released by `releaseWeight`.
    const WeightBuf = struct { buf: *Buffer, owned: bool };

    fn weightBuffer(self: *Backend, data: []const u8) Error!WeightBuf {
        // Residency (memory-mapped weights, --gpu-memory): look the tile up,
        // then alias its host pages when they are page aligned, else copy it
        // once. Streamed and warp modes skip all of this — their host buffers
        // are reused for different tensors, so neither caching by address nor
        // keeping a buffer alive across calls would be sound, and creating a
        // Metal buffer per call would cost more than the staging copy below.
        if (self.residency.enabled()) {
            if (self.residency.get(data)) |h| return .{ .buf = @ptrCast(@alignCast(h)), .owned = false };
            const mutable: *anyopaque = @ptrCast(@constCast(data.ptr));
            if (ditch_mtl_buffer_nocopy(self.ctx, mutable, data.len)) |b| {
                if (self.residency.put(data, data.len, b)) return .{ .buf = b, .owned = false };
                return .{ .buf = b, .owned = true };
            }
            const b = ditch_mtl_buffer(self.ctx, data.len) orelse return error.OutOfMemory;
            const dst: [*]u8 = @ptrCast(ditch_mtl_contents(b) orelse {
                ditch_mtl_release(b);
                return error.DeviceFailed;
            });
            @memcpy(dst[0..data.len], data);
            if (self.residency.put(data, data.len, b)) return .{ .buf = b, .owned = false };
            return .{ .buf = b, .owned = true };
        }
        // Upload, compute, drop: reuse one staging buffer for every tile.
        const s = &self.weight_scratch;
        if (s.buf) |b| {
            if (s.bytes < data.len) {
                ditch_mtl_release(b);
                s.buf = null;
                s.bytes = 0;
            }
        }
        if (s.buf == null) {
            const b = ditch_mtl_buffer(self.ctx, @max(data.len, 4096)) orelse return error.OutOfMemory;
            s.buf = b;
            s.bytes = @max(data.len, 4096);
        }
        const b = s.buf.?;
        const dst: [*]u8 = @ptrCast(ditch_mtl_contents(b) orelse return error.DeviceFailed);
        @memcpy(dst[0..data.len], data);
        return .{ .buf = b, .owned = false };
    }

    fn releaseWeight(_: *Backend, wb: WeightBuf) void {
        if (wb.owned) ditch_mtl_release(wb.buf);
    }

    fn dispatch(self: *Backend, pipe: i32, buffers: []const *Buffer, params: *const anyopaque, params_len: usize, groups: [3]u32, threads: [3]u32) Error!void {
        const rc = ditch_mtl_dispatch(
            self.ctx,
            pipe,
            buffers.ptr,
            @intCast(buffers.len),
            params,
            params_len,
            groups[0],
            groups[1],
            groups[2],
            threads[0],
            threads[1],
            threads[2],
        );
        if (rc != 0) return error.DeviceFailed;
    }
};

/// Kernel name suffix for a floating-point weight dtype.
fn dtypeSuffix(dt: tensor.DType) Error![]const u8 {
    return switch (dt) {
        .f32 => "f32",
        .f16 => "f16",
        .bf16 => "bf16",
        else => error.Unsupported, // quantised and integer tiles stay on the CPU
    };
}

fn kernelName(buf: []u8, comptime base: []const u8, dt: tensor.DType) Error![:0]const u8 {
    const suffix = try dtypeSuffix(dt);
    return std.fmt.bufPrintZ(buf, base ++ "_{s}", .{suffix}) catch error.Unsupported;
}

fn ceilDiv(a: usize, b: usize) u32 {
    return @intCast((a + b - 1) / b);
}

// ---------------------------------------------------------------------------
// Operations
// ---------------------------------------------------------------------------

fn matmulT(ctx: *anyopaque, out: []f32, x: []const f32, n: usize, w: Weight) Error!void {
    const self: *Backend = @ptrCast(@alignCast(ctx));
    if (w.cols == 0 or w.rows == 0 or n == 0) return error.Unsupported;
    var name_buf: [64]u8 = undefined;
    const name = try kernelName(&name_buf, "matmul", w.dtype);
    self.lock.lockUncancelable(self.io);
    defer self.lock.unlock(self.io);
    const pipe = try self.pipeline(name.ptr);
    const xb = try self.upload(0, x[0 .. n * w.cols]);
    const ob = try self.slot(1, n * w.rows * 4);
    const wb = try self.weightBuffer(w.data[0 .. w.rows * w.dtype.rowBytes(w.cols)]);
    defer self.releaseWeight(wb);
    const p = MatmulParams{ .n = @intCast(n), .rows = @intCast(w.rows), .cols = @intCast(w.cols) };
    try self.dispatch(pipe, &.{ ob, xb, wb.buf }, &p, @sizeOf(MatmulParams), .{ ceilDiv(w.rows, tile), ceilDiv(n, tile), 1 }, .{ tile, tile, 1 });
    try self.download(ob, out[0 .. n * w.rows]);
}

fn matvecTMulti(ctx: *anyopaque, out: []f32, w: Weight, y: []const f32, q: usize) Error!void {
    const self: *Backend = @ptrCast(@alignCast(ctx));
    if (w.cols == 0 or w.rows == 0 or q == 0) return error.Unsupported;
    var name_buf: [64]u8 = undefined;
    const name = try kernelName(&name_buf, "matvec", w.dtype);
    self.lock.lockUncancelable(self.io);
    defer self.lock.unlock(self.io);
    const pipe = try self.pipeline(name.ptr);
    const yb = try self.upload(0, y[0 .. q * w.rows]);
    const ob = try self.slot(1, q * w.cols * 4);
    const wb = try self.weightBuffer(w.data[0 .. w.rows * w.dtype.rowBytes(w.cols)]);
    defer self.releaseWeight(wb);
    const p = MatvecParams{ .q = @intCast(q), .rows = @intCast(w.rows), .cols = @intCast(w.cols) };
    try self.dispatch(pipe, &.{ ob, yb, wb.buf }, &p, @sizeOf(MatvecParams), .{ ceilDiv(w.cols, flat_threads), @intCast(q), 1 }, .{ flat_threads, 1, 1 });
    try self.download(ob, out[0 .. q * w.cols]);
}

fn rowNorms(ctx: *anyopaque, out: []f32, w: Weight) Error!void {
    const self: *Backend = @ptrCast(@alignCast(ctx));
    if (w.cols == 0 or w.rows == 0) return error.Unsupported;
    var name_buf: [64]u8 = undefined;
    const name = try kernelName(&name_buf, "row_norms", w.dtype);
    self.lock.lockUncancelable(self.io);
    defer self.lock.unlock(self.io);
    const pipe = try self.pipeline(name.ptr);
    const ob = try self.slot(1, w.rows * 4);
    const wb = try self.weightBuffer(w.data[0 .. w.rows * w.dtype.rowBytes(w.cols)]);
    defer self.releaseWeight(wb);
    const p = RowNormParams{ .rows = @intCast(w.rows), .cols = @intCast(w.cols) };
    try self.dispatch(pipe, &.{ ob, wb.buf }, &p, @sizeOf(RowNormParams), .{ @intCast(w.rows), 1, 1 }, .{ reduce_threads, 1, 1 });
    try self.download(ob, out[0..w.rows]);
}

fn attentionScores(ctx: *anyopaque, scores: []f32, q: []const f32, k: []const f32, stride: usize, scale: f32) Error!void {
    const self: *Backend = @ptrCast(@alignCast(ctx));
    if (scores.len == 0 or q.len == 0) return error.Unsupported;
    self.lock.lockUncancelable(self.io);
    defer self.lock.unlock(self.io);
    const pipe = try self.pipeline("attn_scores");
    const qb = try self.upload(0, q);
    const kb = try self.upload(2, k);
    const ob = try self.slot(1, scores.len * 4);
    const p = AttnScoreParams{ .keys = @intCast(scores.len), .hd = @intCast(q.len), .stride = @intCast(stride), .scale = scale };
    try self.dispatch(pipe, &.{ ob, qb, kb }, &p, @sizeOf(AttnScoreParams), .{ ceilDiv(scores.len, flat_threads), 1, 1 }, .{ flat_threads, 1, 1 });
    try self.download(ob, scores);
}

fn attentionValues(ctx: *anyopaque, out: []f32, scores: []const f32, v: []const f32, stride: usize) Error!void {
    const self: *Backend = @ptrCast(@alignCast(ctx));
    if (out.len == 0 or scores.len == 0) return error.Unsupported;
    self.lock.lockUncancelable(self.io);
    defer self.lock.unlock(self.io);
    const pipe = try self.pipeline("attn_values");
    const sb = try self.upload(0, scores);
    const vb = try self.upload(2, v);
    const ob = try self.slot(1, out.len * 4);
    const p = AttnValueParams{ .keys = @intCast(scores.len), .vd = @intCast(out.len), .stride = @intCast(stride) };
    try self.dispatch(pipe, &.{ ob, sb, vb }, &p, @sizeOf(AttnValueParams), .{ ceilDiv(out.len, flat_threads), 1, 1 }, .{ flat_threads, 1, 1 });
    try self.download(ob, out);
}

fn gatedActivation(ctx: *anyopaque, act: compute.ActivationCode, out: []f32, gate: []const f32, up: ?[]const f32, n: usize, len: usize, in_stride: usize, out_stride: usize) Error!void {
    const self: *Backend = @ptrCast(@alignCast(ctx));
    if (n == 0 or len == 0) return error.Unsupported;
    self.lock.lockUncancelable(self.io);
    defer self.lock.unlock(self.io);
    const pipe = try self.pipeline("gated");
    const gb = try self.upload(0, gate[0 .. (n - 1) * in_stride + len]);
    const ub = if (up) |u| try self.upload(2, u[0 .. (n - 1) * in_stride + len]) else gb;
    const ob = try self.slot(1, ((n - 1) * out_stride + len) * 4);
    const p = GatedParams{
        .act = act,
        .n = @intCast(n),
        .len = @intCast(len),
        .in_stride = @intCast(in_stride),
        .out_stride = @intCast(out_stride),
        .has_up = if (up == null) 0 else 1,
    };
    try self.dispatch(pipe, &.{ ob, gb, ub }, &p, @sizeOf(GatedParams), .{ ceilDiv(len, flat_threads), @intCast(n), 1 }, .{ flat_threads, 1, 1 });
    try self.download(ob, out[0 .. (n - 1) * out_stride + len]);
}

fn normFlags(weight: []const f32, one_plus: bool, bias: ?[]const f32) u32 {
    var flags: u32 = 0;
    if (weight.len > 0) flags |= 1;
    if (one_plus) flags |= 2;
    if (bias != null) flags |= 4;
    return flags;
}

fn rmsnormRows(ctx: *anyopaque, out: []f32, x: []const f32, weight: []const f32, n: usize, len: usize, eps: f32, gemma_style: bool) Error!void {
    const self: *Backend = @ptrCast(@alignCast(ctx));
    if (n == 0 or len == 0) return error.Unsupported;
    self.lock.lockUncancelable(self.io);
    defer self.lock.unlock(self.io);
    const pipe = try self.pipeline("rmsnorm_rows");
    const xb = try self.upload(0, x[0 .. n * len]);
    const wb = try self.upload(2, weight);
    const ob = try self.slot(1, n * len * 4);
    const p = NormParams{ .n = @intCast(n), .len = @intCast(len), .eps = eps, .flags = normFlags(weight, gemma_style, null) };
    try self.dispatch(pipe, &.{ ob, xb, wb }, &p, @sizeOf(NormParams), .{ @intCast(n), 1, 1 }, .{ reduce_threads, 1, 1 });
    try self.download(ob, out[0 .. n * len]);
}

fn layernormRows(ctx: *anyopaque, out: []f32, x: []const f32, weight: []const f32, bias: ?[]const f32, n: usize, len: usize, eps: f32, one_plus: bool) Error!void {
    const self: *Backend = @ptrCast(@alignCast(ctx));
    if (n == 0 or len == 0) return error.Unsupported;
    self.lock.lockUncancelable(self.io);
    defer self.lock.unlock(self.io);
    const pipe = try self.pipeline("layernorm_rows");
    const xb = try self.upload(0, x[0 .. n * len]);
    const wb = try self.upload(2, weight);
    const bb = if (bias) |b| try self.upload(3, b) else wb;
    const ob = try self.slot(1, n * len * 4);
    const p = NormParams{ .n = @intCast(n), .len = @intCast(len), .eps = eps, .flags = normFlags(weight, one_plus, bias) };
    try self.dispatch(pipe, &.{ ob, xb, wb, bb }, &p, @sizeOf(NormParams), .{ @intCast(n), 1, 1 }, .{ reduce_threads, 1, 1 });
    try self.download(ob, out[0 .. n * len]);
}

fn softmaxRows(ctx: *anyopaque, x: []f32, n: usize, len: usize) Error!void {
    const self: *Backend = @ptrCast(@alignCast(ctx));
    if (n == 0 or len == 0) return error.Unsupported;
    self.lock.lockUncancelable(self.io);
    defer self.lock.unlock(self.io);
    const pipe = try self.pipeline("softmax_rows");
    const xb = try self.upload(0, x[0 .. n * len]);
    const p = SoftmaxParams{ .n = @intCast(n), .len = @intCast(len) };
    try self.dispatch(pipe, &.{xb}, &p, @sizeOf(SoftmaxParams), .{ @intCast(n), 1, 1 }, .{ reduce_threads, 1, 1 });
    try self.download(xb, x[0 .. n * len]);
}

fn ropeRows(ctx: *anyopaque, x: []f32, n: usize, dim: usize, cos: []const f32, sin: []const f32, half: usize, pos: []const u32, style: compute.RopeStyle) Error!void {
    const self: *Backend = @ptrCast(@alignCast(ctx));
    if (n == 0 or half == 0) return error.Unsupported;
    self.lock.lockUncancelable(self.io);
    defer self.lock.unlock(self.io);
    const pipe = try self.pipeline("rope_rows");
    const xb = try self.upload(0, x[0 .. n * dim]);
    const cb = try self.upload(1, cos);
    const sb = try self.upload(2, sin);
    const pb = try self.uploadU32(3, pos[0..n]);
    const p = RopeParams{ .n = @intCast(n), .dim = @intCast(dim), .half_dim = @intCast(half), .style = @intFromEnum(style) };
    try self.dispatch(pipe, &.{ xb, cb, sb, pb }, &p, @sizeOf(RopeParams), .{ ceilDiv(half, flat_threads), @intCast(n), 1 }, .{ flat_threads, 1, 1 });
    try self.download(xb, x[0 .. n * dim]);
}

fn deinit(ctx: *anyopaque) void {
    const self: *Backend = @ptrCast(@alignCast(ctx));
    self.residency.deinit();
    for (&self.scratch) |*s| {
        if (s.buf) |b| ditch_mtl_release(b);
        s.* = .{};
    }
    if (self.weight_scratch.buf) |b| ditch_mtl_release(b);
    ditch_mtl_close(self.ctx);
    const gpa = self.gpa;
    gpa.free(self.name);
    gpa.destroy(self);
}

fn forgetWeights(ctx: *anyopaque) void {
    const self: *Backend = @ptrCast(@alignCast(ctx));
    self.lock.lockUncancelable(self.io);
    defer self.lock.unlock(self.io);
    self.residency.clear();
}

const vtable = compute.VTable{
    .deinit = deinit,
    .forgetWeights = forgetWeights,
    .matmulT = matmulT,
    .matvecTMulti = matvecTMulti,
    .rowNorms = rowNorms,
    .attentionScores = attentionScores,
    .attentionValues = attentionValues,
    .gatedActivation = gatedActivation,
    .rmsnormRows = rmsnormRows,
    .layernormRows = layernormRows,
    .softmaxRows = softmaxRows,
    .ropeRows = ropeRows,
};

fn releaseBuffer(_: *anyopaque, handle: *anyopaque) void {
    ditch_mtl_release(@ptrCast(@alignCast(handle)));
}

/// Opens the default Metal device and compiles the shaders. `memory_budget` is
/// the device memory the residency cache may hold (0 = upload, compute, drop).
pub fn open(gpa: Allocator, io: std.Io, memory_budget: u64) !compute.Device {
    var err_buf: [512]u8 = undefined;
    err_buf[0] = 0;
    const ctx = ditch_mtl_open(shaders, &err_buf, err_buf.len) orelse {
        return error.DeviceUnavailable;
    };
    errdefer ditch_mtl_close(ctx);
    const dev_name = std.mem.span(ditch_mtl_name(ctx));
    const name = try std.fmt.allocPrint(gpa, "{s} (Metal)", .{dev_name});
    errdefer gpa.free(name);
    const self = try gpa.create(Backend);
    self.* = .{
        .gpa = gpa,
        .io = io,
        .ctx = ctx,
        .name = name,
        .residency = undefined,
    };
    // The cache is capped by the device's own recommended working set.
    const ws = ditch_mtl_working_set(ctx);
    const budget = if (ws > 0) @min(memory_budget, ws) else memory_budget;
    self.residency = compute.Residency.init(gpa, budget, self, releaseBuffer);
    return .{
        .kind = .metal,
        .name = name,
        .vtable = &vtable,
        .ctx = self,
        .memory_budget = budget,
    };
}

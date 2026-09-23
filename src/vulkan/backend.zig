//! The Vulkan backend: a `compute.Device` backed by Vulkan compute shaders, for
//! NVIDIA, AMD and Intel GPUs on Linux and Windows.
//!
//! It mirrors the Metal backend (`src/metal/backend.zig`): the same kernels,
//! the same parameter blocks, the same "upload, compute, drop" handling of
//! weight tiles and the same residency cache behind `--gpu-memory`. The
//! kernels are GLSL (`shaders/*.comp`) compiled ahead of time to the SPIR-V in
//! `spirv/`, which is embedded with `@embedFile`; `tools/gen_spirv.sh`
//! regenerates it. Vulkan itself is loaded at run time (`vk.zig`), so a
//! machine without a Vulkan driver runs the same binary on the CPU.
//!
//! Device choice, memory types and command submission are in `device.zig`;
//! this file is the kernels: which SPIR-V module, which buffers, which grid.
//!
//! Weight tiles are copied as the raw bytes of their on-disk dtype (f32, f16
//! or bf16) and converted inside the kernel, reading 32-bit words: f16 goes
//! through `unpackHalf2x16`, so neither `shaderFloat16` nor 16-bit storage is
//! required and every Vulkan 1.0 device can take every dtype. Quantised tiles
//! stay on the CPU, as with Metal.
//!
//! Every call is synchronous: the backend records one command buffer (copies
//! in, one dispatch, copies out), submits it and waits on a fence, under a
//! lock, so it may be called from any pool thread.

const std = @import("std");
const compute = @import("../compute.zig");
const tensor = @import("../tensor.zig");
const vk = @import("vk.zig");
const device = @import("device.zig");

const Context = device.Context;
const Buf = device.Buf;
const Bind = Context.Bind;

const Allocator = std.mem.Allocator;
const Weight = tensor.Weight;
const Error = compute.Error;

// ---------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------

pub const Kernel = enum {
    matmul_f32,
    matmul_f16,
    matmul_bf16,
    matvec_f32,
    matvec_f16,
    matvec_bf16,
    row_norms_f32,
    row_norms_f16,
    row_norms_bf16,
    attn_scores,
    attn_values,
    gated,
    rmsnorm_rows,
    layernorm_rows,
    softmax_rows,
    rope_rows,

    /// The embedded SPIR-V module (see `tools/gen_spirv.sh`).
    pub fn spirv(self: Kernel) []const u8 {
        return switch (self) {
            inline else => |k| @embedFile("spirv/" ++ @tagName(k) ++ ".spv"),
        };
    }

    /// The weight-reading kernel `base` (matmul, matvec, row_norms) for `dt`.
    fn forDtype(comptime base: []const u8, dt: tensor.DType) Error!Kernel {
        return switch (dt) {
            .f32 => @field(Kernel, base ++ "_f32"),
            .f16 => @field(Kernel, base ++ "_f16"),
            .bf16 => @field(Kernel, base ++ "_bf16"),
            else => error.Unsupported, // quantised and integer tiles stay on the CPU
        };
    }
};

const n_kernels = @typeInfo(Kernel).@"enum".fields.len;

// Parameter blocks: the push constant blocks of the shaders, and the same
// structs as the Metal backend's. `spirv_test.zig` checks each against the
// `layout(push_constant)` block of its shader, field by field.
pub const MatmulParams = extern struct { n: u32, rows: u32, cols: u32 };
pub const MatvecParams = extern struct { q: u32, rows: u32, cols: u32 };
pub const RowNormParams = extern struct { rows: u32, cols: u32 };
pub const AttnScoreParams = extern struct { keys: u32, hd: u32, stride: u32, scale: f32 };
pub const AttnValueParams = extern struct { keys: u32, vd: u32, stride: u32 };
pub const GatedParams = extern struct { act: u32, n: u32, len: u32, in_stride: u32, out_stride: u32, has_up: u32 };
pub const NormParams = extern struct { n: u32, len: u32, eps: f32, flags: u32 };
pub const SoftmaxParams = extern struct { n: u32, len: u32 };
pub const RopeParams = extern struct { n: u32, dim: u32, half_dim: u32, style: u32 };

/// The parameter block of kernel `k`.
pub fn Params(comptime k: Kernel) type {
    return switch (k) {
        .matmul_f32, .matmul_f16, .matmul_bf16 => MatmulParams,
        .matvec_f32, .matvec_f16, .matvec_bf16 => MatvecParams,
        .row_norms_f32, .row_norms_f16, .row_norms_bf16 => RowNormParams,
        .attn_scores => AttnScoreParams,
        .attn_values => AttnValueParams,
        .gated => GatedParams,
        .rmsnorm_rows, .layernorm_rows => NormParams,
        .softmax_rows => SoftmaxParams,
        .rope_rows => RopeParams,
    };
}

comptime {
    std.debug.assert(n_kernels <= device.max_pipelines);
    for (.{ MatmulParams, MatvecParams, RowNormParams, AttnScoreParams, AttnValueParams, GatedParams, NormParams, SoftmaxParams, RopeParams }) |T| {
        std.debug.assert(@sizeOf(T) <= device.push_bytes);
    }
}

/// Workgroup sizes, matching the `local_size` of the shaders.
pub const tile = 16;
pub const reduce_threads = 256;
pub const flat_threads = 64;

/// `local_size_x` of kernel `k`'s shader (checked by `spirv_test.zig`).
pub fn workgroupWidth(k: Kernel) u32 {
    return switch (k) {
        .matmul_f32, .matmul_f16, .matmul_bf16 => tile,
        .row_norms_f32, .row_norms_f16, .row_norms_bf16, .rmsnorm_rows, .layernorm_rows, .softmax_rows => reduce_threads,
        else => flat_threads,
    };
}

/// The GLSL source a kernel is compiled from (`shaders/<name>.comp`).
pub fn sourceName(k: Kernel) []const u8 {
    const name = @tagName(k);
    for ([_][]const u8{ "_f32", "_f16", "_bf16" }) |suffix| {
        if (std.mem.endsWith(u8, name, suffix)) return name[0 .. name.len - suffix.len];
    }
    return name;
}

// ---------------------------------------------------------------------------
// Backend
// ---------------------------------------------------------------------------

pub const Backend = struct {
    gpa: Allocator,
    io: std.Io,
    /// Held across a whole call, including the wait for the GPU, so a
    /// contending thread must sleep rather than spin.
    lock: std.Io.Mutex = .init,
    name: []u8,
    ctx: Context,
    residency: compute.Residency,
    /// Reused kernel buffers for inputs, outputs and small tables.
    scratch: [5]Buf = [_]Buf{.{}} ** 5,
    /// Weight tiles that are not resident are staged here.
    weight_scratch: Buf = .{},

    fn run(self: *Backend, k: Kernel, binds: []const Bind, params: []const u8, groups: [3]u32, reads: []const Context.Read) Error!void {
        const pipe = try self.ctx.pipeline(@intFromEnum(k), k.spirv());
        return self.ctx.run(pipe, binds, params, groups, reads);
    }

    /// A scratch kernel buffer of at least `bytes`.
    fn slot(self: *Backend, i: usize, bytes: u64) Error!*Buf {
        try self.ctx.fits(bytes);
        try self.ctx.ensure(&self.scratch[i], @max(bytes, 4), .kernel);
        return &self.scratch[i];
    }

    /// The device buffer for a weight tile: resident (`--gpu-memory` with
    /// memory-mapped weights), or the reused weight scratch buffer.
    const WeightBuf = struct {
        bind: Bind,
        /// A new buffer to insert into the residency cache once the upload
        /// has run, or to free if the call fails.
        pending: ?*Buf = null,
    };

    fn weightBuffer(self: *Backend, data: []const u8) Error!WeightBuf {
        try self.ctx.fits(data.len);
        if (self.residency.enabled() and data.len <= self.residency.budget) {
            if (self.residency.get(data)) |h| {
                const b: *Buf = @ptrCast(@alignCast(h));
                return .{ .bind = .{ .buf = b, .bytes = data.len } };
            }
            const b = try self.gpa.create(Buf);
            errdefer self.gpa.destroy(b);
            b.* = try self.ctx.newBuf(std.mem.alignForward(u64, @max(data.len, 4), 4), .kernel);
            return .{ .bind = .{ .buf = b, .bytes = data.len, .upload = data }, .pending = b };
        }
        try self.ctx.ensure(&self.weight_scratch, @max(data.len, 4), .kernel);
        return .{ .bind = .{ .buf = &self.weight_scratch, .bytes = data.len, .upload = data } };
    }

    /// After the call: cache a freshly uploaded tile, or free it.
    fn settleWeight(self: *Backend, wb: WeightBuf, ok: bool) void {
        const b = wb.pending orelse return;
        if (ok and self.residency.put(wb.bind.upload.?, b.size, b)) return;
        self.ctx.freeBuf(b);
        self.gpa.destroy(b);
    }
};

fn ceilDiv(a: usize, b: usize) Error!u32 {
    return std.math.cast(u32, (a + b - 1) / b) orelse error.Unsupported;
}

fn u32Of(v: usize) Error!u32 {
    return std.math.cast(u32, v) orelse error.Unsupported;
}

fn bytesOf(s: anytype) []const u8 {
    return std.mem.sliceAsBytes(s);
}

fn paramBytes(p: anytype) []const u8 {
    return std.mem.asBytes(p);
}

// ---------------------------------------------------------------------------
// Operations
// ---------------------------------------------------------------------------

fn matmulT(ctx: *anyopaque, out: []f32, x: []const f32, n: usize, w: Weight) Error!void {
    const self: *Backend = @ptrCast(@alignCast(ctx));
    if (w.cols == 0 or w.rows == 0 or n == 0) return error.Unsupported;
    const k = try Kernel.forDtype("matmul", w.dtype);
    const p = MatmulParams{ .n = try u32Of(n), .rows = try u32Of(w.rows), .cols = try u32Of(w.cols) };
    const groups = [3]u32{ try ceilDiv(w.rows, tile), try ceilDiv(n, tile), 1 };
    self.lock.lockUncancelable(self.io);
    defer self.lock.unlock(self.io);
    const xs = x[0 .. n * w.cols];
    const os = out[0 .. n * w.rows];
    const xb = try self.slot(0, xs.len * 4);
    const ob = try self.slot(1, os.len * 4);
    const wb = try self.weightBuffer(w.data[0 .. w.rows * w.dtype.rowBytes(w.cols)]);
    var ok = false;
    defer self.settleWeight(wb, ok);
    try self.run(k, &.{
        .{ .buf = ob, .bytes = os.len * 4 },
        .{ .buf = xb, .bytes = xs.len * 4, .upload = bytesOf(xs) },
        wb.bind,
    }, paramBytes(&p), groups, &.{.{ .buf = ob, .dst = std.mem.sliceAsBytes(os) }});
    ok = true;
}

fn matvecTMulti(ctx: *anyopaque, out: []f32, w: Weight, y: []const f32, q: usize) Error!void {
    const self: *Backend = @ptrCast(@alignCast(ctx));
    if (w.cols == 0 or w.rows == 0 or q == 0) return error.Unsupported;
    const k = try Kernel.forDtype("matvec", w.dtype);
    const p = MatvecParams{ .q = try u32Of(q), .rows = try u32Of(w.rows), .cols = try u32Of(w.cols) };
    const groups = [3]u32{ try ceilDiv(w.cols, flat_threads), try u32Of(q), 1 };
    self.lock.lockUncancelable(self.io);
    defer self.lock.unlock(self.io);
    const ys = y[0 .. q * w.rows];
    const os = out[0 .. q * w.cols];
    const yb = try self.slot(0, ys.len * 4);
    const ob = try self.slot(1, os.len * 4);
    const wb = try self.weightBuffer(w.data[0 .. w.rows * w.dtype.rowBytes(w.cols)]);
    var ok = false;
    defer self.settleWeight(wb, ok);
    try self.run(k, &.{
        .{ .buf = ob, .bytes = os.len * 4 },
        .{ .buf = yb, .bytes = ys.len * 4, .upload = bytesOf(ys) },
        wb.bind,
    }, paramBytes(&p), groups, &.{.{ .buf = ob, .dst = std.mem.sliceAsBytes(os) }});
    ok = true;
}

fn rowNorms(ctx: *anyopaque, out: []f32, w: Weight) Error!void {
    const self: *Backend = @ptrCast(@alignCast(ctx));
    if (w.cols == 0 or w.rows == 0) return error.Unsupported;
    const k = try Kernel.forDtype("row_norms", w.dtype);
    const p = RowNormParams{ .rows = try u32Of(w.rows), .cols = try u32Of(w.cols) };
    self.lock.lockUncancelable(self.io);
    defer self.lock.unlock(self.io);
    const groups = try self.ctx.rowGrid(w.rows);
    const os = out[0..w.rows];
    const ob = try self.slot(1, os.len * 4);
    const wb = try self.weightBuffer(w.data[0 .. w.rows * w.dtype.rowBytes(w.cols)]);
    var ok = false;
    defer self.settleWeight(wb, ok);
    try self.run(k, &.{ .{ .buf = ob, .bytes = os.len * 4 }, wb.bind }, paramBytes(&p), groups, &.{.{ .buf = ob, .dst = std.mem.sliceAsBytes(os) }});
    ok = true;
}

fn attentionScores(ctx: *anyopaque, scores: []f32, q: []const f32, k: []const f32, stride: usize, scale: f32) Error!void {
    const self: *Backend = @ptrCast(@alignCast(ctx));
    if (scores.len == 0 or q.len == 0) return error.Unsupported;
    const p = AttnScoreParams{ .keys = try u32Of(scores.len), .hd = try u32Of(q.len), .stride = try u32Of(stride), .scale = scale };
    self.lock.lockUncancelable(self.io);
    defer self.lock.unlock(self.io);
    const qb = try self.slot(0, q.len * 4);
    const kb = try self.slot(2, k.len * 4);
    const ob = try self.slot(1, scores.len * 4);
    try self.run(.attn_scores, &.{
        .{ .buf = ob, .bytes = scores.len * 4 },
        .{ .buf = qb, .bytes = q.len * 4, .upload = bytesOf(q) },
        .{ .buf = kb, .bytes = k.len * 4, .upload = bytesOf(k) },
    }, paramBytes(&p), .{ try ceilDiv(scores.len, flat_threads), 1, 1 }, &.{.{ .buf = ob, .dst = std.mem.sliceAsBytes(scores) }});
}

fn attentionValues(ctx: *anyopaque, out: []f32, scores: []const f32, v: []const f32, stride: usize) Error!void {
    const self: *Backend = @ptrCast(@alignCast(ctx));
    if (out.len == 0 or scores.len == 0) return error.Unsupported;
    const p = AttnValueParams{ .keys = try u32Of(scores.len), .vd = try u32Of(out.len), .stride = try u32Of(stride) };
    self.lock.lockUncancelable(self.io);
    defer self.lock.unlock(self.io);
    const sb = try self.slot(0, scores.len * 4);
    const vb = try self.slot(2, v.len * 4);
    const ob = try self.slot(1, out.len * 4);
    try self.run(.attn_values, &.{
        .{ .buf = ob, .bytes = out.len * 4 },
        .{ .buf = sb, .bytes = scores.len * 4, .upload = bytesOf(scores) },
        .{ .buf = vb, .bytes = v.len * 4, .upload = bytesOf(v) },
    }, paramBytes(&p), .{ try ceilDiv(out.len, flat_threads), 1, 1 }, &.{.{ .buf = ob, .dst = std.mem.sliceAsBytes(out) }});
}

fn gatedActivation(ctx: *anyopaque, act: compute.ActivationCode, out: []f32, gate: []const f32, up: ?[]const f32, n: usize, len: usize, in_stride: usize, out_stride: usize) Error!void {
    const self: *Backend = @ptrCast(@alignCast(ctx));
    if (n == 0 or len == 0) return error.Unsupported;
    const p = GatedParams{
        .act = act,
        .n = try u32Of(n),
        .len = try u32Of(len),
        .in_stride = try u32Of(in_stride),
        .out_stride = try u32Of(out_stride),
        .has_up = if (up == null) 0 else 1,
    };
    self.lock.lockUncancelable(self.io);
    defer self.lock.unlock(self.io);
    const gs = gate[0 .. (n - 1) * in_stride + len];
    const os = out[0 .. (n - 1) * out_stride + len];
    const gb = try self.slot(0, gs.len * 4);
    const ob = try self.slot(1, os.len * 4);
    // Without `up` the kernel never reads binding 2; bind the gate again.
    const ub: Bind = if (up) |u| blk: {
        const us = u[0 .. (n - 1) * in_stride + len];
        break :blk .{ .buf = try self.slot(2, us.len * 4), .bytes = us.len * 4, .upload = bytesOf(us) };
    } else .{ .buf = gb, .bytes = gs.len * 4 };
    try self.run(.gated, &.{
        .{ .buf = ob, .bytes = os.len * 4 },
        .{ .buf = gb, .bytes = gs.len * 4, .upload = bytesOf(gs) },
        ub,
    }, paramBytes(&p), .{ try ceilDiv(len, flat_threads), try u32Of(n), 1 }, &.{.{ .buf = ob, .dst = std.mem.sliceAsBytes(os) }});
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
    const p = NormParams{ .n = try u32Of(n), .len = try u32Of(len), .eps = eps, .flags = normFlags(weight, gemma_style, null) };
    self.lock.lockUncancelable(self.io);
    defer self.lock.unlock(self.io);
    const groups = try self.ctx.rowGrid(n);
    const xs = x[0 .. n * len];
    const os = out[0 .. n * len];
    const xb = try self.slot(0, xs.len * 4);
    const wb = try self.slot(2, weight.len * 4);
    const ob = try self.slot(1, os.len * 4);
    try self.run(.rmsnorm_rows, &.{
        .{ .buf = ob, .bytes = os.len * 4 },
        .{ .buf = xb, .bytes = xs.len * 4, .upload = bytesOf(xs) },
        .{ .buf = wb, .bytes = weight.len * 4, .upload = bytesOf(weight) },
    }, paramBytes(&p), groups, &.{.{ .buf = ob, .dst = std.mem.sliceAsBytes(os) }});
}

fn layernormRows(ctx: *anyopaque, out: []f32, x: []const f32, weight: []const f32, bias: ?[]const f32, n: usize, len: usize, eps: f32, one_plus: bool) Error!void {
    const self: *Backend = @ptrCast(@alignCast(ctx));
    if (n == 0 or len == 0) return error.Unsupported;
    const p = NormParams{ .n = try u32Of(n), .len = try u32Of(len), .eps = eps, .flags = normFlags(weight, one_plus, bias) };
    self.lock.lockUncancelable(self.io);
    defer self.lock.unlock(self.io);
    const groups = try self.ctx.rowGrid(n);
    const xs = x[0 .. n * len];
    const os = out[0 .. n * len];
    const xb = try self.slot(0, xs.len * 4);
    const wb = try self.slot(2, weight.len * 4);
    const ob = try self.slot(1, os.len * 4);
    const bb: Bind = if (bias) |b|
        .{ .buf = try self.slot(3, b.len * 4), .bytes = b.len * 4, .upload = bytesOf(b) }
    else
        .{ .buf = wb, .bytes = weight.len * 4 };
    try self.run(.layernorm_rows, &.{
        .{ .buf = ob, .bytes = os.len * 4 },
        .{ .buf = xb, .bytes = xs.len * 4, .upload = bytesOf(xs) },
        .{ .buf = wb, .bytes = weight.len * 4, .upload = bytesOf(weight) },
        bb,
    }, paramBytes(&p), groups, &.{.{ .buf = ob, .dst = std.mem.sliceAsBytes(os) }});
}

fn softmaxRows(ctx: *anyopaque, x: []f32, n: usize, len: usize) Error!void {
    const self: *Backend = @ptrCast(@alignCast(ctx));
    if (n == 0 or len == 0) return error.Unsupported;
    const p = SoftmaxParams{ .n = try u32Of(n), .len = try u32Of(len) };
    self.lock.lockUncancelable(self.io);
    defer self.lock.unlock(self.io);
    const groups = try self.ctx.rowGrid(n);
    const xs = x[0 .. n * len];
    const xb = try self.slot(0, xs.len * 4);
    try self.run(.softmax_rows, &.{
        .{ .buf = xb, .bytes = xs.len * 4, .upload = bytesOf(xs) },
    }, paramBytes(&p), groups, &.{.{ .buf = xb, .dst = std.mem.sliceAsBytes(xs) }});
}

fn ropeRows(ctx: *anyopaque, x: []f32, n: usize, dim: usize, cos: []const f32, sin: []const f32, half: usize, pos: []const u32, style: compute.RopeStyle) Error!void {
    const self: *Backend = @ptrCast(@alignCast(ctx));
    if (n == 0 or half == 0) return error.Unsupported;
    const p = RopeParams{ .n = try u32Of(n), .dim = try u32Of(dim), .half_dim = try u32Of(half), .style = @intFromEnum(style) };
    self.lock.lockUncancelable(self.io);
    defer self.lock.unlock(self.io);
    const xs = x[0 .. n * dim];
    const ps = pos[0..n];
    const xb = try self.slot(0, xs.len * 4);
    const cb = try self.slot(1, cos.len * 4);
    const sb = try self.slot(2, sin.len * 4);
    const pb = try self.slot(3, ps.len * 4);
    try self.run(.rope_rows, &.{
        .{ .buf = xb, .bytes = xs.len * 4, .upload = bytesOf(xs) },
        .{ .buf = cb, .bytes = cos.len * 4, .upload = bytesOf(cos) },
        .{ .buf = sb, .bytes = sin.len * 4, .upload = bytesOf(sin) },
        .{ .buf = pb, .bytes = ps.len * 4, .upload = bytesOf(ps) },
    }, paramBytes(&p), .{ try ceilDiv(half, flat_threads), try u32Of(n), 1 }, &.{.{ .buf = xb, .dst = std.mem.sliceAsBytes(xs) }});
}

// ---------------------------------------------------------------------------
// Lifetime
// ---------------------------------------------------------------------------

fn deinit(ctx: *anyopaque) void {
    const self: *Backend = @ptrCast(@alignCast(ctx));
    _ = self.ctx.dfn.vkDeviceWaitIdle(self.ctx.device);
    self.residency.deinit();
    for (&self.scratch) |*b| self.ctx.freeBuf(b);
    self.ctx.freeBuf(&self.weight_scratch);
    self.ctx.deinit();
    const gpa = self.gpa;
    gpa.free(self.name);
    gpa.destroy(self);
}

const vtable = compute.VTable{
    .deinit = deinit,
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

fn releaseBuffer(ctx: *anyopaque, handle: *anyopaque) void {
    const self: *Backend = @ptrCast(@alignCast(ctx));
    const b: *Buf = @ptrCast(@alignCast(handle));
    self.ctx.freeBuf(b);
    self.gpa.destroy(b);
}

// ---------------------------------------------------------------------------
// Opening
// ---------------------------------------------------------------------------

pub const OpenOptions = struct {
    /// Device memory the residency cache may hold (0 = upload, compute, drop).
    memory_budget: u64 = 0,
    /// Accept a software (CPU-type) Vulkan device; see `device.OpenOptions`.
    allow_cpu: bool = false,
};

/// Why the last `open` failed ("" when it did not).
pub const lastError = device.lastError;

/// Loads Vulkan, picks a device and builds the backend. On failure,
/// `lastError()` says why in one line.
pub fn open(gpa: Allocator, io: std.Io, opts: OpenOptions) !compute.Device {
    var ctx = try device.open(gpa, .{ .allow_cpu = opts.allow_cpu });
    errdefer ctx.deinit();
    // Build one pipeline now, so a driver that cannot compile the kernels is
    // refused here (and `auto` falls back to the CPU) rather than on the
    // first matmul.
    _ = ctx.pipeline(@intFromEnum(Kernel.matmul_f32), Kernel.matmul_f32.spirv()) catch
        return device.fail("the Vulkan driver could not build the compute pipelines", .{});

    const self = try gpa.create(Backend);
    errdefer gpa.destroy(self);
    const name = try std.fmt.allocPrint(gpa, "{s} (Vulkan, {s})", .{ device.cstr(&ctx.props.deviceName), device.typeName(ctx.props.deviceType) });
    errdefer gpa.free(name);
    // The cache is capped at three quarters of the largest device-local heap.
    const heap = device.deviceLocalBytes(&ctx.mem);
    const budget = if (heap > 0) @min(opts.memory_budget, heap / 4 * 3) else opts.memory_budget;
    self.* = .{
        .gpa = gpa,
        .io = io,
        .name = name,
        .ctx = ctx,
        .residency = compute.Residency.init(gpa, budget, self, releaseBuffer),
    };
    return .{
        .kind = .vulkan,
        .name = name,
        .vtable = &vtable,
        .ctx = self,
        .memory_budget = budget,
    };
}

/// One line about the device for reports (see `device.describe`).
pub fn describe(dev: *const compute.Device, w: *std.Io.Writer) !void {
    const self: *Backend = @ptrCast(@alignCast(dev.ctx orelse return));
    return device.describe(&self.ctx, w);
}

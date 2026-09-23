//! Mixture-of-experts layers: loading (separate or fused expert tensors),
//! routed forward pass, per-expert abliteration deltas, expert ranking and
//! expert-selective abliteration.
//!
//! Tensor names and routing rules come from the architecture registry
//! (`arch.zig`): Qwen2/3-MoE, Mixtral, DeepSeek V2/V3 (sigmoid or softmax
//! scoring, group-limited top-k, correction bias, shared experts), Llama 4
//! (top-1 routing scaling the expert input) and gpt-oss (interleaved fused
//! experts with biases, clamped swiglu). Every expert's down projection is
//! exposed as a `tensor.Weight` view regardless of the on-disk layout, so the
//! abliteration kernels address a target by (layer, expert index) only.
//!
//! Every expert matrix also carries a `MatrixRef` describing where it lives on
//! disk (a whole tensor, a slice of a stacked expert tensor, or a transposed
//! column range of such a slice). In mapped mode the `Weight` views are always
//! valid; in streamed mode they carry only the shape and the matrices are made
//! resident per layer through the model's `WeightStore` (`acquireLayer`), or
//! one at a time for abliteration (`MoeLayer.acquireDown`). With an expert
//! cache ("warp mode", see expert_cache.zig) only the trunk of a layer is
//! acquired with the layer; the routed experts a token batch selects are
//! made resident on demand through the cache, from `forward` and from the
//! abliteration entry points alike, so a trial never reads an expert twice.

const std = @import("std");
const Io = std.Io;
const tensor = @import("tensor.zig");
const compute = @import("compute.zig");
const model_mod = @import("model.zig");
const stream = @import("stream.zig");
const abliterate = @import("abliterate.zig");
const search = @import("search.zig");
const arch = @import("arch.zig");
const expert_cache = @import("expert_cache.zig");

const Allocator = std.mem.Allocator;
const Weight = tensor.Weight;
const Delta = tensor.Delta;
const Model = model_mod.Model;
const WeightRef = stream.WeightRef;
const Lease = stream.Lease;

/// Where an expert matrix lives on disk and how to make it resident.
pub const MatrixRef = struct {
    /// The on-disk block (`[rows][cols]`).
    ref: WeightRef,
    /// null: the block is the matrix. Otherwise the matrix is columns `[lo, hi)`
    /// of the block (every `stride`-th one), transposed (`[(hi - lo) / stride][block rows]`).
    transposed: ?[2]usize = null,
    stride: usize = 1,

    pub fn rows(self: MatrixRef) usize {
        return if (self.transposed) |t| (t[1] - t[0] + self.stride - 1) / self.stride else self.ref.rows;
    }

    pub fn cols(self: MatrixRef) usize {
        return if (self.transposed != null) self.ref.rows else self.ref.cols;
    }

    /// Bytes of the resident matrix.
    pub fn residentBytes(self: MatrixRef) u64 {
        return @as(u64, self.rows()) * self.ref.dtype.rowBytes(self.cols());
    }

    /// Placeholder view carrying the shape only.
    pub fn shapeOnly(self: MatrixRef) Weight {
        return .{ .data = &.{}, .dtype = self.ref.dtype, .rows = self.rows(), .cols = self.cols() };
    }

    pub fn columns(self: MatrixRef) stream.ColumnSpec {
        const t = self.transposed.?;
        return .{ .lo = t[0], .hi = t[1], .stride = self.stride };
    }

    /// Makes the matrix resident (streamed mode); release with `store.release`.
    pub fn acquire(self: MatrixRef, store: *stream.WeightStore) !Lease {
        if (self.transposed == null) return store.acquire(self.ref);
        var out: [1]Lease = undefined;
        try store.acquireColumns(self.ref, &.{self.columns()}, &out);
        return out[0];
    }
};

/// On-disk layout of the routed experts.
pub const Layout = enum {
    /// One tensor per expert and projection (`experts.{i}.down_proj.weight`).
    separate,
    /// `experts.gate_up_proj` [E, 2I, hidden] and `experts.down_proj` [E, hidden, I];
    /// each expert is a contiguous [rows][cols] block.
    fused,
    /// `experts.gate_up_proj` [E, hidden, 2I] and `experts.down_proj` [E, I, hidden];
    /// expert blocks are transposed on load into arena-owned copies.
    fused_transposed,
};

pub const Expert = struct {
    gate: Weight, // [I, hidden]
    up: Weight, // [I, hidden]
    down: Weight, // [hidden, I]
    /// Abliteration delta on the down projection (null = identity).
    down_delta: ?Delta = null,
    /// On-disk locations of the three matrices.
    gate_ref: MatrixRef,
    up_ref: MatrixRef,
    down_ref: MatrixRef,
    /// Projection biases (gpt-oss).
    gate_bias: ?[]const f32 = null,
    up_bias: ?[]const f32 = null,
    down_bias: ?[]const f32 = null,
    /// False for a plain `down(act(up(x)))` expert (Nemotron-H): `gate` /
    /// `gate_ref` then alias `up` and there is no second input matrix.
    gated: bool = true,

    /// Fused, transposed layout: gate and up are column ranges of one block
    /// and come out of a single read.
    pub fn sharesGateUpBlock(self: *const Expert) bool {
        return self.gated and self.gate_ref.transposed != null and self.up_ref.transposed != null and self.gate_ref.ref.offset == self.up_ref.ref.offset and self.gate_ref.ref.file == self.up_ref.ref.file;
    }

    /// Matrices to make resident: two for a non-gated expert, three otherwise.
    pub fn matrixCount(self: *const Expert) usize {
        return if (self.gated) 3 else 2;
    }
};

/// Bytes resident for one routed expert (its two or three matrices).
pub fn expertBytes(ex: *const Expert) u64 {
    return ex.gate_ref.residentBytes() + (if (ex.gated) ex.up_ref.residentBytes() else 0) + ex.down_ref.residentBytes();
}

/// Transient bytes needed while an expert is read (the block buffer of a
/// transposed fused slice; 0 for the other layouts).
pub fn expertTransientBytes(ex: *const Expert) u64 {
    var t: u64 = 0;
    for ([_]MatrixRef{ ex.gate_ref, ex.up_ref, ex.down_ref }) |r| {
        if (r.transposed != null) t = @max(t, r.ref.byteLen());
    }
    return t;
}

/// Always-active expert (Qwen2-MoE, DeepSeek, Llama 4). Its output is scaled
/// by `sigmoid(x · gate_vec)` when a gate vector exists.
pub const SharedExpert = struct {
    gate: Weight,
    up: Weight,
    down: Weight,
    gate_vec: ?[]f32,
    down_delta: ?Delta = null,
    gate_ref: MatrixRef,
    up_ref: MatrixRef,
    down_ref: MatrixRef,
    /// As `Expert.gated`.
    gated: bool = true,

    pub fn matrixCount(self: *const SharedExpert) usize {
        return if (self.gated) 3 else 2;
    }
};

/// Latent MoE (Kimi K3 `routed_expert_hidden_size`): the routed experts read
/// `down(x)` and their weighted sum, RMS-normalised when `norm` is set, is
/// written back into the residual through `up`. `up` is therefore the routed
/// experts' write into the residual stream and the matrix abliteration edits
/// on such a layer (the per-expert `w2` matrices live in latent space, where
/// a residual direction has no meaning); `down` and `norm` are never edited.
pub const Latent = struct {
    width: usize,
    /// `[width][hidden]`.
    down: Weight,
    /// `[hidden][width]`.
    up: Weight,
    norm: ?[]const f32,
    down_ref: MatrixRef,
    up_ref: MatrixRef,
    up_delta: ?Delta = null,
};

pub const MoeLayer = struct {
    /// Index of the layer this block belongs to (the expert cache key).
    layer_index: usize,
    /// Router `[E, hidden]`.
    router: Weight,
    router_ref: WeightRef,
    router_bias: ?[]const f32,
    /// Selection bias added to the scores when choosing experts (DeepSeek V3 `e_score_correction_bias`).
    correction_bias: ?[]const f32,
    top_k: usize,
    norm_topk_prob: bool,
    routing: arch.MoeConfig,
    activation: tensor.Activation,
    /// Expert intermediate size.
    inter: usize,
    layout: Layout,
    experts: []Expert,
    shared: ?SharedExpert,
    /// Latent MoE projections (null: the experts read and write `hidden`).
    latent: ?Latent = null,
    /// Tensor-name suffix (after `layers.{i}.`) of the fused down tensor, for export.
    fused_down_suffix: ?[]const u8,
    /// Name templates of the family (for export lookups).
    names: *const arch.Names,
    /// Hash routing (DeepSeek V4 `hash_moe`): expert `[token][k]` is fixed by
    /// the input token id; the router scores only weight the selection.
    hash_table: ?[]const u32 = null,

    /// Number of editable down projections: routed experts plus the shared expert.
    pub fn numDown(self: *const MoeLayer) usize {
        return self.experts.len + @as(usize, if (self.shared != null) 1 else 0);
    }

    /// Width of the routed experts' input and output (`hidden`, or the latent width).
    pub fn expertWidth(self: *const MoeLayer, hidden: usize) usize {
        return if (self.latent) |l| l.width else hidden;
    }

    /// True when the routed experts' down projections cannot carry a residual
    /// edit (latent MoE); `latent.up` is edited in their place.
    pub fn routedInLatent(self: *const MoeLayer) bool {
        return self.latent != null;
    }

    /// Makes the latent up projection resident; release with `DownLease.release`.
    pub fn acquireLatentUp(self: *const MoeLayer, model: *const Model) !DownLease {
        const lat = self.latent.?;
        if (!model.streamed()) return .{ .weight = lat.up };
        const lease = try lat.up_ref.acquire(@constCast(&model.store));
        return .{ .weight = lease.weight, .lease = lease };
    }

    /// Index used for the shared expert in `downWeight`/`downDelta` (== experts.len).
    pub fn sharedIndex(self: *const MoeLayer) ?usize {
        return if (self.shared != null) self.experts.len else null;
    }

    pub fn downWeight(self: *const MoeLayer, idx: usize) Weight {
        if (idx < self.experts.len) return self.experts[idx].down;
        return self.shared.?.down;
    }

    pub fn downRef(self: *const MoeLayer, idx: usize) MatrixRef {
        if (idx < self.experts.len) return self.experts[idx].down_ref;
        return self.shared.?.down_ref;
    }

    /// Makes down projection `idx` resident: a view in mapped mode, a budgeted
    /// buffer in streamed mode, or a pinned expert-cache entry in warp mode
    /// (all three matrices of the expert stay resident for later use).
    /// Release with `DownLease.release`.
    pub fn acquireDown(self: *const MoeLayer, model: *const Model, idx: usize) !DownLease {
        if (!model.streamed()) return .{ .weight = self.downWeight(idx) };
        if (model.expert_cache) |cache| {
            if (idx < self.experts.len) {
                const e = try cache.acquire(self.layer_index, self, idx);
                return .{ .weight = e.down, .entry = e };
            }
        }
        const lease = try self.downRef(idx).acquire(@constCast(&model.store));
        return .{ .weight = lease.weight, .lease = lease };
    }

    /// Bytes of the layer's trunk part in streamed mode: the shared expert (if
    /// any) and the latent projections.
    pub fn trunkBytes(self: *const MoeLayer) u64 {
        var total: u64 = 0;
        if (self.shared) |sh| {
            total += sh.gate_ref.residentBytes() + sh.down_ref.residentBytes();
            if (sh.gated) total += sh.up_ref.residentBytes();
        }
        if (self.latent) |l| total += l.down_ref.residentBytes() + l.up_ref.residentBytes();
        return total;
    }

    /// Bytes of the largest routed expert (0 without experts).
    pub fn maxExpertBytes(self: *const MoeLayer) u64 {
        var max: u64 = 0;
        for (self.experts) |*ex| max = @max(max, expertBytes(ex));
        return max;
    }

    /// Bytes resident while this layer computes in streamed mode: the shared
    /// expert plus either every routed expert (`warp == false`) or the
    /// experts one token selects (`warp == true`, the expert cache holds
    /// them), plus the transient block buffer used to transpose fused slices.
    pub fn residentBytes(self: *const MoeLayer, warp: bool) u64 {
        var total: u64 = self.trunkBytes();
        var transient: u64 = 0;
        for (self.experts) |*ex| {
            if (!warp) total += expertBytes(ex);
            transient = @max(transient, expertTransientBytes(ex));
        }
        if (warp) total += @as(u64, self.top_k) * self.maxExpertBytes();
        return total + transient;
    }

    pub fn downDelta(self: *MoeLayer, idx: usize) *?Delta {
        if (idx < self.experts.len) return &self.experts[idx].down_delta;
        return &self.shared.?.down_delta;
    }

    pub fn getDownDelta(self: *const MoeLayer, idx: usize) ?Delta {
        if (idx < self.experts.len) return self.experts[idx].down_delta;
        return self.shared.?.down_delta;
    }

    pub fn resetDeltas(self: *MoeLayer, gpa: Allocator) void {
        var i: usize = 0;
        while (i < self.numDown()) : (i += 1) {
            const slot = self.downDelta(i);
            if (slot.*) |d| {
                gpa.free(d.a);
                gpa.free(d.b);
                slot.* = null;
            }
        }
        if (self.latent) |*l| if (l.up_delta) |d| {
            gpa.free(d.a);
            gpa.free(d.b);
            l.up_delta = null;
        };
    }

    /// Which down projection a tensor-name suffix (after `layers.{i}.`) refers to.
    pub const ExportTarget = union(enum) {
        /// Index into `downWeight`/`getDownDelta` (routed expert or shared expert).
        expert: usize,
        /// The fused down tensor holding every routed expert.
        fused_down,
        /// The latent up projection (`Latent.up`).
        latent_up,
    };

    pub fn exportTarget(self: *const MoeLayer, suffix: []const u8) ?ExportTarget {
        if (self.fused_down_suffix) |fd| {
            if (std.mem.eql(u8, suffix, fd)) return .fused_down;
        }
        const names = self.names;
        if (self.latent != null) {
            if (names.latent_up) |lu| if (std.mem.eql(u8, suffix, lu)) return .latent_up;
        }
        if (self.shared != null) {
            if (names.shared_expert) |sp| {
                if (std.mem.startsWith(u8, suffix, sp) and std.mem.eql(u8, suffix[sp.len..], names.shared_down orelse names.expert_down)) return .{ .expert = self.experts.len };
            }
        }
        // `mlp.experts.{e}.` → prefix before `{e}` and the text after it.
        const marker = std.mem.indexOf(u8, names.expert, "{e}") orelse return null;
        const head = names.expert[0..marker];
        const mid = names.expert[marker + 3 ..];
        if (!std.mem.startsWith(u8, suffix, head)) return null;
        const rest = suffix[head.len..];
        const end = std.mem.indexOf(u8, rest, mid) orelse return null;
        if (end == 0) return null;
        const idx = std.fmt.parseInt(usize, rest[0..end], 10) catch return null;
        if (idx >= self.experts.len) return null;
        const tail = rest[end + mid.len ..];
        if (std.mem.eql(u8, tail, names.expert_down)) return .{ .expert = idx };
        return null;
    }

    /// True if any routed expert of this layer carries a delta.
    pub fn anyExpertDelta(self: *const MoeLayer) bool {
        for (self.experts) |ex| if (ex.down_delta != null) return true;
        return false;
    }

    /// Merges routed expert `e`'s delta (if any) into `block`, the f32 copy of
    /// that expert's slice of the fused down tensor (`[hidden][I]` for the
    /// `fused` layout, `[I][hidden]` for `fused_transposed`).
    pub fn mergeExpertDown(self: *const MoeLayer, e: usize, block: []f32) void {
        const hidden = self.experts[0].down.rows;
        const inter = self.inter;
        const d = self.experts[e].down_delta orelse return;
        std.debug.assert(block.len == hidden * inter);
        switch (self.layout) {
            .fused => for (0..hidden) |i| {
                const row = block[i * inter ..][0..inter];
                for (0..d.rank) |k| tensor.axpy(row, d.b[i * d.rank + k], d.a[k * inter ..][0..inter]);
            },
            .fused_transposed => for (0..inter) |j| {
                const row = block[j * hidden ..][0..hidden];
                for (0..hidden) |i| {
                    var acc: f32 = 0;
                    for (0..d.rank) |k| acc += d.b[i * d.rank + k] * d.a[k * inter + j];
                    row[i] += acc;
                }
            },
            .separate => unreachable,
        }
    }
};

/// A down projection made resident by `MoeLayer.acquireDown`.
pub const DownLease = struct {
    weight: Weight,
    /// Streamed mode without an expert cache: the owned buffer.
    lease: Lease = .{ .weight = .{ .data = &.{}, .dtype = .f32, .rows = 0, .cols = 0 } },
    /// Warp mode: the pinned cache entry.
    entry: ?*expert_cache.Entry = null,

    pub fn release(self: DownLease, model: *const Model) void {
        if (self.entry) |e| {
            model.expert_cache.?.release(e);
        } else {
            @constCast(&model.store).release(self.lease);
        }
    }
};

/// An MoE layer whose expert matrices are resident (streamed mode). `layer` is
/// a copy of the model's `MoeLayer` whose experts point at `experts`. In warp
/// mode only the shared expert is acquired here; the routed experts stay
/// shape-only and `forward` fetches them through the model's expert cache.
pub const MoeLease = struct {
    layer: MoeLayer,
    gpa: Allocator,
    experts: []Expert,
    leases: []Lease,

    pub fn release(self: *MoeLease, store: *stream.WeightStore) void {
        store.releaseSet(self.leases);
        if (self.leases.len > 0) self.gpa.free(self.leases);
        if (self.experts.len > 0) self.gpa.free(self.experts);
    }
};

fn acquireInto(store: *stream.WeightStore, r: MatrixRef, leases: []Lease, n: *usize) !Weight {
    leases[n.*] = try r.acquire(store);
    n.* += 1;
    return leases[n.* - 1].weight;
}

/// Number of leases the trunk of `m` takes: the shared expert's matrices
/// (two when it has no gate) and the two latent projections.
fn trunkLeaseCount(m: *const MoeLayer) usize {
    return (if (m.shared) |sh| sh.matrixCount() else 0) + @as(usize, if (m.latent != null) 2 else 0);
}

/// Acquires the trunk matrices of `m` (shared expert, latent projections) into `lease.layer`.
fn acquireTrunk(store: *stream.WeightStore, m: *const MoeLayer, lease: *MoeLease, leases: []Lease, n: *usize) !void {
    if (m.shared) |sh| {
        var s = sh;
        s.gate = try acquireInto(store, sh.gate_ref, leases, n);
        s.up = if (sh.gated) try acquireInto(store, sh.up_ref, leases, n) else s.gate;
        s.down = try acquireInto(store, sh.down_ref, leases, n);
        lease.layer.shared = s;
    }
    if (m.latent) |lat| {
        var l = lat;
        l.down = try acquireInto(store, lat.down_ref, leases, n);
        l.up = try acquireInto(store, lat.up_ref, leases, n);
        lease.layer.latent = l;
    }
}

/// Makes every expert matrix of `m` resident through the model's store (only
/// the trunk, i.e. the shared expert and the latent projections, in warp
/// mode). In mapped mode the views are already valid and nothing is allocated.
pub fn acquireLayer(model: *const Model, m: *const MoeLayer) !MoeLease {
    const gpa = model.gpa;
    var lease = MoeLease{ .layer = m.*, .gpa = gpa, .experts = &.{}, .leases = &.{} };
    if (!model.streamed()) return lease;
    const store: *stream.WeightStore = @constCast(&model.store);
    if (model.expert_cache != null) {
        const count = trunkLeaseCount(m);
        if (count == 0) return lease;
        const trunk_leases = try gpa.alloc(Lease, count);
        errdefer gpa.free(trunk_leases);
        var n: usize = 0;
        errdefer store.releaseSet(trunk_leases[0..n]);
        try acquireTrunk(store, m, &lease, trunk_leases, &n);
        lease.leases = trunk_leases;
        return lease;
    }
    const experts = try gpa.alloc(Expert, m.experts.len);
    errdefer gpa.free(experts);
    // Three matrices per routed expert (gate and up of a transposed fused block
    // come out of one read but are still two leases; a non-gated expert has
    // two matrices) plus the trunk.
    var n_leases: usize = trunkLeaseCount(m);
    for (m.experts) |*ex| n_leases += ex.matrixCount();
    const all_leases = try gpa.alloc(Lease, n_leases);
    errdefer gpa.free(all_leases);
    var n: usize = 0;
    errdefer store.releaseSet(all_leases[0..n]);
    for (m.experts, 0..) |ex, e| {
        experts[e] = ex;
        if (ex.sharesGateUpBlock()) {
            // Fused, transposed layout: gate and up share one block; read it once.
            try store.acquireColumns(ex.gate_ref.ref, &.{ ex.gate_ref.columns(), ex.up_ref.columns() }, all_leases[n..][0..2]);
            experts[e].gate = all_leases[n].weight;
            experts[e].up = all_leases[n + 1].weight;
            n += 2;
        } else {
            experts[e].gate = try acquireInto(store, ex.gate_ref, all_leases, &n);
            experts[e].up = if (ex.gated) try acquireInto(store, ex.up_ref, all_leases, &n) else experts[e].gate;
        }
        experts[e].down = try acquireInto(store, ex.down_ref, all_leases, &n);
    }
    try acquireTrunk(store, m, &lease, all_leases, &n);
    std.debug.assert(n == all_leases.len);
    lease.layer.experts = experts;
    lease.experts = experts;
    lease.leases = all_leases;
    return lease;
}

// ---------------------------------------------------------------------------
// Loading
// ---------------------------------------------------------------------------

fn cat(arena: Allocator, parts: []const []const u8) ![]const u8 {
    return std.mem.concat(arena, u8, parts);
}

/// Copies the sub-block `src[rows][col_lo..col_hi]` (every `stride`-th column,
/// row-major, element size `es`) transposed into a new `[n][rows]` buffer.
fn transposeSub(arena: Allocator, es: usize, src: []const u8, rows: usize, cols: usize, col_lo: usize, col_hi: usize, stride: usize) ![]u8 {
    const n = (col_hi - col_lo + stride - 1) / stride;
    const out = try arena.alloc(u8, n * rows * es);
    var r: usize = 0;
    while (r < rows) : (r += 1) {
        var j: usize = 0;
        while (j < n) : (j += 1) {
            const c = col_lo + j * stride;
            @memcpy(out[(j * rows + r) * es ..][0..es], src[(r * cols + c) * es ..][0..es]);
        }
    }
    return out;
}

fn blockWeight(t: Weight, block: usize, block_elems: usize, rows: usize, cols: usize, row_off: usize) Weight {
    const es = t.dtype.size();
    return .{
        .data = t.data[(block * block_elems + row_off * cols) * es ..][0 .. rows * cols * es],
        .dtype = t.dtype,
        .rows = rows,
        .cols = cols,
    };
}

/// Expert `e`'s `[rows][cols]` block of a stacked `[E, ...]` tensor ref.
fn blockRef(stacked: WeightRef, e: usize, rows: usize, cols: usize) WeightRef {
    var r = stacked.rowSlice(e, 1);
    std.debug.assert(r.cols == rows * cols);
    r.rows = rows;
    r.cols = cols;
    return r;
}

const Stacked = struct { ref: WeightRef, shape: []const usize };

/// The stacked 3-D tensor `name` if present.
fn findStacked(model: *const Model, name: []const u8) !?Stacked {
    const t = model.find(name) orelse return null;
    if (t.shape.len != 3) return error.InvalidConfig;
    return .{ .ref = try model.ref(name), .shape = t.shape };
}

/// Row `e` of a stacked `[E][n]` bias tensor as f32 (null if absent).
fn biasRow(model: *Model, arena: Allocator, name: []const u8, e: usize, n: usize) !?[]f32 {
    const r = model.store.lookup(name) orelse return null;
    if (r.cols != n) return error.InvalidConfig;
    return try model.store.readVecF32(arena, r.rowSlice(e, 1));
}

/// Splits an interleaved `[2I]` vector into its even and odd entries.
fn deinterleave(arena: Allocator, v: []const f32) ![2][]f32 {
    const n = v.len / 2;
    const a = try arena.alloc(f32, n);
    const b = try arena.alloc(f32, n);
    for (0..n) |i| {
        a[i] = v[2 * i];
        b[i] = v[2 * i + 1];
    }
    return .{ a, b };
}

/// The router's `e_score_correction_bias`, under the family's name or the
/// other of its two on-disk spellings: `mlp.gate.e_score_correction_bias`
/// (DeepSeek V3) and `mlp.e_score_correction_bias` (K-EXAONE's release,
/// which transformers renames on load). A missing bias is not an error, since
/// a router without one is valid, so a misspelt name would otherwise route
/// silently without it.
fn loadCorrectionBias(model: anytype, arena: Allocator, lp: []const u8, name: ?[]const u8) !?[]const f32 {
    const t = name orelse return null;
    if ((try model.loadVecOpt(try cat(arena, &.{ lp, t })))) |v| return v;
    const gate = ".gate.e_score_correction_bias";
    const bare = ".e_score_correction_bias";
    const alt = if (std.mem.endsWith(u8, t, gate))
        try std.mem.concat(arena, u8, &.{ t[0 .. t.len - gate.len], bare })
    else if (std.mem.endsWith(u8, t, bare))
        try std.mem.concat(arena, u8, &.{ t[0 .. t.len - bare.len], gate })
    else
        return null;
    return (try model.loadVecOpt(try cat(arena, &.{ lp, alt })));
}

/// Loads the MoE block of layer `li` with tensor prefix `lp` (e.g. "model.layers.3.").
pub fn loadLayer(model: *Model, arena: Allocator, li: usize, lp: []const u8) !MoeLayer {
    const c = &model.config;
    var names: *const arch.Names = &c.arch.names;
    const hidden = c.hidden_size;
    const inter = c.moe_intermediate_size;
    const n_experts = c.num_experts;
    const mapped = !model.streamed();

    // Checkpoints that predate the family's Hugging Face module layout keep
    // the MoE under other names (Kimi Linear's `block_sparse_moe`).
    if (names.moe_alt) |alt| {
        if (model.store.lookup(try cat(arena, &.{ lp, names.router })) == null and model.store.lookup(try cat(arena, &.{ lp, alt.router })) != null) names = alt;
    }
    const router_name = try cat(arena, &.{ lp, names.router });
    const router_ref = model.store.lookup(router_name) orelse {
        std.log.err("missing router tensor {s}", .{router_name});
        return error.MissingWeights;
    };
    var self = MoeLayer{
        .layer_index = li,
        .router = try model.loadMat(router_name),
        .router_ref = router_ref,
        .router_bias = (try model.loadVecOpt(try model_mod.biasName(arena, router_name))),
        .correction_bias = try loadCorrectionBias(model, arena, lp, names.router_correction_bias),
        .top_k = @min(c.num_experts_per_tok, n_experts),
        .norm_topk_prob = c.norm_topk_prob,
        .routing = c.moe,
        .activation = c.activation,
        .inter = inter,
        .layout = .separate,
        .experts = try arena.alloc(Expert, n_experts),
        .shared = null,
        .fused_down_suffix = null,
        .names = names,
    };
    if (self.router.rows != n_experts or self.router.cols != hidden) {
        std.log.err("router {s} is [{d}][{d}], expected [{d}][{d}]", .{ router_name, self.router.rows, self.router.cols, n_experts, hidden });
        return error.InvalidConfig;
    }
    if (c.dsv4) |d| if (d.hash_moe_layers[li]) {
        const table_name = try cat(arena, &.{ lp, "mlp.gate.tid2eid" });
        const ref = model.store.lookup(table_name) orelse {
            std.log.err("missing hash routing table {s}", .{table_name});
            return error.MissingWeights;
        };
        if (ref.dtype != .i64 or ref.cols != self.top_k or ref.rows < c.vocab_size) {
            std.log.err("hash routing table {s} is {s} [{d}][{d}], expected I64 [{d}][{d}]", .{ table_name, ref.dtype.safetensorsName(), ref.rows, ref.cols, c.vocab_size, self.top_k });
            return error.InvalidConfig;
        }
        const raw = try model.store.readVecF32(arena, ref);
        const table = try arena.alloc(u32, raw.len);
        for (raw, 0..) |v, i| {
            if (v < 0 or v >= @as(f32, @floatFromInt(n_experts)) or v != @floor(v)) return error.InvalidConfig;
            table[i] = @intFromFloat(v);
        }
        self.hash_table = table;
    };

    // Latent MoE: the routed experts run in `routed_expert_hidden_size` space.
    if (c.moe_latent > 0) {
        const down_name = try cat(arena, &.{ lp, names.latent_down orelse return error.InvalidConfig });
        const up_name = try cat(arena, &.{ lp, names.latent_up orelse return error.InvalidConfig });
        var lat = Latent{
            .width = c.moe_latent,
            .down = try model.loadMat(down_name),
            .up = try model.loadMat(up_name),
            .norm = null,
            .down_ref = .{ .ref = try model.ref(down_name) },
            .up_ref = .{ .ref = try model.ref(up_name) },
        };
        if (lat.down.rows != c.moe_latent or lat.down.cols != hidden or lat.up.rows != hidden or lat.up.cols != c.moe_latent) {
            std.log.err("layer {d}: latent projections {s} [{d}][{d}] / {s} [{d}][{d}] do not match routed_expert_hidden_size {d}", .{ li, down_name, lat.down.rows, lat.down.cols, up_name, lat.up.rows, lat.up.cols, c.moe_latent });
            return error.InvalidConfig;
        }
        if (c.moe_latent_norm) {
            const norm_name = try cat(arena, &.{ lp, names.latent_norm orelse return error.InvalidConfig });
            lat.norm = (try model.loadVecOpt(norm_name)) orelse {
                std.log.err("missing tensor: {s}", .{norm_name});
                return error.MissingWeights;
            };
            if (lat.norm.?.len != c.moe_latent) return error.InvalidConfig;
        }
        self.latent = lat;
    }
    // Width of the routed experts' input and output.
    const ew = self.expertWidth(hidden);

    // Fused layout?
    var fused_gu: ?WeightRef = null;
    var fused_gu_name: []const u8 = "";
    for (names.fused_gate_up) |gn| {
        const full = try cat(arena, &.{ lp, gn });
        if (try findStacked(model, full)) |st| {
            const shape = st.shape;
            if (shape[0] != n_experts) return error.InvalidConfig;
            if (shape[1] == 2 * inter and shape[2] == ew) self.layout = .fused else if (shape[1] == ew and shape[2] == 2 * inter) self.layout = .fused_transposed else {
                std.log.err("unexpected fused expert shape [{d}, {d}, {d}]", .{ shape[0], shape[1], shape[2] });
                return error.InvalidConfig;
            }
            fused_gu = st.ref;
            fused_gu_name = full;
            break;
        }
    }
    if (fused_gu) |gu| {
        var down: ?WeightRef = null;
        var down_shape: []const usize = &.{};
        var down_name: []const u8 = "";
        for (names.fused_down) |dn| {
            if (try findStacked(model, try cat(arena, &.{ lp, dn }))) |st| {
                down = st.ref;
                down_shape = st.shape;
                down_name = try cat(arena, &.{ lp, dn });
                self.fused_down_suffix = dn;
                break;
            }
        }
        const dw = down orelse return error.MissingWeights;
        if (down_shape[0] != n_experts) return error.InvalidConfig;
        const es = gu.dtype.size();
        const interleaved = c.moe.gate_up_interleaved;
        if (interleaved and self.layout != .fused_transposed) {
            std.log.err("interleaved gate/up experts are only supported in the [E, ew, 2I] layout", .{});
            return error.InvalidConfig;
        }
        // Views of the whole stacked tensors (data valid in mapped mode only).
        const gu_w: Weight = if (mapped) model.find(gu.name).?.asWeight() else gu.shapeOnly();
        const dw_w: Weight = if (mapped) model.find(dw.name).?.asWeight() else dw.shapeOnly();
        const gu_bias_name = try model_mod.biasName(arena, fused_gu_name);
        const dn_bias_name = try model_mod.biasName(arena, down_name);
        switch (self.layout) {
            .fused => {
                if (down_shape[1] != ew or down_shape[2] != inter) return error.InvalidConfig;
                for (self.experts, 0..) |*ex, e| {
                    const gu_block = blockRef(gu, e, 2 * inter, ew);
                    ex.* = .{
                        .gate_ref = .{ .ref = gu_block.rowSlice(0, inter) },
                        .up_ref = .{ .ref = gu_block.rowSlice(inter, inter) },
                        .down_ref = .{ .ref = blockRef(dw, e, ew, inter) },
                        .gate = undefined,
                        .up = undefined,
                        .down = undefined,
                    };
                    if (mapped) {
                        ex.gate = blockWeight(gu_w, e, 2 * inter * ew, inter, ew, 0);
                        ex.up = blockWeight(gu_w, e, 2 * inter * ew, inter, ew, inter);
                        ex.down = blockWeight(dw_w, e, ew * inter, ew, inter, 0);
                    }
                }
            },
            .fused_transposed => {
                if (down_shape[1] != inter or down_shape[2] != ew) return error.InvalidConfig;
                for (self.experts, 0..) |*ex, e| {
                    const gu_block = blockRef(gu, e, ew, 2 * inter);
                    ex.* = .{
                        .gate_ref = if (interleaved) .{ .ref = gu_block, .transposed = .{ 0, 2 * inter }, .stride = 2 } else .{ .ref = gu_block, .transposed = .{ 0, inter } },
                        .up_ref = if (interleaved) .{ .ref = gu_block, .transposed = .{ 1, 2 * inter }, .stride = 2 } else .{ .ref = gu_block, .transposed = .{ inter, 2 * inter } },
                        .down_ref = .{ .ref = blockRef(dw, e, inter, ew), .transposed = .{ 0, ew } },
                        .gate = undefined,
                        .up = undefined,
                        .down = undefined,
                    };
                    if (mapped) {
                        // Transposed once on load into arena-owned copies.
                        const gu_bytes = gu_w.data[e * ew * 2 * inter * es ..][0 .. ew * 2 * inter * es];
                        const dn_bytes = dw_w.data[e * inter * ew * es ..][0 .. inter * ew * es];
                        const gcols = ex.gate_ref.columns();
                        const ucols = ex.up_ref.columns();
                        ex.gate = .{ .data = try transposeSub(arena, es, gu_bytes, ew, 2 * inter, gcols.lo, gcols.hi, gcols.stride), .dtype = gu.dtype, .rows = inter, .cols = ew };
                        ex.up = .{ .data = try transposeSub(arena, es, gu_bytes, ew, 2 * inter, ucols.lo, ucols.hi, ucols.stride), .dtype = gu.dtype, .rows = inter, .cols = ew };
                        ex.down = .{ .data = try transposeSub(arena, es, dn_bytes, inter, ew, 0, ew, 1), .dtype = dw.dtype, .rows = ew, .cols = inter };
                    }
                }
            },
            .separate => unreachable,
        }
        if (!mapped) {
            for (self.experts) |*ex| {
                ex.gate = ex.gate_ref.shapeOnly();
                ex.up = ex.up_ref.shapeOnly();
                ex.down = ex.down_ref.shapeOnly();
            }
        }
        // Expert biases (`gate_up_proj_bias` [E, 2I], `down_proj_bias` [E, ew]).
        for (self.experts, 0..) |*ex, e| {
            if (try biasRow(model, arena, gu_bias_name, e, 2 * inter)) |gub| {
                if (interleaved) {
                    const parts = try deinterleave(arena, gub);
                    ex.gate_bias = parts[0];
                    ex.up_bias = parts[1];
                } else {
                    ex.gate_bias = gub[0..inter];
                    ex.up_bias = gub[inter..];
                }
            }
            ex.down_bias = try biasRow(model, arena, dn_bias_name, e, ew);
        }
    } else if (c.moe.dense_experts) {
        // Non-gated experts (Nemotron-H): `up_proj` and `down_proj` only.
        for (self.experts, 0..) |*ex, e| {
            const ep = try cat(arena, &.{ lp, try model_mod.resolveName(arena, names.expert, "", 0, e) });
            const up_name = try cat(arena, &.{ ep, names.expert_up });
            const down_name = try cat(arena, &.{ ep, names.expert_down });
            const up = try model.loadMat(up_name);
            const up_ref = MatrixRef{ .ref = try model.ref(up_name) };
            ex.* = .{
                .gate = up,
                .up = up,
                .down = try model.loadMat(down_name),
                .gate_ref = up_ref,
                .up_ref = up_ref,
                .down_ref = .{ .ref = try model.ref(down_name) },
                .gate_bias = (try model.loadVecOpt(try model_mod.biasName(arena, up_name))),
                .down_bias = (try model.loadVecOpt(try model_mod.biasName(arena, down_name))),
                .gated = false,
            };
        }
    } else {
        for (self.experts, 0..) |*ex, e| {
            const ep = try cat(arena, &.{ lp, try model_mod.resolveName(arena, names.expert, "", 0, e) });
            const gate_name = try cat(arena, &.{ ep, names.expert_gate });
            const up_name = try cat(arena, &.{ ep, names.expert_up });
            const down_name = try cat(arena, &.{ ep, names.expert_down });
            ex.* = .{
                .gate = try model.loadMat(gate_name),
                .up = try model.loadMat(up_name),
                .down = try model.loadMat(down_name),
                .gate_ref = .{ .ref = try model.ref(gate_name) },
                .up_ref = .{ .ref = try model.ref(up_name) },
                .down_ref = .{ .ref = try model.ref(down_name) },
                .gate_bias = (try model.loadVecOpt(try model_mod.biasName(arena, gate_name))),
                .up_bias = (try model.loadVecOpt(try model_mod.biasName(arena, up_name))),
                .down_bias = (try model.loadVecOpt(try model_mod.biasName(arena, down_name))),
            };
        }
    }
    for (self.experts) |ex| {
        if (ex.gate.rows != inter or ex.gate.cols != ew or ex.down.rows != ew or ex.down.cols != inter) return error.InvalidConfig;
    }

    // Shared expert(s): one fused block (DeepSeek `shared_experts`, Qwen2-MoE
    // `shared_expert`, Llama 4 `shared_expert`); its width comes from the
    // weights. GraniteMoeShared and MiniMax M3 store its gate and up rows in
    // one `[2I][hidden]` tensor: the two halves become row slices of it.
    // Nemotron-H's shared expert has no gate projection.
    if (names.shared_expert) |sp_t| {
        const sp = try cat(arena, &.{ lp, sp_t });
        const down_name = try cat(arena, &.{ sp, names.shared_down orelse names.expert_down });
        if (model.find(down_name)) |_| {
            var sh = SharedExpert{
                .gate = undefined,
                .up = undefined,
                .down = try model.loadMat(down_name),
                .gate_ref = undefined,
                .up_ref = undefined,
                .down_ref = .{ .ref = try model.ref(down_name) },
                .gate_vec = if (names.shared_expert_gate) |g| (try model.loadVecOpt(try cat(arena, &.{ lp, g }))) else null,
            };
            const fused_name: ?[]const u8 = if (names.shared_gate_up) |t| try cat(arena, &.{ sp, t }) else null;
            if (fused_name != null and model.find(fused_name.?) != null) {
                const gu_ref = try model.ref(fused_name.?);
                if (gu_ref.rows % 2 != 0 or gu_ref.rows / 2 != sh.down.cols) {
                    std.log.err("shared expert {s} is [{d}][{d}], expected [{d}][{d}]", .{ fused_name.?, gu_ref.rows, gu_ref.cols, 2 * sh.down.cols, hidden });
                    return error.InvalidConfig;
                }
                const si = gu_ref.rows / 2;
                sh.gate_ref = .{ .ref = gu_ref.rowSlice(0, si) };
                sh.up_ref = .{ .ref = gu_ref.rowSlice(si, si) };
                if (mapped) {
                    const gu_w = model.find(fused_name.?).?.asWeight();
                    sh.gate = blockWeight(gu_w, 0, 0, si, hidden, 0);
                    sh.up = blockWeight(gu_w, 0, 0, si, hidden, si);
                } else {
                    sh.gate = sh.gate_ref.shapeOnly();
                    sh.up = sh.up_ref.shapeOnly();
                }
            } else if (c.moe.dense_experts) {
                // `down(act(up(x)))`: `gate` aliases `up` and is not read twice.
                const up_name = try cat(arena, &.{ sp, names.shared_up orelse names.expert_up });
                sh.up = try model.loadMat(up_name);
                sh.gate = sh.up;
                sh.up_ref = .{ .ref = try model.ref(up_name) };
                sh.gate_ref = sh.up_ref;
                sh.gated = false;
            } else {
                const gate_name = try cat(arena, &.{ sp, names.shared_gate orelse names.expert_gate });
                const up_name = try cat(arena, &.{ sp, names.shared_up orelse names.expert_up });
                sh.gate = try model.loadMat(gate_name);
                sh.up = try model.loadMat(up_name);
                sh.gate_ref = .{ .ref = try model.ref(gate_name) };
                sh.up_ref = .{ .ref = try model.ref(up_name) };
            }
            if (sh.gate.cols != hidden or sh.down.rows != hidden or sh.gate.rows != sh.down.cols or sh.up.rows != sh.down.cols) {
                std.log.err("shared expert of layer {d} has inconsistent shapes", .{li});
                return error.InvalidConfig;
            }
            self.shared = sh;
        }
    }
    return self;
}

// ---------------------------------------------------------------------------
// Forward
// ---------------------------------------------------------------------------

inline fn sigmoid(x: f32) f32 {
    return 1.0 / (1.0 + @exp(-x));
}

/// `out[ne][hidden] = (act(x Wgᵀ + bg) ⊙ (x Wuᵀ + bu)) Wdᵀ + bd` for `ne` rows
/// of `x`; a non-gated expert (`gated == false`) computes `act(x Wgᵀ + bg) Wdᵀ + bd`.
fn runExpert(model: *const Model, m: *const MoeLayer, gated: bool, gate_w: Weight, up_w: Weight, down_w: Weight, biases: [3]?[]const f32, delta: ?*const Delta, x: []const f32, ne: usize, gate: []f32, up: []f32, out: []f32) !void {
    const gpa = model.gpa;
    const inter = gate_w.rows;
    const hidden = down_w.rows;
    try compute.matmulT(model.pool, gpa, gate, x, ne, gate_w, null);
    if (!gated) {
        for (0..ne) |i| {
            if (biases[0]) |b| tensor.axpy(gate[i * inter ..][0..inter], 1.0, b[0..inter]);
        }
        compute.gatedActivation(model.pool, m.activation, gate, gate, null, ne, inter, inter, inter);
        try compute.matmulT(model.pool, gpa, out, gate, ne, down_w, delta);
        if (biases[2]) |b| for (0..ne) |i| tensor.axpy(out[i * hidden ..][0..hidden], 1.0, b[0..hidden]);
        return;
    }
    try compute.matmulT(model.pool, gpa, up, x, ne, up_w, null);
    for (0..ne) |i| {
        if (biases[0]) |b| tensor.axpy(gate[i * inter ..][0..inter], 1.0, b[0..inter]);
        if (biases[1]) |b| tensor.axpy(up[i * inter ..][0..inter], 1.0, b[0..inter]);
    }
    if (m.routing.swiglu) |sw| {
        // gpt-oss: clamp, gate * sigmoid(alpha * gate), (up + 1) factor.
        for (gate[0 .. ne * inter], 0..) |*g, j| {
            const gc = @min(g.*, sw.limit);
            const uc = std.math.clamp(up[j], -sw.limit, sw.limit);
            g.* = (uc + 1.0) * gc * sigmoid(sw.alpha * gc);
        }
    } else if (m.routing.situ) |st| {
        situGlu(st, gate, gate, up, ne, inter, inter);
    } else if (m.routing.swiglu_limit) |limit| {
        // DeepSeek V4: the gate is clamped from above, the up projection on both sides.
        for (gate[0 .. ne * inter], 0..) |*g, j| {
            g.* = @min(g.*, limit);
            up[j] = std.math.clamp(up[j], -limit, limit);
        }
        compute.gatedActivation(model.pool, m.activation, gate, gate, up, ne, inter, inter, inter);
    } else {
        compute.gatedActivation(model.pool, m.activation, gate, gate, up, ne, inter, inter, inter);
    }
    try compute.matmulT(model.pool, gpa, out, gate, ne, down_w, delta);
    if (biases[2]) |b| for (0..ne) |i| tensor.axpy(out[i * hidden ..][0..hidden], 1.0, b[0..hidden]);
}

/// Kimi K3 SiTU on `n` rows of `gate` and `up` (row stride `in_stride`):
/// `out = beta · tanh(gate / beta) · σ(gate) · up'`, with `up' = linear_beta ·
/// tanh(up / linear_beta)` when a linear beta is set; written densely (`[n][inter]`).
pub fn situGlu(st: arch.Situ, out: []f32, gate: []const f32, up: []const f32, n: usize, inter: usize, in_stride: usize) void {
    for (0..n) |i| {
        const g = gate[i * in_stride ..][0..inter];
        const u = up[i * in_stride ..][0..inter];
        const o = out[i * inter ..][0..inter];
        for (o, 0..) |*v, j| {
            const a = st.beta * std.math.tanh(g[j] / st.beta) * sigmoid(g[j]);
            const x = if (st.linear_beta) |lb| lb * std.math.tanh(u[j] / lb) else u[j];
            v.* = a * x;
        }
    }
}

/// Picks the top-k experts of one token from `scores` (routing weights) using
/// `sel` (selection scores, possibly biased and group-masked). Returns the
/// selected indices and their weights (before renormalisation).
fn topk(scores: []const f32, sel: []f32, k: usize, idx: []usize, wts: []f32) void {
    for (0..k) |j| {
        var best: usize = 0;
        var bv: f32 = -std.math.inf(f32);
        for (sel, 0..) |s, e| {
            if (s > bv) {
                bv = s;
                best = e;
            }
        }
        idx[j] = best;
        wts[j] = scores[best];
        sel[best] = -std.math.inf(f32);
    }
}

/// `sqrt(softplus(x))` (DeepSeek V4 router scores; softplus is linear above 20 like PyTorch's).
fn sqrtSoftplus(x: f32) f32 {
    const sp = if (x > 20.0) x else std.math.log1p(@exp(x));
    return @sqrt(sp);
}

/// Routed MoE MLP of layer `li`: `out[n][hidden]` from normalised inputs
/// `h[n][hidden]`. In warp mode the union of the experts selected for the
/// `n` tokens is fetched through the model's expert cache (misses of the
/// layer as one prefetch group), one expert pinned at a time while it runs.
/// `tokens` (the input token ids of the rows) drives hash-routed layers.
pub fn forward(model: *const Model, m: *const MoeLayer, li: usize, out: []f32, h: []const f32, n: usize, tokens: ?[]const u32) !void {
    const gpa = model.gpa;
    const hidden = model.config.hidden_size;
    const n_experts = m.experts.len;
    const k = m.top_k;
    const inter = m.inter;
    const r = &m.routing;

    // Router scores (softmax or sigmoid), then top-k on the selection scores
    // (scores plus the correction bias, restricted to the best groups when
    // routing is group-limited) with optional renormalisation and scaling.
    const logits = try gpa.alloc(f32, n * n_experts);
    defer gpa.free(logits);
    try compute.matmulT(model.pool, gpa, logits, h, n, m.router, null);
    const sel = try gpa.alloc(usize, n * k);
    defer gpa.free(sel);
    const selw = try gpa.alloc(f32, n * k);
    defer gpa.free(selw);
    const choice = try gpa.alloc(f32, n_experts);
    defer gpa.free(choice);
    const n_group = if (r.topk_method == .group_limited) r.n_group else 1;
    const group_scores = try gpa.alloc(f32, n_group);
    defer gpa.free(group_scores);
    for (0..n) |t| {
        const row = logits[t * n_experts ..][0..n_experts];
        if (m.router_bias) |b| tensor.axpy(row, 1.0, b[0..n_experts]);
        if (r.gate_temp != 1.0) tensor.scale(row, 1.0 / r.gate_temp);
        if (r.router_softcap) |cap| compute.softcap(row, cap);
        switch (r.scoring) {
            .softmax => compute.softmaxInPlace(row),
            .sigmoid => for (row) |*v| {
                v.* = sigmoid(v.*);
            },
            .sqrtsoftplus => for (row) |*v| {
                v.* = sqrtSoftplus(v.*);
            },
        }
        if (m.hash_table) |table| {
            // The token id fixes the experts; the scores only weight them.
            const toks = tokens orelse return error.MissingTokens;
            const tid = @min(toks[t], @as(u32, @intCast(table.len / k - 1)));
            for (0..k) |j| {
                const e = table[tid * k + j];
                sel[t * k + j] = e;
                selw[t * k + j] = row[e];
            }
            var hsum: f32 = 0;
            for (selw[t * k ..][0..k]) |w| hsum += w;
            for (0..k) |j| selw[t * k + j] = selw[t * k + j] / (hsum + 1e-20) * r.routed_scaling_factor;
            continue;
        }
        for (choice, 0..) |*cv, e| cv.* = row[e] + (if (m.correction_bias) |b| b[e] else 0);
        if (r.topk_method == .group_limited and n_group > 1) {
            const per_group = n_experts / n_group;
            for (0..n_group) |g| {
                const gs = choice[g * per_group ..][0..per_group];
                if (m.correction_bias != null or r.group_score_top2) {
                    // Sum of the two best selection scores of the group (DeepSeek V3, Kimi Linear).
                    var b1: f32 = -std.math.inf(f32);
                    var b2: f32 = -std.math.inf(f32);
                    for (gs) |s| {
                        if (s > b1) {
                            b2 = b1;
                            b1 = s;
                        } else if (s > b2) b2 = s;
                    }
                    group_scores[g] = b1 + (if (per_group > 1) b2 else 0);
                } else {
                    var best: f32 = -std.math.inf(f32);
                    for (gs) |s| best = @max(best, s);
                    group_scores[g] = best;
                }
            }
            // Keep the `topk_group` best groups.
            const keep = @min(r.topk_group, n_group);
            var kept: usize = 0;
            while (kept < keep) : (kept += 1) {
                var best: usize = 0;
                var bv: f32 = -std.math.inf(f32);
                for (group_scores, 0..) |s, g| {
                    if (s > bv) {
                        bv = s;
                        best = g;
                    }
                }
                group_scores[best] = std.math.nan(f32); // marked as kept
            }
            for (group_scores, 0..) |s, g| {
                if (!std.math.isNan(s)) @memset(choice[g * per_group ..][0..per_group], -std.math.inf(f32));
            }
        }
        topk(row, choice, k, sel[t * k ..][0..k], selw[t * k ..][0..k]);
        var sum: f32 = 0;
        for (selw[t * k ..][0..k]) |w| sum += w;
        if (m.norm_topk_prob and r.norm_eps_floor) {
            for (0..k) |j| selw[t * k + j] /= sum + 1e-20;
        } else if (m.norm_topk_prob and sum > 0) {
            for (0..k) |j| selw[t * k + j] /= sum;
        }
        if (r.routed_scaling_factor != 1.0) for (0..k) |j| {
            selw[t * k + j] *= r.routed_scaling_factor;
        };
    }

    @memset(out[0 .. n * hidden], 0);
    // Latent MoE: the experts read `down(h)` and accumulate in latent space;
    // otherwise they read `h` and accumulate straight into `out`.
    const ew = m.expertWidth(hidden);
    var xin: []const f32 = h;
    var acc: []f32 = out;
    var latent_bufs: ?[2][]f32 = null;
    defer if (latent_bufs) |bufs| {
        gpa.free(bufs[0]);
        gpa.free(bufs[1]);
    };
    if (m.latent) |lat| {
        const xl = try gpa.alloc(f32, n * ew);
        errdefer gpa.free(xl);
        const al = try gpa.alloc(f32, n * ew);
        latent_bufs = .{ xl, al };
        try compute.matmulT(model.pool, gpa, xl, h, n, lat.down, null);
        @memset(al, 0);
        xin = xl;
        acc = al;
    }
    const xg = try gpa.alloc(f32, n * ew);
    defer gpa.free(xg);
    const gate = try gpa.alloc(f32, n * inter);
    defer gpa.free(gate);
    const up = try gpa.alloc(f32, n * inter);
    defer gpa.free(up);
    const eo = try gpa.alloc(f32, n * hidden);
    defer gpa.free(eo);
    const tok = try gpa.alloc(usize, n);
    defer gpa.free(tok);
    const wts = try gpa.alloc(f32, n);
    defer gpa.free(wts);

    // The union of selected experts, in index order (the accumulation order
    // is the same in every mode, so results are bitwise identical).
    const used = try gpa.alloc(bool, n_experts);
    defer gpa.free(used);
    @memset(used, false);
    for (sel[0 .. n * k]) |e| used[e] = true;
    const union_list = try gpa.alloc(usize, n_experts);
    defer gpa.free(union_list);
    var n_union: usize = 0;
    for (used, 0..) |u, e| {
        if (u) {
            union_list[n_union] = e;
            n_union += 1;
        }
    }
    const cache = model.expert_cache;
    // From a remote source, every routed expert that is not resident starts
    // downloading now, in the order they run, over the source's parallel
    // connections: the ones beyond what the expert cache can prefetch are
    // then in the chunk cache (or on their way) by the time they are needed,
    // instead of each one being fetched on its own when its turn comes.
    if (model.warp()) {
        var refs = std.ArrayList(stream.WeightRef).empty;
        defer refs.deinit(gpa);
        for (union_list[0..n_union]) |e| {
            if (cache) |c| if (c.resident(li, e)) continue;
            const ex = &m.experts[e];
            refs.appendSlice(gpa, &.{ ex.gate_ref.ref, ex.up_ref.ref, ex.down_ref.ref }) catch break;
        }
        const store: *stream.WeightStore = @constCast(&model.store);
        store.prefetchRemote(refs.items);
    }
    if (cache) |c| c.prefetch(li, m, union_list[0..n_union]);

    for (union_list[0..n_union]) |e| {
        const ex = &m.experts[e];
        var ne: usize = 0;
        for (0..n) |t| {
            for (0..k) |j| {
                if (sel[t * k + j] == e) {
                    tok[ne] = t;
                    wts[ne] = selw[t * k + j];
                    ne += 1;
                }
            }
        }
        for (0..ne) |j| {
            const dst = xg[j * ew ..][0..ew];
            @memcpy(dst, xin[tok[j] * ew ..][0..ew]);
            if (r.scale_input) tensor.scale(dst, wts[j]);
        }
        const delta: ?*const Delta = if (ex.down_delta) |*d| d else null;
        const biases = [3]?[]const f32{ ex.gate_bias, ex.up_bias, ex.down_bias };
        if (cache) |c| {
            const entry = try c.acquire(li, m, e);
            defer c.release(entry);
            try runExpert(model, m, ex.gated, entry.gate, entry.up, entry.down, biases, delta, xg, ne, gate, up, eo);
        } else {
            try runExpert(model, m, ex.gated, ex.gate, ex.up, ex.down, biases, delta, xg, ne, gate, up, eo);
        }
        for (0..ne) |j| tensor.axpy(acc[tok[j] * ew ..][0..ew], if (r.scale_input) 1.0 else wts[j], eo[j * ew ..][0..ew]);
    }

    if (m.latent) |*lat| {
        // Latent norm on the summed routed output, then the up projection
        // (with its delta) writes it into the residual.
        if (lat.norm) |nw| {
            const tmp = try gpa.alloc(f32, ew);
            defer gpa.free(tmp);
            for (0..n) |t| {
                const row = acc[t * ew ..][0..ew];
                compute.rmsnorm(tmp, row, nw, model.config.rms_norm_eps, false);
                @memcpy(row, tmp);
            }
        }
        try compute.matmulT(model.pool, gpa, out, acc, n, lat.up, if (lat.up_delta) |*d| d else null);
    }

    if (m.shared) |*sh| {
        const s_inter = sh.gate.rows;
        const sg = try gpa.alloc(f32, n * s_inter);
        defer gpa.free(sg);
        const su = try gpa.alloc(f32, n * s_inter);
        defer gpa.free(su);
        try runExpert(model, m, sh.gated, sh.gate, sh.up, sh.down, .{ null, null, null }, if (sh.down_delta) |*d| d else null, h, n, sg, su, eo);
        for (0..n) |t| {
            const s: f32 = if (sh.gate_vec) |g| sigmoid(tensor.dot(h[t * hidden ..][0..hidden], g[0..hidden])) else 1.0;
            tensor.axpy(out[t * hidden ..][0..hidden], s, eo[t * hidden ..][0..hidden]);
        }
    }
}

// ---------------------------------------------------------------------------
// Expert ranking and selection
// ---------------------------------------------------------------------------

/// Ranking heuristic: `score = ||Vᵀ W||_F / ||W||_F`, the fraction of a down
/// projection's output energy that lies along the refusal direction(s) `V`
/// (one or more orthonormal vectors in residual space, `v.len == K * rows`).
/// Experts whose down projection writes more strongly along `V` are assumed
/// to carry more of the refusal behaviour.
pub fn scoreDown(pool: *const tensor.Pool, gpa: Allocator, w: Weight, v: []const f32) !f32 {
    std.debug.assert(v.len > 0 and v.len % w.rows == 0);
    const k = v.len / w.rows;
    const proj = try gpa.alloc(f32, k * w.cols);
    defer gpa.free(proj);
    try compute.matvecTMulti(pool, gpa, proj, w, v, k);
    const norms = try gpa.alloc(f32, w.rows);
    defer gpa.free(norms);
    try compute.rowNorms(pool, gpa, norms, w);
    const fro = tensor.norm2(norms);
    return if (fro > 0) tensor.norm2(proj) / fro else 0;
}

/// Scores every routed expert of a layer whose down projections are resident
/// (mapped mode). Only routed experts are scored; the shared expert is never a candidate.
pub fn scoreLayer(pool: *const tensor.Pool, gpa: Allocator, m: *const MoeLayer, v: []const f32) ![]f32 {
    const scores = try gpa.alloc(f32, m.experts.len);
    errdefer gpa.free(scores);
    for (m.experts, 0..) |ex, e| scores[e] = try scoreDown(pool, gpa, ex.down, v);
    return scores;
}

/// Scores every routed expert of `layer`, making one down projection resident
/// at a time (through the expert cache in warp mode). With `visited_only`,
/// experts that no forward pass of this run routed to are not read; they
/// score -1 and therefore rank last.
pub fn scoreExperts(model: *const Model, layer: usize, v: []const f32, visited_only: bool) ![]f32 {
    const m = &(model.layers[layer].moe orelse return error.NotMoeLayer);
    // Latent experts write into latent space, where a residual direction has
    // no projection; the layer's routed edit goes through `Latent.up` instead.
    if (m.routedInLatent()) return error.LatentExperts;
    const gpa = model.gpa;
    const scores = try gpa.alloc(f32, m.experts.len);
    errdefer gpa.free(scores);
    for (0..m.experts.len) |e| {
        if (visited_only and !model.expertVisited(layer, e)) {
            scores[e] = -1;
            continue;
        }
        const lease = try m.acquireDown(model, e);
        defer lease.release(model);
        scores[e] = try scoreDown(model.pool, gpa, lease.weight, v);
    }
    return scores;
}

/// Number of routed experts of `layer` that are candidates for an edit:
/// all of them, or only the visited ones (warp mode with `visited_only`).
pub fn candidateCount(model: *const Model, layer: usize, visited_only: bool) usize {
    const m = &(model.layers[layer].moe orelse return 0);
    if (!visited_only) return m.experts.len;
    var n: usize = 0;
    for (0..m.experts.len) |e| n += @intFromBool(model.expertVisited(layer, e));
    return n;
}

/// Expert indices sorted by descending score (ties keep index order).
pub fn rankScores(gpa: Allocator, scores: []const f32) ![]usize {
    const order = try gpa.alloc(usize, scores.len);
    for (order, 0..) |*o, i| o.* = i;
    std.mem.sort(usize, order, scores, struct {
        fn lt(s: []const f32, a: usize, b: usize) bool {
            if (s[a] != s[b]) return s[a] > s[b];
            return a < b;
        }
    }.lt);
    return order;
}

/// Routed experts of `layer` ranked by descending alignment with `v` (caller frees).
pub fn rankExperts(model: *const Model, layer: usize, v: []const f32) ![]usize {
    const scores = try scoreExperts(model, layer, v, false);
    defer model.gpa.free(scores);
    return rankScores(model.gpa, scores);
}

pub const Strategy = union(enum) {
    /// The `n` best-ranked experts.
    ranked,
    /// `n` uniformly random experts (validation baseline), seeded.
    random: u64,
};

/// Picks `n` experts from `ranking` (caller frees the result).
pub fn selectExperts(gpa: Allocator, ranking: []const usize, n: usize, strategy: Strategy) ![]usize {
    const count = @min(n, ranking.len);
    const out = try gpa.alloc(usize, count);
    switch (strategy) {
        .ranked => @memcpy(out, ranking[0..count]),
        .random => |seed| {
            const pool_idx = try gpa.alloc(usize, ranking.len);
            defer gpa.free(pool_idx);
            for (pool_idx, 0..) |*p, i| p.* = i;
            var prng = std.Random.DefaultPrng.init(seed);
            prng.random().shuffle(usize, pool_idx);
            @memcpy(out, pool_idx[0..count]);
        },
    }
    return out;
}

/// Prints the per-expert score table for one layer.
pub fn printExpertTable(out: *Io.Writer, layer: usize, scores: []const f32, ranking: []const usize, selected: []const usize) !void {
    try out.print("  layer {d}: expert alignment with refusal direction (||vᵀW||/||W||_F)\n", .{layer});
    try out.writeAll("    rank  expert   score  selected\n");
    for (ranking, 0..) |e, r| {
        var is_sel = false;
        for (selected) |s| is_sel = is_sel or s == e;
        try out.print("    {d:>4}  {d:>6}  {d:.4}  {s}\n", .{ r, e, scores[e], if (is_sel) "*" else "" });
    }
    try out.flush();
}

// ---------------------------------------------------------------------------
// Expert-selective abliteration
// ---------------------------------------------------------------------------

/// Applies `cfg` to the model: attention edits and dense-layer MLP edits exactly
/// as `abliterate.apply`; for each MoE layer with a non-null MLP kernel weight λ,
/// only the `n_experts` best-ranked (or random, per `opts.expert_selection`)
/// routed experts receive a down-projection edit of weight `max(0, λ·strength)`.
/// `n_experts == 0` edits every expert (broad). Routing and unselected experts
/// are untouched; the shared expert is edited only in broad mode. Deltas are
/// always recomputed from the base weights. With `opts.visited_experts_only`
/// (warp mode) only experts that a forward pass of this run routed to are
/// scored and edited: the others cannot have influenced any refusal, and
/// skipping them saves their reads.
///
/// On a latent-MoE layer (Kimi K3) the routed experts share one write into
/// the residual, `routed_expert_up_proj`: there is nothing to rank, so that
/// matrix takes the edit every routed expert would have received (weight λ
/// in broad mode, `λ·strength` otherwise, nothing when `n_experts == 0` in a
/// selective mode would edit nothing); the per-expert `w2` matrices are never
/// touched.
pub fn applyExpertSelective(model: *Model, dirs: []const f32, cfg: search.TrialConfig, opts: abliterate.Options) !void {
    if (!model.isMoe()) return error.NotMoeModel;
    const sel = cfg.experts orelse search.ExpertSelection{ .n_experts = 0, .strength = 1.0 };
    const gpa = model.gpa;
    const hidden = model.config.hidden_size;
    const stride = opts.n_directions * hidden;
    var global_dir: ?[]f32 = null;
    defer if (global_dir) |g| gpa.free(g);
    if (cfg.direction_index) |di| global_dir = try abliterate.interpolateBasis(gpa, dirs, opts.n_directions, hidden, di);

    const store: *stream.WeightStore = @constCast(&model.store);
    const visited_only = opts.visited_experts_only and model.visitedExpertsKnown();
    model.resetDeltas();
    var seed_counter: u64 = 0;
    for (model.layers, 0..) |*layer, li| {
        if (model.budget) |b| try b.checkTime();
        const v = if (global_dir) |g| g else dirs[(li + 1) * stride ..][0..stride];
        if (cfg.parameters.get(.attn_o_proj)) |p| {
            if (abliterate.kernelWeight(p, li)) |weight| {
                if (model.hasComponent(li, .attn_o_proj)) {
                    // One matrix resident at a time (streamed mode reads it from disk).
                    const lease = try model.acquireComponent(li, .attn_o_proj);
                    defer store.release(lease);
                    const delta = try abliterate.computeDelta(model.pool, gpa, lease.weight, v, weight, opts, opts.seed +% seed_counter);
                    seed_counter += 1;
                    model.setDelta(li, .attn_o_proj, delta);
                }
                if (model.hasSsmOut(li)) {
                    const lease = try model.acquireSsmOut(li);
                    defer store.release(lease);
                    const delta = try abliterate.computeDelta(model.pool, gpa, lease.weight, v, weight, opts, opts.seed +% seed_counter);
                    seed_counter += 1;
                    model.setSsmOutDelta(li, delta);
                }
            }
        }
        const p = cfg.parameters.get(.mlp_down_proj) orelse continue;
        const lambda = abliterate.kernelWeight(p, li) orelse continue;
        if (!model.hasComponent(li, .mlp_down_proj)) continue;
        const m = &(layer.moe orelse {
            // Dense layer inside a hybrid model.
            const lease = try model.acquireComponent(li, .mlp_down_proj);
            defer store.release(lease);
            const delta = try abliterate.computeDelta(model.pool, gpa, lease.weight, v, lambda, opts, opts.seed +% seed_counter);
            seed_counter += 1;
            model.setDelta(li, .mlp_down_proj, delta);
            continue;
        });
        const n_candidates = candidateCount(model, li, visited_only);
        const n = @min(sel.n_experts, n_candidates);
        const broad = sel.n_experts == 0 or opts.expert_selection == .broad;
        if (m.routedInLatent()) {
            const weight = if (broad) lambda else @max(0.0, lambda * sel.strength);
            if (weight != 0) {
                const lease = try m.acquireLatentUp(model);
                defer lease.release(model);
                const delta = try abliterate.computeDelta(model.pool, gpa, lease.weight, v, weight, opts, opts.seed +% seed_counter);
                seed_counter += 1;
                model.setLatentDelta(li, delta);
            }
            if (broad) if (m.sharedIndex()) |si| {
                const lease = try m.acquireDown(model, si);
                defer lease.release(model);
                const delta = try abliterate.computeDelta(model.pool, gpa, lease.weight, v, lambda, opts, opts.seed +% seed_counter);
                seed_counter += 1;
                model.setExpertDelta(li, si, delta);
            };
            continue;
        }
        if (broad) {
            var idx: usize = 0;
            while (idx < m.numDown()) : (idx += 1) {
                if (visited_only and idx < m.experts.len and !model.expertVisited(li, idx)) continue;
                const lease = try m.acquireDown(model, idx);
                defer lease.release(model);
                const delta = try abliterate.computeDelta(model.pool, gpa, lease.weight, v, lambda, opts, opts.seed +% seed_counter);
                seed_counter += 1;
                model.setExpertDelta(li, idx, delta);
            }
            continue;
        }
        const weight = @max(0.0, lambda * sel.strength);
        if (weight == 0 or n == 0) continue;
        const scores = try scoreExperts(model, li, v, visited_only);
        defer gpa.free(scores);
        const ranking = try rankScores(gpa, scores);
        defer gpa.free(ranking);
        const strategy: Strategy = switch (opts.expert_selection) {
            .random => .{ .random = opts.seed +% 0x9e3779b97f4a7c15 *% (li + 1) },
            else => .ranked,
        };
        // Unvisited experts score -1 and sit at the end of the ranking; the
        // candidates are its first `n_candidates` entries.
        const selected = try selectExperts(gpa, ranking[0..n_candidates], n, strategy);
        defer gpa.free(selected);
        if (opts.debug_writer) |w| try printExpertTable(w, li, scores, ranking[0..n_candidates], selected);
        for (selected) |e| {
            const lease = try m.acquireDown(model, e);
            defer lease.release(model);
            const delta = try abliterate.computeDelta(model.pool, gpa, lease.weight, v, weight, opts, opts.seed +% seed_counter);
            seed_counter += 1;
            model.setExpertDelta(li, e, delta);
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

test "expert ranking prefers the aligned expert" {
    const gpa = std.testing.allocator;
    const pool = tensor.Pool.init(std.testing.io, 1);
    const hidden = 8;
    const inter = 5;
    var prng = std.Random.DefaultPrng.init(3);
    const rand = prng.random();
    var v: [hidden]f32 = undefined;
    for (&v) |*x| x.* = rand.floatNorm(f32);
    tensor.normalize(&v);
    var u: [inter]f32 = undefined;
    for (&u) |*x| x.* = rand.floatNorm(f32);
    // Expert 1's down projection is v uᵀ (all output energy along v); the others are random.
    var mats: [3][hidden * inter]f32 = undefined;
    for (&mats, 0..) |*mat, e| {
        for (mat, 0..) |*x, idx| {
            x.* = if (e == 1) v[idx / inter] * u[idx % inter] else rand.floatNorm(f32);
        }
    }
    var experts: [3]Expert = undefined;
    const dummy_ref = MatrixRef{ .ref = .{ .name = "", .file = 0, .offset = 0, .rows = hidden, .cols = inter, .dtype = .f32 } };
    for (&experts, 0..) |*ex, e| {
        const w = Weight{ .data = std.mem.sliceAsBytes(&mats[e]), .dtype = .f32, .rows = hidden, .cols = inter };
        ex.* = .{ .gate = w, .up = w, .down = w, .gate_ref = dummy_ref, .up_ref = dummy_ref, .down_ref = dummy_ref };
    }
    const layer = MoeLayer{ .layer_index = 0, .router = experts[0].gate, .router_ref = dummy_ref.ref, .router_bias = null, .correction_bias = null, .top_k = 1, .norm_topk_prob = true, .routing = .{}, .activation = .silu, .inter = inter, .layout = .separate, .experts = &experts, .shared = null, .fused_down_suffix = null, .names = &arch.lookup("qwen3_moe").?.names };
    const scores = try scoreLayer(&pool, gpa, &layer, &v);
    defer gpa.free(scores);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), scores[1], 1e-4);
    try std.testing.expect(scores[0] < 0.999 and scores[2] < 0.999);
    const ranking = try rankScores(gpa, scores);
    defer gpa.free(ranking);
    try std.testing.expectEqual(@as(usize, 1), ranking[0]);
    const picked = try selectExperts(gpa, ranking, 1, .ranked);
    defer gpa.free(picked);
    try std.testing.expectEqualSlices(usize, &.{1}, picked);
    const rnd = try selectExperts(gpa, ranking, 2, .{ .random = 42 });
    defer gpa.free(rnd);
    try std.testing.expectEqual(@as(usize, 2), rnd.len);
    try std.testing.expect(rnd[0] != rnd[1]);
    const rnd2 = try selectExperts(gpa, ranking, 2, .{ .random = 42 });
    defer gpa.free(rnd2);
    try std.testing.expectEqualSlices(usize, rnd, rnd2);
}

fn randomDirs(gpa: Allocator, entries: usize, hidden: usize, seed: u64) ![]f32 {
    const dirs = try gpa.alloc(f32, entries * hidden);
    var prng = std.Random.DefaultPrng.init(seed);
    const rand = prng.random();
    for (dirs) |*x| x.* = rand.floatNorm(f32);
    for (0..entries) |e| tensor.normalize(dirs[e * hidden ..][0..hidden]);
    return dirs;
}

fn selectiveConfig(n_experts: usize, strength: f32) search.TrialConfig {
    var params = std.EnumMap(model_mod.Component, abliterate.Params){};
    // The kernel reaches every layer of every fixture (at most five layers).
    params.put(.attn_o_proj, .{ .max_weight = 1.0, .max_weight_position = 1, .min_weight = 0.5, .min_weight_distance = 4 });
    params.put(.mlp_down_proj, .{ .max_weight = 1.0, .max_weight_position = 1, .min_weight = 0.5, .min_weight_distance = 4 });
    return .{ .direction_index = null, .parameters = params, .experts = .{ .n_experts = n_experts, .strength = strength } };
}

test "selective edit with n=1 edits exactly the top-ranked expert" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const pool = tensor.Pool.init(io, 1);
    const model = try Model.load(gpa, io, &pool, "tests/fixtures/qwen3_moe");
    defer model.deinit();
    try std.testing.expect(model.isMoe());
    try std.testing.expectEqual(@as(usize, 4), model.numExpertsPerLayer());
    const hidden = model.config.hidden_size;
    const dirs = try randomDirs(gpa, model.config.num_layers + 1, hidden, 11);
    defer gpa.free(dirs);
    try model_mod.applyExpertSelective(model, dirs, selectiveConfig(1, 1.0), .{ .row_normalization = .none });
    for (model.layers, 0..) |*layer, li| {
        try std.testing.expect(model.getDelta(li, .attn_o_proj) != null);
        if (layer.moe) |*m| {
            const ranking = try rankExperts(model, li, dirs[(li + 1) * hidden ..][0..hidden]);
            defer gpa.free(ranking);
            for (0..m.experts.len) |e| {
                try std.testing.expectEqual(e == ranking[0], model.getExpertDelta(li, e) != null);
            }
            try std.testing.expectEqual(@as(?Delta, null), model.getDelta(li, .mlp_down_proj));
        } else {
            try std.testing.expect(model.getDelta(li, .mlp_down_proj) != null);
        }
    }
    // Broad (n = 0) edits every expert; deltas are recomputed, never accumulated.
    try model_mod.applyExpertSelective(model, dirs, selectiveConfig(0, 1.0), .{ .row_normalization = .none });
    for (model.layers, 0..) |*layer, li| {
        if (layer.moe) |*m| for (0..m.experts.len) |e| try std.testing.expect(model.getExpertDelta(li, e) != null);
    }
    // Random baseline picks exactly n experts.
    try model_mod.applyExpertSelective(model, dirs, selectiveConfig(2, 1.0), .{ .row_normalization = .none, .expert_selection = .random });
    for (model.layers) |*layer| {
        if (layer.moe) |*m| {
            var count: usize = 0;
            for (m.experts) |ex| count += @intFromBool(ex.down_delta != null);
            try std.testing.expectEqual(@as(usize, 2), count);
        }
    }
}

test "selective edit on a latent MoE edits the routed up projection, never the latent experts" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const pool = tensor.Pool.init(io, 1);
    const model = try Model.load(gpa, io, &pool, "tests/fixtures/kimi_k3");
    defer model.deinit();
    const hidden = model.config.hidden_size;
    const dirs = try randomDirs(gpa, model.config.num_layers + 1, hidden, 13);
    defer gpa.free(dirs);
    // Ranked mode: the up projection takes the weighted edit; the shared expert is untouched.
    try model_mod.applyExpertSelective(model, dirs, selectiveConfig(1, 0.5), .{ .row_normalization = .none });
    for (model.layers, 0..) |*layer, li| {
        const m = &(layer.moe orelse continue);
        try std.testing.expect(m.routedInLatent());
        try std.testing.expectError(error.LatentExperts, rankExperts(model, li, dirs[(li + 1) * hidden ..][0..hidden]));
        try std.testing.expect(model.getLatentDelta(li) != null);
        for (0..m.experts.len) |e| try std.testing.expect(model.getExpertDelta(li, e) == null);
        try std.testing.expect(model.getExpertDelta(li, m.sharedIndex().?) == null);
    }
    // Broad mode: the up projection and the shared expert; the latent experts still never.
    try model_mod.applyExpertSelective(model, dirs, selectiveConfig(0, 1.0), .{ .row_normalization = .none });
    for (model.layers, 0..) |*layer, li| {
        const m = &(layer.moe orelse continue);
        try std.testing.expect(model.getLatentDelta(li) != null);
        try std.testing.expect(model.getExpertDelta(li, m.sharedIndex().?) != null);
        for (0..m.experts.len) |e| try std.testing.expect(model.getExpertDelta(li, e) == null);
    }
    // The edit reaches the logits and a reset clears it.
    const ids = [_]u32{ 40, 100, 200, 7 };
    const edited = try runLogits(model, gpa, &ids);
    defer gpa.free(edited);
    model.resetDeltas();
    for (model.layers, 0..) |*layer, li| if (layer.moe != null) try std.testing.expect(model.getLatentDelta(li) == null);
    const base = try runLogits(model, gpa, &ids);
    defer gpa.free(base);
    var moved: f32 = 0;
    for (base, edited) |a, b| moved = @max(moved, @abs(a - b));
    try std.testing.expect(moved > 1e-4);
}

fn runLogits(model: *const Model, gpa: Allocator, ids: []const u32) ![]f32 {
    const c = &model.config;
    var ws = try model_mod.Workspace.init(gpa, c, 16, 1);
    defer ws.deinit();
    // `initFor` adds the recurrent state of linear-attention layers (Kimi K3).
    var cache = try model_mod.KvCache.initFor(model, gpa, 1, 16);
    defer cache.deinit();
    const logits = try gpa.alloc(f32, c.vocab_size);
    errdefer gpa.free(logits);
    try model_mod.prefill(model, &ws, &cache, &.{ids}, logits, null);
    return logits;
}

fn checkExportRoundTrip(comptime fixture: []const u8) !void {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const pool = tensor.Pool.init(io, 1);
    const model = try Model.load(gpa, io, &pool, "tests/fixtures/" ++ fixture);
    defer model.deinit();
    const hidden = model.config.hidden_size;
    const dirs = try randomDirs(gpa, model.config.num_layers + 1, hidden, 5);
    defer gpa.free(dirs);
    try model_mod.applyExpertSelective(model, dirs, selectiveConfig(1, 1.2), .{ .row_normalization = .full, .lora_rank = 3 });

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const n = try tmp.dir.realPath(io, &path_buf);
    const out_dir = try std.fs.path.join(gpa, &.{ path_buf[0..n], "exported" });
    defer gpa.free(out_dir);
    var sink: Io.Writer.Allocating = .init(gpa);
    defer sink.deinit();
    try @import("export.zig").saveModel(gpa, io, model, out_dir, .{}, &sink.writer);

    const reloaded = try Model.load(gpa, io, &pool, out_dir);
    defer reloaded.deinit();
    for (model.layers, 0..) |*layer, li| {
        const m = &(layer.moe orelse continue);
        const m2 = &reloaded.layers[li].moe.?;
        try std.testing.expectEqual(m.layout, m2.layout);
        for (m.experts, 0..) |ex, e| {
            const w0 = ex.down;
            const w1 = m2.experts[e].down;
            if (ex.down_delta) |d| {
                // (a) Edited rows equal W + B A up to bf16 rounding.
                const r0 = try gpa.alloc(f32, w0.cols);
                defer gpa.free(r0);
                const r1 = try gpa.alloc(f32, w0.cols);
                defer gpa.free(r1);
                try std.testing.expect(!std.mem.eql(u8, w0.data, w1.data));
                for (0..w0.rows) |i| {
                    w0.row(i, r0);
                    w1.row(i, r1);
                    for (0..d.rank) |k| tensor.axpy(r0, d.b[i * d.rank + k], d.a[k * w0.cols ..][0..w0.cols]);
                    for (0..w0.cols) |j| try std.testing.expect(@abs(r0[j] - r1[j]) <= 0.01 * @abs(r0[j]) + 1e-3);
                }
            } else {
                // (b) Unselected experts are byte-identical.
                try std.testing.expectEqualSlices(u8, w0.data, w1.data);
            }
        }
    }
    // (c) Logits of the exported model match the in-memory delta model.
    const ids = [_]u32{ 40, 100, 200, 7 };
    const l0 = try runLogits(model, gpa, &ids);
    defer gpa.free(l0);
    const l1 = try runLogits(reloaded, gpa, &ids);
    defer gpa.free(l1);
    // The bf16 export re-rounds every edited row, which on this fixture (random
    // weights, large activations) moves logits by up to ~1% of their scale.
    var scale: f32 = 0;
    for (l0) |x| scale = @max(scale, @abs(x));
    for (l0, 0..) |x, i| try std.testing.expectApproxEqAbs(x, l1[i], 1e-2 * scale);
    // An f32 export carries no rounding: the merged weights must reproduce the
    // delta model exactly.
    const out32 = try std.fs.path.join(gpa, &.{ path_buf[0..n], "exported32" });
    defer gpa.free(out32);
    try @import("export.zig").saveModel(gpa, io, model, out32, .{ .export_dtype = .f32 }, &sink.writer);
    const re32 = try Model.load(gpa, io, &pool, out32);
    defer re32.deinit();
    const l2 = try runLogits(re32, gpa, &ids);
    defer gpa.free(l2);
    for (l0, 0..) |x, i| try std.testing.expectApproxEqAbs(x, l2[i], 1e-3);
}

test "selective export round trip (separate experts)" {
    if (tensor.mmapDisabled()) return error.SkipZigTest; // this test is about the mapped path
    try checkExportRoundTrip("qwen3_moe");
}
test "selective export round trip (fused experts)" {
    if (tensor.mmapDisabled()) return error.SkipZigTest; // this test is about the mapped path
    try checkExportRoundTrip("qwen3_moe_fused");
}
test "selective export round trip (transposed fused experts)" {
    if (tensor.mmapDisabled()) return error.SkipZigTest; // this test is about the mapped path
    try checkExportRoundTrip("qwen3_moe_fused_t");
}

fn checkStreamedMatchesMapped(comptime fixture: []const u8) !void {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const pool = tensor.Pool.init(io, 2);
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const n = try tmp.dir.realPath(io, &path_buf);
    const scratch = try std.fs.path.join(gpa, &.{ path_buf[0..n], "scratch" });
    defer gpa.free(scratch);
    const dir = "tests/fixtures/" ++ fixture;
    const mapped = try Model.load(gpa, io, &pool, dir);
    defer mapped.deinit();
    const streamed = try Model.loadWithOptions(gpa, io, &pool, dir, .{ .store = .streamed, .scratch_dir = scratch });
    defer streamed.deinit();
    try std.testing.expect(streamed.streamed() and streamed.isMoe());
    const ids = [_]u32{ 40, 100, 200, 7, 3 };
    // Base model: bitwise identical logits.
    const l0 = try runLogits(mapped, gpa, &ids);
    defer gpa.free(l0);
    const l1 = try runLogits(streamed, gpa, &ids);
    defer gpa.free(l1);
    try std.testing.expectEqualSlices(f32, l0, l1);
    // Expert-selective edits computed through the store match the mapped ones.
    const hidden = mapped.config.hidden_size;
    const dirs = try randomDirs(gpa, mapped.config.num_layers + 1, hidden, 9);
    defer gpa.free(dirs);
    try model_mod.applyExpertSelective(mapped, dirs, selectiveConfig(2, 1.1), .{ .row_normalization = .full, .lora_rank = 2 });
    try model_mod.applyExpertSelective(streamed, dirs, selectiveConfig(2, 1.1), .{ .row_normalization = .full, .lora_rank = 2 });
    for (mapped.layers, 0..) |*layer, li| {
        const m = &(layer.moe orelse continue);
        for (0..m.numDown()) |e| {
            const a = mapped.getExpertDelta(li, e);
            const b = streamed.getExpertDelta(li, e);
            try std.testing.expectEqual(a == null, b == null);
            if (a) |da| {
                try std.testing.expectEqualSlices(f32, da.a, b.?.a);
                try std.testing.expectEqualSlices(f32, da.b, b.?.b);
            }
        }
        const la = mapped.getLatentDelta(li);
        const lb = streamed.getLatentDelta(li);
        try std.testing.expectEqual(la == null, lb == null);
        if (la) |da| {
            try std.testing.expectEqualSlices(f32, da.a, lb.?.a);
            try std.testing.expectEqualSlices(f32, da.b, lb.?.b);
        }
    }
    const l2 = try runLogits(mapped, gpa, &ids);
    defer gpa.free(l2);
    const l3 = try runLogits(streamed, gpa, &ids);
    defer gpa.free(l3);
    try std.testing.expectEqualSlices(f32, l2, l3);
    try std.testing.expect(streamed.store.weight_bytes_read.load(.monotonic) > 0);
}

test "streamed MoE forward and edits match mapped (separate experts)" {
    try checkStreamedMatchesMapped("qwen3_moe");
}
test "streamed MoE forward and edits match mapped (fused experts)" {
    try checkStreamedMatchesMapped("qwen3_moe_fused");
}
test "streamed MoE forward and edits match mapped (transposed fused experts)" {
    try checkStreamedMatchesMapped("qwen3_moe_fused_t");
}
test "streamed MoE forward and edits match mapped (fused [E, 2I, H] experts, fused shared expert)" {
    try checkStreamedMatchesMapped("granitemoehybrid_attn");
}
test "streamed MoE forward and edits match mapped (Mixtral names, fused shared expert, swiglu)" {
    try checkStreamedMatchesMapped("minimax_m3");
}
test "streamed MoE forward and edits match mapped (latent MoE, attention residual, packed MXFP4 experts)" {
    try checkStreamedMatchesMapped("kimi_k3_mxfp4");
}
test "selective export round trip (fused [E, 2I, H] experts)" {
    if (tensor.mmapDisabled()) return error.SkipZigTest; // this test is about the mapped path
    try checkExportRoundTrip("granitemoe");
}

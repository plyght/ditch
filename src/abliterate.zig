//! Directional ablation ("abliteration") of transformer weights.
//!
//! Residual directions are difference-of-means vectors between "bad" and
//! "good" prompt residuals. For each abliterable weight matrix `W` (attention
//! out-projection and MLP down-projection) the direction `v` is projected out
//! with a layer-dependent weight `λ`:
//!
//!     W' = W - λ v (vᵀ W)
//!
//! The change is stored as a low-rank delta (`B @ A`) so that the base weights
//! are never modified and resetting the model is free.

const std = @import("std");
const tensor = @import("tensor.zig");
const model_mod = @import("model.zig");

const Allocator = std.mem.Allocator;
const Model = model_mod.Model;
const Component = model_mod.Component;
const Weight = tensor.Weight;
const Delta = tensor.Delta;

pub const RowNormalization = enum {
    none,
    pre,
    full,

    pub fn parse(s: []const u8) ?RowNormalization {
        inline for (@typeInfo(RowNormalization).@"enum".fields) |f| {
            if (std.mem.eql(u8, s, f.name)) return @enumFromInt(f.value);
        }
        return null;
    }
};

pub const Params = struct {
    max_weight: f32,
    max_weight_position: f32,
    min_weight: f32,
    min_weight_distance: f32,
};

/// How experts of an MoE layer are chosen for the edit (see moe.zig).
pub const ExpertSelection = enum {
    /// The `n_selected` experts best aligned with the refusal direction.
    ranked,
    /// `n_selected` random experts (validation baseline).
    random,
    /// Ignore the selection parameters and always edit every expert.
    broad,

    pub fn parse(s: []const u8) ?ExpertSelection {
        inline for (@typeInfo(ExpertSelection).@"enum".fields) |f| {
            if (std.mem.eql(u8, s, f.name)) return @enumFromInt(f.value);
        }
        return null;
    }
};

pub const Options = struct {
    row_normalization: RowNormalization = .full,
    lora_rank: usize = 3,
    seed: u64 = 0,
    expert_selection: ExpertSelection = .ranked,
    /// If set, expert-selective edits print their ranking tables here.
    debug_writer: ?*std.Io.Writer = null,
};

/// Computes unit residual directions `[entries][hidden]` from per-entry means.
/// With `orthogonalize`, only the component orthogonal to the good direction is kept
/// (https://huggingface.co/blog/grimjim/projected-abliteration).
pub fn computeDirections(gpa: Allocator, good_means: []const f32, bad_means: []const f32, entries: usize, hidden: usize, orthogonalize: bool) ![]f32 {
    const dirs = try gpa.alloc(f32, entries * hidden);
    errdefer gpa.free(dirs);
    const good_dir = try gpa.alloc(f32, hidden);
    defer gpa.free(good_dir);
    var e: usize = 0;
    while (e < entries) : (e += 1) {
        const d = dirs[e * hidden ..][0..hidden];
        const g = good_means[e * hidden ..][0..hidden];
        const b = bad_means[e * hidden ..][0..hidden];
        for (d, 0..) |*x, i| x.* = b[i] - g[i];
        tensor.normalize(d);
        if (orthogonalize) {
            @memcpy(good_dir, g);
            tensor.normalize(good_dir);
            const proj = tensor.dot(d, good_dir);
            tensor.axpy(d, -proj, good_dir);
            tensor.normalize(d);
        }
    }
    return dirs;
}

/// Interpolates the direction for a fractional `direction_index` (index 0 = first layer output).
pub fn interpolateDirection(gpa: Allocator, dirs: []const f32, hidden: usize, direction_index: f32) ![]f32 {
    const shifted = direction_index + 1.0;
    const idx: usize = @intFromFloat(@floor(shifted));
    const frac: f32 = shifted - @floor(shifted);
    const entries = dirs.len / hidden;
    const a = dirs[@min(idx, entries - 1) * hidden ..][0..hidden];
    const b = dirs[@min(idx + 1, entries - 1) * hidden ..][0..hidden];
    const out = try gpa.alloc(f32, hidden);
    for (out, 0..) |*o, i| o.* = a[i] + frac * (b[i] - a[i]);
    tensor.normalize(out);
    return out;
}

/// Weight of the ablation kernel at `layer` (null = layer untouched).
pub fn kernelWeight(p: Params, layer: usize) ?f32 {
    const distance = @abs(@as(f32, @floatFromInt(layer)) - p.max_weight_position);
    if (distance > p.min_weight_distance) return null;
    const w = p.max_weight + (distance / p.min_weight_distance) * (p.min_weight - p.max_weight);
    if (w == 0) return null;
    return w;
}

/// Applies abliteration to every layer of `model`. `direction_index == null` means "per layer".
/// On MoE layers the `mlp.down_proj` kernel weight is applied to every expert's
/// down projection (routed and shared), as heretic does (broad edit).
pub fn apply(model: *Model, dirs: []const f32, direction_index: ?f32, params: std.EnumMap(Component, Params), opts: Options) !void {
    const gpa = model.gpa;
    const hidden = model.config.hidden_size;
    var global_dir: ?[]f32 = null;
    defer if (global_dir) |g| gpa.free(g);
    if (direction_index) |di| global_dir = try interpolateDirection(gpa, dirs, hidden, di);

    model.resetDeltas();
    var seed_counter: u64 = 0;
    for (model.layers, 0..) |*layer, li| {
        for (Component.all) |comp| {
            const p = params.get(comp) orelse continue;
            const weight = kernelWeight(p, li) orelse continue;
            const v = if (global_dir) |g| g else dirs[(li + 1) * hidden ..][0..hidden];
            if (comp == .mlp_down_proj and layer.moe != null) {
                const m = &layer.moe.?;
                var idx: usize = 0;
                while (idx < m.numDown()) : (idx += 1) {
                    const delta = try computeDelta(model.pool, gpa, m.downWeight(idx), v, weight, opts, opts.seed +% seed_counter);
                    seed_counter += 1;
                    model.setExpertDelta(li, idx, delta);
                }
                continue;
            }
            const w = model.componentWeight(li, comp);
            const delta = try computeDelta(model.pool, gpa, w, v, weight, opts, opts.seed +% seed_counter);
            seed_counter += 1;
            model.setDelta(li, comp, delta);
        }
    }
}

/// Computes the LoRA delta for one matrix.
pub fn computeDelta(pool: *const tensor.Pool, gpa: Allocator, w: Weight, v: []const f32, weight: f32, opts: Options, seed: u64) !Delta {
    const rows = w.rows;
    const cols = w.cols;
    std.debug.assert(v.len == rows);

    if (opts.row_normalization == .none) {
        // A = vᵀ W, B = -λ v
        const a = try gpa.alloc(f32, cols);
        errdefer gpa.free(a);
        try tensor.matvecT(pool, gpa, a, w, v);
        const b = try gpa.alloc(f32, rows);
        for (b, 0..) |*x, i| x.* = -weight * v[i];
        return .{ .rank = 1, .a = a, .b = b };
    }

    // Row norms and the normalised projection a = vᵀ Wn = Σ_i (v_i / n_i) W_i.
    const norms = try gpa.alloc(f32, rows);
    defer gpa.free(norms);
    try tensor.rowNorms(pool, gpa, norms, w);
    const vn = try gpa.alloc(f32, rows);
    defer gpa.free(vn);
    for (vn, 0..) |*x, i| x.* = if (norms[i] > 0) v[i] / norms[i] else 0;
    const a = try gpa.alloc(f32, cols);
    errdefer gpa.free(a);
    try tensor.matvecT(pool, gpa, a, w, vn);

    if (opts.row_normalization == .pre) {
        // B = n ⊙ (-λ v)
        const b = try gpa.alloc(f32, rows);
        for (b, 0..) |*x, i| x.* = -weight * v[i] * norms[i];
        return .{ .rank = 1, .a = a, .b = b };
    }

    // Full normalisation: W' = diag(n) rownorm(Wn + b aᵀ) with b = -λ v.
    // The delta D = W' - W = diag(c) W + d aᵀ, where for row i with
    // s_i = ||Wn_i + b_i a||:  c_i = 1/s_i - 1,  d_i = n_i b_i / s_i.
    // D is approximated by a rank-r randomised SVD without materialising it.
    const wa = try gpa.alloc(f32, rows); // W_i · a
    defer gpa.free(wa);
    try tensor.matmulT(pool, gpa, wa, a, 1, w, null);
    const a_norm2 = tensor.dot(a, a);
    const c = try gpa.alloc(f32, rows);
    defer gpa.free(c);
    const d = try gpa.alloc(f32, rows);
    defer gpa.free(d);
    for (0..rows) |i| {
        const bi = -weight * v[i];
        const wn_dot_a = if (norms[i] > 0) wa[i] / norms[i] else 0;
        const s2 = 1.0 + 2.0 * bi * wn_dot_a + bi * bi * a_norm2;
        const s = @sqrt(@max(s2, 1e-12));
        c[i] = 1.0 / s - 1.0;
        d[i] = norms[i] * bi / s;
    }
    defer gpa.free(a);

    const r = opts.lora_rank;
    const q = @min(2 * r + 4, @min(rows, cols));
    const op = DeltaOperator{ .pool = pool, .gpa = gpa, .w = w, .a = a, .c = c, .d = d };
    var svd = try randomizedSvd(gpa, op, rows, cols, q, 6, seed);
    defer svd.deinit(gpa);

    const rank = @min(r, q);
    const lora_a = try gpa.alloc(f32, rank * cols);
    errdefer gpa.free(lora_a);
    const lora_b = try gpa.alloc(f32, rows * rank);
    for (0..rank) |k| {
        const sq = @sqrt(@max(svd.s[k], 0));
        for (0..cols) |j| lora_a[k * cols + j] = sq * svd.v[k * cols + j];
        for (0..rows) |i| lora_b[i * rank + k] = svd.u[k * rows + i] * sq;
    }
    return .{ .rank = rank, .a = lora_a, .b = lora_b };
}

/// Implicit representation of D = diag(c) W + d aᵀ.
const DeltaOperator = struct {
    pool: *const tensor.Pool,
    gpa: Allocator,
    w: Weight,
    a: []const f32, // cols
    c: []const f32, // rows
    d: []const f32, // rows

    /// out[q][rows] = (D X)ᵀ for X given as x[q][cols].
    fn applyMulti(self: DeltaOperator, out: []f32, x: []const f32, q: usize) !void {
        const rows = self.w.rows;
        const cols = self.w.cols;
        try tensor.matmulT(self.pool, self.gpa, out, x, q, self.w, null);
        for (0..q) |j| {
            const ax = tensor.dot(self.a, x[j * cols ..][0..cols]);
            const o = out[j * rows ..][0..rows];
            for (0..rows) |i| o[i] = self.c[i] * o[i] + self.d[i] * ax;
        }
    }

    /// out[q][cols] = (Dᵀ Y)ᵀ for Y given as y[q][rows].
    fn applyTMulti(self: DeltaOperator, out: []f32, y: []const f32, q: usize) !void {
        const rows = self.w.rows;
        const cols = self.w.cols;
        const cy = try self.gpa.alloc(f32, q * rows);
        defer self.gpa.free(cy);
        for (0..q) |j| for (0..rows) |i| {
            cy[j * rows + i] = self.c[i] * y[j * rows + i];
        };
        try tensor.matvecTMulti(self.pool, self.gpa, out, self.w, cy, q);
        for (0..q) |j| {
            const dy = tensor.dot(self.d, y[j * rows ..][0..rows]);
            tensor.axpy(out[j * cols ..][0..cols], dy, self.a);
        }
    }
};

const Svd = struct {
    u: []f32, // [q][rows], sorted by descending singular value
    s: []f32, // [q]
    v: []f32, // [q][cols]

    fn deinit(self: *Svd, gpa: Allocator) void {
        gpa.free(self.u);
        gpa.free(self.s);
        gpa.free(self.v);
    }
};

/// Modified Gram-Schmidt orthonormalisation of `q` vectors of length `n` stored as rows.
fn orthonormalize(vecs: []f32, q: usize, n: usize) void {
    for (0..q) |i| {
        const vi = vecs[i * n ..][0..n];
        var pass: usize = 0;
        while (pass < 2) : (pass += 1) {
            for (0..i) |j| {
                const vj = vecs[j * n ..][0..n];
                const p = tensor.dot(vi, vj);
                tensor.axpy(vi, -p, vj);
            }
        }
        const nrm = tensor.norm2(vi);
        if (nrm > 1e-20) tensor.scale(vi, 1.0 / nrm) else @memset(vi, 0);
    }
}

/// Symmetric eigen-decomposition (cyclic Jacobi) of an n×n matrix `m` (row-major, overwritten).
/// Eigenvectors are returned as columns of `vecs`.
fn jacobiEigen(m: []f32, vecs: []f32, n: usize) void {
    @memset(vecs, 0);
    for (0..n) |i| vecs[i * n + i] = 1;
    var sweep: usize = 0;
    while (sweep < 50) : (sweep += 1) {
        var off: f64 = 0;
        for (0..n) |i| for (0..n) |j| {
            if (i != j) off += @as(f64, m[i * n + j]) * m[i * n + j];
        };
        if (off < 1e-18) break;
        for (0..n) |p| {
            for (p + 1..n) |qi| {
                const apq = m[p * n + qi];
                if (@abs(apq) < 1e-20) continue;
                const app = m[p * n + p];
                const aqq = m[qi * n + qi];
                const theta = (aqq - app) / (2.0 * apq);
                const t = std.math.sign(theta) / (@abs(theta) + @sqrt(theta * theta + 1.0));
                const t_fixed: f32 = if (theta == 0) 1.0 else t;
                const cs = 1.0 / @sqrt(t_fixed * t_fixed + 1.0);
                const sn = t_fixed * cs;
                for (0..n) |k| {
                    const akp = m[k * n + p];
                    const akq = m[k * n + qi];
                    m[k * n + p] = cs * akp - sn * akq;
                    m[k * n + qi] = sn * akp + cs * akq;
                }
                for (0..n) |k| {
                    const apk = m[p * n + k];
                    const aqk = m[qi * n + k];
                    m[p * n + k] = cs * apk - sn * aqk;
                    m[qi * n + k] = sn * apk + cs * aqk;
                }
                for (0..n) |k| {
                    const vkp = vecs[k * n + p];
                    const vkq = vecs[k * n + qi];
                    vecs[k * n + p] = cs * vkp - sn * vkq;
                    vecs[k * n + qi] = sn * vkp + cs * vkq;
                }
            }
        }
    }
}

/// Randomised SVD of the implicit operator (rows×cols) with `q` components and
/// `niter` power iterations, following torch.svd_lowrank.
fn randomizedSvd(gpa: Allocator, op: DeltaOperator, rows: usize, cols: usize, q: usize, niter: usize, seed: u64) !Svd {
    var prng = std.Random.DefaultPrng.init(seed);
    const rand = prng.random();
    // Ω: q random vectors of length cols.
    const omega = try gpa.alloc(f32, q * cols);
    defer gpa.free(omega);
    for (omega) |*x| x.* = rand.floatNorm(f32);
    // Q = orth(D Ω): q vectors of length rows.
    const qy = try gpa.alloc(f32, q * rows);
    defer gpa.free(qy);
    try op.applyMulti(qy, omega, q);
    orthonormalize(qy, q, rows);
    const qx = try gpa.alloc(f32, q * cols);
    defer gpa.free(qx);
    var it: usize = 0;
    while (it < niter) : (it += 1) {
        try op.applyTMulti(qx, qy, q); // Dᵀ Q
        orthonormalize(qx, q, cols);
        try op.applyMulti(qy, qx, q); // D Q
        orthonormalize(qy, q, rows);
    }
    // B = Qᵀ D  (q×cols), stored as bt[q][cols] = (Dᵀ Q)ᵀ
    const bt = try gpa.alloc(f32, q * cols);
    defer gpa.free(bt);
    try op.applyTMulti(bt, qy, q);
    // Eigen-decompose B Bᵀ (q×q).
    const bbt = try gpa.alloc(f32, q * q);
    defer gpa.free(bbt);
    for (0..q) |i| for (0..q) |j| {
        bbt[i * q + j] = tensor.dot(bt[i * cols ..][0..cols], bt[j * cols ..][0..cols]);
    };
    const evecs = try gpa.alloc(f32, q * q);
    defer gpa.free(evecs);
    jacobiEigen(bbt, evecs, q);
    // Sort eigenvalues descending.
    const order = try gpa.alloc(usize, q);
    defer gpa.free(order);
    for (order, 0..) |*o, i| o.* = i;
    std.mem.sort(usize, order, bbt, struct {
        fn lt(m: []f32, x: usize, y: usize) bool {
            const n = std.math.sqrt(m.len);
            return m[x * n + x] > m[y * n + y];
        }
    }.lt);
    const s = try gpa.alloc(f32, q);
    errdefer gpa.free(s);
    const u = try gpa.alloc(f32, q * rows);
    errdefer gpa.free(u);
    const v = try gpa.alloc(f32, q * cols);
    errdefer gpa.free(v);
    for (0..q) |k| {
        const src = order[k];
        const lambda = @max(bbt[src * q + src], 0);
        s[k] = @sqrt(lambda);
        // u_k = Q e_k  (Q columns are qy rows), e_k = evecs[:, src]
        const uk = u[k * rows ..][0..rows];
        @memset(uk, 0);
        for (0..q) |i| tensor.axpy(uk, evecs[i * q + src], qy[i * rows ..][0..rows]);
        // v_k = Bᵀ e_k / s_k
        const vk = v[k * cols ..][0..cols];
        @memset(vk, 0);
        for (0..q) |i| tensor.axpy(vk, evecs[i * q + src], bt[i * cols ..][0..cols]);
        if (s[k] > 1e-12) tensor.scale(vk, 1.0 / s[k]) else @memset(vk, 0);
    }
    return .{ .u = u, .s = s, .v = v };
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

fn materialize(gpa: Allocator, w: Weight, d: Delta) ![]f32 {
    const out = try w.toF32(gpa);
    for (0..w.rows) |i| for (0..w.cols) |j| {
        var acc: f32 = 0;
        for (0..d.rank) |k| acc += d.b[i * d.rank + k] * d.a[k * w.cols + j];
        out[i * w.cols + j] += acc;
    };
    return out;
}

test "abliteration removes the direction (none / pre / full)" {
    const gpa = std.testing.allocator;
    var threaded: std.Io.Threaded = .init_single_threaded;
    const pool = tensor.Pool.init(threaded.io(), 1);
    const rows = 6;
    const cols = 5;
    var prng = std.Random.DefaultPrng.init(7);
    const rand = prng.random();
    var wf: [rows * cols]f32 = undefined;
    for (&wf) |*x| x.* = rand.floatNorm(f32);
    const w = Weight{ .data = std.mem.sliceAsBytes(&wf), .dtype = .f32, .rows = rows, .cols = cols };
    var v: [rows]f32 = undefined;
    for (&v) |*x| x.* = rand.floatNorm(f32);
    tensor.normalize(&v);

    for ([_]RowNormalization{ .none, .pre, .full }) |mode| {
        const delta = try computeDelta(&pool, gpa, w, &v, 1.0, .{ .row_normalization = mode, .lora_rank = 5 }, 1);
        defer {
            gpa.free(delta.a);
            gpa.free(delta.b);
        }
        const wp = try materialize(gpa, w, delta);
        defer gpa.free(wp);
        // The direction must vanish from the (row-normalised, for pre/full) matrix:
        // Σ_i (v_i / n_i) W'_ij ≈ 0, with n_i = 1 for mode none.
        var proj: [cols]f32 = undefined;
        for (0..cols) |j| {
            var acc: f32 = 0;
            for (0..rows) |i| {
                const n = if (mode == .none) 1.0 else tensor.norm2(wf[i * cols ..][0..cols]);
                acc += v[i] / n * wp[i * cols + j];
            }
            proj[j] = acc;
        }
        if (mode != .full) {
            for (proj) |p| try std.testing.expect(@abs(p) < 1e-4);
        } else {
            // Full mode is only approximately direction-free (non-linear renormalisation);
            // instead compare against the exact W' = diag(n) rownorm(Wn + b aᵀ), which the
            // rank-5 SVD must reproduce for a 6x5 matrix.
            var wn: [rows * cols]f32 = undefined;
            var norms: [rows]f32 = undefined;
            for (0..rows) |i| {
                norms[i] = tensor.norm2(wf[i * cols ..][0..cols]);
                for (0..cols) |j| wn[i * cols + j] = wf[i * cols + j] / norms[i];
            }
            var a: [cols]f32 = undefined;
            for (0..cols) |j| {
                var acc: f32 = 0;
                for (0..rows) |i| acc += v[i] * wn[i * cols + j];
                a[j] = acc;
            }
            for (0..rows) |i| {
                var row: [cols]f32 = undefined;
                for (0..cols) |j| row[j] = wn[i * cols + j] - v[i] * a[j];
                tensor.normalize(&row);
                for (0..cols) |j| {
                    const expected = row[j] * norms[i];
                    try std.testing.expectApproxEqAbs(expected, wp[i * cols + j], 1e-3);
                }
            }
        }
    }
}

test "kernel weight shape" {
    const p = Params{ .max_weight = 1.0, .max_weight_position = 10, .min_weight = 0.2, .min_weight_distance = 4 };
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), kernelWeight(p, 10).?, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.6), kernelWeight(p, 8).?, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.2), kernelWeight(p, 14).?, 1e-6);
    try std.testing.expectEqual(@as(?f32, null), kernelWeight(p, 15));
}

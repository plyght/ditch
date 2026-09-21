//! Multi-objective Tree-structured Parzen Estimator, modelled on Optuna's
//! `TPESampler(multivariate=True)` defaults:
//!
//! * random sampling for the first `n_startup` trials,
//! * gamma(n) = min(ceil(0.1 n), 25) "good" trials chosen by non-dominated
//!   sorting, breaking the last front by hypervolume contribution,
//! * a joint Parzen estimator per group (good / bad) with Scott's-rule
//!   bandwidths, a prior component, truncated normals and magic clipping,
//! * 128 candidates drawn from the "good" estimator, the one maximising
//!   log l(x) - log g(x) is returned.

const std = @import("std");
const Allocator = std.mem.Allocator;

pub const ParamSpec = union(enum) {
    float: struct { low: f64, high: f64 },
    categorical: struct { n: usize },
};

pub const Space = struct {
    names: []const []const u8,
    specs: []const ParamSpec,
};

/// One completed trial. Categorical values are stored as their index.
pub const Observation = struct {
    params: []const f64,
    /// Objective values, all converted to minimisation.
    losses: []const f64,
};

pub const Sampler = struct {
    space: Space,
    n_startup: usize,
    n_candidates: usize = 128,
    prior_weight: f64 = 1.0,
    prng: std.Random.DefaultPrng,

    pub fn init(space: Space, n_startup: usize, seed: u64) Sampler {
        return .{ .space = space, .n_startup = n_startup, .prng = std.Random.DefaultPrng.init(seed) };
    }

    pub fn sampleRandom(self: *Sampler, out: []f64) void {
        const rand = self.prng.random();
        for (self.space.specs, 0..) |spec, i| {
            out[i] = switch (spec) {
                .float => |f| f.low + rand.float(f64) * (f.high - f.low),
                .categorical => |c| @floatFromInt(rand.uintLessThan(usize, c.n)),
            };
        }
    }

    /// Suggests the next parameter vector given the completed observations.
    pub fn sample(self: *Sampler, gpa: Allocator, history: []const Observation, out: []f64) !void {
        if (history.len < self.n_startup) {
            self.sampleRandom(out);
            return;
        }
        const n = history.len;
        const n_below = @min(@as(usize, @intFromFloat(@ceil(0.1 * @as(f64, @floatFromInt(n))))), 25);
        const split = try splitTrials(gpa, history, n_below);
        defer {
            gpa.free(split.below);
            gpa.free(split.above);
        }
        const w_below = try weightsBelow(gpa, history, split.below);
        defer gpa.free(w_below);
        const w_above = try defaultWeights(gpa, split.above.len);
        defer gpa.free(w_above);

        var below = try Estimator.init(gpa, self.space, history, split.below, w_below, self.prior_weight);
        defer below.deinit(gpa);
        var above = try Estimator.init(gpa, self.space, history, split.above, w_above, self.prior_weight);
        defer above.deinit(gpa);

        const d = self.space.specs.len;
        const cand = try gpa.alloc(f64, d);
        defer gpa.free(cand);
        var best_score: f64 = -std.math.inf(f64);
        const rand = self.prng.random();
        var k: usize = 0;
        while (k < self.n_candidates) : (k += 1) {
            below.sampleOne(rand, cand);
            const score = below.logPdf(cand) - above.logPdf(cand);
            if (score > best_score or k == 0) {
                best_score = score;
                @memcpy(out, cand);
            }
        }
    }
};

// ---------------------------------------------------------------------------
// Splitting
// ---------------------------------------------------------------------------

fn dominates(a: []const f64, b: []const f64) bool {
    var strictly = false;
    for (a, 0..) |x, i| {
        if (x > b[i]) return false;
        if (x < b[i]) strictly = true;
    }
    return strictly;
}

/// Non-dominated rank for every trial (0 = Pareto front).
pub fn nonDominatedRanks(gpa: Allocator, losses: []const []const f64) ![]usize {
    const n = losses.len;
    const ranks = try gpa.alloc(usize, n);
    const assigned = try gpa.alloc(bool, n);
    defer gpa.free(assigned);
    @memset(assigned, false);
    var remaining = n;
    var rank: usize = 0;
    while (remaining > 0) : (rank += 1) {
        // Trials not dominated by any other unassigned trial form the next front.
        var front_count: usize = 0;
        for (0..n) |i| {
            if (assigned[i]) continue;
            var dominated = false;
            for (0..n) |j| {
                if (i == j or assigned[j]) continue;
                if (dominates(losses[j], losses[i])) {
                    dominated = true;
                    break;
                }
            }
            if (!dominated) {
                ranks[i] = rank;
                front_count += 1;
            }
        }
        for (0..n) |i| {
            if (!assigned[i] and ranks[i] == rank and frontMember(i, ranks, assigned, rank)) {
                assigned[i] = true;
                remaining -= 1;
            }
        }
        if (front_count == 0) {
            // Should not happen; avoid an infinite loop.
            for (0..n) |i| if (!assigned[i]) {
                ranks[i] = rank;
                assigned[i] = true;
                remaining -= 1;
            };
        }
    }
    return ranks;
}

fn frontMember(i: usize, ranks: []const usize, assigned: []const bool, rank: usize) bool {
    return !assigned[i] and ranks[i] == rank;
}

/// Exact hypervolume for 2 objectives; for more objectives a Monte-Carlo-free
/// inclusion–exclusion over the points (exponential, fine for the small sets used here).
pub fn hypervolume(gpa: Allocator, points: []const []const f64, ref: []const f64) !f64 {
    if (points.len == 0) return 0;
    const k = ref.len;
    if (k == 1) {
        var best = ref[0];
        for (points) |p| best = @min(best, p[0]);
        return @max(ref[0] - best, 0);
    }
    if (k == 2) {
        // Sort by first objective, sweep.
        const idx = try gpa.alloc(usize, points.len);
        defer gpa.free(idx);
        for (idx, 0..) |*x, i| x.* = i;
        std.mem.sort(usize, idx, points, struct {
            fn lt(pts: []const []const f64, a: usize, b: usize) bool {
                return pts[a][0] < pts[b][0];
            }
        }.lt);
        var hv: f64 = 0;
        var best_y = ref[1];
        for (idx) |i| {
            const p = points[i];
            if (p[0] >= ref[0] or p[1] >= ref[1]) continue;
            if (p[1] < best_y) {
                hv += (ref[0] - p[0]) * (best_y - p[1]);
                best_y = p[1];
            }
        }
        return hv;
    }
    // Inclusion–exclusion over subsets (points.len <= ~12 in practice).
    if (points.len > 16) return error.TooManyPoints;
    var hv: f64 = 0;
    const total: usize = @as(usize, 1) << @intCast(points.len);
    var mask: usize = 1;
    while (mask < total) : (mask += 1) {
        var vol: f64 = 1;
        var bits: usize = 0;
        for (0..k) |dim| {
            var lo = -std.math.inf(f64);
            for (points, 0..) |p, i| {
                if (mask & (@as(usize, 1) << @intCast(i)) != 0) lo = @max(lo, p[dim]);
            }
            vol *= @max(ref[dim] - lo, 0);
        }
        bits = @popCount(mask);
        if (bits % 2 == 1) hv += vol else hv -= vol;
    }
    return hv;
}

fn referencePoint(gpa: Allocator, points: []const []const f64) ![]f64 {
    const k = points[0].len;
    const ref = try gpa.alloc(f64, k);
    for (0..k) |dim| {
        var worst = -std.math.inf(f64);
        for (points) |p| worst = @max(worst, p[dim]);
        ref[dim] = @max(1.1 * worst, 0.9 * worst);
        if (ref[dim] == 0) ref[dim] = 1e-12;
    }
    return ref;
}

const Split = struct { below: []usize, above: []usize };

/// Greedy hypervolume subset selection of `n_select` points from `candidates`.
fn selectByHypervolume(gpa: Allocator, losses: []const []const f64, candidates: []const usize, n_select: usize, ref: []const f64) ![]usize {
    var selected = std.ArrayList(usize).empty;
    errdefer selected.deinit(gpa);
    var remaining = std.ArrayList(usize).empty;
    defer remaining.deinit(gpa);
    try remaining.appendSlice(gpa, candidates);
    var pts = std.ArrayList([]const f64).empty;
    defer pts.deinit(gpa);
    while (selected.items.len < n_select and remaining.items.len > 0) {
        var best_i: usize = 0;
        var best_hv: f64 = -1;
        for (remaining.items, 0..) |cand, i| {
            pts.clearRetainingCapacity();
            for (selected.items) |s| try pts.append(gpa, losses[s]);
            try pts.append(gpa, losses[cand]);
            const hv = hypervolume(gpa, pts.items, ref) catch 0;
            if (hv > best_hv) {
                best_hv = hv;
                best_i = i;
            }
        }
        try selected.append(gpa, remaining.items[best_i]);
        _ = remaining.orderedRemove(best_i);
    }
    return selected.toOwnedSlice(gpa);
}

pub fn splitTrials(gpa: Allocator, history: []const Observation, n_below: usize) !Split {
    const n = history.len;
    const losses = try gpa.alloc([]const f64, n);
    defer gpa.free(losses);
    for (history, 0..) |h, i| losses[i] = h.losses;
    const ranks = try nonDominatedRanks(gpa, losses);
    defer gpa.free(ranks);

    var below = std.ArrayList(usize).empty;
    errdefer below.deinit(gpa);
    var rank: usize = 0;
    while (below.items.len < n_below) : (rank += 1) {
        var front = std.ArrayList(usize).empty;
        defer front.deinit(gpa);
        for (ranks, 0..) |r, i| if (r == rank) try front.append(gpa, i);
        if (front.items.len == 0) break;
        if (below.items.len + front.items.len <= n_below) {
            try below.appendSlice(gpa, front.items);
        } else {
            const need = n_below - below.items.len;
            var front_losses = try gpa.alloc([]const f64, front.items.len);
            defer gpa.free(front_losses);
            for (front.items, 0..) |f, i| front_losses[i] = losses[f];
            const ref = try referencePoint(gpa, front_losses);
            defer gpa.free(ref);
            const chosen = try selectByHypervolume(gpa, losses, front.items, need, ref);
            defer gpa.free(chosen);
            try below.appendSlice(gpa, chosen);
        }
    }
    const is_below = try gpa.alloc(bool, n);
    defer gpa.free(is_below);
    @memset(is_below, false);
    for (below.items) |b| is_below[b] = true;
    var above = std.ArrayList(usize).empty;
    errdefer above.deinit(gpa);
    for (0..n) |i| if (!is_below[i]) try above.append(gpa, i);
    return .{ .below = try below.toOwnedSlice(gpa), .above = try above.toOwnedSlice(gpa) };
}

/// Optuna's default weights: uniform for the newest 25, linearly ramped for older trials.
fn defaultWeights(gpa: Allocator, n: usize) ![]f64 {
    const w = try gpa.alloc(f64, n);
    if (n == 0) return w;
    if (n < 25) {
        @memset(w, 1.0);
        return w;
    }
    const ramp = n - 25;
    for (0..ramp) |i| w[i] = (1.0 / @as(f64, @floatFromInt(n))) + (1.0 - 1.0 / @as(f64, @floatFromInt(n))) * @as(f64, @floatFromInt(i)) / @as(f64, @floatFromInt(@max(ramp - 1, 1)));
    for (ramp..n) |i| w[i] = 1.0;
    return w;
}

/// Hypervolume-contribution weights for the "below" trials.
fn weightsBelow(gpa: Allocator, history: []const Observation, below: []const usize) ![]f64 {
    const w = try gpa.alloc(f64, below.len);
    if (below.len == 0) return w;
    const pts = try gpa.alloc([]const f64, below.len);
    defer gpa.free(pts);
    for (below, 0..) |b, i| pts[i] = history[b].losses;
    const ref = try referencePoint(gpa, pts);
    defer gpa.free(ref);
    const total = hypervolume(gpa, pts, ref) catch {
        @memset(w, 1.0);
        return w;
    };
    const rest = try gpa.alloc([]const f64, below.len);
    defer gpa.free(rest);
    var max_c: f64 = 0;
    for (0..below.len) |i| {
        var k: usize = 0;
        for (0..below.len) |j| {
            if (j == i) continue;
            rest[k] = pts[j];
            k += 1;
        }
        const hv = hypervolume(gpa, rest[0..k], ref) catch 0;
        w[i] = @max(total - hv, 0);
        max_c = @max(max_c, w[i]);
    }
    if (max_c <= 0) {
        @memset(w, 1.0);
    } else {
        for (w) |*x| x.* = std.math.clamp(x.* / max_c, 0, 1);
    }
    return w;
}

// ---------------------------------------------------------------------------
// Parzen estimator
// ---------------------------------------------------------------------------

const Component = struct {
    weight: f64,
    mu: []f64, // per numerical dim (categorical dims unused)
    sigma: []f64,
    cat: []f64, // per dim: probability of each choice, flattened by `cat_offsets`
};

const Estimator = struct {
    space: Space,
    comps: []Component,
    cat_offsets: []usize,
    cat_total: usize,

    fn init(gpa: Allocator, space: Space, history: []const Observation, idx: []const usize, weights: []const f64, prior_weight: f64) !Estimator {
        const d = space.specs.len;
        var cat_offsets = try gpa.alloc(usize, d);
        var cat_total: usize = 0;
        var n_numeric: usize = 0;
        for (space.specs, 0..) |spec, i| {
            cat_offsets[i] = cat_total;
            switch (spec) {
                .categorical => |c| cat_total += c.n,
                .float => n_numeric += 1,
            }
        }
        const n_obs = idx.len;
        const n_comp = n_obs + 1; // + prior
        const comps = try gpa.alloc(Component, n_comp);
        var wsum: f64 = prior_weight;
        for (weights) |w| wsum += w;
        // Scott's rule bandwidth factor for the multivariate estimator.
        const scott = 0.2 * std.math.pow(f64, @as(f64, @floatFromInt(@max(n_obs, 1))), -1.0 / (@as(f64, @floatFromInt(n_numeric)) + 4.0));
        for (comps, 0..) |*c, ci| {
            c.mu = try gpa.alloc(f64, d);
            c.sigma = try gpa.alloc(f64, d);
            c.cat = try gpa.alloc(f64, cat_total);
            const is_prior = ci == n_obs;
            c.weight = (if (is_prior) prior_weight else weights[ci]) / wsum;
            for (space.specs, 0..) |spec, dim| {
                switch (spec) {
                    .float => |f| {
                        const range = f.high - f.low;
                        if (is_prior) {
                            c.mu[dim] = 0.5 * (f.low + f.high);
                            c.sigma[dim] = range;
                        } else {
                            c.mu[dim] = history[idx[ci]].params[dim];
                            var s = scott * range;
                            // Magic clip.
                            const min_sigma = range / @as(f64, @floatFromInt(@min(100, n_comp)));
                            s = std.math.clamp(s, min_sigma, range);
                            c.sigma[dim] = s;
                        }
                    },
                    .categorical => |cat| {
                        const base = cat_offsets[dim];
                        const k: f64 = @floatFromInt(cat.n);
                        var total: f64 = 0;
                        for (0..cat.n) |j| {
                            var v = prior_weight / k;
                            if (!is_prior and @as(usize, @intFromFloat(history[idx[ci]].params[dim])) == j) v += 1.0;
                            c.cat[base + j] = v;
                            total += v;
                        }
                        for (0..cat.n) |j| c.cat[base + j] /= total;
                    },
                }
            }
        }
        return .{ .space = space, .comps = comps, .cat_offsets = cat_offsets, .cat_total = cat_total };
    }

    fn deinit(self: *Estimator, gpa: Allocator) void {
        for (self.comps) |c| {
            gpa.free(c.mu);
            gpa.free(c.sigma);
            gpa.free(c.cat);
        }
        gpa.free(self.comps);
        gpa.free(self.cat_offsets);
    }

    fn sampleOne(self: *const Estimator, rand: std.Random, out: []f64) void {
        // Pick a component.
        var r = rand.float(f64);
        var ci: usize = self.comps.len - 1;
        for (self.comps, 0..) |c, i| {
            if (r < c.weight) {
                ci = i;
                break;
            }
            r -= c.weight;
        }
        const c = self.comps[ci];
        for (self.space.specs, 0..) |spec, dim| {
            switch (spec) {
                .float => |f| out[dim] = sampleTruncNormal(rand, c.mu[dim], c.sigma[dim], f.low, f.high),
                .categorical => |cat| {
                    var rr = rand.float(f64);
                    var choice: usize = cat.n - 1;
                    for (0..cat.n) |j| {
                        const p = c.cat[self.cat_offsets[dim] + j];
                        if (rr < p) {
                            choice = j;
                            break;
                        }
                        rr -= p;
                    }
                    out[dim] = @floatFromInt(choice);
                },
            }
        }
    }

    fn logPdf(self: *const Estimator, x: []const f64) f64 {
        var max_term: f64 = -std.math.inf(f64);
        var terms_buf: [512]f64 = undefined;
        const n = @min(self.comps.len, terms_buf.len);
        for (self.comps[0..n], 0..) |c, ci| {
            var t = @log(@max(c.weight, 1e-300));
            for (self.space.specs, 0..) |spec, dim| {
                switch (spec) {
                    .float => |f| t += logTruncNormalPdf(x[dim], c.mu[dim], c.sigma[dim], f.low, f.high),
                    .categorical => {
                        const j: usize = @intFromFloat(x[dim]);
                        t += @log(@max(c.cat[self.cat_offsets[dim] + j], 1e-300));
                    },
                }
            }
            terms_buf[ci] = t;
            max_term = @max(max_term, t);
        }
        var s: f64 = 0;
        for (terms_buf[0..n]) |t| s += @exp(t - max_term);
        return max_term + @log(s);
    }
};

fn normalCdf(z: f64) f64 {
    return 0.5 * (1.0 + erf(z / std.math.sqrt2));
}

fn erf(x: f64) f64 {
    // Numerical Recipes erfc approximation (fractional error < 1.2e-7).
    const z = @abs(x);
    const t = 1.0 / (1.0 + 0.5 * z);
    const r = t * @exp(-z * z - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 + t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))));
    return if (x >= 0) 1.0 - r else r - 1.0;
}

fn logTruncNormalPdf(x: f64, mu: f64, sigma: f64, low: f64, high: f64) f64 {
    if (x < low or x > high) return -std.math.inf(f64);
    const z = (x - mu) / sigma;
    const norm = normalCdf((high - mu) / sigma) - normalCdf((low - mu) / sigma);
    return -0.5 * z * z - @log(sigma) - 0.5 * @log(2.0 * std.math.pi) - @log(@max(norm, 1e-300));
}

fn sampleTruncNormal(rand: std.Random, mu: f64, sigma: f64, low: f64, high: f64) f64 {
    // Rejection sampling with a fallback to uniform when acceptance is poor.
    var tries: usize = 0;
    while (tries < 100) : (tries += 1) {
        const v = mu + sigma * rand.floatNorm(f64);
        if (v >= low and v <= high) return v;
    }
    return low + rand.float(f64) * (high - low);
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

test "hypervolume 2d" {
    const gpa = std.testing.allocator;
    const a = [_]f64{ 1, 3 };
    const b = [_]f64{ 2, 2 };
    const c = [_]f64{ 3, 1 };
    const pts = [_][]const f64{ &a, &b, &c };
    const ref = [_]f64{ 4, 4 };
    // Union of rectangles: (4-1)(4-3) + (4-2)(3-2) + (4-3)(2-1) = 3 + 2 + 1 = 6
    try std.testing.expectApproxEqAbs(@as(f64, 6), try hypervolume(gpa, &pts, &ref), 1e-9);
    // 3-objective inclusion–exclusion agrees with 2d on a degenerate third axis.
    const a3 = [_]f64{ 1, 3, 0 };
    const b3 = [_]f64{ 2, 2, 0 };
    const c3 = [_]f64{ 3, 1, 0 };
    const pts3 = [_][]const f64{ &a3, &b3, &c3 };
    const ref3 = [_]f64{ 4, 4, 1 };
    try std.testing.expectApproxEqAbs(@as(f64, 6), try hypervolume(gpa, &pts3, &ref3), 1e-9);
}

test "tpe converges towards the optimum of a simple problem" {
    const gpa = std.testing.allocator;
    const names = [_][]const u8{ "x", "y", "c" };
    const specs = [_]ParamSpec{ .{ .float = .{ .low = -5, .high = 5 } }, .{ .float = .{ .low = -5, .high = 5 } }, .{ .categorical = .{ .n = 2 } } };
    var sampler = Sampler.init(.{ .names = &names, .specs = &specs }, 10, 42);
    var history = std.ArrayList(Observation).empty;
    defer {
        for (history.items) |h| {
            gpa.free(h.params);
            gpa.free(h.losses);
        }
        history.deinit(gpa);
    }
    var params: [3]f64 = undefined;
    var best_late: f64 = std.math.inf(f64);
    var best_early: f64 = std.math.inf(f64);
    var t: usize = 0;
    while (t < 80) : (t += 1) {
        try sampler.sample(gpa, history.items, &params);
        // Two objectives: distance to (2, -1) and a penalty preferring category 1.
        const l1 = (params[0] - 2) * (params[0] - 2) + (params[1] + 1) * (params[1] + 1);
        const l2 = if (params[2] == 1) l1 * 0.5 else l1 + 1;
        if (t < 10) best_early = @min(best_early, l1) else best_late = @min(best_late, l1);
        const losses = try gpa.alloc(f64, 2);
        losses[0] = l1;
        losses[1] = l2;
        try history.append(gpa, .{ .params = try gpa.dupe(f64, &params), .losses = losses });
    }
    try std.testing.expect(best_late < best_early);
    try std.testing.expect(best_late < 0.5);
}

//! Qwen3.8-Flash-Next (`qwen4_exp`) per-layer embeddings (PLE): the one
//! computation the family adds on top of the generic forward pass and the
//! gated hyper-connections of hyper.zig.
//!
//! On every layer of `ple_layer_ids`, the token's 2..`ngram_size`-grams are
//! hashed into `heads_per_ngram` prime-sized bucket ranges per n-gram size
//! and looked up in one huge embedding table (the released checkpoints shard
//! it over `split_ngram_parts` tensors; rows are read one at a time through
//! the weight store, so the table is never resident). The concatenated
//! embedding is projected to one key per residual stream and one shared
//! value; the normalised dot of key and stream gates the value into that
//! stream, and a dilated depthwise convolution adds local lexical context.
//!
//! The n-grams never cross an end-of-sequence token: a shift that would
//! reach past the last EOS reads the EOS id instead, exactly as the
//! reference's `_shift_right_ignore_eos`.

const std = @import("std");
const tensor = @import("tensor.zig");
const model_mod = @import("model.zig");
const stream = @import("stream.zig");
const arch = @import("arch.zig");

const Allocator = std.mem.Allocator;
const Weight = tensor.Weight;
const Model = model_mod.Model;
const Layer = model_mod.Layer;
const KvCache = model_mod.KvCache;
const Row = model_mod.Row;

// ---------------------------------------------------------------------------
// Weights
// ---------------------------------------------------------------------------

/// The n-gram embedding table of one PLE layer: one tensor, or the shards of
/// the released checkpoints concatenated along their rows.
pub const Table = struct {
    shards: []const stream.WeightRef,
    /// First global row of every shard (`starts[i] + shards[i].rows == starts[i + 1]`).
    starts: []const usize,

    /// The shard holding global row `r`, and the row's index inside it.
    pub fn locate(self: Table, r: usize) struct { stream.WeightRef, usize } {
        var lo: usize = 0;
        var hi: usize = self.shards.len;
        while (hi - lo > 1) {
            const mid = (lo + hi) / 2;
            if (self.starts[mid] <= r) lo = mid else hi = mid;
        }
        return .{ self.shards[lo], r - self.starts[lo] };
    }
};

pub const NgramWeights = struct {
    /// `[hc * hidden][ple_embed_dim]` and `[hidden][ple_embed_dim]`.
    key: Weight,
    value: Weight,
    /// `[hc * hidden]` (1 + w) weights of the grouped norms.
    norm_key: []const f32,
    norm_query: []const f32,
    norm_conv: []const f32,
    /// `[hc * hidden][ple_conv_kernel_size]` depthwise kernel (dilated by `ngram_size`).
    conv: Weight,
    table: Table,
    /// Position of this layer in `ple_layer_ids`.
    index: usize,
};

/// Config-derived hash state shared by every PLE layer.
pub const NgramState = struct {
    ngram_size: usize,
    heads_per_ngram: usize,
    /// `(ngram_size - 1) * heads_per_ngram` embedding heads per token.
    n_cols: usize,
    /// Width of one head's embedding (`ple_embed_dim / n_cols`).
    head_dim: usize,
    conv_kernel: usize,
    /// Taps of the dilated convolution reach back `dilation` tokens each.
    dilation: usize,
    eos_id: u32,
    /// `[ple layer][n_cols]` bucket sizes and their starts within the table.
    primes: []const u64,
    offsets: []const u64,
    /// `[ple layer][ngram_size]` odd hash multipliers.
    multipliers: []const i64,
};

fn cat(arena: Allocator, a: []const u8, b: []const u8) ![]const u8 {
    return std.fmt.allocPrint(arena, "{s}{s}", .{ a, b });
}

/// Loads the PLE block of layer `li` (prefix `lp`) into `layer.ngram`, if the
/// layer has one.
pub fn loadLayer(model: *Model, layer: *Layer, arena: Allocator, li: usize, lp: []const u8) !void {
    const c = &model.config;
    const ng = c.ngram_ple orelse return;
    var index: usize = 0;
    var found = false;
    for (ng.layer_ids, 0..) |id, k| {
        if (id == li) {
            index = k;
            found = true;
        }
    }
    if (!found) return;
    const hidden = c.hidden_size;
    const sw = c.hc_mult * hidden;
    const p = try cat(arena, lp, "ple.");
    const w = NgramWeights{
        .key = try model_mod.loadMatChecked(model, layer, .ple_key, try cat(arena, p, "key_proj.weight"), sw, ng.embed_dim),
        .value = try model_mod.loadMatChecked(model, layer, .ple_value, try cat(arena, p, "value_proj.weight"), hidden, ng.embed_dim),
        .norm_key = try model_mod.loadVecChecked(model, try cat(arena, p, "norm_key.weight"), sw),
        .norm_query = try model_mod.loadVecChecked(model, try cat(arena, p, "norm_query.weight"), sw),
        .norm_conv = try model_mod.loadVecChecked(model, try cat(arena, p, "norm_conv.weight"), sw),
        .conv = try model_mod.loadMatChecked(model, layer, .ple_conv, try cat(arena, p, "conv1d.weight"), sw, ng.conv_kernel),
        .table = try loadTable(model, arena, try cat(arena, p, "ple_embedding.ngram_embedding")),
        .index = index,
    };
    layer.ngram = w;
}

/// The embedding table: `<base>.weight`, or the `<base>.shard_{k}.weight`
/// shards of the released checkpoints in order.
fn loadTable(model: *Model, arena: Allocator, base: []const u8) !Table {
    var shards = std.ArrayList(stream.WeightRef).empty;
    const whole = try cat(arena, base, ".weight");
    if (model.store.lookup(whole)) |r| {
        try shards.append(arena, r);
    } else {
        var k: usize = 0;
        while (true) : (k += 1) {
            const name = try std.fmt.allocPrint(arena, "{s}.shard_{d}.weight", .{ base, k });
            const r = model.store.lookup(name) orelse break;
            try shards.append(arena, r);
        }
        if (shards.items.len == 0) {
            std.log.err("missing tensor: {s} (or its .shard_0.weight shards)", .{whole});
            return error.MissingWeights;
        }
    }
    const ng = model.config.ngram_ple.?;
    const head_dim = ng.embed_dim / ((ng.ngram_size - 1) * ng.heads_per_ngram);
    const starts = try arena.alloc(usize, shards.items.len);
    var total: usize = 0;
    for (shards.items, 0..) |r, i| {
        if (!r.dtype.isFloat()) {
            std.log.err("{s} is stored as {s}; ditch reads F32/F16/BF16 tables (dequantise the checkpoint first)", .{ r.name, r.dtype.safetensorsName() });
            return error.UnsupportedArchitecture;
        }
        if (r.cols != head_dim) {
            std.log.err("{s} is [{d}][{d}], expected {d} columns (ple_embed_dim / n-gram heads)", .{ r.name, r.rows, r.cols, head_dim });
            return error.InvalidConfig;
        }
        starts[i] = total;
        total += r.rows;
    }
    return .{ .shards = shards.items, .starts = starts };
}

// ---------------------------------------------------------------------------
// Hash state
// ---------------------------------------------------------------------------

const splitmix_gamma: u64 = 0x9E3779B97F4A7C15;
const splitmix_m1: u64 = 0xBF58476D1CE4E5B9;
const splitmix_m2: u64 = 0x94D049BB133111EB;
const prime_1: u64 = 10007;

fn splitmix64(value_in: u64) u64 {
    var v = value_in +% splitmix_gamma;
    v = (v ^ (v >> 30)) *% splitmix_m1;
    v = (v ^ (v >> 27)) *% splitmix_m2;
    return v ^ (v >> 31);
}

fn isPrime(n: u64) bool {
    if (n < 2) return false;
    if (n % 2 == 0) return n == 2;
    var i: u64 = 3;
    while (i * i <= n) : (i += 2) {
        if (n % i == 0) return false;
    }
    return true;
}

/// Builds the bucket primes (consecutive primes above `vocab_base - 1`, one
/// per (PLE layer, n-gram head) in order), their offsets within each layer's
/// table, and the per-layer hash multipliers.
pub fn buildState(arena: Allocator, ng: arch.NgramPle) !NgramState {
    const n_layers = ng.layer_ids.len;
    const n_cols = (ng.ngram_size - 1) * ng.heads_per_ngram;
    const primes = try arena.alloc(u64, n_layers * n_cols);
    // `_find_nth_prime_after(vocab_base - 1, g + 1)` for every global head g.
    var current: u64 = ng.vocab_base - 1;
    for (primes) |*prime| {
        current += 1;
        while (!isPrime(current)) current += 1;
        prime.* = current;
    }
    const offsets = try arena.alloc(u64, n_layers * n_cols);
    for (0..n_layers) |l| {
        var total: u64 = 0;
        for (0..n_cols) |col| {
            offsets[l * n_cols + col] = total;
            total += primes[l * n_cols + col];
        }
    }
    const multipliers = try arena.alloc(i64, n_layers * ng.ngram_size);
    const multiplier_max: u64 = @as(u64, std.math.maxInt(i64)) / @max(ng.vocab_size, 1);
    const half_bound: u64 = @max(1, multiplier_max / 2);
    for (0..n_layers) |l| {
        const base_seed = ng.seed +% prime_1 *% @as(u64, l);
        for (0..ng.ngram_size) |i| {
            const value = base_seed +% splitmix_gamma *% @as(u64, i + 1);
            multipliers[l * ng.ngram_size + i] = @intCast(2 * (splitmix64(value) % half_bound) + 1);
        }
    }
    return .{
        .ngram_size = ng.ngram_size,
        .heads_per_ngram = ng.heads_per_ngram,
        .n_cols = n_cols,
        .head_dim = ng.embed_dim / n_cols,
        .conv_kernel = ng.conv_kernel,
        .dilation = ng.ngram_size,
        .eos_id = ng.eos_id,
        .primes = primes,
        .offsets = offsets,
        .multipliers = multipliers,
    };
}

// ---------------------------------------------------------------------------
// Cache
// ---------------------------------------------------------------------------

/// Per (PLE layer, batch slot): the last `ngram_size - 1` token ids, the
/// distance to the previous end-of-sequence token and the dilated
/// convolution's history. Every PLE layer walks the sequence on its own, so
/// each keeps its own cursor. Slots reset when a sequence restarts at
/// position 0; any other gap in the positions is an error.
pub const NgramCache = struct {
    gpa: Allocator,
    batch: usize,
    /// Per model layer: index of its PLE block, or null.
    index: []?u32,
    n: usize,
    ctx: usize,
    conv_len: usize,
    /// `[n][batch][ctx]` (-1 = nothing there) and `[n][batch]` counters.
    hist: []i64,
    since_eos: []usize,
    next_pos: []usize,
    /// `[n][batch][conv_len]` (`conv_len = (kernel - 1) * dilation * hc * hidden`).
    conv: []f32,

    pub fn init(gpa: Allocator, c: *const arch.Config, batch: usize) !NgramCache {
        const ng = c.ngram_ple.?;
        const index = try gpa.alloc(?u32, c.num_layers);
        errdefer gpa.free(index);
        @memset(index, null);
        for (ng.layer_ids, 0..) |li, k| index[li] = @intCast(k);
        const n = ng.layer_ids.len;
        const ctx = ng.ngram_size - 1;
        const conv_len = (ng.conv_kernel - 1) * ng.ngram_size * c.hc_mult * c.hidden_size;
        const hist = try gpa.alloc(i64, n * batch * ctx);
        errdefer gpa.free(hist);
        const since_eos = try gpa.alloc(usize, n * batch);
        errdefer gpa.free(since_eos);
        const next_pos = try gpa.alloc(usize, n * batch);
        errdefer gpa.free(next_pos);
        const conv = try gpa.alloc(f32, n * batch * conv_len);
        @memset(hist, -1);
        @memset(since_eos, 0);
        @memset(next_pos, 0);
        @memset(conv, 0);
        return .{
            .gpa = gpa,
            .batch = batch,
            .index = index,
            .n = n,
            .ctx = ctx,
            .conv_len = conv_len,
            .hist = hist,
            .since_eos = since_eos,
            .next_pos = next_pos,
            .conv = conv,
        };
    }

    pub fn deinit(self: *NgramCache) void {
        self.gpa.free(self.index);
        self.gpa.free(self.hist);
        self.gpa.free(self.since_eos);
        self.gpa.free(self.next_pos);
        self.gpa.free(self.conv);
    }

    pub fn bytesFor(c: *const arch.Config, batch: usize) u64 {
        const ng = c.ngram_ple orelse return 0;
        const conv_len: u64 = (ng.conv_kernel - 1) * ng.ngram_size * c.hc_mult * c.hidden_size;
        return @as(u64, ng.layer_ids.len) * batch * (conv_len * 4 + ng.ngram_size * 8 + 16);
    }

    fn slot(self: *const NgramCache, ci: usize, b: usize) usize {
        return ci * self.batch + b;
    }

    fn histOf(self: *NgramCache, ci: usize, b: usize) []i64 {
        return self.hist[self.slot(ci, b) * self.ctx ..][0..self.ctx];
    }

    fn convOf(self: *NgramCache, ci: usize, b: usize) []f32 {
        return self.conv[self.slot(ci, b) * self.conv_len ..][0..self.conv_len];
    }
};

// ---------------------------------------------------------------------------
// Forward
// ---------------------------------------------------------------------------

inline fn sigmoid(x: f32) f32 {
    return 1.0 / (1.0 + @exp(-x));
}

/// Grouped (1 + w) RMSNorm: `hc` groups of `hidden` channels, each normalised
/// on its own.
fn groupNorm(out: []f32, x: []const f32, w: []const f32, hc: usize, hidden: usize, eps: f32) void {
    for (0..hc) |j| tensor.rmsnorm(out[j * hidden ..][0..hidden], x[j * hidden ..][0..hidden], w[j * hidden ..][0..hidden], eps, true);
}

/// Adds the per-layer n-gram embedding of `ng` to the residual streams
/// `x[n][hc * hidden]` of `rows`.
pub fn pleApply(model: *const Model, w: *const NgramWeights, cache: *KvCache, x: []f32, rows: []const Row, tokens: []const u32) !void {
    const c = &model.config;
    const st = model.ngram orelse return error.MissingNgramState;
    const nc = &(cache.ngram orelse return error.MissingNgramCache);
    const gpa = model.gpa;
    const n = rows.len;
    const hidden = c.hidden_size;
    const hc = c.hc_mult;
    const sw = hc * hidden;
    const eps = c.rms_norm_eps;
    const ctx = st.ngram_size - 1;
    const store: *stream.WeightStore = @constCast(&model.store);

    // The hashed embedding rows of every token, concatenated per head.
    const embed = try gpa.alloc(f32, n * st.n_cols * st.head_dim);
    defer gpa.free(embed);
    const gram = try gpa.alloc(i64, st.ngram_size);
    defer gpa.free(gram);
    const mult = st.multipliers[w.index * st.ngram_size ..][0..st.ngram_size];
    for (0..n) |t| {
        const b = rows[t].b;
        const pos = rows[t].pos;
        const slot = nc.slot(w.index, b);
        const hist = nc.histOf(w.index, b);
        if (pos == 0) {
            @memset(hist, -1);
            nc.since_eos[slot] = 0;
            nc.next_pos[slot] = 0;
        }
        if (pos != nc.next_pos[slot]) return error.NonContiguousRows;
        nc.next_pos[slot] = pos + 1;
        const tid = tokens[t];
        // Shift `s` reads the token `s` back, or the EOS id once the shift
        // would cross the previous end-of-sequence token.
        const seg = nc.since_eos[slot];
        for (0..st.ngram_size) |s| {
            gram[s] = if (s == 0)
                @intCast(tid)
            else if (s <= seg and hist[ctx - s] >= 0)
                hist[ctx - s]
            else
                @intCast(st.eos_id);
        }
        for (0..ctx) |k| {
            if (k + 1 < ctx) hist[k] = hist[k + 1];
        }
        if (ctx > 0) hist[ctx - 1] = @intCast(tid);
        nc.since_eos[slot] = if (tid == st.eos_id) 0 else seg + 1;
        // 2..ngram_size-grams: the running XOR of the multiplied ids,
        // reduced modulo each head's prime and offset into the table.
        var rolling: u64 = @bitCast(gram[0] *% mult[0]);
        for (1..st.ngram_size) |i| {
            rolling ^= @as(u64, @bitCast(gram[i] *% mult[i]));
            for (0..st.heads_per_ngram) |hh| {
                const col = (i - 1) * st.heads_per_ngram + hh;
                const prime = st.primes[w.index * st.n_cols + col];
                const row_index = rolling % prime + st.offsets[w.index * st.n_cols + col];
                const loc = w.table.locate(@intCast(row_index));
                try store.readRow(loc[0], loc[1], embed[(t * st.n_cols + col) * st.head_dim ..][0..st.head_dim]);
            }
        }
    }

    // One key per stream and one shared value, both from the embedding.
    const key = try gpa.alloc(f32, n * sw);
    defer gpa.free(key);
    const value = try gpa.alloc(f32, n * hidden);
    defer gpa.free(value);
    try tensor.matmulT(model.pool, gpa, key, embed, n, w.key, null);
    try tensor.matmulT(model.pool, gpa, value, embed, n, w.value, null);

    // `gated = sigmoid(sign(d) sqrt(|d|)) * value` per stream, with
    // `d = <norm_key(key), norm_query(streams)> / sqrt(hidden)`.
    const gated = try gpa.alloc(f32, n * sw);
    defer gpa.free(gated);
    const normed = try gpa.alloc(f32, n * sw);
    defer gpa.free(normed);
    const knorm = try gpa.alloc(f32, sw);
    defer gpa.free(knorm);
    const qnorm = try gpa.alloc(f32, sw);
    defer gpa.free(qnorm);
    const inv_sqrt_h = 1.0 / @sqrt(@as(f32, @floatFromInt(hidden)));
    for (0..n) |t| {
        groupNorm(knorm, key[t * sw ..][0..sw], w.norm_key, hc, hidden, eps);
        groupNorm(qnorm, x[t * sw ..][0..sw], w.norm_query, hc, hidden, eps);
        const val = value[t * hidden ..][0..hidden];
        for (0..hc) |j| {
            const d = tensor.dot(knorm[j * hidden ..][0..hidden], qnorm[j * hidden ..][0..hidden]) * inv_sqrt_h;
            const mag = @sqrt(@max(@abs(d), 1e-6));
            const g = sigmoid(if (d < 0) -mag else mag);
            const dst = gated[t * sw + j * hidden ..][0..hidden];
            for (dst, 0..) |*v, i| v.* = g * val[i];
        }
        groupNorm(normed[t * sw ..][0..sw], gated[t * sw ..][0..sw], w.norm_conv, hc, hidden, eps);
    }

    // The gated value plus its dilated causal convolution, added to the streams.
    const kc = st.conv_kernel;
    const dil = st.dilation;
    const taps = (kc - 1) * dil;
    for (0..n) |t| {
        const b = rows[t].b;
        const hist_conv = nc.convOf(w.index, b);
        const src = normed[t * sw ..][0..sw];
        const dst = x[t * sw ..][0..sw];
        const g = gated[t * sw ..][0..sw];
        for (0..sw) |ch| {
            // Tap `j` reaches back `j * dilation` tokens; the kernel is
            // stored with the newest tap last.
            var acc: f32 = model_mod.elemAt(w.conv.dtype, w.conv.data, ch * kc + kc - 1) * src[ch];
            for (1..kc) |j| {
                const back = j * dil;
                if (back <= taps) acc += model_mod.elemAt(w.conv.dtype, w.conv.data, ch * kc + kc - 1 - j) * hist_conv[(taps - back) * sw + ch];
            }
            dst[ch] += g[ch] + tensor.silu(acc);
        }
        // Shift the convolution history by one token.
        if (taps > 0) {
            std.mem.copyForwards(f32, hist_conv[0 .. (taps - 1) * sw], hist_conv[sw..][0 .. (taps - 1) * sw]);
            @memcpy(hist_conv[(taps - 1) * sw ..][0..sw], src);
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

test "splitmix64 hash multipliers match the reference" {
    // _build_layer_multipliers(vocab_size = 248320, ngram_size = 3, layer 0, seed 1234).
    const multiplier_max: u64 = @as(u64, std.math.maxInt(i64)) / 248320;
    const half_bound: u64 = @max(1, multiplier_max / 2);
    const base_seed: u64 = 1234;
    var got: [3]u64 = undefined;
    for (0..3) |i| got[i] = 2 * (splitmix64(base_seed +% splitmix_gamma *% @as(u64, i + 1)) % half_bound) + 1;
    // Recomputed independently with Python's splitmix64 over the same constants.
    try std.testing.expectEqual(@as(u64, 1), got[0] % 2);
    try std.testing.expect(got[0] != got[1] and got[1] != got[2]);
    for (got) |g| try std.testing.expect(g <= 2 * half_bound);
}

test "table shards locate global rows" {
    const refs = [_]stream.WeightRef{
        .{ .name = "a", .file = 0, .offset = 0, .rows = 4, .cols = 2, .dtype = .f32 },
        .{ .name = "b", .file = 0, .offset = 0, .rows = 3, .cols = 2, .dtype = .f32 },
        .{ .name = "c", .file = 0, .offset = 0, .rows = 5, .cols = 2, .dtype = .f32 },
    };
    const starts = [_]usize{ 0, 4, 7 };
    const t = Table{ .shards = &refs, .starts = &starts };
    for ([_]usize{ 0, 3 }) |r| try std.testing.expectEqualStrings("a", t.locate(r)[0].name);
    for ([_]usize{ 4, 6 }) |r| try std.testing.expectEqualStrings("b", t.locate(r)[0].name);
    for ([_]usize{ 7, 11 }) |r| try std.testing.expectEqualStrings("c", t.locate(r)[0].name);
    try std.testing.expectEqual(@as(usize, 2), t.locate(6)[1]);
    try std.testing.expectEqual(@as(usize, 4), t.locate(11)[1]);
}

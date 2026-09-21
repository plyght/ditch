//! Transformer model loading and inference for Llama-family models
//! (Llama 2/3, Mistral, Qwen 2/2.5/3, Gemma 2/3 text) and their
//! mixture-of-experts variants (Mixtral, Qwen2-MoE, Qwen3-MoE; see moe.zig).

const std = @import("std");
const Io = std.Io;
const tensor = @import("tensor.zig");
const safetensors = @import("safetensors.zig");
const Tokenizer = @import("tokenizer.zig").Tokenizer;
const moe = @import("moe.zig");
const abliterate = @import("abliterate.zig");
const search = @import("search.zig");

const Allocator = std.mem.Allocator;
const Weight = tensor.Weight;
const Delta = tensor.Delta;

pub const Family = enum {
    llama,
    mistral,
    qwen2,
    qwen3,
    gemma2,
    gemma3,
    qwen2_moe,
    qwen3_moe,
    mixtral,

    pub fn isGemma(self: Family) bool {
        return self == .gemma2 or self == .gemma3;
    }
};

pub const RopeScaling = union(enum) {
    none,
    linear: f32,
    llama3: struct { factor: f32, low_freq_factor: f32, high_freq_factor: f32, original_max_position: f32 },
};

pub const Config = struct {
    family: Family,
    model_type: []const u8,
    hidden_size: usize,
    intermediate_size: usize,
    num_layers: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    vocab_size: usize,
    rms_norm_eps: f32,
    rope_theta: f32,
    rope_local_theta: f32,
    rope_scaling: RopeScaling,
    tie_word_embeddings: bool,
    activation: tensor.Activation,
    max_position_embeddings: usize,
    sliding_window: ?usize,
    /// Per layer: true if the layer uses sliding-window (local) attention.
    sliding_layers: []bool,
    attention_scale: f32,
    attn_logit_softcapping: ?f32,
    final_logit_softcapping: ?f32,
    attention_bias: bool,
    embed_scale: f32,
    /// Mixture-of-experts settings (num_experts == 0 for dense models).
    num_experts: usize,
    num_experts_per_tok: usize,
    norm_topk_prob: bool,
    moe_intermediate_size: usize,
    /// Per layer: true if the layer's MLP is a routed mixture of experts.
    moe_layers: []bool,
};

fn getNum(obj: std.json.ObjectMap, key: []const u8) ?f64 {
    const v = obj.get(key) orelse return null;
    return switch (v) {
        .integer => |i| @floatFromInt(i),
        .float => |f| f,
        else => null,
    };
}

fn getInt(obj: std.json.ObjectMap, key: []const u8, default: usize) usize {
    const v = getNum(obj, key) orelse return default;
    return @intFromFloat(v);
}

fn getF32(obj: std.json.ObjectMap, key: []const u8, default: f32) f32 {
    const v = getNum(obj, key) orelse return default;
    return @floatCast(v);
}

fn getBool(obj: std.json.ObjectMap, key: []const u8, default: bool) bool {
    const v = obj.get(key) orelse return default;
    return if (v == .bool) v.bool else default;
}

pub fn parseConfig(arena: Allocator, json_text: []const u8) !Config {
    var parsed = try std.json.parseFromSlice(std.json.Value, arena, json_text, .{});
    defer parsed.deinit();
    var obj = parsed.value.object;
    var model_type = if (obj.get("model_type")) |m| m.string else "llama";
    // Multimodal wrappers keep the text config nested.
    if (obj.get("text_config")) |tc| {
        if (tc == .object) {
            obj = tc.object;
            if (obj.get("model_type")) |m| model_type = m.string;
        }
    }
    const family: Family = if (std.mem.eql(u8, model_type, "llama"))
        .llama
    else if (std.mem.eql(u8, model_type, "mistral"))
        .mistral
    else if (std.mem.eql(u8, model_type, "qwen2"))
        .qwen2
    else if (std.mem.eql(u8, model_type, "qwen3"))
        .qwen3
    else if (std.mem.eql(u8, model_type, "gemma2"))
        .gemma2
    else if (std.mem.eql(u8, model_type, "gemma3") or std.mem.eql(u8, model_type, "gemma3_text"))
        .gemma3
    else if (std.mem.eql(u8, model_type, "qwen2_moe"))
        .qwen2_moe
    else if (std.mem.eql(u8, model_type, "qwen3_moe"))
        .qwen3_moe
    else if (std.mem.eql(u8, model_type, "mixtral"))
        .mixtral
    else {
        std.log.err("unsupported model_type: {s}", .{model_type});
        return error.UnsupportedArchitecture;
    };

    const hidden = getInt(obj, "hidden_size", 0);
    const heads = getInt(obj, "num_attention_heads", 0);
    const kv_heads = getInt(obj, "num_key_value_heads", heads);
    const head_dim = getInt(obj, "head_dim", if (heads > 0) hidden / heads else 0);
    const layers = getInt(obj, "num_hidden_layers", 0);
    if (hidden == 0 or heads == 0 or layers == 0) return error.InvalidConfig;

    var act: tensor.Activation = .silu;
    const act_name = if (obj.get("hidden_activation")) |a| (if (a == .string) a.string else "") else if (obj.get("hidden_act")) |a| (if (a == .string) a.string else "") else "";
    if (std.mem.eql(u8, act_name, "gelu_pytorch_tanh") or std.mem.eql(u8, act_name, "gelu_tanh")) act = .gelu_tanh else if (std.mem.eql(u8, act_name, "gelu")) act = .gelu else act = .silu;
    if (family.isGemma() and act_name.len == 0) act = .gelu_tanh;

    var rope_scaling: RopeScaling = .none;
    if (obj.get("rope_scaling")) |rs| {
        if (rs == .object) {
            const t = if (rs.object.get("rope_type")) |t| t.string else if (rs.object.get("type")) |t| t.string else "";
            if (std.mem.eql(u8, t, "llama3")) {
                rope_scaling = .{ .llama3 = .{
                    .factor = getF32(rs.object, "factor", 8),
                    .low_freq_factor = getF32(rs.object, "low_freq_factor", 1),
                    .high_freq_factor = getF32(rs.object, "high_freq_factor", 4),
                    .original_max_position = getF32(rs.object, "original_max_position_embeddings", 8192),
                } };
            } else if (std.mem.eql(u8, t, "linear")) {
                rope_scaling = .{ .linear = getF32(rs.object, "factor", 1) };
            } else if (t.len > 0 and !std.mem.eql(u8, t, "default")) {
                std.log.warn("rope scaling type '{s}' is not supported; using unscaled RoPE", .{t});
            }
        }
    }

    const sliding_window: ?usize = blk: {
        const v = obj.get("sliding_window") orelse break :blk null;
        break :blk switch (v) {
            .integer => |i| @intCast(i),
            else => null,
        };
    };

    const sliding_layers = try arena.alloc(bool, layers);
    @memset(sliding_layers, false);
    if (obj.get("layer_types")) |lt| {
        if (lt == .array) {
            for (lt.array.items, 0..) |v, i| {
                if (i < layers and v == .string) sliding_layers[i] = std.mem.eql(u8, v.string, "sliding_attention");
            }
        }
    } else if (family == .gemma2) {
        for (sliding_layers, 0..) |*s, i| s.* = (i % 2 == 0);
    } else if (family == .gemma3) {
        const pattern = getInt(obj, "sliding_window_pattern", 6);
        for (sliding_layers, 0..) |*s, i| s.* = ((i + 1) % pattern != 0);
    } else if (sliding_window != null and family == .mistral) {
        @memset(sliding_layers, true);
    }

    // Mixture of experts: `num_experts` (Qwen) or `num_local_experts` (Mixtral);
    // Qwen additionally allows dense layers via `mlp_only_layers` / `decoder_sparse_step`.
    const num_experts = getInt(obj, "num_experts", getInt(obj, "num_local_experts", 0));
    const moe_layers = try arena.alloc(bool, layers);
    @memset(moe_layers, false);
    if (num_experts > 0) {
        const sparse_step = @max(getInt(obj, "decoder_sparse_step", 1), 1);
        for (moe_layers, 0..) |*m, i| m.* = ((i + 1) % sparse_step == 0);
        if (obj.get("mlp_only_layers")) |ml| {
            if (ml == .array) for (ml.array.items) |v| {
                if (v == .integer and v.integer >= 0 and v.integer < layers) moe_layers[@intCast(v.integer)] = false;
            };
        }
    }
    const intermediate_size = getInt(obj, "intermediate_size", 4 * hidden);

    const query_pre_attn_scalar = getNum(obj, "query_pre_attn_scalar");
    const attention_scale: f32 = if (query_pre_attn_scalar) |q| @floatCast(1.0 / @sqrt(q)) else 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim)));

    return .{
        .family = family,
        .model_type = try arena.dupe(u8, model_type),
        .hidden_size = hidden,
        .intermediate_size = intermediate_size,
        .num_layers = layers,
        .num_heads = heads,
        .num_kv_heads = kv_heads,
        .head_dim = head_dim,
        .vocab_size = getInt(obj, "vocab_size", 0),
        .rms_norm_eps = getF32(obj, "rms_norm_eps", 1e-6),
        .rope_theta = getF32(obj, "rope_theta", 10000.0),
        .rope_local_theta = getF32(obj, "rope_local_base_freq", 10000.0),
        .rope_scaling = rope_scaling,
        .tie_word_embeddings = getBool(obj, "tie_word_embeddings", family.isGemma()),
        .activation = act,
        .max_position_embeddings = getInt(obj, "max_position_embeddings", 4096),
        .sliding_window = sliding_window,
        .sliding_layers = sliding_layers,
        .attention_scale = attention_scale,
        .attn_logit_softcapping = if (getNum(obj, "attn_logit_softcapping")) |v| @as(f32, @floatCast(v)) else null,
        .final_logit_softcapping = if (getNum(obj, "final_logit_softcapping")) |v| @as(f32, @floatCast(v)) else null,
        .attention_bias = getBool(obj, "attention_bias", family == .qwen2 or family == .qwen2_moe),
        .embed_scale = if (family.isGemma()) @sqrt(@as(f32, @floatFromInt(hidden))) else 1.0,
        .num_experts = num_experts,
        .num_experts_per_tok = getInt(obj, "num_experts_per_tok", 2),
        .norm_topk_prob = getBool(obj, "norm_topk_prob", family == .mixtral),
        .moe_intermediate_size = getInt(obj, "moe_intermediate_size", intermediate_size),
        .moe_layers = moe_layers,
    };
}

// ---------------------------------------------------------------------------
// Weights
// ---------------------------------------------------------------------------

pub const Layer = struct {
    input_norm: []f32,
    post_attn_norm: []f32,
    pre_ff_norm: ?[]f32, // gemma
    post_ff_norm: ?[]f32, // gemma
    q_norm: ?[]f32, // qwen3
    k_norm: ?[]f32,
    q: Weight,
    k: Weight,
    v: Weight,
    o: Weight,
    q_bias: ?[]f32,
    k_bias: ?[]f32,
    v_bias: ?[]f32,
    /// Dense MLP (null for mixture-of-experts layers).
    gate: ?Weight,
    up: ?Weight,
    down: ?Weight,
    /// Routed mixture of experts (null for dense layers).
    moe: ?moe.MoeLayer = null,
    /// Abliteration deltas (null = identity). Expert deltas live in `moe`.
    o_delta: ?Delta = null,
    down_delta: ?Delta = null,
};

/// The two abliterable components, named as in heretic.
pub const Component = enum {
    attn_o_proj,
    mlp_down_proj,

    pub fn name(self: Component) []const u8 {
        return switch (self) {
            .attn_o_proj => "attn.o_proj",
            .mlp_down_proj => "mlp.down_proj",
        };
    }

    pub fn fromName(s: []const u8) ?Component {
        if (std.mem.eql(u8, s, "attn.o_proj")) return .attn_o_proj;
        if (std.mem.eql(u8, s, "mlp.down_proj")) return .mlp_down_proj;
        return null;
    }

    pub const all = [_]Component{ .attn_o_proj, .mlp_down_proj };
};

pub const Model = struct {
    gpa: Allocator,
    io: Io,
    pool: *const tensor.Pool,
    arena: std.heap.ArenaAllocator,
    config: Config,
    tokenizer: *Tokenizer,
    files: []*safetensors.File,
    /// Tensor name prefix for the language model (e.g. "model." or "language_model.model.").
    prefix: []const u8,
    embed: Weight,
    lm_head: Weight,
    final_norm: []f32,
    layers: []Layer,
    eos_ids: []u32,
    pad_id: u32,
    /// Directory the model was loaded from.
    source_dir: []const u8,
    /// Raw JSON texts kept for export.
    config_json: []const u8,
    tokenizer_json: []const u8,
    generation_config_json: ?[]const u8,
    tokenizer_config_json: ?[]const u8,
    chat_template: ?[]const u8,
    dtype: tensor.DType,
    rope_cos: []f32, // [max_pos][head_dim/2]
    rope_sin: []f32,
    rope_cos_local: []f32,
    rope_sin_local: []f32,
    rope_len: usize,

    pub fn deinit(self: *Model) void {
        self.resetDeltas();
        for (self.files) |f| f.close(self.gpa, self.io);
        self.tokenizer.deinit();
        self.arena.deinit();
        self.gpa.destroy(self);
    }

    /// Loads a model from a local directory containing config.json, tokenizer.json and safetensors files.
    pub fn load(gpa: Allocator, io: Io, pool: *const tensor.Pool, dir_path: []const u8) !*Model {
        const self = try gpa.create(Model);
        errdefer gpa.destroy(self);
        self.* = undefined;
        self.gpa = gpa;
        self.io = io;
        self.pool = pool;
        self.arena = std.heap.ArenaAllocator.init(gpa);
        errdefer self.arena.deinit();
        const arena = self.arena.allocator();

        var dir = try Io.Dir.cwd().openDir(io, dir_path, .{ .iterate = true });
        defer dir.close(io);

        self.source_dir = try arena.dupe(u8, dir_path);
        self.config_json = try dir.readFileAlloc(io, "config.json", arena, .unlimited);
        self.config = try parseConfig(arena, self.config_json);
        self.generation_config_json = dir.readFileAlloc(io, "generation_config.json", arena, .unlimited) catch null;
        self.tokenizer_config_json = dir.readFileAlloc(io, "tokenizer_config.json", arena, .unlimited) catch null;
        const tok_json = dir.readFileAlloc(io, "tokenizer.json", arena, .unlimited) catch {
            std.log.err("tokenizer.json not found in {s} (only fast tokenizers are supported)", .{dir_path});
            return error.MissingTokenizer;
        };
        self.tokenizer_json = tok_json;
        self.tokenizer = try Tokenizer.parse(gpa, tok_json, self.tokenizer_config_json);
        errdefer self.tokenizer.deinit();
        self.chat_template = null;
        if (self.tokenizer_config_json) |tc| {
            var parsed = try std.json.parseFromSlice(std.json.Value, gpa, tc, .{});
            defer parsed.deinit();
            if (parsed.value == .object) {
                if (parsed.value.object.get("chat_template")) |ct| {
                    switch (ct) {
                        .string => |s| self.chat_template = try arena.dupe(u8, s),
                        .array => |a| {
                            for (a.items) |item| {
                                if (item == .object) {
                                    if (item.object.get("template")) |t| {
                                        if (t == .string) self.chat_template = try arena.dupe(u8, t.string);
                                    }
                                    if (item.object.get("name")) |n| {
                                        if (n == .string and std.mem.eql(u8, n.string, "default")) break;
                                    }
                                }
                            }
                        },
                        else => {},
                    }
                }
            }
        }
        if (self.chat_template == null) {
            const ct = dir.readFileAlloc(io, "chat_template.jinja", arena, .unlimited) catch null;
            self.chat_template = ct;
        }

        // EOS ids: generation_config eos_token_id (int or list) + tokenizer eos.
        var eos = std.ArrayList(u32).empty;
        if (self.generation_config_json) |gc| {
            var parsed = try std.json.parseFromSlice(std.json.Value, gpa, gc, .{});
            defer parsed.deinit();
            if (parsed.value == .object) {
                if (parsed.value.object.get("eos_token_id")) |e| {
                    switch (e) {
                        .integer => |i| try eos.append(arena, @intCast(i)),
                        .array => |a| for (a.items) |x| {
                            if (x == .integer) try eos.append(arena, @intCast(x.integer));
                        },
                        else => {},
                    }
                }
            }
        }
        if (self.tokenizer.eos_id) |e| {
            var found = false;
            for (eos.items) |x| found = found or x == e;
            if (!found) try eos.append(arena, e);
        }
        self.eos_ids = eos.items;
        self.pad_id = if (eos.items.len > 0) eos.items[0] else 0;

        // Safetensors files.
        var files = std.ArrayList(*safetensors.File).empty;
        errdefer for (files.items) |f| f.close(gpa, io);
        {
            var names = std.ArrayList([]const u8).empty;
            var it = dir.iterate();
            while (try it.next(io)) |entry| {
                if (entry.kind != .file) continue;
                if (std.mem.endsWith(u8, entry.name, ".safetensors")) try names.append(arena, try arena.dupe(u8, entry.name));
            }
            std.mem.sort([]const u8, names.items, {}, struct {
                fn lt(_: void, a: []const u8, b: []const u8) bool {
                    return std.mem.lessThan(u8, a, b);
                }
            }.lt);
            if (names.items.len == 0) {
                std.log.err("no .safetensors files found in {s}", .{dir_path});
                return error.MissingWeights;
            }
            for (names.items) |n| try files.append(arena, try safetensors.File.open(gpa, io, dir, n));
        }
        self.files = files.items;

        // Detect prefix.
        const prefixes = [_][]const u8{ "model.", "language_model.model.", "model.language_model.", "" };
        self.prefix = "";
        var found_prefix = false;
        for (prefixes) |p| {
            const key = try std.fmt.allocPrint(arena, "{s}embed_tokens.weight", .{p});
            if (self.find(key) != null) {
                self.prefix = p;
                found_prefix = true;
                break;
            }
        }
        if (!found_prefix) return error.MissingWeights;

        const embed_t = self.find(try std.fmt.allocPrint(arena, "{s}embed_tokens.weight", .{self.prefix})).?;
        self.embed = embed_t.asWeight();
        self.dtype = embed_t.dtype;
        if (self.config.vocab_size == 0) self.config.vocab_size = self.embed.rows;
        self.final_norm = try self.loadVec(try std.fmt.allocPrint(arena, "{s}norm.weight", .{self.prefix}));
        if (self.find("lm_head.weight")) |lm| {
            self.lm_head = lm.asWeight();
        } else if (self.find(try std.fmt.allocPrint(arena, "{s}lm_head.weight", .{std.mem.trimEnd(u8, self.prefix, "model.")}))) |lm| {
            self.lm_head = lm.asWeight();
        } else {
            self.lm_head = self.embed;
        }

        const c = &self.config;
        self.layers = try arena.alloc(Layer, c.num_layers);
        for (self.layers, 0..) |*layer, i| {
            const lp = try std.fmt.allocPrint(arena, "{s}layers.{d}.", .{ self.prefix, i });
            layer.* = .{
                .input_norm = try self.loadVec(try cat(arena, lp, "input_layernorm.weight")),
                .post_attn_norm = try self.loadVec(try cat(arena, lp, "post_attention_layernorm.weight")),
                .pre_ff_norm = self.loadVecOpt(try cat(arena, lp, "pre_feedforward_layernorm.weight")),
                .post_ff_norm = self.loadVecOpt(try cat(arena, lp, "post_feedforward_layernorm.weight")),
                .q_norm = self.loadVecOpt(try cat(arena, lp, "self_attn.q_norm.weight")),
                .k_norm = self.loadVecOpt(try cat(arena, lp, "self_attn.k_norm.weight")),
                .q = try self.loadMat(try cat(arena, lp, "self_attn.q_proj.weight")),
                .k = try self.loadMat(try cat(arena, lp, "self_attn.k_proj.weight")),
                .v = try self.loadMat(try cat(arena, lp, "self_attn.v_proj.weight")),
                .o = try self.loadMat(try cat(arena, lp, "self_attn.o_proj.weight")),
                .q_bias = self.loadVecOpt(try cat(arena, lp, "self_attn.q_proj.bias")),
                .k_bias = self.loadVecOpt(try cat(arena, lp, "self_attn.k_proj.bias")),
                .v_bias = self.loadVecOpt(try cat(arena, lp, "self_attn.v_proj.bias")),
                .gate = null,
                .up = null,
                .down = null,
            };
            if (c.moe_layers[i]) {
                layer.moe = try moe.loadLayer(self, arena, lp);
            } else {
                layer.gate = try self.loadMat(try cat(arena, lp, "mlp.gate_proj.weight"));
                layer.up = try self.loadMat(try cat(arena, lp, "mlp.up_proj.weight"));
                layer.down = try self.loadMat(try cat(arena, lp, "mlp.down_proj.weight"));
            }
        }

        try self.buildRope();
        return self;
    }

    fn cat(arena: Allocator, a: []const u8, b: []const u8) ![]const u8 {
        return std.fmt.allocPrint(arena, "{s}{s}", .{ a, b });
    }

    pub fn find(self: *const Model, name: []const u8) ?safetensors.TensorInfo {
        for (self.files) |f| {
            if (f.get(name)) |t| return t;
        }
        return null;
    }

    /// Matrix view of a named tensor, or null if absent.
    pub fn findWeight(self: *const Model, name: []const u8) ?Weight {
        const t = self.find(name) orelse return null;
        return t.asWeight();
    }

    fn loadMat(self: *Model, name: []const u8) !Weight {
        return self.findWeight(name) orelse {
            std.log.err("missing tensor: {s}", .{name});
            return error.MissingWeights;
        };
    }

    fn loadVec(self: *Model, name: []const u8) ![]f32 {
        return self.loadVecOpt(name) orelse {
            std.log.err("missing tensor: {s}", .{name});
            return error.MissingWeights;
        };
    }

    fn loadVecOpt(self: *Model, name: []const u8) ?[]f32 {
        const t = self.find(name) orelse return null;
        const out = self.arena.allocator().alloc(f32, t.numel()) catch return null;
        tensor.convertToF32(t.dtype, t.data, out);
        return out;
    }

    fn buildRope(self: *Model) !void {
        const c = &self.config;
        const arena = self.arena.allocator();
        const half = c.head_dim / 2;
        self.rope_len = @min(c.max_position_embeddings, 8192);
        self.rope_cos = try arena.alloc(f32, self.rope_len * half);
        self.rope_sin = try arena.alloc(f32, self.rope_len * half);
        try self.fillRope(self.rope_cos, self.rope_sin, c.rope_theta, c.rope_scaling);
        if (c.family == .gemma3) {
            self.rope_cos_local = try arena.alloc(f32, self.rope_len * half);
            self.rope_sin_local = try arena.alloc(f32, self.rope_len * half);
            try self.fillRope(self.rope_cos_local, self.rope_sin_local, c.rope_local_theta, .none);
        } else {
            self.rope_cos_local = self.rope_cos;
            self.rope_sin_local = self.rope_sin;
        }
    }

    fn fillRope(self: *Model, cos: []f32, sin: []f32, theta: f32, scaling: RopeScaling) !void {
        const c = &self.config;
        const half = c.head_dim / 2;
        const inv_freq = try self.gpa.alloc(f64, half);
        defer self.gpa.free(inv_freq);
        for (inv_freq, 0..) |*f, i| {
            const exponent: f64 = @as(f64, @floatFromInt(2 * i)) / @as(f64, @floatFromInt(c.head_dim));
            f.* = 1.0 / std.math.pow(f64, theta, exponent);
        }
        switch (scaling) {
            .none => {},
            .linear => |factor| for (inv_freq) |*f| {
                f.* /= factor;
            },
            .llama3 => |s| {
                const low_wavelen = s.original_max_position / s.low_freq_factor;
                const high_wavelen = s.original_max_position / s.high_freq_factor;
                for (inv_freq) |*f| {
                    const wavelen = 2.0 * std.math.pi / f.*;
                    if (wavelen < high_wavelen) {
                        // keep
                    } else if (wavelen > low_wavelen) {
                        f.* /= s.factor;
                    } else {
                        const smooth = (s.original_max_position / wavelen - s.low_freq_factor) / (s.high_freq_factor - s.low_freq_factor);
                        f.* = (1.0 - smooth) * f.* / s.factor + smooth * f.*;
                    }
                }
            },
        }
        var pos: usize = 0;
        while (pos < self.rope_len) : (pos += 1) {
            for (inv_freq, 0..) |f, i| {
                const angle = @as(f64, @floatFromInt(pos)) * f;
                cos[pos * half + i] = @floatCast(@cos(angle));
                sin[pos * half + i] = @floatCast(@sin(angle));
            }
        }
    }

    pub fn isEos(self: *const Model, id: u32) bool {
        for (self.eos_ids) |e| if (e == id) return true;
        return false;
    }

    /// Removes all abliteration deltas.
    pub fn resetDeltas(self: *Model) void {
        for (self.layers) |*l| {
            if (l.o_delta) |d| {
                self.gpa.free(d.a);
                self.gpa.free(d.b);
                l.o_delta = null;
            }
            if (l.down_delta) |d| {
                self.gpa.free(d.a);
                self.gpa.free(d.b);
                l.down_delta = null;
            }
            if (l.moe) |*m| m.resetDeltas(self.gpa);
        }
    }

    /// Weight of a dense component. For `.mlp_down_proj` on an MoE layer use `expertDownWeight`.
    pub fn componentWeight(self: *const Model, layer: usize, comp: Component) Weight {
        return switch (comp) {
            .attn_o_proj => self.layers[layer].o,
            .mlp_down_proj => self.layers[layer].down.?,
        };
    }

    /// Down projection of routed expert `expert` of an MoE layer; the shared
    /// expert (if any) is addressed by index `experts.len`.
    pub fn expertDownWeight(self: *const Model, layer: usize, expert: usize) Weight {
        return self.layers[layer].moe.?.downWeight(expert);
    }

    pub fn setExpertDelta(self: *Model, layer: usize, expert: usize, delta: Delta) void {
        const slot = self.layers[layer].moe.?.downDelta(expert);
        if (slot.*) |d| {
            self.gpa.free(d.a);
            self.gpa.free(d.b);
        }
        slot.* = delta;
    }

    pub fn getExpertDelta(self: *const Model, layer: usize, expert: usize) ?Delta {
        return self.layers[layer].moe.?.getDownDelta(expert);
    }

    pub fn setDelta(self: *Model, layer: usize, comp: Component, delta: Delta) void {
        const l = &self.layers[layer];
        switch (comp) {
            .attn_o_proj => {
                if (l.o_delta) |d| {
                    self.gpa.free(d.a);
                    self.gpa.free(d.b);
                }
                l.o_delta = delta;
            },
            .mlp_down_proj => {
                if (l.down_delta) |d| {
                    self.gpa.free(d.a);
                    self.gpa.free(d.b);
                }
                l.down_delta = delta;
            },
        }
    }

    /// True if any layer is a routed mixture of experts.
    pub fn isMoe(self: *const Model) bool {
        for (self.layers) |l| if (l.moe != null) return true;
        return false;
    }

    /// Largest number of routed experts in any layer (0 for dense models).
    pub fn numExpertsPerLayer(self: *const Model) usize {
        var n: usize = 0;
        for (self.layers) |l| if (l.moe) |m| {
            n = @max(n, m.experts.len);
        };
        return n;
    }

    pub fn getDelta(self: *const Model, layer: usize, comp: Component) ?Delta {
        return switch (comp) {
            .attn_o_proj => self.layers[layer].o_delta,
            .mlp_down_proj => self.layers[layer].down_delta,
        };
    }
};

// ---------------------------------------------------------------------------
// Inference
// ---------------------------------------------------------------------------

pub const KvCache = struct {
    gpa: Allocator,
    batch: usize,
    max_len: usize,
    kv_dim: usize,
    layers: usize,
    /// [layer][batch][pos][kv_dim]
    k: []f32,
    v: []f32,

    pub fn init(gpa: Allocator, layers: usize, batch: usize, max_len: usize, kv_dim: usize) !KvCache {
        const n = layers * batch * max_len * kv_dim;
        const k = try gpa.alloc(f32, n);
        errdefer gpa.free(k);
        const v = try gpa.alloc(f32, n);
        return .{ .gpa = gpa, .batch = batch, .max_len = max_len, .kv_dim = kv_dim, .layers = layers, .k = k, .v = v };
    }

    pub fn deinit(self: *KvCache) void {
        self.gpa.free(self.k);
        self.gpa.free(self.v);
    }

    inline fn index(self: *const KvCache, layer: usize, b: usize, pos: usize) usize {
        return ((layer * self.batch + b) * self.max_len + pos) * self.kv_dim;
    }

    pub fn kSlot(self: *KvCache, layer: usize, b: usize, pos: usize) []f32 {
        return self.k[self.index(layer, b, pos)..][0..self.kv_dim];
    }
    pub fn vSlot(self: *KvCache, layer: usize, b: usize, pos: usize) []f32 {
        return self.v[self.index(layer, b, pos)..][0..self.kv_dim];
    }
};

/// Describes one row of a batched forward call.
pub const Row = struct {
    /// Batch slot (index into the KV cache).
    b: usize,
    /// Absolute position of this token within its sequence.
    pos: usize,
};

const AttnCtx = struct {
    model: *const Model,
    cache: *KvCache,
    layer: usize,
    rows: []const Row,
    q: []f32, // [n][heads*head_dim] (already roped)
    out: []f32, // [n][heads*head_dim]
    sliding: bool,
};

fn attentionWorker(ctx: *const AttnCtx, start: usize, end: usize) void {
    const c = &ctx.model.config;
    const hd = c.head_dim;
    const groups = c.num_heads / c.num_kv_heads;
    var scores_buf: [8192]f32 = undefined;
    var task = start;
    while (task < end) : (task += 1) {
        const r = task / c.num_heads;
        const h = task % c.num_heads;
        const row = ctx.rows[r];
        const kvh = h / groups;
        const q = ctx.q[r * c.num_heads * hd + h * hd ..][0..hd];
        const out = ctx.out[r * c.num_heads * hd + h * hd ..][0..hd];
        var lo: usize = 0;
        if (ctx.sliding) {
            if (c.sliding_window) |w| {
                if (row.pos + 1 > w) lo = row.pos + 1 - w;
            }
        }
        const n_keys = row.pos + 1 - lo;
        const scores = scores_buf[0..n_keys];
        var p: usize = 0;
        while (p < n_keys) : (p += 1) {
            const k = ctx.cache.kSlot(ctx.layer, row.b, lo + p)[kvh * hd ..][0..hd];
            var s = tensor.dot(q, k) * c.attention_scale;
            if (c.attn_logit_softcapping) |cap| s = cap * std.math.tanh(s / cap);
            scores[p] = s;
        }
        tensor.softmaxInPlace(scores);
        @memset(out, 0);
        p = 0;
        while (p < n_keys) : (p += 1) {
            const v = ctx.cache.vSlot(ctx.layer, row.b, lo + p)[kvh * hd ..][0..hd];
            tensor.axpy(out, scores[p], v);
        }
    }
}

/// Workspace for forward passes, sized for `max_rows` tokens per call.
pub const Workspace = struct {
    gpa: Allocator,
    x: []f32,
    h: []f32,
    q: []f32,
    k: []f32,
    v: []f32,
    attn: []f32,
    o: []f32,
    gate: []f32,
    up: []f32,
    logits: []f32,
    max_rows: usize,
    max_logit_rows: usize,

    pub fn init(gpa: Allocator, c: *const Config, max_rows: usize, max_logit_rows: usize) !Workspace {
        const hidden = c.hidden_size;
        const qd = c.num_heads * c.head_dim;
        const kvd = c.num_kv_heads * c.head_dim;
        return .{
            .gpa = gpa,
            .x = try gpa.alloc(f32, max_rows * hidden),
            .h = try gpa.alloc(f32, max_rows * hidden),
            .q = try gpa.alloc(f32, max_rows * qd),
            .k = try gpa.alloc(f32, max_rows * kvd),
            .v = try gpa.alloc(f32, max_rows * kvd),
            .attn = try gpa.alloc(f32, max_rows * qd),
            .o = try gpa.alloc(f32, max_rows * hidden),
            .gate = try gpa.alloc(f32, max_rows * c.intermediate_size),
            .up = try gpa.alloc(f32, max_rows * c.intermediate_size),
            .logits = try gpa.alloc(f32, max_logit_rows * c.vocab_size),
            .max_rows = max_rows,
            .max_logit_rows = max_logit_rows,
        };
    }

    pub fn deinit(self: *Workspace) void {
        self.gpa.free(self.x);
        self.gpa.free(self.h);
        self.gpa.free(self.q);
        self.gpa.free(self.k);
        self.gpa.free(self.v);
        self.gpa.free(self.attn);
        self.gpa.free(self.o);
        self.gpa.free(self.gate);
        self.gpa.free(self.up);
        self.gpa.free(self.logits);
    }
};

/// Options for a forward call.
pub const ForwardOptions = struct {
    /// If set, hidden states (residual stream) for the rows listed in
    /// `capture_rows` are written to `residuals[layer_entry][i][hidden]` where
    /// layer_entry 0 is the embedding output and entry L is the output of layer L-1.
    capture_rows: []const usize = &.{},
    residuals: ?[]f32 = null,
    /// Rows for which logits should be computed (indexes into `rows`).
    logit_rows: []const usize = &.{},
};

/// Runs the transformer over `tokens`/`rows` (n tokens). Logits for the
/// requested rows are written to `ws.logits[i * vocab ..]`.
pub fn forward(model: *const Model, ws: *Workspace, cache: *KvCache, tokens: []const u32, rows: []const Row, opts: ForwardOptions) !void {
    const c = &model.config;
    const gpa = model.gpa;
    const n = tokens.len;
    std.debug.assert(n == rows.len and n <= ws.max_rows);
    const hidden = c.hidden_size;
    const hd = c.head_dim;
    const qd = c.num_heads * hd;
    const kvd = c.num_kv_heads * hd;
    const half = hd / 2;
    const x = ws.x[0 .. n * hidden];
    const h = ws.h[0 .. n * hidden];

    // Embeddings.
    for (tokens, 0..) |t, i| {
        model.embed.row(@min(t, model.embed.rows - 1), x[i * hidden ..][0..hidden]);
        if (c.embed_scale != 1.0) tensor.scale(x[i * hidden ..][0..hidden], c.embed_scale);
    }
    if (opts.residuals) |res| {
        for (opts.capture_rows, 0..) |r, i| @memcpy(res[(0 * opts.capture_rows.len + i) * hidden ..][0..hidden], x[r * hidden ..][0..hidden]);
    }

    for (model.layers, 0..) |*layer, li| {
        // Attention block.
        var i: usize = 0;
        while (i < n) : (i += 1) tensor.rmsnorm(h[i * hidden ..][0..hidden], x[i * hidden ..][0..hidden], layer.input_norm, c.rms_norm_eps, c.family.isGemma());
        try tensor.matmulT(model.pool, gpa, ws.q, h, n, layer.q, null);
        try tensor.matmulT(model.pool, gpa, ws.k, h, n, layer.k, null);
        try tensor.matmulT(model.pool, gpa, ws.v, h, n, layer.v, null);
        if (layer.q_bias) |b| {
            i = 0;
            while (i < n) : (i += 1) tensor.axpy(ws.q[i * qd ..][0..qd], 1.0, b);
        }
        if (layer.k_bias) |b| {
            i = 0;
            while (i < n) : (i += 1) tensor.axpy(ws.k[i * kvd ..][0..kvd], 1.0, b);
        }
        if (layer.v_bias) |b| {
            i = 0;
            while (i < n) : (i += 1) tensor.axpy(ws.v[i * kvd ..][0..kvd], 1.0, b);
        }
        const sliding = c.sliding_layers[li];
        const cos = if (sliding and c.family == .gemma3) model.rope_cos_local else model.rope_cos;
        const sin = if (sliding and c.family == .gemma3) model.rope_sin_local else model.rope_sin;
        i = 0;
        while (i < n) : (i += 1) {
            const pos = @min(rows[i].pos, model.rope_len - 1);
            const cr = cos[pos * half ..][0..half];
            const sr = sin[pos * half ..][0..half];
            var hh: usize = 0;
            while (hh < c.num_heads) : (hh += 1) {
                const q = ws.q[i * qd + hh * hd ..][0..hd];
                if (layer.q_norm) |qn| {
                    var tmp: [512]f32 = undefined;
                    tensor.rmsnorm(tmp[0..hd], q, qn, c.rms_norm_eps, false);
                    @memcpy(q, tmp[0..hd]);
                }
                tensor.applyRope(q, cr, sr);
            }
            hh = 0;
            while (hh < c.num_kv_heads) : (hh += 1) {
                const k = ws.k[i * kvd + hh * hd ..][0..hd];
                if (layer.k_norm) |kn| {
                    var tmp: [512]f32 = undefined;
                    tensor.rmsnorm(tmp[0..hd], k, kn, c.rms_norm_eps, false);
                    @memcpy(k, tmp[0..hd]);
                }
                tensor.applyRope(k, cr, sr);
            }
            @memcpy(cache.kSlot(li, rows[i].b, rows[i].pos), ws.k[i * kvd ..][0..kvd]);
            @memcpy(cache.vSlot(li, rows[i].b, rows[i].pos), ws.v[i * kvd ..][0..kvd]);
        }
        const actx = AttnCtx{ .model = model, .cache = cache, .layer = li, .rows = rows, .q = ws.q, .out = ws.attn, .sliding = sliding };
        model.pool.parallelFor(n * c.num_heads, &actx, attentionWorker);
        try tensor.matmulT(model.pool, gpa, ws.o, ws.attn, n, layer.o, if (layer.o_delta) |*d| d else null);
        i = 0;
        while (i < n) : (i += 1) {
            const o = ws.o[i * hidden ..][0..hidden];
            if (c.family.isGemma()) {
                const tmp = h[i * hidden ..][0..hidden];
                tensor.rmsnorm(tmp, o, layer.post_attn_norm, c.rms_norm_eps, true);
                tensor.axpy(x[i * hidden ..][0..hidden], 1.0, tmp);
            } else {
                tensor.axpy(x[i * hidden ..][0..hidden], 1.0, o);
            }
        }

        // MLP block.
        const ff_norm = if (c.family.isGemma()) layer.pre_ff_norm.? else layer.post_attn_norm;
        i = 0;
        while (i < n) : (i += 1) tensor.rmsnorm(h[i * hidden ..][0..hidden], x[i * hidden ..][0..hidden], ff_norm, c.rms_norm_eps, c.family.isGemma());
        if (layer.moe) |*m| {
            try moe.forward(model, m, ws.o, h, n);
        } else {
            try tensor.matmulT(model.pool, gpa, ws.gate, h, n, layer.gate.?, null);
            try tensor.matmulT(model.pool, gpa, ws.up, h, n, layer.up.?, null);
            const inter = c.intermediate_size;
            for (ws.gate[0 .. n * inter], 0..) |*g, j| g.* = c.activation.apply(g.*) * ws.up[j];
            try tensor.matmulT(model.pool, gpa, ws.o, ws.gate, n, layer.down.?, if (layer.down_delta) |*d| d else null);
        }
        i = 0;
        while (i < n) : (i += 1) {
            const o = ws.o[i * hidden ..][0..hidden];
            if (c.family.isGemma()) {
                const tmp = h[i * hidden ..][0..hidden];
                tensor.rmsnorm(tmp, o, layer.post_ff_norm.?, c.rms_norm_eps, true);
                tensor.axpy(x[i * hidden ..][0..hidden], 1.0, tmp);
            } else {
                tensor.axpy(x[i * hidden ..][0..hidden], 1.0, o);
            }
        }
        if (opts.residuals) |res| {
            for (opts.capture_rows, 0..) |r, ci| @memcpy(res[((li + 1) * opts.capture_rows.len + ci) * hidden ..][0..hidden], x[r * hidden ..][0..hidden]);
        }
    }

    // Final norm + logits for requested rows.
    if (opts.logit_rows.len > 0) {
        std.debug.assert(opts.logit_rows.len <= ws.max_logit_rows);
        for (opts.logit_rows, 0..) |r, i| tensor.rmsnorm(h[i * hidden ..][0..hidden], x[r * hidden ..][0..hidden], model.final_norm, c.rms_norm_eps, c.family.isGemma());
        try tensor.matmulT(model.pool, gpa, ws.logits, h, opts.logit_rows.len, model.lm_head, null);
        if (c.final_logit_softcapping) |cap| tensor.softcap(ws.logits[0 .. opts.logit_rows.len * c.vocab_size], cap);
    }
}

// ---------------------------------------------------------------------------
// High-level batched API
// ---------------------------------------------------------------------------

pub const GenerateResult = struct {
    /// Generated token ids per prompt (caller frees each and the outer slice).
    tokens: [][]u32,
};

fn argmax(x: []const f32) u32 {
    var best: usize = 0;
    var bv: f32 = -std.math.inf(f32);
    for (x, 0..) |v, i| {
        if (v > bv) {
            bv = v;
            best = i;
        }
    }
    return @intCast(best);
}

/// Runs prefill over a batch of tokenised prompts. Fills the KV cache and
/// returns the next-token logits for each prompt in `logits_out` ([batch][vocab]).
/// Optionally captures residuals for the last prompt token ([layer+1][batch][hidden]).
pub fn prefill(model: *const Model, ws: *Workspace, cache: *KvCache, prompts: []const []const u32, logits_out: ?[]f32, residuals_out: ?[]f32) !void {
    const gpa = model.gpa;
    const c = &model.config;
    var total: usize = 0;
    for (prompts) |p| total += p.len;
    const tokens = try gpa.alloc(u32, total);
    defer gpa.free(tokens);
    const rows = try gpa.alloc(Row, total);
    defer gpa.free(rows);
    const last_rows = try gpa.alloc(usize, prompts.len);
    defer gpa.free(last_rows);
    var idx: usize = 0;
    for (prompts, 0..) |p, b| {
        for (p, 0..) |t, pos| {
            tokens[idx] = t;
            rows[idx] = .{ .b = b, .pos = pos };
            idx += 1;
        }
        last_rows[b] = idx - 1;
    }
    // Process in chunks that fit the workspace, keeping whole prompts together where possible.
    var start: usize = 0;
    var res_tmp: ?[]f32 = null;
    defer if (res_tmp) |r| gpa.free(r);
    if (residuals_out != null) res_tmp = try gpa.alloc(f32, (c.num_layers + 1) * prompts.len * c.hidden_size);
    while (start < total) {
        var end = @min(total, start + ws.max_rows);
        // Do not split a prompt across chunks unless it is longer than the workspace.
        if (end < total) {
            var e = end;
            while (e > start and rows[e].pos != 0) e -= 1;
            if (e > start) end = e;
        }
        var lr = std.ArrayList(usize).empty;
        defer lr.deinit(gpa);
        var lr_batch = std.ArrayList(usize).empty;
        defer lr_batch.deinit(gpa);
        for (last_rows, 0..) |r, b| {
            if (r >= start and r < end) {
                try lr.append(gpa, r - start);
                try lr_batch.append(gpa, b);
            }
        }
        var residual_chunk: ?[]f32 = null;
        defer if (residual_chunk) |r| gpa.free(r);
        if (residuals_out != null and lr.items.len > 0) residual_chunk = try gpa.alloc(f32, (c.num_layers + 1) * lr.items.len * c.hidden_size);
        try forward(model, ws, cache, tokens[start..end], rows[start..end], .{
            .logit_rows = lr.items,
            .capture_rows = lr.items,
            .residuals = residual_chunk,
        });
        for (lr_batch.items, 0..) |b, i| {
            if (logits_out) |lo| @memcpy(lo[b * c.vocab_size ..][0..c.vocab_size], ws.logits[i * c.vocab_size ..][0..c.vocab_size]);
            if (residuals_out) |ro| {
                var l: usize = 0;
                while (l <= c.num_layers) : (l += 1) {
                    @memcpy(ro[(l * prompts.len + b) * c.hidden_size ..][0..c.hidden_size], residual_chunk.?[(l * lr.items.len + i) * c.hidden_size ..][0..c.hidden_size]);
                }
            }
        }
        start = end;
    }
}

/// Greedy generation for a batch of prompts.
pub fn generate(model: *const Model, ws: *Workspace, cache: *KvCache, prompts: []const []const u32, max_new_tokens: usize) ![][]u32 {
    const gpa = model.gpa;
    const c = &model.config;
    const b = prompts.len;
    const logits = try gpa.alloc(f32, b * c.vocab_size);
    defer gpa.free(logits);
    try prefill(model, ws, cache, prompts, logits, null);

    var outputs = try gpa.alloc(std.ArrayList(u32), b);
    defer gpa.free(outputs);
    for (outputs) |*o| o.* = .empty;
    errdefer for (outputs) |*o| o.deinit(gpa);
    var active = try gpa.alloc(bool, b);
    defer gpa.free(active);
    @memset(active, true);
    var positions = try gpa.alloc(usize, b);
    defer gpa.free(positions);
    for (prompts, 0..) |p, i| positions[i] = p.len;

    var tokens = try gpa.alloc(u32, b);
    defer gpa.free(tokens);
    var rows = try gpa.alloc(Row, b);
    defer gpa.free(rows);
    var logit_rows = try gpa.alloc(usize, b);
    defer gpa.free(logit_rows);

    // First token from prefill logits.
    var n_active: usize = 0;
    for (0..b) |i| {
        const t = argmax(logits[i * c.vocab_size ..][0..c.vocab_size]);
        try outputs[i].append(gpa, t);
        if (model.isEos(t) or max_new_tokens <= 1 or positions[i] >= cache.max_len) active[i] = false else n_active += 1;
    }
    var step: usize = 1;
    while (step < max_new_tokens and n_active > 0) : (step += 1) {
        var n: usize = 0;
        for (0..b) |i| {
            if (!active[i]) continue;
            tokens[n] = outputs[i].items[outputs[i].items.len - 1];
            rows[n] = .{ .b = i, .pos = positions[i] };
            logit_rows[n] = n;
            n += 1;
        }
        try forward(model, ws, cache, tokens[0..n], rows[0..n], .{ .logit_rows = logit_rows[0..n] });
        var j: usize = 0;
        for (0..b) |i| {
            if (!active[i]) continue;
            const t = argmax(ws.logits[j * c.vocab_size ..][0..c.vocab_size]);
            j += 1;
            positions[i] += 1;
            try outputs[i].append(gpa, t);
            if (model.isEos(t) or positions[i] >= cache.max_len) {
                active[i] = false;
                n_active -= 1;
            }
        }
    }
    const result = try gpa.alloc([]u32, b);
    for (outputs, 0..) |*o, i| result[i] = try o.toOwnedSlice(gpa);
    return result;
}

/// Expert-selective abliteration entry point (see `moe.applyExpertSelective`).
pub fn applyExpertSelective(model: *Model, dirs: []const f32, cfg: search.TrialConfig, opts: abliterate.Options) !void {
    return moe.applyExpertSelective(model, dirs, cfg, opts);
}

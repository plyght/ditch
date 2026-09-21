//! GGUF files as model input. Locates the file, rebuilds `config.json`,
//! `tokenizer.json`, `tokenizer_config.json` and `generation_config.json`
//! from the metadata (or takes the copies a ditch export embedded), and
//! presents every tensor under its Hugging Face name, so `model.zig`,
//! `moe.zig`, `export.zig` and the kernels work unchanged: llama.cpp's
//! q/k permutation is undone, gemma norms lose their `+1`, stacked expert
//! tensors are split into per-expert views and quantised tensors keep their
//! block dtype (dequantised row by row by `tensor.Weight.row`).

const std = @import("std");
const Io = std.Io;
const gguf = @import("gguf.zig");
const tensor = @import("tensor.zig");
const safetensors = @import("safetensors.zig");
const model_mod = @import("model.zig");
const Tokenizer = @import("tokenizer.zig").Tokenizer;

const Allocator = std.mem.Allocator;
const Model = model_mod.Model;
const Family = model_mod.Family;

/// Metadata keys ditch adds to the files it writes (all optional on read).
pub const key_hf_config = "ditch.hf.config_json";
pub const key_hf_tokenizer = "tokenizer.huggingface.json";
pub const key_hf_tokenizer_config = "ditch.hf.tokenizer_config_json";
pub const key_hf_generation_config = "ditch.hf.generation_config_json";

/// llama.cpp token types (`tokenizer.ggml.token_type`).
pub const TokenType = enum(i32) {
    normal = 1,
    unknown = 2,
    control = 3,
    user_defined = 4,
    unused = 5,
    byte = 6,
};

pub const Source = struct {
    file: *gguf.File,
    /// Directory holding the file, and the file name inside it.
    dir_path: []const u8,
    file_name: []const u8,
    arch: []const u8,
    family: Family,
    /// Tensors under their Hugging Face names (expert entries are views into the stacked tensors).
    tensors: std.StringArrayHashMapUnmanaged(safetensors.TensorInfo),
    /// GGUF tensors without a Hugging Face counterpart (`rope_freqs.weight`, ...), copied verbatim on GGUF export.
    extra: []const gguf.TensorInfo,
    /// Sum of all tensor bytes in the file.
    total_bytes: u64,
    /// Whether the config/tokenizer came from embedded copies rather than the ggml metadata.
    embedded_config: bool,
    embedded_tokenizer: bool,

    pub fn close(self: *Source, gpa: Allocator, io: Io) void {
        self.file.close(gpa, io);
        // `self` and the tables live in the model arena.
    }
};

/// Options for `attach`, used by tests to force the metadata-only paths.
pub const AttachOptions = struct {
    ignore_embedded: bool = false,
};

// ---------------------------------------------------------------------------
// Locating
// ---------------------------------------------------------------------------

/// Returns the path of the GGUF file to load if `path` is a `.gguf` file, or a
/// directory without `config.json` containing exactly one `.gguf` file.
pub fn locate(io: Io, arena: Allocator, path: []const u8) !?[]const u8 {
    const cwd = Io.Dir.cwd();
    if (std.mem.endsWith(u8, path, ".gguf")) {
        if (cwd.openFile(io, path, .{})) |f| {
            f.close(io);
            return path;
        } else |_| {}
    }
    var dir = cwd.openDir(io, path, .{ .iterate = true }) catch return null;
    defer dir.close(io);
    if (dir.access(io, "config.json", .{})) |_| return null else |_| {}
    var found: ?[]const u8 = null;
    var count: usize = 0;
    var it = dir.iterate();
    while (try it.next(io)) |entry| {
        if (entry.kind != .file or !std.mem.endsWith(u8, entry.name, ".gguf")) continue;
        count += 1;
        if (found == null or std.mem.lessThan(u8, entry.name, found.?)) found = try arena.dupe(u8, entry.name);
    }
    if (found == null) return null;
    if (count > 1) {
        std.log.err("{s} contains {d} .gguf files; pass the file path instead of the directory", .{ path, count });
        return error.AmbiguousModel;
    }
    return try std.fs.path.join(arena, &.{ path, found.? });
}

// ---------------------------------------------------------------------------
// Attaching to a model
// ---------------------------------------------------------------------------

fn familyFromArch(arch: []const u8, has_experts: bool) ?Family {
    if (std.mem.eql(u8, arch, "llama")) return if (has_experts) .mixtral else .llama;
    if (std.mem.eql(u8, arch, "qwen2")) return .qwen2;
    if (std.mem.eql(u8, arch, "qwen3")) return .qwen3;
    if (std.mem.eql(u8, arch, "gemma2")) return .gemma2;
    if (std.mem.eql(u8, arch, "gemma3")) return .gemma3;
    if (std.mem.eql(u8, arch, "qwen2moe")) return .qwen2_moe;
    if (std.mem.eql(u8, arch, "qwen3moe")) return .qwen3_moe;
    return null;
}

/// llama.cpp architecture name of a family.
pub fn archName(family: Family) []const u8 {
    return switch (family) {
        .llama, .mistral, .mixtral => "llama",
        .qwen2 => "qwen2",
        .qwen3 => "qwen3",
        .gemma2 => "gemma2",
        .gemma3 => "gemma3",
        .qwen2_moe => "qwen2moe",
        .qwen3_moe => "qwen3moe",
    };
}

fn modelTypeName(family: Family) []const u8 {
    return switch (family) {
        .llama => "llama",
        .mistral => "mistral",
        .mixtral => "mixtral",
        .qwen2 => "qwen2",
        .qwen3 => "qwen3",
        .gemma2 => "gemma2",
        .gemma3 => "gemma3_text",
        .qwen2_moe => "qwen2_moe",
        .qwen3_moe => "qwen3_moe",
    };
}

/// Fills `model` (a freshly created, otherwise empty `Model`) from a GGUF file.
pub fn attach(model: *Model, gguf_path: []const u8) !void {
    return attachWithOptions(model, gguf_path, .{});
}

pub fn attachWithOptions(model: *Model, gguf_path: []const u8, opts: AttachOptions) !void {
    const gpa = model.gpa;
    const io = model.io;
    const arena = model.arena.allocator();
    const dir_path = std.fs.path.dirname(gguf_path) orelse ".";
    const file_name = std.fs.path.basename(gguf_path);
    var dir = try Io.Dir.cwd().openDir(io, dir_path, .{});
    defer dir.close(io);
    const file = try gguf.File.open(gpa, io, dir, file_name);
    errdefer file.close(gpa, io);

    const arch = file.architecture() orelse {
        std.log.err("{s}: general.architecture missing", .{gguf_path});
        return error.InvalidGguf;
    };
    const has_experts = (file.archInt("expert_count") orelse 0) > 0;
    const family = familyFromArch(arch, has_experts) orelse {
        std.log.err("unsupported GGUF architecture: {s}", .{arch});
        return error.UnsupportedArchitecture;
    };
    const src = try arena.create(Source);
    src.* = .{
        .file = file,
        .dir_path = try arena.dupe(u8, dir_path),
        .file_name = try arena.dupe(u8, file_name),
        .arch = arch,
        .family = family,
        .tensors = .{},
        .extra = &.{},
        .total_bytes = 0,
        .embedded_config = false,
        .embedded_tokenizer = false,
    };
    model.source_dir = src.dir_path;
    model.files = &.{};

    // Configuration.
    if (!opts.ignore_embedded and file.getStr(key_hf_config) != null) {
        model.config_json = try arena.dupe(u8, file.getStr(key_hf_config).?);
        src.embedded_config = true;
    } else {
        model.config_json = try buildConfigJson(arena, file, family);
    }
    model.config = try model_mod.parseConfig(arena, model.config_json);
    if (model.config.rope_scaling == .none) {
        if (file.getTensor("rope_freqs.weight")) |rf| {
            if (rf.dtype()) |dt| {
                const factors = try arena.alloc(f32, rf.numel());
                tensor.convertToF32(dt, rf.data, factors);
                model.config.rope_scaling = .{ .factors = factors };
            }
        }
    }

    // Tokenizer.
    if (!opts.ignore_embedded and file.getStr(key_hf_tokenizer) != null) {
        model.tokenizer_json = try arena.dupe(u8, file.getStr(key_hf_tokenizer).?);
        model.tokenizer_config_json = if (file.getStr(key_hf_tokenizer_config)) |s| try arena.dupe(u8, s) else try buildTokenizerConfigJson(arena, file);
        model.generation_config_json = if (file.getStr(key_hf_generation_config)) |s| try arena.dupe(u8, s) else try buildGenerationConfigJson(arena, file);
        src.embedded_tokenizer = true;
    } else {
        model.tokenizer_json = try buildTokenizerJson(arena, file);
        model.tokenizer_config_json = try buildTokenizerConfigJson(arena, file);
        model.generation_config_json = try buildGenerationConfigJson(arena, file);
    }
    model.tokenizer = try Tokenizer.parse(gpa, model.tokenizer_json, model.tokenizer_config_json);
    errdefer model.tokenizer.deinit();
    model.chat_template = if (file.getStr("tokenizer.chat_template")) |t| try arena.dupe(u8, t) else null;

    // EOS ids: eos, eot, eom.
    var eos = std.ArrayList(u32).empty;
    for ([_][]const u8{ "tokenizer.ggml.eos_token_id", "tokenizer.ggml.eot_token_id", "tokenizer.ggml.eom_token_id" }) |key| {
        const v = file.getInt(key) orelse continue;
        if (v < 0) continue;
        const id: u32 = @intCast(v);
        var found = false;
        for (eos.items) |x| found = found or x == id;
        if (!found) try eos.append(arena, id);
    }
    if (model.tokenizer.eos_id) |e| {
        var found = false;
        for (eos.items) |x| found = found or x == e;
        if (!found) try eos.append(arena, e);
    }
    model.eos_ids = eos.items;
    model.pad_id = if (file.getInt("tokenizer.ggml.padding_token_id")) |p| (if (p >= 0) @intCast(p) else 0) else if (eos.items.len > 0) eos.items[0] else 0;

    try mapTensors(src, arena, &model.config);
    model.gguf = src;
}

// ---------------------------------------------------------------------------
// config.json from the metadata
// ---------------------------------------------------------------------------

fn jsonStr(w: *Io.Writer, s: []const u8) !void {
    try std.json.Stringify.encodeJsonString(s, .{}, w);
}

fn buildConfigJson(a: Allocator, f: *const gguf.File, family: Family) ![]const u8 {
    var out: Io.Writer.Allocating = .init(a);
    const w = &out.writer;
    const hidden = f.archInt("embedding_length") orelse return error.InvalidGguf;
    const layers = f.archInt("block_count") orelse return error.InvalidGguf;
    const heads = f.archInt("attention.head_count") orelse return error.InvalidGguf;
    if (hidden <= 0 or layers <= 0 or heads <= 0) return error.InvalidGguf;
    const kv_heads = f.archInt("attention.head_count_kv") orelse heads;
    const head_dim = f.archInt("attention.key_length") orelse @divTrunc(hidden, heads);
    const inter = f.archInt("feed_forward_length") orelse 4 * hidden;
    const vocab: i64 = if (f.getArray("tokenizer.ggml.tokens")) |t| @intCast(t.items.len) else if (f.getTensor("token_embd.weight")) |e| @intCast(e.dims[1]) else (f.archInt("vocab_size") orelse 0);
    const eps = f.archFloat("attention.layer_norm_rms_epsilon") orelse 1e-6;
    const theta = f.archFloat("rope.freq_base") orelse 10000.0;
    const ctx = f.archInt("context_length") orelse 4096;
    const tied = f.getTensor("output.weight") == null;
    const is_gemma = family == .gemma2 or family == .gemma3;

    try w.writeAll("{");
    try w.writeAll("\"model_type\":");
    try jsonStr(w, modelTypeName(family));
    try w.print(",\"hidden_size\":{d},\"intermediate_size\":{d},\"num_hidden_layers\":{d},\"num_attention_heads\":{d},\"num_key_value_heads\":{d},\"head_dim\":{d},\"vocab_size\":{d}", .{ hidden, inter, layers, heads, kv_heads, head_dim, vocab });
    try w.print(",\"rms_norm_eps\":{d},\"rope_theta\":{d},\"max_position_embeddings\":{d},\"tie_word_embeddings\":{s}", .{ eps, theta, ctx, if (tied) "true" else "false" });
    try w.print(",\"hidden_act\":\"{s}\"", .{if (is_gemma) "gelu_pytorch_tanh" else "silu"});
    try w.print(",\"attention_bias\":{s}", .{if (f.getTensor("blk.0.attn_q.bias") != null) "true" else "false"});
    if (f.archStr("rope.scaling.type")) |t| {
        if (std.mem.eql(u8, t, "linear")) {
            try w.print(",\"rope_scaling\":{{\"rope_type\":\"linear\",\"factor\":{d}}}", .{f.archFloat("rope.scaling.factor") orelse 1.0});
        } else if (!std.mem.eql(u8, t, "none")) {
            std.log.warn("rope scaling type '{s}' is not supported; using unscaled RoPE", .{t});
        }
    }
    if (f.archFloat("attn_logit_softcapping")) |v| try w.print(",\"attn_logit_softcapping\":{d}", .{v});
    if (f.archFloat("final_logit_softcapping")) |v| try w.print(",\"final_logit_softcapping\":{d}", .{v});
    if (is_gemma or family == .mistral) {
        if (f.archInt("attention.sliding_window")) |sw| try w.print(",\"sliding_window\":{d}", .{sw});
    }
    if (family == .gemma3) {
        try w.print(",\"sliding_window_pattern\":{d}", .{f.archInt("attention.sliding_window_pattern") orelse 6});
        try w.print(",\"rope_local_base_freq\":{d}", .{f.archFloat("rope.freq_base_swa") orelse 10000.0});
    }
    if (is_gemma) {
        // llama.cpp derives the attention scale from the model size: the 27B
        // variants (46 / 62 blocks) use hidden / heads, every other one head_dim.
        const is_27b = (family == .gemma2 and layers == 46) or (family == .gemma3 and layers == 62);
        try w.print(",\"query_pre_attn_scalar\":{d}", .{if (is_27b) @divTrunc(hidden, heads) else head_dim});
    }
    if (f.archInt("expert_count")) |n_exp| {
        if (n_exp > 0) {
            try w.print(",\"num_experts\":{d},\"num_experts_per_tok\":{d},\"moe_intermediate_size\":{d}", .{ n_exp, f.archInt("expert_used_count") orelse 2, f.archInt("expert_feed_forward_length") orelse inter });
            if (f.archInt("expert_shared_feed_forward_length")) |s| try w.print(",\"shared_expert_intermediate_size\":{d}", .{s});
            try w.print(",\"norm_topk_prob\":{s}", .{if (family == .qwen3_moe or family == .mixtral) "true" else "false"});
            // Dense layers of a MoE model have no stacked expert tensors.
            try w.writeAll(",\"mlp_only_layers\":[");
            var first = true;
            var i: i64 = 0;
            while (i < layers) : (i += 1) {
                var buf: [64]u8 = undefined;
                const name = try std.fmt.bufPrint(&buf, "blk.{d}.ffn_gate_exps.weight", .{i});
                if (f.getTensor(name) == null) {
                    if (!first) try w.writeAll(",");
                    first = false;
                    try w.print("{d}", .{i});
                }
            }
            try w.writeAll("]");
        }
    }
    try w.writeAll("}");
    return out.toOwnedSlice();
}

// ---------------------------------------------------------------------------
// tokenizer.json from the vocabulary
// ---------------------------------------------------------------------------

/// Pre-tokeniser regular expressions that `tokenizer.zig` recognises.
pub const regex_qwen2 = "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";
pub const regex_llama3 = "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";
pub const regex_gpt2 = "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+";

/// The regular expression behind a `tokenizer.ggml.pre` name.
pub fn regexForPre(pre: []const u8) []const u8 {
    if (std.mem.eql(u8, pre, "llama-bpe") or std.mem.eql(u8, pre, "llama3") or std.mem.eql(u8, pre, "llama-v3")) return regex_llama3;
    if (std.mem.eql(u8, pre, "qwen2") or std.mem.eql(u8, pre, "deepseek-r1-qwen")) return regex_qwen2;
    if (!std.mem.eql(u8, pre, "gpt-2") and !std.mem.eql(u8, pre, "default")) {
        std.log.warn("pre-tokenizer '{s}' is not known; using the GPT-2 pattern", .{pre});
    }
    return regex_gpt2;
}

fn isLlamaBpe(pre: []const u8) bool {
    return std.mem.eql(u8, pre, "llama-bpe") or std.mem.eql(u8, pre, "llama3") or std.mem.eql(u8, pre, "llama-v3");
}

const Vocab = struct {
    tokens: []const []const u8,
    types: []const TokenType,
    scores: ?[]const f32,
    merges: ?[]const []const u8,
    model: []const u8,
    pre: []const u8,
    bos: ?u32,
    eos: ?u32,
    unk: ?u32,
    pad: ?u32,
    add_bos: bool,
    add_space_prefix: bool,

    fn load(a: Allocator, f: *const gguf.File) !Vocab {
        const toks = f.getArray("tokenizer.ggml.tokens") orelse {
            std.log.err("GGUF file has no tokenizer.ggml.tokens", .{});
            return error.MissingTokenizer;
        };
        const tokens = try a.alloc([]const u8, toks.items.len);
        for (toks.items, 0..) |t, i| tokens[i] = t.asString() orelse return error.InvalidGguf;
        const types = try a.alloc(TokenType, tokens.len);
        @memset(types, .normal);
        if (f.getArray("tokenizer.ggml.token_type")) |tt| {
            for (tt.items, 0..) |t, i| {
                if (i >= types.len) break;
                const v = t.asInt() orelse 1;
                types[i] = if (v >= 1 and v <= 6) @enumFromInt(@as(i32, @intCast(v))) else .normal;
            }
        }
        var scores: ?[]const f32 = null;
        if (f.getArray("tokenizer.ggml.scores")) |sc| {
            const s = try a.alloc(f32, tokens.len);
            @memset(s, 0);
            for (sc.items, 0..) |v, i| {
                if (i >= s.len) break;
                s[i] = @floatCast(v.asFloat() orelse 0);
            }
            scores = s;
        }
        var merges: ?[]const []const u8 = null;
        if (f.getArray("tokenizer.ggml.merges")) |m| {
            const list = try a.alloc([]const u8, m.items.len);
            for (m.items, 0..) |v, i| list[i] = v.asString() orelse return error.InvalidGguf;
            merges = list;
        }
        const model = f.getStr("tokenizer.ggml.model") orelse "gpt2";
        const is_spm = std.mem.eql(u8, model, "llama");
        const pre = f.getStr("tokenizer.ggml.pre") orelse "default";
        return .{
            .tokens = tokens,
            .types = types,
            .scores = scores,
            .merges = merges,
            .model = model,
            .pre = pre,
            .bos = idOf(f, "tokenizer.ggml.bos_token_id", tokens.len),
            .eos = idOf(f, "tokenizer.ggml.eos_token_id", tokens.len),
            .unk = idOf(f, "tokenizer.ggml.unknown_token_id", tokens.len),
            .pad = idOf(f, "tokenizer.ggml.padding_token_id", tokens.len),
            .add_bos = f.getBool("tokenizer.ggml.add_bos_token") orelse (is_spm or isLlamaBpe(pre)),
            .add_space_prefix = f.getBool("tokenizer.ggml.add_space_prefix") orelse is_spm,
        };
    }

    fn idOf(f: *const gguf.File, key: []const u8, n: usize) ?u32 {
        const v = f.getInt(key) orelse return null;
        if (v < 0 or v >= n) return null;
        return @intCast(v);
    }
};

fn buildTokenizerJson(a: Allocator, f: *const gguf.File) ![]const u8 {
    const v = try Vocab.load(a, f);
    const is_spm = std.mem.eql(u8, v.model, "llama");
    if (!is_spm and !std.mem.eql(u8, v.model, "gpt2")) {
        std.log.err("unsupported tokenizer.ggml.model: {s} (expected gpt2 or llama)", .{v.model});
        return error.UnsupportedTokenizer;
    }
    var out: Io.Writer.Allocating = .init(a);
    const w = &out.writer;
    try w.writeAll("{\"version\":\"1.0\",\"added_tokens\":[");
    var first = true;
    for (v.tokens, 0..) |t, i| {
        const special = switch (v.types[i]) {
            .control => true,
            .user_defined => false,
            else => continue,
        };
        if (!first) try w.writeAll(",");
        first = false;
        try w.print("{{\"id\":{d},\"content\":", .{i});
        try jsonStr(w, t);
        try w.print(",\"single_word\":false,\"lstrip\":false,\"rstrip\":false,\"normalized\":false,\"special\":{s}}}", .{if (special) "true" else "false"});
    }
    try w.writeAll("]");
    if (is_spm) {
        try w.writeAll(",\"normalizer\":{\"type\":\"Sequence\",\"normalizers\":[");
        if (v.add_space_prefix) try w.writeAll("{\"type\":\"Prepend\",\"prepend\":\"\xe2\x96\x81\"},");
        try w.writeAll("{\"type\":\"Replace\",\"pattern\":{\"String\":\" \"},\"content\":\"\xe2\x96\x81\"}]}");
        try w.writeAll(",\"pre_tokenizer\":null");
        try w.writeAll(",\"decoder\":{\"type\":\"Sequence\",\"decoders\":[{\"type\":\"Replace\",\"pattern\":{\"String\":\"\xe2\x96\x81\"},\"content\":\" \"},{\"type\":\"ByteFallback\"},{\"type\":\"Fuse\"}");
        if (v.add_space_prefix) try w.writeAll(",{\"type\":\"Strip\",\"content\":\" \",\"start\":1,\"stop\":0}");
        try w.writeAll("]}");
    } else {
        try w.writeAll(",\"normalizer\":null,\"pre_tokenizer\":{\"type\":\"Sequence\",\"pretokenizers\":[{\"type\":\"Split\",\"pattern\":{\"Regex\":");
        try jsonStr(w, regexForPre(v.pre));
        try w.writeAll("},\"behavior\":\"Isolated\",\"invert\":false},{\"type\":\"ByteLevel\",\"add_prefix_space\":false,\"trim_offsets\":false,\"use_regex\":false}]}");
        try w.writeAll(",\"decoder\":{\"type\":\"ByteLevel\",\"add_prefix_space\":true,\"trim_offsets\":true,\"use_regex\":true}");
    }
    if (v.add_bos and v.bos != null) {
        const bos = v.tokens[v.bos.?];
        try w.writeAll(",\"post_processor\":{\"type\":\"TemplateProcessing\",\"single\":[{\"SpecialToken\":{\"id\":");
        try jsonStr(w, bos);
        try w.writeAll(",\"type_id\":0}},{\"Sequence\":{\"id\":\"A\",\"type_id\":0}}],\"pair\":[],\"special_tokens\":{");
        try jsonStr(w, bos);
        try w.writeAll(":{\"id\":");
        try jsonStr(w, bos);
        try w.print(",\"ids\":[{d}],\"tokens\":[", .{v.bos.?});
        try jsonStr(w, bos);
        try w.writeAll("]}}}");
    } else {
        try w.writeAll(",\"post_processor\":null");
    }
    // Model.
    try w.writeAll(",\"model\":{\"type\":\"BPE\",\"dropout\":null,\"continuing_subword_prefix\":null,\"end_of_word_suffix\":null,\"fuse_unk\":");
    try w.writeAll(if (is_spm) "true" else "false");
    try w.print(",\"byte_fallback\":{s},\"ignore_merges\":{s},\"unk_token\":", .{ if (is_spm) "true" else "false", if (isLlamaBpe(v.pre)) "true" else "false" });
    if (v.unk) |u| try jsonStr(w, v.tokens[u]) else try w.writeAll("null");
    try w.writeAll(",\"vocab\":{");
    var seen = std.StringHashMapUnmanaged(u32){};
    first = true;
    for (v.tokens, 0..) |t, i| {
        if (seen.contains(t)) continue;
        try seen.put(a, t, @intCast(i));
        if (!first) try w.writeAll(",");
        first = false;
        try jsonStr(w, t);
        try w.print(":{d}", .{i});
    }
    try w.writeAll("},\"merges\":[");
    if (v.merges) |merges| {
        for (merges, 0..) |m, i| {
            if (i > 0) try w.writeAll(",");
            try jsonStr(w, m);
        }
    } else if (is_spm) {
        try writeMergesFromScores(a, w, v, &seen);
    }
    try w.writeAll("]}}");
    return out.toOwnedSlice();
}

/// A SentencePiece vocabulary carries scores instead of merges: llama.cpp merges
/// the adjacent pair whose concatenation has the highest score. Every split of
/// every multi-character token becomes a merge, ordered by descending score.
fn writeMergesFromScores(a: Allocator, w: *Io.Writer, v: Vocab, seen: *const std.StringHashMapUnmanaged(u32)) !void {
    const Cand = struct { score: f32, id: u32, left: []const u8, right: []const u8 };
    var cands = std.ArrayList(Cand).empty;
    for (v.tokens, 0..) |t, i| {
        if (v.types[i] != .normal or t.len < 2) continue;
        var cut: usize = 1;
        while (cut < t.len) : (cut += 1) {
            const l = t[0..cut];
            const r = t[cut..];
            if (seen.contains(l) and seen.contains(r)) {
                try cands.append(a, .{ .score = if (v.scores) |s| s[i] else 0, .id = @intCast(i), .left = l, .right = r });
            }
        }
    }
    std.mem.sort(Cand, cands.items, {}, struct {
        fn lt(_: void, x: Cand, y: Cand) bool {
            if (x.score != y.score) return x.score > y.score;
            return x.id < y.id;
        }
    }.lt);
    for (cands.items, 0..) |c, i| {
        if (i > 0) try w.writeAll(",");
        var buf: Io.Writer.Allocating = .init(a);
        try buf.writer.print("{s} {s}", .{ c.left, c.right });
        try jsonStr(w, buf.written());
    }
}

fn buildTokenizerConfigJson(a: Allocator, f: *const gguf.File) ![]const u8 {
    const v = try Vocab.load(a, f);
    var out: Io.Writer.Allocating = .init(a);
    const w = &out.writer;
    try w.writeAll("{\"tokenizer_class\":\"PreTrainedTokenizerFast\"");
    if (v.bos) |b| {
        try w.writeAll(",\"bos_token\":");
        try jsonStr(w, v.tokens[b]);
    }
    if (v.eos) |e| {
        try w.writeAll(",\"eos_token\":");
        try jsonStr(w, v.tokens[e]);
    }
    if (v.unk) |u| {
        try w.writeAll(",\"unk_token\":");
        try jsonStr(w, v.tokens[u]);
    }
    if (v.pad) |p| {
        try w.writeAll(",\"pad_token\":");
        try jsonStr(w, v.tokens[p]);
    }
    try w.print(",\"add_bos_token\":{s}", .{if (v.add_bos and v.bos != null) "true" else "false"});
    if (f.getStr("tokenizer.chat_template")) |t| {
        try w.writeAll(",\"chat_template\":");
        try jsonStr(w, t);
    }
    try w.writeAll("}");
    return out.toOwnedSlice();
}

fn buildGenerationConfigJson(a: Allocator, f: *const gguf.File) ![]const u8 {
    var out: Io.Writer.Allocating = .init(a);
    const w = &out.writer;
    try w.writeAll("{\"eos_token_id\":[");
    var first = true;
    var seen: [3]i64 = .{ -1, -1, -1 };
    var n: usize = 0;
    for ([_][]const u8{ "tokenizer.ggml.eos_token_id", "tokenizer.ggml.eot_token_id", "tokenizer.ggml.eom_token_id" }) |key| {
        const v = f.getInt(key) orelse continue;
        if (v < 0) continue;
        var dup = false;
        for (seen[0..n]) |s| dup = dup or s == v;
        if (dup) continue;
        seen[n] = v;
        n += 1;
        if (!first) try w.writeAll(",");
        first = false;
        try w.print("{d}", .{v});
    }
    try w.writeAll("]");
    if (f.getInt("tokenizer.ggml.bos_token_id")) |b| try w.print(",\"bos_token_id\":{d}", .{b});
    try w.writeAll(",\"do_sample\":false}");
    return out.toOwnedSlice();
}

// ---------------------------------------------------------------------------
// Tensor names
// ---------------------------------------------------------------------------

/// llama.cpp's `permute` for the llama family: within every head the rows
/// `[2][head_dim/2]` become `[head_dim/2][2]`. `forward` maps Hugging Face
/// order to GGUF order; the inverse undoes it. Works on raw row bytes, so it
/// applies to quantised rows too.
pub fn permuteRows(out: []u8, in: []const u8, rows: usize, row_bytes: usize, n_head: usize, forward: bool) void {
    std.debug.assert(rows % (2 * n_head) == 0);
    const hd = rows / n_head;
    const half = hd / 2;
    var h: usize = 0;
    while (h < n_head) : (h += 1) {
        var i: usize = 0;
        while (i < half) : (i += 1) {
            var j: usize = 0;
            while (j < 2) : (j += 1) {
                const hf_row = h * hd + j * half + i;
                const gguf_row = h * hd + 2 * i + j;
                const src_row = if (forward) hf_row else gguf_row;
                const dst_row = if (forward) gguf_row else hf_row;
                @memcpy(out[dst_row * row_bytes ..][0..row_bytes], in[src_row * row_bytes ..][0..row_bytes]);
            }
        }
    }
}

/// Whether llama.cpp permutes q/k for this family.
pub fn permutesQk(family: Family) bool {
    return family == .llama or family == .mistral or family == .mixtral;
}

fn rowBytesOf(t: gguf.TensorInfo, dt: tensor.DType) !usize {
    if (t.dims.len >= 2) return dt.rowBytes(@intCast(t.dims[0]));
    // 1-D: one element per "row".
    if (dt.blockSize() != 1) return error.InvalidGguf;
    return dt.blockBytes();
}

fn mapTensors(src: *Source, arena: Allocator, config: *const model_mod.Config) !void {
    const f = src.file;
    const family = src.family;
    const is_gemma = family == .gemma2 or family == .gemma3;
    const mixtral = family == .mixtral;
    var extra = std.ArrayList(gguf.TensorInfo).empty;
    var it = f.tensors.iterator();
    while (it.next()) |kv| {
        const t = kv.value_ptr.*;
        src.total_bytes += t.data.len;
        const dt = t.dtype() orelse continue;
        var hf: ?[]const u8 = null;
        var layer: ?usize = null;
        var permute_heads: ?usize = null;
        var experts: ?struct { kind: []const u8 } = null;
        if (std.mem.eql(u8, t.name, "token_embd.weight")) {
            hf = "model.embed_tokens.weight";
        } else if (std.mem.eql(u8, t.name, "output_norm.weight")) {
            hf = "model.norm.weight";
        } else if (std.mem.eql(u8, t.name, "output.weight")) {
            hf = "lm_head.weight";
        } else if (std.mem.startsWith(u8, t.name, "blk.")) {
            const rest = t.name["blk.".len..];
            const dot = std.mem.indexOfScalar(u8, rest, '.') orelse continue;
            layer = std.fmt.parseInt(usize, rest[0..dot], 10) catch continue;
            const tail = rest[dot + 1 ..];
            const mlp: []const u8 = if (mixtral) "block_sparse_moe." else "mlp.";
            const Pair = struct { g: []const u8, h: []const u8 };
            const simple = [_]Pair{
                .{ .g = "attn_norm.weight", .h = "input_layernorm.weight" },
                .{ .g = "attn_v.weight", .h = "self_attn.v_proj.weight" },
                .{ .g = "attn_v.bias", .h = "self_attn.v_proj.bias" },
                .{ .g = "attn_output.weight", .h = "self_attn.o_proj.weight" },
                .{ .g = "attn_output.bias", .h = "self_attn.o_proj.bias" },
                .{ .g = "attn_q_norm.weight", .h = "self_attn.q_norm.weight" },
                .{ .g = "attn_k_norm.weight", .h = "self_attn.k_norm.weight" },
                .{ .g = "post_attention_norm.weight", .h = "post_attention_layernorm.weight" },
                .{ .g = "post_ffw_norm.weight", .h = "post_feedforward_layernorm.weight" },
                .{ .g = "ffn_gate.weight", .h = "mlp.gate_proj.weight" },
                .{ .g = "ffn_up.weight", .h = "mlp.up_proj.weight" },
                .{ .g = "ffn_down.weight", .h = "mlp.down_proj.weight" },
                .{ .g = "ffn_gate_shexp.weight", .h = "mlp.shared_expert.gate_proj.weight" },
                .{ .g = "ffn_up_shexp.weight", .h = "mlp.shared_expert.up_proj.weight" },
                .{ .g = "ffn_down_shexp.weight", .h = "mlp.shared_expert.down_proj.weight" },
                .{ .g = "ffn_gate_inp_shexp.weight", .h = "mlp.shared_expert_gate.weight" },
            };
            var suffix: ?[]const u8 = null;
            for (simple) |p| {
                if (std.mem.eql(u8, tail, p.g)) {
                    suffix = p.h;
                    break;
                }
            }
            if (suffix == null) {
                if (std.mem.eql(u8, tail, "attn_q.weight") or std.mem.eql(u8, tail, "attn_q.bias")) {
                    suffix = if (tail[tail.len - 1] == 't') "self_attn.q_proj.weight" else "self_attn.q_proj.bias";
                    if (permutesQk(family)) permute_heads = config.num_heads;
                } else if (std.mem.eql(u8, tail, "attn_k.weight") or std.mem.eql(u8, tail, "attn_k.bias")) {
                    suffix = if (tail[tail.len - 1] == 't') "self_attn.k_proj.weight" else "self_attn.k_proj.bias";
                    if (permutesQk(family)) permute_heads = config.num_kv_heads;
                } else if (std.mem.eql(u8, tail, "ffn_norm.weight")) {
                    suffix = if (is_gemma) "pre_feedforward_layernorm.weight" else "post_attention_layernorm.weight";
                } else if (std.mem.eql(u8, tail, "ffn_gate_inp.weight")) {
                    suffix = try std.mem.concat(arena, u8, &.{ mlp, "gate.weight" });
                } else if (std.mem.eql(u8, tail, "ffn_gate_exps.weight")) {
                    experts = .{ .kind = if (mixtral) "w1" else "gate_proj" };
                } else if (std.mem.eql(u8, tail, "ffn_up_exps.weight")) {
                    experts = .{ .kind = if (mixtral) "w3" else "up_proj" };
                } else if (std.mem.eql(u8, tail, "ffn_down_exps.weight")) {
                    experts = .{ .kind = if (mixtral) "w2" else "down_proj" };
                }
            }
            if (suffix) |sfx| hf = try std.fmt.allocPrint(arena, "model.layers.{d}.{s}", .{ layer.?, sfx });
        }

        if (experts) |ex| {
            if (t.dims.len != 3) return error.InvalidGguf;
            const n_exp: usize = @intCast(t.dims[2]);
            const rows: usize = @intCast(t.dims[1]);
            const cols: usize = @intCast(t.dims[0]);
            const rb = dt.rowBytes(cols);
            for (0..n_exp) |e| {
                const name = try std.fmt.allocPrint(arena, "model.layers.{d}.{s}experts.{d}.{s}.weight", .{ layer.?, mlp_prefix(mixtral), e, ex.kind });
                const shape = try arena.alloc(usize, 2);
                shape[0] = rows;
                shape[1] = cols;
                try src.tensors.put(arena, name, .{ .name = name, .dtype = dt, .shape = shape, .data = t.data[e * rows * rb ..][0 .. rows * rb] });
            }
            continue;
        }
        const name = hf orelse {
            try extra.append(arena, t);
            continue;
        };
        const shape = try t.shape(arena);
        var data = t.data;
        var dtype = dt;
        if (permute_heads) |n_head| {
            const rows: usize = if (shape.len >= 2) shape[0] else t.numel();
            const rb = try rowBytesOf(t, dt);
            const copy = try arena.alloc(u8, data.len);
            permuteRows(copy, data, rows, rb, n_head, false);
            data = copy;
        }
        if (is_gemma and std.mem.endsWith(u8, name, "norm.weight")) {
            // llama.cpp stores gemma norms as (1 + w); Hugging Face stores w.
            const vals = try arena.alloc(f32, t.numel());
            tensor.convertToF32(dt, data, vals);
            for (vals) |*v| v.* -= 1.0;
            data = std.mem.sliceAsBytes(vals);
            dtype = .f32;
        }
        try src.tensors.put(arena, name, .{ .name = name, .dtype = dtype, .shape = shape, .data = data });
    }
    src.extra = extra.items;
}

fn mlp_prefix(mixtral: bool) []const u8 {
    return if (mixtral) "block_sparse_moe." else "mlp.";
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

test "llama q/k permutation round trip" {
    const rows = 8; // 2 heads of head_dim 4
    const rb = 3;
    var in: [rows * rb]u8 = undefined;
    for (&in, 0..) |*b, i| b.* = @intCast(i);
    var perm: [rows * rb]u8 = undefined;
    var back: [rows * rb]u8 = undefined;
    permuteRows(&perm, &in, rows, rb, 2, true);
    // head 0: HF rows [0,1,2,3] -> GGUF rows [0,2,1,3]
    try std.testing.expectEqualSlices(u8, in[2 * rb ..][0..rb], perm[1 * rb ..][0..rb]);
    try std.testing.expectEqualSlices(u8, in[1 * rb ..][0..rb], perm[2 * rb ..][0..rb]);
    try std.testing.expectEqualSlices(u8, in[6 * rb ..][0..rb], perm[5 * rb ..][0..rb]);
    permuteRows(&back, &perm, rows, rb, 2, false);
    try std.testing.expectEqualSlices(u8, &in, &back);
}

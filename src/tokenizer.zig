//! A Hugging Face `tokenizer.json` compatible BPE tokenizer.
//!
//! Supports the tokenizer configurations shipped by decoder-only models:
//! byte-level BPE with the GPT-2 / Qwen2 / Llama-3 / o200k / DeepSeek style
//! regex pre-tokenisers (plus the `Digits`, `Punctuation`, `Whitespace` and
//! `ByteLevel(add_prefix_space)` steps some families chain in a `Sequence`),
//! and SentencePiece-derived BPE with Metaspace handling and byte fallback.
//! Unicode normalisers (NFC/NFKC/Precompiled) are approximated by the
//! identity with a warning.

const std = @import("std");
const uni = @import("unicode_tables.zig");

const Allocator = std.mem.Allocator;

pub const AddedToken = struct {
    id: u32,
    content: []const u8,
    special: bool,
};

/// The `Split` regular expressions the splitter implements natively.
const RegexKind = enum {
    gpt2,
    qwen2,
    llama3,
    /// tiktoken o200k (gpt-oss): words split at lower→upper case changes, contractions as suffixes.
    o200k,
    /// DeepSeek V3 main pattern (`[\p{P}\p{S}]` classes; digits are split by an earlier `\p{N}{1,3}` step).
    deepseek3,
    /// `\p{N}{1,3}` alone.
    digits3,
    /// `[\r\n]`
    newlines,
    /// DeepSeek V2 letters: `\s?[A-Za-z...]+`
    ds2_letters,
    /// DeepSeek V2 punctuation: `\s?[!-/:-~！-／：-～‘-‟　-。]+`
    ds2_punct,
    /// `\s+$`
    trailing_ws,
    /// CJK runs `[一-龥ࠀ-一가-퟿]+`
    cjk,
};

/// One pre-tokenisation step of a `Sequence`.
const Step = union(enum) {
    regex: RegexKind,
    split_string: struct { pattern: []const u8, removed: bool },
    /// `Digits(individual_digits)`
    digits: bool,
    /// `Punctuation` (true: contiguous runs stay together)
    punctuation: bool,
    whitespace,
    whitespace_split,
    /// `ByteLevel(add_prefix_space)`; the regex (when `use_regex`) is a separate step.
    byte_level: bool,
    metaspace: struct { prepend: bool, split: bool },
};

const Decoder = enum { byte_level, metaspace, plain };

pub const Tokenizer = struct {
    arena: std.heap.ArenaAllocator,
    gpa: Allocator,

    vocab: std.StringHashMapUnmanaged(u32),
    id_to_token: [][]const u8,
    merges: std.StringHashMapUnmanaged(u32), // "a b" -> rank
    added: []AddedToken,
    added_by_id: std.AutoHashMapUnmanaged(u32, usize),
    special_ids: std.AutoHashMapUnmanaged(u32, void),

    byte_fallback: bool,
    ignore_merges: bool,
    unk_id: ?u32,
    // normalizer
    prepend: ?[]const u8,
    replace_space: ?[]const u8, // replacement for " " (typically "▁")
    lowercase: bool,
    /// Pre-tokenisation steps applied in order to every segment.
    steps: []const Step,
    /// Pieces are mapped through the GPT-2 byte→unicode table before BPE.
    byte_level: bool,
    has_metaspace: bool,
    decoder: Decoder,
    strip_leading_space: bool,
    bos_id: ?u32,
    add_bos: bool,
    eos_id: ?u32,

    /// Word -> ids cache. The tokenizer is not thread-safe; encode from one thread.
    cache: std.StringHashMapUnmanaged([]const u32),

    byte_encoder: [256]u21,
    byte_decoder: std.AutoHashMapUnmanaged(u21, u8),

    pub fn deinit(self: *Tokenizer) void {
        const gpa = self.gpa;
        self.arena.deinit();
        gpa.destroy(self);
    }

    /// Parses `tokenizer.json`. `tokenizer_config_json` (optional) is used for bos/eos token names.
    pub fn parse(gpa: Allocator, json_text: []const u8, tokenizer_config_json: ?[]const u8) !*Tokenizer {
        const self = try gpa.create(Tokenizer);
        errdefer gpa.destroy(self);
        self.* = .{
            .arena = std.heap.ArenaAllocator.init(gpa),
            .gpa = gpa,
            .vocab = .{},
            .id_to_token = &.{},
            .merges = .{},
            .added = &.{},
            .added_by_id = .{},
            .special_ids = .{},
            .byte_fallback = false,
            .ignore_merges = false,
            .unk_id = null,
            .prepend = null,
            .replace_space = null,
            .lowercase = false,
            .steps = &.{},
            .byte_level = false,
            .has_metaspace = false,
            .decoder = .plain,
            .strip_leading_space = false,
            .bos_id = null,
            .add_bos = false,
            .eos_id = null,
            .cache = .{},
            .byte_encoder = undefined,
            .byte_decoder = .{},
        };
        errdefer self.arena.deinit();
        const arena = self.arena.allocator();

        // GPT-2 byte <-> unicode mapping.
        {
            var n: u21 = 0;
            var b: usize = 0;
            while (b < 256) : (b += 1) {
                const printable = (b >= '!' and b <= '~') or (b >= 0xA1 and b <= 0xAC) or (b >= 0xAE and b <= 0xFF);
                if (printable) {
                    self.byte_encoder[b] = @intCast(b);
                } else {
                    self.byte_encoder[b] = 256 + n;
                    n += 1;
                }
                try self.byte_decoder.put(arena, self.byte_encoder[b], @intCast(b));
            }
        }

        var parsed = try std.json.parseFromSlice(std.json.Value, gpa, json_text, .{});
        defer parsed.deinit();
        const root = parsed.value.object;

        // --- model ---
        const model = (root.get("model") orelse return error.InvalidTokenizer).object;
        const model_type = if (model.get("type")) |t| t.string else "BPE";
        if (!std.mem.eql(u8, model_type, "BPE")) {
            std.log.err("unsupported tokenizer model type: {s}", .{model_type});
            return error.UnsupportedTokenizer;
        }
        if (model.get("byte_fallback")) |v| self.byte_fallback = v == .bool and v.bool;
        if (model.get("ignore_merges")) |v| self.ignore_merges = v == .bool and v.bool;

        const vocab = (model.get("vocab") orelse return error.InvalidTokenizer).object;
        var max_id: usize = 0;
        {
            var it = vocab.iterator();
            while (it.next()) |e| {
                const id: u32 = @intCast(e.value_ptr.integer);
                try self.vocab.put(arena, try arena.dupe(u8, e.key_ptr.*), id);
                max_id = @max(max_id, id);
            }
        }
        if (root.get("added_tokens")) |at| {
            for (at.array.items) |t| {
                const id: u32 = @intCast(t.object.get("id").?.integer);
                max_id = @max(max_id, id);
            }
        }
        self.id_to_token = try arena.alloc([]const u8, max_id + 1);
        @memset(self.id_to_token, "");
        {
            var it = self.vocab.iterator();
            while (it.next()) |e| self.id_to_token[e.value_ptr.*] = e.key_ptr.*;
        }
        if (model.get("unk_token")) |u| {
            if (u == .string) self.unk_id = self.vocab.get(u.string);
        }

        if (model.get("merges")) |m| {
            for (m.array.items, 0..) |item, rank| {
                const key = switch (item) {
                    .string => |s| try arena.dupe(u8, s),
                    .array => |a| try std.fmt.allocPrint(arena, "{s} {s}", .{ a.items[0].string, a.items[1].string }),
                    else => return error.InvalidTokenizer,
                };
                try self.merges.put(arena, key, @intCast(rank));
            }
        }

        // --- added tokens ---
        if (root.get("added_tokens")) |at| {
            var list = std.ArrayList(AddedToken).empty;
            for (at.array.items) |t| {
                const obj = t.object;
                const id: u32 = @intCast(obj.get("id").?.integer);
                const content = try arena.dupe(u8, obj.get("content").?.string);
                const special = if (obj.get("special")) |s| s.bool else false;
                try list.append(arena, .{ .id = id, .content = content, .special = special });
                self.id_to_token[id] = content;
                if (!self.vocab.contains(content)) try self.vocab.put(arena, content, id);
                if (special) try self.special_ids.put(arena, id, {});
            }
            // Longest first so greedy matching prefers longer tokens.
            std.mem.sort(AddedToken, list.items, {}, struct {
                fn lt(_: void, a: AddedToken, b: AddedToken) bool {
                    return a.content.len > b.content.len;
                }
            }.lt);
            self.added = list.items;
            for (self.added, 0..) |a, i| try self.added_by_id.put(arena, a.id, i);
        }

        // --- normalizer ---
        if (root.get("normalizer")) |n| try self.parseNormalizer(n);

        // --- pre-tokenizer ---
        if (root.get("pre_tokenizer")) |p| {
            var steps = std.ArrayList(Step).empty;
            try self.parsePreTokenizer(p, &steps);
            self.steps = steps.items;
            for (self.steps) |st| {
                if (st == .byte_level) self.byte_level = true;
                if (st == .metaspace) self.has_metaspace = true;
            }
        }

        // --- decoder ---
        if (root.get("decoder")) |d| self.parseDecoder(d);
        if (self.byte_level) self.decoder = .byte_level;

        // --- post-processor (BOS) ---
        if (root.get("post_processor")) |pp| self.parsePostProcessor(pp);

        // --- tokenizer_config.json: bos/eos names ---
        if (tokenizer_config_json) |cfg_text| {
            var cfg = try std.json.parseFromSlice(std.json.Value, gpa, cfg_text, .{});
            defer cfg.deinit();
            if (cfg.value == .object) {
                if (tokenName(cfg.value.object.get("bos_token"))) |name| {
                    if (self.vocab.get(name)) |id| self.bos_id = id;
                }
                if (tokenName(cfg.value.object.get("eos_token"))) |name| {
                    if (self.vocab.get(name)) |id| self.eos_id = id;
                }
                if (cfg.value.object.get("add_bos_token")) |v| {
                    if (v == .bool) self.add_bos = v.bool and self.bos_id != null;
                }
            }
        }
        return self;
    }

    fn tokenName(v: ?std.json.Value) ?[]const u8 {
        const val = v orelse return null;
        return switch (val) {
            .string => |s| s,
            .object => |o| if (o.get("content")) |c| (if (c == .string) c.string else null) else null,
            else => null,
        };
    }

    fn parseNormalizer(self: *Tokenizer, n: std.json.Value) !void {
        if (n != .object) return;
        const obj = n.object;
        const t = (obj.get("type") orelse return).string;
        if (std.mem.eql(u8, t, "Sequence")) {
            for (obj.get("normalizers").?.array.items) |sub| try self.parseNormalizer(sub);
        } else if (std.mem.eql(u8, t, "Prepend")) {
            self.prepend = try self.arena.allocator().dupe(u8, obj.get("prepend").?.string);
        } else if (std.mem.eql(u8, t, "Replace")) {
            const pattern = obj.get("pattern").?.object;
            if (pattern.get("String")) |s| {
                if (std.mem.eql(u8, s.string, " ")) {
                    self.replace_space = try self.arena.allocator().dupe(u8, obj.get("content").?.string);
                }
            }
        } else if (std.mem.eql(u8, t, "Lowercase")) {
            self.lowercase = true;
        } else if (std.mem.eql(u8, t, "Strip") or std.mem.eql(u8, t, "NFC")) {
            // Whitespace stripping of the whole input and NFC are identities for the prompts ditch builds.
        } else {
            std.log.warn("normalizer '{s}' is approximated by the identity (text that is not already normalised may tokenize differently)", .{t});
        }
    }

    fn parsePreTokenizer(self: *Tokenizer, p: std.json.Value, steps: *std.ArrayList(Step)) !void {
        if (p != .object) return;
        const arena = self.arena.allocator();
        const obj = p.object;
        const t = (obj.get("type") orelse return).string;
        if (std.mem.eql(u8, t, "Sequence")) {
            for (obj.get("pretokenizers").?.array.items) |sub| try self.parsePreTokenizer(sub, steps);
        } else if (std.mem.eql(u8, t, "Split")) {
            const pattern = obj.get("pattern").?.object;
            const behavior = if (obj.get("behavior")) |b| (if (b == .string) b.string else "isolated") else "isolated";
            if (pattern.get("Regex")) |r| {
                try steps.append(arena, .{ .regex = classifyRegex(r.string) });
            } else if (pattern.get("String")) |str| {
                try steps.append(arena, .{ .split_string = .{ .pattern = try arena.dupe(u8, str.string), .removed = std.ascii.eqlIgnoreCase(behavior, "removed") } });
            }
        } else if (std.mem.eql(u8, t, "ByteLevel")) {
            const use_regex = if (obj.get("use_regex")) |u| u.bool else true;
            const add_prefix_space = if (obj.get("add_prefix_space")) |u| u.bool else false;
            try steps.append(arena, .{ .byte_level = add_prefix_space });
            if (use_regex) try steps.append(arena, .{ .regex = .gpt2 });
        } else if (std.mem.eql(u8, t, "Metaspace")) {
            const scheme = if (obj.get("prepend_scheme")) |s| s.string else "always";
            const split = if (obj.get("split")) |s| s.bool else true;
            try steps.append(arena, .{ .metaspace = .{ .prepend = !std.mem.eql(u8, scheme, "never"), .split = split } });
            if (obj.get("replacement")) |r| self.replace_space = try arena.dupe(u8, r.string);
        } else if (std.mem.eql(u8, t, "Digits")) {
            const individual = if (obj.get("individual_digits")) |v| v.bool else false;
            try steps.append(arena, .{ .digits = individual });
        } else if (std.mem.eql(u8, t, "Punctuation")) {
            const behavior = if (obj.get("behavior")) |b| (if (b == .string) b.string else "isolated") else "isolated";
            try steps.append(arena, .{ .punctuation = std.ascii.eqlIgnoreCase(behavior, "contiguous") });
        } else if (std.mem.eql(u8, t, "Whitespace")) {
            try steps.append(arena, .whitespace);
        } else if (std.mem.eql(u8, t, "WhitespaceSplit")) {
            try steps.append(arena, .whitespace_split);
        } else {
            std.log.warn("ignoring unsupported pre-tokenizer: {s}", .{t});
        }
    }

    fn classifyRegex(r: []const u8) RegexKind {
        const has = struct {
            fn f(hay: []const u8, needle: []const u8) bool {
                return std.mem.indexOf(u8, hay, needle) != null;
            }
        }.f;
        if (std.mem.eql(u8, r, "\\p{N}{1,3}")) return .digits3;
        if (std.mem.eql(u8, r, "[\\r\\n]")) return .newlines;
        if (std.mem.eql(u8, r, "\\s+$")) return .trailing_ws;
        if (std.mem.startsWith(u8, r, "\\s?[A-Za-z")) return .ds2_letters;
        if (std.mem.startsWith(u8, r, "\\s?[!-/")) return .ds2_punct;
        if (std.mem.startsWith(u8, r, "[\xe4\xb8\x80-")) return .cjk;
        if (has(r, "\\p{Lu}")) return .o200k;
        if (has(r, "\\p{P}\\p{S}")) return .deepseek3;
        if (has(r, "{1,3}")) return .llama3;
        if (has(r, "[^\\r\\n\\p{L}\\p{N}]?\\p{L}+")) return .qwen2;
        if (!has(r, "'s|'t|'re")) std.log.warn("pre-tokenizer regex is not recognised; using the GPT-2 pattern: {s}", .{r});
        return .gpt2;
    }

    fn parseDecoder(self: *Tokenizer, d: std.json.Value) void {
        if (d != .object) return;
        const obj = d.object;
        const t = (obj.get("type") orelse return).string;
        if (std.mem.eql(u8, t, "Sequence")) {
            for (obj.get("decoders").?.array.items) |sub| self.parseDecoder(sub);
        } else if (std.mem.eql(u8, t, "ByteLevel")) {
            self.decoder = .byte_level;
        } else if (std.mem.eql(u8, t, "Metaspace") or std.mem.eql(u8, t, "Replace") or std.mem.eql(u8, t, "ByteFallback")) {
            if (self.decoder != .byte_level) self.decoder = .metaspace;
        } else if (std.mem.eql(u8, t, "Strip")) {
            if (obj.get("start")) |s| self.strip_leading_space = s.integer > 0;
        }
    }

    fn parsePostProcessor(self: *Tokenizer, pp: std.json.Value) void {
        if (pp != .object) return;
        const obj = pp.object;
        const t = (obj.get("type") orelse return).string;
        if (std.mem.eql(u8, t, "Sequence")) {
            for (obj.get("processors").?.array.items) |sub| self.parsePostProcessor(sub);
        } else if (std.mem.eql(u8, t, "TemplateProcessing")) {
            const single = obj.get("single") orelse return;
            if (single != .array or single.array.items.len == 0) return;
            const first = single.array.items[0];
            if (first != .object) return;
            if (first.object.get("SpecialToken")) |st| {
                const id_name = st.object.get("id").?.string;
                if (self.vocab.get(id_name)) |id| {
                    self.bos_id = id;
                    self.add_bos = true;
                }
            }
        } else if (std.mem.eql(u8, t, "RobertaProcessing") or std.mem.eql(u8, t, "BertProcessing")) {
            // Not used by supported models.
        }
    }

    /// The llama.cpp `tokenizer.ggml.pre` name of the byte-level regex in use (null for SentencePiece-style tokenizers).
    pub fn ggmlPreName(self: *const Tokenizer) ?[]const u8 {
        if (!self.byte_level) return null;
        for (self.steps) |st| {
            if (st == .regex) return switch (st.regex) {
                .qwen2 => "qwen2",
                .llama3 => "llama-bpe",
                .o200k => "gpt-4o",
                .deepseek3, .digits3 => "deepseek-v3",
                .ds2_letters, .newlines, .ds2_punct, .trailing_ws, .cjk => "deepseek-llm",
                .gpt2 => "gpt-2",
            };
        }
        return "gpt-2";
    }

    pub fn vocabSize(self: *const Tokenizer) usize {
        return self.id_to_token.len;
    }

    pub fn tokenToId(self: *const Tokenizer, token: []const u8) ?u32 {
        return self.vocab.get(token);
    }

    pub fn isSpecial(self: *const Tokenizer, id: u32) bool {
        return self.special_ids.contains(id);
    }

    // -----------------------------------------------------------------------
    // Encoding
    // -----------------------------------------------------------------------

    /// Encodes text into token ids. `add_special` prepends BOS when the
    /// tokenizer is configured to do so and the text does not already start with it.
    pub fn encode(self: *Tokenizer, gpa: Allocator, text: []const u8, add_special: bool) ![]u32 {
        var out = std.ArrayList(u32).empty;
        errdefer out.deinit(gpa);
        if (add_special and self.add_bos) {
            if (self.bos_id) |bos| {
                const bos_text = self.id_to_token[bos];
                if (!std.mem.startsWith(u8, text, bos_text)) try out.append(gpa, bos);
            }
        }
        // Split on added tokens.
        var pos: usize = 0;
        var seg_start: usize = 0;
        while (pos < text.len) {
            var matched: ?AddedToken = null;
            for (self.added) |a| {
                if (a.content.len == 0) continue;
                if (std.mem.startsWith(u8, text[pos..], a.content)) {
                    matched = a;
                    break;
                }
            }
            if (matched) |a| {
                if (pos > seg_start) try self.encodeSegment(gpa, text[seg_start..pos], &out);
                try out.append(gpa, a.id);
                pos += a.content.len;
                seg_start = pos;
            } else {
                pos += 1;
            }
        }
        if (seg_start < text.len) try self.encodeSegment(gpa, text[seg_start..], &out);
        return out.toOwnedSlice(gpa);
    }

    fn encodeSegment(self: *Tokenizer, gpa: Allocator, raw: []const u8, out: *std.ArrayList(u32)) !void {
        var scratch = std.heap.ArenaAllocator.init(gpa);
        defer scratch.deinit();
        const a = scratch.allocator();
        // Normalise.
        var norm = std.ArrayList(u8).empty;
        if (self.prepend) |p| try norm.appendSlice(a, p);
        for (raw) |c| {
            if (c == ' ' and self.replace_space != null and !self.has_metaspace) {
                try norm.appendSlice(a, self.replace_space.?);
            } else if (self.lowercase) {
                try norm.append(a, std.ascii.toLower(c));
            } else {
                try norm.append(a, c);
            }
        }

        // Pre-tokenise: every step splits (or rewrites) the pieces of the previous one.
        var pieces = std.ArrayList([]const u8).empty;
        try pieces.append(a, norm.items);
        for (self.steps) |step| {
            var next = std.ArrayList([]const u8).empty;
            for (pieces.items) |piece| try self.applyStep(a, step, piece, &next);
            pieces = next;
        }
        for (pieces.items) |piece| {
            if (piece.len == 0) continue;
            if (self.byte_level) {
                const mapped = try self.byteLevelEncode(a, piece);
                try self.bpeWord(gpa, mapped, out);
            } else {
                try self.bpeWord(gpa, piece, out);
            }
        }
    }

    fn applyStep(self: *Tokenizer, a: Allocator, step: Step, piece: []const u8, out: *std.ArrayList([]const u8)) !void {
        switch (step) {
            .regex => |kind| {
                var it = RegexSplitter{ .text = piece, .kind = kind };
                while (it.next()) |w| try out.append(a, w);
            },
            .split_string => |sp| {
                var start: usize = 0;
                while (std.mem.indexOfPos(u8, piece, start, sp.pattern)) |idx| {
                    if (idx > start) try out.append(a, piece[start..idx]);
                    if (!sp.removed) try out.append(a, piece[idx .. idx + sp.pattern.len]);
                    start = idx + sp.pattern.len;
                }
                if (start < piece.len) try out.append(a, piece[start..]);
            },
            .digits => |individual| try splitClass(a, piece, isNumber, individual, true, out),
            .punctuation => |contiguous| try splitClass(a, piece, isPunctuation, !contiguous, true, out),
            .whitespace => {
                // `\w+|[^\w\s]+`: words, then runs of everything else; whitespace is dropped.
                var i: usize = 0;
                while (i < piece.len) {
                    const c = cpAtSlice(piece, i);
                    if (isWhitespace(c.cp)) {
                        i += c.len;
                        continue;
                    }
                    const word = isWordChar(c.cp);
                    const start = i;
                    while (i < piece.len) {
                        const n = cpAtSlice(piece, i);
                        if (isWhitespace(n.cp) or isWordChar(n.cp) != word) break;
                        i += n.len;
                    }
                    try out.append(a, piece[start..i]);
                }
            },
            .whitespace_split => {
                var i: usize = 0;
                while (i < piece.len) {
                    const c = cpAtSlice(piece, i);
                    if (isWhitespace(c.cp)) {
                        i += c.len;
                        continue;
                    }
                    const start = i;
                    while (i < piece.len and !isWhitespace(cpAtSlice(piece, i).cp)) i += cpAtSlice(piece, i).len;
                    try out.append(a, piece[start..i]);
                }
            },
            .byte_level => |add_prefix_space| {
                if (add_prefix_space and piece.len > 0 and piece[0] != ' ') {
                    try out.append(a, try std.mem.concat(a, u8, &.{ " ", piece }));
                } else {
                    try out.append(a, piece);
                }
            },
            .metaspace => |ms| {
                const rep = self.replace_space orelse "\xe2\x96\x81";
                var buf = std.ArrayList(u8).empty;
                if (ms.prepend and (piece.len == 0 or !std.mem.startsWith(u8, piece, rep)) and (piece.len == 0 or piece[0] != ' ')) {
                    try buf.appendSlice(a, rep);
                }
                for (piece) |c| {
                    if (c == ' ') try buf.appendSlice(a, rep) else try buf.append(a, c);
                }
                if (!ms.split) {
                    try out.append(a, buf.items);
                } else {
                    // Split so each piece starts with the replacement (MergedWithNext).
                    var start: usize = 0;
                    var i: usize = 0;
                    while (i < buf.items.len) {
                        if (i > start and std.mem.startsWith(u8, buf.items[i..], rep)) {
                            try out.append(a, buf.items[start..i]);
                            start = i;
                        }
                        i += std.unicode.utf8ByteSequenceLength(buf.items[i]) catch 1;
                    }
                    if (start < buf.items.len) try out.append(a, buf.items[start..]);
                }
            },
        }
    }

    fn byteLevelEncode(self: *const Tokenizer, gpa: Allocator, bytes: []const u8) ![]u8 {
        var out = std.ArrayList(u8).empty;
        errdefer out.deinit(gpa);
        for (bytes) |b| {
            var buf: [4]u8 = undefined;
            const n = try std.unicode.utf8Encode(self.byte_encoder[b], &buf);
            try out.appendSlice(gpa, buf[0..n]);
        }
        return out.toOwnedSlice(gpa);
    }

    const Symbol = struct { start: usize, end: usize };

    /// Runs byte-pair merging over one pre-tokenised word and appends ids.
    fn bpeWord(self: *Tokenizer, gpa: Allocator, word: []const u8, out: *std.ArrayList(u32)) !void {
        if (word.len == 0) return;
        if (self.cache.get(word)) |ids| {
            try out.appendSlice(gpa, ids);
            return;
        }
        var ids = std.ArrayList(u32).empty;
        defer ids.deinit(gpa);

        if (self.ignore_merges) {
            if (self.vocab.get(word)) |id| {
                try ids.append(gpa, id);
                try self.cacheWord(word, ids.items);
                try out.appendSlice(gpa, ids.items);
                return;
            }
        }

        // Initial symbols: one per UTF-8 character.
        var syms = std.ArrayList(Symbol).empty;
        defer syms.deinit(gpa);
        {
            var i: usize = 0;
            while (i < word.len) {
                const n = std.unicode.utf8ByteSequenceLength(word[i]) catch 1;
                const end = @min(word.len, i + n);
                try syms.append(gpa, .{ .start = i, .end = end });
                i = end;
            }
        }

        // Byte fallback for characters missing from the vocab happens after merging
        // in HF; a missing char can never merge, so handle it at emission time.
        while (syms.items.len > 1) {
            var best_rank: u32 = std.math.maxInt(u32);
            var best_i: usize = 0;
            var key_buf: [512]u8 = undefined;
            var i: usize = 0;
            while (i + 1 < syms.items.len) : (i += 1) {
                const a = word[syms.items[i].start..syms.items[i].end];
                const b = word[syms.items[i + 1].start..syms.items[i + 1].end];
                if (a.len + b.len + 1 > key_buf.len) continue;
                const key = std.fmt.bufPrint(&key_buf, "{s} {s}", .{ a, b }) catch continue;
                if (self.merges.get(key)) |rank| {
                    if (rank < best_rank) {
                        best_rank = rank;
                        best_i = i;
                    }
                }
            }
            if (best_rank == std.math.maxInt(u32)) break;
            syms.items[best_i].end = syms.items[best_i + 1].end;
            _ = syms.orderedRemove(best_i + 1);
        }

        for (syms.items) |s| {
            const piece = word[s.start..s.end];
            if (self.vocab.get(piece)) |id| {
                try ids.append(gpa, id);
            } else if (self.byte_fallback) {
                for (piece) |b| {
                    var nb: [8]u8 = undefined;
                    const name = std.fmt.bufPrint(&nb, "<0x{X:0>2}>", .{b}) catch unreachable;
                    if (self.vocab.get(name)) |id| {
                        try ids.append(gpa, id);
                    } else if (self.unk_id) |u| {
                        try ids.append(gpa, u);
                    }
                }
            } else if (self.unk_id) |u| {
                try ids.append(gpa, u);
            }
        }
        try self.cacheWord(word, ids.items);
        try out.appendSlice(gpa, ids.items);
    }

    fn cacheWord(self: *Tokenizer, word: []const u8, ids: []const u32) !void {
        if (self.cache.contains(word)) return;
        const arena = self.arena.allocator();
        try self.cache.put(arena, try arena.dupe(u8, word), try arena.dupe(u32, ids));
    }

    // -----------------------------------------------------------------------
    // Decoding
    // -----------------------------------------------------------------------

    pub fn decode(self: *const Tokenizer, gpa: Allocator, ids: []const u32, skip_special: bool) ![]u8 {
        var out = std.ArrayList(u8).empty;
        errdefer out.deinit(gpa);
        var first = true;
        for (ids) |id| {
            if (id >= self.id_to_token.len) continue;
            if (skip_special and self.special_ids.contains(id)) continue;
            const tok = self.id_to_token[id];
            const is_added = self.added_by_id.contains(id);
            if (is_added) {
                try out.appendSlice(gpa, tok);
            } else switch (self.decoder) {
                .byte_level => {
                    var it = std.unicode.Utf8View.initUnchecked(tok).iterator();
                    while (it.nextCodepoint()) |cp| {
                        if (self.byte_decoder.get(cp)) |b| try out.append(gpa, b) else {
                            var buf: [4]u8 = undefined;
                            const n = std.unicode.utf8Encode(cp, &buf) catch 0;
                            try out.appendSlice(gpa, buf[0..n]);
                        }
                    }
                },
                .metaspace, .plain => {
                    if (tok.len == 6 and std.mem.startsWith(u8, tok, "<0x") and tok[5] == '>') {
                        const b = std.fmt.parseInt(u8, tok[3..5], 16) catch 0;
                        try out.append(gpa, b);
                    } else {
                        const rep = self.replace_space orelse "\xe2\x96\x81";
                        var i: usize = 0;
                        while (i < tok.len) {
                            if (std.mem.startsWith(u8, tok[i..], rep)) {
                                if (!(first and self.strip_leading_space and out.items.len == 0)) try out.append(gpa, ' ');
                                i += rep.len;
                            } else {
                                try out.append(gpa, tok[i]);
                                i += 1;
                            }
                        }
                    }
                },
            }
            first = false;
        }
        return out.toOwnedSlice(gpa);
    }
};

// ---------------------------------------------------------------------------
// Regex-style pre-tokenisation for GPT-2 / Qwen2 / Llama-3 patterns
// ---------------------------------------------------------------------------

fn inRanges(cp: u21, ranges: []const uni.Range) bool {
    var lo: usize = 0;
    var hi: usize = ranges.len;
    while (lo < hi) {
        const mid = (lo + hi) / 2;
        const r = ranges[mid];
        if (cp < r.lo) {
            hi = mid;
        } else if (cp > r.hi) {
            lo = mid + 1;
        } else return true;
    }
    return false;
}

pub fn isLetter(cp: u21) bool {
    return inRanges(cp, &uni.letters);
}
pub fn isNumber(cp: u21) bool {
    return inRanges(cp, &uni.numbers);
}
pub fn isWhitespace(cp: u21) bool {
    return inRanges(cp, &uni.whitespace);
}

/// Unicode punctuation (approximated: ASCII punctuation, Latin-1 symbols and
/// the general / CJK / full-width punctuation blocks).
pub fn isPunctuation(cp: u21) bool {
    if (cp < 128) return std.ascii.isPunctuation(@intCast(cp));
    if (cp >= 0xA1 and cp <= 0xBF) return true;
    if (cp == 0xD7 or cp == 0xF7) return true;
    if (cp >= 0x2000 and cp <= 0x206F) return true;
    if (cp >= 0x3000 and cp <= 0x303F) return true;
    if (cp >= 0xFF00 and cp <= 0xFF0F) return true;
    if (cp >= 0xFF1A and cp <= 0xFF20) return true;
    if (cp >= 0xFF3B and cp <= 0xFF40) return true;
    if (cp >= 0xFF5B and cp <= 0xFF65) return true;
    return false;
}

fn isWordChar(cp: u21) bool {
    return cp == '_' or isLetter(cp) or isNumber(cp);
}

const Cp = struct { cp: u21, len: usize };

fn cpAtSlice(text: []const u8, i: usize) Cp {
    const n = std.unicode.utf8ByteSequenceLength(text[i]) catch return .{ .cp = text[i], .len = 1 };
    if (i + n > text.len) return .{ .cp = text[i], .len = 1 };
    const cp = std.unicode.utf8Decode(text[i .. i + n]) catch return .{ .cp = text[i], .len = 1 };
    return .{ .cp = cp, .len = n };
}

/// Splits `piece` around characters of a class: matching characters become
/// pieces of their own (`individual`) or runs; the rest stays as runs.
fn splitClass(a: Allocator, piece: []const u8, comptime pred: fn (u21) bool, individual: bool, keep: bool, out: *std.ArrayList([]const u8)) !void {
    var i: usize = 0;
    while (i < piece.len) {
        const c = cpAtSlice(piece, i);
        const start = i;
        if (pred(c.cp)) {
            i += c.len;
            if (!individual) {
                while (i < piece.len) {
                    const n = cpAtSlice(piece, i);
                    if (!pred(n.cp)) break;
                    i += n.len;
                }
            }
            if (keep) try out.append(a, piece[start..i]);
        } else {
            while (i < piece.len) {
                const n = cpAtSlice(piece, i);
                if (pred(n.cp)) break;
                i += n.len;
            }
            try out.append(a, piece[start..i]);
        }
    }
}

const RegexSplitter = struct {
    text: []const u8,
    kind: RegexKind,
    pos: usize = 0,

    fn cpAt(self: *const RegexSplitter, i: usize) ?Cp {
        if (i >= self.text.len) return null;
        return cpAtSlice(self.text, i);
    }

    fn isNewline(cp: u21) bool {
        return cp == '\r' or cp == '\n';
    }

    /// The next piece: a match of the pattern, or the run of text before the
    /// next match (`Isolated` behaviour) for patterns that do not cover every character.
    pub fn next(self: *RegexSplitter) ?[]const u8 {
        if (self.pos >= self.text.len) return null;
        const start = self.pos;
        var i = start;
        while (i < self.text.len) {
            const end = self.matchOne(i);
            if (end > i) {
                if (i > start) {
                    self.pos = i;
                    return self.text[start..i];
                }
                self.pos = end;
                return self.text[start..end];
            }
            i += self.cpAt(i).?.len;
        }
        self.pos = self.text.len;
        return self.text[start..];
    }

    fn isUpperAscii(cp: u21) bool {
        return cp >= 'A' and cp <= 'Z';
    }

    /// Optional English contraction (`'s`, `'ll`, ...) at `i`, case-insensitive; returns its end or `i`.
    fn contractionAt(self: *const RegexSplitter, i: usize) usize {
        const q = self.cpAt(i) orelse return i;
        if (q.cp != '\'') return i;
        const c1 = self.cpAt(i + 1) orelse return i;
        const l1 = std.ascii.toLower(@as(u8, if (c1.cp < 128) @intCast(c1.cp) else 0));
        if (l1 == 's' or l1 == 't' or l1 == 'm' or l1 == 'd') return i + 1 + c1.len;
        const c2 = self.cpAt(i + 1 + c1.len) orelse return i;
        const l2 = std.ascii.toLower(@as(u8, if (c2.cp < 128) @intCast(c2.cp) else 0));
        if ((l1 == 'r' and l2 == 'e') or (l1 == 'v' and l2 == 'e') or (l1 == 'l' and l2 == 'l')) return i + 1 + c1.len + c2.len;
        return i;
    }

    fn matchOne(self: *const RegexSplitter, start: usize) usize {
        const first = self.cpAt(start).?;
        switch (self.kind) {
            .gpt2, .qwen2, .llama3 => {},
            .o200k => return self.matchO200k(start, first),
            .deepseek3 => return self.matchDeepseek3(start, first),
            .digits3 => {
                if (!isNumber(first.cp)) return start;
                var i = start;
                var count: usize = 0;
                while (self.cpAt(i)) |n| {
                    if (!isNumber(n.cp) or count == 3) break;
                    i += n.len;
                    count += 1;
                }
                return i;
            },
            .newlines => return if (isNewline(first.cp)) start + first.len else start,
            .ds2_letters, .ds2_punct => {
                var i = start;
                var c = first;
                if (isWhitespace(c.cp)) {
                    const n = self.cpAt(i + c.len) orelse return start;
                    i += c.len;
                    c = n;
                }
                const letters = self.kind == .ds2_letters;
                if (!(if (letters) isLetter(c.cp) else isDs2Punct(c.cp))) return start;
                while (self.cpAt(i)) |n| {
                    if (!(if (letters) isLetter(n.cp) else isDs2Punct(n.cp))) break;
                    i += n.len;
                }
                return i;
            },
            .trailing_ws => {
                if (!isWhitespace(first.cp)) return start;
                var i = start;
                while (self.cpAt(i)) |n| {
                    if (!isWhitespace(n.cp)) return start;
                    i += n.len;
                }
                return i;
            },
            .cjk => {
                if (!isCjk(first.cp)) return start;
                var i = start;
                while (self.cpAt(i)) |n| {
                    if (!isCjk(n.cp)) break;
                    i += n.len;
                }
                return i;
            },
        }
        // 1. Contractions: 's 't 're 've 'm 'll 'd (case-insensitive for qwen2/llama3)
        if (first.cp == '\'') {
            if (self.cpAt(start + 1)) |c1| {
                const l1 = std.ascii.toLower(@as(u8, if (c1.cp < 128) @intCast(c1.cp) else 0));
                const ci = self.kind != .gpt2;
                const ok1 = if (ci) l1 else @as(u8, if (c1.cp < 128) @intCast(c1.cp) else 0);
                if (ok1 == 's' or ok1 == 't' or ok1 == 'm' or ok1 == 'd') return start + 1 + c1.len;
                if (self.cpAt(start + 1 + c1.len)) |c2| {
                    const l2 = std.ascii.toLower(@as(u8, if (c2.cp < 128) @intCast(c2.cp) else 0));
                    const ok2 = if (ci) l2 else @as(u8, if (c2.cp < 128) @intCast(c2.cp) else 0);
                    if ((ok1 == 'r' and ok2 == 'e') or (ok1 == 'v' and ok2 == 'e') or (ok1 == 'l' and ok2 == 'l')) return start + 1 + c1.len + c2.len;
                }
            }
        }
        switch (self.kind) {
            .gpt2 => {
                // ' ?\p{L}+'
                var i = start;
                var c = first;
                if (c.cp == ' ') {
                    if (self.cpAt(i + 1)) |n| {
                        if (isLetter(n.cp)) {
                            i += 1;
                            c = n;
                        }
                    }
                }
                if (isLetter(c.cp)) {
                    while (self.cpAt(i)) |n| {
                        if (!isLetter(n.cp)) break;
                        i += n.len;
                    }
                    return i;
                }
                // ' ?\p{N}+'
                i = start;
                c = first;
                if (c.cp == ' ') {
                    if (self.cpAt(i + 1)) |n| {
                        if (isNumber(n.cp)) {
                            i += 1;
                            c = n;
                        }
                    }
                }
                if (isNumber(c.cp)) {
                    while (self.cpAt(i)) |n| {
                        if (!isNumber(n.cp)) break;
                        i += n.len;
                    }
                    return i;
                }
                // ' ?[^\s\p{L}\p{N}]+'
                i = start;
                c = first;
                if (c.cp == ' ') {
                    if (self.cpAt(i + 1)) |n| {
                        if (!isWhitespace(n.cp) and !isLetter(n.cp) and !isNumber(n.cp)) {
                            i += 1;
                            c = n;
                        }
                    }
                }
                if (!isWhitespace(c.cp) and !isLetter(c.cp) and !isNumber(c.cp)) {
                    while (self.cpAt(i)) |n| {
                        if (isWhitespace(n.cp) or isLetter(n.cp) or isNumber(n.cp)) break;
                        i += n.len;
                    }
                    return i;
                }
                return self.matchWhitespace(start);
            },
            .o200k, .deepseek3, .digits3, .newlines, .ds2_letters, .ds2_punct, .trailing_ws, .cjk => unreachable,
            .qwen2, .llama3 => {
                // '[^\r\n\p{L}\p{N}]?\p{L}+'
                var i = start;
                var c = first;
                if (!isNewline(c.cp) and !isLetter(c.cp) and !isNumber(c.cp)) {
                    if (self.cpAt(i + c.len)) |n| {
                        if (isLetter(n.cp)) {
                            i += c.len;
                            c = n;
                        }
                    }
                }
                if (isLetter(c.cp)) {
                    while (self.cpAt(i)) |n| {
                        if (!isLetter(n.cp)) break;
                        i += n.len;
                    }
                    return i;
                }
                // '\p{N}' or '\p{N}{1,3}'
                if (isNumber(first.cp)) {
                    const max: usize = if (self.kind == .llama3) 3 else 1;
                    i = start;
                    var count: usize = 0;
                    while (self.cpAt(i)) |n| {
                        if (!isNumber(n.cp) or count == max) break;
                        i += n.len;
                        count += 1;
                    }
                    return i;
                }
                // ' ?[^\s\p{L}\p{N}]+[\r\n]*'
                i = start;
                c = first;
                if (c.cp == ' ') {
                    if (self.cpAt(i + 1)) |n| {
                        if (!isWhitespace(n.cp) and !isLetter(n.cp) and !isNumber(n.cp)) {
                            i += 1;
                            c = n;
                        }
                    }
                }
                if (!isWhitespace(c.cp) and !isLetter(c.cp) and !isNumber(c.cp)) {
                    while (self.cpAt(i)) |n| {
                        if (isWhitespace(n.cp) or isLetter(n.cp) or isNumber(n.cp)) break;
                        i += n.len;
                    }
                    while (self.cpAt(i)) |n| {
                        if (!isNewline(n.cp)) break;
                        i += n.len;
                    }
                    return i;
                }
                // '\s*[\r\n]+'
                i = start;
                var saw_nl = false;
                var j = start;
                while (self.cpAt(j)) |n| {
                    if (!isWhitespace(n.cp)) break;
                    j += n.len;
                    if (isNewline(n.cp)) {
                        saw_nl = true;
                        i = j;
                    }
                }
                if (saw_nl) return i;
                return self.matchWhitespace(start);
            },
        }
    }

    /// o200k: `[^\r\n\p{L}\p{N}]?[Upper]*[Lower]+contraction? | [^\r\n\p{L}\p{N}]?[Upper]+[Lower]*contraction? |
    /// \p{N}{1,3} | ?[^\s\p{L}\p{N}]+[\r\n/]* | \s*[\r\n]+ | \s+(?!\S) | \s+`. Non-ASCII letters count as
    /// both cases (the pattern's Lm/Lo classes), so only ASCII case changes split a word.
    fn matchO200k(self: *const RegexSplitter, start: usize, first: Cp) usize {
        var i = start;
        var c = first;
        if (!isNewline(c.cp) and !isLetter(c.cp) and !isNumber(c.cp)) {
            if (self.cpAt(i + c.len)) |n| {
                if (isLetter(n.cp)) {
                    i += c.len;
                    c = n;
                }
            }
        }
        if (isLetter(c.cp)) {
            // Upper run (ASCII capitals or case-less letters), then lower run (everything but ASCII capitals).
            while (self.cpAt(i)) |n| {
                if (!isLetter(n.cp) or (n.cp < 128 and !isUpperAscii(n.cp))) break;
                i += n.len;
            }
            while (self.cpAt(i)) |n| {
                if (!isLetter(n.cp) or isUpperAscii(n.cp)) break;
                i += n.len;
            }
            return self.contractionAt(i);
        }
        if (isNumber(first.cp)) {
            i = start;
            var count: usize = 0;
            while (self.cpAt(i)) |n| {
                if (!isNumber(n.cp) or count == 3) break;
                i += n.len;
                count += 1;
            }
            return i;
        }
        // ` ?[^\s\p{L}\p{N}]+[\r\n/]*`
        i = start;
        c = first;
        if (c.cp == ' ') {
            if (self.cpAt(i + 1)) |n| {
                if (!isWhitespace(n.cp) and !isLetter(n.cp) and !isNumber(n.cp)) {
                    i += 1;
                    c = n;
                }
            }
        }
        if (!isWhitespace(c.cp) and !isLetter(c.cp) and !isNumber(c.cp)) {
            while (self.cpAt(i)) |n| {
                if (isWhitespace(n.cp) or isLetter(n.cp) or isNumber(n.cp)) break;
                i += n.len;
            }
            while (self.cpAt(i)) |n| {
                if (!isNewline(n.cp) and n.cp != '/') break;
                i += n.len;
            }
            return i;
        }
        return self.matchNewlinesOrWhitespace(start);
    }

    /// DeepSeek V3: `[ascii punct][A-Za-z]+ | [^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+ | ?[\p{P}\p{S}]+[\r\n]* |
    /// \s*[\r\n]+ | \s+(?!\S) | \s+` (symbols approximated as "neither letter, number nor whitespace").
    fn matchDeepseek3(self: *const RegexSplitter, start: usize, first: Cp) usize {
        var i = start;
        var c = first;
        if (first.cp < 128 and std.ascii.isPunctuation(@intCast(first.cp))) {
            if (self.cpAt(i + 1)) |n| {
                if (n.cp < 128 and std.ascii.isAlphabetic(@intCast(n.cp))) {
                    i += 1;
                    while (self.cpAt(i)) |m| {
                        if (!(m.cp < 128 and std.ascii.isAlphabetic(@intCast(m.cp)))) break;
                        i += m.len;
                    }
                    return i;
                }
            }
        }
        if (!isNewline(c.cp) and !isLetter(c.cp) and !isPunctOrSymbol(c.cp)) {
            if (self.cpAt(i + c.len)) |n| {
                if (isLetter(n.cp)) {
                    i += c.len;
                    c = n;
                }
            }
        }
        if (isLetter(c.cp)) {
            while (self.cpAt(i)) |n| {
                if (!isLetter(n.cp)) break;
                i += n.len;
            }
            return i;
        }
        i = start;
        c = first;
        if (c.cp == ' ') {
            if (self.cpAt(i + 1)) |n| {
                if (isPunctOrSymbol(n.cp)) {
                    i += 1;
                    c = n;
                }
            }
        }
        if (isPunctOrSymbol(c.cp)) {
            while (self.cpAt(i)) |n| {
                if (!isPunctOrSymbol(n.cp)) break;
                i += n.len;
            }
            while (self.cpAt(i)) |n| {
                if (!isNewline(n.cp)) break;
                i += n.len;
            }
            return i;
        }
        return self.matchNewlinesOrWhitespace(start);
    }

    /// `\s*[\r\n]+ | \s+(?!\S) | \s+`
    fn matchNewlinesOrWhitespace(self: *const RegexSplitter, start: usize) usize {
        var i = start;
        var saw_nl = false;
        var j = start;
        while (self.cpAt(j)) |n| {
            if (!isWhitespace(n.cp)) break;
            j += n.len;
            if (isNewline(n.cp)) {
                saw_nl = true;
                i = j;
            }
        }
        if (saw_nl) return i;
        return self.matchWhitespace(start);
    }

    fn isPunctOrSymbol(cp: u21) bool {
        return !isWhitespace(cp) and !isLetter(cp) and !isNumber(cp) and !isNewline(cp);
    }

    fn isDs2Punct(cp: u21) bool {
        return (cp >= 0x21 and cp <= 0x2F) or (cp >= 0x3A and cp <= 0x7E) or (cp >= 0xFF01 and cp <= 0xFF0F) or
            (cp >= 0xFF1A and cp <= 0xFF5E) or (cp >= 0x2018 and cp <= 0x201F) or (cp >= 0x3000 and cp <= 0x3002);
    }

    fn isCjk(cp: u21) bool {
        return (cp >= 0x4E00 and cp <= 0x9FA5) or (cp >= 0x0800 and cp < 0x4E00) or (cp >= 0xAC00 and cp <= 0xD7FF);
    }

    /// '\s+(?!\S)|\s+'
    fn matchWhitespace(self: *const RegexSplitter, start: usize) usize {
        var i = start;
        var last_start = start;
        var count: usize = 0;
        while (self.cpAt(i)) |n| {
            if (!isWhitespace(n.cp)) break;
            last_start = i;
            i += n.len;
            count += 1;
        }
        if (count == 0) return start;
        // \s+(?!\S): if followed by non-space, give back the last whitespace char
        // (so it can prefix the next word) unless that would leave nothing.
        if (i < self.text.len and count > 1) return last_start;
        return i;
    }
};

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

test "regex splitter qwen2" {
    var it = RegexSplitter{ .text = "Hello world, it's 2024!\n\nBye", .kind = .qwen2 };
    const expected = [_][]const u8{ "Hello", " world", ",", " it", "'s", " ", "2", "0", "2", "4", "!\n\n", "Bye" };
    for (expected) |e| {
        const got = it.next().?;
        try std.testing.expectEqualStrings(e, got);
    }
    try std.testing.expect(it.next() == null);
}

test "regex splitter llama3 digits" {
    var it = RegexSplitter{ .text = "a 12345", .kind = .llama3 };
    try std.testing.expectEqualStrings("a", it.next().?);
    try std.testing.expectEqualStrings(" ", it.next().?);
    try std.testing.expectEqualStrings("123", it.next().?);
    try std.testing.expectEqualStrings("45", it.next().?);
}

test "pre-tokenizer steps" {
    var it = RegexSplitter{ .text = "HelloWorld it's 2024/a", .kind = .o200k };
    const expected = [_][]const u8{ "Hello", "World", " it's", " ", "202", "4", "/a" };
    for (expected) |e| try std.testing.expectEqualStrings(e, it.next().?);
    try std.testing.expect(it.next() == null);
    var ds = RegexSplitter{ .text = "abc 12 é.x", .kind = .deepseek3 };
    const exp2 = [_][]const u8{ "abc", " ", "12", " é", ".x" };
    for (exp2) |e| try std.testing.expectEqualStrings(e, ds.next().?);
    var nl = RegexSplitter{ .text = "a\nb", .kind = .newlines };
    try std.testing.expectEqualStrings("a", nl.next().?);
    try std.testing.expectEqualStrings("\n", nl.next().?);
    try std.testing.expectEqualStrings("b", nl.next().?);
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var out = std.ArrayList([]const u8).empty;
    try splitClass(arena.allocator(), "ab12,c", isNumber, true, true, &out);
    try std.testing.expectEqual(@as(usize, 4), out.items.len);
    try std.testing.expectEqualStrings("1", out.items[1]);
    out.clearRetainingCapacity();
    try splitClass(arena.allocator(), "a,,b", isPunctuation, false, true, &out);
    try std.testing.expectEqualStrings(",,", out.items[1]);
}

test "byte-level bpe round trip" {
    const gpa = std.testing.allocator;
    const json =
        \\{"model":{"type":"BPE","vocab":{"h":0,"e":1,"l":2,"o":3,"he":4,"ll":5,"hell":6,"hello":7,"Ġ":8,"Ġw":9,"<|end|>":10},
        \\"merges":["h e","l l","he ll","hell o","Ġ w"]},
        \\"pre_tokenizer":{"type":"Sequence","pretokenizers":[{"type":"Split","pattern":{"Regex":"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"},"behavior":"Isolated"},{"type":"ByteLevel","add_prefix_space":false,"use_regex":false}]},
        \\"decoder":{"type":"ByteLevel"},
        \\"added_tokens":[{"id":10,"content":"<|end|>","special":true}]}
    ;
    const tok = try Tokenizer.parse(gpa, json, null);
    defer tok.deinit();
    const ids = try tok.encode(gpa, "hello w<|end|>", true);
    defer gpa.free(ids);
    try std.testing.expectEqualSlices(u32, &.{ 7, 9, 10 }, ids);
    const text = try tok.decode(gpa, ids, false);
    defer gpa.free(text);
    try std.testing.expectEqualStrings("hello w<|end|>", text);
    const text2 = try tok.decode(gpa, ids, true);
    defer gpa.free(text2);
    try std.testing.expectEqualStrings("hello w", text2);
}

test "sentencepiece style bpe with byte fallback" {
    const gpa = std.testing.allocator;
    const json =
        \\{"model":{"type":"BPE","byte_fallback":true,"vocab":{"<s>":0,"▁":1,"h":2,"i":3,"▁h":4,"▁hi":5,"<0xE2>":6,"<0x82>":7,"<0xAC>":8},
        \\"merges":["▁ h","▁h i"]},
        \\"normalizer":{"type":"Sequence","normalizers":[{"type":"Prepend","prepend":"▁"},{"type":"Replace","pattern":{"String":" "},"content":"▁"}]},
        \\"pre_tokenizer":null,
        \\"decoder":{"type":"Sequence","decoders":[{"type":"Replace","pattern":{"String":"▁"},"content":" "},{"type":"ByteFallback"},{"type":"Fuse"},{"type":"Strip","content":" ","start":1,"stop":0}]},
        \\"post_processor":{"type":"TemplateProcessing","single":[{"SpecialToken":{"id":"<s>","type_id":0}},{"Sequence":{"id":"A","type_id":0}}]},
        \\"added_tokens":[{"id":0,"content":"<s>","special":true}]}
    ;
    const tok = try Tokenizer.parse(gpa, json, null);
    defer tok.deinit();
    const ids = try tok.encode(gpa, "hi €", true);
    defer gpa.free(ids);
    try std.testing.expectEqualSlices(u32, &.{ 0, 5, 1, 6, 7, 8 }, ids);
    const text = try tok.decode(gpa, ids, true);
    defer gpa.free(text);
    try std.testing.expectEqualStrings("hi €", text);
}

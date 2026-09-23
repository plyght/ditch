//! A Hugging Face `tokenizer.json` compatible BPE tokenizer.
//!
//! Supports the tokenizer configurations shipped by decoder-only models:
//! byte-level BPE with the GPT-2 / Qwen2 / Llama-3 / o200k / DeepSeek style
//! regex pre-tokenisers (plus the `Digits`, `Punctuation`, `Whitespace` and
//! `ByteLevel(add_prefix_space)` steps some families chain in a `Sequence`),
//! and SentencePiece-derived BPE with Metaspace handling and byte fallback.
//! NFKC and NFC are applied (per code point, then canonical composition;
//! only the reordering of several combining marks is left out); the rest of
//! SentencePiece's `Precompiled` map is approximated by the identity.
//!
//! Models without a `tokenizer.json` that ship a tiktoken rank file instead
//! (Moonshot Kimi's `tiktoken.model`, Meta's Llama 3 `tokenizer.model`) are
//! loaded by `parseTiktoken`: byte-level BPE whose merges are the ranks of
//! the vocabulary itself (the pair whose concatenation has the lowest rank is
//! merged first; a whole pre-token that is in the vocabulary is one token).

const std = @import("std");
const uni = @import("unicode_tables.zig");

const Allocator = std.mem.Allocator;
const Io = std.Io;

/// Pre-tokeniser of Moonshot's `tokenization_kimi.py` (Kimi K2 / K2.5 / K3 / Kimi-Linear).
pub const pattern_kimi = "[\\p{Han}]+|[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}&&[^\\p{Han}]]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}&&[^\\p{Han}]]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}&&[^\\p{Han}]]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}&&[^\\p{Han}]]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";
/// Pre-tokeniser of Meta's Llama 3 `tokenizer.py`.
pub const pattern_llama3 = "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";

/// The conventions a tiktoken rank file is read with: the pre-tokeniser and
/// the names of the special-token block that follows the base vocabulary.
pub const TiktokenKind = enum {
    /// Moonshot `tiktoken.model`: 256 special ids named by `added_tokens_decoder`, else `<|reserved_token_N|>`.
    kimi,
    /// Meta `tokenizer.model`: 256 special ids named by `added_tokens_decoder`, else the Llama 3 reference list.
    llama3,
};

/// The Llama 3 reference tokenizer's special tokens, in id order after the base vocabulary.
fn llama3SpecialName(arena: Allocator, index: usize) ![]const u8 {
    const named = [_][]const u8{ "<|begin_of_text|>", "<|end_of_text|>", "<|reserved_special_token_0|>", "<|reserved_special_token_1|>", "<|reserved_special_token_2|>", "<|reserved_special_token_3|>", "<|start_header_id|>", "<|end_header_id|>", "<|reserved_special_token_4|>", "<|eot_id|>" };
    if (index < named.len) return named[index];
    return std.fmt.allocPrint(arena, "<|reserved_special_token_{d}|>", .{index - named.len + 5});
}

pub const AddedToken = struct {
    id: u32,
    content: []const u8,
    special: bool,
    /// The token absorbs the whitespace before (`lstrip`) or after
    /// (`rstrip`) it, as `tokenizers`' `AddedToken` does (Phi-3's
    /// `<|user|>`, `<|end|>`: `rstrip`).
    lstrip: bool = false,
    rstrip: bool = false,
};

/// The `Split` regular expressions the splitter implements natively.
const RegexKind = enum {
    gpt2,
    qwen2,
    /// Seed-OSS: Qwen 2's pattern whose punctuation run takes no line breaks
    /// after it (` ?[^\s\p{L}\p{N}\r\n]+` for ` ?[^\s\p{L}\p{N}]+[\r\n]*`).
    qwen2_bare_punct,
    /// Qwen 3.5: Qwen 2's pattern with marks in words and out of punctuation
    /// (`[\p{L}\p{M}]+`, `[^\s\p{L}\p{M}\p{N}]`).
    qwen3_5,
    /// K-EXAONE's phrase pattern: a word runs over letters (each with its
    /// marks) *and single spaces between them*
    /// (`(?:\p{L}\p{M}*(?: \p{L}\p{M}*)*)+`), digits are single, and
    /// punctuation takes at most one line break or `/` after it.
    phrase,
    llama3,
    /// nanochat's GPT-4-style split: Llama 3's with digit runs of at most two
    /// (`\p{N}{1,2}`). Its possessive quantifiers and single-`[\r\n]` line
    /// break alternative match exactly what Llama 3's pattern does.
    nanochat,
    /// tiktoken o200k (gpt-oss): words split at lower→upper case changes, contractions as suffixes.
    o200k,
    /// o200k with single digits (`\p{N}` for `\p{N}{1,3}`; Falcon-H1).
    o200k_digit1,
    /// Kimi (`tokenization_kimi.py`): Han runs first, then o200k-style words excluding Han characters.
    kimi,
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
    /// CJK runs `[一-龥ࠀ-一가-퟿]+` (DeepSeek V2)
    cjk,
    /// CJK and kana runs `[一-龥぀-ゟ゠-ヿ]+` (DeepSeek V3)
    cjk_kana,
    /// AFMoE's digit handling, three `Split`s in a row: decimal-digit runs cut
    /// into right-aligned chunks of 510 (`\p{Nd}{1,510}(?=(?>\p{Nd}{510})*(?:\P{Nd}|$))|\G\p{Nd}{510}`),
    /// the leading 1–2 digits of an all-digit piece whose length is not a
    /// multiple of three (`\A\p{Nd}{1,2}(?=\p{Nd}{3}+\z)`), then groups of
    /// three from the start (`\A\p{Nd}{3}|\G\p{Nd}{3}`): `1234567` -> `1 234 567`.
    nd_chunks510,
    nd_lead,
    nd_groups3,
    /// AFMoE's script runs: CJK / kana, Thai, Lao, Khmer, Myanmar and Hangul,
    /// each a run of its own class.
    script_runs,
};

/// One pre-tokenisation step of a `Sequence`.
const Step = union(enum) {
    regex: RegexKind,
    split_string: struct { pattern: []const u8, removed: bool },
    /// `Split(Regex("(?:\\r?\\n)+(?!\\r?\\n)"), MergedWithNext)` (Laguna): each
    /// maximal run of line breaks starts a new piece, followed by the text up
    /// to the next run.
    newline_runs,
    /// `Digits(individual_digits)`
    digits: bool,
    /// `Punctuation` (true: contiguous runs stay together)
    punctuation: bool,
    whitespace,
    whitespace_split,
    /// `ByteLevel(add_prefix_space)`; the regex (when `use_regex`) is a separate step.
    byte_level: bool,
    /// `prepend`: `always`, `first` (only at the start of the input — not
    /// after a special token) or `never`.
    metaspace: struct { prepend: MetaspacePrepend, split: bool },
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
    /// tiktoken semantics: a pair merges when its concatenation is in the
    /// vocabulary, lowest rank first (`merges` then only holds the merges
    /// reconstructed for exports).
    rank_bpe: bool,
    /// Set by `parseTiktoken`; selects the pattern written by `toJson`.
    tiktoken_kind: ?TiktokenKind,
    unk_id: ?u32,
    /// Unigram (SentencePiece) model: one log-probability per id, and the
    /// longest vocabulary entry in bytes. Empty for a BPE tokenizer.
    unigram_scores: []const f32,
    unigram_max_len: usize,
    // normalizer
    prepend: ?[]const u8,
    replace_space: ?[]const u8, // replacement for " " (typically "▁")
    lowercase: bool,
    /// `NFKC` / `NFC` normalizer (see `normalizeUnicode`); null: neither.
    unicode_form: ?UnicodeForm,
    /// The letter class of DeepSeek V2's `\s?[A-Za-z…]+` split, read from the
    /// pattern itself: a hand-picked subset of the cased letters that no
    /// Unicode property reproduces.
    ds2_letter_class: []const uni.Range = &.{},
    /// While encoding: whether the segment being encoded starts the input (a
    /// `Metaspace` with `prepend_scheme: first` prepends only there).
    segment_at_start: bool = true,
    /// Pre-tokenisation steps applied in order to every segment.
    steps: []const Step,
    /// SentencePiece's `Precompiled` charsmap: only its whitespace rules are
    /// applied (control characters and the Unicode separators become a space,
    /// runs of spaces collapse, the ends are stripped), which is what it does
    /// to ordinary text; the rest of the charsmap is the identity.
    sp_whitespace: bool,
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

    /// An empty tokenizer with the GPT-2 byte <-> unicode mapping set up.
    fn create(gpa: Allocator) !*Tokenizer {
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
            .rank_bpe = false,
            .tiktoken_kind = null,
            .unk_id = null,
            .unigram_scores = &.{},
            .unigram_max_len = 0,
            .prepend = null,
            .replace_space = null,
            .lowercase = false,
            .unicode_form = null,
            .steps = &.{},
            .sp_whitespace = false,
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
        return self;
    }

    /// Reads bos/eos token names and `add_bos_token` from `tokenizer_config.json`.
    fn applyConfig(self: *Tokenizer, tokenizer_config_json: []const u8) !void {
        var cfg = try std.json.parseFromSlice(std.json.Value, self.gpa, tokenizer_config_json, .{});
        defer cfg.deinit();
        if (cfg.value != .object) return;
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

    /// Parses `tokenizer.json`. `tokenizer_config_json` (optional) is used for bos/eos token names.
    pub fn parse(gpa: Allocator, json_text: []const u8, tokenizer_config_json: ?[]const u8) !*Tokenizer {
        const self = try create(gpa);
        errdefer self.deinit();
        const arena = self.arena.allocator();

        var parsed = try std.json.parseFromSlice(std.json.Value, gpa, json_text, .{});
        defer parsed.deinit();
        const root = parsed.value.object;

        // --- model ---
        const model = (root.get("model") orelse return error.InvalidTokenizer).object;
        const model_type = if (model.get("type")) |t| t.string else "BPE";
        const unigram = std.mem.eql(u8, model_type, "Unigram");
        if (!std.mem.eql(u8, model_type, "BPE") and !unigram) {
            std.log.err("unsupported tokenizer model type: {s}", .{model_type});
            return error.UnsupportedTokenizer;
        }
        if (model.get("byte_fallback")) |v| self.byte_fallback = v == .bool and v.bool;
        if (model.get("ignore_merges")) |v| self.ignore_merges = v == .bool and v.bool;

        const vocab_value = model.get("vocab") orelse return error.InvalidTokenizer;
        var max_id: usize = 0;
        if (unigram) {
            // `vocab` is `[[token, log_prob], ...]`; the id is the index.
            const list = if (vocab_value == .array) vocab_value.array.items else return error.InvalidTokenizer;
            const scores = try arena.alloc(f32, list.len);
            for (list, 0..) |entry, i| {
                if (entry != .array or entry.array.items.len < 2) return error.InvalidTokenizer;
                const token = entry.array.items[0];
                if (token != .string) return error.InvalidTokenizer;
                const score = entry.array.items[1];
                scores[i] = switch (score) {
                    .float => |f| @floatCast(f),
                    .integer => |n| @floatFromInt(n),
                    else => return error.InvalidTokenizer,
                };
                const id: u32 = @intCast(i);
                const key = try arena.dupe(u8, token.string);
                // The first entry of a duplicated token wins, as in sentencepiece.
                if (!self.vocab.contains(key)) try self.vocab.put(arena, key, id);
                self.unigram_max_len = @max(self.unigram_max_len, key.len);
                max_id = @max(max_id, i);
            }
            self.unigram_scores = scores;
            if (model.get("unk_id")) |u| {
                if (u == .integer and u.integer >= 0) self.unk_id = @intCast(u.integer);
            }
        } else {
            const vocab = if (vocab_value == .object) vocab_value.object else return error.InvalidTokenizer;
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
                const lstrip = if (obj.get("lstrip")) |v| (v == .bool and v.bool) else false;
                const rstrip = if (obj.get("rstrip")) |v| (v == .bool and v.bool) else false;
                try list.append(arena, .{ .id = id, .content = content, .special = special, .lstrip = lstrip, .rstrip = rstrip });
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
        if (tokenizer_config_json) |cfg_text| try self.applyConfig(cfg_text);
        return self;
    }

    /// Whether `bytes` is a tiktoken rank file (`<base64 token> <rank>` lines)
    /// rather than, say, a SentencePiece protobuf.
    pub fn looksLikeTiktoken(bytes: []const u8) bool {
        var it = std.mem.splitScalar(u8, bytes, '\n');
        while (it.next()) |raw| {
            const line = std.mem.trimEnd(u8, raw, "\r ");
            if (line.len == 0) continue;
            const sp = std.mem.indexOfScalar(u8, line, ' ') orelse return false;
            for (line[0..sp]) |c| if (!(std.ascii.isAlphanumeric(c) or c == '+' or c == '/' or c == '=')) return false;
            _ = std.fmt.parseInt(u32, line[sp + 1 ..], 10) catch return false;
            return true;
        }
        return false;
    }

    /// Parses a tiktoken rank file (`tiktoken.model` / `tokenizer.model`: one
    /// `<base64 token> <rank>` line per token). The 256 ids after the base
    /// vocabulary are special tokens, named by `added_tokens_decoder` of
    /// `tokenizer_config_json` and otherwise by the conventions of `kind`.
    pub fn parseTiktoken(gpa: Allocator, model_text: []const u8, tokenizer_config_json: ?[]const u8, kind: TiktokenKind) !*Tokenizer {
        const self = try create(gpa);
        errdefer self.deinit();
        const arena = self.arena.allocator();
        self.rank_bpe = true;
        self.tiktoken_kind = kind;
        self.byte_level = true;
        self.decoder = .byte_level;
        const steps = try arena.alloc(Step, 2);
        steps[0] = .{ .regex = switch (kind) {
            .kimi => .kimi,
            .llama3 => .llama3,
        } };
        steps[1] = .{ .byte_level = false };
        self.steps = steps;

        // --- ranks ---
        var max_rank: u32 = 0;
        var n_base: usize = 0;
        var line_no: usize = 0;
        var it = std.mem.splitScalar(u8, model_text, '\n');
        while (it.next()) |raw| {
            line_no += 1;
            const line = std.mem.trimEnd(u8, raw, "\r ");
            if (line.len == 0) continue;
            const sp = std.mem.indexOfScalar(u8, line, ' ') orelse {
                std.log.err("tiktoken vocabulary line {d} is not '<base64> <rank>'", .{line_no});
                return error.InvalidTokenizer;
            };
            const rank = std.fmt.parseInt(u32, line[sp + 1 ..], 10) catch {
                std.log.err("tiktoken vocabulary line {d} has an invalid rank", .{line_no});
                return error.InvalidTokenizer;
            };
            const decoder = std.base64.standard.Decoder;
            const n = decoder.calcSizeForSlice(line[0..sp]) catch return error.InvalidTokenizer;
            const bytes = try arena.alloc(u8, n);
            decoder.decode(bytes, line[0..sp]) catch {
                std.log.err("tiktoken vocabulary line {d} is not base64", .{line_no});
                return error.InvalidTokenizer;
            };
            const mapped = try self.byteLevelEncode(arena, bytes);
            try self.vocab.put(arena, mapped, rank);
            max_rank = @max(max_rank, rank);
            n_base += 1;
        }
        if (n_base == 0) return error.InvalidTokenizer;
        const first_special: u32 = max_rank + 1;
        const n_special: u32 = 256;

        // --- special tokens ---
        var list = std.ArrayList(AddedToken).empty;
        var max_id: u32 = first_special + n_special - 1;
        if (tokenizer_config_json) |cfg_text| {
            var cfg = try std.json.parseFromSlice(std.json.Value, gpa, cfg_text, .{});
            defer cfg.deinit();
            if (cfg.value == .object) {
                if (cfg.value.object.get("added_tokens_decoder")) |atd| if (atd == .object) {
                    var e = atd.object.iterator();
                    while (e.next()) |entry| {
                        const id = std.fmt.parseInt(u32, entry.key_ptr.*, 10) catch continue;
                        if (entry.value_ptr.* != .object) continue;
                        const content = tokenName(entry.value_ptr.*) orelse continue;
                        if (id < first_special) {
                            std.log.warn("added token {d} ('{s}') overlaps the tiktoken vocabulary; ignored", .{ id, content });
                            continue;
                        }
                        const special = if (entry.value_ptr.object.get("special")) |s| s == .bool and s.bool else true;
                        try list.append(arena, .{ .id = id, .content = try arena.dupe(u8, content), .special = special });
                        max_id = @max(max_id, id);
                    }
                };
            }
        }
        var i: u32 = 0;
        while (i < n_special) : (i += 1) {
            const id = first_special + i;
            var present = false;
            for (list.items) |t| present = present or t.id == id;
            if (present) continue;
            const content = switch (kind) {
                .kimi => try std.fmt.allocPrint(arena, "<|reserved_token_{d}|>", .{id}),
                .llama3 => try llama3SpecialName(arena, i),
            };
            try list.append(arena, .{ .id = id, .content = content, .special = true });
        }
        self.id_to_token = try arena.alloc([]const u8, max_id + 1);
        @memset(self.id_to_token, "");
        {
            var v = self.vocab.iterator();
            while (v.next()) |e| self.id_to_token[e.value_ptr.*] = e.key_ptr.*;
        }
        for (list.items) |t| {
            self.id_to_token[t.id] = t.content;
            if (!self.vocab.contains(t.content)) try self.vocab.put(arena, t.content, t.id);
            if (t.special) try self.special_ids.put(arena, t.id, {});
        }
        std.mem.sort(AddedToken, list.items, {}, struct {
            fn lt(_: void, a: AddedToken, b: AddedToken) bool {
                return a.content.len > b.content.len;
            }
        }.lt);
        self.added = list.items;
        for (self.added, 0..) |a, idx| try self.added_by_id.put(arena, a.id, idx);

        // --- bos / eos ---
        switch (kind) {
            .llama3 => {
                self.bos_id = self.vocab.get("<|begin_of_text|>");
                self.eos_id = self.vocab.get("<|end_of_text|>");
                self.add_bos = self.bos_id != null;
            },
            .kimi => {},
        }
        if (tokenizer_config_json) |cfg_text| try self.applyConfig(cfg_text);

        try self.reconstructMerges();
        return self;
    }

    /// Rebuilds the merge list from the ranks the way Hugging Face's tiktoken
    /// converter does (for each token, the merge that produces it when only
    /// lower ranks apply), so exports carry a `merges` list.
    fn reconstructMerges(self: *Tokenizer) !void {
        const arena = self.arena.allocator();
        const Pair = struct { rank: u32, key: []const u8 };
        var pairs = std.ArrayList(Pair).empty;
        var syms = std.ArrayList(Symbol).empty;
        defer syms.deinit(self.gpa);
        var v = self.vocab.iterator();
        while (v.next()) |e| {
            const word = e.key_ptr.*;
            const rank = e.value_ptr.*;
            if (self.added_by_id.contains(rank)) continue;
            syms.clearRetainingCapacity();
            var i: usize = 0;
            while (i < word.len) {
                const n = std.unicode.utf8ByteSequenceLength(word[i]) catch 1;
                const end = @min(word.len, i + n);
                try syms.append(self.gpa, .{ .start = i, .end = end });
                i = end;
            }
            if (syms.items.len < 2) continue;
            self.rankMerge(word, &syms, rank);
            if (syms.items.len != 2) continue;
            const key = try std.fmt.allocPrint(arena, "{s} {s}", .{ word[syms.items[0].start..syms.items[0].end], word[syms.items[1].start..syms.items[1].end] });
            try pairs.append(arena, .{ .rank = rank, .key = key });
        }
        std.mem.sort(Pair, pairs.items, {}, struct {
            fn lt(_: void, a: Pair, b: Pair) bool {
                return a.rank < b.rank;
            }
        }.lt);
        for (pairs.items, 0..) |p, dense| try self.merges.put(arena, p.key, @intCast(dense));
    }

    /// A `tokenizer.json` equivalent of a tiktoken vocabulary (byte-level
    /// vocabulary, reconstructed merges with `ignore_merges`, the pattern of
    /// the kind), for exports.
    pub fn toJson(self: *const Tokenizer, gpa: Allocator) ![]u8 {
        const kind = self.tiktoken_kind orelse return error.NotTiktoken;
        var out: Io.Writer.Allocating = .init(gpa);
        errdefer out.deinit();
        const w = &out.writer;
        try w.writeAll("{\"version\":\"1.0\",\"truncation\":null,\"padding\":null,\"added_tokens\":[");
        // In id order for a readable file.
        const ids = try gpa.alloc(u32, self.added.len);
        defer gpa.free(ids);
        for (self.added, 0..) |a, i| ids[i] = a.id;
        std.mem.sort(u32, ids, {}, std.sort.asc(u32));
        for (ids, 0..) |id, i| {
            const a = self.added[self.added_by_id.get(id).?];
            if (i > 0) try w.writeAll(",");
            try w.print("{{\"id\":{d},\"content\":\"", .{a.id});
            try std.json.Stringify.encodeJsonStringChars(a.content, .{}, w);
            try w.print("\",\"single_word\":false,\"lstrip\":false,\"rstrip\":false,\"normalized\":false,\"special\":{s}}}", .{if (a.special) "true" else "false"});
        }
        try w.writeAll("],\"normalizer\":null,\"pre_tokenizer\":{\"type\":\"Sequence\",\"pretokenizers\":[{\"type\":\"Split\",\"pattern\":{\"Regex\":\"");
        try std.json.Stringify.encodeJsonStringChars(switch (kind) {
            .kimi => pattern_kimi,
            .llama3 => pattern_llama3,
        }, .{}, w);
        try w.writeAll("\"},\"behavior\":\"Isolated\",\"invert\":false},{\"type\":\"ByteLevel\",\"add_prefix_space\":false,\"trim_offsets\":true,\"use_regex\":false}]},\"post_processor\":");
        if (self.add_bos and self.bos_id != null) {
            const bos = self.id_to_token[self.bos_id.?];
            try w.writeAll("{\"type\":\"TemplateProcessing\",\"single\":[{\"SpecialToken\":{\"id\":\"");
            try std.json.Stringify.encodeJsonStringChars(bos, .{}, w);
            try w.writeAll("\",\"type_id\":0}},{\"Sequence\":{\"id\":\"A\",\"type_id\":0}}],\"pair\":[],\"special_tokens\":{\"");
            try std.json.Stringify.encodeJsonStringChars(bos, .{}, w);
            try w.print("\":{{\"id\":\"", .{});
            try std.json.Stringify.encodeJsonStringChars(bos, .{}, w);
            try w.print("\",\"ids\":[{d}],\"tokens\":[\"", .{self.bos_id.?});
            try std.json.Stringify.encodeJsonStringChars(bos, .{}, w);
            try w.writeAll("\"]}}}");
        } else {
            try w.writeAll("null");
        }
        try w.writeAll(",\"decoder\":{\"type\":\"ByteLevel\",\"add_prefix_space\":true,\"trim_offsets\":true,\"use_regex\":true},\"model\":{\"type\":\"BPE\",\"dropout\":null,\"unk_token\":null,\"continuing_subword_prefix\":null,\"end_of_word_suffix\":null,\"fuse_unk\":false,\"byte_fallback\":false,\"ignore_merges\":true,\"vocab\":{");
        var first = true;
        for (self.id_to_token, 0..) |tok, id| {
            if (tok.len == 0 or self.added_by_id.contains(@intCast(id))) continue;
            if (!first) try w.writeAll(",");
            first = false;
            try w.writeAll("\"");
            try std.json.Stringify.encodeJsonStringChars(tok, .{}, w);
            try w.print("\":{d}", .{id});
        }
        try w.writeAll("},\"merges\":[");
        const merges = try gpa.alloc([]const u8, self.merges.count());
        defer gpa.free(merges);
        @memset(merges, "");
        var m = self.merges.iterator();
        while (m.next()) |e| merges[e.value_ptr.*] = e.key_ptr.*;
        for (merges, 0..) |key, i| {
            if (i > 0) try w.writeAll(",");
            try w.writeAll("\"");
            try std.json.Stringify.encodeJsonStringChars(key, .{}, w);
            try w.writeAll("\"");
        }
        try w.writeAll("]}}");
        return out.toOwnedSlice();
    }

    pub const Loaded = struct {
        tokenizer: *Tokenizer,
        /// `tokenizer.json` as read, or synthesised from the tiktoken vocabulary (for exports).
        json: []const u8,
    };

    /// Loads the tokenizer of a model directory: `tokenizer.json` when present,
    /// else a tiktoken rank file (`tiktoken.model`, Moonshot's conventions, or
    /// `tokenizer.model`, Meta's). A SentencePiece `tokenizer.model` alone is
    /// an error naming the conversion.
    pub fn loadDir(gpa: Allocator, io: Io, arena: Allocator, dir: Io.Dir, dir_path: []const u8, tokenizer_config_json: ?[]const u8) !Loaded {
        if (dir.readFileAlloc(io, "tokenizer.json", arena, .unlimited)) |json| {
            return .{ .tokenizer = try parse(gpa, json, tokenizer_config_json), .json = json };
        } else |_| {}
        const candidates = [_]struct { name: []const u8, kind: TiktokenKind }{
            .{ .name = "tiktoken.model", .kind = .kimi },
            .{ .name = "tokenizer.model", .kind = .llama3 },
        };
        for (candidates) |c| {
            const text = dir.readFileAlloc(io, c.name, arena, .unlimited) catch continue;
            if (!looksLikeTiktoken(text)) {
                std.log.err("{s}/{s} is not a tiktoken vocabulary (a SentencePiece model needs a tokenizer.json: save one with AutoTokenizer.from_pretrained(...).save_pretrained(...))", .{ dir_path, c.name });
                return error.UnsupportedTokenizer;
            }
            var kind = c.kind;
            if (tokenizer_config_json) |cfg| {
                if (std.mem.indexOf(u8, cfg, "tokenization_kimi") != null or std.mem.indexOf(u8, cfg, "<|im_middle|>") != null) kind = .kimi;
            }
            const tok = try parseTiktoken(gpa, text, tokenizer_config_json, kind);
            errdefer tok.deinit();
            std.log.info("tokenizer: tiktoken vocabulary {s} ({s} conventions, {d} tokens)", .{ c.name, @tagName(kind), tok.vocabSize() });
            const json = try tok.toJson(gpa);
            defer gpa.free(json);
            return .{ .tokenizer = tok, .json = try arena.dupe(u8, json) };
        }
        std.log.err("no tokenizer.json, tiktoken.model or tokenizer.model in {s}", .{dir_path});
        return error.MissingTokenizer;
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
        } else if (std.mem.eql(u8, t, "NFKC")) {
            self.unicode_form = .nfkc;
        } else if (std.mem.eql(u8, t, "NFC")) {
            self.unicode_form = .nfc;
        } else if (std.mem.eql(u8, t, "Strip")) {
            // Whitespace stripping of the whole input is the identity for the prompts ditch builds.
        } else if (std.mem.eql(u8, t, "Precompiled")) {
            // The charsmap is a compiled trie; ditch applies the part of it
            // that ordinary text actually hits (see `sp_whitespace`) and
            // leaves the rest — the compatibility foldings — as the identity.
            self.sp_whitespace = true;
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
                if (std.mem.eql(u8, r.string, "(?:\\r?\\n)+(?!\\r?\\n)")) {
                    if (!std.ascii.eqlIgnoreCase(behavior, "mergedwithnext")) std.log.warn("pre-tokenizer: line-break split with behaviour {s} is treated as MergedWithNext", .{behavior});
                    try steps.append(arena, .newline_runs);
                    return;
                }
                const kind = classifyRegex(r.string);
                if (kind == .ds2_letters) self.ds2_letter_class = try parseCharClass(arena, r.string);
                try steps.append(arena, .{ .regex = kind });
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
            const prepend: MetaspacePrepend = if (std.mem.eql(u8, scheme, "never")) .never else if (std.mem.eql(u8, scheme, "first")) .first else .always;
            try steps.append(arena, .{ .metaspace = .{ .prepend = prepend, .split = split } });
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

    /// The ranges of the first bracketed class of `pattern` (`[A-Za-zµÀ-Ö…]`):
    /// single code points and `a-b` ranges, `\x` escaping `x`; sorted and merged.
    fn parseCharClass(arena: Allocator, pattern: []const u8) ![]const uni.Range {
        const open = std.mem.indexOfScalar(u8, pattern, '[') orelse return error.InvalidTokenizer;
        var cps = std.ArrayList(u21).empty;
        var it = (try std.unicode.Utf8View.init(pattern[open + 1 ..])).iterator();
        var escaped = false;
        while (it.nextCodepoint()) |cp| {
            if (!escaped and cp == '\\') {
                escaped = true;
                continue;
            }
            if (!escaped and cp == ']') break;
            escaped = false;
            try cps.append(arena, cp);
        }
        var ranges = std.ArrayList(uni.Range).empty;
        var i: usize = 0;
        while (i < cps.items.len) {
            if (i + 2 < cps.items.len and cps.items[i + 1] == '-') {
                try ranges.append(arena, .{ .lo = cps.items[i], .hi = cps.items[i + 2] });
                i += 3;
            } else {
                try ranges.append(arena, .{ .lo = cps.items[i], .hi = cps.items[i] });
                i += 1;
            }
        }
        std.mem.sort(uni.Range, ranges.items, {}, struct {
            fn lt(_: void, x: uni.Range, y: uni.Range) bool {
                return x.lo < y.lo;
            }
        }.lt);
        var merged = std.ArrayList(uni.Range).empty;
        for (ranges.items) |r| {
            if (merged.items.len > 0 and r.lo <= merged.items[merged.items.len - 1].hi + 1) {
                const last = &merged.items[merged.items.len - 1];
                last.hi = @max(last.hi, r.hi);
            } else try merged.append(arena, r);
        }
        return merged.items;
    }

    fn classifyRegex(raw: []const u8) RegexKind {
        // Some tokenizer.json files (StableLM 2) carry the line breaks of a
        // character class as literal CR/LF rather than as `\r` / `\n`
        // escapes; the patterns below are written with the escapes.
        var buf: [2048]u8 = undefined;
        var n: usize = 0;
        const r = for (raw) |ch| {
            if (n + 2 > buf.len) break raw;
            if (ch == '\r' or ch == '\n') {
                buf[n] = '\\';
                buf[n + 1] = if (ch == '\r') 'r' else 'n';
                n += 2;
            } else {
                buf[n] = ch;
                n += 1;
            }
        } else buf[0..n];
        const has = struct {
            fn f(hay: []const u8, needle: []const u8) bool {
                return std.mem.indexOf(u8, hay, needle) != null;
            }
        }.f;
        if (std.mem.eql(u8, r, "\\p{N}{1,3}")) return .digits3;
        if (std.mem.eql(u8, r, "\\p{Nd}{1,510}(?=(?>\\p{Nd}{510})*(?:\\P{Nd}|$))|\\G\\p{Nd}{510}")) return .nd_chunks510;
        if (std.mem.eql(u8, r, "\\A\\p{Nd}{1,2}(?=\\p{Nd}{3}+\\z)")) return .nd_lead;
        if (std.mem.eql(u8, r, "\\A\\p{Nd}{3}|\\G\\p{Nd}{3}")) return .nd_groups3;
        if (std.mem.eql(u8, r, "(?:[\\u4E00-\\u9FFF\\u3040-\\u309F\\u30A0-\\u30FF\\u3400-\\u4DBF\\uf900-\\uFAFF\\uFF65-\\uFF9F\\u2F00-\\u2FDF]+|[\\u0E00-\\u0E7F]+|[\\u0E80-\\u0EFF]+|[\\u1780-\\u17FF]+|[\\u1000-\\u109F\\uAA60-\\uAA7F\\uA9E0-\\uA9FF]+|[\\uAC00-\\uD7AF\\u1100-\\u11FF]+)")) return .script_runs;
        if (std.mem.eql(u8, r, "[\\r\\n]")) return .newlines;
        if (std.mem.eql(u8, r, "\\s+$")) return .trailing_ws;
        if (std.mem.startsWith(u8, r, "\\s?[A-Za-z")) return .ds2_letters;
        if (std.mem.startsWith(u8, r, "\\s?[!-/")) return .ds2_punct;
        if (std.mem.eql(u8, r, "[\u{4e00}-\u{9fa5}\u{3040}-\u{309f}\u{30a0}-\u{30ff}]+")) return .cjk_kana;
        if (std.mem.startsWith(u8, r, "[\xe4\xb8\x80-")) return .cjk;
        if (has(r, "\\p{Han}")) return .kimi;
        if (has(r, "\\p{Lu}")) return if (has(r, "|\\p{N}|")) .o200k_digit1 else .o200k;
        if (has(r, "\\p{P}\\p{S}")) return .deepseek3;
        if (has(r, "\\p{N}{1,2}") and has(r, "\\p{L}+")) return .nanochat;
        if (has(r, "{1,3}")) return .llama3;
        if (has(r, "(?:\\p{L}\\p{M}*(?: \\p{L}\\p{M}*)*)+")) return .phrase;
        if (has(r, "[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+") and has(r, "[^\\s\\p{L}\\p{M}\\p{N}]")) return .qwen3_5;
        if (has(r, "[^\\r\\n\\p{L}\\p{N}]?\\p{L}+") and has(r, " ?[^\\s\\p{L}\\p{N}\\r\\n]+|")) return .qwen2_bare_punct;
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
                .qwen2, .qwen2_bare_punct, .qwen3_5, .phrase => "qwen2",
                .llama3 => "llama-bpe",
                // No llama.cpp pre-tokenizer splits digits in pairs; the nearest
                // one (nanochat has no llama.cpp architecture to export to anyway).
                .nanochat => "llama-bpe",
                .o200k, .o200k_digit1 => "gpt-4o",
                .kimi => "kimi-k2",
                .deepseek3, .digits3, .nd_chunks510, .nd_lead, .nd_groups3, .script_runs => "deepseek-v3",
                .ds2_letters, .newlines, .ds2_punct, .trailing_ws, .cjk => "deepseek-llm",
                .cjk_kana => "deepseek-v3",
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
                var seg_end = pos;
                if (a.lstrip) {
                    while (seg_end > seg_start) {
                        var k = seg_end - 1;
                        while (k > seg_start and (text[k] & 0xC0) == 0x80) k -= 1;
                        if (!isWhitespace(cpAtSlice(text, k).cp)) break;
                        seg_end = k;
                    }
                }
                self.segment_at_start = seg_start == 0;
                if (seg_end > seg_start) try self.encodeSegment(gpa, text[seg_start..seg_end], &out);
                try out.append(gpa, a.id);
                pos += a.content.len;
                if (a.rstrip) {
                    while (pos < text.len) {
                        const c = cpAtSlice(text, pos);
                        if (!isWhitespace(c.cp)) break;
                        pos += c.len;
                    }
                }
                seg_start = pos;
            } else {
                pos += 1;
            }
        }
        self.segment_at_start = seg_start == 0;
        if (seg_start < text.len) try self.encodeSegment(gpa, text[seg_start..], &out);
        self.segment_at_start = true;
        return out.toOwnedSlice(gpa);
    }

    fn encodeSegment(self: *Tokenizer, gpa: Allocator, raw: []const u8, out: *std.ArrayList(u32)) !void {
        var scratch = std.heap.ArenaAllocator.init(gpa);
        defer scratch.deinit();
        const a = scratch.allocator();
        // Normalise. The prepended text goes through the replacement too: a
        // `Prepend " "` in front of a `Replace " " -> "▁"` (Helium) means the
        // prefix is a metaspace, not a literal space.
        const folded = if (self.unicode_form) |f| try normalizeUnicode(a, raw, f) else raw;
        const text = if (self.sp_whitespace) try spWhitespace(a, folded) else folded;
        var norm = std.ArrayList(u8).empty;
        for ([_][]const u8{ self.prepend orelse "", text }) |part| {
            for (part) |c| {
                if (c == ' ' and self.replace_space != null and !self.has_metaspace) {
                    try norm.appendSlice(a, self.replace_space.?);
                } else if (self.lowercase) {
                    try norm.append(a, std.ascii.toLower(c));
                } else {
                    try norm.append(a, c);
                }
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
                var it = RegexSplitter{ .text = piece, .kind = kind, .class = self.ds2_letter_class };
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
            .newline_runs => {
                var start: usize = 0;
                var i: usize = 0;
                while (i < piece.len) {
                    const run = newlineRunLen(piece[i..]);
                    if (run == 0) {
                        i += 1;
                        continue;
                    }
                    if (i > start) try out.append(a, piece[start..i]);
                    start = i;
                    i += run;
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
                const prepend = switch (ms.prepend) {
                    .always => true,
                    .first => self.segment_at_start,
                    .never => false,
                };
                if (prepend and (piece.len == 0 or !std.mem.startsWith(u8, piece, rep)) and (piece.len == 0 or piece[0] != ' ')) {
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

    /// NFKC (EXAONE) or NFC (Qwen, MiMo, Seed-OSS, MiniMax-M2, dots1): every
    /// code point is replaced by its own NFKC / NFC — for NFKC no-break and
    /// thin spaces, full-width letters, ligatures, superscripts; for NFC only
    /// singletons such as U+212B and composition exclusions — and the result
    /// is canonically composed, so a letter followed by a combining accent,
    /// or half-width katakana followed by a voiced mark, becomes the
    /// precomposed character. The one step of full NFKC / NFC left out is
    /// canonical reordering of several combining marks. Bytes that are not
    /// UTF-8 are copied and block composition.
    fn normalizeUnicode(a: Allocator, s: []const u8, form: UnicodeForm) ![]const u8 {
        const table: []const uni.FoldMulti = switch (form) {
            .nfkc => &uni.nfkc_map,
            .nfc => &uni.nfc_map,
        };
        // Code points, or `raw_byte | b` for a byte that does not decode.
        const raw_byte: u32 = 1 << 31;
        var cps = std.ArrayList(u32).empty;
        var i: usize = 0;
        while (i < s.len) {
            const len = std.unicode.utf8ByteSequenceLength(s[i]) catch 0;
            const cp = if (len > 0 and i + len <= s.len) (std.unicode.utf8Decode(s[i .. i + len]) catch null) else null;
            if (cp) |c| {
                if (mapOf(table, c)) |m| {
                    var it = std.unicode.Utf8View.initUnchecked(m).iterator();
                    while (it.nextCodepoint()) |mc| try cps.append(a, mc);
                } else try cps.append(a, c);
                i += len;
            } else {
                try cps.append(a, raw_byte | s[i]);
                i += 1;
            }
        }
        // Canonical composition: a code point composes with the last starter
        // unless a code point in between has class 0 or a class >= its own.
        var out = std.ArrayList(u32).empty;
        var starter: ?usize = null;
        var last_cc: i32 = -1; // class of the last code point after the starter; -1: none
        for (cps.items) |c| {
            const cc: i32 = if (c & raw_byte != 0) 0 else combiningClass(@intCast(c));
            if (starter) |si| {
                if (c & raw_byte == 0 and out.items[si] & raw_byte == 0 and (last_cc == -1 or (last_cc != 0 and last_cc < cc))) {
                    if (composePair(@intCast(out.items[si]), @intCast(c))) |comp| {
                        out.items[si] = comp;
                        continue;
                    }
                }
            }
            if (cc == 0) {
                starter = out.items.len;
                last_cc = -1;
            } else last_cc = cc;
            try out.append(a, c);
        }
        var bytes = std.ArrayList(u8).empty;
        for (out.items) |c| {
            if (c & raw_byte != 0) {
                try bytes.append(a, @truncate(c));
            } else {
                var buf: [4]u8 = undefined;
                const n = std.unicode.utf8Encode(@intCast(c), &buf) catch unreachable;
                try bytes.appendSlice(a, buf[0..n]);
            }
        }
        return bytes.items;
    }

    fn combiningClass(cp: u21) i32 {
        const table = &uni.combining_class;
        var lo: usize = 0;
        var hi: usize = table.len;
        while (lo < hi) {
            const mid = (lo + hi) / 2;
            if (table[mid].cp == cp) return table[mid].ccc;
            if (table[mid].cp < cp) lo = mid + 1 else hi = mid;
        }
        return 0;
    }

    /// The primary composite of `first + second`, if any (Hangul algorithmically).
    fn composePair(first: u21, second: u21) ?u32 {
        if (first >= 0x1100 and first <= 0x1112 and second >= 0x1161 and second <= 0x1175) {
            return 0xAC00 + (@as(u32, first - 0x1100) * 21 + (second - 0x1161)) * 28;
        }
        if (first >= 0xAC00 and first <= 0xD7A3 and (first - 0xAC00) % 28 == 0 and second >= 0x11A8 and second <= 0x11C2) {
            return @as(u32, first) + (second - 0x11A7);
        }
        const table = &uni.compose_pairs;
        var lo: usize = 0;
        var hi: usize = table.len;
        while (lo < hi) {
            const mid = (lo + hi) / 2;
            const e = table[mid];
            if (e.first == first and e.second == second) return e.out;
            if (e.first < first or (e.first == first and e.second < second)) lo = mid + 1 else hi = mid;
        }
        return null;
    }

    fn mapOf(table: []const uni.FoldMulti, cp: u21) ?[]const u8 {
        var lo: usize = 0;
        var hi: usize = table.len;
        while (lo < hi) {
            const mid = (lo + hi) / 2;
            if (table[mid].cp == cp) return table[mid].out;
            if (table[mid].cp < cp) lo = mid + 1 else hi = mid;
        }
        return null;
    }

    /// Length of the run of `\r?\n` line breaks at the start of `s` (0 if none).
    fn newlineRunLen(s: []const u8) usize {
        var i: usize = 0;
        while (true) {
            if (i < s.len and s[i] == '\n') {
                i += 1;
            } else if (i + 1 < s.len and s[i] == '\r' and s[i + 1] == '\n') {
                i += 2;
            } else return i;
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
    /// SentencePiece's whitespace normalisation (`nmt_nfkc` plus
    /// `remove_extra_whitespaces`): the C0 controls, the Unicode separators
    /// and the zero-width joiners become a space, runs of spaces collapse to
    /// one, and a leading run is dropped (Metaspace supplies the prefix). The
    /// zero-width space and the byte order mark are removed outright.
    fn spWhitespace(a: Allocator, raw: []const u8) ![]const u8 {
        var out = std.ArrayList(u8).empty;
        var i: usize = 0;
        var pending_space = false;
        while (i < raw.len) {
            const len = utf8Len(raw[i]);
            const cp: u21 = if (len == 1) raw[i] else (std.unicode.utf8Decode(raw[i..][0..@min(len, raw.len - i)]) catch raw[i]);
            i += len;
            const space = switch (cp) {
                0x09...0x0D, 0x20, 0x85, 0xA0, 0x1680 => true,
                0x2000...0x200A => true,
                0x2028, 0x2029, 0x202F, 0x205F, 0x3000 => true,
                0x200C, 0x200D => true, // ZWNJ / ZWJ
                else => false,
            };
            const drop = switch (cp) {
                0x00...0x08, 0x0E...0x1F, 0x7F => true,
                0x200B, 0xFEFF => true, // zero-width space, BOM
                else => false,
            };
            if (drop) continue;
            if (space) {
                pending_space = out.items.len > 0; // strip leading
                continue;
            }
            if (pending_space) {
                try out.append(a, ' ');
                pending_space = false;
            }
            var buf: [4]u8 = undefined;
            const n = std.unicode.utf8Encode(cp, &buf) catch {
                try out.append(a, raw[i - len]);
                continue;
            };
            try out.appendSlice(a, buf[0..n]);
        }
        // Leading whitespace is dropped (Metaspace supplies the prefix), a
        // trailing run collapses to one space, which sentencepiece keeps.
        if (pending_space) try out.append(a, ' ');
        return out.items;
    }

    fn utf8Len(first: u8) usize {
        return std.unicode.utf8ByteSequenceLength(first) catch 1;
    }

    /// Unigram (SentencePiece) segmentation: the path through `word` whose
    /// vocabulary pieces have the highest total log-probability, by Viterbi
    /// over the UTF-8 character boundaries. A position no piece covers falls
    /// back to one character as `unk_id`, scored below every vocabulary entry
    /// so a covered path always wins.
    fn unigramWord(self: *Tokenizer, gpa: Allocator, word: []const u8, out: *std.ArrayList(u32)) !void {
        var scratch = std.heap.ArenaAllocator.init(gpa);
        defer scratch.deinit();
        const a = scratch.allocator();
        const n = word.len;
        const best = try a.alloc(f32, n + 1);
        const prev = try a.alloc(usize, n + 1);
        const piece_id = try a.alloc(u32, n + 1);
        const reachable = try a.alloc(bool, n + 1);
        const is_unk = try a.alloc(bool, n + 1);
        @memset(best, 0);
        @memset(reachable, false);
        @memset(is_unk, false);
        reachable[0] = true;
        var min_score: f32 = 0;
        for (self.unigram_scores) |sc| min_score = @min(min_score, sc);
        const unk_score = min_score - 10.0; // sentencepiece's kUnkPenalty

        var i: usize = 0;
        while (i < n) : (i += utf8Len(word[i])) {
            if (!reachable[i]) continue;
            var covered = false;
            var end = i + utf8Len(word[i]);
            const limit = @min(n, i + @max(self.unigram_max_len, 1));
            while (end <= limit) {
                if (self.vocab.get(word[i..end])) |id| {
                    const score = best[i] + self.unigram_scores[id];
                    if (!reachable[end] or score > best[end]) {
                        best[end] = score;
                        prev[end] = i;
                        piece_id[end] = id;
                        is_unk[end] = false;
                        reachable[end] = true;
                    }
                    covered = true;
                }
                if (end == n) break;
                end += utf8Len(word[end]);
            }
            if (!covered) {
                // One character as the unknown token (or its bytes, with
                // `byte_fallback`), so the path continues.
                const one = i + utf8Len(word[i]);
                const score = best[i] + unk_score;
                if (!reachable[one] or score > best[one]) {
                    best[one] = score;
                    prev[one] = i;
                    piece_id[one] = self.unk_id orelse 0;
                    is_unk[one] = true;
                    reachable[one] = true;
                }
            }
        }
        if (!reachable[n]) return; // no path: nothing to emit

        var rev = std.ArrayList(u32).empty;
        defer rev.deinit(a);
        var pos = n;
        while (pos > 0) {
            const from = prev[pos];
            if (is_unk[pos] and self.byte_fallback) {
                // sentencepiece replaces an unknown piece with its bytes,
                // highest byte last so the reversal puts them back in order.
                var b = pos;
                while (b > from) : (b -= 1) {
                    var nb: [8]u8 = undefined;
                    const name = std.fmt.bufPrint(&nb, "<0x{X:0>2}>", .{word[b - 1]}) catch unreachable;
                    try rev.append(a, self.vocab.get(name) orelse self.unk_id orelse 0);
                }
            } else {
                try rev.append(a, piece_id[pos]);
            }
            pos = from;
        }
        std.mem.reverse(u32, rev.items);
        try self.cacheWord(word, rev.items);
        try out.appendSlice(gpa, rev.items);
    }

    fn bpeWord(self: *Tokenizer, gpa: Allocator, word: []const u8, out: *std.ArrayList(u32)) !void {
        if (word.len == 0) return;
        if (self.cache.get(word)) |ids| {
            try out.appendSlice(gpa, ids);
            return;
        }
        if (self.unigram_scores.len > 0) return self.unigramWord(gpa, word, out);
        // A character with no vocabulary entry, no byte fallback and no unknown
        // token is left out before merging, as `tokenizers`' BPE does, so its
        // neighbours can merge across it ("a h" with the space dropped).
        if (!self.rank_bpe and !self.byte_fallback and self.unk_id == null) {
            var kept = std.ArrayList(u8).empty;
            defer kept.deinit(gpa);
            var dropped = false;
            var i: usize = 0;
            while (i < word.len) {
                const n = std.unicode.utf8ByteSequenceLength(word[i]) catch 1;
                const end = @min(word.len, i + n);
                if (self.vocab.get(word[i..end]) != null) try kept.appendSlice(gpa, word[i..end]) else dropped = true;
                i = end;
            }
            if (dropped) {
                var sub = std.ArrayList(u32).empty;
                defer sub.deinit(gpa);
                try self.bpeWord(gpa, kept.items, &sub);
                try self.cacheWord(word, sub.items);
                try out.appendSlice(gpa, sub.items);
                return;
            }
        }
        var ids = std.ArrayList(u32).empty;
        defer ids.deinit(gpa);

        // tiktoken encodes a whole pre-token that is in the vocabulary as that
        // token without merging; HF models mirror this with `ignore_merges`.
        if (self.ignore_merges or self.rank_bpe) {
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
        if (self.rank_bpe) self.rankMerge(word, &syms, std.math.maxInt(u32));
        while (!self.rank_bpe and syms.items.len > 1) {
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

    /// tiktoken's merge loop: repeatedly merges the adjacent pair whose
    /// concatenation has the lowest rank below `max_rank` (leftmost on ties).
    fn rankMerge(self: *const Tokenizer, word: []const u8, syms: *std.ArrayList(Symbol), max_rank: u32) void {
        while (syms.items.len > 1) {
            var best_rank: u32 = max_rank;
            var best_i: usize = 0;
            var i: usize = 0;
            while (i + 1 < syms.items.len) : (i += 1) {
                const pair = word[syms.items[i].start..syms.items[i + 1].end];
                if (self.vocab.get(pair)) |rank| {
                    if (rank < best_rank) {
                        best_rank = rank;
                        best_i = i;
                    }
                }
            }
            if (best_rank == max_rank) break;
            syms.items[best_i].end = syms.items[best_i + 1].end;
            _ = syms.orderedRemove(best_i + 1);
        }
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
/// The `Punctuation` pre-tokenizer's test (`tokenizers`' `is_punc`): ASCII
/// punctuation, which includes `$+<=>^`|~`, or Unicode `P*` — not the other
/// symbols (`©`, `€`, `×`).
pub fn isPunctuation(cp: u21) bool {
    if (cp < 128) return std.ascii.isPunctuation(@intCast(cp));
    return inRanges(cp, &uni.punctuation);
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

const UnicodeForm = enum { nfkc, nfc };
const MetaspacePrepend = enum { always, first, never };

const RegexSplitter = struct {
    text: []const u8,
    kind: RegexKind,
    pos: usize = 0,
    /// End of the previous match (`\G`); a pattern may match there contiguously.
    g: usize = 0,
    /// Character class of `ds2_letters` (see `Tokenizer.ds2_letter_class`).
    class: []const uni.Range = &.{},

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
                self.g = end;
                return self.text[start..end];
            }
            i += self.cpAt(i).?.len;
        }
        self.pos = self.text.len;
        return self.text[start..];
    }

    /// Decimal digits (`\p{Nd}`) from `i`: how many, and where they end.
    fn ndRun(self: *const RegexSplitter, i: usize) struct { count: usize, end: usize } {
        var j = i;
        var count: usize = 0;
        while (self.cpAt(j)) |n| {
            if (!inRanges(n.cp, &uni.decimal_digits)) break;
            j += n.len;
            count += 1;
        }
        return .{ .count = count, .end = j };
    }

    fn skipCps(self: *const RegexSplitter, i: usize, count: usize) usize {
        var j = i;
        for (0..count) |_| j += self.cpAt(j).?.len;
        return j;
    }

    /// The `script_runs` class of `cp` (0: none).
    fn scriptClass(cp: u21) u8 {
        const in = struct {
            fn f(c: u21, lo: u21, hi: u21) bool {
                return c >= lo and c <= hi;
            }
        }.f;
        if (in(cp, 0x4E00, 0x9FFF) or in(cp, 0x3040, 0x309F) or in(cp, 0x30A0, 0x30FF) or in(cp, 0x3400, 0x4DBF) or
            in(cp, 0xF900, 0xFAFF) or in(cp, 0xFF65, 0xFF9F) or in(cp, 0x2F00, 0x2FDF)) return 1;
        if (in(cp, 0x0E00, 0x0E7F)) return 2;
        if (in(cp, 0x0E80, 0x0EFF)) return 3;
        if (in(cp, 0x1780, 0x17FF)) return 4;
        if (in(cp, 0x1000, 0x109F) or in(cp, 0xAA60, 0xAA7F) or in(cp, 0xA9E0, 0xA9FF)) return 5;
        if (in(cp, 0xAC00, 0xD7AF) or in(cp, 0x1100, 0x11FF)) return 6;
        return 0;
    }

    /// `(?:\p{L}\p{M}*(?: \p{L}\p{M}*)*)+` from `i` (a letter): letters with
    /// their marks, and single spaces that are followed by a letter.
    fn phraseRun(self: *const RegexSplitter, from: usize) usize {
        var i = from;
        while (self.cpAt(i)) |n| {
            if (isLetter(n.cp) or inRanges(n.cp, &uni.marks)) {
                i += n.len;
            } else if (n.cp == ' ') {
                const after = self.cpAt(i + 1) orelse break;
                if (!isLetter(after.cp)) break;
                i += 1;
            } else break;
        }
        return i;
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
            .gpt2, .qwen2, .qwen2_bare_punct, .qwen3_5, .phrase, .llama3, .nanochat => {},
            .o200k, .o200k_digit1 => return self.matchO200k(start, first),
            .kimi => return self.matchKimi(start, first),
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
            .nd_chunks510 => {
                // Greedy 1..510 digits such that what is left of the run is a
                // whole number of 510-digit chunks.
                const run = self.ndRun(start);
                if (run.count == 0) return start;
                return self.skipCps(start, (run.count - 1) % 510 + 1);
            },
            .nd_lead => {
                if (start != 0) return start;
                const run = self.ndRun(0);
                if (run.end != self.text.len or run.count % 3 == 0) return start;
                return self.skipCps(0, run.count % 3);
            },
            .nd_groups3 => {
                if (start != 0 and start != self.g) return start;
                if (self.ndRun(start).count < 3) return start;
                return self.skipCps(start, 3);
            },
            .script_runs => {
                const class = scriptClass(first.cp);
                if (class == 0) return start;
                var i = start;
                while (self.cpAt(i)) |n| {
                    if (scriptClass(n.cp) != class) break;
                    i += n.len;
                }
                return i;
            },
            .ds2_letters, .ds2_punct => {
                var i = start;
                var c = first;
                if (isWhitespace(c.cp)) {
                    const n = self.cpAt(i + c.len) orelse return start;
                    i += c.len;
                    c = n;
                }
                const letters = self.kind == .ds2_letters;
                if (!(if (letters) inRanges(c.cp, self.class) else isDs2Punct(c.cp))) return start;
                while (self.cpAt(i)) |n| {
                    if (!(if (letters) inRanges(n.cp, self.class) else isDs2Punct(n.cp))) break;
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
            .cjk_kana => {
                const in = struct {
                    fn f(cp: u21) bool {
                        return (cp >= 0x4E00 and cp <= 0x9FA5) or (cp >= 0x3040 and cp <= 0x30FF);
                    }
                }.f;
                if (!in(first.cp)) return start;
                var i = start;
                while (self.cpAt(i)) |n| {
                    if (!in(n.cp)) break;
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
            .o200k, .o200k_digit1, .kimi, .deepseek3, .digits3, .newlines, .ds2_letters, .ds2_punct, .trailing_ws, .cjk, .cjk_kana, .nd_chunks510, .nd_lead, .nd_groups3, .script_runs => unreachable,
            .qwen2, .qwen2_bare_punct, .qwen3_5, .phrase, .llama3, .nanochat => {
                // '[^\r\n\p{L}\p{N}]?\p{L}+' (Qwen 3.5: `[\p{L}\p{M}]+`; phrase: see `phrase`)
                const marks = self.kind == .qwen3_5;
                var i = start;
                var c = first;
                const starts_word = struct {
                    fn f(k: RegexKind, cp: u21) bool {
                        return isLetter(cp) or (k == .qwen3_5 and inRanges(cp, &uni.marks));
                    }
                }.f;
                if (!isNewline(c.cp) and !isLetter(c.cp) and !isNumber(c.cp)) {
                    if (self.cpAt(i + c.len)) |n| {
                        if (starts_word(self.kind, n.cp)) {
                            i += c.len;
                            c = n;
                        }
                    }
                }
                if (starts_word(self.kind, c.cp)) {
                    if (self.kind == .phrase) return self.phraseRun(i);
                    while (self.cpAt(i)) |n| {
                        if (!isLetter(n.cp) and !(marks and inRanges(n.cp, &uni.marks))) break;
                        i += n.len;
                    }
                    return i;
                }
                // '\p{N}' or '\p{N}{1,3}'
                if (isNumber(first.cp)) {
                    const max: usize = switch (self.kind) {
                        .llama3 => 3,
                        .nanochat => 2,
                        else => 1,
                    };
                    i = start;
                    var count: usize = 0;
                    while (self.cpAt(i)) |n| {
                        if (!isNumber(n.cp) or count == max) break;
                        i += n.len;
                        count += 1;
                    }
                    return i;
                }
                // ' ?[^\s\p{L}\p{N}]+[\r\n]*' (Qwen 3.5 also excludes marks)
                i = start;
                c = first;
                const punct = struct {
                    fn f(m: bool, cp: u21) bool {
                        return !isWhitespace(cp) and !isLetter(cp) and !isNumber(cp) and !(m and inRanges(cp, &uni.marks));
                    }
                }.f;
                if (c.cp == ' ') {
                    if (self.cpAt(i + 1)) |n| {
                        if (punct(marks, n.cp)) {
                            i += 1;
                            c = n;
                        }
                    }
                }
                if (punct(marks, c.cp)) {
                    while (self.cpAt(i)) |n| {
                        if (!punct(marks, n.cp)) break;
                        i += n.len;
                    }
                    if (self.kind == .qwen2_bare_punct) return i;
                    if (self.kind == .phrase) {
                        // `[\r\n/]?`
                        if (self.cpAt(i)) |n| {
                            if (isNewline(n.cp) or n.cp == '/') i += n.len;
                        }
                        return i;
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

    /// o200k (gpt-oss, Phi-4, Nemotron Nano): `[^\r\n\p{L}\p{N}]?U*L+c? | [^\r\n\p{L}\p{N}]?U+L*c? |
    /// \p{N}{1,3} | ?[^\s\p{L}\p{N}]+[\r\n/]* | \s*[\r\n]+ | \s+(?!\S) | \s+` with
    /// `U` = `[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]` and `L` = `[\p{Ll}\p{Lm}\p{Lo}\p{M}]`: Kimi's pattern
    /// without its Han alternative, and `/` may follow punctuation.
    fn matchO200k(self: *const RegexSplitter, start: usize, first: Cp) usize {
        return self.matchCased(start, first);
    }

    /// Kimi: `[\p{Han}]+ | [^\r\n\p{L}\p{N}]?U*L+c? | [^\r\n\p{L}\p{N}]?U+L*c? | \p{N}{1,3} |
    /// ?[^\s\p{L}\p{N}]+[\r\n]* | \s*[\r\n]+ | \s+(?!\S) | \s+` with `U` = `[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]`,
    /// `L` = `[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]` and `c` the contraction. The alternatives are tried
    /// in order with the regex's backtracking (greedy `U*`, then the optional leading character).
    fn matchKimi(self: *const RegexSplitter, start: usize, first: Cp) usize {
        if (isHan(first.cp)) {
            var i = start;
            while (self.cpAt(i)) |n| {
                if (!isHan(n.cp)) break;
                i += n.len;
            }
            return i;
        }
        return self.matchCased(start, first);
    }

    /// The cased-word pattern shared by Kimi and o200k (see both).
    fn matchCased(self: *const RegexSplitter, start: usize, first: Cp) usize {
        const lead = !isNewline(first.cp) and !isLetter(first.cp) and !isNumber(first.cp);
        const after_lead = start + first.len;
        if (lead) if (self.kimiWord(after_lead, true)) |e| return self.contractionAt(e);
        if (self.kimiWord(start, true)) |e| return self.contractionAt(e);
        if (lead) if (self.kimiWord(after_lead, false)) |e| return self.contractionAt(e);
        if (self.kimiWord(start, false)) |e| return self.contractionAt(e);
        if (isNumber(first.cp)) {
            var i = start;
            var count: usize = 0;
            while (self.cpAt(i)) |n| {
                if (!isNumber(n.cp) or count == @as(usize, if (self.kind == .o200k_digit1) 1 else 3)) break;
                i += n.len;
                count += 1;
            }
            return i;
        }
        // ` ?[^\s\p{L}\p{N}]+[\r\n]*`
        var i = start;
        var c = first;
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
                if (!isNewline(n.cp) and !((self.kind == .o200k or self.kind == .o200k_digit1) and n.cp == '/')) break;
                i += n.len;
            }
            return i;
        }
        return self.matchNewlinesOrWhitespace(start);
    }

    /// End of `U*L+` (`need_lower`) or `U+L*` at `i` as the regex engine finds
    /// it: `U*` takes the longest run and gives characters back one at a time
    /// until the rest matches. Null when nothing matches.
    fn kimiWord(self: *const RegexSplitter, i: usize, need_lower: bool) ?usize {
        var k = i;
        while (self.cpAt(k)) |n| {
            if (!self.isCasedUpper(n.cp)) break;
            k += n.len;
        }
        if (!need_lower) {
            if (k == i) return null;
            return self.kimiLowerRun(k);
        }
        var s = k;
        while (true) {
            if (self.cpAt(s)) |n| {
                if (self.isCasedLower(n.cp)) return self.kimiLowerRun(s);
            }
            if (s == i) return null;
            s -= 1;
            while (s > i and (self.text[s] & 0xC0) == 0x80) s -= 1;
        }
    }

    fn kimiLowerRun(self: *const RegexSplitter, start: usize) usize {
        var i = start;
        while (self.cpAt(i)) |n| {
            if (!self.isCasedLower(n.cp)) break;
            i += n.len;
        }
        return i;
    }

    fn isHan(cp: u21) bool {
        return inRanges(cp, &uni.han);
    }

    /// `[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]`
    fn isCasedUpper(self: *const RegexSplitter, cp: u21) bool {
        if (self.kind == .kimi and isHan(cp)) return false;
        return (isLetter(cp) and !inRanges(cp, &uni.lower)) or inRanges(cp, &uni.marks);
    }

    /// `[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]`
    fn isCasedLower(self: *const RegexSplitter, cp: u21) bool {
        if (self.kind == .kimi and isHan(cp)) return false;
        return (isLetter(cp) and !inRanges(cp, &uni.upper)) or inRanges(cp, &uni.marks);
    }

    /// DeepSeek V3: `[ascii punct][A-Za-z]+ | [^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+ | ?[\p{P}\p{S}]+[\r\n]* |
    /// \s*[\r\n]+ | \s+(?!\S) | \s+` (symbols approximated as "neither letter, number nor whitespace").
    /// DeepSeek V3's main pattern (AFMoE's too):
    /// `[!"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~][A-Za-z]+ | [^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+ |
    /// ?[\p{P}\p{S}]+[\r\n]* | \s*[\r\n]+ | \s+(?!\S) | \s+`. Words run over letters *and*
    /// combining marks; punctuation is exactly `P` and `S`.
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
        if (!isLetterOrMark(c.cp) and !isNewline(c.cp) and !isPunctSymbol(c.cp)) {
            if (self.cpAt(i + c.len)) |n| {
                if (isLetterOrMark(n.cp)) {
                    i += c.len;
                    c = n;
                }
            }
        }
        if (isLetterOrMark(c.cp)) {
            while (self.cpAt(i)) |n| {
                if (!isLetterOrMark(n.cp)) break;
                i += n.len;
            }
            return i;
        }
        i = start;
        c = first;
        if (c.cp == ' ') {
            if (self.cpAt(i + 1)) |n| {
                if (isPunctSymbol(n.cp)) {
                    i += 1;
                    c = n;
                }
            }
        }
        if (isPunctSymbol(c.cp)) {
            while (self.cpAt(i)) |n| {
                if (!isPunctSymbol(n.cp)) break;
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

    fn isLetterOrMark(cp: u21) bool {
        return isLetter(cp) or inRanges(cp, &uni.marks);
    }

    fn isPunctSymbol(cp: u21) bool {
        return inRanges(cp, &uni.punct_symbols);
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

test "regex splitter nanochat: digit pairs, punctuation keeps its line breaks" {
    const r = "'(?i:[sdmt]|ll|ve|re)|[^\\r\\n\\p{L}\\p{N}]?+\\p{L}+|\\p{N}{1,2}| ?[^\\s\\p{L}\\p{N}]++[\\r\\n]*|\\s*[\\r\\n]|\\s+(?!\\S)|\\s+";
    try std.testing.expectEqual(RegexKind.nanochat, Tokenizer.classifyRegex(r));
    // From `tokenizers`' pre_tokenize_str on nanochat-students/nanochat-d20.
    var it = RegexSplitter{ .text = "assistant.\n\nThe 12345 capital", .kind = .nanochat };
    const want = [_][]const u8{ "assistant", ".\n\n", "The", " ", "12", "34", "5", " capital" };
    for (want) |w| try std.testing.expectEqualStrings(w, it.next().?);
    try std.testing.expect(it.next() == null);
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

test "NFKC: compatibility forms, then canonical composition" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    // (input, Python's unicodedata.normalize("NFKC", input))
    const cases = [_][2][]const u8{
        .{ "a\u{a0}b\u{2009}c\u{200b}d", "a b c\u{200b}d" },
        .{ "\u{ff26}\u{ff35}\u{3000}\u{fb01}ne x\u{b2}", "FU fine x2" },
        .{ "cafe\u{301}", "caf\u{e9}" },
        .{ "\u{ff76}\u{ff9e}", "\u{30ac}" },
        .{ "\u{3131}\u{314f}", "\u{ac00}" },
        .{ "\u{1100}\u{1161}\u{11a8}", "\u{ac01}" },
        .{ "o\u{308}\u{304}", "\u{22b}" },
        .{ "\xff\u{301}", "\xff\u{301}" },
    };
    for (cases) |c| try std.testing.expectEqualStrings(c[1], try Tokenizer.normalizeUnicode(a, c[0], .nfkc));
    // NFC composes but keeps compatibility forms.
    try std.testing.expectEqualStrings("caf\u{e9} \u{fb01} \u{c5}", try Tokenizer.normalizeUnicode(a, "cafe\u{301} \u{fb01} \u{212b}", .nfc));
}

test "a line-break split merged with the next piece (Laguna)" {
    const gpa = std.testing.allocator;
    const json =
        \\{"model":{"type":"BPE","vocab":{"a":0},"merges":[]},
        \\"pre_tokenizer":{"type":"Sequence","pretokenizers":[{"type":"Split","pattern":{"Regex":"(?:\\r?\\n)+(?!\\r?\\n)"},"behavior":"MergedWithNext","invert":false},{"type":"Split","pattern":{"Regex":"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"},"behavior":"Isolated","invert":false},{"type":"ByteLevel","add_prefix_space":false,"use_regex":false}]},
        \\"decoder":{"type":"ByteLevel"}}
    ;
    const tok = try Tokenizer.parse(gpa, json, null);
    defer tok.deinit();
    try std.testing.expect(tok.steps[0] == .newline_runs);
    try std.testing.expect(tok.steps[1] == .regex and tok.steps[1].regex == .qwen2);
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var pieces = std.ArrayList([]const u8).empty;
    try pieces.append(a, "f(x):\r\n\n  y\rz\n");
    for (tok.steps) |step| {
        var next = std.ArrayList([]const u8).empty;
        for (pieces.items) |piece| try tok.applyStep(a, step, piece, &next);
        pieces = next;
    }
    // Without the first step Qwen 2's regex would keep "):\r\n\n" together.
    const want = [_][]const u8{ "f", "(x", "):", "\r\n\n", " ", " y", "\r", "z", "\n" };
    try std.testing.expectEqual(want.len, pieces.items.len);
    for (want, pieces.items) |w, got| try std.testing.expectEqualStrings(w, got);
}

test "a regex with literal line breaks is still recognised (StableLM 2)" {
    try std.testing.expectEqual(RegexKind.qwen2, Tokenizer.classifyRegex("(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\r\n]*|\\s*[\r\n]+|\\s+(?!\\S)|\\s+"));
}

test "AFMoE digit groups and script runs, and DeepSeek V3 words with marks" {
    // Pieces from `tokenizers`' pre_tokenize_str on arcee-ai/Trinity-Nano-Preview.
    const Case = struct { kind: RegexKind, text: []const u8, want: []const []const u8 };
    const cases = [_]Case{
        .{ .kind = .nd_lead, .text = "1234567", .want = &.{ "1", "234567" } },
        .{ .kind = .nd_lead, .text = "123456", .want = &.{"123456"} },
        .{ .kind = .nd_lead, .text = "12a", .want = &.{"12a"} },
        .{ .kind = .nd_groups3, .text = "234567", .want = &.{ "234", "567" } },
        .{ .kind = .nd_groups3, .text = "2345", .want = &.{ "234", "5" } },
        .{ .kind = .script_runs, .text = "ab\u{0e20}\u{0e32}\u{1780}\u{17d2}\u{d55c}x", .want = &.{ "ab", "\u{0e20}\u{0e32}", "\u{1780}\u{17d2}", "\u{d55c}", "x" } },
        .{ .kind = .deepseek3, .text = "\u{1781}\u{17d2}\u{1798}\u{17c2}\u{179a} cafe\u{301}!", .want = &.{ "\u{1781}\u{17d2}\u{1798}\u{17c2}\u{179a}", " cafe\u{301}", "!" } },
    };
    for (cases) |c| {
        var it = RegexSplitter{ .text = c.text, .kind = c.kind };
        for (c.want) |w| try std.testing.expectEqualStrings(w, it.next().?);
        try std.testing.expect(it.next() == null);
    }
    var long: [1100]u8 = undefined;
    @memset(&long, '9');
    var it = RegexSplitter{ .text = &long, .kind = .nd_chunks510 };
    try std.testing.expectEqual(@as(usize, 80), it.next().?.len);
    try std.testing.expectEqual(@as(usize, 510), it.next().?.len);
    try std.testing.expectEqual(@as(usize, 510), it.next().?.len);
    try std.testing.expect(it.next() == null);
}

test "pre-tokenizer variants found by the tokenizer sweep" {
    // Pieces from `tokenizers`' pre_tokenize_str on each release's tokenizer.
    const Case = struct { kind: RegexKind, text: []const u8, want: []const []const u8 };
    const cases = [_]Case{
        // o200k (openai/gpt-oss-20b): marks belong to words.
        .{ .kind = .o200k, .text = "\u{939}\u{93f}\u{928}\u{94d}\u{926}\u{940} \u{92e}\u{947}\u{902}", .want = &.{ "\u{939}\u{93f}\u{928}\u{94d}\u{926}\u{940}", " \u{92e}\u{947}\u{902}" } },
        // Falcon-H1: single digits.
        .{ .kind = .o200k_digit1, .text = "x123", .want = &.{ "x", "1", "2", "3" } },
        .{ .kind = .o200k, .text = "x123", .want = &.{ "x", "123" } },
        // Seed-OSS: no line breaks after punctuation.
        .{ .kind = .qwen2_bare_punct, .text = "):\n x", .want = &.{ "):", "\n", " x" } },
        .{ .kind = .qwen2, .text = "):\n x", .want = &.{ "):\n", " x" } },
        // DeepSeek V3's CJK and kana split does not take the em dash.
        .{ .kind = .cjk_kana, .text = "\u{65e5}\u{672c}\u{3042}\u{2014}x", .want = &.{ "\u{65e5}\u{672c}\u{3042}", "\u{2014}x" } },
    };
    for (cases) |c| {
        var it = RegexSplitter{ .text = c.text, .kind = c.kind };
        for (c.want) |w| try std.testing.expectEqualStrings(w, it.next().?);
        try std.testing.expect(it.next() == null);
    }
    try std.testing.expectEqual(RegexKind.o200k_digit1, Tokenizer.classifyRegex("[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+|[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"));
    try std.testing.expectEqual(RegexKind.qwen2_bare_punct, Tokenizer.classifyRegex("(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1}| ?[^\\s\\p{L}\\p{N}\r\n]+|\\s*[\r\n]+|\\s+(?!\\S)|\\s+"));
    try std.testing.expectEqual(RegexKind.cjk_kana, Tokenizer.classifyRegex("[\u{4e00}-\u{9fa5}\u{3040}-\u{309f}\u{30a0}-\u{30ff}]+"));
    // Punctuation is ASCII punctuation or Unicode P, not the other symbols.
    try std.testing.expect(isPunctuation('$') and isPunctuation(0xAB) and isPunctuation(0x2014));
    try std.testing.expect(!isPunctuation(0xA9) and !isPunctuation(0x20AC) and !isPunctuation(0xD7));
    // DeepSeek V2's letter class comes from the pattern itself.
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const class = try Tokenizer.parseCharClass(arena.allocator(), "\\s?[A-Za-z\u{b5}\u{c0}-\u{d6}\\-]+");
    try std.testing.expect(inRanges('q', class) and inRanges(0xB5, class) and inRanges(0xC3, class) and inRanges('-', class));
    try std.testing.expect(!inRanges(0xD7, class) and !inRanges(0x939, class) and !inRanges('[', class));
}

test "added tokens absorb the whitespace their lstrip / rstrip flags name (Phi-3)" {
    const gpa = std.testing.allocator;
    const json =
        \\{"model":{"type":"BPE","vocab":{"a":0,"b":1," ":2,"\n":3},"merges":[]},
        \\"added_tokens":[{"id":4,"content":"<|user|>","rstrip":true,"special":true},{"id":5,"content":"<|end|>","lstrip":true,"special":true}]}
    ;
    const tok = try Tokenizer.parse(gpa, json, null);
    defer tok.deinit();
    const ids = try tok.encode(gpa, "a<|user|>\n  b \n<|end|>\nb", false);
    defer gpa.free(ids);
    try std.testing.expectEqualSlices(u32, &.{ 0, 4, 1, 5, 3, 1 }, ids);
}

test "byte-level bpe round trip" {
    const gpa = std.testing.allocator;
    const json =
        \\{"model":{"type":"BPE","vocab":{"h":0,"e":1,"l":2,"o":3,"he":4,"ll":5,"hell":6,"hello":7,"Ġ":8,"Ġw":9,"<|end|>":10,"w":11},
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

test "regex splitter kimi" {
    var it = RegexSplitter{ .text = "Kimi是Moonshot的模型，你好 world's 1234 אבגA  \n x", .kind = .kimi };
    const expected = [_][]const u8{ "Kimi", "是", "Moonshot", "的模型", "，", "你好", " world's", " ", "123", "4", " אבג", "A", "  \n", " x" };
    for (expected) |e| try std.testing.expectEqualStrings(e, it.next().?);
    try std.testing.expect(it.next() == null);
}

/// Encodes every reference string of a tiktoken fixture with the rank-based
/// tokenizer and with the `tokenizer.json` synthesised for exports.
fn checkTiktokenFixture(comptime dir: []const u8, comptime file: []const u8, kind: TiktokenKind) !void {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const cwd = Io.Dir.cwd();
    const model_text = try cwd.readFileAlloc(io, dir ++ "/" ++ file, arena, .unlimited);
    const cfg = try cwd.readFileAlloc(io, dir ++ "/tokenizer_config.json", arena, .unlimited);
    const ref_text = try cwd.readFileAlloc(io, dir ++ "/reference.json", arena, .unlimited);
    var ref = try std.json.parseFromSlice(std.json.Value, gpa, ref_text, .{});
    defer ref.deinit();
    try std.testing.expect(Tokenizer.looksLikeTiktoken(model_text));
    try std.testing.expect(!Tokenizer.looksLikeTiktoken("\x0a\x05\x0a\x03<s>"));

    var d = try cwd.openDir(io, dir, .{});
    defer d.close(io);
    const loaded = try Tokenizer.loadDir(gpa, io, arena, d, dir, cfg);
    const tok = loaded.tokenizer;
    defer tok.deinit();
    try std.testing.expect(tok.rank_bpe);
    try std.testing.expectEqual(kind, tok.tiktoken_kind.?);
    const n_base: usize = @intCast(ref.value.object.get("n_base").?.integer);
    try std.testing.expectEqual(@as(usize, @intCast(ref.value.object.get("n_vocab").?.integer)), tok.vocabSize());
    switch (kind) {
        .kimi => {
            try std.testing.expectEqualStrings("kimi-k2", tok.ggmlPreName().?);
            try std.testing.expectEqual(@as(?u32, @intCast(n_base)), tok.bos_id);
            try std.testing.expectEqual(@as(?u32, @intCast(n_base + 1)), tok.eos_id);
            try std.testing.expect(!tok.add_bos);
            try std.testing.expect(tok.isSpecial(@intCast(n_base + 17)));
            try std.testing.expect(!tok.isSpecial(@intCast(n_base + 11))); // <|tool_calls_section_begin|>, special: false
            try std.testing.expectEqualStrings("<|reserved_token_" ++ "670|>", tok.id_to_token[n_base + 13]);
        },
        .llama3 => {
            try std.testing.expectEqualStrings("llama-bpe", tok.ggmlPreName().?);
            try std.testing.expectEqual(@as(?u32, @intCast(n_base)), tok.bos_id);
            try std.testing.expectEqual(@as(?u32, @intCast(n_base + 1)), tok.eos_id);
            try std.testing.expect(tok.add_bos);
            try std.testing.expectEqual(@as(?u32, @intCast(n_base + 9)), tok.tokenToId("<|eot_id|>"));
            try std.testing.expectEqual(@as(?u32, @intCast(n_base + 10)), tok.tokenToId("<|reserved_special_token_5|>"));
        },
    }

    const tok2 = try Tokenizer.parse(gpa, loaded.json, cfg);
    defer tok2.deinit();
    try std.testing.expect(tok2.ignore_merges and !tok2.rank_bpe);
    try std.testing.expectEqual(tok.vocabSize(), tok2.vocabSize());

    for (ref.value.object.get("cases").?.array.items) |case| {
        const text = case.object.get("text").?.string;
        const want = try arena.alloc(u32, case.object.get("ids").?.array.items.len);
        for (case.object.get("ids").?.array.items, 0..) |v, i| want[i] = @intCast(v.integer);
        const ids = try tok.encode(gpa, text, false);
        defer gpa.free(ids);
        try std.testing.expectEqualSlices(u32, want, ids);
        const back = try tok.decode(gpa, ids, false);
        defer gpa.free(back);
        try std.testing.expectEqualStrings(text, back);
        const ids2 = try tok2.encode(gpa, text, false);
        defer gpa.free(ids2);
        try std.testing.expectEqualSlices(u32, want, ids2);
    }
    // BOS is prepended only when the conventions ask for it.
    const with_bos = try tok.encode(gpa, "hi", true);
    defer gpa.free(with_bos);
    try std.testing.expectEqual(kind == .llama3, with_bos.len > 0 and with_bos[0] == n_base);
}

test "tiktoken kimi fixture matches the tiktoken package" {
    try checkTiktokenFixture("tests/fixtures/tiktoken_kimi", "tiktoken.model", .kimi);
}

test "tiktoken llama3 fixture matches the tiktoken package" {
    try checkTiktokenFixture("tests/fixtures/tiktoken_llama3", "tokenizer.model", .llama3);
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

test "unigram segmentation picks the highest-scoring path" {
    const gpa = std.testing.allocator;
    // "\u2581ab" can be cut as "\u2581a"+"b" (-1.5 -3.0) or "\u2581"+"ab" (-4.0 -0.5):
    // the second path wins on total log-probability, not on longest match.
    const json =
        \\{"model":{"type":"Unigram","unk_id":0,"vocab":[["<unk>",0.0],["\u2581",-4.0],["a",-3.5],["b",-3.0],["ab",-0.5],["\u2581a",-1.5],["\u2581ab",-9.0]]},
        \\"normalizer":{"type":"Sequence","normalizers":[{"type":"Precompiled","precompiled_charsmap":""}]},
        \\"pre_tokenizer":{"type":"Metaspace","replacement":"\u2581","prepend_scheme":"always"},
        \\"decoder":{"type":"Metaspace","replacement":"\u2581","prepend_scheme":"always"},
        \\"added_tokens":[{"id":0,"content":"<unk>","special":true}]}
    ;
    const tok = try Tokenizer.parse(gpa, json, null);
    defer tok.deinit();
    const ids = try tok.encode(gpa, "ab", false);
    defer gpa.free(ids);
    try std.testing.expectEqualSlices(u32, &.{ 1, 4 }, ids);
    // A character no piece covers becomes the unknown token, and the path continues.
    const unk = try tok.encode(gpa, "aZb", false);
    defer gpa.free(unk);
    try std.testing.expectEqualSlices(u32, &.{ 5, 0, 3 }, unk);
}

test "unigram byte fallback and a prepended space that the normalizer replaces" {
    const gpa = std.testing.allocator;
    // Helium's normalizer: `Prepend " "` in front of `Replace " " -> "\u2581"`,
    // so the prefix is a metaspace, not a literal space. `\u00e9` is not in the
    // vocabulary, so `byte_fallback` spells it out as its two UTF-8 bytes.
    const json =
        \\{"model":{"type":"Unigram","unk_id":0,"byte_fallback":true,
        \\"vocab":[["<unk>",0.0],["\u2581",-3.0],["a",-4.0],["\u2581a",-2.0],["<0xC3>",-9.0],["<0xA9>",-9.0]]},
        \\"normalizer":{"type":"Sequence","normalizers":[{"type":"Prepend","prepend":" "},{"type":"Replace","pattern":{"String":" "},"content":"\u2581"}]},
        \\"pre_tokenizer":null,
        \\"decoder":{"type":"Sequence","decoders":[{"type":"Replace","pattern":{"String":"\u2581"},"content":" "},{"type":"ByteFallback"},{"type":"Fuse"}]},
        \\"added_tokens":[{"id":0,"content":"<unk>","special":true}]}
    ;
    const tok = try Tokenizer.parse(gpa, json, null);
    defer tok.deinit();
    // " a" -> "\u2581a", one piece; without the fix the prefix stayed a literal space.
    const ids = try tok.encode(gpa, "a", false);
    defer gpa.free(ids);
    try std.testing.expectEqualSlices(u32, &.{3}, ids);
    const bytes = try tok.encode(gpa, "a\u{e9}", false);
    defer gpa.free(bytes);
    try std.testing.expectEqualSlices(u32, &.{ 3, 4, 5 }, bytes);
}

test "sentencepiece whitespace normalisation collapses runs and drops the leading one" {
    const gpa = std.testing.allocator;
    const json =
        \\{"model":{"type":"Unigram","unk_id":0,"vocab":[["<unk>",0.0],["\u2581",-4.0],["a",-3.5],["b",-3.0]]},
        \\"normalizer":{"type":"Precompiled","precompiled_charsmap":""},
        \\"pre_tokenizer":{"type":"Metaspace","replacement":"\u2581","prepend_scheme":"always"},
        \\"decoder":{"type":"Metaspace","replacement":"\u2581","prepend_scheme":"always"}}
    ;
    const tok = try Tokenizer.parse(gpa, json, null);
    defer tok.deinit();
    // "  a\t\tb " -> "a b " -> pieces "\u2581a", "\u2581b", "\u2581".
    const ids = try tok.encode(gpa, "  a\t\tb ", false);
    defer gpa.free(ids);
    try std.testing.expectEqualSlices(u32, &.{ 1, 2, 1, 3, 1 }, ids);
}

test "BPE leaves out a character it cannot spell before merging" {
    const gpa = std.testing.allocator;
    // No pre-tokenizer, no byte fallback, no unk token, and no " " in the
    // vocabulary: tokenizers drops the space and merges "a" with "h".
    const json =
        \\{"model":{"type":"BPE","byte_fallback":false,"vocab":{"a":0,"h":1,"ah":2,"x":3},"merges":["a h"]},
        \\"normalizer":null,"pre_tokenizer":null,"added_tokens":[]}
    ;
    const tok = try Tokenizer.parse(gpa, json, null);
    defer tok.deinit();
    const ids = try tok.encode(gpa, "a hx", false);
    defer gpa.free(ids);
    try std.testing.expectEqualSlices(u32, &.{ 2, 3 }, ids);
}

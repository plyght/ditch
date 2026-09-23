//! `ditch add-model MODEL`: drafts a Lua model definition (docs/models.md)
//! for a checkpoint, then checks it.
//!
//! 1. **Read.** What describes the model without its weights: config.json,
//!    the tokenizer files and chat template, and every safetensors header
//!    (range requests for a Hub id, `hf://` or an http(s) URL; the files
//!    themselves for a directory). Nothing else is downloaded.
//! 2. **Shape-only copy.** The small files, and each shard as its real
//!    header followed by a hole the size of its data (a sparse file: every
//!    tensor has its name, dtype and shape, and reads as zeros).
//! 3. **Match.** The copy is loaded as each plausible known family with
//!    ditch's own loader, which names the first tensor a family needs that
//!    the checkpoint lacks, while `safetensors.lookup_log` records which
//!    checkpoint tensors it read. A family that loads and reads every tensor
//!    of the text model is a match. A missing tensor is matched to an unread
//!    one of the same place, shape and role, and the load is retried with
//!    that name.
//! 4. **Config keys.** A config.json key is read by the definition when
//!    changing or removing it changes the parsed configuration; the others
//!    are listed, less the ones that never matter to a forward pass.
//! 5. **Draft and check.** The definition, with a comment on every guess and
//!    on everything left unmapped, goes to the user models directory
//!    (`$XDG_CONFIG_HOME/ditch/models`, or `--models-dir`), and `ditch
//!    verify` runs on the model with it (skipped with `--dry-run`).

const std = @import("std");
const Io = std.Io;
const arch = @import("arch.zig");
const models = @import("models.zig");
const chat = @import("chat.zig");
const config = @import("config.zig");
const hf = @import("hf.zig");
const remote = @import("remote.zig");
const safetensors = @import("safetensors.zig");
const tensor = @import("tensor.zig");
const model_mod = @import("model.zig");
const verify = @import("verify.zig");

const Allocator = std.mem.Allocator;
const Arch = arch.Arch;
const Names = arch.Names;

pub const Ctx = struct {
    gpa: Allocator,
    arena: Allocator,
    io: Io,
    env: *std.process.Environ.Map,
    settings: *const config.Settings,
    http: *hf.Http,
    cache_root: []const u8,
    pool: *const tensor.Pool,
    out: *Io.Writer,
    result: *Io.Writer,
};

// ---------------------------------------------------------------------------
// Log capture
// ---------------------------------------------------------------------------

/// While set, log messages are collected here instead of printed (see
/// `logFn` in main.zig): a trial load's "missing tensor" is data, not an error.
pub var capture: ?*Capture = null;

pub const Capture = struct {
    arena: Allocator,
    lines: std.ArrayList(Line) = .empty,
    lock: std.atomic.Mutex = .unlocked,

    pub const Line = struct { level: std.log.Level, text: []const u8 };

    pub fn take(self: *Capture, comptime level: std.log.Level, comptime format: []const u8, args: anytype) bool {
        const text = std.fmt.allocPrint(self.arena, format, args) catch return false;
        while (!self.lock.tryLock()) std.atomic.spinLoopHint();
        defer self.lock.unlock();
        self.lines.append(self.arena, .{ .level = level, .text = text }) catch return false;
        return true;
    }

    fn errors(self: *const Capture, a: Allocator) ![]const u8 {
        var out: std.Io.Writer.Allocating = .init(a);
        for (self.lines.items) |l| if (l.level == .err) {
            if (out.written().len > 0) try out.writer.writeAll("; ");
            try out.writer.writeAll(l.text);
        };
        return out.written();
    }
};

// ---------------------------------------------------------------------------
// Reading the checkpoint
// ---------------------------------------------------------------------------

const SmallFile = struct { name: []const u8, bytes: []const u8 };

const Shard = struct {
    name: []const u8,
    /// The 8-byte length and the JSON header, as stored.
    head: []const u8,
    /// Bytes of tensor data after the header.
    data_len: u64,
};

const Tensor = struct {
    name: []const u8,
    dtype: []const u8,
    shape: []const usize,

    fn is(self: Tensor, shape: []const usize) bool {
        return std.mem.eql(usize, self.shape, shape);
    }
};

const Checkpoint = struct {
    source: []const u8,
    small: []const SmallFile,
    shards: []const Shard,
    tensors: []const Tensor,

    fn file(self: *const Checkpoint, name: []const u8) ?[]const u8 {
        for (self.small) |f| if (std.mem.eql(u8, f.name, name)) return f.bytes;
        return null;
    }

    fn find(self: *const Checkpoint, name: []const u8) ?Tensor {
        for (self.tensors) |t| if (std.mem.eql(u8, t.name, name)) return t;
        return null;
    }
};

const small_names = [_][]const u8{
    "config.json",           "generation_config.json", "tokenizer.json",          "tokenizer_config.json", "tiktoken.model",
    "tokenizer.model",       "chat_template.jinja",    "special_tokens_map.json", "model.safetensors.index.json",
};

/// Shard file names from an index's `weight_map` (sorted, unique).
fn shardsFromIndex(a: Allocator, text: []const u8) ![]const []const u8 {
    const v = try std.json.parseFromSliceLeaky(std.json.Value, a, text, .{});
    const wm = if (v == .object) v.object.get("weight_map") else null;
    if (wm == null or wm.? != .object) return error.InvalidIndex;
    var names: std.ArrayList([]const u8) = .empty;
    var it = wm.?.object.iterator();
    while (it.next()) |kv| {
        if (kv.value_ptr.* != .string) continue;
        const n = kv.value_ptr.string;
        for (names.items) |x| {
            if (std.mem.eql(u8, x, n)) break;
        } else try names.append(a, n);
    }
    sortStrings(names.items);
    return names.items;
}

fn sortStrings(items: [][]const u8) void {
    std.mem.sort([]const u8, items, {}, struct {
        fn lt(_: void, x: []const u8, y: []const u8) bool {
            return std.mem.lessThan(u8, x, y);
        }
    }.lt);
}

/// Parses a shard's header into tensors; returns the data length.
fn parseHeader(a: Allocator, header: []const u8, out: *std.ArrayList(Tensor)) !u64 {
    const v = try std.json.parseFromSliceLeaky(std.json.Value, a, header, .{});
    if (v != .object) return error.InvalidSafetensors;
    var end: u64 = 0;
    var it = v.object.iterator();
    while (it.next()) |kv| {
        if (std.mem.eql(u8, kv.key_ptr.*, "__metadata__")) continue;
        const t = kv.value_ptr.*;
        if (t != .object) return error.InvalidSafetensors;
        const dt = t.object.get("dtype") orelse return error.InvalidSafetensors;
        const sh = t.object.get("shape") orelse return error.InvalidSafetensors;
        const offs = t.object.get("data_offsets") orelse return error.InvalidSafetensors;
        if (dt != .string or sh != .array or offs != .array or offs.array.items.len != 2) return error.InvalidSafetensors;
        const shape = try a.alloc(usize, sh.array.items.len);
        for (sh.array.items, shape) |d, *s| s.* = if (d == .integer and d.integer >= 0) @intCast(d.integer) else return error.InvalidSafetensors;
        if (offs.array.items[1] != .integer) return error.InvalidSafetensors;
        end = @max(end, @as(u64, @intCast(@max(0, offs.array.items[1].integer))));
        try out.append(a, .{ .name = kv.key_ptr.*, .dtype = dt.string, .shape = shape });
    }
    return end;
}

fn addShard(a: Allocator, name: []const u8, head: []const u8, shards: *std.ArrayList(Shard), tensors: *std.ArrayList(Tensor)) !void {
    const n = std.mem.readInt(u64, head[0..8], .little);
    const data_len = try parseHeader(a, head[8..][0..@intCast(n)], tensors);
    try shards.append(a, .{ .name = name, .head = head, .data_len = data_len });
}

/// A checkpoint in a local directory: the small files and the shard headers.
fn readLocal(ctx: Ctx, path: []const u8) !Checkpoint {
    const a = ctx.arena;
    const io = ctx.io;
    var dir = try Io.Dir.cwd().openDir(io, path, .{ .iterate = true });
    defer dir.close(io);
    var small: std.ArrayList(SmallFile) = .empty;
    for (small_names) |n| {
        const bytes = dir.readFileAlloc(io, n, a, .limited(1 << 30)) catch continue;
        try small.append(a, .{ .name = n, .bytes = bytes });
    }
    var names: std.ArrayList([]const u8) = .empty;
    var it = dir.iterate();
    while (try it.next(io)) |e| {
        if (std.mem.endsWith(u8, e.name, ".safetensors") and !std.mem.startsWith(u8, e.name, "consolidated")) try names.append(a, try a.dupe(u8, e.name));
    }
    sortStrings(names.items);
    var shards: std.ArrayList(Shard) = .empty;
    var tensors: std.ArrayList(Tensor) = .empty;
    for (names.items) |n| {
        var f = try dir.openFile(io, n, .{});
        defer f.close(io);
        var len_buf: [8]u8 = undefined;
        if (try f.readPositionalAll(io, &len_buf, 0) != 8) return error.InvalidSafetensors;
        const hl = std.mem.readInt(u64, &len_buf, .little);
        if (hl == 0 or hl > 1 << 30) return error.InvalidSafetensors;
        const head = try a.alloc(u8, 8 + @as(usize, @intCast(hl)));
        if (try f.readPositionalAll(io, head, 0) != head.len) return error.InvalidSafetensors;
        try addShard(a, n, head, &shards, &tensors);
    }
    return .{ .source = path, .small = small.items, .shards = shards.items, .tensors = tensors.items };
}

/// A checkpoint on the Hub (or any server of model files): the small files,
/// then each shard's header through two range requests.
fn readRemote(ctx: Ctx, model: []const u8, out: *Io.Writer) !Checkpoint {
    const a = ctx.arena;
    const http = ctx.http;
    const rev = ctx.settings.model_commit orelse "main";
    var base: []const u8 = undefined;
    var listed: ?[]const []const u8 = null;
    if (std.mem.startsWith(u8, model, "http://") or std.mem.startsWith(u8, model, "https://")) {
        base = if (std.mem.endsWith(u8, model, "/")) model else try std.fmt.allocPrint(a, "{s}/", .{model});
    } else {
        const id = remote.hubId(model);
        if (std.mem.indexOfScalar(u8, id, '/') == null) {
            std.log.err("not a local directory, and not a Hub id (owner/name): {s}", .{model});
            return error.ModelNotFound;
        }
        base = try std.fmt.allocPrint(a, "https://huggingface.co/{s}/resolve/{s}/", .{ id, rev });
        // The repository listing, so that absent files are not requested.
        const api = try std.fmt.allocPrint(a, "https://huggingface.co/api/models/{s}/revision/{s}", .{ id, rev });
        const text = http.get(api) catch |err| switch (err) {
            error.NotFound => {
                std.log.err("{s} is not on the Hub (revision {s})", .{ id, rev });
                return error.ModelNotFound;
            },
            error.Forbidden => {
                std.log.err("access to {s} denied: a gated model needs HF_TOKEN", .{id});
                return error.ModelNotFound;
            },
            else => return err,
        };
        const v = try std.json.parseFromSliceLeaky(std.json.Value, a, text, .{});
        var names: std.ArrayList([]const u8) = .empty;
        if (v == .object) if (v.object.get("siblings")) |sib| if (sib == .array) for (sib.array.items) |s| {
            if (s != .object) continue;
            const f = s.object.get("rfilename") orelse continue;
            if (f == .string) try names.append(a, f.string);
        };
        listed = names.items;
    }
    try out.print("* Reading the config, tokenizer and safetensors headers of {s}\n", .{base});
    try out.flush();
    var small: std.ArrayList(SmallFile) = .empty;
    for (small_names) |n| {
        if (listed) |l| if (!contains(l, n)) continue;
        const url = try std.fmt.allocPrint(a, "{s}{s}", .{ base, n });
        const bytes = http.get(url) catch |err| switch (err) {
            error.NotFound => continue,
            else => return err,
        };
        try small.append(a, .{ .name = n, .bytes = try a.dupe(u8, bytes) });
        ctx.gpa.free(bytes);
    }
    var shard_names: []const []const u8 = &.{};
    for (small.items) |f| if (std.mem.eql(u8, f.name, "model.safetensors.index.json")) {
        shard_names = try shardsFromIndex(a, f.bytes);
    };
    if (shard_names.len == 0) {
        var names: std.ArrayList([]const u8) = .empty;
        if (listed) |l| {
            for (l) |n| if (std.mem.endsWith(u8, n, ".safetensors") and std.mem.indexOfScalar(u8, n, '/') == null and !std.mem.startsWith(u8, n, "consolidated")) try names.append(a, n);
        } else try names.append(a, "model.safetensors");
        sortStrings(names.items);
        shard_names = names.items;
    }
    var shards: std.ArrayList(Shard) = .empty;
    var tensors: std.ArrayList(Tensor) = .empty;
    for (shard_names) |n| {
        const url = try std.fmt.allocPrint(a, "{s}{s}", .{ base, n });
        const len_bytes = try http.getRange(url, 0, 7);
        defer ctx.gpa.free(len_bytes);
        if (len_bytes.len != 8) return error.InvalidSafetensors;
        const hl = std.mem.readInt(u64, len_bytes[0..8], .little);
        if (hl == 0 or hl > 1 << 30) return error.InvalidSafetensors;
        const header = try http.getRange(url, 8, 8 + hl - 1);
        defer ctx.gpa.free(header);
        const head = try a.alloc(u8, 8 + header.len);
        @memcpy(head[0..8], len_bytes[0..8]);
        @memcpy(head[8..], header);
        try addShard(a, n, head, &shards, &tensors);
    }
    return .{ .source = base, .small = small.items, .shards = shards.items, .tensors = tensors.items };
}

fn contains(list: []const []const u8, s: []const u8) bool {
    for (list) |x| if (std.mem.eql(u8, x, s)) return true;
    return false;
}

/// Writes the shape-only copy into `dir`: the small files (config.json as
/// given) and every shard as a sparse file.
fn writeCopy(ctx: Ctx, ck: *const Checkpoint, dir_path: []const u8) !void {
    const io = ctx.io;
    const cwd = Io.Dir.cwd();
    cwd.deleteTree(io, dir_path) catch {};
    try cwd.createDirPath(io, dir_path);
    var dir = try cwd.openDir(io, dir_path, .{});
    defer dir.close(io);
    for (ck.small) |f| try dir.writeFile(io, .{ .sub_path = f.name, .data = f.bytes });
    for (ck.shards) |s| {
        var f = try dir.createFile(io, s.name, .{});
        defer f.close(io);
        try f.writePositionalAll(io, s.head, 0);
        try f.setLength(io, s.head.len + s.data_len);
    }
}

// ---------------------------------------------------------------------------
// Trial loads
// ---------------------------------------------------------------------------

const Trial = struct {
    family: *const Arch,
    /// The family as loaded (with the overrides applied).
    used: *const Arch,
    /// Templates of `used` that name no tensor of the checkpoint (a tighter
    /// fit has fewer), and the tie-breaker of `affinity`.
    absent: usize = 0,
    affinity: f64 = 0,
    /// Name overrides tried on top of the family (`names` field -> template).
    overrides: []const Override,
    ok: bool,
    /// What the loader said when it refused (empty when it loaded).
    why: []const u8,
    missing: ?[]const u8,
    /// Checkpoint tensors the loader never looked up.
    unread: []const []const u8,
};

const Override = struct { field: []const u8, value: []const u8, why: []const u8 };

/// config.json with the model_type (and a wrapper's text_config model_type) replaced.
fn retype(a: Allocator, config_text: []const u8, model_type: []const u8) ![]const u8 {
    const v = try std.json.parseFromSliceLeaky(std.json.Value, a, try arch.sanitizeJson(a, config_text), .{});
    if (v != .object) return error.InvalidConfig;
    var obj = try v.object.clone(a);
    try obj.put(a, "model_type", .{ .string = model_type });
    _ = obj.orderedRemove("architectures");
    if (obj.get("text_config")) |tc| if (tc == .object) {
        var inner = try tc.object.clone(a);
        try inner.put(a, "model_type", .{ .string = model_type });
        try obj.put(a, "text_config", .{ .object = inner });
    };
    return std.json.Stringify.valueAlloc(a, std.json.Value{ .object = obj }, .{ .whitespace = .indent_2 });
}

/// A family `base`d on `f` under `name`, with `overrides` applied, as Lua source.
fn familySource(a: Allocator, name: []const u8, f: *const Arch, overrides: []const Override) ![]const u8 {
    var out: std.Io.Writer.Allocating = .init(a);
    const w = &out.writer;
    try w.print("return {{\n  model_type = \"{s}\",\n  base = \"{s}\",\n", .{ name, f.model_type });
    if (overrides.len > 0) {
        try w.writeAll("  names = {\n");
        for (overrides) |o| try w.print("    {s} = {s},\n", .{ o.field, o.value });
        try w.writeAll("  },\n");
    }
    try w.writeAll("}\n");
    return out.written();
}

/// Loads the shape-only copy as `f` (with `overrides`) and records what the loader did.
fn tryLoad(ctx: Ctx, ck: *const Checkpoint, dir_path: []const u8, f: *const Arch, overrides: []const Override, trial_type: []const u8) !Trial {
    const a = ctx.arena;
    var result: Trial = .{ .family = f, .used = f, .overrides = overrides, .ok = false, .why = "", .missing = null, .unread = &.{} };
    var family = f;
    if (overrides.len > 0) {
        var diag: models.Diagnostic = .{};
        const src = try familySource(a, trial_type, f, overrides);
        const got = models.parseSource(src, "trial.lua", &diag) catch {
            result.why = diag.message orelse "the overrides do not load";
            return result;
        };
        family = got[0];
        result.used = family;
        _ = try models.loadSource(src, "trial.lua", &diag);
    }
    const cfg_text = try retype(a, ck.file("config.json").?, if (overrides.len > 0) trial_type else f.model_type);
    {
        var dir = try Io.Dir.cwd().openDir(ctx.io, dir_path, .{});
        defer dir.close(ctx.io);
        try dir.writeFile(ctx.io, .{ .sub_path = "config.json", .data = cfg_text });
    }
    var cap: Capture = .{ .arena = a };
    var log: safetensors.LookupLog = .{ .arena = .init(ctx.gpa) };
    defer log.arena.deinit();
    capture = &cap;
    safetensors.lookup_log = &log;
    const loaded = model_mod.Model.loadWithOptions(ctx.gpa, ctx.io, ctx.pool, dir_path, .{ .store = .streamed, .prefetch = false, .expert_cache = 0 });
    safetensors.lookup_log = null;
    capture = null;
    const model = loaded catch |err| {
        result.why = try cap.errors(a);
        if (result.why.len == 0) result.why = @errorName(err);
        for (cap.lines.items) |l| if (std.mem.startsWith(u8, l.text, "missing tensor: ")) {
            result.missing = l.text["missing tensor: ".len..];
        };
        return result;
    };
    defer model.deinit();
    result.ok = true;
    var unread: std.ArrayList([]const u8) = .empty;
    for (model.files) |file| {
        var it = file.tensors.iterator();
        while (it.next()) |kv| if (!log.names.contains(kv.key_ptr.*)) try unread.append(a, try a.dupe(u8, kv.key_ptr.*));
    }
    sortStrings(unread.items);
    result.unread = unread.items;
    return result;
}

// ---------------------------------------------------------------------------
// Tensor names and roles
// ---------------------------------------------------------------------------

/// Tensors that are not part of the text model a definition describes.
fn outsideTextModel(name: []const u8) ?[]const u8 {
    const parts = [_]struct { []const u8, []const u8 }{
        .{ "vision", "vision tower" },      .{ "visual", "vision tower" },         .{ "image", "vision tower" },
        .{ "vit.", "vision tower" },        .{ "patch_embed", "vision tower" },    .{ "multi_modal_projector", "multimodal projector" },
        .{ "mm_projector", "multimodal projector" }, .{ "audio", "audio tower" }, .{ "speech", "audio tower" },
        .{ "talker", "speech decoder" },   .{ "token2wav", "speech decoder" },    .{ "mtp", "multi-token prediction head" },
        .{ "nextn", "multi-token prediction head" }, .{ "rotary_emb.inv_freq", "precomputed RoPE table" },
    };
    for (parts) |p| if (std.mem.indexOf(u8, name, p[0]) != null) return p[1];
    return null;
}

/// `name` with every numeric path segment replaced by `{n}`, and the numbers.
fn pattern(a: Allocator, name: []const u8) !struct { pat: []const u8, nums: []const usize } {
    var out: std.ArrayList(u8) = .empty;
    var nums: std.ArrayList(usize) = .empty;
    var it = std.mem.splitScalar(u8, name, '.');
    var first = true;
    while (it.next()) |seg| {
        if (!first) try out.append(a, '.');
        first = false;
        if (seg.len > 0 and std.ascii.isDigit(seg[0]) and (std.fmt.parseInt(usize, seg, 10) catch null) != null) {
            try out.appendSlice(a, "{n}");
            try nums.append(a, std.fmt.parseInt(usize, seg, 10) catch 0);
        } else try out.appendSlice(a, seg);
    }
    return .{ .pat = out.items, .nums = nums.items };
}

const Group = struct { pat: []const u8, count: usize, first: []const u8, shape: []const usize, dtype: []const u8, why: ?[]const u8 };

/// Groups tensor names by pattern (layer and expert indices folded).
fn groupNames(a: Allocator, ck: *const Checkpoint, names: []const []const u8) ![]Group {
    var groups: std.ArrayList(Group) = .empty;
    for (names) |n| {
        const p = try pattern(a, n);
        for (groups.items) |*g| {
            if (std.mem.eql(u8, g.pat, p.pat)) {
                g.count += 1;
                break;
            }
        } else {
            const t = ck.find(n);
            try groups.append(a, .{ .pat = p.pat, .count = 1, .first = n, .shape = if (t) |x| x.shape else &.{}, .dtype = if (t) |x| x.dtype else "?", .why = outsideTextModel(n) });
        }
    }
    return groups.items;
}

fn fmtShape(a: Allocator, shape: []const usize) ![]const u8 {
    var out: std.Io.Writer.Allocating = .init(a);
    try out.writer.writeByte('[');
    for (shape, 0..) |d, i| {
        if (i > 0) try out.writer.writeAll(", ");
        try out.writer.print("{d}", .{d});
    }
    try out.writer.writeByte(']');
    return out.written();
}

/// Where a tensor name sits: the model prefix, the layer (and its prefix),
/// the expert, and the rest relative to them.
const Place = struct {
    prefix: []const u8,
    layer: ?usize = null,
    layer_prefix: []const u8 = "",
    expert: ?usize = null,
    expert_prefix: []const u8 = "",
    rel: []const u8,
};

fn expand(a: Allocator, template: []const u8, prefix: []const u8, i: ?usize, e: ?usize) ![]const u8 {
    var out: std.ArrayList(u8) = .empty;
    var k: usize = 0;
    while (k < template.len) {
        if (std.mem.startsWith(u8, template[k..], "{p}")) {
            try out.appendSlice(a, prefix);
            k += 3;
        } else if (std.mem.startsWith(u8, template[k..], "{i}")) {
            try out.print(a, "{d}", .{i orelse 0});
            k += 3;
        } else if (std.mem.startsWith(u8, template[k..], "{e}")) {
            try out.print(a, "{d}", .{e orelse 0});
            k += 3;
        } else {
            try out.append(a, template[k]);
            k += 1;
        }
    }
    return out.items;
}

/// The model prefix the loader would pick: the first of `prefixes` under
/// which the embedding exists (else the first one).
fn modelPrefix(a: Allocator, ck: *const Checkpoint, names: *const Names) ![]const u8 {
    for (names.prefixes) |p| if (ck.find(try expand(a, names.embed, p, null, null)) != null) return p;
    for (names.prefixes) |p| {
        const lp = try expand(a, names.layer, p, 0, null);
        for (ck.tensors) |t| if (std.mem.startsWith(u8, t.name, lp)) return p;
    }
    return names.prefixes[0];
}

fn locate(a: Allocator, names: *const Names, prefix: []const u8, layers: usize, experts: usize, name: []const u8) !Place {
    var place: Place = .{ .prefix = prefix, .rel = name };
    for (0..layers) |i| {
        const lp = try expand(a, names.layer, prefix, i, null);
        if (!std.mem.startsWith(u8, name, lp)) continue;
        place.layer = i;
        place.layer_prefix = lp;
        place.rel = name[lp.len..];
        for (0..experts) |e| {
            const ep = try expand(a, names.expert, prefix, i, e);
            if (std.mem.startsWith(u8, place.rel, ep)) {
                place.expert = e;
                place.expert_prefix = ep;
                place.rel = place.rel[ep.len..];
                break;
            }
        }
        return place;
    }
    return place;
}

/// The `names` field whose template produced `rel` at `place` (a model-level
/// template is matched with its prefix expanded).
fn fieldOf(a: Allocator, names: *const Names, place: Place, full: []const u8) !?[]const u8 {
    inline for (@typeInfo(Names).@"struct".fields) |f| {
        const skip = comptime std.mem.eql(u8, f.name, "prefixes") or std.mem.eql(u8, f.name, "layer") or std.mem.eql(u8, f.name, "expert");
        if (!skip) {
            const v = @field(names.*, f.name);
            const T = @TypeOf(v);
            if (T == []const u8 or T == ?[]const u8) {
                if (@as(?[]const u8, v)) |t| if (try templateIs(a, t, place, full)) return f.name;
            } else if (T == []const []const u8) {
                for (v) |t| if (try templateIs(a, t, place, full)) return f.name;
            }
        }
    }
    return null;
}

fn templateIs(a: Allocator, t: []const u8, place: Place, full: []const u8) !bool {
    if (std.mem.indexOf(u8, t, "{p}") != null) return std.mem.eql(u8, try expand(a, t, place.prefix, place.layer, place.expert), full);
    return std.mem.eql(u8, t, place.rel);
}

/// Words that name a slot in the checkpoints ditch has met, for telling
/// apart two unread tensors of the same shape.
const role_words = [_]struct { field: []const u8, words: []const []const u8 }{
    .{ .field = "q", .words = &.{ "q_proj", "wq", "query", ".q." } },
    .{ .field = "k", .words = &.{ "k_proj", "wk", "key", ".k." } },
    .{ .field = "v", .words = &.{ "v_proj", "wv", "value", ".v." } },
    .{ .field = "o", .words = &.{ "o_proj", "wo", "out_proj", "dense", "c_proj", "proj" } },
    .{ .field = "gate", .words = &.{ "gate", "w1", "wi_0", "fc1" } },
    .{ .field = "up", .words = &.{ "up", "w3", "wi_1", "fc_in" } },
    .{ .field = "down", .words = &.{ "down", "w2", "wo", "fc2", "fc_out" } },
    .{ .field = "expert_gate", .words = &.{ "gate", "w1" } },
    .{ .field = "expert_up", .words = &.{ "up", "w3" } },
    .{ .field = "expert_down", .words = &.{ "down", "w2" } },
    .{ .field = "input_norm", .words = &.{ "input", "attn_norm", "attention_norm", "ln_1", "pre_attn", "norm1" } },
    .{ .field = "pre_ff_norm", .words = &.{ "post_attention", "ffn_norm", "mlp_norm", "ln_2", "pre_mlp", "norm2" } },
    .{ .field = "post_attn_norm", .words = &.{ "post_attn", "post_attention" } },
    .{ .field = "post_ff_norm", .words = &.{ "post_ff", "post_mlp", "post_feedforward" } },
    .{ .field = "q_norm", .words = &.{ "q_norm", "q_layernorm", "query_norm" } },
    .{ .field = "k_norm", .words = &.{ "k_norm", "k_layernorm", "key_norm" } },
    .{ .field = "router", .words = &.{ "gate", "router" } },
    .{ .field = "embed", .words = &.{ "embed", "wte", "tok" } },
    .{ .field = "final_norm", .words = &.{ "norm", "ln_f" } },
    .{ .field = "lm_head", .words = &.{ "lm_head", "output", "head" } },
};

fn roleScore(field: []const u8, name: []const u8) usize {
    for (role_words) |r| if (std.mem.eql(u8, r.field, field)) {
        var s: usize = 0;
        for (r.words) |w| {
            if (std.mem.indexOf(u8, name, w) != null) s += 1;
        }
        return s;
    };
    return 0;
}

/// The shape ditch expects in `field`, from the parsed configuration (null
/// when the slot has no simple expected shape).
fn expectedShape(a: Allocator, c: *const arch.Config, field: []const u8, li: usize) !?[]const usize {
    const H = c.hidden_size;
    const hd = if (li < c.layer_head_dim.len) c.layer_head_dim[li] else c.head_dim;
    const nh = if (li < c.layer_heads.len) c.layer_heads[li] else c.num_heads;
    const kvh = if (li < c.layer_kv_heads.len) c.layer_kv_heads[li] else c.num_kv_heads;
    const I = c.intermediate_size;
    const Im = c.moe_intermediate_size;
    const table = [_]struct { []const u8, [2]usize }{
        .{ "embed", .{ c.vocab_size, H } },    .{ "lm_head", .{ c.vocab_size, H } },
        .{ "q", .{ nh * hd, H } },             .{ "k", .{ kvh * hd, H } },
        .{ "v", .{ kvh * c.layerVDim(li), H } }, .{ "o", .{ H, nh * c.layerVDim(li) } },
        .{ "gate", .{ I, H } },                .{ "up", .{ I, H } },
        .{ "down", .{ H, I } },                .{ "gate_up", .{ 2 * I, H } },
        .{ "router", .{ c.num_experts, H } },  .{ "expert_gate", .{ Im, H } },
        .{ "expert_up", .{ Im, H } },          .{ "expert_down", .{ H, Im } },
    };
    for (table) |e| if (std.mem.eql(u8, e[0], field)) return try a.dupe(usize, &e[1]);
    const vectors = [_]struct { []const u8, usize }{
        .{ "final_norm", H }, .{ "input_norm", H }, .{ "pre_ff_norm", H }, .{ "post_attn_norm", H },
        .{ "post_ff_norm", H }, .{ "mlp_norm", H }, .{ "q_norm", hd }, .{ "k_norm", hd },
    };
    for (vectors) |e| if (std.mem.eql(u8, e[0], field)) return try a.dupe(usize, &.{e[1]});
    return null;
}

/// For a tensor the family needs and the checkpoint lacks, an unread tensor
/// of the same place with the expected shape (and, among several, the one
/// whose name says the role): the name override to try next.
fn proposeRename(a: Allocator, ck: *const Checkpoint, f: *const Arch, c: *const arch.Config, missing: []const u8, unread: []const []const u8) !?Override {
    const names = &f.names;
    const prefix = try modelPrefix(a, ck, names);
    const place = try locate(a, names, prefix, c.num_layers, c.num_experts, missing);
    const field = (try fieldOf(a, names, place, missing)) orelse return null;
    const want = try expectedShape(a, c, field, place.layer orelse 0);
    var best: ?[]const u8 = null;
    var best_score: usize = 0;
    var ties: usize = 0;
    for (unread) |n| {
        const p = try locate(a, names, prefix, c.num_layers, c.num_experts, n);
        if ((p.layer == null) != (place.layer == null) or (p.layer != null and p.layer.? != place.layer.?)) continue;
        if ((p.expert == null) != (place.expert == null)) continue;
        const t = ck.find(n) orelse continue;
        if (want) |w| if (!t.is(w)) continue;
        if (want == null and !std.mem.eql(u8, std.fs.path.extension(n), std.fs.path.extension(missing))) continue;
        const s = roleScore(field, p.rel) + 1;
        if (s > best_score) {
            best = n;
            best_score = s;
            ties = 1;
        } else if (s == best_score) ties += 1;
    }
    const chosen = best orelse return null;
    if (ties > 1) return null;
    const p = try locate(a, names, prefix, c.num_layers, c.num_experts, chosen);
    // The new template, in the form the field takes.
    var template: []const u8 = p.rel;
    if (place.layer == null) template = if (std.mem.startsWith(u8, chosen, prefix)) try std.fmt.allocPrint(a, "{{p}}{s}", .{chosen[prefix.len..]}) else chosen;
    const is_list = inline for (@typeInfo(Names).@"struct".fields) |nf| {
        if (std.mem.eql(u8, nf.name, field)) break nf.type == []const []const u8;
    } else false;
    const t = ck.find(chosen).?;
    const value = if (is_list) try std.fmt.allocPrint(a, "{{ \"{s}\" }}", .{template}) else try std.fmt.allocPrint(a, "\"{s}\"", .{template});
    const why = try std.fmt.allocPrint(a, "guess: `{s}` {s} is the {s} tensor of this place with the shape ditch expects in {s} (the base family names it {s})", .{
        chosen, try fmtShape(a, t.shape), if (ties == 1 and best_score > 1) "one" else "only unread", field, missing,
    });
    return .{ .field = field, .value = value, .why = why };
}

// ---------------------------------------------------------------------------
// Candidates
// ---------------------------------------------------------------------------

const Candidate = struct {
    family: *const Arch,
    score: f64,
    cfg: arch.Config,
    /// Fields and tensor names that differ from the schema defaults: among
    /// equally covered families the plainest is tried first.
    plainness: usize,
};

fn plainness(f: *const Arch) usize {
    @setEvalBranchQuota(20000);
    const default: Arch = .{ .model_type = "", .llama_cpp = null };
    var n: usize = 0;
    inline for (@typeInfo(Arch).@"struct".fields) |af| {
        const skip = comptime std.mem.eql(u8, af.name, "model_type") or std.mem.eql(u8, af.name, "aliases") or std.mem.eql(u8, af.name, "llama_cpp") or
            std.mem.eql(u8, af.name, "chat") or std.mem.eql(u8, af.name, "verified") or std.mem.eql(u8, af.name, "notes") or std.mem.eql(u8, af.name, "names") or std.mem.eql(u8, af.name, "script");
        if (!skip and !std.meta.eql(@field(f.*, af.name), @field(default, af.name))) n += 1;
    }
    // A hook or a config function reads keys of its own.
    if (f.extra != null) n += 3;
    if (f.script != null) n += 3;
    const dn: Names = .{};
    inline for (@typeInfo(Names).@"struct".fields) |nf| {
        const v = @field(f.names, nf.name);
        const d = @field(dn, nf.name);
        const T = @TypeOf(v);
        if (T == []const u8) {
            if (!std.mem.eql(u8, v, d)) n += 1;
        } else if (T == ?[]const u8) {
            if ((v == null) != (d == null) or (v != null and !std.mem.eql(u8, v.?, d.?))) n += 1;
        } else if (T == []const []const u8) {
            var same = v.len == d.len;
            if (same) for (v, d) |x, y| {
                same = same and std.mem.eql(u8, x, y);
            };
            if (!same) n += 1;
        } else if (!std.meta.eql(v, d)) n += 1;
    }
    return n;
}

/// Every template string of a family, relative to its layer or expert
/// prefix, for a first cheap ranking.
fn coverage(a: Allocator, ck: *const Checkpoint, f: *const Arch, c: *const arch.Config) !f64 {
    const prefix = try modelPrefix(a, ck, &f.names);
    var known: usize = 0;
    var total: usize = 0;
    for (ck.tensors) |t| {
        if (outsideTextModel(t.name) != null) continue;
        total += 1;
        const p = try locate(a, &f.names, prefix, @min(c.num_layers, 4096), @min(c.num_experts, 4096), t.name);
        const probe = if (std.mem.endsWith(u8, p.rel, ".bias")) try std.fmt.allocPrint(a, "{s}.weight", .{p.rel[0 .. p.rel.len - ".bias".len]}) else p.rel;
        var q = p;
        q.rel = probe;
        const full = if (std.mem.endsWith(u8, t.name, ".bias")) try std.fmt.allocPrint(a, "{s}.weight", .{t.name[0 .. t.name.len - ".bias".len]}) else t.name;
        if (try fieldOf(a, &f.names, q, full) != null) known += 1;
    }
    return if (total == 0) 0 else @as(f64, @floatFromInt(known)) / @as(f64, @floatFromInt(total));
}

/// A tie-breaker between families that read the same tensors: the one the
/// config's `architectures` class names (`Qwen2ForCausalLM` is `qwen2`),
/// then one whose name shares the model_type's stem (`qwen2_foo`, `qwen`).
fn affinity(a: Allocator, config_text: []const u8, own_type: []const u8, f: *const Arch) f64 {
    var bonus: f64 = 0;
    const v = std.json.parseFromSliceLeaky(std.json.Value, a, config_text, .{}) catch return 0;
    if (v == .object) if (v.object.get("architectures")) |archs| if (archs == .array and archs.array.items.len > 0 and archs.array.items[0] == .string) {
        const cls = archs.array.items[0].string;
        for ([_][]const u8{ "ForCausalLM", "LMHeadModel", "ForConditionalGeneration" }) |suffix| {
            if (!std.mem.endsWith(u8, cls, suffix)) continue;
            const lower = std.ascii.allocLowerString(a, cls[0 .. cls.len - suffix.len]) catch return 0;
            if (std.mem.eql(u8, lower, f.model_type) or contains(f.aliases, lower)) bonus += 0.2;
        }
    };
    const stem = std.mem.trimEnd(u8, own_type[0 .. std.mem.indexOfAny(u8, own_type, "_-") orelse own_type.len], "0123456789.");
    if (stem.len >= 3 and std.mem.startsWith(u8, f.model_type, stem)) bonus += 0.1;
    if (std.mem.startsWith(u8, own_type, f.model_type)) bonus += 0.05;
    return bonus;
}

// ---------------------------------------------------------------------------
// Config keys
// ---------------------------------------------------------------------------

/// Keys that do not describe the computation (generation defaults, token
/// ids, training settings, bookkeeping).
const ignorable_keys = [_][]const u8{
    "architectures",     "auto_map",               "model_type",          "torch_dtype",          "dtype",                 "transformers_version",
    "_name_or_path",     "bos_token_id",           "eos_token_id",        "pad_token_id",         "sep_token_id",          "decoder_start_token_id",
    "use_cache",         "initializer_range",      "attention_dropout",   "hidden_dropout",       "dropout",               "embd_pdrop",
    "resid_pdrop",       "attn_pdrop",             "summary_type",        "summary_use_proj",     "summary_activation",    "summary_proj_to_labels",
    "summary_first_dropout", "output_attentions",  "output_hidden_states", "return_dict",         "pretraining_tp",        "is_decoder",
    "is_encoder_decoder", "tokenizer_class",       "router_aux_loss_coef", "output_router_logits", "router_z_loss_coef",   "aux_loss_alpha",
    "seq_aux",           "use_flash_attn",         "_attn_implementation", "attn_implementation", "image_token_id",       "video_token_id",
    "vision_start_token_id", "vision_end_token_id", "vision_token_id",    "chunk_size_feed_forward", "gradient_checkpointing", "num_nextn_predict_layers",
    "mtp_num_layers",    "task_specific_params",   "id2label",            "label2id",             "problem_type",          "use_return_dict",
    "ffn_dropout",       "hidden_dropout_prob",    "attention_probs_dropout_prob", "classifier_dropout", "layerdrop",       "mlp_dropout",
    "quantization_config", "vision_config",        "audio_config",        "text_config",          "thinker_config",        "talker_config",
};

const KeyReport = struct { unread: []const []const u8, read: usize };

fn dumpConfig(a: Allocator, text: []const u8) []const u8 {
    const cfg = arch.parseConfig(a, text) catch |err| return @errorName(err);
    var out: std.Io.Writer.Allocating = .init(a);
    models.dump(arch.Config, &out.writer, cfg, "") catch return "?";
    return out.written();
}

/// The keys of the (text) config that no part of the definition reads:
/// neither removing them nor changing their value changes the parse.
fn classifyKeys(a: Allocator, config_text: []const u8) !KeyReport {
    var quiet: Capture = .{ .arena = a };
    capture = &quiet;
    defer capture = null;
    const root = try std.json.parseFromSliceLeaky(std.json.Value, a, try arch.sanitizeJson(a, config_text), .{});
    if (root != .object) return error.InvalidConfig;
    const nested = if (root.object.get("text_config")) |tc| (if (tc == .object) tc.object else null) else null;
    const obj = nested orelse root.object;
    const base = dumpConfig(a, config_text);
    var unread: std.ArrayList([]const u8) = .empty;
    var read: usize = 0;
    var it = obj.iterator();
    while (it.next()) |kv| {
        const key = kv.key_ptr.*;
        if (contains(&ignorable_keys, key)) continue;
        var changed = false;
        for (0..1 + perturbations.len) |variant| {
            if (changed) break;
            var o = try obj.clone(a);
            if (variant == 0) {
                _ = o.orderedRemove(key);
            } else try o.put(a, key, perturb(a, kv.value_ptr.*, variant - 1));
            var r = try root.object.clone(a);
            if (nested != null) try r.put(a, "text_config", .{ .object = o }) else r = o;
            const text = try std.json.Stringify.valueAlloc(a, std.json.Value{ .object = r }, .{});
            if (!std.mem.eql(u8, dumpConfig(a, text), base)) changed = true;
        }
        if (changed) read += 1 else try unread.append(a, key);
    }
    sortStrings(unread.items);
    return .{ .unread = unread.items, .read = read };
}

/// String values a key may take that ditch knows (activations, rope and
/// layer types, router scores): a changed value is only noticed when the
/// parser accepts it.
const perturbations = [_][]const u8{ "gelu", "relu", "sigmoid", "yarn", "linear", "layer_norm", "ditch_probe_value" };

fn perturb(a: Allocator, v: std.json.Value, k: usize) std.json.Value {
    return switch (v) {
        .integer => |n| .{ .integer = if (n == 0) 3 + @as(i64, @intCast(k)) else n * 2 + 1 + @as(i64, @intCast(k)) },
        .float => |f| .{ .float = f * 1.5 + 0.25 + @as(f64, @floatFromInt(k)) },
        .bool => |b| .{ .bool = !b },
        .string => |s| .{ .string = if (std.mem.eql(u8, s, perturbations[k])) "ditch_probe_value" else perturbations[k] },
        .array => |arr| blk: {
            var copy = std.json.Array.init(a);
            if (arr.items.len > 1) copy.appendSlice(arr.items[0 .. arr.items.len - 1]) catch {};
            break :blk .{ .array = copy };
        },
        .object => .{ .object = .empty },
        else => .{ .integer = 1 },
    };
}

// ---------------------------------------------------------------------------
// The command
// ---------------------------------------------------------------------------

fn configType(a: Allocator, text: []const u8) !struct { top: ?[]const u8, text: ?[]const u8 } {
    const v = try std.json.parseFromSliceLeaky(std.json.Value, a, try arch.sanitizeJson(a, text), .{});
    if (v != .object) return error.InvalidConfig;
    const top = arch.getStr(v.object, "model_type");
    var inner: ?[]const u8 = null;
    if (v.object.get("text_config")) |tc| if (tc == .object) {
        inner = arch.getStr(tc.object, "model_type");
    };
    return .{ .top = top, .text = inner };
}

fn userModelsDir(ctx: Ctx) ![]const u8 {
    if (ctx.settings.models_dir) |d| return d;
    const base = (try config.configDir(ctx.arena, ctx.env)) orelse {
        std.log.err("no configuration directory (set XDG_CONFIG_HOME or HOME, or pass --models-dir)", .{});
        return error.NoConfigDir;
    };
    return std.fs.path.join(ctx.arena, &.{ base, "models" });
}

pub fn run(ctx: Ctx) !u8 {
    const a = ctx.arena;
    const io = ctx.io;
    const out = ctx.out;
    const model = ctx.settings.model;
    if (model.len == 0) {
        try out.writeAll("Usage: ditch add-model MODEL [--models-dir DIR] [--force] [--dry-run]\n\nMODEL is a Hub id (owner/name), hf://owner/name, an http(s) URL of the model files or a local directory.\n");
        return 2;
    }
    if (std.mem.endsWith(u8, model, ".gguf")) {
        std.log.err("add-model reads safetensors checkpoints: a GGUF file is loaded through ditch's GGUF path, which knows its architectures by name (see docs/models.md, GGUF per family)", .{});
        return 2;
    }

    // 1. Read.
    const ck = if (hf.isLocalDir(io, model)) try readLocal(ctx, model) else try readRemote(ctx, model, out);
    const config_text = ck.file("config.json") orelse {
        std.log.err("{s} has no config.json", .{model});
        return 1;
    };
    if (ck.tensors.len == 0) {
        std.log.err("{s} has no safetensors weights (add-model needs their headers)", .{model});
        return 1;
    }
    const types = try configType(a, config_text);
    const own_type = types.text orelse types.top orelse {
        std.log.err("config.json names no model_type", .{});
        return 1;
    };
    try out.print("* config.json: model_type {s}{s}{s}; {d} tensors in {d} safetensors file(s)\n", .{ own_type, if (types.text != null) " (text config of " else "", if (types.text != null) try std.fmt.allocPrint(a, "{s})", .{types.top orelse "?"}) else "", ck.tensors.len, ck.shards.len });
    const known = models.lookup(own_type) orelse if (types.top) |t| models.lookup(t) else null;
    if (known) |k| try out.print("* {s} is already defined ({s}, {s}): drafting without it, to compare the draft with it\n", .{ own_type, k.model_type, models.origin(k) });

    // 2. The shape-only copy.
    var name_buf: std.ArrayList(u8) = .empty;
    for (model) |ch| try name_buf.append(a, if (std.ascii.isAlphanumeric(ch) or ch == '.' or ch == '-') ch else '_');
    const work = try std.fs.path.join(a, &.{ ctx.cache_root, "add-model", name_buf.items });
    try writeCopy(ctx, &ck, work);
    try out.print("* Shape-only copy (headers and holes, no weights) in {s}\n", .{work});
    try out.flush();

    // 3. Match.
    const chosen = (try match(ctx, &ck, work, own_type, config_text, known)) orelse {
        std.log.err("no known family reads this config.json (every one refused it); a new family needs a definition written by hand (docs/models.md)", .{});
        return 1;
    };

    // 4. Config keys, as the chosen family reads them.
    var keys: KeyReport = .{ .unread = &.{}, .read = 0 };
    if (chosen.ok) keys = classifyKeys(a, try retype(a, config_text, chosen.family.model_type)) catch keys;

    // 5. The draft.
    const draft = try writeDraft(ctx, &ck, own_type, chosen, keys, known);
    const dir_path = try userModelsDir(ctx);
    const file_name = try std.fmt.allocPrint(a, "{s}.lua", .{own_type});
    const path = try std.fs.path.join(a, &.{ dir_path, file_name });
    try out.writeAll("\n");
    try out.flush();
    try ctx.result.writeAll(draft);
    try ctx.result.flush();
    const exists = if (Io.Dir.cwd().access(io, path, .{})) |_| true else |_| false;
    if ((exists or known != null) and !ctx.settings.force) {
        try out.print("\n{s} not written: {s}; --force writes it anyway\n", .{ path, if (exists) "the file exists" else "the model_type is already defined, and the draft would shadow that definition" });
        return if (chosen.ok) 0 else 1;
    }
    try Io.Dir.cwd().createDirPath(io, dir_path);
    try Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = draft });
    try out.print("\nWrote {s}\n", .{path});

    // The draft loads as a definition, and parses as the family it names.
    var diag: models.Diagnostic = .{};
    _ = models.loadSource(draft, path, &diag) catch |err| {
        std.log.err("the draft does not load: {s}", .{diag.message orelse @errorName(err)});
        return 1;
    };
    if (known) |k| try compareWithKnown(ctx, config_text, own_type, k);
    if (!chosen.ok) {
        try out.writeAll("The draft does not load this checkpoint yet (see its comments); edit it, then run ditch verify.\n");
        return 1;
    }
    if (ctx.settings.dry_run) {
        try out.print("Dry run: not checking; run ditch verify {s} to compare it with the official implementation.\n", .{model});
        return 0;
    }
    try out.print("\nChecking the draft: ditch verify {s}\n", .{model});
    try out.flush();
    try ctx.env.put("DITCH_MODELS_DIR", dir_path);
    const verify_args = [_][]const u8{model};
    return verify.run(ctx.gpa, ctx.arena, io, ctx.env, &verify_args, out, ctx.result);
}

/// Ranks the known families by how well they read the checkpoint (see
/// the module comment) and returns the best trial load, or null when no
/// family reads its config.json. `known` (the definition ditch already has
/// for this model_type, if any) is left out, so that the draft is made
/// without it and can be compared with it.
fn match(ctx: Ctx, ck: *const Checkpoint, work: []const u8, own_type: []const u8, config_text: []const u8, known: ?*const Arch) !?Trial {
    const a = ctx.arena;
    const out = ctx.out;
    // Rank the families whose reading of config.json succeeds.
    var candidates: std.ArrayList(Candidate) = .empty;
    var quiet: Capture = .{ .arena = a };
    capture = &quiet;
    for (models.families()) |f| {
        if (known != null and f == known.?) continue;
        if (f.inherits != null or std.mem.startsWith(u8, f.model_type, "add_model_trial")) continue;
        for (candidates.items) |cand| {
            if (cand.family == f) break;
        } else {
            const text = try retype(a, config_text, f.model_type);
            const cfg = arch.parseConfig(a, text) catch continue;
            const score = try coverage(a, ck, f, &cfg) + affinity(a, config_text, own_type, f);
            try candidates.append(a, .{ .family = f, .score = score, .cfg = cfg, .plainness = plainness(f) });
        }
    }
    capture = null;
    std.mem.sort(Candidate, candidates.items, {}, struct {
        fn gt(_: void, x: Candidate, y: Candidate) bool {
            if (x.score != y.score) return x.score > y.score;
            return x.plainness < y.plainness;
        }
    }.gt);
    if (candidates.items.len == 0) return null;
    try out.print("* {d} families read config.json; by tensor-name coverage:", .{candidates.items.len});
    for (candidates.items[0..@min(5, candidates.items.len)]) |cand| try out.print(" {s} {d:.0}%", .{ cand.family.model_type, @min(cand.score, 1.0) * 100 });
    try out.writeAll("\n");
    try out.flush();

    // Trial loads: the best-covered families as they are, then with the
    // renames the loader's complaints suggest. Among those that read every
    // tensor, the fewest renames, then the tightest fit, then the affinity.
    var best: ?Trial = null;
    var trial_no: usize = 0;
    for (candidates.items[0..@min(10, candidates.items.len)]) |cand| {
        var overrides: std.ArrayList(Override) = .empty;
        var round: usize = 0;
        while (round < 24) : (round += 1) {
            trial_no += 1;
            const trial_type = try std.fmt.allocPrint(a, "add_model_trial_{d}", .{trial_no});
            const t = try tryLoad(ctx, ck, work, cand.family, overrides.items, trial_type);
            if (t.ok) {
                var trial = t;
                trial.absent = try absentTemplates(a, ck, t.used, &cand.cfg);
                trial.affinity = affinity(a, config_text, own_type, cand.family);
                try out.print("  {s}{s}: loads; {d} tensor(s) of the text model unread, {d} of its names absent\n", .{ cand.family.model_type, if (overrides.items.len > 0) " (renamed)" else "", countText(t.unread), trial.absent });
                if (best == null or better(trial, best.?)) best = trial;
                break;
            }
            const missing = t.missing orelse {
                try out.print("  {s}: refused: {s}\n", .{ cand.family.model_type, t.why });
                if (best == null) best = t;
                break;
            };
            const all = try allNames(a, ck);
            const prop = try proposeRename(a, ck, cand.family, &cand.cfg, missing, try unreadExcept(a, all, ck, cand.family, &cand.cfg, overrides.items));
            if (prop == null) {
                try out.print("  {s}: needs {s}, which the checkpoint lacks\n", .{ cand.family.model_type, missing });
                if (best == null) best = t;
                break;
            }
            try overrides.append(a, prop.?);
        }
    }
    return best;
}

fn better(x: Trial, y: Trial) bool {
    if (x.ok != y.ok) return x.ok;
    const xu = countText(x.unread);
    const yu = countText(y.unread);
    if (xu != yu) return xu < yu;
    if (x.overrides.len != y.overrides.len) return x.overrides.len < y.overrides.len;
    if (x.absent != y.absent) return x.absent < y.absent;
    if (x.affinity != y.affinity) return x.affinity > y.affinity;
    return x.family.verified and !y.family.verified;
}

/// How many of a family's tensor templates name nothing in the checkpoint
/// (checked at layer 0 and expert 0): a family with optional tensors the
/// checkpoint lacks (BitNet's sub-norms for a Llama checkpoint) fits less
/// tightly than one without them.
fn absentTemplates(a: Allocator, ck: *const Checkpoint, f: *const Arch, c: *const arch.Config) !usize {
    @setEvalBranchQuota(20000);
    const names = &f.names;
    const prefix = try modelPrefix(a, ck, names);
    const lp = try expand(a, names.layer, prefix, 0, null);
    const ep = try expand(a, names.expert, prefix, 0, 0);
    var absent: usize = 0;
    inline for (@typeInfo(Names).@"struct".fields) |nf| {
        const skip = comptime std.mem.eql(u8, nf.name, "prefixes") or std.mem.eql(u8, nf.name, "layer") or std.mem.eql(u8, nf.name, "expert") or
            std.mem.eql(u8, nf.name, "hc_attn") or std.mem.eql(u8, nf.name, "hc_ffn") or std.mem.eql(u8, nf.name, "hc_attn_flat") or std.mem.eql(u8, nf.name, "hc_ffn_flat");
        if (!skip) {
            const v = @field(names.*, nf.name);
            const T = @TypeOf(v);
            const expert_field = comptime std.mem.startsWith(u8, nf.name, "expert_");
            var list: []const []const u8 = &.{};
            if (T == []const u8) list = &.{v} else if (T == ?[]const u8) {
                if (v) |x| list = &.{x};
            } else if (T == []const []const u8) list = v;
            if (list.len > 0 and !(expert_field and c.num_experts == 0)) {
                var found = false;
                for (list) |t| {
                    const full = if (std.mem.indexOf(u8, t, "{p}") != null) try expand(a, t, prefix, 0, 0) else try std.fmt.allocPrint(a, "{s}{s}{s}", .{ lp, if (expert_field) ep[lp.len..] else "", t });
                    if (ck.find(full) != null) found = true;
                    if (!found and std.mem.endsWith(u8, full, ".weight")) {
                        if (ck.find(try std.fmt.allocPrint(a, "{s}.bias", .{full[0 .. full.len - ".weight".len]})) != null) found = true;
                    }
                }
                if (!found) absent += 1;
            }
        }
    }
    return absent;
}

fn countText(names: []const []const u8) usize {
    var n: usize = 0;
    for (names) |x| {
        if (outsideTextModel(x) == null) n += 1;
    }
    return n;
}

fn allNames(a: Allocator, ck: *const Checkpoint) ![]const []const u8 {
    const names = try a.alloc([]const u8, ck.tensors.len);
    for (ck.tensors, names) |t, *n| n.* = t.name;
    return names;
}

/// Tensors a family's templates do not name (with the overrides already
/// chosen), as candidates for a rename.
fn unreadExcept(a: Allocator, all: []const []const u8, ck: *const Checkpoint, f: *const Arch, c: *const arch.Config, overrides: []const Override) ![]const []const u8 {
    var out: std.ArrayList([]const u8) = .empty;
    const prefix = try modelPrefix(a, ck, &f.names);
    outer: for (all) |n| {
        if (outsideTextModel(n) != null) continue;
        const p = try locate(a, &f.names, prefix, c.num_layers, c.num_experts, n);
        if (try fieldOf(a, &f.names, p, n) != null) continue;
        for (overrides) |o| if (std.mem.indexOf(u8, o.value, p.rel) != null and p.rel.len > 0) continue :outer;
        try out.append(a, n);
    }
    return out.items;
}

/// Parses the config with the draft and with the definition ditch already
/// has for its model_type, and says whether they agree.
fn compareWithKnown(ctx: Ctx, config_text: []const u8, own_type: []const u8, known: *const Arch) !void {
    const a = ctx.arena;
    const draft_family = models.lookup(own_type).?;
    const as_draft = dumpConfig(a, config_text);
    const as_known = dumpConfig(a, try retype(a, config_text, known.model_type));
    if (std.mem.eql(u8, as_draft, as_known) and std.mem.eql(u8, draft_family.names.layer, known.names.layer)) {
        try ctx.out.print("* The draft parses this config.json exactly as the built-in {s} does.\n", .{known.model_type});
        return;
    }
    try ctx.out.print("* The draft parses this config.json differently from the built-in {s}:\n", .{known.model_type});
    var da = std.mem.splitScalar(u8, as_draft, '\n');
    var ka = std.mem.splitScalar(u8, as_known, '\n');
    var shown: usize = 0;
    while (da.next()) |x| {
        const y = ka.next() orelse "";
        if (!std.mem.eql(u8, x, y) and shown < 20) {
            try ctx.out.print("    draft {s}\n    known {s}\n", .{ x, y });
            shown += 1;
        }
    }
}

fn writeDraft(ctx: Ctx, ck: *const Checkpoint, own_type: []const u8, t: Trial, keys: KeyReport, known: ?*const Arch) ![]const u8 {
    const a = ctx.arena;
    var out: std.Io.Writer.Allocating = .init(a);
    const w = &out.writer;
    const f = t.family;
    try w.print("-- {s}: drafted by `ditch add-model {s}`\n", .{ own_type, ctx.settings.model });
    try w.print("-- from {s} ({d} tensors in {d} safetensors file(s)).\n", .{ ck.source, ck.tensors.len, ck.shards.len });
    if (known) |k| try w.print("-- ditch already defines {s} ({s}); this draft was made without that definition.\n", .{ own_type, k.model_type });
    try w.writeAll("--\n");
    if (t.ok) {
        const text_unread = countText(t.unread);
        try w.print("-- Match: ditch's loader reads this checkpoint as `{s}`{s}", .{ f.model_type, if (t.overrides.len > 0) " with the renamed tensors below" else "" });
        if (text_unread == 0) try w.writeAll(", and every tensor of the text model is read.\n") else try w.print("; {d} tensor(s) of the text model are left unread (listed at the end).\n", .{text_unread});
        try w.print("-- `{s}`: {s}\n", .{ f.model_type, if (f.verified) "verified against a reference forward pass" else "not yet verified" });
    } else {
        try w.print("-- No match: the closest family, `{s}`, does not load this checkpoint:\n--   {s}\n", .{ f.model_type, t.why });
    }
    try w.writeAll("return {\n");
    try w.print("  model_type = \"{s}\",\n", .{own_type});
    try w.print("  base = \"{s}\",", .{f.model_type});
    if (!t.ok) try w.writeAll(" -- guess: the closest family, see above") else if (t.overrides.len > 0) try w.writeAll(" -- the layout that reads the checkpoint once the tensors below are renamed");
    try w.writeAll("\n");
    // Chat template: the checkpoint's own, if ditch recognises it.
    const template = chatTemplate(a, ck);
    if (template) |tpl| {
        const detected = chat.detect(tpl, "");
        if (detected != .raw) {
            try w.print("  -- chat: the checkpoint's own chat template is recognised as `{s}` and used; `chat` only matters without it.\n", .{@tagName(detected)});
        } else {
            try w.print("  chat = \"{s}\", -- guess: the checkpoint's chat template is not one ditch recognises; this is {s}'s fallback. Check the prompt ditch verify renders.\n", .{ f.chat, f.model_type });
        }
    } else try w.print("  -- chat: the checkpoint has no chat template; ditch uses {s}'s fallback `{s}`.\n", .{ f.model_type, f.chat });
    try w.print("  notes = \"Drafted by ditch add-model from {s}; not yet verified.\",\n", .{ctx.settings.model});
    if (t.overrides.len > 0) {
        try w.writeAll("  names = {\n");
        for (t.overrides) |o| try w.print("    {s} = {s}, -- {s}\n", .{ o.field, o.value, o.why });
        try w.writeAll("  },\n");
    }
    try w.writeAll("}\n");

    // What is left.
    const groups = try groupNames(a, ck, t.unread);
    var text_groups: usize = 0;
    for (groups) |g| {
        if (g.why == null) text_groups += 1;
    }
    if (text_groups > 0) {
        try w.writeAll("\n-- Tensors of the text model ditch does not read with this definition\n-- (`{n}` stands for layer and expert indices):\n");
        for (groups) |g| if (g.why == null) {
            try w.print("--   {s} {s} {s} ({d}x)", .{ g.pat, g.dtype, try fmtShape(a, g.shape), g.count });
            if (try slotElsewhere(a, g.first)) |hint| try w.print(": {s}", .{hint});
            try w.writeAll("\n");
        };
    }
    var outside: usize = 0;
    for (groups) |g| {
        if (g.why != null) outside += g.count;
    }
    if (outside > 0) {
        try w.writeAll("\n-- Not part of the text model, left unread:\n");
        for (groups) |*g| if (g.why) |why| try w.print("--   {s} ({d}x): {s}\n", .{ g.pat, g.count, why });
    }
    if (keys.unread.len > 0) {
        try w.writeAll("\n-- config.json keys no part of this definition reads (changing them changes nothing);\n-- check that none of them changes the computation:\n--  ");
        var col: usize = 3;
        for (keys.unread) |k| {
            if (col + k.len + 1 > 78) {
                try w.writeAll("\n--  ");
                col = 3;
            }
            try w.print(" {s}", .{k});
            col += k.len + 1;
        }
        try w.writeAll("\n");
    }
    return out.written();
}

fn chatTemplate(a: Allocator, ck: *const Checkpoint) ?[]const u8 {
    if (ck.file("tokenizer_config.json")) |tc| {
        const v = std.json.parseFromSliceLeaky(std.json.Value, a, tc, .{}) catch return null;
        if (v == .object) if (v.object.get("chat_template")) |ct| switch (ct) {
            .string => |s| return s,
            .array => |arr| for (arr.items) |item| {
                if (item == .object) if (item.object.get("template")) |t| if (t == .string) return t.string;
            },
            else => {},
        };
    }
    return ck.file("chat_template.jinja");
}

/// The building block a tensor name fills in another family, if any: the
/// name is a template (relative to a layer) of some family's `names`.
fn slotElsewhere(a: Allocator, full: []const u8) !?[]const u8 {
    const p = try pattern(a, full);
    // Relative part after the last `{n}.` segment.
    const idx = std.mem.lastIndexOf(u8, p.pat, "{n}.") orelse return null;
    const rel = p.pat[idx + 4 ..];
    for (models.builtins()) |f| {
        inline for (@typeInfo(Names).@"struct".fields) |nf| {
            const v = @field(f.names, nf.name);
            const T = @TypeOf(v);
            var hit = false;
            if (T == []const u8 or T == ?[]const u8) {
                if (@as(?[]const u8, v)) |x| hit = std.mem.eql(u8, x, rel);
            } else if (T == []const []const u8) {
                for (v) |x| hit = hit or std.mem.eql(u8, x, rel);
            }
            if (hit) return try std.fmt.allocPrint(a, "`names.{s}` of {s} reads this name", .{ nf.name, f.model_type });
        }
    }
    return null;
}

/// A fixture read as add-model reads a checkpoint, with its model_type
/// replaced and tensor names renamed (`renames` pairs of substrings).
fn testCheckpoint(ctx: Ctx, fixture: []const u8, model_type: []const u8, renames: []const [2][]const u8) !Checkpoint {
    const a = ctx.arena;
    const ck = try readLocal(ctx, fixture);
    var small: std.ArrayList(SmallFile) = .empty;
    for (ck.small) |f| {
        if (std.mem.eql(u8, f.name, "model.safetensors.index.json")) continue;
        try small.append(a, if (std.mem.eql(u8, f.name, "config.json")) .{ .name = f.name, .bytes = try retype(a, f.bytes, model_type) } else f);
    }
    var shards: std.ArrayList(Shard) = .empty;
    var tensors: std.ArrayList(Tensor) = .empty;
    for (ck.shards) |sh| {
        var header: []const u8 = sh.head[8..];
        for (renames) |r| header = try std.mem.replaceOwned(u8, a, header, r[0], r[1]);
        const head = try a.alloc(u8, 8 + header.len);
        std.mem.writeInt(u64, head[0..8], header.len, .little);
        @memcpy(head[8..], header);
        try addShard(a, sh.name, head, &shards, &tensors);
    }
    return .{ .source = fixture, .small = small.items, .shards = shards.items, .tensors = tensors.items };
}

test "add-model reads a known family under a new model_type, and renames tensors by place and shape" {
    // Only loads that succeed run here: a refused trial load logs its
    // error, which the test runner counts as a failure (the binary collects
    // it instead); tests/e2e.sh runs the whole command.
    const gpa = std.testing.allocator;
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    const io = std.testing.io;
    const pool = tensor.Pool.init(io, 2);
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const tmp_path = path_buf[0..try tmp.dir.realPath(io, &path_buf)];
    var discard_buf: [256]u8 = undefined;
    var discard: Io.Writer.Discarding = .init(&discard_buf);
    const ctx: Ctx = .{ .gpa = gpa, .arena = a, .io = io, .env = undefined, .settings = undefined, .http = undefined, .cache_root = tmp_path, .pool = &pool, .out = &discard.writer, .result = &discard.writer };

    // Qwen3 under another name loads as qwen3 with every tensor read.
    {
        const ck = try testCheckpoint(ctx, "tests/fixtures/qwen3", "qwen_renamed_family", &.{});
        const work = try std.fs.path.join(a, &.{ tmp_path, "qwen3" });
        try writeCopy(ctx, &ck, work);
        const qwen3 = models.lookup("qwen3").?;
        const t = try tryLoad(ctx, &ck, work, qwen3, &.{}, "add_model_trial_test_1");
        try std.testing.expect(t.ok);
        try std.testing.expectEqual(@as(usize, 0), countText(t.unread));
    }
    // Llama with its MLP renamed w1/w3/w2: each missing name is matched to
    // the unread tensor of its layer with the expected shape and role, and
    // the renamed family reads everything.
    {
        const ck = try testCheckpoint(ctx, "tests/fixtures/llama", "llama_renamed_mlp", &.{ .{ "mlp.gate_proj", "mlp.w1" }, .{ "mlp.up_proj", "mlp.w3" }, .{ "mlp.down_proj", "mlp.w2" } });
        const work = try std.fs.path.join(a, &.{ tmp_path, "llama" });
        try writeCopy(ctx, &ck, work);
        const llama = models.lookup("llama").?;
        const cfg = try arch.parseConfig(a, try retype(a, ck.file("config.json").?, "llama"));
        const all = try allNames(a, &ck);
        var overrides: std.ArrayList(Override) = .empty;
        for ([_][2][]const u8{ .{ "gate_proj", "\"mlp.w1.weight\"" }, .{ "up_proj", "\"mlp.w3.weight\"" }, .{ "down_proj", "\"mlp.w2.weight\"" } }) |want| {
            const missing = try std.fmt.allocPrint(a, "model.layers.1.mlp.{s}.weight", .{want[0]});
            const o = (try proposeRename(a, &ck, llama, &cfg, missing, try unreadExcept(a, all, &ck, llama, &cfg, overrides.items))).?;
            try std.testing.expectEqualStrings(want[1], o.value);
            try overrides.append(a, o);
        }
        const t = try tryLoad(ctx, &ck, work, llama, overrides.items, "add_model_trial_test_2");
        try std.testing.expect(t.ok);
        try std.testing.expectEqual(@as(usize, 0), countText(t.unread));
    }
}

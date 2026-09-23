//! `ditch render-template <DIR> <CASES.json>`, left out of the help: renders
//! conversations with a model directory's chat template and encodes them as
//! a study would, for `tools/chat_template_check.py` to compare with
//! transformers' `apply_chat_template`. DIR needs only the tokenizer files
//! (tokenizer_config.json, chat_template.jinja / .json,
//! special_tokens_map.json, tokenizer.json, config.json for the model type).
//!
//! CASES.json is a list of `{"messages": [...], "add_generation_prompt":
//! bool, "kwargs": {...}}`. Output is one JSON object on stdout:
//! `{"renderer": "jinja" | family, "fold_system": bool, "content_parts":
//! bool, "warnings": [], "results": [{"text": ..., "ids": [...]} |
//! {"error": ...}]}`; the ids are present when a tokenizer loads.

const std = @import("std");
const Io = std.Io;
const chat = @import("chat.zig");
const jinja = @import("jinja.zig");
const engine = @import("engine.zig");
const Tokenizer = @import("tokenizer.zig").Tokenizer;

const Allocator = std.mem.Allocator;

pub fn run(gpa: Allocator, arena: Allocator, io: Io, args: []const []const u8, out: *Io.Writer) !void {
    if (args.len != 2) {
        std.log.err("usage: ditch render-template <DIR> <CASES.json>", .{});
        return error.InvalidArguments;
    }
    const cwd = Io.Dir.cwd();
    var dir = try cwd.openDir(io, args[0], .{});
    defer dir.close(io);
    const read = struct {
        fn f(d: Io.Dir, i: Io, a: Allocator, name: []const u8) ?[]const u8 {
            return d.readFileAlloc(i, name, a, .unlimited) catch null;
        }
    }.f;
    const tokenizer_config = read(dir, io, arena, "tokenizer_config.json");
    const source = try chat.pickTemplate(arena, tokenizer_config, read(dir, io, arena, "chat_template.jinja"), read(dir, io, arena, "chat_template.json"));
    const tokens = try chat.specialTokens(arena, tokenizer_config, read(dir, io, arena, "special_tokens_map.json"));
    var model_type: []const u8 = "";
    if (read(dir, io, arena, "config.json")) |cfg| {
        const v = std.json.parseFromSliceLeaky(std.json.Value, arena, cfg, .{}) catch null;
        if (v) |c| if (c == .object) if (c.object.get("model_type")) |t| if (t == .string) {
            model_type = t.string;
        };
    }
    const now = Io.Clock.real.now(io);
    chat.now_seconds = @intCast(@divFloor(now.nanoseconds, std.time.ns_per_s));
    chat.today = chat.Date.fromUnix(chat.now_seconds);
    var format = try chat.Format.init(gpa, source, tokens, chat.detect(source, model_type));
    defer format.deinit();

    const tok: ?*Tokenizer = if (Tokenizer.loadDir(gpa, io, arena, dir, args[0], tokenizer_config)) |l| l.tokenizer else |_| null;
    defer if (tok) |t| t.deinit();
    const encoding: ?engine.Engine.Encoding = if (tok) |t| engine.Engine.encodingFor(gpa, t, source, format) else null;

    const cases_text = try cwd.readFileAlloc(io, args[1], arena, .unlimited);
    const cases = try std.json.parseFromSliceLeaky(std.json.Value, arena, cases_text, .{});
    if (cases != .array) return error.InvalidCases;

    var js: std.json.Stringify = .{ .writer = out };
    try js.beginObject();
    try js.objectField("renderer");
    try js.write(if (format.template != null) "jinja" else @tagName(format.family));
    try js.objectField("fold_system");
    try js.write(format.fold_system);
    try js.objectField("content_parts");
    try js.write(format.content_parts);
    try js.objectField("warnings");
    try js.write(&[_][]const u8{});
    try js.objectField("results");
    try js.beginArray();
    for (cases.array.items) |case| {
        try js.beginObject();
        if (renderCase(gpa, arena, &format, case)) |text| {
            try js.objectField("text");
            try js.write(text);
            if (tok) |t| {
                const e = encoding.?;
                const full = try std.mem.concat(arena, u8, &.{ e.bos_prefix, text });
                const ids = try t.encode(gpa, full, e.add_special);
                defer gpa.free(ids);
                try js.objectField("ids");
                try js.write(ids);
            }
        } else |err| {
            try js.objectField("error");
            try js.write(switch (err) {
                error.Render => last_error,
                else => @errorName(err),
            });
        }
        try js.endObject();
    }
    try js.endArray();
    try js.endObject();
    try out.writeAll("\n");
    try out.flush();
}

var last_error: []const u8 = "";

fn renderCase(gpa: Allocator, arena: Allocator, format: *const chat.Format, case: std.json.Value) ![]const u8 {
    if (case != .object) return error.InvalidCases;
    const obj = case.object;
    const agp = if (obj.get("add_generation_prompt")) |v| v == .bool and v.bool else true;
    const msgs_json = obj.get("messages") orelse return error.InvalidCases;
    if (format.template != null) {
        var kwargs = std.ArrayList(jinja.Var).empty;
        if (obj.get("kwargs")) |kw| if (kw == .object) {
            var it = kw.object.iterator();
            while (it.next()) |e| try kwargs.append(arena, .{ .name = e.key_ptr.*, .value = try jinja.Value.fromJson(arena, e.value_ptr.*) });
        };
        var diag: jinja.Diagnostic = .{};
        const msgs = try jinja.Value.fromJson(arena, msgs_json);
        return format.renderValues(arena, msgs, .{ .add_generation_prompt = agp, .kwargs = kwargs.items }, &diag) catch |e| {
            if (e == error.OutOfMemory) return e;
            last_error = try arena.dupe(u8, diag.message());
            return error.Render;
        };
    }
    // A named family: system / user / assistant turns, always with the generation prompt.
    var msgs = std.ArrayList(chat.Message).empty;
    for (msgs_json.array.items) |m| {
        const role = std.meta.stringToEnum(chat.Role, m.object.get("role").?.string) orelse return error.UnsupportedRole;
        try msgs.append(arena, .{ .role = role, .content = m.object.get("content").?.string });
    }
    const text = try chat.render(gpa, format.family, msgs.items);
    defer gpa.free(text);
    return arena.dupe(u8, text);
}

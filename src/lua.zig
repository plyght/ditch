//! Lua configuration files. A `config.lua` is run in a sandboxed Lua 5.4
//! state (base, string, table, math and utf8 libraries plus `os.getenv`) and
//! must either `return` a table of settings or assign them as globals.
//! The result is converted into the generic config value tree.

const std = @import("std");
const tree = @import("tree.zig");

/// The Lua C API, shared with the model-definition loader (models.zig).
pub const c = @cImport({
    @cInclude("lua.h");
    @cInclude("lauxlib.h");
    @cInclude("lualib.h");
});

const Allocator = std.mem.Allocator;

pub const Error = error{ LuaSyntax, LuaRuntime, LuaNotTable, OutOfMemory };

/// Returns the last Lua error message, allocated in `a`.
fn takeError(L: *c.lua_State, a: Allocator) ![]const u8 {
    var len: usize = 0;
    const msg = c.lua_tolstring(L, -1, &len);
    const s = if (msg) |m| m[0..len] else "unknown Lua error";
    const out = try a.dupe(u8, s);
    c.lua_settop(L, -2);
    return out;
}

pub const Result = struct {
    parsed: tree.Parsed,
    /// Error description when parsing failed (allocated in the parsed arena).
    err: ?[]const u8 = null,
};

/// Opens the sandboxed standard libraries: base, string, table, math, utf8
/// and `os.getenv`; nothing that reads or writes files or loads code.
pub fn openSandbox(L: *c.lua_State) void {
    c.luaL_requiref(L, "_G", c.luaopen_base, 1);
    c.lua_settop(L, -2);
    c.luaL_requiref(L, "string", c.luaopen_string, 1);
    c.lua_settop(L, -2);
    c.luaL_requiref(L, "table", c.luaopen_table, 1);
    c.lua_settop(L, -2);
    c.luaL_requiref(L, "math", c.luaopen_math, 1);
    c.lua_settop(L, -2);
    c.luaL_requiref(L, "utf8", c.luaopen_utf8, 1);
    c.lua_settop(L, -2);
    // Only os.getenv from the os library.
    c.luaL_requiref(L, "os", c.luaopen_os, 0);
    _ = c.lua_getfield(L, -1, "getenv");
    c.lua_createtable(L, 0, 1);
    c.lua_pushvalue(L, -2);
    c.lua_setfield(L, -2, "getenv");
    c.lua_setglobal(L, "os");
    c.lua_settop(L, -3);
    // Remove functions that could touch the outside world.
    for ([_][*:0]const u8{ "dofile", "loadfile", "load", "require", "collectgarbage" }) |name| {
        c.lua_pushnil(L);
        c.lua_setglobal(L, name);
    }
}

/// Evaluates Lua source text and returns the configuration tree.
pub fn parse(gpa: Allocator, source: []const u8, chunk_name: []const u8) !Result {
    var arena = std.heap.ArenaAllocator.init(gpa);
    errdefer arena.deinit();
    const a = arena.allocator();
    const root = try a.create(tree.Table);
    root.* = .{};

    const L = c.luaL_newstate() orelse return error.OutOfMemory;
    defer c.lua_close(L);
    openSandbox(L);

    // Remember the initial globals so assignments made by the chunk can be collected.
    _ = c.lua_getglobal(L, "_G");
    c.lua_createtable(L, 0, 32);
    c.lua_pushnil(L);
    while (c.lua_next(L, -3) != 0) {
        c.lua_settop(L, -2); // drop value
        c.lua_pushvalue(L, -1);
        c.lua_pushboolean(L, 1);
        c.lua_settable(L, -4);
    }
    c.lua_setfield(L, c.LUA_REGISTRYINDEX, "ditch_initial_globals");
    c.lua_settop(L, -2);

    const name_z = try a.dupeZ(u8, chunk_name);
    if (c.luaL_loadbufferx(L, source.ptr, source.len, name_z.ptr, "t") != c.LUA_OK) {
        return .{ .parsed = .{ .arena = arena, .root = root }, .err = try takeError(L, a) };
    }
    if (c.lua_pcallk(L, 0, 1, 0, 0, null) != c.LUA_OK) {
        return .{ .parsed = .{ .arena = arena, .root = root }, .err = try takeError(L, a) };
    }
    if (c.lua_type(L, -1) == c.LUA_TTABLE) {
        try fillTable(L, a, root, -1, 0);
    } else if (c.lua_type(L, -1) == c.LUA_TNIL) {
        // Collect globals assigned by the chunk.
        c.lua_settop(L, -2);
        _ = c.lua_getglobal(L, "_G");
        _ = c.lua_getfield(L, c.LUA_REGISTRYINDEX, "ditch_initial_globals");
        c.lua_pushnil(L);
        while (c.lua_next(L, -3) != 0) {
            // stack: _G, initial, key, value
            c.lua_pushvalue(L, -2);
            _ = c.lua_gettable(L, -4);
            const was_initial = c.lua_type(L, -1) != c.LUA_TNIL;
            c.lua_settop(L, -2);
            if (!was_initial and c.lua_type(L, -2) == c.LUA_TSTRING) {
                const key = try luaString(L, a, -2);
                const v = try convert(L, a, -1, 0);
                try root.map.put(a, key, v);
            }
            c.lua_settop(L, -2);
        }
    } else {
        return .{ .parsed = .{ .arena = arena, .root = root }, .err = try a.dupe(u8, "the configuration chunk must return a table") };
    }
    return .{ .parsed = .{ .arena = arena, .root = root } };
}

fn luaString(L: *c.lua_State, a: Allocator, idx: c_int) ![]const u8 {
    var len: usize = 0;
    const p = c.lua_tolstring(L, idx, &len) orelse return a.dupe(u8, "");
    return a.dupe(u8, p[0..len]);
}

fn convert(L: *c.lua_State, a: Allocator, idx: c_int, depth: usize) Error!tree.Value {
    const abs = c.lua_absindex(L, idx);
    switch (c.lua_type(L, abs)) {
        c.LUA_TBOOLEAN => return .{ .boolean = c.lua_toboolean(L, abs) != 0 },
        c.LUA_TNUMBER => {
            if (c.lua_isinteger(L, abs) != 0) return .{ .integer = @intCast(c.lua_tointegerx(L, abs, null)) };
            return .{ .float = c.lua_tonumberx(L, abs, null) };
        },
        c.LUA_TSTRING => return .{ .string = try luaString(L, a, abs) },
        c.LUA_TTABLE => {
            if (depth > 64) return error.LuaRuntime;
            const n = c.lua_rawlen(L, abs);
            if (n > 0) {
                // Sequence: keys 1..n only.
                var is_seq = true;
                c.lua_pushnil(L);
                while (c.lua_next(L, abs) != 0) {
                    if (c.lua_isinteger(L, -2) == 0 or c.lua_tointegerx(L, -2, null) < 1 or c.lua_tointegerx(L, -2, null) > @as(c.lua_Integer, @intCast(n))) is_seq = false;
                    c.lua_settop(L, -2);
                }
                if (is_seq) {
                    const arr = try a.alloc(tree.Value, @intCast(n));
                    var i: c.lua_Integer = 1;
                    while (i <= @as(c.lua_Integer, @intCast(n))) : (i += 1) {
                        _ = c.lua_rawgeti(L, abs, i);
                        arr[@intCast(i - 1)] = try convert(L, a, -1, depth + 1);
                        c.lua_settop(L, -2);
                    }
                    return .{ .array = arr };
                }
            }
            const t = try a.create(tree.Table);
            t.* = .{};
            try fillTable(L, a, t, abs, depth + 1);
            return .{ .table = t };
        },
        else => return error.LuaNotTable,
    }
}

fn fillTable(L: *c.lua_State, a: Allocator, t: *tree.Table, idx: c_int, depth: usize) Error!void {
    const abs = c.lua_absindex(L, idx);
    c.lua_pushnil(L);
    while (c.lua_next(L, abs) != 0) {
        const key: []const u8 = switch (c.lua_type(L, -2)) {
            c.LUA_TSTRING => try luaString(L, a, -2),
            c.LUA_TNUMBER => try std.fmt.allocPrint(a, "{d}", .{c.lua_tointegerx(L, -2, null)}),
            else => {
                c.lua_settop(L, -2);
                continue;
            },
        };
        const v = convert(L, a, -1, depth) catch |err| switch (err) {
            error.LuaNotTable => {
                // Functions, userdata etc. are skipped.
                c.lua_settop(L, -2);
                continue;
            },
            else => return err,
        };
        try t.map.put(a, key, v);
        c.lua_settop(L, -2);
    }
}

test "lua config returning a table" {
    const gpa = std.testing.allocator;
    const src =
        \\local trials = 100
        \\return {
        \\  model = "Qwen/Qwen2.5-0.5B-Instruct",
        \\  n_trials = trials * 2,
        \\  winsorization_quantile = 0.95,
        \\  orthogonalize_direction = true,
        \\  scorers = {
        \\    { plugin = "keyword_rate", optimization = "minimize" },
        \\    { plugin = "kl_divergence", optimization = "minimize" },
        \\  },
        \\  good_prompts = { dataset = "mlabonne/harmless_alpaca", split = "train[:400]", column = "text" },
        \\  scorer = { KeywordRate = { keyword_markers = { "sorry", "i cannot" } } },
        \\  chain_of_thought_skips = { { "<think>", "<think></think>" } },
        \\  greeting = string.upper("hi"),
        \\}
    ;
    var r = try parse(gpa, src, "test.lua");
    defer r.parsed.deinit();
    try std.testing.expect(r.err == null);
    const root = r.parsed.root;
    try std.testing.expectEqualStrings("Qwen/Qwen2.5-0.5B-Instruct", root.get("model").?.string);
    try std.testing.expectEqual(@as(i64, 200), root.get("n_trials").?.integer);
    try std.testing.expectEqual(@as(f64, 0.95), root.get("winsorization_quantile").?.float);
    try std.testing.expect(root.get("orthogonalize_direction").?.boolean);
    try std.testing.expectEqualStrings("kl_divergence", root.get("scorers").?.array[1].table.get("plugin").?.string);
    try std.testing.expectEqualStrings("train[:400]", root.getPath("good_prompts.split").?.string);
    try std.testing.expectEqualStrings("i cannot", root.getPath("scorer.KeywordRate.keyword_markers").?.array[1].string);
    try std.testing.expectEqualStrings("<think></think>", root.get("chain_of_thought_skips").?.array[0].array[1].string);
    try std.testing.expectEqualStrings("HI", root.get("greeting").?.string);
}

test "lua config via globals and error reporting" {
    const gpa = std.testing.allocator;
    var r = try parse(gpa, "n_trials = 7\nsystem_prompt = 'x'\n", "g.lua");
    defer r.parsed.deinit();
    try std.testing.expect(r.err == null);
    try std.testing.expectEqual(@as(i64, 7), r.parsed.root.get("n_trials").?.integer);
    try std.testing.expectEqualStrings("x", r.parsed.root.get("system_prompt").?.string);

    var bad = try parse(gpa, "return {", "bad.lua");
    defer bad.parsed.deinit();
    try std.testing.expect(bad.err != null);
    var sandboxed = try parse(gpa, "return { x = io ~= nil, y = require ~= nil }", "s.lua");
    defer sandboxed.parsed.deinit();
    try std.testing.expect(!sandboxed.parsed.root.get("x").?.boolean);
    try std.testing.expect(!sandboxed.parsed.root.get("y").?.boolean);
}

//! Model definitions in Lua. A definition describes one Hugging Face
//! `model_type` entirely in terms of ditch's building blocks: the layout
//! fields of `arch.Arch` (norms, positional encoding, attention and MLP
//! layouts, tensor-name templates), a named Zig hook for config.json keys
//! that need Zig (`hook`), and a Lua `config` function for the rest. The
//! schema is documented in docs/models.md.
//!
//! The built-in definitions (src/models/*.lua) are compiled into the binary;
//! user definitions are loaded from `$XDG_CONFIG_HOME/ditch/models/*.lua`
//! and `--models-dir` and take precedence over built-in ones with the same
//! `model_type`. All definitions share one sandboxed Lua state (the one
//! config.lua runs in, see lua.zig), kept for the life of the process so that
//! `config` functions can run whenever a config.json is parsed.

const std = @import("std");
const arch = @import("arch.zig");
const lua = @import("lua.zig");
const chat = @import("chat.zig");
const definitions = @import("model_definitions");
const builtin_files = definitions.files;

const c = lua.c;
const Allocator = std.mem.Allocator;
const Arch = arch.Arch;
const Names = arch.Names;
const Config = arch.Config;
const Hook = *const fn (*Config, Allocator, std.json.ObjectMap) anyerror!void;

pub const Error = error{ InvalidModelDefinition, OutOfMemory };

const prelude = definitions.prelude;

/// Where a load error is described (allocated in the registry's arena).
pub const Diagnostic = struct {
    message: ?[]const u8 = null,
};

const Registry = struct {
    arena: std.heap.ArenaAllocator,
    L: *c.lua_State,
    families: std.ArrayList(*Arch) = .empty,
    /// The file each family came from, parallel to `families`.
    origins: std.ArrayList([]const u8) = .empty,
    /// The first `builtin_count` families are the compiled-in ones.
    builtin_count: usize = 0,
};

var state_ptr: ?*Registry = null;
var state_lock: std.atomic.Mutex = .unlocked;

fn lock() void {
    while (!state_lock.tryLock()) std.atomic.spinLoopHint();
}

/// The registry, loading the built-in definitions on first use. Callers hold the lock.
fn stateLocked() *Registry {
    if (state_ptr) |r| return r;
    var diag: Diagnostic = .{};
    const r = initRegistry(&diag) catch |err| {
        std.debug.panic("built-in model definitions: {s}", .{diag.message orelse @errorName(err)});
    };
    state_ptr = r;
    return r;
}

fn initRegistry(diag: *Diagnostic) !*Registry {
    const gpa = std.heap.page_allocator;
    const r = try gpa.create(Registry);
    r.* = .{ .arena = .init(gpa), .L = c.luaL_newstate() orelse return error.OutOfMemory };
    const L = r.L;
    lua.openSandbox(L);
    // `null` stands for a JSON null in the `cfg` table of a config function.
    c.lua_createtable(L, 0, 0);
    c.lua_createtable(L, 0, 1);
    _ = c.lua_pushstring(L, "null");
    c.lua_setfield(L, -2, "__name");
    _ = c.lua_setmetatable(L, -2);
    c.lua_pushvalue(L, -1);
    c.lua_setfield(L, c.LUA_REGISTRYINDEX, "ditch_null");
    c.lua_setglobal(L, "null");
    // The metatable of JSON arrays (see `pushJson`), for `is_array(v)`.
    c.lua_createtable(L, 0, 1);
    _ = c.lua_pushstring(L, "json_array");
    c.lua_setfield(L, -2, "__name");
    c.lua_setfield(L, c.LUA_REGISTRYINDEX, "ditch_json_array");
    c.lua_pushcclosure(L, luaWarn, 0);
    c.lua_setglobal(L, "warn");
    c.lua_pushcclosure(L, luaUnsupported, 0);
    c.lua_setglobal(L, "unsupported");
    c.lua_pushcclosure(L, luaInvalid, 0);
    c.lua_setglobal(L, "invalid");
    c.lua_pushcclosure(L, luaRequire, 0);
    c.lua_setglobal(L, "require");
    c.lua_pushcclosure(L, luaRopeScaling, 0);
    c.lua_setglobal(L, "rope_scaling");
    c.lua_pushcclosure(L, luaYarnMscale, 0);
    c.lua_setglobal(L, "yarn_mscale");
    c.lua_pushcclosure(L, luaLog32, 0);
    c.lua_setglobal(L, "log32");
    c.lua_pushcclosure(L, luaPow, 0);
    c.lua_setglobal(L, "pow");
    if (c.luaL_loadbufferx(L, prelude.ptr, prelude.len, "prelude.lua", "t") != c.LUA_OK or c.lua_pcallk(L, 0, 0, 0, 0, null) != c.LUA_OK) {
        diag.message = try luaMessage(L, r.arena.allocator());
        return error.InvalidModelDefinition;
    }
    for (builtin_files) |f| _ = try loadInto(r, f.source, f.name, diag);
    r.builtin_count = r.families.items.len;
    return r;
}

/// Finds the definition of a `model_type` (or one of its aliases). A user
/// definition loaded later shadows a built-in one.
pub fn lookup(model_type: []const u8) ?*const Arch {
    lock();
    defer state_lock.unlock();
    return lookupLocked(stateLocked(), model_type);
}

fn lookupLocked(r: *Registry, model_type: []const u8) ?*Arch {
    var i = r.families.items.len;
    while (i > 0) {
        i -= 1;
        const a = r.families.items[i];
        if (std.mem.eql(u8, a.model_type, model_type)) return a;
        for (a.aliases) |alias| if (std.mem.eql(u8, alias, model_type)) return a;
    }
    return null;
}

/// Every loaded definition, built-in ones first, in load order. Definitions
/// are only added at start-up (`loadDir`), so the slice stays valid.
pub fn families() []const *Arch {
    lock();
    defer state_lock.unlock();
    return stateLocked().families.items;
}

/// The compiled-in definitions (the start of `families`).
pub fn builtins() []const *Arch {
    lock();
    defer state_lock.unlock();
    const r = stateLocked();
    return r.families.items[0..r.builtin_count];
}

/// The file a definition was loaded from (`llama.lua` for a built-in one).
pub fn origin(a: *const Arch) []const u8 {
    lock();
    defer state_lock.unlock();
    const r = stateLocked();
    for (r.families.items, r.origins.items) |f, o| if (f == a) return o;
    return "";
}

/// Loads the definitions of one Lua chunk; returns how many it defined.
pub fn loadSource(source: []const u8, chunk_name: []const u8, diag: *Diagnostic) !usize {
    lock();
    defer state_lock.unlock();
    return loadInto(stateLocked(), source, chunk_name, diag);
}

/// Loads every `*.lua` file of `dir_path` in name order. A missing directory
/// is not an error; a broken file is reported through `report` and skipped.
/// Returns the number of definitions loaded.
pub fn loadDir(io: std.Io, gpa: Allocator, dir_path: []const u8, report: *const fn ([]const u8, []const u8) void) !usize {
    var dir = std.Io.Dir.cwd().openDir(io, dir_path, .{ .iterate = true }) catch |err| switch (err) {
        error.FileNotFound, error.NotDir => return 0,
        else => return err,
    };
    defer dir.close(io);
    var names: std.ArrayList([]const u8) = .empty;
    defer {
        for (names.items) |n| gpa.free(n);
        names.deinit(gpa);
    }
    var it = dir.iterate();
    while (try it.next(io)) |entry| {
        if (entry.kind == .directory or !std.mem.endsWith(u8, entry.name, ".lua")) continue;
        try names.append(gpa, try gpa.dupe(u8, entry.name));
    }
    std.mem.sort([]const u8, names.items, {}, struct {
        fn lt(_: void, a: []const u8, b: []const u8) bool {
            return std.mem.lessThan(u8, a, b);
        }
    }.lt);
    var count: usize = 0;
    for (names.items) |name| {
        const path = try std.fs.path.join(gpa, &.{ dir_path, name });
        defer gpa.free(path);
        const text = dir.readFileAlloc(io, name, gpa, .limited(1 << 20)) catch |err| {
            report(path, @errorName(err));
            continue;
        };
        defer gpa.free(text);
        var diag: Diagnostic = .{};
        count += loadSource(text, path, &diag) catch |err| {
            report(path, diag.message orelse @errorName(err));
            continue;
        };
    }
    return count;
}

// ---------------------------------------------------------------------------
// Loading
// ---------------------------------------------------------------------------

fn luaMessage(L: *c.lua_State, a: Allocator) ![]const u8 {
    var len: usize = 0;
    const msg = c.lua_tolstring(L, -1, &len);
    const s = if (msg) |m| m[0..len] else "unknown Lua error";
    const out = try a.dupe(u8, s);
    c.lua_settop(L, -2);
    return out;
}

fn loadInto(r: *Registry, source: []const u8, chunk_name: []const u8, diag: *Diagnostic) !usize {
    const list = try evalDefinitions(r, source, chunk_name, diag);
    const name = try r.arena.allocator().dupe(u8, std.fs.path.basename(chunk_name));
    for (list) |a| {
        try r.families.append(r.arena.allocator(), a);
        try r.origins.append(r.arena.allocator(), name);
    }
    return list.len;
}

/// Evaluates a definition chunk into families without registering them
/// (`base` still resolves against the registered ones).
pub fn parseSource(source: []const u8, chunk_name: []const u8, diag: *Diagnostic) ![]const *Arch {
    lock();
    defer state_lock.unlock();
    return evalDefinitions(stateLocked(), source, chunk_name, diag);
}

fn evalDefinitions(r: *Registry, source: []const u8, chunk_name: []const u8, diag: *Diagnostic) ![]*Arch {
    const L = r.L;
    const a = r.arena.allocator();
    const top = c.lua_gettop(L);
    defer c.lua_settop(L, top);
    const name_z = try a.dupeZ(u8, chunk_name);
    if (c.luaL_loadbufferx(L, source.ptr, source.len, name_z.ptr, "t") != c.LUA_OK) {
        diag.message = try luaMessage(L, a);
        return error.InvalidModelDefinition;
    }
    // Each file gets its own globals, falling back to the shared ones.
    c.lua_createtable(L, 0, 0);
    c.lua_createtable(L, 0, 1);
    _ = c.lua_getglobal(L, "_G");
    c.lua_setfield(L, -2, "__index");
    _ = c.lua_setmetatable(L, -2);
    _ = c.lua_setupvalue(L, -2, 1);
    if (c.lua_pcallk(L, 0, 1, 0, 0, null) != c.LUA_OK) {
        diag.message = try luaMessage(L, a);
        return error.InvalidModelDefinition;
    }
    if (c.lua_type(L, -1) != c.LUA_TTABLE) {
        diag.message = try std.fmt.allocPrint(a, "{s}: a model definition must return a table", .{chunk_name});
        return error.InvalidModelDefinition;
    }
    _ = c.lua_getfield(L, -1, "model_type");
    const single = c.lua_type(L, -1) != c.LUA_TNIL;
    c.lua_settop(L, -2);
    var ctx = Ctx{ .L = L, .a = a, .diag = diag, .where = chunk_name };
    if (single) {
        const out = try a.alloc(*Arch, 1);
        out[0] = try a.create(Arch);
        out[0].* = try readFamily(r, &ctx, -1);
        return out;
    }
    const n = c.lua_rawlen(L, -1);
    if (n == 0) return ctx.fail("a model definition must return a family table (with model_type) or a list of them", .{});
    const out = try a.alloc(*Arch, n);
    for (out, 0..) |*f, k| {
        _ = c.lua_rawgeti(L, -1, @intCast(k + 1));
        defer c.lua_settop(L, -2);
        f.* = try a.create(Arch);
        f.*.* = try readFamily(r, &ctx, -1);
    }
    return out;
}

const arch_fields_special = [_][]const u8{ "base", "hook", "config", "names" };

fn readFamily(r: *Registry, ctx: *Ctx, idx: c_int) !Arch {
    const L = ctx.L;
    const t = c.lua_absindex(L, idx);
    if (c.lua_type(L, t) != c.LUA_TTABLE) return ctx.fail("a family must be a table", .{});
    var result: Arch = .{ .model_type = "", .llama_cpp = null };
    // `base`: start from another family's definition. Its identity (type,
    // aliases, notes) and verification status are not inherited.
    _ = c.lua_getfield(L, t, "base");
    if (c.lua_type(L, -1) != c.LUA_TNIL) {
        const base = try ctx.string(-1, "base");
        const parent = lookupLocked(r, base) orelse return ctx.fail("base '{s}' is not a known model_type (built-in definitions and earlier files only)", .{base});
        result = parent.*;
        result.aliases = &.{};
        result.notes = "";
        result.verified = false;
        result.inherits = parent.inherits orelse parent.model_type;
    }
    c.lua_settop(L, -2);
    _ = c.lua_getfield(L, t, "model_type");
    const mt = if (c.lua_type(L, -1) == c.LUA_TSTRING) try ctx.string(-1, "model_type") else "";
    c.lua_settop(L, -2);
    if (mt.len == 0) return ctx.fail("a family needs a model_type string", .{});
    ctx.family = mt;
    defer ctx.family = null;

    c.lua_pushnil(L);
    while (c.lua_next(L, t) != 0) {
        defer c.lua_settop(L, -2);
        if (c.lua_type(L, -2) != c.LUA_TSTRING) return ctx.fail("family keys must be names", .{});
        const key = try ctx.string(-2, "key");
        if (std.mem.eql(u8, key, "base")) continue;
        if (std.mem.eql(u8, key, "hook")) {
            if (c.lua_type(L, -1) == c.LUA_TBOOLEAN and c.lua_toboolean(L, -1) == 0) {
                result.extra = null;
                continue;
            }
            const name = try ctx.string(-1, "hook");
            result.extra = arch.hookByName(name) orelse return ctx.fail("hook '{s}' is not a Zig building block (known: {s})", .{ name, try hookList(ctx.a) });
            continue;
        }
        if (std.mem.eql(u8, key, "config")) {
            if (c.lua_type(L, -1) == c.LUA_TBOOLEAN and c.lua_toboolean(L, -1) == 0) {
                result.script = null;
                continue;
            }
            if (c.lua_type(L, -1) != c.LUA_TFUNCTION) return ctx.fail("config must be a function(cfg, c)", .{});
            c.lua_pushvalue(L, -1);
            result.script = c.luaL_ref(L, c.LUA_REGISTRYINDEX);
            continue;
        }
        if (std.mem.eql(u8, key, "names")) {
            ctx.path = "names";
            defer ctx.path = null;
            result.names = try readValue(Names, ctx, -1, result.names, .partial);
            continue;
        }
        var found = false;
        inline for (@typeInfo(Arch).@"struct".fields) |f| {
            if (comptime settable(f.type) and !isSpecial(f.name)) {
                if (std.mem.eql(u8, key, f.name)) {
                    ctx.path = f.name;
                    defer ctx.path = null;
                    @field(result, f.name) = try readValue(f.type, ctx, -1, @field(result, f.name), .partial);
                    found = true;
                }
            }
        }
        if (!found) return ctx.fail("unknown field '{s}' (see docs/models.md)", .{key});
    }
    if (chat.Template.parse(result.chat) == null) return ctx.fail("chat '{s}' is not a known template family", .{result.chat});
    return result;
}

fn isSpecial(name: []const u8) bool {
    for (arch_fields_special) |s| if (std.mem.eql(u8, s, name)) return true;
    return std.mem.eql(u8, name, "extra") or std.mem.eql(u8, name, "script") or std.mem.eql(u8, name, "inherits");
}

fn hookList(a: Allocator) ![]const u8 {
    var out: std.Io.Writer.Allocating = .init(a);
    for (arch.hooks, 0..) |h, i| {
        if (i > 0) try out.writer.writeAll(", ");
        try out.writer.writeAll(h.name);
    }
    return out.written();
}

// ---------------------------------------------------------------------------
// Lua <-> Zig values by reflection
// ---------------------------------------------------------------------------

const Ctx = struct {
    L: *c.lua_State,
    a: Allocator,
    diag: *Diagnostic,
    where: []const u8,
    family: ?[]const u8 = null,
    path: ?[]const u8 = null,

    fn fail(self: *Ctx, comptime fmt: []const u8, args: anytype) error{ InvalidModelDefinition, OutOfMemory } {
        var out: std.Io.Writer.Allocating = .init(self.a);
        out.writer.print("{s}", .{self.where}) catch return error.OutOfMemory;
        if (self.family) |f| out.writer.print(" ({s})", .{f}) catch return error.OutOfMemory;
        if (self.path) |p| out.writer.print(", {s}", .{p}) catch return error.OutOfMemory;
        out.writer.writeAll(": ") catch return error.OutOfMemory;
        out.writer.print(fmt, args) catch return error.OutOfMemory;
        self.diag.message = out.written();
        return error.InvalidModelDefinition;
    }

    fn string(self: *Ctx, idx: c_int, what: []const u8) ![]const u8 {
        if (c.lua_type(self.L, idx) != c.LUA_TSTRING) return self.fail("{s}: expected a string, found {s}", .{ what, typeName(self.L, idx) });
        var len: usize = 0;
        const p = c.lua_tolstring(self.L, idx, &len);
        return self.a.dupe(u8, p[0..len]);
    }
};

fn typeName(L: *c.lua_State, idx: c_int) []const u8 {
    return std.mem.span(c.lua_typename(L, c.lua_type(L, idx)));
}

/// `.partial`: only the keys present change the value (a definition).
/// `.full`: every field is read back (the `c` table of a config function); a
/// missing key clears an optional field.
const Mode = enum { partial, full };

/// Types a definition or a config function can set; pointers to the whole
/// family (`Config.arch`) and the hook function are not values.
fn settable(comptime T: type) bool {
    return switch (@typeInfo(T)) {
        .pointer => |p| p.size == .slice or (p.size == .one and @typeInfo(p.child) == .@"struct" and p.child != Arch),
        .@"fn" => false,
        .optional => |o| settable(o.child),
        else => true,
    };
}

fn readValue(comptime T: type, ctx: *Ctx, idx: c_int, cur: T, mode: Mode) Error!T {
    const L = ctx.L;
    const i = c.lua_absindex(L, idx);
    const ty = c.lua_type(L, i);
    switch (@typeInfo(T)) {
        .bool => {
            if (ty != c.LUA_TBOOLEAN) return ctx.fail("expected a boolean, found {s}", .{typeName(L, i)});
            return c.lua_toboolean(L, i) != 0;
        },
        .int => |info| {
            if (ty != c.LUA_TNUMBER) return ctx.fail("expected an integer, found {s}", .{typeName(L, i)});
            var isnum: c_int = 0;
            const v = c.lua_tointegerx(L, i, &isnum);
            if (isnum == 0) return ctx.fail("expected an integer, found {d}", .{c.lua_tonumberx(L, i, null)});
            if (info.signedness == .unsigned and info.bits == 64) return @bitCast(v);
            return std.math.cast(T, v) orelse ctx.fail("{d} is out of range", .{v});
        },
        .float => {
            if (ty != c.LUA_TNUMBER) return ctx.fail("expected a number, found {s}", .{typeName(L, i)});
            return @floatCast(c.lua_tonumberx(L, i, null));
        },
        .@"enum" => |info| {
            const s = try ctx.string(i, "value");
            return std.meta.stringToEnum(T, s) orelse {
                var names: std.Io.Writer.Allocating = .init(ctx.a);
                inline for (info.fields, 0..) |f, k| {
                    if (k > 0) names.writer.writeAll(", ") catch return error.OutOfMemory;
                    names.writer.writeAll(f.name) catch return error.OutOfMemory;
                }
                return ctx.fail("'{s}' is not one of {s}", .{ s, names.written() });
            };
        },
        .optional => |o| {
            if (ty == c.LUA_TNIL or (ty == c.LUA_TBOOLEAN and o.child != bool and c.lua_toboolean(L, i) == 0)) return null;
            if (@typeInfo(o.child) == .pointer and @typeInfo(o.child).pointer.size == .one) {
                const P = @typeInfo(o.child).pointer.child;
                const out = try ctx.a.create(P);
                out.* = try readValue(P, ctx, i, if (cur) |p| p.* else defaultOf(P), mode);
                return out;
            }
            const base: o.child = if (cur) |v| v else defaultOf(o.child);
            return try readValue(o.child, ctx, i, base, mode);
        },
        .pointer => |p| {
            if (p.size == .slice) {
                if (p.child == u8) return ctx.string(i, "value");
                if (ty != c.LUA_TTABLE) return ctx.fail("expected a list, found {s}", .{typeName(L, i)});
                const n = c.lua_rawlen(L, i);
                const out = try ctx.a.alloc(p.child, n);
                for (out, 0..) |*e, k| {
                    _ = c.lua_rawgeti(L, i, @intCast(k + 1));
                    defer c.lua_settop(L, -2);
                    e.* = try readValue(p.child, ctx, -1, if (k < cur.len) cur[k] else defaultOf(p.child), mode);
                }
                return out;
            }
            const out = try ctx.a.create(p.child);
            out.* = try readValue(p.child, ctx, i, cur.*, mode);
            return out;
        },
        .array => |arr| {
            if (ty != c.LUA_TTABLE or c.lua_rawlen(L, i) != arr.len) return ctx.fail("expected a list of {d}", .{arr.len});
            var out: T = undefined;
            for (&out, 0..) |*e, k| {
                _ = c.lua_rawgeti(L, i, @intCast(k + 1));
                defer c.lua_settop(L, -2);
                e.* = try readValue(arr.child, ctx, -1, cur[k], mode);
            }
            return out;
        },
        .@"struct" => |info| {
            if (ty != c.LUA_TTABLE) return ctx.fail("expected a table, found {s}", .{typeName(L, i)});
            var out = cur;
            // Unknown keys are mistakes (a misspelt field would be ignored).
            c.lua_pushnil(L);
            while (c.lua_next(L, i) != 0) {
                defer c.lua_settop(L, -2);
                const key = if (c.lua_type(L, -2) == c.LUA_TSTRING) try ctx.string(-2, "key") else return ctx.fail("expected named fields", .{});
                var known = false;
                inline for (info.fields) |f| {
                    if (comptime settable(f.type)) {
                        if (std.mem.eql(u8, key, f.name)) {
                            known = true;
                            if (mode == .partial) @field(out, f.name) = try readValue(f.type, ctx, -1, @field(out, f.name), mode);
                        }
                    }
                }
                if (!known) return ctx.fail("unknown field '{s}'", .{key});
            }
            if (mode == .full) {
                inline for (info.fields) |f| {
                    if (comptime settable(f.type)) {
                        _ = c.lua_getfield(L, i, f.name.ptr);
                        defer c.lua_settop(L, -2);
                        if (c.lua_type(L, -1) == c.LUA_TNIL) {
                            if (@typeInfo(f.type) == .optional) @field(out, f.name) = null else return ctx.fail("{s} was removed", .{f.name});
                        } else @field(out, f.name) = try readValue(f.type, ctx, -1, @field(out, f.name), mode);
                    }
                }
            }
            return out;
        },
        .@"union" => |info| {
            if (ty != c.LUA_TTABLE) return ctx.fail("expected a table with a type field, found {s}", .{typeName(L, i)});
            _ = c.lua_getfield(L, i, "type");
            const tag = try ctx.string(-1, "type");
            c.lua_settop(L, -2);
            inline for (info.fields) |f| {
                if (std.mem.eql(u8, tag, f.name)) {
                    switch (@typeInfo(f.type)) {
                        .void => return @unionInit(T, f.name, {}),
                        .@"struct" => |s| {
                            var payload: f.type = defaultOf(f.type);
                            inline for (s.fields) |pf| {
                                _ = c.lua_getfield(L, i, pf.name.ptr);
                                defer c.lua_settop(L, -2);
                                if (c.lua_type(L, -1) != c.LUA_TNIL) @field(payload, pf.name) = try readValue(pf.type, ctx, -1, @field(payload, pf.name), mode);
                            }
                            return @unionInit(T, f.name, payload);
                        },
                        else => {
                            _ = c.lua_getfield(L, i, "value");
                            defer c.lua_settop(L, -2);
                            return @unionInit(T, f.name, try readValue(f.type, ctx, -1, defaultOf(f.type), mode));
                        },
                    }
                }
            }
            return ctx.fail("unknown type '{s}'", .{tag});
        },
        else => @compileError("no Lua reading for " ++ @typeName(T)),
    }
}

fn defaultOf(comptime T: type) T {
    return switch (@typeInfo(T)) {
        .@"struct" => |s| blk: {
            var v: T = undefined;
            inline for (s.fields) |f| @field(v, f.name) = if (comptime f.defaultValue()) |d| d else defaultOf(f.type);
            break :blk v;
        },
        .@"union" => |u| @unionInit(T, u.fields[0].name, defaultOf(u.fields[0].type)),
        .void => {},
        .pointer => |p| if (p.size == .slice) &.{} else @compileError("no default for " ++ @typeName(T)),
        .optional => null,
        else => std.mem.zeroes(T),
    };
}

/// Pushes `v` as a Lua value: structs as tables, slices and arrays as lists,
/// enums as their names, unions as `{ type = tag, ... }`, null as nil (as
/// `false` inside a list, which cannot hold nil).
fn pushValue(comptime T: type, L: *c.lua_State, v: T, in_list: bool) void {
    switch (@typeInfo(T)) {
        .bool => c.lua_pushboolean(L, @intFromBool(v)),
        .int => |info| c.lua_pushinteger(L, if (info.signedness == .unsigned and info.bits == 64) @bitCast(v) else @intCast(v)),
        .float => c.lua_pushnumber(L, @floatCast(v)),
        .@"enum" => _ = c.lua_pushlstring(L, @tagName(v).ptr, @tagName(v).len),
        .optional => if (v) |x| pushValue(@TypeOf(x), L, x, in_list) else if (in_list) c.lua_pushboolean(L, 0) else c.lua_pushnil(L),
        .pointer => |p| {
            if (p.size == .slice) {
                if (p.child == u8) {
                    _ = c.lua_pushlstring(L, v.ptr, v.len);
                    return;
                }
                c.lua_createtable(L, @intCast(v.len), 0);
                for (v, 0..) |e, k| {
                    pushValue(p.child, L, e, true);
                    c.lua_rawseti(L, -2, @intCast(k + 1));
                }
            } else pushValue(p.child, L, v.*, in_list);
        },
        .array => |arr| {
            c.lua_createtable(L, arr.len, 0);
            for (v, 0..) |e, k| {
                pushValue(arr.child, L, e, true);
                c.lua_rawseti(L, -2, @intCast(k + 1));
            }
        },
        .@"struct" => |info| {
            c.lua_createtable(L, 0, info.fields.len);
            inline for (info.fields) |f| {
                if (comptime settable(f.type)) {
                    pushValue(f.type, L, @field(v, f.name), false);
                    c.lua_setfield(L, -2, f.name.ptr);
                }
            }
        },
        .@"union" => |info| {
            c.lua_createtable(L, 0, 4);
            inline for (info.fields) |f| {
                if (v == @field(std.meta.Tag(T), f.name)) {
                    _ = c.lua_pushstring(L, f.name.ptr);
                    c.lua_setfield(L, -2, "type");
                    const payload = @field(v, f.name);
                    switch (@typeInfo(f.type)) {
                        .void => {},
                        .@"struct" => |s| inline for (s.fields) |pf| {
                            pushValue(pf.type, L, @field(payload, pf.name), false);
                            c.lua_setfield(L, -2, pf.name.ptr);
                        },
                        else => {
                            pushValue(f.type, L, payload, false);
                            c.lua_setfield(L, -2, "value");
                        },
                    }
                }
            }
        },
        else => @compileError("no Lua value for " ++ @typeName(T)),
    }
}

fn pushJson(L: *c.lua_State, v: std.json.Value) void {
    switch (v) {
        .null => _ = c.lua_getfield(L, c.LUA_REGISTRYINDEX, "ditch_null"),
        .bool => |b| c.lua_pushboolean(L, @intFromBool(b)),
        .integer => |n| c.lua_pushinteger(L, n),
        .float => |f| c.lua_pushnumber(L, f),
        // A number std.json could not hold (an integer beyond 64 bits): ditch's
        // readers see no number there, so neither does a definition.
        .number_string => _ = c.lua_getfield(L, c.LUA_REGISTRYINDEX, "ditch_null"),
        .string => |s| _ = c.lua_pushlstring(L, s.ptr, s.len),
        .array => |arr| {
            c.lua_createtable(L, @intCast(arr.items.len), 0);
            for (arr.items, 0..) |e, k| {
                pushJson(L, e);
                c.lua_rawseti(L, -2, @intCast(k + 1));
            }
            // Marked, so that an empty array is not taken for an empty object.
            _ = c.lua_getfield(L, c.LUA_REGISTRYINDEX, "ditch_json_array");
            _ = c.lua_setmetatable(L, -2);
        },
        .object => |o| {
            c.lua_createtable(L, 0, @intCast(o.count()));
            // The keys in document order, for `keys(v)`.
            c.lua_createtable(L, 0, 2);
            _ = c.lua_pushstring(L, "json_object");
            c.lua_setfield(L, -2, "__name");
            c.lua_createtable(L, @intCast(o.count()), 0);
            var it = o.iterator();
            var k: c.lua_Integer = 1;
            while (it.next()) |kv| : (k += 1) {
                _ = c.lua_pushlstring(L, kv.key_ptr.ptr, kv.key_ptr.len);
                c.lua_rawseti(L, -2, k);
            }
            c.lua_setfield(L, -2, "order");
            _ = c.lua_setmetatable(L, -2);
            it = o.iterator();
            while (it.next()) |kv| {
                _ = c.lua_pushlstring(L, kv.key_ptr.ptr, kv.key_ptr.len);
                pushJson(L, kv.value_ptr.*);
                c.lua_settable(L, -3);
            }
        },
    }
}

// ---------------------------------------------------------------------------
// Config functions
// ---------------------------------------------------------------------------

const unsupported_marker = "\x01unsupported\x01";
const invalid_marker = "\x01invalid\x01";

fn luaWarn(L: ?*c.lua_State) callconv(.c) c_int {
    var len: usize = 0;
    const p = c.luaL_tolstring(L, 1, &len);
    std.log.warn("{s}", .{p[0..len]});
    return 0;
}

fn raiseMarked(L: ?*c.lua_State, marker: []const u8) c_int {
    var len: usize = 0;
    const p = c.luaL_tolstring(L, 1, &len);
    _ = c.lua_pushlstring(L, marker.ptr, marker.len);
    _ = c.lua_pushlstring(L, p, len);
    c.lua_concat(L, 2);
    return c.lua_error(L);
}

fn luaUnsupported(L: ?*c.lua_State) callconv(.c) c_int {
    return raiseMarked(L, unsupported_marker);
}

fn luaInvalid(L: ?*c.lua_State) callconv(.c) c_int {
    return raiseMarked(L, invalid_marker);
}

/// `require(name)`: a built-in library of src/models/lib, run once in its own
/// environment; its return value is cached.
fn luaRequire(L_opt: ?*c.lua_State) callconv(.c) c_int {
    const L = L_opt.?;
    var len: usize = 0;
    const p = c.luaL_checklstring(L, 1, &len);
    const name = p[0..len];
    _ = c.lua_getfield(L, c.LUA_REGISTRYINDEX, "ditch_libs");
    if (c.lua_type(L, -1) != c.LUA_TTABLE) {
        c.lua_settop(L, -2);
        c.lua_createtable(L, 0, 8);
        c.lua_pushvalue(L, -1);
        c.lua_setfield(L, c.LUA_REGISTRYINDEX, "ditch_libs");
    }
    const cache = c.lua_gettop(L);
    c.lua_pushvalue(L, 1);
    if (c.lua_gettable(L, cache) != c.LUA_TNIL) return 1;
    c.lua_settop(L, cache);
    for (definitions.libs) |lib| {
        if (!std.mem.eql(u8, lib.name[0 .. lib.name.len - ".lua".len], name)) continue;
        if (c.luaL_loadbufferx(L, lib.source.ptr, lib.source.len, lib.name.ptr, "t") != c.LUA_OK) return c.lua_error(L);
        c.lua_createtable(L, 0, 0);
        c.lua_createtable(L, 0, 1);
        _ = c.lua_getglobal(L, "_G");
        c.lua_setfield(L, -2, "__index");
        _ = c.lua_setmetatable(L, -2);
        _ = c.lua_setupvalue(L, -2, 1);
        c.lua_callk(L, 0, 1, 0, null);
        c.lua_pushvalue(L, 1);
        c.lua_pushvalue(L, -2);
        c.lua_settable(L, cache);
        return 1;
    }
    return c.luaL_error(L, "require: no library '%s' (the built-in ones are src/models/lib/*.lua)", p);
}

/// Converts a Lua value (as built by `pushJson`) back into JSON.
fn toJson(L: *c.lua_State, a: Allocator, idx: c_int, depth: usize) !std.json.Value {
    const i = c.lua_absindex(L, idx);
    switch (c.lua_type(L, i)) {
        c.LUA_TNIL => return .null,
        c.LUA_TBOOLEAN => return .{ .bool = c.lua_toboolean(L, i) != 0 },
        c.LUA_TNUMBER => {
            if (c.lua_isinteger(L, i) != 0) return .{ .integer = c.lua_tointegerx(L, i, null) };
            return .{ .float = c.lua_tonumberx(L, i, null) };
        },
        c.LUA_TSTRING => {
            var len: usize = 0;
            const p = c.lua_tolstring(L, i, &len);
            return .{ .string = try a.dupe(u8, p[0..len]) };
        },
        c.LUA_TTABLE => {
            if (depth > 32) return error.OutOfMemory;
            _ = c.lua_getfield(L, c.LUA_REGISTRYINDEX, "ditch_null");
            const is_null = c.lua_rawequal(L, -1, i) != 0;
            c.lua_settop(L, -2);
            if (is_null) return .null;
            const n = c.lua_rawlen(L, i);
            if (n > 0) {
                var arr = std.json.Array.init(a);
                for (0..n) |k| {
                    _ = c.lua_rawgeti(L, i, @intCast(k + 1));
                    defer c.lua_settop(L, -2);
                    try arr.append(try toJson(L, a, -1, depth + 1));
                }
                return .{ .array = arr };
            }
            var obj: std.json.ObjectMap = .empty;
            c.lua_pushnil(L);
            while (c.lua_next(L, i) != 0) {
                defer c.lua_settop(L, -2);
                if (c.lua_type(L, -2) != c.LUA_TSTRING) continue;
                var len: usize = 0;
                const p = c.lua_tolstring(L, -2, &len);
                try obj.put(a, try a.dupe(u8, p[0..len]), try toJson(L, a, -1, depth + 1));
            }
            return .{ .object = obj };
        },
        else => return .null,
    }
}

/// `rope_scaling(rs, cfg, rotary_dim, max_positions)`: ditch's reading of a
/// `rope_scaling` / `rope_parameters` table, as `c.rope_scaling` holds it.
fn luaRopeScaling(L_opt: ?*c.lua_State) callconv(.c) c_int {
    const L = L_opt.?;
    var arena_state = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    const rs = toJson(L, a, 1, 0) catch return c.luaL_error(L, "rope_scaling: out of memory");
    const cfg = toJson(L, a, 2, 0) catch return c.luaL_error(L, "rope_scaling: out of memory");
    if (rs != .object or cfg != .object) return c.luaL_error(L, "rope_scaling(rs, cfg, rotary_dim, max_positions): rs and cfg must be tables");
    const rotary: usize = @intCast(@max(0, c.luaL_checkinteger(L, 3)));
    const max_pos: usize = @intCast(@max(0, c.luaL_checkinteger(L, 4)));
    const scaling = arch.parseRopeScaling(a, cfg.object, rs.object, rotary, max_pos) catch |err| return c.luaL_error(L, "rope_scaling: %s", @errorName(err).ptr);
    pushValue(arch.RopeScaling, L, scaling, false);
    return 1;
}

/// `yarn_mscale(scale, mscale)`: YaRN's attention factor, in ditch's f32.
fn luaYarnMscale(L_opt: ?*c.lua_State) callconv(.c) c_int {
    const L = L_opt.?;
    const scale: f32 = @floatCast(c.luaL_checknumber(L, 1));
    const mscale: f32 = @floatCast(c.luaL_checknumber(L, 2));
    c.lua_pushnumber(L, arch.yarnMscale(scale, mscale));
    return 1;
}

/// `log32(x)`: the natural logarithm in single precision, as ditch computes it.
fn luaLog32(L_opt: ?*c.lua_State) callconv(.c) c_int {
    const L = L_opt.?;
    const x: f32 = @floatCast(c.luaL_checknumber(L, 1));
    c.lua_pushnumber(L, @log(x));
    return 1;
}

/// `pow(x, y)`: `x^y` in double precision, as ditch computes it.
fn luaPow(L_opt: ?*c.lua_State) callconv(.c) c_int {
    const L = L_opt.?;
    c.lua_pushnumber(L, std.math.pow(f64, c.luaL_checknumber(L, 1), c.luaL_checknumber(L, 2)));
    return 1;
}

/// Runs a definition's `config` function on a parsed configuration: `obj` is
/// the model's config.json object (its text config for a wrapper). Errors
/// from `unsupported(msg)` and `invalid(msg)` become
/// `error.UnsupportedArchitecture` and `error.InvalidConfig`.
pub fn runConfig(ref: c_int, cfg: *Config, arena: Allocator, obj: std.json.ObjectMap) !void {
    lock();
    defer state_lock.unlock();
    const r = stateLocked();
    const L = r.L;
    const top = c.lua_gettop(L);
    defer c.lua_settop(L, top);
    pushValue(Config, L, cfg.*, false);
    const t = c.lua_gettop(L);
    _ = c.lua_rawgeti(L, c.LUA_REGISTRYINDEX, ref);
    pushJson(L, .{ .object = obj });
    c.lua_pushvalue(L, t);
    if (c.lua_pcallk(L, 2, 0, 0, 0, null) != c.LUA_OK) {
        var len: usize = 0;
        const p = c.lua_tolstring(L, -1, &len);
        const msg: []const u8 = if (p) |m| m[0..len] else "unknown Lua error";
        if (std.mem.indexOf(u8, msg, unsupported_marker)) |k| {
            arch.logErr("unsupported model: {s}", .{msg[k + unsupported_marker.len ..]});
            return error.UnsupportedArchitecture;
        }
        if (std.mem.indexOf(u8, msg, invalid_marker)) |k| {
            arch.logErr("invalid config.json: {s}", .{msg[k + invalid_marker.len ..]});
            return error.InvalidConfig;
        }
        arch.logErr("{s}: config function failed: {s}", .{ cfg.arch.model_type, msg });
        return error.InvalidConfig;
    }
    var diag: Diagnostic = .{};
    var ctx = Ctx{ .L = L, .a = arena, .diag = &diag, .where = "config", .family = cfg.arch.model_type };
    const layers = cfg.num_layers;
    var out = readValue(Config, &ctx, t, cfg.*, .full) catch |err| {
        arch.logErr("{s}", .{diag.message orelse @errorName(err)});
        return error.InvalidConfig;
    };
    out.arch = cfg.arch;
    // Per-layer tables keep their length: the parser indexes them by layer.
    inline for (@typeInfo(Config).@"struct".fields) |f| {
        if (@typeInfo(f.type) == .pointer and @typeInfo(f.type).pointer.size == .slice and @typeInfo(f.type).pointer.child != u8) {
            if (@field(cfg.*, f.name).len == layers and @field(out, f.name).len != layers) {
                arch.logErr("{s}: config function changed the length of c.{s} ({d} layers)", .{ cfg.arch.model_type, f.name, layers });
                return error.InvalidConfig;
            }
        }
    }
    if (out.num_layers != layers) {
        arch.logErr("{s}: a config function cannot change num_layers", .{cfg.arch.model_type});
        return error.InvalidConfig;
    }
    cfg.* = out;
}

// ---------------------------------------------------------------------------
// Writing definitions and comparing families
// ---------------------------------------------------------------------------

/// Writes `v` as a Lua string literal.
fn writeString(w: *std.Io.Writer, s: []const u8) std.Io.Writer.Error!void {
    try w.writeByte('"');
    for (s) |ch| switch (ch) {
        '"' => try w.writeAll("\\\""),
        '\\' => try w.writeAll("\\\\"),
        '\n' => try w.writeAll("\\n"),
        '\t' => try w.writeAll("\\t"),
        0...8, 11...31, 127 => try w.print("\\{d:0>3}", .{ch}),
        else => try w.writeByte(ch),
    };
    try w.writeByte('"');
}

fn writeLuaValue(comptime T: type, w: *std.Io.Writer, v: T, indent: usize) std.Io.Writer.Error!void {
    switch (@typeInfo(T)) {
        .bool => try w.writeAll(if (v) "true" else "false"),
        .int => try w.print("{d}", .{v}),
        .float => {
            if (v == @trunc(v) and @abs(v) < 1e15) try w.print("{d}.0", .{v}) else try w.print("{e}", .{v});
        },
        .@"enum" => try writeString(w, @tagName(v)),
        .optional => if (v) |x| try writeLuaValue(@TypeOf(x), w, x, indent) else try w.writeAll("false"),
        .pointer => |p| {
            if (p.size == .slice) {
                if (p.child == u8) return writeString(w, v);
                if (v.len == 0) return w.writeAll("{}");
                try w.writeAll("{ ");
                for (v, 0..) |e, k| {
                    if (k > 0) try w.writeAll(", ");
                    try writeLuaValue(p.child, w, e, indent);
                }
                try w.writeAll(" }");
            } else try writeFields(p.child, w, v.*, defaultOf(p.child), indent);
        },
        .@"struct" => try writeFields(T, w, v, defaultOf(T), indent),
        else => @compileError("no Lua text for " ++ @typeName(T)),
    }
}

/// Writes the fields of `v` that differ from `base` as a Lua table.
fn writeFields(comptime T: type, w: *std.Io.Writer, v: T, base: T, indent: usize) std.Io.Writer.Error!void {
    try w.writeAll("{\n");
    inline for (@typeInfo(T).@"struct".fields) |f| {
        if (comptime settable(f.type)) {
            if (!valueEql(f.type, @field(v, f.name), @field(base, f.name))) {
                try w.splatByteAll(' ', indent + 2);
                try w.print("{s} = ", .{f.name});
                try writeLuaValue(f.type, w, @field(v, f.name), indent + 2);
                try w.writeAll(",\n");
            }
        }
    }
    try w.splatByteAll(' ', indent);
    try w.writeAll("}");
}

/// Writes a family as a Lua definition file (what `ditch add-model` drafts
/// and the built-in definitions were generated from). Only fields that
/// differ from the schema defaults are written; a Lua `config` function
/// cannot be written back and is noted instead.
pub fn writeDefinition(w: *std.Io.Writer, a: *const Arch) !void {
    const default: Arch = .{ .model_type = "", .llama_cpp = null };
    try w.writeAll("return {\n");
    if (a.inherits) |b| {
        try w.writeAll("  base = ");
        try writeString(w, b);
        try w.writeAll(",\n");
    }
    inline for (@typeInfo(Arch).@"struct".fields) |f| {
        if (comptime settable(f.type) and !std.mem.eql(u8, f.name, "names") and !std.mem.eql(u8, f.name, "script") and !std.mem.eql(u8, f.name, "inherits")) {
            const always = comptime std.mem.eql(u8, f.name, "model_type") or std.mem.eql(u8, f.name, "llama_cpp");
            if (always or !valueEql(f.type, @field(a.*, f.name), @field(default, f.name))) {
                try w.print("  {s} = ", .{f.name});
                if (comptime std.mem.eql(u8, f.name, "llama_cpp")) {
                    if (a.llama_cpp) |l| try writeString(w, l) else try w.writeAll("nil");
                } else try writeLuaValue(f.type, w, @field(a.*, f.name), 2);
                try w.writeAll(",\n");
            }
        }
    }
    if (!valueEql(Names, a.names, .{})) {
        try w.writeAll("  names = ");
        try writeFields(Names, w, a.names, .{}, 2);
        try w.writeAll(",\n");
    }
    if (a.extra) |f| {
        try w.writeAll("  hook = ");
        try writeString(w, arch.hookName(f) orelse "?");
        try w.writeAll(",\n");
    }
    if (a.script != null) try w.writeAll("  -- config = function(cfg, c) ... end (not reproducible from the loaded family)\n");
    try w.writeAll("}\n");
}

fn valueEql(comptime T: type, x: T, y: T) bool {
    switch (@typeInfo(T)) {
        .float => return @as(std.meta.Int(.unsigned, @bitSizeOf(T)), @bitCast(x)) == @as(std.meta.Int(.unsigned, @bitSizeOf(T)), @bitCast(y)),
        .optional => {
            if (x == null or y == null) return x == null and y == null;
            return valueEql(@TypeOf(x.?), x.?, y.?);
        },
        .pointer => |p| {
            if (p.size == .slice) {
                if (x.len != y.len) return false;
                for (x, y) |a, b| if (!valueEql(p.child, a, b)) return false;
                return true;
            }
            if (@typeInfo(p.child) == .@"fn" or p.child == Arch) return x == y;
            return valueEql(p.child, x.*, y.*);
        },
        .array => |arr| {
            for (x, y) |a, b| if (!valueEql(arr.child, a, b)) return false;
            return true;
        },
        .@"struct" => |info| {
            inline for (info.fields) |f| if (!valueEql(f.type, @field(x, f.name), @field(y, f.name))) return false;
            return true;
        },
        .@"union" => {
            if (std.meta.activeTag(x) != std.meta.activeTag(y)) return false;
            switch (x) {
                inline else => |payload, tag| return valueEql(@TypeOf(payload), payload, @field(y, @tagName(tag))),
            }
        },
        else => return x == y,
    }
}

/// Writes every field of `v` (families, parsed configurations) as one
/// `path = value` line each: the canonical form two definitions are
/// compared in.
pub fn dump(comptime T: type, w: *std.Io.Writer, v: T, path: []const u8) std.Io.Writer.Error!void {
    switch (@typeInfo(T)) {
        .bool => try w.print("{s} = {}\n", .{ path, v }),
        .int => try w.print("{s} = {d}\n", .{ path, v }),
        .float => try w.print("{s} = {e}\n", .{ path, v }),
        .@"enum" => try w.print("{s} = {s}\n", .{ path, @tagName(v) }),
        .void => try w.print("{s}\n", .{path}),
        .optional => if (v) |x| try dump(@TypeOf(x), w, x, path) else try w.print("{s} = null\n", .{path}),
        .pointer => |p| {
            if (p.size == .slice) {
                if (p.child == u8) return w.print("{s} = \"{s}\"\n", .{ path, v });
                if (@typeInfo(p.child) != .@"struct" and @typeInfo(p.child) != .pointer and @typeInfo(p.child) != .optional) {
                    try w.print("{s} = [", .{path});
                    for (v, 0..) |e, k| {
                        if (k > 0) try w.writeAll(" ");
                        switch (@typeInfo(p.child)) {
                            .bool => try w.writeAll(if (e) "1" else "0"),
                            .float => try w.print("{e}", .{e}),
                            .@"enum" => try w.writeAll(@tagName(e)),
                            else => try w.print("{d}", .{e}),
                        }
                    }
                    return w.writeAll("]\n");
                }
                if (v.len == 0) return w.print("{s} = []\n", .{path});
                var buf: [256]u8 = undefined;
                for (v, 0..) |e, k| try dump(p.child, w, e, std.fmt.bufPrint(&buf, "{s}[{d}]", .{ path, k }) catch path);
                return;
            }
            if (@typeInfo(p.child) == .@"fn") return w.print("{s} = {s}\n", .{ path, hookNameAny(v) });
            if (p.child == Arch) return w.print("{s} = {s}\n", .{ path, v.model_type });
            try dump(p.child, w, v.*, path);
        },
        .array => |arr| {
            var buf: [256]u8 = undefined;
            for (v, 0..) |e, k| try dump(arr.child, w, e, std.fmt.bufPrint(&buf, "{s}[{d}]", .{ path, k }) catch path);
        },
        .@"struct" => |info| inline for (info.fields) |f| {
            var buf: [256]u8 = undefined;
            const sub = if (path.len == 0) f.name else std.fmt.bufPrint(&buf, "{s}.{s}", .{ path, f.name }) catch f.name;
            try dump(f.type, w, @field(v, f.name), sub);
        },
        .@"union" => switch (v) {
            inline else => |payload, tag| {
                var buf: [256]u8 = undefined;
                try dump(@TypeOf(payload), w, payload, std.fmt.bufPrint(&buf, "{s}.{s}", .{ path, @tagName(tag) }) catch path);
            },
        },
        else => @compileError("no dump for " ++ @typeName(T)),
    }
}

fn hookNameAny(f: anytype) []const u8 {
    if (@TypeOf(f) == Hook) return arch.hookName(f) orelse "?";
    return "fn";
}

/// A family as its canonical dump, without the Lua function reference
/// (which differs between loads and is compared through parsed configs).
pub fn dumpFamily(a: Allocator, f: *const Arch) ![]u8 {
    var out: std.Io.Writer.Allocating = .init(a);
    var copy = f.*;
    copy.script = null;
    try dump(Arch, &out.writer, copy, "");
    return out.toOwnedSlice();
}


test "a family written as Lua loads back unchanged" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    for (families()) |f| {
        if (f.script != null) continue;
        var out: std.Io.Writer.Allocating = .init(a);
        try writeDefinition(&out.writer, f);
        var diag: Diagnostic = .{};
        const back = parseSource(out.written(), "roundtrip.lua", &diag) catch |err| {
            std.debug.print("{s}\n{s}\n", .{ out.written(), diag.message orelse @errorName(err) });
            return err;
        };
        try std.testing.expectEqual(@as(usize, 1), back.len);
        try std.testing.expectEqualStrings(try dumpFamily(a, f), try dumpFamily(a, back[0]));
    }
}

test "definitions report their mistakes" {
    const cases = [_]struct { src: []const u8, msg: []const u8 }{
        .{ .src = "return { model_type = 'x', nrom = 'rms' }", .msg = "unknown field 'nrom'" },
        .{ .src = "return { model_type = 'x', norm = 'rsm' }", .msg = "'rsm' is not one of" },
        .{ .src = "return { model_type = 'x', names = { qq = 'a' } }", .msg = "unknown field 'qq'" },
        .{ .src = "return { model_type = 'x', hook = 'nope' }", .msg = "hook 'nope' is not a Zig building block" },
        .{ .src = "return { model_type = 'x', base = 'nope' }", .msg = "base 'nope' is not a known model_type" },
        .{ .src = "return { model_type = 'x', chat = 'nope' }", .msg = "chat 'nope' is not a known template family" },
        .{ .src = "return { model_type = 'x', verified = 1 }", .msg = "expected a boolean" },
        .{ .src = "return 3", .msg = "must return a table" },
        .{ .src = "return {", .msg = "expected" },
    };
    for (cases) |case| {
        var diag: Diagnostic = .{};
        try std.testing.expectError(error.InvalidModelDefinition, parseSource(case.src, "bad.lua", &diag));
        std.testing.expect(std.mem.indexOf(u8, diag.message.?, case.msg) != null) catch |err| {
            std.debug.print("{s}\n", .{diag.message.?});
            return err;
        };
    }
}

test "base inherits a family's layout and names" {
    var diag: Diagnostic = .{};
    const got = try parseSource(
        \\return {
        \\  model_type = "my_qwen3",
        \\  base = "qwen3",
        \\  names = { gate = "mlp.w1.weight", q = false },
        \\}
    , "mine.lua", &diag);
    const parent = lookup("qwen3").?;
    const f = got[0];
    try std.testing.expectEqualStrings("my_qwen3", f.model_type);
    try std.testing.expect(!f.verified and f.aliases.len == 0);
    try std.testing.expectEqual(parent.qk_norm, f.qk_norm);
    try std.testing.expectEqualStrings(parent.names.q_norm.?, f.names.q_norm.?);
    try std.testing.expectEqualStrings("mlp.w1.weight", f.names.gate.?);
    try std.testing.expect(f.names.q == null);
    try std.testing.expectEqualStrings(parent.llama_cpp.?, f.llama_cpp.?);
}

test "the worked example of docs/models.md" {
    var diag: Diagnostic = .{};
    _ = loadSource(
        \\return {
        \\  model_type = "acme_lm_example",
        \\  base = "qwen3",
        \\  chat = "chatml",
        \\  names = { gate = "mlp.w1.weight", up = "mlp.w3.weight", down = "mlp.w2.weight" },
        \\  config = function(cfg, c)
        \\    c.residual_multiplier = num(cfg.residual_scale, 1.0)
        \\    if c.sliding_window then
        \\      each_layer(c.sliding_layers, function(i) return (i + 1) % 4 ~= 0 end)
        \\    end
        \\  end,
        \\}
    , "acme_lm.lua", &diag) catch |err| {
        std.debug.print("{s}\n", .{diag.message orelse @errorName(err)});
        return err;
    };
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const cfg = try arch.parseConfig(arena_state.allocator(),
        \\{"model_type": "acme_lm_example", "hidden_size": 64, "num_attention_heads": 4, "num_key_value_heads": 2,
        \\ "num_hidden_layers": 8, "intermediate_size": 128, "vocab_size": 100, "sliding_window": 16, "residual_scale": 0.5}
    );
    try std.testing.expectEqual(@as(f32, 0.5), cfg.residual_multiplier);
    try std.testing.expectEqual(arch.QkNorm.head, cfg.qk_norm);
    try std.testing.expectEqualStrings("mlp.w1.weight", cfg.arch.names.gate.?);
    for (cfg.sliding_layers, 0..) |s, i| try std.testing.expectEqual((i + 1) % 4 != 0, s);
}

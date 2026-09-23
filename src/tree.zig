//! The value tree a Lua configuration file, model definition or
//! reproducibility manifest evaluates to (see lua.zig): tables of strings,
//! numbers, booleans, arrays and nested tables.

const std = @import("std");
const Allocator = std.mem.Allocator;

pub const Value = union(enum) {
    string: []const u8,
    integer: i64,
    float: f64,
    boolean: bool,
    array: []Value,
    table: *Table,

    pub fn asFloat(self: Value) ?f64 {
        return switch (self) {
            .integer => |i| @floatFromInt(i),
            .float => |f| f,
            else => null,
        };
    }
};

pub const Table = struct {
    map: std.StringArrayHashMapUnmanaged(Value) = .{},

    pub fn get(self: *const Table, key: []const u8) ?Value {
        return self.map.get(key);
    }

    /// Looks up a dotted path like "scorer.KeywordRate.prompts".
    pub fn getPath(self: *const Table, path: []const u8) ?Value {
        var cur: *const Table = self;
        var it = std.mem.splitScalar(u8, path, '.');
        var last: ?Value = null;
        while (it.next()) |seg| {
            const v = cur.get(seg) orelse return null;
            last = v;
            if (it.peek() != null) {
                if (v != .table) return null;
                cur = v.table;
            }
        }
        return last;
    }

    pub fn getTable(self: *const Table, key: []const u8) ?*Table {
        const v = self.get(key) orelse return null;
        return if (v == .table) v.table else null;
    }
};

pub const Parsed = struct {
    arena: std.heap.ArenaAllocator,
    root: *Table,

    pub fn deinit(self: *Parsed) void {
        self.arena.deinit();
    }
};

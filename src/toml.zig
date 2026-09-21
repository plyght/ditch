//! A small TOML parser covering the subset used by ditch configuration files:
//! tables, dotted keys, strings (basic / literal / multi-line), integers,
//! floats, booleans, arrays (including nested and multi-line), inline tables
//! and comments.

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

pub const Error = error{ Syntax, OutOfMemory };

pub fn parse(gpa: Allocator, text: []const u8) Error!Parsed {
    var arena = std.heap.ArenaAllocator.init(gpa);
    errdefer arena.deinit();
    const a = arena.allocator();
    const root = try a.create(Table);
    root.* = .{};
    var p = Parser{ .a = a, .text = text, .pos = 0, .root = root, .current = root };
    try p.parseDocument();
    return .{ .arena = arena, .root = root };
}

const Parser = struct {
    a: Allocator,
    text: []const u8,
    pos: usize,
    root: *Table,
    current: *Table,

    fn peek(self: *Parser) ?u8 {
        return if (self.pos < self.text.len) self.text[self.pos] else null;
    }

    fn skipWs(self: *Parser) void {
        while (self.peek()) |c| {
            if (c == ' ' or c == '\t') self.pos += 1 else break;
        }
    }

    fn skipWsAndComments(self: *Parser) void {
        while (self.peek()) |c| {
            if (c == ' ' or c == '\t' or c == '\r' or c == '\n') {
                self.pos += 1;
            } else if (c == '#') {
                while (self.peek()) |d| {
                    if (d == '\n') break;
                    self.pos += 1;
                }
            } else break;
        }
    }

    fn expectLineEnd(self: *Parser) Error!void {
        self.skipWs();
        if (self.peek()) |c| {
            if (c == '#') {
                while (self.peek()) |d| {
                    if (d == '\n') break;
                    self.pos += 1;
                }
            } else if (c != '\n' and c != '\r') return error.Syntax;
        }
    }

    fn parseDocument(self: *Parser) Error!void {
        while (true) {
            self.skipWsAndComments();
            const c = self.peek() orelse return;
            if (c == '[') {
                try self.parseTableHeader();
            } else {
                try self.parseKeyValue(self.current);
                try self.expectLineEnd();
            }
        }
    }

    fn parseTableHeader(self: *Parser) Error!void {
        self.pos += 1;
        var is_array = false;
        if (self.peek() == '[') {
            is_array = true;
            self.pos += 1;
        }
        self.skipWs();
        const keys = try self.parseKeyPath();
        self.skipWs();
        if (self.peek() != ']') return error.Syntax;
        self.pos += 1;
        if (is_array) {
            if (self.peek() != ']') return error.Syntax;
            self.pos += 1;
        }
        try self.expectLineEnd();
        var t = self.root;
        for (keys, 0..) |k, i| {
            const last = i == keys.len - 1;
            if (last and is_array) {
                const gop = try t.map.getOrPut(self.a, k);
                if (!gop.found_existing) gop.value_ptr.* = .{ .array = &.{} };
                if (gop.value_ptr.* != .array) return error.Syntax;
                const nt = try self.a.create(Table);
                nt.* = .{};
                const old = gop.value_ptr.array;
                const arr = try self.a.alloc(Value, old.len + 1);
                @memcpy(arr[0..old.len], old);
                arr[old.len] = .{ .table = nt };
                gop.value_ptr.* = .{ .array = arr };
                t = nt;
            } else {
                t = try self.subTable(t, k);
            }
        }
        self.current = t;
    }

    fn subTable(self: *Parser, t: *Table, key: []const u8) Error!*Table {
        const gop = try t.map.getOrPut(self.a, key);
        if (!gop.found_existing) {
            const nt = try self.a.create(Table);
            nt.* = .{};
            gop.value_ptr.* = .{ .table = nt };
            return nt;
        }
        return switch (gop.value_ptr.*) {
            .table => |tt| tt,
            .array => |arr| if (arr.len > 0 and arr[arr.len - 1] == .table) arr[arr.len - 1].table else error.Syntax,
            else => error.Syntax,
        };
    }

    fn parseKeyPath(self: *Parser) Error![][]const u8 {
        var keys = std.ArrayList([]const u8).empty;
        while (true) {
            self.skipWs();
            const key = try self.parseKey();
            try keys.append(self.a, key);
            self.skipWs();
            if (self.peek() == '.') {
                self.pos += 1;
                continue;
            }
            break;
        }
        return keys.toOwnedSlice(self.a);
    }

    fn parseKey(self: *Parser) Error![]const u8 {
        const c = self.peek() orelse return error.Syntax;
        if (c == '"' or c == '\'') return self.parseString();
        const start = self.pos;
        while (self.peek()) |d| {
            if (std.ascii.isAlphanumeric(d) or d == '_' or d == '-') self.pos += 1 else break;
        }
        if (self.pos == start) return error.Syntax;
        return self.text[start..self.pos];
    }

    fn parseKeyValue(self: *Parser, table: *Table) Error!void {
        const keys = try self.parseKeyPath();
        self.skipWs();
        if (self.peek() != '=') return error.Syntax;
        self.pos += 1;
        self.skipWs();
        const value = try self.parseValue();
        var t = table;
        for (keys[0 .. keys.len - 1]) |k| t = try self.subTable(t, k);
        try t.map.put(self.a, keys[keys.len - 1], value);
    }

    fn parseValue(self: *Parser) Error!Value {
        const c = self.peek() orelse return error.Syntax;
        switch (c) {
            '"', '\'' => return .{ .string = try self.parseString() },
            '[' => return self.parseArray(),
            '{' => return self.parseInlineTable(),
            't', 'f' => {
                if (std.mem.startsWith(u8, self.text[self.pos..], "true")) {
                    self.pos += 4;
                    return .{ .boolean = true };
                }
                if (std.mem.startsWith(u8, self.text[self.pos..], "false")) {
                    self.pos += 5;
                    return .{ .boolean = false };
                }
                return error.Syntax;
            },
            else => return self.parseNumber(),
        }
    }

    fn parseNumber(self: *Parser) Error!Value {
        const start = self.pos;
        while (self.peek()) |d| {
            if (std.ascii.isAlphanumeric(d) or d == '_' or d == '-' or d == '+' or d == '.') self.pos += 1 else break;
        }
        if (self.pos == start) return error.Syntax;
        var buf = std.ArrayList(u8).empty;
        for (self.text[start..self.pos]) |d| if (d != '_') try buf.append(self.a, d);
        const s = buf.items;
        if (std.mem.eql(u8, s, "inf") or std.mem.eql(u8, s, "+inf")) return .{ .float = std.math.inf(f64) };
        if (std.mem.eql(u8, s, "-inf")) return .{ .float = -std.math.inf(f64) };
        if (std.mem.eql(u8, s, "nan") or std.mem.eql(u8, s, "+nan") or std.mem.eql(u8, s, "-nan")) return .{ .float = std.math.nan(f64) };
        const is_float = std.mem.indexOfAny(u8, s, ".eE") != null and !std.mem.startsWith(u8, s, "0x");
        if (is_float) {
            const f = std.fmt.parseFloat(f64, s) catch return error.Syntax;
            return .{ .float = f };
        }
        const i = std.fmt.parseInt(i64, s, 0) catch return error.Syntax;
        return .{ .integer = i };
    }

    fn parseString(self: *Parser) Error![]const u8 {
        const q = self.peek().?;
        const text = self.text;
        // Multi-line?
        if (self.pos + 2 < text.len and text[self.pos + 1] == q and text[self.pos + 2] == q) {
            self.pos += 3;
            if (self.peek() == '\n') self.pos += 1 else if (self.peek() == '\r') self.pos += 2;
            var out = std.ArrayList(u8).empty;
            while (self.pos < text.len) {
                if (text[self.pos] == q and self.pos + 2 < text.len and text[self.pos + 1] == q and text[self.pos + 2] == q) {
                    self.pos += 3;
                    return out.toOwnedSlice(self.a);
                }
                if (q == '"' and text[self.pos] == '\\') {
                    if (self.pos + 1 < text.len and (text[self.pos + 1] == '\n' or text[self.pos + 1] == '\r')) {
                        self.pos += 1;
                        while (self.peek()) |d| {
                            if (d == ' ' or d == '\t' or d == '\n' or d == '\r') self.pos += 1 else break;
                        }
                        continue;
                    }
                    try self.parseEscape(&out);
                    continue;
                }
                try out.append(self.a, text[self.pos]);
                self.pos += 1;
            }
            return error.Syntax;
        }
        self.pos += 1;
        var out = std.ArrayList(u8).empty;
        while (self.pos < text.len) {
            const d = text[self.pos];
            if (d == q) {
                self.pos += 1;
                return out.toOwnedSlice(self.a);
            }
            if (d == '\n') return error.Syntax;
            if (q == '"' and d == '\\') {
                try self.parseEscape(&out);
                continue;
            }
            try out.append(self.a, d);
            self.pos += 1;
        }
        return error.Syntax;
    }

    fn parseEscape(self: *Parser, out: *std.ArrayList(u8)) Error!void {
        // self.text[self.pos] == '\\'
        self.pos += 1;
        const e = self.peek() orelse return error.Syntax;
        self.pos += 1;
        switch (e) {
            'n' => try out.append(self.a, '\n'),
            't' => try out.append(self.a, '\t'),
            'r' => try out.append(self.a, '\r'),
            'b' => try out.append(self.a, 0x08),
            'f' => try out.append(self.a, 0x0C),
            '"' => try out.append(self.a, '"'),
            '\\' => try out.append(self.a, '\\'),
            'u', 'U' => {
                const n: usize = if (e == 'u') 4 else 8;
                if (self.pos + n > self.text.len) return error.Syntax;
                const cp = std.fmt.parseInt(u21, self.text[self.pos .. self.pos + n], 16) catch return error.Syntax;
                self.pos += n;
                var buf: [4]u8 = undefined;
                const len = std.unicode.utf8Encode(cp, &buf) catch return error.Syntax;
                try out.appendSlice(self.a, buf[0..len]);
            },
            else => return error.Syntax,
        }
    }

    fn parseArray(self: *Parser) Error!Value {
        self.pos += 1;
        var items = std.ArrayList(Value).empty;
        while (true) {
            self.skipWsAndComments();
            if (self.peek() == ']') {
                self.pos += 1;
                break;
            }
            try items.append(self.a, try self.parseValue());
            self.skipWsAndComments();
            if (self.peek() == ',') {
                self.pos += 1;
                continue;
            }
            self.skipWsAndComments();
            if (self.peek() == ']') {
                self.pos += 1;
                break;
            }
            return error.Syntax;
        }
        return .{ .array = try items.toOwnedSlice(self.a) };
    }

    fn parseInlineTable(self: *Parser) Error!Value {
        self.pos += 1;
        const t = try self.a.create(Table);
        t.* = .{};
        self.skipWs();
        if (self.peek() == '}') {
            self.pos += 1;
            return .{ .table = t };
        }
        while (true) {
            self.skipWs();
            try self.parseKeyValue(t);
            self.skipWs();
            if (self.peek() == ',') {
                self.pos += 1;
                continue;
            }
            if (self.peek() == '}') {
                self.pos += 1;
                break;
            }
            return error.Syntax;
        }
        return .{ .table = t };
    }
};

test "toml basics" {
    const src =
        \\# comment
        \\name = "ditch"
        \\n_trials = 200
        \\ratio = 0.5
        \\flag = true
        \\list = [
        \\  "a", # c
        \\  "b",
        \\]
        \\pairs = [["x", "y"], ["z", "w"]]
        \\scorers = [ { plugin = "kw", optimization = "minimize" }, { plugin = "kl" } ]
        \\max_memory = { "0" = "20GB", cpu = "64GB" }
        \\
        \\[good_prompts]
        \\dataset = "mlabonne/harmless_alpaca"
        \\split = "train[:400]"
        \\
        \\[scorer.KeywordRate.prompts]
        \\dataset = 'x'
        \\title = '''multi
        \\line'''
    ;
    var p = try parse(std.testing.allocator, src);
    defer p.deinit();
    try std.testing.expectEqualStrings("ditch", p.root.get("name").?.string);
    try std.testing.expectEqual(@as(i64, 200), p.root.get("n_trials").?.integer);
    try std.testing.expectEqual(@as(f64, 0.5), p.root.get("ratio").?.float);
    try std.testing.expect(p.root.get("flag").?.boolean);
    try std.testing.expectEqual(@as(usize, 2), p.root.get("list").?.array.len);
    try std.testing.expectEqualStrings("y", p.root.get("pairs").?.array[0].array[1].string);
    try std.testing.expectEqualStrings("kl", p.root.get("scorers").?.array[1].table.get("plugin").?.string);
    try std.testing.expectEqualStrings("20GB", p.root.get("max_memory").?.table.get("0").?.string);
    try std.testing.expectEqualStrings("train[:400]", p.root.getPath("good_prompts.split").?.string);
    try std.testing.expectEqualStrings("x", p.root.getPath("scorer.KeywordRate.prompts.dataset").?.string);
    try std.testing.expectEqualStrings("multi\nline", p.root.getPath("scorer.KeywordRate.prompts.title").?.string);
}

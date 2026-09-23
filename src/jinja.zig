//! A Jinja2 interpreter for chat templates. It covers what Hugging Face chat
//! templates use and renders them the way transformers does: jinja2's
//! sandboxed environment with `trim_blocks` and `lstrip_blocks`, loop
//! controls, the `{% generation %}` tag, a `json.dumps` `tojson` and the
//! `raise_exception` / `strftime_now` globals. Values follow Python's rules:
//! `str(None)` is "None", `/` is true division, strings index by code point,
//! `and` / `or` return an operand, and `map` / `select` return one-shot
//! generators (always true, no length).

const std = @import("std");
const Allocator = std.mem.Allocator;

pub const Error = error{ TemplateError, OutOfMemory };

/// Where a parse or render error is described (`message()`).
pub const Diagnostic = struct {
    buf: [512]u8 = undefined,
    len: usize = 0,
    /// The error came from the template's own `raise_exception`.
    raised: bool = false,

    pub fn message(self: *const Diagnostic) []const u8 {
        return self.buf[0..self.len];
    }

    fn set(self: *Diagnostic, comptime fmt: []const u8, args: anytype) error{TemplateError} {
        const s = std.fmt.bufPrint(&self.buf, fmt, args) catch self.buf[0..];
        self.len = s.len;
        return error.TemplateError;
    }
};

// ---------------------------------------------------------------------------
// Values
// ---------------------------------------------------------------------------

pub const Value = union(enum) {
    /// A missing name, key or attribute; carries the name for error messages.
    undefined: []const u8,
    none,
    boolean: bool,
    int: i64,
    float: f64,
    string: []const u8,
    list: *List,
    dict: *Dict,
    namespace: *Dict,
    macro: *Macro,
    builtin: Builtin,
    method: *Method,
    loop: *Loop,

    pub fn str(s: []const u8) Value {
        return .{ .string = s };
    }

    /// A list of `items` (not copied).
    pub fn listOf(a: Allocator, items: []const Value) !Value {
        const l = try a.create(List);
        l.* = .{ .items = .empty };
        try l.items.appendSlice(a, items);
        return .{ .list = l };
    }

    /// Converts parsed JSON (tool schemas, `--kwargs`, messages) to a value.
    pub fn fromJson(a: Allocator, j: std.json.Value) !Value {
        return switch (j) {
            .null => .none,
            .bool => |b| .{ .boolean = b },
            .integer => |i| .{ .int = i },
            .float => |f| .{ .float = f },
            .number_string => |s| .{ .float = std.fmt.parseFloat(f64, s) catch 0 },
            .string => |s| .{ .string = try a.dupe(u8, s) },
            .array => |arr| blk: {
                const l = try a.create(List);
                l.* = .{ .items = .empty };
                for (arr.items) |x| try l.items.append(a, try fromJson(a, x));
                break :blk .{ .list = l };
            },
            .object => |o| blk: {
                const d = try a.create(Dict);
                d.* = .{};
                var it = o.iterator();
                while (it.next()) |e| try d.put(a, .{ .string = try a.dupe(u8, e.key_ptr.*) }, try fromJson(a, e.value_ptr.*));
                break :blk .{ .dict = d };
            },
        };
    }

    fn isUndefined(v: Value) bool {
        return v == .undefined;
    }
};

pub const List = struct {
    items: std.ArrayList(Value),
    kind: Kind = .list,
    /// A generator that has been iterated once (iterating it again gives nothing).
    consumed: bool = false,

    pub const Kind = enum { list, tuple, generator };
};

/// An insertion-ordered mapping (Python's dict).
pub const Dict = struct {
    keys: std.ArrayList(Value) = .empty,
    values: std.ArrayList(Value) = .empty,

    pub fn get(self: *const Dict, key: Value) ?Value {
        for (self.keys.items, 0..) |k, i| if (equal(k, key)) return self.values.items[i];
        return null;
    }

    pub fn getStr(self: *const Dict, key: []const u8) ?Value {
        for (self.keys.items, 0..) |k, i| if (k == .string and std.mem.eql(u8, k.string, key)) return self.values.items[i];
        return null;
    }

    pub fn put(self: *Dict, a: Allocator, key: Value, value: Value) !void {
        for (self.keys.items, 0..) |k, i| if (equal(k, key)) {
            self.values.items[i] = value;
            return;
        };
        try self.keys.append(a, key);
        try self.values.append(a, value);
    }

    pub fn putStr(self: *Dict, a: Allocator, key: []const u8, value: Value) !void {
        return self.put(a, .{ .string = key }, value);
    }
};

pub const Macro = struct {
    name: []const u8,
    params: []const Param,
    body: []const Node,
    closure: *Scope,
    /// The body of the `{% call %}` block, for `caller()`.
    caller: ?*Macro = null,
};

pub const Method = struct {
    name: []const u8,
    self: Value,
};

pub const Loop = struct {
    index0: usize,
    items: []const Value,
    last_changed: ?[]const Value = null,
};

pub const Builtin = enum { range, dict, namespace, raise_exception, strftime_now, joiner_call };

// ---------------------------------------------------------------------------
// Lexer
// ---------------------------------------------------------------------------

const Tok = struct {
    kind: Kind,
    s: []const u8 = "",
    int: i64 = 0,
    float: f64 = 0,
    line: u32,

    const Kind = enum { text, var_begin, var_end, block_begin, block_end, name, string, int, float, op, eof };

    fn isOp(t: Tok, op: []const u8) bool {
        return t.kind == .op and std.mem.eql(u8, t.s, op);
    }

    fn isName(t: Tok, n: []const u8) bool {
        return t.kind == .name and std.mem.eql(u8, t.s, n);
    }
};

/// Python's `str.isspace` over ASCII (what `rstrip()` removes next to tags).
fn isWs(c: u8) bool {
    return c == ' ' or c == '\t' or c == '\n' or c == '\r' or c == 0x0b or c == 0x0c;
}

fn lstripWs(s: []const u8) []const u8 {
    var i: usize = 0;
    while (i < s.len and isWs(s[i])) i += 1;
    return s[i..];
}

fn rstripWs(s: []const u8) []const u8 {
    var i = s.len;
    while (i > 0 and isWs(s[i - 1])) i -= 1;
    return s[0..i];
}

const Lexer = struct {
    a: Allocator,
    src: []const u8,
    diag: *Diagnostic,
    toks: std.ArrayList(Tok) = .empty,
    pos: usize = 0,
    /// `lineAt`'s last answer, counted on from there (positions only grow).
    line_pos: usize = 0,
    line: u32 = 1,

    fn lineAt(self: *Lexer, pos: usize) u32 {
        const p = @min(pos, self.src.len);
        if (p < self.line_pos) return @intCast(std.mem.count(u8, self.src[0..p], "\n") + 1);
        self.line += @intCast(std.mem.count(u8, self.src[self.line_pos..p], "\n"));
        self.line_pos = p;
        return self.line;
    }

    fn fail(self: *Lexer, pos: usize, comptime fmt: []const u8, args: anytype) error{TemplateError} {
        var buf: [400]u8 = undefined;
        const msg = std.fmt.bufPrint(&buf, fmt, args) catch "?";
        return self.diag.set("line {d}: {s}", .{ self.lineAt(pos), msg });
    }

    fn push(self: *Lexer, t: Tok) !void {
        try self.toks.append(self.a, t);
    }

    /// Appends template text, with line breaks normalised to "\n" as jinja2 does.
    fn emitText(self: *Lexer, text: []const u8, at: usize) !void {
        if (text.len == 0) return;
        var out = text;
        if (std.mem.indexOfScalar(u8, text, '\r') != null) {
            var buf = try std.ArrayList(u8).initCapacity(self.a, text.len);
            var i: usize = 0;
            while (i < text.len) : (i += 1) {
                if (text[i] == '\r') {
                    buf.appendAssumeCapacity('\n');
                    if (i + 1 < text.len and text[i + 1] == '\n') i += 1;
                } else buf.appendAssumeCapacity(text[i]);
            }
            out = buf.items;
        }
        try self.push(.{ .kind = .text, .s = out, .line = self.lineAt(at) });
    }

    fn findOpen(self: *Lexer, from: usize) ?usize {
        var i = from;
        while (std.mem.indexOfScalarPos(u8, self.src, i, '{')) |p| {
            if (p + 1 >= self.src.len) return null;
            const c = self.src[p + 1];
            if (c == '{' or c == '%' or c == '#') return p;
            i = p + 1;
        }
        return null;
    }

    fn run(self: *Lexer) !void {
        const src = self.src;
        // jinja2's `line_starting`: the previous tag's end consumed a line break.
        var line_starting = true;
        while (true) {
            const open = self.findOpen(self.pos) orelse {
                try self.emitText(src[self.pos..], self.pos);
                break;
            };
            var text = src[self.pos..open];
            const kind = src[open + 1];
            var p = open + 2;
            var sign: u8 = 0;
            if (p < src.len and (src[p] == '-' or src[p] == '+')) {
                sign = src[p];
                p += 1;
            }
            if (sign == '-') {
                text = rstripWs(text);
            } else if (sign != '+' and kind != '{') {
                // lstrip_blocks: drop the indentation before a block or comment tag.
                const l_pos = if (std.mem.lastIndexOfScalar(u8, text, '\n')) |i| i + 1 else 0;
                if ((l_pos > 0 or line_starting) and l_pos < text.len and isAllWs(text[l_pos..])) text = text[0..l_pos];
            }
            try self.emitText(text, self.pos);
            switch (kind) {
                '#' => {
                    const end = std.mem.indexOfPos(u8, src, p, "#}") orelse return self.fail(open, "unclosed comment", .{});
                    self.pos = end + 2;
                    line_starting = self.afterTag(end > p and src[end - 1] == '-', false, true);
                },
                '%' => {
                    if (try self.rawBlock(p, open)) {
                        line_starting = false;
                        continue;
                    }
                    try self.push(.{ .kind = .block_begin, .line = self.lineAt(open) });
                    const end_sign = try self.lexExpr(p, "%}");
                    try self.push(.{ .kind = .block_end, .line = self.lineAt(self.pos) });
                    line_starting = self.afterTag(end_sign == '-', end_sign == '+', true);
                },
                else => {
                    try self.push(.{ .kind = .var_begin, .line = self.lineAt(open) });
                    const end_sign = try self.lexExpr(p, "}}");
                    try self.push(.{ .kind = .var_end, .line = self.lineAt(self.pos) });
                    line_starting = self.afterTag(end_sign == '-', false, false);
                },
            }
        }
        try self.push(.{ .kind = .eof, .line = self.lineAt(src.len) });
    }

    fn isAllWs(s: []const u8) bool {
        for (s) |c| if (!isWs(c)) return false;
        return true;
    }

    /// Whitespace handling after a tag's end (`self.pos` is just past it):
    /// `-` strips all of it, trim_blocks drops one line break after a block
    /// or comment. Returns whether what was consumed ended a line.
    fn afterTag(self: *Lexer, strip: bool, keep: bool, block: bool) bool {
        const src = self.src;
        if (strip) {
            const start = self.pos;
            while (self.pos < src.len and isWs(src[self.pos])) self.pos += 1;
            return self.pos > start and src[self.pos - 1] == '\n';
        }
        if (block and !keep and self.pos < src.len and src[self.pos] == '\n') {
            self.pos += 1;
            return true;
        }
        return false;
    }

    /// `{% raw %}...{% endraw %}`: the content is text. Returns false when the tag is not `raw`.
    fn rawBlock(self: *Lexer, p0: usize, open: usize) !bool {
        const src = self.src;
        var p = p0;
        while (p < src.len and isWs(src[p])) p += 1;
        if (!std.mem.startsWith(u8, src[p..], "raw")) return false;
        p += 3;
        while (p < src.len and isWs(src[p])) p += 1;
        var strip_after = false;
        if (p < src.len and src[p] == '-') {
            strip_after = true;
            p += 1;
        }
        if (!std.mem.startsWith(u8, src[p..], "%}")) return false;
        p += 2;
        // Find `{%[-+]? endraw [-]?%}`.
        var q = p;
        while (std.mem.indexOfPos(u8, src, q, "{%")) |s| {
            var r = s + 2;
            var strip_before = false;
            if (r < src.len and (src[r] == '-' or src[r] == '+')) {
                strip_before = src[r] == '-';
                r += 1;
            }
            while (r < src.len and isWs(src[r])) r += 1;
            if (std.mem.startsWith(u8, src[r..], "endraw")) {
                r += 6;
                while (r < src.len and isWs(src[r])) r += 1;
                var strip_end = false;
                if (r < src.len and src[r] == '-') {
                    strip_end = true;
                    r += 1;
                }
                if (!std.mem.startsWith(u8, src[r..], "%}")) return self.fail(s, "malformed endraw", .{});
                var body = src[p..s];
                if (strip_after) body = lstripWs(body);
                if (strip_before) body = rstripWs(body);
                try self.emitText(body, p);
                self.pos = r + 2;
                _ = self.afterTag(strip_end, false, true);
                return true;
            }
            q = s + 2;
        }
        return self.fail(open, "unclosed raw block", .{});
    }

    /// Lexes expression tokens up to `closer` at bracket depth 0. Leaves
    /// `self.pos` after the closer and returns its whitespace sign (0, '-' or '+').
    fn lexExpr(self: *Lexer, start: usize, comptime closer: []const u8) !u8 {
        const src = self.src;
        var p = start;
        var depth: usize = 0;
        while (true) {
            while (p < src.len and isWs(src[p])) p += 1;
            if (p >= src.len) return self.fail(start, "unclosed tag (expected '{s}')", .{closer});
            if (depth == 0) {
                if (std.mem.startsWith(u8, src[p..], closer)) {
                    self.pos = p + 2;
                    return 0;
                }
                if ((src[p] == '-' or (src[p] == '+' and closer[0] == '%')) and std.mem.startsWith(u8, src[p + 1 ..], closer)) {
                    self.pos = p + 3;
                    return src[p];
                }
            }
            const c = src[p];
            const line = self.lineAt(p);
            if (std.ascii.isAlphabetic(c) or c == '_' or c >= 0x80) {
                const s = p;
                while (p < src.len and (std.ascii.isAlphanumeric(src[p]) or src[p] == '_' or src[p] >= 0x80)) p += 1;
                try self.push(.{ .kind = .name, .s = src[s..p], .line = line });
            } else if (std.ascii.isDigit(c)) {
                p = try self.lexNumber(p, line);
            } else if (c == '\'' or c == '"') {
                const s = p + 1;
                var q = s;
                while (q < src.len and src[q] != c) : (q += 1) {
                    if (src[q] == '\\') q += 1;
                }
                if (q >= src.len) return self.fail(p, "unterminated string", .{});
                try self.push(.{ .kind = .string, .s = try unescape(self.a, src[s..q]), .line = line });
                p = q + 1;
            } else {
                const ops2 = [_][]const u8{ "//", "**", "==", "!=", "<=", ">=" };
                var len: usize = 0;
                for (ops2) |op| if (std.mem.startsWith(u8, src[p..], op)) {
                    len = 2;
                };
                if (len == 0) {
                    if (std.mem.indexOfScalar(u8, "+-/*%~<>=.,:|()[]{}", c) == null) return self.fail(p, "unexpected character '{c}'", .{c});
                    len = 1;
                }
                switch (c) {
                    '(', '[', '{' => depth += 1,
                    ')', ']', '}' => depth -|= 1,
                    else => {},
                }
                try self.push(.{ .kind = .op, .s = src[p .. p + len], .line = line });
                p += len;
            }
        }
    }

    fn lexNumber(self: *Lexer, start: usize, line: u32) !usize {
        const src = self.src;
        var p = start;
        var is_float = false;
        while (p < src.len and (std.ascii.isDigit(src[p]) or src[p] == '_')) p += 1;
        if (p + 1 < src.len and src[p] == '.' and std.ascii.isDigit(src[p + 1])) {
            is_float = true;
            p += 1;
            while (p < src.len and (std.ascii.isDigit(src[p]) or src[p] == '_')) p += 1;
        }
        if (p < src.len and (src[p] == 'e' or src[p] == 'E')) {
            var q = p + 1;
            if (q < src.len and (src[q] == '+' or src[q] == '-')) q += 1;
            if (q < src.len and std.ascii.isDigit(src[q])) {
                is_float = true;
                p = q;
                while (p < src.len and std.ascii.isDigit(src[p])) p += 1;
            }
        }
        var digits = std.ArrayList(u8).empty;
        for (src[start..p]) |ch| if (ch != '_') try digits.append(self.a, ch);
        if (is_float) {
            const f = std.fmt.parseFloat(f64, digits.items) catch return self.fail(start, "bad number", .{});
            try self.push(.{ .kind = .float, .float = f, .line = line });
        } else {
            const i = std.fmt.parseInt(i64, digits.items, 10) catch return self.fail(start, "integer out of range", .{});
            try self.push(.{ .kind = .int, .int = i, .line = line });
        }
        return p;
    }
};

/// Python's `unicode-escape` decoding of a string literal's body, as jinja2
/// applies it (unknown escapes are kept with their backslash).
fn unescape(a: Allocator, s: []const u8) ![]const u8 {
    if (std.mem.indexOfScalar(u8, s, '\\') == null and std.mem.indexOfScalar(u8, s, '\r') == null) return s;
    var out = std.ArrayList(u8).empty;
    var i: usize = 0;
    while (i < s.len) {
        const c = s[i];
        if (c == '\r') {
            try out.append(a, '\n');
            i += if (i + 1 < s.len and s[i + 1] == '\n') 2 else 1;
            continue;
        }
        if (c != '\\' or i + 1 >= s.len) {
            try out.append(a, c);
            i += 1;
            continue;
        }
        const e = s[i + 1];
        i += 2;
        switch (e) {
            '\n' => {},
            '\\' => try out.append(a, '\\'),
            '\'' => try out.append(a, '\''),
            '"' => try out.append(a, '"'),
            'a' => try out.append(a, 0x07),
            'b' => try out.append(a, 0x08),
            'f' => try out.append(a, 0x0c),
            'n' => try out.append(a, '\n'),
            'r' => try out.append(a, '\r'),
            't' => try out.append(a, '\t'),
            'v' => try out.append(a, 0x0b),
            '0'...'7' => {
                var v: u21 = e - '0';
                var n: usize = 1;
                while (n < 3 and i < s.len and s[i] >= '0' and s[i] <= '7') : (n += 1) {
                    v = v * 8 + (s[i] - '0');
                    i += 1;
                }
                try appendCodepoint(a, &out, v);
            },
            'x', 'u', 'U' => {
                const n: usize = switch (e) {
                    'x' => 2,
                    'u' => 4,
                    else => 8,
                };
                if (i + n > s.len) return error.TemplateError;
                const v = std.fmt.parseInt(u21, s[i .. i + n], 16) catch return error.TemplateError;
                i += n;
                try appendCodepoint(a, &out, v);
            },
            else => {
                try out.append(a, '\\');
                try out.append(a, e);
            },
        }
    }
    return out.items;
}

fn appendCodepoint(a: Allocator, out: *std.ArrayList(u8), cp: u21) !void {
    var buf: [4]u8 = undefined;
    const n = std.unicode.utf8Encode(cp, &buf) catch {
        // A lone surrogate: Python keeps it; UTF-8 cannot, so write U+FFFD.
        try out.appendSlice(a, "\u{fffd}");
        return;
    };
    try out.appendSlice(a, buf[0..n]);
}

// ---------------------------------------------------------------------------
// Syntax tree
// ---------------------------------------------------------------------------

const BinOp = enum { add, sub, mul, div, floordiv, mod, pow, concat, @"and", @"or", eq, ne, lt, le, gt, ge, in, not_in };

const Kwarg = struct { name: []const u8, value: *Expr };

const Args = struct {
    pos: []*Expr = &.{},
    kw: []Kwarg = &.{},
    /// `*items` / `**mapping` expanded into the call.
    star: ?*Expr = null,
    dstar: ?*Expr = null,
};

const Expr = union(enum) {
    literal: Value,
    name: []const u8,
    list: []*Expr,
    tuple: []*Expr,
    dict: []const [2]*Expr,
    getattr: struct { obj: *Expr, name: []const u8 },
    getitem: struct { obj: *Expr, key: *Expr },
    slice: struct { obj: *Expr, start: ?*Expr, stop: ?*Expr, step: ?*Expr },
    call: struct { callee: *Expr, args: Args },
    filter: struct { input: *Expr, name: []const u8, args: Args },
    @"test": struct { input: *Expr, name: []const u8, args: Args, negate: bool },
    not_: *Expr,
    neg: *Expr,
    pos: *Expr,
    binary: struct { op: BinOp, l: *Expr, r: *Expr },
    cond: struct { cond: *Expr, then: *Expr, otherwise: ?*Expr },
    /// The rendered body of a `{% set x | filter %}` / `{% filter %}` block.
    block_value,
};

const Param = struct { name: []const u8, default: ?*Expr };

/// An assignment target: a name, a tuple of names, or `namespace.attr`.
const Target = union(enum) {
    name: []const u8,
    tuple: []const []const u8,
    attr: struct { obj: []const u8, name: []const u8 },
};

const Node = union(enum) {
    text: []const u8,
    output: *Expr,
    @"if": struct { branches: []const Branch, otherwise: []const Node },
    @"for": struct { target: Target, iter: *Expr, filter: ?*Expr, body: []const Node, otherwise: []const Node },
    set: struct { target: Target, value: *Expr },
    set_block: struct { target: Target, filter: ?*Expr, body: []const Node },
    macro: struct { name: []const u8, params: []const Param, body: []const Node },
    call_block: struct { call: *Expr, params: []const Param, body: []const Node },
    filter_block: struct { filter: *Expr, body: []const Node },
    with: struct { names: []const []const u8, values: []const *Expr, body: []const Node },
    @"break",
    @"continue",
};

const Branch = struct { cond: *Expr, body: []const Node };

// ---------------------------------------------------------------------------
// Parser
// ---------------------------------------------------------------------------

// jinja2 rejects an unknown filter or test when it compiles the template, so
// these are checked while parsing; they are the ones `Interp` implements.
const known_filters = [_][]const u8{ "safe", "string", "trim", "upper", "lower", "capitalize", "title", "length", "count", "tojson", "replace", "join", "first", "last", "default", "d", "list", "items", "dictsort", "sort", "reverse", "unique", "min", "max", "sum", "map", "select", "reject", "selectattr", "rejectattr", "indent", "int", "float", "abs", "round", "center", "wordcount", "format", "attr", "pprint", "batch", "escape", "e", "forceescape" };
const known_tests = [_][]const u8{ "defined", "undefined", "none", "boolean", "true", "false", "integer", "float", "number", "string", "mapping", "iterable", "sequence", "callable", "odd", "even", "divisibleby", "lower", "upper", "escaped", "sameas", "eq", "equalto", "==", "ne", "!=", "lt", "lessthan", "<", "le", "<=", "gt", "greaterthan", ">", "ge", ">=", "in" };

fn isOneOf(name: []const u8, set: []const []const u8) bool {
    for (set) |x| if (std.mem.eql(u8, x, name)) return true;
    return false;
}

const Parser = struct {
    a: Allocator,
    toks: []const Tok,
    i: usize = 0,
    diag: *Diagnostic,

    fn cur(self: *Parser) Tok {
        return self.toks[self.i];
    }

    fn peek(self: *Parser, n: usize) Tok {
        return self.toks[@min(self.i + n, self.toks.len - 1)];
    }

    fn next(self: *Parser) Tok {
        const t = self.toks[self.i];
        if (self.i + 1 < self.toks.len) self.i += 1;
        return t;
    }

    fn fail(self: *Parser, comptime fmt: []const u8, args: anytype) error{TemplateError} {
        var buf: [400]u8 = undefined;
        const msg = std.fmt.bufPrint(&buf, fmt, args) catch "?";
        return self.diag.set("line {d}: {s}", .{ self.cur().line, msg });
    }

    fn describe(t: Tok) []const u8 {
        return switch (t.kind) {
            .name, .op => t.s,
            .string => "a string",
            .int, .float => "a number",
            .text => "text",
            .var_begin => "'{{'",
            .var_end => "'}}'",
            .block_begin => "'{%'",
            .block_end => "'%}'",
            .eof => "the end of the template",
        };
    }

    fn expectOp(self: *Parser, op: []const u8) !void {
        if (!self.cur().isOp(op)) return self.fail("expected '{s}', got {s}", .{ op, describe(self.cur()) });
        _ = self.next();
    }

    fn expectKind(self: *Parser, kind: Tok.Kind) !Tok {
        if (self.cur().kind != kind) return self.fail("expected {s}, got {s}", .{ @tagName(kind), describe(self.cur()) });
        return self.next();
    }

    fn expectName(self: *Parser) ![]const u8 {
        return (try self.expectKind(.name)).s;
    }

    fn expectKeyword(self: *Parser, n: []const u8) !void {
        if (!self.cur().isName(n)) return self.fail("expected '{s}', got {s}", .{ n, describe(self.cur()) });
        _ = self.next();
    }

    fn new(self: *Parser, e: Expr) !*Expr {
        const p = try self.a.create(Expr);
        p.* = e;
        return p;
    }

    /// Parses nodes until a block tag named in `ends` (left unconsumed after
    /// its `{%`) or the end of the template (only when `ends` is empty).
    fn subparse(self: *Parser, ends: []const []const u8) Error![]const Node {
        var nodes = std.ArrayList(Node).empty;
        while (true) {
            const t = self.cur();
            switch (t.kind) {
                .eof => {
                    if (ends.len > 0) return self.fail("missing '{{% {s} %}}'", .{ends[ends.len - 1]});
                    break;
                },
                .text => {
                    _ = self.next();
                    try nodes.append(self.a, .{ .text = t.s });
                },
                .var_begin => {
                    _ = self.next();
                    const e = try self.parseTuple(true);
                    _ = try self.expectKind(.var_end);
                    try nodes.append(self.a, .{ .output = e });
                },
                .block_begin => {
                    const name = self.peek(1);
                    if (name.kind == .name) {
                        for (ends) |e| if (std.mem.eql(u8, e, name.s)) {
                            _ = self.next();
                            return nodes.items;
                        };
                    }
                    _ = self.next();
                    try self.statement(&nodes);
                },
                else => return self.fail("unexpected {s}", .{describe(t)}),
            }
        }
        return nodes.items;
    }

    /// Consumes an end tag's name (already current) and the tag's close.
    fn endTag(self: *Parser) !void {
        _ = self.next();
        // `{% endmacro name %}` / `{% endblock name %}` may repeat the name.
        if (self.cur().kind == .name) _ = self.next();
        _ = try self.expectKind(.block_end);
    }

    fn statement(self: *Parser, nodes: *std.ArrayList(Node)) Error!void {
        const kw = try self.expectName();
        const eql = std.mem.eql;
        if (eql(u8, kw, "if")) {
            var branches = std.ArrayList(Branch).empty;
            var cond = try self.parseTuple(false);
            _ = try self.expectKind(.block_end);
            var otherwise: []const Node = &.{};
            while (true) {
                const body = try self.subparse(&.{ "elif", "else", "endif" });
                try branches.append(self.a, .{ .cond = cond, .body = body });
                const tag = self.next().s;
                if (eql(u8, tag, "elif")) {
                    cond = try self.parseTuple(false);
                    _ = try self.expectKind(.block_end);
                } else if (eql(u8, tag, "else")) {
                    _ = try self.expectKind(.block_end);
                    otherwise = try self.subparse(&.{"endif"});
                    try self.endTag();
                    break;
                } else {
                    _ = try self.expectKind(.block_end);
                    break;
                }
            }
            try nodes.append(self.a, .{ .@"if" = .{ .branches = branches.items, .otherwise = otherwise } });
        } else if (eql(u8, kw, "for")) {
            const target = try self.parseTarget(false);
            try self.expectKeyword("in");
            const iter = try self.parseTupleNoCond();
            var filter: ?*Expr = null;
            if (self.cur().isName("if")) {
                _ = self.next();
                filter = try self.parseExpression(true);
            }
            if (self.cur().isName("recursive")) return self.fail("recursive loops are not supported", .{});
            _ = try self.expectKind(.block_end);
            const body = try self.subparse(&.{ "endfor", "else" });
            var otherwise: []const Node = &.{};
            if (self.next().isName("else")) {
                _ = try self.expectKind(.block_end);
                otherwise = try self.subparse(&.{"endfor"});
                try self.endTag();
            } else _ = try self.expectKind(.block_end);
            try nodes.append(self.a, .{ .@"for" = .{ .target = target, .iter = iter, .filter = filter, .body = body, .otherwise = otherwise } });
        } else if (eql(u8, kw, "set")) {
            const target = try self.parseTarget(true);
            if (self.cur().isOp("=")) {
                _ = self.next();
                const value = try self.parseTuple(true);
                _ = try self.expectKind(.block_end);
                try nodes.append(self.a, .{ .set = .{ .target = target, .value = value } });
            } else {
                var filter: ?*Expr = null;
                if (self.cur().isOp("|")) filter = try self.parseFilterChain();
                _ = try self.expectKind(.block_end);
                const body = try self.subparse(&.{"endset"});
                try self.endTag();
                try nodes.append(self.a, .{ .set_block = .{ .target = target, .filter = filter, .body = body } });
            }
        } else if (eql(u8, kw, "macro")) {
            const name = try self.expectName();
            const params = try self.parseParams();
            _ = try self.expectKind(.block_end);
            const body = try self.subparse(&.{"endmacro"});
            try self.endTag();
            try nodes.append(self.a, .{ .macro = .{ .name = name, .params = params, .body = body } });
        } else if (eql(u8, kw, "call")) {
            var params: []const Param = &.{};
            if (self.cur().isOp("(")) params = try self.parseParams();
            const call = try self.parseExpression(true);
            if (call.* != .call) return self.fail("expected a macro call after 'call'", .{});
            _ = try self.expectKind(.block_end);
            const body = try self.subparse(&.{"endcall"});
            try self.endTag();
            try nodes.append(self.a, .{ .call_block = .{ .call = call, .params = params, .body = body } });
        } else if (eql(u8, kw, "filter")) {
            const filter = try self.parseFilterChainFrom(try self.new(.block_value), false);
            _ = try self.expectKind(.block_end);
            const body = try self.subparse(&.{"endfilter"});
            try self.endTag();
            try nodes.append(self.a, .{ .filter_block = .{ .filter = filter, .body = body } });
        } else if (eql(u8, kw, "with")) {
            var names = std.ArrayList([]const u8).empty;
            var values = std.ArrayList(*Expr).empty;
            while (self.cur().kind != .block_end) {
                if (names.items.len > 0) try self.expectOp(",");
                try names.append(self.a, try self.expectName());
                try self.expectOp("=");
                try values.append(self.a, try self.parseExpression(true));
            }
            _ = self.next();
            const body = try self.subparse(&.{"endwith"});
            try self.endTag();
            try nodes.append(self.a, .{ .with = .{ .names = names.items, .values = values.items, .body = body } });
        } else if (eql(u8, kw, "generation")) {
            // transformers' assistant-mask tag: renders its body unchanged.
            _ = try self.expectKind(.block_end);
            const body = try self.subparse(&.{"endgeneration"});
            try self.endTag();
            try nodes.appendSlice(self.a, body);
        } else if (eql(u8, kw, "break") or eql(u8, kw, "continue")) {
            _ = try self.expectKind(.block_end);
            try nodes.append(self.a, if (eql(u8, kw, "break")) .@"break" else .@"continue");
        } else {
            return self.fail("unknown tag '{s}'", .{kw});
        }
    }

    fn parseParams(self: *Parser) ![]const Param {
        try self.expectOp("(");
        var params = std.ArrayList(Param).empty;
        while (!self.cur().isOp(")")) {
            if (params.items.len > 0) try self.expectOp(",");
            if (self.cur().isOp(")")) break;
            const name = try self.expectName();
            var default: ?*Expr = null;
            if (self.cur().isOp("=")) {
                _ = self.next();
                default = try self.parseExpression(true);
            }
            try params.append(self.a, .{ .name = name, .default = default });
        }
        _ = self.next();
        return params.items;
    }

    fn parseTarget(self: *Parser, with_namespace: bool) !Target {
        if (with_namespace and self.cur().kind == .name and self.peek(1).isOp(".") and self.peek(2).kind == .name) {
            const obj = self.next().s;
            _ = self.next();
            return .{ .attr = .{ .obj = obj, .name = self.next().s } };
        }
        const paren = self.cur().isOp("(");
        if (paren) _ = self.next();
        var names = std.ArrayList([]const u8).empty;
        try names.append(self.a, try self.expectName());
        var tuple = paren;
        while (self.cur().isOp(",")) {
            _ = self.next();
            tuple = true;
            if (self.cur().kind != .name) break;
            try names.append(self.a, try self.expectName());
        }
        if (paren) try self.expectOp(")");
        return if (tuple) .{ .tuple = names.items } else .{ .name = names.items[0] };
    }

    /// Whether the current token ends a tuple (`a, b` without parentheses).
    fn tupleEnds(self: *Parser) bool {
        const t = self.cur();
        return switch (t.kind) {
            .var_end, .block_end, .eof => true,
            .op => t.isOp(")") or t.isOp("]") or t.isOp("}") or t.isOp("="),
            .name => t.isName("in") or t.isName("if") or t.isName("recursive"),
            else => false,
        };
    }

    fn parseTuple(self: *Parser, with_condexpr: bool) Error!*Expr {
        var items = std.ArrayList(*Expr).empty;
        var saw_comma = false;
        while (true) {
            if (items.items.len > 0) {
                if (!self.cur().isOp(",")) break;
                _ = self.next();
                saw_comma = true;
            }
            if (self.tupleEnds()) break;
            try items.append(self.a, try self.parseExpression(with_condexpr));
        }
        if (!saw_comma) {
            if (items.items.len == 0) return self.fail("expected an expression, got {s}", .{describe(self.cur())});
            return items.items[0];
        }
        return self.new(.{ .tuple = items.items });
    }

    fn parseTupleNoCond(self: *Parser) Error!*Expr {
        return self.parseTuple(false);
    }

    fn parseExpression(self: *Parser, with_condexpr: bool) Error!*Expr {
        if (!with_condexpr) return self.parseOr();
        var e = try self.parseOr();
        while (self.cur().isName("if")) {
            _ = self.next();
            const c = try self.parseOr();
            var otherwise: ?*Expr = null;
            if (self.cur().isName("else")) {
                _ = self.next();
                otherwise = try self.parseExpression(true);
            }
            e = try self.new(.{ .cond = .{ .cond = c, .then = e, .otherwise = otherwise } });
        }
        return e;
    }

    fn binary(self: *Parser, op: BinOp, l: *Expr, r: *Expr) !*Expr {
        return self.new(.{ .binary = .{ .op = op, .l = l, .r = r } });
    }

    fn parseOr(self: *Parser) Error!*Expr {
        var l = try self.parseAnd();
        while (self.cur().isName("or")) {
            _ = self.next();
            l = try self.binary(.@"or", l, try self.parseAnd());
        }
        return l;
    }

    fn parseAnd(self: *Parser) Error!*Expr {
        var l = try self.parseNot();
        while (self.cur().isName("and")) {
            _ = self.next();
            l = try self.binary(.@"and", l, try self.parseNot());
        }
        return l;
    }

    fn parseNot(self: *Parser) Error!*Expr {
        if (self.cur().isName("not")) {
            _ = self.next();
            return self.new(.{ .not_ = try self.parseNot() });
        }
        return self.parseCompare();
    }

    fn parseCompare(self: *Parser) Error!*Expr {
        const first = try self.parseMath1();
        var result: ?*Expr = null;
        var left = first;
        while (true) {
            const t = self.cur();
            var op: BinOp = undefined;
            if (t.isOp("==")) op = .eq else if (t.isOp("!=")) op = .ne else if (t.isOp("<")) op = .lt else if (t.isOp("<=")) op = .le else if (t.isOp(">")) op = .gt else if (t.isOp(">=")) op = .ge else if (t.isName("in")) op = .in else if (t.isName("not") and self.peek(1).isName("in")) {
                _ = self.next();
                op = .not_in;
            } else break;
            _ = self.next();
            const right = try self.parseMath1();
            const cmp = try self.binary(op, left, right);
            // `a < b < c` is `a < b and b < c`.
            result = if (result) |r| try self.binary(.@"and", r, cmp) else cmp;
            left = right;
        }
        return result orelse first;
    }

    fn parseMath1(self: *Parser) Error!*Expr {
        var l = try self.parseConcat();
        while (true) {
            const op: BinOp = if (self.cur().isOp("+")) .add else if (self.cur().isOp("-")) .sub else break;
            _ = self.next();
            l = try self.binary(op, l, try self.parseConcat());
        }
        return l;
    }

    fn parseConcat(self: *Parser) Error!*Expr {
        var l = try self.parseMath2();
        while (self.cur().isOp("~")) {
            _ = self.next();
            l = try self.binary(.concat, l, try self.parseMath2());
        }
        return l;
    }

    fn parseMath2(self: *Parser) Error!*Expr {
        var l = try self.parsePow();
        while (true) {
            const t = self.cur();
            const op: BinOp = if (t.isOp("*")) .mul else if (t.isOp("/")) .div else if (t.isOp("//")) .floordiv else if (t.isOp("%")) .mod else break;
            _ = self.next();
            l = try self.binary(op, l, try self.parsePow());
        }
        return l;
    }

    fn parsePow(self: *Parser) Error!*Expr {
        var l = try self.parseUnary(true);
        while (self.cur().isOp("**")) {
            _ = self.next();
            l = try self.binary(.pow, l, try self.parseUnary(true));
        }
        return l;
    }

    fn parseUnary(self: *Parser, with_filter: bool) Error!*Expr {
        var e: *Expr = undefined;
        if (self.cur().isOp("-")) {
            _ = self.next();
            e = try self.new(.{ .neg = try self.parseUnary(false) });
        } else if (self.cur().isOp("+")) {
            _ = self.next();
            e = try self.new(.{ .pos = try self.parseUnary(false) });
        } else e = try self.parsePrimary();
        e = try self.parsePostfix(e);
        if (with_filter) e = try self.parseFilterExpr(e);
        return e;
    }

    fn parsePrimary(self: *Parser) Error!*Expr {
        const t = self.next();
        switch (t.kind) {
            .name => {
                const eql = std.mem.eql;
                if (eql(u8, t.s, "true") or eql(u8, t.s, "True")) return self.new(.{ .literal = .{ .boolean = true } });
                if (eql(u8, t.s, "false") or eql(u8, t.s, "False")) return self.new(.{ .literal = .{ .boolean = false } });
                if (eql(u8, t.s, "none") or eql(u8, t.s, "None")) return self.new(.{ .literal = .none });
                return self.new(.{ .name = t.s });
            },
            .string => {
                // Adjacent string literals concatenate.
                if (self.cur().kind != .string) return self.new(.{ .literal = .{ .string = t.s } });
                var buf = std.ArrayList(u8).empty;
                try buf.appendSlice(self.a, t.s);
                while (self.cur().kind == .string) try buf.appendSlice(self.a, self.next().s);
                return self.new(.{ .literal = .{ .string = buf.items } });
            },
            .int => return self.new(.{ .literal = .{ .int = t.int } }),
            .float => return self.new(.{ .literal = .{ .float = t.float } }),
            .op => {
                if (t.isOp("(")) {
                    if (self.cur().isOp(")")) {
                        _ = self.next();
                        return self.new(.{ .tuple = &.{} });
                    }
                    const first = try self.parseExpression(true);
                    if (self.cur().isOp(")")) {
                        _ = self.next();
                        return first;
                    }
                    var items = std.ArrayList(*Expr).empty;
                    try items.append(self.a, first);
                    while (self.cur().isOp(",")) {
                        _ = self.next();
                        if (self.cur().isOp(")")) break;
                        try items.append(self.a, try self.parseExpression(true));
                    }
                    try self.expectOp(")");
                    return self.new(.{ .tuple = items.items });
                }
                if (t.isOp("[")) {
                    var items = std.ArrayList(*Expr).empty;
                    while (!self.cur().isOp("]")) {
                        if (items.items.len > 0) {
                            try self.expectOp(",");
                            if (self.cur().isOp("]")) break;
                        }
                        try items.append(self.a, try self.parseExpression(true));
                    }
                    _ = self.next();
                    return self.new(.{ .list = items.items });
                }
                if (t.isOp("{")) {
                    var items = std.ArrayList([2]*Expr).empty;
                    while (!self.cur().isOp("}")) {
                        if (items.items.len > 0) {
                            try self.expectOp(",");
                            if (self.cur().isOp("}")) break;
                        }
                        const k = try self.parseExpression(true);
                        try self.expectOp(":");
                        try items.append(self.a, .{ k, try self.parseExpression(true) });
                    }
                    _ = self.next();
                    return self.new(.{ .dict = items.items });
                }
            },
            else => {},
        }
        self.i -= 1;
        return self.fail("unexpected {s}", .{describe(t)});
    }

    fn parsePostfix(self: *Parser, e0: *Expr) Error!*Expr {
        var e = e0;
        while (true) {
            const t = self.cur();
            if (t.isOp(".")) {
                _ = self.next();
                const n = self.next();
                if (n.kind == .name) {
                    e = try self.new(.{ .getattr = .{ .obj = e, .name = n.s } });
                } else if (n.kind == .int) {
                    e = try self.new(.{ .getitem = .{ .obj = e, .key = try self.new(.{ .literal = .{ .int = n.int } }) } });
                } else return self.fail("expected an attribute name after '.'", .{});
            } else if (t.isOp("[")) {
                _ = self.next();
                e = try self.parseSubscript(e);
                try self.expectOp("]");
            } else if (t.isOp("(")) {
                e = try self.new(.{ .call = .{ .callee = e, .args = try self.parseCallArgs() } });
            } else break;
        }
        return e;
    }

    fn parseSubscript(self: *Parser, obj: *Expr) Error!*Expr {
        var start: ?*Expr = null;
        if (!self.cur().isOp(":")) {
            const k = try self.parseExpression(true);
            if (!self.cur().isOp(":")) return self.new(.{ .getitem = .{ .obj = obj, .key = k } });
            start = k;
        }
        _ = self.next();
        var stop: ?*Expr = null;
        var step: ?*Expr = null;
        if (!self.cur().isOp("]") and !self.cur().isOp(":")) stop = try self.parseExpression(true);
        if (self.cur().isOp(":")) {
            _ = self.next();
            if (!self.cur().isOp("]")) step = try self.parseExpression(true);
        }
        return self.new(.{ .slice = .{ .obj = obj, .start = start, .stop = stop, .step = step } });
    }

    fn parseCallArgs(self: *Parser) Error!Args {
        try self.expectOp("(");
        var pos = std.ArrayList(*Expr).empty;
        var kw = std.ArrayList(Kwarg).empty;
        var args: Args = .{};
        var count: usize = 0;
        while (!self.cur().isOp(")")) : (count += 1) {
            if (count > 0) {
                try self.expectOp(",");
                if (self.cur().isOp(")")) break;
            }
            if (self.cur().isOp("*")) {
                _ = self.next();
                args.star = try self.parseExpression(true);
            } else if (self.cur().isOp("**")) {
                _ = self.next();
                args.dstar = try self.parseExpression(true);
            } else if (self.cur().kind == .name and self.peek(1).isOp("=")) {
                const n = self.next().s;
                _ = self.next();
                try kw.append(self.a, .{ .name = n, .value = try self.parseExpression(true) });
            } else {
                if (kw.items.len > 0) return self.fail("positional argument after a keyword argument", .{});
                try pos.append(self.a, try self.parseExpression(true));
            }
        }
        _ = self.next();
        args.pos = pos.items;
        args.kw = kw.items;
        return args;
    }

    fn parseFilterExpr(self: *Parser, e0: *Expr) Error!*Expr {
        var e = e0;
        while (true) {
            if (self.cur().isOp("|")) {
                e = try self.parseFilterChainFrom(e, true);
            } else if (self.cur().isName("is")) {
                _ = self.next();
                var negate = false;
                if (self.cur().isName("not")) {
                    _ = self.next();
                    negate = true;
                }
                const name = try self.expectName();
                if (!isOneOf(name, &known_tests)) return self.fail("no test named '{s}'", .{name});
                var args: Args = .{};
                const t = self.cur();
                if (t.isOp("(")) {
                    args = try self.parseCallArgs();
                } else if ((t.kind == .name or t.kind == .string or t.kind == .int or t.kind == .float or t.isOp("[") or t.isOp("{")) and
                    !(t.isName("else") or t.isName("or") or t.isName("and")))
                {
                    if (t.isName("is")) return self.fail("tests cannot be chained with 'is'", .{});
                    const arg = try self.parsePostfix(try self.parsePrimary());
                    const one = try self.a.alloc(*Expr, 1);
                    one[0] = arg;
                    args = .{ .pos = one };
                }
                e = try self.new(.{ .@"test" = .{ .input = e, .name = name, .args = args, .negate = negate } });
            } else if (self.cur().isOp("(")) {
                e = try self.new(.{ .call = .{ .callee = e, .args = try self.parseCallArgs() } });
            } else break;
        }
        return e;
    }

    /// `| name(args) | name ...` applied to `input` (the current token is `|`
    /// when `leading_pipe`, otherwise the first filter's name).
    fn parseFilterChainFrom(self: *Parser, input: *Expr, leading_pipe: bool) Error!*Expr {
        var e = input;
        var first = true;
        while (true) {
            if (!first or leading_pipe) {
                if (!self.cur().isOp("|")) break;
                _ = self.next();
            }
            first = false;
            var name = try self.expectName();
            while (self.cur().isOp(".") and self.peek(1).kind == .name) {
                _ = self.next();
                name = try std.fmt.allocPrint(self.a, "{s}.{s}", .{ name, self.next().s });
            }
            if (!isOneOf(name, &known_filters)) return self.fail("no filter named '{s}'", .{name});
            var args: Args = .{};
            if (self.cur().isOp("(")) args = try self.parseCallArgs();
            e = try self.new(.{ .filter = .{ .input = e, .name = name, .args = args } });
            if (!self.cur().isOp("|")) break;
        }
        return e;
    }

    fn parseFilterChain(self: *Parser) Error!*Expr {
        return self.parseFilterChainFrom(try self.new(.block_value), true);
    }
};

// ---------------------------------------------------------------------------
// Templates
// ---------------------------------------------------------------------------

pub const Template = struct {
    arena: std.heap.ArenaAllocator,
    nodes: []const Node,

    /// Parses `source`; on error.TemplateError `diag` says where and why.
    pub fn parse(gpa: Allocator, source: []const u8, diag: *Diagnostic) Error!*Template {
        const self = try gpa.create(Template);
        self.* = .{ .arena = .init(gpa), .nodes = &.{} };
        errdefer self.deinit();
        const a = self.arena.allocator();
        var src = try a.dupe(u8, source);
        // keep_trailing_newline=False: one final line break is dropped.
        if (std.mem.endsWith(u8, src, "\r\n")) src = src[0 .. src.len - 2] else if (src.len > 0 and (src[src.len - 1] == '\n' or src[src.len - 1] == '\r')) src = src[0 .. src.len - 1];
        var lx: Lexer = .{ .a = a, .src = src, .diag = diag };
        lx.run() catch |e| return if (e == error.OutOfMemory) error.OutOfMemory else error.TemplateError;
        var p: Parser = .{ .a = a, .toks = lx.toks.items, .diag = diag };
        self.nodes = try p.subparse(&.{});
        return self;
    }

    pub fn deinit(self: *Template) void {
        const gpa = self.arena.child_allocator;
        self.arena.deinit();
        gpa.destroy(self);
    }

    /// Renders with `vars` as the context (values built with `arena`, which
    /// must outlive the call). The result is allocated with `arena` too.
    pub fn render(self: *const Template, arena: Allocator, vars: []const Var, options: Options, diag: *Diagnostic) Error![]const u8 {
        const root = try Scope.create(arena, null);
        for (vars) |v| try root.vars.put(arena, v.name, v.value);
        var it: Interp = .{ .a = arena, .root = root, .diag = diag, .now = options.now };
        var out = std.ArrayList(u8).empty;
        it.out = &out;
        _ = try it.body(self.nodes, root);
        return out.items;
    }
};

pub const Var = struct { name: []const u8, value: Value };

pub const Options = struct {
    /// Seconds since the Unix epoch for `strftime_now` (UTC).
    now: i64 = 0,
};

const Scope = struct {
    parent: ?*Scope,
    vars: std.StringHashMapUnmanaged(Value) = .empty,

    fn create(a: Allocator, parent: ?*Scope) !*Scope {
        const s = try a.create(Scope);
        s.* = .{ .parent = parent };
        return s;
    }

    fn lookup(self: *Scope, name: []const u8) ?Value {
        var s: ?*Scope = self;
        while (s) |sc| : (s = sc.parent) if (sc.vars.get(name)) |v| return v;
        return null;
    }
};

const Flow = enum { normal, @"break", @"continue" };

const Interp = struct {
    a: Allocator,
    root: *Scope,
    diag: *Diagnostic,
    now: i64,
    out: *std.ArrayList(u8) = undefined,
    block_value: Value = .none,
    depth: usize = 0,

    fn fail(self: *Interp, comptime fmt: []const u8, args: anytype) error{TemplateError} {
        return self.diag.set(fmt, args);
    }

    fn write(self: *Interp, s: []const u8) !void {
        try self.out.appendSlice(self.a, s);
    }

    fn body(self: *Interp, nodes: []const Node, scope: *Scope) Error!Flow {
        for (nodes) |*n| {
            const f = try self.node(n, scope);
            if (f != .normal) return f;
        }
        return .normal;
    }

    /// Renders `nodes` into a fresh buffer and returns it.
    fn capture(self: *Interp, nodes: []const Node, scope: *Scope) Error![]const u8 {
        var buf = std.ArrayList(u8).empty;
        const saved = self.out;
        self.out = &buf;
        defer self.out = saved;
        _ = try self.body(nodes, scope);
        return buf.items;
    }

    fn node(self: *Interp, n: *const Node, scope: *Scope) Error!Flow {
        switch (n.*) {
            .text => |t| try self.write(t),
            .output => |e| {
                const v = try self.eval(e, scope);
                try self.writeStr(v);
            },
            .@"if" => |s| {
                for (s.branches) |b| {
                    if (truthy(try self.eval(b.cond, scope))) return self.body(b.body, scope);
                }
                return self.body(s.otherwise, scope);
            },
            .@"for" => |s| return self.forLoop(s.target, s.iter, s.filter, s.body, s.otherwise, scope),
            .set => |s| try self.assign(s.target, try self.eval(s.value, scope), scope),
            .set_block => |s| {
                var v: Value = .{ .string = try self.capture(s.body, scope) };
                if (s.filter) |f| v = try self.withBlockValue(f, v, scope);
                try self.assign(s.target, v, scope);
            },
            .macro => |m| {
                const mac = try self.a.create(Macro);
                mac.* = .{ .name = m.name, .params = m.params, .body = m.body, .closure = scope };
                try scope.vars.put(self.a, m.name, .{ .macro = mac });
            },
            .call_block => |c| {
                const caller = try self.a.create(Macro);
                caller.* = .{ .name = "caller", .params = c.params, .body = c.body, .closure = scope };
                const cexpr = c.call.call;
                const callee = try self.eval(cexpr.callee, scope);
                if (callee != .macro) return self.fail("'call' needs a macro", .{});
                const with_caller = try self.a.create(Macro);
                with_caller.* = callee.macro.*;
                with_caller.caller = caller;
                const args = try self.evalArgs(cexpr.args, scope);
                const v = try self.callMacro(with_caller, args);
                try self.writeStr(v);
            },
            .filter_block => |f| {
                const v = try self.withBlockValue(f.filter, .{ .string = try self.capture(f.body, scope) }, scope);
                try self.writeStr(v);
            },
            .with => |w| {
                const inner = try Scope.create(self.a, scope);
                for (w.names, w.values) |name, e| try inner.vars.put(self.a, name, try self.eval(e, scope));
                return self.body(w.body, inner);
            },
            .@"break" => return .@"break",
            .@"continue" => return .@"continue",
        }
        return .normal;
    }

    fn withBlockValue(self: *Interp, filter: *Expr, v: Value, scope: *Scope) Error!Value {
        const saved = self.block_value;
        self.block_value = v;
        defer self.block_value = saved;
        return self.eval(filter, scope);
    }

    fn writeStr(self: *Interp, v: Value) !void {
        switch (v) {
            .string => |s| try self.write(s),
            else => try self.write(try self.toStr(v)),
        }
    }

    fn assign(self: *Interp, target: Target, v: Value, scope: *Scope) Error!void {
        switch (target) {
            .name => |n| try scope.vars.put(self.a, n, v),
            .tuple => |names| {
                const items = try self.iterate(v);
                if (items.len != names.len) return self.fail("cannot unpack {d} values into {d} names", .{ items.len, names.len });
                for (names, items) |n, x| try scope.vars.put(self.a, n, x);
            },
            .attr => |t| {
                const obj = scope.lookup(t.obj) orelse return self.fail("'{s}' is undefined", .{t.obj});
                if (obj != .namespace) return self.fail("cannot assign attribute on non-namespace object", .{});
                try obj.namespace.putStr(self.a, t.name, v);
            },
        }
    }

    fn forLoop(self: *Interp, target: Target, iter_e: *Expr, filter: ?*Expr, nodes: []const Node, otherwise: []const Node, scope: *Scope) Error!Flow {
        const iter_v = try self.eval(iter_e, scope);
        var items = try self.iterate(iter_v);
        if (filter) |f| {
            // `for x in xs if cond`: the condition sees the loop variable but not `loop`.
            var kept = std.ArrayList(Value).empty;
            for (items) |x| {
                const s = try Scope.create(self.a, scope);
                try self.assign(target, x, s);
                if (truthy(try self.eval(f, s))) try kept.append(self.a, x);
            }
            items = kept.items;
        }
        if (items.len == 0) return self.body(otherwise, scope);
        const loop = try self.a.create(Loop);
        loop.* = .{ .index0 = 0, .items = items };
        for (items, 0..) |x, i| {
            loop.index0 = i;
            // Each iteration gets its own scope: a `set` inside the loop does not outlive it.
            const s = try Scope.create(self.a, scope);
            try self.assign(target, x, s);
            try s.vars.put(self.a, "loop", .{ .loop = loop });
            const f = try self.body(nodes, s);
            if (f == .@"break") break;
        }
        return .normal;
    }

    /// The items a `for` loop (or `list()`) sees: a dict yields its keys, a
    /// string its characters, undefined nothing. A generator can be iterated once.
    fn iterate(self: *Interp, v: Value) Error![]const Value {
        switch (v) {
            .list => |l| {
                if (l.kind == .generator) {
                    if (l.consumed) return &.{};
                    l.consumed = true;
                }
                return l.items.items;
            },
            .dict => |d| return d.keys.items,
            .string => |s| {
                var out = std.ArrayList(Value).empty;
                var it = Utf8Iter{ .s = s };
                while (it.next()) |c| try out.append(self.a, .{ .string = c });
                return out.items;
            },
            .undefined => return &.{},
            else => return self.fail("'{s}' object is not iterable", .{typeName(v)}),
        }
    }

    // ----- expressions -----

    fn eval(self: *Interp, e: *const Expr, scope: *Scope) Error!Value {
        switch (e.*) {
            .literal => |v| return v,
            .name => |n| return scope.lookup(n) orelse self.global(n),
            .list, .tuple => |items| {
                const l = try self.a.create(List);
                l.* = .{ .items = .empty, .kind = if (e.* == .tuple) .tuple else .list };
                for (items) |x| try l.items.append(self.a, try self.eval(x, scope));
                return .{ .list = l };
            },
            .dict => |pairs| {
                const d = try self.a.create(Dict);
                d.* = .{};
                for (pairs) |p| try d.put(self.a, try self.eval(p[0], scope), try self.eval(p[1], scope));
                return .{ .dict = d };
            },
            .getattr => |g| return self.getAttr(try self.eval(g.obj, scope), g.name),
            .getitem => |g| return self.getItem(try self.eval(g.obj, scope), try self.eval(g.key, scope)),
            .slice => |s| {
                const obj = try self.eval(s.obj, scope);
                const start = if (s.start) |x| try self.eval(x, scope) else Value.none;
                const stop = if (s.stop) |x| try self.eval(x, scope) else Value.none;
                const step = if (s.step) |x| try self.eval(x, scope) else Value.none;
                return self.slice(obj, start, stop, step);
            },
            .call => |c| {
                const callee = try self.eval(c.callee, scope);
                const args = try self.evalArgs(c.args, scope);
                return self.call(callee, args);
            },
            .filter => |f| {
                const input = try self.eval(f.input, scope);
                const args = try self.evalArgs(f.args, scope);
                return self.applyFilter(f.name, input, args);
            },
            .@"test" => |t| {
                const input = try self.eval(t.input, scope);
                const args = try self.evalArgs(t.args, scope);
                const r = try self.applyTest(t.name, input, args);
                return .{ .boolean = r != t.negate };
            },
            .not_ => |x| return .{ .boolean = !truthy(try self.eval(x, scope)) },
            .neg => |x| {
                const v = try self.eval(x, scope);
                return switch (v) {
                    .int => |i| .{ .int = -i },
                    .float => |f| .{ .float = -f },
                    .boolean => |b| .{ .int = -@as(i64, @intFromBool(b)) },
                    else => self.fail("bad operand type for unary -: '{s}'", .{typeName(v)}),
                };
            },
            .pos => |x| {
                const v = try self.eval(x, scope);
                if (v != .int and v != .float) return self.fail("bad operand type for unary +: '{s}'", .{typeName(v)});
                return v;
            },
            .binary => |b| {
                if (b.op == .@"and") {
                    const l = try self.eval(b.l, scope);
                    return if (!truthy(l)) l else self.eval(b.r, scope);
                }
                if (b.op == .@"or") {
                    const l = try self.eval(b.l, scope);
                    return if (truthy(l)) l else self.eval(b.r, scope);
                }
                return self.binop(b.op, try self.eval(b.l, scope), try self.eval(b.r, scope));
            },
            .cond => |c| {
                if (truthy(try self.eval(c.cond, scope))) return self.eval(c.then, scope);
                if (c.otherwise) |o| return self.eval(o, scope);
                return .{ .undefined = "" };
            },
            .block_value => return self.block_value,
        }
    }

    fn global(self: *Interp, n: []const u8) Value {
        _ = self;
        const eql = std.mem.eql;
        if (eql(u8, n, "range")) return .{ .builtin = .range };
        if (eql(u8, n, "dict")) return .{ .builtin = .dict };
        if (eql(u8, n, "namespace")) return .{ .builtin = .namespace };
        if (eql(u8, n, "raise_exception")) return .{ .builtin = .raise_exception };
        if (eql(u8, n, "strftime_now")) return .{ .builtin = .strftime_now };
        return .{ .undefined = n };
    }

    const CallArgs = struct {
        pos: []const Value,
        kw: []const KwValue,

        fn get(self: CallArgs, i: usize, name: []const u8) ?Value {
            if (i < self.pos.len) return self.pos[i];
            for (self.kw) |k| if (std.mem.eql(u8, k.name, name)) return k.value;
            return null;
        }
    };
    const KwValue = struct { name: []const u8, value: Value };

    fn evalArgs(self: *Interp, args: Args, scope: *Scope) Error!CallArgs {
        var pos = std.ArrayList(Value).empty;
        for (args.pos) |x| try pos.append(self.a, try self.eval(x, scope));
        if (args.star) |x| try pos.appendSlice(self.a, try self.iterate(try self.eval(x, scope)));
        var kw = std.ArrayList(KwValue).empty;
        for (args.kw) |k| try kw.append(self.a, .{ .name = k.name, .value = try self.eval(k.value, scope) });
        if (args.dstar) |x| {
            const d = try self.eval(x, scope);
            if (d != .dict) return self.fail("argument after ** must be a mapping, not {s}", .{typeName(d)});
            for (d.dict.keys.items, d.dict.values.items) |k, v| try kw.append(self.a, .{ .name = try self.toStr(k), .value = v });
        }
        return .{ .pos = pos.items, .kw = kw.items };
    }

    fn call(self: *Interp, callee: Value, args: CallArgs) Error!Value {
        switch (callee) {
            .macro => |m| return self.callMacro(m, args),
            .method => |m| return self.callMethod(m.self, m.name, args),
            .builtin => |b| switch (b) {
                .range => {
                    var start: i64 = 0;
                    var stop: i64 = 0;
                    var step: i64 = 1;
                    const p = args.pos;
                    if (p.len == 1) stop = try self.toInt(p[0]) else if (p.len >= 2) {
                        start = try self.toInt(p[0]);
                        stop = try self.toInt(p[1]);
                        if (p.len >= 3) step = try self.toInt(p[2]);
                    } else return self.fail("range expected at least 1 argument", .{});
                    if (step == 0) return self.fail("range() arg 3 must not be zero", .{});
                    const l = try self.a.create(List);
                    l.* = .{ .items = .empty };
                    var i = start;
                    while ((step > 0 and i < stop) or (step < 0 and i > stop)) : (i += step) {
                        if (l.items.items.len > 1_000_000) return self.fail("range too large", .{});
                        try l.items.append(self.a, .{ .int = i });
                    }
                    return .{ .list = l };
                },
                .dict, .namespace => {
                    const d = try self.a.create(Dict);
                    d.* = .{};
                    for (args.pos) |p| {
                        if (p != .dict) return self.fail("{s}() takes a mapping", .{@tagName(b)});
                        for (p.dict.keys.items, p.dict.values.items) |k, v| try d.put(self.a, k, v);
                    }
                    for (args.kw) |k| try d.putStr(self.a, k.name, k.value);
                    return if (b == .dict) .{ .dict = d } else .{ .namespace = d };
                },
                .raise_exception => {
                    const msg = if (args.pos.len > 0) try self.toStr(args.pos[0]) else "";
                    self.diag.raised = true;
                    return self.fail("{s}", .{msg});
                },
                .strftime_now => {
                    const fmt = if (args.pos.len > 0) try self.toStr(args.pos[0]) else return self.fail("strftime_now() takes a format", .{});
                    return .{ .string = try strftime(self.a, fmt, self.now) };
                },
                .joiner_call => unreachable,
            },
            .undefined => |n| return self.fail("'{s}' is undefined", .{n}),
            else => return self.fail("'{s}' object is not callable", .{typeName(callee)}),
        }
    }

    fn callMacro(self: *Interp, m: *Macro, args: CallArgs) Error!Value {
        self.depth += 1;
        defer self.depth -= 1;
        if (self.depth > 200) return self.fail("macro recursion too deep", .{});
        const s = try Scope.create(self.a, m.closure);
        var extra = std.ArrayList(Value).empty;
        for (args.pos, 0..) |v, i| {
            if (i < m.params.len) try s.vars.put(self.a, m.params[i].name, v) else try extra.append(self.a, v);
        }
        const kwargs = try self.a.create(Dict);
        kwargs.* = .{};
        for (args.kw) |k| {
            var found = false;
            for (m.params) |p| if (std.mem.eql(u8, p.name, k.name)) {
                found = true;
            };
            if (found) try s.vars.put(self.a, k.name, k.value) else try kwargs.putStr(self.a, k.name, k.value);
        }
        for (m.params) |p| {
            if (s.vars.contains(p.name)) continue;
            const v: Value = if (p.default) |d| try self.eval(d, s) else .{ .undefined = p.name };
            try s.vars.put(self.a, p.name, v);
        }
        const varargs = try Value.listOf(self.a, extra.items);
        varargs.list.kind = .tuple;
        try s.vars.put(self.a, "varargs", varargs);
        try s.vars.put(self.a, "kwargs", .{ .dict = kwargs });
        if (m.caller) |c| try s.vars.put(self.a, "caller", .{ .macro = c });
        return .{ .string = try self.capture(m.body, s) };
    }

    // ----- attributes, items, slices -----

    const str_methods = [_][]const u8{ "strip", "lstrip", "rstrip", "split", "rsplit", "splitlines", "startswith", "endswith", "upper", "lower", "title", "capitalize", "replace", "find", "rfind", "index", "count", "join", "format", "isdigit", "isalpha", "isalnum", "isspace", "isupper", "islower", "removeprefix", "removesuffix", "casefold", "swapcase", "zfill" };
    const dict_methods = [_][]const u8{ "items", "keys", "values", "get", "copy" };
    const list_methods = [_][]const u8{ "index", "count", "copy" };

    fn bound(self: *Interp, v: Value, name: []const u8) !Value {
        const m = try self.a.create(Method);
        m.* = .{ .name = name, .self = v };
        return .{ .method = m };
    }

    /// `obj.name`: jinja2 tries the Python attribute first, then the item.
    fn getAttr(self: *Interp, obj: Value, name: []const u8) Error!Value {
        switch (obj) {
            .undefined => |n| return self.fail("'{s}' is undefined", .{if (n.len > 0) n else name}),
            .dict => |d| {
                if (isOneOf(name, &dict_methods)) return self.bound(obj, name);
                return d.getStr(name) orelse .{ .undefined = name };
            },
            .namespace => |d| return d.getStr(name) orelse .{ .undefined = name },
            .string => return if (isOneOf(name, &str_methods)) self.bound(obj, name) else .{ .undefined = name },
            .list => |l| {
                if (l.kind != .generator and isOneOf(name, &list_methods)) return self.bound(obj, name);
                if (std.mem.eql(u8, name, "append") or std.mem.eql(u8, name, "extend") or std.mem.eql(u8, name, "pop") or std.mem.eql(u8, name, "insert"))
                    return self.fail("access to attribute '{s}' of 'list' object is unsafe", .{name});
                return .{ .undefined = name };
            },
            .loop => |l| {
                const n = l.items.len;
                const i = l.index0;
                const eql = std.mem.eql;
                if (eql(u8, name, "index")) return .{ .int = @intCast(i + 1) };
                if (eql(u8, name, "index0")) return .{ .int = @intCast(i) };
                if (eql(u8, name, "revindex")) return .{ .int = @intCast(n - i) };
                if (eql(u8, name, "revindex0")) return .{ .int = @intCast(n - i - 1) };
                if (eql(u8, name, "first")) return .{ .boolean = i == 0 };
                if (eql(u8, name, "last")) return .{ .boolean = i + 1 == n };
                if (eql(u8, name, "length")) return .{ .int = @intCast(n) };
                if (eql(u8, name, "depth")) return .{ .int = 1 };
                if (eql(u8, name, "depth0")) return .{ .int = 0 };
                if (eql(u8, name, "previtem")) return if (i > 0) l.items[i - 1] else .{ .undefined = "previtem" };
                if (eql(u8, name, "nextitem")) return if (i + 1 < n) l.items[i + 1] else .{ .undefined = "nextitem" };
                if (eql(u8, name, "cycle") or eql(u8, name, "changed")) return self.bound(obj, name);
                return .{ .undefined = name };
            },
            else => return .{ .undefined = name },
        }
    }

    /// `obj[key]`: the item first, then (for a string key) the attribute.
    fn getItem(self: *Interp, obj: Value, key: Value) Error!Value {
        switch (obj) {
            .undefined => |n| return self.fail("'{s}' is undefined", .{n}),
            .dict => |d| {
                if (d.get(key)) |v| return v;
                if (key == .string) return self.getAttr(obj, key.string);
                return .{ .undefined = "" };
            },
            .namespace => |d| {
                if (key == .string) return d.getStr(key.string) orelse .{ .undefined = key.string };
                return .{ .undefined = "" };
            },
            .list => |l| {
                if (l.kind == .generator) return .{ .undefined = "" };
                if (key == .int or key == .boolean) {
                    const i = if (key == .int) key.int else @intFromBool(key.boolean);
                    const n: i64 = @intCast(l.items.items.len);
                    const j = if (i < 0) i + n else i;
                    if (j < 0 or j >= n) return .{ .undefined = "" };
                    return l.items.items[@intCast(j)];
                }
                if (key == .string) return self.getAttr(obj, key.string);
                return .{ .undefined = "" };
            },
            .string => |s| {
                if (key == .int) {
                    const cps = try codepoints(self.a, s);
                    const n: i64 = @intCast(cps.len);
                    const j = if (key.int < 0) key.int + n else key.int;
                    if (j < 0 or j >= n) return .{ .undefined = "" };
                    return .{ .string = cps[@intCast(j)] };
                }
                if (key == .string) return self.getAttr(obj, key.string);
                return .{ .undefined = "" };
            },
            .loop => if (key == .string) return self.getAttr(obj, key.string),
            else => {},
        }
        return .{ .undefined = "" };
    }

    fn slice(self: *Interp, obj: Value, start: Value, stop: Value, step: Value) Error!Value {
        const st: i64 = if (step == .none) 1 else try self.toInt(step);
        if (st == 0) return self.fail("slice step cannot be zero", .{});
        var items: []const Value = undefined;
        var is_str = false;
        switch (obj) {
            .list => |l| items = l.items.items,
            .string => |s| {
                is_str = true;
                const cps = try codepoints(self.a, s);
                const vals = try self.a.alloc(Value, cps.len);
                for (cps, 0..) |c, i| vals[i] = .{ .string = c };
                items = vals;
            },
            .undefined => |n| return self.fail("'{s}' is undefined", .{n}),
            else => return self.fail("'{s}' object is not subscriptable", .{typeName(obj)}),
        }
        const n: i64 = @intCast(items.len);
        // Python's slice.indices().
        var lo: i64 = undefined;
        var hi: i64 = undefined;
        if (st > 0) {
            lo = if (start == .none) 0 else clampIndex(try self.toInt(start), n, 0, n);
            hi = if (stop == .none) n else clampIndex(try self.toInt(stop), n, 0, n);
        } else {
            lo = if (start == .none) n - 1 else clampIndex(try self.toInt(start), n, -1, n - 1);
            hi = if (stop == .none) -1 else clampIndex(try self.toInt(stop), n, -1, n - 1);
        }
        var out = std.ArrayList(Value).empty;
        var i = lo;
        while ((st > 0 and i < hi) or (st < 0 and i > hi)) : (i += st) try out.append(self.a, items[@intCast(i)]);
        if (is_str) {
            var buf = std.ArrayList(u8).empty;
            for (out.items) |c| try buf.appendSlice(self.a, c.string);
            return .{ .string = buf.items };
        }
        const l = try self.a.create(List);
        l.* = .{ .items = out, .kind = if (obj.list.kind == .tuple) .tuple else .list };
        return .{ .list = l };
    }

    fn clampIndex(i: i64, n: i64, lo: i64, hi: i64) i64 {
        const j = if (i < 0) i + n else i;
        return std.math.clamp(j, lo, hi);
    }

    // ----- operators -----

    fn binop(self: *Interp, op: BinOp, l: Value, r: Value) Error!Value {
        switch (op) {
            .concat => {
                const ls = try self.toStr(l);
                const rs = try self.toStr(r);
                return .{ .string = try std.mem.concat(self.a, u8, &.{ ls, rs }) };
            },
            .eq => return .{ .boolean = equal(l, r) },
            .ne => return .{ .boolean = !equal(l, r) },
            .lt, .le, .gt, .ge => {
                const c = try self.compare(l, r);
                return .{ .boolean = switch (op) {
                    .lt => c == .lt,
                    .le => c != .gt,
                    .gt => c == .gt,
                    else => c != .lt,
                } };
            },
            .in => return .{ .boolean = try self.contains(r, l) },
            .not_in => return .{ .boolean = !try self.contains(r, l) },
            else => {},
        }
        if (l == .undefined) return self.fail("'{s}' is undefined", .{l.undefined});
        if (r == .undefined) return self.fail("'{s}' is undefined", .{r.undefined});
        if (op == .add) {
            if (l == .string and r == .string) return .{ .string = try std.mem.concat(self.a, u8, &.{ l.string, r.string }) };
            if (l == .list and r == .list and l.list.kind != .generator and r.list.kind != .generator) {
                const out = try self.a.create(List);
                out.* = .{ .items = .empty, .kind = l.list.kind };
                try out.items.appendSlice(self.a, l.list.items.items);
                try out.items.appendSlice(self.a, r.list.items.items);
                return .{ .list = out };
            }
        }
        if (op == .mul) {
            if ((l == .string or l == .list) and (r == .int or r == .boolean)) return self.repeat(l, try self.toInt(r));
            if ((r == .string or r == .list) and (l == .int or l == .boolean)) return self.repeat(r, try self.toInt(l));
        }
        if (op == .mod and l == .string) return .{ .string = try self.percentFormat(l.string, r) };
        const ln = numeric(l) orelse return self.fail("unsupported operand types for {s}: '{s}' and '{s}'", .{ @tagName(op), typeName(l), typeName(r) });
        const rn = numeric(r) orelse return self.fail("unsupported operand types for {s}: '{s}' and '{s}'", .{ @tagName(op), typeName(l), typeName(r) });
        if (ln == .int and rn == .int and op != .div) {
            const a = ln.int;
            const b = rn.int;
            switch (op) {
                .add => return .{ .int = std.math.add(i64, a, b) catch return self.fail("integer overflow", .{}) },
                .sub => return .{ .int = std.math.sub(i64, a, b) catch return self.fail("integer overflow", .{}) },
                .mul => return .{ .int = std.math.mul(i64, a, b) catch return self.fail("integer overflow", .{}) },
                .floordiv => {
                    if (b == 0) return self.fail("integer division or modulo by zero", .{});
                    return .{ .int = @divFloor(a, b) };
                },
                .mod => {
                    if (b == 0) return self.fail("integer division or modulo by zero", .{});
                    return .{ .int = @mod(a, b) };
                },
                .pow => {
                    if (b < 0) return .{ .float = std.math.pow(f64, @floatFromInt(a), @floatFromInt(b)) };
                    return .{ .int = std.math.powi(i64, a, b) catch return self.fail("integer overflow", .{}) };
                },
                else => unreachable,
            }
        }
        const a = asFloat(ln);
        const b = asFloat(rn);
        return .{ .float = switch (op) {
            .add => a + b,
            .sub => a - b,
            .mul => a * b,
            .div => if (b == 0) return self.fail("division by zero", .{}) else a / b,
            .floordiv => if (b == 0) return self.fail("float divmod()", .{}) else @floor(a / b),
            .mod => if (b == 0) return self.fail("float modulo", .{}) else a - b * @floor(a / b),
            .pow => std.math.pow(f64, a, b),
            else => unreachable,
        } };
    }

    fn repeat(self: *Interp, v: Value, n: i64) Error!Value {
        const times: usize = if (n < 0) 0 else @intCast(n);
        if (v == .string) {
            var buf = std.ArrayList(u8).empty;
            for (0..times) |_| try buf.appendSlice(self.a, v.string);
            return .{ .string = buf.items };
        }
        const out = try self.a.create(List);
        out.* = .{ .items = .empty, .kind = v.list.kind };
        for (0..times) |_| try out.items.appendSlice(self.a, v.list.items.items);
        return .{ .list = out };
    }

    fn compare(self: *Interp, l: Value, r: Value) Error!std.math.Order {
        if (numeric(l)) |a| if (numeric(r)) |b| {
            if (a == .int and b == .int) return std.math.order(a.int, b.int);
            return std.math.order(asFloat(a), asFloat(b));
        };
        if (l == .string and r == .string) return std.mem.order(u8, l.string, r.string);
        if (l == .list and r == .list) {
            const x = l.list.items.items;
            const y = r.list.items.items;
            for (0..@min(x.len, y.len)) |i| {
                if (equal(x[i], y[i])) continue;
                return self.compare(x[i], y[i]);
            }
            return std.math.order(x.len, y.len);
        }
        if (l == .undefined) return self.fail("'{s}' is undefined", .{l.undefined});
        if (r == .undefined) return self.fail("'{s}' is undefined", .{r.undefined});
        return self.fail("'<' not supported between instances of '{s}' and '{s}'", .{ typeName(l), typeName(r) });
    }

    /// Python's `needle in hay`.
    fn contains(self: *Interp, hay: Value, needle: Value) Error!bool {
        switch (hay) {
            .string => |s| {
                if (needle != .string) return self.fail("'in <string>' requires string as left operand, not {s}", .{typeName(needle)});
                return std.mem.indexOf(u8, s, needle.string) != null;
            },
            .list => |l| {
                for (try self.iterate(hay)) |x| if (equal(x, needle)) return true;
                _ = l;
                return false;
            },
            .dict => |d| return d.get(needle) != null,
            .namespace => |d| return d.get(needle) != null,
            .undefined => return false,
            else => return self.fail("argument of type '{s}' is not iterable", .{typeName(hay)}),
        }
    }

    fn toInt(self: *Interp, v: Value) Error!i64 {
        return switch (v) {
            .int => |i| i,
            .boolean => |b| @intFromBool(b),
            else => self.fail("'{s}' object cannot be interpreted as an integer", .{typeName(v)}),
        };
    }

    // ----- str() and repr() -----

    fn toStr(self: *Interp, v: Value) Error![]const u8 {
        return switch (v) {
            .string => |s| s,
            .undefined => "",
            else => blk: {
                var buf = std.ArrayList(u8).empty;
                try self.writeRepr(&buf, v, false);
                break :blk buf.items;
            },
        };
    }

    /// Python's `str(v)` (top level) / `repr(v)` (inside containers).
    fn writeRepr(self: *Interp, buf: *std.ArrayList(u8), v: Value, quote: bool) Error!void {
        const a = self.a;
        switch (v) {
            .undefined => {},
            .none => try buf.appendSlice(a, "None"),
            .boolean => |b| try buf.appendSlice(a, if (b) "True" else "False"),
            .int => |i| try buf.print(a, "{d}", .{i}),
            .float => |f| try writePyFloat(a, buf, f),
            .string => |s| if (quote) try writePyStr(a, buf, s) else try buf.appendSlice(a, s),
            .list => |l| {
                if (l.kind == .generator) return buf.appendSlice(a, "<generator object>");
                try buf.append(a, if (l.kind == .tuple) '(' else '[');
                for (l.items.items, 0..) |x, i| {
                    if (i > 0) try buf.appendSlice(a, ", ");
                    try self.writeRepr(buf, x, true);
                }
                if (l.kind == .tuple and l.items.items.len == 1) try buf.append(a, ',');
                try buf.append(a, if (l.kind == .tuple) ')' else ']');
            },
            .dict, .namespace => |d| {
                if (v == .namespace) try buf.appendSlice(a, "<Namespace ");
                try buf.append(a, '{');
                for (d.keys.items, d.values.items, 0..) |k, x, i| {
                    if (i > 0) try buf.appendSlice(a, ", ");
                    try self.writeRepr(buf, k, true);
                    try buf.appendSlice(a, ": ");
                    try self.writeRepr(buf, x, true);
                }
                try buf.append(a, '}');
                if (v == .namespace) try buf.append(a, '>');
            },
            .macro => |m| try buf.print(a, "<Macro '{s}'>", .{m.name}),
            .builtin, .method => try buf.appendSlice(a, "<built-in function>"),
            .loop => try buf.appendSlice(a, "<LoopContext>"),
        }
    }

    // ----- methods -----

    fn callMethod(self: *Interp, obj: Value, name: []const u8, args: CallArgs) Error!Value {
        const eql = std.mem.eql;
        const a = self.a;
        switch (obj) {
            .string => |s| {
                if (eql(u8, name, "strip") or eql(u8, name, "lstrip") or eql(u8, name, "rstrip")) {
                    const chars = args.get(0, "chars");
                    const cs: ?[]const u8 = if (chars) |c| (if (c == .none) null else try self.toStr(c)) else null;
                    return .{ .string = pyStrip(s, cs, !eql(u8, name, "rstrip"), !eql(u8, name, "lstrip")) };
                }
                if (eql(u8, name, "split") or eql(u8, name, "rsplit")) {
                    const sep = args.get(0, "sep");
                    const maxsplit: i64 = if (args.get(1, "maxsplit")) |m| try self.toInt(m) else -1;
                    const sep_s: ?[]const u8 = if (sep) |x| (if (x == .none) null else try self.toStr(x)) else null;
                    if (sep_s != null and sep_s.?.len == 0) return self.fail("empty separator", .{});
                    return self.listFromStrings(try pySplit(a, s, sep_s, maxsplit, eql(u8, name, "rsplit")));
                }
                if (eql(u8, name, "splitlines")) {
                    var parts = std.ArrayList([]const u8).empty;
                    var it = std.mem.splitScalar(u8, s, '\n');
                    while (it.next()) |line| try parts.append(a, std.mem.trimEnd(u8, line, "\r"));
                    if (parts.items.len > 0 and parts.items[parts.items.len - 1].len == 0) parts.items.len -= 1;
                    return self.listFromStrings(parts.items);
                }
                if (eql(u8, name, "startswith") or eql(u8, name, "endswith")) {
                    const p = args.get(0, "prefix") orelse return self.fail("{s}() takes an argument", .{name});
                    const start = eql(u8, name, "startswith");
                    if (p == .list) {
                        for (p.list.items.items) |x| {
                            const xs = try self.toStr(x);
                            if (if (start) std.mem.startsWith(u8, s, xs) else std.mem.endsWith(u8, s, xs)) return .{ .boolean = true };
                        }
                        return .{ .boolean = false };
                    }
                    if (p != .string) return self.fail("{s} first arg must be str or a tuple of str, not {s}", .{ name, typeName(p) });
                    return .{ .boolean = if (start) std.mem.startsWith(u8, s, p.string) else std.mem.endsWith(u8, s, p.string) };
                }
                if (eql(u8, name, "upper")) return .{ .string = try upper(a, s) };
                if (eql(u8, name, "lower") or eql(u8, name, "casefold")) return .{ .string = try lower(a, s) };
                if (eql(u8, name, "title")) return .{ .string = try title(a, s) };
                if (eql(u8, name, "capitalize")) return .{ .string = try capitalize(a, s) };
                if (eql(u8, name, "swapcase")) {
                    const out = try a.dupe(u8, s);
                    for (out) |*c| c.* = if (std.ascii.isUpper(c.*)) std.ascii.toLower(c.*) else std.ascii.toUpper(c.*);
                    return .{ .string = out };
                }
                if (eql(u8, name, "replace")) {
                    const old = try self.toStr(args.get(0, "old") orelse return self.fail("replace() takes 2 arguments", .{}));
                    const new_s = try self.toStr(args.get(1, "new") orelse return self.fail("replace() takes 2 arguments", .{}));
                    const count: i64 = if (args.get(2, "count")) |c| try self.toInt(c) else -1;
                    return .{ .string = try pyReplace(a, s, old, new_s, count) };
                }
                if (eql(u8, name, "find") or eql(u8, name, "rfind") or eql(u8, name, "index")) {
                    const sub = try self.toStr(args.get(0, "sub") orelse return self.fail("{s}() takes an argument", .{name}));
                    const at = if (eql(u8, name, "rfind")) std.mem.lastIndexOf(u8, s, sub) else std.mem.indexOf(u8, s, sub);
                    if (at) |i| return .{ .int = @intCast(std.unicode.utf8CountCodepoints(s[0..i]) catch i) };
                    if (eql(u8, name, "index")) return self.fail("substring not found", .{});
                    return .{ .int = -1 };
                }
                if (eql(u8, name, "count")) {
                    const sub = try self.toStr(args.get(0, "sub") orelse return self.fail("count() takes an argument", .{}));
                    if (sub.len == 0) return .{ .int = @intCast((std.unicode.utf8CountCodepoints(s) catch s.len) + 1) };
                    return .{ .int = @intCast(std.mem.count(u8, s, sub)) };
                }
                if (eql(u8, name, "join")) {
                    const items = try self.iterate(args.get(0, "iterable") orelse return self.fail("join() takes an argument", .{}));
                    var buf = std.ArrayList(u8).empty;
                    for (items, 0..) |x, i| {
                        if (i > 0) try buf.appendSlice(a, s);
                        if (x != .string) return self.fail("sequence item {d}: expected str instance, {s} found", .{ i, typeName(x) });
                        try buf.appendSlice(a, x.string);
                    }
                    return .{ .string = buf.items };
                }
                if (eql(u8, name, "format")) return .{ .string = try self.braceFormat(s, args) };
                if (eql(u8, name, "removeprefix")) {
                    const p = try self.toStr(args.get(0, "prefix") orelse return self.fail("removeprefix() takes an argument", .{}));
                    return .{ .string = if (std.mem.startsWith(u8, s, p)) s[p.len..] else s };
                }
                if (eql(u8, name, "removesuffix")) {
                    const p = try self.toStr(args.get(0, "suffix") orelse return self.fail("removesuffix() takes an argument", .{}));
                    return .{ .string = if (p.len > 0 and std.mem.endsWith(u8, s, p)) s[0 .. s.len - p.len] else s };
                }
                if (eql(u8, name, "zfill")) {
                    const width: usize = @intCast(@max(try self.toInt(args.get(0, "width") orelse return self.fail("zfill() takes an argument", .{})), 0));
                    const n = std.unicode.utf8CountCodepoints(s) catch s.len;
                    if (n >= width) return .{ .string = s };
                    var buf = std.ArrayList(u8).empty;
                    var rest = s;
                    if (rest.len > 0 and (rest[0] == '-' or rest[0] == '+')) {
                        try buf.append(a, rest[0]);
                        rest = rest[1..];
                    }
                    for (0..width - n) |_| try buf.append(a, '0');
                    try buf.appendSlice(a, rest);
                    return .{ .string = buf.items };
                }
                if (eql(u8, name, "isdigit") or eql(u8, name, "isalpha") or eql(u8, name, "isalnum") or eql(u8, name, "isspace") or eql(u8, name, "isupper") or eql(u8, name, "islower")) {
                    return .{ .boolean = strPredicate(name, s) };
                }
            },
            .dict => |d| {
                if (eql(u8, name, "items")) {
                    var out = std.ArrayList(Value).empty;
                    for (d.keys.items, d.values.items) |k, v| {
                        const pair = try self.a.create(List);
                        pair.* = .{ .items = .empty, .kind = .tuple };
                        try pair.items.appendSlice(a, &.{ k, v });
                        try out.append(a, .{ .list = pair });
                    }
                    const l = try a.create(List);
                    l.* = .{ .items = out };
                    return .{ .list = l };
                }
                if (eql(u8, name, "keys")) return Value.listOf(a, d.keys.items);
                if (eql(u8, name, "values")) return Value.listOf(a, d.values.items);
                if (eql(u8, name, "get")) {
                    const k = args.get(0, "key") orelse return self.fail("get() takes an argument", .{});
                    return d.get(k) orelse (args.get(1, "default") orelse .none);
                }
                if (eql(u8, name, "copy")) {
                    const c = try a.create(Dict);
                    c.* = .{};
                    for (d.keys.items, d.values.items) |k, v| try c.put(a, k, v);
                    return .{ .dict = c };
                }
            },
            .list => |l| {
                if (eql(u8, name, "index")) {
                    const x = args.get(0, "value") orelse return self.fail("index() takes an argument", .{});
                    for (l.items.items, 0..) |y, i| if (equal(x, y)) return .{ .int = @intCast(i) };
                    return self.fail("value is not in list", .{});
                }
                if (eql(u8, name, "count")) {
                    const x = args.get(0, "value") orelse return self.fail("count() takes an argument", .{});
                    var n: i64 = 0;
                    for (l.items.items) |y| n += @intFromBool(equal(x, y));
                    return .{ .int = n };
                }
                if (eql(u8, name, "copy")) return Value.listOf(a, l.items.items);
            },
            .loop => |l| {
                if (eql(u8, name, "cycle")) {
                    if (args.pos.len == 0) return self.fail("no items for cycling given", .{});
                    return args.pos[l.index0 % args.pos.len];
                }
                if (eql(u8, name, "changed")) {
                    const prev = l.last_changed;
                    l.last_changed = args.pos;
                    if (prev) |p| {
                        if (p.len == args.pos.len) {
                            var same = true;
                            for (p, args.pos) |x, y| same = same and equal(x, y);
                            if (same) return .{ .boolean = false };
                        }
                    }
                    return .{ .boolean = true };
                }
            },
            else => {},
        }
        return self.fail("'{s}' object has no attribute '{s}'", .{ typeName(obj), name });
    }

    fn listFromStrings(self: *Interp, parts: []const []const u8) Error!Value {
        const l = try self.a.create(List);
        l.* = .{ .items = .empty };
        for (parts) |p| try l.items.append(self.a, .{ .string = p });
        return .{ .list = l };
    }

    /// `'%s: %d' % args` for the conversions templates use (s, r, d, i, f, x, %).
    fn percentFormat(self: *Interp, fmt: []const u8, args_v: Value) Error![]const u8 {
        const args: []const Value = if (args_v == .list and args_v.list.kind == .tuple) args_v.list.items.items else &.{args_v};
        var buf = std.ArrayList(u8).empty;
        var ai: usize = 0;
        var i: usize = 0;
        while (i < fmt.len) : (i += 1) {
            if (fmt[i] != '%') {
                try buf.append(self.a, fmt[i]);
                continue;
            }
            i += 1;
            if (i >= fmt.len) return self.fail("incomplete format", .{});
            var zero = false;
            var left = false;
            while (i < fmt.len and (fmt[i] == '0' or fmt[i] == '-')) : (i += 1) {
                if (fmt[i] == '0') zero = true else left = true;
            }
            var width: usize = 0;
            while (i < fmt.len and std.ascii.isDigit(fmt[i])) : (i += 1) width = width * 10 + (fmt[i] - '0');
            var prec: ?usize = null;
            if (i < fmt.len and fmt[i] == '.') {
                i += 1;
                var p: usize = 0;
                while (i < fmt.len and std.ascii.isDigit(fmt[i])) : (i += 1) p = p * 10 + (fmt[i] - '0');
                prec = p;
            }
            if (i >= fmt.len) return self.fail("incomplete format", .{});
            const conv = fmt[i];
            if (conv == '%') {
                try buf.append(self.a, '%');
                continue;
            }
            if (ai >= args.len) return self.fail("not enough arguments for format string", .{});
            const arg = args[ai];
            ai += 1;
            var piece = std.ArrayList(u8).empty;
            switch (conv) {
                's' => try piece.appendSlice(self.a, try self.toStr(arg)),
                'r' => try self.writeRepr(&piece, arg, true),
                'd', 'i' => {
                    const n = numeric(arg) orelse return self.fail("%d format: a number is required, not {s}", .{typeName(arg)});
                    const iv: i64 = if (n == .int) n.int else @intFromFloat(@trunc(n.float));
                    try piece.print(self.a, "{d}", .{iv});
                },
                'f' => {
                    const n = numeric(arg) orelse return self.fail("%f format: a number is required", .{});
                    try piece.print(self.a, "{d:.[1]}", .{ asFloat(n), prec orelse 6 });
                },
                'x' => try piece.print(self.a, "{x}", .{try self.toInt(arg)}),
                else => return self.fail("unsupported format character '{c}'", .{conv}),
            }
            const plen = std.unicode.utf8CountCodepoints(piece.items) catch piece.items.len;
            if (plen < width and !left) {
                const neg = zero and piece.items.len > 0 and piece.items[0] == '-';
                if (neg) try buf.append(self.a, '-');
                for (0..width - plen) |_| try buf.append(self.a, if (zero) '0' else ' ');
                try buf.appendSlice(self.a, if (neg) piece.items[1..] else piece.items);
            } else {
                try buf.appendSlice(self.a, piece.items);
                if (plen < width) for (0..width - plen) |_| try buf.append(self.a, ' ');
            }
        }
        if (ai < args.len and !(args_v == .dict)) return self.fail("not all arguments converted during string formatting", .{});
        return buf.items;
    }

    /// `'{} {name}'.format(...)` without format specs beyond `{}` / `{0}` / `{name}`.
    fn braceFormat(self: *Interp, fmt: []const u8, args: CallArgs) Error![]const u8 {
        var buf = std.ArrayList(u8).empty;
        var auto: usize = 0;
        var i: usize = 0;
        while (i < fmt.len) : (i += 1) {
            const c = fmt[i];
            if (c == '{' and i + 1 < fmt.len and fmt[i + 1] == '{') {
                try buf.append(self.a, '{');
                i += 1;
            } else if (c == '}' and i + 1 < fmt.len and fmt[i + 1] == '}') {
                try buf.append(self.a, '}');
                i += 1;
            } else if (c == '{') {
                const end = std.mem.indexOfScalarPos(u8, fmt, i, '}') orelse return self.fail("unmatched '{{' in format string", .{});
                var field = fmt[i + 1 .. end];
                if (std.mem.indexOfScalar(u8, field, ':')) |colon| field = field[0..colon];
                var v: Value = undefined;
                if (field.len == 0) {
                    if (auto >= args.pos.len) return self.fail("format: not enough arguments", .{});
                    v = args.pos[auto];
                    auto += 1;
                } else if (std.fmt.parseInt(usize, field, 10)) |n| {
                    if (n >= args.pos.len) return self.fail("format: index out of range", .{});
                    v = args.pos[n];
                } else |_| {
                    v = args.get(std.math.maxInt(usize), field) orelse return self.fail("format: missing key '{s}'", .{field});
                }
                try buf.appendSlice(self.a, try self.toStr(v));
                i = end;
            } else try buf.append(self.a, c);
        }
        return buf.items;
    }

    // ----- filters -----

    /// An item's attribute for `map` / `selectattr` / `sort(attribute=...)`:
    /// dotted, with integer parts as indexes.
    fn attrPath(self: *Interp, v: Value, path: []const u8) Error!Value {
        var cur = v;
        var it = std.mem.splitScalar(u8, path, '.');
        while (it.next()) |part| {
            const key: Value = if (std.fmt.parseInt(i64, part, 10)) |n| .{ .int = n } else |_| .{ .string = part };
            cur = try self.getItem(cur, key);
        }
        return cur;
    }

    fn generator(self: *Interp, items: []const Value) Error!Value {
        const l = try self.a.create(List);
        l.* = .{ .items = .empty, .kind = .generator };
        try l.items.appendSlice(self.a, items);
        return .{ .list = l };
    }

    fn length(self: *Interp, v: Value) Error!usize {
        return switch (v) {
            .string => |s| std.unicode.utf8CountCodepoints(s) catch s.len,
            .list => |l| if (l.kind == .generator) self.fail("object of type 'generator' has no len()", .{}) else l.items.items.len,
            .dict, .namespace => |d| d.keys.items.len,
            .undefined => 0,
            else => self.fail("object of type '{s}' has no len()", .{typeName(v)}),
        };
    }

    fn sortValues(self: *Interp, items: []Value, attribute: ?[]const u8, case_sensitive: bool, reverse: bool) Error!void {
        const Ctx = struct {
            it: *Interp,
            attribute: ?[]const u8,
            case_sensitive: bool,
            err: ?Error = null,

            fn key(c: *@This(), v: Value) Value {
                var k = v;
                if (c.attribute) |p| k = c.it.attrPath(v, p) catch |e| blk: {
                    c.err = e;
                    break :blk v;
                };
                if (!c.case_sensitive and k == .string) k = .{ .string = lower(c.it.a, k.string) catch k.string };
                return k;
            }

            fn less(c: *@This(), x: Value, y: Value) bool {
                const o = c.it.compare(c.key(x), c.key(y)) catch |e| {
                    c.err = e;
                    return false;
                };
                return o == .lt;
            }
        };
        var ctx: Ctx = .{ .it = self, .attribute = attribute, .case_sensitive = case_sensitive };
        std.sort.insertion(Value, items, &ctx, Ctx.less);
        if (ctx.err) |e| return e;
        if (reverse) std.mem.reverse(Value, items);
    }

    fn applyFilter(self: *Interp, name: []const u8, v: Value, args: CallArgs) Error!Value {
        const eql = std.mem.eql;
        const a = self.a;
        if (eql(u8, name, "safe")) return v;
        if (eql(u8, name, "string")) return .{ .string = try self.toStr(v) };
        if (eql(u8, name, "trim")) {
            const chars = args.get(0, "chars");
            const cs: ?[]const u8 = if (chars) |c| (if (c == .none) null else try self.toStr(c)) else null;
            return .{ .string = pyStrip(try self.toStr(v), cs, true, true) };
        }
        if (eql(u8, name, "upper")) return .{ .string = try upper(a, try self.toStr(v)) };
        if (eql(u8, name, "lower")) return .{ .string = try lower(a, try self.toStr(v)) };
        if (eql(u8, name, "capitalize")) return .{ .string = try capitalize(a, try self.toStr(v)) };
        if (eql(u8, name, "title")) return .{ .string = try title(a, try self.toStr(v)) };
        if (eql(u8, name, "length") or eql(u8, name, "count")) return .{ .int = @intCast(try self.length(v)) };
        if (eql(u8, name, "tojson")) {
            const ensure_ascii = if (args.get(0, "ensure_ascii")) |x| truthy(x) else false;
            const indent = args.get(1, "indent") orelse Value.none;
            const separators = args.get(2, "separators") orelse Value.none;
            const sort_keys = if (args.get(3, "sort_keys")) |x| truthy(x) else false;
            var j: Json = .{ .it = self, .ensure_ascii = ensure_ascii, .sort_keys = sort_keys };
            switch (indent) {
                .none => {},
                .int => |n| {
                    j.indent = try a.alloc(u8, @intCast(@max(n, 0)));
                    @memset(@constCast(j.indent.?), ' ');
                },
                .string => |s| j.indent = s,
                else => return self.fail("tojson: bad indent", .{}),
            }
            if (j.indent != null) j.item_sep = ",";
            if (separators == .list and separators.list.items.items.len == 2) {
                j.item_sep = try self.toStr(separators.list.items.items[0]);
                j.key_sep = try self.toStr(separators.list.items.items[1]);
            }
            var buf = std.ArrayList(u8).empty;
            try j.write(&buf, v, 0);
            return .{ .string = buf.items };
        }
        if (eql(u8, name, "replace")) {
            const old = try self.toStr(args.get(0, "old") orelse return self.fail("replace needs 2 arguments", .{}));
            const new_s = try self.toStr(args.get(1, "new") orelse return self.fail("replace needs 2 arguments", .{}));
            const count: i64 = if (args.get(2, "count")) |c| (if (c == .none) -1 else try self.toInt(c)) else -1;
            return .{ .string = try pyReplace(a, try self.toStr(v), old, new_s, count) };
        }
        if (eql(u8, name, "join")) {
            const sep = if (args.get(0, "d")) |d| try self.toStr(d) else "";
            const attribute = args.get(1, "attribute");
            var buf = std.ArrayList(u8).empty;
            for (try self.iterate(v), 0..) |x0, i| {
                const x = if (attribute) |p| try self.attrPath(x0, try self.toStr(p)) else x0;
                if (i > 0) try buf.appendSlice(a, sep);
                try buf.appendSlice(a, try self.toStr(x));
            }
            return .{ .string = buf.items };
        }
        if (eql(u8, name, "first")) {
            if (v == .list and v.list.kind == .generator) {
                const items = try self.iterate(v);
                return if (items.len > 0) items[0] else .{ .undefined = "first" };
            }
            const items = try self.iterate(v);
            return if (items.len > 0) items[0] else .{ .undefined = "first" };
        }
        if (eql(u8, name, "last")) {
            if (v == .list and v.list.kind == .generator) return self.fail("'generator' object is not reversible", .{});
            const items = try self.iterate(v);
            return if (items.len > 0) items[items.len - 1] else .{ .undefined = "last" };
        }
        if (eql(u8, name, "default") or eql(u8, name, "d")) {
            const dflt = args.get(0, "default_value") orelse Value{ .string = "" };
            const boolean = if (args.get(1, "boolean")) |b| truthy(b) else false;
            if (v == .undefined or (boolean and !truthy(v))) return dflt;
            return v;
        }
        if (eql(u8, name, "list")) return Value.listOf(a, try self.iterate(v));
        if (eql(u8, name, "items")) {
            if (v == .undefined) return self.generator(&.{});
            if (v != .dict) return self.fail("items: expected a mapping, got '{s}'", .{typeName(v)});
            const pairs = try self.callMethod(v, "items", .{ .pos = &.{}, .kw = &.{} });
            return self.generator(pairs.list.items.items);
        }
        if (eql(u8, name, "dictsort")) {
            if (v != .dict) return self.fail("dictsort: expected a mapping", .{});
            const case_sensitive = if (args.get(0, "case_sensitive")) |x| truthy(x) else false;
            const by = if (args.get(1, "by")) |x| try self.toStr(x) else "key";
            const reverse = if (args.get(2, "reverse")) |x| truthy(x) else false;
            const pairs = try self.callMethod(v, "items", .{ .pos = &.{}, .kw = &.{} });
            const items = pairs.list.items.items;
            try self.sortValues(items, if (eql(u8, by, "value")) "1" else "0", case_sensitive, reverse);
            return pairs;
        }
        if (eql(u8, name, "sort")) {
            const reverse = if (args.get(0, "reverse")) |x| truthy(x) else false;
            const case_sensitive = if (args.get(1, "case_sensitive")) |x| truthy(x) else false;
            const attribute = if (args.get(2, "attribute")) |x| try self.toStr(x) else null;
            const items = try a.dupe(Value, try self.iterate(v));
            try self.sortValues(items, attribute, case_sensitive, reverse);
            return Value.listOf(a, items);
        }
        if (eql(u8, name, "reverse")) {
            if (v == .string) {
                const cps = try codepoints(a, v.string);
                var buf = std.ArrayList(u8).empty;
                var i = cps.len;
                while (i > 0) : (i -= 1) try buf.appendSlice(a, cps[i - 1]);
                return .{ .string = buf.items };
            }
            const items = try a.dupe(Value, try self.iterate(v));
            std.mem.reverse(Value, items);
            return self.generator(items);
        }
        if (eql(u8, name, "unique")) {
            const case_sensitive = if (args.get(0, "case_sensitive")) |x| truthy(x) else false;
            const attribute = if (args.get(1, "attribute")) |x| try self.toStr(x) else null;
            var out = std.ArrayList(Value).empty;
            var seen = std.ArrayList(Value).empty;
            for (try self.iterate(v)) |x| {
                var k = if (attribute) |p| try self.attrPath(x, p) else x;
                if (!case_sensitive and k == .string) k = .{ .string = try lower(a, k.string) };
                var dup = false;
                for (seen.items) |s| dup = dup or equal(s, k);
                if (dup) continue;
                try seen.append(a, k);
                try out.append(a, x);
            }
            return self.generator(out.items);
        }
        if (eql(u8, name, "min") or eql(u8, name, "max")) {
            const case_sensitive = if (args.get(0, "case_sensitive")) |x| truthy(x) else false;
            const attribute = if (args.get(1, "attribute")) |x| try self.toStr(x) else null;
            const items = try a.dupe(Value, try self.iterate(v));
            if (items.len == 0) return .{ .undefined = name };
            try self.sortValues(items, attribute, case_sensitive, false);
            return if (eql(u8, name, "min")) items[0] else items[items.len - 1];
        }
        if (eql(u8, name, "sum")) {
            const attribute = if (args.get(0, "attribute")) |x| try self.toStr(x) else null;
            var total = args.get(1, "start") orelse Value{ .int = 0 };
            for (try self.iterate(v)) |x| total = try self.binop(.add, total, if (attribute) |p| try self.attrPath(x, p) else x);
            return total;
        }
        if (eql(u8, name, "map")) {
            var out = std.ArrayList(Value).empty;
            const attribute = args.get(std.math.maxInt(usize), "attribute");
            if (attribute) |p| {
                const path = try self.toStr(p);
                const dflt = args.get(std.math.maxInt(usize), "default");
                for (try self.iterate(v)) |x| {
                    var y = try self.attrPath(x, path);
                    if (y == .undefined) if (dflt) |d| {
                        y = d;
                    };
                    try out.append(a, y);
                }
            } else {
                if (args.pos.len == 0) return self.fail("map: no filter given", .{});
                const fname = try self.toStr(args.pos[0]);
                const rest: CallArgs = .{ .pos = args.pos[1..], .kw = args.kw };
                for (try self.iterate(v)) |x| try out.append(a, try self.applyFilter(fname, x, rest));
            }
            return self.generator(out.items);
        }
        if (eql(u8, name, "select") or eql(u8, name, "reject") or eql(u8, name, "selectattr") or eql(u8, name, "rejectattr")) {
            const by_attr = std.mem.endsWith(u8, name, "attr");
            const want = std.mem.startsWith(u8, name, "select");
            var pos = args.pos;
            var path: ?[]const u8 = null;
            if (by_attr) {
                if (pos.len == 0) return self.fail("{s}: missing attribute", .{name});
                path = try self.toStr(pos[0]);
                pos = pos[1..];
            }
            var out = std.ArrayList(Value).empty;
            for (try self.iterate(v)) |x| {
                const y = if (path) |p| try self.attrPath(x, p) else x;
                const ok = if (pos.len == 0) truthy(y) else try self.applyTest(try self.toStr(pos[0]), y, .{ .pos = pos[1..], .kw = args.kw });
                if (ok == want) try out.append(a, x);
            }
            return self.generator(out.items);
        }
        if (eql(u8, name, "indent")) {
            const width = args.get(0, "width") orelse Value{ .int = 4 };
            const first = if (args.get(1, "first")) |x| truthy(x) else false;
            const blank = if (args.get(2, "blank")) |x| truthy(x) else false;
            const ind: []const u8 = if (width == .string) width.string else blk: {
                const n: usize = @intCast(@max(try self.toInt(width), 0));
                const s = try a.alloc(u8, n);
                @memset(s, ' ');
                break :blk s;
            };
            const s = try std.mem.concat(a, u8, &.{ try self.toStr(v), "\n" });
            var lines = std.ArrayList([]const u8).empty;
            var it = std.mem.splitScalar(u8, s, '\n');
            while (it.next()) |line| try lines.append(a, std.mem.trimEnd(u8, line, "\r"));
            lines.items.len -= 1; // splitlines() drops the empty piece after the added "\n"
            var buf = std.ArrayList(u8).empty;
            for (lines.items, 0..) |line, i| {
                if (i > 0) {
                    try buf.append(a, '\n');
                    if (blank or line.len > 0) try buf.appendSlice(a, ind);
                }
                try buf.appendSlice(a, line);
            }
            if (first) return .{ .string = try std.mem.concat(a, u8, &.{ ind, buf.items }) };
            return .{ .string = buf.items };
        }
        if (eql(u8, name, "int")) {
            const dflt = args.get(0, "default") orelse Value{ .int = 0 };
            return switch (v) {
                .int => v,
                .boolean => |b| .{ .int = @intFromBool(b) },
                .float => |f| .{ .int = @intFromFloat(@trunc(f)) },
                .string => |s| blk: {
                    const t = pyStrip(s, null, true, true);
                    if (std.fmt.parseInt(i64, t, 10)) |n| break :blk .{ .int = n } else |_| {}
                    if (std.fmt.parseFloat(f64, t)) |f| break :blk .{ .int = @intFromFloat(@trunc(f)) } else |_| {}
                    break :blk dflt;
                },
                else => dflt,
            };
        }
        if (eql(u8, name, "float")) {
            const dflt = args.get(0, "default") orelse Value{ .float = 0 };
            return switch (v) {
                .float => v,
                .int => |i| .{ .float = @floatFromInt(i) },
                .boolean => |b| .{ .float = @floatFromInt(@intFromBool(b)) },
                .string => |s| if (std.fmt.parseFloat(f64, pyStrip(s, null, true, true))) |f| .{ .float = f } else |_| dflt,
                else => dflt,
            };
        }
        if (eql(u8, name, "abs")) {
            return switch (v) {
                .int => |i| .{ .int = if (i < 0) -i else i },
                .float => |f| .{ .float = @abs(f) },
                else => self.fail("bad operand type for abs(): '{s}'", .{typeName(v)}),
            };
        }
        if (eql(u8, name, "round")) {
            const precision = if (args.get(0, "precision")) |p| try self.toInt(p) else 0;
            const method = if (args.get(1, "method")) |m| try self.toStr(m) else "common";
            const x = asFloat(numeric(v) orelse return self.fail("round: a number is required", .{}));
            const scale = std.math.pow(f64, 10, @floatFromInt(precision));
            const y = x * scale;
            // "common" is Python's round(): halves go to the even neighbour.
            const r = if (eql(u8, method, "floor")) @floor(y) else if (eql(u8, method, "ceil")) @ceil(y) else roundHalfEven(y);
            return .{ .float = r / scale };
        }
        if (eql(u8, name, "center")) {
            const width: usize = @intCast(@max(if (args.get(0, "width")) |w| try self.toInt(w) else 80, 0));
            const s = try self.toStr(v);
            const n = std.unicode.utf8CountCodepoints(s) catch s.len;
            if (n >= width) return .{ .string = s };
            const total = width - n;
            // Python's str.center puts the odd space on the right unless the width is odd.
            const left = total / 2 + (total & width & 1);
            var buf = std.ArrayList(u8).empty;
            for (0..left) |_| try buf.append(a, ' ');
            try buf.appendSlice(a, s);
            for (0..total - left) |_| try buf.append(a, ' ');
            return .{ .string = buf.items };
        }
        if (eql(u8, name, "wordcount")) {
            const parts = try pySplit(a, try self.toStr(v), null, -1, false);
            return .{ .int = @intCast(parts.len) };
        }
        if (eql(u8, name, "format")) {
            const tuple = try a.create(List);
            tuple.* = .{ .items = .empty, .kind = .tuple };
            try tuple.items.appendSlice(a, args.pos);
            return .{ .string = try self.percentFormat(try self.toStr(v), .{ .list = tuple }) };
        }
        if (eql(u8, name, "attr")) return self.getAttr(v, try self.toStr(args.get(0, "name") orelse return self.fail("attr needs a name", .{})));
        if (eql(u8, name, "pprint")) {
            var buf = std.ArrayList(u8).empty;
            try self.writeRepr(&buf, v, true);
            return .{ .string = buf.items };
        }
        if (eql(u8, name, "batch")) {
            const n: usize = @intCast(@max(try self.toInt(args.get(0, "linecount") orelse return self.fail("batch needs a size", .{})), 1));
            const fill = args.get(1, "fill_with");
            var out = std.ArrayList(Value).empty;
            const items = try self.iterate(v);
            var i: usize = 0;
            while (i < items.len) : (i += n) {
                var chunk = std.ArrayList(Value).empty;
                try chunk.appendSlice(a, items[i..@min(i + n, items.len)]);
                if (fill) |f| while (chunk.items.len < n) try chunk.append(a, f);
                try out.append(a, try Value.listOf(a, chunk.items));
            }
            return self.generator(out.items);
        }
        if (eql(u8, name, "escape") or eql(u8, name, "e") or eql(u8, name, "forceescape")) {
            var buf = std.ArrayList(u8).empty;
            for (try self.toStr(v)) |c| switch (c) {
                '&' => try buf.appendSlice(a, "&amp;"),
                '<' => try buf.appendSlice(a, "&lt;"),
                '>' => try buf.appendSlice(a, "&gt;"),
                '"' => try buf.appendSlice(a, "&#34;"),
                '\'' => try buf.appendSlice(a, "&#39;"),
                else => try buf.append(a, c),
            };
            return .{ .string = buf.items };
        }
        return self.fail("no filter named '{s}'", .{name});
    }

    // ----- tests -----

    fn applyTest(self: *Interp, name: []const u8, v: Value, args: CallArgs) Error!bool {
        const eql = std.mem.eql;
        const arg = args.get(0, "other");
        if (eql(u8, name, "defined")) return v != .undefined;
        if (eql(u8, name, "undefined")) return v == .undefined;
        if (eql(u8, name, "none")) return v == .none;
        if (eql(u8, name, "boolean")) return v == .boolean;
        if (eql(u8, name, "true")) return v == .boolean and v.boolean;
        if (eql(u8, name, "false")) return v == .boolean and !v.boolean;
        if (eql(u8, name, "integer")) return v == .int;
        if (eql(u8, name, "float")) return v == .float;
        if (eql(u8, name, "number")) return v == .int or v == .float or v == .boolean;
        if (eql(u8, name, "string")) return v == .string;
        if (eql(u8, name, "mapping")) return v == .dict;
        if (eql(u8, name, "iterable")) return v == .string or v == .list or v == .dict or v == .undefined;
        if (eql(u8, name, "sequence")) return v == .string or (v == .list and v.list.kind != .generator) or v == .dict or v == .undefined;
        if (eql(u8, name, "callable")) return v == .macro or v == .builtin or v == .method;
        if (eql(u8, name, "odd") or eql(u8, name, "even")) {
            const i = try self.toInt(v);
            return (@mod(i, 2) == 1) == eql(u8, name, "odd");
        }
        if (eql(u8, name, "divisibleby")) {
            const d = try self.toInt(arg orelse return self.fail("divisibleby needs an argument", .{}));
            if (d == 0) return self.fail("integer division or modulo by zero", .{});
            return @mod(try self.toInt(v), d) == 0;
        }
        if (eql(u8, name, "lower")) return v == .string and eql(u8, v.string, try lower(self.a, v.string));
        if (eql(u8, name, "upper")) return v == .string and eql(u8, v.string, try upper(self.a, v.string));
        if (eql(u8, name, "escaped")) return false;
        if (eql(u8, name, "sameas")) {
            const o = arg orelse return self.fail("sameas needs an argument", .{});
            return switch (v) {
                .none => o == .none,
                .boolean => |b| o == .boolean and o.boolean == b,
                .list => |l| o == .list and o.list == l,
                .dict => |d| o == .dict and o.dict == d,
                else => equal(v, o),
            };
        }
        const other = arg orelse Value{ .undefined = "" };
        if (eql(u8, name, "eq") or eql(u8, name, "equalto") or eql(u8, name, "==")) return equal(v, other);
        if (eql(u8, name, "ne") or eql(u8, name, "!=")) return !equal(v, other);
        if (eql(u8, name, "lt") or eql(u8, name, "lessthan") or eql(u8, name, "<")) return (try self.compare(v, other)) == .lt;
        if (eql(u8, name, "le") or eql(u8, name, "<=")) return (try self.compare(v, other)) != .gt;
        if (eql(u8, name, "gt") or eql(u8, name, "greaterthan") or eql(u8, name, ">")) return (try self.compare(v, other)) == .gt;
        if (eql(u8, name, "ge") or eql(u8, name, ">=")) return (try self.compare(v, other)) != .lt;
        if (eql(u8, name, "in")) return self.contains(arg orelse return self.fail("in needs an argument", .{}), v);
        return self.fail("no test named '{s}'", .{name});
    }
};

// ---------------------------------------------------------------------------
// Python semantics
// ---------------------------------------------------------------------------

const Num = union(enum) { int: i64, float: f64 };

fn numeric(v: Value) ?Num {
    return switch (v) {
        .int => |i| .{ .int = i },
        .float => |f| .{ .float = f },
        .boolean => |b| .{ .int = @intFromBool(b) },
        else => null,
    };
}

fn asFloat(n: Num) f64 {
    return switch (n) {
        .int => |i| @floatFromInt(i),
        .float => |f| f,
    };
}

/// Python's `==`.
pub fn equal(x: Value, y: Value) bool {
    if (numeric(x)) |a| if (numeric(y)) |b| {
        if (a == .int and b == .int) return a.int == b.int;
        return asFloat(a) == asFloat(b);
    };
    return switch (x) {
        .undefined => y == .undefined,
        .none => y == .none,
        .string => |s| y == .string and std.mem.eql(u8, s, y.string),
        .list => |l| blk: {
            if (y != .list) break :blk false;
            if (l.kind == .generator or y.list.kind == .generator) break :blk l == y.list;
            if (l.kind != y.list.kind or l.items.items.len != y.list.items.items.len) break :blk false;
            for (l.items.items, y.list.items.items) |p, q| if (!equal(p, q)) break :blk false;
            break :blk true;
        },
        .dict => |d| blk: {
            if (y != .dict or d.keys.items.len != y.dict.keys.items.len) break :blk false;
            for (d.keys.items, d.values.items) |k, v| {
                const o = y.dict.get(k) orelse break :blk false;
                if (!equal(v, o)) break :blk false;
            }
            break :blk true;
        },
        .namespace => |d| y == .namespace and y.namespace == d,
        .macro => |m| y == .macro and y.macro == m,
        .builtin => |b| y == .builtin and y.builtin == b,
        .method => |m| y == .method and y.method == m,
        .loop => |l| y == .loop and y.loop == l,
        else => false,
    };
}

pub fn truthy(v: Value) bool {
    return switch (v) {
        .undefined, .none => false,
        .boolean => |b| b,
        .int => |i| i != 0,
        .float => |f| f != 0,
        .string => |s| s.len > 0,
        .list => |l| l.kind == .generator or l.items.items.len > 0,
        .dict => |d| d.keys.items.len > 0,
        else => true,
    };
}

fn typeName(v: Value) []const u8 {
    return switch (v) {
        .undefined => "Undefined",
        .none => "NoneType",
        .boolean => "bool",
        .int => "int",
        .float => "float",
        .string => "str",
        .list => |l| @tagName(l.kind),
        .dict => "dict",
        .namespace => "Namespace",
        .macro => "Macro",
        .builtin, .method => "builtin_function_or_method",
        .loop => "LoopContext",
    };
}

const Utf8Iter = struct {
    s: []const u8,
    i: usize = 0,

    /// The next code point's bytes (an invalid byte on its own).
    fn next(self: *Utf8Iter) ?[]const u8 {
        if (self.i >= self.s.len) return null;
        const n = std.unicode.utf8ByteSequenceLength(self.s[self.i]) catch 1;
        const end = @min(self.i + n, self.s.len);
        const out = self.s[self.i..end];
        self.i = end;
        return out;
    }
};

fn codepoints(a: Allocator, s: []const u8) ![]const []const u8 {
    var out = std.ArrayList([]const u8).empty;
    var it = Utf8Iter{ .s = s };
    while (it.next()) |c| try out.append(a, c);
    return out.items;
}

fn decodeCp(c: []const u8) u21 {
    return std.unicode.utf8Decode(c) catch c[0];
}

/// Python's `str.isspace` for one code point.
fn isPySpace(cp: u21) bool {
    return switch (cp) {
        0x09...0x0d, 0x1c...0x20, 0x85, 0xa0, 0x1680, 0x2000...0x200a, 0x2028, 0x2029, 0x202f, 0x205f, 0x3000 => true,
        else => false,
    };
}

/// Python's `strip` / `lstrip` / `rstrip`: whitespace, or any code point of `chars`.
fn pyStrip(s: []const u8, chars: ?[]const u8, left: bool, right: bool) []const u8 {
    const Set = struct {
        fn has(set: ?[]const u8, c: []const u8) bool {
            if (set) |cs| {
                var it = Utf8Iter{ .s = cs };
                while (it.next()) |x| if (std.mem.eql(u8, x, c)) return true;
                return false;
            }
            return isPySpace(decodeCp(c));
        }
    };
    var start: usize = 0;
    var end = s.len;
    if (left) {
        var it = Utf8Iter{ .s = s };
        while (it.next()) |c| {
            if (!Set.has(chars, c)) break;
            start = it.i;
        }
    }
    if (right) {
        while (end > start) {
            var b = end - 1;
            while (b > start and (s[b] & 0xc0) == 0x80) b -= 1;
            if (!Set.has(chars, s[b..end])) break;
            end = b;
        }
    }
    return s[start..end];
}

/// Python's `split` / `rsplit` (whitespace runs when `sep` is null).
fn pySplit(a: Allocator, s: []const u8, sep: ?[]const u8, maxsplit: i64, from_right: bool) ![]const []const u8 {
    var parts = std.ArrayList([]const u8).empty;
    var splits: i64 = 0;
    if (sep) |p| {
        if (!from_right) {
            var rest = s;
            while (maxsplit < 0 or splits < maxsplit) {
                const i = std.mem.indexOf(u8, rest, p) orelse break;
                try parts.append(a, rest[0..i]);
                rest = rest[i + p.len ..];
                splits += 1;
            }
            try parts.append(a, rest);
        } else {
            var rest = s;
            while (maxsplit < 0 or splits < maxsplit) {
                const i = std.mem.lastIndexOf(u8, rest, p) orelse break;
                try parts.append(a, rest[i + p.len ..]);
                rest = rest[0..i];
                splits += 1;
            }
            try parts.append(a, rest);
            std.mem.reverse([]const u8, parts.items);
        }
        return parts.items;
    }
    // Whitespace: runs of spaces separate, leading/trailing ones are dropped.
    var words = std.ArrayList([2]usize).empty;
    var it = Utf8Iter{ .s = s };
    var word_start: ?usize = null;
    var pos: usize = 0;
    while (it.next()) |c| {
        const space = isPySpace(decodeCp(c));
        if (space) {
            if (word_start) |ws| try words.append(a, .{ ws, pos });
            word_start = null;
        } else if (word_start == null) word_start = pos;
        pos = it.i;
    }
    if (word_start) |ws| try words.append(a, .{ ws, s.len });
    const w = words.items;
    if (maxsplit < 0 or w.len <= @as(usize, @intCast(maxsplit))) {
        for (w) |r| try parts.append(a, s[r[0]..r[1]]);
        return parts.items;
    }
    const m: usize = @intCast(maxsplit);
    if (!from_right) {
        for (w[0..m]) |r| try parts.append(a, s[r[0]..r[1]]);
        try parts.append(a, s[w[m][0]..]);
    } else {
        try parts.append(a, s[0..w[w.len - m - 1][1]]);
        for (w[w.len - m ..]) |r| try parts.append(a, s[r[0]..r[1]]);
    }
    return parts.items;
}

fn pyReplace(a: Allocator, s: []const u8, old: []const u8, new: []const u8, count: i64) ![]const u8 {
    var buf = std.ArrayList(u8).empty;
    var done: i64 = 0;
    if (old.len == 0) {
        // Python inserts `new` around every code point.
        var it = Utf8Iter{ .s = s };
        if (count != 0) {
            try buf.appendSlice(a, new);
            done += 1;
        }
        while (it.next()) |c| {
            try buf.appendSlice(a, c);
            if (count < 0 or done < count) {
                try buf.appendSlice(a, new);
                done += 1;
            }
        }
        return buf.items;
    }
    var rest = s;
    while (count < 0 or done < count) {
        const i = std.mem.indexOf(u8, rest, old) orelse break;
        try buf.appendSlice(a, rest[0..i]);
        try buf.appendSlice(a, new);
        rest = rest[i + old.len ..];
        done += 1;
    }
    try buf.appendSlice(a, rest);
    return buf.items;
}

fn roundHalfEven(y: f64) f64 {
    const r = @round(y);
    if (@abs(y - @trunc(y)) == 0.5 and @mod(r, 2) != 0) return r - std.math.sign(y);
    return r;
}

/// Simple case mapping for ASCII, Latin-1, Latin Extended-A, Greek and
/// Cyrillic; other code points map to themselves.
fn caseMap(cp: u21, to_upper: bool) u21 {
    if (to_upper) {
        return switch (cp) {
            'a'...'z' => cp - 32,
            0xe0...0xf6, 0xf8...0xfe => cp - 32,
            0xff => 0x178,
            0x101...0x137, 0x14b...0x177 => if (cp & 1 == 1) cp - 1 else cp,
            0x13a...0x148, 0x17a...0x17e => if (cp & 1 == 0) cp - 1 else cp,
            0x3b1...0x3c1, 0x3c3...0x3c9 => cp - 32,
            0x3c2 => 0x3a3,
            0x430...0x44f => cp - 32,
            0x450...0x45f => cp - 80,
            else => cp,
        };
    }
    return switch (cp) {
        'A'...'Z' => cp + 32,
        0xc0...0xd6, 0xd8...0xde => cp + 32,
        0x178 => 0xff,
        0x100...0x136, 0x14a...0x176 => if (cp & 1 == 0) cp + 1 else cp,
        0x139...0x147, 0x179...0x17d => if (cp & 1 == 1) cp + 1 else cp,
        0x391...0x3a1, 0x3a3...0x3a9 => cp + 32,
        0x410...0x42f => cp + 32,
        0x400...0x40f => cp + 80,
        else => cp,
    };
}

fn isCased(cp: u21) bool {
    return caseMap(cp, true) != cp or caseMap(cp, false) != cp;
}

/// Maps every code point of `s`: `mode` 0 lower, 1 upper, 2 title (upper after
/// an uncased character), 3 capitalize (first upper, the rest lower).
fn mapCase(a: Allocator, s: []const u8, mode: u2) ![]const u8 {
    var out = try std.ArrayList(u8).initCapacity(a, s.len);
    var it = Utf8Iter{ .s = s };
    var prev_cased = false;
    var first = true;
    while (it.next()) |c| {
        if (c.len == 1 and c[0] < 0x80 and !std.ascii.isAlphabetic(c[0])) {
            try out.append(a, c[0]);
            prev_cased = false;
            first = false;
            continue;
        }
        const cp = std.unicode.utf8Decode(c) catch {
            try out.appendSlice(a, c);
            continue;
        };
        const up = switch (mode) {
            0 => false,
            1 => true,
            2 => !prev_cased,
            3 => first,
        };
        try appendCodepoint(a, &out, caseMap(cp, up));
        prev_cased = isCased(cp);
        first = false;
    }
    return out.items;
}

fn upper(a: Allocator, s: []const u8) ![]const u8 {
    return mapCase(a, s, 1);
}

fn lower(a: Allocator, s: []const u8) ![]const u8 {
    return mapCase(a, s, 0);
}

fn capitalize(a: Allocator, s: []const u8) ![]const u8 {
    return mapCase(a, s, 3);
}

/// Python's `str.title`: a cased letter after an uncased character is upper-cased, the rest lower-cased.
fn title(a: Allocator, s: []const u8) ![]const u8 {
    return mapCase(a, s, 2);
}

fn strPredicate(name: []const u8, s: []const u8) bool {
    const eql = std.mem.eql;
    if (s.len == 0) return false;
    var has_cased = false;
    for (s) |c| {
        if (eql(u8, name, "isdigit") and !std.ascii.isDigit(c)) return false;
        if (eql(u8, name, "isalpha") and !std.ascii.isAlphabetic(c)) return false;
        if (eql(u8, name, "isalnum") and !std.ascii.isAlphanumeric(c)) return false;
        if (eql(u8, name, "isspace") and !isWs(c)) return false;
        if (eql(u8, name, "isupper") and std.ascii.isLower(c)) return false;
        if (eql(u8, name, "islower") and std.ascii.isUpper(c)) return false;
        if (std.ascii.isAlphabetic(c)) has_cased = true;
    }
    if (eql(u8, name, "isupper") or eql(u8, name, "islower")) return has_cased;
    return true;
}

/// Python's `repr(float)`: the shortest round-trip digits, positional for
/// exponents in [-4, 16), scientific (`1e+16`, `1.5e-05`) otherwise.
fn writePyFloat(a: Allocator, buf: *std.ArrayList(u8), f: f64) !void {
    if (std.math.isNan(f)) return buf.appendSlice(a, "nan");
    if (std.math.isInf(f)) return buf.appendSlice(a, if (f < 0) "-inf" else "inf");
    if (f == 0) return buf.appendSlice(a, if (std.math.signbit(f)) "-0.0" else "0.0");
    var tmp: [64]u8 = undefined;
    const sci = std.fmt.bufPrint(&tmp, "{e}", .{@abs(f)}) catch unreachable;
    const e_at = std.mem.indexOfScalar(u8, sci, 'e').?;
    var digits_buf: [32]u8 = undefined;
    var nd: usize = 0;
    for (sci[0..e_at]) |c| if (c != '.') {
        digits_buf[nd] = c;
        nd += 1;
    };
    while (nd > 1 and digits_buf[nd - 1] == '0') nd -= 1;
    const digits = digits_buf[0..nd];
    const exp = std.fmt.parseInt(i32, sci[e_at + 1 ..], 10) catch 0;
    if (f < 0) try buf.append(a, '-');
    if (exp >= -4 and exp < 16) {
        if (exp < 0) {
            try buf.appendSlice(a, "0.");
            for (0..@intCast(-exp - 1)) |_| try buf.append(a, '0');
            try buf.appendSlice(a, digits);
        } else {
            const int_len: usize = @intCast(exp + 1);
            if (digits.len <= int_len) {
                try buf.appendSlice(a, digits);
                for (0..int_len - digits.len) |_| try buf.append(a, '0');
                try buf.appendSlice(a, ".0");
            } else {
                try buf.appendSlice(a, digits[0..int_len]);
                try buf.append(a, '.');
                try buf.appendSlice(a, digits[int_len..]);
            }
        }
    } else {
        try buf.append(a, digits[0]);
        if (digits.len > 1) {
            try buf.append(a, '.');
            try buf.appendSlice(a, digits[1..]);
        }
        try buf.print(a, "e{c}{d:0>2}", .{ @as(u8, if (exp < 0) '-' else '+'), @abs(exp) });
    }
}

/// Python's `repr(str)`.
fn writePyStr(a: Allocator, buf: *std.ArrayList(u8), s: []const u8) !void {
    const q: u8 = if (std.mem.indexOfScalar(u8, s, '\'') != null and std.mem.indexOfScalar(u8, s, '"') == null) '"' else '\'';
    try buf.append(a, q);
    for (s) |c| {
        switch (c) {
            '\\' => try buf.appendSlice(a, "\\\\"),
            '\n' => try buf.appendSlice(a, "\\n"),
            '\r' => try buf.appendSlice(a, "\\r"),
            '\t' => try buf.appendSlice(a, "\\t"),
            0...8, 11, 12, 14...31, 127 => try buf.print(a, "\\x{x:0>2}", .{c}),
            else => {
                if (c == q) try buf.append(a, '\\');
                try buf.append(a, c);
            },
        }
    }
    try buf.append(a, q);
}

/// `json.dumps` with transformers' defaults (`ensure_ascii=False`).
const Json = struct {
    it: *Interp,
    ensure_ascii: bool,
    sort_keys: bool,
    indent: ?[]const u8 = null,
    item_sep: []const u8 = ", ",
    key_sep: []const u8 = ": ",

    fn newline(self: *Json, buf: *std.ArrayList(u8), level: usize) !void {
        const ind = self.indent orelse return;
        try buf.append(self.it.a, '\n');
        for (0..level) |_| try buf.appendSlice(self.it.a, ind);
    }

    fn write(self: *Json, buf: *std.ArrayList(u8), v: Value, level: usize) Error!void {
        const a = self.it.a;
        switch (v) {
            .none => try buf.appendSlice(a, "null"),
            .boolean => |b| try buf.appendSlice(a, if (b) "true" else "false"),
            .int => |i| try buf.print(a, "{d}", .{i}),
            .float => |f| {
                if (std.math.isNan(f)) return buf.appendSlice(a, "NaN");
                if (std.math.isInf(f)) return buf.appendSlice(a, if (f < 0) "-Infinity" else "Infinity");
                try writePyFloat(a, buf, f);
            },
            .string => |s| try self.string(buf, s),
            .list => |l| {
                if (l.kind == .generator) return self.it.fail("Object of type generator is not JSON serializable", .{});
                const items = l.items.items;
                if (items.len == 0) return buf.appendSlice(a, "[]");
                try buf.append(a, '[');
                for (items, 0..) |x, i| {
                    if (i > 0) try buf.appendSlice(a, self.item_sep);
                    try self.newline(buf, level + 1);
                    try self.write(buf, x, level + 1);
                }
                try self.newline(buf, level);
                try buf.append(a, ']');
            },
            .dict => |d| {
                if (d.keys.items.len == 0) return buf.appendSlice(a, "{}");
                const order = try a.alloc(usize, d.keys.items.len);
                for (order, 0..) |*o, i| o.* = i;
                if (self.sort_keys) {
                    const Ctx = struct {
                        keys: []const Value,
                        fn less(c: @This(), x: usize, y: usize) bool {
                            const p = c.keys[x];
                            const q = c.keys[y];
                            if (p == .string and q == .string) return std.mem.order(u8, p.string, q.string) == .lt;
                            return false;
                        }
                    };
                    std.sort.insertion(usize, order, Ctx{ .keys = d.keys.items }, Ctx.less);
                }
                try buf.append(a, '{');
                for (order, 0..) |idx, i| {
                    if (i > 0) try buf.appendSlice(a, self.item_sep);
                    try self.newline(buf, level + 1);
                    const k = d.keys.items[idx];
                    switch (k) {
                        .string => |s| try self.string(buf, s),
                        .int => |n| try buf.print(a, "\"{d}\"", .{n}),
                        .boolean => |b| try buf.appendSlice(a, if (b) "\"true\"" else "\"false\""),
                        .none => try buf.appendSlice(a, "\"null\""),
                        .float => |f| {
                            try buf.append(a, '"');
                            try writePyFloat(a, buf, f);
                            try buf.append(a, '"');
                        },
                        else => return self.it.fail("keys must be str, int, float, bool or None, not {s}", .{typeName(k)}),
                    }
                    try buf.appendSlice(a, self.key_sep);
                    try self.write(buf, d.values.items[idx], level + 1);
                }
                try self.newline(buf, level);
                try buf.append(a, '}');
            },
            else => return self.it.fail("Object of type {s} is not JSON serializable", .{typeName(v)}),
        }
    }

    fn string(self: *Json, buf: *std.ArrayList(u8), s: []const u8) !void {
        const a = self.it.a;
        try buf.append(a, '"');
        var it = Utf8Iter{ .s = s };
        while (it.next()) |c| {
            if (c.len == 1) {
                switch (c[0]) {
                    '"' => try buf.appendSlice(a, "\\\""),
                    '\\' => try buf.appendSlice(a, "\\\\"),
                    '\n' => try buf.appendSlice(a, "\\n"),
                    '\r' => try buf.appendSlice(a, "\\r"),
                    '\t' => try buf.appendSlice(a, "\\t"),
                    0x08 => try buf.appendSlice(a, "\\b"),
                    0x0c => try buf.appendSlice(a, "\\f"),
                    0...7, 0x0b, 0x0e...0x1f => try buf.print(a, "\\u{x:0>4}", .{c[0]}),
                    0x7f => if (self.ensure_ascii) try buf.appendSlice(a, "\\u007f") else try buf.append(a, 0x7f),
                    else => try buf.append(a, c[0]),
                }
            } else if (self.ensure_ascii) {
                const cp = decodeCp(c);
                if (cp >= 0x10000) {
                    const v = cp - 0x10000;
                    try buf.print(a, "\\u{x:0>4}\\u{x:0>4}", .{ 0xd800 + (v >> 10), 0xdc00 + (v & 0x3ff) });
                } else try buf.print(a, "\\u{x:0>4}", .{cp});
            } else try buf.appendSlice(a, c);
        }
        try buf.append(a, '"');
    }
};

const day_names = [_][]const u8{ "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday" };
const month_names = [_][]const u8{ "January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December" };

/// C's `strftime` (glibc's, which Python uses on Linux) for UTC `now`,
/// including the `%-d` no-padding flag.
fn strftime(a: Allocator, fmt: []const u8, now: i64) ![]const u8 {
    const epoch = std.time.epoch;
    const secs: u64 = @intCast(@max(now, 0));
    const es = epoch.EpochSeconds{ .secs = secs };
    const day = es.getEpochDay();
    const yd = day.calculateYearDay();
    const md = yd.calculateMonthDay();
    const ds = es.getDaySeconds();
    const year: u32 = yd.year;
    const month: u32 = md.month.numeric();
    const mday: u32 = md.day_index + 1;
    const hour: u32 = ds.getHoursIntoDay();
    const minute: u32 = ds.getMinutesIntoHour();
    const second: u32 = ds.getSecondsIntoMinute();
    const wday: usize = @intCast((day.day + 3) % 7); // 1970-01-01 was a Thursday
    var buf = std.ArrayList(u8).empty;
    var i: usize = 0;
    while (i < fmt.len) : (i += 1) {
        if (fmt[i] != '%' or i + 1 >= fmt.len) {
            try buf.append(a, fmt[i]);
            continue;
        }
        i += 1;
        var pad = true;
        if (fmt[i] == '-' and i + 1 < fmt.len) {
            pad = false;
            i += 1;
        }
        const Num2 = struct {
            fn put(b: *std.ArrayList(u8), al: Allocator, n: u32, width: usize, p: bool) !void {
                if (p) {
                    if (width == 3) try b.print(al, "{d:0>3}", .{n}) else try b.print(al, "{d:0>2}", .{n});
                } else try b.print(al, "{d}", .{n});
            }
        };
        switch (fmt[i]) {
            'Y' => try buf.print(a, "{d}", .{year}),
            'y' => try Num2.put(&buf, a, year % 100, 2, pad),
            'm' => try Num2.put(&buf, a, month, 2, pad),
            'd' => try Num2.put(&buf, a, mday, 2, pad),
            'e' => if (pad) try buf.print(a, "{d: >2}", .{mday}) else try buf.print(a, "{d}", .{mday}),
            'H' => try Num2.put(&buf, a, hour, 2, pad),
            'I' => try Num2.put(&buf, a, if (hour % 12 == 0) 12 else hour % 12, 2, pad),
            'M' => try Num2.put(&buf, a, minute, 2, pad),
            'S' => try Num2.put(&buf, a, second, 2, pad),
            'p' => try buf.appendSlice(a, if (hour < 12) "AM" else "PM"),
            'j' => try Num2.put(&buf, a, @as(u32, yd.day) + 1, 3, pad),
            'B' => try buf.appendSlice(a, month_names[month - 1]),
            'b', 'h' => try buf.appendSlice(a, month_names[month - 1][0..3]),
            'A' => try buf.appendSlice(a, day_names[wday]),
            'a' => try buf.appendSlice(a, day_names[wday][0..3]),
            'u' => try buf.print(a, "{d}", .{wday + 1}),
            'w' => try buf.print(a, "{d}", .{(wday + 1) % 7}),
            'F' => try buf.print(a, "{d}-{d:0>2}-{d:0>2}", .{ year, month, mday }),
            'D' => try buf.print(a, "{d:0>2}/{d:0>2}/{d:0>2}", .{ month, mday, year % 100 }),
            'T' => try buf.print(a, "{d:0>2}:{d:0>2}:{d:0>2}", .{ hour, minute, second }),
            'Z' => try buf.appendSlice(a, "UTC"),
            'z' => try buf.appendSlice(a, "+0000"),
            '%' => try buf.append(a, '%'),
            else => {
                try buf.append(a, '%');
                try buf.append(a, fmt[i]);
            },
        }
    }
    return buf.items;
}

fn renderForTest(a: Allocator, src: []const u8, vars_json: []const u8, now: i64, diag: *Diagnostic) ![]const u8 {
    const t = try Template.parse(std.testing.allocator, src, diag);
    defer t.deinit();
    const parsed = try std.json.parseFromSliceLeaky(std.json.Value, a, vars_json, .{});
    var vars = std.ArrayList(Var).empty;
    var it = parsed.object.iterator();
    while (it.next()) |e| try vars.append(a, .{ .name = e.key_ptr.*, .value = try Value.fromJson(a, e.value_ptr.*) });
    return a.dupe(u8, try t.render(a, vars.items, .{ .now = now }, diag));
}

test "jinja: expressions, statements and whitespace against jinja2" {
    // Each `want` is jinja2's output in transformers' environment (null: jinja2 raises).
    const vars =
        \\{"messages": [{"role": "system", "content": "Sys  \n"}, {"role": "user", "content": " Hi é {x} 'q' \"d\""}, {"role": "assistant", "content": "<think>a</think>Ans"}, {"role": "user", "content": "Again"}], "bos_token": "<s>", "eos_token": "</s>", "add_generation_prompt": true}
    ;
    const cases = [_]struct { t: []const u8, want: ?[]const u8 }{
        .{ .t = "{% for i in [1,2,3] %}{% if i==1 %}{% set y = 5 %}{% endif %}{{ y }},{% endfor %}", .want = "5,,," },
        .{ .t = "{% set y=1 %}{% for i in [1,2] %}{% set y = y+1 %}{{y}}{% endfor %}{{y}}", .want = "221" },
        .{ .t = "{{ none }}|{{ true }}|{{ 1.0 }}|{{ [1,'a',none,1.5] }}|{{ {'a':1} }}|{{ 1e20 }}|{{ 0.00001 }}|{{ (1,2) }}|{{ \"it's\" }}|{{ [\"it's\"] }}|{{ (1,) }}|{{ 1/3 }}|{{ 2.5e-7 }}|{{ 123456789012345678.0 }}", .want = "None|True|1.0|[1, 'a', None, 1.5]|{'a': 1}|1e+20|1e-05|(1, 2)|it's|[\"it's\"]|(1,)|0.3333333333333333|2.5e-07|1.2345678901234568e+17" },
        .{ .t = "{{ x.y }}", .want = null },
        .{ .t = "{{ x.y.z }}", .want = null },
        .{ .t = "{{ 'a' in x }}|{{ x|length }}|{{ x is defined }}|{{ x ~ 1 }}", .want = "False|0|False|1" },
        .{ .t = "{{ 7/2 }} {{ 7//2 }} {{ -7//2 }} {{ -7 % 3 }} {{ 2**10 }} {{ 4/2 }} {{ 7.5 // 2 }} {{ -7.5 % 2 }}", .want = "3.5 3 -4 2 1024 2.0 3.0 0.5" },
        .{ .t = "{{ [3,1,2]|sort }} {{ 'abc'[::-1] }} {{ 'héllo'[1:3] }} {{ 'héllo'|length }} {{ [1,2,3,4][1:] }} {{ [1,2,3][-1] }} {{ 'abc'[-2:] }} {{ [1,2,3,4][::2] }} {{ [1,2,3][5] }}", .want = "[1, 2, 3] cba él 5 [2, 3, 4] 3 bc [1, 3] " },
        .{ .t = "{{ {'b':1,'a':[1,{'x':'é'}]}|tojson }}|{{ {'b':1}|tojson(indent=2) }}|{{ 'a\\nb\\t\"\\\\'|tojson }}|{{ [1,[2,{}],[]]|tojson(indent=4) }}|{{ {'b':1,'a':2}|tojson(sort_keys=true) }}|{{ 'é'|tojson(ensure_ascii=true) }}|{{ {1:2, none: 3, true: 4}|tojson }}|{{ [1.0, 0.5, none, true]|tojson }}", .want = "{\"b\": 1, \"a\": [1, {\"x\": \"é\"}]}|{\n  \"b\": 1\n}|\"a\\nb\\t\\\"\\\\\"|[\n    1,\n    [\n        2,\n        {}\n    ],\n    []\n]|{\"a\": 2, \"b\": 1}|\"\\u00e9\"|{\"1\": 4, \"null\": 3}|[1.0, 0.5, null, true]" },
        .{ .t = "{{ x | tojson }}", .want = null },
        .{ .t = "  {% if true %}\n  x\n  {% endif %}\n  y  {%- if true %} z {% endif -%}  \n w", .want = "  x\n  y z w" },
        .{ .t = "a\n  {# c #}\nb {{ 1 }}\n  {{ 2 }}\n", .want = "a\nb 1\n  2" },
        .{ .t = "{{ '%s-%d' % ('a', 3) }} {{ '{}'.format(2) }} {{ '%05.2f|%-4s|%4s|%%' % (3.14159, 'ab', 'cd') }} {{ '%s' % 'x' }}", .want = "a-3 2 03.14|ab  |  cd|% x" },
        .{ .t = "{{ 'a b  c'.split() }} {{ 'a,b'.split(',') }} {{ ' x '.strip() }} {{ 'xax'.strip('x') }} {{ 'a b c'.split(' ', 1) }} {{ 'a,b,c'.rsplit(',', 1) }} {{ '  a  b '.split(none, 1) }}", .want = "['a', 'b', 'c'] ['a', 'b'] x a ['a', 'b c'] ['a,b', 'c'] ['a', 'b ']" },
        .{ .t = "{% set ns = namespace(a=1) %}{% for i in range(3) %}{% set ns.a = ns.a + i %}{% endfor %}{{ ns.a }}", .want = "4" },
        .{ .t = "{{ undefined_thing.items() }}", .want = null },
        .{ .t = "{% for m in [1,2,3] %}{{ loop.cycle('a','b') }}{{ loop.revindex }}{{ loop.index0 }}{{ loop.first }}{{ loop.last }}{{ loop.length }}{% endfor %}", .want = "a30TrueFalse3b21FalseFalse3a12FalseTrue3" },
        .{ .t = "{{ True }}{{ False }}{{ None }}", .want = "TrueFalseNone" },
        .{ .t = "{% for m in messages %}{{ m.role|upper }}:{{ m['content']|trim }};{% endfor %}", .want = "SYSTEM:Sys;USER:Hi é {x} 'q' \"d\";ASSISTANT:<think>a</think>Ans;USER:Again;" },
        .{ .t = "{% for m in messages if m.role != 'system' %}{{ loop.index }}{{ m.role }}{% else %}none{% endfor %}", .want = "1user2assistant3user" },
        .{ .t = "{% for m in [] %}x{% else %}empty{% endfor %}", .want = "empty" },
        .{ .t = "{{ messages|selectattr('role', 'equalto', 'user')|list|length }} {{ messages|map(attribute='role')|join(',') }} {{ messages|rejectattr('role','in',['user','system'])|map(attribute='content')|first }}", .want = "2 system,user,assistant,user <think>a</think>Ans" },
        .{ .t = "{{ messages|selectattr('role', 'equalto', 'user')|length }}", .want = null },
        .{ .t = "{% if messages|selectattr('role','equalto','nobody') %}gen-true{% endif %}", .want = "gen-true" },
        .{ .t = "{% set g = messages|map(attribute='role') %}{{ g|join }}|{{ g|join }}", .want = "systemuserassistantuser|" },
        .{ .t = "{{ messages[0].content.startswith('Sys') }} {{ messages[1].content.endswith(('x', '\"')) }} {{ 'Abc'.lower() }} {{ 'abc def'.title() }} {{ 'aBC'.capitalize() }} {{ 'a-b'.replace('-', '+') }}", .want = "True True abc Abc Def Abc a+b" },
        .{ .t = "{{ {'a':1,'b':2}.items()|list }} {{ {'a':1}.get('a') }} {{ {'a':1}.get('z', 'd') }} {{ {'a':1}.get('z') }} {{ {'a':1}.keys()|list }} {% for k, v in {'x': 1, 'y': 2}.items() %}{{k}}={{v}};{% endfor %}", .want = "[('a', 1), ('b', 2)] 1 d None ['a'] x=1;y=2;" },
        .{ .t = "{% for k, v in {'x': 1, 'y': 2}|dictsort %}{{k}}{{v}}{% endfor %}{% for k, v in {'b': 1, 'a': 2}|dictsort(by='value', reverse=true) %}{{k}}{% endfor %}{% for k in {'q':1,'p':2}|items %}{{ k }}{% endfor %}", .want = "x1y2ab('q', 1)('p', 2)" },
        .{ .t = "{{ x|default('d') }} {{ ''|default('e') }} {{ ''|default('e', true) }} {{ none|default('n') }} {{ x|d('dd') }}", .want = "d  e None dd" },
        .{ .t = "{{ '  hi  '|trim }}|{{ 'xxhixx'|trim('x') }}|{{ 'a\\nb\\n\\nc'|indent(2) }}|{{ 'a\\nb'|indent(2, true) }}|{{ 'a\\n\\nb'|indent(2, blank=true) }}|{{ 'a\\nb'|indent('> ') }}", .want = "hi|hi|a\n  b\n\n  c|  a\n  b|a\n  \n  b|a\n> b" },
        .{ .t = "{{ [1,2,3]|first }} {{ [1,2,3]|last }} {{ 'abc'|first }} {{ []|first }} {{ [3,1,2]|max }} {{ [3,1,2]|min }} {{ [1,2,3]|sum }} {{ ['b','A','a']|sort }} {{ ['b','A','a']|unique|list }} {{ [1,2]|reverse|list }} {{ 'abc'|reverse }}", .want = "1 3 a  3 1 6 ['A', 'a', 'b'] ['b', 'A'] [2, 1] cba" },
        .{ .t = "{{ 3|string }}{{ '3'|int + 1 }}{{ 'x'|int }}{{ 3.7|int }}{{ '2.5'|float }}{{ -3|abs }}{{ 2.567|round(2) }}{{ 2.5|round }}{{ 3|float }}", .want = "34032.532.572.03.0" },
        .{ .t = "{{ 'a' ~ 1 ~ none ~ true }} {{ 'ab' * 3 }} {{ [1] * 2 }} {{ [1,2] + [3] }} {{ 'a' + 'b' }}", .want = "a1NoneTrue ababab [1, 1] [1, 2, 3] ab" },
        .{ .t = "{{ 1 < 2 < 3 }} {{ 1 == 1.0 }} {{ true == 1 }} {{ 'a' < 'b' }} {{ [1,2] == [1,2] }} {{ (1,2) == [1,2] }} {{ {'a':1} == {'a':1} }} {{ none == none }} {{ x == none }} {{ x is none }}", .want = "True True True True True False True True False False" },
        .{ .t = "{{ 'ab' in 'cabd' }} {{ 1 in [1,2] }} {{ 'a' in {'a':1} }} {{ 3 not in [1] }} {{ not true }} {{ true and 'x' }} {{ false or 'y' }} {{ 0 or none }} {{ '' and 1 }}", .want = "True True True True False x y None " },
        .{ .t = "{{ 1 if true else 2 }} {{ 1 if false else 2 }} {{ (1 if false) }} {{ 'a' if x is defined else 'b' }}", .want = "1 2  b" },
        .{ .t = "{% macro f(a, b='B') %}[{{a}}{{b}}{{ varargs }}{{ kwargs }}]{% endmacro %}{{ f(1) }}{{ f(1, 2) }}{{ f(1, b=3) }}{{ f(1,2,3,z=4) }}", .want = "[1B(){}][12(){}][13(){}][12(3,){'z': 4}]" },
        .{ .t = "{% macro w() %}<{{ caller() }}>{% endmacro %}{% call w() %}inner{% endcall %}", .want = "<inner>" },
        .{ .t = "{% set x %}a {{ 1 }} b{% endset %}[{{ x }}]{% set y | upper %}q{% endset %}{{ y }}", .want = "[a 1 b]Q" },
        .{ .t = "{% filter upper %}abc{% endfilter %}", .want = "ABC" },
        .{ .t = "{% for i in range(5) %}{% if i == 1 %}{% continue %}{% endif %}{% if i == 3 %}{% break %}{% endif %}{{ i }}{% endfor %}", .want = "02" },
        .{ .t = "{% for i in range(1, 7, 2) %}{{ i }}{% endfor %}{{ range(3)|list }}", .want = "135[0, 1, 2]" },
        .{ .t = "{% for a, b in [[1,2],[3,4]] %}{{ a }}{{ b }}{% endfor %}{% set p, q = 1, 2 %}{{ p }}{{ q }}", .want = "123412" },
        .{ .t = "{{ raise_exception('boom') }}", .want = null },
        .{ .t = "{{ strftime_now('%d %b %Y') }}|{{ strftime_now('%Y-%m-%d %H:%M:%S %A %a %B %j %-d %e %I %p') }}", .want = "21 Sep 2026|2026-09-21 14:13:20 Monday Mon September 264 21 21 02 PM" },
        .{ .t = "{{ bos_token }}{% for m in messages %}{{ '<|im_start|>' + m['role'] + '\\n' + m['content'] + '<|im_end|>' + '\\n' }}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}", .want = "<s><|im_start|>system\nSys  \n<|im_end|>\n<|im_start|>user\n Hi é {x} 'q' \"d\"<|im_end|>\n<|im_start|>assistant\n<think>a</think>Ans<|im_end|>\n<|im_start|>user\nAgain<|im_end|>\n<|im_start|>assistant\n" },
        .{ .t = "{%- if messages[0]['role'] == 'system' -%}\n    {%- set system_message = messages[0]['content'] -%}\n    {%- set loop_messages = messages[1:] -%}\n{%- else -%}\n    {%- set loop_messages = messages -%}\n{%- endif -%}\n{{ system_message }}|{{ loop_messages|length }}", .want = "Sys  \n|3" },
        .{ .t = "{{ messages[-1]['content'] }} {{ messages.0.role }} {{ messages|length > 2 }}", .want = "Again system True" },
        .{ .t = "{% raw %}{{ not rendered }}{% endraw %}", .want = "{{ not rendered }}" },
        .{ .t = "{{ 'x' is string }}{{ 1 is number }}{{ true is number }}{{ 1 is integer }}{{ true is integer }}{{ {} is mapping }}{{ [] is iterable }}{{ 'a' is iterable }}{{ x is iterable }}{{ 3 is odd }}{{ 4 is even }}{{ 6 is divisibleby 3 }}{{ 'a' is in 'abc' }}{{ 1 is equalto 1 }}{{ x is sequence }}{{ none is none }}{{ 'ABC' is upper }}{{ messages[0] is mapping }}", .want = "TrueTrueTrueTrueFalseTrueTrueTrueTrueTrueTrueTrueTrueTrueTrueTrueTrueTrue" },
        .{ .t = "{% set d = {'k': [1, 2]} %}{{ d.k[1] }} {{ d['k']|length }}", .want = "2 2" },
        .{ .t = "{{ 'multi' ' concat' }} {{ \"a\\u00e9\\x41\" }} {{ 'a\\'b' }}", .want = "multi concat aéA a'b" },
        .{ .t = "{% for x in 'ab' %}{{ x }}{{ loop.previtem }}{{ loop.nextitem }}|{% endfor %}", .want = "ab|ba|" },
        .{ .t = "{{ messages | map('upper') | list }}", .want = "[\"{'ROLE': 'SYSTEM', 'CONTENT': 'SYS  \\\\N'}\", '{\\'ROLE\\': \\'USER\\', \\'CONTENT\\': \\' HI É {X} \\\\\\'Q\\\\\\' \"D\"\\'}', \"{'ROLE': 'ASSISTANT', 'CONTENT': '<THINK>A</THINK>ANS'}\", \"{'ROLE': 'USER', 'CONTENT': 'AGAIN'}\"]" },
        .{ .t = "{{ ['a','b']|map('upper')|list }} {{ [' a ', 'b ']|map('trim')|join(',') }} {{ ['a', 1, none]|select('string')|list }} {{ [0,1,2]|select|list }} {{ [0,1,2]|reject|list }}", .want = "['A', 'B'] a,b ['a'] [1, 2] [0]" },
        .{ .t = "{{ x.y is defined }}", .want = null },
        .{ .t = "{{ none.x }}|{{ (none.x) is defined }}", .want = "|False" },
        .{ .t = "{{ 1 + x }}", .want = null },
        .{ .t = "{% if x %}yes{% elif 1 %}elif{% else %}no{% endif %}", .want = "elif" },
        .{ .t = "{% with a = 1, b = 2 %}{{ a + b }}{% endwith %}", .want = "3" },
        .{ .t = "{{- '  a  ' -}}   {{- 'b' }}  \n{%- if true %}\n c\n{%- endif %}\n", .want = "  a  b c" },
        .{ .t = "line1\n    {%- if true -%}\n    x\n    {%- endif -%}\n\nline2", .want = "line1xline2" },
        .{ .t = "a\n{% if true %}\nb\n{% endif %}\nc\n", .want = "a\nb\nc" },
        .{ .t = "a  {% if true %}b{% endif %}  c\n  {%+ if true %}d{% endif %}", .want = "a  b  c\n  d" },
        .{ .t = "{{ \"%d items\"|format(3) }} {{ '{0}-{1}'.format('a', 'b') }} {{ '{x}'.format(x=1) }}", .want = "3 items a-b 1" },
        .{ .t = "{{ 'abc'.upper().lower() }} {{ ' a b '.strip().split(' ') }} {{ 'a\\nb\\r\\nc'.splitlines() }}", .want = "abc ['a', 'b'] ['a', 'b', 'c']" },
        .{ .t = "{{ [1,2,3]|batch(2)|list }} {{ 'ab'|center(6) }}|{{ 'abc'|center(6) }}|", .want = "[[1, 2], [3]]   ab  | abc  |" },
        .{ .t = "{{ tools }}{% if tools is not none %}T{% endif %}{% if tools %}X{% endif %}", .want = "T" },
        .{ .t = "{{ [{'a': {'b': 1}}, {'a': {'b': 0}}]|selectattr('a.b')|list }}", .want = "[{'a': {'b': 1}}]" },
        .{ .t = "{% set x = [1, 2] %}{{ x.append(3) }}", .want = null },
        .{ .t = "{{ foo(1) }}", .want = null },
        .{ .t = "{{ 'a' + 1 }}", .want = null },
        .{ .t = "{{ [1,2]|length }}{{ {'a':1,'b':2}|length }}{{ ''|length }}", .want = "220" },
        .{ .t = "{{ messages|last|tojson }}", .want = "{\"role\": \"user\", \"content\": \"Again\"}" },
        .{ .t = "{% for m in messages %}{% if loop.index0 == 0 and m.role == 'system' %}{% continue %}{% endif %}{{ m.role[0] }}{% endfor %}", .want = "uau" },
        .{ .t = "{{ 'a' if 'b' in 'abc' and not false else 'c' }}", .want = "a" },
        .{ .t = "{{ -1 }} {{ - 2 + 3 }} {{ -(2) }} {{ 2 ** -1 }} {{ 10 % 3 }} {{ 1_000 }}", .want = "-1 1 -2 0.5 1 1000" },
        .{ .t = "{% if not messages %}e{% endif %}{% if messages is defined and messages|length > 0 %}ne{% endif %}", .want = "ne" },
        .{ .t = "{{ messages[1]['content'][:3] }}|{{ messages[1]['content'][3:] }}", .want = " Hi| é {x} 'q' \"d\"" },
        .{ .t = "{{ 'x' ~ ['a'] ~ {'k': 'v'} }}", .want = "x['a']{'k': 'v'}" },
        .{ .t = "{{ 'Hello {name}'.replace('{name}', 'W') }}", .want = "Hello W" },
        .{ .t = "{%- for message in messages %}\n{{- '<|' + message.role + '|>\\n' }}\n{%- endfor %}", .want = "<|system|>\n<|user|>\n<|assistant|>\n<|user|>\n" },
        .{ .t = "{% set content = messages[2].content %}{% if '</think>' in content %}{{ content.split('</think>')[-1].lstrip('\\n') }}|{{ content.split('</think>')[0].rstrip('\\n').split('<think>')[-1].lstrip('\\n') }}{% endif %}", .want = "Ans|a" },
        .{ .t = "{{ [1, 2, 3] | join(', ') }} {{ [1,2] | join }}", .want = "1, 2, 3 12" },
        .{ .t = "{{ {'role': 'user'} | items | list }}", .want = "[('role', 'user')]" },
        .{ .t = "{%- set ns = namespace(found=false, idx=0) -%}{%- for m in messages[::-1] -%}{%- if not ns.found and m.role == 'user' -%}{%- set ns.found = true -%}{%- set ns.idx = messages|length - 1 - loop.index0 -%}{%- endif -%}{%- endfor -%}{{ ns.idx }}", .want = "3" },
    };
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    for (cases) |c| {
        var diag: Diagnostic = .{};
        const got = renderForTest(arena.allocator(), c.t, vars, 1790000000, &diag) catch |e| {
            if (e == error.TemplateError and c.want == null) continue;
            std.debug.print("{s}\n  failed: {s}\n", .{ c.t, diag.message() });
            return e;
        };
        if (c.want) |w| {
            std.testing.expectEqualStrings(w, got) catch |e| {
                std.debug.print("template: {s}\n", .{c.t});
                return e;
            };
        } else {
            std.debug.print("{s}\n  rendered, but jinja2 raises\n", .{c.t});
            return error.TestExpectedError;
        }
    }
}

test "jinja: real chat templates render as transformers renders them" {
    // tests/fixtures/chat_templates: releases' own templates and transformers'
    // renderings of them (tools/make_template_fixtures.py).
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    const dir = "tests/fixtures/chat_templates";
    const cwd = std.Io.Dir.cwd();
    const expected = try std.json.parseFromSliceLeaky(std.json.Value, a, try cwd.readFileAlloc(io, dir ++ "/expected.json", a, .unlimited), .{});
    const now = expected.object.get("now").?.integer;
    var checked: usize = 0;
    for (expected.object.get("cases").?.array.items) |case| {
        const c = case.object;
        const name = c.get("template").?.string;
        const path = try std.fmt.allocPrint(a, dir ++ "/{s}.jinja", .{name});
        const src = try cwd.readFileAlloc(io, path, a, .unlimited);
        var diag: Diagnostic = .{};
        const t = try Template.parse(gpa, src, &diag);
        defer t.deinit();
        var vars = std.ArrayList(Var).empty;
        var tok_it = c.get("tokens").?.object.iterator();
        while (tok_it.next()) |e| try vars.append(a, .{ .name = e.key_ptr.*, .value = .{ .string = e.value_ptr.*.string } });
        try vars.append(a, .{ .name = "messages", .value = try Value.fromJson(a, c.get("messages").?) });
        try vars.append(a, .{ .name = "tools", .value = .none });
        try vars.append(a, .{ .name = "documents", .value = .none });
        try vars.append(a, .{ .name = "add_generation_prompt", .value = .{ .boolean = c.get("add_generation_prompt").?.bool } });
        var kw_it = c.get("kwargs").?.object.iterator();
        while (kw_it.next()) |e| try vars.append(a, .{ .name = e.key_ptr.*, .value = try Value.fromJson(a, e.value_ptr.*) });
        const got = t.render(a, vars.items, .{ .now = now }, &diag);
        if (c.get("want")) |want| {
            const text = got catch |e| {
                std.debug.print("{s}: {s}\n", .{ name, diag.message() });
                return e;
            };
            std.testing.expectEqualStrings(want.string, text) catch |e| {
                std.debug.print("template {s}\n", .{name});
                return e;
            };
        } else {
            try std.testing.expectError(error.TemplateError, got);
        }
        checked += 1;
    }
    try std.testing.expect(checked >= 100);
}

test "jinja: parse errors carry the line" {
    var diag: Diagnostic = .{};
    try std.testing.expectError(error.TemplateError, Template.parse(std.testing.allocator, "a\n{% if x %}\nb", &diag));
    try std.testing.expect(std.mem.indexOf(u8, diag.message(), "endif") != null);
    try std.testing.expectError(error.TemplateError, Template.parse(std.testing.allocator, "{{ x | nosuchfilter }}", &diag));
    try std.testing.expectError(error.TemplateError, Template.parse(std.testing.allocator, "\n\n{{ 'a' ", &diag));
    try std.testing.expect(std.mem.startsWith(u8, diag.message(), "line 3"));
}

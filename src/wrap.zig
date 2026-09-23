//! Reflows the help text to the terminal's width. The help is written for a
//! wide terminal: entries are a term (a command or an option) with its
//! description in a column after two or more spaces, and prose is indented
//! text. Lines that continue an entry's description or a paragraph are joined
//! first, then every item is wrapped at word boundaries: a description keeps
//! its column while at least `min_description` columns remain beside it and
//! otherwise moves below its term.

const std = @import("std");

/// Never wrap wider than this, however wide the terminal: long lines read badly.
pub const max_columns: usize = 100;
const min_description = 28;

const Item = struct {
    indent: usize,
    /// The command or option of an entry; empty for prose.
    term: []const u8 = "",
    /// The column the description starts in (entries only).
    column: usize = 0,
    /// Words of the description or the paragraph.
    words: std.ArrayList([]const u8) = .empty,
    blank: bool = false,
};

/// Visible width of `s`, not counting ANSI escape sequences.
fn visibleWidth(s: []const u8) usize {
    var n: usize = 0;
    var i: usize = 0;
    while (i < s.len) {
        if (s[i] == 0x1b) {
            while (i < s.len and s[i] != 'm') i += 1;
            i += 1;
            continue;
        }
        // Count UTF-8 lead bytes only.
        if (s[i] & 0xC0 != 0x80) n += 1;
        i += 1;
    }
    return n;
}

fn appendWords(a: std.mem.Allocator, list: *std.ArrayList([]const u8), text: []const u8) !void {
    var it = std.mem.tokenizeScalar(u8, text, ' ');
    while (it.next()) |word| try list.append(a, word);
}

/// Writes `text` wrapped to `columns` (capped at `max_columns`).
pub fn write(a: std.mem.Allocator, w: *std.Io.Writer, text: []const u8, columns: usize) !void {
    const cols = @max(@min(columns, max_columns), 20);
    var items: std.ArrayList(Item) = .empty;
    var lines = std.mem.splitScalar(u8, text, '\n');
    while (lines.next()) |raw| {
        const line = std.mem.trimEnd(u8, raw, " ");
        if (line.len == 0) {
            try items.append(a, .{ .indent = 0, .blank = true });
            continue;
        }
        const indent = line.len - std.mem.trimStart(u8, line, " ").len;
        const body = line[indent..];
        // An entry: a term, then two or more spaces, then the description.
        const gap = std.mem.indexOf(u8, body, "  ");
        if (gap) |g| {
            const rest = std.mem.trimStart(u8, body[g..], " ");
            if (rest.len > 0) {
                var item: Item = .{ .indent = indent, .term = body[0..g], .column = indent + body.len - rest.len };
                try appendWords(a, &item.words, rest);
                try items.append(a, item);
                continue;
            }
        }
        // A continuation: indented to the previous entry's description, or at
        // the indent of the previous paragraph (not a heading).
        if (items.items.len > 0) {
            const prev = &items.items[items.items.len - 1];
            const continues = !prev.blank and if (prev.term.len > 0)
                indent == prev.column
            else
                indent == prev.indent and prev.words.items.len > 0 and
                    !std.mem.endsWith(u8, prev.words.items[prev.words.items.len - 1], ":");
            if (continues) {
                try appendWords(a, &prev.words, body);
                continue;
            }
        }
        var item: Item = .{ .indent = indent };
        try appendWords(a, &item.words, body);
        try items.append(a, item);
    }
    // The split leaves one empty item for the final newline.
    if (items.items.len > 0 and items.items[items.items.len - 1].blank) _ = items.pop();

    for (items.items) |item| {
        if (item.blank) {
            try w.writeAll("\n");
            continue;
        }
        try w.splatByteAll(' ', item.indent);
        var start = item.indent;
        if (item.term.len > 0) {
            var term: std.ArrayList([]const u8) = .empty;
            try appendWords(a, &term, item.term);
            const after = try writeWords(w, term.items, item.indent, item.indent + 2, cols);
            if (item.column >= after + 2 and item.column + min_description <= cols) {
                try w.splatByteAll(' ', item.column - after);
                start = item.column;
            } else {
                start = item.indent + 4;
                try w.writeAll("\n");
                try w.splatByteAll(' ', start);
            }
        }
        _ = try writeWords(w, item.words.items, start, start, cols);
        try w.writeAll("\n");
    }
}

/// Writes `words` from column `x`, breaking lines before `cols` and indenting
/// continuations to `cont`; returns the column after the last word.
fn writeWords(w: *std.Io.Writer, words: []const []const u8, x0: usize, cont: usize, cols: usize) !usize {
    var x = x0;
    for (words, 0..) |word, k| {
        const len = visibleWidth(word);
        if (k > 0) {
            if (x + 1 + len > cols) {
                try w.writeAll("\n");
                try w.splatByteAll(' ', cont);
                x = cont;
            } else {
                try w.writeAll(" ");
                x += 1;
            }
        }
        try w.writeAll(word);
        x += len;
    }
    return x;
}

fn wrapped(text: []const u8, cols: usize) ![]u8 {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    var out: std.Io.Writer.Allocating = .init(std.testing.allocator);
    errdefer out.deinit();
    try write(arena.allocator(), &out.writer, text, cols);
    return out.toOwnedSlice();
}

test "an entry keeps its description column and wraps under it" {
    const got = try wrapped(
        \\  --max-ram <size>     Keep resident memory under this budget, for example 8GB or 12GB.
        \\
    , 64);
    defer std.testing.allocator.free(got);
    try std.testing.expectEqualStrings(
        \\  --max-ram <size>     Keep resident memory under this budget,
        \\                       for example 8GB or 12GB.
        \\
    , got);
}

test "continuation lines are joined before wrapping" {
    const got = try wrapped(
        \\  --accelerate <bool>  Use Apple's Accelerate framework
        \\                       on macOS (default: on).
        \\
    , 100);
    defer std.testing.allocator.free(got);
    try std.testing.expectEqualStrings(
        \\  --accelerate <bool>  Use Apple's Accelerate framework on macOS (default: on).
        \\
    , got);
}

test "a narrow terminal moves the description below its term" {
    const got = try wrapped(
        \\Common options:
        \\  --n-trials <n>                 Total trials (default: 200).
        \\
    , 40);
    defer std.testing.allocator.free(got);
    try std.testing.expectEqualStrings(
        \\Common options:
        \\  --n-trials <n>
        \\      Total trials (default: 200).
        \\
    , got);
}

test "paragraphs reflow and headings stay on their own line" {
    const got = try wrapped(
        \\Usage:
        \\A model id, a local directory
        \\or a .gguf file.
        \\
    , 24);
    defer std.testing.allocator.free(got);
    try std.testing.expectEqualStrings(
        \\Usage:
        \\A model id, a local
        \\directory or a .gguf
        \\file.
        \\
    , got);
}

test "no line of the help is wider than the terminal" {
    const help = @import("config.zig").help_text;
    for ([_]usize{ 40, 60, 80, 100 }) |cols| {
        const got = try wrapped(help, cols);
        defer std.testing.allocator.free(got);
        var it = std.mem.splitScalar(u8, got, '\n');
        while (it.next()) |line| {
            // A single word longer than the line (a URL) is left whole.
            if (std.mem.indexOfScalar(u8, std.mem.trim(u8, line, " "), ' ') == null) continue;
            try std.testing.expect(visibleWidth(line) <= cols);
        }
    }
}

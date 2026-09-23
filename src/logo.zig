//! The ditch wordmark: the word drawn in shaded characters with a thread woven
//! through it. `logo.txt` holds it with colour marks: `{W}` starts the letters,
//! `{R}` the thread and `{0}` resets. install.sh and install.ps1 carry copies.
//! A terminal narrower than the art gets a one-line form instead of wrapped rows.

const std = @import("std");
const builtin = @import("builtin");

const art = @embedFile("logo.txt");
const compact = "  {W}ditch {R}--.__.-'`{0}\n";

/// The widest row of the art, without its colour marks.
pub const width: usize = blk: {
    @setEvalBranchQuota(100_000);
    var widest: usize = 0;
    var row: usize = 0;
    var i: usize = 0;
    while (i < art.len) : (i += 1) {
        if (art[i] == '\n') {
            widest = @max(widest, row);
            row = 0;
        } else if (art[i] == '{' and i + 2 < art.len and art[i + 2] == '}') {
            i += 2;
        } else row += 1;
    }
    break :blk @max(widest, row);
};

/// Light grey letters and a soft red thread, from the 256-colour palette so
/// that terminals without 24-bit colour show them too.
const letters = "\x1b[38;5;252m";
const thread = "\x1b[38;5;174m";
const reset = "\x1b[0m";

/// Writes the wordmark; `columns` is the terminal's width when known, and the
/// one-line form is written when the art would not fit in it.
pub fn write(w: *std.Io.Writer, color: bool, columns: ?u16) std.Io.Writer.Error!void {
    const fits = if (columns) |c| c > width else true;
    try writeMarked(w, if (fits) art else compact, color);
}

fn writeMarked(w: *std.Io.Writer, text: []const u8, color: bool) std.Io.Writer.Error!void {
    var i: usize = 0;
    while (i < text.len) {
        if (text[i] == '{' and i + 2 < text.len and text[i + 2] == '}') {
            if (color) try w.writeAll(switch (text[i + 1]) {
                'W' => letters,
                'R' => thread,
                else => reset,
            });
            i += 3;
            continue;
        }
        const end = std.mem.indexOfScalarPos(u8, text, i + 1, '{') orelse text.len;
        try w.writeAll(text[i..end]);
        i = end;
    }
}

/// The width of the terminal `file` is attached to, or null when it is not a
/// terminal or cannot be asked.
pub fn terminalColumns(io: std.Io, file: std.Io.File) ?u16 {
    if (builtin.os.tag == .windows) {
        var info = std.os.windows.CONSOLE.USER_IO.GET_SCREEN_BUFFER_INFO;
        const status = info.operate(io, file) catch return null;
        if (status != .SUCCESS) return null;
        const cols = info.Data.dwWindowSize.X;
        return if (cols > 0) @intCast(cols) else null;
    } else {
        var ws: std.posix.winsize = .{ .row = 0, .col = 0, .xpixel = 0, .ypixel = 0 };
        const result = io.operate(.{ .device_io_control = .{
            .file = file,
            .code = std.posix.T.IOCGWINSZ,
            .arg = &ws,
        } }) catch return null;
        if (result.device_io_control < 0 or ws.col == 0) return null;
        return ws.col;
    }
}

test "the plain logo has no colour marks and every row fits in its width" {
    var buf: [4096]u8 = undefined;
    var w: std.Io.Writer = .fixed(&buf);
    try write(&w, false, null);
    const plain = w.buffered();
    try std.testing.expect(std.mem.indexOfAny(u8, plain, "{}\x1b") == null);
    var rows = std.mem.splitScalar(u8, plain, '\n');
    while (rows.next()) |row| try std.testing.expect(row.len <= width);
    try std.testing.expect(width < 80);
}

test "a narrow terminal gets the one-line wordmark" {
    var buf: [4096]u8 = undefined;
    var w: std.Io.Writer = .fixed(&buf);
    try write(&w, false, 60);
    try std.testing.expectEqualStrings("  ditch --.__.-'`\n", w.buffered());
    w = .fixed(&buf);
    try write(&w, false, @intCast(width + 1));
    try std.testing.expect(w.buffered().len > 200);
}

test "the coloured logo draws the same characters" {
    var a: [4096]u8 = undefined;
    var b: [8192]u8 = undefined;
    var plain: std.Io.Writer = .fixed(&a);
    var colored: std.Io.Writer = .fixed(&b);
    try write(&plain, false, null);
    try write(&colored, true, null);
    var stripped: [4096]u8 = undefined;
    var n: usize = 0;
    var i: usize = 0;
    const c = colored.buffered();
    while (i < c.len) {
        if (c[i] == 0x1b) {
            i = std.mem.indexOfScalarPos(u8, c, i, 'm').? + 1;
            continue;
        }
        stripped[n] = c[i];
        n += 1;
        i += 1;
    }
    try std.testing.expectEqualStrings(plain.buffered(), stripped[0..n]);
}

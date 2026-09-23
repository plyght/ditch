//! The ditch wordmark: the word drawn in braille dots with a thread woven
//! through it. `logo.txt` holds it with colour marks: `{W}` starts the letters,
//! `{R}` the thread and `{0}` resets. install.sh and install.ps1 carry copies.
//! A terminal narrower than the art, or one that cannot show braille, gets a
//! one-line form instead.

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
        } else if (art[i] & 0xC0 != 0x80) row += 1;
    }
    break :blk @max(widest, row);
};

/// Light grey letters and a soft red thread, from the 256-colour palette so
/// that terminals without 24-bit colour show them too.
const letters = "\x1b[38;5;252m";
const thread = "\x1b[38;5;174m";
const reset = "\x1b[0m";

/// Writes the wordmark; `columns` is the terminal's width when known, and the
/// one-line form is written when the art would not fit in it or `unicode` is
/// false (a terminal that cannot show braille).
pub fn write(w: *std.Io.Writer, color: bool, columns: ?u16, unicode: bool) std.Io.Writer.Error!void {
    const fits = if (columns) |c| c > width else true;
    try writeMarked(w, if (fits and unicode) art else compact, color);
}

/// Whether the terminal can be expected to show braille: a UTF-8 locale, or
/// none named (macOS and most terminal emulators decode UTF-8 regardless),
/// and not the Linux virtual console. Windows is decided by `enableUtf8`.
pub fn unicodeTerminal(env: *const std.process.Environ.Map) bool {
    if (builtin.os.tag == .windows) return enableUtf8();
    if (env.get("TERM")) |t| if (std.mem.eql(u8, t, "linux") or std.mem.eql(u8, t, "dumb")) return false;
    if (builtin.os.tag == .macos) return true;
    for ([_][]const u8{ "LC_ALL", "LC_CTYPE", "LANG" }) |name| {
        const v = env.get(name) orelse continue;
        if (v.len == 0) continue;
        return containsIgnoreCase(v, "utf-8") or containsIgnoreCase(v, "utf8");
    }
    return true;
}

fn containsIgnoreCase(haystack: []const u8, needle: []const u8) bool {
    if (needle.len > haystack.len) return false;
    for (0..haystack.len - needle.len + 1) |i| {
        if (std.ascii.eqlIgnoreCase(haystack[i..][0..needle.len], needle)) return true;
    }
    return false;
}

extern "kernel32" fn SetConsoleOutputCP(code_page: c_uint) callconv(.winapi) c_int;

/// Windows consoles show UTF-8 only once the output code page is 65001.
fn enableUtf8() bool {
    if (builtin.os.tag != .windows) return true;
    return SetConsoleOutputCP(65001) != 0;
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
    try write(&w, false, null, true);
    const plain = w.buffered();
    try std.testing.expect(std.mem.indexOfAny(u8, plain, "{}\x1b") == null);
    var rows = std.mem.splitScalar(u8, plain, '\n');
    while (rows.next()) |row| try std.testing.expect(try std.unicode.utf8CountCodepoints(row) <= width);
    try std.testing.expect(width <= 60);
}

test "a narrow or non-UTF-8 terminal gets the one-line wordmark" {
    var buf: [4096]u8 = undefined;
    var w: std.Io.Writer = .fixed(&buf);
    try write(&w, false, @intCast(width), true);
    try std.testing.expectEqualStrings("  ditch --.__.-'`\n", w.buffered());
    w = .fixed(&buf);
    try write(&w, false, @intCast(width + 1), false);
    try std.testing.expectEqualStrings("  ditch --.__.-'`\n", w.buffered());
    w = .fixed(&buf);
    try write(&w, false, @intCast(width + 1), true);
    try std.testing.expect(w.buffered().len > 200);
}

test "the coloured logo draws the same characters" {
    var a: [4096]u8 = undefined;
    var b: [8192]u8 = undefined;
    var plain: std.Io.Writer = .fixed(&a);
    var colored: std.Io.Writer = .fixed(&b);
    try write(&plain, false, null, true);
    try write(&colored, true, null, true);
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

//! The ditch wordmark: the word drawn in shaded characters with a thread woven
//! through it. `logo.txt` holds it with colour marks: `{W}` starts the letters,
//! `{R}` the thread and `{0}` resets. install.sh and install.ps1 carry copies.

const std = @import("std");

const art = @embedFile("logo.txt");

/// Light grey letters and a soft red thread, from the 256-colour palette so
/// that terminals without 24-bit colour show them too.
const letters = "\x1b[38;5;252m";
const thread = "\x1b[38;5;174m";
const reset = "\x1b[0m";

pub fn write(w: *std.Io.Writer, color: bool) std.Io.Writer.Error!void {
    var i: usize = 0;
    while (i < art.len) {
        if (art[i] == '{' and i + 2 < art.len and art[i + 2] == '}') {
            if (color) try w.writeAll(switch (art[i + 1]) {
                'W' => letters,
                'R' => thread,
                else => reset,
            });
            i += 3;
            continue;
        }
        const end = std.mem.indexOfScalarPos(u8, art, i + 1, '{') orelse art.len;
        try w.writeAll(art[i..end]);
        i = end;
    }
}

test "the plain logo has no colour marks and every row fits in 80 columns" {
    var buf: [4096]u8 = undefined;
    var w: std.Io.Writer = .fixed(&buf);
    try write(&w, false);
    const plain = w.buffered();
    try std.testing.expect(std.mem.indexOfAny(u8, plain, "{}\x1b") == null);
    var rows = std.mem.splitScalar(u8, plain, '\n');
    while (rows.next()) |row| try std.testing.expect(row.len <= 80);
}

test "the coloured logo draws the same characters" {
    var a: [4096]u8 = undefined;
    var b: [8192]u8 = undefined;
    var plain: std.Io.Writer = .fixed(&a);
    var colored: std.Io.Writer = .fixed(&b);
    try write(&plain, false);
    try write(&colored, true);
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

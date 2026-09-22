//! Static checks of `shaders.metal`, the best that can be done without Apple's
//! Metal compiler: the shader source is embedded in the binary, so these run
//! everywhere, including on the Linux machines where ditch is developed.
//!
//! What is checked here: that every kernel the Zig backend asks for by name is
//! defined in the shader source (directly or through one of the dtype macros),
//! that braces, parentheses and brackets balance, that no `#define` is left
//! open and that the tile and reduction widths the two sides assume agree.
//!
//! What is NOT checked here and needs a Mac: Metal Shading Language syntax and
//! semantics, buffer-index and attribute correctness, and threadgroup memory
//! limits. `xcrun -sdk macosx metal -c` in the macOS CI job compiles the file
//! for real, and `ditch selftest --device metal` runs every kernel against the
//! CPU reference.

const std = @import("std");

const shaders = @embedFile("shaders.metal");
const backend_src = @embedFile("backend.zig");

/// Kernel names the backend requests, taken from the Zig source so that a
/// rename on either side is caught.
fn collectRequested(gpa: std.mem.Allocator) !std.ArrayList([]const u8) {
    var out: std.ArrayList([]const u8) = .empty;
    errdefer out.deinit(gpa);
    // `self.pipeline("name")`
    var i: usize = 0;
    while (std.mem.indexOfPos(u8, backend_src, i, ".pipeline(\"")) |pos| {
        const start = pos + ".pipeline(\"".len;
        const end = std.mem.indexOfScalarPos(u8, backend_src, start, '"') orelse break;
        try out.append(gpa, backend_src[start..end]);
        i = end;
    }
    // `kernelName(&buf, "base", dtype)` expands to base_f32 / _f16 / _bf16.
    i = 0;
    while (std.mem.indexOfPos(u8, backend_src, i, "kernelName(&name_buf, \"")) |pos| {
        const start = pos + "kernelName(&name_buf, \"".len;
        const end = std.mem.indexOfScalarPos(u8, backend_src, start, '"') orelse break;
        const base = backend_src[start..end];
        for ([_][]const u8{ "f32", "f16", "bf16" }) |suffix| {
            try out.append(gpa, try std.fmt.allocPrint(gpa, "{s}_{s}", .{ base, suffix }));
        }
        i = end;
    }
    return out;
}

fn defined(name: []const u8) bool {
    var buf: [128]u8 = undefined;
    const direct = std.fmt.bufPrint(&buf, "kernel void {s}(", .{name}) catch return false;
    if (std.mem.indexOf(u8, shaders, direct) != null) return true;
    var buf2: [128]u8 = undefined;
    const via_macro = std.fmt.bufPrint(&buf2, "({s},", .{name}) catch return false;
    return std.mem.indexOf(u8, shaders, via_macro) != null;
}

test "every kernel the Metal backend asks for is defined in the shader source" {
    const gpa = std.testing.allocator;
    var requested = try collectRequested(gpa);
    defer {
        // Only the generated `base_dtype` names were allocated; the literals
        // point into the embedded source and must not be freed.
        for (requested.items) |n| {
            if (@intFromPtr(n.ptr) < @intFromPtr(backend_src.ptr) or
                @intFromPtr(n.ptr) >= @intFromPtr(backend_src.ptr) + backend_src.len) gpa.free(n);
        }
        requested.deinit(gpa);
    }
    try std.testing.expect(requested.items.len >= 16);
    for (requested.items) |name| {
        if (!defined(name)) {
            std.debug.print("shaders.metal defines no kernel named {s}\n", .{name});
            return error.MissingKernel;
        }
    }
}

test "the shader source is structurally well formed" {
    var braces: i64 = 0;
    var parens: i64 = 0;
    var brackets: i64 = 0;
    var line_start = true;
    var in_line_comment = false;
    var i: usize = 0;
    var defines: usize = 0;
    while (i < shaders.len) : (i += 1) {
        const c = shaders[i];
        if (c == '\n') {
            in_line_comment = false;
            line_start = true;
            continue;
        }
        if (in_line_comment) continue;
        if (c == '/' and i + 1 < shaders.len and shaders[i + 1] == '/') {
            in_line_comment = true;
            continue;
        }
        if (line_start and c == '#') {
            if (std.mem.startsWith(u8, shaders[i..], "#define")) defines += 1;
        }
        if (!std.ascii.isWhitespace(c)) line_start = false;
        switch (c) {
            '{' => braces += 1,
            '}' => braces -= 1,
            '(' => parens += 1,
            ')' => parens -= 1,
            '[' => brackets += 1,
            ']' => brackets -= 1,
            else => {},
        }
        // A closing delimiter never outnumbers its opening one.
        try std.testing.expect(braces >= 0 and parens >= 0 and brackets >= 0);
    }
    try std.testing.expectEqual(@as(i64, 0), braces);
    try std.testing.expectEqual(@as(i64, 0), parens);
    try std.testing.expectEqual(@as(i64, 0), brackets);
    try std.testing.expect(defines >= 5);
    try std.testing.expect(std.mem.indexOf(u8, shaders, "#include <metal_stdlib>") != null);
    try std.testing.expect(std.mem.indexOf(u8, shaders, "using namespace metal;") != null);
}

test "the tile and reduction widths agree between the shaders and the backend" {
    try std.testing.expect(std.mem.indexOf(u8, shaders, "#define TS 16") != null);
    try std.testing.expect(std.mem.indexOf(u8, shaders, "#define RED 256") != null);
    try std.testing.expect(std.mem.indexOf(u8, backend_src, "const tile = 16;") != null);
    try std.testing.expect(std.mem.indexOf(u8, backend_src, "const reduce_threads = 256;") != null);
}

test "every parameter block in the shaders has a Zig counterpart" {
    // `struct XParams {` in the shader, `const XParams = extern struct` in Zig.
    var i: usize = 0;
    var found: usize = 0;
    while (std.mem.indexOfPos(u8, shaders, i, "struct ")) |pos| {
        const start = pos + "struct ".len;
        const end = std.mem.indexOfScalarPos(u8, shaders, start, ' ') orelse break;
        const name = shaders[start..end];
        i = end;
        if (!std.mem.endsWith(u8, name, "Params")) continue;
        found += 1;
        var buf: [128]u8 = undefined;
        const decl = try std.fmt.bufPrint(&buf, "const {s} = extern struct", .{name});
        if (std.mem.indexOf(u8, backend_src, decl) == null) {
            std.debug.print("backend.zig declares no counterpart for {s}\n", .{name});
            return error.MissingParams;
        }
    }
    try std.testing.expect(found >= 9);
}

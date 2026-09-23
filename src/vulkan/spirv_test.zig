//! Checks of the Vulkan shaders that run everywhere, without a GPU or a
//! shader compiler:
//!
//! * the committed SPIR-V is current: `spirv/SHA256SUMS` (written by
//!   `tools/gen_spirv.sh`) records the hash of every GLSL source and every
//!   module, and both must match what is embedded in the binary, so a shader
//!   edited without regenerating fails `zig build test`;
//! * every kernel the backend can ask for has a module, and each is SPIR-V;
//! * each shader's `layout(push_constant)` block has exactly the fields of
//!   the Zig parameter struct the backend pushes, in order, with the same
//!   scalar types;
//! * each shader's `local_size_x` is the workgroup width the backend assumes
//!   when it sizes the dispatch.
//!
//! Compiling the GLSL and validating the modules is `tools/gen_spirv.sh
//! --check` (CI); running them against the CPU reference is `ditch selftest
//! --device vulkan` (CI, on Mesa's lavapipe) and `backend_test.zig` wherever
//! a Vulkan device exists.

const std = @import("std");
const backend = @import("backend.zig");
const Kernel = backend.Kernel;

const sums = @embedFile("spirv/SHA256SUMS");

/// Every GLSL file `tools/gen_spirv.sh` hashes.
const sources = .{
    .{ "attn_scores.comp", @embedFile("shaders/attn_scores.comp") },
    .{ "attn_values.comp", @embedFile("shaders/attn_values.comp") },
    .{ "gated.comp", @embedFile("shaders/gated.comp") },
    .{ "layernorm_rows.comp", @embedFile("shaders/layernorm_rows.comp") },
    .{ "matmul.comp", @embedFile("shaders/matmul.comp") },
    .{ "matvec.comp", @embedFile("shaders/matvec.comp") },
    .{ "rmsnorm_rows.comp", @embedFile("shaders/rmsnorm_rows.comp") },
    .{ "rope_rows.comp", @embedFile("shaders/rope_rows.comp") },
    .{ "row_norms.comp", @embedFile("shaders/row_norms.comp") },
    .{ "softmax_rows.comp", @embedFile("shaders/softmax_rows.comp") },
    .{ "weights.glsl", @embedFile("shaders/weights.glsl") },
};

fn source(name: []const u8) ?[]const u8 {
    inline for (sources) |s| {
        if (std.mem.eql(u8, s[0], name)) return s[1];
    }
    return null;
}

fn hex(data: []const u8) [64]u8 {
    var digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(data, &digest, .{});
    return std.fmt.bytesToHex(digest, .lower);
}

/// The hash SHA256SUMS records for `path`.
fn recorded(path: []const u8) ?[]const u8 {
    var lines = std.mem.tokenizeScalar(u8, sums, '\n');
    while (lines.next()) |line| {
        if (line.len < 66) continue;
        if (std.mem.eql(u8, line[66..], path)) return line[0..64];
    }
    return null;
}

test "the committed SPIR-V was generated from the current GLSL sources" {
    var buf: [128]u8 = undefined;
    // Every source is recorded with its current hash...
    inline for (sources) |s| {
        const path = try std.fmt.bufPrint(&buf, "src/vulkan/shaders/{s}", .{s[0]});
        const want = recorded(path) orelse {
            std.debug.print("{s} is not in src/vulkan/spirv/SHA256SUMS: run bash tools/gen_spirv.sh\n", .{path});
            return error.TestUnexpectedResult;
        };
        const got = hex(s[1]);
        if (!std.mem.eql(u8, want, &got)) {
            std.debug.print("{s} changed since the SPIR-V was generated: run bash tools/gen_spirv.sh\n", .{path});
            return error.TestUnexpectedResult;
        }
    }
    // ...and so is every module the backend embeds.
    inline for (@typeInfo(Kernel).@"enum".fields) |f| {
        const k: Kernel = @enumFromInt(f.value);
        const path = try std.fmt.bufPrint(&buf, "src/vulkan/spirv/{s}.spv", .{f.name});
        const want = recorded(path) orelse {
            std.debug.print("{s} is not in SHA256SUMS: run bash tools/gen_spirv.sh\n", .{path});
            return error.TestUnexpectedResult;
        };
        const got = hex(k.spirv());
        try std.testing.expectEqualStrings(want, &got);
    }
    // Nothing is recorded that the backend does not embed.
    var lines = std.mem.tokenizeScalar(u8, sums, '\n');
    var n: usize = 0;
    while (lines.next()) |_| n += 1;
    try std.testing.expectEqual(@as(usize, sources.len + @typeInfo(Kernel).@"enum".fields.len), n);
}

test "every kernel has a SPIR-V module and a GLSL source" {
    inline for (@typeInfo(Kernel).@"enum".fields) |f| {
        const k: Kernel = @enumFromInt(f.value);
        const code = k.spirv();
        try std.testing.expect(code.len > 20 and code.len % 4 == 0);
        try std.testing.expectEqual(@as(u32, 0x07230203), std.mem.readInt(u32, code[0..4], .little));
        var buf: [64]u8 = undefined;
        const name = try std.fmt.bufPrint(&buf, "{s}.comp", .{backend.sourceName(k)});
        try std.testing.expect(source(name) != null);
    }
}

/// The `(type, name)` pairs of the push constant block of `src`.
fn pushFields(gpa: std.mem.Allocator, src: []const u8) !std.ArrayList([2][]const u8) {
    var out: std.ArrayList([2][]const u8) = .empty;
    errdefer out.deinit(gpa);
    const start = std.mem.indexOf(u8, src, "layout(push_constant) uniform P {") orelse return error.NoPushConstants;
    const open = start + "layout(push_constant) uniform P {".len;
    const close = std.mem.indexOfScalarPos(u8, src, open, '}') orelse return error.NoPushConstants;
    var decls = std.mem.tokenizeScalar(u8, src[open..close], ';');
    while (decls.next()) |d| {
        var words = std.mem.tokenizeAny(u8, d, " \t\n");
        const ty = words.next() orelse continue;
        const name = words.next() orelse return error.BadDeclaration;
        try out.append(gpa, .{ ty, name });
    }
    return out;
}

test "each shader's push constants are the backend's parameter struct" {
    const gpa = std.testing.allocator;
    inline for (@typeInfo(Kernel).@"enum".fields) |f| {
        const k: Kernel = @enumFromInt(f.value);
        var buf: [64]u8 = undefined;
        const src = source(try std.fmt.bufPrint(&buf, "{s}.comp", .{backend.sourceName(k)})).?;
        var fields = try pushFields(gpa, src);
        defer fields.deinit(gpa);
        const P = backend.Params(k);
        const zig_fields = @typeInfo(P).@"struct".fields;
        if (fields.items.len != zig_fields.len) {
            std.debug.print("{s}: shader pushes {d} fields, {s} has {d}\n", .{ f.name, fields.items.len, @typeName(P), zig_fields.len });
            return error.TestUnexpectedResult;
        }
        inline for (zig_fields, 0..) |zf, i| {
            const glsl_type = switch (zf.type) {
                u32 => "uint",
                f32 => "float",
                else => @compileError("unexpected parameter type"),
            };
            try std.testing.expectEqualStrings(glsl_type, fields.items[i][0]);
            try std.testing.expectEqualStrings(zf.name, fields.items[i][1]);
        }
    }
}

/// `local_size_x` of `src`, resolving one `#define` level.
fn localSizeX(src: []const u8) !u32 {
    const key = "local_size_x = ";
    const pos = std.mem.indexOf(u8, src, key) orelse return error.NoLocalSize;
    const start = pos + key.len;
    var end = start;
    while (end < src.len and (std.ascii.isAlphanumeric(src[end]) or src[end] == '_')) end += 1;
    const tok = src[start..end];
    if (std.fmt.parseInt(u32, tok, 10)) |v| return v else |_| {}
    var buf: [64]u8 = undefined;
    const def = try std.fmt.bufPrint(&buf, "#define {s} ", .{tok});
    const dpos = std.mem.indexOf(u8, src, def) orelse return error.NoLocalSize;
    const vstart = dpos + def.len;
    var vend = vstart;
    while (vend < src.len and std.ascii.isDigit(src[vend])) vend += 1;
    return std.fmt.parseInt(u32, src[vstart..vend], 10);
}

test "each shader's workgroup width is the one the backend dispatches for" {
    inline for (@typeInfo(Kernel).@"enum".fields) |f| {
        const k: Kernel = @enumFromInt(f.value);
        var buf: [64]u8 = undefined;
        const src = source(try std.fmt.bufPrint(&buf, "{s}.comp", .{backend.sourceName(k)})).?;
        try std.testing.expectEqual(backend.workgroupWidth(k), try localSizeX(src));
    }
}

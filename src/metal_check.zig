//! Root of `zig build metal-check`: compiles the Zig half of the Metal backend
//! for an Apple silicon target on any host. The Objective-C shim is only
//! declared there, never linked here, so this needs no Apple SDK — it type
//! checks the backend, its extern signatures and its use of the compute seam.
//! The shaders themselves are checked by `src/metal/shaders_test.zig` and, for
//! real, by `xcrun -sdk macosx metal -c` in the macOS CI job.
//!
//! Naming a declaration is not enough: Zig resolves a struct's field types
//! only when its layout is needed and analyses a function body only when the
//! function is called or its address is taken. So every struct is laid out
//! and every function's address is taken, which is what a real `-Dmetal`
//! build would do.

const backend = @import("metal/backend.zig");

comptime {
    for (@typeInfo(backend).@"struct".decls) |d| {
        const v = @field(backend, d.name);
        const T = @TypeOf(v);
        if (T == type) {
            switch (@typeInfo(v)) {
                .@"struct", .@"union", .@"enum" => _ = @sizeOf(v),
                else => {},
            }
        } else if (@typeInfo(T) == .@"fn") {
            _ = &@field(backend, d.name);
        }
    }
}

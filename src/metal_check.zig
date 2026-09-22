//! Root of `zig build metal-check`: compiles the Zig half of the Metal backend
//! for an Apple silicon target on any host. The Objective-C shim is only
//! declared there, never linked here, so this needs no Apple SDK — it type
//! checks the backend, its extern signatures and its use of the compute seam.
//! The shaders themselves are checked by `src/metal/shaders_test.zig` and, for
//! real, by `xcrun -sdk macosx metal -c` in the macOS CI job.

const backend = @import("metal/backend.zig");

comptime {
    for (@typeInfo(backend).@"struct".decls) |d| _ = @field(backend, d.name);
}

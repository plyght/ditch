//! Aggregates unit tests from every module.
comptime {
    _ = @import("tensor.zig");
    _ = @import("safetensors.zig");
    _ = @import("tokenizer.zig");
    _ = @import("model.zig");
    _ = @import("model_test.zig");
}

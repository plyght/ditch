//! Aggregates unit tests from every module.
comptime {
    _ = @import("tensor.zig");
    _ = @import("quant.zig");
    _ = @import("gguf.zig");
    _ = @import("gguf_model.zig");
    _ = @import("gguf_export.zig");
    _ = @import("gguf_test.zig");
    _ = @import("safetensors.zig");
    _ = @import("tokenizer.zig");
    _ = @import("model.zig");
    _ = @import("model_test.zig");
    _ = @import("moe.zig");
    _ = @import("abliterate.zig");
    _ = @import("directions.zig");
    _ = @import("tpe.zig");
    _ = @import("toml.zig");
    _ = @import("config.zig");
    _ = @import("chat.zig");
    _ = @import("hf.zig");
    _ = @import("engine.zig");
    _ = @import("scorers.zig");
    _ = @import("study.zig");
    _ = @import("export.zig");
    _ = @import("search.zig");
    _ = @import("lua.zig");
    _ = @import("budget.zig");
    _ = @import("stream.zig");
    _ = @import("stream_test.zig");
    _ = @import("reproduce.zig");
    _ = @import("bench.zig");
}

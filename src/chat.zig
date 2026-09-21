//! Chat prompt formatting. Instead of a Jinja engine, ditch ships the handful
//! of template families used by supported models and detects which one a
//! model's `chat_template` corresponds to.

const std = @import("std");
const Allocator = std.mem.Allocator;

pub const Template = enum {
    chatml,
    llama3,
    llama2,
    mistral,
    gemma,
    raw,

    pub fn parse(name: []const u8) ?Template {
        inline for (@typeInfo(Template).@"enum".fields) |f| {
            if (std.mem.eql(u8, name, f.name)) return @enumFromInt(f.value);
        }
        return null;
    }
};

pub const Role = enum { system, user, assistant };

pub const Message = struct {
    role: Role,
    content: []const u8,
};

/// Detects the template family from a model's Jinja chat template and model type.
pub fn detect(chat_template: ?[]const u8, model_type: []const u8) Template {
    if (chat_template) |t| {
        if (std.mem.indexOf(u8, t, "<|im_start|>") != null) return .chatml;
        if (std.mem.indexOf(u8, t, "<|start_header_id|>") != null) return .llama3;
        if (std.mem.indexOf(u8, t, "<start_of_turn>") != null) return .gemma;
        if (std.mem.indexOf(u8, t, "<<SYS>>") != null) return .llama2;
        if (std.mem.indexOf(u8, t, "[INST]") != null) return .mistral;
    }
    if (std.mem.startsWith(u8, model_type, "qwen")) return .chatml;
    if (std.mem.startsWith(u8, model_type, "llama")) return .llama3;
    if (std.mem.startsWith(u8, model_type, "gemma")) return .gemma;
    if (std.mem.startsWith(u8, model_type, "mistral")) return .mistral;
    return .raw;
}

fn trim(s: []const u8) []const u8 {
    return std.mem.trim(u8, s, " \t\r\n");
}

/// Renders a conversation with the generation prompt appended (`add_generation_prompt=True`).
pub fn render(gpa: Allocator, template: Template, messages: []const Message) ![]u8 {
    var out: std.Io.Writer.Allocating = .init(gpa);
    errdefer out.deinit();
    const w = &out.writer;
    switch (template) {
        .chatml => {
            for (messages) |m| try w.print("<|im_start|>{s}\n{s}<|im_end|>\n", .{ @tagName(m.role), m.content });
            try w.writeAll("<|im_start|>assistant\n");
        },
        .llama3 => {
            try w.writeAll("<|begin_of_text|>");
            for (messages) |m| try w.print("<|start_header_id|>{s}<|end_header_id|>\n\n{s}<|eot_id|>", .{ @tagName(m.role), trim(m.content) });
            try w.writeAll("<|start_header_id|>assistant<|end_header_id|>\n\n");
        },
        .gemma => {
            // Gemma has no system role: the system prompt is prepended to the first user turn.
            try w.writeAll("<bos>");
            var system: ?[]const u8 = null;
            var first_user = true;
            for (messages) |m| {
                switch (m.role) {
                    .system => system = m.content,
                    .user => {
                        try w.writeAll("<start_of_turn>user\n");
                        if (first_user) {
                            if (system) |s| try w.print("{s}\n\n", .{trim(s)});
                            first_user = false;
                        }
                        try w.print("{s}<end_of_turn>\n", .{trim(m.content)});
                    },
                    .assistant => try w.print("<start_of_turn>model\n{s}<end_of_turn>\n", .{trim(m.content)}),
                }
            }
            try w.writeAll("<start_of_turn>model\n");
        },
        .llama2 => {
            try w.writeAll("<s>");
            var system: ?[]const u8 = null;
            var first_user = true;
            for (messages) |m| {
                switch (m.role) {
                    .system => system = m.content,
                    .user => {
                        try w.writeAll("[INST] ");
                        if (first_user) {
                            if (system) |s| try w.print("<<SYS>>\n{s}\n<</SYS>>\n\n", .{trim(s)});
                            first_user = false;
                        }
                        try w.print("{s} [/INST]", .{trim(m.content)});
                    },
                    .assistant => try w.print(" {s} </s><s>", .{trim(m.content)}),
                }
            }
        },
        .mistral => {
            try w.writeAll("<s>");
            var system: ?[]const u8 = null;
            var first_user = true;
            for (messages) |m| {
                switch (m.role) {
                    .system => system = m.content,
                    .user => {
                        try w.writeAll("[INST] ");
                        if (first_user) {
                            if (system) |s| try w.print("{s}\n\n", .{trim(s)});
                            first_user = false;
                        }
                        try w.print("{s}[/INST]", .{trim(m.content)});
                    },
                    .assistant => try w.print(" {s}</s>", .{trim(m.content)}),
                }
            }
        },
        .raw => {
            for (messages) |m| {
                switch (m.role) {
                    .system => try w.print("{s}\n\n", .{trim(m.content)}),
                    .user => try w.print("User: {s}\n", .{trim(m.content)}),
                    .assistant => try w.print("Assistant: {s}\n", .{trim(m.content)}),
                }
            }
            try w.writeAll("Assistant:");
        },
    }
    return out.toOwnedSlice();
}

/// Convenience for the common system + user case.
pub fn renderPrompt(gpa: Allocator, template: Template, system: []const u8, user: []const u8) ![]u8 {
    const msgs = [_]Message{ .{ .role = .system, .content = system }, .{ .role = .user, .content = user } };
    return render(gpa, template, &msgs);
}

test "template detection and rendering" {
    const gpa = std.testing.allocator;
    try std.testing.expectEqual(Template.chatml, detect("{% ... <|im_start|> ... %}", "qwen2"));
    try std.testing.expectEqual(Template.llama3, detect(null, "llama"));
    try std.testing.expectEqual(Template.gemma, detect("<start_of_turn>", "gemma3_text"));
    const p = try renderPrompt(gpa, .chatml, "Sys.", "Hi");
    defer gpa.free(p);
    try std.testing.expectEqualStrings("<|im_start|>system\nSys.<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n", p);
    const g = try renderPrompt(gpa, .gemma, "Sys.", "Hi");
    defer gpa.free(g);
    try std.testing.expectEqualStrings("<bos><start_of_turn>user\nSys.\n\nHi<end_of_turn>\n<start_of_turn>model\n", g);
}

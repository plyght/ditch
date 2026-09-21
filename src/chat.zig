//! Chat prompt formatting. Instead of a Jinja engine, ditch ships the handful
//! of template families used by supported models and detects which one a
//! model's `chat_template` corresponds to.

const std = @import("std");
const arch = @import("arch.zig");
const Allocator = std.mem.Allocator;

pub const Template = enum {
    chatml,
    llama3,
    llama2,
    mistral,
    gemma,
    /// Phi-3 / Phi-4: `<|user|>\n...<|end|>\n<|assistant|>\n`.
    phi3,
    /// Zephyr / StableLM-chat: `<|user|>\n...<|endoftext|>\n<|assistant|>\n`.
    zephyr,
    /// OLMo / Tülu: `<|user|>\n...\n<|assistant|>\n`.
    olmo,
    /// GLM-4 / ChatGLM3: `[gMASK]<sop><|user|>\n...<|assistant|>\n`.
    glm4,
    /// Command R: `<|START_OF_TURN_TOKEN|><|USER_TOKEN|>...<|END_OF_TURN_TOKEN|>`.
    cohere,
    /// DeepSeek V2/V3: `<｜User｜>...<｜Assistant｜>`.
    deepseek,
    /// gpt-oss harmony: `<|start|>user<|message|>...<|end|><|start|>assistant`.
    harmony,
    /// Llama 4: `<|header_start|>user<|header_end|>\n\n...<|eot|>`.
    llama4,
    /// EXAONE: `[|user|]...[|endofturn|]\n[|assistant|]`.
    exaone,
    /// Granite 3: `<|start_of_role|>user<|end_of_role|>...<|end_of_text|>`.
    granite,
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

/// Detects the template family from a model's Jinja chat template (by its
/// role markers) and otherwise from the architecture registry's hint for the
/// model type.
pub fn detect(chat_template: ?[]const u8, model_type: []const u8) Template {
    if (chat_template) |t| {
        const has = struct {
            fn f(hay: []const u8, needle: []const u8) bool {
                return std.mem.indexOf(u8, hay, needle) != null;
            }
        }.f;
        if (has(t, "<|im_start|>")) return .chatml;
        if (has(t, "<|start_header_id|>")) return .llama3;
        if (has(t, "<|header_start|>")) return .llama4;
        if (has(t, "<start_of_turn>")) return .gemma;
        if (has(t, "<|START_OF_TURN_TOKEN|>")) return .cohere;
        if (has(t, "<|start|>") and has(t, "<|message|>")) return .harmony;
        if (has(t, "<|start_of_role|>")) return .granite;
        if (has(t, "[|user|]")) return .exaone;
        if (has(t, "<\xef\xbd\x9cUser\xef\xbd\x9c>")) return .deepseek;
        if (has(t, "[gMASK]")) return .glm4;
        if (has(t, "<|user|>") and has(t, "<|end|>")) return .phi3;
        if (has(t, "<|user|>") and has(t, "<|endoftext|>")) return .zephyr;
        if (has(t, "<|user|>")) return .olmo;
        if (has(t, "<<SYS>>")) return .llama2;
        if (has(t, "[INST]")) return .mistral;
    }
    if (arch.lookup(model_type)) |a| {
        if (Template.parse(a.chat)) |tpl| return tpl;
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
        .phi3, .zephyr, .olmo => {
            // `<|role|>` headers; the turn terminator differs per family.
            const end: []const u8 = switch (template) {
                .phi3 => "<|end|>\n",
                .zephyr => "<|endoftext|>\n",
                else => "\n",
            };
            for (messages) |m| try w.print("<|{s}|>\n{s}{s}", .{ @tagName(m.role), trim(m.content), end });
            try w.writeAll("<|assistant|>\n");
        },
        .glm4 => {
            try w.writeAll("[gMASK]<sop>");
            for (messages) |m| try w.print("<|{s}|>\n{s}", .{ @tagName(m.role), trim(m.content) });
            try w.writeAll("<|assistant|>\n");
        },
        .cohere => {
            try w.writeAll("<BOS_TOKEN>");
            for (messages) |m| {
                const role: []const u8 = switch (m.role) {
                    .system => "<|SYSTEM_TOKEN|>",
                    .user => "<|USER_TOKEN|>",
                    .assistant => "<|CHATBOT_TOKEN|>",
                };
                try w.print("<|START_OF_TURN_TOKEN|>{s}{s}<|END_OF_TURN_TOKEN|>", .{ role, trim(m.content) });
            }
            try w.writeAll("<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>");
        },
        .deepseek => {
            try w.writeAll("<\xef\xbd\x9cbegin\xe2\x96\x81of\xe2\x96\x81sentence\xef\xbd\x9c>");
            for (messages) |m| {
                switch (m.role) {
                    .system => try w.print("{s}", .{trim(m.content)}),
                    .user => try w.print("<\xef\xbd\x9cUser\xef\xbd\x9c>{s}", .{trim(m.content)}),
                    .assistant => try w.print("<\xef\xbd\x9cAssistant\xef\xbd\x9c>{s}<\xef\xbd\x9cend\xe2\x96\x81of\xe2\x96\x81sentence\xef\xbd\x9c>", .{trim(m.content)}),
                }
            }
            try w.writeAll("<\xef\xbd\x9cAssistant\xef\xbd\x9c>");
        },
        .harmony => {
            for (messages) |m| try w.print("<|start|>{s}<|message|>{s}<|end|>", .{ @tagName(m.role), trim(m.content) });
            try w.writeAll("<|start|>assistant");
        },
        .llama4 => {
            try w.writeAll("<|begin_of_text|>");
            for (messages) |m| try w.print("<|header_start|>{s}<|header_end|>\n\n{s}<|eot|>", .{ @tagName(m.role), trim(m.content) });
            try w.writeAll("<|header_start|>assistant<|header_end|>\n\n");
        },
        .exaone => {
            for (messages) |m| {
                switch (m.role) {
                    .system => try w.print("[|system|]{s}[|endofturn|]\n", .{trim(m.content)}),
                    .user => try w.print("[|user|]{s}\n", .{trim(m.content)}),
                    .assistant => try w.print("[|assistant|]{s}[|endofturn|]\n", .{trim(m.content)}),
                }
            }
            try w.writeAll("[|assistant|]");
        },
        .granite => {
            for (messages) |m| try w.print("<|start_of_role|>{s}<|end_of_role|>{s}<|end_of_text|>\n", .{ @tagName(m.role), trim(m.content) });
            try w.writeAll("<|start_of_role|>assistant<|end_of_role|>");
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
    try std.testing.expectEqual(Template.phi3, detect(null, "phi3"));
    try std.testing.expectEqual(Template.harmony, detect("{{ '<|start|>' + role + '<|message|>' }}", "gpt_oss"));
    try std.testing.expectEqual(Template.deepseek, detect(null, "deepseek_v3"));
    try std.testing.expectEqual(Template.raw, detect(null, "gpt2"));
    const ph = try renderPrompt(gpa, .phi3, "Sys.", "Hi");
    defer gpa.free(ph);
    try std.testing.expectEqualStrings("<|system|>\nSys.<|end|>\n<|user|>\nHi<|end|>\n<|assistant|>\n", ph);
    const p = try renderPrompt(gpa, .chatml, "Sys.", "Hi");
    defer gpa.free(p);
    try std.testing.expectEqualStrings("<|im_start|>system\nSys.<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n", p);
    const g = try renderPrompt(gpa, .gemma, "Sys.", "Hi");
    defer gpa.free(g);
    try std.testing.expectEqualStrings("<bos><start_of_turn>user\nSys.\n\nHi<end_of_turn>\n<start_of_turn>model\n", g);
}

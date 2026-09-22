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
    /// Mistral's V7 (tekken) template — Ministral 3, Mistral Small 3.x,
    /// Magistral, Devstral: `<s>[SYSTEM_PROMPT]...[/SYSTEM_PROMPT][INST]...[/INST]`.
    mistral_v7,
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
    /// Command R7B (cohere2): Command R's turns, but the chatbot's opens with
    /// `<|START_RESPONSE|>` (closed by `<|END_RESPONSE|>`), and an empty
    /// system turn stands in when the conversation has none.
    cohere_response,
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
    /// Kimi K2 / K2.5: `<|im_user|>user<|im_middle|>...<|im_end|>` (a default
    /// system message is inserted when the conversation has none).
    kimi,
    /// Kimi K3 (XTML): `<|open|>message role="user"<|sep|>...<|close|>message<|sep|><|end_of_msg|>`.
    kimi_k3,
    /// Jamba: `<|startoftext|><|bom|><|user|> ...<|eom|><|bom|><|assistant|>`.
    jamba,
    /// ERNIE 4.5: `<|begin_of_sentence|>{system}\nUser: ...\nAssistant: `.
    ernie,
    /// Hunyuan (V1 dense): `<｜hy_begin▁of▁sentence｜>{system}<｜hy_place▁holder▁no▁3｜><｜hy_User｜>...<｜hy_Assistant｜>`.
    hunyuan,
    /// nanochat: `<|user_start|>...<|user_end|><|assistant_start|>`; a system
    /// message is prepended to the first user turn, followed by a blank line.
    nanochat,
    /// Laguna (Poolside): `〈|EOS|〉<system>\n\n...\n</system>\n<user>\n...\n</user>\n<assistant>\n</think>`
    /// (a default system message is inserted when the conversation has none;
    /// `</think>` is the non-thinking generation prompt).
    laguna,
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
        if (has(t, "<|bom|>")) return .jamba;
        if (has(t, "<\u{ff5c}hy_User\u{ff5c}>")) return .hunyuan;
        if (has(t, "<|begin_of_sentence|>") and has(t, "Assistant: ")) return .ernie;
        if (has(t, "<|user_start|>")) return .nanochat;
        if (has(t, "</user>") and has(t, "<assistant>")) return .laguna;
        if (has(t, "<|im_middle|>")) return .kimi;
        if (has(t, "<|end_of_msg|>")) return .kimi_k3;
        if (has(t, "<|im_start|>")) return .chatml;
        if (has(t, "<|start_header_id|>")) return .llama3;
        if (has(t, "<|header_start|>")) return .llama4;
        if (has(t, "<start_of_turn>")) return .gemma;
        if (has(t, "<|START_OF_TURN_TOKEN|>")) return if (has(t, "<|START_RESPONSE|>")) .cohere_response else .cohere;
        if (has(t, "<|start|>") and has(t, "<|message|>")) return .harmony;
        if (has(t, "<|start_of_role|>")) return .granite;
        if (has(t, "[|user|]")) return .exaone;
        if (has(t, "<\xef\xbd\x9cUser\xef\xbd\x9c>")) return .deepseek;
        if (has(t, "[gMASK]")) return .glm4;
        if (has(t, "<|user|>") and has(t, "<|end|>")) return .phi3;
        if (has(t, "<|user|>") and has(t, "<|endoftext|>")) return .zephyr;
        if (has(t, "<|user|>")) return .olmo;
        if (has(t, "<<SYS>>")) return .llama2;
        if (has(t, "[INST]")) return if (has(t, "[SYSTEM_PROMPT]")) .mistral_v7 else .mistral;
    }
    if (arch.lookup(model_type)) |a| {
        if (Template.parse(a.chat)) |tpl| return tpl;
    }
    if (std.mem.startsWith(u8, model_type, "qwen")) return .chatml;
    if (std.mem.eql(u8, model_type, "kimi_k3")) return .kimi_k3;
    if (std.mem.startsWith(u8, model_type, "kimi")) return .kimi;
    if (std.mem.startsWith(u8, model_type, "llama")) return .llama3;
    if (std.mem.startsWith(u8, model_type, "gemma")) return .gemma;
    if (std.mem.startsWith(u8, model_type, "mistral")) return .mistral;
    return .raw;
}

/// The BOS text a model's own Jinja template emits before the first turn, or
/// "" when it emits none. ditch's fixed families hardcode a BOS where the
/// family always has one (llama3, gemma, llama2, mistral, cohere), but a
/// checkpoint can add one to a family that usually has none — Falcon-H1 puts
/// `{{bos_token}}` in front of a ChatML body — and its tokenizer does not add
/// one either (no `add_bos_token`), so without this the prompt loses its BOS.
pub fn templateBos(chat_template: ?[]const u8, bos_token: ?[]const u8) []const u8 {
    const bos = bos_token orelse return "";
    if (bos.len == 0) return "";
    const t = chat_template orelse return "";
    // Only what precedes the first control block counts as "before the first turn".
    const head = t[0 .. std.mem.indexOf(u8, t, "{%") orelse t.len];
    if (std.mem.indexOf(u8, head, "bos_token") != null) return bos;
    if (std.mem.indexOf(u8, head, bos) != null) return bos;
    return "";
}

const laguna_default_system = "You are a helpful, conversationally-fluent assistant made by Poolside. You are here to be helpful to users through natural language conversations.";

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
        .mistral_v7 => {
            // Contents verbatim; nothing follows the last `[/INST]`. Without a
            // system message the template inserts a long, model-specific
            // default (dated), which ditch, always passing one, leaves out.
            try w.writeAll("<s>");
            for (messages) |m| {
                switch (m.role) {
                    .system => try w.print("[SYSTEM_PROMPT]{s}[/SYSTEM_PROMPT]", .{m.content}),
                    .user => try w.print("[INST]{s}[/INST]", .{m.content}),
                    .assistant => try w.print("{s}</s>", .{m.content}),
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
        .cohere_response => {
            // CohereLabs/c4ai-command-r7b-12-2024's plain-chat branch (no tools or
            // documents): the system message is not trimmed, the turns are.
            try w.writeAll("<BOS_TOKEN>");
            if (messages.len == 0 or messages[0].role != .system) try w.writeAll("<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|><|END_OF_TURN_TOKEN|>");
            for (messages) |m| {
                switch (m.role) {
                    .system => try w.print("<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>{s}<|END_OF_TURN_TOKEN|>", .{m.content}),
                    .user => try w.print("<|START_OF_TURN_TOKEN|><|USER_TOKEN|>{s}<|END_OF_TURN_TOKEN|>", .{trim(m.content)}),
                    .assistant => try w.print("<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|><|START_RESPONSE|>{s}<|END_RESPONSE|><|END_OF_TURN_TOKEN|>", .{trim(m.content)}),
                }
            }
            try w.writeAll("<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|><|START_RESPONSE|>");
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
        .jamba => {
            // One `<|eom|>` *between* messages (never after the last), and the
            // role header carries a trailing space except on the generation turn.
            try w.writeAll("<|startoftext|>");
            for (messages, 0..) |m, i| {
                if (i > 0) try w.writeAll("<|eom|>");
                try w.print("<|bom|><|{s}|> {s}", .{ @tagName(m.role), m.content });
            }
            if (messages.len > 0) try w.writeAll("<|eom|>");
            try w.writeAll("<|bom|><|assistant|>");
        },
        .laguna => {
            try w.writeAll("\xe3\x80\x88|EOS|\xe3\x80\x89");
            const has_system = messages.len > 0 and messages[0].role == .system;
            const system = if (has_system) messages[0].content else laguna_default_system;
            if (trim(system).len > 0) try w.print("<system>\n\n{s}\n</system>\n", .{std.mem.trimEnd(u8, system, " \t\r\n")});
            for (messages, 0..) |m, i| {
                switch (m.role) {
                    .system => if (i != 0) try w.print("<system>\n{s}\n</system>\n", .{m.content}),
                    .user => try w.print("<user>\n{s}\n</user>\n", .{m.content}),
                    .assistant => {
                        try w.writeAll("<assistant>\n</think>\n");
                        if (trim(m.content).len > 0) try w.print("{s}\n", .{trim(m.content)});
                        try w.writeAll("</assistant>\n");
                    },
                }
            }
            try w.writeAll("<assistant>\n</think>");
        },
        .nanochat => {
            // Contents verbatim; the template's `bos_token` comes from `templateBos`.
            var system: ?[]const u8 = null;
            for (messages, 0..) |m, i| {
                switch (m.role) {
                    .system => if (i == 0) {
                        system = m.content;
                    },
                    .user => {
                        try w.writeAll("<|user_start|>");
                        if (system) |sys| try w.print("{s}\n\n", .{sys});
                        system = null;
                        try w.print("{s}<|user_end|>", .{m.content});
                    },
                    .assistant => try w.print("<|assistant_start|>{s}<|assistant_end|>", .{m.content}),
                }
            }
            try w.writeAll("<|assistant_start|>");
        },
        .ernie => {
            // baidu/ERNIE-4.5-*-PT: contents verbatim, no trimming.
            try w.writeAll("<|begin_of_sentence|>");
            for (messages) |m| {
                switch (m.role) {
                    .system => try w.print("{s}\n", .{m.content}),
                    .user => try w.print("User: {s}\n", .{m.content}),
                    .assistant => try w.print("Assistant: {s}<|end_of_sentence|>", .{m.content}),
                }
            }
            try w.writeAll("Assistant: ");
        },
        .hunyuan => {
            // tencent/Hunyuan-*-Instruct: every system message joined by a
            // blank line in front of the first turn; the user turn carries the
            // assistant header, and a finished assistant turn ends with the EOS.
            try w.writeAll("<\u{ff5c}hy_begin\u{2581}of\u{2581}sentence\u{ff5c}>");
            var any_system = false;
            for (messages) |m| if (m.role == .system) {
                if (any_system) try w.writeAll("\n\n");
                try w.writeAll(m.content);
                any_system = true;
            };
            if (any_system) try w.writeAll("<\u{ff5c}hy_place\u{2581}holder\u{2581}no\u{2581}3\u{ff5c}>");
            var last_user = false;
            for (messages) |m| {
                switch (m.role) {
                    .system => {},
                    .user => {
                        try w.print("<\u{ff5c}hy_User\u{ff5c}>{s}<\u{ff5c}hy_Assistant\u{ff5c}>", .{m.content});
                        last_user = true;
                    },
                    .assistant => {
                        try w.print("{s}<\u{ff5c}hy_place\u{2581}holder\u{2581}no\u{2581}2\u{ff5c}>", .{m.content});
                        last_user = false;
                    },
                }
            }
            if (!last_user) try w.writeAll("<\u{ff5c}hy_Assistant\u{ff5c}>");
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
        .kimi => {
            // moonshotai/Kimi-K2-Instruct: content is not trimmed; a missing system message becomes the default one.
            if (messages.len == 0 or messages[0].role != .system) try w.writeAll("<|im_system|>system<|im_middle|>You are a helpful assistant<|im_end|>");
            for (messages) |m| {
                const header: []const u8 = switch (m.role) {
                    .system => "<|im_system|>system<|im_middle|>",
                    .user => "<|im_user|>user<|im_middle|>",
                    .assistant => "<|im_assistant|>assistant<|im_middle|>",
                };
                try w.print("{s}{s}<|im_end|>", .{ header, m.content });
            }
            try w.writeAll("<|im_assistant|>assistant<|im_middle|>");
        },
        .kimi_k3 => {
            // Kimi K3's XTML: the structural markers are special tokens, the tag names plain text.
            // The assistant's channels (`<|open|>think<|sep|>...`) are produced by the model itself.
            for (messages) |m| {
                try w.print("<|open|>message role=\"{s}\"<|sep|>{s}<|close|>message<|sep|><|end_of_msg|>", .{ @tagName(m.role), m.content });
            }
            try w.writeAll("<|open|>message role=\"assistant\"<|sep|>");
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
    try std.testing.expectEqual(Template.kimi, detect("{{ '<|im_user|>user<|im_middle|>' }}", "deepseek_v3"));
    try std.testing.expectEqual(Template.kimi, detect(null, "kimi_k25"));
    try std.testing.expectEqual(Template.kimi_k3, detect(null, "kimi_k3"));
    try std.testing.expectEqual(Template.kimi_k3, detect("<|open|>message<|sep|><|close|>message<|sep|><|end_of_msg|>", "kimi_k3"));
    const k = try renderPrompt(gpa, .kimi, "Sys.", "Hi");
    defer gpa.free(k);
    try std.testing.expectEqualStrings("<|im_system|>system<|im_middle|>Sys.<|im_end|><|im_user|>user<|im_middle|>Hi<|im_end|><|im_assistant|>assistant<|im_middle|>", k);
    const k_no_sys = try render(gpa, .kimi, &.{.{ .role = .user, .content = "Hi" }});
    defer gpa.free(k_no_sys);
    try std.testing.expectEqualStrings("<|im_system|>system<|im_middle|>You are a helpful assistant<|im_end|><|im_user|>user<|im_middle|>Hi<|im_end|><|im_assistant|>assistant<|im_middle|>", k_no_sys);
    const k3 = try renderPrompt(gpa, .kimi_k3, "Sys.", "Hi");
    defer gpa.free(k3);
    try std.testing.expectEqualStrings("<|open|>message role=\"system\"<|sep|>Sys.<|close|>message<|sep|><|end_of_msg|><|open|>message role=\"user\"<|sep|>Hi<|close|>message<|sep|><|end_of_msg|><|open|>message role=\"assistant\"<|sep|>", k3);
    // Jamba: `<|eom|>` between messages only, a space after the role header
    // except on the generation turn. (ai21labs/Jamba-tiny-dev, verified
    // token-for-token against transformers' apply_chat_template.)
    try std.testing.expectEqual(Template.jamba, detect("{{- bom_str + handle_role(role) }} <|bom|>", "jamba"));
    try std.testing.expectEqual(Template.jamba, detect(null, "jamba"));
    const jb = try renderPrompt(gpa, .jamba, "Sys.", "Hi");
    defer gpa.free(jb);
    try std.testing.expectEqualStrings("<|startoftext|><|bom|><|system|> Sys.<|eom|><|bom|><|user|> Hi<|eom|><|bom|><|assistant|>", jb);
    // Laguna (poolside/Laguna-XS.2, verified token-for-token against
    // transformers' apply_chat_template).
    try std.testing.expectEqual(Template.laguna, detect("{{- \"<user>\\n\" + content + \"\\n</user>\\n\" -}}{{- \"<assistant>\\n\" -}}", "laguna"));
    try std.testing.expectEqual(Template.laguna, detect(null, "laguna"));
    const lg = try renderPrompt(gpa, .laguna, "Sys. ", "Hi");
    defer gpa.free(lg);
    try std.testing.expectEqualStrings("\xe3\x80\x88|EOS|\xe3\x80\x89<system>\n\nSys.\n</system>\n<user>\nHi\n</user>\n<assistant>\n</think>", lg);
    const lg_no_sys = try render(gpa, .laguna, &.{ .{ .role = .user, .content = "Hi" }, .{ .role = .assistant, .content = " Yo " }, .{ .role = .user, .content = "Bye" } });
    defer gpa.free(lg_no_sys);
    try std.testing.expectEqualStrings("\xe3\x80\x88|EOS|\xe3\x80\x89<system>\n\n" ++ laguna_default_system ++ "\n</system>\n<user>\nHi\n</user>\n<assistant>\n</think>\nYo\n</assistant>\n<user>\nBye\n</user>\n<assistant>\n</think>", lg_no_sys);
    // Command R7B (estrogen/c4ai-command-r7b-12-2024, a copy of the gated
    // CohereLabs release; token-for-token against apply_chat_template).
    try std.testing.expectEqual(Template.cohere_response, detect("<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|><|START_RESPONSE|>", "cohere2"));
    try std.testing.expectEqual(Template.cohere, detect("<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>", "cohere"));
    const r7 = try render(gpa, .cohere_response, &.{ .{ .role = .user, .content = " Hi " }, .{ .role = .assistant, .content = "Yo" }, .{ .role = .user, .content = "Bye" } });
    defer gpa.free(r7);
    try std.testing.expectEqualStrings("<BOS_TOKEN><|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|><|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|USER_TOKEN|>Hi<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|><|START_RESPONSE|>Yo<|END_RESPONSE|><|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|USER_TOKEN|>Bye<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|><|START_RESPONSE|>", r7);
    // nanochat (nanochat-students/nanochat-d20; token-for-token against
    // apply_chat_template): the system message joins the first user turn.
    try std.testing.expectEqual(Template.nanochat, detect("{{- '<|user_start|>' }}", "nanochat"));
    const nc = try renderPrompt(gpa, .nanochat, "Sys.", " Hi ");
    defer gpa.free(nc);
    try std.testing.expectEqualStrings("<|user_start|>Sys.\n\n Hi <|user_end|><|assistant_start|>", nc);
    const nc2 = try render(gpa, .nanochat, &.{ .{ .role = .user, .content = "Hi" }, .{ .role = .assistant, .content = " Yo " }, .{ .role = .user, .content = "Bye" } });
    defer gpa.free(nc2);
    try std.testing.expectEqualStrings("<|user_start|>Hi<|user_end|><|assistant_start|> Yo <|assistant_end|><|user_start|>Bye<|user_end|><|assistant_start|>", nc2);
    // ERNIE 4.5 and Hunyuan V1 dense (baidu/ERNIE-4.5-0.3B-PT,
    // tencent/Hunyuan-0.5B-Instruct; token-for-token against apply_chat_template).
    try std.testing.expectEqual(Template.ernie, detect("{{- \"Assistant: \" -}}{%- set cls_token = \"<|begin_of_sentence|>\" -%}", "ernie4_5"));
    try std.testing.expectEqual(Template.hunyuan, detect("{{- '<\u{ff5c}hy_User\u{ff5c}>' + message['content'] }}", "hunyuan_v1_dense"));
    const er = try render(gpa, .ernie, &.{ .{ .role = .system, .content = "Sys." }, .{ .role = .user, .content = " Hi " }, .{ .role = .assistant, .content = " Yo " }, .{ .role = .user, .content = "Bye" } });
    defer gpa.free(er);
    try std.testing.expectEqualStrings("<|begin_of_sentence|>Sys.\nUser:  Hi \nAssistant:  Yo <|end_of_sentence|>User: Bye\nAssistant: ", er);
    const hy = try render(gpa, .hunyuan, &.{ .{ .role = .system, .content = "Sys." }, .{ .role = .user, .content = " Hi " }, .{ .role = .assistant, .content = " Yo " }, .{ .role = .user, .content = "Bye" } });
    defer gpa.free(hy);
    try std.testing.expectEqualStrings("<\u{ff5c}hy_begin\u{2581}of\u{2581}sentence\u{ff5c}>Sys.<\u{ff5c}hy_place\u{2581}holder\u{2581}no\u{2581}3\u{ff5c}><\u{ff5c}hy_User\u{ff5c}> Hi <\u{ff5c}hy_Assistant\u{ff5c}> Yo <\u{ff5c}hy_place\u{2581}holder\u{2581}no\u{2581}2\u{ff5c}><\u{ff5c}hy_User\u{ff5c}>Bye<\u{ff5c}hy_Assistant\u{ff5c}>", hy);
    const hy_ns = try render(gpa, .hunyuan, &.{.{ .role = .user, .content = "Hi" }});
    defer gpa.free(hy_ns);
    try std.testing.expectEqualStrings("<\u{ff5c}hy_begin\u{2581}of\u{2581}sentence\u{ff5c}><\u{ff5c}hy_User\u{ff5c}>Hi<\u{ff5c}hy_Assistant\u{ff5c}>", hy_ns);
    // Mistral V7 (mistralai/Ministral-3-3B-Instruct-2512; token-for-token
    // against apply_chat_template).
    try std.testing.expectEqual(Template.mistral_v7, detect("{{- '[SYSTEM_PROMPT]' }}{{- '[INST]' }}", "ministral3"));
    try std.testing.expectEqual(Template.mistral, detect("{{ '[INST] ' + message['content'] }}", "mistral"));
    const m7 = try render(gpa, .mistral_v7, &.{ .{ .role = .system, .content = "Sys.\n" }, .{ .role = .user, .content = " Hi " }, .{ .role = .assistant, .content = "Yo" }, .{ .role = .user, .content = "Bye" } });
    defer gpa.free(m7);
    try std.testing.expectEqualStrings("<s>[SYSTEM_PROMPT]Sys.\n[/SYSTEM_PROMPT][INST] Hi [/INST]Yo</s>[INST]Bye[/INST]", m7);
    // A template that emits the BOS in front of a family that does not.
    try std.testing.expectEqualStrings("<|begin_of_text|>", templateBos("{{bos_token}}\n{%- if tools %}<|im_start|>", "<|begin_of_text|>"));
    try std.testing.expectEqualStrings("", templateBos("{% for m in messages %}<|im_start|>{{ bos_token }}", "<|begin_of_text|>"));
    try std.testing.expectEqualStrings("", templateBos("{{bos_token}}<|im_start|>", null));
}

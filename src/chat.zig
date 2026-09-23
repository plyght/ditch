//! Chat prompt formatting. A model is prompted with its own `chat_template`,
//! rendered by `jinja.zig` the way transformers' `apply_chat_template` renders
//! it (`Format`). The named template families below are the fallback, for a
//! model that ships no template or one that cannot be rendered, and can be
//! forced with the `chat_template` setting.

const std = @import("std");
const arch = @import("arch.zig");
const jinja = @import("jinja.zig");
const Allocator = std.mem.Allocator;

pub const Template = enum {
    chatml,
    llama3,
    /// Llama 3.1 / 3.3: Llama 3 with a system block that always opens with
    /// `Cutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\n`.
    llama31,
    /// Llama 3.2: the same, dated today (`strftime_now("%d %b %Y")`).
    llama32,
    llama2,
    mistral,
    /// Mistral's V7 (tekken) template — Ministral 3, Mistral Small 3.x,
    /// Magistral, Devstral: `<s>[SYSTEM_PROMPT]...[/SYSTEM_PROMPT][INST]...[/INST]`.
    mistral_v7,
    /// Mistral v0.1 / v0.2, Mixtral v0.1: `<s> [INST] ... [/INST]`, with spaces.
    mistral_spaced,
    /// DeepSeek V2: `<｜begin▁of▁sentence｜>{system}\n\nUser: ...\n\nAssistant:`.
    deepseek_v2,
    gemma,
    /// Gemma 4: `<|turn>user\n...<turn|>\n<|turn>model\n<|channel>thought\n<channel|>`,
    /// with a system turn of its own (Gemma 2 / 3 fold the system prompt into the user's).
    gemma4,
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
    /// AI21 Jamba Reasoning: ChatML after `<|startoftext|>`, a thinking
    /// instruction in front of the last user turn, and `<think>\n` opened.
    jamba_reasoning,
    /// Qwen 3.5: ChatML with trimmed contents and `<think>\n` opened.
    qwen3_5,
    /// MiMo V2: ChatML without newlines between turns, `<think></think>` closed
    /// at once, and a default system message.
    mimo,
    /// SmolLM3: a dated `## Metadata` system block, reasoning mode `/think`.
    smollm3,
    /// Phi-4: `<|im_start|>user<|im_sep|>...<|im_end|>`.
    phi4,
    /// Phi-4-mini: `<|user|>...<|end|><|assistant|>`, no newlines.
    phi4_mini,
    /// DeepSeek V3.1 / V3.2: DeepSeek V3's turns, the assistant opening with `</think>` (thinking off).
    deepseek_v31,
    /// GLM-4.7: GLM-4's headers without newlines, `<think>` opened.
    glm47,
    /// EXAONE 4: `[|user|]\n...[|endofturn|]\n[|assistant|]\n<think>\n\n</think>\n\n`.
    exaone4,
    /// K-EXAONE (EXAONE MoE): `<|user|>\n...<|endofturn|>\n<|assistant|>\n<think>\n`.
    k_exaone,
    /// dots.llm1: `<|system|>...<|endofsystem|><|userprompt|>...<|endofuserprompt|><|response|>`.
    dots,
    /// Seed-OSS: `<seed:bos>user\n...<seed:eos><seed:bos>assistant\n`.
    seed,
    /// Hunyuan A13B (V1 MoE): `<|startoftext|>{system}<|extra_4|>{user}<|extra_0|>`.
    hunyuan_moe,
    /// MiniMax-M2: `]~!b[]~b]system\n...[e~[\n]~b]user\n...[e~[\n]~b]ai\n<think>\n`.
    minimax_m2,
    /// Nemotron Nano 2 (Nemotron-H): `<SPECIAL_10>System\n...\n<SPECIAL_11>User\n...\n<SPECIAL_11>Assistant\n<think>\n`.
    nemotron_nano,
    /// Nemotron Mini: `<extra_id_0>System\n...\n\n<extra_id_1>User\n...\n<extra_id_1>Assistant\n`.
    nemotron_mini,
    /// Solar Open: `<|begin|>user<|content|>...<|end|><|begin|>assistant`, after a dated provider prompt.
    solar_open,
    /// ERNIE 4.5: `<|begin_of_sentence|>{system}\nUser: ...\nAssistant: `.
    ernie,
    /// Hunyuan (V1 dense): `<｜hy_begin▁of▁sentence｜>{system}<｜hy_place▁holder▁no▁3｜><｜hy_User｜>...<｜hy_Assistant｜>`.
    hunyuan,
    /// Apertus: `<|system_start|>...<|system_end|><|developer_start|>Deliberation: disabled\n
    /// Tool Capabilities: disabled<|developer_end|><|user_start|>...<|user_end|><|assistant_start|>`.
    apertus,
    /// nanochat: `<|user_start|>...<|user_end|><|assistant_start|>`; a system
    /// message is prepended to the first user turn, followed by a blank line.
    nanochat,
    /// Laguna (Poolside): `〈|EOS|〉<system>\n\n...\n</system>\n<user>\n...\n</user>\n<assistant>\n</think>`
    /// (a default system message is inserted when the conversation has none;
    /// `</think>` is the non-thinking generation prompt).
    laguna,
    /// Falcon (7B/40B instruct): `{system}\n\nUser: ...\n\nAssistant:`, contents
    /// stripped and with blank lines collapsed to single line breaks.
    falcon,
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
        if (has(t, "Begin by thinking about the reasoning process")) return .jamba_reasoning;
        if (has(t, "<|im_sep|>")) return .phi4;
        if (has(t, "## Metadata") and has(t, "Reasoning Mode")) return .smollm3;
        if (has(t, "<|im_start|>") and has(t, "<think></think>")) return .mimo;
        if (has(t, "<|im_start|>") and has(t, "{{- '<think>\\n' }}")) return .qwen3_5;
        if (has(t, "<|endofuserprompt|>")) return .dots;
        if (has(t, "<seed:bos>")) return .seed;
        if (has(t, "<|extra_4|>")) return .hunyuan_moe;
        if (has(t, "]~b]")) return .minimax_m2;
        if (has(t, "<SPECIAL_11>")) return .nemotron_nano;
        if (has(t, "<extra_id_1>")) return .nemotron_mini;
        if (has(t, "<|begin|>") and has(t, "<|content|>")) return .solar_open;
        if (has(t, "<|endofturn|>") and has(t, "<|user|>")) return .k_exaone;
        if (has(t, "'<|' + message['role'] + '|>' + message['content'] + '<|end|>'")) return .phi4_mini;
        if (has(t, "<|user|>\n' + message['content'] + eos_token")) return .zephyr;
        if (has(t, "<\u{ff5c}hy_User\u{ff5c}>")) return .hunyuan;
        if (has(t, "<|begin_of_sentence|>") and has(t, "Assistant: ")) return .ernie;
        if (has(t, "<|system_start|>") and has(t, "<|developer_start|>")) return .apertus;
        if (has(t, "<|user_start|>")) return .nanochat;
        if (has(t, "</user>") and has(t, "<assistant>")) return .laguna;
        if (has(t, "<|im_middle|>")) return .kimi;
        if (has(t, "<|end_of_msg|>")) return .kimi_k3;
        if (has(t, "<|im_start|>")) return .chatml;
        if (has(t, "<|start_header_id|>") and has(t, "Cutting Knowledge Date")) return if (has(t, "strftime_now")) .llama32 else .llama31;
        if (has(t, "<|start_header_id|>")) return .llama3;
        if (has(t, "<|header_start|>")) return .llama4;
        if (has(t, "<|turn>")) return .gemma4;
        if (has(t, "<start_of_turn>")) return .gemma;
        if (has(t, "<|START_OF_TURN_TOKEN|>")) return if (has(t, "<|START_RESPONSE|>")) .cohere_response else .cohere;
        if (has(t, "<|start|>") and has(t, "<|message|>")) return .harmony;
        if (has(t, "<|start_of_role|>")) return .granite;
        if (has(t, "[|user|]") or has(t, "'[|' + message['role'] + '|]'")) return if (has(t, "<think>")) .exaone4 else .exaone;
        if (has(t, "<\xef\xbd\x9cUser\xef\xbd\x9c>")) return if (has(t, "{{'</think>'}}")) .deepseek_v31 else .deepseek;
        if (has(t, "[gMASK]")) return if (has(t, "else '<think>'")) .glm47 else .glm4;
        if (has(t, "<|user|>") and has(t, "<|end|>")) return .phi3;
        if (has(t, "<|user|>") and has(t, "<|endoftext|>")) return .zephyr;
        if (has(t, "<|user|>")) return .olmo;
        if (has(t, "<<SYS>>")) return .llama2;
        if (has(t, "[INST]")) return if (has(t, "[SYSTEM_PROMPT]")) .mistral_v7 else if (has(t, "' [INST] '")) .mistral_spaced else .mistral;
        if ((has(t, "'\n\nUser: '") and has(t, "'\n\nAssistant:'")) or (has(t, "'\\n\\nUser: '") and has(t, "'\\n\\nAssistant:'"))) return .falcon;
        if (has(t, "'User: '") and has(t, "'Assistant:'")) return .deepseek_v2;
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

/// A calendar date, for templates that write today's (gpt-oss, SmolLM3, Solar Open).
pub const Date = struct {
    year: u16 = 2025,
    month: u4 = 1,
    day: u5 = 1,

    /// The UTC date of `seconds` since the Unix epoch.
    pub fn fromUnix(seconds: i64) Date {
        const epoch = std.time.epoch;
        const es = epoch.EpochSeconds{ .secs = @intCast(@max(seconds, 0)) };
        const yd = es.getEpochDay().calculateYearDay();
        const md = yd.calculateMonthDay();
        return .{ .year = yd.year, .month = @intCast(md.month.numeric()), .day = @intCast(md.day_index + 1) };
    }
};

/// The date `strftime_now` would give (Jinja's is process-global too); the
/// engine sets it from the real clock.
pub var today: Date = .{};

const month_names = [_][]const u8{ "January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December" };

/// Falcon's `.replace('\r\n', '\n').replace('\n\n', '\n')`, applied in that order
/// (one left-to-right pass each, as Python's replace does).
fn writeFalconContent(w: *std.Io.Writer, s: []const u8) !void {
    var crlf: std.Io.Writer.Allocating = .init(std.heap.page_allocator);
    defer crlf.deinit();
    var i: usize = 0;
    while (i < s.len) : (i += 1) {
        if (s[i] == '\r' and i + 1 < s.len and s[i + 1] == '\n') {
            try crlf.writer.writeByte('\n');
            i += 1;
        } else try crlf.writer.writeByte(s[i]);
    }
    const t = crlf.written();
    i = 0;
    while (i < t.len) : (i += 1) {
        if (t[i] == '\n' and i + 1 < t.len and t[i + 1] == '\n') {
            try w.writeByte('\n');
            i += 1;
        } else try w.writeByte(t[i]);
    }
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
        .llama31, .llama32 => {
            // The system block is always written, with an empty message when
            // the conversation has none.
            try w.writeAll("<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: ");
            if (template == .llama32) {
                try w.print("{d:0>2} {s} {d}", .{ today.day, month_names[today.month - 1][0..3], today.year });
            } else try w.writeAll("26 Jul 2024");
            try w.writeAll("\n\n");
            var rest = messages;
            if (rest.len > 0 and rest[0].role == .system) {
                try w.writeAll(trim(rest[0].content));
                rest = rest[1..];
            }
            try w.writeAll("<|eot_id|>");
            for (rest) |m| try w.print("<|start_header_id|>{s}<|end_header_id|>\n\n{s}<|eot_id|>", .{ @tagName(m.role), trim(m.content) });
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
        .gemma4 => {
            try w.writeAll("<bos>");
            var rest = messages;
            if (rest.len > 0 and rest[0].role == .system) {
                try w.print("<|turn>system\n{s}<turn|>\n", .{trim(rest[0].content)});
                rest = rest[1..];
            }
            for (rest) |m| switch (m.role) {
                .system, .user => try w.print("<|turn>{s}\n{s}<turn|>\n", .{ @tagName(m.role), trim(m.content) }),
                .assistant => {
                    // `strip_thinking`: what lies between `<|channel>` and `<channel|>` goes.
                    try w.writeAll("<|turn>model\n");
                    var kept: std.Io.Writer.Allocating = .init(gpa);
                    defer kept.deinit();
                    var parts = std.mem.splitSequence(u8, m.content, "<channel|>");
                    while (parts.next()) |part| {
                        const cut = std.mem.indexOf(u8, part, "<|channel>") orelse part.len;
                        try kept.writer.writeAll(part[0..cut]);
                    }
                    try w.print("{s}<turn|>\n", .{trim(kept.written())});
                },
            };
            try w.writeAll("<|turn>model\n<|channel>thought\n<channel|>");
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
            // mistralai/Mistral-7B-Instruct-v0.3: the system message goes in
            // front of the *last* user turn; contents verbatim.
            try w.writeAll("<s>");
            var system: ?[]const u8 = null;
            var last_user: usize = messages.len;
            for (messages, 0..) |m, i| switch (m.role) {
                .system => system = m.content,
                .user => last_user = i,
                .assistant => {},
            };
            for (messages, 0..) |m, i| {
                switch (m.role) {
                    .system => {},
                    .user => {
                        try w.writeAll("[INST] ");
                        if (i == last_user) if (system) |s| try w.print("{s}\n\n", .{s});
                        try w.print("{s}[/INST]", .{m.content});
                    },
                    .assistant => try w.print(" {s}</s>", .{trim(m.content)}),
                }
            }
        },
        .mistral_spaced => {
            // mistralai/Mixtral-8x7B-Instruct-v0.1: the system message joins
            // the first user turn.
            try w.writeAll("<s>");
            var system: ?[]const u8 = null;
            var first_user = true;
            for (messages) |m| {
                switch (m.role) {
                    .system => system = m.content,
                    .user => {
                        try w.writeAll(" [INST] ");
                        if (first_user) if (system) |s| try w.print("{s}\n\n", .{s});
                        first_user = false;
                        try w.print("{s} [/INST]", .{m.content});
                    },
                    .assistant => try w.print(" {s}</s>", .{m.content}),
                }
            }
        },
        .deepseek_v2 => {
            // deepseek-ai/DeepSeek-V2-Lite-Chat: contents verbatim.
            try w.writeAll("<\u{ff5c}begin\u{2581}of\u{2581}sentence\u{ff5c}>");
            for (messages) |m| {
                switch (m.role) {
                    .system => try w.print("{s}\n\n", .{m.content}),
                    .user => try w.print("User: {s}\n\n", .{m.content}),
                    .assistant => try w.print("Assistant: {s}<\u{ff5c}end\u{2581}of\u{2581}sentence\u{ff5c}>", .{m.content}),
                }
            }
            try w.writeAll("Assistant:");
        },
        .phi3, .zephyr, .olmo => {
            // `<|role|>` headers; the turn terminator differs per family.
            const end: []const u8 = switch (template) {
                .phi3 => "<|end|>\n",
                .zephyr => "<|endoftext|>\n",
                else => "\n",
            };
            // Contents verbatim (Phi-3.5, StableLM Zephyr, OLMo 2, Falcon 3).
            for (messages) |m| try w.print("<|{s}|>\n{s}{s}", .{ @tagName(m.role), m.content, end });
            try w.writeAll("<|assistant|>\n");
        },
        .glm4 => {
            // GLM-4-0414 / GLM-4.5 / glm-4-9b-chat-hf: contents verbatim, and
            // the generation header ends the prompt (the model writes the newline).
            try w.writeAll("[gMASK]<sop>");
            for (messages) |m| try w.print("<|{s}|>\n{s}", .{ @tagName(m.role), m.content });
            try w.writeAll("<|assistant|>");
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
                    .system => try w.print("{s}", .{m.content}),
                    .user => try w.print("<\xef\xbd\x9cUser\xef\xbd\x9c>{s}", .{m.content}),
                    .assistant => try w.print("<\xef\xbd\x9cAssistant\xef\xbd\x9c>{s}<\xef\xbd\x9cend\xe2\x96\x81of\xe2\x96\x81sentence\xef\xbd\x9c>", .{trim(m.content)}),
                }
            }
            try w.writeAll("<\xef\xbd\x9cAssistant\xef\xbd\x9c>");
        },
        .harmony => {
            // openai/gpt-oss: the model-identity system block (dated, reasoning
            // medium) always comes first; the conversation's system message is
            // the developer's instructions.
            try w.print("<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.\nKnowledge cutoff: 2024-06\nCurrent date: {d:0>4}-{d:0>2}-{d:0>2}\n\nReasoning: medium\n\n# Valid channels: analysis, commentary, final. Channel must be included for every message.<|end|>", .{ today.year, today.month, today.day });
            for (messages, 0..) |m, i| {
                switch (m.role) {
                    .system => if (i == 0) try w.print("<|start|>developer<|message|># Instructions\n\n{s}\n\n<|end|>", .{m.content}),
                    .user => try w.print("<|start|>user<|message|>{s}<|end|>", .{m.content}),
                    .assistant => try w.print("<|start|>assistant<|channel|>final<|message|>{s}<|end|>", .{m.content}),
                }
            }
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
        .apertus => {
            // swiss-ai/Apertus-*-Instruct, with deliberation and tools off.
            // Without a system message the template inserts a dated default,
            // which ditch, always passing one, leaves out. The BOS comes from
            // the tokenizer (`add_bos_token`).
            var i: usize = 0;
            if (messages.len > 0 and messages[0].role == .system) {
                try w.print("<|system_start|>{s}<|system_end|>", .{messages[0].content});
                i = 1;
            }
            try w.writeAll("<|developer_start|>Deliberation: disabled\nTool Capabilities: disabled<|developer_end|>");
            var in_assistant = false;
            for (messages[i..]) |m| {
                switch (m.role) {
                    .system => {},
                    .user => {
                        if (in_assistant) try w.writeAll("<|assistant_end|>");
                        in_assistant = false;
                        try w.print("<|user_start|>{s}<|user_end|>", .{m.content});
                    },
                    .assistant => {
                        if (!in_assistant) try w.writeAll("<|assistant_start|>");
                        in_assistant = true;
                        try w.writeAll(m.content);
                    },
                }
            }
            if (in_assistant) try w.writeAll("<|assistant_end|>");
            try w.writeAll("<|assistant_start|>");
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
        .jamba_reasoning => {
            // ai21labs/AI21-Jamba-Reasoning-3B.
            try w.writeAll("<|startoftext|>");
            var last_user: usize = messages.len;
            for (messages, 0..) |m, i| if (m.role == .user) {
                last_user = i;
            };
            for (messages, 0..) |m, i| {
                const pre: []const u8 = if (i == last_user) "Begin by thinking about the reasoning process in the mind within <think> </think> tags and then proceed to give your response.\n" else "";
                try w.print("<|im_start|>{s}\n{s}{s}<|im_end|>\n", .{ @tagName(m.role), pre, m.content });
            }
            try w.writeAll("<|im_start|>assistant\n<think>\n");
        },
        .qwen3_5 => {
            for (messages) |m| try w.print("<|im_start|>{s}\n{s}<|im_end|>\n", .{ @tagName(m.role), trim(m.content) });
            try w.writeAll("<|im_start|>assistant\n<think>\n");
        },
        .mimo => {
            if (messages.len == 0 or messages[0].role != .system) try w.writeAll("<|im_start|>system\nYou are MiMo, a helpful AI assistant engineered by Xiaomi.<|im_end|>");
            for (messages) |m| {
                switch (m.role) {
                    .assistant => try w.print("<|im_start|>assistant\n<think></think>{s}<|im_end|>", .{m.content}),
                    else => try w.print("<|im_start|>{s}\n{s}<|im_end|>", .{ @tagName(m.role), m.content }),
                }
            }
            try w.writeAll("<|im_start|>assistant\n<think></think>");
        },
        .smollm3 => {
            // HuggingFaceTB/SmolLM3-3B: the system turn has no `<|im_end|>`.
            // Without a system message the template writes its long default
            // instructions, which ditch, always passing one, leaves out.
            try w.print("<|im_start|>system\n## Metadata\n\nKnowledge Cutoff Date: June 2025\nToday Date: {d:0>2} {s} {d}\nReasoning Mode: /think\n\n## Custom Instructions\n\n", .{ today.day, month_names[today.month - 1], today.year });
            var i: usize = 0;
            if (messages.len > 0 and messages[0].role == .system) {
                try w.writeAll(std.mem.trimEnd(u8, messages[0].content, " \t\r\n"));
                i = 1;
            }
            try w.writeAll("\n\n");
            for (messages[i..]) |m| try w.print("<|im_start|>{s}\n{s}<|im_end|>\n", .{ @tagName(m.role), m.content });
            try w.writeAll("<|im_start|>assistant\n");
        },
        .phi4 => {
            for (messages) |m| try w.print("<|im_start|>{s}<|im_sep|>{s}<|im_end|>", .{ @tagName(m.role), m.content });
            try w.writeAll("<|im_start|>assistant<|im_sep|>");
        },
        .phi4_mini => {
            for (messages) |m| try w.print("<|{s}|>{s}<|end|>", .{ @tagName(m.role), m.content });
            try w.writeAll("<|assistant|>");
        },
        .deepseek_v31 => {
            try w.writeAll("<\u{ff5c}begin\u{2581}of\u{2581}sentence\u{ff5c}>");
            for (messages) |m| {
                switch (m.role) {
                    .system => try w.writeAll(m.content),
                    .user => try w.print("<\u{ff5c}User\u{ff5c}>{s}", .{m.content}),
                    .assistant => try w.print("<\u{ff5c}Assistant\u{ff5c}></think>{s}<\u{ff5c}end\u{2581}of\u{2581}sentence\u{ff5c}>", .{m.content}),
                }
            }
            try w.writeAll("<\u{ff5c}Assistant\u{ff5c}></think>");
        },
        .glm47 => {
            try w.writeAll("[gMASK]<sop>");
            for (messages) |m| {
                switch (m.role) {
                    .assistant => try w.print("<|assistant|></think>{s}", .{m.content}),
                    else => try w.print("<|{s}|>{s}", .{ @tagName(m.role), m.content }),
                }
            }
            try w.writeAll("<|assistant|><think>");
        },
        .exaone4 => {
            for (messages) |m| {
                switch (m.role) {
                    .assistant => try w.print("[|assistant|]\n<think>\n\n</think>\n\n{s}[|endofturn|]\n", .{m.content}),
                    else => try w.print("[|{s}|]\n{s}[|endofturn|]\n", .{ @tagName(m.role), m.content }),
                }
            }
            try w.writeAll("[|assistant|]\n<think>\n\n</think>\n\n");
        },
        .k_exaone => {
            for (messages) |m| {
                switch (m.role) {
                    .assistant => try w.print("<|assistant|>\n<think>\n\n</think>\n\n{s}<|endofturn|>\n", .{m.content}),
                    else => try w.print("<|{s}|>\n{s}<|endofturn|>\n", .{ @tagName(m.role), m.content }),
                }
            }
            try w.writeAll("<|assistant|>\n<think>\n");
        },
        .dots => {
            var i: usize = 0;
            if (messages.len > 0 and messages[0].role == .system) {
                try w.print("<|system|>{s}<|endofsystem|>", .{messages[0].content});
                i = 1;
            } else try w.writeAll("<|system|>You are a helpful assistant.<|endofsystem|>");
            for (messages[i..]) |m| {
                switch (m.role) {
                    .system => {},
                    .user => try w.print("<|userprompt|>{s}<|endofuserprompt|>", .{m.content}),
                    .assistant => try w.print("<|response|>{s}<|endofresponse|>", .{m.content}),
                }
            }
            if (messages.len == 0 or messages[messages.len - 1].role == .user) try w.writeAll("<|response|>");
        },
        .seed => {
            for (messages) |m| try w.print("<seed:bos>{s}\n{s}<seed:eos>", .{ @tagName(m.role), m.content });
            try w.writeAll("<seed:bos>assistant\n");
        },
        .hunyuan_moe => {
            // tencent/Hunyuan-A13B-Instruct: no generation header; the model
            // answers after `<|extra_0|>`.
            const has_head = messages.len > 0 and messages[0].content.len > 0;
            for (messages, 0..) |m, i| {
                switch (m.role) {
                    .system => if (i == 0 and has_head) try w.print("<|startoftext|>{s}<|extra_4|>", .{m.content}) else try w.writeAll(m.content),
                    .user => if (i == 1 and has_head and messages[0].role == .system)
                        try w.print("{s}<|extra_0|>", .{m.content})
                    else
                        try w.print("<|startoftext|>{s}<|extra_0|>", .{m.content}),
                    .assistant => try w.print("{s}<|eos|>", .{m.content}),
                }
            }
        },
        .minimax_m2 => {
            try w.writeAll("]~!b[]~b]system\n");
            var i: usize = 0;
            if (messages.len > 0 and messages[0].role == .system and messages[0].content.len > 0) {
                try w.writeAll(messages[0].content);
                i = 1;
            } else {
                if (messages.len > 0 and messages[0].role == .system) i = 1;
                try w.writeAll("You are MiniMax-M2, a helpful AI assistant built by MiniMax. Knowledge cutoff: 2025-06.");
            }
            try w.writeAll("[e~[\n");
            for (messages[i..]) |m| {
                switch (m.role) {
                    .system => {},
                    .user => try w.print("]~b]user\n{s}[e~[\n", .{m.content}),
                    .assistant => try w.print("]~b]ai\n{s}[e~[\n", .{m.content}),
                }
            }
            try w.writeAll("]~b]ai\n<think>\n");
        },
        .nemotron_nano => {
            var i: usize = 0;
            try w.writeAll("<SPECIAL_10>System\n");
            if (messages.len > 0 and messages[0].role == .system) {
                try w.writeAll(trim(messages[0].content));
                i = 1;
            }
            try w.writeAll("\n");
            for (messages[i..]) |m| {
                switch (m.role) {
                    .system => {},
                    .user => try w.print("<SPECIAL_11>User\n{s}\n", .{trim(m.content)}),
                    .assistant => try w.print("<SPECIAL_11>Assistant\n{s}\n<SPECIAL_12>\n", .{trim(m.content)}),
                }
            }
            try w.writeAll("<SPECIAL_11>Assistant\n<think>\n");
        },
        .nemotron_mini => {
            try w.writeAll("<extra_id_0>System");
            for (messages) |m| if (m.role == .system) try w.print("\n{s}", .{trim(m.content)});
            try w.writeAll("\n\n");
            for (messages) |m| {
                switch (m.role) {
                    .system => {},
                    .user => try w.print("<extra_id_1>User\n{s}\n", .{trim(m.content)}),
                    .assistant => try w.print("<extra_id_1>Assistant\n{s}\n", .{trim(m.content)}),
                }
            }
            try w.writeAll("<extra_id_1>Assistant\n");
        },
        .solar_open => {
            // upstage/Solar-Open-100B: a dated provider prompt always opens the
            // system turn; the conversation's own follows under its heading.
            try w.print("<|begin|>system<|content|>## Provider System Prompt\n\nYou are Solar Open 100B, a large language model trained by Upstage AI, a Korean startup. Your knowledge cutoff is 2025-07. The current date is {d:0>4}-{d:0>2}-{d:0>2}.", .{ today.year, today.month, today.day });
            var i: usize = 0;
            if (messages.len > 0 and messages[0].role == .system) {
                try w.print("\n\n## System Prompt\n\n{s}", .{messages[0].content});
                i = 1;
            }
            try w.writeAll("<|end|>");
            for (messages[i..]) |m| {
                switch (m.role) {
                    .system => {},
                    .user => try w.print("<|begin|>user<|content|>{s}<|end|>", .{m.content}),
                    .assistant => try w.print("<|begin|>assistant<|content|>{s}<|end|>", .{m.content}),
                }
            }
            try w.writeAll("<|begin|>assistant");
        },
        .llama4 => {
            try w.writeAll("<|begin_of_text|>");
            for (messages) |m| try w.print("<|header_start|>{s}<|header_end|>\n\n{s}<|eot|>", .{ @tagName(m.role), trim(m.content) });
            try w.writeAll("<|header_start|>assistant<|header_end|>\n\n");
        },
        .exaone => {
            // LGAI-EXAONE/EXAONE-3.5-*: contents verbatim; an empty system turn
            // when the conversation does not open with one.
            if (messages.len == 0 or messages[0].role != .system) try w.writeAll("[|system|][|endofturn|]\n");
            for (messages) |m| {
                switch (m.role) {
                    .system => try w.print("[|system|]{s}[|endofturn|]\n", .{m.content}),
                    .user => try w.print("[|user|]{s}\n", .{m.content}),
                    .assistant => try w.print("[|assistant|]{s}[|endofturn|]\n", .{m.content}),
                }
            }
            try w.writeAll("[|assistant|]");
        },
        .granite => {
            for (messages) |m| try w.print("<|start_of_role|>{s}<|end_of_role|>{s}<|end_of_text|>\n", .{ @tagName(m.role), m.content });
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
        .falcon => {
            var rest = messages;
            if (rest.len > 0 and rest[0].role == .system) {
                try w.writeAll(trim(rest[0].content));
                rest = rest[1..];
            }
            for (rest) |m| {
                try w.writeAll(if (m.role == .assistant) "\n\nAssistant: " else "\n\nUser: ");
                try writeFalconContent(w, trim(m.content));
            }
            try w.writeAll("\n\nAssistant:");
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

/// Seconds since the Unix epoch for templates that call `strftime_now`; the
/// engine sets it from the real clock with `today`.
pub var now_seconds: i64 = 1735689600;

/// A model's own chat template, chosen as transformers chooses it:
/// `chat_template.jinja`, else `tokenizer_config.json`'s `chat_template` (a
/// string, or a list of named templates of which "default" is taken), else
/// a processor's `chat_template.json`.
pub fn pickTemplate(arena: Allocator, tokenizer_config_json: ?[]const u8, jinja_file: ?[]const u8, processor_json: ?[]const u8) !?[]const u8 {
    if (jinja_file) |t| return t;
    if (tokenizer_config_json) |tc| {
        const parsed = std.json.parseFromSliceLeaky(std.json.Value, arena, tc, .{}) catch null;
        if (parsed) |v| if (v == .object) if (v.object.get("chat_template")) |ct| switch (ct) {
            .string => |t| return t,
            .array => |list| {
                var first: ?[]const u8 = null;
                for (list.items) |item| {
                    if (item != .object) continue;
                    const t = item.object.get("template") orelse continue;
                    if (t != .string) continue;
                    if (first == null) first = t.string;
                    if (item.object.get("name")) |n| if (n == .string and std.mem.eql(u8, n.string, "default")) return t.string;
                }
                if (first) |t| return t;
            },
            else => {},
        };
    }
    if (processor_json) |pj| {
        const parsed = std.json.parseFromSliceLeaky(std.json.Value, arena, pj, .{}) catch return null;
        if (parsed == .object) if (parsed.object.get("chat_template")) |t| if (t == .string) return t.string;
    }
    return null;
}

/// The special tokens transformers passes to a chat template: every
/// top-level `*_token` of `tokenizer_config.json` that is a string or an
/// added token (`{"content": ...}`), and the named `extra_special_tokens`.
/// `special_tokens_map.json` overrides them when the config is the legacy
/// kind without `added_tokens_decoder`, as transformers reads it.
pub fn specialTokens(arena: Allocator, tokenizer_config_json: ?[]const u8, special_tokens_map_json: ?[]const u8) ![]const jinja.Var {
    var out = std.ArrayList(jinja.Var).empty;
    const Add = struct {
        fn content(v: std.json.Value) ?[]const u8 {
            return switch (v) {
                .string => |s| s,
                .object => |o| if (o.get("content")) |c| (if (c == .string) c.string else null) else null,
                else => null,
            };
        }
        fn put(a: Allocator, list: *std.ArrayList(jinja.Var), name: []const u8, value: []const u8) !void {
            for (list.items) |*x| if (std.mem.eql(u8, x.name, name)) {
                x.value = .{ .string = value };
                return;
            };
            try list.append(a, .{ .name = name, .value = .{ .string = value } });
        }
        fn fromObject(a: Allocator, list: *std.ArrayList(jinja.Var), o: std.json.ObjectMap) !void {
            var it = o.iterator();
            while (it.next()) |e| {
                if (!std.mem.endsWith(u8, e.key_ptr.*, "_token")) continue;
                if (content(e.value_ptr.*)) |c| try put(a, list, e.key_ptr.*, c);
            }
            if (o.get("extra_special_tokens")) |x| if (x == .object) {
                var xi = x.object.iterator();
                while (xi.next()) |e| if (content(e.value_ptr.*)) |c| try put(a, list, e.key_ptr.*, c);
            };
        }
    };
    var legacy = true;
    if (tokenizer_config_json) |tc| {
        const v = std.json.parseFromSliceLeaky(std.json.Value, arena, tc, .{}) catch null;
        if (v) |cfg| if (cfg == .object) {
            try Add.fromObject(arena, &out, cfg.object);
            legacy = cfg.object.get("added_tokens_decoder") == null;
        };
    }
    if (legacy) if (special_tokens_map_json) |sm| {
        const v = std.json.parseFromSliceLeaky(std.json.Value, arena, sm, .{}) catch null;
        if (v) |m| if (m == .object) try Add.fromObject(arena, &out, m.object);
    };
    return out.items;
}

/// How ditch prompts a model: its own chat template when it has one that
/// renders, otherwise a named family.
pub const Format = struct {
    family: Template,
    template: ?*jinja.Template = null,
    /// The tokenizer's named special tokens, passed to the template as
    /// transformers passes `special_tokens_map`.
    tokens: []const jinja.Var = &.{},
    /// The template refuses a system message (Gemma 2, Mistral v0.1 raise):
    /// the system prompt opens the first user message instead, followed by a
    /// blank line, as Gemma 3's own template does.
    fold_system: bool = false,
    /// The template reads message content only as a list of parts
    /// (`[{"type": "text", "text": ...}]`, MiniMax-Text-01 and M1), as a
    /// processor passes it; plain strings are wrapped that way.
    content_parts: bool = false,

    /// A named family only.
    pub fn named(family: Template) Format {
        return .{ .family = family };
    }

    /// Parses `source` (the model's `chat_template`, if any) and checks that
    /// it renders the system + user conversation ditch builds. A template
    /// that cannot be parsed or rendered is dropped for `fallback`, with a
    /// warning saying why.
    pub fn init(gpa: Allocator, source: ?[]const u8, tokens: []const jinja.Var, fallback: Template) !Format {
        var f: Format = .{ .family = fallback, .tokens = tokens };
        const src = source orelse return f;
        var diag: jinja.Diagnostic = .{};
        f.template = jinja.Template.parse(gpa, src, &diag) catch |e| {
            if (e == error.OutOfMemory) return e;
            std.log.warn("the model's chat template could not be parsed ({s}); prompting with the {s} template instead", .{ diag.message(), @tagName(fallback) });
            return f;
        };
        // The first of: as is, contents as parts, the system prompt folded
        // into the user message, both.
        const probe = [_]Message{ .{ .role = .system, .content = "S" }, .{ .role = .user, .content = "U" } };
        var first_error: ?[]const u8 = null;
        defer if (first_error) |m| gpa.free(m);
        for ([_][2]bool{ .{ false, false }, .{ false, true }, .{ true, false }, .{ true, true } }) |mode| {
            f.fold_system = mode[0];
            f.content_parts = mode[1];
            if (f.renderJinja(gpa, &probe, .{}, &diag)) |text| {
                gpa.free(text);
                if (f.fold_system) std.log.warn("the model's chat template refuses a system message (\"{s}\"); the system prompt opens the first user message instead", .{first_error.?});
                return f;
            } else |e| if (e == error.OutOfMemory) return e;
            if (first_error == null) first_error = try gpa.dupe(u8, diag.message());
        }
        std.log.warn("the model's chat template failed to render ({s}); prompting with the {s} template instead", .{ first_error.?, @tagName(fallback) });
        f.template.?.deinit();
        f.template = null;
        f.fold_system = false;
        f.content_parts = false;
        return f;
    }

    pub fn deinit(self: *Format) void {
        if (self.template) |t| t.deinit();
        self.template = null;
    }

    /// The name recorded in manifests and accepted by the `chat_template`
    /// setting: "model" for the model's own template, else the family's.
    pub fn name(self: *const Format) []const u8 {
        return if (self.template != null) "model" else @tagName(self.family);
    }

    pub const Options = struct {
        add_generation_prompt: bool = true,
        /// Extra template variables (`enable_thinking`, ...), overriding the defaults.
        kwargs: []const jinja.Var = &.{},
    };

    /// Renders a conversation. A render error with the model's template
    /// (a conversation shape it refuses) falls back to the family, with a
    /// warning the first time.
    pub fn chat(self: *const Format, gpa: Allocator, messages: []const Message, options: Options) ![]u8 {
        if (self.template != null) {
            var diag: jinja.Diagnostic = .{};
            if (self.renderJinja(gpa, messages, options, &diag)) |text| return text else |e| {
                if (e == error.OutOfMemory) return e;
                if (!warned_render_failure.swap(true, .monotonic))
                    std.log.warn("the model's chat template failed to render a conversation ({s}); using the {s} template for it", .{ diag.message(), @tagName(self.family) });
            }
        }
        return render(gpa, self.family, messages);
    }

    /// The common system + user prompt with the generation prompt.
    pub fn prompt(self: *const Format, gpa: Allocator, system: []const u8, user: []const u8) ![]u8 {
        const msgs = [_]Message{ .{ .role = .system, .content = system }, .{ .role = .user, .content = user } };
        return self.chat(gpa, &msgs, .{});
    }

    fn renderJinja(self: *const Format, gpa: Allocator, messages: []const Message, options: Options, diag: *jinja.Diagnostic) ![]u8 {
        var arena_state: std.heap.ArenaAllocator = .init(gpa);
        defer arena_state.deinit();
        const a = arena_state.allocator();
        var list = std.ArrayList(jinja.Value).empty;
        for (messages) |m| {
            const d = try a.create(jinja.Dict);
            d.* = .{};
            try d.putStr(a, "role", .{ .string = @tagName(m.role) });
            try d.putStr(a, "content", .{ .string = m.content });
            try list.append(a, .{ .dict = d });
        }
        const text = try self.renderValues(a, try jinja.Value.listOf(a, list.items), options, diag);
        return gpa.dupe(u8, text);
    }

    /// Renders `messages` (a list of message dicts, built with `arena`) as
    /// transformers does: the special tokens, `messages`, `tools` and
    /// `documents` (none), `add_generation_prompt`, then the kwargs.
    pub fn renderValues(self: *const Format, arena: Allocator, messages: jinja.Value, options: Options, diag: *jinja.Diagnostic) ![]const u8 {
        const t = self.template orelse return error.TemplateError;
        var msgs = messages;
        if (self.fold_system) msgs = try foldSystem(arena, msgs);
        if (self.content_parts) msgs = try contentParts(arena, msgs);
        var vars = std.ArrayList(jinja.Var).empty;
        try vars.appendSlice(arena, self.tokens);
        try vars.appendSlice(arena, &.{
            .{ .name = "messages", .value = msgs },
            .{ .name = "tools", .value = .none },
            .{ .name = "documents", .value = .none },
            .{ .name = "add_generation_prompt", .value = .{ .boolean = options.add_generation_prompt } },
        });
        try vars.appendSlice(arena, options.kwargs);
        return t.render(arena, vars.items, .{ .now = now_seconds }, diag);
    }
};

var warned_render_failure: std.atomic.Value(bool) = .init(false);

/// Wraps every string `content` as `[{"type": "text", "text": content}]`.
fn contentParts(a: Allocator, messages: jinja.Value) !jinja.Value {
    if (messages != .list) return messages;
    var out = std.ArrayList(jinja.Value).empty;
    for (messages.list.items.items) |m| {
        const content = if (m == .dict) m.dict.getStr("content") else null;
        if (content == null or content.? != .string) {
            try out.append(a, m);
            continue;
        }
        const part = try a.create(jinja.Dict);
        part.* = .{};
        try part.putStr(a, "type", .{ .string = "text" });
        try part.putStr(a, "text", content.?);
        const d = try a.create(jinja.Dict);
        d.* = .{};
        for (m.dict.keys.items, m.dict.values.items) |k, v| try d.put(a, k, v);
        try d.putStr(a, "content", try jinja.Value.listOf(a, &.{.{ .dict = part }}));
        try out.append(a, .{ .dict = d });
    }
    return jinja.Value.listOf(a, out.items);
}

/// Moves a leading system message into the first user message ("{system}\n\n{user}").
fn foldSystem(a: Allocator, messages: jinja.Value) !jinja.Value {
    if (messages != .list) return messages;
    const items = messages.list.items.items;
    if (items.len == 0 or items[0] != .dict) return messages;
    const role = items[0].dict.getStr("role") orelse return messages;
    if (role != .string or !std.mem.eql(u8, role.string, "system")) return messages;
    const system = items[0].dict.getStr("content") orelse return messages;
    if (system != .string) return messages;
    var out = std.ArrayList(jinja.Value).empty;
    var folded = false;
    for (items[1..]) |m| {
        if (!folded and m == .dict) if (m.dict.getStr("role")) |r| if (r == .string and std.mem.eql(u8, r.string, "user")) {
            const content = m.dict.getStr("content") orelse jinja.Value{ .string = "" };
            if (content == .string) {
                const d = try a.create(jinja.Dict);
                d.* = .{};
                for (m.dict.keys.items, m.dict.values.items) |k, v| try d.put(a, k, v);
                const merged = if (system.string.len > 0) try std.fmt.allocPrint(a, "{s}\n\n{s}", .{ system.string, content.string }) else content.string;
                try d.putStr(a, "content", .{ .string = merged });
                try out.append(a, .{ .dict = d });
                folded = true;
                continue;
            }
        };
        try out.append(a, m);
    }
    return jinja.Value.listOf(a, out.items);
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
    try std.testing.expectEqual(Template.gemma4, detect("{{- '<|turn>' + role + '\\n' }}<start_of_turn>", "gemma4_unified_text"));
    try std.testing.expectEqual(Template.gemma4, detect(null, "gemma4_unified"));
    const g4 = try renderPrompt(gpa, .gemma4, "Sys.", "Hi");
    defer gpa.free(g4);
    try std.testing.expectEqualStrings("<bos><|turn>system\nSys.<turn|>\n<|turn>user\nHi<turn|>\n<|turn>model\n<|channel>thought\n<channel|>", g4);
    // The rendering of google/gemma-4-12B-it's template for a four-message
    // conversation (transformers' apply_chat_template).
    const g4_multi = try render(gpa, .gemma4, &.{
        .{ .role = .system, .content = "You are a helpful assistant." },
        .{ .role = .user, .content = "Hi" },
        .{ .role = .assistant, .content = "<|channel>thought\nhmm<channel|>Hello! " },
        .{ .role = .user, .content = " Q" },
    });
    defer gpa.free(g4_multi);
    try std.testing.expectEqualStrings("<bos><|turn>system\nYou are a helpful assistant.<turn|>\n<|turn>user\nHi<turn|>\n<|turn>model\nHello!<turn|>\n<|turn>user\nQ<turn|>\n<|turn>model\n<|channel>thought\n<channel|>", g4_multi);
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
    // Apertus (swiss-ai/Apertus-8B-Instruct-2509; token-for-token against
    // apply_chat_template), which also has `<|user_start|>`.
    try std.testing.expectEqual(Template.apertus, detect("{%- set system_token = '<|system_start|>' -%}{%- set developer_token = '<|developer_start|>' -%}{%- set user_token = '<|user_start|>' -%}", "apertus"));
    const ap = try render(gpa, .apertus, &.{ .{ .role = .system, .content = "Sys." }, .{ .role = .user, .content = " Hi " }, .{ .role = .assistant, .content = "Yo" }, .{ .role = .user, .content = "Bye" } });
    defer gpa.free(ap);
    try std.testing.expectEqualStrings("<|system_start|>Sys.<|system_end|><|developer_start|>Deliberation: disabled\nTool Capabilities: disabled<|developer_end|><|user_start|> Hi <|user_end|><|assistant_start|>Yo<|assistant_end|><|user_start|>Bye<|user_end|><|assistant_start|>", ap);
    // The sweep's families: each expectation is transformers' own
    // apply_chat_template output for (system SYS, user U1), on the release
    // named, rendered on 2026-09-22.
    today = .{ .year = 2026, .month = 9, .day = 22 };
    defer today = .{};
    const expected = [_]struct { t: Template, want: []const u8 }{
        .{ .t = .seed, .want = "<seed:bos>system\nSYS<seed:eos><seed:bos>user\nU1<seed:eos><seed:bos>assistant\n" }, // ByteDance-Seed/Seed-OSS-36B-Instruct
        .{ .t = .minimax_m2, .want = "]~!b[]~b]system\nSYS[e~[\n]~b]user\nU1[e~[\n]~b]ai\n<think>\n" }, // MiniMaxAI/MiniMax-M2
        .{ .t = .nemotron_nano, .want = "<SPECIAL_10>System\nSYS\n<SPECIAL_11>User\nU1\n<SPECIAL_11>Assistant\n<think>\n" }, // nvidia/NVIDIA-Nemotron-Nano-9B-v2
        .{ .t = .k_exaone, .want = "<|system|>\nSYS<|endofturn|>\n<|user|>\nU1<|endofturn|>\n<|assistant|>\n<think>\n" }, // LGAI-EXAONE/K-EXAONE-236B-A23B
        .{ .t = .exaone4, .want = "[|system|]\nSYS[|endofturn|]\n[|user|]\nU1[|endofturn|]\n[|assistant|]\n<think>\n\n</think>\n\n" }, // LGAI-EXAONE/EXAONE-4.0-1.2B
        .{ .t = .harmony, .want = "<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.\nKnowledge cutoff: 2024-06\nCurrent date: 2026-09-22\n\nReasoning: medium\n\n# Valid channels: analysis, commentary, final. Channel must be included for every message.<|end|><|start|>developer<|message|># Instructions\n\nSYS\n\n<|end|><|start|>user<|message|>U1<|end|><|start|>assistant" }, // openai/gpt-oss-20b
        .{ .t = .smollm3, .want = "<|im_start|>system\n## Metadata\n\nKnowledge Cutoff Date: June 2025\nToday Date: 22 September 2026\nReasoning Mode: /think\n\n## Custom Instructions\n\nSYS\n\n<|im_start|>user\nU1<|im_end|>\n<|im_start|>assistant\n" }, // HuggingFaceTB/SmolLM3-3B
        .{ .t = .solar_open, .want = "<|begin|>system<|content|>## Provider System Prompt\n\nYou are Solar Open 100B, a large language model trained by Upstage AI, a Korean startup. Your knowledge cutoff is 2025-07. The current date is 2026-09-22.\n\n## System Prompt\n\nSYS<|end|><|begin|>user<|content|>U1<|end|><|begin|>assistant" }, // upstage/Solar-Open-100B
        .{ .t = .qwen3_5, .want = "<|im_start|>system\nSYS<|im_end|>\n<|im_start|>user\nU1<|im_end|>\n<|im_start|>assistant\n<think>\n" }, // Qwen/Qwen3.5-397B-A17B
        .{ .t = .mimo, .want = "<|im_start|>system\nSYS<|im_end|><|im_start|>user\nU1<|im_end|><|im_start|>assistant\n<think></think>" }, // XiaomiMiMo/MiMo-V2-Flash
        .{ .t = .deepseek_v31, .want = "<\u{ff5c}begin\u{2581}of\u{2581}sentence\u{ff5c}>SYS<\u{ff5c}User\u{ff5c}>U1<\u{ff5c}Assistant\u{ff5c}></think>" }, // deepseek-ai/DeepSeek-V3.1
        .{ .t = .glm47, .want = "[gMASK]<sop><|system|>SYS<|user|>U1<|assistant|><think>" }, // zai-org/GLM-4.7-Flash
        .{ .t = .jamba_reasoning, .want = "<|startoftext|><|im_start|>system\nSYS<|im_end|>\n<|im_start|>user\nBegin by thinking about the reasoning process in the mind within <think> </think> tags and then proceed to give your response.\nU1<|im_end|>\n<|im_start|>assistant\n<think>\n" }, // ai21labs/AI21-Jamba-Reasoning-3B
        .{ .t = .phi4, .want = "<|im_start|>system<|im_sep|>SYS<|im_end|><|im_start|>user<|im_sep|>U1<|im_end|><|im_start|>assistant<|im_sep|>" }, // microsoft/phi-4
        .{ .t = .phi4_mini, .want = "<|system|>SYS<|end|><|user|>U1<|end|><|assistant|>" }, // microsoft/Phi-4-mini-instruct
        .{ .t = .dots, .want = "<|system|>SYS<|endofsystem|><|userprompt|>U1<|endofuserprompt|><|response|>" }, // rednote-hilab/dots.llm1.inst
        .{ .t = .falcon, .want = "SYS\n\nUser: U1\n\nAssistant:" }, // tiiuae/falcon-7b-instruct
        .{ .t = .llama32, .want = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: 22 Sep 2026\n\nSYS<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nU1<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n" }, // unsloth/Llama-3.2-1B-Instruct
        .{ .t = .llama31, .want = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\nSYS<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nU1<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n" }, // unsloth/Meta-Llama-3.1-8B-Instruct, unsloth/Llama-3.3-70B-Instruct
        .{ .t = .hunyuan_moe, .want = "<|startoftext|>SYS<|extra_4|>U1<|extra_0|>" }, // tencent/Hunyuan-A13B-Instruct
        .{ .t = .nemotron_mini, .want = "<extra_id_0>System\nSYS\n\n<extra_id_1>User\nU1\n<extra_id_1>Assistant\n" }, // nvidia/Nemotron-Mini-4B-Instruct
        .{ .t = .deepseek_v2, .want = "<\u{ff5c}begin\u{2581}of\u{2581}sentence\u{ff5c}>SYS\n\nUser: U1\n\nAssistant:" }, // deepseek-ai/DeepSeek-V2-Lite-Chat
        .{ .t = .mistral_spaced, .want = "<s> [INST] SYS\n\nU1 [/INST]" }, // mistralai/Mixtral-8x7B-Instruct-v0.1
    };
    for (expected) |e| {
        const got = try renderPrompt(gpa, e.t, "SYS", "U1");
        defer gpa.free(got);
        std.testing.expectEqualStrings(e.want, got) catch |err| {
            std.debug.print("template {s}\n", .{@tagName(e.t)});
            return err;
        };
    }
    // Multi-turn shapes (transformers, SYS / U1 / A1 / U2).
    const turns = [_]Message{ .{ .role = .system, .content = "SYS" }, .{ .role = .user, .content = "U1" }, .{ .role = .assistant, .content = "A1" }, .{ .role = .user, .content = "U2" } };
    const multi = [_]struct { t: Template, want: []const u8 }{
        .{ .t = .nemotron_nano, .want = "<SPECIAL_10>System\nSYS\n<SPECIAL_11>User\nU1\n<SPECIAL_11>Assistant\nA1\n<SPECIAL_12>\n<SPECIAL_11>User\nU2\n<SPECIAL_11>Assistant\n<think>\n" },
        .{ .t = .k_exaone, .want = "<|system|>\nSYS<|endofturn|>\n<|user|>\nU1<|endofturn|>\n<|assistant|>\n<think>\n\n</think>\n\nA1<|endofturn|>\n<|user|>\nU2<|endofturn|>\n<|assistant|>\n<think>\n" },
        .{ .t = .deepseek_v31, .want = "<\u{ff5c}begin\u{2581}of\u{2581}sentence\u{ff5c}>SYS<\u{ff5c}User\u{ff5c}>U1<\u{ff5c}Assistant\u{ff5c}></think>A1<\u{ff5c}end\u{2581}of\u{2581}sentence\u{ff5c}><\u{ff5c}User\u{ff5c}>U2<\u{ff5c}Assistant\u{ff5c}></think>" },
        .{ .t = .glm47, .want = "[gMASK]<sop><|system|>SYS<|user|>U1<|assistant|></think>A1<|user|>U2<|assistant|><think>" },
        .{ .t = .jamba_reasoning, .want = "<|startoftext|><|im_start|>system\nSYS<|im_end|>\n<|im_start|>user\nU1<|im_end|>\n<|im_start|>assistant\nA1<|im_end|>\n<|im_start|>user\nBegin by thinking about the reasoning process in the mind within <think> </think> tags and then proceed to give your response.\nU2<|im_end|>\n<|im_start|>assistant\n<think>\n" },
        .{ .t = .harmony, .want = "<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.\nKnowledge cutoff: 2024-06\nCurrent date: 2026-09-22\n\nReasoning: medium\n\n# Valid channels: analysis, commentary, final. Channel must be included for every message.<|end|><|start|>developer<|message|># Instructions\n\nSYS\n\n<|end|><|start|>user<|message|>U1<|end|><|start|>assistant<|channel|>final<|message|>A1<|end|><|start|>user<|message|>U2<|end|><|start|>assistant" },
    };
    for (multi) |e| {
        const got = try render(gpa, e.t, &turns);
        defer gpa.free(got);
        std.testing.expectEqualStrings(e.want, got) catch |err| {
            std.debug.print("template {s} (multi-turn)\n", .{@tagName(e.t)});
            return err;
        };
    }
    try std.testing.expectEqual(Date{ .year = 2026, .month = 9, .day = 22 }, Date.fromUnix(1790035200));
    // A template that emits the BOS in front of a family that does not.
    try std.testing.expectEqualStrings("<|begin_of_text|>", templateBos("{{bos_token}}\n{%- if tools %}<|im_start|>", "<|begin_of_text|>"));
    try std.testing.expectEqualStrings("", templateBos("{% for m in messages %}<|im_start|>{{ bos_token }}", "<|begin_of_text|>"));
    try std.testing.expectEqualStrings("", templateBos("{{bos_token}}<|im_start|>", null));
}

test "falcon template: detection from the release, stripped and collapsed contents" {
    const gpa = std.testing.allocator;
    try std.testing.expectEqual(Template.falcon, detect("{% for message in loop_messages %}{% if loop.index0 == 0 %}{{ system_message.strip() }}{% endif %}{% if message['role'] == 'user' %}{{ '\n\nUser: ' + message['content'].strip().replace('\r\n', '\n').replace('\n\n', '\n') }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ '\n\nAssistant:' }}{% endif %}", "falcon"));
    const out = try render(gpa, .falcon, &.{ .{ .role = .system, .content = " S " }, .{ .role = .user, .content = " a\r\n\r\nb\n\n\nc " }, .{ .role = .assistant, .content = "x" }, .{ .role = .user, .content = "y" } });
    defer gpa.free(out);
    // Python: 'a\r\n\r\nb\n\n\nc' -> 'a\n\nb\n\n\nc' -> 'a\nb\n\nc'
    try std.testing.expectEqualStrings("S\n\nUser: a\nb\n\nc\n\nAssistant: x\n\nUser: y\n\nAssistant:", out);
}

test "the model's own template: where it comes from and its special tokens" {
    var arena_state: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    const cfg =
        \\{"chat_template": [{"name": "tool_use", "template": "T"}, {"name": "default", "template": "D"}],
        \\ "bos_token": {"content": "<s>", "lstrip": false}, "eos_token": "</s>", "pad_token": null,
        \\ "image_token": "<img>", "extra_special_tokens": {"audio_token": "<aud>"}}
    ;
    try std.testing.expectEqualStrings("J", (try pickTemplate(a, cfg, "J", null)).?);
    try std.testing.expectEqualStrings("D", (try pickTemplate(a, cfg, null, "{\"chat_template\": \"P\"}")).?);
    try std.testing.expectEqualStrings("P", (try pickTemplate(a, "{}", null, "{\"chat_template\": \"P\"}")).?);
    try std.testing.expectEqual(null, try pickTemplate(a, null, null, null));

    const toks = try specialTokens(a, cfg, "{\"bos_token\": \"<B>\"}");
    const want = [_][2][]const u8{ .{ "bos_token", "<B>" }, .{ "eos_token", "</s>" }, .{ "image_token", "<img>" }, .{ "audio_token", "<aud>" } };
    try std.testing.expectEqual(want.len, toks.len);
    for (want) |w| {
        var found = false;
        for (toks) |t| if (std.mem.eql(u8, t.name, w[0])) {
            try std.testing.expectEqualStrings(w[1], t.value.string);
            found = true;
        };
        try std.testing.expect(found);
    }
    // A config with `added_tokens_decoder` is not overridden by special_tokens_map.json.
    const modern = try specialTokens(a, "{\"bos_token\": \"<s>\", \"added_tokens_decoder\": {}}", "{\"bos_token\": \"<B>\"}");
    try std.testing.expectEqualStrings("<s>", modern[0].value.string);
}

test "the model's own template: rendering, refusals and the fallback" {
    const gpa = std.testing.allocator;
    const tokens = [_]jinja.Var{.{ .name = "bos_token", .value = .{ .string = "<s>" } }};
    const chatml_src = "{{ bos_token }}{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}";
    var own = try Format.init(gpa, chatml_src, &tokens, .chatml);
    defer own.deinit();
    try std.testing.expectEqualStrings("model", own.name());
    const p = try own.prompt(gpa, "Sys.", "Hi");
    defer gpa.free(p);
    try std.testing.expectEqualStrings("<s><|im_start|>system\nSys.<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n", p);

    // Gemma 2 raises on a system message: it is folded into the user turn.
    const gemma2_src = "{{ bos_token }}{% if messages[0]['role'] == 'system' %}{{ raise_exception('System role not supported') }}{% endif %}{% for m in messages %}<start_of_turn>{{ m.role }}\n{{ m.content | trim }}<end_of_turn>\n{% endfor %}<start_of_turn>model\n";
    var folded = try Format.init(gpa, gemma2_src, &tokens, .gemma);
    defer folded.deinit();
    try std.testing.expect(folded.fold_system);
    const g = try folded.prompt(gpa, "Sys.", "Hi");
    defer gpa.free(g);
    // The final line break of a template is dropped (keep_trailing_newline=False).
    try std.testing.expectEqualStrings("<s><start_of_turn>user\nSys.\n\nHi<end_of_turn>\n<start_of_turn>model", g);

    // MiniMax-Text-01 reads contents as lists of parts.
    const parts_src = "{% for m in messages %}{{ m.role }}:{% for p in m.content %}{{ p.text + ';' }}{% endfor %}{% endfor %}";
    var parts = try Format.init(gpa, parts_src, &tokens, .raw);
    defer parts.deinit();
    try std.testing.expect(parts.content_parts and !parts.fold_system);
    const mp = try parts.prompt(gpa, "Sys.", "Hi");
    defer gpa.free(mp);
    try std.testing.expectEqualStrings("system:Sys.;user:Hi;", mp);

    // A template that cannot be parsed (or never renders) falls back to the family.
    var broken = try Format.init(gpa, "{% if %}", &tokens, .chatml);
    defer broken.deinit();
    try std.testing.expect(broken.template == null);
    try std.testing.expectEqualStrings("chatml", broken.name());
    const b = try broken.prompt(gpa, "Sys.", "Hi");
    defer gpa.free(b);
    try std.testing.expectEqualStrings("<|im_start|>system\nSys.<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n", b);
}

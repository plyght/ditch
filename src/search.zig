//! The optimisation search space: how a TPE parameter vector maps to an
//! abliteration configuration, and how that configuration is applied.
//!
//! Dense models expose the heretic parameters (direction scope/index and a
//! weight kernel per component). MoE models additionally expose the
//! expert-selection parameters (see `moe.zig`).

const std = @import("std");
const tpe = @import("tpe.zig");
const abliterate = @import("abliterate.zig");
const model_mod = @import("model.zig");

const Allocator = std.mem.Allocator;
const Model = model_mod.Model;
const Component = model_mod.Component;

/// Fully decoded trial configuration.
pub const TrialConfig = struct {
    /// null = "per layer" direction scope.
    direction_index: ?f32,
    parameters: std.EnumMap(Component, abliterate.Params),
    /// Expert selection for MoE models (null = broad edit of every expert).
    experts: ?ExpertSelection = null,
};

/// Placeholder contract for expert-selective abliteration; filled in by moe.zig.
pub const ExpertSelection = struct {
    /// Number of experts per layer that receive the edit (0 = all experts, i.e. broad edit).
    n_experts: usize,
    /// Multiplier applied to the MLP kernel weight for selected experts.
    strength: f32,
};

pub const Space = struct {
    arena: std.heap.ArenaAllocator,
    space: tpe.Space,
    n_layers: usize,
    components: []Component,
    is_moe: bool,

    pub fn deinit(self: *Space) void {
        self.arena.deinit();
    }

    pub fn dims(self: *const Space) usize {
        return self.space.specs.len;
    }
};

/// Builds the parameter space for `model`, mirroring heretic's ranges.
pub fn buildSpace(gpa: Allocator, model: *const Model) !Space {
    var arena = std.heap.ArenaAllocator.init(gpa);
    errdefer arena.deinit();
    const a = arena.allocator();
    const last: f64 = @floatFromInt(model.config.num_layers - 1);
    var names = std.ArrayList([]const u8).empty;
    var specs = std.ArrayList(tpe.ParamSpec).empty;
    try names.append(a, "direction_scope");
    try specs.append(a, .{ .categorical = .{ .n = 2 } }); // 0 = global, 1 = per layer
    try names.append(a, "direction_index");
    try specs.append(a, .{ .float = .{ .low = 0.4 * last, .high = 0.9 * last } });
    const components = try a.dupe(Component, &Component.all);
    for (components) |comp| {
        const lower: f64 = if (comp == .mlp_down_proj) -0.25 else 0.8;
        try names.append(a, try std.fmt.allocPrint(a, "{s}.max_weight", .{comp.name()}));
        try specs.append(a, .{ .float = .{ .low = lower, .high = 1.5 } });
        try names.append(a, try std.fmt.allocPrint(a, "{s}.max_weight_position", .{comp.name()}));
        try specs.append(a, .{ .float = .{ .low = 0.6 * last, .high = 1.0 * last } });
        try names.append(a, try std.fmt.allocPrint(a, "{s}.min_weight", .{comp.name()}));
        try specs.append(a, .{ .float = .{ .low = 0.0, .high = 1.0 } });
        try names.append(a, try std.fmt.allocPrint(a, "{s}.min_weight_distance", .{comp.name()}));
        try specs.append(a, .{ .float = .{ .low = 1.0, .high = @max(0.6 * last, 1.0) } });
    }
    const is_moe = model.isMoe();
    if (is_moe) {
        try names.append(a, "experts.n_selected");
        try specs.append(a, .{ .float = .{ .low = 0.0, .high = @floatFromInt(model.numExpertsPerLayer()) } });
        try names.append(a, "experts.strength");
        try specs.append(a, .{ .float = .{ .low = 0.5, .high = 1.5 } });
    }
    return .{
        .arena = arena,
        .space = .{ .names = names.items, .specs = specs.items },
        .n_layers = model.config.num_layers,
        .components = components,
        .is_moe = is_moe,
    };
}

/// Decodes a sampled parameter vector into a trial configuration.
pub fn decode(space: *const Space, vector: []const f64) TrialConfig {
    var i: usize = 0;
    const scope: usize = @intFromFloat(vector[i]);
    i += 1;
    const direction_index: f32 = @floatCast(vector[i]);
    i += 1;
    var params = std.EnumMap(Component, abliterate.Params){};
    for (space.components) |comp| {
        const max_weight: f32 = @floatCast(@max(0.0, vector[i]));
        const max_pos: f32 = @floatCast(vector[i + 1]);
        const min_frac: f32 = @floatCast(vector[i + 2]);
        const min_dist: f32 = @floatCast(vector[i + 3]);
        i += 4;
        params.put(comp, .{
            .max_weight = max_weight,
            .max_weight_position = max_pos,
            .min_weight = min_frac * max_weight,
            .min_weight_distance = min_dist,
        });
    }
    var experts: ?ExpertSelection = null;
    if (space.is_moe) {
        experts = .{ .n_experts = @intFromFloat(@floor(vector[i])), .strength = @floatCast(vector[i + 1]) };
        i += 2;
    }
    return .{
        .direction_index = if (scope == 1) null else direction_index,
        .parameters = params,
        .experts = experts,
    };
}

/// Human-readable "name = value" lines for a parameter vector.
pub fn describe(space: *const Space, vector: []const f64, out: *std.Io.Writer) !void {
    for (space.space.names, 0..) |name, i| {
        switch (space.space.specs[i]) {
            .categorical => {
                const v: usize = @intFromFloat(vector[i]);
                if (std.mem.eql(u8, name, "direction_scope")) {
                    try out.print("  * {s} = {s}\n", .{ name, if (v == 1) "per layer" else "global" });
                } else try out.print("  * {s} = {d}\n", .{ name, v });
            },
            .float => try out.print("  * {s} = {d:.4}\n", .{ name, vector[i] }),
        }
    }
}

/// Applies a decoded trial configuration to the model.
pub fn applyTrial(model: *Model, dirs: []const f32, cfg: TrialConfig, opts: abliterate.Options) !void {
    if (cfg.experts) |sel| {
        if (sel.n_experts > 0) {
            return model_mod.applyExpertSelective(model, dirs, cfg, opts);
        }
    }
    try abliterate.apply(model, dirs, cfg.direction_index, cfg.parameters, opts);
}

test "space round trip (dense)" {
    const gpa = std.testing.allocator;
    const pool = @import("tensor.zig").Pool.init(std.testing.io, 1);
    const model = try Model.load(gpa, std.testing.io, &pool, "tests/fixtures/llama");
    defer model.deinit();
    var space = try buildSpace(gpa, model);
    defer space.deinit();
    try std.testing.expectEqual(@as(usize, 2 + 4 * 2), space.dims());
    var v = [_]f64{ 0, 1.2, 1.0, 2.0, 0.5, 1.0, -0.1, 2.0, 0.5, 1.0 };
    const cfg = decode(&space, &v);
    try std.testing.expectEqual(@as(?f32, 1.2), cfg.direction_index);
    try std.testing.expectEqual(@as(f32, 0.5), cfg.parameters.get(.attn_o_proj).?.min_weight);
    try std.testing.expectEqual(@as(f32, 0.0), cfg.parameters.get(.mlp_down_proj).?.max_weight);
    v[0] = 1;
    try std.testing.expectEqual(@as(?f32, null), decode(&space, &v).direction_index);
}

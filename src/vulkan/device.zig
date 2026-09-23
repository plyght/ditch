//! The Vulkan device under the compute backend: loading Vulkan, choosing a
//! physical device, memory types, buffers and the one synchronous
//! submission path every kernel call goes through. The kernels themselves and
//! the `compute.VTable` are in `backend.zig`.
//!
//! Device choice: a discrete GPU first, then an integrated one (then virtual
//! and other types), the one with the largest device-local heap among equals.
//! A software (CPU-type) device such as Mesa's lavapipe is taken only when
//! asked for explicitly (`--device vulkan`), since the CPU backend beats it.
//! `DITCH_VULKAN_DEVICE=<index>` picks a device by its position in Vulkan's
//! enumeration instead.
//!
//! Memory
//! ------
//! * Discrete GPUs: every buffer a kernel reads or writes is device-local.
//!   Inputs are written into one host-visible staging buffer and copied over
//!   in the same command buffer as the dispatch; outputs come back through a
//!   host-visible (cached where possible) read-back buffer.
//! * Integrated GPUs and software devices ("unified memory"): when a memory
//!   type is device-local and host-visible at once, kernel buffers are
//!   allocated there and written and read in place, with no staging copies.
//!   `DITCH_VULKAN_STAGING=1` forces the discrete path on such a device,
//!   which is how CI tests that path without a discrete GPU.
//!
//! Allocation failures surface as `error.Unsupported`, so a tile the device
//! has no room for is computed by the CPU instead of failing the run.

const std = @import("std");
const builtin = @import("builtin");
const compute = @import("../compute.zig");
const vk = @import("vk.zig");

const Allocator = std.mem.Allocator;
const Error = compute.Error;

/// Storage buffers a kernel may bind (the descriptor set layout has this
/// many bindings, and a kernel uses the first few).
pub const max_bindings = 4;
/// Push constant bytes every pipeline layout reserves.
pub const push_bytes = 32;
/// Pipelines a context caches (at least the backend's kernel count).
pub const max_pipelines = 32;

/// The least a device must offer for the kernels in `backend.zig`: 256
/// invocations for the row reductions, 16x16 for the matmul tiles, and the
/// matmul's two 16x16 f32 tiles of shared memory.
const min_invocations = 256;
const min_workgroup_y = 16;
const min_shared_bytes = 2 * 16 * 16 * 4;

// ---------------------------------------------------------------------------
// Why a device is unavailable, for the one line ditch prints
// ---------------------------------------------------------------------------

var reason_buf: [256]u8 = undefined;
var reason_len: usize = 0;

/// Why the last `open` failed ("" when it did not).
pub fn lastError() []const u8 {
    return reason_buf[0..reason_len];
}

pub fn fail(comptime fmt: []const u8, args: anytype) error{DeviceUnavailable} {
    const s = std.fmt.bufPrint(&reason_buf, fmt, args) catch reason_buf[0..];
    reason_len = s.len;
    return error.DeviceUnavailable;
}

fn getenv(name: [*:0]const u8) ?[]const u8 {
    if (!builtin.link_libc) return null;
    const v = std.c.getenv(name) orelse return null;
    return std.mem.span(v);
}

// ---------------------------------------------------------------------------
// Buffers and submission
// ---------------------------------------------------------------------------

pub const Buf = struct {
    buffer: vk.Buffer = vk.null_handle,
    memory: vk.DeviceMemory = vk.null_handle,
    size: u64 = 0,
    /// Host address of the buffer when its memory is host-visible.
    map: ?[*]u8 = null,
};

pub const Use = enum {
    /// Read and written by kernels (device-local; also host-visible in
    /// unified-memory mode).
    kernel,
    /// Host-to-device staging.
    upload,
    /// Device-to-host read-back.
    readback,
};

pub const Context = struct {
    gpa: Allocator,
    loader: vk.Loader,
    instance: vk.Instance,
    ifn: vk.InstanceFns,
    phys: vk.PhysicalDevice,
    props: vk.PhysicalDeviceProperties,
    mem: vk.PhysicalDeviceMemoryProperties,
    device: vk.Device,
    dfn: vk.DeviceFns,
    queue: vk.Queue,
    family: u32,
    /// Unified memory: kernel buffers are host-visible and written in place.
    direct: bool,

    cmd_pool: vk.CommandPool = vk.null_handle,
    cmd: vk.CommandBuffer = undefined,
    fence: vk.Fence = vk.null_handle,
    set_layout: vk.DescriptorSetLayout = vk.null_handle,
    pipe_layout: vk.PipelineLayout = vk.null_handle,
    desc_pool: vk.DescriptorPool = vk.null_handle,
    desc_set: vk.DescriptorSet = vk.null_handle,
    /// Built on first use (0 = not yet).
    pipelines: [max_pipelines]vk.Pipeline = [_]vk.Pipeline{vk.null_handle} ** max_pipelines,
    /// Staging (discrete GPUs only).
    up: Buf = .{},
    down: Buf = .{},

    pub fn check(r: vk.Result) Error!void {
        if (r == vk.SUCCESS) return;
        if (r == vk.ERROR_OUT_OF_DEVICE_MEMORY or r == vk.ERROR_OUT_OF_HOST_MEMORY) return error.Unsupported;
        return error.DeviceFailed;
    }

    pub fn memoryType(self: *const Context, bits: u32, required: vk.Flags, preferred: vk.Flags) ?u32 {
        for ([_]vk.Flags{ required | preferred, required }) |want| {
            for (0..self.mem.memoryTypeCount) |i| {
                if (bits & (@as(u32, 1) << @intCast(i)) == 0) continue;
                if (self.mem.memoryTypes[i].propertyFlags & want == want) return @intCast(i);
            }
        }
        return null;
    }

    pub fn newBuf(self: *Context, size: u64, use: Use) Error!Buf {
        const d = self.device;
        const usage: vk.Flags = switch (use) {
            .kernel => vk.BUFFER_USAGE_STORAGE_BUFFER_BIT | vk.BUFFER_USAGE_TRANSFER_SRC_BIT | vk.BUFFER_USAGE_TRANSFER_DST_BIT,
            .upload => vk.BUFFER_USAGE_TRANSFER_SRC_BIT,
            .readback => vk.BUFFER_USAGE_TRANSFER_DST_BIT,
        };
        var b = Buf{ .size = size };
        try check(self.dfn.vkCreateBuffer(d, &.{ .size = size, .usage = usage }, null, &b.buffer));
        errdefer self.dfn.vkDestroyBuffer(d, b.buffer, null);
        var req: vk.MemoryRequirements = undefined;
        self.dfn.vkGetBufferMemoryRequirements(d, b.buffer, &req);
        const host = vk.MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk.MEMORY_PROPERTY_HOST_COHERENT_BIT;
        const type_index = switch (use) {
            .kernel => if (self.direct)
                self.memoryType(req.memoryTypeBits, host, vk.MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
            else
                self.memoryType(req.memoryTypeBits, vk.MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0),
            .upload => self.memoryType(req.memoryTypeBits, host, 0),
            .readback => self.memoryType(req.memoryTypeBits, host, vk.MEMORY_PROPERTY_HOST_CACHED_BIT),
        } orelse return error.Unsupported;
        try check(self.dfn.vkAllocateMemory(d, &.{ .allocationSize = req.size, .memoryTypeIndex = type_index }, null, &b.memory));
        errdefer self.dfn.vkFreeMemory(d, b.memory, null);
        try check(self.dfn.vkBindBufferMemory(d, b.buffer, b.memory, 0));
        if (self.mem.memoryTypes[type_index].propertyFlags & vk.MEMORY_PROPERTY_HOST_VISIBLE_BIT != 0) {
            var p: ?*anyopaque = null;
            try check(self.dfn.vkMapMemory(d, b.memory, 0, vk.whole_size, 0, &p));
            b.map = @ptrCast(p orelse return error.DeviceFailed);
        } else if (use != .kernel) return error.DeviceFailed;
        return b;
    }

    pub fn freeBuf(self: *Context, b: *Buf) void {
        if (b.buffer == vk.null_handle) return;
        if (b.map != null) self.dfn.vkUnmapMemory(self.device, b.memory);
        self.dfn.vkDestroyBuffer(self.device, b.buffer, null);
        self.dfn.vkFreeMemory(self.device, b.memory, null);
        b.* = .{};
    }

    /// Grows `b` to at least `bytes` (at least 64 KiB, with a quarter of
    /// headroom so a slowly growing input does not reallocate every call),
    /// dropping its contents.
    pub fn ensure(self: *Context, b: *Buf, bytes: u64, use: Use) Error!void {
        if (b.buffer != vk.null_handle and b.size >= bytes) return;
        const want = std.mem.alignForward(u64, @max(bytes + bytes / 4, 64 * 1024), 256);
        self.freeBuf(b);
        b.* = self.newBuf(want, use) catch |e| {
            // Doubling past what the device can give: try the exact size.
            if (want == bytes) return e;
            b.* = try self.newBuf(std.mem.alignForward(u64, bytes, 256), use);
            return;
        };
    }

    /// Whether one binding of `bytes` is addressable by a kernel.
    pub fn fits(self: *const Context, bytes: u64) Error!void {
        if (std.mem.alignForward(u64, bytes, 4) > self.props.limits.maxStorageBufferRange) return error.Unsupported;
    }

    /// The compute pipeline for SPIR-V module `code`, built on first use and
    /// cached in slot `index`.
    pub fn pipeline(self: *Context, index: usize, code: []const u8) Error!vk.Pipeline {
        if (self.pipelines[index] != vk.null_handle) return self.pipelines[index];
        // SPIR-V must be handed over 4-byte aligned; @embedFile is not.
        const words = try self.gpa.alloc(u32, code.len / 4);
        defer self.gpa.free(words);
        @memcpy(std.mem.sliceAsBytes(words), code[0 .. words.len * 4]);
        var module: vk.ShaderModule = vk.null_handle;
        try check(self.dfn.vkCreateShaderModule(self.device, &.{ .codeSize = words.len * 4, .pCode = words.ptr }, null, &module));
        defer self.dfn.vkDestroyShaderModule(self.device, module, null);
        const info = vk.ComputePipelineCreateInfo{
            .stage = .{ .stage = vk.SHADER_STAGE_COMPUTE_BIT, .module = module, .pName = "main" },
            .layout = self.pipe_layout,
        };
        var p: vk.Pipeline = vk.null_handle;
        try check(self.dfn.vkCreateComputePipelines(self.device, vk.null_handle, 1, @ptrCast(&info), null, @ptrCast(&p)));
        self.pipelines[index] = p;
        return p;
    }

    /// One kernel binding: a buffer, the bytes the kernel may address, and
    /// optionally host bytes to copy into it first.
    pub const Bind = struct { buf: *Buf, bytes: u64, upload: ?[]const u8 = null };
    /// A buffer to copy back into host memory after the dispatch.
    pub const Read = struct { buf: *Buf, dst: []u8 };

    /// Uploads, one dispatch, read-backs: one command buffer, submitted and
    /// waited for.
    pub fn run(self: *Context, pipe: vk.Pipeline, binds: []const Bind, params: []const u8, groups: [3]u32, reads: []const Read) Error!void {
        const lim = self.props.limits.maxComputeWorkGroupCount;
        for (groups, lim) |g, l| if (g == 0 or g > l) return error.Unsupported;
        for (binds) |b| try self.fits(b.bytes);
        const f = &self.dfn;

        // Host side of the uploads.
        if (self.direct) {
            for (binds) |b| if (b.upload) |src| if (src.len > 0) @memcpy(b.buf.map.?[0..src.len], src);
        } else {
            var total: u64 = 0;
            for (binds) |b| if (b.upload) |src| {
                total += alignUp(src.len);
            };
            if (total > 0) {
                try self.ensure(&self.up, total, .upload);
                var off: u64 = 0;
                for (binds) |b| if (b.upload) |src| {
                    @memcpy(self.up.map.?[off..][0..src.len], src);
                    off += alignUp(src.len);
                };
            }
            var down: u64 = 0;
            for (reads) |r| down += alignUp(r.dst.len);
            if (down > 0) try self.ensure(&self.down, down, .readback);
        }

        // Descriptors, before recording (updating a bound set would
        // invalidate the command buffer).
        var infos: [max_bindings]vk.DescriptorBufferInfo = undefined;
        var writes: [max_bindings]vk.WriteDescriptorSet = undefined;
        for (binds, 0..) |b, i| {
            infos[i] = .{ .buffer = b.buf.buffer, .offset = 0, .range = std.mem.alignForward(u64, @max(b.bytes, 4), 4) };
            writes[i] = .{
                .dstSet = self.desc_set,
                .dstBinding = @intCast(i),
                .descriptorCount = 1,
                .descriptorType = vk.DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .pBufferInfo = @ptrCast(&infos[i]),
            };
        }
        f.vkUpdateDescriptorSets(self.device, @intCast(binds.len), &writes, 0, null);

        const cmd = self.cmd;
        try check(f.vkResetCommandBuffer(cmd, 0));
        try check(f.vkBeginCommandBuffer(cmd, &.{ .flags = vk.COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT }));
        const all_stages = vk.PIPELINE_STAGE_COMPUTE_SHADER_BIT | vk.PIPELINE_STAGE_TRANSFER_BIT;
        // Earlier submissions wrote the same scratch buffers.
        barrier(f, cmd, all_stages, all_stages, vk.ACCESS_SHADER_WRITE_BIT | vk.ACCESS_TRANSFER_WRITE_BIT, vk.ACCESS_SHADER_READ_BIT | vk.ACCESS_SHADER_WRITE_BIT | vk.ACCESS_TRANSFER_READ_BIT | vk.ACCESS_TRANSFER_WRITE_BIT);
        if (!self.direct) {
            var off: u64 = 0;
            var any = false;
            for (binds) |b| if (b.upload) |src| {
                if (src.len > 0) {
                    const region = vk.BufferCopy{ .srcOffset = off, .dstOffset = 0, .size = src.len };
                    f.vkCmdCopyBuffer(cmd, self.up.buffer, b.buf.buffer, 1, @ptrCast(&region));
                    any = true;
                }
                off += alignUp(src.len);
            };
            if (any) barrier(f, cmd, vk.PIPELINE_STAGE_TRANSFER_BIT, vk.PIPELINE_STAGE_COMPUTE_SHADER_BIT, vk.ACCESS_TRANSFER_WRITE_BIT, vk.ACCESS_SHADER_READ_BIT | vk.ACCESS_SHADER_WRITE_BIT);
        } else {
            barrier(f, cmd, vk.PIPELINE_STAGE_HOST_BIT, vk.PIPELINE_STAGE_COMPUTE_SHADER_BIT, vk.ACCESS_HOST_WRITE_BIT, vk.ACCESS_SHADER_READ_BIT | vk.ACCESS_SHADER_WRITE_BIT);
        }
        f.vkCmdBindPipeline(cmd, vk.PIPELINE_BIND_POINT_COMPUTE, pipe);
        f.vkCmdBindDescriptorSets(cmd, vk.PIPELINE_BIND_POINT_COMPUTE, self.pipe_layout, 0, 1, @ptrCast(&self.desc_set), 0, null);
        var pc: [push_bytes]u8 = [_]u8{0} ** push_bytes;
        @memcpy(pc[0..params.len], params);
        f.vkCmdPushConstants(cmd, self.pipe_layout, vk.SHADER_STAGE_COMPUTE_BIT, 0, push_bytes, &pc);
        f.vkCmdDispatch(cmd, groups[0], groups[1], groups[2]);
        if (!self.direct) {
            if (reads.len > 0) {
                barrier(f, cmd, vk.PIPELINE_STAGE_COMPUTE_SHADER_BIT, vk.PIPELINE_STAGE_TRANSFER_BIT, vk.ACCESS_SHADER_WRITE_BIT, vk.ACCESS_TRANSFER_READ_BIT);
                var off: u64 = 0;
                for (reads) |r| {
                    if (r.dst.len > 0) {
                        const region = vk.BufferCopy{ .srcOffset = 0, .dstOffset = off, .size = r.dst.len };
                        f.vkCmdCopyBuffer(cmd, r.buf.buffer, self.down.buffer, 1, @ptrCast(&region));
                    }
                    off += alignUp(r.dst.len);
                }
                barrier(f, cmd, vk.PIPELINE_STAGE_TRANSFER_BIT, vk.PIPELINE_STAGE_HOST_BIT, vk.ACCESS_TRANSFER_WRITE_BIT, vk.ACCESS_HOST_READ_BIT);
            }
        } else {
            barrier(f, cmd, vk.PIPELINE_STAGE_COMPUTE_SHADER_BIT, vk.PIPELINE_STAGE_HOST_BIT, vk.ACCESS_SHADER_WRITE_BIT, vk.ACCESS_HOST_READ_BIT);
        }
        try check(f.vkEndCommandBuffer(cmd));

        const submit = vk.SubmitInfo{ .commandBufferCount = 1, .pCommandBuffers = @ptrCast(&self.cmd) };
        try check(f.vkQueueSubmit(self.queue, 1, @ptrCast(&submit), self.fence));
        const waited = f.vkWaitForFences(self.device, 1, @ptrCast(&self.fence), 1, std.math.maxInt(u64));
        try check(f.vkResetFences(self.device, 1, @ptrCast(&self.fence)));
        if (waited != vk.SUCCESS) return error.DeviceFailed;

        // Host side of the read-backs.
        if (self.direct) {
            for (reads) |r| if (r.dst.len > 0) @memcpy(r.dst, r.buf.map.?[0..r.dst.len]);
        } else {
            var off: u64 = 0;
            for (reads) |r| {
                if (r.dst.len > 0) @memcpy(r.dst, self.down.map.?[off..][0..r.dst.len]);
                off += alignUp(r.dst.len);
            }
        }
    }

    /// A row-per-workgroup grid for `rows` rows: rows beyond the first
    /// dimension's limit continue in the second (the shaders number rows as
    /// `y * width + x`).
    pub fn rowGrid(self: *const Context, rows: usize) Error![3]u32 {
        const max_x: usize = self.props.limits.maxComputeWorkGroupCount[0];
        const gx = @min(rows, max_x);
        const gy = (rows + gx - 1) / gx;
        if (gy > self.props.limits.maxComputeWorkGroupCount[1]) return error.Unsupported;
        return .{ @intCast(gx), @intCast(gy), 1 };
    }

    /// Releases everything: staging, pipelines, layouts, pools, the device,
    /// the instance and the loader. Safe on a partially built context.
    pub fn deinit(self: *Context) void {
        _ = self.dfn.vkDeviceWaitIdle(self.device);
        self.freeBuf(&self.up);
        self.freeBuf(&self.down);
        const d = self.device;
        const f = &self.dfn;
        for (self.pipelines) |p| if (p != vk.null_handle) f.vkDestroyPipeline(d, p, null);
        if (self.desc_pool != vk.null_handle) f.vkDestroyDescriptorPool(d, self.desc_pool, null);
        if (self.pipe_layout != vk.null_handle) f.vkDestroyPipelineLayout(d, self.pipe_layout, null);
        if (self.set_layout != vk.null_handle) f.vkDestroyDescriptorSetLayout(d, self.set_layout, null);
        if (self.fence != vk.null_handle) f.vkDestroyFence(d, self.fence, null);
        if (self.cmd_pool != vk.null_handle) f.vkDestroyCommandPool(d, self.cmd_pool, null);
        f.vkDestroyDevice(d, null);
        self.ifn.vkDestroyInstance(self.instance, null);
        self.loader.close();
    }
};

/// Offsets inside the staging buffers.
fn alignUp(x: u64) u64 {
    return std.mem.alignForward(u64, x, 16);
}

fn barrier(f: *const vk.DeviceFns, cmd: vk.CommandBuffer, src_stage: vk.Flags, dst_stage: vk.Flags, src: vk.Flags, dst: vk.Flags) void {
    const mb = vk.MemoryBarrier{ .srcAccessMask = src, .dstAccessMask = dst };
    f.vkCmdPipelineBarrier(cmd, src_stage, dst_stage, 0, 1, @ptrCast(&mb), 0, null, 0, null);
}

// ---------------------------------------------------------------------------
// Device selection
// ---------------------------------------------------------------------------

pub const OpenOptions = struct {
    /// Also accept a device whose type is CPU (a software rasteriser such as
    /// Mesa's lavapipe). `--device vulkan` does, so the backend can be tested
    /// without a GPU; `--device auto` does not, since the CPU backend is
    /// faster than a software Vulkan device.
    allow_cpu: bool = false,
};

fn typeRank(t: vk.PhysicalDeviceType) u32 {
    return switch (t) {
        .discrete_gpu => 4,
        .integrated_gpu => 3,
        .virtual_gpu => 2,
        .other => 1,
        .cpu => 0,
        _ => 0,
    };
}

pub fn typeName(t: vk.PhysicalDeviceType) []const u8 {
    return switch (t) {
        .discrete_gpu => "discrete GPU",
        .integrated_gpu => "integrated GPU",
        .virtual_gpu => "virtual GPU",
        .cpu => "CPU",
        else => "other",
    };
}

/// The largest device-local heap, in bytes.
pub fn deviceLocalBytes(mem: *const vk.PhysicalDeviceMemoryProperties) u64 {
    var best: u64 = 0;
    for (mem.memoryHeaps[0..mem.memoryHeapCount]) |h| {
        if (h.flags & vk.MEMORY_HEAP_DEVICE_LOCAL_BIT != 0) best = @max(best, h.size);
    }
    return best;
}

fn computeFamily(ifn: *const vk.InstanceFns, pd: vk.PhysicalDevice) ?u32 {
    var count: u32 = 0;
    ifn.vkGetPhysicalDeviceQueueFamilyProperties(pd, &count, null);
    var fams: [32]vk.QueueFamilyProperties = undefined;
    count = @min(count, fams.len);
    ifn.vkGetPhysicalDeviceQueueFamilyProperties(pd, &count, &fams);
    for (fams[0..count], 0..) |q, i| {
        if (q.queueFlags & vk.QUEUE_COMPUTE_BIT != 0 and q.queueCount > 0) return @intCast(i);
    }
    return null;
}

/// Why `pd` cannot run the kernels, or null when it can.
fn unsuitable(props: *const vk.PhysicalDeviceProperties) ?[]const u8 {
    const l = &props.limits;
    if (l.maxComputeWorkGroupInvocations < min_invocations or l.maxComputeWorkGroupSize[0] < min_invocations) return "fewer than 256 invocations per workgroup";
    if (l.maxComputeWorkGroupSize[1] < min_workgroup_y) return "workgroups narrower than 16 in y";
    if (l.maxPushConstantsSize < push_bytes) return "too little push constant space";
    if (l.maxComputeSharedMemorySize < min_shared_bytes) return "too little shared memory";
    if (l.maxPerStageDescriptorStorageBuffers < max_bindings) return "too few storage buffers per stage";
    return null;
}

pub fn cstr(s: []const u8) []const u8 {
    return s[0 .. std.mem.indexOfScalar(u8, s, 0) orelse s.len];
}

/// Loads Vulkan, picks a device and sets up a queue, a command buffer and the
/// shared pipeline layout. On failure, `lastError()` says why in one line.
pub fn open(gpa: Allocator, opts: OpenOptions) !Context {
    reason_len = 0;
    var loader = vk.Loader.open() catch |e| switch (e) {
        error.Unsupported => return fail("this binary cannot load Vulkan (statically linked; use a glibc build)", .{}),
        error.LoaderMissing => return fail("no Vulkan loader ({s} not found)", .{vk.Loader.names}),
    };
    var loader_owned = true;
    defer if (loader_owned) loader.close();

    const gipa = loader.getInstanceProcAddr;
    const createInstance: *const fn (*const vk.InstanceCreateInfo, ?*const anyopaque, *vk.Instance) callconv(vk.cc) vk.Result =
        @ptrCast(gipa(null, "vkCreateInstance") orelse return fail("the Vulkan loader has no vkCreateInstance", .{}));
    const app = vk.ApplicationInfo{ .pApplicationName = "ditch", .pEngineName = "ditch", .apiVersion = vk.API_VERSION_1_0 };
    var instance: vk.Instance = undefined;
    const rc = createInstance(&.{ .pApplicationInfo = &app }, null, &instance);
    if (rc != vk.SUCCESS) {
        if (rc == vk.ERROR_INCOMPATIBLE_DRIVER) return fail("no Vulkan driver installed (the loader found no ICD)", .{});
        return fail("vkCreateInstance failed ({d})", .{rc});
    }
    const ifn = vk.resolve(vk.InstanceFns, .{ gipa, instance }, struct {
        fn get(c: anytype, name: [*:0]const u8) ?vk.PfnVoid {
            return c[0](c[1], name);
        }
    }.get) orelse {
        const destroy: ?*const fn (vk.Instance, ?*const anyopaque) callconv(vk.cc) void = @ptrCast(gipa(instance, "vkDestroyInstance"));
        if (destroy) |d| d(instance, null);
        return fail("the Vulkan loader lacks a core 1.0 function", .{});
    };
    var instance_owned = true;
    defer if (instance_owned) ifn.vkDestroyInstance(instance, null);

    var count: u32 = 0;
    _ = ifn.vkEnumeratePhysicalDevices(instance, &count, null);
    var devices: [16]vk.PhysicalDevice = undefined;
    count = @min(count, devices.len);
    if (count == 0) return fail("Vulkan reports no devices", .{});
    if (ifn.vkEnumeratePhysicalDevices(instance, &count, &devices) < 0) return fail("vkEnumeratePhysicalDevices failed", .{});

    // DITCH_VULKAN_DEVICE=<index> picks a device by its position in
    // Vulkan's enumeration (the order `vulkaninfo --summary` lists them).
    const forced: ?usize = if (getenv("DITCH_VULKAN_DEVICE")) |s| std.fmt.parseInt(usize, s, 10) catch null else null;
    if (forced) |i| if (i >= count) return fail("DITCH_VULKAN_DEVICE={d}, but Vulkan lists {d} device(s)", .{ i, count });

    var best: ?usize = null;
    var best_rank: u32 = 0;
    var best_mem: u64 = 0;
    var props: [16]vk.PhysicalDeviceProperties = undefined;
    var rejected: []const u8 = "";
    for (devices[0..count], 0..) |pd, i| {
        ifn.vkGetPhysicalDeviceProperties(pd, &props[i]);
        if (forced) |fi| if (fi != i) continue;
        if (unsuitable(&props[i])) |why| {
            rejected = why;
            continue;
        }
        if (computeFamily(&ifn, pd) == null) {
            rejected = "no compute queue";
            continue;
        }
        const t = props[i].deviceType;
        if (t == .cpu and !opts.allow_cpu and forced == null) {
            rejected = "only a software (CPU) Vulkan device";
            continue;
        }
        var mp: vk.PhysicalDeviceMemoryProperties = undefined;
        ifn.vkGetPhysicalDeviceMemoryProperties(pd, &mp);
        const rank = typeRank(t);
        const local = deviceLocalBytes(&mp);
        if (best == null or rank > best_rank or (rank == best_rank and local > best_mem)) {
            best = i;
            best_rank = rank;
            best_mem = local;
        }
    }
    const chosen = best orelse {
        if (rejected.len > 0) return fail("no suitable Vulkan GPU ({s})", .{rejected});
        return fail("no suitable Vulkan GPU", .{});
    };
    const pd = devices[chosen];
    const family = computeFamily(&ifn, pd).?;

    const priority = [_]f32{1.0};
    const qci = [_]vk.DeviceQueueCreateInfo{.{ .queueFamilyIndex = family, .queueCount = 1, .pQueuePriorities = &priority }};
    var device: vk.Device = undefined;
    if (ifn.vkCreateDevice(pd, &.{ .queueCreateInfoCount = 1, .pQueueCreateInfos = &qci }, null, &device) != vk.SUCCESS) {
        return fail("vkCreateDevice failed on {s}", .{cstr(&props[chosen].deviceName)});
    }
    const dfn = vk.resolve(vk.DeviceFns, .{ ifn.vkGetDeviceProcAddr, device }, struct {
        fn get(c: anytype, name: [*:0]const u8) ?vk.PfnVoid {
            return c[0](c[1], name);
        }
    }.get) orelse {
        const destroy: ?*const fn (vk.Device, ?*const anyopaque) callconv(vk.cc) void = @ptrCast(ifn.vkGetDeviceProcAddr(device, "vkDestroyDevice"));
        if (destroy) |d| d(device, null);
        return fail("the Vulkan driver lacks a core 1.0 function", .{});
    };
    var device_owned = true;
    defer if (device_owned) dfn.vkDestroyDevice(device, null);

    var ctx = Context{
        .gpa = gpa,
        .loader = loader,
        .instance = instance,
        .ifn = ifn,
        .phys = pd,
        .props = props[chosen],
        .mem = undefined,
        .device = device,
        .dfn = dfn,
        .queue = undefined,
        .family = family,
        .direct = false,
    };
    const self = &ctx;
    // From here on `Context.deinit` owns the device, the instance and the loader.
    loader_owned = false;
    instance_owned = false;
    device_owned = false;
    errdefer self.deinit();
    ifn.vkGetPhysicalDeviceMemoryProperties(pd, &self.mem);
    dfn.vkGetDeviceQueue(device, family, 0, &self.queue);

    // Unified memory: an integrated (or software) device with a memory type
    // that is device-local and host-visible at once. DITCH_VULKAN_STAGING=1
    // forces the discrete-GPU path (staging copies) on such a device, which
    // is how that path is tested without a discrete GPU.
    const t = self.props.deviceType;
    const uma_type = for (self.mem.memoryTypes[0..self.mem.memoryTypeCount]) |mt| {
        const want = vk.MEMORY_PROPERTY_DEVICE_LOCAL_BIT | vk.MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk.MEMORY_PROPERTY_HOST_COHERENT_BIT;
        if (mt.propertyFlags & want == want) break true;
    } else false;
    const force_staging = if (getenv("DITCH_VULKAN_STAGING")) |s| std.mem.eql(u8, s, "1") else false;
    self.direct = uma_type and (t == .integrated_gpu or t == .cpu) and !force_staging;

    const f = &self.dfn;
    if (f.vkCreateCommandPool(device, &.{ .flags = vk.COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, .queueFamilyIndex = family }, null, &self.cmd_pool) != vk.SUCCESS) return fail("vkCreateCommandPool failed", .{});
    if (f.vkAllocateCommandBuffers(device, &.{ .commandPool = self.cmd_pool, .commandBufferCount = 1 }, @ptrCast(&self.cmd)) != vk.SUCCESS) return fail("vkAllocateCommandBuffers failed", .{});
    if (f.vkCreateFence(device, &.{}, null, &self.fence) != vk.SUCCESS) return fail("vkCreateFence failed", .{});
    var bindings: [max_bindings]vk.DescriptorSetLayoutBinding = undefined;
    for (&bindings, 0..) |*b, i| b.* = .{ .binding = @intCast(i), .descriptorType = vk.DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1, .stageFlags = vk.SHADER_STAGE_COMPUTE_BIT };
    if (f.vkCreateDescriptorSetLayout(device, &.{ .bindingCount = max_bindings, .pBindings = &bindings }, null, &self.set_layout) != vk.SUCCESS) return fail("vkCreateDescriptorSetLayout failed", .{});
    const range = vk.PushConstantRange{ .stageFlags = vk.SHADER_STAGE_COMPUTE_BIT, .offset = 0, .size = push_bytes };
    if (f.vkCreatePipelineLayout(device, &.{ .setLayoutCount = 1, .pSetLayouts = @ptrCast(&self.set_layout), .pushConstantRangeCount = 1, .pPushConstantRanges = @ptrCast(&range) }, null, &self.pipe_layout) != vk.SUCCESS) return fail("vkCreatePipelineLayout failed", .{});
    const pool_size = vk.DescriptorPoolSize{ .type = vk.DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = max_bindings };
    if (f.vkCreateDescriptorPool(device, &.{ .maxSets = 1, .poolSizeCount = 1, .pPoolSizes = @ptrCast(&pool_size) }, null, &self.desc_pool) != vk.SUCCESS) return fail("vkCreateDescriptorPool failed", .{});
    if (f.vkAllocateDescriptorSets(device, &.{ .descriptorPool = self.desc_pool, .descriptorSetCount = 1, .pSetLayouts = @ptrCast(&self.set_layout) }, @ptrCast(&self.desc_set)) != vk.SUCCESS) return fail("vkAllocateDescriptorSets failed", .{});

    return ctx;
}

/// A description of the device for reports: API and driver version, memory
/// mode and the limits that bound what the backend takes.
pub fn describe(self: *const Context, w: *std.Io.Writer) !void {
    const p = &self.props;
    try w.print("Vulkan {d}.{d}, vendor 0x{x:0>4}, device 0x{x:0>4}, driver version 0x{x}, {s} memory, largest device-local heap {d} MiB, max storage buffer {d} MiB", .{
        vk.apiMajor(p.apiVersion),
        vk.apiMinor(p.apiVersion),
        p.vendorID,
        p.deviceID,
        p.driverVersion,
        if (self.direct) "unified" else "staged",
        deviceLocalBytes(&self.mem) >> 20,
        p.limits.maxStorageBufferRange >> 20,
    });
}

test "opening a Vulkan device either works or says why" {
    if (!vk.loadable) return error.SkipZigTest;
    var ctx = open(std.testing.allocator, .{ .allow_cpu = true }) catch |e| {
        try std.testing.expectEqual(error.DeviceUnavailable, e);
        try std.testing.expect(lastError().len > 0);
        return;
    };
    defer ctx.deinit();
    try std.testing.expect(cstr(&ctx.props.deviceName).len > 0);
    // A buffer round trip through whichever memory path the device uses.
    var b = try ctx.newBuf(4096, .kernel);
    defer ctx.freeBuf(&b);
    try std.testing.expect(b.size == 4096);
    if (ctx.direct) try std.testing.expect(b.map != null);
    var line: [512]u8 = undefined;
    var w: std.Io.Writer = .fixed(&line);
    try describe(&ctx, &w);
    try std.testing.expect(std.mem.startsWith(u8, w.buffered(), "Vulkan 1."));
}

//! Pins the hand-written Vulkan structures in `vk.zig` to the layout of
//! `vulkan_core.h` on 64-bit targets (the only ones ditch ships). The numbers
//! are `sizeof` and `offsetof` from the real header, compiled with a C
//! compiler; a field missed or mistyped in the long limits struct would move
//! every offset after it and show up here.

const std = @import("std");
const vk = @import("vk.zig");

const expectEqual = std.testing.expectEqual;

test "Vulkan structures have the C layout" {
    if (@sizeOf(usize) != 8) return error.SkipZigTest;
    try expectEqual(@as(usize, 504), @sizeOf(vk.PhysicalDeviceLimits));
    try expectEqual(@as(usize, 824), @sizeOf(vk.PhysicalDeviceProperties));
    try expectEqual(@as(usize, 520), @sizeOf(vk.PhysicalDeviceMemoryProperties));
    try expectEqual(@as(usize, 24), @sizeOf(vk.QueueFamilyProperties));
    try expectEqual(@as(usize, 96), @sizeOf(vk.ComputePipelineCreateInfo));
    try expectEqual(@as(usize, 64), @sizeOf(vk.WriteDescriptorSet));
    try expectEqual(@as(usize, 72), @sizeOf(vk.SubmitInfo));
    try expectEqual(@as(usize, 56), @sizeOf(vk.BufferCreateInfo));
    try expectEqual(@as(usize, 72), @sizeOf(vk.DeviceCreateInfo));
    try expectEqual(@as(usize, 64), @sizeOf(vk.InstanceCreateInfo));
    try expectEqual(@as(usize, 48), @sizeOf(vk.ApplicationInfo));

    try expectEqual(@as(usize, 28), @offsetOf(vk.PhysicalDeviceLimits, "maxStorageBufferRange"));
    try expectEqual(@as(usize, 216), @offsetOf(vk.PhysicalDeviceLimits, "maxComputeSharedMemorySize"));
    try expectEqual(@as(usize, 220), @offsetOf(vk.PhysicalDeviceLimits, "maxComputeWorkGroupCount"));
    try expectEqual(@as(usize, 232), @offsetOf(vk.PhysicalDeviceLimits, "maxComputeWorkGroupInvocations"));
    try expectEqual(@as(usize, 304), @offsetOf(vk.PhysicalDeviceLimits, "minMemoryMapAlignment"));
    try expectEqual(@as(usize, 496), @offsetOf(vk.PhysicalDeviceLimits, "nonCoherentAtomSize"));
    try expectEqual(@as(usize, 296), @offsetOf(vk.PhysicalDeviceProperties, "limits"));
    try expectEqual(@as(usize, 800), @offsetOf(vk.PhysicalDeviceProperties, "sparseProperties"));
    try expectEqual(@as(usize, 264), @offsetOf(vk.PhysicalDeviceMemoryProperties, "memoryHeaps"));
}

test "Vulkan enumerants have the header's values" {
    try expectEqual(@as(i32, 29), @intFromEnum(vk.StructureType.compute_pipeline_create_info));
    try expectEqual(@as(i32, 35), @intFromEnum(vk.StructureType.write_descriptor_set));
    try expectEqual(@as(i32, 46), @intFromEnum(vk.StructureType.memory_barrier));
    try expectEqual(@as(u32, 7), vk.DESCRIPTOR_TYPE_STORAGE_BUFFER);
    try expectEqual(@as(u32, 0x800), vk.PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    try expectEqual(@as(u32, 0x4000), vk.PIPELINE_STAGE_HOST_BIT);
    try expectEqual(@as(u32, 0x2000), vk.ACCESS_HOST_READ_BIT);
    try expectEqual(@as(u32, 4194304), vk.API_VERSION_1_0);
}

//! The part of the Vulkan 1.0 API the compute backend uses, declared by hand
//! and loaded at run time.
//!
//! Nothing here links against the Vulkan SDK or the loader: `Loader.open`
//! finds `libvulkan.so.1` (Linux) or `vulkan-1.dll` (Windows) with
//! `dlopen`/`LoadLibraryA`, and every other entry point is resolved through
//! `vkGetInstanceProcAddr` / `vkGetDeviceProcAddr`. A machine without Vulkan
//! therefore runs the same binary; the backend reports itself unavailable and
//! ditch stays on the CPU.
//!
//! Only the types, constants and functions the backend calls are declared.
//! The struct layouts follow `vulkan_core.h`; `src/vulkan/vk_test.zig` pins
//! their sizes and the offsets that matter.

const std = @import("std");
const builtin = @import("builtin");

/// `VKAPI_PTR`: the C calling convention on every target ditch ships for
/// (it is `__stdcall` only on 32-bit Windows, which is not one of them).
pub const cc: std.builtin.CallingConvention = .c;

/// Whether this build can load a Vulkan loader at all. A statically linked
/// musl binary has no dynamic linker to load `libvulkan.so.1` (and the loader
/// and the drivers it loads are built against the system libc), so the
/// release builds for `*-linux-musl` report the backend unavailable; a
/// glibc-linked build (`-Dtarget=x86_64-linux-gnu`, or a native build on a
/// glibc distribution) and the Windows build load it.
pub const loadable = switch (builtin.os.tag) {
    .linux => builtin.link_libc and !(builtin.abi.isMusl() and builtin.link_mode == .static),
    .windows => true,
    else => false,
};

// ---------------------------------------------------------------------------
// Handles and scalars
// ---------------------------------------------------------------------------

pub const Instance = *opaque {};
pub const PhysicalDevice = *opaque {};
pub const Device = *opaque {};
pub const Queue = *opaque {};
pub const CommandBuffer = *opaque {};

/// Non-dispatchable handles are 64-bit on every platform.
pub const Buffer = u64;
pub const DeviceMemory = u64;
pub const ShaderModule = u64;
pub const DescriptorSetLayout = u64;
pub const PipelineLayout = u64;
pub const Pipeline = u64;
pub const DescriptorPool = u64;
pub const DescriptorSet = u64;
pub const CommandPool = u64;
pub const Fence = u64;
pub const PipelineCache = u64;
pub const null_handle: u64 = 0;

pub const Bool32 = u32;
pub const DeviceSize = u64;
pub const Flags = u32;
pub const Result = i32;

pub const whole_size: u64 = ~@as(u64, 0);

pub const SUCCESS: Result = 0;
pub const TIMEOUT: Result = 2;
pub const ERROR_OUT_OF_HOST_MEMORY: Result = -1;
pub const ERROR_OUT_OF_DEVICE_MEMORY: Result = -2;
pub const ERROR_INITIALIZATION_FAILED: Result = -3;
pub const ERROR_DEVICE_LOST: Result = -4;
pub const ERROR_INCOMPATIBLE_DRIVER: Result = -9;

pub fn makeApiVersion(major: u32, minor: u32, patch: u32) u32 {
    return (major << 22) | (minor << 12) | patch;
}
pub const API_VERSION_1_0 = (1 << 22);

pub fn apiMajor(v: u32) u32 {
    return (v >> 22) & 0x7f;
}
pub fn apiMinor(v: u32) u32 {
    return (v >> 12) & 0x3ff;
}

pub const StructureType = enum(i32) {
    application_info = 0,
    instance_create_info = 1,
    device_queue_create_info = 2,
    device_create_info = 3,
    submit_info = 4,
    memory_allocate_info = 5,
    fence_create_info = 8,
    buffer_create_info = 12,
    shader_module_create_info = 16,
    pipeline_shader_stage_create_info = 18,
    compute_pipeline_create_info = 29,
    pipeline_layout_create_info = 30,
    descriptor_set_layout_create_info = 32,
    descriptor_pool_create_info = 33,
    descriptor_set_allocate_info = 34,
    write_descriptor_set = 35,
    command_pool_create_info = 39,
    command_buffer_allocate_info = 40,
    command_buffer_begin_info = 42,
    memory_barrier = 46,
};

pub const PhysicalDeviceType = enum(i32) {
    other = 0,
    integrated_gpu = 1,
    discrete_gpu = 2,
    virtual_gpu = 3,
    cpu = 4,
    _,
};

pub const QUEUE_COMPUTE_BIT: Flags = 0x2;

pub const MEMORY_PROPERTY_DEVICE_LOCAL_BIT: Flags = 0x1;
pub const MEMORY_PROPERTY_HOST_VISIBLE_BIT: Flags = 0x2;
pub const MEMORY_PROPERTY_HOST_COHERENT_BIT: Flags = 0x4;
pub const MEMORY_PROPERTY_HOST_CACHED_BIT: Flags = 0x8;
pub const MEMORY_HEAP_DEVICE_LOCAL_BIT: Flags = 0x1;

pub const BUFFER_USAGE_TRANSFER_SRC_BIT: Flags = 0x1;
pub const BUFFER_USAGE_TRANSFER_DST_BIT: Flags = 0x2;
pub const BUFFER_USAGE_STORAGE_BUFFER_BIT: Flags = 0x20;

pub const SHARING_MODE_EXCLUSIVE: u32 = 0;
pub const DESCRIPTOR_TYPE_STORAGE_BUFFER: u32 = 7;
pub const SHADER_STAGE_COMPUTE_BIT: Flags = 0x20;
pub const PIPELINE_BIND_POINT_COMPUTE: u32 = 1;
pub const COMMAND_BUFFER_LEVEL_PRIMARY: u32 = 0;
pub const COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT: Flags = 0x2;
pub const COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT: Flags = 0x1;

pub const PIPELINE_STAGE_COMPUTE_SHADER_BIT: Flags = 0x800;
pub const PIPELINE_STAGE_TRANSFER_BIT: Flags = 0x1000;
pub const PIPELINE_STAGE_HOST_BIT: Flags = 0x4000;

pub const ACCESS_SHADER_READ_BIT: Flags = 0x20;
pub const ACCESS_SHADER_WRITE_BIT: Flags = 0x40;
pub const ACCESS_TRANSFER_READ_BIT: Flags = 0x800;
pub const ACCESS_TRANSFER_WRITE_BIT: Flags = 0x1000;
pub const ACCESS_HOST_READ_BIT: Flags = 0x2000;
pub const ACCESS_HOST_WRITE_BIT: Flags = 0x4000;

pub const MAX_PHYSICAL_DEVICE_NAME_SIZE = 256;
pub const UUID_SIZE = 16;
pub const MAX_MEMORY_TYPES = 32;
pub const MAX_MEMORY_HEAPS = 16;

// ---------------------------------------------------------------------------
// Structures
// ---------------------------------------------------------------------------

pub const ApplicationInfo = extern struct {
    sType: StructureType = .application_info,
    pNext: ?*const anyopaque = null,
    pApplicationName: ?[*:0]const u8 = null,
    applicationVersion: u32 = 0,
    pEngineName: ?[*:0]const u8 = null,
    engineVersion: u32 = 0,
    apiVersion: u32 = API_VERSION_1_0,
};

pub const InstanceCreateInfo = extern struct {
    sType: StructureType = .instance_create_info,
    pNext: ?*const anyopaque = null,
    flags: Flags = 0,
    pApplicationInfo: ?*const ApplicationInfo = null,
    enabledLayerCount: u32 = 0,
    ppEnabledLayerNames: ?[*]const [*:0]const u8 = null,
    enabledExtensionCount: u32 = 0,
    ppEnabledExtensionNames: ?[*]const [*:0]const u8 = null,
};

pub const PhysicalDeviceLimits = extern struct {
    maxImageDimension1D: u32,
    maxImageDimension2D: u32,
    maxImageDimension3D: u32,
    maxImageDimensionCube: u32,
    maxImageArrayLayers: u32,
    maxTexelBufferElements: u32,
    maxUniformBufferRange: u32,
    maxStorageBufferRange: u32,
    maxPushConstantsSize: u32,
    maxMemoryAllocationCount: u32,
    maxSamplerAllocationCount: u32,
    bufferImageGranularity: DeviceSize,
    sparseAddressSpaceSize: DeviceSize,
    maxBoundDescriptorSets: u32,
    maxPerStageDescriptorSamplers: u32,
    maxPerStageDescriptorUniformBuffers: u32,
    maxPerStageDescriptorStorageBuffers: u32,
    maxPerStageDescriptorSampledImages: u32,
    maxPerStageDescriptorStorageImages: u32,
    maxPerStageDescriptorInputAttachments: u32,
    maxPerStageResources: u32,
    maxDescriptorSetSamplers: u32,
    maxDescriptorSetUniformBuffers: u32,
    maxDescriptorSetUniformBuffersDynamic: u32,
    maxDescriptorSetStorageBuffers: u32,
    maxDescriptorSetStorageBuffersDynamic: u32,
    maxDescriptorSetSampledImages: u32,
    maxDescriptorSetStorageImages: u32,
    maxDescriptorSetInputAttachments: u32,
    maxVertexInputAttributes: u32,
    maxVertexInputBindings: u32,
    maxVertexInputAttributeOffset: u32,
    maxVertexInputBindingStride: u32,
    maxVertexOutputComponents: u32,
    maxTessellationGenerationLevel: u32,
    maxTessellationPatchSize: u32,
    maxTessellationControlPerVertexInputComponents: u32,
    maxTessellationControlPerVertexOutputComponents: u32,
    maxTessellationControlPerPatchOutputComponents: u32,
    maxTessellationControlTotalOutputComponents: u32,
    maxTessellationEvaluationInputComponents: u32,
    maxTessellationEvaluationOutputComponents: u32,
    maxGeometryShaderInvocations: u32,
    maxGeometryInputComponents: u32,
    maxGeometryOutputComponents: u32,
    maxGeometryOutputVertices: u32,
    maxGeometryTotalOutputComponents: u32,
    maxFragmentInputComponents: u32,
    maxFragmentOutputAttachments: u32,
    maxFragmentDualSrcAttachments: u32,
    maxFragmentCombinedOutputResources: u32,
    maxComputeSharedMemorySize: u32,
    maxComputeWorkGroupCount: [3]u32,
    maxComputeWorkGroupInvocations: u32,
    maxComputeWorkGroupSize: [3]u32,
    subPixelPrecisionBits: u32,
    subTexelPrecisionBits: u32,
    mipmapPrecisionBits: u32,
    maxDrawIndexedIndexValue: u32,
    maxDrawIndirectCount: u32,
    maxSamplerLodBias: f32,
    maxSamplerAnisotropy: f32,
    maxViewports: u32,
    maxViewportDimensions: [2]u32,
    viewportBoundsRange: [2]f32,
    viewportSubPixelBits: u32,
    minMemoryMapAlignment: usize,
    minTexelBufferOffsetAlignment: DeviceSize,
    minUniformBufferOffsetAlignment: DeviceSize,
    minStorageBufferOffsetAlignment: DeviceSize,
    minTexelOffset: i32,
    maxTexelOffset: u32,
    minTexelGatherOffset: i32,
    maxTexelGatherOffset: u32,
    minInterpolationOffset: f32,
    maxInterpolationOffset: f32,
    subPixelInterpolationOffsetBits: u32,
    maxFramebufferWidth: u32,
    maxFramebufferHeight: u32,
    maxFramebufferLayers: u32,
    framebufferColorSampleCounts: Flags,
    framebufferDepthSampleCounts: Flags,
    framebufferStencilSampleCounts: Flags,
    framebufferNoAttachmentsSampleCounts: Flags,
    maxColorAttachments: u32,
    sampledImageColorSampleCounts: Flags,
    sampledImageIntegerSampleCounts: Flags,
    sampledImageDepthSampleCounts: Flags,
    sampledImageStencilSampleCounts: Flags,
    storageImageSampleCounts: Flags,
    maxSampleMaskWords: u32,
    timestampComputeAndGraphics: Bool32,
    timestampPeriod: f32,
    maxClipDistances: u32,
    maxCullDistances: u32,
    maxCombinedClipAndCullDistances: u32,
    discreteQueuePriorities: u32,
    pointSizeRange: [2]f32,
    lineWidthRange: [2]f32,
    pointSizeGranularity: f32,
    lineWidthGranularity: f32,
    strictLines: Bool32,
    standardSampleLocations: Bool32,
    optimalBufferCopyOffsetAlignment: DeviceSize,
    optimalBufferCopyRowPitch: DeviceSize,
    nonCoherentAtomSize: DeviceSize,
};

pub const PhysicalDeviceSparseProperties = extern struct {
    residencyStandard2DBlockShape: Bool32,
    residencyStandard2DMultisampleBlockShape: Bool32,
    residencyStandard3DBlockShape: Bool32,
    residencyAlignedMipSize: Bool32,
    residencyNonResidentStrict: Bool32,
};

pub const PhysicalDeviceProperties = extern struct {
    apiVersion: u32,
    driverVersion: u32,
    vendorID: u32,
    deviceID: u32,
    deviceType: PhysicalDeviceType,
    deviceName: [MAX_PHYSICAL_DEVICE_NAME_SIZE]u8,
    pipelineCacheUUID: [UUID_SIZE]u8,
    limits: PhysicalDeviceLimits,
    sparseProperties: PhysicalDeviceSparseProperties,
};

pub const MemoryType = extern struct {
    propertyFlags: Flags,
    heapIndex: u32,
};

pub const MemoryHeap = extern struct {
    size: DeviceSize,
    flags: Flags,
};

pub const PhysicalDeviceMemoryProperties = extern struct {
    memoryTypeCount: u32,
    memoryTypes: [MAX_MEMORY_TYPES]MemoryType,
    memoryHeapCount: u32,
    memoryHeaps: [MAX_MEMORY_HEAPS]MemoryHeap,
};

pub const Extent3D = extern struct { width: u32, height: u32, depth: u32 };

pub const QueueFamilyProperties = extern struct {
    queueFlags: Flags,
    queueCount: u32,
    timestampValidBits: u32,
    minImageTransferGranularity: Extent3D,
};

pub const DeviceQueueCreateInfo = extern struct {
    sType: StructureType = .device_queue_create_info,
    pNext: ?*const anyopaque = null,
    flags: Flags = 0,
    queueFamilyIndex: u32,
    queueCount: u32,
    pQueuePriorities: [*]const f32,
};

pub const DeviceCreateInfo = extern struct {
    sType: StructureType = .device_create_info,
    pNext: ?*const anyopaque = null,
    flags: Flags = 0,
    queueCreateInfoCount: u32,
    pQueueCreateInfos: [*]const DeviceQueueCreateInfo,
    enabledLayerCount: u32 = 0,
    ppEnabledLayerNames: ?[*]const [*:0]const u8 = null,
    enabledExtensionCount: u32 = 0,
    ppEnabledExtensionNames: ?[*]const [*:0]const u8 = null,
    pEnabledFeatures: ?*const anyopaque = null,
};

pub const BufferCreateInfo = extern struct {
    sType: StructureType = .buffer_create_info,
    pNext: ?*const anyopaque = null,
    flags: Flags = 0,
    size: DeviceSize,
    usage: Flags,
    sharingMode: u32 = SHARING_MODE_EXCLUSIVE,
    queueFamilyIndexCount: u32 = 0,
    pQueueFamilyIndices: ?[*]const u32 = null,
};

pub const MemoryRequirements = extern struct {
    size: DeviceSize,
    alignment: DeviceSize,
    memoryTypeBits: u32,
};

pub const MemoryAllocateInfo = extern struct {
    sType: StructureType = .memory_allocate_info,
    pNext: ?*const anyopaque = null,
    allocationSize: DeviceSize,
    memoryTypeIndex: u32,
};

pub const ShaderModuleCreateInfo = extern struct {
    sType: StructureType = .shader_module_create_info,
    pNext: ?*const anyopaque = null,
    flags: Flags = 0,
    codeSize: usize,
    pCode: [*]const u32,
};

pub const DescriptorSetLayoutBinding = extern struct {
    binding: u32,
    descriptorType: u32,
    descriptorCount: u32,
    stageFlags: Flags,
    pImmutableSamplers: ?*const anyopaque = null,
};

pub const DescriptorSetLayoutCreateInfo = extern struct {
    sType: StructureType = .descriptor_set_layout_create_info,
    pNext: ?*const anyopaque = null,
    flags: Flags = 0,
    bindingCount: u32,
    pBindings: [*]const DescriptorSetLayoutBinding,
};

pub const PushConstantRange = extern struct {
    stageFlags: Flags,
    offset: u32,
    size: u32,
};

pub const PipelineLayoutCreateInfo = extern struct {
    sType: StructureType = .pipeline_layout_create_info,
    pNext: ?*const anyopaque = null,
    flags: Flags = 0,
    setLayoutCount: u32,
    pSetLayouts: [*]const DescriptorSetLayout,
    pushConstantRangeCount: u32,
    pPushConstantRanges: [*]const PushConstantRange,
};

pub const PipelineShaderStageCreateInfo = extern struct {
    sType: StructureType = .pipeline_shader_stage_create_info,
    pNext: ?*const anyopaque = null,
    flags: Flags = 0,
    stage: Flags,
    module: ShaderModule,
    pName: [*:0]const u8,
    pSpecializationInfo: ?*const anyopaque = null,
};

pub const ComputePipelineCreateInfo = extern struct {
    sType: StructureType = .compute_pipeline_create_info,
    pNext: ?*const anyopaque = null,
    flags: Flags = 0,
    stage: PipelineShaderStageCreateInfo,
    layout: PipelineLayout,
    basePipelineHandle: Pipeline = null_handle,
    basePipelineIndex: i32 = -1,
};

pub const DescriptorPoolSize = extern struct {
    type: u32,
    descriptorCount: u32,
};

pub const DescriptorPoolCreateInfo = extern struct {
    sType: StructureType = .descriptor_pool_create_info,
    pNext: ?*const anyopaque = null,
    flags: Flags = 0,
    maxSets: u32,
    poolSizeCount: u32,
    pPoolSizes: [*]const DescriptorPoolSize,
};

pub const DescriptorSetAllocateInfo = extern struct {
    sType: StructureType = .descriptor_set_allocate_info,
    pNext: ?*const anyopaque = null,
    descriptorPool: DescriptorPool,
    descriptorSetCount: u32,
    pSetLayouts: [*]const DescriptorSetLayout,
};

pub const DescriptorBufferInfo = extern struct {
    buffer: Buffer,
    offset: DeviceSize,
    range: DeviceSize,
};

pub const WriteDescriptorSet = extern struct {
    sType: StructureType = .write_descriptor_set,
    pNext: ?*const anyopaque = null,
    dstSet: DescriptorSet,
    dstBinding: u32,
    dstArrayElement: u32 = 0,
    descriptorCount: u32,
    descriptorType: u32,
    pImageInfo: ?*const anyopaque = null,
    pBufferInfo: ?[*]const DescriptorBufferInfo,
    pTexelBufferView: ?*const anyopaque = null,
};

pub const CommandPoolCreateInfo = extern struct {
    sType: StructureType = .command_pool_create_info,
    pNext: ?*const anyopaque = null,
    flags: Flags = 0,
    queueFamilyIndex: u32,
};

pub const CommandBufferAllocateInfo = extern struct {
    sType: StructureType = .command_buffer_allocate_info,
    pNext: ?*const anyopaque = null,
    commandPool: CommandPool,
    level: u32 = COMMAND_BUFFER_LEVEL_PRIMARY,
    commandBufferCount: u32,
};

pub const CommandBufferBeginInfo = extern struct {
    sType: StructureType = .command_buffer_begin_info,
    pNext: ?*const anyopaque = null,
    flags: Flags = 0,
    pInheritanceInfo: ?*const anyopaque = null,
};

pub const BufferCopy = extern struct {
    srcOffset: DeviceSize,
    dstOffset: DeviceSize,
    size: DeviceSize,
};

pub const MemoryBarrier = extern struct {
    sType: StructureType = .memory_barrier,
    pNext: ?*const anyopaque = null,
    srcAccessMask: Flags,
    dstAccessMask: Flags,
};

pub const SubmitInfo = extern struct {
    sType: StructureType = .submit_info,
    pNext: ?*const anyopaque = null,
    waitSemaphoreCount: u32 = 0,
    pWaitSemaphores: ?*const anyopaque = null,
    pWaitDstStageMask: ?*const Flags = null,
    commandBufferCount: u32,
    pCommandBuffers: [*]const CommandBuffer,
    signalSemaphoreCount: u32 = 0,
    pSignalSemaphores: ?*const anyopaque = null,
};

pub const FenceCreateInfo = extern struct {
    sType: StructureType = .fence_create_info,
    pNext: ?*const anyopaque = null,
    flags: Flags = 0,
};

// ---------------------------------------------------------------------------
// Entry points
// ---------------------------------------------------------------------------

pub const PfnVoid = *const fn () callconv(cc) void;
pub const PfnGetInstanceProcAddr = *const fn (instance: ?Instance, name: [*:0]const u8) callconv(cc) ?PfnVoid;

/// Instance-level functions, resolved with `vkGetInstanceProcAddr`.
pub const InstanceFns = struct {
    vkDestroyInstance: *const fn (Instance, ?*const anyopaque) callconv(cc) void,
    vkEnumeratePhysicalDevices: *const fn (Instance, *u32, ?[*]PhysicalDevice) callconv(cc) Result,
    vkGetPhysicalDeviceProperties: *const fn (PhysicalDevice, *PhysicalDeviceProperties) callconv(cc) void,
    vkGetPhysicalDeviceMemoryProperties: *const fn (PhysicalDevice, *PhysicalDeviceMemoryProperties) callconv(cc) void,
    vkGetPhysicalDeviceQueueFamilyProperties: *const fn (PhysicalDevice, *u32, ?[*]QueueFamilyProperties) callconv(cc) void,
    vkCreateDevice: *const fn (PhysicalDevice, *const DeviceCreateInfo, ?*const anyopaque, *Device) callconv(cc) Result,
    vkGetDeviceProcAddr: *const fn (Device, [*:0]const u8) callconv(cc) ?PfnVoid,
};

/// Device-level functions, resolved with `vkGetDeviceProcAddr`.
pub const DeviceFns = struct {
    vkDestroyDevice: *const fn (Device, ?*const anyopaque) callconv(cc) void,
    vkGetDeviceQueue: *const fn (Device, u32, u32, *Queue) callconv(cc) void,
    vkDeviceWaitIdle: *const fn (Device) callconv(cc) Result,
    vkCreateBuffer: *const fn (Device, *const BufferCreateInfo, ?*const anyopaque, *Buffer) callconv(cc) Result,
    vkDestroyBuffer: *const fn (Device, Buffer, ?*const anyopaque) callconv(cc) void,
    vkGetBufferMemoryRequirements: *const fn (Device, Buffer, *MemoryRequirements) callconv(cc) void,
    vkAllocateMemory: *const fn (Device, *const MemoryAllocateInfo, ?*const anyopaque, *DeviceMemory) callconv(cc) Result,
    vkFreeMemory: *const fn (Device, DeviceMemory, ?*const anyopaque) callconv(cc) void,
    vkBindBufferMemory: *const fn (Device, Buffer, DeviceMemory, DeviceSize) callconv(cc) Result,
    vkMapMemory: *const fn (Device, DeviceMemory, DeviceSize, DeviceSize, Flags, *?*anyopaque) callconv(cc) Result,
    vkUnmapMemory: *const fn (Device, DeviceMemory) callconv(cc) void,
    vkCreateShaderModule: *const fn (Device, *const ShaderModuleCreateInfo, ?*const anyopaque, *ShaderModule) callconv(cc) Result,
    vkDestroyShaderModule: *const fn (Device, ShaderModule, ?*const anyopaque) callconv(cc) void,
    vkCreateDescriptorSetLayout: *const fn (Device, *const DescriptorSetLayoutCreateInfo, ?*const anyopaque, *DescriptorSetLayout) callconv(cc) Result,
    vkDestroyDescriptorSetLayout: *const fn (Device, DescriptorSetLayout, ?*const anyopaque) callconv(cc) void,
    vkCreatePipelineLayout: *const fn (Device, *const PipelineLayoutCreateInfo, ?*const anyopaque, *PipelineLayout) callconv(cc) Result,
    vkDestroyPipelineLayout: *const fn (Device, PipelineLayout, ?*const anyopaque) callconv(cc) void,
    vkCreateComputePipelines: *const fn (Device, PipelineCache, u32, [*]const ComputePipelineCreateInfo, ?*const anyopaque, [*]Pipeline) callconv(cc) Result,
    vkDestroyPipeline: *const fn (Device, Pipeline, ?*const anyopaque) callconv(cc) void,
    vkCreateDescriptorPool: *const fn (Device, *const DescriptorPoolCreateInfo, ?*const anyopaque, *DescriptorPool) callconv(cc) Result,
    vkDestroyDescriptorPool: *const fn (Device, DescriptorPool, ?*const anyopaque) callconv(cc) void,
    vkAllocateDescriptorSets: *const fn (Device, *const DescriptorSetAllocateInfo, [*]DescriptorSet) callconv(cc) Result,
    vkUpdateDescriptorSets: *const fn (Device, u32, [*]const WriteDescriptorSet, u32, ?*const anyopaque) callconv(cc) void,
    vkCreateCommandPool: *const fn (Device, *const CommandPoolCreateInfo, ?*const anyopaque, *CommandPool) callconv(cc) Result,
    vkDestroyCommandPool: *const fn (Device, CommandPool, ?*const anyopaque) callconv(cc) void,
    vkAllocateCommandBuffers: *const fn (Device, *const CommandBufferAllocateInfo, [*]CommandBuffer) callconv(cc) Result,
    vkBeginCommandBuffer: *const fn (CommandBuffer, *const CommandBufferBeginInfo) callconv(cc) Result,
    vkEndCommandBuffer: *const fn (CommandBuffer) callconv(cc) Result,
    vkResetCommandBuffer: *const fn (CommandBuffer, Flags) callconv(cc) Result,
    vkCmdBindPipeline: *const fn (CommandBuffer, u32, Pipeline) callconv(cc) void,
    vkCmdBindDescriptorSets: *const fn (CommandBuffer, u32, PipelineLayout, u32, u32, [*]const DescriptorSet, u32, ?[*]const u32) callconv(cc) void,
    vkCmdPushConstants: *const fn (CommandBuffer, PipelineLayout, Flags, u32, u32, *const anyopaque) callconv(cc) void,
    vkCmdDispatch: *const fn (CommandBuffer, u32, u32, u32) callconv(cc) void,
    vkCmdCopyBuffer: *const fn (CommandBuffer, Buffer, Buffer, u32, [*]const BufferCopy) callconv(cc) void,
    vkCmdPipelineBarrier: *const fn (CommandBuffer, Flags, Flags, Flags, u32, ?[*]const MemoryBarrier, u32, ?*const anyopaque, u32, ?*const anyopaque) callconv(cc) void,
    vkQueueSubmit: *const fn (Queue, u32, [*]const SubmitInfo, Fence) callconv(cc) Result,
    vkCreateFence: *const fn (Device, *const FenceCreateInfo, ?*const anyopaque, *Fence) callconv(cc) Result,
    vkDestroyFence: *const fn (Device, Fence, ?*const anyopaque) callconv(cc) void,
    vkWaitForFences: *const fn (Device, u32, [*]const Fence, Bool32, u64) callconv(cc) Result,
    vkResetFences: *const fn (Device, u32, [*]const Fence) callconv(cc) Result,
};

/// Resolves every field of `T` by name through `getter`; null when one is
/// missing (a broken or too old driver).
pub fn resolve(comptime T: type, ctx: anytype, comptime getter: anytype) ?T {
    var fns: T = undefined;
    inline for (@typeInfo(T).@"struct".fields) |f| {
        const p = getter(ctx, @ptrCast(f.name.ptr)) orelse return null;
        @field(fns, f.name) = @ptrCast(p);
    }
    return fns;
}

// ---------------------------------------------------------------------------
// The loader library
// ---------------------------------------------------------------------------

const win = struct {
    extern "kernel32" fn LoadLibraryA(name: [*:0]const u8) callconv(.winapi) ?*anyopaque;
    extern "kernel32" fn GetProcAddress(module: *anyopaque, name: [*:0]const u8) callconv(.winapi) ?*anyopaque;
    extern "kernel32" fn FreeLibrary(module: *anyopaque) callconv(.winapi) i32;
};

/// The Vulkan loader (`libvulkan.so.1`, `vulkan-1.dll`), opened at run time.
pub const Loader = struct {
    handle: *anyopaque,
    getInstanceProcAddr: PfnGetInstanceProcAddr,

    /// The file names tried, in order, for messages.
    pub const names = switch (builtin.os.tag) {
        .windows => "vulkan-1.dll",
        else => "libvulkan.so.1",
    };

    pub fn open() error{ LoaderMissing, Unsupported }!Loader {
        if (!loadable) return error.Unsupported;
        switch (builtin.os.tag) {
            .windows => {
                const h = win.LoadLibraryA("vulkan-1.dll") orelse return error.LoaderMissing;
                const p = win.GetProcAddress(h, "vkGetInstanceProcAddr") orelse {
                    _ = win.FreeLibrary(h);
                    return error.LoaderMissing;
                };
                return .{ .handle = h, .getInstanceProcAddr = @ptrCast(p) };
            },
            else => {
                const h = std.c.dlopen("libvulkan.so.1", .{ .LAZY = true }) orelse
                    std.c.dlopen("libvulkan.so", .{ .LAZY = true }) orelse
                    return error.LoaderMissing;
                const p = std.c.dlsym(h, "vkGetInstanceProcAddr") orelse {
                    _ = std.c.dlclose(h);
                    return error.LoaderMissing;
                };
                return .{ .handle = h, .getInstanceProcAddr = @ptrCast(p) };
            },
        }
    }

    pub fn close(self: *Loader) void {
        if (!loadable) return;
        switch (builtin.os.tag) {
            .windows => _ = win.FreeLibrary(self.handle),
            else => _ = std.c.dlclose(self.handle),
        }
    }
};

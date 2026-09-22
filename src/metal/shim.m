// Objective-C implementation of ditch's Metal surface (see shim.h). Compiled
// only for macOS targets and only with -Dmetal (build.zig), because it needs
// Apple's SDK headers.
//
// Design notes:
//   * Buffers are MTLResourceStorageModeShared: on Apple silicon the CPU and
//     GPU address the same pages, so a converted or mapped weight tile is
//     written once and read by the GPU in place, never copied twice.
//   * One command queue, one pipeline cache, one mutex: the backend is called
//     from the thread that drives the forward pass, and the mutex keeps a
//     concurrent caller (the selftest harness) from interleaving encoders.
//   * Dispatch is synchronous (waitUntilCompleted) because ditch's callers
//     need the result in host memory immediately.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <pthread.h>
#include <unistd.h>
#include <string.h>

#include "shim.h"

#define DITCH_MAX_PIPELINES 64
#define DITCH_MAX_BUFFERS 8

struct ditch_mtl {
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    id<MTLLibrary> library;
    char name[256];
    pthread_mutex_t lock;
    int32_t n_pipelines;
    char pipeline_names[DITCH_MAX_PIPELINES][64];
    id<MTLComputePipelineState> pipelines[DITCH_MAX_PIPELINES];
};

static void ditch_set_err(char *err, size_t err_len, const char *msg) {
    if (!err || err_len == 0) return;
    strncpy(err, msg ? msg : "unknown error", err_len - 1);
    err[err_len - 1] = '\0';
}

ditch_mtl *ditch_mtl_open(const char *source, char *err, size_t err_len) {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            ditch_set_err(err, err_len, "no Metal device (MTLCreateSystemDefaultDevice returned nil)");
            return NULL;
        }
        id<MTLCommandQueue> queue = [device newCommandQueue];
        if (!queue) {
            [device release];
            ditch_set_err(err, err_len, "could not create a Metal command queue");
            return NULL;
        }
        NSError *nserr = nil;
        MTLCompileOptions *opts = [MTLCompileOptions new];
        // Keep the numerics close to the CPU reference. `mathMode` replaced
        // `fastMathEnabled` in the macOS 15 SDK; both spellings are compiled
        // conditionally so the shim builds against old and new SDKs alike.
#if defined(__MAC_OS_X_VERSION_MAX_ALLOWED) && __MAC_OS_X_VERSION_MAX_ALLOWED >= 150000
        opts.mathMode = MTLMathModeSafe;
#else
        opts.fastMathEnabled = NO;
#endif
        NSString *src = [NSString stringWithUTF8String:source];
        id<MTLLibrary> library = [device newLibraryWithSource:src options:opts error:&nserr];
        if (!library) {
            ditch_set_err(err, err_len, nserr ? [[nserr localizedDescription] UTF8String]
                                              : "could not compile the Metal shaders");
            [queue release];
            [device release];
            return NULL;
        }
        ditch_mtl *m = (ditch_mtl *)calloc(1, sizeof(ditch_mtl));
        if (!m) {
            ditch_set_err(err, err_len, "out of memory");
            return NULL;
        }
        m->device = device;
        m->queue = queue;
        m->library = library;
        // `MTLCreateSystemDefaultDevice` and the `new...` methods already
        // return owned references (this file is compiled without ARC).
        const char *nm = [[device name] UTF8String];
        strncpy(m->name, nm ? nm : "Metal device", sizeof(m->name) - 1);
        pthread_mutex_init(&m->lock, NULL);
        return m;
    }
}

void ditch_mtl_close(ditch_mtl *m) {
    if (!m) return;
    @autoreleasepool {
        for (int32_t i = 0; i < m->n_pipelines; ++i) {
            if (m->pipelines[i]) [m->pipelines[i] release];
        }
        [m->library release];
        [m->queue release];
        [m->device release];
    }
    pthread_mutex_destroy(&m->lock);
    free(m);
}

const char *ditch_mtl_name(ditch_mtl *m) {
    return m ? m->name : "";
}

uint64_t ditch_mtl_working_set(ditch_mtl *m) {
    if (!m) return 0;
    return (uint64_t)[m->device recommendedMaxWorkingSetSize];
}

int32_t ditch_mtl_pipeline(ditch_mtl *m, const char *name) {
    if (!m || !name) return -1;
    pthread_mutex_lock(&m->lock);
    for (int32_t i = 0; i < m->n_pipelines; ++i) {
        if (strncmp(m->pipeline_names[i], name, sizeof(m->pipeline_names[i])) == 0) {
            pthread_mutex_unlock(&m->lock);
            return i;
        }
    }
    int32_t index = -1;
    @autoreleasepool {
        if (m->n_pipelines < DITCH_MAX_PIPELINES) {
            id<MTLFunction> fn = [m->library newFunctionWithName:[NSString stringWithUTF8String:name]];
            if (fn) {
                NSError *nserr = nil;
                id<MTLComputePipelineState> pso = [m->device newComputePipelineStateWithFunction:fn error:&nserr];
                [fn release];
                if (pso) {
                    index = m->n_pipelines;
                    m->pipelines[index] = pso;
                    strncpy(m->pipeline_names[index], name, sizeof(m->pipeline_names[index]) - 1);
                    m->n_pipelines += 1;
                }
            }
        }
    }
    pthread_mutex_unlock(&m->lock);
    return index;
}

void *ditch_mtl_buffer(ditch_mtl *m, uint64_t bytes) {
    if (!m || bytes == 0) return NULL;
    id<MTLBuffer> buf = [m->device newBufferWithLength:(NSUInteger)bytes
                                               options:MTLResourceStorageModeShared];
    return (void *)buf;
}

void *ditch_mtl_buffer_nocopy(ditch_mtl *m, void *ptr, uint64_t bytes) {
    if (!m || !ptr || bytes == 0) return NULL;
    const size_t page = (size_t)getpagesize();
    if (((uintptr_t)ptr % page) != 0 || (bytes % page) != 0) return NULL;
    id<MTLBuffer> buf = [m->device newBufferWithBytesNoCopy:ptr
                                                     length:(NSUInteger)bytes
                                                    options:MTLResourceStorageModeShared
                                                deallocator:nil];
    return (void *)buf;
}

void *ditch_mtl_contents(void *buffer) {
    if (!buffer) return NULL;
    id<MTLBuffer> buf = (id<MTLBuffer>)buffer;
    return [buf contents];
}

void ditch_mtl_release(void *buffer) {
    if (!buffer) return;
    id<MTLBuffer> buf = (id<MTLBuffer>)buffer;
    [buf release];
}

uint32_t ditch_mtl_max_threads(ditch_mtl *m, int32_t pipeline) {
    if (!m || pipeline < 0 || pipeline >= m->n_pipelines) return 0;
    return (uint32_t)[m->pipelines[pipeline] maxTotalThreadsPerThreadgroup];
}

int32_t ditch_mtl_dispatch(ditch_mtl *m, int32_t pipeline,
                           void *const *buffers, int32_t n_buffers,
                           const void *params, uint64_t params_len,
                           uint32_t gx, uint32_t gy, uint32_t gz,
                           uint32_t tx, uint32_t ty, uint32_t tz) {
    if (!m || pipeline < 0 || pipeline >= m->n_pipelines) return -1;
    if (n_buffers < 0 || n_buffers > DITCH_MAX_BUFFERS) return -1;
    int32_t status = 0;
    pthread_mutex_lock(&m->lock);
    @autoreleasepool {
        id<MTLCommandBuffer> cmd = [m->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        if (!cmd || !enc) {
            pthread_mutex_unlock(&m->lock);
            return -1;
        }
        [enc setComputePipelineState:m->pipelines[pipeline]];
        for (int32_t i = 0; i < n_buffers; ++i) {
            [enc setBuffer:(id<MTLBuffer>)buffers[i] offset:0 atIndex:(NSUInteger)i];
        }
        if (params && params_len > 0) {
            [enc setBytes:params length:(NSUInteger)params_len atIndex:(NSUInteger)n_buffers];
        }
        MTLSize groups = MTLSizeMake(gx, gy, gz);
        MTLSize threads = MTLSizeMake(tx, ty, tz);
        [enc dispatchThreadgroups:groups threadsPerThreadgroup:threads];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
        if ([cmd status] != MTLCommandBufferStatusCompleted) status = -1;
    }
    pthread_mutex_unlock(&m->lock);
    return status;
}

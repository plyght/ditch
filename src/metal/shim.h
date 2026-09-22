// C surface of ditch's Metal backend. Implemented in shim.m (Objective-C,
// compiled by Zig's C compiler with -x objective-c, see build.zig) and called
// from backend.zig. Keeping the Objective-C runtime behind this header keeps
// the Zig side free of objc_msgSend casts.
//
// Every function is safe to call with a null context (it then fails cleanly);
// none of them raise Objective-C exceptions across the boundary.

#ifndef DITCH_METAL_SHIM_H
#define DITCH_METAL_SHIM_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ditch_mtl ditch_mtl;

/// Opens the default Metal device and compiles `source` with
/// newLibraryWithSource:. Returns NULL on failure and writes a NUL-terminated
/// message into `err` (which may be NULL).
ditch_mtl *ditch_mtl_open(const char *source, char *err, size_t err_len);

void ditch_mtl_close(ditch_mtl *m);

/// The device name (owned by the context, valid until ditch_mtl_close).
const char *ditch_mtl_name(ditch_mtl *m);

/// The device's recommended working-set size in bytes (0 if unknown).
uint64_t ditch_mtl_working_set(ditch_mtl *m);

/// Index of a compute pipeline by kernel function name, or -1. Pipelines are
/// built lazily and cached.
int32_t ditch_mtl_pipeline(ditch_mtl *m, const char *name);

/// A new shared-storage buffer of `bytes` (unified memory: the CPU writes and
/// the GPU reads the same pages). NULL on failure.
void *ditch_mtl_buffer(ditch_mtl *m, uint64_t bytes);

/// A shared-storage buffer aliasing `ptr` without copying, or NULL when the
/// pointer or length is not page aligned (the caller then copies instead).
void *ditch_mtl_buffer_nocopy(ditch_mtl *m, void *ptr, uint64_t bytes);

/// The CPU-visible contents of a buffer (shared storage, so no sync needed).
void *ditch_mtl_contents(void *buffer);

void ditch_mtl_release(void *buffer);

/// Encodes `pipeline` with `n_buffers` buffers bound at indices 0.., the
/// `params_len` bytes of `params` bound at index `n_buffers`, dispatches
/// `gx * gy * gz` threadgroups of `tx * ty * tz` threads and waits.
/// Returns 0 on success, -1 on failure.
int32_t ditch_mtl_dispatch(ditch_mtl *m, int32_t pipeline,
                           void *const *buffers, int32_t n_buffers,
                           const void *params, uint64_t params_len,
                           uint32_t gx, uint32_t gy, uint32_t gz,
                           uint32_t tx, uint32_t ty, uint32_t tz);

/// The largest threadgroup the pipeline supports (0 if unknown).
uint32_t ditch_mtl_max_threads(ditch_mtl *m, int32_t pipeline);

#ifdef __cplusplus
}
#endif

#endif // DITCH_METAL_SHIM_H

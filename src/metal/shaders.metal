// Metal compute kernels for ditch's hot paths. Compiled at runtime with
// newLibraryWithSource: (see shim.m), so no Apple toolchain is needed to build
// ditch itself; `xcrun -sdk macosx metal -c` in CI type-checks this file.
//
// Everything accumulates in f32, whatever the storage dtype of the weights, so
// the results match the CPU reference to f32 rounding. Weight tiles arrive as
// raw bytes in their on-disk dtype (f32, f16 or bf16) and are converted in the
// kernel: a tile is therefore uploaded once, never converted host-side and
// never copied twice.
//
// Tolerances: the summation order differs from the CPU kernels (which reduce
// 16 lanes at a time), so results agree to f32 rounding, not bitwise. The
// reference harness (`ditch selftest --device metal`) measures the actual
// error per kernel.

#include <metal_stdlib>
using namespace metal;

// ---------------------------------------------------------------------------
// Parameter blocks (must match the extern structs in backend.zig)
// ---------------------------------------------------------------------------

struct MatmulParams {
    uint n;     // input rows
    uint rows;  // weight rows (outputs)
    uint cols;  // shared dimension
};

struct MatvecParams {
    uint q;     // vectors
    uint rows;
    uint cols;
};

struct RowNormParams {
    uint rows;
    uint cols;
};

struct AttnScoreParams {
    uint keys;
    uint hd;
    uint stride;
    float scale;
};

struct AttnValueParams {
    uint keys;
    uint vd;
    uint stride;
};

struct GatedParams {
    uint act;
    uint n;
    uint len;
    uint in_stride;
    uint out_stride;
    uint has_up;
};

struct NormParams {
    uint n;
    uint len;
    float eps;
    uint flags; // bit 0: weighted, bit 1: (1 + w) scaling, bit 2: bias
};

struct SoftmaxParams {
    uint n;
    uint len;
};

struct RopeParams {
    uint n;
    uint dim;
    uint half_dim;
    uint style; // 0 = neox (rotate_half), 1 = gptj (interleaved pairs)
};

// ---------------------------------------------------------------------------
// Weight loads
// ---------------------------------------------------------------------------

inline float load_f32(device const float *p, uint i) { return p[i]; }
inline float load_f16(device const half *p, uint i) { return float(p[i]); }
inline float load_bf16(device const ushort *p, uint i) { return as_type<float>(uint(p[i]) << 16); }

// ---------------------------------------------------------------------------
// Tiled matmul: out[n][rows] = x[n][cols] @ W^T, f32 accumulation
// ---------------------------------------------------------------------------

#define TS 16

#define DEFINE_MATMUL(NAME, WTYPE, LOAD)                                        \
kernel void NAME(device float *out [[buffer(0)]],                               \
                 device const float *x [[buffer(1)]],                           \
                 device const WTYPE *w [[buffer(2)]],                           \
                 constant MatmulParams &p [[buffer(3)]],                        \
                 uint2 lid [[thread_position_in_threadgroup]],                  \
                 uint2 tgid [[threadgroup_position_in_grid]])                   \
{                                                                               \
    threadgroup float xs[TS][TS];                                               \
    threadgroup float ws[TS][TS];                                               \
    const uint r = tgid.x * TS + lid.x;                                         \
    const uint i = tgid.y * TS + lid.y;                                         \
    float acc = 0.0f;                                                           \
    const uint tiles = (p.cols + TS - 1) / TS;                                  \
    for (uint t = 0; t < tiles; ++t) {                                          \
        const uint kx = t * TS + lid.x;                                         \
        xs[lid.y][lid.x] = (i < p.n && kx < p.cols) ? x[i * p.cols + kx] : 0.0f; \
        const uint kw = t * TS + lid.y;                                         \
        ws[lid.x][lid.y] = (r < p.rows && kw < p.cols)                          \
            ? LOAD(w, r * p.cols + kw) : 0.0f;                                  \
        threadgroup_barrier(mem_flags::mem_threadgroup);                        \
        for (uint k = 0; k < TS; ++k) acc = fma(xs[lid.y][k], ws[lid.x][k], acc); \
        threadgroup_barrier(mem_flags::mem_threadgroup);                        \
    }                                                                           \
    if (r < p.rows && i < p.n) out[i * p.rows + r] = acc;                        \
}

DEFINE_MATMUL(matmul_f32, float, load_f32)
DEFINE_MATMUL(matmul_f16, half, load_f16)
DEFINE_MATMUL(matmul_bf16, ushort, load_bf16)

// ---------------------------------------------------------------------------
// out[q][cols] = y[q][rows] @ W
// ---------------------------------------------------------------------------

#define DEFINE_MATVEC(NAME, WTYPE, LOAD)                                        \
kernel void NAME(device float *out [[buffer(0)]],                               \
                 device const float *y [[buffer(1)]],                           \
                 device const WTYPE *w [[buffer(2)]],                           \
                 constant MatvecParams &p [[buffer(3)]],                        \
                 uint2 gid [[thread_position_in_grid]])                         \
{                                                                               \
    const uint c = gid.x;                                                       \
    const uint j = gid.y;                                                       \
    if (c >= p.cols || j >= p.q) return;                                        \
    float acc = 0.0f;                                                           \
    for (uint r = 0; r < p.rows; ++r) {                                         \
        const float yr = y[j * p.rows + r];                                     \
        if (yr != 0.0f) acc = fma(yr, LOAD(w, r * p.cols + c), acc);            \
    }                                                                           \
    out[j * p.cols + c] = acc;                                                  \
}

DEFINE_MATVEC(matvec_f32, float, load_f32)
DEFINE_MATVEC(matvec_f16, half, load_f16)
DEFINE_MATVEC(matvec_bf16, ushort, load_bf16)

// ---------------------------------------------------------------------------
// Row L2 norms: one threadgroup per row
// ---------------------------------------------------------------------------

#define RED 256

#define DEFINE_ROWNORM(NAME, WTYPE, LOAD)                                       \
kernel void NAME(device float *out [[buffer(0)]],                               \
                 device const WTYPE *w [[buffer(1)]],                           \
                 constant RowNormParams &p [[buffer(2)]],                       \
                 uint lid [[thread_position_in_threadgroup]],                   \
                 uint tgid [[threadgroup_position_in_grid]])                    \
{                                                                               \
    threadgroup float red[RED];                                                 \
    const uint r = tgid;                                                        \
    float acc = 0.0f;                                                           \
    if (r < p.rows) {                                                           \
        for (uint c = lid; c < p.cols; c += RED) {                              \
            const float v = LOAD(w, r * p.cols + c);                            \
            acc = fma(v, v, acc);                                               \
        }                                                                       \
    }                                                                           \
    red[lid] = acc;                                                             \
    threadgroup_barrier(mem_flags::mem_threadgroup);                            \
    for (uint s = RED / 2; s > 0; s >>= 1) {                                    \
        if (lid < s) red[lid] += red[lid + s];                                  \
        threadgroup_barrier(mem_flags::mem_threadgroup);                        \
    }                                                                           \
    if (lid == 0 && r < p.rows) out[r] = sqrt(red[0]);                          \
}

DEFINE_ROWNORM(row_norms_f32, float, load_f32)
DEFINE_ROWNORM(row_norms_f16, half, load_f16)
DEFINE_ROWNORM(row_norms_bf16, ushort, load_bf16)

// ---------------------------------------------------------------------------
// Attention
// ---------------------------------------------------------------------------

kernel void attn_scores(device float *scores [[buffer(0)]],
                        device const float *q [[buffer(1)]],
                        device const float *k [[buffer(2)]],
                        constant AttnScoreParams &p [[buffer(3)]],
                        uint gid [[thread_position_in_grid]])
{
    if (gid >= p.keys) return;
    float acc = 0.0f;
    device const float *kp = k + uint(gid) * p.stride;
    for (uint d = 0; d < p.hd; ++d) acc = fma(q[d], kp[d], acc);
    scores[gid] = acc * p.scale;
}

kernel void attn_values(device float *out [[buffer(0)]],
                        device const float *scores [[buffer(1)]],
                        device const float *v [[buffer(2)]],
                        constant AttnValueParams &p [[buffer(3)]],
                        uint gid [[thread_position_in_grid]])
{
    if (gid >= p.vd) return;
    float acc = 0.0f;
    for (uint i = 0; i < p.keys; ++i) acc = fma(scores[i], v[i * p.stride + gid], acc);
    out[gid] = acc;
}

// ---------------------------------------------------------------------------
// Gated activations
// ---------------------------------------------------------------------------

// Abramowitz-Stegun 7.1.26, the same approximation `tensor.erf` uses. Metal
// Shading Language has no `erf`, and matching the host polynomial term for
// term is what keeps the gelu kernel equal to the CPU one rather than merely
// close to it.
inline float ditch_erf(float x) {
    const float a = fabs(x);
    const float t = 1.0f / (1.0f + 0.3275911f * a);
    const float y = 1.0f - (((((1.061405429f * t - 1.453152027f) * t) + 1.421413741f) * t - 0.284496736f) * t + 0.254829592f) * t * exp(-x * x);
    return x >= 0.0f ? y : -y;
}

// Must match tensor.Activation's declaration order.
inline float activate(uint act, float x) {
    switch (act) {
        case 0: return x / (1.0f + exp(-x));                      // silu
        case 1: {                                                 // gelu_tanh
            const float u = 0.7978845608028654f * (x + 0.044715f * x * x * x);
            return 0.5f * x * (1.0f + tanh(u));
        }
        case 2: return 0.5f * x * (1.0f + ditch_erf(x * 0.7071067811865476f)); // gelu
        case 3: return fmax(x, 0.0f);                             // relu
        case 4: { const float r = fmax(x, 0.0f); return r * r; }  // relu2
        default: return x / (1.0f + exp(-1.702f * x));            // quick_gelu
    }
}

kernel void gated(device float *out [[buffer(0)]],
                  device const float *gate [[buffer(1)]],
                  device const float *up [[buffer(2)]],
                  constant GatedParams &p [[buffer(3)]],
                  uint2 gid [[thread_position_in_grid]])
{
    const uint j = gid.x;
    const uint row = gid.y;
    if (j >= p.len || row >= p.n) return;
    float v = activate(p.act, gate[row * p.in_stride + j]);
    if (p.has_up != 0) v *= up[row * p.in_stride + j];
    out[row * p.out_stride + j] = v;
}

// ---------------------------------------------------------------------------
// Normalisation and softmax: one threadgroup per row
// ---------------------------------------------------------------------------

kernel void rmsnorm_rows(device float *out [[buffer(0)]],
                         device const float *x [[buffer(1)]],
                         device const float *weight [[buffer(2)]],
                         constant NormParams &p [[buffer(3)]],
                         uint lid [[thread_position_in_threadgroup]],
                         uint tgid [[threadgroup_position_in_grid]])
{
    threadgroup float red[RED];
    const uint row = tgid;
    device const float *xr = x + uint(row) * p.len;
    device float *outr = out + uint(row) * p.len;
    float acc = 0.0f;
    for (uint i = lid; i < p.len; i += RED) acc = fma(xr[i], xr[i], acc);
    red[lid] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = RED / 2; s > 0; s >>= 1) {
        if (lid < s) red[lid] += red[lid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float inv = rsqrt(red[0] / float(p.len) + p.eps);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const bool weighted = (p.flags & 1u) != 0u;
    const bool one_plus = (p.flags & 2u) != 0u;
    for (uint i = lid; i < p.len; i += RED) {
        float v = xr[i] * inv;
        if (weighted) v *= one_plus ? (1.0f + weight[i]) : weight[i];
        outr[i] = v;
    }
}

kernel void layernorm_rows(device float *out [[buffer(0)]],
                           device const float *x [[buffer(1)]],
                           device const float *weight [[buffer(2)]],
                           device const float *bias [[buffer(3)]],
                           constant NormParams &p [[buffer(4)]],
                           uint lid [[thread_position_in_threadgroup]],
                           uint tgid [[threadgroup_position_in_grid]])
{
    threadgroup float red[RED];
    const uint row = tgid;
    device const float *xr = x + uint(row) * p.len;
    device float *outr = out + uint(row) * p.len;
    float acc = 0.0f;
    for (uint i = lid; i < p.len; i += RED) acc += xr[i];
    red[lid] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = RED / 2; s > 0; s >>= 1) {
        if (lid < s) red[lid] += red[lid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float mean = red[0] / float(p.len);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float vacc = 0.0f;
    for (uint i = lid; i < p.len; i += RED) {
        const float d = xr[i] - mean;
        vacc = fma(d, d, vacc);
    }
    red[lid] = vacc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = RED / 2; s > 0; s >>= 1) {
        if (lid < s) red[lid] += red[lid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float inv = 1.0f / sqrt(red[0] / float(p.len) + p.eps);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const bool weighted = (p.flags & 1u) != 0u;
    const bool one_plus = (p.flags & 2u) != 0u;
    const bool biased = (p.flags & 4u) != 0u;
    for (uint i = lid; i < p.len; i += RED) {
        float v = (xr[i] - mean) * inv;
        if (weighted) v *= one_plus ? (1.0f + weight[i]) : weight[i];
        if (biased) v += bias[i];
        outr[i] = v;
    }
}

kernel void softmax_rows(device float *x [[buffer(0)]],
                         constant SoftmaxParams &p [[buffer(1)]],
                         uint lid [[thread_position_in_threadgroup]],
                         uint tgid [[threadgroup_position_in_grid]])
{
    threadgroup float red[RED];
    device float *xr = x + uint(tgid) * p.len;
    float m = -INFINITY;
    for (uint i = lid; i < p.len; i += RED) m = fmax(m, xr[i]);
    red[lid] = m;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = RED / 2; s > 0; s >>= 1) {
        if (lid < s) red[lid] = fmax(red[lid], red[lid + s]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float mx = red[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float sum = 0.0f;
    for (uint i = lid; i < p.len; i += RED) {
        const float e = exp(xr[i] - mx);
        xr[i] = e;
        sum += e;
    }
    red[lid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = RED / 2; s > 0; s >>= 1) {
        if (lid < s) red[lid] += red[lid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float inv = 1.0f / red[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = lid; i < p.len; i += RED) xr[i] *= inv;
}

// ---------------------------------------------------------------------------
// Rotary embeddings
// ---------------------------------------------------------------------------

kernel void rope_rows(device float *x [[buffer(0)]],
                      device const float *cosv [[buffer(1)]],
                      device const float *sinv [[buffer(2)]],
                      device const uint *pos [[buffer(3)]],
                      constant RopeParams &p [[buffer(4)]],
                      uint2 gid [[thread_position_in_grid]])
{
    const uint i = gid.x;
    const uint row = gid.y;
    if (i >= p.half_dim || row >= p.n) return;
    device float *xr = x + uint(row) * p.dim;
    const uint base = pos[row] * p.half_dim;
    const float c = cosv[base + i];
    const float s = sinv[base + i];
    if (p.style == 0) {
        const float x1 = xr[i];
        const float x2 = xr[i + p.half_dim];
        xr[i] = x1 * c - x2 * s;
        xr[i + p.half_dim] = x2 * c + x1 * s;
    } else {
        const float x1 = xr[2 * i];
        const float x2 = xr[2 * i + 1];
        xr[2 * i] = x1 * c - x2 * s;
        xr[2 * i + 1] = x2 * c + x1 * s;
    }
}

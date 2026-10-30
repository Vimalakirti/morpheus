// Morpheus S&P'27: GPU kernels of the PCS authentication data path
// (mirrors zk-torch-4/src/morpheus/pcs_maps.rs and auth.rs).
//
//  * wtable / ev_partial / ev_reduce: Ev_raw via the eq tensor factorization,
//      P_m = Σ_{l,j} (e_l e_j) W_l[j + 64 m],  m = 0..63  (K coefficients);
//    the host finishes Ev = g_k(X)·P(X) (one 64x64 A product).
//  * decode: Σ_h 2^h col[h + 64 m] mod q for every column and m (H and E_O).
//  * combine: Z_r += β_{r,e} · V_e for both rows at once (S action X = Y^64
//    is a negacyclic 64-tap convolution along k of each j-slice).
//  * norms: per-block squared norm (exact 128-bit) and max |coefficient|.
#pragma once
#include <cstdint>
#include "almost_goldilocks.cuh"
#include "morpheus_k8.cuh"

namespace mauth {

typedef unsigned long long u64;
constexpr u64 Q = 18446744069414584289ULL;

struct K8d { u64 c[8]; };

__device__ __forceinline__ u64 fmul(u64 a, u64 b) { return agl_mul(AlmostGoldilocksField(a), AlmostGoldilocksField(b)).value; }
__device__ __forceinline__ u64 fadd(u64 a, u64 b) { return agl_add(AlmostGoldilocksField(a), AlmostGoldilocksField(b)).value; }
__device__ __forceinline__ u64 fcanon(u64 a) { return agl_canonicalize(a); }
__device__ __forceinline__ u64 to_field(int64_t v) { return v >= 0 ? (u64)v : Q - (u64)(-v); }

/// K = F_q[Θ]/(Θ^8 − 3) product (lazy reduction, canonical output).
__device__ __forceinline__ K8d k8_mul(const K8d& a, const K8d& b) {
    K8d r;
    mk8::mul(a.c, b.c, r.c);
    return r;
}

/// w[l][j] = e_l · e_j for l < cols, j < 64.
__global__ void wtable_kernel(const K8d* __restrict__ el, const K8d* __restrict__ ej, int cols, K8d* __restrict__ w) {
    for (size_t id = blockIdx.x * (size_t)blockDim.x + threadIdx.x; id < (size_t)cols * 64; id += (size_t)gridDim.x * blockDim.x)
        w[id] = k8_mul(el[id / 64], ej[id % 64]);
}

/// Per-block partial sums P_m over a grid-stride set of columns.
/// 256 threads: m = t % 64, quarter q = t / 64 handles j in [16q, 16q+16).
template <typename TW>
__global__ void ev_partial_kernel(const TW* __restrict__ W, int cols, const K8d* __restrict__ w, K8d* __restrict__ partial) {
    // The column tile and the final 4x64x8 reduction share one buffer
    // (int64 tile 33,280 B; reduction 16,384 B), keeping the int64 variant
    // under the 48 KB static limit.
    constexpr size_t TILE = sizeof(TW) * 64 * 65, RED = 4 * 64 * 8 * 8;
    __shared__ __align__(16) unsigned char raw[TILE > RED ? TILE : RED];
    TW* s = reinterpret_cast<TW*>(raw);
    __shared__ K8d sw[64];
    const int t = threadIdx.x, m = t & 63, qtr = t >> 6;
    u64 acc[8];
#pragma unroll
    for (int a = 0; a < 8; a++) acc[a] = 0;
    for (int l = blockIdx.x; l < cols; l += gridDim.x) {
        const TW* col = W + (size_t)l * 4096;
        for (int i = t; i < 4096; i += 256) s[(i >> 6) * 65 + (i & 63)] = col[i];  // phys i = j + 64 m -> s[m][j]
        if (t < 64) sw[t] = w[(size_t)l * 64 + t];
        __syncthreads();
        for (int j = qtr * 16; j < qtr * 16 + 16; j++) {
            int64_t v = (int64_t)s[m * 65 + j];
            if (v == 0) continue;
            u64 x = to_field(v);
#pragma unroll
            for (int a = 0; a < 8; a++) acc[a] = fadd(acc[a], fmul(sw[j].c[a], x));
        }
        __syncthreads();
    }
    u64 (*red)[64][8] = reinterpret_cast<u64 (*)[64][8]>(raw);
#pragma unroll
    for (int a = 0; a < 8; a++) red[qtr][m][a] = acc[a];
    __syncthreads();
    if (qtr == 0) {
        K8d o;
#pragma unroll
        for (int a = 0; a < 8; a++) o.c[a] = fcanon(fadd(fadd(red[0][m][a], red[1][m][a]), fadd(red[2][m][a], red[3][m][a])));
        partial[(size_t)blockIdx.x * 64 + m] = o;
    }
}

/// out[m] = Σ_b partial[b][m]; 512 threads = 64 m x 8 coordinates.
__global__ void ev_reduce_kernel(const K8d* __restrict__ partial, int blocks, K8d* __restrict__ out) {
    int t = threadIdx.x, m = t >> 3, a = t & 7;
    u64 s = 0;
    for (int b = 0; b < blocks; b++) s = fadd(s, partial[(size_t)b * 64 + m].c[a]);
    out[m].c[a] = fcanon(s);
}

/// out[c][m] = Σ_h 2^h col_c[h + 64 m] mod q (int64 columns).
__global__ void decode_kernel(const int64_t* __restrict__ cols, int ncols, u64* __restrict__ out) {
    for (size_t id = blockIdx.x * (size_t)blockDim.x + threadIdx.x; id < (size_t)ncols * 64; id += (size_t)gridDim.x * blockDim.x) {
        size_t c = id / 64; int m = (int)(id % 64);
        const int64_t* col = cols + c * 4096 + 64 * m;
        u64 s = 0, p2 = 1;
        for (int h = 0; h < 64; h++) {
            int64_t v = col[h];
            if (v) s = fadd(s, fmul(to_field(v), p2));
            p2 = fadd(p2, p2);
        }
        out[id] = fcanon(s);
    }
}

/// Z_r[col][i] += Σ_m β_r[m] · sgn · V[col][i − 64 m (mod 4096)], r = 0, 1.
/// One block per column; V is int8 (source/link/aux witnesses are digits).
__global__ void combine_kernel(const int8_t* __restrict__ V, int cols, const int8_t* __restrict__ beta2 /* [2][64] */,
                               int64_t* __restrict__ Z0, int64_t* __restrict__ Z1) {
    __shared__ int8_t s[4096];
    __shared__ int8_t b[2][64];
    if (threadIdx.x < 128) b[threadIdx.x >> 6][threadIdx.x & 63] = beta2[threadIdx.x];
    for (int l = blockIdx.x; l < cols; l += gridDim.x) {
        const int8_t* col = V + (size_t)l * 4096;
        for (int i = threadIdx.x; i < 4096; i += blockDim.x) s[i] = col[i];
        __syncthreads();
        for (int i = threadIdx.x; i < 4096; i += blockDim.x) {
            int j = i & 63, k = i >> 6;
            int a0 = 0, a1 = 0;
#pragma unroll 8
            for (int m = 0; m < 64; m++) {
                int kk = k - m;
                int v = kk >= 0 ? s[j + 64 * kk] : -s[j + 64 * (kk + 64)];
                a0 += b[0][m] * v;
                a1 += b[1][m] * v;
            }
            Z0[(size_t)l * 4096 + i] += a0;
            Z1[(size_t)l * 4096 + i] += a1;
        }
        __syncthreads();
    }
}

/// Wire codec of the responses: v ∈ (−B_Z, B_Z) is sent as the 37-bit
/// unsigned u = v + (B_Z − 1), LSB first, eight coefficients per 37 bytes.
__global__ void pack37_kernel(const int64_t* __restrict__ z, u64 groups, int64_t off, uint8_t* __restrict__ out) {
    for (u64 g = blockIdx.x * (u64)blockDim.x + threadIdx.x; g < groups; g += (u64)gridDim.x * blockDim.x) {
        u64 w[5] = {0, 0, 0, 0, 0};
        for (int t = 0; t < 8; t++) {
            u64 u = (u64)(z[g * 8 + t] + off) & ((1ULL << 37) - 1);
            int bp = 37 * t, i = bp >> 6, sh = bp & 63;
            w[i] |= u << sh;
            if (sh + 37 > 64) w[i + 1] |= u >> (64 - sh);
        }
        uint8_t* o = out + g * 37;
        for (int b = 0; b < 37; b++) o[b] = (uint8_t)(w[b >> 3] >> (8 * (b & 7)));
    }
}

/// Inverse of pack37; a code above `umax` (= 2 B_Z − 2) sets *bad.
__global__ void unpack37_kernel(const uint8_t* __restrict__ in, u64 groups, int64_t off, u64 umax, int64_t* __restrict__ z, int* __restrict__ bad) {
    for (u64 g = blockIdx.x * (u64)blockDim.x + threadIdx.x; g < groups; g += (u64)gridDim.x * blockDim.x) {
        u64 w[5] = {0, 0, 0, 0, 0};
        const uint8_t* p = in + g * 37;
        for (int b = 0; b < 37; b++) w[b >> 3] |= (u64)p[b] << (8 * (b & 7));
        for (int t = 0; t < 8; t++) {
            int bp = 37 * t, i = bp >> 6, sh = bp & 63;
            u64 u = w[i] >> sh;
            if (sh + 37 > 64) u |= w[i + 1] << (64 - sh);
            u &= (1ULL << 37) - 1;
            if (u > umax) *bad = 1;
            z[g * 8 + t] = (int64_t)u - off;
        }
    }
}

/// Per-block partial squared norms (lo, hi words of a 128-bit sum) and max |v|.
__global__ void norm_kernel(const int64_t* __restrict__ z, size_t n, u64* __restrict__ part /* [grid][3] */) {
    u64 lo = 0, hi = 0, mx = 0;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        int64_t v = z[i];
        u64 a = (u64)(v < 0 ? -v : v);
        u64 sl = a * a, sh = __umul64hi(a, a);
        u64 nl = lo + sl; hi += sh + (nl < lo); lo = nl;
        mx = a > mx ? a : mx;
    }
    __shared__ u64 L[256], H[256], M[256];
    L[threadIdx.x] = lo; H[threadIdx.x] = hi; M[threadIdx.x] = mx;
    __syncthreads();
    if (threadIdx.x == 0) {
        u64 rl = 0, rh = 0, rm = 0;
        for (int t = 0; t < blockDim.x; t++) { u64 nl = rl + L[t]; rh += H[t] + (nl < rl); rl = nl; rm = M[t] > rm ? M[t] : rm; }
        part[blockIdx.x * 3] = rl; part[blockIdx.x * 3 + 1] = rh; part[blockIdx.x * 3 + 2] = rm;
    }
}

/// Logical raw digits -> physical columns: dst[c·4096 + perm(l)] = src[c·4096 + l],
/// perm(l) = j + 64k for l = 64j + k.
__global__ void permute_kernel(const int8_t* __restrict__ src, int8_t* __restrict__ dst, size_t n) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        size_t c = i >> 12; int l = (int)(i & 4095);
        int j = l >> 6, k = l & 63;
        dst[(c << 12) + j + 64 * k] = src[i];
    }
}

}  // namespace mauth

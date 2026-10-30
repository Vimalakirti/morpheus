// Morpheus S&P'27: structured Π_link prover, GPU v2.
//
// Same rounds as zk-torch-4/src/morpheus/link_fast.rs (the CPU reference);
// the work is organised by region instead of by (block, term):
//   * per-block constants come from split tables: c_alpha(id) =
//     CA_lo[id_lo]·CA_hi[id_hi]; a term's high-index factor hi_T(k) =
//     HL_T[k_lo]·HH_T[k_hi] (k = block index inside the region);
//   * one-hot regions (lookup selectors) in their plane rounds (i < p): an
//     index with selected plane t* has folded value e = E_i[t* mod 2^i] on the
//     side bit_i(t*) of its pair, so with λ_0 = 1 − t, λ_1 = t,
//       Q(λe) = λ³e³ − λe,  λe(λe − 1) = λ²e² − λe,
//     and per index only ea·e, ea·e², ea·e³ (and the query terms) are
//     accumulated, split by bit_i(t*);
//   * other rounds: folded digits (and dense query tables) are rebuilt in
//     shared memory, nonzero pairs are compacted, warp t evaluates point t;
//   * coarse rounds (i ≥ B0) run on dense per-source K tables.
#pragma once
#include <cstdint>
#include "morpheus_link.cuh"

namespace mlink2 {

using mauth::K8d;
using mlink::kzero; using mlink::kadd; using mlink::ksub; using mlink::kzero_p; using mlink::kcanon; using mlink::shfl_down_k;
typedef unsigned long long u64;
constexpr int MAXD = 4;
constexpr int NACC = 2 * (MAXD + 1);
constexpr int MAXT = 32;          // terms per region

struct Region {
    uint32_t src, base_blk, h, p;      // h: block-index bits inside the region
    uint32_t flags, toff, tcnt, qd_off; // flags: 1 chi, 2 one-hot, 4 dense-q; qd_off: start of this region's entries in qidx
};
struct Term {
    uint32_t pa, aoff, boff, hl_off, hh_off, h1, pad0, pad1;
    K8d cs;                             // coef·scale
};
struct Blk { uint32_t src, id, region, k; };   // region 0xffffffff: no terms, no flags

__device__ __forceinline__ K8d kmul(const K8d& a, const K8d& b) { return mauth::k8_mul(a, b); }
__device__ __forceinline__ K8d kscale(const K8d& a, u64 s) { K8d r; for (int j = 0; j < 8; j++) r.c[j] = mauth::fmul(a.c[j], s); return r; }

struct Consts {
    const K8d* ca_lo; const K8d* ca_hi; int ca_lo_bits;
    const K8d* g_range; const K8d* g_bit;
    const Region* regions; const Term* terms; const K8d* hl; const K8d* hh;
    const K8d* A; const K8d* B;            // per-round term tables
    const K8d* qd;                         // dense query tables (level 0: 2^b0 K each; rounds: folded to level i)
    const uint32_t* qidx;                  // per region block: table number or 0xffffffff
    const u64* bco;                        // per block (gen_all order): offset of its constants in bc (K units)
    u64 oh_base;                           // index of the first one-hot block in gen_all (one-hot pre pass)
};

/// Per-block constants: c_alpha·γ (cr), c_alpha·γ'·χ (cb), term factors c_T.
__device__ __forceinline__ void block_consts(const Consts& C, const Blk& b, K8d& cr, K8d& cb, K8d* ct, int& tcnt, uint32_t& flags) {
    const uint32_t mlo = (1u << C.ca_lo_bits) - 1;
    K8d ca = kmul(C.ca_lo[b.id & mlo], C.ca_hi[b.id >> C.ca_lo_bits]);
    cr = kmul(ca, C.g_range[b.src]);
    flags = 0; tcnt = 0;
    if (b.region != 0xffffffffu) {
        const Region R = C.regions[b.region];
        flags = R.flags;
        tcnt = (int)R.tcnt;
        for (int t = 0; t < tcnt; t++) {
            const Term T = C.terms[R.toff + t];
            uint32_t klo = b.k & ((1u << T.h1) - 1), khi = b.k >> T.h1;
            ct[t] = kmul(T.cs, kmul(C.hl[T.hl_off + klo], C.hh[T.hh_off + khi]));
        }
    }
    cb = (flags & 1) ? kmul(ca, C.g_bit[b.src]) : kzero();
}

// ------------------------------------------------------------- constants
/// Per-block constants, once per link: bc[b·(2+MAXT)+0] = cr, +1 = cb,
/// +2+k = c_T for the region's k-th term (k < tcnt ≤ MAXT).
__global__ void block_consts_kernel(const Blk* __restrict__ blks, uint32_t nblk, Consts C, K8d* __restrict__ bc) {
    for (uint32_t bi = blockIdx.x * blockDim.x + threadIdx.x; bi < nblk; bi += gridDim.x * blockDim.x) {
        const Blk b = blks[bi];
        K8d* o = bc + C.bco[bi];
        int tc; uint32_t fl;
        block_consts(C, b, o[0], o[1], o + 2, tc, fl);
    }
}

// ------------------------------------------------------------- one-hot
/// Plane round i < p of one-hot regions. Output per CUDA block: NACC K.
__global__ void __launch_bounds__(256) onehot_kernel(
    const int8_t* __restrict__ raw, uint64_t src_len, int b0, int i, int d,
    const Blk* __restrict__ blks, uint32_t nblk, Consts C,
    const K8d* __restrict__ E1, const K8d* __restrict__ E2, const K8d* __restrict__ E3,
    const K8d* __restrict__ e_alpha, K8d* __restrict__ partial)
{
    __shared__ K8d s_ct[MAXT];
    __shared__ K8d s_cr, s_cb;
    __shared__ int s_tcnt, s_p;
    __shared__ uint32_t s_toff;
    __shared__ K8d red[8][10];
    __shared__ K8d tot[NACC];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (threadIdx.x < NACC) tot[threadIdx.x] = kzero();
    for (uint32_t bi = blockIdx.x; bi < nblk; bi += gridDim.x) {
        const Blk b = blks[bi];
        if (threadIdx.x == 0) {
            int tc; uint32_t fl;
            K8d cr, cb;
            block_consts(C, b, cr, cb, s_ct, tc, fl);
            s_cr = cr; s_cb = cb; s_tcnt = tc;
            const Region R = C.regions[b.region];
            s_p = (int)R.p; s_toff = R.toff;
        }
        __syncthreads();
        const int p = s_p;
        const int nidx = 1 << (b0 - p);
        const int8_t* dg = raw + (uint64_t)b.src * src_len + ((uint64_t)b.id << b0);
        K8d acc[10];  // S1[2], S2[2], S3[2], P0[2], P1[2]
        for (int k = 0; k < 10; k++) acc[k] = kzero();
        for (int idx = threadIdx.x; idx < nidx; idx += blockDim.x) {
            const int8_t* pl = dg + ((size_t)idx << p);
            int ts = -1;
            for (int q = 0; q < (1 << p); q++) if (pl[q] == 1) { ts = q; break; }
            if (ts < 0) continue;
            const int bit = (ts >> i) & 1, low = ts & ((1 << i) - 1), pr = ts >> (i + 1);
            const int pa = p - i - 1;
            const int rest = pr + (idx << pa);
            const K8d ea = e_alpha[rest];
            const K8d e = E1[low];
            acc[0 + bit] = kadd(acc[0 + bit], kmul(ea, e));
            acc[2 + bit] = kadd(acc[2 + bit], kmul(ea, E2[low]));
            acc[4 + bit] = kadd(acc[4 + bit], kmul(ea, E3[low]));
            K8d qa = kzero(), qd = kzero();
            for (int t = 0; t < s_tcnt; t++) {
                const Term T = C.terms[s_toff + t];
                K8d bv = kmul(C.B[T.boff + idx], s_ct[t]);
                K8d a0 = C.A[T.aoff + pr], a1 = C.A[T.aoff + (1 << T.pa) + pr];
                qa = kadd(qa, kmul(a0, bv));
                qd = kadd(qd, kmul(ksub(a1, a0), bv));
            }
            acc[6 + bit] = kadd(acc[6 + bit], kmul(qa, e));
            acc[8 + bit] = kadd(acc[8 + bit], kmul(qd, e));
        }
        for (int k = 0; k < 10; k++) {
            K8d v = kcanon(acc[k]);
            for (int off = 16; off > 0; off >>= 1) v = kadd(v, shfl_down_k(v, off));
            if (lane == 0) red[warp][k] = kcanon(v);
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            K8d S[10];
            for (int k = 0; k < 10; k++) { S[k] = kzero(); for (int w = 0; w < (int)(blockDim.x >> 5); w++) S[k] = kadd(S[k], red[w][k]); }
            for (int t = 0; t <= d; t++) {
                // λ_0 = 1 − t, λ_1 = t as field elements.
                u64 l[2] = { t == 0 ? 1ULL : (t == 1 ? 0ULL : mauth::Q - (u64)(t - 1)), (u64)t };
                K8d r1 = kzero(), r2 = kzero(), qv = kzero();
                for (int bt = 0; bt < 2; bt++) {
                    u64 l1 = l[bt], l2 = mauth::fmul(l1, l1), l3 = mauth::fmul(l2, l1);
                    // cr·(λ³S3 − λS1) + cb·(λ²S2 − λS1)
                    r1 = kadd(r1, ksub(kscale(S[4 + bt], l3), kscale(S[0 + bt], l1)));
                    r2 = kadd(r2, ksub(kscale(S[2 + bt], l2), kscale(S[0 + bt], l1)));
                    qv = kadd(qv, kscale(kadd(S[6 + bt], kscale(S[8 + bt], (u64)t)), l1));
                }
                tot[t] = kadd(tot[t], kadd(kmul(s_cr, r1), kmul(s_cb, r2)));
                tot[d + 1 + t] = kadd(tot[d + 1 + t], qv);
            }
        }
        __syncthreads();
    }
    if (threadIdx.x < NACC) partial[(size_t)blockIdx.x * NACC + threadIdx.x] = kcanon(tot[threadIdx.x]);
}

// ------------------------------------------------------------- generic
/// Generic block-local round, blockDim = 32·(d+1), warp t evaluates point t.
/// A data block's pairs are processed in chunks of GCH:
///   fold the chunk's digits into F (i ≥ 1; round 0 reads digits directly),
///   compact the nonzero pairs, build their query weights QF at t = 0, 1 once
///   (level-i dense table + Σ_T c_T B_T[ir] A_T[side][pr]; all weights are
///   multilinear, so linear in t), then evaluate.
/// The range constants cr, cb are applied once per block per lane:
///   Σ ea·(cr(w³ − w) + cb(w² − w)) = cr·Σ ea(w³ − w) + cb·Σ ea(w² − w).
/// Reduced mode (blockDim 96): warps evaluate t = 0, 2, 3 (range) and t = 0, 2
/// (query); the host gets t = 1 from the previous round's claims.
constexpr int GCH = 128;
__global__ void __launch_bounds__(160, 3) generic_kernel(
    const int8_t* __restrict__ raw, uint64_t src_len, int b0, int i, int d, int reduced,
    const Blk* __restrict__ blks, uint32_t nblk, Consts C, const K8d* __restrict__ bc,
    const K8d* __restrict__ e_pre, const K8d* __restrict__ e_alpha, K8d* __restrict__ partial)
{
    __shared__ K8d F[2 * GCH], QF[2 * GCH];
    __shared__ uint16_t nz[GCH];
    __shared__ K8d s_ct[MAXT];
    __shared__ K8d s_cr, s_cb;
    __shared__ int s_tcnt, s_cnt;
    __shared__ uint32_t s_toff, s_qd;
    const int ny = 1 << (b0 - i), nrest = ny >> 1;
    const int lane = threadIdx.x & 31, wp = threadIdx.x >> 5;
    const int t = reduced ? (wp == 0 ? 0 : wp + 1) : wp;
    const bool doq = !reduced || t < 3;
    const u64 tq = (u64)t;
    K8d acc_r = kzero(), acc_q = kzero();
    if (threadIdx.x == 0) s_cnt = 0;
    for (uint32_t bi = blockIdx.x; bi < nblk; bi += gridDim.x) {
        const Blk b = blks[bi];
        if (threadIdx.x == 0) {
            uint32_t fl = 0; int tc = 0; uint32_t toff = 0, qdo = 0;
            if (b.region != 0xffffffffu) { const Region R = C.regions[b.region]; fl = R.flags; tc = (int)R.tcnt; toff = R.toff; qdo = R.qd_off; }
            s_tcnt = tc; s_toff = toff;
            uint32_t qi = (fl & 4) ? C.qidx[qdo + b.k] : 0xffffffffu;
            s_qd = qi == 0xffffffffu ? 0xffffffffu : (qi << (b0 - i));   // level-i folded tables
        }
        __syncthreads();
        {
            const K8d* o = bc + C.bco[bi];
            if (threadIdx.x == 0) s_cr = o[0];
            if (threadIdx.x == 1) s_cb = o[1];
            for (int k = (int)threadIdx.x - 2; k < s_tcnt; k += blockDim.x) if (k >= 0) s_ct[k] = o[2 + k];
        }
        const int8_t* dg = raw + (uint64_t)b.src * src_len + ((uint64_t)b.id << b0);
        const int tcnt = s_tcnt;
        const uint32_t toff = s_toff, qd = s_qd;
        K8d b3 = kzero(), b2 = kzero();          // this lane's Σ ea(w³ − w), Σ ea(w² − w) in the block
        for (int c0 = 0; c0 < nrest; c0 += GCH) {
            const int cp = nrest - c0 < GCH ? nrest - c0 : GCH, cy = 2 * cp, y0 = 2 * c0;
            if (i > 0) {
                // A group of G threads (G ≤ 32, inside a warp) shares one y.
                int G = 1;
                while (G < 32 && G * cy < (int)blockDim.x && G < (1 << i)) G <<= 1;
                const int groups = blockDim.x / G;
                const int gid = threadIdx.x / G, gl = threadIdx.x % G;
                for (int yr = 0; yr < cy; yr += groups) {
                    const int yl = yr + gid;
                    const bool live = yl < cy;
                    K8d sacc = kzero();
                    if (live) {
                        const int8_t* p0 = dg + ((size_t)(y0 + yl) << i);
                        for (int p = gl; p < (1 << i); p += G) { int8_t v = p0[p]; if (v == 1) sacc = kadd(sacc, e_pre[p]); else if (v == -1) sacc = ksub(sacc, e_pre[p]); }
                    }
                    if (G > 1) {
                        sacc = kcanon(sacc);
                        for (int off = G >> 1; off > 0; off >>= 1) {
                            K8d x;
                            for (int w = 0; w < 8; w++) x.c[w] = __shfl_down_sync(0xffffffffu, sacc.c[w], off, G);
                            sacc = kadd(sacc, x);
                        }
                    }
                    if (live && gl == 0) F[yl] = kcanon(sacc);
                }
                __syncthreads();
            }
            // Compact the nonzero pairs.
            for (int r = threadIdx.x; r < cp; r += blockDim.x) {
                bool z;
                if (i == 0) z = dg[y0 + 2 * r] == 0 && dg[y0 + 2 * r + 1] == 0;
                else z = kzero_p(F[2 * r]) && kzero_p(F[2 * r + 1]);
                if (!z) { int pos = atomicAdd(&s_cnt, 1); nz[pos] = (uint16_t)r; }
            }
            __syncthreads();
            const int cnt = s_cnt;
            // Query weights of the compacted pairs at t = 0, 1.
            if (qd != 0xffffffffu || tcnt > 0) {
                for (int j = threadIdx.x; j < cnt; j += blockDim.x) {
                    const int rest = c0 + nz[j];
                    K8d q0 = qd != 0xffffffffu ? C.qd[qd + 2 * rest] : kzero();
                    K8d q1 = qd != 0xffffffffu ? C.qd[qd + 2 * rest + 1] : kzero();
                    for (int k = 0; k < tcnt; k++) {
                        const Term T = C.terms[toff + k];
                        const int pr = rest & ((1 << T.pa) - 1), ir = rest >> T.pa;
                        const K8d bv = kmul(C.B[T.boff + ir], s_ct[k]);
                        q0 = kadd(q0, kmul(C.A[T.aoff + pr], bv));
                        q1 = kadd(q1, kmul(C.A[T.aoff + (1 << T.pa) + pr], bv));
                    }
                    QF[2 * j] = kcanon(q0); QF[2 * j + 1] = kcanon(q1);
                }
                __syncthreads();
            }
            const bool hasq = (qd != 0xffffffffu || tcnt > 0) && doq;
            {
                for (int j = lane; j < cnt; j += 32) {
                    const int rl = nz[j], rest = c0 + rl;
                    const K8d ea = e_alpha[rest];
                    K8d qt = kzero();
                    if (hasq) {
                        const K8d q0 = QF[2 * j], dq = ksub(QF[2 * j + 1], q0);
                        for (int a = 0; a < 8; a++) qt.c[a] = mauth::fadd(q0.c[a], mauth::fmul(dq.c[a], tq));
                    }
                    if (i == 0) {
                        // Digits: w_t = a + t (c − a) is a small integer.
                        const long long a = dg[y0 + 2 * rl], c = dg[y0 + 2 * rl + 1];
                        const long long wv = a + (long long)t * (c - a);
                        auto fe = [](long long x) -> u64 { return x >= 0 ? (u64)x : mauth::Q - (u64)(-x); };
                        if (wv != 0) {
                            b3 = kadd(b3, kscale(ea, fe(wv * wv * wv - wv)));
                            b2 = kadd(b2, kscale(ea, fe(wv * wv - wv)));
                            if (hasq) acc_q = kadd(acc_q, kscale(qt, fe(wv)));
                        }
                    } else {
                        const K8d w0 = F[2 * rl], dw = ksub(F[2 * rl + 1], w0);
                        K8d wt;
                        for (int a = 0; a < 8; a++) wt.c[a] = mauth::fadd(w0.c[a], mauth::fmul(dw.c[a], tq));
                        const K8d w2 = kmul(wt, wt), w3 = kmul(w2, wt);
                        b3 = kadd(b3, kmul(ea, ksub(w3, wt)));
                        b2 = kadd(b2, kmul(ea, ksub(w2, wt)));
                        if (hasq) acc_q = kadd(acc_q, kmul(qt, wt));
                    }
                }
            }
            __syncthreads();
            if (threadIdx.x == 0) s_cnt = 0;
            __syncthreads();
        }
        if (!(kzero_p(b3) && kzero_p(b2))) acc_r = kadd(acc_r, kadd(kmul(s_cr, b3), kmul(s_cb, b2)));
        __syncthreads();
    }
    K8d vr = kcanon(acc_r), vq = kcanon(acc_q);
    for (int off = 16; off > 0; off >>= 1) { vr = kadd(vr, shfl_down_k(vr, off)); vq = kadd(vq, shfl_down_k(vq, off)); }
    if (lane == 0) {
        partial[(size_t)blockIdx.x * NACC + t] = kcanon(vr);
        if (doq) partial[(size_t)blockIdx.x * NACC + d + 1 + t] = kcanon(vq);
    }
}

// ------------------------------------------------------------- coarse init
/// w_coarse[src][id] = Σ_p e_full[p]·digit[p] for each nonzero block.
__global__ void __launch_bounds__(256) fold_kernel(
    const int8_t* __restrict__ raw, uint64_t src_len, int b0, const Blk* __restrict__ blks, uint32_t nblk,
    const K8d* __restrict__ e_full, uint64_t coarse, K8d* __restrict__ w)
{
    __shared__ K8d wsum[8];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    for (uint32_t bi = blockIdx.x; bi < nblk; bi += gridDim.x) {
        const Blk b = blks[bi];
        const int8_t* dg = raw + (uint64_t)b.src * src_len + ((uint64_t)b.id << b0);
        K8d s = kzero();
        for (int p = threadIdx.x; p < (1 << b0); p += blockDim.x) { int8_t v = dg[p]; if (v == 1) s = kadd(s, e_full[p]); else if (v == -1) s = ksub(s, e_full[p]); }
        s = kcanon(s);
        for (int off = 16; off > 0; off >>= 1) s = kadd(s, shfl_down_k(s, off));
        if (lane == 0) wsum[warp] = kcanon(s);
        __syncthreads();
        if (threadIdx.x == 0) { K8d tt = kzero(); for (int x = 0; x < (int)(blockDim.x >> 5); x++) tt = kadd(tt, wsum[x]); w[(uint64_t)b.src * coarse + b.id] = kcanon(tt); }
        __syncthreads();
    }
}

/// q_coarse and χ_coarse for every block of every region (thread per block).
/// Block g of the flattened list belongs to region rid[g] with index k[g].
__global__ void coarse_q_kernel(Consts C, const uint32_t* __restrict__ rid, const uint32_t* __restrict__ kk, uint64_t n,
                                const K8d* __restrict__ tfold, const K8d* __restrict__ e_full, int b0, uint64_t coarse,
                                K8d* __restrict__ q, K8d* __restrict__ chi)
{
    for (uint64_t g = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x; g < n; g += (uint64_t)gridDim.x * blockDim.x) {
        const Region R = C.regions[rid[g]];
        const uint32_t k = kk[g];
        K8d acc = kzero();
        for (uint32_t t = 0; t < R.tcnt; t++) {
            const Term T = C.terms[R.toff + t];
            uint32_t klo = k & ((1u << T.h1) - 1), khi = k >> T.h1;
            K8d c = kmul(T.cs, kmul(C.hl[T.hl_off + klo], C.hh[T.hh_off + khi]));
            acc = kadd(acc, kmul(c, tfold[R.toff + t]));
        }
        uint32_t qi = (R.flags & 4) ? C.qidx[R.qd_off + k] : 0xffffffffu;
        if (qi != 0xffffffffu) {
            const K8d* qb = C.qd + ((uint64_t)qi << b0);
            for (int p = 0; p < (1 << b0); p++) if (!kzero_p(qb[p])) acc = kadd(acc, kmul(qb[p], e_full[p]));
        }
        const uint64_t at = (uint64_t)R.src * coarse + R.base_blk + k;
        q[at] = kcanon(acc);
        K8d c1 = kzero(); if (R.flags & 1) c1.c[0] = 1;
        chi[at] = c1;
    }
}

// ------------------------------------------------------------- coarse rounds
/// One coarse round over len-entry tables per source (pairs k: (2k, 2k+1)).
__global__ void __launch_bounds__(160) coarse_round_kernel(
    const K8d* __restrict__ w, const K8d* __restrict__ q, const K8d* __restrict__ chi, uint32_t nsrc, uint64_t len,
    const K8d* __restrict__ g_range, const K8d* __restrict__ g_bit, const K8d* __restrict__ e_alpha, int d,
    K8d* __restrict__ partial)
{
    const int lane = threadIdx.x & 31, t = threadIdx.x >> 5;
    const u64 tq = (u64)t;
    K8d acc_r = kzero(), acc_q = kzero();
    const uint64_t half = len >> 1, total = half * nsrc;
    if (t <= d) {
        for (uint64_t g = blockIdx.x * 32ULL + lane; g < total; g += (uint64_t)gridDim.x * 32) {
            const uint64_t s = g / half, k = g % half;
            const K8d* ws = w + s * len;
            K8d w0 = ws[2 * k], w1 = ws[2 * k + 1];
            if (kzero_p(w0) && kzero_p(w1)) continue;
            const K8d* qs = q + s * len; const K8d* cs = chi + s * len;
            K8d dw = ksub(w1, w0), dq = ksub(qs[2 * k + 1], qs[2 * k]), dc = ksub(cs[2 * k + 1], cs[2 * k]);
            K8d wt, qt, ct;
            for (int a = 0; a < 8; a++) {
                wt.c[a] = mauth::fadd(w0.c[a], mauth::fmul(dw.c[a], tq));
                qt.c[a] = mauth::fadd(qs[2 * k].c[a], mauth::fmul(dq.c[a], tq));
                ct.c[a] = mauth::fadd(cs[2 * k].c[a], mauth::fmul(dc.c[a], tq));
            }
            K8d w2 = kmul(wt, wt), w3 = kmul(w2, wt);
            K8d r = kadd(kmul(g_range[s], ksub(w3, wt)), kmul(kmul(g_bit[s], ct), ksub(w2, wt)));
            acc_r = kadd(acc_r, kmul(e_alpha[k], r));
            acc_q = kadd(acc_q, kmul(qt, wt));
        }
    }
    K8d vr = kcanon(acc_r), vq = kcanon(acc_q);
    for (int off = 16; off > 0; off >>= 1) { vr = kadd(vr, shfl_down_k(vr, off)); vq = kadd(vq, shfl_down_k(vq, off)); }
    if (lane == 0 && t <= d) {
        partial[(size_t)blockIdx.x * NACC + t] = kcanon(vr);
        partial[(size_t)blockIdx.x * NACC + d + 1 + t] = kcanon(vq);
    }
}

/// Fold every per-source table with challenge r: out[s][k] = a + r (b − a).
__global__ void coarse_fold_kernel(const K8d* __restrict__ in, K8d* __restrict__ out, uint32_t nsrc, uint64_t len, K8d r) {
    const uint64_t half = len >> 1, total = half * nsrc;
    for (uint64_t g = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x; g < total; g += (uint64_t)gridDim.x * blockDim.x) {
        const uint64_t s = g / half, k = g % half;
        K8d a = in[s * len + 2 * k], b = in[s * len + 2 * k + 1];
        out[s * half + k] = (kzero_p(a) && kzero_p(b)) ? kzero() : kcanon(kadd(a, kmul(r, ksub(b, a))));
    }
}

// ------------------------------------------------------------- one-hot v3
/// Plane round i < p of one-hot regions, warp per data block, lane t owns
/// plane bin t (t < 2^p ≤ 32). In round i the range/query sums only depend on
/// the bins C[t] = Σ_{idx: sel = t} eq(α_[p..B0), idx) and, per term,
/// CB_T[t] = Σ_{idx: sel = t} B_T[idx] (both independent of i). Each lane
/// accumulates polynomial coefficients in t:
///   range:  cr·H·(λ³e³ − λe) + cb·H·(λ²e² − λe),  H = eq(α_(i..p), t>>(i+1))·C[t]
///   query:  Σ_T c_T CB_T (A_T[0][pr] + t·dA_T[pr])·λe
/// with λ = 1 − t (bit 0) or t (bit 1). Coefficients (degree ≤ 3) are reduced
/// once per CUDA block: partial[block][0..3] range, [4..6] query.
__global__ void __launch_bounds__(128) onehot3_kernel(
    const int8_t* __restrict__ raw, uint64_t src_len, int b0, int i,
    const Blk* __restrict__ blks, uint32_t nblk, Consts C,
    const K8d* __restrict__ E1, const K8d* __restrict__ E2, const K8d* __restrict__ E3,
    const K8d* __restrict__ eai,      // per p: eq(α_[p..B0), idx), at offsets eai_off[p]
    const uint32_t* __restrict__ eai_off,
    const K8d* __restrict__ eapr,     // eq(α_(i..p), pr) per p, at offsets eapr_off[p]
    const uint32_t* __restrict__ eapr_off,
    K8d* __restrict__ partial)
{
    __shared__ K8d s_ct[4][MAXT];
    __shared__ K8d s_cr[4], s_cb[4];
    __shared__ uint32_t s_tcnt[4], s_toff[4], s_p[4];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int gw = blockIdx.x * 4 + warp, nw = gridDim.x * 4;
    K8d cr3 = kzero(), cr2 = kzero(), cr1 = kzero(), cr0 = kzero();   // range coefficients
    K8d cq2 = kzero(), cq1 = kzero(), cq0 = kzero();                  // query coefficients
    for (uint32_t bi = gw; bi < nblk; bi += nw) {
        const Blk b = blks[bi];
        const Region R = C.regions[b.region];
        const int p = (int)R.p, np = 1 << p;
        if (lane == 0) {
            const uint32_t mlo = (1u << C.ca_lo_bits) - 1;
            K8d ca = kmul(C.ca_lo[b.id & mlo], C.ca_hi[b.id >> C.ca_lo_bits]);
            s_cr[warp] = kmul(ca, C.g_range[b.src]);
            s_cb[warp] = (R.flags & 1) ? kmul(ca, C.g_bit[b.src]) : kzero();
            s_tcnt[warp] = R.tcnt; s_toff[warp] = R.toff; s_p[warp] = R.p;
        }
        if (lane < (int)R.tcnt) {
            const Term T = C.terms[R.toff + lane];
            uint32_t klo = b.k & ((1u << T.h1) - 1), khi = b.k >> T.h1;
            s_ct[warp][lane] = kmul(T.cs, kmul(C.hl[T.hl_off + klo], C.hh[T.hh_off + khi]));
        }
        __syncwarp();
        if (lane >= np) continue;
        // Bin t = lane: scan the block's indices.
        const int nidx = 1 << (b0 - p);
        const int8_t* dg = raw + (uint64_t)b.src * src_len + ((uint64_t)b.id << b0);
        const K8d* ea_i = eai + eai_off[p];
        K8d Cb = kzero();
        K8d CB[4] = {kzero(), kzero(), kzero(), kzero()};   // up to 4 structured terms here
        const int tc = (int)s_tcnt[warp] < 4 ? (int)s_tcnt[warp] : 4;
        bool any = false;
        for (int idx = 0; idx < nidx; idx++) {
            if (dg[((size_t)idx << p) + lane] != 1) continue;
            any = true;
            Cb = kadd(Cb, ea_i[idx]);
            for (int k = 0; k < tc; k++) {
                const Term T = C.terms[s_toff[warp] + k];
                CB[k] = kadd(CB[k], C.B[T.boff + idx]);
            }
        }
        if (!any) continue;
        const int t = lane;
        const int bit = (t >> i) & 1, low = t & ((1 << i) - 1), pr = t >> (i + 1);
        const K8d e = E1[low];
        const K8d H = kmul(eapr[eapr_off[p] + pr], Cb);
        // Range: X3·λ³ − X1r·λ + X2·λ² − X1b·λ
        const K8d X3 = kmul(kmul(s_cr[warp], H), E3[low]);
        const K8d X2 = kmul(kmul(s_cb[warp], H), E2[low]);
        const K8d X1 = kmul(kmul(kadd(s_cr[warp], s_cb[warp]), H), e);
        // Query: Σ_T c_T CB_T (a0 + t·da)·λ·e = λ·(Y0 + t·Y1)
        K8d Y0 = kzero(), Y1 = kzero();
        for (int k = 0; k < tc; k++) {
            const Term T = C.terms[s_toff[warp] + k];
            K8d cbk = kmul(s_ct[warp][k], CB[k]);
            K8d a0 = C.A[T.aoff + pr], a1 = C.A[T.aoff + (1 << T.pa) + pr];
            Y0 = kadd(Y0, kmul(cbk, a0));
            Y1 = kadd(Y1, kmul(cbk, ksub(a1, a0)));
        }
        Y0 = kmul(Y0, e); Y1 = kmul(Y1, e);
        if (bit == 1) {
            // λ = t: range = X3 t³ + X2 t² − X1 t; query = Y0 t + Y1 t²
            cr3 = kadd(cr3, X3); cr2 = kadd(cr2, X2); cr1 = ksub(cr1, X1);
            cq1 = kadd(cq1, Y0); cq2 = kadd(cq2, Y1);
        } else {
            // λ = 1 − t: λ³ = 1 − 3t + 3t² − t³, λ² = 1 − 2t + t².
            cr0 = kadd(cr0, ksub(kadd(X3, X2), X1));
            cr1 = kadd(cr1, kadd(ksub(kscale(X1, 1), kscale(X3, 3)), ksub(kzero(), kscale(X2, 2))));
            cr2 = kadd(cr2, kadd(kscale(X3, 3), X2));
            cr3 = ksub(cr3, X3);
            // (1 − t)(Y0 + t Y1) = Y0 + t (Y1 − Y0) − t² Y1
            cq0 = kadd(cq0, Y0); cq1 = kadd(cq1, ksub(Y1, Y0)); cq2 = ksub(cq2, Y1);
        }
    }
    // Block reduction of the 7 coefficients.
    __shared__ K8d red[4][7];
    K8d v[7] = {cr0, cr1, cr2, cr3, cq0, cq1, cq2};
    for (int k = 0; k < 7; k++) {
        K8d x = kcanon(v[k]);
        for (int off = 16; off > 0; off >>= 1) x = kadd(x, shfl_down_k(x, off));
        if (lane == 0) red[warp][k] = kcanon(x);
    }
    __syncthreads();
    if (threadIdx.x < 7) {
        K8d sacc = kzero();
        for (int w = 0; w < 4; w++) sacc = kadd(sacc, red[w][threadIdx.x]);
        partial[(size_t)blockIdx.x * 7 + threadIdx.x] = kcanon(sacc);
    }
}

// ------------------------------------------------------------- one-hot sums
/// One-hot regions, all plane rounds at once. With C[t] = Σ_{idx: sel = t}
/// eq(α_[p..B0), idx) and CB_T[t] = Σ_{idx: sel = t} B_T[idx] (B_T is fixed
/// while i < p), every round's one-hot contribution is a function of
///   G3[p][t] = Σ_b cr_b·C_b[t],  G2[p][t] = Σ_b cb_b·C_b[t],
///   W[T][t]  = Σ_b c_T(b)·CB_T,b[t].
/// Each warp takes a contiguous run of blocks, lane t owns bin t, and flushes
/// its sums on a region change as 32-bit halves into u64 atomics (16 per K).
/// bc_oh: block constants of this list in reverse order.
__device__ __forceinline__ void kacc_atomic(u64* dst, const K8d& v) {
    for (int w = 0; w < 8; w++) {
        atomicAdd(dst + 2 * w, v.c[w] & 0xffffffffULL);
        atomicAdd(dst + 2 * w + 1, v.c[w] >> 32);
    }
}
__global__ void __launch_bounds__(128) onehot_pre_kernel(
    const int8_t* __restrict__ raw, uint64_t src_len, int b0,
    const Blk* __restrict__ blks, uint32_t nblk, Consts C, const K8d* __restrict__ bc_oh,
    const K8d* __restrict__ eai, const uint32_t* __restrict__ eai_off, uint32_t per_warp,
    u64* __restrict__ G, u64* __restrict__ W)
{
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const uint64_t s0 = (uint64_t)(blockIdx.x * 4 + warp) * per_warp;
    const uint64_t e0 = s0 + per_warp < nblk ? s0 + per_warp : nblk;
    int cur = -1, cp = 0, ctc = 0;
    uint32_t ctoff = 0, cw = 0;   // cw: compact per-term bin base (one-hot regions keep it in qd_off)
    K8d g3 = kzero(), g2 = kzero(), w0 = kzero(), w1 = kzero(), w2 = kzero(), w3 = kzero();
    auto flush = [&]() {
        if (cur < 0 || lane >= (1 << cp)) return;
        kacc_atomic(G + ((size_t)(cp * 32 + lane) * 2 + 0) * 16, kcanon(g3));
        kacc_atomic(G + ((size_t)(cp * 32 + lane) * 2 + 1) * 16, kcanon(g2));
        if (ctc > 0) kacc_atomic(W + ((size_t)(cw + 0) * 32 + lane) * 16, kcanon(w0));
        if (ctc > 1) kacc_atomic(W + ((size_t)(cw + 1) * 32 + lane) * 16, kcanon(w1));
        if (ctc > 2) kacc_atomic(W + ((size_t)(cw + 2) * 32 + lane) * 16, kcanon(w2));
        if (ctc > 3) kacc_atomic(W + ((size_t)(cw + 3) * 32 + lane) * 16, kcanon(w3));
    };
    for (uint64_t bi = s0; bi < e0; bi++) {
        const Blk b = blks[bi];
        const Region R = C.regions[b.region];
        if ((int)b.region != cur) {
            flush();
            g3 = g2 = w0 = w1 = w2 = w3 = kzero();
            cur = (int)b.region; cp = (int)R.p; ctoff = R.toff; cw = R.qd_off; ctc = R.tcnt < 4 ? (int)R.tcnt : 4;
        }
        const int p = cp;
        if (lane >= (1 << p)) continue;
        const int nidx = 1 << (b0 - p);
        const int8_t* dg = raw + (uint64_t)b.src * src_len + ((uint64_t)b.id << b0);
        const K8d* ea_i = eai + eai_off[p];
        K8d Cb = kzero(), CB0 = kzero(), CB1 = kzero(), CB2 = kzero(), CB3 = kzero();
        const Term* Ts = C.terms + ctoff;
        bool any = false;
        for (int idx = 0; idx < nidx; idx++) {
            if (dg[((size_t)idx << p) + lane] != 1) continue;
            any = true;
            Cb = kadd(Cb, ea_i[idx]);
            if (ctc > 0) CB0 = kadd(CB0, C.B[Ts[0].boff + idx]);
            if (ctc > 1) CB1 = kadd(CB1, C.B[Ts[1].boff + idx]);
            if (ctc > 2) CB2 = kadd(CB2, C.B[Ts[2].boff + idx]);
            if (ctc > 3) CB3 = kadd(CB3, C.B[Ts[3].boff + idx]);
        }
        if (!any) continue;
        const K8d* o = bc_oh + C.bco[C.oh_base + nblk - 1 - bi];
        g3 = kadd(g3, kmul(o[0], Cb));
        if (!kzero_p(o[1])) g2 = kadd(g2, kmul(o[1], Cb));
        if (ctc > 0) w0 = kadd(w0, kmul(o[2], CB0));
        if (ctc > 1) w1 = kadd(w1, kmul(o[3], CB1));
        if (ctc > 2) w2 = kadd(w2, kmul(o[4], CB2));
        if (ctc > 3) w3 = kadd(w3, kmul(o[5], CB3));
    }
    flush();
}

// ------------------------------------------------------------- index rounds
/// Late block-local rounds (2^{b0−i} ≤ 128, 2^i ≥ 16): one warp per data
/// block, folded digits F and query weights QF in a per-warp shared slice.
///   A: digits → F[y] = Σ_p e_pre[p]·d[(y << i) + p]; lane owns 2^b0/32
///      contiguous bytes (16-byte loads), lanes sharing a y reduce.
///   B: QF[y] = folded dense table (level i) + Σ_T c_T B_T[ir] A_T[side][pr];
///      lanes split pairs, then terms.
///   C: lanes 0..29 (pair slot pl = lane/3, point t ∈ {0, 2, 3}) add the range
///      sum at t and (t < 3) the query sum at t. Two accumulators per lane.
/// The host gets t = 1 from the previous round's claims and extrapolates
/// range (degree 3) and query (degree 2) to 5 points.
/// partial[block][t] range at t (slot 1 zero), [4 + t] query at t (slot 5 zero).
constexpr int IDX_WARPS = 4;
constexpr int IDX_MAXY = 128;
__device__ __forceinline__ K8d shfl_down_seg(const K8d& v, int off, int width) {
    K8d r; for (int w = 0; w < 8; w++) r.c[w] = __shfl_down_sync(0xffffffffu, v.c[w], off, width); return r;
}
__global__ void __launch_bounds__(128, 4) index_kernel(
    const int8_t* __restrict__ raw, uint64_t src_len, int b0, int i,
    const Blk* __restrict__ blks, uint32_t nblk, Consts C, const K8d* __restrict__ bc,
    const K8d* __restrict__ e_pre, const K8d* __restrict__ e_alpha, K8d* __restrict__ partial)
{
    extern __shared__ K8d dyn_sh[];   // [IDX_WARPS][2][ny]
    __shared__ K8d s_c[IDX_WARPS][2 + MAXT];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int gw = blockIdx.x * IDX_WARPS + warp, nw = gridDim.x * IDX_WARPS;
    const int ny = 1 << (b0 - i), nrest = ny >> 1;
    K8d* Fw = dyn_sh + (size_t)warp * 2 * ny;
    K8d* QFw = Fw + ny;
    // Phase A geometry.
    const int P = 1 << (b0 - 5);                       // bytes per lane
    const int span = P < (1 << i) ? P : (1 << i);      // bytes of one y per lane
    const int G = (1 << i) > P ? (1 << i) / P : 1;     // lanes per y
    const int nys = P >> (31 - __clz(span));           // ys per lane
    // Phase B geometry.
    const int GB = nrest >= 32 ? 1 : 32 / nrest;
    // Phase C slot.
    const int tl = lane < 30 ? (lane % 3 == 0 ? 0 : lane % 3 + 1) : 0;
    const int pl = lane < 30 ? lane / 3 : 1 << 20;
    const u64 tq = (u64)tl;
    K8d ar = kzero(), aq = kzero();
    for (uint32_t bi = gw; bi < nblk; bi += nw) {
        const Blk b = blks[bi];
        uint32_t tcnt = 0, toff = 0, qd = 0xffffffffu;
        if (b.region != 0xffffffffu) {
            const Region R = C.regions[b.region];
            tcnt = R.tcnt; toff = R.toff;
            if (R.flags & 4) { uint32_t qi = C.qidx[R.qd_off + b.k]; qd = qi == 0xffffffffu ? 0xffffffffu : (qi << (b0 - i)); }
        }
        const K8d* o = bc + C.bco[bi];
        for (int k = lane; k < 2 + (int)tcnt; k += 32) s_c[warp][k] = o[k];
        // ---- A: fold the digits.
        const int8_t* dg = raw + (uint64_t)b.src * src_len + ((uint64_t)b.id << b0) + (size_t)lane * P;
        for (int s = 0; s < nys; s++) {
            const int start = lane * P + s * span;
            const int poff = start & ((1 << i) - 1);
            K8d acc = kzero();
            for (int c = 0; c < span; c += 16) {
                const int4 v = __ldg(reinterpret_cast<const int4*>(dg + s * span + c));
                const uint32_t wv[4] = {(uint32_t)v.x, (uint32_t)v.y, (uint32_t)v.z, (uint32_t)v.w};
                #pragma unroll
                for (int q = 0; q < 4; q++) {
                    uint32_t x = wv[q];
                    while (x) {
                        const int bt = __ffs(x) - 1, by = bt >> 3;
                        const int8_t dv = (int8_t)((x >> (by * 8)) & 0xff);
                        x &= ~(0xffu << (by * 8));
                        const K8d e = e_pre[poff + c + q * 4 + by];
                        acc = dv == 1 ? kadd(acc, e) : ksub(acc, e);
                    }
                }
            }
            if (G > 1) {
                acc = kcanon(acc);
                for (int off = G >> 1; off > 0; off >>= 1) acc = kadd(acc, shfl_down_seg(acc, off, G));
            }
            if (lane % G == 0) Fw[start >> i] = kcanon(acc);
        }
        // ---- B: query weights.
        for (int y = lane; y < ny; y += 32) QFw[y] = qd != 0xffffffffu ? C.qd[qd + y] : kzero();
        __syncwarp();
        if (tcnt > 0) {
            for (int r0 = 0; r0 < nrest; r0 += 32 / GB) {
                const int rest = r0 + lane / GB, gl = lane % GB;
                K8d q0 = kzero(), q1 = kzero();
                if (rest < nrest) {
                    for (int k = gl; k < (int)tcnt; k += GB) {
                        const Term T = C.terms[toff + k];
                        const int pr = rest & ((1 << T.pa) - 1), ir = rest >> T.pa;
                        const K8d bv = kmul(C.B[T.boff + ir], s_c[warp][2 + k]);
                        q0 = kadd(q0, kmul(C.A[T.aoff + pr], bv));
                        q1 = kadd(q1, kmul(C.A[T.aoff + (1 << T.pa) + pr], bv));
                    }
                }
                if (GB > 1) {
                    q0 = kcanon(q0); q1 = kcanon(q1);
                    for (int off = GB >> 1; off > 0; off >>= 1) { q0 = kadd(q0, shfl_down_seg(q0, off, GB)); q1 = kadd(q1, shfl_down_seg(q1, off, GB)); }
                }
                if (rest < nrest && gl == 0) {
                    QFw[2 * rest] = kcanon(kadd(QFw[2 * rest], q0));
                    QFw[2 * rest + 1] = kcanon(kadd(QFw[2 * rest + 1], q1));
                }
            }
            __syncwarp();
        }
        // ---- C: evaluations.
        const K8d cr = s_c[warp][0], cb = s_c[warp][1];
        for (int rest = pl; rest < nrest; rest += 10) {
            const K8d w0 = Fw[2 * rest], w1 = Fw[2 * rest + 1];
            if (kzero_p(w0) && kzero_p(w1)) continue;
            const K8d dw = ksub(w1, w0);
            K8d wt;
            for (int a = 0; a < 8; a++) wt.c[a] = mauth::fadd(w0.c[a], mauth::fmul(dw.c[a], tq));
            const K8d w2 = kmul(wt, wt), w3 = kmul(w2, wt);
            const K8d r = kadd(kmul(cr, ksub(w3, wt)), kmul(cb, ksub(w2, wt)));
            ar = kadd(ar, kmul(e_alpha[rest], r));
            if (tl < 3) {
                const K8d q0 = QFw[2 * rest], q1 = QFw[2 * rest + 1];
                const K8d dq = ksub(q1, q0);
                K8d qt;
                for (int a = 0; a < 8; a++) qt.c[a] = mauth::fadd(q0.c[a], mauth::fmul(dq.c[a], tq));
                aq = kadd(aq, kmul(qt, wt));
            }
        }
        __syncwarp();
    }
    // Lanes with equal t reduce through shared memory (once per kernel; the
    // dynamic slice holds at least IDX_WARPS·32·2 K).
    __syncthreads();
    K8d (*red)[2] = reinterpret_cast<K8d (*)[2]>(dyn_sh);
    red[threadIdx.x][0] = kcanon(ar); red[threadIdx.x][1] = kcanon(aq);
    __syncthreads();
    if (threadIdx.x < 7) {
        const int slot = threadIdx.x;
        const int tt = slot < 4 ? slot : slot - 4, which = slot < 4 ? 0 : 1;
        K8d sacc = kzero();
        if (tt != 1 && !(which == 1 && tt == 3))
            for (int l = 0; l < IDX_WARPS * 32; l++) {
                const int ln = l & 31;
                if (ln >= 30) continue;
                const int lt = ln % 3 == 0 ? 0 : ln % 3 + 1;
                if (lt == tt) sacc = kcanon(kadd(sacc, red[l][which]));
            }
        partial[(size_t)blockIdx.x * 7 + slot] = kcanon(sacc);
    }
}

}  // namespace mlink2

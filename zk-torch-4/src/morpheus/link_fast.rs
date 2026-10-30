//! Structured prover for Π_link (same transcript as `link::prove`).
//!
//! The reference prover folds dense 2^ℓ-entry K tables per source and per
//! query term. This one uses the structure of the statement:
//!   * every region (and every declared bit block) is aligned and at least
//!     2^B0 long, so each aligned 2^B0 *block* of a source lies inside one
//!     region; blocks with no digits and no terms contribute nothing;
//!   * rounds i < B0 bind block-local bits. A block recomputes its folded
//!     values w(r_<i, b, rest) from its ±1 digits with the prefix eq weights
//!     (additions only), so no per-round tables are stored. eq(α, ·) is a
//!     shared table over the remaining local bits times a per-block constant;
//!     χ_b is constant on the block; each query term is
//!     c_{T,block}·A_T[t][plane rest]·B_T[index rest];
//!   * from round B0 every block is one (w, q, χ) value: the remaining rounds
//!     are a dense sumcheck over 2^{ℓ−B0} entries per source.
//! `phase1_block` is the per-block kernel; `cuda_almost_goldilocks/
//! morpheus_link.cuh` runs the same computation on the GPU.

use super::auth::{f_ox, f_x, Point, SourceVec, Tuple};
use super::coins::{Coins, Purpose};
use super::field8::{eq_table, K8};
use super::keys::KeySeed;
use super::link::{interpolate, raw_to_cols, LinkOutput, LinkProof, Statement};
use super::params::{Profile, A_ENC_COLS};
use super::pcs_maps::enc;
use super::ring::{AElem, DS};
use super::transcript::{Item, MTranscript, Stage};
use almost_goldilocks_cuda::field::ALMOST_GOLDILOCKS_PRIME as Q;
use rayon::prelude::*;
use std::collections::HashMap;

pub(crate) fn half() -> K8 { K8::from_base((Q + 1) / 2) }
fn q_of(v: &K8) -> K8 { v.mul(v).mul(v).sub(v) }
fn b_of(v: &K8) -> K8 { v.mul(v).sub(v) }
pub(crate) fn eqb(a: &K8, bit: usize) -> K8 { if bit == 1 { *a } else { K8::ONE.sub(a) } }
/// eq(a, t) for the small integer t.
fn eqt(a: &K8, t: usize) -> K8 { let tk = K8::from_base(t as u64); a.mul(&tk).add(&K8::ONE.sub(a).mul(&K8::ONE.sub(&tk))) }
pub(crate) fn m_term(a: &[K8], u: &K8) -> K8 {
    let mut acc = K8::ZERO; let mut up = K8::ONE;
    for ak in a { up = up.mul(u); acc = acc.add(&ak.mul(&up.sub(&half()))); }
    acc
}

/// One query term restricted to one block.
#[derive(Clone)]
pub struct BlockTerm {
    /// Term index (into the per-round A/B tables).
    pub term: usize,
    /// coef·scale·eq(r_T[high index bits], block bits).
    pub c: K8,
}

/// A nonzero block of a source.
#[derive(Clone)]
pub struct Block {
    pub source: usize,
    /// Block number (raw index >> B0).
    pub id: usize,
    pub chi: bool,
    pub c_alpha: K8,
    pub terms: Vec<BlockTerm>,
    /// Dense query weights over the block (regions with many planes, i.e.
    /// the graph-mask bank), 2^B0 entries.
    pub qd: Option<std::sync::Arc<Vec<K8>>>,
}

/// Regions with more plane bits than this use dense per-block query tables.
pub const DENSE_P: usize = 6;

/// Per-term data: plane bits p, plane weights pw, index point r_T (local part).
pub struct TermInfo { pub p: usize, pub pw: std::sync::Arc<Vec<K8>>, pub r: Vec<K8> }

/// Per-round tables shared by all blocks.
pub struct RoundTables {
    pub i: usize,
    /// Prefix eq weights E_i[p] = eq(r_<i, p), 2^i entries.
    pub e_pre: Vec<K8>,
    /// eq(α_j, rest_j) over local rest bits i+1..B0, 2^{B0−i−1} entries.
    pub e_alpha: Vec<K8>,
    /// Per term: A[t][plane rest] (len (d+1)·2^{pa}) and B[index rest].
    pub a: Vec<Vec<K8>>,
    pub b: Vec<Vec<K8>>,
    /// Per term: plane-rest bit count pa (so rest = pr + (ir << pa)).
    pub pa: Vec<usize>,
}

/// Per-block contribution to the round message: 2(d+1) values, range[t]
/// (times c_alpha and the source's γ powers, without the α prefix) then
/// query[t].
pub fn phase1_block(
    digits: &[i8], blk: &Block, rt: &RoundTables, d: usize, b0: usize, g_range: &K8, g_bit: &K8,
) -> Vec<K8> {
    let i = rt.i;
    let rest_n = 1usize << (b0 - i - 1);
    let mut acc = vec![K8::ZERO; 2 * (d + 1)];
    let mut w = [K8::ZERO, K8::ZERO];
    for rest in 0..rest_n {
        // w(r_<i, bit, rest) = Σ_p E_i[p]·digit[p + (bit << i) + (rest << (i+1))].
        let mut nz = false;
        for (bit, wb) in w.iter_mut().enumerate() {
            let mut acc_w = K8::ZERO;
            let base = (bit << i) + (rest << (i + 1));
            for (p, e) in rt.e_pre.iter().enumerate() {
                match digits[base + p] { 1 => acc_w = acc_w.add(e), -1 => acc_w = acc_w.sub(e), 0 => {}, _ => panic!("non-ternary digit") }
            }
            if !acc_w.is_zero() { nz = true; }
            *wb = acc_w;
        }
        if !nz { continue; }
        let ea = rt.e_alpha[rest];
        // Query weight per t (linear in t through A).
        let mut qw = vec![K8::ZERO; d + 1];
        if let Some(qd) = &blk.qd {
            let fold = |bit: usize| -> K8 {
                let base = (bit << i) + (rest << (i + 1));
                rt.e_pre.iter().enumerate().fold(K8::ZERO, |a, (p, e)| { let v = &qd[base + p]; if v.is_zero() { a } else { a.add(&e.mul(v)) } })
            };
            let (q0, q1) = (fold(0), fold(1));
            let dq = q1.sub(&q0);
            let mut qt = q0;
            for (t, q) in qw.iter_mut().enumerate() { if t > 0 { qt = qt.add(&dq); } *q = q.add(&qt); }
        }
        for bt in &blk.terms {
            let pa = rt.pa[bt.term];
            let (pr, ir) = (rest & ((1 << pa) - 1), rest >> pa);
            let bv = rt.b[bt.term][ir].mul(&bt.c);
            for (t, q) in qw.iter_mut().enumerate() { *q = q.add(&rt.a[bt.term][t * (1 << pa) + pr].mul(&bv)); }
        }
        let dw = w[1].sub(&w[0]);
        let mut wt = w[0];
        for t in 0..=d {
            if t > 0 { wt = wt.add(&dw); }
            let mut r = g_range.mul(&q_of(&wt));
            if blk.chi { r = r.add(&g_bit.mul(&b_of(&wt))); }
            acc[t] = acc[t].add(&ea.mul(&r));
            acc[d + 1 + t] = acc[d + 1 + t].add(&qw[t].mul(&wt));
        }
    }
    for x in acc[..=d].iter_mut() { *x = x.mul(&blk.c_alpha); }
    acc
}

/// Block-local work of the structured prover (CPU or GPU).
pub trait Phase1 {
    /// Round sums (range[t] then query[t]) over all blocks.
    fn round(&mut self, rt: &RoundTables, terms: &[TermInfo], d: usize) -> Vec<K8>;
    /// Fully folded value of every block, in block order.
    fn fold(&mut self, e_full: &[K8]) -> Vec<K8>;
}

/// Everything a backend needs about the blocks.
pub struct Phase1Input<'a> {
    pub raw: &'a [Vec<i8>],
    pub blocks: &'a [Block],
    pub g_range: &'a [K8],
    pub g_bit: &'a [K8],
    pub b0: usize,
}

pub struct CpuPhase1<'a> { pub inp: Phase1Input<'a> }

impl<'a> Phase1 for CpuPhase1<'a> {
    fn round(&mut self, rt: &RoundTables, _terms: &[TermInfo], d: usize) -> Vec<K8> {
        let b0 = self.inp.b0;
        let raw = self.inp.raw;
        self.inp.blocks.par_iter().map(|blk| {
            let dg = &raw[blk.source][blk.id << b0..(blk.id + 1) << b0];
            phase1_block(dg, blk, rt, d, b0, &self.inp.g_range[blk.source], &self.inp.g_bit[blk.source])
        }).reduce(|| vec![K8::ZERO; 2 * (d + 1)], |a, b| a.iter().zip(&b).map(|(x, y)| x.add(y)).collect())
    }
    fn fold(&mut self, e_full: &[K8]) -> Vec<K8> {
        let b0 = self.inp.b0;
        self.inp.blocks.par_iter().map(|blk| block_fold(&self.inp.raw[blk.source][blk.id << b0..(blk.id + 1) << b0], e_full)).collect()
    }
}

/// Link masks and auxiliary witness.
pub struct Masks { pub u: Vec<AElem>, pub r_cols: Vec<Vec<Vec<i64>>>, pub a_aux: Vec<AElem>, pub s_o: Vec<Vec<i64>> }

/// The non-sumcheck work of the link: private coins, link/aux commitments
/// and the raw evaluation endpoints (CPU reference or GPU).
pub trait LinkServices {
    fn masks(&mut self, p: &Profile, coins: &Coins, opening: u64, n: usize) -> Masks;
    fn commit_link(&mut self, p: &Profile, seed: &KeySeed, r_cols: &[Vec<Vec<i64>>]) -> Vec<Vec<Vec<u64>>>;
    fn commit_aux(&mut self, p: &Profile, seed: &KeySeed, s_o: &[Vec<i64>]) -> Vec<Vec<u64>>;
    /// Ev_raw(W_e; ξ) for every source.
    fn ev(&mut self, p: &Profile, raw: &[Vec<i8>], xi: &[K8]) -> Vec<AElem>;
    /// All sources on the device (logical order, concatenated), if resident.
    fn dev_raw(&self) -> Option<&almost_goldilocks_cuda::DeviceBuffer<i8>> { None }
    /// Streamed sources: where the link keeps its per-device source shards.
    fn kept(&self) -> Option<&std::sync::Mutex<Vec<super::link_gpu::Kept>>> { None }
}

pub struct CpuServices;

impl LinkServices for CpuServices {
    fn masks(&mut self, p: &Profile, coins: &Coins, opening: u64, n: usize) -> Masks {
        let owner = |e: usize| opening * p.n_max as u64 + e as u64;
        let ua = |purpose, owner, c0| super::link::uniform_a_pub(coins, purpose, owner, c0);
        let u: Vec<AElem> = (0..n).map(|e| ua(Purpose::LinkPayload, owner(e), 0)).collect();
        let r_cols = (0..n).map(|e| [enc(&u[e]), super::link::ternary_cols_pub(coins, Purpose::LinkSecret, owner(e), p.hiding), super::link::ternary_cols_pub(coins, Purpose::LinkError, owner(e), p.rank)].concat()).collect();
        let a_aux: Vec<AElem> = (0..p.aux_coeffs).map(|i| ua(Purpose::AuxPayload, opening, (i * 512) as u32)).collect();
        let s_o = [a_aux.iter().flat_map(enc).collect::<Vec<_>>(), super::link::ternary_cols_pub(coins, Purpose::AuxSecret, opening, p.hiding), super::link::ternary_cols_pub(coins, Purpose::AuxError, opening, p.rank)].concat();
        Masks { u, r_cols, a_aux, s_o }
    }
    fn commit_link(&mut self, p: &Profile, seed: &KeySeed, r_cols: &[Vec<Vec<i64>>]) -> Vec<Vec<Vec<u64>>> {
        r_cols.iter().map(|r| super::auth::lin(p, seed, super::keys::KeyKind::Link, A_ENC_COLS, r)).collect()
    }
    fn commit_aux(&mut self, p: &Profile, seed: &KeySeed, s_o: &[Vec<i64>]) -> Vec<Vec<u64>> {
        super::auth::lin(p, seed, super::keys::KeyKind::Aux, A_ENC_COLS * p.aux_coeffs, s_o)
    }
    fn ev(&mut self, _p: &Profile, raw: &[Vec<i8>], xi: &[K8]) -> Vec<AElem> {
        raw.iter().map(|r| super::pcs_maps::ev_raw(&raw_to_cols(r), xi)).collect()
    }
}

/// Builds the backend once the blocks are known.
pub type BackendFactory<'f> = dyn for<'a> Fn(Phase1Input<'a>) -> Box<dyn Phase1 + 'a> + 'f;

pub fn cpu_backend<'a>(inp: Phase1Input<'a>) -> Box<dyn Phase1 + 'a> { Box::new(CpuPhase1 { inp }) }

/// Fully folded block value w(r_<B0) = Σ_p eq(r_<B0, p)·digit[p].
pub fn block_fold(digits: &[i8], e_full: &[K8]) -> K8 {
    digits.iter().zip(e_full).fold(K8::ZERO, |a, (&d, e)| match d { 1 => a.add(e), -1 => a.sub(e), _ => a })
}

/// Plane-rest/index-rest tables of a term at round i (< b0).
pub(crate) fn term_tables(t: &TermInfo, r_pre: &[K8], i: usize, d: usize, b0: usize) -> (Vec<K8>, Vec<K8>, usize) {
    let p = t.p;
    let idx_bits = b0 - p; // index bits inside the block
    if i < p {
        let pa = p - i - 1;
        // A[t][pr] = Σ_{b<2^i} eq(r_<i, b)·pw[b + (t' << i) + (pr << (i+1))] with t' the
        // value of bit i, linearly extended to t.
        let e = eq_table(r_pre);
        let mut a = vec![K8::ZERO; (d + 1) << pa];
        for pr in 0..1usize << pa {
            let f = |bit: usize| -> K8 { (0..1usize << i).fold(K8::ZERO, |acc, b| acc.add(&e[b].mul(&t.pw[b + (bit << i) + (pr << (i + 1))]))) };
            let (f0, f1) = (f(0), f(1));
            for tt in 0..=d { a[(tt << pa) + pr] = f0.add(&K8::from_base(tt as u64).mul(&f1.sub(&f0))); }
        }
        let b = eq_table(&t.r[..idx_bits]);
        (a, b, pa)
    } else {
        // Planes folded: pw(r_[0..p)); index bits bound up to i.
        let ep = eq_table(&r_pre[..p]);
        let pwv = t.pw.iter().zip(&ep).fold(K8::ZERO, |acc, (w, e)| acc.add(&w.mul(e)));
        let j = i - p; // index bit being bound
        let pre = t.r[..j].iter().zip(&r_pre[p..i]).fold(K8::ONE, |acc, (a, b)| acc.mul(&a.mul(b).add(&K8::ONE.sub(a).mul(&K8::ONE.sub(b)))));
        let s = pwv.mul(&pre);
        let a: Vec<K8> = (0..=d).map(|tt| s.mul(&eqt(&t.r[j], tt))).collect();
        let b = eq_table(&t.r[j + 1..idx_bits]);
        (a, b, 0)
    }
}

/// Structured prover; identical output to `link::prove`.
#[allow(clippy::too_many_arguments)]
pub fn prove(
    p: &Profile, seed: &KeySeed, coins: &Coins, opening: u64, tr: &mut MTranscript,
    st: &Statement, raw: &[Vec<i8>], b0: usize,
) -> LinkOutput {
    prove_with(p, seed, coins, opening, tr, st, raw, b0, &cpu_backend, &mut CpuServices, None, false)
}

/// Same, with a chosen block backend.
#[allow(clippy::too_many_arguments)]
pub fn prove_with(
    p: &Profile, seed: &KeySeed, coins: &Coins, opening: u64, tr: &mut MTranscript,
    st: &Statement, raw: &[Vec<i8>], b0: usize, backend: &BackendFactory,
    svc: &mut dyn LinkServices, c_src: Option<&[Vec<Vec<u64>>]>, v2: bool,
) -> LinkOutput {
    p.check_caps(raw);
    let tm = std::env::var("MORPHEUS_TIMING").is_ok();
    let mut t0 = std::time::Instant::now();
    let mut lap = |what: &str| { if tm { eprintln!("[link] {what:24} {:8.1} ms", t0.elapsed().as_secs_f64() * 1e3); } t0 = std::time::Instant::now(); };
    let ell = p.ell();
    let n = st.n;
    let degs = p.degs();
    let Masks { u, r_cols, a_aux, s_o } = svc.masks(p, coins, opening, n);
    lap("masks");
    let g_commit = svc.commit_link(p, seed, &r_cols);
    lap("link commitments");
    let c_o = svc.commit_aux(p, seed, &s_o);
    for g in &g_commit { for c in g { tr.absorb(Item::LinkCommitment, c); } }
    for c in &c_o { tr.absorb(Item::AuxCommitment, c); }
    let (alpha, gamma, eta) = super::link::challenges_scalar(tr, ell);
    let mut gp = Vec::new();
    { let mut x = K8::ONE; for _ in 0..2 * n + st.rows.len() { gp.push(x); x = x.mul(&gamma); } }
    lap("aux commitment + coins");
    let a_coef: Vec<Vec<K8>> = { let mut v = Vec::new(); let mut idx = 0; for &d in &degs { v.push((0..d).map(|k| a_aux[idx + k].0[0]).collect()); idx += d; } v };
    let (rounds, xi, w, q, chi, pre_alpha, mo_prefix) = if v2 {
        let o = super::link_gpu::phases_v2(p, st, raw, b0, &alpha, &eta, &gp, &degs, &a_coef, tr, svc.dev_raw(), svc.kept());
        lap("v2 phases");
        (o.0, o.1, o.2.into_iter().map(|x| vec![x]).collect::<Vec<_>>(), o.3.into_iter().map(|x| vec![x]).collect::<Vec<_>>(), o.4.into_iter().map(|x| vec![x]).collect::<Vec<_>>(), o.5, o.6)
    } else {
    // Terms and blocks.
    let mut terms: Vec<TermInfo> = Vec::new();
    let mut by_block: HashMap<(usize, usize), Vec<BlockTerm>> = HashMap::new();
    let mut dense: HashMap<(usize, usize), Vec<K8>> = HashMap::new();
    for (j, row) in st.rows.iter().enumerate() {
        let coef = eta.mul(&gp[2 * n + j]);
        for t in &row.terms {
            let (m, pp) = (t.region.m, t.region.p);
            assert!(m + pp >= b0 && pp <= b0, "region smaller than the block size");
            if pp > DENSE_P {
                // Dense per-block weights: coef·scale·pw[b]·eq(r_T, index).
                let idx_bits = b0 - pp;
                let el = eq_table(&t.r[..idx_bits]);
                let boolean = t.r[idx_bits..].iter().all(|x| *x == K8::ZERO || *x == K8::ONE);
                let nb = 1usize << (m + pp - b0);
                let ks: Vec<usize> = if boolean {
                    vec![t.r[idx_bits..].iter().enumerate().fold(0usize, |a, (jj, x)| a | (((*x == K8::ONE) as usize) << jj))]
                } else { (0..nb).collect() };
                for k in ks {
                    let hi = t.r[idx_bits..].iter().enumerate().fold(K8::ONE, |a, (jj, x)| a.mul(&eqb(x, (k >> jj) & 1)));
                    let c = coef.mul(&t.scale).mul(&hi);
                    let id = (t.region.base >> b0) + k;
                    let tab = dense.entry((t.region.source, id)).or_insert_with(|| vec![K8::ZERO; 1usize << b0]);
                    for (il, e) in el.iter().enumerate() {
                        if e.is_zero() { continue; }
                        let ce = c.mul(e);
                        for b in 0..1usize << pp { if !t.pw[b].is_zero() { let u = (il << pp) + b; tab[u] = tab[u].add(&ce.mul(&t.pw[b])); } }
                    }
                }
                continue;
            }
            let ti = terms.len();
            terms.push(TermInfo { p: pp, pw: t.pw.clone(), r: t.r.clone() });
            let idx_bits = b0 - pp;
            let nb = 1usize << (m + pp - b0);
            for k in 0..nb {
                // Block k of the region covers index high bits k.
                let hi = t.r[idx_bits..].iter().enumerate().fold(K8::ONE, |a, (jj, x)| a.mul(&eqb(x, (k >> jj) & 1)));
                let c = coef.mul(&t.scale).mul(&hi);
                let id = (t.region.base >> b0) + k;
                by_block.entry((t.region.source, id)).or_default().push(BlockTerm { term: ti, c });
            }
        }
    }
    let mut bit_block: HashMap<(usize, usize), bool> = HashMap::new();
    for (s, bl) in st.bits.iter().enumerate() {
        for &(base, lg) in bl {
            assert!(lg >= b0, "bit block smaller than the block size");
            for k in 0..1usize << (lg - b0) { bit_block.insert((s, (base >> b0) + k), true); }
        }
    }
    let c_alpha = |id: usize| -> K8 { alpha[b0..].iter().enumerate().fold(K8::ONE, |a, (j, x)| a.mul(&eqb(x, (id >> j) & 1))) };
    let mut blocks: Vec<Block> = Vec::new();
    for s in 0..n {
        let nblk = 1usize << (ell - b0);
        let nz: Vec<usize> = (0..nblk).into_par_iter().filter(|&id| raw[s][id << b0..(id + 1) << b0].iter().any(|&d| d != 0)).collect();
        let mut ids: std::collections::BTreeSet<usize> = nz.into_iter().collect();
        for &(ss, id) in by_block.keys().chain(dense.keys()) { if ss == s { ids.insert(id); } }
        for id in ids {
            blocks.push(Block { source: s, id, chi: bit_block.contains_key(&(s, id)), c_alpha: c_alpha(id), terms: by_block.get(&(s, id)).cloned().unwrap_or_default(), qd: dense.remove(&(s, id)).map(std::sync::Arc::new) });
        }
    }
    lap("terms + blocks");

    let nvars = ell + 1;
    let mut rounds = Vec::new();
    let mut xi: Vec<K8> = Vec::new();
    let mut mo_prefix = K8::ZERO;
    let mut pre_alpha = K8::ONE; // Π_{j<i} eq(α_j, r_j)
    let add_v = |a: Vec<K8>, b: Vec<K8>| -> Vec<K8> { a.iter().zip(&b).map(|(x, y)| x.add(y)).collect() };
    let g_range: Vec<K8> = gp[..n].to_vec();
    let g_bit: Vec<K8> = gp[n..2 * n].to_vec();
    let mut be = backend(Phase1Input { raw, blocks: &blocks, g_range: &g_range, g_bit: &g_bit, b0 });
    lap("backend setup");
    // ---- Phase 1: block-local rounds.
    for i in 0..b0 {
        let d = degs[i];
        let mut rt = RoundTables { i, e_pre: eq_table(&xi), e_alpha: eq_table(&alpha[i + 1..b0]), a: Vec::new(), b: Vec::new(), pa: Vec::new() };
        let tabs: Vec<(Vec<K8>, Vec<K8>, usize)> = terms.par_iter().map(|t| term_tables(t, &xi, i, d, b0)).collect();
        for (a, b, pa) in tabs { rt.a.push(a); rt.b.push(b); rt.pa.push(pa); }
        let sum = be.round(&rt, &terms, d);
        let g = finish_round(&sum, &pre_alpha, &alpha[i], d, nvars, i, &mo_prefix, &a_coef[i]);
        tr.absorb_k(Item::RoundPoly, &g);
        let r = tr.challenge_k(Stage::LinkRound, 0, i as u64);
        rounds.push(g);
        mo_prefix = mo_prefix.add(&m_term(&a_coef[i], &r));
        pre_alpha = pre_alpha.mul(&eqt_k(&alpha[i], &r));
        xi.push(r);
    }
    lap("phase 1 rounds");
    // ---- Phase 2: dense coarse tables over 2^{ℓ−B0} blocks per source.
    let e_full = eq_table(&xi);
    let coarse = 1usize << (ell - b0);
    let mut w: Vec<Vec<K8>> = vec![vec![K8::ZERO; coarse]; n];
    let mut q: Vec<Vec<K8>> = vec![vec![K8::ZERO; coarse]; n];
    let mut chi: Vec<Vec<K8>> = vec![vec![K8::ZERO; coarse]; n];
    let tfold: Vec<K8> = terms.iter().map(|t| {
        let ep = eq_table(&xi[..t.p]);
        let pwv = t.pw.iter().zip(&ep).fold(K8::ZERO, |acc, (a, b)| acc.add(&a.mul(b)));
        let idx_bits = b0 - t.p;
        let er = t.r[..idx_bits].iter().zip(&xi[t.p..b0]).fold(K8::ONE, |acc, (a, b)| acc.mul(&eqt_k(a, b)));
        pwv.mul(&er)
    }).collect();
    let folded: Vec<K8> = be.fold(&e_full);
    drop(be);
    for (blk, wf) in blocks.iter().zip(folded) {
        w[blk.source][blk.id] = wf;
        let qd = blk.qd.as_ref().map(|t| t.iter().zip(&e_full).fold(K8::ZERO, |a, (x, e)| a.add(&x.mul(e)))).unwrap_or(K8::ZERO);
        q[blk.source][blk.id] = blk.terms.iter().fold(qd, |a, bt| a.add(&bt.c.mul(&tfold[bt.term])));
        if blk.chi { chi[blk.source][blk.id] = K8::ONE; }
    }
    for i in b0..ell {
        let d = degs[i];
        let ea = eq_table(&alpha[i + 1..]);
        let half_len = w[0].len() / 2;
        let sum: Vec<K8> = (0..n).into_par_iter().map(|s| {
            let mut acc = vec![K8::ZERO; 2 * (d + 1)];
            for k in 0..half_len {
                let (w0, w1) = (w[s][2 * k], w[s][2 * k + 1]);
                if w0.is_zero() && w1.is_zero() { continue; }
                let (q0, q1) = (q[s][2 * k], q[s][2 * k + 1]);
                let (c0, c1) = (chi[s][2 * k], chi[s][2 * k + 1]);
                let (dw, dq, dc) = (w1.sub(&w0), q1.sub(&q0), c1.sub(&c0));
                let (mut wt, mut qt, mut ct) = (w0, q0, c0);
                for t in 0..=d {
                    if t > 0 { wt = wt.add(&dw); qt = qt.add(&dq); ct = ct.add(&dc); }
                    let r = gp[s].mul(&q_of(&wt)).add(&gp[n + s].mul(&ct).mul(&b_of(&wt)));
                    acc[t] = acc[t].add(&ea[k].mul(&r));
                    acc[d + 1 + t] = acc[d + 1 + t].add(&qt.mul(&wt));
                }
            }
            acc
        }).reduce(|| vec![K8::ZERO; 2 * (d + 1)], add_v);
        let g = finish_round(&sum, &pre_alpha, &alpha[i], d, nvars, i, &mo_prefix, &a_coef[i]);
        tr.absorb_k(Item::RoundPoly, &g);
        let r = tr.challenge_k(Stage::LinkRound, 0, i as u64);
        rounds.push(g);
        mo_prefix = mo_prefix.add(&m_term(&a_coef[i], &r));
        pre_alpha = pre_alpha.mul(&eqt_k(&alpha[i], &r));
        let fold = |t: &mut Vec<K8>| { let h = t.len() / 2; *t = (0..h).map(|k| t[2 * k].add(&r.mul(&t[2 * k + 1].sub(&t[2 * k])))).collect(); };
        for s in 0..n { fold(&mut w[s]); fold(&mut q[s]); fold(&mut chi[s]); }
        xi.push(r);
    }
    lap("phase 2 (fold + coarse)");
    (rounds, xi, w, q, chi, pre_alpha, mo_prefix)
    };
    let nvars = ell + 1;
    let mut rounds = rounds;
    // ---- Final variable z (as in the reference).
    let d = degs[ell];
    let mut g = vec![K8::ZERO; d + 1];
    for (t, gt) in g.iter_mut().enumerate() {
        let tk = K8::from_base(t as u64);
        let zz = tk.mul(&K8::ONE.sub(&tk));
        let mut acc = K8::ZERO;
        for s in 0..n {
            let v = w[s][0].add(&zz.mul(&u[s].0[0]));
            let range = gp[s].mul(&q_of(&v)).add(&gp[n + s].mul(&chi[s][0]).mul(&b_of(&v)));
            acc = acc.add(&pre_alpha.mul(&range)).add(&q[s][0].mul(&v));
        }
        *gt = acc.mul(&half()).add(&mo_prefix.add(&m_term(&a_coef[ell], &tk)));
    }
    tr.absorb_k(Item::RoundPoly, &g);
    let rho = tr.challenge_k_not01(Stage::LinkRound, 0, ell as u64);
    rounds.push(g);
    let point = Point { xi, rho };
    let sources: Vec<SourceVec> = if c_src.is_none() {
        raw.iter().zip(&r_cols).map(|(rw, r)| SourceVec { w: raw_to_cols(rw), r: r.clone() }).collect()
    } else { Vec::new() };
    let (targets, target_o) = match c_src {
        None => (sources.iter().map(|s| f_x(p, seed, &point, &s.w, &s.r)).collect::<Vec<Tuple>>(), f_ox(p, seed, &point, &s_o)),
        Some(cs) => {
            let ev = svc.ev(p, raw, &point.xi);
            let rf = point.rho.mul(&K8::ONE.sub(&point.rho));
            let targets = (0..n).map(|e| Tuple {
                c: cs[e].clone(),
                g: g_commit[e].clone(),
                a: ev[e].add(&super::pcs_maps::decode_h(&r_cols[e][..A_ENC_COLS]).scale_k(&rf)),
            }).collect();
            let t = super::pcs_maps::e_o(&s_o[..A_ENC_COLS * p.aux_coeffs], &p.degs(), &point.aux_vars());
            (targets, super::auth::AuxTuple { c: c_o.clone(), t })
        }
    };
    let a_hat: Vec<AElem> = targets.iter().map(|t| t.a.clone()).collect();
    lap("final round + targets");
    for a in &a_hat { tr.absorb_k(Item::Endpoint, &a.0); }
    tr.absorb_k(Item::Endpoint, &target_o.t.0);
    let proof = LinkProof { g_commit, c_o, rounds, a_hat, t_o: target_o.t.clone() };
    let _ = (DS, &a_aux);
    LinkOutput { proof, point, sources: if sources.is_empty() { r_cols.into_iter().map(|r| SourceVec { w: Vec::new(), r }).collect() } else { sources }, s_o, targets, target_o }
}

pub(crate) fn eqt_k(a: &K8, r: &K8) -> K8 { a.mul(r).add(&K8::ONE.sub(a).mul(&K8::ONE.sub(r))) }

/// g(t) = pre·eq(α_i, t)·range[t] + query[t] + 2^{N−1−i}(Σ_{j<i} m_j(r_j) + m_i(t)).
#[allow(clippy::too_many_arguments)]
pub(crate) fn finish_round(sum: &[K8], pre: &K8, a_i: &K8, d: usize, nvars: usize, i: usize, mo: &K8, ac: &[K8]) -> Vec<K8> {
    let scale = K8::from_base(1u64 << (nvars - 1 - i));
    (0..=d).map(|t| {
        let tk = K8::from_base(t as u64);
        pre.mul(&eqt(a_i, t)).mul(&sum[t]).add(&sum[d + 1 + t]).add(&scale.mul(&mo.add(&m_term(ac, &tk))))
    }).collect()
}

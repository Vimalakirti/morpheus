//! GPU backend of the structured Π_link prover
//! (`cuda_almost_goldilocks/morpheus_link.cuh`).

use super::field8::K8;
use super::link_fast::{Phase1, Phase1Input, RoundTables, TermInfo};
use almost_goldilocks_cuda::{ffi, DeviceBuffer};
use std::os::raw::c_void;

#[repr(C)]
#[derive(Clone, Copy)]
struct BlockDesc { src: u32, id: u32, toff: u32, tcnt: u32, qoff: u32, pad: [u32; 3], cr: [u64; 8], cb: [u64; 8] }
#[repr(C)]
#[derive(Clone, Copy)]
struct BlockTerm { term: u32, pad: u32, c: [u64; 8] }
#[repr(C)]
#[derive(Clone, Copy)]
struct TermTab { aoff: u32, boff: u32, pa: u32, pad: u32 }

const NACC: usize = 10;

fn as_bytes<T>(v: &[T]) -> &[u8] { unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) } }
fn upload<T>(v: &[T]) -> DeviceBuffer<u8> {
    let b = as_bytes(v);
    let mut d = DeviceBuffer::<u8>::new(b.len().max(1)).expect("device alloc");
    if !b.is_empty() { d.copy_from_slice(b).expect("htod"); }
    d
}
fn kwords(v: &[K8]) -> Vec<u64> { v.iter().flat_map(|k| k.words()).collect() }

pub struct GpuPhase1 {
    raw: DeviceBuffer<i8>,
    src_len: u64,
    b0: usize,
    nblocks: u32,
    blocks: DeviceBuffer<u8>,
    bterms: DeviceBuffer<u8>,
    qd: DeviceBuffer<u8>,
    pub kernel_ms: f64,
}

impl GpuPhase1 {
    pub fn new(inp: Phase1Input) -> Self {
        let src_len = inp.raw[0].len() as u64;
        let flat: Vec<i8> = inp.raw.iter().flat_map(|r| r.iter().copied()).collect();
        let raw = DeviceBuffer::from_slice(&flat).expect("raw upload");
        let mut descs = Vec::with_capacity(inp.blocks.len());
        let mut bts = Vec::new();
        let mut qd: Vec<u64> = Vec::new();
        for blk in inp.blocks {
            let cr = blk.c_alpha.mul(&inp.g_range[blk.source]);
            let cb = if blk.chi { blk.c_alpha.mul(&inp.g_bit[blk.source]) } else { K8::ZERO };
            let qoff = match &blk.qd { Some(t) => { let o = (qd.len() / 8) as u32; qd.extend(kwords(t)); o } None => u32::MAX };
            descs.push(BlockDesc { src: blk.source as u32, id: blk.id as u32, toff: bts.len() as u32, tcnt: blk.terms.len() as u32, qoff, pad: [0; 3], cr: cr.words(), cb: cb.words() });
            for bt in &blk.terms { bts.push(BlockTerm { term: bt.term as u32, pad: 0, c: bt.c.words() }); }
        }
        if std::env::var("MORPHEUS_TIMING").is_ok() {
            let dense = descs.iter().filter(|d| d.qoff != u32::MAX).count();
            eprintln!("[link]   {} blocks ({} dense-q), {} block terms, {} sources", descs.len(), dense, bts.len(), inp.raw.len());
        }
        GpuPhase1 { raw, src_len, b0: inp.b0, nblocks: descs.len() as u32, blocks: upload(&descs), bterms: upload(&bts), qd: upload(&qd), kernel_ms: 0.0 }
    }
}

impl Phase1 for GpuPhase1 {
    fn round(&mut self, rt: &RoundTables, terms: &[TermInfo], d: usize) -> Vec<K8> {
        let tr0 = std::time::Instant::now();
        let (mut a, mut b, mut tt) = (Vec::new(), Vec::new(), Vec::new());
        for k in 0..terms.len() {
            tt.push(TermTab { aoff: (a.len() / 8) as u32, boff: (b.len() / 8) as u32, pa: rt.pa[k] as u32, pad: 0 });
            a.extend(kwords(&rt.a[k]));
            b.extend(kwords(&rt.b[k]));
        }
        let (da, db, dt) = (upload(&a), upload(&b), upload(&tt));
        let (de, dal) = (upload(&kwords(&rt.e_pre)), upload(&kwords(&rt.e_alpha)));
        let mut out = vec![0u64; 8 * NACC * 8192];
        let t = std::time::Instant::now();
        let grid = unsafe {
            ffi::agl_mlink_phase1(self.raw.as_ptr(), self.src_len, self.b0 as i32, rt.i as i32, d as i32,
                self.blocks.as_ptr() as *const c_void, self.nblocks, self.bterms.as_ptr() as *const c_void, dt.as_ptr() as *const c_void,
                da.as_ptr() as *const u64, db.as_ptr() as *const u64, de.as_ptr() as *const u64, dal.as_ptr() as *const u64,
                self.qd.as_ptr() as *const u64, out.as_mut_ptr())
        };
        self.kernel_ms += t.elapsed().as_secs_f64() * 1e3;
        if std::env::var("MORPHEUS_TIMING").is_ok() { eprintln!("[link]   round {} kernel {:.1} ms, upload {:.1} ms, {} terms", rt.i, t.elapsed().as_secs_f64() * 1e3, (t - tr0).as_secs_f64() * 1e3, terms.len()); }
        assert!(grid > 0 && (grid as usize) <= 8192, "phase1 kernel failed ({grid})");
        let mut sum = vec![K8::ZERO; 2 * (d + 1)];
        for g in 0..grid as usize {
            for (k, sk) in sum.iter_mut().enumerate() {
                let o = (g * NACC + k) * 8;
                *sk = sk.add(&K8(out[o..o + 8].try_into().unwrap()));
            }
        }
        sum
    }
    fn fold(&mut self, e_full: &[K8]) -> Vec<K8> {
        let de = upload(&kwords(e_full));
        let mut dout = DeviceBuffer::<u64>::new(8 * self.nblocks as usize).expect("alloc");
        let rc = unsafe { ffi::agl_mlink_fold(self.raw.as_ptr(), self.src_len, self.b0 as i32, self.blocks.as_ptr() as *const c_void, self.nblocks, de.as_ptr() as *const u64, dout.as_mut_ptr()) };
        assert_eq!(rc, 0, "fold kernel failed");
        dout.to_vec().unwrap().chunks(8).map(|c| K8(c.try_into().unwrap())).collect()
    }
}

pub fn gpu_backend<'a>(inp: Phase1Input<'a>) -> Box<dyn Phase1 + 'a> { Box::new(GpuPhase1::new(inp)) }

// ======================================================================
// v2: region-organised GPU engine (morpheus_link2.cuh)
// ======================================================================

use super::link::{interpolate, Statement};
use std::collections::HashMap;
use super::link_fast::{eqb, eqt_k, finish_round, m_term, term_tables, DENSE_P};
use super::field8::eq_table;
use super::params::Profile;
use super::transcript::{Item, MTranscript, Stage};
use rayon::prelude::*;

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct RegionC { src: u32, base_blk: u32, h: u32, p: u32, flags: u32, toff: u32, tcnt: u32, qd_off: u32 }
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct TermC { pa: u32, aoff: u32, boff: u32, hl_off: u32, hh_off: u32, h1: u32, pad0: u32, pad1: u32, cs: [u64; 8] }
#[repr(C)]
#[derive(Clone, Copy)]
struct Blk { src: u32, id: u32, region: u32, k: u32 }
#[repr(C)]
#[derive(Clone, Copy)]
struct ConstsC {
    ca_lo: u64, ca_hi: u64, ca_lo_bits: i32, pad: i32,
    g_range: u64, g_bit: u64,
    regions: u64, terms: u64, hl: u64, hh: u64,
    a: u64, b: u64, qd: u64, qidx: u64,
    bco: u64, oh_base: u64,
}

fn dptr<T>(b: &DeviceBuffer<T>) -> u64 { b.as_ptr() as u64 }
fn up_k(v: &[K8]) -> DeviceBuffer<u64> { DeviceBuffer::from_slice(&kwords(v)).expect("upload") }
fn kvec(words: &[u64]) -> Vec<K8> { words.chunks(8).map(|c| K8(c.try_into().unwrap())).collect() }

/// Phases 1 and 2 of the structured link on the GPU (v2). Returns the round
/// messages, the point ξ, the final (w, q, χ) per source, Π eq(α_j, r_j) and
/// the M_O prefix — exactly what the CPU phases produce.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn phases_v2(
    p: &Profile, st: &Statement, raw: &[Vec<i8>], b0: usize, alpha: &[K8], eta: &K8, gp: &[K8],
    degs: &[usize], a_coef: &[Vec<K8>], tr: &mut MTranscript, dev_raw: Option<&DeviceBuffer<i8>>,
    keep: Option<&std::sync::Mutex<Vec<Kept>>>,
) -> (Vec<Vec<K8>>, Vec<K8>, Vec<K8>, Vec<K8>, Vec<K8>, K8, K8) {
    // Device shards (MORPHEUS_GPUS): sources go to devices greedily by
    // weighted load, from the raw digits and the declared bit blocks alone,
    // so the uploads run while the host prepares terms and tables. Plane
    // rounds 0..3 run the generic kernel on non-one-hot blocks only (bit
    // blocks stand in for one-hot ones here); streamed mode scales device 0
    // down (MORPHEUS_SHARD0_WEIGHT, default 0.5), it also hosts the
    // authentication.
    let n = raw.len();
    let ngpu = std::env::var("MORPHEUS_GPUS").ok().and_then(|s| s.parse().ok()).unwrap_or(1usize).max(1).min(n.max(1));
    let wgen: usize = std::env::var("MORPHEUS_GEN_WEIGHT").ok().and_then(|s| s.parse().ok()).unwrap_or(5);
    let nblk = raw[0].len() >> b0;
    let per_src: Vec<usize> = (0..n).into_par_iter().map(|s| {
        let mut bit = vec![false; nblk];
        for &(base, lg) in &st.bits[s] { let (a, c) = (base >> b0, 1usize << lg.saturating_sub(b0)); for x in bit.iter_mut().skip(a).take(c) { *x = true; } }
        raw[s].par_chunks(1usize << b0).enumerate().map(|(k, c)| if c.iter().any(|&d| d != 0) { if bit[k] { 1 } else { wgen } } else { 0 }).sum::<usize>()
    }).collect();
    let w0: f64 = if keep.is_some() { std::env::var("MORPHEUS_SHARD0_WEIGHT").ok().and_then(|s| s.parse().ok()).unwrap_or(0.5) } else { 1.0 };
    let cap: Vec<f64> = (0..ngpu).map(|g| if g == 0 { w0 } else { 1.0 }).collect();
    let mut by_load: Vec<usize> = (0..n).collect();
    by_load.sort_by_key(|&s| std::cmp::Reverse(per_src[s]));
    let mut load = vec![0usize; ngpu];
    let mut owner = vec![0usize; n];
    for s in by_load {
        let g = (0..ngpu).min_by(|&a, &b| ((load[a] + per_src[s] + 1) as f64 / cap[a]).partial_cmp(&((load[b] + per_src[s] + 1) as f64 / cap[b])).unwrap()).unwrap();
        owner[s] = g;
        load[g] += per_src[s] + 1;
    }
    let src_len = raw[0].len();
    std::thread::scope(|sc| {
        let ups: Vec<Option<std::thread::ScopedJoinHandle<'_, DeviceBuffer<i8>>>> = (0..ngpu).map(|g| {
            if g == 0 && dev_raw.is_some() { return None; }
            let owner = &owner;
            Some(sc.spawn(move || {
                almost_goldilocks_cuda::set_device(g as i32).expect("set device");
                let srcs: Vec<usize> = (0..n).filter(|&s| owner[s] == g).collect();
                let mut d = DeviceBuffer::<i8>::new((srcs.len() * src_len).max(1)).expect("raw alloc");
                for (j, &s) in srcs.iter().enumerate() { d.copy_from_slice_at(j * src_len, &raw[s]).expect("raw upload"); }
                d
            }))
        }).collect();
        phases_v2_body(p, st, raw, b0, alpha, eta, gp, degs, a_coef, tr, dev_raw, keep, owner.clone(), ups)
    })
}

#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn phases_v2_body<'u>(
    p: &Profile, st: &Statement, raw: &[Vec<i8>], b0: usize, alpha: &[K8], eta: &K8, gp: &[K8],
    degs: &[usize], a_coef: &[Vec<K8>], tr: &mut MTranscript, dev_raw: Option<&DeviceBuffer<i8>>,
    keep: Option<&std::sync::Mutex<Vec<Kept>>>, owner: Vec<usize>, mut ups: Vec<Option<std::thread::ScopedJoinHandle<'u, DeviceBuffer<i8>>>>,
) -> (Vec<Vec<K8>>, Vec<K8>, Vec<K8>, Vec<K8>, Vec<K8>, K8, K8) {
    let tm = std::env::var("MORPHEUS_TIMING").is_ok();
    let mut t0 = std::time::Instant::now();
    let mut lap = |what: &str| { if tm { eprintln!("[link2] {what:28} {:8.1} ms", t0.elapsed().as_secs_f64() * 1e3); } t0 = std::time::Instant::now(); };
    let ell = p.ell();
    let n = st.n;
    let nvars = ell + 1;
    // ---- Regions and terms (grouped by (source, base)).
    let mut reg_index: HashMap<(usize, usize), usize> = HashMap::new();
    struct Reg { src: usize, base: usize, m: usize, p: usize, terms: Vec<usize> }
    let mut regs: Vec<Reg> = Vec::new();
    let mut terms: Vec<TermInfo> = Vec::new();
    let mut tcs: Vec<K8> = Vec::new();
    let mut treg: Vec<usize> = Vec::new();
    // Term data per row in parallel; regions assigned in row order.
    let per_row: Vec<Vec<(TermInfo, K8)>> = st.rows.par_iter().enumerate().map(|(j, row)| {
        let coef = eta.mul(&gp[2 * n + j]);
        row.terms.iter().map(|t| (TermInfo { p: t.region.p, pw: t.pw.clone(), r: t.r.clone() }, coef.mul(&t.scale))).collect()
    }).collect();
    for (row, infos) in st.rows.iter().zip(per_row) {
        for (t, (info, c)) in row.terms.iter().zip(infos) {
            let key = (t.region.source, t.region.base);
            let ri = *reg_index.entry(key).or_insert_with(|| { regs.push(Reg { src: key.0, base: key.1, m: t.region.m, p: t.region.p, terms: Vec::new() }); regs.len() - 1 });
            let ti = terms.len();
            terms.push(info);
            tcs.push(c);
            treg.push(ri);
            regs[ri].terms.push(ti);
        }
    }
    lap("terms");
    let bits: std::collections::HashSet<(usize, usize)> = st.bits.iter().enumerate().flat_map(|(s, bl)| bl.iter().map(move |&(b, _)| (s, b))).collect();
    // Terms in region order (each region's terms contiguous).
    let order: Vec<usize> = regs.iter().flat_map(|r| r.terms.iter().copied()).collect();
    let pos_of: Vec<usize> = { let mut v = vec![0; order.len()]; for (k, &t) in order.iter().enumerate() { v[t] = k; } v };
    let mut terms_c: Vec<TermC> = vec![TermC::default(); order.len()];
    // Split high-index eq tables of every term, written in parallel into
    // flat buffers at precomputed offsets.
    // Dense regions (many planes or many terms) fold their terms into dense
    // query tables; their terms need no index tables.
    let is_dense = |r: &Reg| r.p > DENSE_P || r.terms.len() > 32;
    let hsz: Vec<(usize, usize, usize)> = order.iter().map(|&t| {
        let r = &regs[treg[t]];
        if is_dense(r) { return (0, 0, 0); }
        let h = r.m + r.p - b0;
        (h / 2, 1usize << (h / 2), 1usize << (h - h / 2))
    }).collect();
    let (mut lo_off, mut hi_off) = (Vec::with_capacity(order.len()), Vec::with_capacity(order.len()));
    let (mut nlo, mut nhi) = (0usize, 0usize);
    for &(_, a, b) in &hsz { lo_off.push(nlo); hi_off.push(nhi); nlo += a; nhi += b; }
    let mut hl_all = vec![K8::ZERO; nlo];
    let mut hh_all = vec![K8::ZERO; nhi];
    {
        let (pl, ph) = (SyncPtr(hl_all.as_mut_ptr()), SyncPtr(hh_all.as_mut_ptr()));
        order.par_iter().enumerate().for_each(|(k, &t)| {
            let r = &regs[treg[t]];
            if hsz[k].1 == 0 { return; }
            let idx_bits = b0 - r.p;
            let h1 = hsz[k].0;
            let lo = eq_table(&terms[t].r[idx_bits..idx_bits + h1]);
            let hi = eq_table(&terms[t].r[idx_bits + h1..]);
            // Disjoint ranges [off, off + len) per term.
            unsafe {
                std::ptr::copy_nonoverlapping(lo.as_ptr(), pl.get().add(lo_off[k]), lo.len());
                std::ptr::copy_nonoverlapping(hi.as_ptr(), ph.get().add(hi_off[k]), hi.len());
            }
        });
    }
    for (k, &t) in order.iter().enumerate() {
        terms_c[k] = TermC { pa: 0, aoff: 0, boff: 0, hl_off: lo_off[k] as u32, hh_off: hi_off[k] as u32, h1: hsz[k].0 as u32, pad0: 0, pad1: 0, cs: tcs[t].words() };
    }
    lap("hi tables");
    // One-hot detection (bit-typed regions whose indices each hold one 1).
    let onehot: Vec<bool> = regs.par_iter().map(|r| {
        if r.p == 0 || r.p > DENSE_P || !bits.contains(&(r.src, r.base)) { return false; }
        let np = 1usize << r.p;
        raw[r.src][r.base..r.base + (1usize << (r.m + r.p))].chunks(np).all(|c| c.iter().filter(|&&d| d != 0).count() == 1 && c.iter().all(|&d| d == 0 || d == 1))
    }).collect();
    lap("one-hot detection");
    // Dense query tables (many-plane regions).
    let mut qtabs: Vec<K8> = Vec::new();
    let mut qidx: Vec<u32> = Vec::new();
    let mut regions_c: Vec<RegionC> = Vec::new();
    let mut toff = 0u32;
    for (ri, r) in regs.iter().enumerate() {
        let h = r.m + r.p - b0;
        let dense = r.p > DENSE_P || r.terms.len() > 32;
        let mut flags = 0u32;
        if bits.contains(&(r.src, r.base)) { flags |= 1; }
        if onehot[ri] && r.terms.len() <= 4 && r.p <= 5 { flags |= 2; }
        let qd_off = qidx.len() as u32;
        if dense {
            flags |= 4;
            let idx_bits = b0 - r.p;
            // (block k, term) pairs: a term whose high point bits are boolean
            // addresses one block, others every block. Each block's table is
            // accumulated in place (no per-term contribution vectors).
            let kb = |ti: &TermInfo| -> Option<usize> {
                if ti.r[idx_bits..].iter().all(|x| *x == K8::ZERO || *x == K8::ONE) { Some(ti.r[idx_bits..].iter().enumerate().fold(0usize, |a, (jj, x)| a | (((*x == K8::ONE) as usize) << jj))) } else { None }
            };
            let mut pairs: Vec<(usize, usize)> = r.terms.par_iter().flat_map_iter(|&t| match kb(&terms[t]) {
                Some(k) => vec![(k, t)],
                None => (0..1usize << h).map(|k| (k, t)).collect(),
            }).collect();
            pairs.par_sort_unstable();
            let mut groups: Vec<(usize, usize, usize)> = Vec::new();
            for (q, &(k, _)) in pairs.iter().enumerate() {
                match groups.last_mut() { Some(g) if g.0 == k => g.2 = q + 1, _ => groups.push((k, q, q + 1)) }
            }
            let rp = r.p;
            let tabs: Vec<(usize, Vec<K8>)> = groups.par_iter().map(|&(k, a, b)| {
                let mut tab = vec![K8::ZERO; 1usize << b0];
                for &(_, t) in &pairs[a..b] {
                    let ti = &terms[t];
                    let hi = ti.r[idx_bits..].iter().enumerate().fold(K8::ONE, |acc, (jj, x)| acc.mul(&eqb(x, (k >> jj) & 1)));
                    let c = tcs[t].mul(&hi);
                    let el = eq_table(&ti.r[..idx_bits]);
                    for (il, e) in el.iter().enumerate() {
                        if e.is_zero() { continue; }
                        let ce = c.mul(e);
                        let row = &mut tab[il << rp..(il + 1) << rp];
                        for (bb, w) in ti.pw.iter().enumerate().take(1usize << rp) { if !w.is_zero() { row[bb] = row[bb].add(&ce.mul(w)); } }
                    }
                }
                (k, tab)
            }).collect();
            let mut ix = vec![u32::MAX; 1usize << h];
            for (k, tab) in tabs { ix[k] = (qtabs.len() >> b0) as u32; qtabs.extend(tab); }
            qidx.extend(ix);
        }
        let tcnt = if dense { 0 } else { r.terms.len() as u32 };
        assert!(tcnt as usize <= 32, "region with more than 32 query terms");
        regions_c.push(RegionC { src: r.src as u32, base_blk: (r.base >> b0) as u32, h: h as u32, p: r.p as u32, flags, toff, tcnt, qd_off });
        toff += r.terms.len() as u32;
    }
    // One-hot regions (never dense-q) keep, in qd_off, the base of their
    // terms in the compact per-term bin sums of the one-hot pass.
    let mut n_ohw = 0u32;
    for r in regions_c.iter_mut() { if r.flags & 2 != 0 { r.qd_off = n_ohw; n_ohw += r.tcnt; } }
    lap("dense tables + regions");
    // Nonzero blocks and their regions.
    let mut reg_sorted: Vec<(usize, usize, usize, usize)> = regs.iter().enumerate().map(|(i, r)| (r.src, r.base >> b0, 1usize << (r.m + r.p - b0), i)).collect();
    reg_sorted.sort();
    let nblk_src = 1usize << (ell - b0);
    let find = |s: usize, id: usize| -> Option<(usize, usize)> {
        let i = reg_sorted.partition_point(|x| (x.0, x.1) <= (s, id));
        if i == 0 { return None; }
        let (rs, rb, rn, ri) = reg_sorted[i - 1];
        if rs == s && id < rb + rn { Some((ri, id - rb)) } else { None }
    };
    let blocks: Vec<Blk> = (0..n).into_par_iter().flat_map_iter(|s| {
        let rw = &raw[s];
        (0..nblk_src).filter(move |&id| rw[id << b0..(id + 1) << b0].iter().any(|&d| d != 0)).map(move |id| (s, id)).collect::<Vec<_>>()
    }).map(|(s, id)| match find(s, id) { Some((ri, k)) => Blk { src: s as u32, id: id as u32, region: ri as u32, k: k as u32 }, None => Blk { src: s as u32, id: id as u32, region: u32::MAX, k: 0 } }).collect();
    lap("nonzero block scan");
    let (rid, kk): (Vec<u32>, Vec<u32>) = regions_c.iter().enumerate().flat_map(|(i, r)| (0..1u32 << r.h).map(move |k| (i as u32, k))).unzip();
    lap("region block list");

    // ---- Device shards (MORPHEUS_GPUS, default 1): contiguous source ranges
    // balanced by nonzero blocks; each device holds only its range (shard 0
    // may use the resident `dev_raw` holding every source). The coarse phase
    // also runs per shard.
    let src_len = raw[0].len() as u64;
    let ca_lo_bits = (ell - b0) / 2;
    let calo_h = eq_table(&alpha[b0..b0 + ca_lo_bits]);
    let cahi_h = eq_table(&alpha[b0 + ca_lo_bits..]);
    let mut claims: Option<(K8, K8)> = None;
    let no_claim = std::env::var("MORPHEUS_NO_CLAIM").is_ok();
    // Blocks grouped by region plane bits, so each round's one-hot list is a prefix.
    let is_oh = |b: &Blk| b.region != u32::MAX && regions_c[b.region as usize].flags & 2 != 0;
    let mut oh: Vec<Blk> = blocks.iter().filter(|b| is_oh(b)).copied().collect();
    oh.sort_by_key(|b| std::cmp::Reverse(regions_c[b.region as usize].p));
    let n_gen_base = blocks.len() - oh.len();
    if tm { eprintln!("[link2] {} nonzero blocks ({} one-hot), {} regions, {} terms, {} region blocks, {} dense tables, {} GPUs", blocks.len(), oh.len(), regions_c.len(), order.len(), rid.len(), qtabs.len() >> b0, ups.len()); }
    let ngpu = ups.len();
    let mut shards: Vec<Shard> = std::thread::scope(|sc| {
        let hs: Vec<_> = (0..ngpu).map(|g| {
            let up = ups[g].take();
            let (owner, blocks, oh, regions_c, terms_c, qtabs, qidx, hl_all, hh_all) = (&owner, &blocks, &oh, &regions_c, &terms_c, &qtabs, &qidx, &hl_all, &hh_all);
            let (calo_h, cahi_h) = (&calo_h, &cahi_h);
            sc.spawn(move || {
                almost_goldilocks_cuda::set_device(g as i32).expect("set device");
                // Local numbering: owned sources in order (shard 0 over the
                // resident `dev_raw` keeps the global numbering).
                let global = g == 0 && dev_raw.is_some();
                let srcs: Vec<usize> = (0..n).filter(|&s| owner[s] == g).collect();
                let mut loc = vec![u32::MAX; n];
                if global { for s in 0..n { loc[s] = s as u32; } } else { for (j, &s) in srcs.iter().enumerate() { loc[s] = j as u32; } }
                let raw_own = if global { None } else { Some(up.expect("shard upload").join().expect("raw upload")) };
                let mine = |b: &Blk| owner[b.src as usize] == g;
                let rl = |b: &Blk| Blk { src: loc[b.src as usize], ..*b };
                let gen_base: Vec<Blk> = blocks.iter().filter(|b| mine(b) && !(b.region != u32::MAX && regions_c[b.region as usize].flags & 2 != 0)).map(rl).collect();
                let oh_g: Vec<Blk> = oh.iter().filter(|b| mine(b)).map(rl).collect();
                let mut gen_all = gen_base.clone();
                gen_all.extend(oh_g.iter().rev().copied());
                // Per-block constant slots: cr, cb and one per region term.
                let mut bco: Vec<u64> = Vec::with_capacity(gen_all.len());
                let mut bc_len = 0u64;
                for b in &gen_all {
                    bco.push(bc_len);
                    let tc = if b.region == u32::MAX { 0 } else { regions_c[b.region as usize].tcnt as u64 };
                    bc_len += 2 + tc;
                }
                let bc_len = bc_len as usize;
                let mut regions_l: Vec<RegionC> = regions_c.iter().map(|r| RegionC { src: loc[r.src as usize], ..*r }).collect();
                // Dense query tables of this device's dense regions only.
                let (mut qt_g, mut qi_g): (Vec<K8>, Vec<u32>) = (Vec::new(), Vec::new());
                let tl = 1usize << b0;
                for (ri, r) in regions_c.iter().enumerate() {
                    if r.flags & 4 == 0 || owner[r.src as usize] != g { continue; }
                    let off = qi_g.len() as u32;
                    for k in 0..1usize << r.h {
                        let q = qidx[r.qd_off as usize + k];
                        if q == u32::MAX { qi_g.push(u32::MAX); continue; }
                        qi_g.push((qt_g.len() / tl) as u32);
                        qt_g.extend_from_slice(&qtabs[q as usize * tl..(q as usize + 1) * tl]);
                    }
                    regions_l[ri].qd_off = off;
                }
                let order_l: Vec<usize> = if global { (0..n).collect() } else { srcs.clone() };
                let gr: Vec<K8> = order_l.iter().map(|&s| gp[s]).collect();
                let gb: Vec<K8> = order_l.iter().map(|&s| gp[n + s]).collect();
                let mut sh = Shard {
                    dev: g as i32, raw: raw_own, srcs: order_l, owned: srcs.clone(),
                    calo: up_k(calo_h), cahi: up_k(cahi_h), grange: up_k(if gr.is_empty() { &[K8::ZERO][..] } else { &gr }), gbit: up_k(if gb.is_empty() { &[K8::ZERO][..] } else { &gb }),
                    regions: upload(&regions_l), hl: up_k(hl_all), hh: up_k(hh_all),
                    qd: up_k(if qt_g.is_empty() { &[K8::ZERO] } else { &qt_g }), qdl: None, qdl_len: qt_g.len(),
                    qidx: DeviceBuffer::from_slice(if qi_g.is_empty() { &[0u32][..] } else { &qi_g }).unwrap(),
                    gen_base: gen_base.len(), oh: oh_g, gen_all: upload(&gen_all),
                    bc: DeviceBuffer::<u64>::new(bc_len.max(1) * 8).unwrap(), bco: DeviceBuffer::from_slice(if bco.is_empty() { &[0u64][..] } else { &bco }).unwrap(), ca_lo_bits,
                };
                let d_t0 = upload(terms_c);
                let c0 = sh.consts(&d_t0, 0, 0, dptr(&sh.qd));
                let rc = unsafe { ffi::agl_mlink2_consts(sh.gen_all.as_ptr() as *const c_void, gen_all.len() as u32, &c0 as *const ConstsC as *const c_void, sh.bc.as_mut_ptr()) };
                assert_eq!(rc, 0, "block constants");
                sh
            })
        }).collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    });
    lap("upload");
    // One-hot plane rounds: bin sums from one pass (round 0), then host work.
    let mut oh_sums: Option<(Vec<K8>, Vec<K8>)> = None;
    let mut oh_terms: Vec<Vec<(usize, usize)>> = vec![Vec::new(); 6];
    {
        let mut seen = std::collections::HashSet::new();
        for b in &oh {
            if seen.insert(b.region) {
                let r = &regions_c[b.region as usize];
                for k in 0..r.tcnt as usize { oh_terms[r.p as usize].push((r.toff as usize + k, r.qd_off as usize + k)); }
            }
        }
    }
    let mut rounds = Vec::new();
    let mut xi: Vec<K8> = Vec::new();
    let mut mo_prefix = K8::ZERO;
    let mut pre_alpha = K8::ONE;
    let mut kms = 0.0;
    for i in 0..b0 {
        let d = degs[i];
        let t_round = std::time::Instant::now();
        // Rows t = 0, 1 only (weights are linear in t); dense regions' terms
        // live in the dense query tables.
        let tabs: Vec<(Vec<K8>, Vec<K8>, usize)> = order.par_iter().map(|&t| {
            if regs[treg[t]].p > DENSE_P || regs[treg[t]].terms.len() > 32 { (Vec::new(), Vec::new(), 0) } else { term_tables(&terms[t], &xi, i, 1, b0) }
        }).collect();
        let (mut a_all, mut b_all) = (Vec::new(), Vec::new());
        let mut tc = terms_c.clone();
        for (k, (a, b, pa)) in tabs.into_iter().enumerate() {
            tc[k].pa = pa as u32; tc[k].aoff = (a_all.len()) as u32; tc[k].boff = (b_all.len()) as u32;
            a_all.extend(a); b_all.extend(b);
        }
        let t_tab = t_round.elapsed().as_secs_f64() * 1e3;
        let e1 = eq_table(&xi);
        let e2: Vec<K8> = e1.iter().map(|x| x.mul(x)).collect();
        let e3: Vec<K8> = e1.iter().zip(&e2).map(|(x, y)| x.mul(y)).collect();
        let ea = eq_table(&alpha[i + 1..b0]);
        let n_oh = oh.partition_point(|b| regions_c[b.region as usize].p as usize > i);
        if tm { eprintln!("[link2]   round {i} host: tables {t_tab:.1} ms, prep {:.1} ms (A {} K, B {} K)", t_round.elapsed().as_secs_f64() * 1e3, a_all.len(), b_all.len()); }
        let t = std::time::Instant::now();
        // Reduced rounds: range at t = 0, 2, 3 and query at t = 0, 2 on the
        // device; t = 1 from the previous round's claims
        //   R_{i-1}(r) = (1 − α_i) R_i(0) + α_i R_i(1),  Q_{i-1}(r) = Q_i(0) + Q_i(1).
        let reduced = d == 4 && claims.is_some() && !alpha[i].is_zero() && !no_claim;
        let use_index = reduced && (1usize << (b0 - i)) <= 128 && i >= 4 && b0 >= 9 && std::env::var("MORPHEUS_NO_INDEX").is_err();
        let rin = RoundIn { i, d, b0, src_len, reduced, use_index, r_prev: xi.last().copied(), a_all: &a_all, b_all: &b_all, tc: &tc, e1: &e1, e2: &e2, e3: &e3, ea: &ea };
        let outs: Vec<(Vec<K8>, Option<(Vec<K8>, Vec<K8>)>)> = std::thread::scope(|sc| {
            let hs: Vec<_> = shards.iter_mut().map(|sh| {
                let (rin, regions_c, nterms) = (&rin, &regions_c, n_ohw as usize);
                sc.spawn(move || shard_round(sh, dev_raw, rin, regions_c, alpha, nterms))
            }).collect();
            hs.into_iter().map(|h| h.join().unwrap()).collect()
        });
        let mut sum = vec![K8::ZERO; 2 * (d + 1)];
        let mut ohs: Option<(Vec<K8>, Vec<K8>)> = None;
        for (s, o) in outs {
            for (a, b) in sum.iter_mut().zip(&s) { *a = a.add(b); }
            if let Some((g, w)) = o {
                ohs = Some(match ohs {
                    None => (g, w),
                    Some((g0, w0)) => (g0.iter().zip(&g).map(|(x, y)| x.add(y)).collect(), w0.iter().zip(&w).map(|(x, y)| x.add(y)).collect()),
                });
            }
        }
        let t_gen = t.elapsed().as_secs_f64() * 1e3;
        if i == 0 && !oh.is_empty() { oh_sums = ohs; }
        if n_oh > 0 {
            let (g, w) = oh_sums.as_ref().unwrap();
            let c = onehot_round(g, w, &oh_terms, i, b0, alpha, &e1, &e2, &e3, &a_all, &tc);
            for t in 0..=d {
                let tk = K8::from_base(t as u64);
                let (t2, t3) = (tk.mul(&tk), tk.mul(&tk).mul(&tk));
                sum[t] = sum[t].add(&c[0].add(&c[1].mul(&tk)).add(&c[2].mul(&t2)).add(&c[3].mul(&t3)));
                sum[d + 1 + t] = sum[d + 1 + t].add(&c[4].add(&c[5].mul(&tk)).add(&c[6].mul(&t2)));
            }
        }
        if reduced {
            let (cr, cq) = claims.unwrap();
            let a = alpha[i];
            sum[1] = cr.sub(&K8::ONE.sub(&a).mul(&sum[0])).mul(&a.inv());
            sum[d + 2] = cq.sub(&sum[d + 1]);
            // Extrapolate: range (cubic) from t = 0..3, query (quadratic) from t = 0..2.
            let s = |x: &K8, c: i64| -> K8 { if c >= 0 { x.scale(c as u64) } else { x.scale((-c) as u64).neg() } };
            sum[4] = s(&sum[0], -1).add(&s(&sum[1], 4)).add(&s(&sum[2], -6)).add(&s(&sum[3], 4));
            let (q0, q1, q2) = (sum[d + 1], sum[d + 2], sum[d + 3]);
            sum[d + 4] = s(&q0, 1).add(&s(&q1, -3)).add(&s(&q2, 3));
            sum[d + 5] = s(&q0, 3).add(&s(&q1, -8)).add(&s(&q2, 6));
        }
        kms += t.elapsed().as_secs_f64() * 1e3;
        if tm { eprintln!("[link2]   round {i}: {:.1} ms (devices {:.1} ms; {} one-hot, {} generic blocks)", t.elapsed().as_secs_f64() * 1e3, t_gen, n_oh, n_gen_base + oh.len() - n_oh); }
        let g = finish_round(&sum, &pre_alpha, &alpha[i], d, nvars, i, &mo_prefix, &a_coef[i]);
        tr.absorb_k(Item::RoundPoly, &g);
        let r = tr.challenge_k(Stage::LinkRound, 0, i as u64);
        rounds.push(g);
        claims = Some((interpolate(&sum[..d + 1], &r), interpolate(&sum[d + 1..2 * (d + 1)], &r)));
        mo_prefix = mo_prefix.add(&m_term(&a_coef[i], &r));
        pre_alpha = pre_alpha.mul(&eqt_k(&alpha[i], &r));
        xi.push(r);
    }
    lap(&format!("phase 1 ({kms:.0} ms kernels)"));
    // ---- Coarse phase on the devices, each over its own sources.
    let coarse = 1u64 << (ell - b0);
    let tfold: Vec<K8> = order.par_iter().map(|&t| {
        let rg = &regs[treg[t]];
        if rg.p > DENSE_P || rg.terms.len() > 32 { return K8::ZERO; }
        let tt = &terms[t];
        let ep = eq_table(&xi[..tt.p]);
        let pwv = tt.pw.iter().zip(&ep).fold(K8::ZERO, |acc, (a, b)| acc.add(&a.mul(b)));
        let idx_bits = b0 - tt.p;
        let er = tt.r[..idx_bits].iter().zip(&xi[tt.p..b0]).fold(K8::ONE, |acc, (a, b)| acc.mul(&eqt_k(a, b)));
        pwv.mul(&er)
    }).collect();
    let efull = eq_table(&xi);
    struct Coarse { w: DeviceBuffer<u64>, q: DeviceBuffer<u64>, ch: DeviceBuffer<u64>, w2: DeviceBuffer<u64>, q2: DeviceBuffer<u64>, c2: DeviceBuffer<u64> }
    let mut cs: Vec<Coarse> = std::thread::scope(|sc| {
        let hs: Vec<_> = shards.iter().map(|sh| {
            let (blocks, rid, kk, terms_c, regions_c, tfold, efull, owner) = (&blocks, &rid, &kk, &terms_c, &regions_c, &tfold, &efull, &owner);
            sc.spawn(move || {
                almost_goldilocks_cuda::set_device(sh.dev).expect("set device");
                let ns = sh.srcs.len();
                let tsz = (ns.max(1) as u64 * coarse * 8) as usize;
                let (mut w, mut q, mut ch) = (DeviceBuffer::<u64>::new(tsz).unwrap(), DeviceBuffer::<u64>::new(tsz).unwrap(), DeviceBuffer::<u64>::new(tsz).unwrap());
                let mut loc = vec![u32::MAX; n];
                for (j, &s) in sh.srcs.iter().enumerate() { loc[s] = j as u32; }
                let own = |s: usize| owner[s] as i32 == sh.dev;
                let mine: Vec<Blk> = blocks.iter().filter(|b| own(b.src as usize)).map(|b| Blk { src: loc[b.src as usize], ..*b }).collect();
                let (rid_g, kk_g): (Vec<u32>, Vec<u32>) = rid.iter().zip(kk.iter()).filter(|(r, _)| own(regions_c[**r as usize].src as usize)).map(|(a, b)| (*a, *b)).unzip();
                let d_blocks = upload(&mine);
                let (d_rid, d_kk) = (DeviceBuffer::from_slice(if rid_g.is_empty() { &[0u32][..] } else { &rid_g }).unwrap(), DeviceBuffer::from_slice(if kk_g.is_empty() { &[0u32][..] } else { &kk_g }).unwrap());
                let d_t0 = upload(terms_c);
                let (d_tfold, d_efull) = (up_k(tfold), up_k(efull));
                let consts = sh.consts(&d_t0, 0, 0, dptr(&sh.qd));
                let rc = unsafe {
                    ffi::agl_mlink2_coarse_init2(sh.raw_ptr(dev_raw, src_len), src_len, b0 as i32, &consts as *const ConstsC as *const c_void,
                        d_blocks.as_ptr() as *const c_void, mine.len() as u32, d_rid.as_ptr(), d_kk.as_ptr(), rid_g.len() as u64,
                        dptr(&d_tfold) as *const u64, dptr(&d_efull) as *const u64, coarse, 0, ns as u32, w.as_mut_ptr(), q.as_mut_ptr(), ch.as_mut_ptr())
                };
                assert_eq!(rc, 0, "coarse init failed");
                let h = (tsz / 2).max(8);
                Coarse { w, q, ch, w2: DeviceBuffer::<u64>::new(h).unwrap(), q2: DeviceBuffer::<u64>::new(h).unwrap(), c2: DeviceBuffer::<u64>::new(h).unwrap() }
            })
        }).collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    });
    lap("coarse init");
    let mut len = coarse;
    for i in b0..ell {
        let d = degs[i];
        let ea = eq_table(&alpha[i + 1..]);
        let sums: Vec<Vec<K8>> = std::thread::scope(|sc| {
            let hs: Vec<_> = shards.iter().zip(cs.iter()).map(|(sh, c)| {
                let ea = &ea;
                sc.spawn(move || {
                    almost_goldilocks_cuda::set_device(sh.dev).expect("set device");
                    let ns = sh.srcs.len();
                    if ns == 0 { return vec![K8::ZERO; 2 * (d + 1)]; }
                    let d_ea = up_k(ea);
                    let mut out = vec![0u64; 8 * 10];
                    let rc = unsafe { ffi::agl_mlink2_coarse_round(c.w.as_ptr(), c.q.as_ptr(), c.ch.as_ptr(), ns as u32, len, dptr(&sh.grange) as *const u64, dptr(&sh.gbit) as *const u64, dptr(&d_ea) as *const u64, d as i32, out.as_mut_ptr()) };
                    assert_eq!(rc, 0, "coarse round {i}");
                    kvec(&out)[..2 * (d + 1)].to_vec()
                })
            }).collect();
            hs.into_iter().map(|h| h.join().unwrap()).collect()
        });
        let mut sum = vec![K8::ZERO; 2 * (d + 1)];
        for v in &sums { for (a, b) in sum.iter_mut().zip(v) { *a = a.add(b); } }
        let g = finish_round(&sum, &pre_alpha, &alpha[i], d, nvars, i, &mo_prefix, &a_coef[i]);
        tr.absorb_k(Item::RoundPoly, &g);
        let r = tr.challenge_k(Stage::LinkRound, 0, i as u64);
        rounds.push(g);
        mo_prefix = mo_prefix.add(&m_term(&a_coef[i], &r));
        pre_alpha = pre_alpha.mul(&eqt_k(&alpha[i], &r));
        let rw = r.words();
        std::thread::scope(|sc| {
            for (sh, c) in shards.iter().zip(cs.iter_mut()) {
                let rw = &rw;
                sc.spawn(move || {
                    almost_goldilocks_cuda::set_device(sh.dev).expect("set device");
                    let ns = sh.srcs.len() as u32;
                    if ns == 0 { return; }
                    let Coarse { w, q, ch, w2, q2, c2 } = c;
                    for (src, dst) in [(&*w, &mut *w2), (&*q, &mut *q2), (&*ch, &mut *c2)] {
                        let rc = unsafe { ffi::agl_mlink2_coarse_fold(src.as_ptr(), dst.as_mut_ptr(), ns, len, rw.as_ptr()) };
                        assert_eq!(rc, 0, "coarse fold");
                    }
                    std::mem::swap(w, w2); std::mem::swap(q, q2); std::mem::swap(ch, c2);
                });
            }
        });
        len /= 2;
        xi.push(r);
    }
    lap("coarse rounds");
    let fin = |f: &dyn Fn(&Coarse) -> &DeviceBuffer<u64>| -> Vec<K8> {
        let mut v = vec![K8::ZERO; n];
        for (sh, c) in shards.iter().zip(&cs) {
            almost_goldilocks_cuda::set_device(sh.dev).unwrap();
            let ns = sh.srcs.len();
            if ns == 0 { continue; }
            let x = kvec(&f(c).read_slice(0, ns * 8).unwrap());
            for (j, &s) in sh.srcs.iter().enumerate() { if owner[s] as i32 == sh.dev { v[s] = x[j]; } }
        }
        v
    };
    let (fw, fq, fc) = (fin(&|c| &c.w), fin(&|c| &c.q), fin(&|c| &c.ch));
    drop(cs);
    if let Some(k) = keep {
        let mut k = k.lock().unwrap();
        for sh in shards.into_iter() {
            if let Some(buf) = sh.raw { if !sh.owned.is_empty() { k.push(Kept { dev: sh.dev, srcs: sh.owned, buf, phys: false }); } }
        }
    }
    almost_goldilocks_cuda::set_device(0).unwrap();
    (rounds, xi, fw, fq, fc, pre_alpha, mo_prefix)
}

/// G (per plane count p and bin t: [G3, G2]) and W (per term and bin) of the
/// one-hot regions; see `onehot_pre_kernel`.
#[allow(clippy::too_many_arguments)]
fn onehot_sums(
    raw_ptr: *const i8, src_len: u64, b0: usize, consts: &ConstsC, oh: &[Blk], d_bc: &DeviceBuffer<u64>,
    n_gen_base: usize, regions_c: &[RegionC], alpha: &[K8], nterms: usize,
) -> (Vec<K8>, Vec<K8>) {
    let maxp = regions_c.iter().map(|r| r.p as usize).max().unwrap_or(0);
    let (mut eai, mut eai_off) = (Vec::new(), Vec::new());
    for pp in 0..=maxp {
        eai_off.push(eai.len() as u32);
        if pp <= b0 { eai.extend(eq_table(&alpha[pp..b0])); }
    }
    let d_eai = up_k(&eai);
    let d_eai_off = DeviceBuffer::from_slice(&eai_off).unwrap();
    let d_oh = upload(oh);
    let ng = 6 * 32 * 2;
    let d_g = DeviceBuffer::from_slice(&vec![0u64; ng * 16]).unwrap();
    let d_w = DeviceBuffer::from_slice(&vec![0u64; nterms.max(1) * 32 * 16]).unwrap();
    let bc_oh = d_bc.as_ptr();
    let mut c2 = *consts;
    c2.oh_base = n_gen_base as u64;
    let consts = &c2;
    let rc = unsafe {
        ffi::agl_mlink2_onehot_pre(raw_ptr, src_len, b0 as i32, consts as *const ConstsC as *const c_void,
            d_oh.as_ptr() as *const c_void, oh.len() as u32, bc_oh, dptr(&d_eai) as *const u64, d_eai_off.as_ptr(),
            d_g.as_ptr() as *mut u64, d_w.as_ptr() as *mut u64)
    };
    assert_eq!(rc, 0, "one-hot sums");
    let halves = |v: Vec<u64>| -> Vec<K8> {
        const Q: u128 = 18446744069414584289;
        v.chunks(16).map(|c| K8(std::array::from_fn(|w| ((c[2 * w] as u128 + ((c[2 * w + 1] as u128 % Q) << 32)) % Q) as u64))).collect()
    };
    (halves(d_g.read_slice(0, ng * 16).unwrap()), halves(d_w.read_slice(0, nterms.max(1) * 32 * 16).unwrap()))
}

/// One-hot contribution of plane round i as polynomials in t:
/// [range c0..c3, query c0..c2] (same algebra as `onehot3_kernel`).
#[allow(clippy::too_many_arguments)]
fn onehot_round(
    g: &[K8], w: &[K8], oh_terms: &[Vec<(usize, usize)>], i: usize, b0: usize, alpha: &[K8],
    e1: &[K8], e2: &[K8], e3: &[K8], a_all: &[K8], tc: &[TermC],
) -> [K8; 7] {
    let mut c = [K8::ZERO; 7];
    let three = K8::from_base(3);
    let two = K8::from_base(2);
    for p in (i + 1)..oh_terms.len() {
        if p > b0 { break; }
        let np = 1usize << p;
        let eapr = eq_table(&alpha[i + 1..p]);
        // Query: Σ_T W_T[t]·A_T[0][pr], Σ_T W_T[t]·dA_T[pr] per bin.
        let zero = || vec![(K8::ZERO, K8::ZERO); np];
        let ys = oh_terms[p].par_iter().map(|&(k, kw)| {
            let (aoff, pa) = (tc[k].aoff as usize, tc[k].pa as usize);
            let mut v = zero();
            for t in 0..np {
                let wt = &w[kw * 32 + t];
                if wt.is_zero() { continue; }
                let pr = t >> (i + 1);
                let a0 = a_all[aoff + pr];
                let a1 = a_all[aoff + (1 << pa) + pr];
                v[t] = (wt.mul(&a0), wt.mul(&a1.sub(&a0)));
            }
            v
        }).reduce(zero, |mut x, y| { for (a, b) in x.iter_mut().zip(y) { a.0 = a.0.add(&b.0); a.1 = a.1.add(&b.1); } x });
        for t in 0..np {
            let (g3, g2) = (g[(p * 32 + t) * 2], g[(p * 32 + t) * 2 + 1]);
            let bit = (t >> i) & 1;
            let low = t & ((1 << i) - 1);
            let pr = t >> (i + 1);
            let (e, ep) = (e1[low], eapr[pr]);
            let x3 = g3.mul(&ep).mul(&e3[low]);
            let x2 = g2.mul(&ep).mul(&e2[low]);
            let x1 = g3.add(&g2).mul(&ep).mul(&e);
            let y0 = ys[t].0.mul(&e);
            let y1 = ys[t].1.mul(&e);
            if bit == 1 {
                c[3] = c[3].add(&x3); c[2] = c[2].add(&x2); c[1] = c[1].sub(&x1);
                c[5] = c[5].add(&y0); c[6] = c[6].add(&y1);
            } else {
                c[0] = c[0].add(&x3.add(&x2).sub(&x1));
                c[1] = c[1].add(&x1.sub(&three.mul(&x3)).sub(&two.mul(&x2)));
                c[2] = c[2].add(&three.mul(&x3).add(&x2));
                c[3] = c[3].sub(&x3);
                c[4] = c[4].add(&y0); c[5] = c[5].add(&y1.sub(&y0)); c[6] = c[6].sub(&y1);
            }
        }
    }
    c
}

#[derive(Clone, Copy)]
struct SyncPtr(*mut K8);
unsafe impl Send for SyncPtr {}
unsafe impl Sync for SyncPtr {}
impl SyncPtr { fn get(&self) -> *mut K8 { self.0 } }

/// A device's range of sources [lo, hi), kept after the link (streamed
/// mode) for the endpoints and the authentication; logical raw order until
/// `phys` (then the commitment's physical column order).
pub struct Kept { pub dev: i32, pub srcs: Vec<usize>, pub buf: DeviceBuffer<i8>, pub phys: bool }

/// One device's share of the plane rounds.
struct Shard {
    dev: i32,
    raw: Option<DeviceBuffer<i8>>,
    /// Local numbering: local index j is global source srcs[j] (`raw`,
    /// γ powers, blocks and regions use it); `owned` are this device's sources.
    srcs: Vec<usize>, owned: Vec<usize>,
    calo: DeviceBuffer<u64>, cahi: DeviceBuffer<u64>, grange: DeviceBuffer<u64>, gbit: DeviceBuffer<u64>,
    regions: DeviceBuffer<u8>, hl: DeviceBuffer<u64>, hh: DeviceBuffer<u64>,
    qd: DeviceBuffer<u64>, qdl: Option<DeviceBuffer<u64>>, qdl_len: usize, qidx: DeviceBuffer<u32>,
    gen_base: usize, oh: Vec<Blk>, gen_all: DeviceBuffer<u8>, bc: DeviceBuffer<u64>, bco: DeviceBuffer<u64>, ca_lo_bits: usize,
}

impl Shard {
    fn raw_ptr(&self, dev_raw: Option<&DeviceBuffer<i8>>, _src_len: u64) -> *const i8 {
        match &self.raw { Some(b) => b.as_ptr(), None => dev_raw.expect("raw sources").as_ptr() }
    }
    fn consts(&self, d_terms: &DeviceBuffer<u8>, a: u64, b: u64, qd: u64) -> ConstsC {
        ConstsC {
            ca_lo: dptr(&self.calo), ca_hi: dptr(&self.cahi), ca_lo_bits: self.ca_lo_bits as i32, pad: 0,
            g_range: dptr(&self.grange), g_bit: dptr(&self.gbit), regions: dptr(&self.regions), terms: dptr(d_terms),
            hl: dptr(&self.hl), hh: dptr(&self.hh), a, b, qd, qidx: dptr(&self.qidx),
            bco: dptr(&self.bco), oh_base: 0,
        }
    }
}

/// Host tables of one plane round, shared by the shards.
struct RoundIn<'a> {
    i: usize, d: usize, b0: usize, src_len: u64, reduced: bool, use_index: bool, r_prev: Option<K8>,
    a_all: &'a [K8], b_all: &'a [K8], tc: &'a [TermC], e1: &'a [K8], e2: &'a [K8], e3: &'a [K8], ea: &'a [K8],
}

/// One shard's plane round: fold its dense query tables with the previous
/// challenge, run its blocks; in round 0 also its one-hot bin sums.
#[allow(clippy::type_complexity)]
fn shard_round(sh: &mut Shard, dev_raw: Option<&DeviceBuffer<i8>>, rin: &RoundIn, regions_c: &[RegionC], alpha: &[K8], nterms: usize)
    -> (Vec<K8>, Option<(Vec<K8>, Vec<K8>)>) {
    almost_goldilocks_cuda::set_device(sh.dev).expect("set device");
    let (i, d, b0) = (rin.i, rin.d, rin.b0);
    if let Some(r) = rin.r_prev {
        if sh.qdl_len > 1 {
            let dst = DeviceBuffer::<u64>::new(8 * (sh.qdl_len / 2)).unwrap();
            let src = sh.qdl.as_ref().unwrap_or(&sh.qd);
            let rc = unsafe { ffi::agl_mpiop_sc_fold(src.as_ptr(), dst.as_ptr() as *mut u64, sh.qdl_len as u64, r.words().as_ptr()) };
            assert_eq!(rc, 0, "dense query fold");
            sh.qdl = Some(dst);
            sh.qdl_len /= 2;
        }
    }
    let raw_ptr = sh.raw_ptr(dev_raw, rin.src_len);
    let (d_a, d_b, d_t) = (up_k(rin.a_all), up_k(rin.b_all), upload(rin.tc));
    let (d_e1, d_e2, d_e3, d_ea) = (up_k(rin.e1), up_k(rin.e2), up_k(rin.e3), up_k(rin.ea));
    let n_oh = sh.oh.partition_point(|b| regions_c[b.region as usize].p as usize > i);
    let n_gen = sh.gen_base + (sh.oh.len() - n_oh);
    let consts = sh.consts(&d_t, dptr(&d_a), dptr(&d_b), sh.qdl.as_ref().map(dptr).unwrap_or(dptr(&sh.qd)));
    let sum = if n_gen == 0 { vec![K8::ZERO; 2 * (d + 1)] } else if rin.use_index {
        let mut o7 = vec![0u64; 7 * 8];
        let rc = unsafe {
            ffi::agl_mlink2_index(raw_ptr, rin.src_len, b0 as i32, i as i32, &consts as *const ConstsC as *const c_void,
                sh.gen_all.as_ptr() as *const c_void, n_gen as u32, sh.bc.as_ptr(), dptr(&d_e1) as *const u64, dptr(&d_ea) as *const u64, o7.as_mut_ptr())
        };
        assert_eq!(rc, 0, "link v2 index round {i} failed");
        let v = kvec(&o7);
        vec![v[0], K8::ZERO, v[2], v[3], K8::ZERO, v[4], K8::ZERO, v[6], K8::ZERO, K8::ZERO]
    } else {
        let mut out = vec![0u64; 8 * 10];
        let rc = unsafe {
            ffi::agl_mlink2_round(raw_ptr, rin.src_len, b0 as i32, i as i32, d as i32, rin.reduced as i32, &consts as *const ConstsC as *const c_void,
                std::ptr::null(), 0, sh.gen_all.as_ptr() as *const c_void, n_gen as u32, sh.bc.as_ptr(),
                dptr(&d_e1) as *const u64, dptr(&d_e2) as *const u64, dptr(&d_e3) as *const u64, dptr(&d_ea) as *const u64, out.as_mut_ptr())
        };
        assert_eq!(rc, 0, "link v2 round {i} failed");
        kvec(&out)[..2 * (d + 1)].to_vec()
    };
    let ohs = if i == 0 && !sh.oh.is_empty() {
        Some(onehot_sums(raw_ptr, rin.src_len, b0, &consts, &sh.oh, &sh.bc, sh.gen_base, regions_c, alpha, nterms))
    } else { None };
    (sum, ohs)
}

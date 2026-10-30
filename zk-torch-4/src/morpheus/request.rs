//! One inference request through the Morpheus prover: source commitments,
//! masked graph PIOP, Π_link and authentication; monolithic, or under
//! deferred fixed-weight opening (paper §Deferred Fixed-Weight Opening).
//!
//! Deferral. Model tensors (Role::Constant) are packed into their own
//! sources, committed once at registration. A request commits its own
//! sources, runs the PIOP, and turns every claim (e, r, η, z) on a model
//! tensor into a request row carry_k + η(W_G) = z and a deferred row
//! T̃_e(r) − carry_k = 0 (see `piop::Defer`). The carry bank is one extra
//! request source committed after the PIOP and before the link challenges.
//! The request opens its sources and carry bank. The finalization of an
//! ordered batch opens the model sources together with the carry banks of
//! every queued request against all deferred rows, under a transcript that
//! binds the model commitments, the queue order and each request's carry
//! commitment and transcript digest. Requests are accepted only together
//! with their finalization.

use super::auth::{AuthProof, Point, Tuple, AuxTuple};
use super::auth_gpu::{self, AuthDev, AuthTiming, DevSource};
use super::coins::Coins;
use super::field8::K8;
use super::keys::KeySeed;
use super::link::{self, LinkProof, Row, Statement};
use super::link_fast;
use super::link_gpu;
use super::params::Profile;
use super::pcs_gpu::{AuthReplica, PcsGpu};
use super::piop::{self, layout::{Layout, TKey}, Config, Ctx, Defer, Graph, ProverData, Walk};
use super::transcript::{Item, MTranscript, Stage};
use super::wire;
use crate::dag::{Dag, Witness};
use almost_goldilocks_cuda::DeviceBuffer;
use rayon::prelude::*;
use std::collections::BTreeMap;
use std::ops::Range;
use std::time::Instant;

pub struct Opts {
    pub profile: Profile,
    pub seed: KeySeed,
    pub coins: Coins,
    pub b0: usize,
    pub cfg: Config,
    pub defer: bool,
    pub carry_log: usize,
    pub auth: bool,
    pub verify: bool,
    pub auth_gpus: usize,
    pub timing: bool,
    /// Stream sources instead of keeping two resident copies on device 0:
    /// commitments through every device, the link and the authentication on
    /// per-device source ranges (MORPHEUS_STREAM=1, or automatically above
    /// MORPHEUS_STREAM_GB of sources, default 16).
    pub stream: Option<bool>,
}

/// The prover's private random tape: a fresh 512-bit OS seed per process.
/// MORPHEUS_COIN_SEED (128 hex digits) fixes it for reproducibility tests
/// only; a fixed seed reuses private masks across processes.
fn private_coins() -> Coins {
    use super::coins::CoinProfile;
    match std::env::var("MORPHEUS_COIN_SEED") {
        Ok(h) => {
            let h = h.trim();
            assert!(h.len() == 128 && h.bytes().all(|b| b.is_ascii_hexdigit()), "MORPHEUS_COIN_SEED: 128 hex digits");
            let seed: [u8; 64] = std::array::from_fn(|i| u8::from_str_radix(&h[2 * i..2 * i + 2], 16).unwrap());
            eprintln!("warning: MORPHEUS_COIN_SEED fixes the private random tape (tests only)");
            Coins::new(seed, CoinProfile::Ternary)
        }
        Err(_) => Coins::from_entropy(CoinProfile::Ternary),
    }
}

fn ev(k: &str, d: usize) -> usize { std::env::var(k).ok().and_then(|s| s.parse().ok()).unwrap_or(d) }

impl Opts {
    /// PROFILE, B0, TCL, SLOTS_LOG, DEFER, CARRY_LOG, AUTH, NO_VERIFY,
    /// MORPHEUS_AUTH_GPUS / MORPHEUS_GPUS, MORPHEUS_TIMING.
    pub fn from_env() -> Self {
        let profile = match std::env::var("PROFILE").ok().as_deref() { Some("mini") => Profile::mini(), Some("toy") => Profile::toy(), _ => Profile::ternary() };
        Opts {
            profile,
            seed: KeySeed([1; 8]),
            coins: private_coins(),
            b0: ev("B0", 11),
            cfg: Config { tcl: ev("TCL", 4), slots_log: ev("SLOTS_LOG", 17), req_planes_log: ev("REQ_PLANES_LOG", 4) },
            defer: std::env::var("DEFER").ok().as_deref() == Some("1"),
            carry_log: ev("CARRY_LOG", 14),
            auth: std::env::var("NO_AUTH").is_err(),
            verify: std::env::var("NO_VERIFY").is_err(),
            auth_gpus: std::env::var("MORPHEUS_AUTH_GPUS").or_else(|_| std::env::var("MORPHEUS_GPUS")).ok().and_then(|s| s.parse().ok()).unwrap_or(1),
            timing: std::env::var("MORPHEUS_TIMING").is_ok(),
            stream: std::env::var("MORPHEUS_STREAM").ok().map(|s| s == "1"),
        }
    }
}

/// Opening identifiers (coins, salts): request j and finalization batch b.
pub fn req_opening(j: u64) -> u64 { 1 + j }
pub fn fin_opening(b: u64) -> u64 { (1u64 << 32) + b }
/// Hiding-suffix coin owners of model and request sources: disjoint ranges
/// (model sources set bit 63; request j < 2^39 and source e < 2^24 never do).
fn model_owner(e: usize) -> u64 { (1u64 << 63) | e as u64 }
fn req_owner(j: u64, e: usize) -> u64 {
    assert!(e < 1 << 24 && j < (1 << 39) - 1, "coin owner out of range");
    ((1 + j) << 24) + e as u64
}

pub type Times = BTreeMap<&'static str, f64>;
fn lap(t: &mut Times, k: &'static str, t0: Instant) -> f64 { let s = t0.elapsed().as_secs_f64(); *t.entry(k).or_default() += s; s }

/// A compiled request instance: graph, prover data, source layout.
pub struct Prepared<'a> {
    pub g: Graph<'a>,
    pub pd: ProverData,
    pub lay: Layout,
    pub model: Range<usize>,
    pub req: Range<usize>,
    pub carry: Range<usize>,
    pub specs: Vec<(TKey, usize, usize, bool)>,
}

pub fn prepare<'a>(dag: &'a Dag, wit: &'a [Vec<Witness>], public: &[usize], o: &mut Opts) -> Prepared<'a> {
    // The carry bank (2^carry_log slots of 512 bit planes) fits one source.
    let cap = o.profile.source_payload_cols * super::ring::N;
    let max_lg = (usize::BITS - 1 - cap.leading_zeros()) as usize;
    o.carry_log = o.carry_log.min(max_lg - 9);
    // Graph-mask slots: at least SLOTS_LOG, and about 32 per graph node
    // (GPT-2 uses ~13; reductions draw a few pads and mask coefficients).
    let need = crate::util::arith::log2_ceil(dag.nodes.len().max(1) * 32);
    if need > o.cfg.slots_log { o.cfg.slots_log = need; }
    let t = Instant::now();
    let mut g = Graph::new(dag, wit, o.cfg.clone());
    for &e in public { g.public.insert(e, piop::witness_vals(&wit[e][0])); }
    g.prune_unclaimed();
    let pd = ProverData::new(&g, wit, Vec::new());
    let specs = piop::tensor_specs(&g, &pd);
    let (lay, model, req, carry) = if o.defer {
        let (mut m, mut r) = (Vec::new(), Vec::new());
        for &s in &specs { match s.0 { TKey::Edge(e) if g.model[e] => m.push(s), _ => r.push(s) } }
        let (lay, rg) = Layout::pack_groups(&o.profile, &[m, r, vec![(TKey::Carry, o.carry_log, 9, true)]], o.b0);
        (lay, rg[0].clone(), rg[1].clone(), rg[2].clone())
    } else {
        let lay = Layout::pack_min(&o.profile, &specs, o.b0);
        let n = lay.n_sources;
        (lay, 0..0, 0..n, n..n)
    };
    if o.timing { eprintln!("[prep] graph + prover data + layout {:.2}s: model sources {:?}, request {:?}, carry {:?}", t.elapsed().as_secs_f64(), model, req, carry); }
    Prepared { g, pd, lay, model, req, carry, specs }
}

/// Renumber the sources of `rows` through `map` (global -> local).
/// LINK_ROWS=a..b keeps only those rows (debugging).
fn remap(rows: &[Row], map: &(dyn Fn(usize) -> usize + Sync)) -> Vec<Row> {
    let rows: &[Row] = match std::env::var("LINK_ROWS").ok() {
        Some(r) => { let (a, b) = r.split_once("..").unwrap(); let (a, b): (usize, usize) = (a.parse().unwrap(), b.parse::<usize>().unwrap().min(rows.len())); &rows[a..b] }
        None => rows,
    };
    rows.par_iter().map(|r| {
        let mut r = r.clone();
        for t in r.terms.iter_mut() { t.region.source = map(t.region.source); }
        r
    }).collect()
}

fn fill_suffixes(p: &Profile, coins: &Coins, raw: &mut [Vec<i8>], owner: &dyn Fn(usize) -> u64) {
    for (e, r) in raw.iter_mut().enumerate() { link::fill_source_suffix_gpu(p, coins, owner(e), r); }
}

// ---------------------------------------------------------------- model registration

pub struct ModelReg {
    pub raw: Vec<Vec<i8>>,
    pub c: Vec<Vec<Vec<u64>>>,
    pub secs: f64,
}

/// One-time commitment to the model sources (deferral only).
pub fn register(pp: &Prepared, o: &Opts, pc: &mut PcsGpu, reps: &Replicas) -> ModelReg {
    let t = Instant::now();
    let mut raw = piop::fill_sources_in(&pp.g, &pp.pd, &pp.lay, o.profile.ell(), pp.model.clone());
    fill_suffixes(&o.profile, &o.coins, &mut raw, &model_owner);
    let _ = reps;
    let c = commit_streamed(o, pc, &raw);
    pc.ensure_w();
    ModelReg { raw, c, secs: t.elapsed().as_secs_f64() }
}

fn absorb_model(tr: &mut MTranscript, c_model: &[Vec<Vec<u64>>]) {
    tr.absorb(Item::Statement, &[c_model.len() as u64]);
    for c in c_model { for x in c { tr.absorb(Item::SourceCommitment, x); } }
}

/// The registry's public layout, literally: every tensor's key, index
/// width, planes, bit type and pieces (source, base, size, planes, plane
/// offset, block split).
fn absorb_layout(tr: &mut MTranscript, lay: &Layout) {
    let mut w = vec![lay.n_sources as u64, lay.ell as u64, lay.min_lg as u64, lay.tensors.len() as u64];
    for t in &lay.tensors {
        match t.key {
            TKey::Edge(e) => w.extend([1, e as u64]),
            TKey::Sel { node, chunk } => w.extend([2, node as u64, chunk as u64]),
            TKey::Wg => w.push(3),
            TKey::Carry => w.push(4),
        }
        w.extend([t.n as u64, t.p as u64, t.bits as u64, t.pieces.len() as u64]);
        for pc in &t.pieces {
            let r = &pc.region;
            w.extend([r.source as u64, r.base as u64, r.m as u64, r.p as u64, pc.plane0 as u64, pc.hb as u64, pc.hbits as u64]);
        }
    }
    tr.absorb(Item::Layout, &w);
}

fn request_transcript(o: &Opts, pp: &Prepared, j: u64, salt: &[u8; 64], c_model: Option<&[Vec<Vec<u64>>]>, c_req: &[Vec<Vec<u64>>]) -> MTranscript {
    let mut tr = MTranscript::new(o.profile.name, salt);
    absorb_layout(&mut tr, &pp.lay);
    if let Some(cm) = c_model {
        absorb_model(&mut tr, cm);
        tr.absorb(Item::Statement, &[pp.req.len() as u64, j]);
    } else {
        tr.absorb(Item::Statement, &[pp.lay.n_sources as u64, j]);
    }
    for c in c_req { for x in c { tr.absorb(Item::SourceCommitment, x); } }
    tr
}

/// Request digest bound by the finalization transcript.
fn digest(tr: &MTranscript) -> K8 { tr.clone().challenge_k(Stage::GraphMerge, 0, u64::MAX) }

// ---------------------------------------------------------------- auth plumbing

/// Authentication replicas on devices 1..auth_gpus (key caches persist).
pub struct Replicas { pub r: Vec<AuthReplica> }

impl Replicas {
    pub fn new(o: &Opts) -> Self {
        let r = std::thread::scope(|sc| {
            let hs: Vec<_> = (1..o.auth_gpus.max(1)).map(|g| { let (p, s) = (o.profile.clone(), o.seed); sc.spawn(move || AuthReplica::new(g as i32, &p, &s)) }).collect();
            hs.into_iter().map(|h| h.join().unwrap()).collect()
        });
        almost_goldilocks_cuda::set_device(0).unwrap();
        Replicas { r }
    }
}

/// Devices of this process (MORPHEUS_GPUS) other than 0.
fn extra_devs() -> Vec<i32> {
    let g: i32 = std::env::var("MORPHEUS_GPUS").ok().and_then(|s| s.parse().ok()).unwrap_or(1);
    (1..g.max(1)).collect()
}

/// Streamed commitments: device 0 with its key cache, the other devices
/// with temporary 3-prime caches.
fn commit_streamed(o: &Opts, pc: &mut PcsGpu, raw: &[Vec<i8>]) -> Vec<Vec<Vec<u64>>> {
    let ex = extra_devs();
    if ex.is_empty() { pc.ensure_w(); }
    let devs: Vec<(i32, &auth_gpu::Keys)> = if pc.keys.w.buf.len() > 1 { vec![(0, &pc.keys)] } else { vec![] };
    super::pcs_gpu::commit_streamed_on(&o.profile, Some((&o.seed, &ex)), &devs, raw)
}

impl Opts {
    pub fn streams(&self, nsrc: usize) -> bool {
        let gb = std::env::var("MORPHEUS_STREAM_GB").ok().and_then(|s| s.parse::<f64>().ok()).unwrap_or(12.0);
        self.stream.unwrap_or((nsrc << self.profile.ell()) as f64 > gb * 1e9)
    }
}

fn key_devs<'k>(pc: &'k PcsGpu, reps: &'k [AuthReplica]) -> Vec<(i32, &'k auth_gpu::Keys)> {
    let mut v = vec![(0, &pc.keys)];
    for r in reps { v.push((r.dev, &r.keys)); }
    v
}

/// Streamed authentication over the link's kept per-device source ranges.
fn authenticate_streamed(o: &Opts, opening: u64, pc: &mut PcsGpu, fast: &link::LinkOutput, t1: &MTranscript, times: &mut Times) -> AuthProof {
    let t = Instant::now();
    pc.kept_to_phys();
    pc.ensure_w();
    let k = pc.kept.lock().unwrap();
    let shards: Vec<auth_gpu::ShardSrc> = k.iter().map(|sh| {
        almost_goldilocks_cuda::set_device(sh.dev).unwrap();
        let r: Vec<i8> = sh.srcs.iter().flat_map(|&s| fast.sources[s].r.iter().flatten().map(|&v| v as i8)).collect();
        auth_gpu::ShardSrc { dev: sh.dev, srcs: sh.srcs.clone(), w: sh.buf.as_ptr(), len: sh.buf.len() / sh.srcs.len(), r: DeviceBuffer::from_slice(&r).unwrap() }
    }).collect();
    almost_goldilocks_cuda::set_device(0).unwrap();
    let s_o = DeviceBuffer::from_slice(&fast.s_o.iter().flatten().map(|&v| v as i8).collect::<Vec<_>>()).unwrap();
    let mut tm = AuthTiming::default();
    let (apf, _) = auth_gpu::prove_sharded(&o.profile, &pc.keys, &shards, &s_o, fast.sources.len(), &o.coins, opening, t1, &fast.point, true, &mut tm).expect("authentication");
    drop(k);
    let secs = lap(times, "auth", t);
    if o.timing { eprintln!("[auth, sharded] {:.2}s, attempt {} {:?}", secs, apf.attempt, tm); }
    apf
}

/// Load `raw` onto every replica in background threads; join with `finish`.
fn load_replicas<'s>(sc: &'s std::thread::Scope<'s, '_>, reps: Vec<AuthReplica>, raw: std::sync::Arc<Vec<Vec<i8>>>) -> Vec<std::thread::ScopedJoinHandle<'s, AuthReplica>> {
    reps.into_iter().map(|mut r| { let raw = raw.clone(); sc.spawn(move || { r.load(&raw); r }) }).collect()
}

#[allow(clippy::too_many_arguments)]
fn authenticate(o: &Opts, opening: u64, pc: &mut PcsGpu, reps: &mut [AuthReplica], fast: &link::LinkOutput, t1: &MTranscript, times: &mut Times) -> AuthProof {
    let to_dev = |cols: &[Vec<i64>]| DeviceBuffer::from_slice(&cols.iter().flatten().map(|&v| v as i8).collect::<Vec<_>>()).unwrap();
    let t = Instant::now();
    let r_cols: Vec<Vec<Vec<i64>>> = fast.sources.iter().map(|s| s.r.clone()).collect();
    let dsrc: Vec<DevSource> = pc.dev_sources(&r_cols);
    let rsrc: Vec<Vec<DevSource>> = reps.iter_mut().map(|r| r.sources(&r_cols)).collect();
    let s_os: Vec<DeviceBuffer<i8>> = reps.iter().map(|r| { almost_goldilocks_cuda::set_device(r.dev).unwrap(); to_dev(&fast.s_o) }).collect();
    almost_goldilocks_cuda::set_device(0).unwrap();
    let s_o0 = to_dev(&fast.s_o);
    let mut devs = vec![AuthDev { dev: 0, keys: &pc.keys, sources: &dsrc, s_o: &s_o0 }];
    for (k, r) in reps.iter().enumerate() { devs.push(AuthDev { dev: r.dev, keys: &r.keys, sources: &rsrc[k], s_o: &s_os[k] }); }
    let mut tm = AuthTiming::default();
    let (apf, _) = auth_gpu::prove_multi(&o.profile, &devs, &o.coins, opening, t1, &fast.point, true, &mut tm).expect("authentication");
    let secs = lap(times, "auth", t);
    if o.timing { eprintln!("[auth] {:.2}s, attempt {} {:?}", secs, apf.attempt, tm); }
    apf
}

// ---------------------------------------------------------------- request

pub struct RequestOut {
    /// The serialized proof (`wire::encode_request`): salt, PIOP messages,
    /// request source commitments (carry bank last under deferral), link
    /// and authentication.
    pub bytes: Vec<u8>,
    pub piop_len: usize,
    /// Prover-side copy of the carry commitment for the finalization queue.
    pub c_carry: Option<Vec<Vec<u64>>>,
    pub carry_raw: Option<Vec<i8>>,
    pub deferred: Vec<Row>,
    pub digest: K8,
    pub times: Times,
    pub rows: usize,
    pub carries: usize,
}

/// Prove request `j` (its witness is the one `pp` was prepared with).
pub fn prove_request(pp: &mut Prepared, o: &Opts, j: u64, model: Option<&ModelReg>, pc: &mut PcsGpu, reps: &mut Replicas) -> RequestOut {
    let mut times = Times::new();
    let ell = o.profile.ell();
    let opening = req_opening(j);
    let salt = o.coins.begin_opening(opening);
    let t = Instant::now();
    pp.pd.bank = piop::sample_bank(&o.coins, opening, 1usize << o.cfg.slots_log);
    lap(&mut times, "bank", t);
    let defer = o.defer;
    let Prepared { g, pd, lay, req, carry, .. } = &*pp;
    let (raw_req, mut rep_vec, c_req, piop_out) = std::thread::scope(|sc| {
        let pdr = pd;
        sc.spawn(move || pdr.prefetch());
        let t = Instant::now();
        let mut raw = piop::fill_sources_in(g, pd, lay, ell, req.clone());
        fill_suffixes(&o.profile, &o.coins, &mut raw, &|e| req_owner(j, e));
        lap(&mut times, "fill", t);
        let raw = std::sync::Arc::new(raw);
        let stream = o.streams(raw.len() + defer as usize);
        let jobs = if o.auth && !stream { load_replicas(sc, std::mem::take(&mut reps.r), raw.clone()) } else { Vec::new() };
        let t = Instant::now();
        let c_req = if stream {
            pc.streamed = true;
            let c = commit_streamed(o, pc, &raw);
            // The source key cache is rebuilt for the authentication.
            if !extra_devs().is_empty() { pc.drop_w(); }
            c
        } else {
            pc.streamed = false;
            pc.load_sources(&raw);
            pc.commit_sources()
        };
        lap(&mut times, "commit", t);
        // PIOP.
        let tr = request_transcript(o, pp, j, &salt, model.map(|m| &m.c[..]), &c_req);
        let t = Instant::now();
        let mut ctx = Ctx::prover(tr, &pd.bank, lay, o.cfg.slots_log);
        if defer { ctx.defer = Some(Defer::new(o.carry_log)); }
        let mut walk = Walk::new(g, Some(pd));
        walk.run(&mut ctx);
        lap(&mut times, "piop", t);
        pd.cache.clear();
        if o.timing { for (k, v) in &ctx.stats { eprintln!("  {k:12} {:6} calls {:.3}s", v.0, v.1); } }
        println!("{}", ctx.census_line());
        let reps: Vec<AuthReplica> = if stream { std::mem::take(&mut reps.r) } else { jobs.into_iter().map(|h| h.join().expect("replica")).collect() };
        (raw, reps, c_req, ctx)
    });
    let mut ctx = piop_out;
    if std::env::var("MORPHEUS_UNCLAIMED").is_ok() {
        let mut seen = std::collections::HashSet::new();
        for r in ctx.rows.iter().chain(ctx.defer.iter().flat_map(|d| d.rows.iter())) { for t in &r.terms { seen.insert((t.region.source, t.region.base)); } }
        for t in &lay.tensors {
            if !t.pieces.iter().any(|pc| seen.contains(&(pc.region.source, pc.region.base))) {
                let what = match t.key { TKey::Edge(e) => format!("edge {e} {:?} {:?} producer {:?}", g.shapes[e], pp.g.dag.producers[e].map(|p| format!("{:?}", pp.g.dag.nodes[p].kind).chars().take(40).collect::<String>()), g.model[e]), k => format!("{k:?}") };
                eprintln!("[unclaimed] {what}");
            }
        }
    }
    let mut raw: Vec<Vec<i8>> = std::sync::Arc::try_unwrap(raw_req).unwrap_or_else(|a| (*a).clone());
    let mut c_src = c_req;
    let mut t1 = ctx.tr.clone();
    let (mut carry_raw, mut deferred, mut ncarry) = (None, Vec::new(), 0);
    if let Some(d) = ctx.defer.as_ref() {
        let t = Instant::now();
        let mut cr = piop::fill_carry(lay, &d.carries, ell, carry.start);
        link::fill_source_suffix_gpu(&o.profile, &o.coins, req_owner(j, req.len()), &mut cr);
        let cc = if pc.streamed { commit_streamed(o, pc, std::slice::from_ref(&cr)).pop().unwrap() } else { pc.push_source(&cr) };
        for x in &cc { t1.absorb(Item::SourceCommitment, x); }
        if !pc.streamed { for r in rep_vec.iter_mut() { r.push(&cr); } }
        almost_goldilocks_cuda::set_device(0).unwrap();
        c_src.push(cc);
        raw.push(cr.clone());
        carry_raw = Some(cr);
        deferred = d.rows.clone();
        ncarry = d.used() as usize;
        lap(&mut times, "carry", t);
    }
    // Π_link over the request sources (+ carry bank).
    let base = req.start;
    let span = req.start..if defer { carry.end } else { req.end };
    let st = Statement { n: raw.len(), bits: lay.bit_blocks_in(&span), rows: remap(&ctx.rows, &|s| s - base) };
    let t = Instant::now();
    let fast = match std::env::var("LINK_IMPL").ok().as_deref() {
        Some("ref") => link::prove(&o.profile, &o.seed, &o.coins, opening, &mut t1, &st, &raw),
        Some("cpu") => link_fast::prove(&o.profile, &o.seed, &o.coins, opening, &mut t1, &st, &raw, o.b0),
        Some("v1") => link_fast::prove_with(&o.profile, &o.seed, &o.coins, opening, &mut t1, &st, &raw, o.b0, &link_gpu::gpu_backend, pc, Some(&c_src), false),
        _ => link_fast::prove_with(&o.profile, &o.seed, &o.coins, opening, &mut t1, &st, &raw, o.b0, &link_gpu::gpu_backend, pc, Some(&c_src), true),
    };
    lap(&mut times, "link", t);
    let auth = if !o.auth { None } else if pc.streamed { Some(authenticate_streamed(o, opening, pc, &fast, &t1, &mut times)) } else { Some(authenticate(o, opening, pc, &mut rep_vec, &fast, &t1, &mut times)) };
    reps.r = rep_vec;
    pc.streamed = false;
    let dg = digest(&t1);
    pc.unload();
    // The proof is complete once serialized; this is part of proving.
    let t = Instant::now();
    let c_carry = if defer { c_src.last().cloned() } else { None };
    let piop_len = ctx.proof.len();
    let pf = wire::RequestProof { salt, piop: std::mem::take(&mut ctx.proof), c_src, link: fast.proof, auth };
    let bytes = wire::encode_request(&o.profile, &pf);
    lap(&mut times, "serialize", t);
    RequestOut { bytes, piop_len, c_carry, carry_raw, deferred, digest: dg, times, rows: ctx.rows.len(), carries: ncarry }
}

/// Verifier of request `j`; returns its deferred rows and digest.
pub fn verify_request(pp: &Prepared, o: &Opts, j: u64, model: Option<&[Vec<Vec<u64>>]>, out: &wire::RequestProof, keys: &auth_gpu::Keys, times: &mut Times) -> Option<(Vec<Row>, K8)> {
    let defer = o.defer;
    let nreq = out.c_src.len() - defer as usize;
    // The salt is disclosed with the proof; the verifier never sees the
    // prover's private tape.
    let tr = request_transcript(o, pp, j, &out.salt, model, &out.c_src[..nreq]);
    let t = Instant::now();
    let mut vctx = Ctx::verifier(tr, out.piop.clone(), &pp.lay, o.cfg.slots_log);
    if defer { vctx.defer = Some(Defer::new(o.carry_log)); }
    let mut vw = Walk::new(&pp.g, None);
    let ran = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| vw.run(&mut vctx)));
    lap(times, "verify piop", t);
    if ran.is_err() || !vctx.ok { eprintln!("request {j}: PIOP rejected ({:?})", vctx.fail); return None; }
    if !vctx.consumed() { eprintln!("request {j}: PIOP proof has unread messages"); return None; }
    let mut tv = vctx.tr.clone();
    if defer { for x in &out.c_src[nreq] { tv.absorb(Item::SourceCommitment, x); } }
    let base = pp.req.start;
    let span = pp.req.start..if defer { pp.carry.end } else { pp.req.end };
    let st = Statement { n: out.c_src.len(), bits: pp.lay.bit_blocks_in(&span), rows: remap(&vctx.rows, &|s| s - base) };
    let t = Instant::now();
    let lv = link::verify(&o.profile, &mut tv, &st, &out.c_src, &out.link);
    lap(times, "verify link", t);
    let (pt, targets, target_o) = match lv { Some(x) => x, None => { eprintln!("request {j}: link rejected"); return None; } };
    if let Some(apf) = &out.auth {
        let t = Instant::now();
        let ok = auth_gpu::verify(&o.profile, keys, &tv, &pt, &targets, &target_o, apf);
        lap(times, "verify auth", t);
        if !ok { eprintln!("request {j}: authentication rejected"); return None; }
    }
    let deferred = vctx.defer.map(|d| d.rows).unwrap_or_default();
    Some((deferred, digest(&tv)))
}

// ---------------------------------------------------------------- finalization

#[derive(Clone)]
pub struct QueueEntry<'q> { pub j: u64, pub deferred: &'q [Row], pub c_carry: &'q [Vec<u64>], pub digest: K8 }

/// Finalization output: the serialized proof (`wire::encode_final`).
pub struct FinalOut { pub bytes: Vec<u8>, pub times: Times, pub rows: usize }

fn final_statement(pp: &Prepared, q: &[QueueEntry]) -> Statement {
    let nm = pp.model.len();
    let mut rows = Vec::new();
    for (k, e) in q.iter().enumerate() {
        let model = pp.model.clone();
        let carry = pp.carry.clone();
        rows.extend(remap(e.deferred, &|s| if model.contains(&s) { s - model.start } else { assert!(carry.contains(&s)); nm + k }));
    }
    let mut bits = pp.lay.bit_blocks_in(&pp.model);
    let cb = pp.lay.bit_blocks_in(&pp.carry);
    for _ in q { bits.extend(cb.iter().cloned()); }
    Statement { n: nm + q.len(), bits, rows }
}

fn final_transcript(o: &Opts, lay: &Layout, b: u64, salt: &[u8; 64], c_model: &[Vec<Vec<u64>>], q: &[QueueEntry]) -> MTranscript {
    let mut tr = MTranscript::new(o.profile.name, salt);
    absorb_layout(&mut tr, lay);
    absorb_model(&mut tr, c_model);
    tr.absorb(Item::Statement, &[q.len() as u64, b]);
    for e in q {
        tr.absorb(Item::Statement, &[e.j]);
        tr.absorb_k(Item::Other, &[e.digest]);
        for x in e.c_carry { tr.absorb(Item::SourceCommitment, x); }
    }
    tr
}

/// Finalize batch `b`: open the model sources and the queued carry banks
/// against every deferred row.
pub fn finalize(pp: &Prepared, o: &Opts, b: u64, model: &mut ModelReg, q: &[QueueEntry], carries: Vec<Vec<i8>>, pc: &mut PcsGpu, reps: &mut Replicas) -> FinalOut {
    let mut times = Times::new();
    let opening = fin_opening(b);
    let t = Instant::now();
    let st = final_statement(pp, q);
    let nm = model.raw.len();
    model.raw.extend(carries);
    let raw = std::mem::take(&mut model.raw);
    let raw = std::sync::Arc::new(raw);
    let mut c_src = model.c.clone();
    for e in q { c_src.push(e.c_carry.to_vec()); }
    let salt = o.coins.begin_opening(fin_opening(b));
    let mut tr = final_transcript(o, &pp.lay, b, &salt, &model.c, q);
    let stream = o.streams(raw.len());
    pc.streamed = stream;
    let (mut rep_vec, raw) = std::thread::scope(|sc| {
        if stream { return (std::mem::take(&mut reps.r), raw); }
        let jobs = if o.auth { load_replicas(sc, std::mem::take(&mut reps.r), raw.clone()) } else { Vec::new() };
        pc.load_sources(&raw);
        let v: Vec<AuthReplica> = jobs.into_iter().map(|h| h.join().expect("replica")).collect();
        (v, raw)
    });
    lap(&mut times, "load", t);
    let raw: Vec<Vec<i8>> = std::sync::Arc::try_unwrap(raw).unwrap_or_else(|a| (*a).clone());
    let t = Instant::now();
    let fast = link_fast::prove_with(&o.profile, &o.seed, &o.coins, opening, &mut tr, &st, &raw, o.b0, &link_gpu::gpu_backend, pc, Some(&c_src), true);
    lap(&mut times, "link", t);
    let auth = if !o.auth { None } else if pc.streamed { Some(authenticate_streamed(o, opening, pc, &fast, &tr, &mut times)) } else { Some(authenticate(o, opening, pc, &mut rep_vec, &fast, &tr, &mut times)) };
    reps.r = rep_vec;
    pc.streamed = false;
    pc.unload();
    let mut raw = raw;
    raw.truncate(nm);
    model.raw = raw;
    let t = Instant::now();
    let bytes = wire::encode_final(&o.profile, &wire::FinalProof { salt, link: fast.proof, auth });
    lap(&mut times, "serialize", t);
    FinalOut { bytes, times, rows: st.rows.len() }
}

pub fn verify_final(pp: &Prepared, o: &Opts, b: u64, c_model: &[Vec<Vec<u64>>], q: &[QueueEntry], out: &wire::FinalProof, keys: &auth_gpu::Keys, times: &mut Times) -> bool {
    let st = final_statement(pp, q);
    let mut c_src = c_model.to_vec();
    for e in q { c_src.push(e.c_carry.to_vec()); }
    let mut tv = final_transcript(o, &pp.lay, b, &out.salt, c_model, q);
    let t = Instant::now();
    let lv = link::verify(&o.profile, &mut tv, &st, &c_src, &out.link);
    lap(times, "verify link", t);
    let (pt, targets, target_o) = match lv { Some(x) => x, None => { eprintln!("finalization: link rejected"); return false; } };
    if let Some(apf) = &out.auth {
        let t = Instant::now();
        let ok = auth_gpu::verify(&o.profile, keys, &tv, &pt, &targets, &target_o, apf);
        lap(times, "verify auth", t);
        return ok;
    }
    true
}


#[allow(dead_code)]
fn _types(_: &Point, _: &Tuple, _: &AuxTuple) {}

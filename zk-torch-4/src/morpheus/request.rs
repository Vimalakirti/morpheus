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
pub fn req_opening(j: u64) -> u64 { assert!(j < (1u64 << 32) - 1, "request identifier out of range"); 1 + j }
pub fn fin_opening(b: u64) -> u64 { assert!(b < 1u64 << 32, "batch identifier out of range"); (1u64 << 32) + b }
/// Disjoint hiding-suffix owners: model [2^63,2^63+2^24), registered
/// inputs [2^62,2^62+2^20), requests j<2^32-1 and e<2^24 below2^56.
fn model_owner(e: usize) -> u64 { assert!(e < 1 << 24, "model source owner out of range"); (1u64 << 63) | e as u64 }
fn req_owner(j: u64, e: usize) -> u64 {
    assert!(e < 1 << 24 && j < (1 << 32) - 1, "coin owner out of range");
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
    /// Registered hidden leaf inputs; excludes public literals/inputs and masks.
    pub input: Range<usize>,
    pub req: Range<usize>,
    pub carry: Range<usize>,
    pub specs: Vec<(TKey, usize, usize, bool)>,
    pub ar: Option<super::ar_statement::ArStatement>,
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
    for &e in public {
        let values = piop::witness_vals(&wit[e][0]);
        if let Some(fixed) = g.public.get(&e) { assert_eq!(fixed, &values, "conflicting public input binding"); }
        else { g.public.insert(e, values); }
    }
    let private_inputs: Vec<usize> = (0..dag.num_edges()).filter(|&e|
        dag.producers[e].is_none() && g.edge_meta[e].3 == crate::dag::Role::Input && !g.public.contains_key(&e)).collect();
    g.prune_unclaimed();
    // Registration binds even inputs consumed only by currently-unconstrained
    // advice. This does not assert that the advice relation is correct.
    for &e in &private_inputs { g.committed[e] = true; }
    // A registered model identifies all declared parameters, including those
    // consumed only by advice. Keeping them private does not repair that advice.
    for e in 0..dag.num_edges() { if g.model[e] { g.committed[e] = true; } }
    let pd = ProverData::new(&g, wit, Vec::new());
    let specs = piop::tensor_specs(&g, &pd);
    // Never co-pack immutable inputs/model with proof-local masks/selectors.
    let (mut m, mut x, mut r) = (Vec::new(), Vec::new(), Vec::new());
    for &s in &specs { match s.0 {
        TKey::Edge(e) if g.model[e] => m.push(s),
        TKey::Edge(e) if private_inputs.contains(&e) => x.push(s),
        _ => r.push(s),
    } }
    let (lay, model, input, req, carry) = if o.defer {
        let (lay, rg) = Layout::pack_groups(&o.profile, &[m, x, r, vec![(TKey::Carry, o.carry_log, 9, true)]], o.b0);
        (lay, rg[0].clone(), rg[1].clone(), rg[2].clone(), rg[3].clone())
    } else {
        let (lay, rg) = Layout::pack_groups(&o.profile, &[m, x, r], o.b0);
        let end = lay.n_sources;
        (lay, rg[0].clone(), rg[1].clone(), rg[2].clone(), end..end)
    };
    if o.timing { eprintln!("[prep] graph + prover data + layout {:.2}s: model {:?}, input {:?}, fresh request {:?}, carry {:?}", t.elapsed().as_secs_f64(), model, input, req, carry); }
    Prepared { g, pd, lay, model, input, req, carry, specs, ar: None }
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
    raw: Vec<Vec<i8>>,
    c: Vec<Vec<Vec<u64>>>,
    pub secs: f64,
    /// Canonical immutable program/type identity; contains no private weights.
    statement: Vec<u64>,
    layout: Vec<u64>,
}

/// Public registration descriptor from an independently trusted registry.
/// This contains no model payload or hiding suffix. Do not source it from proof bytes.
#[derive(Clone, Copy)]
pub struct RegisteredModel<'a> {
    pub commitments: &'a [Vec<Vec<u64>>],
    pub statement: &'a [u64],
    pub layout: &'a [u64],
}
impl ModelReg {
    /// Number of immutable registered sources (without exposing openings).
    pub fn source_count(&self) -> usize { self.raw.len() }
    pub fn public(&self) -> RegisteredModel<'_> {
        RegisteredModel { commitments: &self.c, statement: &self.statement, layout: &self.layout }
    }
}
fn registered_descriptor_matches(model: RegisteredModel<'_>, statement: &[u64], layout: &[u64]) -> bool {
    model.statement == statement && model.layout == layout
}
fn registration_matches(pp: &Prepared, o: &Opts, model: RegisteredModel<'_>) -> bool {
    let Ok(statement) = super::statement::model_words(&pp.g, &o.profile, &o.seed, o.b0, o.defer, o.carry_log) else { return false; };
    registered_descriptor_matches(model, &statement, &model_layout_words(pp))
        && model.commitments.len() == pp.model.len()
        && model.commitments.iter().all(|c| wire::commitment_ok(&o.profile, c))
}

/// Prover-owned hidden input openings. Reuse this object to prove the SAME
/// registered input again; a different input requires a fresh registration.
pub struct InputReg {
    raw: Vec<Vec<i8>>,
    c: Vec<Vec<Vec<u64>>>,
    statement: Vec<u64>,
    layout: Vec<u64>,
    pub secs: f64,
}
/// Independently authenticated expected hidden-input identity. No private input
/// values or hiding suffixes are exported. Never obtain this view from the proof.
#[derive(Clone, Copy)]
pub struct RegisteredInput<'a> {
    pub commitments: &'a [Vec<Vec<u64>>],
    pub statement: &'a [u64],
    pub layout: &'a [u64],
}
impl InputReg {
    /// Number of immutable registered sources (without exposing openings).
    pub fn source_count(&self) -> usize { self.raw.len() }
    pub fn public(&self) -> RegisteredInput<'_> {
        RegisteredInput { commitments: &self.c, statement: &self.statement, layout: &self.layout }
    }
}
fn input_descriptor_matches(input: RegisteredInput<'_>, statement: &[u64], layout: &[u64]) -> bool {
    input.statement == statement && input.layout == layout
}
fn input_prefix_matches(expected: &[Vec<Vec<u64>>], model_prefix: usize, sources: &[Vec<Vec<u64>>]) -> bool {
    let Some(end) = model_prefix.checked_add(expected.len()) else { return false; };
    sources.get(model_prefix..end) == Some(expected)
}
fn input_registration_matches(pp: &Prepared, o: &Opts, input: RegisteredInput<'_>) -> bool {
    let Ok(statement) = super::statement::model_words(&pp.g, &o.profile, &o.seed, o.b0, o.defer, o.carry_log) else { return false; };
    input_descriptor_matches(input, &statement, &group_layout_words(pp, &pp.input))
        && input.commitments.len() == pp.input.len()
        && input.commitments.iter().all(|c| wire::commitment_ok(&o.profile, c))
}
/// Register every nonpublic leaf Role::Input, without registering fresh graph
/// masks. Allocated suffix-owner addresses are irreversibly consumed on failure.
pub fn register_inputs(pp: &Prepared, o: &Opts, pc: &mut PcsGpu) -> InputReg {
    let r = register_inputs_with(pp, o, &|raw, owner| fill_suffixes(&o.profile, &o.coins, raw, owner), &mut |raw| if raw.is_empty() { Vec::new() } else { commit_streamed(o, pc, raw) });
    pc.ensure_w(); r
}
pub fn register_inputs_cpu(pp: &Prepared, o: &Opts) -> InputReg {
    register_inputs_with(pp, o, &|raw, owner| fill_suffixes_cpu(o, raw, owner), &mut |raw| commit_cpu(o, raw))
}
fn register_inputs_with(pp: &Prepared, o: &Opts, fill: &dyn Fn(&mut [Vec<i8>], &dyn Fn(usize) -> u64), commit: &mut dyn FnMut(&[Vec<i8>]) -> Vec<Vec<Vec<u64>>>) -> InputReg {
    let t = Instant::now();
    let statement = super::statement::model_words(&pp.g, &o.profile, &o.seed, o.b0, o.defer, o.carry_log)
        .expect("invalid input registration program");
    let owners = o.coins.allocate_input_registration(pp.input.len()).expect("input registration source budget exhausted");
    let mut raw = piop::fill_sources_in(&pp.g, &pp.pd, &pp.lay, o.profile.ell(), pp.input.clone());
    assert_eq!(raw.len() as u64, owners.end - owners.start, "input suffix owner allocation mismatch");
    fill(&mut raw, &|e| owners.start + e as u64);
    let c = commit(&raw);
    InputReg { raw, c, statement, layout: group_layout_words(pp, &pp.input), secs: t.elapsed().as_secs_f64() }
}

/// One-time commitment to the model sources, reused verbatim in both modes.
/// The verifier must obtain the expected commitments from its trusted registry,
/// not from the proof or an untrusted prover-supplied ModelReg.
pub fn register(pp: &Prepared, o: &Opts, pc: &mut PcsGpu, reps: &Replicas) -> ModelReg {
    let _ = reps;
    let r = register_with(pp, o, &|raw, owner| fill_suffixes(&o.profile, &o.coins, raw, owner), &mut |raw| if raw.is_empty() { Vec::new() } else { commit_streamed(o, pc, raw) });
    pc.ensure_w(); r
}
pub fn register_cpu(pp: &Prepared, o: &Opts) -> ModelReg {
    register_with(pp, o, &|raw, owner| fill_suffixes_cpu(o, raw, owner), &mut |raw| commit_cpu(o, raw))
}
fn register_with(pp: &Prepared, o: &Opts, fill: &dyn Fn(&mut [Vec<i8>], &dyn Fn(usize) -> u64), commit: &mut dyn FnMut(&[Vec<i8>]) -> Vec<Vec<Vec<u64>>>) -> ModelReg {
    o.coins.claim_model_registration().expect("model registration namespace already consumed");
    let t = Instant::now();
    let mut raw = piop::fill_sources_in(&pp.g, &pp.pd, &pp.lay, o.profile.ell(), pp.model.clone());
    fill(&mut raw, &model_owner);
    let c = commit(&raw);
    let statement = super::statement::model_words(&pp.g, &o.profile, &o.seed, o.b0, o.defer, o.carry_log)
        .expect("invalid registered program");
    ModelReg { raw, c, secs: t.elapsed().as_secs_f64(), statement, layout: model_layout_words(pp) }
}

/// Stable registered model layout, excluding request-value-derived widths.
fn model_layout_words(pp: &Prepared) -> Vec<u64> { group_layout_words(pp, &pp.model) }
fn group_layout_words(pp: &Prepared, group: &Range<usize>) -> Vec<u64> { source_layout_words(&pp.lay, group) }
fn source_layout_words(lay: &Layout, group: &Range<usize>) -> Vec<u64> {
    let tensors: Vec<_> = lay.tensors.iter().filter(|t| !t.pieces.is_empty() && t.pieces.iter().all(|pc| group.contains(&pc.region.source))).collect();
    let mut w = vec![group.start as u64, group.end as u64, lay.ell as u64, lay.min_lg as u64, tensors.len() as u64];
    for t in tensors {
        let TKey::Edge(e) = t.key else { unreachable!() };
        w.extend([e as u64, t.n as u64, t.p as u64, t.bits as u64, t.pieces.len() as u64]);
        for pc in &t.pieces {
            let r = &pc.region;
            w.extend([r.source as u64, r.base as u64, r.m as u64, r.p as u64, pc.plane0 as u64, pc.hb as u64, pc.hbits as u64]);
        }
    }
    w
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

fn request_transcript(o: &Opts, pp: &Prepared, j: u64, salt: &[u8; 64], c_model: &[Vec<Vec<u64>>], c_input: &[Vec<Vec<u64>>], c_req: &[Vec<Vec<u64>>]) -> Option<MTranscript> {
    let words = super::statement::words(&pp.g, &o.profile, &o.seed, o.b0, o.defer, o.carry_log).ok()?;
    let mut tr = MTranscript::new(o.profile.name, salt);
    tr.absorb(Item::Statement, &[wire::PROTOCOL_VERSION, 0, o.defer as u64]);
    tr.absorb(Item::Statement, &words);
    absorb_layout(&mut tr, &pp.lay);
    if let Some(ar) = &pp.ar {
        assert!(ar.validate(&pp.g), "invalid public AR statement");
        tr.absorb(Item::Statement, &ar.words());
        // Bind exact canonical public selector arrays before graph challenges.
        for e in std::iter::once(&ar.embedding).chain(ar.selectors.iter()).chain(ar.rank_constants.iter().flatten()) {
            tr.absorb(Item::Statement, &[*e as u64]);
            tr.absorb(Item::Statement, &pp.g.public[e]);
        }
    }
    tr.absorb(Item::Statement, &[1]); absorb_model(&mut tr, c_model);
    tr.absorb(Item::Statement, &[2]); absorb_model(&mut tr, c_input);
    tr.absorb(Item::Statement, &[3, c_req.len() as u64, j]);
    for c in c_req { for x in c { tr.absorb(Item::SourceCommitment, x); } }
    Some(tr)
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
    pub literal_prefix: Vec<u8>,
    pub times: Times,
    pub rows: usize,
    pub carries: usize,
}

/// Prove request `j` (its witness is the one `pp` was prepared with).
pub fn prove_request(pp: &mut Prepared, o: &Opts, j: u64, model: Option<&ModelReg>, input: &InputReg, pc: &mut PcsGpu, reps: &mut Replicas) -> RequestOut {
    let model = model.expect("a registered model is required in both modes");
    assert_eq!(model.statement, super::statement::model_words(&pp.g, &o.profile, &o.seed, o.b0, o.defer, o.carry_log)
        .expect("invalid request program"), "registered program identity changed");
    assert_eq!(model.layout, model_layout_words(pp), "registered model layout changed");
    assert_eq!(model.raw.len(), pp.model.len(), "registered model source count");
    assert_eq!(model.c.len(), pp.model.len(), "registered model commitment count");
    assert!(input_registration_matches(pp, o, input.public()), "registered input descriptor mismatch");
    assert_eq!(input.raw.len(), pp.input.len(), "registered input source count");
    let mut times = Times::new();
    let ell = o.profile.ell();
    let opening = req_opening(j);
    let salt = o.coins.begin_opening(opening).expect("opening identifier already consumed or lifetime exhausted");
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
        raw = input.raw.iter().cloned().chain(raw).collect();
        if !defer {
            // Preserve the registered model's payload AND hiding suffix. Only
            // request sources above receive fresh masks/suffixes.
            raw = model.raw.iter().cloned().chain(raw).collect();
        }
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
        let model_prefix = if defer { 0 } else { pp.model.len() };
        if !defer { assert_eq!(&c_req[..model_prefix], &model.c[..], "registered model commitment changed"); }
        let input_end = model_prefix + pp.input.len();
        assert_eq!(&c_req[model_prefix..input_end], &input.c[..], "registered input commitment changed");
        let tr = request_transcript(o, pp, j, &salt, &model.c, &input.c, &c_req[input_end..]).expect("invalid request statement");
        let t = Instant::now();
        let mut ctx = Ctx::prover(tr, &pd.bank, lay, o.cfg.slots_log);
        if defer { ctx.defer = Some(Defer::new(o.carry_log)); }
        let mut walk = Walk::new(g, Some(pd));
        walk.run(&mut ctx);
        lap(&mut times, "piop", t);
        pd.clear_cache();
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
    let base = if defer { pp.input.start } else { pp.model.start };
    let span = base..if defer { carry.end } else { req.end };
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
    RequestOut { bytes, piop_len, c_carry, carry_raw, deferred, digest: dg, literal_prefix: t1.literal_prefix().to_vec(), times, rows: ctx.rows.len(), carries: ncarry }
}

/// Registry identity check, independent of prover data and GPU state.
fn registered_model_matches(n_model: usize, defer: bool, expected: Option<&[Vec<Vec<u64>>]>, sources: &[Vec<Vec<u64>>]) -> bool {
    let Some(expected) = expected else { return false; };
    expected.len() == n_model && (defer || sources.get(..n_model) == Some(expected))
}

/// Verifier of request `j`; returns deferred rows, diagnostic digest, and literal public prefix.
pub fn verify_request(pp: &Prepared, o: &Opts, j: u64, model: Option<RegisteredModel<'_>>, input: RegisteredInput<'_>, out: &wire::RequestProof, keys: &auth_gpu::Keys, times: &mut Times) -> Option<(Vec<Row>, K8, Vec<u8>)> {
    verify_request_with(pp, o, j, model, input, out, times, o.profile.source_cols(), &|tr, pt, targets, target_o, apf| auth_gpu::verify(&o.profile, keys, tr, pt, targets, target_o, apf))
}

/// Host authentication backend; all statement, registry, graph and link gates
/// are shared with the GPU verifier. No witness or private coins are read.
pub fn verify_request_cpu(pp: &Prepared, o: &Opts, j: u64, model: Option<RegisteredModel<'_>>, input: RegisteredInput<'_>, out: &wire::RequestProof, times: &mut Times) -> Option<(Vec<Row>, K8, Vec<u8>)> {
    verify_request_with(pp, o, j, model, input, out, times, o.profile.source_cols(), &|tr, pt, targets, target_o, apf| super::auth::verify(&o.profile, &o.seed, tr, pt, targets, target_o, apf))
}

fn verify_request_with(pp: &Prepared, o: &Opts, j: u64, model: Option<RegisteredModel<'_>>, input: RegisteredInput<'_>, out: &wire::RequestProof, times: &mut Times,
    w_cols: usize, authenticate: &dyn Fn(&MTranscript, &Point, &[Tuple], &AuxTuple, &AuthProof) -> bool,
) -> Option<(Vec<Row>, K8, Vec<u8>)> {
    if j >= (1u64 << 32) - 1 || std::env::var_os("LINK_ROWS").is_some() { return None; }
    if pp.ar.as_ref().map(|ar| !ar.validate(&pp.g)).unwrap_or(false) { return None; }
    // Diagnostic NO_AUTH proofs are never accepted by this API.
    let apf = out.auth.as_ref()?;
    let defer = o.defer;
    let registered = model?;
    if !registration_matches(pp, o, registered) || !input_registration_matches(pp, o, input) { return None; }
    let expected_model = registered.commitments;
    let nreq = out.c_src.len().checked_sub(defer as usize)?;
    let model_prefix = if defer { 0 } else { pp.model.len() };
    let input_end = model_prefix.checked_add(pp.input.len())?;
    let expected_sources = pp.req.len().checked_add(input_end)?;
    if nreq != expected_sources || out.c_src.len() > o.profile.n_max
        || !registered_model_matches(pp.model.len(), defer, Some(expected_model), &out.c_src)
        || !input_prefix_matches(input.commitments, model_prefix, &out.c_src)
        || !expected_model.iter().all(|c| wire::commitment_ok(&o.profile, c))
        || !out.c_src.iter().all(|c| wire::commitment_ok(&o.profile, c))
        || !wire::link_shape_ok(&o.profile, out.c_src.len(), &out.link)
        || !wire::auth_shape_ok_cols(&o.profile, apf, w_cols)
        || !out.piop.iter().all(wire::canonical_k) { return None; }
    // The expected registry commitments, not a prover-selected model identity,
    // are bound before graph challenges. Empty-model graphs still supply a
    // RegisteredModel descriptor, with an empty commitments slice.
    let tr = request_transcript(o, pp, j, &out.salt, expected_model, input.commitments, &out.c_src[input_end..nreq])?;
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
    let base = if defer { pp.input.start } else { pp.model.start };
    let span = base..if defer { pp.carry.end } else { pp.req.end };
    let st = Statement { n: out.c_src.len(), bits: pp.lay.bit_blocks_in(&span), rows: remap(&vctx.rows, &|s| s - base) };
    let t = Instant::now();
    let lv = link::verify(&o.profile, &mut tv, &st, &out.c_src, &out.link);
    lap(times, "verify link", t);
    let (pt, targets, target_o) = match lv { Some(x) => x, None => { eprintln!("request {j}: link rejected"); return None; } };
    if let Some(apf) = &out.auth {
        let t = Instant::now();
        let ok = authenticate(&tv, &pt, &targets, &target_o, apf);
        lap(times, "verify auth", t);
        if !ok { eprintln!("request {j}: authentication rejected"); return None; }
    }
    let deferred = vctx.defer.map(|d| d.rows).unwrap_or_default();
    Some((deferred, digest(&tv), tv.literal_prefix().to_vec()))
}

// ---------------------------------------------------------------- finalization

/// Verifier-owned state from successful `verify_request` for the SAME pinned
/// program/model/layout. Never deserialize an untrusted network queue into this
/// type and pass it directly to `verify_final`; neither prefix nor digest is self-authenticating.
#[derive(Clone)]
pub struct QueueEntry<'q> { pub j: u64, pub deferred: &'q [Row], pub c_carry: &'q [Vec<u64>], pub digest: K8, pub literal_prefix: &'q [u8] }

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

fn final_transcript(o: &Opts, pp: &Prepared, b: u64, salt: &[u8; 64], c_model: &[Vec<Vec<u64>>], q: &[QueueEntry]) -> Option<MTranscript> {
    let words = super::statement::model_words(&pp.g, &o.profile, &o.seed, o.b0, o.defer, o.carry_log).ok()?;
    let mut tr = MTranscript::new(o.profile.name, salt);
    tr.absorb(Item::Statement, &[wire::PROTOCOL_VERSION, 1, o.defer as u64]);
    // Request-specific public values and history are bound literally below.
    // The diagnostic digest is not a substitute for the oracle prefix.
    tr.absorb(Item::Statement, &words);
    absorb_layout(&mut tr, &pp.lay);
    absorb_model(&mut tr, c_model);
    tr.absorb(Item::Statement, &[q.len() as u64, b]);
    for e in q {
        tr.absorb(Item::Statement, &[e.j]);
        tr.absorb_bytes(Item::Other, e.literal_prefix);
        tr.absorb_k(Item::Other, &[e.digest]);
        for x in e.c_carry { tr.absorb(Item::SourceCommitment, x); }
    }
    Some(tr)
}

/// Finalize batch `b`: open the model sources and the queued carry banks
/// against every deferred row.
pub fn finalize(pp: &Prepared, o: &Opts, b: u64, model: &mut ModelReg, q: &[QueueEntry], carries: Vec<Vec<i8>>, pc: &mut PcsGpu, reps: &mut Replicas) -> FinalOut {
    assert!(o.defer && !q.is_empty(), "finalization needs a nonempty deferred queue");
    assert_eq!(model.layout, model_layout_words(pp), "registered model layout changed");
    assert_eq!(model.statement, super::statement::model_words(&pp.g, &o.profile, &o.seed, o.b0, o.defer, o.carry_log)
        .expect("invalid finalization program"), "registered program identity changed");
    let mut times = Times::new();
    let opening = fin_opening(b);
    // Reserve before taking/extending model.raw: duplicate/lifetime failures
    // must leave the reusable registered source intact.
    let salt = o.coins.begin_opening(opening).expect("opening identifier already consumed or lifetime exhausted");
    let t = Instant::now();
    let st = final_statement(pp, q);
    let nm = model.raw.len();
    model.raw.extend(carries);
    let raw = std::mem::take(&mut model.raw);
    let raw = std::sync::Arc::new(raw);
    let mut c_src = model.c.clone();
    for e in q { c_src.push(e.c_carry.to_vec()); }
    let mut tr = final_transcript(o, pp, b, &salt, &model.c, q).expect("invalid finalization statement");
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

/// `q` must contain only verifier-produced entries from accepted requests under
/// this same independently pinned model/program/layout; this function does not
/// re-run request verification. The expected model commitments are external.
pub fn verify_final(pp: &Prepared, o: &Opts, b: u64, model: RegisteredModel<'_>, q: &[QueueEntry], out: &wire::FinalProof, keys: &auth_gpu::Keys, times: &mut Times) -> bool {
    verify_final_with(pp, o, b, model, q, out, times, o.profile.source_cols(), &|tr, pt, targets, target_o, apf| auth_gpu::verify(&o.profile, keys, tr, pt, targets, target_o, apf))
}
/// Host finalization verifier sharing every registry, queue, transcript and link
/// check with the GPU backend. Queue entries must come from verified requests.
pub fn verify_final_cpu(pp: &Prepared, o: &Opts, b: u64, model: RegisteredModel<'_>, q: &[QueueEntry], out: &wire::FinalProof, times: &mut Times) -> bool {
    verify_final_with(pp, o, b, model, q, out, times, o.profile.source_cols(), &|tr, pt, targets, target_o, apf| super::auth::verify(&o.profile, &o.seed, tr, pt, targets, target_o, apf))
}
fn verify_final_with(pp: &Prepared, o: &Opts, b: u64, model: RegisteredModel<'_>, q: &[QueueEntry], out: &wire::FinalProof, times: &mut Times,
    w_cols: usize, authenticate: &dyn Fn(&MTranscript, &Point, &[Tuple], &AuxTuple, &AuthProof) -> bool,
) -> bool {
    if !registration_matches(pp, o, model) { return false; }
    let c_model = model.commitments;
    let Some(apf) = out.auth.as_ref() else { return false; };
    let Some(n) = c_model.len().checked_add(q.len()) else { return false; };
    if b >= 1u64 << 32 || q.iter().any(|e| e.j >= (1u64 << 32) - 1)
        || std::env::var_os("LINK_ROWS").is_some() || q.is_empty() || !o.defer || pp.carry.len() != 1 || c_model.len() != pp.model.len()
        || n > o.profile.n_max || !c_model.iter().all(|c| wire::commitment_ok(&o.profile, c))
        || !wire::link_shape_ok(&o.profile, n, &out.link)
        || !wire::auth_shape_ok_cols(&o.profile, apf, w_cols)
        || !q.iter().all(|e| wire::commitment_ok(&o.profile, e.c_carry)
            && wire::canonical_k(&e.digest)
            && e.deferred.iter().flat_map(|r| &r.terms).all(|t|
                pp.model.contains(&t.region.source) || pp.carry.contains(&t.region.source))) { return false; }
    let st = final_statement(pp, q);
    let mut c_src = c_model.to_vec();
    for e in q { c_src.push(e.c_carry.to_vec()); }
    let Some(mut tv) = final_transcript(o, pp, b, &out.salt, c_model, q) else { return false; };
    let t = Instant::now();
    let lv = link::verify(&o.profile, &mut tv, &st, &c_src, &out.link);
    lap(times, "verify link", t);
    let (pt, targets, target_o) = match lv { Some(x) => x, None => { eprintln!("finalization: link rejected"); return false; } };
    if let Some(apf) = &out.auth {
        let t = Instant::now();
        let ok = authenticate(&tv, &pt, &targets, &target_o, apf);
        lap(times, "verify auth", t);
        return ok;
    }
    true
}


#[allow(dead_code)]
fn _types(_: &Point, _: &Tuple, _: &AuxTuple) {}

#[cfg(test)]
mod registry_tests {
    #[test]
    fn registered_input_prefix_rejects_replacement_and_reordering_in_both_modes() {
        use super::input_prefix_matches;
        let model = vec![vec![1u64]];
        let a = vec![vec![vec![2u64]], vec![vec![3u64]]];
        let mut b = a.clone(); b[0][0][0] = 9;
        let fresh = vec![vec![4u64]];
        let mut mono = vec![model]; mono.extend(a.clone()); mono.push(fresh.clone());
        let mut deferred = a.clone(); deferred.push(fresh);
        assert!(input_prefix_matches(&a, 1, &mono));
        assert!(input_prefix_matches(&a, 0, &deferred));
        assert!(!input_prefix_matches(&b, 1, &mono));
        assert!(!input_prefix_matches(&b, 0, &deferred));
        let mut reversed = a.clone(); reversed.reverse();
        assert!(!input_prefix_matches(&reversed, 1, &mono));
        assert!(!input_prefix_matches(&a, 1, &mono[..2]));
        assert!(!input_prefix_matches(&a, usize::MAX, &mono));
        // Reusing a registered input is valid while fresh request commitments change.
        *mono.last_mut().unwrap() = vec![vec![8u64]];
        assert!(input_prefix_matches(&a, 1, &mono));
        assert!(input_prefix_matches(&[], 0, &[]));
    }

    #[test]
    fn private_input_group_never_contains_masks_selectors_or_public_sources() {
        use super::{source_layout_words, Layout, TKey, Profile, input_descriptor_matches, RegisteredInput};
        let groups = vec![vec![(TKey::Edge(0),2,2,false)],
            vec![(TKey::Edge(1),2,4,false)],
            vec![(TKey::Edge(2),2,1,true),(TKey::Wg,2,9,true),(TKey::Sel{node:0,chunk:0},2,2,true)],
            vec![(TKey::Carry,2,9,true)]];
        let (lay, ranges) = Layout::pack_groups(&Profile::toy(), &groups, 9);
        for (i,g) in groups.iter().enumerate() { for (key,_,_,_) in g {
            assert!(lay.get(key).pieces.iter().all(|p| ranges[i].contains(&p.region.source)));
        }}
        assert_eq!(ranges[0].end, ranges[1].start);
        assert_eq!(ranges[1].end, ranges[2].start);
        assert_eq!(ranges[2].end, ranges[3].start);
        let input_layout = source_layout_words(&lay, &ranges[1]);
        let statement = vec![3u64,9];
        let input = RegisteredInput { commitments: &[], statement: &statement, layout: &input_layout };
        assert!(input_descriptor_matches(input, &statement, &input_layout));
        let mut changed = lay.clone(); changed.tensors[1].p += 1;
        assert!(!input_descriptor_matches(input, &statement, &source_layout_words(&changed, &ranges[1])));
        assert!(!input_descriptor_matches(input, &[3,8], &input_layout));
        assert!(!input_descriptor_matches(input, &statement, &source_layout_words(&lay, &ranges[0])));
    }

    use super::{registered_model_matches, registered_descriptor_matches, RegisteredModel};

    #[test]
    fn registered_program_and_each_layout_word_are_immutable() {
        let commitments = vec![vec![vec![2u64]]];
        let statement = vec![2u64, 1, 9, 4];
        // Representative serialized model range and piece fields.
        let layout = vec![0u64, 1, 12, 4, 1, 3, 8, 4, 0, 1, 0, 0, 8, 4, 0, 0, 0];
        let model = RegisteredModel { commitments: &commitments, statement: &statement, layout: &layout };
        assert!(registered_descriptor_matches(model, &statement, &layout));
        for i in 0..layout.len() {
            let mut changed = layout.clone(); changed[i] += 1;
            assert!(!registered_descriptor_matches(model, &statement, &changed));
        }
        for i in 0..statement.len() {
            let mut changed = statement.clone(); changed[i] += 1;
            assert!(!registered_descriptor_matches(model, &changed, &layout));
        }
        assert!(!registered_descriptor_matches(model, &statement, &layout[..layout.len()-1]));
    }


    #[test]
    fn monolithic_registry_rejects_replacement_missing_and_wrong_count() {
        let a = vec![vec![vec![2u64]]];
        let b = vec![vec![vec![3u64]]];
        let request = vec![vec![9u64]];
        let mut sources = a.clone(); sources.push(request);
        assert!(registered_model_matches(1, false, Some(&a), &sources));
        assert!(!registered_model_matches(1, false, Some(&b), &sources));
        assert!(!registered_model_matches(1, false, None, &sources));
        assert!(!registered_model_matches(2, false, Some(&a), &sources));
        assert!(!registered_model_matches(1, false, Some(&a), &[]));
        // An empty model is explicit, not a missing registration.
        assert!(registered_model_matches(0, false, Some(&[]), &sources[1..]));
        assert!(!registered_model_matches(0, false, None, &sources[1..]));
    }

    #[test]
    fn deferred_registry_identity_is_external_not_a_request_prefix() {
        let a = vec![vec![vec![2u64]]];
        let request = vec![vec![vec![9u64]]];
        assert!(registered_model_matches(1, true, Some(&a), &request));
        assert!(!registered_model_matches(1, true, None, &request));
        assert!(!registered_model_matches(2, true, Some(&a), &request));
    }
}

// CPU reference orchestration. Full authentication; no diagnostic bypass.
fn fill_suffixes_cpu(o: &Opts, raw: &mut [Vec<i8>], owner: &dyn Fn(usize) -> u64) {
    for (e, r) in raw.iter_mut().enumerate() { link::fill_source_suffix(&o.profile, &o.coins, owner(e), r); }
}
fn commit_cpu(o: &Opts, raw: &[Vec<i8>]) -> Vec<Vec<Vec<u64>>> {
    raw.iter().map(|r| {
        // Same zero-skipping commitment optimization as commit::commit, without
        // materializing a full-domain i64 witness. This is NOT noise support.
        let indices: Vec<usize> = (0..o.profile.source_cols()).filter(|&c|
            c >= o.profile.source_payload_cols || r[c * super::ring::N..(c+1)*super::ring::N].iter().any(|&v|v!=0)).collect();
        super::auth_compact::lin_indexed(&o.profile,&o.seed,super::keys::KeyKind::Source,o.profile.source_payload_cols,&indices,&|c|super::auth_compact::raw_column(r,c))
    }).collect()
}
/// Real non-deferred host proof. The eager reference backend is intended for
/// algebra tests, not selected-profile memory/performance claims.
pub fn prove_request_cpu(pp: &mut Prepared, o: &Opts, j: u64, model: &ModelReg, input: &InputReg) -> wire::RequestProof {
    assert!(!o.defer, "use prove_request_cpu_deferred to retain carry witness");
    prove_request_cpu_inner(pp, o, j, model, input, false, None).proof
}
/// Prover output for deferred requests. Only the proof is public; carry_raw is
/// retained privately until finalization. Verifier queue fields are NOT supplied
/// by this object; derive them by verifying the serialized request.
pub struct CpuDeferredRequest { pub proof: wire::RequestProof, pub carry_raw: Option<Vec<i8>> }
pub fn prove_request_cpu_deferred(pp: &mut Prepared, o: &Opts, j: u64, model: &ModelReg, input: &InputReg) -> CpuDeferredRequest {
    assert!(o.defer, "deferred CPU entrypoint requires deferral");
    assert_eq!(pp.carry.len(), 1, "exactly one carry bank per request");
    prove_request_cpu_inner(pp, o, j, model, input, false, None)
}
fn prove_request_cpu_inner(pp: &mut Prepared, o: &Opts, j: u64, model: &ModelReg, input: &InputReg, compact: bool, resource_gate: Option<&dyn Fn(&Profile,&Statement,usize)>) -> CpuDeferredRequest {
    assert!(o.auth, "CPU full proof requires authentication");
    assert!(std::env::var_os("LINK_ROWS").is_none(), "partial link rows are not a full proof");
    assert!(registration_matches(pp, o, model.public()), "registered model mismatch");
    assert!(input_registration_matches(pp, o, input.public()), "registered input mismatch");
    assert_eq!(model.raw.len(), pp.model.len()); assert_eq!(input.raw.len(), pp.input.len());
    let opening = req_opening(j);
    let salt = o.coins.begin_opening(opening).expect("opening already consumed");
    pp.pd.bank = piop::sample_bank(&o.coins, opening, 1usize << o.cfg.slots_log);
    let mut raw = piop::fill_sources_in(&pp.g, &pp.pd, &pp.lay, o.profile.ell(), pp.req.clone());
    fill_suffixes_cpu(o, &mut raw, &|e| req_owner(j, e));
    raw = input.raw.iter().cloned().chain(raw).collect();
    if !o.defer { raw = model.raw.iter().cloned().chain(raw).collect(); }
    let mut c_src = commit_cpu(o, &raw);
    let model_prefix = if o.defer { 0 } else { pp.model.len() };
    let input_end = model_prefix + pp.input.len();
    if !o.defer { assert_eq!(&c_src[..model_prefix], &model.c); }
    assert_eq!(&c_src[model_prefix..input_end], &input.c);
    let tr = request_transcript(o, pp, j, &salt, &model.c, &input.c, &c_src[input_end..]).expect("invalid statement");
    let mut ctx = Ctx::prover(tr, &pp.pd.bank, &pp.lay, o.cfg.slots_log);
    if o.defer { ctx.defer = Some(Defer::new(o.carry_log)); }
    Walk::new(&pp.g, Some(&pp.pd)).run(&mut ctx);
    let mut tr = ctx.tr.clone();
    let carry_raw = if let Some(d) = ctx.defer.as_ref() {
        let mut cr = piop::fill_carry(&pp.lay, &d.carries, o.profile.ell(), pp.carry.start);
        link::fill_source_suffix(&o.profile, &o.coins, req_owner(j, pp.req.len()), &mut cr);
        let cc = commit_cpu(o, std::slice::from_ref(&cr)).pop().unwrap();
        for row in &cc { tr.absorb(Item::SourceCommitment, row); }
        c_src.push(cc);raw.push(cr.clone());Some(cr)
    } else { None };
    let span = if o.defer { pp.input.start..pp.carry.end } else { pp.model.start..pp.req.end };
    let st = Statement { n: raw.len(), bits: pp.lay.bit_blocks_in(&span), rows: remap(&ctx.rows, &|s| s - span.start) };
    if compact { super::auth_compact::check_raw_support(&o.profile,&request_support(pp,o).expect("public support"),&raw); }
    if let Some(gate)=resource_gate {gate(&o.profile,&st,o.b0);}
    let mut svc = link_fast::CpuServices;
    let mut link_verifier_prefix = tr.clone();
    let mut fast = link_fast::prove_with(&o.profile, &o.seed, &o.coins, opening, &mut tr, &st, &raw, o.b0, &link_fast::cpu_backend, &mut svc, Some(&c_src), false);
    assert!(link::verify(&o.profile, &mut link_verifier_prefix, &st, &c_src, &fast.proof).is_some(), "CPU link self-check before expensive authentication");
    eprintln!("CPU_FULLPROOF graph+link complete sources={} rows={}; starting genuine Gaussian auth", raw.len(), ctx.rows.len());
    let apf = if compact {
        let support = request_support(pp,o).expect("invalid public support layout");
        let links: Vec<_> = fast.sources.iter_mut().map(|s|std::mem::take(&mut s.r)).collect();
        super::auth_compact::prove(&o.profile,&o.seed,&o.coins,opening,&tr,&fast.point,&support,&raw,&links,&fast.s_o).expect("compact CPU authentication")
    } else {
        for (s,r) in fast.sources.iter_mut().zip(&raw) {s.w=link::raw_to_cols(r);}
        super::auth::prove(&o.profile, &o.seed, &o.coins, opening, &tr, &fast.point, &fast.sources, &fast.s_o).expect("CPU authentication")
    };
    eprintln!("CPU_FULLPROOF auth accepted attempt={}", apf.attempt);
    CpuDeferredRequest { proof: wire::RequestProof { salt, piop: ctx.proof, c_src, link: fast.proof, auth: Some(apf) }, carry_raw }
}

/// Exact eager CPU finalization. Model registration is borrowed, never consumed
/// or modified, including failed/duplicate-opening paths.
pub fn finalize_cpu(pp: &Prepared, o: &Opts, b: u64, model: &ModelReg, q: &[QueueEntry], carries: &[Vec<i8>]) -> wire::FinalProof {
    finalize_cpu_inner(pp,o,b,model,q,carries,false)
}
fn finalize_cpu_inner(pp: &Prepared, o: &Opts, b: u64, model: &ModelReg, q: &[QueueEntry], carries: &[Vec<i8>], compact:bool) -> wire::FinalProof {
    assert!(o.defer && o.auth && !q.is_empty(), "full deferred finalization requires authentication and a queue");
    assert!(std::env::var_os("LINK_ROWS").is_none(), "partial link rows are not a full proof");
    assert!(registration_matches(pp, o, model.public()), "registered model changed");
    assert_eq!(q.len(), carries.len(), "one private carry bank per verified request");
    let opening = fin_opening(b);
    let salt = o.coins.begin_opening(opening).expect("finalization opening already consumed");
    let raw: Vec<Vec<i8>> = model.raw.iter().cloned().chain(carries.iter().cloned()).collect();
    let mut c_src = model.c.clone();
    let actual_carry = commit_cpu(o, carries);
    for (entry, actual) in q.iter().zip(actual_carry) {
        assert_eq!(entry.c_carry, &actual, "private carry does not open the verified request commitment");
        c_src.push(actual);
    }
    let st = final_statement(pp, q);
    let mut tr = final_transcript(o, pp, b, &salt, &model.c, q).expect("invalid finalization statement");
    let mut svc = link_fast::CpuServices;
    let mut link_verifier_prefix = tr.clone();
    let mut fast = link_fast::prove_with(&o.profile, &o.seed, &o.coins, opening, &mut tr, &st, &raw, o.b0, &link_fast::cpu_backend, &mut svc, Some(&c_src), false);
    assert!(link::verify(&o.profile, &mut link_verifier_prefix, &st, &c_src, &fast.proof).is_some(), "CPU link self-check before expensive authentication");
    eprintln!("CPU_DEFERRED final link complete sources={} rows={}; starting genuine Gaussian auth", raw.len(), st.rows.len());
    let auth = if compact {
        let support=final_support(pp,o).expect("invalid final public support");
        let links:Vec<_>=fast.sources.iter_mut().map(|s|std::mem::take(&mut s.r)).collect();
        super::auth_compact::prove(&o.profile,&o.seed,&o.coins,opening,&tr,&fast.point,&support,&raw,&links,&fast.s_o).expect("compact CPU final authentication")
    }else{
        for(s,r)in fast.sources.iter_mut().zip(&raw){s.w=link::raw_to_cols(r);}
        super::auth::prove(&o.profile, &o.seed, &o.coins, opening, &tr, &fast.point, &fast.sources, &fast.s_o).expect("CPU final authentication")
    };
    eprintln!("CPU_DEFERRED final auth accepted attempt={}", auth.attempt);
    wire::FinalProof { salt, link: fast.proof, auth: Some(auth) }
}

/// Public registry-derived response support for this request, before any coins.
pub fn request_support(pp:&Prepared,o:&Opts)->Option<super::auth_compact::Support>{
    let span=if o.defer {pp.input.start..pp.carry.end}else{pp.model.start..pp.req.end};
    super::public_support::derive(&o.profile,&pp.lay,&[span])
}
pub fn prove_request_cpu_compact(pp:&mut Prepared,o:&Opts,j:u64,model:&ModelReg,input:&InputReg)->CpuDeferredRequest{
    request_support(pp,o).expect("invalid registered public layout");
    prove_request_cpu_inner(pp,o,j,model,input,true,None)
}
pub fn verify_request_cpu_compact(pp:&Prepared,o:&Opts,j:u64,model:Option<RegisteredModel<'_>>,input:RegisteredInput<'_>,out:&wire::RequestProof,times:&mut Times)->Option<(Vec<Row>,K8,Vec<u8>)>{
    let support=request_support(pp,o)?;
    verify_request_with(pp,o,j,model,input,out,times,support.len(),&|tr,pt,targets,target_o,apf|super::auth_compact::verify(&o.profile,&o.seed,tr,pt,&support,targets,target_o,apf))
}

pub fn final_support(pp:&Prepared,o:&Opts)->Option<super::auth_compact::Support>{super::public_support::derive(&o.profile,&pp.lay,&[pp.model.clone(),pp.carry.clone()])}
pub fn finalize_cpu_compact(pp:&Prepared,o:&Opts,b:u64,model:&ModelReg,q:&[QueueEntry],carries:&[Vec<i8>])->wire::FinalProof{final_support(pp,o).expect("invalid final layout");finalize_cpu_inner(pp,o,b,model,q,carries,true)}
pub fn verify_final_cpu_compact(pp:&Prepared,o:&Opts,b:u64,model:RegisteredModel<'_>,q:&[QueueEntry],out:&wire::FinalProof,times:&mut Times)->bool{
    let Some(support)=final_support(pp,o)else{return false};
    verify_final_with(pp,o,b,model,q,out,times,support.len(),&|tr,pt,targets,target_o,apf|super::auth_compact::verify(&o.profile,&o.seed,tr,pt,&support,targets,target_o,apf))
}

/// Local execution-budget gate, run on public PIOP/link structure before link
/// allocations. It cannot alter the statement, witness, transcript or verifier.
pub fn prove_request_cpu_compact_guarded(pp:&mut Prepared,o:&Opts,j:u64,model:&ModelReg,input:&InputReg,gate:&dyn Fn(&Profile,&Statement,usize))->CpuDeferredRequest{
    request_support(pp,o).expect("invalid registered public layout");
    prove_request_cpu_inner(pp,o,j,model,input,true,Some(gate))
}

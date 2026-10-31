//! GPU prover of the masked authentication (same output as `auth::prove`).
//!
//! Everything proportional to the response dimension runs on the device:
//! Gaussian masks (written straight into per-block row buffers), the
//! commitments of both rows at once (V = 2 shares every key read), Ev_raw,
//! the digit decodes behind H and E_O, the β combination of the witnesses
//! and the norms. The host keeps only O(64)-sized algebra (the final A
//! products, K combinations), the transcript and the rejection coin.

use super::auth::{AuthError, AuthProof, AuxTuple, Point, Responses, Tuple};
use super::coins::{Address, BitStream, Coins, Purpose};
use super::commit_gpu::{finish, partial_raw, KeyCache};
use super::field8::{eq_table, K8};
use super::gaussian_fast::FastSigma;
use super::gaussian_gpu::sample_into;
use super::keys::{KeyKind, KeySeed};
use super::params::{Profile, A_ENC_COLS};
use super::pcs_maps::{aux_index, mul_a};
use super::rejection::{self, Coin, Ratio};
use super::ring::{g_of_weights, AElem, DS, N};
use super::transcript::{Item, MTranscript, Stage};
use almost_goldilocks_cuda::{ffi, DeviceBuffer};
use almost_goldilocks_cuda::field::ALMOST_GOLDILOCKS_PRIME as Q;
use num_bigint::{BigInt, BigUint};
use std::os::raw::c_void;
use std::time::Instant;

/// Key caches of the three block kinds (5 primes: Gaussian inputs).
pub struct Keys { pub w: KeyCache, pub l: KeyCache, pub o: KeyCache }

impl Keys {
    pub fn build(p: &Profile, seed: &KeySeed) -> Self {
        Keys {
            w: KeyCache::build(seed, KeyKind::Source, p.rank, 0, p.source_payload_cols + p.hiding, 5),
            l: KeyCache::build(seed, KeyKind::Link, p.rank, 0, A_ENC_COLS + p.hiding, 5),
            o: KeyCache::build(seed, KeyKind::Aux, p.rank, 0, A_ENC_COLS * p.aux_coeffs + p.hiding, 5),
        }
    }
}

/// Point-dependent device state: w[l][j] = e_l e_j and the host factors.
pub struct DevPoint { w: DeviceBuffer<u64>, cols: usize, gk: AElem, rho_f: K8, x: Vec<K8>, degs: Vec<usize> }

impl DevPoint {
    pub fn new(p: &Profile, pt: &Point) -> Self {
        let cols = p.source_cols();
        let el = eq_table(&pt.xi[12..]);
        let ej = eq_table(&pt.xi[6..12]);
        let ek = eq_table(&pt.xi[..6]);
        assert_eq!(el.len(), cols);
        let flat = |v: &[K8]| -> Vec<u64> { v.iter().flat_map(|k| k.0).collect() };
        let mut w = DeviceBuffer::<u64>::new(cols * 64 * 8).unwrap();
        let rc = unsafe { ffi::agl_mauth_wtable(flat(&el).as_ptr(), cols as i32, flat(&ej).as_ptr(), w.as_mut_ptr()) };
        assert_eq!(rc, 0);
        let wk: [K8; DS] = std::array::from_fn(|k| ek[k]);
        DevPoint { w, cols, gk: g_of_weights(&wk), rho_f: pt.rho.mul(&K8::ONE.sub(&pt.rho)), x: pt.aux_vars(), degs: p.degs() }
    }
    /// Ev_raw of `cols` columns at ptr (int8 or int64).
    pub fn ev(&self, ptr: *const c_void, bytes: i32) -> AElem {
        let mut out = vec![0u64; 64 * 8];
        let rc = unsafe { ffi::agl_mauth_ev(ptr, bytes, self.cols as i32, self.w.as_ptr(), out.as_mut_ptr()) };
        assert_eq!(rc, 0);
        let pm = AElem((0..DS).map(|m| K8(std::array::from_fn(|a| out[m * 8 + a]))).collect());
        mul_a(&self.gk, &pm)
    }
}

fn decode(ptr: *const i64, ncols: usize) -> Vec<u64> {
    let mut out = vec![0u64; ncols * 64];
    let rc = unsafe { ffi::agl_mauth_decode(ptr, ncols as i32, out.as_mut_ptr()) };
    assert_eq!(rc, 0);
    out
}
/// H from decoded values of 8 digit columns starting at column `c0`.
fn h_of(dec: &[u64], c0: usize) -> AElem {
    AElem((0..DS).map(|m| K8(std::array::from_fn(|a| dec[(c0 + a) * 64 + m]))).collect())
}

fn ring_rows(v: Vec<u64>, rows: usize) -> Vec<Vec<Vec<u64>>> {
    // [V][rows][N] -> per vector, per row
    v.chunks(rows * N).map(|blk| blk.chunks(N).map(|r| r.to_vec()).collect()).collect()
}

/// Device witnesses of one source (int8): W (source columns) and r (link columns).
pub struct DevSource { pub w: DeviceBuffer<i8>, pub r: DeviceBuffer<i8> }

#[derive(Debug, Default, Clone)]
pub struct AuthTiming { pub attempts: u32, pub sample_ms: f64, pub first_msg_ms: f64, pub respond_ms: f64, pub decide_ms: f64, pub extract_ms: f64 }

fn norm(ptr: *const i64, n: usize) -> (u128, u64) {
    let mut o = [0u64; 3];
    let rc = unsafe { ffi::agl_mauth_norm(ptr, n as u64, o.as_mut_ptr()) };
    assert_eq!(rc, 0);
    (((o[1] as u128) << 64) | o[0] as u128, o[2])
}

/// The device inputs of one authentication prover (the primary device or a
/// replica): key caches, source witnesses and the auxiliary witness.
pub struct AuthDev<'a> { pub dev: i32, pub keys: &'a Keys, pub sources: &'a [DevSource], pub s_o: &'a DeviceBuffer<i8> }

/// One attempt's first messages, verdict and (device) responses.
struct AttemptOut { verdict: Coin, d: Vec<Tuple>, d_o: Vec<AuxTuple>, z: (DeviceBuffer<i64>, DeviceBuffer<i64>, DeviceBuffer<i64>) }

/// Attempt `attempt` on the current device. Its masks, challenges and coin
/// depend on the attempt index only, so attempts are independent.
#[allow(clippy::too_many_arguments)]
fn run_attempt(
    p: &Profile, ad: &AuthDev, coins: &Coins, opening: u64, tr: &MTranscript, pt: &Point, attempt: u16, timing: &mut AuthTiming,
) -> Result<AttemptOut, AuthError> {
    let (keys, sources, s_o) = (ad.keys, ad.sources, ad.s_o);
    let fs = FastSigma::new(p.sigma);
    let dp = DevPoint::new(p, pt);
    let (sw, sl, so) = (p.source_cols(), p.link_cols(), p.aux_cols());
    let row = ((sw + sl + so) * N) as u64;
    let n = sources.len();
    let degs = p.degs();
    let half = K8::ONE.scale((Q + 1) / 2);
    {
        timing.attempts += 1;
        let t0 = Instant::now();
        // U blocks laid out [row][cols][N] so both rows commit in one pass.
        let mut uw = DeviceBuffer::<i64>::new(2 * sw * N).unwrap();
        let mut ul = DeviceBuffer::<i64>::new(2 * sl * N).unwrap();
        let mut uo = DeviceBuffer::<i64>::new(2 * so * N).unwrap();
        for r in 0..p.tau {
            let b = r as u64 * row;
            unsafe {
                sample_into(coins, opening, attempt, &fs, b as u32, (sw * N) as u32, uw.as_mut_ptr().add(r * sw * N)).map_err(AuthError::Sampler)?;
                sample_into(coins, opening, attempt, &fs, (b + (sw * N) as u64) as u32, (sl * N) as u32, ul.as_mut_ptr().add(r * sl * N)).map_err(AuthError::Sampler)?;
                sample_into(coins, opening, attempt, &fs, (b + ((sw + sl) * N) as u64) as u32, (so * N) as u32, uo.as_mut_ptr().add(r * so * N)).map_err(AuthError::Sampler)?;
            }
        }
        timing.sample_ms += t0.elapsed().as_secs_f64() * 1e3;
        let t0 = Instant::now();
        let (d, d_o) = first_message(p, keys, &dp, &uw, &ul, &uo);
        let _ = &dp.degs;
        timing.first_msg_ms += t0.elapsed().as_secs_f64() * 1e3;
        let t0 = Instant::now();
        let beta = super::auth::challenges_pub(tr, &d, &d_o, attempt, n);
        let mut zw = uw.clone_on_device().unwrap();
        let mut zl = ul.clone_on_device().unwrap();
        let mut zo = uo.clone_on_device().unwrap();
        assert_eq!(p.tau, 2);
        let b2 = |e: usize| -> [i8; 128] { std::array::from_fn(|i| beta[i / 64][e][i % 64]) };
        for (e, s) in sources.iter().enumerate() {
            let bb = b2(e);
            unsafe {
                let rc = ffi::agl_mauth_combine(s.w.as_ptr(), sw as i32, bb.as_ptr(), zw.as_mut_ptr(), zw.as_mut_ptr().add(sw * N));
                assert_eq!(rc, 0);
                let rc = ffi::agl_mauth_combine(s.r.as_ptr(), sl as i32, bb.as_ptr(), zl.as_mut_ptr(), zl.as_mut_ptr().add(sl * N));
                assert_eq!(rc, 0);
            }
        }
        let bb = b2(n);
        unsafe { assert_eq!(ffi::agl_mauth_combine(s_o.as_ptr(), so as i32, bb.as_ptr(), zo.as_mut_ptr(), zo.as_mut_ptr().add(so * N)), 0); }
        timing.respond_ms += t0.elapsed().as_secs_f64() * 1e3;
        let t0 = Instant::now();
        // Caps per row and block; joint norms.
        let mut caps = true;
        let (mut nu, mut nz) = (0u128, 0u128);
        for (ub, zb, cols) in [(&uw, &zw, sw), (&ul, &zl, sl), (&uo, &zo, so)] {
            for r in 0..p.tau {
                let (zn, zmax) = norm(unsafe { zb.as_ptr().add(r * cols * N) }, cols * N);
                let (un, _) = norm(unsafe { ub.as_ptr().add(r * cols * N) }, cols * N);
                nu += un;
                nz += zn;
                if zmax >= p.b_z || zn > p.r2(cols * N) { caps = false; }
            }
        }
        let mut verdict = Coin::Reject;
        if caps {
            let x = Ratio { num: BigInt::from(nu) - BigInt::from(nz), den: BigInt::from(2u128 * p.sigma as u128 * p.sigma as u128) };
            let mut a = Address::new(Purpose::ProtocolCoin, opening);
            a.attempt = attempt;
            let mut st = coins.stream(a);
            let mut ub = BigUint::from(0u8);
            for _ in 0..rejection::B { ub = (ub << 1u32) + BigUint::from(st.next_bit().map_err(|_| AuthError::CoinAbort)?); }
            verdict = rejection::decide(&x, &ub);
        }
        timing.decide_ms += t0.elapsed().as_secs_f64() * 1e3;
        Ok(AttemptOut { verdict, d, d_o, z: (zw, zl, zo) })
    }
}

/// A device's sources for the sharded authentication: physical columns of
/// sources `srcs` at `w` (stride `len` bytes, in that order) and their link
/// columns `r` (same order).
pub struct ShardSrc { pub dev: i32, pub srcs: Vec<usize>, pub w: *const i8, pub len: usize, pub r: DeviceBuffer<i8> }
unsafe impl Send for ShardSrc {}
unsafe impl Sync for ShardSrc {}

/// Attempt `attempt` with sources spread over devices: masks, first message
/// and decision on device 0 (`keys`, `s_o`); each device adds its sources'
/// share of the responses, summed on the host. Same output as `run_attempt`.
#[allow(clippy::too_many_arguments)]
fn run_attempt_sharded(
    p: &Profile, keys: &Keys, shards: &[ShardSrc], s_o: &DeviceBuffer<i8>, n: usize, coins: &Coins, opening: u64, tr: &MTranscript, pt: &Point, attempt: u16, timing: &mut AuthTiming,
) -> Result<AttemptOut, AuthError> {
    almost_goldilocks_cuda::set_device(0).expect("set device");
    let fs = FastSigma::new(p.sigma);
    let dp = DevPoint::new(p, pt);
    let (sw, sl, so) = (p.source_cols(), p.link_cols(), p.aux_cols());
    let row = ((sw + sl + so) * N) as u64;
    timing.attempts += 1;
    let t0 = Instant::now();
    let mut uw = DeviceBuffer::<i64>::new(2 * sw * N).unwrap();
    let mut ul = DeviceBuffer::<i64>::new(2 * sl * N).unwrap();
    let mut uo = DeviceBuffer::<i64>::new(2 * so * N).unwrap();
    for r in 0..p.tau {
        let b = r as u64 * row;
        unsafe {
            sample_into(coins, opening, attempt, &fs, b as u32, (sw * N) as u32, uw.as_mut_ptr().add(r * sw * N)).map_err(AuthError::Sampler)?;
            sample_into(coins, opening, attempt, &fs, (b + (sw * N) as u64) as u32, (sl * N) as u32, ul.as_mut_ptr().add(r * sl * N)).map_err(AuthError::Sampler)?;
            sample_into(coins, opening, attempt, &fs, (b + ((sw + sl) * N) as u64) as u32, (so * N) as u32, uo.as_mut_ptr().add(r * so * N)).map_err(AuthError::Sampler)?;
        }
    }
    timing.sample_ms += t0.elapsed().as_secs_f64() * 1e3;
    let t0 = Instant::now();
    let (d, d_o) = first_message(p, keys, &dp, &uw, &ul, &uo);
    timing.first_msg_ms += t0.elapsed().as_secs_f64() * 1e3;
    let t0 = Instant::now();
    let beta = super::auth::challenges_pub(tr, &d, &d_o, attempt, n);
    assert_eq!(p.tau, 2);
    let b2 = |e: usize| -> [i8; 128] { std::array::from_fn(|i| beta[i / 64][e][i % 64]) };
    // Responses: device 0 adds the one global mask and its sources; the other
    // devices add only their sources' shifts from zero and ship them as i32.
    // Bound: every source/link digit has |v| ≤ B_x − 1 = 1 and a C_0
    // challenge has 64 coefficients of size ≤ 2, so one source adds at most
    // 128 per coordinate and a device partial is at most 128·#sources. The
    // guard below keeps that below 2^31; the host sum is in i64.
    for sh in shards.iter().filter(|sh| sh.dev != 0) {
        assert!((sh.srcs.len() as u64) * 128 < (1u64 << 31), "i32 response partial could overflow: {} sources on device {}", sh.srcs.len(), sh.dev);
    }
    let mut zw = uw.clone_on_device().unwrap();
    let mut zl = ul.clone_on_device().unwrap();
    let combine = |sh: &ShardSrc, zw: &mut DeviceBuffer<i64>, zl: &mut DeviceBuffer<i64>| {
        for (j, &e) in sh.srcs.iter().enumerate() {
            let bb = b2(e);
            unsafe {
                let rc = ffi::agl_mauth_combine(sh.w.add(j * sh.len), sw as i32, bb.as_ptr(), zw.as_mut_ptr(), zw.as_mut_ptr().add(sw * N));
                assert_eq!(rc, 0);
                let rc = ffi::agl_mauth_combine(sh.r.as_ptr().add(j * sl * N), sl as i32, bb.as_ptr(), zl.as_mut_ptr(), zl.as_mut_ptr().add(sl * N));
                assert_eq!(rc, 0);
            }
        }
    };
    let parts: Vec<(Vec<i32>, Vec<i32>)> = std::thread::scope(|sc| {
        let hs: Vec<_> = shards.iter().filter(|sh| sh.dev != 0).map(|sh| {
            let combine = &combine;
            sc.spawn(move || {
                almost_goldilocks_cuda::set_device(sh.dev).expect("set device");
                let mut pw = DeviceBuffer::<i64>::new(2 * sw * N).unwrap();
                let mut pl = DeviceBuffer::<i64>::new(2 * sl * N).unwrap();
                pw.zero().unwrap(); pl.zero().unwrap();
                combine(sh, &mut pw, &mut pl);
                let narrow = |b: &DeviceBuffer<i64>| -> Vec<i32> {
                    let mut d = DeviceBuffer::<i32>::new(b.len()).unwrap();
                    assert_eq!(unsafe { ffi::agl_i64_to_i32(b.as_ptr(), d.as_mut_ptr(), b.len() as u64) }, 0);
                    d.to_vec().unwrap()
                };
                (narrow(&pw), narrow(&pl))
            })
        }).collect();
        almost_goldilocks_cuda::set_device(0).expect("set device");
        for sh in shards.iter().filter(|sh| sh.dev == 0) { combine(sh, &mut zw, &mut zl); }
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    });
    almost_goldilocks_cuda::set_device(0).expect("set device");
    for (pw, pl) in &parts {
        for (dst, src) in [(&mut zw, pw), (&mut zl, pl)] {
            let d = DeviceBuffer::from_slice(src).unwrap();
            assert_eq!(unsafe { ffi::agl_add_i32_i64(dst.as_mut_ptr(), d.as_ptr(), src.len() as u64) }, 0);
        }
    }
    let mut zo = uo.clone_on_device().unwrap();
    let bb = b2(n);
    unsafe { assert_eq!(ffi::agl_mauth_combine(s_o.as_ptr(), so as i32, bb.as_ptr(), zo.as_mut_ptr(), zo.as_mut_ptr().add(so * N)), 0); }
    timing.respond_ms += t0.elapsed().as_secs_f64() * 1e3;
    let t0 = Instant::now();
    let mut caps = true;
    let (mut nu, mut nz) = (0u128, 0u128);
    for (ub, zb, cols) in [(&uw, &zw, sw), (&ul, &zl, sl), (&uo, &zo, so)] {
        for r in 0..p.tau {
            let (zn, zmax) = norm(unsafe { zb.as_ptr().add(r * cols * N) }, cols * N);
            let (un, _) = norm(unsafe { ub.as_ptr().add(r * cols * N) }, cols * N);
            nu += un;
            nz += zn;
            if zmax >= p.b_z || zn > p.r2(cols * N) { caps = false; }
        }
    }
    let mut verdict = Coin::Reject;
    if caps {
        let x = Ratio { num: BigInt::from(nu) - BigInt::from(nz), den: BigInt::from(2u128 * p.sigma as u128 * p.sigma as u128) };
        let mut a = Address::new(Purpose::ProtocolCoin, opening);
        a.attempt = attempt;
        let mut st = coins.stream(a);
        let mut ub = BigUint::from(0u8);
        for _ in 0..rejection::B { ub = (ub << 1u32) + BigUint::from(st.next_bit().map_err(|_| AuthError::CoinAbort)?); }
        verdict = rejection::decide(&x, &ub);
    }
    timing.decide_ms += t0.elapsed().as_secs_f64() * 1e3;
    Ok(AttemptOut { verdict, d, d_o, z: (zw, zl, zo) })
}

/// Sharded prover (attempts in order, each over every device); equals
/// `prove` on the same inputs.
#[allow(clippy::too_many_arguments)]
pub fn prove_sharded(
    p: &Profile, keys: &Keys, shards: &[ShardSrc], s_o: &DeviceBuffer<i8>, n: usize, coins: &Coins, opening: u64, tr: &MTranscript, pt: &Point, keep_z: bool, timing: &mut AuthTiming,
) -> Result<AuthOut, AuthError> {
    for attempt in 0..p.attempts as u16 {
        let o = run_attempt_sharded(p, keys, shards, s_o, n, coins, opening, tr, pt, attempt, timing)?;
        match o.verdict {
            Coin::Accept => { let t = Instant::now(); let out = accept(p, attempt, o, keep_z); timing.extract_ms += t.elapsed().as_secs_f64() * 1e3; return Ok(out); }
            Coin::Reject => {}
            Coin::Abort => return Err(AuthError::CoinAbort),
        }
    }
    Err(AuthError::Exhausted)
}

/// (f_x, f_ox) of τ rows of device blocks laid out [row][cols][N] (i64):
/// the prover's first messages on the masks, the verifier's left sides on z.
fn first_message(p: &Profile, keys: &Keys, dp: &DevPoint, uw: &DeviceBuffer<i64>, ul: &DeviceBuffer<i64>, uo: &DeviceBuffer<i64>) -> (Vec<Tuple>, Vec<AuxTuple>) {
    let (sw, sl, so) = (p.source_cols(), p.link_cols(), p.aux_cols());
    let degs = p.degs();
    let half = K8::ONE.scale((Q + 1) / 2);
    let cw = ring_rows(finish(partial_raw(&keys.w, 5, uw.as_ptr() as *const c_void, 8, 2, sw - p.rank, sw * N, None), 2, 5, p.rank, None).to_vec().unwrap(), p.rank);
    let cl = ring_rows(finish(partial_raw(&keys.l, 5, ul.as_ptr() as *const c_void, 8, 2, sl - p.rank, sl * N, None), 2, 5, p.rank, None).to_vec().unwrap(), p.rank);
    let co = ring_rows(finish(partial_raw(&keys.o, 5, uo.as_ptr() as *const c_void, 8, 2, so - p.rank, so * N, None), 2, 5, p.rank, None).to_vec().unwrap(), p.rank);
    let (mut d, mut d_o) = (Vec::new(), Vec::new());
    for r in 0..p.tau {
        // The identity error columns are the last `rank` columns of each block.
        let add_err = |c: &Vec<Vec<u64>>, base: *const i64, cols: usize| -> Vec<Vec<u64>> {
            let mut e = vec![0i64; p.rank * N];
            let src = unsafe { base.add((r * cols + cols - p.rank) * N) };
            let rc = unsafe { ffi::cuda_memcpy_dtoh(e.as_mut_ptr() as *mut c_void, src as *const c_void, p.rank * N * 8) };
            assert_eq!(rc, 0);
            c.iter().enumerate().map(|(i, row)| row.iter().enumerate().map(|(j, &v)| {
                let x = e[i * N + j];
                let xe = if x >= 0 { x as u64 } else { Q - x.unsigned_abs() };
                ((v as u128 + xe as u128) % Q as u128) as u64
            }).collect()).collect()
        };
        let c = add_err(&cw[r], uw.as_ptr(), sw);
        let g = add_err(&cl[r], ul.as_ptr(), sl);
        let hl = h_of(&decode(unsafe { ul.as_ptr().add(r * sl * N) }, A_ENC_COLS), 0);
        let a = dp.ev(unsafe { uw.as_ptr().add(r * sw * N) } as *const c_void, 8).add(&hl.scale_k(&dp.rho_f));
        d.push(Tuple { c, g, a });
        let ncoef = p.aux_coeffs;
        let dec = decode(unsafe { uo.as_ptr().add(r * so * N) }, A_ENC_COLS * ncoef);
        let mut t = AElem::zero();
        for (i, &dg) in degs.iter().enumerate() {
            let mut xp = K8::ONE;
            for k in 1..=dg {
                xp = xp.mul(&dp.x[i]);
                t = t.add(&h_of(&dec, A_ENC_COLS * aux_index(&degs, i, k)).scale_k(&xp.sub(&half)));
            }
        }
        d_o.push(AuxTuple { c: add_err(&co[r], uo.as_ptr(), so), t });
    }
    (d, d_o)
}

type AuthOut = (AuthProof, Option<(DeviceBuffer<i64>, DeviceBuffer<i64>, DeviceBuffer<i64>)>);

fn accept(p: &Profile, attempt: u16, o: AttemptOut, keep_z: bool) -> AuthOut {
    let (zw, zl, zo) = o.z;
    // Serialize on the device (37 bits per coefficient), one D2H copy per block.
    let z = if keep_z { Responses { w: pack_dev(p, &zw), l: pack_dev(p, &zl), o: pack_dev(p, &zo) } } else { Responses::default() };
    (AuthProof { attempt, d: o.d, d_o: o.d_o, z }, Some((zw, zl, zo)))
}

fn pack_dev(p: &Profile, z: &DeviceBuffer<i64>) -> Vec<u8> {
    let mut b = DeviceBuffer::<u8>::new(super::wire::packed_len(z.len())).unwrap();
    assert_eq!(unsafe { ffi::agl_mauth_pack37(z.as_ptr(), z.len() as u64, super::wire::z_off(p), b.as_mut_ptr()) }, 0);
    b.to_vec().unwrap()
}

/// Device responses of one block from the wire; None if a code is outside
/// the support or the length is wrong.
fn unpack_dev(p: &Profile, b: &[u8], cols: usize) -> Option<DeviceBuffer<i64>> {
    let n = p.tau * cols * N;
    if b.len() != super::wire::packed_len(n) { return None; }
    let d = DeviceBuffer::from_slice(b).unwrap();
    let mut z = DeviceBuffer::<i64>::new(n).unwrap();
    let rc = unsafe { ffi::agl_mauth_unpack37(d.as_ptr(), n as u64, super::wire::z_off(p), super::wire::z_umax(p), z.as_mut_ptr()) };
    assert!(rc >= 0, "unpack37 failed");
    (rc == 0).then_some(z)
}

/// GPU prover; equals `auth::prove` on the same inputs.
#[allow(clippy::too_many_arguments)]
pub fn prove(
    p: &Profile, keys: &Keys, coins: &Coins, opening: u64, tr: &MTranscript, pt: &Point,
    sources: &[DevSource], s_o: &DeviceBuffer<i8>, keep_z: bool, timing: &mut AuthTiming,
) -> Result<AuthOut, AuthError> {
    let ad = AuthDev { dev: -1, keys, sources, s_o };
    for attempt in 0..p.attempts as u16 {
        let o = run_attempt(p, &ad, coins, opening, tr, pt, attempt, timing)?;
        match o.verdict {
            Coin::Accept => { let t = Instant::now(); let out = accept(p, attempt, o, keep_z); timing.extract_ms += t.elapsed().as_secs_f64() * 1e3; return Ok(out); }
            Coin::Reject => {}
            Coin::Abort => return Err(AuthError::CoinAbort),
        }
    }
    Err(AuthError::Exhausted)
}

/// Same output as `prove`, with attempts run in batches of `devs.len()`, one
/// per device; the first non-rejected attempt in index order decides.
#[allow(clippy::too_many_arguments)]
pub fn prove_multi(
    p: &Profile, devs: &[AuthDev], coins: &Coins, opening: u64, tr: &MTranscript, pt: &Point, keep_z: bool, timing: &mut AuthTiming,
) -> Result<AuthOut, AuthError> {
    let g = devs.len().max(1);
    let total = p.attempts as u16;
    let mut base = 0u16;
    while base < total {
        let outs: Vec<(u16, Result<AttemptOut, AuthError>, AuthTiming)> = std::thread::scope(|sc| {
            let hs: Vec<_> = devs.iter().enumerate().filter(|(j, _)| base as usize + j < total as usize).map(|(j, ad)| {
                let attempt = base + j as u16;
                sc.spawn(move || {
                    if ad.dev >= 0 { almost_goldilocks_cuda::set_device(ad.dev).expect("set device"); }
                    let mut tm = AuthTiming::default();
                    let r = run_attempt(p, ad, coins, opening, tr, pt, attempt, &mut tm);
                    (attempt, r, tm)
                })
            }).collect();
            hs.into_iter().map(|h| h.join().unwrap()).collect()
        });
        let mut done: Option<Result<AuthOut, AuthError>> = None;
        for (attempt, r, tm) in outs {
            if done.is_some() { continue; }
            timing.attempts += tm.attempts;
            timing.sample_ms += tm.sample_ms; timing.first_msg_ms += tm.first_msg_ms;
            timing.respond_ms += tm.respond_ms; timing.decide_ms += tm.decide_ms;
            match r {
                Err(e) => done = Some(Err(e)),
                Ok(o) => match o.verdict {
                    Coin::Accept => {
                        // The accepted responses live on that attempt's device.
                        let dev = devs[(attempt - base) as usize].dev;
                        if dev >= 0 { almost_goldilocks_cuda::set_device(dev).expect("set device"); }
                        let t = Instant::now();
                        done = Some(Ok(accept(p, attempt, o, keep_z)));
                        timing.extract_ms += t.elapsed().as_secs_f64() * 1e3;
                        if devs[0].dev >= 0 { almost_goldilocks_cuda::set_device(devs[0].dev).expect("set device"); }
                    }
                    Coin::Reject => {}
                    Coin::Abort => done = Some(Err(AuthError::CoinAbort)),
                },
            }
        }
        if let Some(d) = done { return d; }
        base += g as u16;
    }
    Err(AuthError::Exhausted)
}

/// GPU verifier of the masked authentication (same verdict as `auth::verify`).
#[allow(clippy::too_many_arguments)]
pub fn verify(p: &Profile, keys: &Keys, tr: &MTranscript, pt: &Point, targets: &[Tuple], target_o: &AuxTuple, pf: &AuthProof) -> bool {
    use super::auth::{act_tuple, add_ring, add_tuple};
    use super::pcs_maps::{act_c0_a, act_c0_ring_q};
    use rayon::prelude::*;
    if !super::wire::auth_shape_ok(p, pf) || targets.len() > p.n_max
        || pt.xi.len() != p.ell() || !pt.xi.iter().all(super::wire::canonical_k)
        || !super::wire::canonical_k(&pt.rho) || pt.rho == K8::ZERO || pt.rho == K8::ONE
        || !targets.iter().all(|t| super::wire::commitment_ok(p, &t.c)
            && super::wire::commitment_ok(p, &t.g) && super::wire::endpoint_ok(&t.a))
        || !super::wire::commitment_ok(p, &target_o.c) || !super::wire::endpoint_ok(&target_o.t) { return false; }
    let (sw, sl, so) = (p.source_cols(), p.link_cols(), p.aux_cols());
    if (pf.attempt as usize) >= p.attempts || pf.d.len() != p.tau || pf.d_o.len() != p.tau {
        return false;
    }
    let n = targets.len();
    let beta = super::auth::challenges_pub(tr, &pf.d, &pf.d_o, pf.attempt, n);
    let (zw, zl, zo) = match (unpack_dev(p, &pf.z.w, sw), unpack_dev(p, &pf.z.l, sl), unpack_dev(p, &pf.z.o, so)) {
        (Some(a), Some(b), Some(c)) => (a, b, c),
        _ => return false,
    };
    // Caps per row and block.
    for (zb, cols) in [(&zw, sw), (&zl, sl), (&zo, so)] {
        for r in 0..p.tau {
            let (zn, zmax) = norm(unsafe { zb.as_ptr().add(r * cols * N) }, cols * N);
            if zmax >= p.b_z || zn > p.r2(cols * N) { return false; }
        }
    }
    let dp = DevPoint::new(p, pt);
    let (lhs, lhs_o) = first_message(p, keys, &dp, &zw, &zl, &zo);
    for r in 0..p.tau {
        let acted: Vec<Tuple> = targets.par_iter().enumerate().map(|(e, t)| act_tuple(&beta[r][e], t)).collect();
        let rhs = acted.iter().fold(pf.d[r].clone(), |acc, t| add_tuple(&acc, t));
        if lhs[r] != rhs { return false; }
        let bt = &beta[r][n];
        let ro = AuxTuple {
            c: add_ring(&pf.d_o[r].c, &target_o.c.par_iter().map(|x| act_c0_ring_q(bt, x)).collect::<Vec<_>>()),
            t: pf.d_o[r].t.add(&act_c0_a(bt, &target_o.t)),
        };
        if lhs_o[r] != ro { return false; }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::auth::{self, f_ox, f_x, SourceVec};
    use super::super::coins::CoinProfile;
    use super::super::pcs_maps::enc;
    use super::super::sampler::ternary_vec;
    use rand::{Rng, SeedableRng};

    fn rk<R: rand::Rng>(rng: &mut R) -> K8 { K8(std::array::from_fn(|_| rng.gen::<u64>() % Q)) }
    fn to_dev(cols: &[Vec<i64>]) -> DeviceBuffer<i8> {
        DeviceBuffer::from_slice(&cols.iter().flatten().map(|&v| v as i8).collect::<Vec<_>>()).unwrap()
    }

    #[test]
    fn gpu_authentication_equals_cpu_authentication() {
        let p = Profile::toy();
        let seed = KeySeed([3; 8]);
        let coins = Coins::new([9u8; 64], CoinProfile::Ternary);
        let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(2);
        let tern = |rng: &mut rand_chacha::ChaCha20Rng, cols: usize| -> Vec<Vec<i64>> { (0..cols).map(|_| ternary_vec(rng, N).unwrap()).collect() };
        let sources: Vec<SourceVec> = (0..2).map(|_| {
            let w = tern(&mut rng, p.source_cols());
            let u = AElem((0..64).map(|_| rk(&mut rng)).collect());
            SourceVec { w, r: [enc(&u), tern(&mut rng, p.hiding + p.rank)].concat() }
        }).collect();
        let aux_payload: Vec<Vec<i64>> = (0..p.aux_coeffs).flat_map(|_| enc(&AElem((0..64).map(|_| rk(&mut rng)).collect()))).collect();
        let s_o = [aux_payload, tern(&mut rng, p.hiding + p.rank)].concat();
        let pt = Point { xi: (0..p.ell()).map(|_| rk(&mut rng)).collect(), rho: rk(&mut rng) };
        let targets: Vec<Tuple> = sources.iter().map(|s| f_x(&p, &seed, &pt, &s.w, &s.r)).collect();
        let target_o = f_ox(&p, &seed, &pt, &s_o);
        let mut tr = MTranscript::new(p.name, &coins.salt(0));
        for t in &targets { for c in &t.c { tr.absorb(Item::SourceCommitment, c); } }
        let cpu = auth::prove(&p, &seed, &coins, 0, &tr, &pt, &sources, &s_o).unwrap();

        let keys = Keys::build(&p, &seed);
        let dsrc: Vec<DevSource> = sources.iter().map(|s| DevSource { w: to_dev(&s.w), r: to_dev(&s.r) }).collect();
        let mut tm = AuthTiming::default();
        let (gpu, _) = prove(&p, &keys, &coins, 0, &tr, &pt, &dsrc, &to_dev(&s_o), true, &mut tm).unwrap();
        assert_eq!(gpu.attempt, cpu.attempt);
        for r in 0..2 { eprintln!("row {r}: c {} g {} a {} | o.c {} o.t {}", gpu.d[r].c == cpu.d[r].c, gpu.d[r].g == cpu.d[r].g, gpu.d[r].a == cpu.d[r].a, gpu.d_o[r].c == cpu.d_o[r].c, gpu.d_o[r].t == cpu.d_o[r].t); }
        assert_eq!(gpu.d, cpu.d);
        assert_eq!(gpu.d_o, cpu.d_o);
        assert_eq!(gpu.z, cpu.z);
        assert!(auth::verify(&p, &seed, &tr, &pt, &targets, &target_o, &gpu));
        let _ = rng.gen::<u8>();
    }

    #[test]
    #[ignore]
    fn gpu_authentication_full_profile() {
        // Ternary profile, synthetic sources: time per attempt on one GPU.
        let p = Profile::ternary();
        let seed = KeySeed([3; 8]);
        let coins = Coins::new([9u8; 64], CoinProfile::Ternary);
        let n: usize = std::env::var("AUTH_SOURCES").ok().and_then(|s| s.parse().ok()).unwrap_or(2);
        let t = Instant::now();
        let keys = Keys::build(&p, &seed);
        println!("key caches: {:.1} s, {:.1} GB", t.elapsed().as_secs_f64(), (keys.w.bytes() + keys.l.bytes() + keys.o.bytes()) as f64 / 1e9);
        let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(1);
        let mut rnd8 = |len: usize| -> DeviceBuffer<i8> { DeviceBuffer::from_slice(&(0..len).map(|_| rng.gen_range(-1i8..=1)).collect::<Vec<_>>()).unwrap() };
        let dsrc: Vec<DevSource> = (0..n).map(|_| DevSource { w: rnd8(p.source_cols() * N), r: rnd8(p.link_cols() * N) }).collect();
        let s_o = rnd8(p.aux_cols() * N);
        let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(2);
        let pt = Point { xi: (0..p.ell()).map(|_| rk(&mut rng)).collect(), rho: rk(&mut rng) };
        let tr = MTranscript::new(p.name, &coins.salt(0));
        for op in 0..3u64 {
            let mut tm = AuthTiming::default();
            let t = Instant::now();
            let (pf, _) = prove(&p, &keys, &coins, op, &tr, &pt, &dsrc, &s_o, false, &mut tm).unwrap();
            println!("opening {op}: {n} sources, accepted at attempt {} in {:.2} s  {tm:?}", pf.attempt, t.elapsed().as_secs_f64());
        }
    }

    #[test]
    fn ev_and_decode_kernels_match_cpu() {
        let p = Profile::toy();
        let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(5);
        let pt = Point { xi: (0..p.ell()).map(|_| rk(&mut rng)).collect(), rho: rk(&mut rng) };
        let dp = DevPoint::new(&p, &pt);
        for big in [false, true] {
            let cols: Vec<Vec<i64>> = (0..p.source_cols()).map(|_| (0..N).map(|_| if big { rng.gen_range(-(1i64 << 40)..(1i64 << 40)) } else { rng.gen_range(-1i64..=1) }).collect()).collect();
            let flat: Vec<i64> = cols.iter().flatten().copied().collect();
            let d = DeviceBuffer::from_slice(&flat).unwrap();
            assert_eq!(dp.ev(d.as_ptr() as *const c_void, 8), super::super::pcs_maps::ev_raw(&cols, &pt.xi), "big={big}");
            let dec = decode(d.as_ptr(), 8);
            assert_eq!(h_of(&dec, 0), super::super::pcs_maps::decode_h(&cols[..8]), "decode big={big}");
        }
    }
}

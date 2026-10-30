//! Proof wire format (v1). The reported proof size is the length of this
//! encoding, not an estimate.
//!
//! * Field elements of F_q and ring coefficients of commitments: 8 bytes
//!   little-endian, canonical (< q). A K element is 8 of them.
//! * Authentication responses: |Z| < B_Z, so a coefficient has 2·B_Z − 1
//!   permitted values and needs ⌈log2(2·B_Z − 1)⌉ = 37 bits. It is sent as
//!   u = Z + (B_Z − 1), LSB first, eight coefficients per 37 bytes, in the
//!   order block (W, L, O), row, column, coefficient.
//! * Every variable-length list carries a u32 length; the decoder rejects
//!   trailing bytes and any length that does not match the profile.

use super::auth::{AuthProof, AuxTuple, Responses, Tuple};
use super::field8::K8;
use super::link::LinkProof;
use super::params::Profile;
use super::ring::{AElem, N};
use rayon::prelude::*;

pub const MAGIC: &[u8; 8] = b"MRPHSv1\0";
pub const Z_BITS: usize = 37;

/// Offset and largest code of the response codec.
pub fn z_off(p: &Profile) -> i64 { p.b_z as i64 - 1 }
pub fn z_umax(p: &Profile) -> u64 { 2 * p.b_z - 2 }
/// Bytes of `n` packed coefficients (n a multiple of 8).
pub fn packed_len(n: usize) -> usize { assert_eq!(n % 8, 0); n / 8 * Z_BITS }

pub fn pack37(z: &[i64], off: i64) -> Vec<u8> {
    let mut out = vec![0u8; packed_len(z.len())];
    out.par_chunks_mut(Z_BITS).zip(z.par_chunks(8)).for_each(|(o, g)| {
        let mut w = [0u64; 5];
        for (t, &v) in g.iter().enumerate() {
            let u = ((v + off) as u64) & ((1 << Z_BITS) - 1);
            let (bp, i) = (Z_BITS * t, Z_BITS * t / 64);
            let sh = bp % 64;
            w[i] |= u << sh;
            if sh + Z_BITS > 64 { w[i + 1] |= u >> (64 - sh); }
        }
        for (b, x) in o.iter_mut().enumerate() { *x = (w[b / 8] >> (8 * (b % 8))) as u8; }
    });
    out
}

/// Inverse of `pack37`; None if a code exceeds `umax` or the length is off.
pub fn unpack37(b: &[u8], n: usize, off: i64, umax: u64) -> Option<Vec<i64>> {
    if n % 8 != 0 || b.len() != packed_len(n) { return None; }
    let mut z = vec![0i64; n];
    let ok = z.par_chunks_mut(8).zip(b.par_chunks(Z_BITS)).all(|(g, p)| {
        let mut w = [0u64; 5];
        for (i, &x) in p.iter().enumerate() { w[i / 8] |= (x as u64) << (8 * (i % 8)); }
        let mut ok = true;
        for (t, v) in g.iter_mut().enumerate() {
            let (bp, i) = (Z_BITS * t, Z_BITS * t / 64);
            let sh = bp % 64;
            let mut u = w[i] >> sh;
            if sh + Z_BITS > 64 { u |= w[i + 1] << (64 - sh); }
            u &= (1 << Z_BITS) - 1;
            ok &= u <= umax;
            *v = u as i64 - off;
        }
        ok
    });
    ok.then_some(z)
}

/// Rows [row][col][N] of a block from its packed bytes.
pub fn rows_of(p: &Profile, bytes: &[u8], cols: usize) -> Option<Vec<Vec<Vec<i64>>>> {
    let z = unpack37(bytes, p.tau * cols * N, z_off(p), z_umax(p))?;
    Some(z.chunks(cols * N).map(|r| r.chunks(N).map(|c| c.to_vec()).collect()).collect())
}
/// Packed bytes of rows [row][col][N].
pub fn pack_rows(p: &Profile, rows: &[Vec<Vec<i64>>]) -> Vec<u8> {
    let flat: Vec<i64> = rows.iter().flatten().flatten().copied().collect();
    pack37(&flat, z_off(p))
}

// ------------------------------------------------------------------ writer

#[derive(Default)]
pub struct W { pub b: Vec<u8> }

impl W {
    pub fn u8(&mut self, x: u8) { self.b.push(x); }
    pub fn u16(&mut self, x: u16) { self.b.extend_from_slice(&x.to_le_bytes()); }
    pub fn u32(&mut self, x: u32) { self.b.extend_from_slice(&x.to_le_bytes()); }
    pub fn len(&mut self, n: usize) { self.u32(u32::try_from(n).expect("wire length")); }
    pub fn u64s(&mut self, v: &[u64]) {
        self.len(v.len());
        self.b.reserve(8 * v.len());
        for x in v { self.b.extend_from_slice(&x.to_le_bytes()); }
    }
    pub fn ring_vec(&mut self, v: &[Vec<u64>]) { self.len(v.len()); for x in v { self.u64s(x); } }
    pub fn k8(&mut self, x: &K8) { for c in &x.0 { self.b.extend_from_slice(&c.to_le_bytes()); } }
    pub fn k8s(&mut self, v: &[K8]) { self.len(v.len()); for x in v { self.k8(x); } }
    pub fn a(&mut self, a: &AElem) { self.k8s(&a.0); }
    pub fn bytes(&mut self, v: &[u8]) { self.len(v.len()); self.b.extend_from_slice(v); }
}

pub struct R<'a> { b: &'a [u8], pub pos: usize }

impl<'a> R<'a> {
    pub fn new(b: &'a [u8]) -> Self { R { b, pos: 0 } }
    fn take(&mut self, n: usize) -> Option<&'a [u8]> {
        let s = self.b.get(self.pos..self.pos.checked_add(n)?)?;
        self.pos += n;
        Some(s)
    }
    pub fn done(&self) -> bool { self.pos == self.b.len() }
    pub fn u8(&mut self) -> Option<u8> { Some(self.take(1)?[0]) }
    pub fn u16(&mut self) -> Option<u16> { Some(u16::from_le_bytes(self.take(2)?.try_into().ok()?)) }
    pub fn u32(&mut self) -> Option<u32> { Some(u32::from_le_bytes(self.take(4)?.try_into().ok()?)) }
    pub fn len(&mut self) -> Option<usize> { Some(self.u32()? as usize) }
    fn canon(x: u64) -> Option<u64> { (x < almost_goldilocks_cuda::field::ALMOST_GOLDILOCKS_PRIME).then_some(x) }
    pub fn u64s(&mut self) -> Option<Vec<u64>> {
        let n = self.len()?;
        let s = self.take(n.checked_mul(8)?)?;
        s.chunks(8).map(|c| Self::canon(u64::from_le_bytes(c.try_into().unwrap()))).collect()
    }
    pub fn ring_vec(&mut self) -> Option<Vec<Vec<u64>>> { let n = self.len()?; (0..n).map(|_| self.u64s()).collect() }
    pub fn k8(&mut self) -> Option<K8> {
        let s = self.take(64)?;
        let mut c = [0u64; 8];
        for (i, x) in c.iter_mut().enumerate() { *x = Self::canon(u64::from_le_bytes(s[8 * i..8 * i + 8].try_into().unwrap()))?; }
        Some(K8(c))
    }
    pub fn k8s(&mut self) -> Option<Vec<K8>> { let n = self.len()?; (0..n).map(|_| self.k8()).collect() }
    pub fn a(&mut self) -> Option<AElem> { Some(AElem(self.k8s()?)) }
    pub fn bytes(&mut self) -> Option<Vec<u8>> { let n = self.len()?; Some(self.take(n)?.to_vec()) }
}

// ------------------------------------------------------------ proof parts

pub fn put_link(w: &mut W, l: &LinkProof) {
    w.len(l.g_commit.len());
    for g in &l.g_commit { w.ring_vec(g); }
    w.ring_vec(&l.c_o);
    w.len(l.rounds.len());
    for r in &l.rounds { w.k8s(r); }
    w.len(l.a_hat.len());
    for a in &l.a_hat { w.a(a); }
    w.a(&l.t_o);
}
pub fn get_link(r: &mut R) -> Option<LinkProof> {
    let n = r.len()?;
    let g_commit = (0..n).map(|_| r.ring_vec()).collect::<Option<_>>()?;
    let c_o = r.ring_vec()?;
    let nr = r.len()?;
    let rounds = (0..nr).map(|_| r.k8s()).collect::<Option<_>>()?;
    let na = r.len()?;
    let a_hat = (0..na).map(|_| r.a()).collect::<Option<_>>()?;
    let t_o = r.a()?;
    Some(LinkProof { g_commit, c_o, rounds, a_hat, t_o })
}

/// Authentication proof; the response lengths are fixed by the profile.
pub fn put_auth(w: &mut W, a: &AuthProof) {
    w.u16(a.attempt);
    w.len(a.d.len());
    for t in &a.d { w.ring_vec(&t.c); w.ring_vec(&t.g); w.a(&t.a); }
    w.len(a.d_o.len());
    for t in &a.d_o { w.ring_vec(&t.c); w.a(&t.t); }
    for z in [&a.z.w, &a.z.l, &a.z.o] { w.bytes(z); }
}
pub fn get_auth(p: &Profile, r: &mut R) -> Option<AuthProof> {
    let attempt = r.u16()?;
    let nd = r.len()?;
    let d = (0..nd).map(|_| Some(Tuple { c: r.ring_vec()?, g: r.ring_vec()?, a: r.a()? })).collect::<Option<Vec<_>>>()?;
    let no = r.len()?;
    let d_o = (0..no).map(|_| Some(AuxTuple { c: r.ring_vec()?, t: r.a()? })).collect::<Option<Vec<_>>>()?;
    let z = Responses { w: r.bytes()?, l: r.bytes()?, o: r.bytes()? };
    let n = |cols: usize| packed_len(p.tau * cols * N);
    if z.w.len() != n(p.source_cols()) || z.l.len() != n(p.link_cols()) || z.o.len() != n(p.aux_cols()) { return None; }
    Some(AuthProof { attempt, d, d_o, z })
}

fn put_opt_auth(w: &mut W, a: &Option<AuthProof>) {
    match a { Some(a) => { w.u8(1); put_auth(w, a); } None => w.u8(0) }
}
fn get_opt_auth(p: &Profile, r: &mut R) -> Option<Option<AuthProof>> {
    match r.u8()? { 0 => Some(None), 1 => Some(Some(get_auth(p, r)?)), _ => None }
}

/// The transmitted part of a request proof.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RequestProof {
    pub salt: [u8; 64],
    pub piop: Vec<K8>,
    pub c_src: Vec<Vec<Vec<u64>>>,
    pub link: LinkProof,
    pub auth: Option<AuthProof>,
}

pub fn encode_request(p: &Profile, pf: &RequestProof) -> Vec<u8> {
    let mut w = W::default();
    w.b.extend_from_slice(MAGIC);
    w.bytes(p.name.as_bytes());
    w.b.extend_from_slice(&pf.salt);
    w.k8s(&pf.piop);
    w.len(pf.c_src.len());
    for c in &pf.c_src { w.ring_vec(c); }
    put_link(&mut w, &pf.link);
    put_opt_auth(&mut w, &pf.auth);
    w.b
}

pub fn decode_request(p: &Profile, b: &[u8]) -> Option<RequestProof> {
    let mut r = R::new(b);
    if r.take(8)? != MAGIC || r.bytes()? != p.name.as_bytes() { return None; }
    let salt: [u8; 64] = r.take(64)?.try_into().ok()?;
    let piop = r.k8s()?;
    let n = r.len()?;
    let c_src = (0..n).map(|_| r.ring_vec()).collect::<Option<_>>()?;
    let link = get_link(&mut r)?;
    let auth = get_opt_auth(p, &mut r)?;
    r.done().then_some(RequestProof { salt, piop, c_src, link, auth })
}

/// Encoded bytes by part: (header+salt, PIOP messages, source commitments,
/// link round messages, link commitments and endpoints, authentication
/// first messages, responses).
pub fn parts(p: &Profile, pf: &RequestProof) -> [(&'static str, usize); 7] {
    let total = encode_request(p, pf).len();
    let mut w = W::default(); w.k8s(&pf.piop); let piop = w.b.len();
    let mut w = W::default(); w.len(pf.c_src.len()); for c in &pf.c_src { w.ring_vec(c); } let csrc = w.b.len();
    let mut w = W::default(); w.len(pf.link.rounds.len()); for r in &pf.link.rounds { w.k8s(r); } let rounds = w.b.len();
    let mut w = W::default(); put_link(&mut w, &pf.link); let link = w.b.len() - rounds;
    let (mut first, mut resp) = (0, 0);
    if let Some(a) = &pf.auth {
        let mut w = W::default(); put_auth(&mut w, a); let all = w.b.len();
        resp = a.z.w.len() + a.z.l.len() + a.z.o.len() + 12;
        first = all - resp;
    }
    let head = total - piop - csrc - rounds - link - first - resp;
    [("header", head), ("piop", piop), ("source commitments", csrc), ("link rounds", rounds), ("link commitments", link), ("auth first messages", first), ("responses", resp)]
}

/// The transmitted part of a finalization proof.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FinalProof { pub salt: [u8; 64], pub link: LinkProof, pub auth: Option<AuthProof> }

pub fn encode_final(p: &Profile, pf: &FinalProof) -> Vec<u8> {
    let mut w = W::default();
    w.b.extend_from_slice(MAGIC);
    w.bytes(p.name.as_bytes());
    w.b.extend_from_slice(&pf.salt);
    put_link(&mut w, &pf.link);
    put_opt_auth(&mut w, &pf.auth);
    w.b
}

pub fn decode_final(p: &Profile, b: &[u8]) -> Option<FinalProof> {
    let mut r = R::new(b);
    if r.take(8)? != MAGIC || r.bytes()? != p.name.as_bytes() { return None; }
    let salt: [u8; 64] = r.take(64)?.try_into().ok()?;
    let link = get_link(&mut r)?;
    let auth = get_opt_auth(p, &mut r)?;
    r.done().then_some(FinalProof { salt, link, auth })
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{Rng, SeedableRng};

    #[test]
    fn pack37_round_trips_the_full_support_and_rejects_out_of_range_codes() {
        let p = Profile::ternary();
        assert_eq!(64 - (2 * p.b_z - 2).leading_zeros() as usize, Z_BITS);
        let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(5);
        let bz = p.b_z as i64;
        let mut z: Vec<i64> = (0..4096).map(|_| rng.gen_range(-(bz - 1)..bz)).collect();
        z[..4].copy_from_slice(&[-(bz - 1), bz - 1, 0, -1]);
        let b = pack37(&z, z_off(&p));
        assert_eq!(b.len(), 4096 * 37 / 8);
        assert_eq!(unpack37(&b, z.len(), z_off(&p), z_umax(&p)).unwrap(), z);
        // Code 2·B_Z − 1 (Z = B_Z) is outside the support.
        let mut bad = z.clone();
        bad[7] = bz;
        assert!(unpack37(&pack37(&bad, z_off(&p)), z.len(), z_off(&p), z_umax(&p)).is_none());
        assert!(unpack37(&b[1..], z.len(), z_off(&p), z_umax(&p)).is_none());
    }
}

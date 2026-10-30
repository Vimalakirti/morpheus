//! Fiat-Shamir transcript for the S&P'27 protocol (paper §Preliminaries,
//! "raw adapter"; theory source causal_profile finite_interface §2).
//!
//! The Monolith duplex sponge of `crate::transcript` (kept per the user's
//! decision) absorbs the complete public prefix: every item is absorbed as
//! (type tag, length, canonical words), so the encoding is injective, and no
//! digest replaces any part of the prefix. A challenge is the oracle answer
//! at the typed address (result type, stage, attempt, index) appended to
//! that prefix. The opening salt is absorbed first.
//!
//! Raw adapter: squeezed sponge words are uniform base-q digits; a raw word
//! is their little-endian base-q rank R in [0, q^m). A challenge of a type
//! with N elements is unrank(R mod N). K^k uses m = 8k digits (exact: the
//! rank is uniform on q^{8k} = |K^k|). Every other type gets the registered
//! 8192-bit modulo slack: q^m >= N 2^8192, so the modular bias is below
//! 2^-8192. One query answers one typed bundle: the link's initial
//! (alpha, gamma, eta_O) in K^{l+2}, and an authentication attempt's whole
//! terminal tuple (C0^tau)^{n+1}. The paper's 2^30-bit raw width is a proof
//! device; only the slack enters the decoded law.

use super::field8::K8;
use crate::transcript::Transcript;
use almost_goldilocks_cuda::field::ALMOST_GOLDILOCKS_PRIME as Q;
use num_bigint::BigUint;
use num_traits::{ToPrimitive, Zero};

/// Item types (absorbed before each item).
#[repr(u64)]
#[derive(Clone, Copy, Debug)]
pub enum Item {
    Profile = 1,
    Salt = 2,
    Statement = 3,
    SourceCommitment = 4,
    LinkCommitment = 5,
    AuxCommitment = 6,
    RoundPoly = 7,
    Endpoint = 8,
    FirstMessage = 9,
    GraphClaim = 10,
    Layout = 11,
    Other = 99,
}

/// Challenge result types.
#[repr(u64)]
#[derive(Clone, Copy, Debug)]
pub enum ChalType { K = 1, KNot01 = 2, C0 = 3, KBundle = 4, C0Tuple = 5 }

/// Protocol stages (part of every challenge address).
#[repr(u64)]
#[derive(Clone, Copy, Debug)]
pub enum Stage {
    GraphMask = 1,
    GraphReduction = 2,
    GraphMerge = 3,
    LinkRange = 4,
    LinkCombine = 5,
    LinkRound = 6,
    AuthColumn = 7,
    LinkInit = 8,
}

/// Modulo slack (bits) of every non-exact raw decoder.
pub const RAW_SLACK: usize = 8192;
/// Base-q digits whose rank covers 2^bits (each digit carries > 63 bits).
fn digits_for(bits: usize) -> usize { bits.div_ceil(63) }

#[derive(Clone)]
pub struct MTranscript { t: Transcript }

impl MTranscript {
    pub fn new(profile: &str, salt: &[u8; 64]) -> Self {
        let mut t = MTranscript { t: Transcript::new(b"MORPHEUS/FS/v1") };
        t.absorb_bytes(Item::Profile, profile.as_bytes());
        t.absorb_bytes(Item::Salt, salt);
        t
    }
    /// Absorb canonical field words.
    pub fn absorb(&mut self, item: Item, words: &[u64]) {
        self.t.append_u64(b"", item as u64);
        self.t.append_u64(b"", words.len() as u64);
        self.t.append_u64_slice(b"", words);
    }
    pub fn absorb_bytes(&mut self, item: Item, bytes: &[u8]) {
        // 7 bytes per word keeps every word below q; the length disambiguates.
        let words: Vec<u64> = bytes.chunks(7).map(|c| c.iter().rev().fold(0u64, |a, &b| (a << 8) | b as u64)).collect();
        self.t.append_u64(b"", item as u64);
        self.t.append_u64(b"", bytes.len() as u64);
        self.t.append_u64_slice(b"", &words);
    }
    pub fn absorb_k(&mut self, item: Item, ks: &[K8]) {
        let w: Vec<u64> = ks.iter().flat_map(|k| k.0).collect();
        self.absorb(item, &w);
    }

    fn address(&mut self, ty: ChalType, stage: Stage, attempt: u64, index: u64) {
        let a = [0xC4A1u64, ty as u64, stage as u64, attempt, index];
        self.t.append_u64_slice(b"", &a);
    }
    fn digits(&mut self, m: usize) -> Vec<u64> {
        // challenge_vector absorbs nothing when the label is empty.
        self.t.challenge_vector(b"", m).iter().map(|f| f.reduce().0).collect()
    }

    pub fn challenge_k(&mut self, stage: Stage, attempt: u64, index: u64) -> K8 {
        self.address(ChalType::K, stage, attempt, index);
        let d = self.digits(8);
        K8([d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7]])
    }

    /// Uniform element of K \ {0, 1}.
    pub fn challenge_k_not01(&mut self, stage: Stage, attempt: u64, index: u64) -> K8 {
        self.address(ChalType::KNot01, stage, attempt, index);
        let d = self.digits(digits_for(512 + RAW_SLACK));
        let qb = BigUint::from(Q);
        let r = d.iter().rev().fold(BigUint::zero(), |a, &x| a * &qb + BigUint::from(x));
        let card = qb.pow(8) - 2u32;
        let mut v = r % card + 2u32; // rank in [2, q^8): skips 0 and 1
        let mut k = [0u64; 8];
        for c in k.iter_mut() {
            *c = (&v % &qb).to_u64().unwrap();
            v /= &qb;
        }
        K8(k)
    }

    /// One query of type K^k (k elements of K, 8 base-q digits each in
    /// coefficient order); the address carries the bundle length.
    pub fn challenge_k_bundle(&mut self, stage: Stage, attempt: u64, k: usize) -> Vec<K8> {
        self.address(ChalType::KBundle, stage, attempt, k as u64);
        self.digits(8 * k).chunks(8).map(|d| K8([d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7]])).collect()
    }

    /// One query of type (C0)^count, C0 = {-2,-1,1,2}^64 (an S element acting
    /// by X = Y^64): the raw rank mod 2^(128 count), two bits per coefficient
    /// in order, element after element. The address carries the count.
    pub fn challenge_c0_tuple(&mut self, stage: Stage, attempt: u64, count: usize) -> Vec<[i8; 64]> {
        self.address(ChalType::C0Tuple, stage, attempt, count as u64);
        let d = self.digits(digits_for(128 * count + RAW_SLACK));
        let r = rank(&d);
        // Little-endian 64-bit limbs of R mod 2^(128 count).
        let limbs = r.to_u64_digits();
        const MAP: [i8; 4] = [-2, -1, 1, 2];
        (0..count).map(|c| {
            let mut out = [0i8; 64];
            for (h, o) in out.iter_mut().enumerate() {
                let bit = 128 * c + 2 * h;
                let w = limbs.get(bit / 64).copied().unwrap_or(0);
                *o = MAP[((w >> (bit % 64)) & 3) as usize];
            }
            out
        }).collect()
    }
}

/// Little-endian base-q rank of `d`. Splits off the largest block of
/// 16 * 2^k low digits (rank = lo + hi * q^len(lo)), so long words cost a
/// few big multiplications instead of a quadratic Horner scan.
fn rank(d: &[u64]) -> BigUint {
    fn go(d: &[u64], pows: &[BigUint]) -> BigUint {
        if d.len() <= 16 {
            let qb = BigUint::from(Q);
            return d.iter().rev().fold(BigUint::zero(), |a, &x| a * &qb + BigUint::from(x));
        }
        let mut k = 0;
        while (16usize << (k + 1)) < d.len() { k += 1; }
        let (lo, hi) = d.split_at(16 << k);
        go(lo, pows) + go(hi, pows) * &pows[k]
    }
    // pows[k] = q^(16 * 2^k)
    let mut pows = vec![BigUint::from(Q).pow(16)];
    while (16usize << pows.len()) < d.len() { let x = &pows[pows.len() - 1] * &pows[pows.len() - 1]; pows.push(x); }
    go(d, &pows)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn challenges_depend_on_prefix_address_and_are_well_formed() {
        let salt = [5u8; 64];
        let mut a = MTranscript::new("p", &salt);
        let mut b = a.clone();
        assert_eq!(a.challenge_k(Stage::LinkRound, 0, 3), b.challenge_k(Stage::LinkRound, 0, 3));
        let mut c = a.clone();
        assert_ne!(a.challenge_k(Stage::LinkRound, 0, 4), c.challenge_k(Stage::LinkRound, 1, 4));
        b.absorb(Item::Other, &[1]);
        let mut d = a.clone();
        d.absorb(Item::Other, &[2]);
        assert_ne!(b.challenge_k(Stage::AuthColumn, 0, 0), d.challenge_k(Stage::AuthColumn, 0, 0));
        // Salt binds.
        let mut e = MTranscript::new("p", &[6u8; 64]);
        let mut f = MTranscript::new("p", &salt);
        assert_ne!(e.challenge_k(Stage::LinkRound, 0, 0), f.challenge_k(Stage::LinkRound, 0, 0));
        // C0 columns: all four values appear, never zero.
        let mut cnt = [0usize; 5];
        for col in a.challenge_c0_tuple(Stage::AuthColumn, 0, 50) { for v in col { cnt[(v + 2) as usize] += 1; } }
        assert_eq!(cnt[2], 0);
        for &i in &[0usize, 1, 3, 4] { assert!(cnt[i] > 650, "{cnt:?}"); }
        // The fast rank equals Horner.
        let d: Vec<u64> = (0..1000u64).map(|i| (i * 0x9E3779B97F4A7C15) % Q).collect();
        let qb = BigUint::from(Q);
        assert_eq!(rank(&d), d.iter().rev().fold(BigUint::zero(), |a, &x| a * &qb + BigUint::from(x)));
        let bun = a.clone().challenge_k_bundle(Stage::LinkInit, 0, 29);
        assert_eq!(bun.len(), 29);
        let k = a.challenge_k_not01(Stage::LinkRound, 0, 9);
        assert!(k != K8::ZERO && k != K8::ONE && k.0.iter().all(|&x| x < Q));
    }
}

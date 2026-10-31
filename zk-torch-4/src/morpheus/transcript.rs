//! Candidate SHAKE256 honest typed-prefix interface, protocol-incompatible with v1.
//! This implements the TO1 *codec*, not an instantiation theorem for SHAKE.
//! No full quantum-oracle simulation, history erasure, or statistical PRG claim.
use super::field8::K8;
use almost_goldilocks_cuda::field::ALMOST_GOLDILOCKS_PRIME as Q;
use num_bigint::BigUint;
use num_traits::{One, ToPrimitive, Zero};
use sha3::digest::{ExtendableOutput, Update, XofReader};
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


pub const RAW_SLACK: usize = 8192;
pub const FS_DOMAIN: &[u8] = b"MORPHEUS/FS/SHAKE256/v2\0";
const MAX_RANK_BITS: u64 = 1 << 30;
// All u64-length byte strings: no smaller arbitrary protocol restriction.
const MAX_ADDRESS_BYTES: u64 = u64::MAX;

/// Literal, injectively framed prefix. It is deliberately not a hash digest.
#[derive(Clone)]
pub struct MTranscript { prefix: Vec<u8> }

fn frame(dst: &mut Vec<u8>, encoding: u8, item: u64, bytes: &[u8]) {
    let total=dst.len().checked_add(17).and_then(|n|n.checked_add(bytes.len())).expect("address length overflow");
    assert!(u64::try_from(total).unwrap() <= MAX_ADDRESS_BYTES, "address cap exceeded");
    dst.push(encoding);
    dst.extend_from_slice(&item.to_be_bytes());
    dst.extend_from_slice(&u64::try_from(bytes.len()).expect("frame length").to_be_bytes());
    dst.extend_from_slice(bytes);
}

/// TO1: first ell MSB-first bits, interpreted as an unsigned integer, modulo N.
/// The output length is derived from the registered domain, never caller-selected.
fn typed_rank(address: &[u8], cardinality: &BigUint) -> BigUint {
    assert!(!cardinality.is_zero());
    let bits = (cardinality - BigUint::one()).bits();
    assert!(bits <= MAX_RANK_BITS, "unregistered typed width");
    let ell = usize::try_from(bits).unwrap() + RAW_SLACK;
    let mut h = sha3::Shake256::default();
    h.update(address);
    let mut bytes = vec![0u8; ell.div_ceil(8)];
    h.finalize_xof().read(&mut bytes);
    let unused = (8 - ell % 8) % 8;
    (BigUint::from_bytes_be(&bytes) >> unused) % cardinality
}

/// Total registered-domain parser. Malformed byte strings still identify distinct
/// SHAKE inputs; only their typed decoding defaults to a singleton. This parser
/// does not validate the protocol statement: that remains the verifier's job.
pub fn registered_cardinality(address: &[u8]) -> BigUint {
    fn parse(address: &[u8]) -> Option<BigUint> {
        if address.len() as u64 > MAX_ADDRESS_BYTES { return None; }
        let mut rest=address.strip_prefix(FS_DOMAIN)?;
        let mut frame_number=0usize;
        let mut pending: Option<BigUint>=None;
        while !rest.is_empty() {
            if rest.len()<17 { return None; }
            let enc=rest[0];
            let item=u64::from_be_bytes(rest[1..9].try_into().ok()?);
            let len=usize::try_from(u64::from_be_bytes(rest[9..17].try_into().ok()?)).ok()?;
            let end=17usize.checked_add(len)?;
            let bytes=rest.get(17..end)?;
            rest=rest.get(end..)?;
            if frame_number==0 {
                if enc!=2 || item!=Item::Profile as u64 { return None; }
            } else if frame_number==1 {
                if enc!=2 || item!=Item::Salt as u64 || len!=64 { return None; }
            } else if let Some(n)=pending.take() {
                if enc!=5 || item!=0 || bytes.is_empty() || (bytes.len()>1 && bytes[0]==0) || BigUint::from_bytes_be(bytes)>=n { return None; }
            } else {
                match enc {
                    1|2|3|6 => {
                        if !matches!(item,1..=11|99) || ((enc==1 || enc==6) && len%8!=0) || (enc==3 && len%64!=0) { return None; }
                        if matches!(enc,1|3) && bytes.chunks_exact(8).any(|b|u64::from_be_bytes(b.try_into().unwrap())>=Q) { return None; }
                    },
                    4 => {
                        if item!=0 || len!=40 { return None; }
                        let mut fields=[0u64;5];
                        for (w,b) in fields.iter_mut().zip(bytes.chunks_exact(8)) { *w=u64::from_be_bytes(b.try_into().ok()?); }
                        let [ty,stage,_attempt,index,arity]=fields;
                        if matches!(ty,4|5) && index!=0 { return None; }
                        if !(1..=8).contains(&stage) { return None; }
                        let n=match ty {
                            1 if arity==1=>BigUint::from(Q).pow(8),
                            2 if arity==1=>BigUint::from(Q).pow(8)-2u32,
                            4=> {
                                let exp=arity.checked_mul(8)?;
                                if exp.checked_mul(64)?>MAX_RANK_BITS { return None; }
                                BigUint::from(Q).pow(u32::try_from(exp).ok()?)
                            },
                            3|5 if ty==5 || arity==1=> {
                                let bits=arity.checked_mul(128)?;
                                if bits>MAX_RANK_BITS { return None; }
                                BigUint::one()<<usize::try_from(bits).ok()?
                            },
                            _=>return None
                        };
                        pending=Some(n);
                    },
                    _=>return None
                }
            }
            frame_number+=1;
        }
        if frame_number<3 { return None; }
        pending
    }
    parse(address).unwrap_or_else(BigUint::one)
}

impl MTranscript {
    pub fn new(profile: &str, salt: &[u8; 64]) -> Self {
        let mut t = Self { prefix: FS_DOMAIN.to_vec() };
        t.absorb_bytes(Item::Profile, profile.as_bytes());
        t.absorb_bytes(Item::Salt, salt);
        t
    }
    /// Complete public prefix for verifier-owned deferred finalization state.
    pub fn literal_prefix(&self) -> &[u8] { &self.prefix }
    /// Canonical field residues, except Statement/Layout which encode arbitrary
    /// public u64 metadata exactly (including q itself), with a separate tag.
    pub fn absorb(&mut self, item: Item, words: &[u64]) {
        let mut bytes = Vec::with_capacity(words.len().checked_mul(8).unwrap());
        let metadata=matches!(item,Item::Statement|Item::Layout);
        for &w in words { bytes.extend_from_slice(&(if metadata {w} else {w % Q}).to_be_bytes()); }
        frame(&mut self.prefix, if metadata {6} else {1}, item as u64, &bytes);
    }
    pub fn absorb_bytes(&mut self, item: Item, bytes: &[u8]) {
        frame(&mut self.prefix, 2, item as u64, bytes);
    }
    pub fn absorb_k(&mut self, item: Item, ks: &[K8]) {
        let words: Vec<u64> = ks.iter().flat_map(|k| k.0).collect();
        let mut bytes = Vec::with_capacity(words.len().checked_mul(8).unwrap());
        for w in words { bytes.extend_from_slice(&(w % Q).to_be_bytes()); }
        frame(&mut self.prefix, 3, item as u64, &bytes);
    }
    fn query(&mut self, ty: ChalType, stage: Stage, attempt: u64, index: u64, arity: usize) -> BigUint {
        let arity64 = u64::try_from(arity).expect("arity overflow");
        let n = match ty {
            ChalType::K => { assert_eq!(arity, 1); BigUint::from(Q).pow(8) },
            ChalType::KNot01 => { assert_eq!(arity, 1); BigUint::from(Q).pow(8) - 2u32 },
            ChalType::KBundle => {
                let exp = arity.checked_mul(8).and_then(|v| u32::try_from(v).ok()).expect("arity overflow");
                assert!(u64::from(exp) * 64 <= MAX_RANK_BITS);
                BigUint::from(Q).pow(exp)
            },
            ChalType::C0 | ChalType::C0Tuple => {
                if matches!(ty, ChalType::C0) { assert_eq!(arity, 1); }
                let bits = arity.checked_mul(128).expect("arity overflow");
                assert!(u64::try_from(bits).unwrap() <= MAX_RANK_BITS);
                BigUint::one() << bits
            }
        };
        let mut a = Vec::with_capacity(40);
        for w in [ty as u64, stage as u64, attempt, index, arity64] { a.extend_from_slice(&w.to_be_bytes()); }
        frame(&mut self.prefix, 4, 0, &a);
        assert!(u64::try_from(self.prefix.len()).unwrap() <= MAX_ADDRESS_BYTES, "address cap exceeded");
        assert_eq!(registered_cardinality(&self.prefix), n, "builder/parser mismatch");
        let r = typed_rank(&self.prefix, &n);
        // Subsequent challenges bind this decoded result and its exact address.
        // Unused raw bits are NEVER read into a future address or transcript.
        frame(&mut self.prefix, 5, 0, &r.to_bytes_be());
        r
    }
    pub fn challenge_k(&mut self, stage: Stage, attempt: u64, index: u64) -> K8 {
        let r = self.query(ChalType::K, stage, attempt, index, 1);
        unrank_k(r, 1).remove(0)
    }
    pub fn challenge_k_not01(&mut self, stage: Stage, attempt: u64, index: u64) -> K8 {
        let r = self.query(ChalType::KNot01, stage, attempt, index, 1) + 2u32;
        unrank_k(r, 1).remove(0)
    }
    pub fn challenge_k_bundle(&mut self, stage: Stage, attempt: u64, k: usize) -> Vec<K8> {
        let r = self.query(ChalType::KBundle, stage, attempt, 0, k);
        unrank_k(r, k)
    }
    pub fn challenge_c0_tuple(&mut self, stage: Stage, attempt: u64, count: usize) -> Vec<[i8; 64]> {
        let mut r = self.query(ChalType::C0Tuple, stage, attempt, 0, count);
        (0..count).map(|_| {
            let mut out = [0i8; 64];
            for v in &mut out { *v = [-2, -1, 1, 2][(&r % 4u32).to_usize().unwrap()]; r >>= 2; }
            out
        }).collect()
    }
}
fn unrank_k(mut r: BigUint, count: usize) -> Vec<K8> {
    let q = BigUint::from(Q);
    (0..count).map(|_| {
        let mut k = [0u64; 8];
        for c in &mut k { *c = (&r % &q).to_u64().unwrap(); r /= &q; }
        K8(k)
    }).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn framed_encodings_cannot_alias() {
        let mut a = MTranscript::new("test", &[0;64]);
        let mut b = a.clone();
        a.absorb(Item::Other, &[1]); b.absorb_bytes(Item::Other, &1u64.to_be_bytes());
        assert_ne!(a.prefix, b.prefix);
        let mut c = a.clone(); let mut d = a.clone();
        c.query(ChalType::K, Stage::LinkRound, 0, 0, 1);
        d.query(ChalType::K, Stage::LinkRound, Q, 0, 1);
        assert_ne!(c.prefix, d.prefix); // Integer addresses are not field-reduced.
    }
    #[test]
    fn singleton_and_non_byte_aligned_codec() {
        assert_eq!(typed_rank(b"kat", &BigUint::one()), BigUint::zero());
        assert_eq!(typed_rank(b"kat", &BigUint::from(257u32)), BigUint::from(176u32));
    }
    #[test]
    fn shake_standard_empty_kat() {
        let mut h = sha3::Shake256::default(); h.update(b"");
        let mut out = [0;32]; h.finalize_xof().read(&mut out);
        assert_eq!(out, [0x46,0xb9,0xdd,0x2b,0x0b,0xa8,0x8d,0x13,0x23,0x3b,0x3f,0xeb,0x74,0x3e,0xeb,0x24,0x3f,0xcd,0x52,0xea,0x62,0xb8,0x1b,0x82,0xb5,0x0c,0x27,0x64,0x6e,0xd5,0x76,0x2f]);
    }
}

#[cfg(test)]
mod candidate_tests {
 use super::*;
 #[test] fn python_chronological_known_answers() {
   let mut t=MTranscript::new("test",&[0;64]);
   assert_eq!(t.challenge_k(Stage::LinkRound,0,0).0,[7797489697268215020, 10771494639500149469, 9585284757969161066, 8519815873561415581, 14459922377744084684, 5078317245461511668, 2955893371482872527, 4638795857135829957]);
   assert_eq!(t.challenge_k_not01(Stage::LinkRound,0,1).0,[12117270624391117241, 900558997749131788, 1659691970300626709, 12516491091674694104, 12860766057750762003, 15388120984586741696, 14910131183502859555, 18118523938667695142]);
 }
 #[test] fn branch_retry_is_addressed_without_rewriting_prior_queries() {
   let t=MTranscript::new("test",&[0;64]);
   let frozen=t.prefix.clone();
   let a=t.clone().challenge_c0_tuple(Stage::AuthColumn,0,2);
   let b=t.clone().challenge_c0_tuple(Stage::AuthColumn,1,2);
   assert_ne!(a,b); assert_eq!(t.prefix,frozen);
   assert_eq!(a,t.clone().challenge_c0_tuple(Stage::AuthColumn,0,2));
 }
 #[test] fn empty_bundles_and_canonical_field_words() {
   let mut t=MTranscript::new("test",&[0;64]);
   assert!(t.challenge_k_bundle(Stage::LinkInit,0,0).is_empty());
   assert!(t.challenge_c0_tuple(Stage::AuthColumn,0,0).is_empty());
   let mut a=MTranscript::new("test",&[0;64]);let mut b=a.clone();
   a.absorb(Item::Other,&[Q]);b.absorb(Item::Other,&[0]);
   assert_eq!(a.prefix,b.prefix);
 }
}

#[cfg(test)] mod parser_tests {
 use super::*;
 fn address(ty:u64, stage:u64, arity:u64)->Vec<u8> {
  let mut t=MTranscript::new("test",&[0;64]);let mut a=vec![];
  for w in [ty,stage,0,0,arity] { a.extend_from_slice(&w.to_be_bytes()); }
  frame(&mut t.prefix,4,0,&a);t.prefix
 }
 #[test] fn total_defaults_and_registered_domains() {
  assert_eq!(registered_cardinality(b"invalid"),BigUint::one());
  assert_eq!(registered_cardinality(&address(1,6,1)),BigUint::from(Q).pow(8));
  assert_eq!(registered_cardinality(&address(2,6,1)),BigUint::from(Q).pow(8)-2u32);
  assert_eq!(registered_cardinality(&address(5,7,3)),BigUint::one()<<384usize);
  for a in [address(1,6,2),address(99,6,1),address(4,6,u64::MAX),address(4,0,1)] {
   assert_eq!(registered_cardinality(&a),BigUint::one());
  }
 }
 #[test] fn malformed_history_cannot_select_nontrivial_domain() {
  let mut t=MTranscript::new("test",&[0;64]);
  frame(&mut t.prefix,5,0,&[0]); // unsolicited output
  let mut a=vec![];for w in [1u64,6,0,0,1] {a.extend_from_slice(&w.to_be_bytes());}
  frame(&mut t.prefix,4,0,&a);
  assert_eq!(registered_cardinality(&t.prefix),BigUint::one());
 }
}

#[cfg(test)] mod metadata_tests {
 use super::*;
 #[test] fn literal_setup_modulus_is_not_zero() {
  let mut a=MTranscript::new("test",&[0;64]);let mut b=a.clone();
  a.absorb(Item::Statement,&[Q]);b.absorb(Item::Statement,&[0]);
  assert_ne!(a.literal_prefix(),b.literal_prefix());
  assert_ne!(a.challenge_k(Stage::LinkInit,0,0),b.challenge_k(Stage::LinkInit,0,0));
 }
}

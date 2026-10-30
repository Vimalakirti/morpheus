//! Private-coin backend (paper theory source `private_coins`, "A bounded
//! one-seed private-coin backend"). Every honest private draw has a
//! canonical address; its bits are the MSB-first expansion of the 512-bit
//! blocks
//!     B_S(a, j) = F(TAG || S || profile || purpose || owner || attempt ||
//!                   coordinate || outer || trial || stream || j),
//! a fixed 106-byte message. Refinement appends bits from the same address
//! and never restarts a stream, so a GPU first pass and a CPU fallback that
//! re-reads the same address see the same prefix.
//!
//! F is SHAKE-256 truncated to 64 bytes, the reference packet's test
//! instantiation. The packet states this is not a proved random-oracle
//! instantiation; swapping F changes only `block`.
//!
//! The packet's profile byte covers PRIMARY64 (0) and causal1024 (1). The
//! ternary candidate is registered here as profile 2 with the same field
//! layout. Address uniqueness is structural (addresses are built from loop
//! indices that never repeat), not a runtime reuse registry, which would not
//! fit 10^8-10^9 draws per opening.

use sha3::digest::{ExtendableOutput, Update, XofReader};

pub const TAG: &[u8; 17] = b"MORPHEUS/COIN/v1\0";
pub const MESSAGE_BYTES: usize = 106;
pub const OPENINGS: u64 = 1 << 20;
pub const ATTEMPTS: u16 = 512;
/// Gaussian outer proposals R and raw polar trials H.
pub const OUTER: u16 = 3072;
pub const TRIALS: u16 = 8192;
pub const POLAR_BITS: usize = 26172;
pub const COIN_BITS: usize = 8708;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum CoinProfile { Primary64 = 0, Causal1024 = 1, Ternary = 2 }

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Purpose {
    Salt = 1,
    SourceSecret = 2,
    SourceError = 3,
    LinkPayload = 4,
    LinkSecret = 5,
    LinkError = 6,
    AuxPayload = 7,
    AuxSecret = 8,
    AuxError = 9,
    PolarX = 10,
    PolarY = 11,
    Correction = 12,
    ProtocolCoin = 13,
    GaussianCompat = 14,
    /// Graph-mask bank W_G (not in the packet's table; registered here as
    /// 15 with the auxiliary-payload layout: opening owner, coordinate =
    /// slot·8 + residue, uniform-q rejection trials).
    GraphMask = 15,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Address {
    pub purpose: Purpose,
    pub owner: u64,
    pub attempt: u16,
    pub coordinate: u32,
    pub outer: u16,
    pub trial: u16,
    pub stream: u32,
}

impl Address {
    pub fn new(purpose: Purpose, owner: u64) -> Self {
        Address { purpose, owner, attempt: 0, coordinate: 0, outer: 0, trial: 0, stream: 0 }
    }
    /// Bit cap of the stream at this address.
    pub fn cap(&self) -> usize {
        match self.purpose {
            Purpose::Salt => 512,
            Purpose::LinkPayload | Purpose::AuxPayload | Purpose::GraphMask => 64,
            Purpose::PolarX | Purpose::PolarY | Purpose::GaussianCompat => POLAR_BITS,
            Purpose::Correction | Purpose::ProtocolCoin => COIN_BITS,
            // Hiding secrets/errors: six bits for PRIMARY64; the ternary
            // sampler reads two-bit proposals, up to 8192 of them.
            _ => 2 * super::sampler::TERNARY_PROPOSALS,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CoinsExhausted;

/// The fixed 106-byte message for block `j` of address `a`.
pub fn encode(profile: CoinProfile, seed: &[u8; 64], a: &Address, j: u8) -> [u8; MESSAGE_BYTES] {
    let mut m = [0u8; MESSAGE_BYTES];
    let mut o = 0;
    let mut put = |b: &[u8]| { m[o..o + b.len()].copy_from_slice(b); o += b.len(); };
    put(TAG);
    put(seed);
    put(&[profile as u8, a.purpose as u8]);
    put(&a.owner.to_be_bytes());
    put(&a.attempt.to_be_bytes());
    put(&a.coordinate.to_be_bytes());
    put(&a.outer.to_be_bytes());
    put(&a.trial.to_be_bytes());
    put(&a.stream.to_be_bytes());
    put(&[j]);
    debug_assert_eq!(o, MESSAGE_BYTES);
    m
}

/// F: one 512-bit block.
pub fn block(msg: &[u8; MESSAGE_BYTES]) -> [u8; 64] {
    let mut h = sha3::Shake256::default();
    h.update(msg);
    let mut out = [0u8; 64];
    h.finalize_xof().read(&mut out);
    out
}

/// An append-only bit stream.
pub trait BitStream {
    fn next_bit(&mut self) -> Result<u8, CoinsExhausted>;
}

/// The private seed and profile; hands out address streams.
#[derive(Clone)]
pub struct Coins {
    seed: [u8; 64],
    pub profile: CoinProfile,
    /// Openings begun under this seed (shared by clones); at most J_o.
    opened: std::sync::Arc<std::sync::atomic::AtomicU64>,
}

impl Coins {
    pub fn new(seed: [u8; 64], profile: CoinProfile) -> Self { Coins { seed, profile, opened: Default::default() } }
    /// Start opening `opening` (request or finalization): counts it against
    /// the lifetime cap J_o = 2^20 of this seed and returns its salt.
    pub fn begin_opening(&self, opening: u64) -> [u8; 64] {
        let k = self.opened.fetch_add(1, std::sync::atomic::Ordering::SeqCst) + 1;
        assert!(k <= OPENINGS, "seed exhausted: opening {k} exceeds J_o = {OPENINGS}");
        self.salt(opening)
    }
    /// A fresh seed from the OS.
    pub fn from_entropy(profile: CoinProfile) -> Self {
        use rand::RngCore;
        let mut seed = [0u8; 64];
        rand::rngs::OsRng.fill_bytes(&mut seed);
        Coins::new(seed, profile)
    }
    pub fn stream(&self, a: Address) -> Prefix {
        Prefix { coins: self.clone(), cap: a.cap(), address: a, bits: 0, idx: usize::MAX, blk: [0; 64] }
    }
    pub fn block(&self, a: &Address, j: u8) -> [u8; 64] { block(&encode(self.profile, &self.seed, a, j)) }
    /// Keccak rate lanes of the padded Gaussian coin message for (opening,
    /// attempt) with purpose, coordinate, outer and trial zeroed; the GPU
    /// sampler XORs those fields in per thread.
    pub fn gaussian_base_lanes(&self, opening: u64, attempt: u16) -> [u64; 17] {
        let mut a = Address::new(Purpose::PolarX, opening);
        a.attempt = attempt;
        let mut m = [0u8; 136];
        m[..MESSAGE_BYTES].copy_from_slice(&encode(self.profile, &self.seed, &a, 0));
        m[82] = 0;
        m[MESSAGE_BYTES] ^= 0x1f;
        m[135] ^= 0x80;
        let mut l = [0u64; 17];
        for i in 0..17 { l[i] = u64::from_le_bytes(m[8 * i..8 * i + 8].try_into().unwrap()); }
        l
    }
    /// Public salt for an opening (512 bits at its own index).
    pub fn salt(&self, opening: u64) -> [u8; 64] { self.block(&Address::new(Purpose::Salt, opening), 0) }
}

/// MSB-first prefix of one address stream.
pub struct Prefix {
    coins: Coins,
    pub address: Address,
    cap: usize,
    pub bits: usize,
    idx: usize,
    blk: [u8; 64],
}

impl BitStream for Prefix {
    fn next_bit(&mut self) -> Result<u8, CoinsExhausted> {
        if self.bits >= self.cap { return Err(CoinsExhausted); }
        let (j, off) = (self.bits / 512, self.bits % 512);
        if j != self.idx {
            self.blk = self.coins.block(&self.address, j as u8);
            self.idx = j;
        }
        self.bits += 1;
        Ok((self.blk[off / 8] >> (7 - off % 8)) & 1)
    }
}

/// Fixture streams for tests: stream i is SHAKE-256(label || i) read
/// MSB-first. Not a private-coin backend.
pub struct FixtureStream { reader: sha3::Shake256Reader, byte: u8, left: u32 }

impl FixtureStream {
    pub fn new(label: &[u8], i: u64) -> Self {
        let mut h = sha3::Shake256::default();
        h.update(label);
        h.update(&i.to_be_bytes());
        FixtureStream { reader: h.finalize_xof(), byte: 0, left: 0 }
    }
}

impl BitStream for FixtureStream {
    fn next_bit(&mut self) -> Result<u8, CoinsExhausted> {
        if self.left == 0 {
            let mut b = [0u8; 1];
            self.reader.read(&mut b);
            self.byte = b[0];
            self.left = 8;
        }
        self.left -= 1;
        Ok((self.byte >> self.left) & 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn codec_is_fixed_width_and_streams_are_prefix_stable() {
        let c = Coins::new([7u8; 64], CoinProfile::Ternary);
        let mut a = Address::new(Purpose::PolarX, 3);
        a.attempt = 1;
        a.coordinate = 99;
        a.outer = 2;
        a.trial = 5;
        let m = encode(c.profile, &[7u8; 64], &a, 1);
        assert_eq!(m.len(), 106);
        assert_eq!(&m[..17], TAG);
        assert_eq!(m[81], 2);
        assert_eq!(m[82], 10);
        // Bits across a block boundary, and a re-read of the same address.
        let mut s = c.stream(a);
        let v: Vec<u8> = (0..1100).map(|_| s.next_bit().unwrap()).collect();
        let mut s2 = c.stream(a);
        let w: Vec<u8> = (0..1100).map(|_| s2.next_bit().unwrap()).collect();
        assert_eq!(v, w);
        let b1 = c.block(&a, 1);
        for t in 0..512 { assert_eq!(v[512 + t], (b1[t / 8] >> (7 - t % 8)) & 1); }
        // Different trial, different stream.
        let mut b = a;
        b.trial = 6;
        let mut s3 = c.stream(b);
        let x: Vec<u8> = (0..256).map(|_| s3.next_bit().unwrap()).collect();
        assert_ne!(&v[..256], &x[..]);
        // Cap.
        let mut s4 = c.stream(Address::new(Purpose::LinkPayload, 0));
        for _ in 0..64 { s4.next_bit().unwrap(); }
        assert!(s4.next_bit().is_err());
    }
}

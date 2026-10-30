//! PRG-expanded uniform keys over R (user decision: keys expanded from a
//! public seed rather than stored). The block function is the ChaCha8 core of
//! `cuda_almost_goldilocks/ajtai_chacha8.cuh`, ported word for word so CPU and
//! GPU expand identical keys.
//!
//! Key element A_kind[row][col] (one degree-4096 ring element) is drawn from
//! 512 ChaCha8 blocks, each giving 8 rejection-sampled coefficients < q:
//!   nonce   = ((kind << 16) | row, col_lo32, col_hi32)
//!   counter = block + 512 * retry   (retry only on a rejected word)
//! Columns are laid out [payload | hiding | identity]; the identity block is
//! not random (A[row][payload + hiding + i] = δ_{row,i}).

use super::ring::N;
use almost_goldilocks_cuda::field::ALMOST_GOLDILOCKS_PRIME as Q;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum KeyKind { Source = 1, Link = 2, Aux = 3 }

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KeySeed(pub [u32; 8]);

#[inline(always)]
fn qr(x: &mut [u32; 16], a: usize, b: usize, c: usize, d: usize) {
    x[a] = x[a].wrapping_add(x[b]); x[d] ^= x[a]; x[d] = x[d].rotate_left(16);
    x[c] = x[c].wrapping_add(x[d]); x[b] ^= x[c]; x[b] = x[b].rotate_left(12);
    x[a] = x[a].wrapping_add(x[b]); x[d] ^= x[a]; x[d] = x[d].rotate_left(8);
    x[c] = x[c].wrapping_add(x[d]); x[b] ^= x[c]; x[b] = x[b].rotate_left(7);
}

/// One ChaCha8 block (identical to `chacha8_block` in ajtai_chacha8.cuh).
pub fn chacha8_block(key: &[u32; 8], counter: u32, nonce: &[u32; 3]) -> [u32; 16] {
    let s: [u32; 16] = [
        0x61707865, 0x3320646e, 0x79622d32, 0x6b206574,
        key[0], key[1], key[2], key[3], key[4], key[5], key[6], key[7],
        counter, nonce[0], nonce[1], nonce[2],
    ];
    let mut x = s;
    for _ in 0..4 {
        qr(&mut x, 0, 4, 8, 12); qr(&mut x, 1, 5, 9, 13);
        qr(&mut x, 2, 6, 10, 14); qr(&mut x, 3, 7, 11, 15);
        qr(&mut x, 0, 5, 10, 15); qr(&mut x, 1, 6, 11, 12);
        qr(&mut x, 2, 7, 8, 13); qr(&mut x, 3, 4, 9, 14);
    }
    let mut out = [0u32; 16];
    for i in 0..16 { out[i] = x[i].wrapping_add(s[i]); }
    out
}

/// 8 uniform coefficients in [0, q) for block `b` of key element (kind, row, col).
pub fn key_block(seed: &KeySeed, kind: KeyKind, row: u32, col: u64, b: u32) -> [u64; 8] {
    let nonce = [((kind as u32) << 16) | row, col as u32, (col >> 32) as u32];
    let mut out = [0u64; 8];
    let mut written = 0;
    for retry in 0..16u32 {
        let w = chacha8_block(&seed.0, b + 512 * retry, &nonce);
        for k in 0..8 {
            let v = (w[2 * k] as u64) | ((w[2 * k + 1] as u64) << 32);
            if v < Q { out[written] = v; written += 1; if written == 8 { return out; } }
        }
    }
    // Unreachable in practice (each word rejects with probability ~2^-31);
    // unfilled slots stay zero, matching the CUDA reference's fallback.
    out
}

/// Full key ring element A_kind[row][col] for a RANDOM column, canonical.
pub fn key_element(seed: &KeySeed, kind: KeyKind, row: u32, col: u64) -> Vec<u64> {
    let mut v = vec![0u64; N];
    for b in 0..(N / 8) as u32 {
        let blk = key_block(seed, kind, row, col, b);
        v[(b as usize) * 8..(b as usize) * 8 + 8].copy_from_slice(&blk);
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand_chacha::rand_core::{RngCore, SeedableRng};

    #[test]
    fn block_matches_an_independent_chacha8() {
        // rand_chacha's ChaCha8 uses the djb layout (64-bit counter in words
        // 12-13, stream in 14-15); counter | nonce0 << 32 and
        // stream = nonce1 | nonce2 << 32 reproduce our IETF-style block.
        let key = [0x03020100u32, 0x07060504, 0x0b0a0908, 0x0f0e0d0c, 0x13121110, 0x17161514, 0x1b1a1918, 0x1f1e1d1c];
        let mut kb = [0u8; 32];
        for i in 0..8 { kb[4 * i..4 * i + 4].copy_from_slice(&key[i].to_le_bytes()); }
        for &(ctr, nonce) in &[(0u32, [0u32, 0, 0]), (7, [1, 2, 3]), (513, [0x10005, 0xdead, 0xbeef])] {
            let mine = chacha8_block(&key, ctr, &nonce);
            let mut rng = rand_chacha::ChaCha8Rng::from_seed(kb);
            rng.set_stream((nonce[1] as u64) | ((nonce[2] as u64) << 32));
            rng.set_word_pos((((ctr as u64) | ((nonce[0] as u64) << 32)) as u128) * 16);
            let mut theirs = [0u32; 16];
            for t in theirs.iter_mut() { *t = rng.next_u32(); }
            assert_eq!(mine, theirs, "ctr={ctr} nonce={nonce:?}");
        }
    }

    #[test]
    fn key_elements_are_canonical_deterministic_and_separated() {
        let seed = KeySeed([1, 2, 3, 4, 5, 6, 7, 8]);
        let a = key_element(&seed, KeyKind::Source, 0, 5);
        assert_eq!(a, key_element(&seed, KeyKind::Source, 0, 5));
        assert!(a.iter().all(|&v| v < Q));
        assert_ne!(a, key_element(&seed, KeyKind::Link, 0, 5));
        assert_ne!(a, key_element(&seed, KeyKind::Source, 1, 5));
        assert_ne!(a, key_element(&seed, KeyKind::Source, 0, 6));
        // Crude uniformity: the top bit is set about half the time.
        let hi = a.iter().filter(|&&v| v >> 63 == 1).count();
        assert!((1700..2400).contains(&hi), "top-bit count {hi}");
    }
}

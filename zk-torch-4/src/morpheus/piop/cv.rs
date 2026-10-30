//! Convolution-family and spatial index-map reductions of the masked graph
//! PIOP (vision and audio front ends).
//!
//! Layout. These blocks store [C, (D,) H, W] tensors padded per axis with
//! the LAST axis in the lowest index bits (`x[c,h,w]` at `w + Wp(h + Hp c)`).
//!
//! Index maps (ZeroPad*, SubSample2D, Replicate2x2, ChannelSlice) read
//! Y[.., o_d, ..] = X[.., i_d, ..] along mapped axes d, with a relation
//! R_d ∋ (o, i). A claim Ỹ(r) = z becomes the masked sumcheck
//!   z = Σ_i X̃(r_fixed, i) · Π_d T_d(i_d),  T_d(i) = Σ_{(o,i)∈R_d} eq(r_o, o),
//! over the mapped input bits. T_d is public: the verifier builds it from
//! R_d (|R_d| ≤ the axis length) and evaluates its MLE.
//!
//! Convolutions (Conv1D/2D/3D, DepthwiseConv2D, ConvTranspose2D/3D) relate
//! output o, kernel tap k and input i per spatial axis by R_d ∋ (o, k, i)
//! (i = s·o + δ·k, or o = s·i + k when transposed). Two masked sumchecks:
//!   1. z = Σ_{k,c} W̃(.., k, c) · A(k, c),  A(k, c) = Σ_o Π_d eq(r_o, o_d) X(c, i(o, k)),
//!      over the kernel taps and input channels (an eq factor for the shared
//!      channel of a depthwise conv);
//!   2. the claim on A at (x_k, x_c) is the index map above with
//!      T_d(i) = Σ_{(o,k,i)∈R_d} eq(r_o, o)·eq(x_k, k).
//! The weight claim is on the unflattened kernel (FlattenKernel /
//! ChangeShape producers are looked through), so no flattened kernel or
//! full-convolution advice is committed.

use super::*;

fn lb(n: usize) -> usize { log2_ceil(n.max(1)) }

/// MORPHEUS_CONV=pp: prove Conv2D with the polynomial-product construction
/// and a committed full convolution (ablation baseline).
fn kpow(a: &K8, mut e: u64) -> K8 { let (mut r, mut b) = (K8::ONE, *a); while e > 0 { if e & 1 == 1 { r = r.mul(&b); } b = b.mul(&b); e >>= 1; } r }

pub(super) fn conv_pp() -> bool { std::env::var("MORPHEUS_CONV").ok().as_deref() == Some("pp") }

/// One mapped axis: relation triples (o, k, i) and bit widths.
#[derive(Clone, Debug)]
pub(super) struct Rel { pub ob: usize, pub kb: usize, pub ib: usize, pub t: Vec<(usize, usize, usize)> }

impl Rel {
    fn map(ob: usize, ib: usize, pairs: impl Iterator<Item = (usize, usize)>) -> Self {
        Rel { ob, kb: 0, ib, t: pairs.map(|(o, i)| (o, 0, i)).collect() }
    }
    /// T(i) = Σ eq(r_o, o)·eq(r_k, k) over the triples, dense over i.
    fn table(&self, ro: &[K8], rk: &[K8]) -> Vec<K8> {
        let (eo, ek) = (eq_table(ro), eq_table(rk));
        let mut t = vec![K8::ZERO; 1usize << self.ib];
        for &(o, k, i) in &self.t { t[i] = t[i].add(&eo[o].mul(&ek[k])); }
        t
    }
    /// Sparse kernel-contraction matrix M[k][i] = Σ_o eq(r_o, o).
    fn kmat(&self, ro: &[K8]) -> Vec<Vec<(usize, K8)>> {
        let eo = eq_table(ro);
        let mut m: Vec<HashMap<usize, K8>> = vec![HashMap::new(); 1usize << self.kb];
        for &(o, k, i) in &self.t { let e = m[k].entry(i).or_insert(K8::ZERO); *e = e.add(&eo[o]); }
        m.into_iter().map(|h| { let mut v: Vec<(usize, K8)> = h.into_iter().collect(); v.sort_by_key(|x| x.0); v }).collect()
    }
}

/// A mapped axis of the input: its first variable and relation, with the
/// output point r_o (and kernel point r_k).
pub(super) struct MapAxis { pub off: usize, pub rel: Rel, pub ro: Vec<K8>, pub rk: Vec<K8> }

/// MLE of the indicator [c < n] over Boolean c (lowest bit first) at x.
fn prefix_mle(n: usize, x: &[K8]) -> K8 {
    if n >= 1usize << x.len() { return K8::ONE; }
    // Σ_{c<n} eq(x, c): walk the bits of n from the top.
    let (mut acc, mut pre) = (K8::ZERO, K8::ONE);
    for b in (0..x.len()).rev() {
        let one = x[b];
        let zero = K8::ONE.sub(&one);
        if (n >> b) & 1 == 1 {
            // c agrees with n above b and has 0 at b: every lower suffix counts.
            acc = acc.add(&pre.mul(&zero));
            pre = pre.mul(&one);
        } else {
            pre = pre.mul(&zero);
        }
    }
    acc
}

/// MLE at y of the table c ↦ eq(r, c)·[c < n]: Σ_{c<n} eq(r, c)·eq(y, c).
fn masked_eq(r: &[K8], y: &[K8], n: usize) -> K8 {
    if n >= 1usize << r.len() { return eq_pt(r, y); }
    let (er, ey) = (eq_table(r), eq_table(y));
    (0..n).fold(K8::ZERO, |a, c| a.add(&er[c].mul(&ey[c])))
}

/// Conv kinds by weight layout.
#[derive(Clone, Copy, PartialEq, Debug)]
enum WKind { Normal, Transpose, Depthwise }

impl<'g, 'a> Walk<'g, 'a> {
    /// Index-map reduction of a claim (z0, η0) on X̃(fixed, mapped) where
    /// `fixed` assigns every unmapped variable of X.
    pub(super) fn map_reduce(&mut self, ctx: &mut Ctx, x: usize, mut fixed: Vec<(usize, K8)>, axes: Vec<MapAxis>, z0: K8, eta0: &Lin) {
        // Trivial axes become fixed variables: an identity over the whole
        // padded range (T = eq(r_o, ·)), or a map onto one input index i0
        // (T = c·[i = i0], the scale c folded into the claim).
        let mut scale = K8::ONE;
        let mut rest = Vec::new();
        for a in axes {
            let full_id = a.rel.kb == 0 && a.rel.ob == a.rel.ib && a.rel.t.len() == 1usize << a.rel.ib
                && a.rel.t.iter().all(|&(o, _, i)| o == i) && { let mut seen = vec![false; 1usize << a.rel.ib]; a.rel.t.iter().all(|&(_, _, i)| !std::mem::replace(&mut seen[i], true)) };
            let i0 = a.rel.t.first().map(|t| t.2);
            if full_id {
                for b in 0..a.rel.ib { fixed.push((a.off + b, a.ro[b])); }
            } else if i0.is_some() && a.rel.t.iter().all(|t| t.2 == i0.unwrap()) && a.rel.table(&a.ro, &a.rk)[i0.unwrap()] != K8::ZERO {
                // (A zero scale cannot be divided out; such an axis takes the
                // sumcheck path below.)
                let c = a.rel.table(&a.ro, &a.rk)[i0.unwrap()];
                scale = scale.mul(&c);
                for b in 0..a.rel.ib { fixed.push((a.off + b, if (i0.unwrap() >> b) & 1 == 1 { K8::ONE } else { K8::ZERO })); }
            } else {
                rest.push(a);
            }
        }
        let mut axes = rest;
        let inv = scale.inv();
        let (z0, eta0) = (z0.mul(&inv), lin_scaled(eta0, &inv));
        let nx = self.g.nvars[x];
        if axes.is_empty() {
            assert_eq!(fixed.len(), nx, "index map must assign every variable of edge {x}");
            let mut point = vec![K8::ZERO; nx];
            for &(v, c) in &fixed { point[v] = c; }
            self.push(ctx, x, AClaim { point, pad: eta0, z: z0 });
            return;
        }
        let eta0 = &eta0;
        axes.sort_by_key(|a| a.off);
        let n: usize = axes.iter().map(|a| a.rel.ib).sum();
        assert_eq!(n + fixed.len(), nx, "index map must assign every variable of edge {x}");
        let tabs: Vec<Vec<K8>> = axes.iter().map(|a| a.rel.table(&a.ro, &a.rk)).collect();
        let bits: Vec<usize> = axes.iter().map(|a| a.rel.ib).collect();
        let tables = if ctx.prover {
            // Public factor Π_d T_d over the mapped bits (earlier axes low).
            let offs: Vec<usize> = bits.iter().scan(0, |acc, &b| { let o = *acc; *acc += b; Some(o) }).collect();
            let p: Vec<K8> = (0..1usize << n).into_par_iter().with_min_len(1 << 12).map(|x| {
                tabs.iter().zip(&offs).zip(&bits).fold(K8::ONE, |acc, ((t, &o), &b)| acc.mul(&t[(x >> o) & ((1usize << b) - 1)]))
            }).collect();
            let pd = self.pd.unwrap();
            if gpu::enabled() && (1usize << n) >= gpu::threshold() {
                let xp = gpu::partial_eval(&pd.dev(x), nx, &fixed);
                Some(Tables::Dev(vec![gpu::KTab::from_host(&p), xp]))
            } else {
                Some(Tables::Host(vec![p, partial_eval(pd.v(x), nx, &fixed)]))
            }
        } else { None };
        if ctx.prover && selfcheck() {
            if let Some(Tables::Host(t)) = tables.as_ref() {
                let sum = t[0].iter().zip(&t[1]).fold(K8::ZERO, |a, (p, q)| a.add(&p.mul(q)));
                let want = z0.sub(&ctx.lin_val(eta0));
                eprintln!("[selfcheck] map_reduce edge {x}: n {n} fixed {} identity {}", fixed.len(), if sum == want { "ok" } else { "FALSE" });
            }
        }
        let out = ctx.masked_sc(n, 1, true, z0, eta0, tables, &|xs: &[K8]| {
            let mut off = 0;
            tabs.iter().zip(&bits).fold(K8::ONE, |acc, (t, &b)| { let v = tensor::mle_k(t, &xs[off..off + b]); off += b; acc.mul(&v) })
        });
        let rr = out.rho.mul(&out.rho.sub(&K8::ONE));
        let mut point = vec![K8::ZERO; nx];
        for &(v, c) in &fixed { point[v] = c; }
        let mut j = 0;
        for a in &axes { for b in 0..a.rel.ib { point[a.off + b] = out.x[j]; j += 1; } }
        self.push(ctx, x, AClaim { point, pad: lin1(out.pads[0], rr), z: out.v[0] });
    }

    /// The channel axis of an index map: fixed to the output point when the
    /// channel count is a power of two, else an identity map over the real
    /// channels (padded input channels need not be zero).
    fn chan_axis(channels: usize, xoff: usize, nc: usize, rc: &[K8], fixed: &mut Vec<(usize, K8)>, axes: &mut Vec<MapAxis>) {
        assert_eq!(rc.len(), nc);
        if channels.is_power_of_two() || nc == 0 {
            for j in 0..nc { fixed.push((xoff + j, rc[j])); }
        } else {
            axes.push(MapAxis { off: xoff, rel: Rel::map(nc, nc, (0..channels).map(|c| (c, c))), ro: rc.to_vec(), rk: vec![] });
        }
    }

    /// Variable offsets of a block-layout tensor (last axis lowest).
    fn offs(shape: &[usize]) -> Vec<usize> {
        let mut o = vec![0; shape.len()];
        let mut acc = 0;
        for d in (0..shape.len()).rev() { o[d] = acc; acc += lb(shape[d]); }
        o
    }

    pub(super) fn reduce_cv(&mut self, ctx: &mut Ctx, node: usize, c: AClaim) -> Option<&'static str> {
        let nd = self.g.dag.nodes[node].clone();
        let (x, y) = (nd.inputs[0], nd.outputs[0]);
        let r = &c.point;
        match &nd.kind {
            BasicBlockType::ZeroPad(p) => {
                let (hp, wp) = (p.input_h + 2 * p.pad_h, p.input_w + 2 * p.pad_w);
                self.pad_like(ctx, x, y, p.channels, &[p.input_h, p.input_w], &[hp, wp], &[p.pad_h, p.pad_w], c);
                Some("pad")
            }
            BasicBlockType::ZeroPadAsym(p) => {
                let (hp, wp) = (p.input_h + p.pad_h_top + p.pad_h_bottom, p.input_w + p.pad_w_left + p.pad_w_right);
                self.pad_like(ctx, x, y, p.channels, &[p.input_h, p.input_w], &[hp, wp], &[p.pad_h_top, p.pad_w_left], c);
                Some("pad")
            }
            BasicBlockType::ZeroPad3D(p) => {
                let o = [p.input_d + 2 * p.pad_d, p.input_h + 2 * p.pad_h, p.input_w + 2 * p.pad_w];
                self.pad_like(ctx, x, y, p.channels, &[p.input_d, p.input_h, p.input_w], &o, &[p.pad_d, p.pad_h, p.pad_w], c);
                Some("pad")
            }
            BasicBlockType::SubSample2D(s) => {
                let (lwo, lho) = (lb(s.out_w), lb(s.out_h));
                let (lw, lh) = (lb(s.input_w), lb(s.input_h));
                let nc = self.g.nvars[x] - lw - lh;
                assert_eq!(self.g.nvars[y], lwo + lho + nc, "SubSample2D channel bits");
                let rw = Rel::map(lwo, lw, (0..s.out_w).map(|o| (o, o * s.stride_w + s.offset_w)).filter(|&(_, i)| i < s.input_w));
                let rh = Rel::map(lho, lh, (0..s.out_h).map(|o| (o, o * s.stride_h + s.offset_h)).filter(|&(_, i)| i < s.input_h));
                let mut axes = vec![MapAxis { off: 0, rel: rw, ro: r[..lwo].to_vec(), rk: vec![] }, MapAxis { off: lw, rel: rh, ro: r[lwo..lwo + lho].to_vec(), rk: vec![] }];
                let mut fixed = Vec::new();
                Self::chan_axis(s.channels, lw + lh, nc, &r[lwo + lho..], &mut fixed, &mut axes);
                self.map_reduce(ctx, x, fixed, axes, c.z, &c.pad);
                Some("subsample")
            }
            BasicBlockType::Replicate2x2(rp) => {
                let (lwo, lho) = (lb(rp.out_w), lb(rp.out_h));
                let (iw, ih) = (rp.out_w / 2, rp.out_h / 2);
                let (lw, lh) = (lb(iw), lb(ih));
                let nc = self.g.nvars[x] - lw - lh;
                assert_eq!(self.g.nvars[y], lwo + lho + nc, "Replicate2x2 channel bits");
                let rw = Rel::map(lwo, lw, (0..rp.out_w).map(|o| (o, o / 2)).filter(|&(_, i)| i < iw));
                let rh = Rel::map(lho, lh, (0..rp.out_h).map(|o| (o, o / 2)).filter(|&(_, i)| i < ih));
                let mut axes = vec![MapAxis { off: 0, rel: rw, ro: r[..lwo].to_vec(), rk: vec![] }, MapAxis { off: lw, rel: rh, ro: r[lwo..lwo + lho].to_vec(), rk: vec![] }];
                let mut fixed = Vec::new();
                Self::chan_axis(rp.channels, lw + lh, nc, &r[lwo + lho..], &mut fixed, &mut axes);
                self.map_reduce(ctx, x, fixed, axes, c.z, &c.pad);
                Some("replicate")
            }
            BasicBlockType::ChannelSlice(cs) => {
                let s: usize = cs.spatial_dims.iter().map(|&d| lb(d)).sum();
                let (lci, lco) = (lb(cs.channels_in), lb(cs.channels_out));
                assert_eq!(self.g.nvars[x], s + lci);
                assert_eq!(self.g.nvars[y], s + lco);
                let fixed: Vec<(usize, K8)> = (0..s).map(|j| (j, r[j])).collect();
                let rel = Rel::map(lco, lci, (0..cs.channels_out).map(|o| (o, o + cs.channel_start)));
                self.map_reduce(ctx, x, fixed, vec![MapAxis { off: s, rel, ro: r[s..].to_vec(), rk: vec![] }], c.z, &c.pad);
                Some("slice")
            }
            BasicBlockType::Concat(cc) => {
                let b = nd.inputs[1];
                assert!(cc.channels_a.is_power_of_two(), "Concat needs a power-of-two first operand");
                let na = self.g.nvars[x];
                assert_eq!(self.g.nvars[b], na);
                assert_eq!(self.g.nvars[y], na + 1);
                let t = r[na];
                let p = r[..na].to_vec();
                let (sa, va) = self.child_lin(ctx, x, &p);
                let (sb, vb) = self.child_lin(ctx, b, &p);
                // z0 − (1−t) v_a − t v_b = (η0 − (1−t) a_a − t a_b)(W_G).
                let omt = K8::ONE.sub(&t);
                let mut l = c.pad.clone();
                lin_add(&mut l, &lin1(sa, K8::ONE), &omt.neg());
                lin_add(&mut l, &lin1(sb, K8::ONE), &t.neg());
                ctx.row(vec![], &l, c.z.sub(&omt.mul(&va)).sub(&t.mul(&vb)));
                self.push(ctx, x, AClaim { point: p.clone(), pad: lin1(sa, K8::ONE), z: va });
                self.push(ctx, b, AClaim { point: p, pad: lin1(sb, K8::ONE), z: vb });
                Some("concat")
            }
            BasicBlockType::Conv1D(cv) => {
                let w = nd.inputs[1];
                let rel = |_| Rel { ob: lb(cv.l_out), kb: lb(cv.kernel_size), ib: lb(cv.input_len), t: trip_conv(cv.l_out, cv.kernel_size, cv.input_len, cv.conv_stride, 1) };
                self.conv(ctx, x, w, &[cv.c_out, cv.c_in, cv.kernel_size], vec![rel(0)], WKind::Normal, lb(cv.c_in), lb(cv.c_out), c);
                Some("conv")
            }
            BasicBlockType::Conv2D(cv) if conv_pp() && nd.outputs.len() > 1 => {
                assert_eq!(cv.batch.max(1), 1, "Morpheus conv: batch 1");
                self.conv2d_pp(ctx, &nd, cv, c);
                Some("conv")
            }
            BasicBlockType::FlattenKernel(fk) => {
                // W_flat[d, c, kh·s_w + kw] = W[d, c, kh, kw] (reached only by
                // the polynomial-product conv); W_flat's index splits into the
                // kw field (low log2 s_w bits) and the kh field.
                assert!(fk.dilation_h == 1 && fk.dilation_w == 1, "FlattenKernel reduction: no dilation");
                let (lkw, lkh) = (lb(fk.kw), lb(fk.kh));
                let lsw = lb(fk.s_w);
                let lk = self.g.nvars[y] - lb(fk.c_in) - lb(fk.c_out);
                let (ow, oh) = (lsw.min(lk), lk.saturating_sub(lsw));
                let fixed: Vec<(usize, K8)> = (0..lb(fk.c_in) + lb(fk.c_out)).map(|j| (lkw + lkh + j, r[lk + j])).collect();
                let axes = vec![
                    MapAxis { off: 0, rel: Rel::map(ow, lkw, (0..fk.kw).map(|k| (k, k))), ro: r[..ow].to_vec(), rk: vec![] },
                    MapAxis { off: lkw, rel: Rel::map(oh, lkh, (0..fk.kh).map(|k| (k, k))), ro: r[lsw.min(lk)..lk].to_vec(), rk: vec![] },
                ];
                self.map_reduce(ctx, x, fixed, axes, c.z, &c.pad);
                Some("flatten")
            }
            BasicBlockType::Conv2D(cv) => {
                assert_eq!(cv.batch.max(1), 1, "Morpheus conv: batch 1");
                let w = self.unflatten(nd.inputs[1]);
                let rw = Rel { ob: lb(cv.w_out), kb: lb(cv.kernel_w), ib: lb(cv.input_w), t: trip_conv(cv.w_out, cv.kernel_w, cv.input_w, cv.conv_stride_w, cv.dilation_w) };
                let rh = Rel { ob: lb(cv.h_out), kb: lb(cv.kernel_h), ib: lb(cv.input_h), t: trip_conv(cv.h_out, cv.kernel_h, cv.input_h, cv.conv_stride_h, cv.dilation_h) };
                self.conv(ctx, x, w, &[cv.c_out, cv.c_in, cv.kernel_h, cv.kernel_w], vec![rw, rh], WKind::Normal, lb(cv.c_in), lb(cv.c_out), c);
                Some("conv")
            }
            BasicBlockType::Conv3D(cv) => {
                let w = self.unflatten(nd.inputs[1]);
                let rw = Rel { ob: lb(cv.w_out), kb: lb(cv.kernel_w), ib: lb(cv.input_w), t: trip_conv(cv.w_out, cv.kernel_w, cv.input_w, cv.conv_stride_w, 1) };
                let rh = Rel { ob: lb(cv.h_out), kb: lb(cv.kernel_h), ib: lb(cv.input_h), t: trip_conv(cv.h_out, cv.kernel_h, cv.input_h, cv.conv_stride_h, 1) };
                let rd = Rel { ob: lb(cv.d_out), kb: lb(cv.kernel_d), ib: lb(cv.input_d), t: trip_conv(cv.d_out, cv.kernel_d, cv.input_d, cv.conv_stride_d, 1) };
                self.conv(ctx, x, w, &[cv.c_out, cv.c_in, cv.kernel_d, cv.kernel_h, cv.kernel_w], vec![rw, rh, rd], WKind::Normal, lb(cv.c_in), lb(cv.c_out), c);
                Some("conv")
            }
            BasicBlockType::DepthwiseConv2D(cv) => {
                let w = self.unflatten(nd.inputs[1]);
                let rw = Rel { ob: lb(cv.w_out), kb: lb(cv.kernel_w), ib: lb(cv.input_w), t: trip_conv(cv.w_out, cv.kernel_w, cv.input_w, cv.conv_stride_w, 1) };
                let rh = Rel { ob: lb(cv.h_out), kb: lb(cv.kernel_h), ib: lb(cv.input_h), t: trip_conv(cv.h_out, cv.kernel_h, cv.input_h, cv.conv_stride_h, 1) };
                self.conv(ctx, x, w, &[cv.channels, 1, cv.kernel_h, cv.kernel_w], vec![rw, rh], WKind::Depthwise, lb(cv.channels), lb(cv.channels), c);
                Some("conv")
            }
            BasicBlockType::ConvTranspose2D(cv) => {
                let w = self.unflatten(nd.inputs[1]);
                let rw = Rel { ob: lb(cv.w_out), kb: lb(cv.kernel_w), ib: lb(cv.input_w), t: trip_tconv(cv.w_out, cv.kernel_w, cv.input_w, cv.stride_w) };
                let rh = Rel { ob: lb(cv.h_out), kb: lb(cv.kernel_h), ib: lb(cv.input_h), t: trip_tconv(cv.h_out, cv.kernel_h, cv.input_h, cv.stride_h) };
                self.conv(ctx, x, w, &[cv.c_in, cv.c_out, cv.kernel_h, cv.kernel_w], vec![rw, rh], WKind::Transpose, lb(cv.c_in), lb(cv.c_out), c);
                Some("conv")
            }
            BasicBlockType::ConvTranspose3D(cv) => {
                let w = self.unflatten(nd.inputs[1]);
                let rw = Rel { ob: lb(cv.w_out), kb: lb(cv.kernel_w), ib: lb(cv.input_w), t: trip_tconv(cv.w_out, cv.kernel_w, cv.input_w, cv.stride_w) };
                let rh = Rel { ob: lb(cv.h_out), kb: lb(cv.kernel_h), ib: lb(cv.input_h), t: trip_tconv(cv.h_out, cv.kernel_h, cv.input_h, cv.stride_h) };
                let rd = Rel { ob: lb(cv.d_out), kb: lb(cv.kernel_d), ib: lb(cv.input_d), t: trip_tconv(cv.d_out, cv.kernel_d, cv.input_d, cv.stride_d) };
                self.conv(ctx, x, w, &[cv.c_in, cv.c_out, cv.kernel_d, cv.kernel_h, cv.kernel_w], vec![rw, rh, rd], WKind::Transpose, lb(cv.c_in), lb(cv.c_out), c);
                Some("conv")
            }
            _ => None,
        }
    }

    /// Polynomial-product convolution (VerfCNN with the output view), the
    /// ablation baseline (MORPHEUS_CONV=pp): the full convolution Y_full is
    /// committed advice; the output Y is a bit-affine view of Y_full; and
    ///   Σ_m α^m Y_full(d, m) = Σ_c X_rev(c, α) · W_flat(d, c, α)
    /// binds Y_full at a random α and output channel point.
    fn conv2d_pp(&mut self, ctx: &mut Ctx, nd: &crate::dag::Node, cv: &crate::basicblock::conv::Conv2D, c: AClaim) {
        let (x, wf, yf) = (nd.inputs[0], nd.inputs[1], nd.outputs[1]);
        let (lw, lh) = (lb(cv.input_w), lb(cv.input_h));
        let l_si = lw + lh;
        let (lwo, lho) = (lb(cv.w_out), lb(cv.h_out));
        let lco = lb(cv.c_out);
        let lci = self.g.nvars[x] - l_si;
        let lf = self.g.nvars[yf] - lco;
        let lk = self.g.nvars[wf] - lci - lco;
        assert!(cv.conv_stride_h.is_power_of_two() && cv.conv_stride_w.is_power_of_two());
        let wo_shift = cv.conv_stride_w.trailing_zeros() as usize;
        let ho_shift = lw + cv.conv_stride_h.trailing_zeros() as usize;
        assert!(wo_shift + lwo <= ho_shift && ho_shift + lho <= l_si && l_si <= lf, "conv view layout");
        let r = c.point.clone();
        let (r_wo, r_ho, r_d) = (r[..lwo].to_vec(), r[lwo..lwo + lho].to_vec(), r[lwo + lho..].to_vec());
        // ---- View: Y(r) = Σ_{wo,ho} eq·[real]·Y_full(r_d, E(wo, ho)).
        let rw = Rel::map(lwo, lwo, (0..cv.w_out).map(|o| (o, o)));
        let rh = Rel::map(lho, lho, (0..cv.h_out).map(|o| (o, o)));
        let (tw, th) = (rw.table(&r_wo, &[]), rh.table(&r_ho, &[]));
        let nv = lwo + lho;
        let tables = if ctx.prover {
            let pd = self.pd.unwrap();
            let fixed: Vec<(usize, K8)> = (0..lco).map(|j| (lf + j, r_d[j])).collect();
            let yd = partial_eval(pd.v(yf), lf + lco, &fixed);
            let p: Vec<K8> = (0..1usize << nv).map(|i| tw[i & ((1 << lwo) - 1)].mul(&th[i >> lwo])).collect();
            let v: Vec<K8> = (0..1usize << nv).map(|i| { let (wo, ho) = (i & ((1 << lwo) - 1), i >> lwo); yd[(cv.s_in - 1) - (wo << wo_shift) - (ho << ho_shift)] }).collect();
            Some(Tables::Host(vec![p, v]))
        } else { None };
        let (twc, thc) = (tw.clone(), th.clone());
        let out = ctx.masked_sc(nv, 1, true, c.z, &c.pad, tables, &|xs: &[K8]| tensor::mle_k(&twc, &xs[..lwo]).mul(&tensor::mle_k(&thc, &xs[lwo..])));
        let rr = out.rho.mul(&out.rho.sub(&K8::ONE));
        let mut pt = vec![K8::ONE; lf];
        for b in l_si..lf { pt[b] = K8::ZERO; }
        for i in 0..lwo { pt[wo_shift + i] = K8::ONE.sub(&out.x[i]); }
        for i in 0..lho { pt[ho_shift + i] = K8::ONE.sub(&out.x[lwo + i]); }
        pt.extend_from_slice(&r_d);
        self.push(ctx, yf, AClaim { point: pt, pad: lin1(out.pads[0], rr), z: out.v[0] });
        // ---- Polynomial product at (α, r_d2).
        let alpha = ctx.chal();
        let r_d2 = ctx.chals(lco);
        let pows = |n: usize, base: &K8| -> Vec<K8> { let mut v = Vec::with_capacity(n); let mut a = K8::ONE; for _ in 0..n { v.push(a); a = a.mul(base); } v };
        let a_m = pows(1usize << lf, &alpha);
        let a_j = pows(1usize << lk, &alpha);
        let ainv = alpha.inv();
        let (wp, hp) = (1usize << lw, 1usize << lh);
        // X_rev(c, α) = α^{s_in-1} Σ_p α^{-p}[real p] X(c, p), p = iw + w_pad·ih.
        let tw_x: Vec<K8> = { let ai = pows(wp, &ainv); (0..wp).map(|i| if i < cv.input_w { ai[i] } else { K8::ZERO }).collect() };
        let aw = kpow(&ainv, wp as u64);
        let th_x: Vec<K8> = { let ah = pows(hp, &aw); (0..hp).map(|i| if i < cv.input_h { ah[i] } else { K8::ZERO }).collect() };
        let lead = kpow(&alpha, (cv.s_in - 1) as u64);
        let ap: Vec<K8> = (0..1usize << l_si).map(|p| lead.mul(&tw_x[p & (wp - 1)]).mul(&th_x[p >> lw])).collect();
        let (yf2, px, pw, val) = if ctx.prover {
            let pd = self.pd.unwrap();
            let yf2 = partial_eval(pd.v(yf), lf + lco, &(0..lco).map(|j| (lf + j, r_d2[j])).collect::<Vec<_>>());
            let xv = pd.v(x);
            let px: Vec<K8> = (0..1usize << lci).into_par_iter().map(|cc| (0..1usize << l_si).fold(K8::ZERO, |a, p| { let u = xv[p + (cc << l_si)]; if u == 0 { a } else { a.add(&ap[p].scale(u)) } })).collect();
            let wfix: Vec<(usize, K8)> = (0..lco).map(|j| (lk + lci + j, r_d2[j])).collect();
            let wd = partial_eval(pd.v(wf), lk + lci + lco, &wfix);
            let pw: Vec<K8> = (0..1usize << lci).map(|cc| (0..1usize << lk).fold(K8::ZERO, |a, j| a.add(&a_j[j].mul(&wd[j + (cc << lk)])))).collect();
            let val = a_m.iter().zip(&yf2).fold(K8::ZERO, |a, (u, v)| a.add(&u.mul(v)));
            if selfcheck() {
                let rhs = px.iter().zip(&pw).fold(K8::ZERO, |a, (u, v)| a.add(&u.mul(v)));
                assert!(val == rhs, "conv polynomial-product identity");
            }
            (yf2, px, pw, Some(val))
        } else { (vec![], vec![], vec![], None) };
        let a = ctx.alloc("convolution A pad");
        let z = ctx.send(val.map(|v| v.add(&ctx.bank[a as usize])));
        let eta = lin1(a, K8::ONE);
        // Σ_m α^m Y_full(r_d2, m) = z − a.
        let am = a_m.clone();
        let o1 = ctx.masked_sc(lf, 1, true, z, &eta, if ctx.prover { Some(Tables::Host(vec![a_m, yf2])) } else { None }, &|xs: &[K8]| tensor::mle_k(&am, xs));
        let rr1 = o1.rho.mul(&o1.rho.sub(&K8::ONE));
        let mut p1 = o1.x.clone(); p1.extend_from_slice(&r_d2);
        self.push(ctx, yf, AClaim { point: p1, pad: lin1(o1.pads[0], rr1), z: o1.v[0] });
        // Σ_c X_rev(c, α) W_flat(r_d2, c, α) = z − a.
        let o2 = ctx.masked_sc(lci, 2, false, z, &eta, if ctx.prover { Some(Tables::Host(vec![px, pw])) } else { None }, &|_: &[K8]| K8::ONE);
        let rr2 = o2.rho.mul(&o2.rho.sub(&K8::ONE));
        let xc = o2.x.clone();
        // X_rev(x_c, α) -> X.
        let tables3 = if ctx.prover {
            let pd = self.pd.unwrap();
            Some(Tables::Host(vec![ap.clone(), partial_eval(pd.v(x), l_si + lci, &(0..lci).map(|j| (l_si + j, xc[j])).collect::<Vec<_>>())]))
        } else { None };
        let (twx, thx) = (tw_x.clone(), th_x.clone());
        let o3 = ctx.masked_sc(l_si, 1, true, o2.v[0], &lin1(o2.pads[0], rr2), tables3, &|xs: &[K8]| lead.mul(&tensor::mle_k(&twx, &xs[..lw])).mul(&tensor::mle_k(&thx, &xs[lw..])));
        let rr3 = o3.rho.mul(&o3.rho.sub(&K8::ONE));
        let mut px_pt = o3.x.clone(); px_pt.extend_from_slice(&xc);
        self.push(ctx, x, AClaim { point: px_pt, pad: lin1(o3.pads[0], rr3), z: o3.v[0] });
        // W_flat(r_d2, x_c, α) -> W_flat.
        let tables4 = if ctx.prover {
            let pd = self.pd.unwrap();
            let fx: Vec<(usize, K8)> = (0..lci).map(|j| (lk + j, xc[j])).chain((0..lco).map(|j| (lk + lci + j, r_d2[j]))).collect();
            Some(Tables::Host(vec![a_j.clone(), partial_eval(pd.v(wf), lk + lci + lco, &fx)]))
        } else { None };
        let aj = a_j.clone();
        let o4 = ctx.masked_sc(lk, 1, true, o2.v[1], &lin1(o2.pads[1], rr2), tables4, &|xs: &[K8]| tensor::mle_k(&aj, xs));
        let rr4 = o4.rho.mul(&o4.rho.sub(&K8::ONE));
        let mut wpt = o4.x.clone(); wpt.extend_from_slice(&xc); wpt.extend_from_slice(&r_d2);
        self.push(ctx, wf, AClaim { point: wpt, pad: lin1(o4.pads[0], rr4), z: o4.v[0] });
    }

    /// The unflattened kernel behind a conv's weight input.
    fn unflatten(&self, mut e: usize) -> usize {
        loop {
            match self.g.dag.producers[e].map(|p| &self.g.dag.nodes[p].kind) {
                Some(BasicBlockType::FlattenKernel(_)) | Some(BasicBlockType::FlattenKernel3D(_)) | Some(BasicBlockType::ChangeShape(_)) => e = self.g.dag.nodes[self.g.dag.producers[e].unwrap()].inputs[0],
                _ => return e,
            }
        }
    }

    /// ZeroPad-style map: Y[c, o_d] = X[c, o_d − off_d] for o_d − off_d < in_d.
    #[allow(clippy::too_many_arguments)]
    fn pad_like(&mut self, ctx: &mut Ctx, x: usize, y: usize, channels: usize, ins: &[usize], outs: &[usize], off: &[usize], c: AClaim) {
        // Axes are given outermost-first ([d,] h, w); the lowest bits are the last axis.
        let k = ins.len();
        let ib: Vec<usize> = ins.iter().map(|&v| lb(v)).collect();
        let ob: Vec<usize> = outs.iter().map(|&v| lb(v)).collect();
        let (si, so): (usize, usize) = (ib.iter().sum(), ob.iter().sum());
        let nc = self.g.nvars[x] - si;
        assert_eq!(self.g.nvars[y], so + nc, "pad channel bits");
        let r = &c.point;
        let (mut xo, mut yo) = (0, 0);
        let mut axes = Vec::new();
        let mut fixed = Vec::new();
        Self::chan_axis(channels, si, nc, &r[so..], &mut fixed, &mut axes);
        for d in (0..k).rev() {
            let rel = Rel::map(ob[d], ib[d], (0..ins[d]).map(|i| (i + off[d], i)));
            axes.push(MapAxis { off: xo, rel, ro: r[yo..yo + ob[d]].to_vec(), rk: vec![] });
            xo += ib[d]; yo += ob[d];
        }
        self.map_reduce(ctx, x, fixed, axes, c.z, &c.pad);
    }

    /// Two-stage convolution reduction. `wshape` is the unflattened kernel
    /// shape ([C_out, C_in, k..] normal/depthwise, [C_in, C_out, k..]
    /// transposed); `rels` are the spatial axes, lowest first.
    #[allow(clippy::too_many_arguments)]
    fn conv(&mut self, ctx: &mut Ctx, x: usize, w: usize, wshape: &[usize], rels: Vec<Rel>, kind: WKind, lci: usize, lco: usize, c: AClaim) {
        let r = c.point.clone();
        let so: usize = rels.iter().map(|d| d.ob).sum();
        let si: usize = rels.iter().map(|d| d.ib).sum();
        let sk: usize = rels.iter().map(|d| d.kb).sum();
        assert_eq!(self.g.nvars[x], si + lci, "conv input bits (edge {x})");
        assert_eq!(r.len(), so + lco, "conv output bits");
        assert_eq!(self.g.nvars[w], wshape.iter().map(|&s| lb(s)).sum::<usize>(), "conv weight bits (edge {w})");
        let ro: Vec<Vec<K8>> = { let mut off = 0; rels.iter().map(|d| { let v = r[off..off + d.ob].to_vec(); off += d.ob; v }).collect() };
        let rd = r[so..].to_vec();
        // Kernel layout: [k (lowest), channel-in-like, channel-out-like].
        let wo = Self::offs(wshape);
        let (cin_off, cout_off) = match kind { WKind::Normal | WKind::Depthwise => (wo[1], wo[0]), WKind::Transpose => (wo[0], wo[1]) };
        // Stage 1 variables: kernel taps then the summed channel.
        let lc = lci;
        let n1 = sk + lc;
        let tables = if ctx.prover {
            let pd = self.pd.unwrap();
            // A(k, c) by contracting the spatial axes of X one by one.
            let xv = pd.v(x);
            let mut sizes: Vec<usize> = rels.iter().map(|d| d.ib).collect();
            sizes.push(lci);
            let mut cur: Option<Vec<K8>> = None;
            for (a, d) in rels.iter().enumerate() {
                let m = d.kmat(&ro[a]);
                let pre: usize = 1usize << sizes[..a].iter().sum::<usize>();
                let post: usize = 1usize << sizes[a + 1..].iter().sum::<usize>();
                let (ni, nk) = (1usize << d.ib, 1usize << d.kb);
                let out: Vec<K8> = (0..post * nk).into_par_iter().flat_map_iter(|pk| {
                    let (ph, k) = (pk / nk, pk % nk);
                    let row = &m[k];
                    let cur = &cur;
                    (0..pre).map(move |pl| {
                        let mut acc = K8::ZERO;
                        for &(i, coef) in row {
                            let idx = (ph * ni + i) * pre + pl;
                            let v = match cur { Some(t) => t[idx].mul(&coef), None => { let u = xv[idx]; if u == 0 { continue } else { coef.scale(u) } } };
                            acc = acc.add(&v);
                        }
                        acc
                    })
                }).collect();
                sizes[a] = d.kb;
                cur = Some(out);
            }
            let a_tab = cur.expect("conv with no spatial axis");
            // W restricted to the output channel point (or kept, depthwise).
            let wv = pd.v(w);
            let nw = self.g.nvars[w];
            let wfixed: Vec<(usize, K8)> = match kind {
                WKind::Depthwise => vec![],
                _ => (0..lco).map(|j| (cout_off + j, rd[j])).collect(),
            };
            let w_tab = partial_eval(wv, nw, &wfixed);
            assert_eq!(w_tab.len(), 1usize << n1, "kernel table size");
            assert_eq!(a_tab.len(), 1usize << n1, "patch table size");
            let mut t = Vec::new();
            if kind == WKind::Depthwise {
                // Public channel relation G(d, c) = [d = c]·[c < C]: zero on
                // padded channels, independent of the committed padding.
                let e = eq_table(&rd);
                let nch = wshape[0];
                t.push((0..1usize << n1).map(|i| if (i >> sk) < nch { e[i >> sk] } else { K8::ZERO }).collect::<Vec<_>>());
            }
            t.push(w_tab);
            t.push(a_tab);
            Some(Tables::Host(t))
        } else { None };
        let dw = kind == WKind::Depthwise;
        let rd2 = rd.clone();
        let nch = wshape[0];
        let out = ctx.masked_sc(n1, 2, dw, c.z, &c.pad, tables, &|xs: &[K8]| if dw { masked_eq(&rd2, &xs[sk..], nch) } else { K8::ONE });
        let rr = out.rho.mul(&out.rho.sub(&K8::ONE));
        let (xk, xc) = (out.x[..sk].to_vec(), out.x[sk..].to_vec());
        // Kernel claim.
        let mut wp = vec![K8::ZERO; self.g.nvars[w]];
        wp[..sk].copy_from_slice(&xk);
        match kind {
            WKind::Depthwise => { wp[cout_off..cout_off + lc].copy_from_slice(&xc); }
            _ => { wp[cin_off..cin_off + lci].copy_from_slice(&xc); wp[cout_off..cout_off + lco].copy_from_slice(&rd); }
        }
        let _ = cin_off;
        self.push(ctx, w, AClaim { point: wp, pad: lin1(out.pads[0], rr), z: out.v[0] });
        // Patch claim -> input map.
        let fixed: Vec<(usize, K8)> = (0..lci).map(|j| (si + j, xc[j])).collect();
        let (mut xo, mut ko) = (0, 0);
        let mut axes = Vec::new();
        for (a, d) in rels.into_iter().enumerate() {
            let (ib, kb) = (d.ib, d.kb);
            axes.push(MapAxis { off: xo, rel: d, ro: ro[a].clone(), rk: xk[ko..ko + kb].to_vec() });
            xo += ib; ko += kb;
        }
        self.map_reduce(ctx, x, fixed, axes, out.v[1], &lin1(out.pads[1], rr));
    }

    /// ProductZeroCheck roots: Σ_x eq(r, x)·A(x)·B(x) = 0.
    pub(super) fn pzc_roots(&mut self, ctx: &mut Ctx) {
        let nodes: Vec<usize> = self.g.dag.nodes.iter().filter(|n| matches!(n.kind, BasicBlockType::ProductZeroCheck(_))).map(|n| n.id).collect();
        for node in nodes {
            let nd = self.g.dag.nodes[node].clone();
            let (a, b) = (nd.inputs[0], nd.inputs[1]);
            let n = self.g.nvars[a];
            assert_eq!(self.g.nvars[b], n);
            let r = ctx.chals(n);
            let tables = if ctx.prover {
                let pd = self.pd.unwrap();
                if gpu::enabled() && (1usize << n) >= gpu::threshold() {
                    Some(Tables::Dev(vec![gpu::eq(&r), gpu::lift(&pd.dev(a), 1 << n), gpu::lift(&pd.dev(b), 1 << n)]))
                } else {
                    let f = |e: usize| pd.v(e).iter().map(|&v| K8::from_base(v)).collect::<Vec<_>>();
                    Some(Tables::Host(vec![eq_table(&r), f(a), f(b)]))
                }
            } else { None };
            let rc = r.clone();
            let out = ctx.masked_sc(n, 2, true, K8::ZERO, &Lin::new(), tables, &|xs: &[K8]| eq_pt(&rc, xs));
            let rr = out.rho.mul(&out.rho.sub(&K8::ONE));
            self.push(ctx, a, AClaim { point: out.x.clone(), pad: lin1(out.pads[0], rr), z: out.v[0] });
            self.push(ctx, b, AClaim { point: out.x, pad: lin1(out.pads[1], rr), z: out.v[1] });
        }
    }
}

/// Convolution relation along one axis: i = s·o + δ·k (o < O, k < K, i < I).
fn trip_conv(o_n: usize, k_n: usize, i_n: usize, s: usize, dl: usize) -> Vec<(usize, usize, usize)> {
    let mut v = Vec::new();
    for o in 0..o_n { for k in 0..k_n { let i = o * s + dl * k; if i < i_n { v.push((o, k, i)); } } }
    v
}

/// Transposed convolution relation: o = s·i + k (o < O).
fn trip_tconv(o_n: usize, k_n: usize, i_n: usize, s: usize) -> Vec<(usize, usize, usize)> {
    let mut v = Vec::new();
    for i in 0..i_n { for k in 0..k_n { let o = i * s + k; if o < o_n { v.push((o, k, i)); } } }
    v
}

#[cfg(test)]
mod prefix_tests {
    use super::*;
    #[test]
    fn prefix_mle_matches_the_boolean_sum() {
        let x: Vec<K8> = (0..5).map(|i| K8::from_base(7 + 13 * i as u64)).collect();
        for n in 0..=33usize {
            let e = eq_table(&x);
            let want = (0..32usize.min(n)).fold(K8::ZERO, |a, c| a.add(&e[c]));
            assert_eq!(prefix_mle(n, &x), want, "n = {n}");
        }
    }
}

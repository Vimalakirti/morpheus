//! Model instances for the Morpheus prover: the same DAGs, weights and
//! inputs as the paper's previous evaluation binaries
//! (bench_streaming_oneshot_{gpt2,gptj,llama2,llama3}, bench_streaming_bert,
//! ...), so measurements compare at matched workloads.
//!
//! Decoder-only transformers prove a one-shot autoregressive generation:
//! embedding of the SEQ_LEN input tokens, the body, the LM head and the
//! argmax check binding every next token to its logits. The token
//! selectors are request inputs and public.

use crate::dag::{gptj::gpt_j_6b_hidden, gpt2::gpt_2_small, llama::{llama3_8b_hidden, llama_2_7b_hidden}, bert::bert_large};
use crate::dag::{Dag, DagBuilder, DataType, EdgeId, Role, Witness};
use crate::util::arith::f_to_int;
use crate::SF_LOG;
use almost_goldilocks_cuda::field::AlmostGoldilocksField as F;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::time::Instant;

fn ev(k: &str, d: usize) -> usize { std::env::var(k).ok().and_then(|s| s.parse().ok()).unwrap_or(d) }
fn zeros(n: usize) -> Vec<F> { crate::zero_witness_vec(n) }
fn cw(shape: Vec<usize>) -> Witness {
    let n = shape.iter().map(|d| d.next_power_of_two()).product();
    Witness::new(shape, zeros(n), DataType::Float, *SF_LOG, Role::Constant)
}

/// WEIGHTS=dense: quantized-model-like weights instead of the benchmark's
/// zero matrices. Matrices are uniform integers in [-WEIGHT_MAX, WEIGHT_MAX]
/// (default 63, i.e. |w| < 0.0625 at scale 2^10), norm gains are 1.0 and
/// biases 0. The default reproduces the previous evaluation's weights.
fn dense() -> bool { std::env::var("WEIGHTS").ok().as_deref() == Some("dense") }

/// A weight matrix (logical shape, padded storage, last axis lowest).
fn mat(shape: Vec<usize>) -> Witness {
    if !dense() { return cw(shape); }
    static CTR: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let a = ev("WEIGHT_MAX", 63) as i64;
    let mut rng = StdRng::seed_from_u64(7 + CTR.fetch_add(1, std::sync::atomic::Ordering::Relaxed));
    let pads: Vec<usize> = shape.iter().map(|d| d.next_power_of_two()).collect();
    let n: usize = pads.iter().product();
    let mut data = zeros(n);
    // Fill logical entries only (padding stays zero).
    let mut idx = vec![0usize; shape.len()];
    loop {
        let mut flat = 0usize;
        let mut stride = 1usize;
        for d in (0..shape.len()).rev() { flat += idx[d] * stride; stride *= pads[d]; }
        let v: i64 = rng.gen_range(-a..=a);
        data[flat] = F(crate::morpheus::piop::tensor::from_i128(v as i128));
        let mut d = shape.len();
        loop {
            if d == 0 { break; }
            d -= 1;
            idx[d] += 1;
            if idx[d] < shape[d] { break; }
            idx[d] = 0;
            if d == 0 { d = usize::MAX; break; }
        }
        if d == usize::MAX || shape.is_empty() { break; }
    }
    Witness::new(shape, data, DataType::Float, *SF_LOG, Role::Constant)
}

/// A normalization gain: 1.0 (dense) or the benchmark's zeros.
fn norm(dim: usize) -> Witness {
    if !dense() { return cw(vec![dim]); }
    let mut d = zeros(dim.next_power_of_two());
    for x in d.iter_mut().take(dim) { *x = F(1u64 << *SF_LOG); }
    Witness::new(vec![dim], d, DataType::Float, *SF_LOG, Role::Constant)
}

/// A bias: zeros in both modes.
fn bias(dim: usize) -> Witness { cw(vec![dim]) }

/// A built request: the graph, its witness, which edges are public, and the
/// unit count for per-unit reporting (generated tokens, input tokens, 1).
pub struct Instance {
    pub name: String,
    pub dag: Dag,
    pub wit: Vec<Vec<Witness>>,
    pub public: Vec<EdgeId>,
    pub units: usize,
    pub unit: &'static str,
    pub witness_secs: f64,
    pub ar: Option<super::ar_statement::ArStatement>,
}

/// Embedding / head patterns of the previous binaries.
#[derive(Clone, Copy)]
enum Emb { Gpt2, Const1024, Alt2048 }

fn emb_data(kind: Emb, rows: usize, hidden: usize) -> Vec<F> {
    let rp = rows.next_power_of_two();
    let hp = hidden.next_power_of_two();
    let mut data = zeros(rp * hp);
    for d in 0..hp {
        let val = match kind { Emb::Gpt2 => F(1000 + 10 * ((d % 16) as u64)), Emb::Const1024 => F(1024), Emb::Alt2048 => F(if d % 2 == 0 { 0 } else { 2048 }) };
        match kind { Emb::Const1024 => for v in 0..rp { data[v + d * rp] = val; }, _ => for v in 0..rows { data[v + d * rp] = val; } }
    }
    data
}

struct Ar { g: DagBuilder, emb_sel: EdgeId, logits: Vec<EdgeId>, sels: Vec<EdgeId>, ranges: Vec<(usize, usize)>, rank_constants: Vec<[usize;3]> }

/// Embedding, body (closure), LM head (tied for GPT-2), argmax check.
#[allow(clippy::too_many_arguments)]
fn ar_graph(seq: usize, vocab: usize, hidden: usize, emb: Emb, head: Emb, tied: bool, shards: usize, pos: Option<Witness>,
            body: &dyn Fn(&mut DagBuilder, EdgeId) -> EdgeId) -> Ar {
    let init = vec![0usize; seq];
    let mut g = DagBuilder::new();
    let w_e = g.param(Witness::new(vec![vocab, hidden], emb_data(emb, vocab, hidden), DataType::Float, *SF_LOG, Role::Constant));
    let (h0, emb_sel) = g.embedding_lookup(w_e, seq, vocab, &init, Role::Input);
    let h0 = match pos { Some(p) => g.add_positional_encoding(h0, p), None => h0 };
    let h_in = g.change_shape(h0, vec![1, seq, hidden]);
    let h_out = body(&mut g, h_in);
    if shards > 1 {
        let ranges = DagBuilder::vocab_shard_ranges(vocab, shards);
        let lens: Vec<usize> = ranges.iter().map(|&(_, l)| l).collect();
        let ws: Vec<EdgeId> = lens.iter().map(|&l| g.param(Witness::new(vec![l, hidden], emb_data(head, l, hidden), DataType::Float, *SF_LOG, Role::Constant))).collect();
        let lg = g.lm_head_sharded(h_out, &ws, seq, &lens);
        let first_new = g.num_edges;
        let sels = g.argmax_check_sharded(&lg, &ranges, seq, &init, Role::Input);
        let rank_constants = rank_constants(&g, first_new, ranges.len());
        Ar { g, emb_sel, logits: lg, sels, ranges, rank_constants }
    } else {
        let logits = if tied { g.lm_head_weight_tied(h_out, w_e, seq, vocab) } else {
            let w = g.param(Witness::new(vec![vocab, hidden], emb_data(head, vocab, hidden), DataType::Float, *SF_LOG, Role::Constant));
            g.lm_head(h_out, w, seq, vocab)
        };
        let first_new = g.num_edges;
        let sel = g.argmax_check(logits, seq, vocab, &init, Role::Input);
        let rank_constants = rank_constants(&g, first_new, 1);
        Ar { g, emb_sel, logits: vec![logits], sels: vec![sel], ranges: vec![(0, vocab)], rank_constants }
    }
}

/// Run the generation: random input tokens, forward, next tokens = argmax of
/// the logits, set the argmax selectors, forward again.
fn rank_constants(g: &DagBuilder, first: usize, shards: usize) -> Vec<[usize;3]> {
    // Rank constructors emit B,V,index as public inputs before selector inputs.
    let edges: Vec<usize> = (first..g.num_edges).filter(|&e| g.init_values[e].as_ref().map(|w|w.role==Role::Input).unwrap_or(false)).take(3*shards).collect();
    assert_eq!(edges.len(),3*shards);
    edges.chunks_exact(3).map(|e|[e[0],e[1],e[2]]).collect()
}

fn ar_run(name: &str, ar: Ar, seq: usize, vocab: usize, request: u64) -> Instance {
    let Ar { g, emb_sel, logits, sels, ranges, rank_constants } = ar;
    let (dag, mut wit) = g.compile();
    let t = Instant::now();
    let mut rng = StdRng::seed_from_u64(42 + request);
    assert!(seq > 0);
    let prompt_len = 16.min(seq);
    let mut tokens: Vec<usize> = (0..seq).map(|_| rng.gen::<usize>() % vocab).collect();
    // Generate an actual suffix. This is deliberately outside proof timing.
    for position in prompt_len..seq {
        wit[emb_sel] = vec![DagBuilder::build_one_hot_selector_witness(seq, vocab, &tokens, Role::Input)];
        dag.run(&mut wit, &[]);
        let (mut best, mut value) = (0usize, i128::MIN);
        for (k, &(start,len)) in ranges.iter().enumerate() {
            for off in 0..len {
                let v=f_to_int(wit[logits[k]][0].get(&[position-1,off]));
                if v>value { value=v; best=start+off; }
            }
        }
        tokens[position]=best;
    }
    wit[emb_sel] = vec![DagBuilder::build_one_hot_selector_witness(seq, vocab, &tokens, Role::Input)];
    dag.run(&mut wit, &[]);
    let next: Vec<usize> = (0..seq).map(|i| {
        let (mut bt, mut bv) = (0usize, i128::MIN);
        for (k, &(start, len)) in ranges.iter().enumerate() {
            let lg = &wit[logits[k]][0];
            for off in 0..len { let v = f_to_int(lg.get(&[i, off])); if v > bv { bv = v; bt = start + off; } }
        }
        bt
    }).collect();
    if ranges.len() > 1 {
        for (k, &(start, len)) in ranges.iter().enumerate() {
            let local: Vec<Option<usize>> = next.iter().map(|&t| if t >= start && t < start + len { Some(t - start) } else { None }).collect();
            wit[sels[k]] = vec![DagBuilder::build_sharded_one_hot_selector_witness(seq, len, &local, Role::Input)];
        }
    } else {
        wit[sels[0]] = vec![DagBuilder::build_one_hot_selector_witness(seq, vocab, &next, Role::Input)];
    }
    dag.run(&mut wit, &[]);
    let mut public = vec![emb_sel];
    public.extend(&sels);
    public.extend(rank_constants.iter().flatten().copied());
    let ar = Some(super::ar_statement::ArStatement { prompt_len, seq, vocab, embedding: emb_sel, selectors: sels.clone(), ranges: ranges.clone(), rank_constants, logits: logits.clone(), logit_bound: DagBuilder::first_argmax_budget(vocab).0 });
    Instance { name: name.into(), dag, wit, public, units: seq.saturating_sub(16).max(1), unit: "generated token", witness_secs: t.elapsed().as_secs_f64(), ar }
}

pub fn gpt2(request: u64) -> Instance {
    let (layers, seq, vocab) = (ev("NUM_LAYERS", 12), ev("SEQ_LEN", 128), ev("VOCAB_SIZE", 50257));
    let (nh, hd) = (ev("NUM_HEADS", 12), ev("HEAD_DIM", 64));
    let h = nh * hd; let ffn = 4 * h;
    let mut v: [Vec<Witness>; 16] = Default::default();
    for _ in 0..layers {
        v[0].push(norm(h));
        for k in 1..5 { v[k].push(mat(vec![h, h])); }
        for k in 5..10 { v[k].push(bias(h)); }
        v[10].push(norm(h));
        v[11].push(mat(vec![h, ffn])); v[12].push(mat(vec![ffn, h]));
        v[13].push(bias(h)); v[14].push(bias(ffn)); v[15].push(bias(h));
    }
    let ln = (norm(h), bias(h));
    let pos = Witness::new(vec![seq, h], zeros(seq.next_power_of_two() * h.next_power_of_two()), DataType::Float, *SF_LOG, Role::Constant);
    let v = std::cell::RefCell::new(Some((v, ln)));
    let ar = ar_graph(seq, vocab, h, Emb::Gpt2, Emb::Gpt2, true, 1, Some(pos), &|g, x| {
        let (v, ln) = v.borrow_mut().take().unwrap();
        let [a, b, c, d, e, f, gg, hh, i, j, k, l, m, n, o, p] = v;
        g.pipe(&[x], gpt_2_small(a, b, c, d, e, f, gg, hh, i, j, k, l, m, n, o, p, ln.0, ln.1, nh, hd, seq))[0]
    });
    ar_run("GPT-2", ar, seq, vocab, request)
}

pub fn gptj(request: u64) -> Instance {
    let (layers, seq, vocab) = (ev("NUM_LAYERS", 28), ev("SEQ_LEN", 128), ev("VOCAB_SIZE", 50400));
    let (nh, hd, ffn) = (ev("NUM_HEADS", 16), ev("HEAD_DIM", 256), ev("FFN_DIM", 16384));
    let shards = ev("VOCAB_SHARDS", 1);
    let h = nh * hd;
    let mut v: [Vec<Witness>; 10] = Default::default();
    for _ in 0..layers {
        v[0].push(norm(h));
        for k in 1..5 { v[k].push(mat(vec![h, h])); }
        v[5].push(bias(h));
        v[6].push(mat(vec![h, ffn])); v[7].push(mat(vec![ffn, h]));
        v[8].push(bias(ffn)); v[9].push(bias(h));
    }
    let fin = (norm(h), bias(h));
    let v = std::cell::RefCell::new(Some((v, fin)));
    let ar = ar_graph(seq, vocab, h, Emb::Alt2048, Emb::Alt2048, false, shards, None, &|g, x| {
        let ([a, b, c, d, e, f, gg, hh, i, j], fin) = v.borrow_mut().take().unwrap();
        g.pipe(&[x], gpt_j_6b_hidden(a, b, c, d, e, f, gg, hh, i, j, fin.0, fin.1, nh, hd, seq))[0]
    });
    ar_run("GPT-J-6B", ar, seq, vocab, request)
}

pub fn llama2(request: u64) -> Instance {
    let (layers, seq, vocab) = (ev("NUM_LAYERS", 32), ev("SEQ_LEN", 128), ev("VOCAB_SIZE", 32000));
    let (nh, hd, ffn, fs) = (ev("NUM_HEADS", 32), ev("HEAD_DIM", 128), ev("FFN_DIM", 11008), ev("FFN_SHARDS", 1));
    let shards = ev("VOCAB_SHARDS", 1);
    let h = nh * hd;
    let sf = ffn / fs;
    let (mut a, mut b, mut c, mut d, mut e, mut f) = (vec![], vec![], vec![], vec![], vec![], vec![]);
    let (mut p1, mut p2, mut p3): (Vec<Vec<Witness>>, Vec<Vec<Witness>>, Vec<Vec<Witness>>) = (vec![], vec![], vec![]);
    let sfp = ffn.next_power_of_two() / fs;
    let cs = |shape: Vec<usize>, n: usize| Witness::new(shape, zeros(n), DataType::Float, *SF_LOG, Role::Constant);
    for _ in 0..layers {
        a.push(norm(h)); b.push(mat(vec![h, h])); c.push(mat(vec![h, h])); d.push(mat(vec![h, h])); e.push(mat(vec![h, h])); f.push(norm(h));
        let hp = h.next_power_of_two();
        if dense() {
            p1.push((0..fs).map(|_| mat(vec![h, sf])).collect());
            p2.push((0..fs).map(|_| mat(vec![h, sf])).collect());
            p3.push((0..fs).map(|_| mat(vec![sf, h])).collect());
        } else {
            p1.push((0..fs).map(|_| cs(vec![h, sf], hp * sfp)).collect());
            p2.push((0..fs).map(|_| cs(vec![h, sf], hp * sfp)).collect());
            p3.push((0..fs).map(|_| cs(vec![sf, h], sfp * hp)).collect());
        }
    }
    let fin = norm(h);
    let v = std::cell::RefCell::new(Some((a, b, c, d, e, f, p1, p2, p3, fin)));
    let ar = ar_graph(seq, vocab, h, Emb::Const1024, Emb::Const1024, false, shards, None, &|g, x| {
        let (a, b, c, d, e, f, p1, p2, p3, fin) = v.borrow_mut().take().unwrap();
        g.pipe(&[x], llama_2_7b_hidden(a, b, c, d, e, f, p1, p2, p3, fin, nh, hd, seq))[0]
    });
    ar_run("LLaMA2-7B", ar, seq, vocab, request)
}

pub fn llama3(request: u64) -> Instance {
    let (layers, seq, vocab) = (ev("NUM_LAYERS", 32), ev("SEQ_LEN", 128), ev("VOCAB_SIZE", 128256));
    let (nh, nkv, hd, ffn) = (ev("NUM_HEADS", 32), ev("NUM_KV_HEADS", 8), ev("HEAD_DIM", 128), ev("FFN_DIM", 14336));
    let shards = ev("VOCAB_SHARDS", 1);
    let h = nh * hd;
    let kv = nkv * hd;
    let mut v: [Vec<Witness>; 9] = Default::default();
    for _ in 0..layers {
        v[0].push(norm(h));
        v[1].push(mat(vec![h, h])); v[2].push(mat(vec![h, kv])); v[3].push(mat(vec![h, kv])); v[4].push(mat(vec![h, h]));
        v[5].push(norm(h));
        v[6].push(mat(vec![h, ffn])); v[7].push(mat(vec![h, ffn])); v[8].push(mat(vec![ffn, h]));
    }
    let fin = norm(h);
    let v = std::cell::RefCell::new(Some((v, fin)));
    let ar = ar_graph(seq, vocab, h, Emb::Const1024, Emb::Const1024, false, shards, None, &|g, x| {
        let ([a, b, c, d, e, f, p1, p2, p3], fin) = v.borrow_mut().take().unwrap();
        let w = |v: Vec<Witness>| v.into_iter().map(|w| vec![w]).collect();
        g.pipe(&[x], llama3_8b_hidden(a, b, c, d, e, f, w(p1), w(p2), w(p3), fin, seq, hd, nh, nkv))[0]
    });
    ar_run("LLaMA3-8B", ar, seq, vocab, request)
}

pub fn bert(request: u64) -> Instance {
    let (layers, seq) = (ev("NUM_LAYERS", 24), ev("SEQ_LEN", 128));
    let (nh, hd) = (16usize, 64usize);
    let h = nh * hd; let ffn = 4 * h; let ncls = 2;
    let unit = |dim: usize| { let mut d = zeros(dim.next_power_of_two()); for x in d.iter_mut().take(dim) { *x = F(1024); } Witness::new(vec![dim], d, DataType::Float, *SF_LOG, Role::Constant) };
    let mut v: [Vec<Witness>; 16] = Default::default();
    for _ in 0..layers {
        v[0].push(unit(h));
        for k in 1..5 { v[k].push(mat(vec![h, h])); }
        for k in 5..10 { v[k].push(cw(vec![h])); }
        v[10].push(unit(h));
        v[11].push(mat(vec![h, ffn])); v[12].push(mat(vec![ffn, h]));
        v[13].push(cw(vec![h])); v[14].push(cw(vec![ffn])); v[15].push(cw(vec![h]));
    }
    let mut g = DagBuilder::new();
    let x = g.input(vec![1, seq, h], DataType::Float);
    let [a, b, c, d, e, f, gg, hh, i, j, k, l, m, n, o, p] = v;
    let _ = g.pipe(&[x], bert_large(a, b, c, d, e, f, gg, hh, i, j, k, l, m, n, o, p, unit(h), cw(vec![h]), nh, hd, seq, mat(vec![h, ncls]), cw(vec![ncls])));
    let (dag, mut wit) = g.compile();
    let pad: usize = [1usize, seq, h].iter().map(|&s| s.next_power_of_two()).product();
    let stride = seq.next_power_of_two();
    let data: Vec<F> = (0..pad).map(|i| F(if (i / stride + request as usize) % 2 == 0 { 0 } else { 2048 })).collect();
    let input = Witness::new(vec![1, seq, h], data, DataType::Float, *SF_LOG, Role::Input);
    let t = Instant::now();
    dag.run(&mut wit, &[(x, input)]);
    Instance { name: "BERT-Large".into(), dag, wit, public: vec![x], units: seq, unit: "input token", witness_secs: t.elapsed().as_secs_f64(), ar: None }
}

/// MODEL env: gpt2 | gptj | llama2 | llama3 | bert.
pub fn build(name: &str, request: u64) -> Instance {
    match name {
        "gpt2" => gpt2(request),
        "gptj" => gptj(request),
        "llama2" => llama2(request),
        "llama3" => llama3(request),
        "bert" => bert(request),
        other => panic!("unknown MODEL {other}"),
    }
}

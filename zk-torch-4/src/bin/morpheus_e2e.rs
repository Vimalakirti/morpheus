//! Morpheus (S&P'27) end to end on a transformer of the paper (see
//! morpheus::models for MODEL and shape knobs, morpheus::driver for the run
//! options). Vision and audio models run through their benchmark binaries
//! with MORPHEUS_AT_RUN (see Dag::run).

use std::time::Instant;
use zk_torch_4::morpheus::{driver, models};

fn main() {
    env_logger::init();
    almost_goldilocks_cuda::init().expect("CUDA init");
    let model = std::env::var("MODEL").unwrap_or_else(|_| "gpt2".into());
    let t = Instant::now();
    let inst = models::build(&model, 0);
    println!("{}: {} nodes, {} edges; graph {:.2}s, witness {:.2}s; {} {}s per request",
        inst.name, inst.dag.nodes.len(), inst.dag.num_edges(), t.elapsed().as_secs_f64() - inst.witness_secs, inst.witness_secs, inst.units, inst.unit);
    driver::run(&inst.name, &inst.dag, &inst.wit, &inst.public, inst.units, inst.unit);
}

//! CPU PIOP prover/verifier replay tests. These check the actual emitted link
//! rows against raw sources; they do NOT stand in for an end-to-end PCS proof.
use super::*;
use crate::basicblock::{BasicBlock, ExpHelper};
use crate::dag::{DagBuilder, DataType};
use crate::morpheus::{keys::KeySeed, params::Profile, statement};
use crate::util::arith::int_to_f;

fn standalone() -> (Dag, Vec<Vec<Witness>>) {
    let mut b = DagBuilder::new();
    let l = exp_ln2();
    let x = b.committed_input(Witness::new(vec![1], vec![int_to_f(-l)], DataType::Float, *crate::SF_LOG, Role::Input));
    // Deliberately bypass b.exp(): compile must guard the raw primitive too.
    b.add_gkr_node(vec![x], BasicBlockType::ExpHelper(ExpHelper::new(8)));
    b.init_values.push(Some(Witness::new_wo_data(vec![1], DataType::Float, *crate::SF_LOG, Role::Output)));
    b.init_values.push(Some(Witness::new_wo_data(vec![1], DataType::Float, 0, Role::Auxiliary)));
    let (d, mut w) = b.compile();
    for &n in &d.topo {
        let node = &d.nodes[n];
        let ins = node.inputs.iter().map(|&e| &w[e][0]).collect::<Vec<_>>();
        let outs = node.kind.run(&ins);
        for (&e, v) in node.outputs.iter().zip(outs) { w[e] = vec![v]; }
    }
    (d,w)
}

fn rows_hold(rows: &[Row], raw: &[Vec<i8>]) -> bool {
    rows.iter().all(|row| {
        let mut acc = K8::ZERO;
        for t in &row.terms {
            let eq = crate::morpheus::field8::eq_table(&t.r);
            let mut sum = K8::ZERO;
            for (i, e) in eq.iter().enumerate() {
                for (b, pw) in t.pw.iter().enumerate() {
                    let digit = raw[t.region.source][t.region.base + (i << t.region.p) + b];
                    sum = sum.add(&e.mul(pw).mul(&kb(digit as i128)));
                }
            }
            acc = acc.add(&sum.mul(&t.scale));
        }
        acc == row.y
    })
}

fn replay(g: &Graph, pd: &ProverData) -> bool {
    let p = Profile::toy();
    let lay = Layout::pack(&p, &tensor_specs(g,pd));
    let raw = fill_sources(g,pd,&lay,p.ell());
    let tr = MTranscript::new("exp-guard-test", &[0;64]);
    let mut pc = Ctx::prover(tr.clone(), &pd.bank, &lay, g.cfg.slots_log);
    Walk::new(g,Some(pd)).run(&mut pc);
    let mut vc = Ctx::verifier(tr,pc.proof.clone(),&lay,g.cfg.slots_log);
    Walk::new(g,None).run(&mut vc);
    vc.ok && vc.consumed() && rows_hold(&vc.rows,&raw)
}

#[test]
fn standalone_exp_has_canonical_verifier_roots_and_rejects_compensated_k_r() {
    let (d,w) = standalone();
    assert_eq!(d.exp_guard_nodes.len(),1);
    let mut g = Graph::new(&d,&w,Config {tcl:4,slots_log:10,req_planes_log:6});
    let mut pd = ProverData::new(&g,&w,vec![K8::ZERO;1<<10]);
    assert!(replay(&g,&pd));
    let n = &d.nodes[0]; let r=n.outputs[0];
    // Old equation still holds: x=r-Lk with (k,r)=(2,L).
    let l=exp_ln2(); pd.sel.insert(0,vec![2]);
    pd.vals[r]=Some(Arc::new(vec![l as u64]));
    g.outputs=vec![(r,Arc::new(vec![l as u64]))];
    let ids=d.exp_guard_nodes[&0];
    let rho=d.nodes[ids[2]].inputs[0]; let slack=d.nodes[ids[3]].inputs[0];
    pd.vals[rho]=Some(Arc::new(vec![from_i128(l/2-l)]));
    pd.vals[slack]=Some(Arc::new(vec![from_i128(l-1-(l/2-l))]));
    // A malicious prover supplies valid bounded selectors, not a constructor
    // panic. No selection can satisfy the negative canonical remainder root.
    pd.sel.insert(ids[2],vec![0]);
    pd.sel.insert(ids[3],vec![((l-1-(l/2-l)) as u64) % (1u64 << g.lookups[&ids[3]])]);
    assert!(!replay(&g,&pd));
}

#[test]
fn immutable_literal_source_mutation_fails_verifier_rows() {
    let (d,w)=standalone();
    let g=Graph::new(&d,&w,Config {tcl:4,slots_log:10,req_planes_log:6});
    let mut pd=ProverData::new(&g,&w,vec![K8::ZERO;1<<10]);
    assert!(replay(&g,&pd));
    let (&e,v)=d.program_literals.iter().find(|(_,v)|v[0]>1).unwrap();
    pd.vals[e]=Some(Arc::new(vec![v[0]-1]));
    assert!(!replay(&g,&pd));
}

#[test]
fn verifier_statement_rejects_removed_guard_or_changed_literal() {
    let (mut d,w)=standalone();
    let guard=d.exp_guard_nodes[&0][2];
    d.exp_guard_nodes.remove(&0);
    assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| Graph::new(&d,&w,Config::default()))).is_err());
    let (d,w)=standalone();
    let mut g=Graph::new(&d,&w,Config::default());
    let e=d.nodes[guard].inputs[0];
    // Turning a derived root into a committed edge would bypass reduction.
    g.committed[e]=true;
    assert!(statement::words(&g,&Profile::toy(),&KeySeed([1;8]),11,false,4).is_err());
    g.committed[e]=false;
    g.lookups.insert(0,40);
    assert!(statement::words(&g,&Profile::toy(),&KeySeed([1;8]),11,false,4).is_err());
    g.lookups.insert(0,8);
    let (&literal,values)=d.program_literals.iter().next().unwrap();
    g.public.insert(literal,Arc::new(vec![values[0]+1]));
    assert!(statement::words(&g,&Profile::toy(),&KeySeed([1;8]),11,false,4).is_err());
}

#[test]
fn finite_exp_rejects_modular_scaled_quotient_counterfeit() {
    let mut b=DagBuilder::new();
    let x=b.committed_input(Witness::new(vec![1],vec![int_to_f(0)],DataType::Float,*crate::SF_LOG,Role::Input));
    let out=b.exp(x)[0]; let (d,mut w)=b.compile();
    for &n in &d.topo {
        let nd=&d.nodes[n];
        let ins=nd.inputs.iter().map(|&e|&w[e][0]).collect::<Vec<_>>();
        let outs=nd.kind.run(&ins);
        for (&e,v) in nd.outputs.iter().zip(outs) { w[e]=vec![v]; }
    }
    let mut g=Graph::new(&d,&w,Config {tcl:4,slots_log:12,req_planes_log:6});
    let mut pd=ProverData::new(&g,&w,vec![K8::ZERO;1<<12]);
    assert!(replay(&g,&pd));
    let add=|a:u64,b:u64| ((a as u128+b as u128)%Q as u128) as u64;
    let sub=|a:u64,b:u64| ((a as u128+Q as u128-b as u128)%Q as u128) as u64;
    let mul=|a:u64,b:u64| ((a as u128*b as u128)%Q as u128) as u64;
    let mut forged=false;
    // Construct a whole field-consistent false execution. Every scale selector
    // is in range; only the new integer interval roots exclude the counterfeit.
    for &n in &d.topo {
        let nd=&d.nodes[n];
        let v=|e:usize| pd.vals[e].as_ref().unwrap()[0];
        let y=match &nd.kind {
            BasicBlockType::ExpHelper(_) => Some(0),
            BasicBlockType::TwoPow(_) => Some(1<<15),
            BasicBlockType::ChangeShape(_) => Some(v(nd.inputs[0])),
            BasicBlockType::Add(_) => Some(add(v(nd.inputs[0]),v(nd.inputs[1]))),
            BasicBlockType::Sub(_) => Some(sub(v(nd.inputs[0]),v(nd.inputs[1]))),
            BasicBlockType::Einsum(_) => Some(nd.inputs.iter().fold(1,|a,&e|mul(a,v(e)))),
            BasicBlockType::ScaleUp(_) => Some(mul(v(nd.inputs[0]),1<<g.lookups[&n])),
            BasicBlockType::ScaleDown(_) => {
                let s=1u64<<g.lookups[&n]; let mut rem=s/2;
                let input=nd.inputs[0];
                let square=d.producers[input].map(|p| {
                    let p=&d.nodes[p]; matches!(p.kind,BasicBlockType::Einsum(_))
                        && p.inputs.len()==2 && p.inputs[0]==p.inputs[1]
                }).unwrap_or(false);
                if square && !forged { rem+=1; forged=true; }
                let y=mul(sub(add(v(input),s/2),rem),invq(s));
                pd.sel.insert(n,vec![rem]); Some(y)
            }
            BasicBlockType::NonNegative(_) => {
                let a=v(nd.inputs[0]); let limit=1u64<<g.lookups[&n];
                pd.sel.insert(n,vec![if a<limit {a} else {0}]); None
            }
            _=>panic!("unexpected exp operator"),
        };
        if let Some(y)=y { pd.vals[nd.outputs[0]]=Some(Arc::new(vec![y])); }
    }
    assert!(forged);
    g.outputs=vec![(out,pd.vals[out].as_ref().unwrap().clone())];
    assert!(!replay(&g,&pd));
}

#[test]
fn padded_exp_verifier_rejects_nonzero_claimed_padding() {
    let mut b=DagBuilder::new();
    // Positive padding must be masked BEFORE the nonpositive helper guard.
    let mut values=vec![int_to_f(0);5]; values.extend(vec![int_to_f(400);3]);
    let x=b.committed_input(Witness::new(vec![5],values,DataType::Float,*crate::SF_LOG,Role::Input));
    let out=b.exp(x)[0]; let (d,mut w)=b.compile();
    for &n in &d.topo {
        let nd=&d.nodes[n]; let ins=nd.inputs.iter().map(|&e|&w[e][0]).collect::<Vec<_>>();
        let outs=nd.kind.run(&ins);
        for (&e,v) in nd.outputs.iter().zip(outs) {w[e]=vec![v];}
    }
    let mut g=Graph::new(&d,&w,Config {tcl:4,slots_log:12,req_planes_log:6});
    let mut pd=ProverData::new(&g,&w,vec![K8::ZERO;1<<12]);
    assert_eq!(&pd.vals[out].as_ref().unwrap()[5..],&[0;3]);
    assert!(replay(&g,&pd));
    let mut bad=pd.vals[out].as_ref().unwrap().as_ref().clone(); bad[5]=1;
    let bad=Arc::new(bad); pd.vals[out]=Some(bad.clone()); g.outputs=vec![(out,bad)];
    assert!(!replay(&g,&pd));
}

# Morpheus repair draft — not inference certification

Base: `903d6e27be1d4a58c2a36a88917283dc2595b1da` (main rechecked 2026-10-30 UTC).
This is an implementation repair draft, not a reproduction of paper measurements,
not a complete soundness audit, and not ready to merge without the checks below.

## Changes

- Both request and deferred-finalization verifiers require authentication, even
  when diagnostic proof production disables it. Shape/canonical-field guards
  cover decoded and directly constructed proof values before indexing.
- ScaleUp adds the missing exact remainder equation. This fixes its local field
  relation, not the full signed integer no-wrap argument.
- An explicit AR descriptor binds public selector arrays and constants, validates
  contiguous vocabulary shards, unique one-hot selections and suffix shift.
  Logit bounds and vocabulary-scaled/index-offset rank comparisons implement
  lowest-index argmax under the checked modulus budget. Public tensor widths
  derive from their already-public values; private widths remain fixed.
- Sequential suffix generation replaces an unrelated randomly drawn suffix.
  This changes witness-generation cost; historical timings are not applicable.
- Benchmark table admission rejects missing authentication/verification, debug
  flags, wrong counts/IDs/batches, malformed measurements and failed processes.
  This is log admission, not cryptographic provenance or a benchmark rerun.
- CUDA div_z validates each floating estimate by exact multiplication before
  subtraction, bounds correction work, and falls back to restoring division.

## Trust boundary and unresolved correctness

`ArStatement::validate` is NOT a canonical numerical-program validator. The
application must independently pin the trusted DAG, graph wiring, model and
public statement, and explicitly select the AR API. Generic `run` / `ar: None`
proves no AR claim. The descriptor does not establish that arbitrary caller DAGs
connect embedding and selectors to the declared rank equations. It does not
repair unconstrained nonlinear normalization advice, complete causal masking,
all intermediate no-wrap bounds, source/program codec coverage, deterministic
key expansion versus a fresh-uniform setup theorem, or private-seed lifecycle.

In particular, InstanceNorm/Softmax/Sigmoid/RMS need specified deterministic
numerical relations and constrained advice. Choosing an integer approximation
must not be described as preserving existing f64 outputs. The sequential
witness may fail final validation until genuine causal/nonlinear relations are
fixed. No task-accuracy equality, accepted inference proof, GPU timings,
whole-process memory, or paper-row provenance is claimed here.

## Reproducible checks run on this draft

From repository root:

```sh
python3 -m unittest discover -s zk-torch-4/scripts/bench/morpheus -p test_evidence_admission.py -v
python3 zk-torch-4/scripts/audit/test_semantics_reference.py
python3 zk-torch-4/scripts/audit/test_div_z_host.py
bash -n zk-torch-4/scripts/bench/morpheus/run_models.sh zk-torch-4/scripts/bench/morpheus/run_extra.sh
git diff --check
```

Results: 23 synthetic log-admission tests pass; seven independent scalar-model
checks pass; 10,048 boundary/random cases of host-extracted `div_z` and
`div_z_exact` agree with Python arbitrary-precision division (g++ C++17, UBSan).
Shell syntax and whitespace checks pass. Host extraction replaces CUDA qualifiers
and two integer intrinsics; it is neither nvcc compilation nor GPU execution.
Scalar-model checks do not invoke Rust or verify actual proofs.

No Rust compiler/Cargo, nvcc, CUDA toolkit/runtime or GPU was available in the
execution environment. Standard alternate Rust locations were also absent.
No packages were installed and no GPU job was submitted. Added Rust wire tests
are unexecuted. An independent source review caught and corrected a misplaced
link guard before publication; this is not a substitute for compilation.

## Required before merge / correctness claims

1. Build using the project's Rust/CUDA environment (`cd zk-torch-4 && cargo build
   --release`); run Rust unit tests and check external constructors for added
   `Instance.ar` and `Prepared.ar` fields. Audit all warnings/errors.
2. Execute full verifier tests rejecting absent auth at request AND finalization;
   malformed decoded and in-memory proof shapes must reject without panic.
3. Execute complete proof mutations for ScaleUp (x=0,S=2,y=1), altered selectors,
   descriptor-reuse, noncanonical padding, shard boundaries, negative/tied logits,
   changed bound constants and range roots. Scalar tests are insufficient.
4. Compile with nvcc and compare actual GPU div_z/sampler outputs to exact CPU
   references across boundaries and fallback paths; check sanitizer results.
5. Complete numerical/causal/no-wrap repairs, prove canonical program binding,
   then rerun source/mask/query census and authenticated performance workloads.
   Persist full accepted proof bytes, binary/source identities and run manifests.

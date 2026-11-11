# Benchmark development pilot

The twelve files in `development/` cover the twelve templates in
[the benchmark design](../../docs/benchmark-design.md). They are **uncertified
candidates**, not a released evaluation dataset. Their manifests deliberately
record pending solver checks. In particular, T2-03/T2-04 have not been shown to
require their intended deduction, and guess-free completion has not been proved.

Run a scripted grounding infrastructure check from the repository root:

```powershell
python -m vembody run --task tasks/benchmark/development/t1-01.yaml
```

This should finish after one correct click and save both before/after screenshots.
It measures the harness and task predicate, not model performance. The four T1
candidates contain direct scripted target actions. Other candidates have no
reference action sequence yet.

T2/T3 runs require a trusted `LabelProvider`, injected through the Python sandbox
constructor. The CLI does not supply a solver and rejects those runs with a
one-line `error: CertificationRequired: ...` and exit code 1. All `evaluation`
split runs are blocked pending reviewed instance certificates.
Development T4 runs may grade a game outcome but do not establish logical accuracy
or guess-free solvability without a solver.

Candidate regeneration (overwrites the selected candidate directory):

```powershell
python scripts/prepare_benchmark.py
```

The generator selects physically valid starting states. It does not implement a
solver or certify reasoning difficulty. Frozen state hashes detect changes in
placement/setup behavior. See `manifest.json` for instance IDs and outstanding
checks. In-context examples and evaluation seeds must stay separate from these
development instances.

See [the handoff](../../docs/benchmark-handoff.md) for dataset coverage,
demonstration regeneration commands, and solver/model integration dependencies.

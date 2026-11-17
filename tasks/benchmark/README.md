# Benchmark development pilot

The twelve files in `development/` cover the twelve templates in
[the benchmark design](../../docs/benchmark-design.md). They are candidates, not a
released evaluation dataset.

The T2/T3 instances in every split are **solver-certified starts**: each one meets
its template's selection rule in `src/vembody/benchmark/certify.py`, so it always
has a provably correct move. T2-01/T2-02 need a single-clue deduction, T2-04 needs
a subset deduction, and T2-03 needs two clues combined. T3-01's outlined mines are
all provable at the start. T3-02 boards can be won using only provable reveals and
need at least ten clicks to win, so any sequence of provable reveals reaches ten
moves. Each manifest record holds the certificate; `certificate review` is the only
pending check. `tests/test_benchmark_certification.py` plays every T2/T3 instance
through the sandbox with the exact label provider. T1/T4 candidates remain
uncertified.

Run a scripted grounding infrastructure check from the repository root:

```powershell
python -m vembody run --task tasks/benchmark/development/t1-01.yaml
```

This should finish after one correct click and save both before/after screenshots.
It measures the harness and task predicate, not model performance. The four T1
candidates contain direct scripted target actions. Other candidates have no
reference action sequence yet.

T2/T3 runs require a trusted `LabelProvider`. Pass `--label-provider exact` to
`vembody run` or `vembody eval`; without it, those runs stop with a one-line
`error: CertificationRequired: ...` and exit code 1.
Development T4 runs may grade a game outcome but do not establish logical accuracy
or guess-free solvability without a solver.

Candidate regeneration (overwrites the selected candidate directory):

```powershell
python scripts/prepare_benchmark.py
```

Replace only the certified T2/T3 instances, keeping each split's T1/T4 files:

```powershell
$T = "T2-01","T2-02","T2-03","T2-04","T3-01","T3-02"
python scripts/prepare_benchmark.py --templates $T --output tasks/benchmark/development --split development --seed-start 0 --exclude tasks/benchmark/demonstration tasks/benchmark/evaluation
python scripts/prepare_benchmark.py --templates $T --output tasks/benchmark/demonstration --split demonstration --seed-start 10000 --exclude tasks/benchmark/development tasks/benchmark/evaluation
python scripts/prepare_benchmark.py --templates $T --output tasks/benchmark/evaluation --split evaluation --seed-start 20000 --instances 5 --exclude tasks/benchmark/development tasks/benchmark/demonstration
python scripts/export_benchmark_assets.py
```

The generator selects physically valid starting states. It does not implement a
solver or certify reasoning difficulty. Frozen state hashes detect changes in
placement/setup behavior. See `manifest.json` for instance IDs and outstanding
checks. In-context examples and evaluation seeds must stay separate from these
development instances.

See [the handoff](../../docs/benchmark-handoff.md) for dataset coverage,
demonstration regeneration commands, and solver/model integration dependencies.

# Cameron's tasks-and-data handoff

## Current status

The development infrastructure is implemented and independently reviewed. The
final evaluation benchmark is **not ready**: no instance has a reviewed logical
difficulty/guess-free certificate, the logical reference examples are pending,
and no pretrained model is integrated. These are reported dependencies, not
successful experiments. Existing mock demonstrations remain usable.

The branch is `cameron/tasks-and-data`. Changes are grouped into phase commits:

| Phase | Commit | Independent review outcome |
| --- | --- | --- |
| 1: design | `c562304` | Approved with mine-count and certification clarifications |
| 2: schema/reset | `eabe012` | Approved; reproducibility, validation, privacy, geometry checked |
| 3: predicates/pilot | `66da89f` | Infrastructure approved; logical certificates remain pending |
| 4: images/examples | `bfbda26` | Candidate assets approved; final evaluation assets pending |
| 5: integration checks | `d721852` | Infrastructure validation approved; 70 tests passed |
| 6: handoff/report drafts | This document and companion report | Documentation independently approved |

The separate review agent reran tests and inspected changes at each checkpoint.
Issues it found were corrected before phase commits, including malformed config
keys, private evidence filtering, late clicks after solver calls, duplicate visible
starts across splits, and incomplete image validation. This is implementation
review, not a scientific review of a model or an approval from a teammate.

## Where the deliverables are

- [Design](benchmark-design.md): environment rules and all twelve task definitions.
- [Task files](../tasks/benchmark/README.md): twelve development and twelve separate
  demonstration candidates, each with a fixed opening and state checksum.
- [Asset manifest](../data/benchmark/pilot/manifest.json): 48 raw/grid screenshots,
  visible-board reference transcriptions, dimensions, identities, and checksums.
- [Development overview](../data/benchmark/pilot/development-contact-sheet.png)
  and [demonstration overview](../data/benchmark/pilot/demonstration-contact-sheet.png).
- [Report drafts](cameron-report-sections.md): problem, environment, dataset, feasibility.
- `src/vembody/benchmark/`: validated configuration, deterministic setup, logical
  label-provider contract, seven task predicates, and grounding trace replay.
- `scripts/`: candidate preparation, image export, and integration checker.

## Coverage of all twelve tasks

Each row has one development candidate and one separate demonstration candidate.
Positive/negative predicate tests exercise the implementation, not the difficulty
of a generated instance. Logical predicate tests use explicitly synthetic label
fixtures; those are not a solver or certificates.

| Template | Predicate | Predicate tests | Actual mock-loop check | Remaining instance evidence |
| --- | --- | --- | --- | --- |
| T1-01 requested reveal | exact reveal | Positive + wrong target/type | One-click success + replay, both splits | Whole-board guess-free certificate |
| T1-02 requested flag | exact flag | Positive + wrong target | One-click success + replay, both splits | Whole-board guess-free certificate |
| T1-03 relative reveal | exact reveal, unique clue | Positive + wrong target | One-click success + replay, both splits | Whole-board guess-free certificate |
| T1-04 corner reveal | exact reveal | Positive + wrong target | One-click success + replay, both splits | Whole-board guess-free certificate |
| T2-01 safe deduction | one provable reveal | Synthetic proof + no proof | Missing-provider rejection, both splits | Easy-rule proof + reference trace + full-board certificate |
| T2-02 mine deduction | one provable flag | Synthetic proof + no proof | Missing-provider rejection, both splits | Easy-rule proof + reference trace + full-board certificate |
| T2-03 combined constraints | one provable reveal | Synthetic proof + no proof | Missing-provider rejection, both splits | Required multi-constraint difficulty + reference trace + full-board certificate |
| T2-04 subset deduction | one provable reveal | Synthetic proof + no proof | Missing-provider rejection, both splits | Required subset difficulty + reference trace + full-board certificate |
| T3-01 regional flags | flag region mines | Correct targets + safe-cell flag | Missing-provider rejection, both splits | Initial target proofs + reference trace + full-board certificate |
| T3-02 ten reveals | safe reveal count | Ten synthetic justified reveals | Missing-provider rejection, both splits | Ten-move logical path + full-board certificate |
| T4-01 small game | board won | Ground-truth fixture win + mine loss | Premature finish fails, both splits | Guess-free solution trace |
| T4-02 beginner game | board won | Ground-truth fixture win + mine loss | Premature finish fails, both splits | Guess-free solution trace |

All 24 candidates pass loader, reset repetition, state checksum, renderer pixel,
and asset-identity checks. Seeds, hidden layouts, and visible starting inputs are
distinct. T1 candidates also meet target safety/number and unique-clue conditions.
The other candidates meet physical setup requirements but are not verified as
having their intended deductions. Some may need replacement after solver checks.

The four demonstration T1 responses were executed during asset export. The other
eight demonstration responses are `null` with a pending status. No evaluation
split has been generated. The design's 96-instance target remains a future
expansion after the twelve-template pilot is certified.

## Run and verify

From the repository root, using an existing Python 3.11 environment:

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements/benchmark-runtime.txt -e ".[test,lint]"
.venv\Scripts\python.exe scripts/run_tests.py
.venv\Scripts\ruff.exe check src tests scripts
.venv\Scripts\ruff.exe format --check src tests scripts
.venv\Scripts\python.exe scripts/check_benchmark.py --output artifacts/benchmark-review-01
```

A local `.venv` is already prepared on Cameron's machine; its `.tools` Python
download and caches are ignored by Git. On another platform, use that platform's
virtual-environment executable paths. The committed image manifest records Python
3.11.14 and Pillow 13.0.0; runtime dependencies are pinned in
`requirements/benchmark-runtime.txt`. The checker compares actual pixels and
rejects renderer differences instead of assuming cross-platform equality.

The test runner creates a unique ignored `.pytest-tmp-*` directory inside
the project for each invocation and disables pytest's shared cache. This avoids
Windows `WinError 5` when the system `pytest-of-*` folder or `.pytest_cache` was
created under a different execution context. It runs the same full suite and
returns pytest's exit code. Test files stay in that run directory for inspection.
Use this runner instead of plain pytest on this machine; it does not repair or
delete the inaccessible shared directories. Additional pytest arguments may be
appended, such as `scripts/run_tests.py tests/test_loop.py -v`.

Use a fresh output directory per benchmark check because the JSONL logger appends. The
checker saves `report.json`, per-instance logs, and screenshots under that output.
It currently reports:

- 24 candidate instances with repeatable resets and matching assets.
- Eight successful grounding reference traces with replayed state/image evidence.
- Twelve expected missing-provider rejections (six templates in two splits).
- Four unsuccessful premature-finish full-game controls.
- `evaluation_ready: false`.

These are infrastructure counts, not success rates for a learned agent. To enforce
the release gate in automation:

```powershell
.venv\Scripts\python.exe scripts/check_benchmark.py --output artifacts/release-check-01 --require-certified
```

That command currently exits with code 2. It must stay unsuccessful until the
certificate review and release integration are implemented. Ordinary checks exit
successfully when the development infrastructure behaves as specified, even with
the documented provider gates.

## Rebuild candidates and assets

These commands overwrite the specified candidate/assets directories. They create
physically valid *uncertified* positions, not guaranteed solvable puzzles:

```powershell
.venv\Scripts\python.exe scripts/prepare_benchmark.py
.venv\Scripts\python.exe scripts/prepare_benchmark.py --output tasks/benchmark/demonstration --split demonstration --seed-start 10000 --exclude tasks/benchmark/development
.venv\Scripts\python.exe scripts/export_benchmark_assets.py
```

The demonstration generator excludes development visible states. Export also
rejects duplicate IDs, seeds, layouts, or visible states across its supplied sets.
Instance-based filenames allow multiple instances of a template without overwrites.
When replacing candidates, regenerate assets and rerun the checker before committing.
Do not tune prompts on eventual evaluation instances.

## Nicholas: solver and certification handoff

`LabelProvider.label(VisibleBoard)` receives an immutable visible grid and public
mine count; it receives no mine layout. Return `LogicalLabels` with frozensets of
provably safe cells and provable mines, plus a provider/version identifier. Unknown
hidden cells are undetermined. Flags must be treated as assertions, not facts.
Invalid records/provider errors stop logical execution as evaluation infrastructure
errors. Provider calls run synchronously: enforce a computation timeout inside the
solver. The runtime rechecks the episode deadline after labeling and sends no late
click, but cannot interrupt an arbitrary stalled Python call.

For development integration, inject a trusted provider explicitly:

```python
from vembody.sandbox import MinesweeperSandbox
from vembody.tasks import load_task

task = load_task("tasks/benchmark/development/t2-01.yaml")
sandbox = MinesweeperSandbox(task, "artifacts/solver-pilot", label_provider=provider)
```

`provider` is Nicholas's implementation; this example does not define a solver.
The CLI does not automatically discover/import one. Certification must additionally
verify each template's selection rules, preserve reference traces, check the public
mine-count constraint in easy-rule closure, and prove guess-free completion from
the fixed opening. A runtime label provider alone is not an instance certificate.

The evaluation-split runtime gate is deliberately unconditional today. Replace it
only with a reviewed certificate-loading/verifying path bound to instance state,
solver/rules versions, and code revision. A YAML field saying `certified` is not
sufficient. Add solver-backed integration tests before opening this gate.

## Josh and Anant: model/runtime handoff

- Keep the model-facing task/history projection in `AgentLoop`. Private scoring
  evidence, mine layouts, setup moves, and reference answers must not enter prompts.
- Current runnable actions use normalized full-image coordinates. Symbolic G actions
  need a parser/adapter; this branch exports G images but does not implement G/T
  model prompting, state-transcription output, or a pretrained inference provider.
- Benchmark 9x9 images are 336x376; 5x5 images are 208x248. Grid origin is `(36, 76)`,
  cells are 32 pixels, and the outer two pixels of each cell are excluded as input.
  Raw/G images share geometry. Human-play dimensions and chording remain unchanged.
- Preserve the original response, converted target, prompt/configuration, model
  identity, decoding parameters, latency, and any resize settings in the model-side
  trace. Prompt/token/provider metadata logging is not yet implemented here.
- Label cell actions against the exact screenshot geometry used for that proposal.
  Set request deadlines and token limits in the inference client; the loop's
  post-response check alone cannot interrupt stalled inference.
- Task success can end an episode before a whole-game win. Observe `task_done`
  separately from `episode_done`. Early finish, parse errors, action/time limits,
  and ineffective inputs must not count as success.

Grounding replay currently recomputes T1 outcomes only, including before/after
images and state transitions. Logical replay, aggregate metrics, confidence
intervals, Slurm jobs, and repeated learned-model trials remain evaluation-team
integration work. JSONL artifacts and candidate assets stay clearly separate from
published model results.

## Next release conditions

1. Connect Nicholas's bounded logical provider and certify or replace the pilot.
2. Complete solver-backed reference actions/examples for T2-T4 and strengthen
   template-specific integration traces with those certificates.
3. Validate one real-model episode with Josh/Anant and calibrate time budgets on
   development instances.
4. Select the example policy, expand the dataset, deduplicate all splits together,
   and freeze a separate evaluation manifest.
5. Implement/review certificate verification and logical replay before enabling
   evaluation runs. Then run the paired variants and write measured results.

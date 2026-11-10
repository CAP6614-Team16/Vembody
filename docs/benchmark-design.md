# Minesweeper benchmark design

Status: Phase 1 design approved with subsequent independent implementation reviews.
The development infrastructure and candidate screenshots are now implemented.
Solver certificates and measured model results remain pending. See
[the handoff](benchmark-handoff.md) for actual coverage and release status.

This design translates the team's *Pixels to Clicks* proposal into a concrete
tasks-and-data contract. It covers Cameron's responsibilities: task definitions,
repeatable starting positions, screenshots, demonstration examples, and objective
success criteria. Nicholas owns solver certification and evaluation implementation;
Josh owns model integration and prompting; Anant owns runtime integration.

## 1. Scope and current implementation

The benchmark uses Minesweeper only. Its twelve task templates cover grounding,
single deductions, partial solves, and full games. A template describes a skill;
an instance supplies a particular board and starting position. All agent variants
must run on the same frozen instances.

The repository baseline at Phase 1 provided:

- A Python game with seeded, first-click-safe mine placement, flood fill, flags,
  chording, and automatic mine flags when the game is won.
- A Pillow-rendered screen, a browser interface for human play, and an agent
  sandbox that directly delivers pixel clicks to the Python screen.
- YAML task loading, normalized action coordinates, a scripted mock model,
  validation, an episode loop, screenshots before actions, and JSONL logs.
- Win/loss evaluation, but no task-specific success predicates or logical solver.

The proposal describes a forked web game controlled through Playwright, absolute
pixel actions, and a hidden-state endpoint. The proposed implementation for this
benchmark keeps the existing Python renderer, normalized coordinates, and an
evaluator-only state interface. It still tests image observations and pixel input,
but does not establish browser automation performance. This is an explicit change
from the proposal that Cameron and the team should review before Phase 2.

`tasks/example.yaml` and `tasks/minesweeper-example.yaml` remain demonstrations;
they are not members of the evaluation benchmark. The Minesweeper demonstration
ends with a scripted `fail` action and is not a reference solution.

## 2. Environment contract

### Board identity and reset

Each instance fixes all of the following:

- Template ID, unique instance ID, split, instance format version, and rules version.
- Rows, columns, mine count, generation seed, fixed opening cell, and ordered setup
  moves. A seed alone does not identify a board: current placement excludes the
  first clicked cell, so the opening must also be fixed.
- The accepted mine layout and complete starting state, including revealed and
  flagged cells and game status. Store a checksum of a canonical state encoding.
- Renderer version, Python/Pillow versions, font identity, geometry, and screenshot
  pixel checksum. Record the code commit when the instance is generated.

Reset creates a fresh game, applies the fixed opening and setup moves, and checks
the resulting state against the saved instance. All setup happens before agent
step zero and is excluded from the action budget and timed interaction episode.
Record setup duration separately. The opening and setup must never lose or finish
the game, and the task predicate must initially be false.

For released evaluation instances, load the frozen layout/start state or verify
that reconstruction matches it exactly; never silently substitute another board.
Each variant receives the same starting state. A face-button click is not an
approved reset mechanism during an episode.

### Actions and game behavior

- Benchmark actions are `click`, `right_click`, and `finish`, matching current
  action names. A model-facing cell action for the G variant is a later adapter;
  cell syntax is not supported by the existing parser.
- `click` reveals one selected hidden, unflagged cell. Standard zero-cell flood
  fill remains enabled. One reveal action can therefore open multiple cells.
- `right_click` toggles a flag on a hidden cell. Unflagging is allowed, recorded
  separately, and never counts as placing a correct new flag.
- Chording is disabled for benchmark episodes so a click on a revealed number
  cannot reveal neighbors. Human play may retain its current behavior.
- A click on a revealed cell, a reveal on a flagged cell, or a right click on a
  revealed cell is an ineffective input; it consumes one action attempt.
- Only cell interiors are valid input targets. Reject clicks on grid lines,
  margins, counters, or the face button; do not clip them into a different target.
- `finish` requests an independent task check. It does not establish success.
  Premature finish is an unsuccessful episode. Existing nonbenchmark `fail`
  support is unchanged but is not offered as a benchmark action.

Every attempted model action consumes one step, including malformed output and
ineffective input. The initial policy is fail-fast on malformed or disallowed
actions, matching the current loop; there are no repair calls or retries. If the
team later enables bounded recovery, apply the same policy to every variant and
version the benchmark protocol. Effective but incorrect task actions are governed
by the per-task rules below.

### Coordinates and rendering

- Human cell names use letters for rows from top to bottom and numbers for columns
  from left to right: `A1` is the top-left cell. Stored grid indices are zero-based
  `(row, col)`, so `C5` is `(2, 4)`.
- Baseline actions use normalized coordinates of the full screenshot. Convert
  with `px = round(x * (width - 1))` and `py = round(y * (height - 1))`.
- G actions name a cell, which the adapter maps to its center using the geometry
  of that exact observation. Save the original cell response and converted pixels.
- Keep 32-pixel cells, a 12-pixel side/bottom margin, and a 52-pixel header for the
  initial renderer. Human-play/base images are 312x352 for 9x9 and 184x224 for 5x5.
- Reserve identical extra gutters for row/column labels in every variant. Phase 2
  fixed these at 24 extra pixels on the left and top (336x376 for 9x9, 208x248 for
  5x5). G variants draw labels there; baseline
  variants leave them blank. This avoids moving or resizing the grid between
  variants. Final coordinate mappings must use the new screenshot dimensions.
- Freeze font, palette, border widths, and RGB PNG output. Do not rescale screenshots
  between variants. Save any model-side resize settings alongside the run.
- Task highlights are part of the environment image and appear in every variant.
  Grid labels are the G treatment and are not added to baseline images.

Pixel equality is checked within the pinned renderer environment. A different
renderer/runtime must pass the reference-image checks or create a new version;
we do not assume fonts render identically on arbitrary machines.

### Agent and evaluator information

The agent receives the instruction, screenshot, public rules, and configured
history of its own outputs/actions. Public board dimensions and total mine count
may appear in the common prompt for all variants. Mine locations, certified safe
cells, expected solutions, setup moves, and success targets derived from hidden
state are evaluator-only. The text-board oracle is a separately labeled condition
and receives the true *visible* board, never the hidden layout.

The model interface uses `TaskSpec` and `StepResult` history. The implemented
agent-facing projection strips private instance fields, evaluator results, and
execution evidence from both before model calls. Merely putting answers in an
`evaluation` field in YAML does not prevent leakage.

Use `#` for hidden cells, `F` for flags, `.` for revealed zeros, and `1` through `8`
for revealed numbers in visible-board records and transcription examples. A flag
is an agent assertion, not evidence that a mine is actually present.

## 3. Episode evaluation rules

### Success, failure, and ordering

After each executed input, save the state transition, assess loss/task violations,
then check the task predicate. A mine loss or persistent task violation takes
precedence over success. A satisfied predicate ends the episode immediately; the
agent does not need to emit `finish`. `finish` sends no mouse input and checks the
current state. Deadlines are checked before dispatching an action after inference,
as well as between steps; an expired response must not produce a late click.

If no predicate is satisfied, exhaustion of steps or time is unsuccessful. Record
termination separately from success. Missing assets, reset mismatches, renderer
errors, and solver timeouts are infrastructure failures and must be reported
separately rather than assigned fabricated agent correctness labels. Record their
counts and any reruns.

This requires changes to the current loop: it currently evaluates only on finish
or game over, checks time only at step boundaries, and uses `episode_done` for
environment termination. Task completion must remain distinguishable from game
win/loss, and the final post-action screenshot must be saved.

### Logical correctness and certification

The solver derives constraints from the true visible numbers and the public total
mine count. Hidden locations are used to validate outcomes, never to derive the
logical labels. Unproven flags remain Boolean mine variables. Trusted solver
deductions may be reused only if their derivation is preserved. In particular,
subtracting every agent flag as a known mine would incorrectly reward bad flags.

Before each relevant action, classify the target as provably safe, provably a mine,
or undetermined across all layouts consistent with the visible evidence. An
effective reveal is logically correct only if its target was provably safe; a
newly placed flag is logically correct only if its target was provably a mine.
Unflags, ineffective actions, `finish`, and terminal automatic flags are not included
as successful logical moves. Keep their counts separately; publish denominators.

T1 is an instruction-grounding test: logical move accuracy is not its success
criterion and should be reported as not applicable there. T2 through T4 require
solver labels. Labeling deadline failures are unknown, never undetermined by default.

Full-game instances require a saved solver trace proving completion from their
designated opening without guesses. T2 and T3 require traces for their target
deductions/solutions, with whole-board guess-free certification additionally
required before inclusion in the proposal's final benchmark. A board seed and a
safe first click alone are not certification. Candidates remain unpublished as
certified evaluation instances until Nicholas's solver verifies them.

## 4. Twelve task templates

All budgets below are proposed initial values. The action budgets follow the
proposal. Time budgets are generous starting limits, not measured latency claims;
calibrate them on development instances with Josh, then freeze before evaluation.
Times cover screenshot capture, inference, validation, execution, and grading during
the episode. Solver deadlines and environment failures are also logged separately.

| ID | Tier | Instruction | Board | Steps | Seconds |
| --- | --- | --- | --- | ---: | ---: |
| T1-01 | Grounding | Reveal the cell in row C, column 5. | 9x9, 10 mines | 3 | 120 |
| T1-02 | Grounding | Flag the cell in row G, column 2. | 9x9, 10 mines | 3 | 120 |
| T1-03 | Grounding | Reveal the hidden cell directly to the right of the only revealed 3. | 9x9, 10 mines | 3 | 120 |
| T1-04 | Grounding | Reveal the top-right corner cell. | 9x9, 10 mines | 3 | 120 |
| T2-01 | Single deduction | Reveal one hidden cell that is guaranteed safe. | 9x9, 10 mines | 3 | 120 |
| T2-02 | Single deduction | Flag one hidden cell that is guaranteed to contain a mine. | 9x9, 10 mines | 3 | 120 |
| T2-03 | Combined constraints | Reveal one hidden cell that is guaranteed safe. | 9x9, 10 mines | 3 | 120 |
| T2-04 | Subset deduction | Reveal one hidden cell that is guaranteed safe. | 9x9, 10 mines | 3 | 120 |
| T3-01 | Partial solve | Flag every mine among the hidden cells touching the yellow outlined region. | 9x9, 10 mines | 20 | 600 |
| T3-02 | Partial solve | Make 10 safe reveal moves without losing. | 9x9, 10 mines | 25 | 600 |
| T4-01 | Full game | Reveal every safe cell without revealing a mine. | 5x5, 3 mines | 40 | 900 |
| T4-02 | Full game | Reveal every safe cell without revealing a mine. | 9x9, 10 mines | 120 | 1800 |

The next sections are mandatory instance-selection conditions and predicates, not
claims that suitable boards have already been found.

### T1: Grounding (four templates)

All starts are nonterminal prepared positions. Indices count from the board edges
even without G labels. Public instructions and targets stay identical across
variants. All four instances remain solvable for the later whole-board certificate.

- **T1-01:** C5 is hidden, unflagged, safe, and numbered (not zero), avoiding flood
  fill. Success requires an effective reveal of C5 and no other cell changes.
- **T1-02:** G2 is an unflagged, hidden mine. Success requires adding its flag and
  no other cell changes. The agent need not infer that it is a mine; this tests the
  requested target and action type.
- **T1-03:** Exactly one revealed cell displays 3. Its immediate right neighbor
  exists and is hidden, unflagged, safe, and numbered. Success requires revealing
  that neighbor and no other cell changes. Store the clue cell and target privately.
- **T1-04:** The top-right cell (A9) is hidden, unflagged, safe, and numbered.
  Success requires revealing A9 and no other cell changes.

For T1, any effective board change other than the requested operation is a
persistent task violation and ends the episode unsuccessfully. Reversing an
incorrect flag does not erase that violation. Ineffective clicks consume budget.
The required predicates exclude game-winning targets so terminal auto-flags cannot
confound the exact-change requirement.

### T2: One deduction (four templates)

Each start has no pre-applied flags. Success/failure is decided on the first
effective board-changing action. Wrong action type or a target lacking the required
proof ends the task unsuccessfully, even if the revealed cell happens to be safe.
Ineffective clicks do not become successes and consume budget.

- **T2-01:** At least one hidden safe target is certified by a single local clue
  rule (including repeated application of such rules). Accept any effective reveal
  whose target is provably safe in the initial visible position.
- **T2-02:** At least one mine is certified by single-clue propagation. Accept any
  newly placed flag whose target is provably a mine in the initial visible position.
- **T2-03:** Single-clue propagation, including the public total mine-count rule,
  reaches a fixed point with no safe target;
  combining at least two clue constraints proves a safe target. Save a certificate
  identifying those constraints. Accept any provably safe initial target.
- **T2-04:** Single-clue propagation, including the public total mine-count rule,
  reaches a fixed point with no safe target;
  subtracting a proper-subset constraint from another proves a safe target. Save
  both neighbor sets, residual mine counts, and the difference deduction. Accept
  any provably safe initial target.

T2-03 and T2-04 are different instances, but their reasoning categories may overlap:
subset reasoning itself combines constraints. Report them as task categories, not
as mathematically disjoint reasoning abilities. If the solver cannot locate a
candidate meeting the selection condition, report that dependency rather than
quietly substituting an easier board.

Flood fill is allowed: success scores the selected cell before the action, not
the number of cells opened afterward. New constraints from that reveal cannot
retroactively justify the selected move.

### T3: Partial solves (two templates)

- **T3-01:** Highlight a small connected group R of already revealed clue cells
  with a yellow outline that does not cover numbers. Define U0 as the cells hidden
  at episode start that touch R, using eight-neighbor adjacency. Define P as the
  actual mines in U0. Require P to be nonempty, all its members to be provably mines
  initially, and none to be initially flagged. The instruction's target set stays
  U0 for the episode. Success requires flags on every member of P, no flags on
  safe cells anywhere, and no mine loss. Any newly placed safe-cell flag is a
  persistent task violation. Reveals and mine flags outside U0 are allowed but
  do not satisfy the target requirement. The instance manifest records R, U0, P,
  and a reference sequence privately; all variants see the same outline.
- **T3-02:** Start from a fixed opening that permits a reference sequence of at
  least 10 effective, logically safe reveal actions before terminal auto-completion.
  Success requires 10 such agent actions without a loss. Each selected target must
  have been hidden, unflagged, and provably safe immediately before the click.
  Flood fill counts as one move, regardless of cells opened. An undetermined
  reveal is a persistent task violation even if it survives. Flags may assist
  deduction, but a newly placed safe-cell flag is a persistent violation. Unflags
  and repeated clicks do not increase the counter. If the board is won in fewer
  than 10 qualifying actions, this task is unsuccessful; report that outcome
  separately and verify during selection that a valid 10-action path exists.

T3-02 makes "safe moves" mean logically justified reveals, rather than lucky
survival. This is an explicit operational definition for review. Task success and
whole-game success remain distinct.

### T4: Full games (two templates)

T4-01 uses a 5x5 board with three mines; T4-02 uses a 9x9 board with ten mines.
Both start after the same fixed opening for all variants and require a certified
guess-free reference solution. Success is exactly the game state `won`: every
non-mine cell is revealed and no mine was hit. Flags are optional.

An undetermined but lucky reveal does not negate T4 success; it lowers logical
move accuracy. Wrong flags may be corrected. Distinguishing game outcome from
move justification preserves the proposal's success-versus-luck comparison. The
game's automatic terminal flags are not credited as agent flag actions.

## 5. Proposed task/instance records

Extend the existing loader compatibly in Phase 2; the following is a design sketch,
not YAML that the current runtime understands. Avoid executable predicate strings.
Use named predicate types with validated parameters.

| Record | Proposed contents | Visibility |
| --- | --- | --- |
| Template | ID, tier, instruction, board family, budgets, allowed actions, predicate type | Instruction/rules public; scoring configuration private |
| Instance | ID, template ID, split, dimensions, seed, opening, setup moves, versions | Only explicitly selected public fields reach the agent |
| Starting state | Frozen layout, revealed/flagged sets, status, canonical checksum | Evaluator/setup only |
| Task targets | Requested target where applicable, region sets, target sets, violation rules | Private except information expressed by instruction/image |
| Certification | Solver version, rule policy, reference trace, label/proof checks, status | Evaluator only |
| Image manifest | Raw and labeled image paths, geometry, pixel checksums, font/runtime identity | Agent receives selected image only |
| Run evidence | Instance ID, variant, outputs, actions, timing, before/after states and images, labels | Private evaluation trace; own-action history is filtered |

Predicate types correspond to `exact_reveal`, `exact_flag`, `one_provable_reveal`,
`one_provable_flag`, `flag_region_mines`, `safe_reveal_count`, and `board_won`.
Predicate parameters are data. The evaluator maintains counters and persistent
violations from the action trace, not just the final board.

Logical labels and true-visible transcriptions must be captured before each
action. Evidence must distinguish the raw proposed target, resolved cell/pixels,
actually executed operation, state change, and environment-generated effects.
The JSONL trace must retain invalid/ineffective attempts and the final screen so
scores can be recomputed offline. Full private trace data must not enter the
model's history.

## 6. Splits, screenshots, and examples

Start with a twelve-template development pilot (one instance per template).
Certify and validate that pilot before expanding to the following proposed counts:

| Split | Instances per template | Total | Use |
| --- | ---: | ---: | --- |
| Development | 2 | 24 | Debugging, prompt iteration, budget calibration |
| Demonstration | 1 | 12 | Candidate pool for in-context examples |
| Evaluation | 5 | 60 | Frozen paired comparison across agent variants |

This yields 96 prepared instances, not 96 model experiments. Under greedy decoding,
the five evaluation instances are five different boards per template, rather than
five identical repetitions. The four core variants plus the text-board oracle
would produce 300 evaluation episodes. Actual runtime must be measured in a pilot.
If certification effort or compute forces a smaller set, version and document the
reduction before collecting evaluation results.

Use non-overlapping seeds across splits and compare canonical hidden-layout and
visible-start-state identities, ignoring IDs and filenames, to catch duplicate
boards or positions. Do not reuse evaluation positions in demonstrations or
prompt tuning. Freeze the instance manifest and checksums before the main run.

Every instance gets an initial RGB PNG and a manifest entry. Run evidence also
saves pre-action and post-action images, including the terminal transition. Raw
and G-label variants depict the same state at the same scale. T3-01's task outline
must be present in both. The evaluator's answer annotations belong in separate
review images, never agent observations.

Choose a small demonstration subset with Josh after model integration. Each
example contains its instruction, screenshot, verified visible transcription,
short deduction where applicable, and one valid action. Keep the source instance
and reference proof in private metadata. B0/G/T/GT prompts use matched source
positions and a fixed example count; transcription content belongs only in T
conditions. One-shot content and token differences are part of the protocol and
must be reported. Alternatively, start all core variants zero-shot and treat
examples as a separately controlled experiment. The choice is frozen before runs.

## 7. Ownership and integration dependencies

| Owner | Work needed for this benchmark | Handoff |
| --- | --- | --- |
| Cameron | Task catalog, instance selection/setup, split manifest, screenshots, examples, report sections | Versioned templates/instances and acceptance evidence |
| Nicholas | Constraint solver, guess-free certificates, logical labels, predicate scoring, metrics | Solver/evaluator interfaces and certificate format |
| Anant | Compatible loader, deterministic reset, action restrictions, task termination, filtered model view, logging | Working loop on scripted reference cases |
| Josh | Renderer/prompt coordination, local VLM, G/T adapters, inference timing | Image/action contract and development pilot results |

These are coordination boundaries, not claims that teammates have approved or
completed work. Phase 2 can implement Cameron's task/reset support against agreed
interfaces. Solver work remains Nicholas's ownership; unimplemented integrations
must be explicit. Do not silently replace the solver with hidden-state answers.

## 8. Acceptance checks and remaining phases

### Phase 1: this document

- Twelve unique IDs: four T1, four T2, two T3, two T4.
- Every template specifies an instruction, starting requirements, budgets, success
  predicate, and failure rules; all share a deterministic reset contract.
- Proposed behavior is distinguished from current behavior and uncertified data.
- Renderer choice, disabled chording, budgets, T3-02 semantics, and example policy
  are exposed for review before implementation.

### Phase 2: task configuration and resets

Implement compatible configuration validation and deterministic setup. Verify
repeated resets against the canonical state and image pixels in the pinned
environment, reject impossible or already-completed starts, check cell mapping
at edges/centers, and confirm both existing demonstrations still load/run.

### Phase 3: instances and success checks

Implement templates and prepare instances. Coordinate solver certificates and
scoring with Nicholas. For each predicate run positive and negative traces:
correct target, wrong target/type, ineffective input, premature finish, action/time
limit, and mine loss as applicable. Include flood fill, toggled flags, repeated
clicks, task completion without a whole-game win, and automatic terminal flags.
Check that private answers never enter the model input or history.

### Phase 4: screenshots and examples

Produce the split manifest and image/example assets. Inspect images for readable
numbers, unambiguous clues, unclipped labels/highlights, and matching geometry.
Check transcription accuracy, seed/state deduplication, paths, and checksums.

### Phase 5: integration validation

Execute every certified reference solution through screenshot capture and pixel
clicks in the agent loop. Repeat resets, exercise unsuccessful traces, and recompute
scores from saved evidence. Reference solutions are test fixtures, not measured
model performance. The final model pilot is a separate integration dependency.

### Phase 6: report and handoff

Draft problem, environment, dataset, and feasibility sections from the verified
implementation. State instance counts, versions, certification limits, actual
renderer, measured pilot latency, and outstanding dependencies. Do not claim
procedural generation proves absence of training-data contamination or that a
reference solver trace is a minimum-action solution.

All phases stay on `cameron/tasks-and-data`, with one cohesive commit per completed
phase where practical. At Cameron's request, a separate review agent now checks
coverage and evidence at each phase checkpoint. Fix its blocking findings before
proceeding and report approvals and outstanding dependencies to Cameron.

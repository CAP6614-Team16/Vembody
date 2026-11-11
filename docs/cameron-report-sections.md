# Draft report sections: tasks and data

Status: draft grounded in the current development implementation. Candidate
preparation and infrastructure checks are complete; solver certification and
learned-model experiments remain pending. Do not present this as the final
experimental results section. See [the handoff](benchmark-handoff.md) for evidence
and dependencies.

## Problem definition

We study whether a locally hosted pretrained vision-language model can complete
Minesweeper tasks through screenshot observations and discrete mouse actions.
The intended agent receives a natural-language instruction, an RGB image of the
current game, public board dimensions and mine count, and an optional bounded
history of its own actions. It does not receive hidden mine locations, reference
solutions, or the evaluator's logical labels. The implemented runtime currently
uses a scripted mock provider; pretrained inference is a separate integration task.

The benchmark contains twelve task templates grouped into four tiers: four
instruction-grounding tasks, four single-deduction tasks, two partial-solve tasks,
and two full-game tasks. A task template defines its instruction, budget, and
success rule. Each instance supplies a fixed board and prepared starting position.
This distinction supports testing the same capability on multiple independent
positions and pairing identical instances across agent variants.

Agent actions reveal a selected cell, toggle a flag, or request completion. Current
runtime coordinates are normalized to the full screenshot; they are converted to
pixels before input. The proposed G condition instead names a labeled cell and
requires a model-facing adapter. Success is established by the evaluator rather
than by the model's completion declaration. A task ends when its predicate is
satisfied, a disqualifying action occurs, the game ends, the agent requests finish,
or a budget is exhausted. Infrastructure errors are recorded separately.

The planned comparison retains the proposal's direct visual baseline, grid
anchoring, explicit board transcription, their combination, and a separately
labeled visible-text-board oracle. This branch prepares the tasks and image
variants but does not implement or measure those model conditions.

## Environment

We use the repository's existing Python Minesweeper engine and Pillow renderer.
This differs from the proposal's planned forked browser game and Playwright
execution. Automated episodes interact with rendered images and pixel clicks on
the Python screen interface; the repository also provides a browser interface for
human play. Consequently, the current system supports a controlled visual-control
study and does not establish performance on browser event handling or general
desktop automation.

The pilot uses 9x9 boards with ten mines and a 5x5 full-game family with three
mines. Every instance specifies a seed, a fixed opening, optional ordered setup
moves, and a starting-state checksum. Mines are placed relative to the first
reveal, so the opening is part of instance identity. Setup occurs before the agent
episode and is excluded from its interaction budget. Reset reconstructs a fresh
game and rejects terminal or mismatched starting states.

Cells are 32 pixels wide. Benchmark images reserve equal row/column label gutters
for all variants, yielding 336x376 images for 9x9 boards and 208x248 for 5x5 boards.
Raw and labeled observations share geometry; only label pixels differ. A task's
yellow region outline, when present, is displayed in both variants. Runtime/font
identity and image checksums are saved to detect rendering changes.

Standard zero-cell flood fill remains enabled, so one reveal action can open
multiple cells. Benchmark chording is disabled, and clicks on the face button,
margins, or cell borders are rejected: they change nothing, consume one step, and
are recorded as off-target agent actions rather than infrastructure errors, so a
model's grounding misses count against it instead of being excluded from the
success denominator. Human-play behavior is preserved. Reveal
grounding targets are numbered safe cells so the target-only success rule is not
confounded by flood fill. Partial-solve move counts count effective reveal actions,
not the number of newly visible cells.

Task scoring distinguishes actual game outcomes from logical justification. An
eventual solver will classify a cell using only visible clues and the public mine
count. A lucky reveal on an undetermined cell may contribute to full-game success
while lowering logical move accuracy. Hidden truth is used to grade mine hits and
target outcomes, not to manufacture logical proofs. Provider-dependent tasks are
blocked until a trusted label provider is connected; all evaluation-split runs
remain blocked pending a reviewed certificate path.

## Dataset and preparation

The current pilot contains 24 procedurally prepared candidates: one development
and one separate demonstration instance for each of the twelve templates. No
evaluation split has been released. Candidate generation checks physical setup
and target conditions, but does not prove reasoning difficulty or guess-free
completion. Positions that fail the future solver checks must be replaced rather
than relabeled as certified.

Each candidate has a YAML definition, a frozen state identity, an initial raw image,
and a matching grid-labeled image. The exported manifest includes visible-board
transcriptions, pixel/file checksums, geometry, and certification status. It is an
evaluator/development asset; reference transcriptions are not baseline model
inputs. We compare seeds, hidden layouts, and visible starting states across the
two sets, preventing a demonstration from reusing an identical visible position
under a different hidden layout or instance ID.

Four grounding demonstration responses were executed successfully against their
task predicates. The remaining eight logical/full-game demonstration responses
are explicitly pending solver-backed reference traces. Number-class coverage is
not guaranteed by this small pilot. Procedural generation provides reproducibility
and control over task construction; it does not prove absence of overlap with
pretraining distributions or previously seen puzzle patterns.

The planned expanded dataset is 24 development, 12 demonstration, and 60 evaluation
instances, for 96 in total. This is a target rather than an implemented release.
Expansion follows solver certification and a real-model pilot. Evaluation positions
must be excluded from prompt tuning and demonstrations. With five distinct
evaluation instances per template, four core variants and the text-board oracle
would yield 300 episodes before optional experiments.

## Feasibility and current evidence

The task infrastructure runs without model weights or GPU resources. It provides
validated task loading, repeatable setup, seven predicate types, screenshots,
private transition logs, and an interface for a logical label provider. The current
test suite has 75 passing tests, covering positive/negative predicates, reset and
image consistency, invalid configurations, information separation, time/action
limits, malformed responses, off-target input, and failure logging.

An integration checker verifies all 24 candidate assets and resets. It executes
and replays eight grounding traces, verifies twelve missing-provider rejections,
and checks four unsuccessful premature-finish full-game episodes. These checks
validate infrastructure behavior, not learned-agent accuracy. No GPU latency,
VRAM requirement, success-rate comparison, confidence interval, or cluster runtime
has been measured in this work.

The remaining critical dependencies are a bounded constraint solver, instance
certification/reference traces, pretrained model integration, and experiment
orchestration. The solver must certify the intended deduction and whole-board
solvability from the designated opening. The model pilot must establish image
handling, output validity, and measured latency before budgets and compute costs
are finalized. The proposal's compute estimate should remain an estimate until
those measurements are available.

If full-board performance or integration time is limiting, prioritize certified
grounding and single-deduction tasks with matched baseline/G/T/GT comparisons.
Optional verification and model/visual-scale experiments can be deferred. Any
reduction of task count, instance count, or experimental conditions must be stated
explicitly in the report; infrastructure tests must not be substituted for model
results.

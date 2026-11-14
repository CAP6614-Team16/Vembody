# Runtime Overview

- `python -m vembody run`: starts one bounded task episode.
- `--task`: selects the YAML or JSON task file loaded by `src/vembody/tasks.py`.
- `--model`: selects `mock`, `random`, `random-cell`, `solver`, `http`, `ui-mate`, `ui-mate-legacy`, or `ui-mate-brief` through `models/registry.py`.
- `--model-config`: selects Anant's endpoint or the hosted alternative for `--model http` (see [model access](model-access.md)).
- `--executor`: selects how actions are applied, currently `dry-run` from `src/vembody/execution/dry_run.py`.
- `python -m vembody eval`: runs every task for N trials as one experiment under `runs/<name>/` (see `docs/evaluation.md`).
- `eval --tasks tasks/benchmark --split development`: selects that split's manifest; frozen board starts remain unchanged across trials.
- `eval --name`: requires a fresh output folder, including after an interrupted experiment.
- `python -m vembody analyze`: compares experiment folders and writes tables and plots.
- `python -m vembody play`: serves a playable minesweeper game in the browser.
- `minesweeper/game.py`: Minesweeper rules with seeded, first-click-safe mines, flood fill, and chording.
- `minesweeper/screen.py`: draws the game window and maps pixel clicks to cells, the face button, or nothing.
- `minesweeper/webui.py`: standard-library HTTP server that shows the screen and forwards browser clicks.
- `sandbox.py`: the agent harness; saves screenshots, converts normalized clicks to pixels, ends the episode on win or loss, grades from the true game state, and records per-click board progress for evaluation.
- `tasks.py`: loads task instructions, limits, allowed actions, mock action sequences, and minesweeper board settings.
- `observations.py`: defines the typed observation passed to the model.
- `perception/file.py`: provides deterministic observations from an optional screenshot path.
- `models/base.py`: defines the interface every future vision-language model must implement.
- `models/mock.py`: returns the task's predefined actions without model inference.
- `models/baselines.py`: reference policies (random click, random hidden cell, perfect-perception solver).
- `models/registry.py`: builds a model by name for `run` and `eval`.
- `models/settings.py`: validates endpoint profiles and resolves environment variables without storing API keys.
- `models/http.py`: sends the current screenshot to a compatible vision endpoint with bounded socket waits and no retries or redirects.
- `models/prompts.py`: defines the versioned screenshot-only baseline prompt and allowed action parameters.
- `models/ui_mate.py`: safely translates native UI-Mate XML calls into one validated action without executing Python.
- `models/ui_mate_contract.py`: implements the pinned coordinate, image-sizing, and restricted prompt contract independently of the executor.
- `notebooks/kaggle-ui-mate-q6.ipynb`: runs the pinned Q6_K model and sandbox together on two T4s (see [Kaggle setup](kaggle.md)).
- `scripts/kaggle_grounding.py`: evaluates the four T1 development tasks and summarizes first-action accuracy, coordinate errors, validity, and latency.
- `scripts/check_ui_mate_compatibility.py`: checks adapter contracts offline and audits coordinate conversion in archived results without executing their contents.
- `scripts/kaggle_autonomous.py`: runs one high-level complete-game task with independent grading and fixed action/time limits.
- `scripts/kaggle_cache.py`: exports and validates reusable compiled runtimes without model weights or credentials.
- `scripts/kaggle_run.py`: launches prepare, original pilot, grounding, or autonomous evaluation inside the isolated environment.
- `actions.py`: defines the typed, allowlisted action schema and serialization helpers.
- `parsing.py`: converts one model JSON response into a validated typed action.
- `validation.py`: enforces task permissions, normalized coordinates, and safety limits.
- `execution/base.py`: defines the interface every action executor must implement.
- `execution/dry_run.py`: records validated actions without controlling the computer.
- `execution/desktop.py`: reserves the boundary for future live keyboard and mouse input.
- `loop.py`: orchestrates observation, model response, parsing, validation, execution, and termination.
- `loop.py` model history: omits private grading data while preserving usage metadata in valid and malformed response logs.
- `evaluation/base.py`: defines independent task-success evaluation; the current evaluator returns `unknown`.
- `evaluation/harness.py`: runs tasks x trials, records the manifest, per-episode logs, and metric rows.
- `evaluation/metrics.py`: per-episode metrics and aggregate statistics with confidence intervals.
- `evaluation/analysis.py`: merges experiments into CSV/Markdown tables and plots.
- `results.py`: stores typed step, episode, execution, and evaluation results.
- `logging/jsonl.py`: writes one JSONL record for each step and one final episode summary.

## Benchmark development infrastructure

- `benchmark/spec.py`: validates versioned task rules, fixed openings, and predicates.
- `benchmark/environment.py`: reconstructs prepared positions and checks frozen state hashes.
- `benchmark/labels.py`: defines the visible-board-only solver interface; contains no solver.
- `benchmark/scoring.py`: grades seven task predicates, persistent violations, and action counts.
- `benchmark/replay.py`: replays grounding traces and verifies state/image evidence.
- `tasks/benchmark/`: twelve development and twelve demonstration candidates, all uncertified.
- `data/benchmark/pilot/`: raw/grid screenshots, reference transcriptions, and checksums.
- `scripts/check_benchmark.py`: checks candidate coverage and infrastructure; certification remains pending.

Benchmark sandbox executions record private before/after state, resolved targets,
logical labels when supplied, and post-action screenshots. The loop strips private
task data and evaluation/execution evidence from model inputs/history. Task completion
and game termination are separate. Deadlines are checked after inference and after
solver labeling before input; model/solver providers must bound their own calls.

A benchmark click that lands on the face button, a label gutter, a margin, or a
cell border changes nothing, is flagged `off_target` in the step evidence, spends
one step, and lets the episode continue. It is graded as an unsuccessful agent
action rather than an infrastructure error, so grounding misses stay in the
results. Actions that send no input, such as `finish`, never call the label
provider.

T2/T3 tasks require a trusted logical provider. Evaluation-split runs require a
future certificate-verification path and are currently blocked. No pretrained
model is bundled; G/T-specific prompting and a production constraint solver remain unimplemented.
See [the handoff](benchmark-handoff.md) for checks and dependencies.

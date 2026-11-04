# Evaluation protocol

How Vembody experiments are run, what is measured, and how results are compared. Owner: evaluation and experiments (Nicholas).

## Running an experiment

An **experiment** is one configuration (model + settings) run over a task set for N trials.

    python -m vembody eval --tasks <task files or folders> --model <name> --trials N --name <experiment> \
        [--vary-seeds] [--max-steps K] [--timeout S] [--label key=value ...]
    python -m vembody analyze runs/<exp-a> runs/<exp-b> ... --out results/<comparison>

- `--vary-seeds`: trial *k* plays board seed `task_seed + k`, so repeated trials see different boards. Without it every trial replays the task's own board, which only makes sense for stochastic models.
- `--max-steps` / `--timeout`: override every task's limits. These are recorded in the manifest and in each row, so overriding is never silent.
- `--label`: tags the condition for ablations, e.g. `--label memory=off --label crops=on`. Labels appear as `label_<key>` columns.
- Model randomness is seeded with `base_seed + trial`, so a rerun of the same command reproduces the same episodes.

Each experiment folder holds:

| File | Contents |
|---|---|
| `manifest.json` | full config, git commit/branch/dirty flag, Python and platform, command line, SHA-256 of every task file |
| `episodes.jsonl` | one metrics row per episode (the input to `analyze`) |
| `logs/<task>/trial_NNN.jsonl` | the complete step log: raw model output, parsed action, execution, evaluation |
| `screenshots/<run_id>/step_NNN.png` | the exact image the model saw at each step |

The harness refuses to write into a folder that already has results. Rerun under a new name instead of mixing runs.

Rule for the paper: **only report numbers from experiments whose manifest says `"dirty": false`**, so every result maps to a commit.

## Metrics

All metrics come from the true game state, which the model never sees. Per-step board facts are logged in `execution.details` by `MinesweeperSandbox`.

### Per episode

| Metric | Definition |
|---|---|
| `success` | evaluator status is `success`, i.e. every safe cell revealed without hitting a mine |
| `progress` | safe cells revealed ÷ safe cells on the board, at episode end. The main partial-credit metric, since win rate will likely be near zero for an unmodified VLM |
| `mine_hit` | the episode ended with a mine revealed |
| `flags_correct`, `flags_wrong` | flags on mines / on safe cells at episode end (flags placed before the first reveal count as wrong because mines are not yet placed) |
| `steps` | model calls made, including invalid ones |
| `actions_executed` | actions that reached the game |
| `invalid_actions`, `invalid_action_rate` | steps with a parse or validation error ÷ steps |
| `ineffective_actions` | executed actions that did not change the board (clicking an open cell, a click outside the grid, an unsatisfied chord) |
| `off_grid_clicks` | clicks that landed on no cell or button, a direct grounding error |
| `resets` | left clicks on the face button, which start a new board. Progress after a reset counts only the new board |
| `setbacks`, `recoveries`, `recovery_rate` | a setback is an invalid or ineffective step; it is recovered when the next step changes the board. Recovery rate is recoveries ÷ setbacks |
| `wall_seconds` | episode wall-clock time |
| `latency_mean/median/p95` | model call time per step, in seconds |
| `prompt_tokens_*`, `completion_tokens_*`, `total_tokens_*`, `image_count_*` | context usage, read from `ModelResponse.metadata` if the model adapter reports these keys |
| `termination_reason` | why the loop stopped (`game_over`, `action_limit`, `timeout`, `parse_error`, …) |

### Aggregates (per experiment, and per experiment × task)

- **Success rate** with a 95% Wilson score interval. It stays sensible for small *n* and 0% or 100% outcomes.
- **Mean progress** with a 95% percentile bootstrap interval (2,000 resamples, fixed seed).
- **Invalid and ineffective action rates** pooled over all steps.
- **Recovery rate** pooled over all setbacks.
- Mean of per-episode median and p95 latency, mean prompt tokens per step, mean steps, and termination reason counts.

`analyze` writes `episodes.csv`, `summary.csv`, `by_task.csv`, `summary.md`, and `success_rate.png`, `progress.png`, `latency.png`.

## Baselines

| Name | What it is | Role |
|---|---|---|
| `random` | uniform random left or right click anywhere in the window | naive floor; also exercises off-grid clicks and resets |
| `random-cell` | random left click on a hidden cell, found from screenshot pixels | floor for "can click the grid, cannot reason" |
| `solver` | single-cell deduction, random guess when stuck; reads the visible board state directly, never mine positions | ceiling for "perfect perception + simple reasoning"; shows that tasks can be solved within the step limit |
| VLM baseline (Milestone 2) | pretrained local VLM, screenshot + instruction, no memory | the proposal's baseline |
| Vembody (Milestone 4) | VLM + Budgeted Verified Transition Memory | the proposed system |

Reference numbers on the beginner board (9×9, 10 mines, seeds 7–26, `--max-steps 100`):

| Experiment | N | Success (95% CI) | Progress (95% CI) | Mine hit |
|---|---|---|---|---|
| random | 20 | 0.0% [0.0%, 16.1%] | 51.5% [38.6%, 63.5%] | 100% |
| random-cell | 20 | 0.0% [0.0%, 16.1%] | 49.4% [34.2%, 64.5%] | 100% |
| solver | 20 | 75.0% [53.1%, 88.8%] | 84.6% [69.2%, 98.5%] | 25% |

Random progress is high because the first click is always safe and usually opens a large region. **Report VLM progress against the random-cell floor, not against 0.**

## Experiment plan

1. **Baselines:** `random`, `random-cell`, `solver`, and the VLM baseline on the full benchmark, ≥10 trials per task with `--vary-seeds`.
2. **Main comparison:** VLM baseline vs. full Vembody, with the same tasks, seeds, limits and model weights.
3. **Ablations:** remove one memory component at a time, each as its own experiment with `--label`:
   - `verify=off`: no before/after transition verification
   - `crops=off`: no retained region crops
   - `state=off`: no textual task-state / action-history memory
   - `budget=off`: all crops supplied, no budgeted selection, to measure context usage and latency cost
4. **Fine-tuning (if it happens):** fine-tuned vs. prompt-only, same harness.

Trial budget: measure baseline VLM latency first. Episodes × steps × median latency must fit the available GPU time; increase trials before adding conditions.

## Open items needing other owners

- **Retry policy (Anant / Josh):** the loop currently stops an episode on the first parse or validation error, so `invalid_action_rate` and recovery after invalid output can barely be measured. Proposal: allow up to *k* consecutive invalid responses, each logged as its own step.
- **Model adapter metadata (Josh):** report `prompt_tokens`, `completion_tokens`, and `image_count` in `ModelResponse.metadata` so context usage is measured.
- **Ablation switches (Josh):** each memory component needs a config flag the harness can set.
- **Benchmark tasks (Cameron):** the harness accepts a folder, so `tasks/benchmark/` with 10–20 task files is enough. Each task needs a fixed seed and a `max_steps` that the `solver` baseline can finish within.

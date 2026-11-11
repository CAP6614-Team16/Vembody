# Evaluation protocol

How Vembody experiments are run, what is measured, and how results are compared. Owner: evaluation and experiments (Nicholas).

## Running an experiment

An **experiment** is one configuration (model + settings) run over a task set for N trials.

    python -m vembody eval --tasks <task files or folders> --model <name> --trials N --name <experiment> \
        [--split development|demonstration|evaluation] [--vary-seeds] [--max-steps K] [--timeout S] [--label key=value ...]
    python -m vembody analyze runs/<exp-a> runs/<exp-b> ... --out results/<comparison>

- `--tasks` accepts task files, split manifests, or folders; a benchmark root selects only `--split development` by default, and empty/duplicate task sets are rejected.
- `--vary-seeds` changes legacy board seeds to `task_seed + trial`; benchmark boards always retain their frozen opening, setup, and checksum while model seeds vary.
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

The harness reserves a new output folder atomically; any existing folder, including an interrupted run without `episodes.jsonl`, requires a fresh name.

Rule for the paper: **only report numbers from experiments whose manifest says `"dirty": false`**, so every result maps to a commit.

## Metrics

Board metrics come from private game state; both `execution.details` and benchmark `execution.evidence` are saved in logs and stripped from every model history.

### Per episode

| Metric | Definition |
|---|---|
| `success` | evaluator status is `success`: the benchmark task predicate, or a whole-board win for legacy tasks |
| `progress` | safe cells revealed ÷ safe cells on the board, at episode end. The main partial-credit metric, since win rate will likely be near zero for an unmodified VLM |
| `mine_hit` | the episode ended with a mine revealed |
| `flags_correct`, `flags_wrong` | flags on mines / on safe cells at episode end (flags placed before the first reveal count as wrong because mines are not yet placed) |
| `steps` | logged attempts, including invalid output and infrastructure-error records |
| `actions_executed` | executed left/right clicks; `finish` and `fail` send no game input |
| `invalid_actions`, `invalid_action_rate` | steps with a parse or validation error ÷ steps |
| `ineffective_actions` | executed cell-input attempts that did not change the board, including off-grid clicks; terminal requests are excluded |
| `off_grid_clicks` | clicks that landed on no cell or button, a direct grounding error |
| `resets` | left clicks on the face button, which start a new board. Progress after a reset counts only the new board |
| `setbacks`, `recoveries`, `recovery_rate` | an invalid or ineffective attempt is recovered only when the next cell action changes the board without reset, mine loss, or a recorded scoring violation |
| `wall_seconds` | episode wall-clock time |
| `latency_mean/median/p95` | model call time per step, in seconds |
| `prompt_tokens_*`, `completion_tokens_*`, `total_tokens_*`, `image_count_*` | context usage, read from `ModelResponse.metadata` if the model adapter reports these keys |
| `termination_reason` | why the loop stopped (`game_over`, `action_limit`, `timeout`, `parse_error`, …) |
| `infrastructure_failure` | capture, execution, or evaluation errors; counted separately from agent failures |

### Aggregates (per experiment, and per experiment × task)

- **Success rate** with a 95% Wilson score interval. It stays sensible for small *n* and 0% or 100% outcomes.
- `scored_episodes` excludes infrastructure failures; summaries also report total episodes and infrastructure failures explicitly.
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
| `solver` | single-cell deduction, random guess when stuck; reads visible clues/flags directly | perfect-perception reference policy; does not supply logical labels or certify benchmark instances |
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

1. **Baselines:** repeat the reference policies and VLM on the same frozen instances with different model seeds; logical-provider and certificate gates remain enforced.
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
- **Benchmark tasks (Cameron):** task selection follows the chosen split manifest; fixed starts, logical providers, and certificate review remain required by `docs/benchmark-handoff.md`.

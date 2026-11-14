# Vembody

A local vision-language agent for autonomous computer control through screen perception and keyboard/mouse actions.

## Milestone 1

Install with Python 3.10 or newer:

    python -m pip install -r requirements/benchmark-runtime.txt -e ".[test]"

Run the safe mock demonstration:

    python -m vembody run --task tasks/example.yaml --model mock --executor dry-run

The mock model proposes deterministic actions, the parser and validator check them, and the dry-run executor records them without controlling the computer.

Live desktop execution is not implemented in this milestone.

For Anant's RTX 5080 endpoint or a hosted vision-model alternative, see [model access](docs/model-access.md); connection profiles are selected with `--model-config`.
For the selected UI-Mate-9B Q6_K model, start with the [Kaggle pilot](docs/kaggle.md); native actions use `--model ui-mate`.

## Minesweeper sandbox

Play the game yourself in a browser (left click reveals, right click flags, the face starts a new game):

    python -m vembody play

Run the mock model on the Minesweeper task (it replays the actions listed in the task file):

    python -m vembody run --task tasks/minesweeper-example.yaml

Screenshots for every step are saved next to the log under `screenshots/<run_id>/`.

The game (`src/vembody/minesweeper/`) is a normal standalone Minesweeper. The agent harness (`src/vembody/sandbox.py`) only uses the same two things a player has, the window screenshot and pixel mouse clicks. It reads the true game state only to end the episode and grade it.

## Experiments

Install the plotting extra once:

    python -m pip install -r requirements/benchmark-runtime.txt -e ".[test,analysis]"

Run each configuration as a named experiment, then compare them:

    python -m vembody eval --tasks tasks/minesweeper-example.yaml --model random-cell --trials 20 --vary-seeds --name random-cell
    python -m vembody eval --tasks tasks/minesweeper-example.yaml --model solver --trials 20 --vary-seeds --name solver
    python -m vembody analyze runs/random-cell runs/solver --out results/

See `docs/evaluation.md` for metric definitions, baselines, and the ablation protocol.

Benchmark folders use the selected split manifest (`--split development` by default); starts stay frozen across trials, and logical tasks still require their label provider.

    python -m vembody eval --tasks tasks/benchmark/development/t1-01.yaml --model mock --trials 3 --name grounding-smoke
## Development benchmark

The twelve-task development benchmark now has validated resets, task predicates,
and separate development/demo screenshot assets. Its instances are uncertified;
logical tasks require a solver provider and evaluation releases remain blocked.
See [the handoff and coverage table](docs/benchmark-handoff.md),
[the benchmark design](docs/benchmark-design.md), and
[the task guide](tasks/benchmark/README.md).

After installing the pinned image runtime and development tools:

    python -m pip install -r requirements/benchmark-runtime.txt -e ".[test,lint]"
    python scripts/check_benchmark.py --output artifacts/benchmark-check-01

Use a fresh output directory for each check. Passing this check establishes the
development infrastructure; its report explicitly records `evaluation_ready: false`.

Run the automated tests with:

    python scripts/run_tests.py

This runs the full pytest suite with a fresh project-local temporary directory
and no shared pytest cache, avoiding Windows temporary-folder permission conflicts.
Earlier run directories are deleted first so they stop accumulating; pass
`--keep-temp` to keep them.

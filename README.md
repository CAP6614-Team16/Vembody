# Vembody

A local vision-language agent for autonomous computer control through screen perception and keyboard/mouse actions.

## Milestone 1

Install with Python 3.10 or newer:

    python -m pip install -e ".[test]"

Run the safe mock demonstration:

    python -m vembody run --task tasks/example.yaml --model mock --executor dry-run

The mock model proposes deterministic actions, the parser and validator check them, and the dry-run executor records them without controlling the computer.

Live desktop execution is not implemented in this milestone.

## Minesweeper sandbox

Play the game yourself in a browser (left click reveals, right click flags, the face starts a new game):

    python -m vembody play

Run the mock model on the Minesweeper task (it replays the actions listed in the task file):

    python -m vembody run --task tasks/minesweeper-example.yaml

Screenshots for every step are saved next to the log under `screenshots/<run_id>/`.

The game (`src/vembody/minesweeper/`) is a normal standalone Minesweeper. The agent harness (`src/vembody/sandbox.py`) only uses the same two things a player has, the window screenshot and pixel mouse clicks. It reads the true game state only to end the episode and grade it.

The proposed twelve-task benchmark, reset procedures, and scoring rules are in
[the benchmark design](docs/benchmark-design.md). This is a Phase 1 review draft;
the benchmark extensions are not implemented yet.

# Vembody

A local vision-language agent for autonomous computer control through screen perception and keyboard/mouse actions.

## Milestone 1

Install with Python 3.10 or newer:

    python -m pip install -e ".[test]"

Run the safe mock demonstration:

    python -m vembody run --task tasks/example.yaml --model mock --executor dry-run

The mock model proposes deterministic actions, the parser and validator check them, and the dry-run executor records them without controlling the computer.

Live desktop execution is not implemented in this milestone.

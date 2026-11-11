"""Module entry point for python -m vembody."""

from .cli import main

if __name__ == "__main__":
    # Propagate the exit code; a refused run must not look successful to a script.
    raise SystemExit(main())

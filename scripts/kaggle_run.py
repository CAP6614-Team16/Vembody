"""Run notebook stages in the isolated interpreter rather than Kaggle's shared kernel."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# -I intentionally removes implicit script/cwd imports; explicitly trust this bundle's scripts.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.check_ui_mate_compatibility import check_contract
from scripts.kaggle_autonomous import AUTONOMOUS_PROFILES, run_autonomous
from scripts.kaggle_cache import export_runtime
from scripts.kaggle_grounding import run_grounding
from scripts.kaggle_ui_mate import (
    Runtime,
    build_server,
    gpu_info,
    prepare_runtime,
    run_pilot,
    serve_model,
    smoke_model,
)
from vembody.models.registry import MODEL_PROMPT_VERSIONS


def save_runtime(path: Path, runtime: Runtime) -> None:
    """Persist prepared paths and provenance without downloading or loading the model again."""
    data = {
        "binary": str(runtime.binary),
        "model": str(runtime.model),
        "projector": str(runtime.projector),
        "spec": runtime.spec,
        "gpu_info": runtime.gpu_info,
    }
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def load_runtime(path: Path, spec: dict) -> Runtime:
    """Reject stale settings or missing prepared files and recheck current GPU access."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if data["spec"] != spec:
        raise RuntimeError("Prepared settings differ from this bundle; rerun the prepare cell")

    paths = [Path(data[name]) for name in ("binary", "model", "projector")]
    if any(not item.is_file() for item in paths):
        raise RuntimeError("Prepared files are missing; rerun the prepare cell")

    return Runtime(*paths, spec, gpu_info())


def main(argv: list[str] | None = None) -> int:
    """Build/export a runtime, prepare weights, or evaluate with explicit notebook paths."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage", choices=("build-runtime", "prepare", "pilot", "grounding", "autonomous")
    )
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--cache-output", type=Path)
    parser.add_argument("--trials", type=int, choices=range(1, 11))
    parser.add_argument("--adapter", choices=("ui-mate", "ui-mate-legacy"), default="ui-mate")
    parser.add_argument("--autonomous-profile", choices=AUTONOMOUS_PROFILES, default="brief")
    args = parser.parse_args(argv)
    trials = args.trials if args.trials is not None else (1 if args.stage == "autonomous" else 3)
    if args.stage not in {"prepare", "build-runtime"} and args.output is None:
        parser.error("evaluation stages require --output")

    if args.stage == "build-runtime" and args.cache_output is None:
        parser.error("build-runtime requires --cache-output")

    if (
        args.stage == "autonomous"
        and args.autonomous_profile == "brief"
        and args.adapter != "ui-mate"
    ):
        parser.error("The brief autonomous profile requires --adapter ui-mate")

    config = args.repo / "configs/kaggle-ui-mate.json"
    state = args.work / "prepared-runtime.json"
    if args.stage == "build-runtime":
        print(gpu_info(), flush=True)
        args.work.mkdir(parents=True, exist_ok=True)
        spec = json.loads(config.read_text(encoding="utf-8"))
        binary = build_server(args.work, spec)
        export_runtime(binary, spec, args.cache_output)
        print("Runtime exported; no model weights downloaded or experiments run", flush=True)
    elif args.stage == "prepare":
        runtime = prepare_runtime(args.work, config)
        save_runtime(state, runtime)
        if args.cache_output is not None:
            export_runtime(runtime.binary, runtime.spec, args.cache_output)

        print("Prepared Q6_K runtime; continue to the evaluation cell", flush=True)
    else:
        runtime = load_runtime(state, json.loads(config.read_text(encoding="utf-8")))
        selected_adapter = (
            "ui-mate-brief"
            if args.stage == "autonomous" and args.autonomous_profile == "brief"
            else args.adapter
        )
        with serve_model(runtime, args.output):
            source = args.repo / "bundle-manifest.json"
            (args.output / "source-snapshot.json").write_bytes(source.read_bytes())
            (args.output / "compatibility.json").write_text(
                json.dumps(
                    {
                        **check_contract(),
                        "selected_adapter": selected_adapter,
                        "selected_prompt_version": MODEL_PROMPT_VERSIONS[selected_adapter],
                        "autonomous_profile": args.autonomous_profile
                        if args.stage == "autonomous"
                        else None,
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            if args.stage == "grounding":
                smoke_model(args.repo, args.output, args.adapter)
                run_grounding(args.repo, args.output / "grounding", trials, adapter=args.adapter)
            elif args.stage == "autonomous":
                run_autonomous(
                    args.repo,
                    args.output / "autonomous",
                    trials,
                    args.adapter,
                    profile=args.autonomous_profile,
                )
            else:
                smoke_model(args.repo, args.output, args.adapter)
                run_pilot(args.repo, args.output, args.adapter)

        print(f"Server stopped. Results: {args.output}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

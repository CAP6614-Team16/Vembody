"""Merge a LoRA adapter into the bf16 base and convert it to GGUF for LM Studio.

Writes the text model as Q8_0 and the vision projector as BF16 (mmproj-*.gguf).
The merge goes one tensor at a time to keep memory low; the merged checkpoint is
deleted afterwards unless --keep-merged is set.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

CONVERT = Path(__file__).resolve().parents[1] / ".tools/llama.cpp/convert_hf_to_gguf.py"
SHARD_BYTES = 4 * 1024**3


def merge(base: Path, adapter: Path, merged: Path) -> None:
    """Write base + scale * B @ A for every adapted weight, copying everything else."""
    config = json.loads((adapter / "adapter_config.json").read_text())
    scale = config["lora_alpha"] / config["r"]
    lora = safe_open(adapter / "adapter_model.safetensors", "pt")
    pending = {
        name.removeprefix("base_model.model.").replace(".lora_A.weight", ".weight")
        for name in lora.keys()
        if ".lora_A." in name
    }
    merged.mkdir(parents=True)
    for item in base.iterdir():  # config, tokenizer, processor, chat template
        if item.is_file() and not item.name.startswith("model.safetensors"):
            shutil.copy(item, merged / item.name)

    weight_map: dict[str, str] = {}
    shard: dict[str, torch.Tensor] = {}
    size = 0

    def flush() -> None:
        nonlocal shard, size
        if shard:
            name = f"model-{len(set(weight_map.values())) + 1:05d}.safetensors"
            save_file(shard, merged / name, metadata={"format": "pt"})
            weight_map.update(dict.fromkeys(shard, name))
            shard, size = {}, 0

    for path in sorted(base.glob("*.safetensors")):
        with safe_open(path, "pt") as source:
            for name in source.keys():
                tensor = source.get_tensor(name)
                if name in pending:
                    key = "base_model.model." + name.removesuffix(".weight")
                    a = lora.get_tensor(key + ".lora_A.weight").float()
                    b = lora.get_tensor(key + ".lora_B.weight").float()
                    tensor = (tensor.float() + scale * (b @ a)).to(tensor.dtype)
                    pending.discard(name)
                shard[name] = tensor
                size += tensor.numel() * tensor.element_size()
                if size >= SHARD_BYTES:
                    flush()
    flush()
    if pending:
        example = pending.pop()
        raise SystemExit(f"{len(pending) + 1} adapter weights have no base tensor, e.g. {example}")
    index = {"metadata": {}, "weight_map": weight_map}
    (merged / "model.safetensors.index.json").write_text(json.dumps(index, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", type=Path, required=True, help="bf16 base checkpoint")
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="GGUF folder for LM Studio")
    parser.add_argument("--keep-merged", action="store_true")
    args = parser.parse_args()

    name = args.out.name.removesuffix("-GGUF")
    merged = args.adapter.parent / "merged"
    shutil.rmtree(merged, ignore_errors=True)  # never reuse a partial merge
    merge(args.model, args.adapter, merged)

    args.out.mkdir(parents=True, exist_ok=True)
    for outtype, extra, filename in (
        ("q8_0", [], f"{name}-Q8_0.gguf"),
        ("bf16", ["--mmproj"], f"mmproj-{name}-BF16.gguf"),  # LM Studio needs the prefix
    ):
        subprocess.run(
            [sys.executable, str(CONVERT), str(merged), "--outtype", outtype]
            + ["--outfile", str(args.out / filename), *extra],
            check=True,
        )
    if not args.keep_merged:
        shutil.rmtree(merged)
    print(f"GGUF files written to {args.out}")


if __name__ == "__main__":
    main()

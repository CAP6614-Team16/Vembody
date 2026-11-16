"""Score a served model on held-out generated examples, per board size.

Sends validation examples the same way as the lmstudio-sft profile and reports valid
JSON, exact board copy, whether the click hits the cell named in the reasoning, and
whether the move is correct.

    python training/eval_lmstudio.py --model <model id> --data data/sft
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import re
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

from vembody.minesweeper.deduce import deductions
from vembody.minesweeper.screen import MinesweeperScreen
from vembody.models.http import BICUBIC, gemma_image_size


def ask(url: str, model: str, record: dict, image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, "PNG")
    data = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()
    body = {
        "model": model,
        "temperature": 0,
        "max_tokens": 1500,
        "messages": [
            {"role": "system", "content": record["system"]},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": record["prompt"]},
                    {"type": "image_url", "image_url": {"url": data}},
                ],
            },
        ],
    }
    request = urllib.request.Request(
        url + "/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        return json.load(response)["choices"][0]["message"]["content"] or ""


def score(record: dict, reply: str) -> Counter:
    target = json.loads(record["response"])
    board = [row.split() for row in target["board"]]
    screen = MinesweeperScreen(len(board), len(board[0]), 1, benchmark=True)

    def cell(action):
        x = round(action["x"] * (screen.width - 1))
        return screen.cell_at(x, round(action["y"] * (screen.height - 1)))

    result = Counter(n=1)
    try:
        answer = json.loads(reply)
        action, summary = answer["action"], answer["reasoning_summary"]
        clicked = cell(action)
    except (ValueError, KeyError, TypeError):
        return result
    result["valid"] = 1
    result["board"] = answer.get("board") == target["board"]
    named = re.findall(r"(?:reveal|flag|names) ([A-I]\d+)", summary)
    if named:
        name = named[-1]
        result["lands_on_named"] = clicked == (ord(name[0]) - 65, int(name[1:]) - 1)
    if "names" in target["reasoning_summary"]:  # grounding example: the requested cell
        result["correct"] = clicked == cell(target["action"])
    else:
        proved = deductions(board).get(clicked) if clicked else None
        result["correct"] = proved is not None and proved[0] == action["type"]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", type=Path, default=Path("data/sft"))
    parser.add_argument("--url", default="http://127.0.0.1:1234/v1")
    parser.add_argument("--count", type=int, default=150)
    args = parser.parse_args()

    records = [json.loads(line) for line in (args.data / "val.jsonl").open(encoding="utf-8")]
    step = max(1, len(records) // args.count)
    sizes: dict[str, Counter] = defaultdict(Counter)
    for record in records[::step][: args.count]:
        image = Image.open(args.data / "images" / record["image"]).convert("RGB")
        image = image.resize(gemma_image_size(image.size, 280), BICUBIC)
        rows = len(json.loads(record["response"])["board"])
        sizes[f"{rows}x{rows}"] += score(record, ask(args.url, args.model, record, image))
    total = Counter()
    for size in sorted(sizes):
        total += sizes[size]
    for name, counts in [*sorted(sizes.items()), ("all", total)]:
        n = counts["n"]
        print(
            f"{name:>4}: n={n:3d}  valid {counts['valid'] / n:5.1%}"
            f"  board {counts['board'] / n:5.1%}"
            f"  lands-on-named {counts['lands_on_named'] / n:5.1%}"
            f"  correct {counts['correct'] / n:5.1%}"
        )


if __name__ == "__main__":
    main()

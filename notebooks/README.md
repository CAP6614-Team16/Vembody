# Kaggle notebooks: teammate quick start

This WIP runs pretrained UI-Mate-9B Q6_K and our screenshot-driven Minesweeper sandbox together on Kaggle; no fine-tuning or external endpoint is involved.

## Get the branch and source ZIP

From your existing Vembody clone, preserve any local changes before switching:

```powershell
git fetch origin
git switch --track origin/wip/ui-mate-kaggle-autonomy
python scripts/package_kaggle.py
```

- If you already have the branch locally, use `git switch wip/ui-mate-kaggle-autonomy` followed by `git pull --ff-only` instead.
- Packaging requires Git and Python 3.10+ only; run it from the repo root to produce `artifacts/vembody-kaggle.zip`.
- Import notebooks from this same branch/version; after editing source, rebuild the ZIP and update the attached source dataset.

## What Josh has done

- Connected the pretrained Q6_K weights/projector to strict screenshot-to-action adapters, deterministic sandbox resets, independent grading, and JSONL logs.
- Built/exported a reusable T4 runtime and completed real inference pilots; v2 grounding succeeded on 6/12 trials across four fixed tasks repeated three times.
- Diagnosed the first autonomous attempt: output hit 2,048 tokens before an action; the new brief/4,096-token profile is implemented but still needs GPU validation.
- No training, reliable full-game play, certified final benchmark, or cross-game generalization is established; see [finished/remaining work](../docs/ui-mate-wip.md).

## Choose the notebook

| File | Purpose | Required private inputs | Output |
| --- | --- | --- | --- |
| [kaggle-build-runtime-t4.ipynb](kaggle-build-runtime-t4.ipynb) | Build/export the pinned CUDA server only; no model weights or inference | Source ZIP | `vembody-runtime-t4.zip` |
| [kaggle-ui-mate-q6.ipynb](kaggle-ui-mate-q6.ipynb) | Download weights, run grounding or autonomous checks, export results | Source ZIP; runtime cache strongly recommended | `vembody-q6-<id>.zip` |

- Both notebooks require Kaggle GPU T4 x2 and Internet enabled; run code cells from top to bottom in a fresh session.
- Dataset titles can be anything, but preserve the filenames `vembody-kaggle.zip` and `vembody-runtime-t4.zip`; Kaggle-unpacked forms are supported too.
- Attach exactly one source bundle/version and at most one trusted runtime cache; do not attach old result ZIPs as inputs.
- Source, cache, and results are different artifacts; none contains model weights, which are downloaded from the pinned Hugging Face revision during experiment preparation.

## If you already have Josh's runtime ZIP

1. Ask Josh to share `vembody-runtime-t4.zip` privately, or provide access to its private Kaggle dataset; it is intentionally not committed to Git.
2. Upload the source ZIP and runtime ZIP as separate private input datasets, and attach both to the experiment notebook.
3. Skip the runtime-generator notebook; experiment preparation validates/restores the cache and downloads approximately 8.6 GB of model/projector weights.

## If you need to generate the runtime

1. Import `kaggle-build-runtime-t4.ipynb`, attach only the source dataset, select GPU T4 x2, enable Internet, and run all three code cells.
2. Expect CPU-heavy CUDA compilation that may take an hour or more; low GPU utilization during compilation is normal and session quota is still consumed.
3. Download `/kaggle/working/vembody-runtime-t4.zip`, then stop the session; upload that ZIP as a private input for the experiment notebook.
4. This generator does not run a model, train anything, or certify cache portability; incompatible CUDA/libc/CPU environments require an explicit rebuild.

## Run the experiment

1. Import `kaggle-ui-mate-q6.ipynb`, attach the source and runtime datasets, and run all five code cells; preparation should reuse a compatible cache without compilation.
2. Setup defaults to `MODE = "autonomous"`, `AUTONOMOUS_PROFILE = "brief"`, `PLAY_TRIALS = 1`: one fixed 5x5 game with a 40-action/15-minute ceiling.
3. For visual grounding, set `MODE = "grounding"`; `TRIALS = 3` runs the four T1 tasks three times plus a non-executed smoke call, without changing grounding's v2 prompt/budget.
4. Autonomous controls are `"original"` (v2/2,048 tokens), `"budget-only"` (v2/4,096), and `"brief"` (v3/4,096); manifests record the selected profile and request timeout.
5. In the same prepared session, change `MODE`/`AUTONOMOUS_PROFILE` in a new cell and rerun only evaluation (code cell 4) and ZIP export (code cell 5); no setup/build rerun is needed.
6. Download the result ZIP from the Output sidebar and stop the session; stopping the model server alone does not stop GPU quota consumption.

## Read results and troubleshoot

- `autonomous/autonomous-summary.json` reports executed actions, truncations, wins/failures, and latency; `grounding/grounding-summary.json` reports target accuracy and coordinate error.
- Per-step JSONL logs, screenshots, `server.log`, model/runtime identity, and the source manifest are retained in each result ZIP; share those privately for review.
- A correct click on an instructed cell is grounding, not autonomous strategy; a complete game requires independent board-win grading, not the model's completion claim.
- `finish_reason = "length"` means output was truncated; the adapter rejects it without executing an action or retrying automatically.
- If a runtime cache is incompatible, stop, detach it, and use the generator to rebuild; never bypass its checks or execute caches from untrusted sources.
- If a download link returns 404, use the Output sidebar file menu; download results/cache before stopping because temporary weights/builds may disappear.
- For implementation details and known setup issues, see [Kaggle notes](../docs/kaggle.md) and [adapter compatibility](../docs/ui-mate-compatibility.md).

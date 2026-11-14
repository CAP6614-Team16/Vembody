# UI-Mate integration — work in progress

Status: 2026-11-14; development experiments only, not a finished autonomous agent or certified benchmark.

## Finished

- Added environment-based endpoint profiles, safe vision HTTP requests, and CLI/evaluation integration without bundled credentials or weights.
- Added versioned UI-Mate adapters with strict single-action parsing, screenshot resizing, coordinate conversion, and independent sandbox grading.
- Added Kaggle packaging, an isolated Python environment, pinned Q6_K inference, reusable validated T4 runtime caches, and result export.
- Added four-task grounding checks and bounded autonomous play on the existing 5×5 development board.
- Measured v2 grounding at 6/12 successes across four tasks repeated three times, 100% valid responses, and 10.45 seconds average response time; target errors remain.
- Diagnosed the v2 autonomous run: its first response exhausted 2,048 output tokens after 70.5 seconds, with zero executed actions; starting reveals were preset.
- Implemented separately logged `original` (v2/2,048 tokens/180 seconds), `budget-only` (v2/4,096/240), and `brief` (v3/4,096/240) profiles without changing game limits or the CUDA runtime.
- Added explicit truncation rejection/reporting and verified 263 local tests, Ruff lint/format checks, and offline adapter checks; GPU validation of the new profiles is pending.

## Next

- Josh: upload a fresh source bundle and matching notebook, attach the existing runtime cache, run the new autonomous profile, and download results before stopping Kaggle.
- Compare `original` → `budget-only` → `brief` on identical development starts to separate budget and prompt effects; a short-response request does not guarantee completion.
- Investigate remaining visual grounding errors and the G2 regression; expand beyond four distinct tasks before claiming reliable grounding.
- Verify successful multi-step autonomous play, response completion, latency, and failure behavior before deciding whether fine-tuning is justified.
- Anant: establish the RTX 5080 endpoint and verify Q6_K/projector fit, runtime compatibility, and client access; no working endpoint is claimed here.
- Complete solver-backed logical certification, T2/T3 provider integration, and held-out evaluation splits before final benchmark claims.
- Collect training demonstrations and implement adaptation, controlled comparisons, and cross-game transfer experiments only after the baseline is established.

See [Kaggle instructions](kaggle.md), [adapter compatibility](ui-mate-compatibility.md), and [benchmark handoff](benchmark-handoff.md); raw results, runtime ZIPs, weights, and keys stay outside Git.

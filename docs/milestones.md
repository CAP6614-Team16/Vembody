# Project Milestones

The original roadmap below describes the overall project. For current tasks/data
coverage, see [the benchmark handoff](benchmark-handoff.md). The twelve-template
development infrastructure and UI-Mate integration are implemented; the original
Kaggle inference pilot ran, while solver certification, final evaluation splits,
fine-tuning, and game-transfer experiments remain pending.

For the latest implemented pieces, measured limitations, and remaining work, see [UI-Mate WIP status](ui-mate-wip.md).

- **Milestone 1 — Safe vertical slice (current):** Run a bounded mock task through task loading, observation, parsing, validation, dry-run execution, logging, and tests; live capture and input remain deferred.
- **Milestone 2 — Local VLM:** Connect one feasible pretrained local vision-language model and measure structured-response validity, visual grounding, latency, and hardware requirements.
- **Milestone 3 — Reproducible environment:** Select one controlled application or game and define approximately 10–20 resettable tasks with objective success conditions.
- **Milestone 4 — Research improvement:** Choose one evidence-based improvement, such as short-term memory, action verification, visual grounding, or failure recovery.
- **Milestone 5 — Controlled experiments:** Compare the baseline, improvement, and ablations across repeated trials using fixed tasks, limits, configurations, and metrics.
- **Final deliverable:** Preserve code, configurations, raw logs, results, demonstrations, and the technical report/presentation in a reproducible form.

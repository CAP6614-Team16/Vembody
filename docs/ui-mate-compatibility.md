# UI-Mate compatibility

- Reference: the [official agent at revision `1cb9e1e`](https://github.com/Tencent/UI-Mate/blob/1cb9e1e44ce856e23b593992b02efbd489943fcb/agents/ui_mate_agent.py); this is an independent restricted adapter, not a verbatim copy or full benchmark reproduction.
- Coordinates already used the correct 0–999 grid; v2 now matches upstream `int(coordinate * original_dimension / 999)` before normalization, clipping only the far image edge.
- The archived 12-trial run contained 24 clicks; auditing upstream versus legacy scaling changed three clicks by one pixel, insufficient to explain the large observed misses.
- Client images now follow the upstream factor-32 resize with no added padding/cropping; original and encoded dimensions are both recorded.
- The 336×376 grounding screenshots become 320×384; 208×248 gameplay screenshots become 192×256 before server-side preprocessing.
- The prompt now uses an embedded function schema and plain `Instruction`/`Previous actions` text rather than the v1 JSON instruction wrapper.
- Safety differences remain intentional: one allowlisted call, no generated Python, no implicit-success fallback, and no arbitrary desktop controls.
- History remains off, generation budgets remain smaller, and decoding remains greedy; the complete upstream history/prompt configuration is not reproduced.
- Client-side checks cannot prove parity with llama.cpp's vision encoder, resizing, padding, or template processing; real model accuracy must be measured again.
- Run `python scripts/check_ui_mate_compatibility.py --results PATH_TO_RESULTS_ZIP --output artifacts/ui-mate-compatibility.json` for a no-GPU audit; archived instructions are never executed.
- Use `ADAPTER = 'ui-mate'` for v2 or `'ui-mate-legacy'` for v1; keep their results separate because prompt, client sizing, and rounding changed together.
- Use `MODE = 'autonomous'` for the existing fixed T4-01 development game; the agent receives one high-level goal, not target cells or hidden state.
- The autonomous task retains its 40-action/900-second limits; a mine hit alone cannot prove a strategy error because logical certification is pending.
- The v2 autonomous pilot exhausted 2,048 output tokens before any action; v3-brief requests a one-sentence decision and complete tool call, with the same screenshot/coordinate contract.
- Autonomous profiles preserve `original` (v2/2,048 tokens/180 seconds), `budget-only` (v2/4,096/240), and `brief` (v3/4,096/240) as separately logged comparisons; grounding stays unchanged.
- A brief response is a prompt request, not a guaranteed reasoning limit; truncated outputs remain failures without action execution or automatic retry.
- No fine-tuning, cross-game transfer evaluation, CUDA rebuild, or accuracy improvement is implied by these compatibility changes.

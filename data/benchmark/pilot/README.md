# Pilot image assets (uncertified)

These are reference images for 12 development candidates and 12 separate
demonstration candidates. They are not the final evaluation dataset. The
`manifest.json` records the renderer/runtime, source instance IDs, geometry,
state/layout hashes, file/pixel hashes, and reference visible transcriptions.

For each instance:

- `<instance-id>-raw.png` has blank label gutters for the baseline.
- `<instance-id>-grid.png` has row/column labels at the same geometry and scale.
- The yellow region outline for T3-01 is present in both images.

The manifest and demonstration responses are evaluator/prompt-development assets.
Do not supply its reference transcriptions, seeds, or hashes to the baseline model.
Select only the configured image and public instruction. The T condition should
produce its own transcription; it must not receive the reference answer.

Four T1 demonstration responses were executed and checked against the grounding
predicates during export. The remaining eight logical/full-game examples are
pending solver-backed reference traces. No asset has a guess-free or reasoning
difficulty certificate. The visible cell alphabet is `#`, `F`, `.`, and `1`-`8`;
this small pilot does not guarantee coverage of every number class.

Regenerate from the repository root with the recorded Python/Pillow versions:

```powershell
python scripts/export_benchmark_assets.py
```

This overwrites generated assets. A renderer/runtime change requires reviewing the
new images and checksums; it must not silently replace frozen evaluation inputs.
See `development-contact-sheet.png` for a visual overview of the twelve templates.

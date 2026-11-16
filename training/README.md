# Fine-tuning

LoRA fine-tuning on solver-labelled Minesweeper screenshots. Uses its own venv
(`.venv-train`, setup at the top of `requirements.txt`).

1. Download the base model:
   `.venv-train/bin/hf download google/gemma-4-12B-it --local-dir <models>/gemma-4-12B-it`
2. Generate data (project venv):
   `python scripts/generate_sft_data.py --episodes 800 --out data/sft`
3. Train:
   `.venv-train/bin/python training/train_lora.py --model <models>/gemma-4-12B-it --out runs/lora-12b --vision`
   Rerun the same command to resume from the last checkpoint.
   `--limit 200 --max-steps 30` does a quick smoke run.
4. Export to GGUF:
   `.venv-train/bin/python training/export_gguf.py --model <models>/gemma-4-12B-it --adapter runs/lora-12b/adapter --out <models>/gemma-4-12B-it-minesweeper-GGUF`
5. Load it in LM Studio with thinking off, set `VEMBODY_LMSTUDIO_MODEL`, and evaluate
   with `configs/models/lmstudio-sft.yaml`.

Training prompts come from `vembody.models.prompts`, so regenerate the data after any
prompt change.

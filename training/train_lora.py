"""QLoRA fine-tune of a Gemma vision model on data from scripts/generate_sft_data.py.

Runs in the separate training environment (see training/README.md), not the project venv:

    .venv-train/bin/python training/train_lora.py --model <models>/gemma-4-12B-it \
        --data data/sft --out runs/lora-12b --vision --limit 200 --max-steps 30

Loss is on the reply only. The prompt is the exact system/user text the HTTP model
sends at inference, with the screenshot after the text, as LM Studio receives it.
"""

# ruff: noqa: I001  (unsloth must be imported before transformers/trl)
from __future__ import annotations

import argparse
import json
from pathlib import Path

from unsloth import FastVisionModel  # must be imported before transformers/trl
from unsloth.trainer import UnslothVisionDataCollator
import torch
from datasets import Dataset
from PIL import Image
from trl import SFTConfig, SFTTrainer


def examples(data: Path, split: str, limit: int | None) -> Dataset:
    """Chat-format examples; screenshots are decoded per item to keep RAM flat."""
    with (data / f"{split}.jsonl").open(encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle][:limit]

    def to_chat(batch: dict) -> dict:
        chats = []
        for system, prompt, response, name in zip(
            batch["system"], batch["prompt"], batch["response"], batch["image"], strict=True
        ):
            image = Image.open(data / "images" / name).convert("RGB")
            chats.append(
                [
                    {"role": "system", "content": [{"type": "text", "text": system}]},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image", "image": image},
                        ],
                    },
                    {"role": "assistant", "content": [{"type": "text", "text": response}]},
                ]
            )
        return {"messages": chats}

    return Dataset.from_list(records).with_transform(to_chat)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", type=Path, default=Path("data/sft"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, help="use only the first N training examples")
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--epochs", type=float, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--vision", action="store_true", help="also train vision layers")
    args = parser.parse_args()

    # bf16 base with batch 8 on large GPUs, otherwise 4-bit with gradient accumulation
    big_gpu = torch.cuda.get_device_properties(0).total_memory > 40 * 1024**3
    batch = 8 if big_gpu else 1
    model, processor = FastVisionModel.from_pretrained(
        args.model, load_in_4bit=not big_gpu, use_gradient_checkpointing="unsloth"
    )
    model = FastVisionModel.get_peft_model(
        model,
        r=args.rank,
        lora_alpha=args.rank,
        finetune_vision_layers=args.vision,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        random_state=0,
    )
    FastVisionModel.for_training(model)

    train = examples(args.data, "train", args.limit)
    val = examples(args.data, "val", 100)
    collator = UnslothVisionDataCollator(
        model,
        processor,
        resize="max",  # leave sizing to the Gemma processor, as at inference
        train_on_responses_only=True,
        instruction_part="<|turn>user\n",
        response_part="<|turn>model\n",
    )
    trainer = SFTTrainer(
        model=model,
        processing_class=processor,
        data_collator=collator,
        train_dataset=train,
        eval_dataset=val,
        args=SFTConfig(
            output_dir=str(args.out),
            per_device_train_batch_size=batch,
            per_device_eval_batch_size=batch,
            gradient_accumulation_steps=8 // batch,
            num_train_epochs=args.epochs,
            max_steps=args.max_steps,
            learning_rate=args.lr,
            lr_scheduler_type="cosine",
            warmup_ratio=0.05,
            weight_decay=0.0,
            bf16=True,
            optim="adamw_8bit",
            logging_steps=5,
            eval_strategy="steps",
            eval_steps=100,
            save_steps=100,
            save_total_limit=2,
            report_to="none",
            seed=0,
            max_length=4096,
            remove_unused_columns=False,
            dataset_text_field="",
            dataset_kwargs={"skip_prepare_dataset": True},
        ),
    )
    # resume from the latest checkpoint if there is one
    resume = any(args.out.glob("checkpoint-*"))
    trainer.train(resume_from_checkpoint=resume or None)
    model.save_pretrained(args.out / "adapter")
    processor.save_pretrained(args.out / "adapter")
    print(f"Saved LoRA adapter to {args.out / 'adapter'}")


if __name__ == "__main__":
    main()

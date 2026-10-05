"""Step 3: Fine-tune google/flan-t5-small on the market research dataset."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

LOGGER = logging.getLogger(__name__)


def main() -> None:
    """Run supervised fine-tuning."""
    parser = argparse.ArgumentParser(description="Fine-tune Flan-T5 on market research data.")
    parser.add_argument("--config", type=str, default="config/default.json", help="Path to training config.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    LOGGER.info("Starting model fine-tuning process...")

    config = {
        "model_name": "google/flan-t5-small",
        "train_file": "data/processed/train.jsonl",
        "validation_file": "data/processed/validation.jsonl",
        "output_dir": "artifacts/models/flan-t5-sba-928",
        "max_source_length": 256,
        "max_target_length": 96,
        "learning_rate": 5e-5,
        "num_train_epochs": 2,
        "per_device_train_batch_size": 4,
        "per_device_eval_batch_size": 4,
        "gradient_accumulation_steps": 2,
        "seed": 42,
    }

    config_path = Path(args.config)
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            config.update(json.load(f))
    else:
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

    LOGGER.info("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    model = AutoModelForSeq2SeqLM.from_pretrained(config["model_name"])

    LOGGER.info("Loading datasets...")
    dataset = load_dataset(
        "json",
        data_files={
            "train": config["train_file"],
            "validation": config["validation_file"],
        },
    )

    def preprocess_function(examples):
        inputs = [
            f"{instr.strip()}\n\nContext:\n{ctx.strip()}"
            for instr, ctx in zip(examples["instruction"], examples["context"])
        ]
        model_inputs = tokenizer(
            inputs,
            max_length=config["max_source_length"],
            truncation=True,
        )

        labels = tokenizer(
            text_target=examples["target"],
            max_length=config["max_target_length"],
            truncation=True,
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    LOGGER.info("Tokenizing datasets...")
    tokenized_datasets = dataset.map(preprocess_function, batched=True)

    # Setup data collator for dynamic padding of both inputs and labels
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    training_args = Seq2SeqTrainingArguments(
        output_dir=config["output_dir"],
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=config["learning_rate"],
        per_device_train_batch_size=config["per_device_train_batch_size"],
        per_device_eval_batch_size=config["per_device_eval_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        num_train_epochs=config["num_train_epochs"],
        weight_decay=0.01,
        save_total_limit=1,
        predict_with_generate=True,
        seed=config["seed"],
        report_to="none",
        dataloader_drop_last=True,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        processing_class=tokenizer,
    )

    LOGGER.info("Beginning training loop...")
    train_result = trainer.train()

    final_dir = Path(config["output_dir"]) / "final"
    trainer.save_model(str(final_str := str(final_dir)))
    tokenizer.save_pretrained(str(final_dir))

    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

    summary_path = Path(config["output_dir"]) / "training_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({"config": config, "metrics": metrics}, f, indent=2)

    LOGGER.info("Fine-tuning complete! Saved final model to %s", final_dir)


if __name__ == "__main__":
    main()
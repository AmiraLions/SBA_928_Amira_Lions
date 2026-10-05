"""Step 2: Compare prompt variations using a pretrained model without fine-tuning."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from sba_928.prompts import build_prompt_variants

LOGGER = logging.getLogger(__name__)


def main() -> None:
    """Run prompt engineering evaluation."""
    parser = argparse.ArgumentParser(description="Run prompt engineering evaluation.")
    parser.add_argument("--limit", type=int, default=10, help="Number of test records to process.")
    parser.add_argument("--device", type=str, default="auto", help="Device to run inference on ('cpu', 'cuda', or 'auto').")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    LOGGER.info("Starting prompt engineering baseline evaluation...")

    test_file = Path("data/processed/test.jsonl")
    if not test_file.exists():
        raise FileNotFoundError(f"Test split not found at {test_file}. Run step_01 first.")

    records = []
    with open(test_file, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))

    records = records[: args.limit]

    # Determine device
    if args.device == "cpu":
        device = torch.device("cpu")
    elif args.device == "cuda":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    LOGGER.info("Loading model and tokenizer for google/flan-t5-small...")
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small").to(device)

    artifacts_dir = Path("artifacts/prompt_baseline")
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    predictions_file = artifacts_dir / "prompt_predictions.jsonl"

    with open(predictions_file, "w", encoding="utf-8") as out_f:
        for rec in records:
            context = rec["context"]
            variants = build_prompt_variants(context)

            for variant_name, prompt_text in variants.items():
                inputs = tokenizer(prompt_text, return_tensors="pt", max_length=256, truncation=True).to(device)
                
                with torch.no_grad():
                    outputs = model.generate(**inputs, max_new_tokens=96, do_sample=False)
                
                response = tokenizer.decode(outputs[0], skip_special_tokens=True)

                result_entry = {
                    "review_id": rec["review_id"],
                    "variant": variant_name,
                    "prompt": prompt_text,
                    "response": response,
                    "target": rec["target"],
                }
                out_f.write(json.dumps(result_entry) + "\n")

    LOGGER.info("Prompt engineering evaluation complete. Wrote results to %s", predictions_file)


if __name__ == "__main__":
    main()
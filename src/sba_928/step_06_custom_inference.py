"""Step 6: Run custom inference on new market research queries using the fine-tuned model."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

LOGGER = logging.getLogger(__name__)


def main() -> None:
    """Run inference on sample custom market research inputs."""
    parser = argparse.ArgumentParser(description="Run custom model inference.")
    parser.add_argument(
        "--model-path",
        type=str,
        default="artifacts/models/flan-t5-sba-928/final",
        help="Path to fine-tuned model.",
    )
    parser.add_argument(
        "--query",
        type=str,
        default="Product: Ergonomic Standing Desk | Category: furniture | Rating: 2/5 | Review: The motor makes a loud grinding noise and frequently sticks halfway up.",
        help="Custom market research review context query.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    LOGGER.info("Starting custom inference...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    LOGGER.info("Loading fine-tuned model from %s...", args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_path).to(device)

    instruction = "Identify the positive feature, primary problem, and recommendation."
    prompt = f"{instruction}\n\nContext:\n{args.query}"

    inputs = tokenizer(prompt, return_tensors="pt", max_length=256, truncation=True).to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=96,
            do_sample=False,
            repetition_penalty=2.5,
            no_repeat_ngram_size=3,
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print("\n" + "=" * 50)
    print("CUSTOM INFERENCE RESULT")
    print("=" * 50)
    print(f"Query / Context:\n{args.query}\n")
    print(f"Generated Recommendation / Analysis:\n{response}")
    print("=" * 50 + "\n")

    # Save inference artifact
    inference_dir = Path("artifacts/inference")
    inference_dir.mkdir(parents=True, exist_ok=True)
    output_file = inference_dir / "custom_inference_output.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({"query": args.query, "response": response}, f, indent=2)

    LOGGER.info("Custom inference complete. Saved output to %s", output_file)


if __name__ == "__main__":
    main()
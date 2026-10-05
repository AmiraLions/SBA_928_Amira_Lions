"""Step 4: Compare base and fine-tuned models on held-out test records."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

LOGGER = logging.getLogger(__name__)


def generate_response(model, tokenizer, prompt: str, device, max_new_tokens: int = 96) -> str:
    """Generate response given a prompt using a loaded model and tokenizer."""
    inputs = tokenizer(prompt, return_tensors="pt", max_length=256, truncation=True).to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def main() -> None:
    """Run base vs fine-tuned model comparison."""
    parser = argparse.ArgumentParser(description="Compare base and fine-tuned models.")
    parser.add_argument(
        "--fine-tuned-model",
        type=str,
        default="artifacts/models/flan-t5-sba-928/final",
        help="Path to fine-tuned model checkpoint.",
    )
    parser.add_argument("--device", type=str, default="auto", help="Device to use ('cpu', 'cuda', or 'auto').")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    LOGGER.info("Starting model comparison evaluation...")

    # Determine device
    if args.device == "cpu":
        device = torch.device("cpu")
    elif args.device == "cuda":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_file = Path("data/processed/test.jsonl")
    if not test_file.exists():
        raise FileNotFoundError(f"Test split not found at {test_file}. Run step_01 first.")

    records = []
    with open(test_file, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))

    LOGGER.info("Loading base model (google/flan-t5-small)...")
    base_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
    base_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small").to(device)

    LOGGER.info("Loading fine-tuned model from %s...", args.fine_tuned_model)
    ft_path = args.fine_tuned_model
    ft_tokenizer = AutoTokenizer.from_pretrained(ft_path)
    ft_model = AutoModelForSeq2SeqLM.from_pretrained(ft_path).to(device)

    eval_dir = Path("artifacts/evaluation")
    eval_dir.mkdir(parents=True, exist_ok=True)
    comparison_file = eval_dir / "model_comparison.jsonl"

    comparisons = []
    with open(comparison_file, "w", encoding="utf-8") as out_f:
        for rec in records:
            # Format input consistently with training
            prompt = f"{rec['instruction'].strip()}\n\nContext:\n{rec['context'].strip()}"

            base_resp = generate_response(base_model, base_tokenizer, prompt, device)
            ft_resp = generate_response(ft_model, ft_tokenizer, prompt, device)

            entry = {
                "review_id": rec["review_id"],
                "product": rec["product"],
                "prompt": prompt,
                "target": rec["target"],
                "base_response": base_resp,
                "fine_tuned_response": ft_resp,
            }
            out_f.write(json.dumps(entry) + "\n")
            comparisons.append(entry)

    summary = {
        "total_evaluated": len(comparisons),
        "base_model": "google/flan-t5-small",
        "fine_tuned_model": args.fine_tuned_model,
    }
    summary_file = eval_dir / "model_comparison_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    LOGGER.info("Model comparison complete. Wrote results to %s", comparison_file)


if __name__ == "__main__":
    main()
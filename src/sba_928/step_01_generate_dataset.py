"""Step 1: Generate and split the market research dataset."""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path

LOGGER = logging.getLogger(__name__)


def generate_market_research_records() -> list[dict]:
    """Generate synthetic market research review records."""
    products = [
        ("Reusable Water Bottle", "drinkware"),
        ("Smart Thermostat", "smart_home"),
        ("Ergonomic Office Chair", "furniture"),
        ("Wireless Noise-Canceling Headphones", "audio"),
        ("Robot Vacuum", "appliances"),
    ]
    
    instructions = [
        "Identify the positive feature, primary problem, and recommendation.",
        "Analyze consumer sentiment, key complaint, and pricing feedback.",
        "Summarize market trend, product feature strength, and suggested improvement."
    ]

    raw_reviews = [
        (2, "The bottle keeps drinks cold for hours, but the lid leaks constantly when tipped."),
        (5, "Exceptional battery life and supreme comfort, though the price point is a bit high."),
        (3, "Setup was straightforward, but the Wi-Fi connection drops every single evening."),
        (1, "Stopped working after two weeks of light use. Customer support was unhelpful."),
        (4, "Stunning audio clarity and deep bass, but the earcups feel tight during long sessions.")
    ]

    records = []
    record_id = 1

    for i in range(150):
        prod, category = products[i % len(products)]
        rating, review_text = raw_reviews[i % len(raw_reviews)]
        instruction = instructions[i % len(instructions)]

        if rating >= 4:
            target = "Positive feature: High quality and performance. Primary problem: Minor ergonomic or pricing notes. Recommendation: Maintain current standards while evaluating value."
        elif rating == 3:
            target = "Positive feature: Functional core features. Primary problem: Reliability or connectivity inconsistencies. Recommendation: Improve software stability and hardware durability."
        else:
            target = "Positive feature: Initial design intent. Primary problem: Premature failure or poor build quality. Recommendation: Enhance quality control and customer support responsiveness."

        context = f"Product: {prod} | Category: {category} | Rating: {rating}/5 | Review: {review_text}"

        record = {
            "review_id": f"MR-{record_id:04d}",
            "product": prod,
            "category": category,
            "rating": rating,
            "review": review_text,
            "instruction": instruction,
            "context": context,
            "target": target,
        }
        records.append(record)
        record_id += 1

    return records


def main() -> None:
    """Run dataset generation and splitting."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    LOGGER.info("Starting market research dataset generation...")

    records = generate_market_research_records()

    rng = random.Random(42)
    rng.shuffle(records)

    train_end = int(len(records) * 0.80)
    val_end = int(len(records) * 0.90)

    train_records = records[:train_end]
    val_records = records[train_end:val_end]
    test_records = records[val_end:]

    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)

    for split_name, split_data in [("train", train_records), ("validation", val_records), ("test", test_records)]:
        file_path = processed_dir / f"{split_name}.jsonl"
        with open(file_path, "w", encoding="utf-8") as f:
            for rec in split_data:
                f.write(json.dumps(rec) + "\n")
        LOGGER.info("Wrote %d records to %s", len(split_data), file_path)

    LOGGER.info("Dataset generation and splitting complete successfully!")


if __name__ == "__main__":
    main()
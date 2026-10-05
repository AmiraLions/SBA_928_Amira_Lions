"""Step 5: Evaluate model performance across product subgroups for potential bias."""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

LOGGER = logging.getLogger(__name__)


def main() -> None:
    """Run subgroup bias analysis on model comparison results."""
    parser = argparse.ArgumentParser(description="Evaluate subgroup performance.")
    parser.add_argument(
        "--comparison-file",
        type=str,
        default="artifacts/evaluation/model_comparison.jsonl",
        help="Path to model comparison jsonl file.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    LOGGER.info("Starting subgroup bias audit...")

    comp_path = Path(args.comparison_file)
    if not comp_path.exists():
        raise FileNotFoundError(f"Comparison file not found at {comp_path}. Run step_04 first.")

    subgroups = defaultdict(lambda: {"count": 0, "base_len": 0, "ft_len": 0})

    with open(comp_path, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            product = record.get("product", "Unknown")
            subgroups[product]["count"] += 1
            subgroups[product]["base_len"] += len(record.get("base_response", ""))
            subgroups[product]["ft_len"] += len(record.get("fine_tuned_response", ""))

    # Compute averages per subgroup
    summary = {}
    for product, stats in subgroups.items():
        count = stats["count"]
        summary[product] = {
            "sample_count": count,
            "avg_base_response_length": round(stats["base_len"] / count, 2) if count > 0 else 0,
            "avg_fine_tuned_response_length": round(stats["ft_len"] / count, 2) if count > 0 else 0,
        }

    eval_dir = Path("artifacts/evaluation")
    eval_dir.mkdir(parents=True, exist_ok=True)
    bias_report_path = eval_dir / "subgroup_bias_report.json"

    with open(bias_report_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    LOGGER.info("Subgroup bias audit complete. Report saved to %s", bias_report_path)


if __name__ == "__main__":
    main()
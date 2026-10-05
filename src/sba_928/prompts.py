"""Prompt variations for baseline comparison."""

from __future__ import annotations


def build_prompt_variants(context: str) -> dict[str, str]:
    """Return controlled prompt variations for a given review context."""
    return {
        "minimal": f"Analyze this market review.\n\n{context}",
        "structured": (
            "Identify the Positive feature, Primary problem, and Recommendation. "
            f"Use those labels exactly.\n\n{context}"
        ),
        "grounded": (
            "Use only the supplied review. Do not invent facts. Identify the "
            "Positive feature, Primary problem, and Recommendation.\n\n"
            f"{context}"
        ),
    }
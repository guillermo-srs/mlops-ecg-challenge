# File: src/reporting.py

from typing import List, Any, Tuple
import numpy as np


def print_results_summary(
    results_list: List[Tuple[Any, float]],
    top_n_count: int,
    search_type: str
):
    """
    Prints a summary of the top N similarity results.
    """
    if not results_list:
        print(f"\nNo similarity results to display for {search_type} search.")
        return

    print(f"\n--- Top {min(top_n_count, len(results_list))} Most Similar Rows ({search_type} Search) ---")
    for rank, (row_id, score) in enumerate(results_list):
        # The list should already be limited by top_n_count from get_top_n_from_rdd
        score_str = f"{score:.4f}" if score is not None and not np.isnan(score) else "N/A"
        print(f"  {rank + 1}. Row ID: '{row_id}', Similarity Score: {score_str}")

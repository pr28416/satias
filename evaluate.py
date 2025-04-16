import argparse
import json
import os
import sys
import math
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from scipy.stats import wilcoxon
from tqdm import tqdm

# --- Adjust path to import from parent directory --- START
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
# --- Adjust path --- END

import config
from search_engine import indexer, query_parser, searcher, utils
from search_engine.indexer import InvertedIndexType

# --- Helper Functions ---


def calculate_precision_at_k(ranked_ids: List[str], target_id: str, k: int) -> float:
    """Calculates Precision@k.

    Args:
        ranked_ids: List of retrieved image IDs, ordered by rank.
        target_id: The ground truth relevant image ID for the query.
        k: The cutoff rank (must be > 0).

    Returns:
        Precision@k score (float between 0.0 and 1.0).
    """
    if k <= 0:
        return 0.0
    top_k_ids = ranked_ids[:k]
    relevant_count = sum(1 for img_id in top_k_ids if img_id == target_id)
    return float(relevant_count) / k


def calculate_average_precision(ranked_ids: List[str], target_id: str) -> float:
    """Calculates Average Precision (AP) for a single query.
       Since we have only one relevant item, AP = 1/rank if found, else 0.

    Args:
        ranked_ids: List of retrieved image IDs, ordered by rank.
        target_id: The ground truth relevant image ID for the query.

    Returns:
        Average Precision score (float between 0.0 and 1.0).
    """
    try:
        # Find the 1-based rank of the target ID
        rank = ranked_ids.index(target_id) + 1
        return 1.0 / rank
    except ValueError:
        # Target ID was not found in the ranked list
        return 0.0


# --- Main Evaluation Logic ---


def run_evaluation(queries_path: str, index_path: str, k: int) -> Tuple[
    Optional[float],
    Optional[float],
    Optional[float],
    Optional[float],
    Optional[float],
    Optional[int],
]:
    """Runs the evaluation process and returns MAP and P@k for both configs and p-value.

    Args:
        queries_path: Path to the queries JSON file.
        index_path: Path to the search index file.
        k: The cutoff rank for P@k and MAP.

    Returns:
        A tuple (map_config, pk_config, map_baseline, pk_baseline, p_value, num_valid_queries).
        Returns (None, ..., None) on error.
    """
    print("--- Starting Evaluation ---")

    # --- Load Queries ---
    print(f"Loading queries from: {queries_path}...")
    try:
        with open(queries_path, "r") as f:
            queries: List[Dict[str, Any]] = json.load(f)
        print(f"Loaded {len(queries)} queries.")
    except FileNotFoundError:
        print(f"Error: Queries file not found at {queries_path}")
        return None, None, None, None, None, 0
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {queries_path}")
        return None, None, None, None, None, 0

    # --- Load Index ---
    print(f"Loading index from: {index_path}...")
    inverted_index: Optional[InvertedIndexType] = indexer.load_index(index_path)
    if inverted_index is None:
        print("Failed to load index. Exiting evaluation.")
        return None, None, None, None, None, 0
    print("Index loaded successfully.")

    # --- Store results for each config ---
    config_ap_scores: List[float] = []
    config_p_at_k_scores: List[float] = []
    baseline_ap_scores: List[float] = []
    baseline_p_at_k_scores: List[float] = []
    num_valid_queries: int = 0

    query_count: int = len(queries)
    print(f"Evaluating {query_count} queries...")

    # --- Run Evaluation for Both Configurations ---
    for i, query in enumerate(tqdm(queries, desc="Evaluating Queries", unit="query")):
        # --- Extract query info (same for both runs) ---
        query_text: Optional[str] = query.get("query_text")
        query_bbox_norm: Optional[List[float]] = query.get("query_region_norm")
        target_image_id: Optional[str] = query.get("image_id")

        if not query_text or query_bbox_norm is None or not target_image_id:
            # print(f"Warning: Skipping query {i+1} due to missing data: {query}")
            continue

        query_ngrams: List[str] = utils.generate_ngrams_from_text(query_text)
        if not query_ngrams:
            # print(f"Warning: No n-grams generated for query {i+1}. Skipping.")
            continue

        num_valid_queries += 1  # Count queries processed for both configs

        # --- Run 1: Current Configuration ---
        image_scores_config: Dict[str, float] = searcher.search_images(
            query_ngrams, query_bbox_norm, inverted_index, is_baseline_search=False
        )
        ranked_results_config: List[Tuple[str, float]] = searcher.rank_results(
            image_scores_config
        )
        ranked_ids_config: List[str] = [res[0] for res in ranked_results_config]
        ap_config = calculate_average_precision(ranked_ids_config, target_image_id)
        pk_config = calculate_precision_at_k(ranked_ids_config, target_image_id, k)
        config_ap_scores.append(ap_config)
        config_p_at_k_scores.append(pk_config)

        # --- Run 2: Baseline (Non-Spatial) Configuration ---
        image_scores_baseline: Dict[str, float] = searcher.search_images(
            query_ngrams, query_bbox_norm, inverted_index, is_baseline_search=True
        )
        ranked_results_baseline: List[Tuple[str, float]] = searcher.rank_results(
            image_scores_baseline
        )
        ranked_ids_baseline: List[str] = [res[0] for res in ranked_results_baseline]
        ap_baseline = calculate_average_precision(ranked_ids_baseline, target_image_id)
        pk_baseline = calculate_precision_at_k(ranked_ids_baseline, target_image_id, k)
        baseline_ap_scores.append(ap_baseline)
        baseline_p_at_k_scores.append(pk_baseline)

    # --- Calculate Overall Metrics ---
    if num_valid_queries == 0:  # Check if any queries were successfully processed
        print("Error: No queries were successfully evaluated.")
        return None, None, None, None, None, 0

    # Use np.mean for calculating averages, cast to float for type consistency
    map_config: float = float(np.mean(config_ap_scores))
    pk_config_mean: float = float(np.mean(config_p_at_k_scores))
    map_baseline: float = float(np.mean(baseline_ap_scores))
    pk_baseline_mean: float = float(np.mean(baseline_p_at_k_scores))

    # --- Perform Statistical Test (Wilcoxon signed-rank test on AP scores) ---
    print(
        "\nPerforming statistical significance test (Wilcoxon signed-rank) on AP scores..."
    )
    # Calculate differences, filtering pairs where both are 0 (no info)
    ap_differences: List[float] = [
        c - b
        for c, b in zip(config_ap_scores, baseline_ap_scores)
        if not (c == 0 and b == 0)
    ]

    p_value: Optional[float] = None
    wilcoxon_statistic: Optional[float] = None
    if not ap_differences:
        print(
            "  Warning: No non-zero differences found between config and baseline AP scores. Cannot perform Wilcoxon test."
        )
        p_value = 1.0  # Assign non-significant p-value
    else:
        try:
            # Use alternative='greater' to test if config scores are significantly GREATER than baseline
            wilcoxon_statistic, p_value = wilcoxon(
                ap_differences, alternative="greater"
            )
            print(
                f"  Wilcoxon test statistic: {wilcoxon_statistic:.4f}, p-value: {p_value:.4f}"
            )
        except ValueError as e:
            print(f"  Warning: Could not perform Wilcoxon test: {e}")
            # This might happen if all differences are identical (e.g., all 0 after filtering)
            p_value = 1.0  # Assign non-significant p-value

    print("--- Evaluation Complete ---")
    return (
        map_config,
        pk_config_mean,
        map_baseline,
        pk_baseline_mean,
        p_value,
        num_valid_queries,
    )


# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate Spatially-Aware Textual Image Search"
    )
    parser.add_argument(
        "--queries",
        default=config.QUERIES_FILE,
        help=f"Path to the queries JSON file (default: {config.QUERIES_FILE})",
    )
    parser.add_argument(
        "--index",
        default=indexer.DEFAULT_INDEX_PATH,
        help=f"Path to the search index file (default: {indexer.DEFAULT_INDEX_PATH})",
    )
    parser.add_argument(
        "-k",
        type=int,
        default=10,
        help="Value of k for Precision@k and MAP calculation (default: 10)",
    )

    args = parser.parse_args()

    map_config, pk_config, map_baseline, pk_baseline, p_value, num_queries_eval = (
        run_evaluation(args.queries, args.index, args.k)
    )

    if map_config is not None:  # Check if evaluation ran successfully
        print(f"\n--- Evaluation Results (k={args.k}, N={num_queries_eval}) --- ")
        # --- Print Table Header ---
        print("+----------------------------+------------------+----------------+")
        print("| Metric                     | Current Config   | Baseline       |")
        print("+----------------------------+------------------+----------------+")
        # --- Print MAP Results ---
        print(
            f"| Mean Average Precision (MAP@{args.k}) | {map_config:<16.4f} | {map_baseline:<14.4f} |"
        )
        # --- Print P@k Results ---
        print(
            f"| Mean Precision@k (P@{args.k})       | {pk_config:<16.4f} | {pk_baseline:<14.4f} |"
        )
        print("+----------------------------+------------------+----------------+")

        # --- Print Statistical Significance ---
        if p_value is not None:
            print(f"\nStatistical Significance (Wilcoxon Test comparing AP scores):")
            print(f"  p-value = {p_value:.4f}")
            if p_value < 0.05:
                print(
                    "  Result: The improvement of the Current Config over Baseline is STATISTICALLY SIGNIFICANT (p < 0.05)."
                )
            else:
                print(
                    "  Result: The difference between Current Config and Baseline is NOT statistically significant (p >= 0.05)."
                )
        else:
            print("\nStatistical Significance test could not be performed.")
        print("----------------------------------------------------------------")

        # --- Print Interpretation Guidance (Contextual) ---
        print("\n--- Interpretation Guidance --- ")
        print(
            " * Baseline = Non-spatial scoring (ignores query region, score based only on n-gram presence/length)."
        )
        print(
            " * Current Config = Spatial scoring using weights from config.py (IOU: {config.IOU_WEIGHT}, Proximity: {config.PROXIMITY_WEIGHT})."
        )
        print(" * MAP closer to 1.0 is better (perfect ranking).")
        print(
            f" * P@{args.k} closer to {1.0/args.k:.2f} is better (correct item usually in top {args.k})."
        )
        print(
            " * Statistically Significant means the observed difference in MAP is unlikely due to random chance alone for this query set."
        )
        print("----------------------------------------------------------------")

    else:
        print("\nEvaluation could not be completed due to errors.")
        sys.exit(1)

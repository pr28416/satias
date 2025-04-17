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
    Optional[float],  # Config MAP, P@k
    Optional[float],
    Optional[float],  # N-gram Baseline MAP, P@k
    Optional[float],
    Optional[float],  # Keyword Baseline MAP, P@k
    Optional[float],
    Optional[float],  # p-value (config vs ngram), p-value (ngram vs keyword)
    Optional[int],  # num_valid_queries
]:
    """Runs the evaluation process and returns MAP/P@k for all configs and p-values.

    Args:
        queries_path: Path to the queries JSON file.
        index_path: Path to the search index file.
        k: The cutoff rank for P@k and MAP.

    Returns:
        A tuple (map_cfg, pk_cfg, map_ngram, pk_ngram, map_key, pk_key, pval1, pval2, num_valid).
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
        return None, None, None, None, None, None, None, None, 0
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {queries_path}")
        return None, None, None, None, None, None, None, None, 0

    # --- Load Index ---
    print(f"Loading index from: {index_path}...")
    inverted_index: Optional[InvertedIndexType] = indexer.load_index(index_path)
    if inverted_index is None:
        print("Failed to load index. Exiting evaluation.")
        return None, None, None, None, None, None, None, None, 0
    print("Index loaded successfully.")

    # --- Store results for each config ---
    config_ap_scores: List[float] = []
    config_p_at_k_scores: List[float] = []
    ngram_baseline_ap_scores: List[float] = []
    ngram_baseline_p_at_k_scores: List[float] = []
    keyword_baseline_ap_scores: List[float] = []
    keyword_baseline_p_at_k_scores: List[float] = []
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

        # --- Run 1: Current Configuration (Spatial) ---
        image_scores_config: Dict[str, float] = searcher.search_images(
            query_ngrams, query_bbox_norm, inverted_index, search_mode="spatial"
        )
        ranked_results_config: List[Tuple[str, float]] = searcher.rank_results(
            image_scores_config
        )
        ranked_ids_config: List[str] = [res[0] for res in ranked_results_config]
        ap_config = calculate_average_precision(ranked_ids_config, target_image_id)
        pk_config = calculate_precision_at_k(ranked_ids_config, target_image_id, k)
        config_ap_scores.append(ap_config)
        config_p_at_k_scores.append(pk_config)

        # --- Run 2: N-gram Baseline (Text Only) ---
        image_scores_ngram_baseline: Dict[str, float] = searcher.search_images(
            query_ngrams,
            query_bbox_norm,  # Bbox needed for function call, but ignored internally by mode
            inverted_index,
            search_mode="ngram_text_only",
        )
        ranked_results_ngram_baseline: List[Tuple[str, float]] = searcher.rank_results(
            image_scores_ngram_baseline
        )
        ranked_ids_ngram_baseline: List[str] = [
            res[0] for res in ranked_results_ngram_baseline
        ]
        ap_ngram_baseline = calculate_average_precision(
            ranked_ids_ngram_baseline, target_image_id
        )
        pk_ngram_baseline = calculate_precision_at_k(
            ranked_ids_ngram_baseline, target_image_id, k
        )
        ngram_baseline_ap_scores.append(ap_ngram_baseline)
        ngram_baseline_p_at_k_scores.append(pk_ngram_baseline)

        # --- Run 3: Keyword Baseline (Simple Text) ---
        # Note: Passes ngrams list, but searcher extracts words from it for keyword mode
        image_scores_keyword_baseline: Dict[str, float] = searcher.search_images(
            query_ngrams,
            query_bbox_norm,  # Ignored internally by mode
            inverted_index,
            search_mode="keyword_only",
        )
        ranked_results_keyword_baseline: List[Tuple[str, float]] = (
            searcher.rank_results(image_scores_keyword_baseline)
        )
        ranked_ids_keyword_baseline: List[str] = [
            res[0] for res in ranked_results_keyword_baseline
        ]
        ap_keyword_baseline = calculate_average_precision(
            ranked_ids_keyword_baseline, target_image_id
        )
        pk_keyword_baseline = calculate_precision_at_k(
            ranked_ids_keyword_baseline, target_image_id, k
        )
        keyword_baseline_ap_scores.append(ap_keyword_baseline)
        keyword_baseline_p_at_k_scores.append(pk_keyword_baseline)

    # --- Calculate Overall Metrics ---
    if num_valid_queries == 0:  # Check if any queries were successfully processed
        print("Error: No queries were successfully evaluated.")
        return None, None, None, None, None, None, None, None, 0

    # Use np.mean for calculating averages, cast to float for type consistency
    map_config: float = float(np.mean(config_ap_scores))
    pk_config_mean: float = float(np.mean(config_p_at_k_scores))
    map_ngram_baseline: float = float(np.mean(ngram_baseline_ap_scores))
    pk_ngram_baseline_mean: float = float(np.mean(ngram_baseline_p_at_k_scores))
    map_keyword_baseline: float = float(np.mean(keyword_baseline_ap_scores))
    pk_keyword_baseline_mean: float = float(np.mean(keyword_baseline_p_at_k_scores))

    # --- Perform Statistical Tests (Wilcoxon signed-rank test on AP scores) ---
    print(
        "\nPerforming statistical significance tests (Wilcoxon signed-rank) on AP scores..."
    )

    # Test 1: Config vs N-gram Baseline
    p_value_config_vs_ngram: Optional[float] = None
    print("  Test 1: Current Config vs. N-gram Baseline")
    ap_diff_config_ngram: List[float] = [
        c - b
        for c, b in zip(config_ap_scores, ngram_baseline_ap_scores)
        if not (c == 0 and b == 0)
    ]
    wilcoxon_statistic: Optional[float] = None
    if not ap_diff_config_ngram:
        print(
            "    Warning: No non-zero differences found. Cannot perform Wilcoxon test."
        )
        p_value_config_vs_ngram = 1.0
    else:
        try:
            wilcoxon_statistic, p_value_config_vs_ngram = wilcoxon(
                ap_diff_config_ngram, alternative="greater"
            )
            print(
                f"    Wilcoxon test statistic: {wilcoxon_statistic:.4f}, p-value: {p_value_config_vs_ngram:.4f}"
            )
        except ValueError as e:
            print(f"    Warning: Could not perform Wilcoxon test: {e}")
            p_value_config_vs_ngram = 1.0

    # Test 2: N-gram Baseline vs Keyword Baseline
    p_value_ngram_vs_keyword: Optional[float] = None
    print("\n  Test 2: N-gram Baseline vs. Keyword Baseline")
    ap_diff_ngram_keyword: List[float] = [
        n - k
        for n, k in zip(ngram_baseline_ap_scores, keyword_baseline_ap_scores)
        if not (n == 0 and k == 0)
    ]
    if not ap_diff_ngram_keyword:
        print(
            "    Warning: No non-zero differences found. Cannot perform Wilcoxon test."
        )
        p_value_ngram_vs_keyword = 1.0
    else:
        try:
            wilcoxon_statistic, p_value_ngram_vs_keyword = wilcoxon(
                ap_diff_ngram_keyword, alternative="greater"
            )
            print(
                f"    Wilcoxon test statistic: {wilcoxon_statistic:.4f}, p-value: {p_value_ngram_vs_keyword:.4f}"
            )
        except ValueError as e:
            print(f"    Warning: Could not perform Wilcoxon test: {e}")
            p_value_ngram_vs_keyword = 1.0

    print("--- Evaluation Complete ---")
    return (
        map_config,
        pk_config_mean,
        map_ngram_baseline,
        pk_ngram_baseline_mean,
        map_keyword_baseline,
        pk_keyword_baseline_mean,
        p_value_config_vs_ngram,
        p_value_ngram_vs_keyword,
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

    (
        map_config,
        pk_config,
        map_ngram,
        pk_ngram,
        map_keyword,
        pk_keyword,
        p_val1,
        p_val2,
        num_queries_eval,
    ) = run_evaluation(args.queries, args.index, args.k)

    if map_config is not None:  # Check if evaluation ran successfully
        print(f"\n--- Evaluation Results (k={args.k}, N={num_queries_eval}) --- ")
        # --- Print Table Header ---
        print(
            "+----------------------------+------------------+----------------+----------------+"
        )
        print(
            "| Metric                     | Spatial N-gram   | N-gram Baseline| Keyword Baseline|"
        )
        print(
            "+----------------------------+------------------+----------------+----------------+"
        )
        # --- Print MAP Results ---
        print(
            f"| Mean Average Precision (MAP@{args.k}) | {map_config:<16.4f} | {map_ngram:<14.4f} | {map_keyword:<15.4f} |"
        )
        # --- Print P@k Results ---
        print(
            f"| Mean Precision@k (P@{args.k})       | {pk_config:<16.4f} | {pk_ngram:<14.4f} | {pk_keyword:<15.4f} |"
        )
        print(
            "+----------------------------+------------------+----------------+----------------+"
        )

        # --- Print Statistical Significance ---
        if p_val1 is not None:
            print(
                f"\nStat. Significance (Config vs N-gram Baseline - Wilcoxon on AP scores):"
            )
            print(f"  p-value = {p_val1:.4f}")
            if p_val1 < 0.05:
                print(
                    "  Result: The improvement of the Spatial N-gram Config over N-gram Baseline is STATISTICALLY SIGNIFICANT (p < 0.05)."
                )
            else:
                print(
                    "  Result: The difference between Spatial N-gram Config and N-gram Baseline is NOT statistically significant (p >= 0.05)."
                )
        else:
            print(
                "\nStatistical Significance test (Config vs N-gram) could not be performed."
            )

        if p_val2 is not None:
            print(
                f"\nStat. Significance (N-gram Baseline vs Keyword Baseline - Wilcoxon on AP scores):"
            )
            print(f"  p-value = {p_val2:.4f}")
            if p_val2 < 0.05:
                print(
                    "  Result: The improvement of the N-gram Baseline over Keyword Baseline is STATISTICALLY SIGNIFICANT (p < 0.05)."
                )
            else:
                print(
                    "  Result: The difference between N-gram Baseline and Keyword Baseline is NOT statistically significant (p >= 0.05)."
                )
        else:
            print(
                "\nStatistical Significance test (N-gram vs Keyword) could not be performed."
            )
        print("----------------------------------------------------------------")

        # --- Print Interpretation Guidance (Contextual) ---
        print("\n--- Interpretation Guidance --- ")
        print(
            " * Keyword Baseline = Simple word matching (score = # query words found)."
        )
        print(
            " * N-gram Baseline = Non-spatial scoring (score based only on n-gram presence/length)."
        )
        print(
            " * Current Config = Spatial scoring using weights from config.py (IOU: {config.IOU_WEIGHT}, Proximity: {config.PROXIMITY_WEIGHT})."
        )
        print(
            f" * MAP closer to 1.0 is better (perfect ranking). P@{args.k} closer to {1.0/args.k:.2f} is better (correct item usually in top {args.k})."
        )
        print(
            " * Statistically Significant means the observed difference in MAP is unlikely due to random chance alone for this query set."
        )
        print("----------------------------------------------------------------")

    else:
        print("\nEvaluation could not be completed due to errors.")
        sys.exit(1)

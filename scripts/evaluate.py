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

# For storing sensitivity analysis results
sensitivity_results = {
    "weight_configs": [],  # Will store [iou_weight, prox_weight] pairs
    "metrics": []          # Will store corresponding metrics for each config
}

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


K_VALUES = [1, 5, 10]  # Define the k values to evaluate


def run_evaluation(queries_path: str, index_path: str) -> Tuple[
    Optional[float],  # Config MAP
    Optional[Dict[int, float]],  # Config P@k dict
    Optional[float],  # N-gram Baseline MAP
    Optional[Dict[int, float]],  # N-gram Baseline P@k dict
    Optional[float],  # Keyword Baseline MAP
    Optional[Dict[int, float]],  # Keyword Baseline P@k dict
    Optional[float],
    Optional[float],
    Optional[float],  # p-value (cfg vs ngram), (ngram vs key), (cfg vs key)
    Optional[int],  # num_valid_queries
    Optional[List[float]]  # AP scores for statistical comparison
]:
    """Runs the evaluation process and returns MAP/P@k for all configs and p-values.

    Args:
        queries_path: Path to the queries JSON file.
        index_path: Path to the search index file.

    Returns:
        A tuple (map_cfg, pk_cfg_dict, map_ngram, pk_ngram_dict, map_key, pk_key_dict,
                 pval1, pval2, pval3, num_valid).
        Returns (None, ..., None) on error.
    """
    print("--- Starting Evaluation ---")
    print(f"Evaluating for k values: {K_VALUES}")

    # --- Load Queries ---
    print(f"Loading queries from: {queries_path}...")
    try:
        with open(queries_path, "r") as f:
            queries: List[Dict[str, Any]] = json.load(f)
        print(f"Loaded {len(queries)} queries.")
    except FileNotFoundError:
        print(f"Error: Queries file not found at {queries_path}")
        return None, None, None, None, None, None, None, None, None, 0, None
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {queries_path}")
        return None, None, None, None, None, None, None, None, None, 0, None

    # --- Load Index ---
    print(f"Loading index from: {index_path}...")
    inverted_index: Optional[InvertedIndexType] = indexer.load_index(index_path)
    if inverted_index is None:
        print("Failed to load index. Exiting evaluation.")
        return None, None, None, None, None, None, None, None, None, 0, None
    print("Index loaded successfully.")

    # --- Store results for each config ---
    config_ap_scores: List[float] = []
    ngram_baseline_ap_scores: List[float] = []
    keyword_baseline_ap_scores: List[float] = []

    # Store P@k scores as dictionaries {k: [list of scores]}
    config_p_at_k_scores: Dict[int, List[float]] = {k: [] for k in K_VALUES}
    ngram_baseline_p_at_k_scores: Dict[int, List[float]] = {k: [] for k in K_VALUES}
    keyword_baseline_p_at_k_scores: Dict[int, List[float]] = {k: [] for k in K_VALUES}

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
        config_ranked_ids: List[str] = [res[0] for res in ranked_results_config]

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
        ngram_ranked_ids: List[str] = [res[0] for res in ranked_results_ngram_baseline]

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
        keyword_ranked_ids: List[str] = [
            res[0] for res in ranked_results_keyword_baseline
        ]

        # Calculate AP (k-independent)
        ap_config = calculate_average_precision(config_ranked_ids, target_image_id)
        ap_ngram = calculate_average_precision(ngram_ranked_ids, target_image_id)
        ap_keyword = calculate_average_precision(keyword_ranked_ids, target_image_id)

        config_ap_scores.append(ap_config)
        ngram_baseline_ap_scores.append(ap_ngram)
        keyword_baseline_ap_scores.append(ap_keyword)

        # Calculate P@k for each k in K_VALUES
        for k_val in K_VALUES:
            pk_config = calculate_precision_at_k(
                config_ranked_ids, target_image_id, k_val
            )
            pk_ngram = calculate_precision_at_k(
                ngram_ranked_ids, target_image_id, k_val
            )
            pk_keyword = calculate_precision_at_k(
                keyword_ranked_ids, target_image_id, k_val
            )

            config_p_at_k_scores[k_val].append(pk_config)
            ngram_baseline_p_at_k_scores[k_val].append(pk_ngram)
            keyword_baseline_p_at_k_scores[k_val].append(pk_keyword)

    print("\nFinished processing queries.")
    if num_valid_queries == 0:  # Check if any queries were successfully processed
        print("Error: No queries were successfully evaluated.")
        return None, None, None, None, None, None, None, None, None, 0, None

    # --- Calculate Mean Scores ---
    mean_ap_config = float(np.mean(config_ap_scores).item())
    mean_ap_ngram = float(np.mean(ngram_baseline_ap_scores).item())
    mean_ap_keyword = float(np.mean(keyword_baseline_ap_scores).item())

    # Calculate mean P@k for each k
    mean_pk_config_dict: Dict[int, float] = {
        k: float(np.mean(scores).item()) for k, scores in config_p_at_k_scores.items()
    }
    mean_pk_ngram_dict: Dict[int, float] = {
        k: float(np.mean(scores).item())
        for k, scores in ngram_baseline_p_at_k_scores.items()
    }
    mean_pk_keyword_dict: Dict[int, float] = {
        k: float(np.mean(scores).item())
        for k, scores in keyword_baseline_p_at_k_scores.items()
    }

    # --- Perform Statistical Tests ---
    p_value_config_vs_ngram = None
    p_value_ngram_vs_keyword = None
    p_value_config_vs_keyword = None

    try:
        # Test 1: Config vs N-gram Baseline
        diff_config_ngram = np.array(config_ap_scores) - np.array(
            ngram_baseline_ap_scores
        )
        # Filter out zero differences for Wilcoxon
        non_zero_diff_config_ngram = diff_config_ngram[diff_config_ngram != 0]
        if len(non_zero_diff_config_ngram) > 10:  # Need sufficient non-zero differences
            stat, p_value_config_vs_ngram = wilcoxon(
                non_zero_diff_config_ngram, alternative="greater"
            )
        else:
            print("Warning: Not enough non-zero differences for Config vs N-gram test.")

        # Test 2: N-gram Baseline vs Keyword Baseline
        diff_ngram_keyword = np.array(ngram_baseline_ap_scores) - np.array(
            keyword_baseline_ap_scores
        )
        non_zero_diff_ngram_keyword = diff_ngram_keyword[diff_ngram_keyword != 0]
        if len(non_zero_diff_ngram_keyword) > 10:
            stat, p_value_ngram_vs_keyword = wilcoxon(
                non_zero_diff_ngram_keyword, alternative="greater"
            )
        else:
            print(
                "Warning: Not enough non-zero differences for N-gram vs Keyword test."
            )

        # Test 3: Config vs Keyword Baseline
        diff_config_keyword = np.array(config_ap_scores) - np.array(
            keyword_baseline_ap_scores
        )
        non_zero_diff_config_keyword = diff_config_keyword[diff_config_keyword != 0]
        if len(non_zero_diff_config_keyword) > 10:
            stat, p_value_config_vs_keyword = wilcoxon(
                non_zero_diff_config_keyword, alternative="greater"
            )
        else:
            print(
                "Warning: Not enough non-zero differences for Config vs Keyword test."
            )

    except Exception as e:
        print(f"Error during statistical tests: {e}")
        # Continue without p-values if tests fail

    print("--- Evaluation Complete ---")

    return (
        mean_ap_config,
        mean_pk_config_dict,
        mean_ap_ngram,
        mean_pk_ngram_dict,
        mean_ap_keyword,
        mean_pk_keyword_dict,
        p_value_config_vs_ngram,
        p_value_ngram_vs_keyword,
        p_value_config_vs_keyword,
        num_valid_queries,
        config_ap_scores
    )


# --- Main Execution ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate SATIAS on validation queries")
    parser.add_argument(
        "--queries",
        type=str,
        default="synthetic_data/metadata/queries.json",  # Correct default path
        help="Path to the generated queries JSON file (default: synthetic_data/metadata/queries.json)",
    )
    parser.add_argument(
        "--index",
        type=str,
        default="synthetic_data/index.pkl",  # Correct default path
        help="Path to the search index file (default: synthetic_data/index.pkl)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="evaluation_results.json",
        help="Path to save the evaluation results JSON (default: evaluation_results.json)",
    )
    # Removed --k argument
    args = parser.parse_args()

    # Start sensitivity analysis
    print("\nStarting Sensitivity Analysis on IoU/Proximity Weights (Original: 0.5/0.5)")
    
    # Original weights
    original_iou_weight = config.IOU_WEIGHT
    original_prox_weight = config.PROXIMITY_WEIGHT
    
    # Weight configurations to test
    weight_configs = [
        (0.0, 1.0),   # Only proximity
        (0.25, 0.75), # More proximity, less IoU
        (0.5, 0.5),   # Original balanced weights
        (0.75, 0.25), # More IoU, less proximity
        (1.0, 0.0),   # Only IoU
    ]
    
    # Store AP scores from the default config for comparison
    default_ap_scores = None
    
    # Loop through each weight configuration
    for iou_weight, prox_weight in weight_configs:
        config.IOU_WEIGHT = iou_weight
        config.PROXIMITY_WEIGHT = prox_weight
        
        print(f"\n===== Running Evaluation with IoU Weight: {iou_weight:.2f}, Proximity Weight: {prox_weight:.2f} =====")
        
        (
            map_config,
            pk_config_dict,
            map_ngram,
            pk_ngram_dict,
            map_keyword,
            pk_keyword_dict,
            p_val1,  # Cfg vs Ng
            p_val2,  # Ng vs Key
            p_val3,  # Cfg vs Key
            num_queries_eval,
            ap_scores,  # Store the AP scores for statistical comparison
        ) = run_evaluation(args.queries, args.index)
        
        # Save metrics for this weight configuration
        config_metrics = {
            "iou_weight": iou_weight,
            "proximity_weight": prox_weight,
            "map": map_config,
            "p@k": pk_config_dict,
            "map_ngram": map_ngram,
            "p@k_ngram": pk_ngram_dict,
            "map_keyword": map_keyword,
            "p@k_keyword": pk_keyword_dict
        }
        
        # Add to sensitivity results
        sensitivity_results["weight_configs"].append([iou_weight, prox_weight])
        sensitivity_results["metrics"].append(config_metrics)
        
        # If this is the default 0.5/0.5 configuration, save the AP scores for comparison
        if iou_weight == 0.5 and prox_weight == 0.5:
            default_ap_scores = ap_scores
            
        if all(
            v is not None
            for v in [
                map_config,
                pk_config_dict,
                map_ngram,
                pk_ngram_dict,
                map_keyword,
                pk_keyword_dict,
            ]
        ):
            # Display results as before
            print(f"\n--- Evaluation Results (N={num_queries_eval}) --- ")
            # --- Print Table Header ---
            # Adjusted table width for readability
            print(
                "+----------------------------+------------------+-------------------+------------------+"
            )
            print(
                "| Metric                     | SATIAS           | N-gram Baseline   | Keyword Baseline |"
            )
            print(
                "+----------------------------+------------------+-------------------+------------------+"
            )
            # --- Print MAP Results ---
            # Renamed to MAP as it's k-independent here
            print(
                f"| Mean Average Precision (MAP) | {map_config:<16.4f} | {map_ngram:<17.4f} | {map_keyword:<16.4f} |"
            )
            print(
                "+----------------------------+------------------+-------------------+------------------+"
            )
            # --- Print P@k Results for each k ---
            for k_val in K_VALUES:
                pk_c = pk_config_dict.get(k_val, float("nan"))
                pk_n = pk_ngram_dict.get(k_val, float("nan"))  # Use safe var
                pk_k = pk_keyword_dict.get(k_val, float("nan"))  # Use safe var
                print(
                    f"| Mean Precision@{k_val:<10} | {pk_c:<16.4f} | {pk_n:<17.4f} | {pk_k:<16.4f} |"
                )

            print(
                "+----------------------------+------------------+-------------------+------------------+"
            )

            # --- Print Statistical Significance ---
            print("\nStatistical Significance (Wilcoxon Signed-Rank Test on AP scores):")
            if p_val1 is not None:
                result_1 = "SIGNIFICANT" if p_val1 < 0.05 else "NOT significant"
                print(f"  * SATIAS vs. N-gram Baseline:      p={p_val1:<.4f} ({result_1})")
            else:
                print("  * SATIAS vs. N-gram Baseline:      Test could not be performed.")

            if p_val2 is not None:
                result_2 = "SIGNIFICANT" if p_val2 < 0.05 else "NOT significant"
                print(
                    f"  * N-gram Baseline vs. Keyword Baseline: p={p_val2:<.4f} ({result_2})"
                )
            else:
                print(
                    "  * N-gram Baseline vs. Keyword Baseline: Test could not be performed."
                )

            if p_val3 is not None:
                result_3 = "SIGNIFICANT" if p_val3 < 0.05 else "NOT significant"
                print(
                    f"  * SATIAS vs. Keyword Baseline:       p={p_val3:<.4f} ({result_3})"
                )
            else:
                print("  * SATIAS vs. Keyword Baseline:       Test could not be performed.")
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
                f" * SATIAS = Spatial scoring using weights from config.py (IOU: {config.IOU_WEIGHT}, Proximity: {config.PROXIMITY_WEIGHT})."
            )
            print(" * MAP closer to 1.0 is better (perfect ranking).")
            for k_val in K_VALUES:
                print(
                    f" * P@{k_val} closer to {1.0/k_val:.2f} is better (correct item usually in top {k_val}). Max value is {1.0/k_val:.2f}."
                )
            print(
                " * Statistically Significant means the observed difference in MAP is unlikely due to random chance alone for this query set."
            )
            print("----------------------------------------------------------------")

    # Restore original weights
    config.IOU_WEIGHT = original_iou_weight
    config.PROXIMITY_WEIGHT = original_prox_weight
    print(f"\n===== Sensitivity Analysis Complete. Restored original weights (IoU: {original_iou_weight}, Prox: {original_prox_weight}) =====")

    # Statistical comparison against the default (0.5/0.5) for all configs
    if default_ap_scores is not None:
        print("\n===== Statistical Significance of Weight Differences (vs. 0.5/0.5) ====")
        
        # Add a new section in the results for statistical comparisons
        sensitivity_results["statistical_comparisons"] = []
        
        for i, (iou_weight, prox_weight) in enumerate(weight_configs):
            if iou_weight == 0.5 and prox_weight == 0.5:
                continue  # Skip comparing default with itself
                
            config_ap_scores = sensitivity_results["metrics"][i].get("ap_scores", [])
            
            if config_ap_scores:  # Only if we have AP scores
                try:
                    diff = np.array(default_ap_scores) - np.array(config_ap_scores)
                    non_zero_diff = diff[diff != 0]
                    
                    if len(non_zero_diff) > 10:
                        stat, p_value = wilcoxon(non_zero_diff, alternative="greater")
                        significant = p_value < 0.05
                        print(f"  * 0.5/0.5 vs. {iou_weight:.2f}/{prox_weight:.2f}: p={p_value:.4f} ({'SIGNIFICANT' if significant else 'NOT significant'})")
                        
                        # Add to results
                        sensitivity_results["statistical_comparisons"].append({
                            "config1": [0.5, 0.5],
                            "config2": [iou_weight, prox_weight],
                            "p_value": float(p_value),
                            "significant": significant
                        })
                    else:
                        print(f"  * 0.5/0.5 vs. {iou_weight:.2f}/{prox_weight:.2f}: Not enough non-zero differences for test")
                except Exception as e:
                    print(f"  * Error comparing 0.5/0.5 vs. {iou_weight:.2f}/{prox_weight:.2f}: {e}")
    
    # Save sensitivity analysis results to JSON file
    try:
        with open(args.output, 'w') as f:
            json.dump(sensitivity_results, f, indent=2)
        print(f"\nSaved evaluation results to {args.output}")
    except Exception as e:
        print(f"Error saving results to JSON: {e}")

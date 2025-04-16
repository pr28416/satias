import argparse
import sys
import os
from typing import List, Tuple, Optional, DefaultDict, Any

# Adjust path to import from parent directory and search_engine package
# This allows running the script from the root directory
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
# Temporarily add parent to path if script is run directly
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import config  # Should now be importable from parent
from search_engine import indexer, query_parser, searcher, utils
from search_engine.indexer import InvertedIndexType


def main() -> None:
    parser = argparse.ArgumentParser(description="Spatially-Aware Textual Image Search")
    parser.add_argument(
        "query_text", type=str, help="The text to search for within images."
    )
    parser.add_argument(
        "-r",
        "--region",
        help="Optional spatial region query string (e.g., 'top: 0-20, left: 50-100')",
        default=None,
    )
    parser.add_argument(
        "-i",
        "--index",
        help=f"Path to the index file (default: {indexer.DEFAULT_INDEX_PATH})",
        default=indexer.DEFAULT_INDEX_PATH,
    )
    parser.add_argument(
        "-n",
        "--num_results",
        type=int,
        help="Number of results to display (default: 10)",
        default=10,
    )
    parser.add_argument(
        "--build",
        action="store_true",
        help="Build the index from metadata before searching (use if index file is missing or outdated).",
    )

    args = parser.parse_args()

    inverted_index: Optional[InvertedIndexType] = None

    # --- Build Index if Requested --- (or if index file doesn't exist)
    should_build: bool = args.build or not os.path.exists(args.index)
    if should_build:
        print("--- Building Index --- ")
        # Use default metadata path from config for building
        inverted_index = indexer.build_index(config.IMAGE_METADATA_FILE)
        if inverted_index is not None:
            indexer.save_index(inverted_index, args.index)
        else:
            print("Failed to build index. Exiting.")
            sys.exit(1)
        print("---------------------")

    # --- Load Index --- (if not just built)
    if inverted_index is None:
        inverted_index = indexer.load_index(args.index)
        if inverted_index is None:
            print("Failed to load index. Use --build to create it. Exiting.")
            sys.exit(1)

    # --- Parse Query ---
    print("--- Parsing Query --- ")
    query_ngrams: List[str]
    query_bbox_norm: List[float]
    query_ngrams, query_bbox_norm = query_parser.parse_query(
        args.query_text, args.region
    )
    print(f"Query Text: '{args.query_text}'")
    print(
        f"Target Region: {args.region if args.region else 'Full Image'} -> BBox: {query_bbox_norm}"
    )
    print(f"Query N-grams: {query_ngrams}")
    print("---------------------")

    if not query_ngrams:
        print("No valid n-grams generated from query text. Exiting.")
        sys.exit(0)

    # --- Search Images ---
    print("--- Searching --- ")
    image_scores: DefaultDict[str, float] = searcher.search_images(
        query_ngrams, query_bbox_norm, inverted_index
    )
    print(f"Found {len(image_scores)} potentially relevant images.")
    print("---------------------")

    # --- Rank Results ---
    print("--- Ranking Results --- ")
    ranked_results: List[Tuple[str, float]] = searcher.rank_results(image_scores)
    print("---------------------")

    # --- Display Results ---
    print(f"\n--- Top {args.num_results} Results --- ")
    if not ranked_results:
        print("No matching images found.")
    else:
        for i, (image_id, score) in enumerate(ranked_results[: args.num_results]):
            # Construct the expected image path based on config and image_id
            image_path: str = os.path.join(config.IMAGE_DIR, f"{image_id}.png")
            print(
                f"{i+1}. Image ID: {image_id} (Score: {score:.4f}) - Path: {image_path}"
            )
    print("---------------------")


if __name__ == "__main__":
    main()

import os
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from typing import List, Tuple, Dict, Any

# --- Adjust path to import from parent directory --- START
# This allows running the script from the root directory or its own directory
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
# --- Adjust path --- END

import config
import data_generator
from tqdm import tqdm


# --- Worker Function for Parallel Processing ---
def process_single_image(
    image_id: str, sentence_pool: List[str]
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Generates data for a single image. Intended to be run in a separate process."""
    # 1. Generate Image and Word Metadata
    image_meta = data_generator.generate_image_and_metadata(image_id, sentence_pool)

    # 2. Calculate N-grams
    if image_meta["word_data"]:
        ngrams_data = data_generator.calculate_ngrams(image_meta["word_data"])
    else:
        ngrams_data = []
    # Note: image_meta now contains the raw word_data which might include objects
    # We need the cleaned version for JSON serialization later.

    # Store image metadata (without word objects to keep JSON clean)
    cleaned_image_meta = {
        "image_id": image_meta["image_id"],
        "width": image_meta["width"],
        "height": image_meta["height"],
        "image_path": image_meta["image_path"],
        "word_data": [
            {"text": wd["text"], "bbox_pixels": wd["bbox_pixels"]}
            for wd in image_meta["word_data"]
        ],
        "ngrams": [
            {"text": ng["text"], "bbox_pixels": ng["bbox_pixels"]} for ng in ngrams_data
        ],
    }

    # 3. Generate Queries
    image_queries = data_generator.generate_queries_for_image(
        image_id, config.W, config.H, ngrams_data  # Pass ngrams with objects
    )

    return cleaned_image_meta, image_queries


def main():
    print("Starting synthetic data generation...")

    # --- Setup Output Directories ---
    print(
        f"Ensuring output directories exist: {config.IMAGE_DIR}, {config.METADATA_DIR}"
    )
    os.makedirs(config.IMAGE_DIR, exist_ok=True)
    os.makedirs(config.METADATA_DIR, exist_ok=True)

    # --- Generate Sentence Pool ---
    print(f"Generating sentence pool ({config.NUM_SENTENCES_IN_POOL} sentences)...")
    sentence_pool = data_generator.generate_sentence_pool()
    print("Sentence pool generated.")

    all_image_metadata: List[Dict[str, Any]] = []
    all_queries: List[Dict[str, Any]] = []

    # --- Main Generation Loop (Parallelized) ---
    print(f"Generating {config.NUM_IMAGES} images and associated data (in parallel)...")
    image_ids = [f"synth_{i:03d}" for i in range(config.NUM_IMAGES)]

    # Prepare partial function with fixed sentence_pool argument
    worker_func = partial(process_single_image, sentence_pool=sentence_pool)

    # Use ProcessPoolExecutor to run tasks in parallel
    with ProcessPoolExecutor() as executor:
        # Use executor.map and wrap with tqdm for progress
        # The map function returns results in the order tasks were submitted
        results_iterator = executor.map(worker_func, image_ids)

        # Process results as they complete
        for result in tqdm(
            results_iterator,
            total=len(image_ids),
            desc="Processing Images",
            unit="image",
        ):
            cleaned_meta, queries = result
            all_image_metadata.append(cleaned_meta)
            all_queries.extend(queries)

    # --- Save Metadata and Queries ---
    print(f"Saving metadata to {config.IMAGE_METADATA_FILE}...")
    with open(config.IMAGE_METADATA_FILE, "w") as f:
        json.dump(all_image_metadata, f, indent=2)

    print(f"Saving queries to {config.QUERIES_FILE}...")
    with open(config.QUERIES_FILE, "w") as f:
        json.dump(all_queries, f, indent=2)

    print("Synthetic data generation complete.")
    print(f"Generated {len(all_image_metadata)} images and {len(all_queries)} queries.")
    print(f"Images saved in: {config.IMAGE_DIR}")
    print(f"Metadata saved in: {config.METADATA_DIR}")


if __name__ == "__main__":
    main()

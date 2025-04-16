import os
import json
import sys  # Added import

# --- Adjust path to import from parent directory --- START
# This allows running the script from the root directory or its own directory
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
# --- Adjust path --- END

import config
import data_generator
from tqdm import tqdm  # Optional progress bar


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

    all_image_metadata = []
    all_queries = []

    # --- Main Generation Loop ---
    print(f"Generating {config.NUM_IMAGES} images and associated data...")
    # Use tqdm for progress bar if available, otherwise just loop
    image_ids = [f"synth_{i:03d}" for i in range(config.NUM_IMAGES)]
    iterator = tqdm(image_ids) if "tqdm" in globals() else image_ids

    for image_id in iterator:
        # 1. Generate Image and Word Metadata
        image_meta = data_generator.generate_image_and_metadata(image_id, sentence_pool)

        # 2. Calculate N-grams
        if image_meta["word_data"]:
            ngrams_data = data_generator.calculate_ngrams(image_meta["word_data"])
        else:
            ngrams_data = []
        image_meta["ngrams"] = ngrams_data  # Add ngrams to the image metadata

        # Store image metadata (without word objects to keep JSON clean)
        # The essential info is bbox and text
        clean_image_meta = {
            "image_id": image_meta["image_id"],
            "width": image_meta["width"],
            "height": image_meta["height"],
            "image_path": image_meta["image_path"],
            "word_data": [
                {"text": wd["text"], "bbox_pixels": wd["bbox_pixels"]}
                for wd in image_meta["word_data"]
            ],
            "ngrams": [
                {"text": ng["text"], "bbox_pixels": ng["bbox_pixels"]}
                for ng in ngrams_data
            ],
        }
        all_image_metadata.append(clean_image_meta)

        # 3. Generate Queries
        image_queries = data_generator.generate_queries_for_image(
            image_id, config.W, config.H, ngrams_data
        )
        all_queries.extend(image_queries)

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

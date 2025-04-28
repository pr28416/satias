import json
import pickle
import os
import sys
from collections import defaultdict
from typing import List, Tuple, Dict, Optional, DefaultDict, Any, Union

# --- Adjust path to import from parent directory --- START
# This allows running the script from the root directory or its own directory
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
# --- Adjust path --- END

import config
from search_engine import utils  # Changed to absolute import

# Type alias for the complex index structure
InvertedIndexType = DefaultDict[str, List[Tuple[str, List[float]]]]

# Default path for the index file within the output directory
DEFAULT_INDEX_PATH: str = os.path.join(config.OUTPUT_DIR, "index.pkl")


def build_index(
    metadata_path: str = config.IMAGE_METADATA_FILE,
) -> Optional[InvertedIndexType]:
    """Builds the inverted index from image metadata.

    Args:
        metadata_path (str): Path to the image_metadata.json file.

    Returns:
        defaultdict(list): The populated inverted index.
        Returns None if metadata file not found or is invalid.
    """
    print(f"Building index from: {metadata_path}")
    try:
        with open(metadata_path, "r") as f:
            all_image_metadata = json.load(f)
    except FileNotFoundError:
        print(f"Error: Metadata file not found at {metadata_path}")
        return None
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {metadata_path}")
        return None

    inverted_index: InvertedIndexType = defaultdict(list)
    images_processed: int = 0
    ngrams_indexed: int = 0

    for image_meta in all_image_metadata:
        image_id = image_meta.get("image_id")
        width = image_meta.get("width")
        height = image_meta.get("height")
        ngrams = image_meta.get("ngrams", [])

        if not all([image_id, width, height]):
            print(
                f"Warning: Skipping image due to missing data: {image_meta.get('image_path', 'Unknown Path')}"
            )
            continue

        for ngram_data in ngrams:
            ngram_text: Optional[str] = ngram_data.get("text")
            bbox_pixels: Optional[List[Union[int, float]]] = ngram_data.get(
                "bbox_pixels"
            )

            if not ngram_text or bbox_pixels is None:
                # print(f"Warning: Skipping ngram due to missing data in image {image_id}: {ngram_data}")
                continue

            # Normalize the bounding box
            normalized_bbox = utils.normalize_bbox(bbox_pixels, width, height)

            # Append (image_id, normalized_bbox) to the index
            inverted_index[ngram_text].append((image_id, normalized_bbox))
            ngrams_indexed += 1

        images_processed += 1

    print(
        f"Index built successfully. Processed {images_processed} images, indexed {ngrams_indexed} n-gram occurrences."
    )
    return inverted_index


def save_index(
    index: Optional[InvertedIndexType], output_path: str = DEFAULT_INDEX_PATH
) -> None:
    """Saves the inverted index to a file using pickle.

    Args:
        index (dict): The inverted index to save.
        output_path (str): The path to save the index file to.
    """
    if index is None:
        print("Error: Cannot save a None index.")
        return

    print(f"Saving index to: {output_path}...")
    try:
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "wb") as f:
            pickle.dump(index, f)
        print("Index saved successfully.")
    except Exception as e:
        print(f"Error saving index to {output_path}: {e}")


def load_index(input_path: str = DEFAULT_INDEX_PATH) -> Optional[InvertedIndexType]:
    """Loads the inverted index from a pickle file.

    Args:
        input_path (str): The path to the index file.

    Returns:
        defaultdict(list): The loaded inverted index.
        Returns None if the file is not found or cannot be loaded.
    """
    print(f"Loading index from: {input_path}...")
    if not os.path.exists(input_path):
        print(f"Error: Index file not found at {input_path}")
        print("Hint: You may need to build the index first.")
        return None

    try:
        with open(input_path, "rb") as f:
            index = pickle.load(f)
        # Ensure it's a defaultdict for consistent behavior
        loaded_index: InvertedIndexType
        if not isinstance(index, defaultdict):
            # Convert loaded dict back to defaultdict
            loaded_index = defaultdict(list, index)
        else:
            loaded_index = index
        print("Index loaded successfully.")
        return loaded_index
    except Exception as e:
        print(f"Error loading index from {input_path}: {e}")
        return None


# --- Optional: Script to build and save the index directly ---
if __name__ == "__main__":
    print("Running indexer directly to build and save the index.")
    # Build the index using default metadata path from config
    built_index: Optional[InvertedIndexType] = build_index()
    # Save the index to the default path
    if built_index is not None:
        save_index(built_index)

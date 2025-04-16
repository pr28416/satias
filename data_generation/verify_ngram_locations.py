import json
import config
from collections import defaultdict
import math

# --- Configuration ---
METADATA_FILE = config.IMAGE_METADATA_FILE
NUM_EXAMPLES_TO_SHOW = 5  # How many n-grams to show as examples
MIN_DISTANCE_THRESHOLD = 50  # Minimum pixel distance between centers for locations to be considered 'different'


# --- Helper Function: Calculate Bbox Center ---
def get_bbox_center(bbox_pixels):
    """Calculates the (x, y) center of a [top, left, bottom, right] bbox."""
    t, l, b, r = bbox_pixels
    center_y = t + (b - t) / 2
    center_x = l + (r - l) / 2
    return center_x, center_y


# --- Helper Function: Calculate Distance ---
def calculate_distance(center1, center2):
    """Calculates Euclidean distance between two (x, y) points."""
    return math.sqrt((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2)


# --- Main Verification Logic ---
def verify_locations():
    print(f"Verifying n-gram locations in: {METADATA_FILE}")

    try:
        with open(METADATA_FILE, "r") as f:
            all_image_metadata = json.load(f)
    except FileNotFoundError:
        print(f"Error: Metadata file not found at {METADATA_FILE}")
        print("Please run generate_synthetic_data.py first.")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {METADATA_FILE}")
        return

    ngram_locations = defaultdict(list)

    # 1. Build index of n-gram locations
    print("Building n-gram location index...")
    for image_meta in all_image_metadata:
        image_id = image_meta["image_id"]
        for ngram_data in image_meta.get("ngrams", []):
            ngram_text = ngram_data["text"]
            bbox_pixels = ngram_data["bbox_pixels"]
            ngram_locations[ngram_text].append(
                {"image_id": image_id, "bbox_pixels": bbox_pixels}
            )

    print(
        f"Found {len(ngram_locations)} unique n-grams across {len(all_image_metadata)} images."
    )

    # 2. Filter for n-grams appearing in multiple, distinct locations
    print(
        f"Filtering for n-grams appearing in multiple distinct locations (distance > {MIN_DISTANCE_THRESHOLD}px)..."
    )
    multi_location_ngrams = {}
    for text, locations in ngram_locations.items():
        if len(locations) < 2:
            continue  # Need at least two occurrences

        distinct_locations = []
        # Check pairs of locations for distance
        has_distinct_pair = False
        if len(locations) > 1:
            # Simple check: just see if *any* pair is far apart enough
            # Could be more robust, but sufficient for demonstration
            first_center = get_bbox_center(locations[0]["bbox_pixels"])
            for i in range(1, len(locations)):
                other_center = get_bbox_center(locations[i]["bbox_pixels"])
                distance = calculate_distance(first_center, other_center)
                # Also consider if they are in different images as inherently distinct
                if (
                    distance > MIN_DISTANCE_THRESHOLD
                    or locations[i]["image_id"] != locations[0]["image_id"]
                ):
                    has_distinct_pair = True
                    break

        if has_distinct_pair:
            # For simplicity, just store all locations if at least one pair is distinct enough
            multi_location_ngrams[text] = locations

    print(
        f"Found {len(multi_location_ngrams)} n-grams appearing in multiple distinct locations."
    )

    # 3. Print examples
    print("\n--- Examples of N-grams in Different Locations ---")
    count = 0

    # Sort n-grams by length (number of words) descending, then alphabetically
    sorted_ngrams = sorted(
        multi_location_ngrams.items(),
        key=lambda item: (len(item[0].split()), item[0]),
        reverse=True,
    )

    for text, locations in sorted_ngrams:
        if count >= NUM_EXAMPLES_TO_SHOW:
            break
        print(f'\nN-gram: "{text}"')
        for loc in locations:
            print(f"  - Image: {loc['image_id']}, BBox (pixels): {loc['bbox_pixels']}")
        count += 1

    if count == 0:
        print("\nNo clear examples found with the current settings/data.")
        print("Try generating more images or adjusting parameters.")


if __name__ == "__main__":
    verify_locations()

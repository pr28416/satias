import re
import math
import config
from typing import List, Tuple, Optional, Union, Dict


def normalize_bbox(
    bbox_pixels: List[Union[int, float]],
    width: Union[int, float],
    height: Union[int, float],
) -> List[float]:
    """Converts pixel bbox [t, l, b, r] to normalized [0-100]."""
    if height == 0 or width == 0:
        return [0.0, 0.0, 0.0, 0.0]  # Avoid division by zero
    t, l, b, r = bbox_pixels
    return [
        (t / height) * 100.0,
        (l / width) * 100.0,
        (b / height) * 100.0,
        (r / width) * 100.0,
    ]


def calculate_iou(boxA: List[float], boxB: List[float]) -> float:
    """Calculates Intersection over Union for two bounding boxes.
    Boxes are expected in [top, left, bottom, right] format (normalized 0-100)."""
    # Determine the coordinates of the intersection rectangle
    top_int: float = max(boxA[0], boxB[0])
    left_int: float = max(boxA[1], boxB[1])
    bottom_int: float = min(boxA[2], boxB[2])
    right_int: float = min(boxA[3], boxB[3])

    # Compute the area of intersection rectangle
    # Ensure the intersection is valid (positive width and height)
    inter_width: float = max(0, right_int - left_int)
    inter_height: float = max(0, bottom_int - top_int)
    interArea: float = inter_width * inter_height
    if interArea == 0:
        return 0.0

    # Compute the area of both bounding boxes
    boxAArea: float = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea: float = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    if boxAArea <= 0 or boxBArea <= 0:
        return 0.0  # Avoid division by zero if boxes have zero area

    # Compute the area of the union
    unionArea: float = boxAArea + boxBArea - interArea

    if unionArea == 0:
        # This case can happen if both boxes are identical and have zero area
        return 0.0

    # Compute the Intersection over Union
    iou: float = interArea / unionArea
    return iou


def generate_ngrams_from_text(
    text: str, n_min: int = config.NGRAM_MIN, n_max: int = config.NGRAM_MAX
) -> List[str]:
    """Generates n-grams (as strings) from a given text string."""
    words: List[str] = text.split()
    ngrams: List[str] = []
    num_words: int = len(words)
    if num_words == 0:
        return ngrams

    for n in range(n_min, n_max + 1):
        for i in range(num_words - n + 1):
            ngram: str = " ".join(words[i : i + n])
            ngrams.append(ngram)
    return ngrams


def parse_region_string(region_str: Optional[str]) -> List[float]:
    """Parses a region string like 'top: 10-30, left: 60-80'
    into a normalized bbox [top, left, bottom, right].
    Returns default [0, 0, 100, 100] if string is None or invalid.
    """
    default_bbox: List[float] = [0.0, 0.0, 100.0, 100.0]
    if not region_str:
        return default_bbox

    # Regex to capture dimension, first value (int or float), and optional second value (int or float)
    pattern = re.compile(
        r"(top|left|bottom|right)\s*:\s*(\d+(?:\.\d+)?)\s*(?:-\s*(\d+(?:\.\d+)?))?"
    )
    matches = pattern.findall(region_str.lower())

    coords: Dict[str, float] = {
        "top": 0.0,
        "left": 0.0,
        "bottom": 100.0,
        "right": 100.0,
    }
    found_any: bool = False
    print(f"[DEBUG UTILS] Initial coords: {coords}")  # DEBUG 1

    for match in matches:
        dim: str
        val1_str: str
        val2_str: Optional[str]
        dim, val1_str, val2_str = match
        try:
            val1: float = float(val1_str)
            if not (0 <= val1 <= 100):
                print(
                    f"Warning: Value {val1} for {dim} out of range [0, 100]. Ignoring."
                )
                continue

            if val2_str:
                # --- Handle Range Input ---
                val2: float = float(val2_str)
                if not (0 <= val2 <= 100):
                    print(
                        f"Warning: Value {val2} for {dim} range end out of range [0, 100]. Ignoring."
                    )
                    continue
                # Ensure min-max order
                min_val: float = min(val1, val2)
                max_val: float = max(val1, val2)

                # Explicitly update coords dictionary based on dimension
                if dim == "top":
                    coords["top"] = min_val
                    coords["bottom"] = max_val  # Set bottom boundary from range end
                elif dim == "left":
                    coords["left"] = min_val
                    coords["right"] = max_val  # Set right boundary from range end
                elif dim == "bottom":
                    # Allow setting bottom explicitly with range (though less common)
                    # This will override any value set by a 'top' range
                    coords["bottom"] = max_val
                    if coords["top"] < min_val:  # Basic check if top wasn't set lower
                        coords["top"] = min_val  # Adjust top if needed?
                elif dim == "right":
                    # Allow setting right explicitly with range
                    coords["right"] = max_val
                    if coords["left"] < min_val:
                        coords["left"] = min_val  # Adjust left if needed?
            else:  # Single value (e.g., top: 10, left: 20, bottom: 90, right: 50)
                # --- Handle Single Value Input ---
                coords[dim] = val1  # Set the specific boundary provided
            print(
                f"[DEBUG UTILS] After processing match {match}, Coords: {coords}"
            )  # DEBUG 2

            found_any = True
        except ValueError:
            print(f"Warning: Could not parse value(s) in '{match}'. Ignoring.")

    if not found_any:
        print(
            f"Warning: Could not parse region string '{region_str}'. Using default full region."
        )
        return default_bbox

    # Final check for valid bbox (top < bottom, left < right)
    if coords["bottom"] <= coords["top"] or coords["right"] <= coords["left"]:
        print(
            f"Warning: Invalid region parsed from '{region_str}' resulting in {coords}. Using default."
        )
        return default_bbox

    print(f"[DEBUG UTILS] Final coords before return: {coords}")  # DEBUG 3
    return [coords["top"], coords["left"], coords["bottom"], coords["right"]]


def get_bbox_center(bbox_norm: List[float]) -> Tuple[float, float]:
    """Calculates the (x, y) center of a normalized [t, l, b, r] bbox."""
    t, l, b, r = bbox_norm
    center_y: float = t + (b - t) / 2.0
    center_x: float = l + (r - l) / 2.0
    return center_x, center_y


def calculate_proximity_score(
    center1: Tuple[float, float],
    center2: Tuple[float, float],
    scale: float = config.PROXIMITY_DISTANCE_SCALE,
) -> float:
    """Calculates a proximity score based on Euclidean distance.
    Score is 1 for 0 distance, decaying exponentially towards 0.
    Uses scale factor k from config for exp(-k * distance).
    """
    distance: float = math.sqrt(
        (center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2
    )
    return math.exp(-scale * distance)

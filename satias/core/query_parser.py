import config
from . import utils  # Relative import
from typing import List, Tuple, Optional


def parse_query(
    query_text: str, query_region_str: Optional[str] = None
) -> Tuple[List[str], List[float]]:
    """Parses the raw query text and region string.

    Args:
        query_text (str): The text the user wants to search for.
        query_region_str (str, optional): A string describing the target region
            (e.g., "top: 10-30, left: 60-80"). Defaults to None (full image).

    Returns:
        tuple: A tuple containing:
            - list[str]: A list of query n-grams derived from query_text.
            - list[float]: The normalized query bounding box [t, l, b, r].
    """
    if not query_text:
        print("Warning: Empty query text received.")
        return [], utils.parse_region_string(
            None
        )  # Return default region for empty text

    # 1. Generate query n-grams
    query_ngrams: List[str] = utils.generate_ngrams_from_text(
        query_text, n_min=config.NGRAM_MIN, n_max=config.NGRAM_MAX
    )

    # 2. Parse the region string (handles None or invalid strings internally)
    query_bbox_norm: List[float] = utils.parse_region_string(query_region_str)

    # print(f"Parsed Query: N-grams={query_ngrams}, Region={query_bbox_norm}")
    return query_ngrams, query_bbox_norm

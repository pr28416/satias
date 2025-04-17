from collections import defaultdict
from . import utils  # Relative import
from typing import List, Tuple, Dict, DefaultDict, Optional, Set  # Added Set import
import config  # Added import

# Import the type alias from indexer
from .indexer import InvertedIndexType


def search_images(
    query_ngrams: List[str],
    query_bbox_norm: List[float],
    inverted_index: Optional[InvertedIndexType],
    search_mode: str = "spatial",
) -> DefaultDict[str, float]:
    """Searches the index for query terms and calculates scores for images based on search mode.

    Args:
        query_ngrams (list[str]): The list of n-grams from the parsed query.
        query_bbox_norm (list[float]): The normalized query bounding box [t, l, b, r].
        inverted_index (defaultdict(list)): The loaded inverted index.

    Returns:
        defaultdict(float): A dictionary mapping image_id to its accumulated score.
                          Returns empty dict if index is None or query_ngrams is empty.
    """
    if not inverted_index:
        return defaultdict(float)

    image_scores: DefaultDict[str, float] = defaultdict(float)
    default_bbox: List[float] = [0.0, 0.0, 100.0, 100.0]
    is_non_spatial_query: bool = query_bbox_norm == default_bbox

    # --- Keyword Only Logic ---
    if search_mode == "keyword_only":
        if not query_ngrams:
            return image_scores  # Need some text input
        # Derive unique words from the first ngram (original query text)
        # Assumes query_ngrams list isn't empty if we reached here
        original_query_text = (
            query_ngrams[0]
            if len(query_ngrams[0].split()) == 1
            else next(
                (ng for ng in query_ngrams if len(ng.split()) > 1), query_ngrams[0]
            )
        )  # Heuristic to get multi-word text if possible
        # TODO: A better way might be to pass the original query text directly into this function
        unique_query_words: Set[str] = set(original_query_text.split())
        if not unique_query_words:
            return image_scores

        # print(f"[DEBUG] Keyword search for: {unique_query_words}")
        for word in unique_query_words:
            if word in inverted_index:
                # Find all unique image IDs where this word appears
                image_ids_with_word: Set[str] = {loc[0] for loc in inverted_index[word]}
                for image_id in image_ids_with_word:
                    image_scores[
                        image_id
                    ] += 1.0  # Increment score for each unique query word found
        return image_scores

    # --- N-gram Based Logic (Spatial or Ngram Text Only) ---
    if not query_ngrams:  # Check for n-grams needed for other modes
        return image_scores

    # print(f"Searching for {len(query_ngrams)} n-grams...")
    # print(f"Query region: {query_bbox_norm}, Non-spatial: {is_non_spatial_query}")

    for ngram in query_ngrams:
        if ngram in inverted_index:
            locations: List[Tuple[str, List[float]]] = inverted_index[ngram]
            ngram_length: int = len(ngram.split())  # Weight by n-gram length

            # print(f"  N-gram '{ngram}' (length {ngram_length}) found in {len(locations)} locations.")

            for image_id, ngram_bbox_norm in locations:
                spatial_score_component: float
                if (
                    search_mode == "ngram_text_only"
                ):  # N-gram baseline ignores spatial info
                    spatial_score_component = 1.0
                elif is_non_spatial_query:
                    # For non-spatial query, spatial component is effectively 1.0
                    spatial_score_component = 1.0
                else:
                    # For spatial query, calculate weighted combination of IoU and Proximity
                    iou: float = utils.calculate_iou(query_bbox_norm, ngram_bbox_norm)

                    query_center = utils.get_bbox_center(query_bbox_norm)
                    ngram_center = utils.get_bbox_center(ngram_bbox_norm)
                    proximity_score = utils.calculate_proximity_score(
                        query_center, ngram_center
                    )

                    spatial_score_component = (config.IOU_WEIGHT * iou) + (
                        config.PROXIMITY_WEIGHT * proximity_score
                    )

                # Final score contribution combines spatial component and ngram length
                score_contribution: float = spatial_score_component * ngram_length

                # Accumulate score for the image
                image_scores[image_id] += score_contribution
                # if score_contribution > 0:
                #     print(f"    -> Image: {image_id}, Ngram Bbox: {ngram_bbox_norm}, IoU: {iou:.3f}, Contribution: {score_contribution:.3f}, New Total: {image_scores[image_id]:.3f}")

    # print(f"Search complete. Found scores for {len(image_scores)} images.")
    return image_scores


def rank_results(image_scores: Dict[str, float]) -> List[Tuple[str, float]]:
    """Ranks the images based on their accumulated scores.

    Args:
        image_scores (dict): Dictionary mapping image_id to score.

    Returns:
        list[tuple[str, float]]: A list of (image_id, score) tuples, sorted by
                                 score in descending order.
    """
    if not image_scores:
        return []

    # Sort by score (the second element of the tuple item) in descending order
    ranked_list: List[Tuple[str, float]] = sorted(
        image_scores.items(), key=lambda item: item[1], reverse=True
    )

    return ranked_list

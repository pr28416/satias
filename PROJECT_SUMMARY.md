# Project Summary: Spatially-Aware Textual Image Search

## 1. Introduction

This document summarizes the implementation of a spatially-aware textual image search engine. The goal was to build and evaluate a system capable of finding images containing specific text within user-defined spatial regions, going beyond simple keyword matching. The project involved generating synthetic test data, implementing the core search algorithm, developing visualization tools, and establishing a rigorous evaluation framework.

## 2. Synthetic Data Generation (`data_generation/`)

To enable controlled testing and evaluation without relying on complex OCR setups initially, a synthetic dataset generation pipeline was created.

**Purpose:** Generate images with known text content and precise ground-truth locations, along with corresponding test queries designed to probe various aspects of the spatial search algorithm.

**Methodology:**

- **Configuration (`config.py`):** Centralized control over parameters like image dimensions (`W`, `H`), number of images (`NUM_IMAGES`), font settings (`FONT_PATH`, `FONT_SIZE`), text density (`MIN/MAX_SENTENCES_PER_IMAGE`), repetition control (`NUM_SENTENCES_IN_POOL`, `TEST_PHRASES`), word distinctiveness (`NOTABLE_WORDS`), n-gram range (`NGRAM_MIN`, `NGRAM_MAX`), query generation (`NUM_QUERIES_PER_IMAGE`, `QUERY_TYPE_DISTRIBUTION`), output paths, etc.
- **Core Logic (`data_generator.py`):**
  - **Sentence Pool:** A pool of unique sentences (`generate_sentence_pool`) is created using `Faker` to ensure controlled repetition of words and phrases across different images.
  - **Image Creation:** For each image (`generate_image_and_metadata`):
    - A blank image is created using `Pillow`.
    - A random subset of sentences is selected from the pool.
    - **Test Phrase Injection:** Specific `TEST_PHRASES` (e.g., "special offer") are probabilistically injected multiple times at random locations within the selected text.
    - **Notable Word Replacement:** Some common words are probabilistically replaced with more distinctive words from the `NOTABLE_WORDS` list to aid later visual inspection/annotation.
    - **Text Rendering:** Words are drawn onto the image sequentially (top-down, left-right) using `Pillow`, handling line wrapping based on margins and word width. The text is allowed to bleed off the bottom edge to ensure full vertical coverage.
    - **Ground Truth Bounding Boxes:** Crucially, the precise pixel bounding box `[top, left, bottom, right]` for _each individual word_ is calculated using `ImageDraw.textbbox` _before_ drawing and stored.
  - **N-gram Calculation:** (`calculate_ngrams`):
    - Iterates through the generated words for an image.
    - Generates all n-grams within the configured range (e.g., 1 to 3 words).
    - Calculates the union bounding box (in pixels) for each n-gram based on the exact pixel bounding boxes of its constituent words.
  - **Query Generation (`generate_queries_for_image`):**
    - For each generated image, a set number (`NUM_QUERIES_PER_IMAGE`) of queries is created.
    - A random n-gram already placed in that image is selected as the textual target (`query_text`). Its location (`target_ngram_bbox_pixels`, `target_ngram_bbox_norm`) serves as the ground truth for this query.
    - A query region type is chosen randomly based on the configured `QUERY_TYPE_DISTRIBUTION` (e.g., No Region, Exact Match, High IoU, Low IoU, Nearby, Distant).
    - A corresponding normalized query region (`query_region_norm`) is generated based on the target n-gram's location and the chosen type (e.g., calculating overlaps, offsets).
  - **Parallelization (`generate_synthetic_data.py`):** The main script uses `concurrent.futures.ProcessPoolExecutor` to parallelize the generation process across multiple CPU cores, significantly speeding up the creation of large datasets. A `tqdm` progress bar monitors the progress.

**Output:**

- Image files (`./synthetic_data/images/synth_XXX.png`)
- Metadata (`./synthetic_data/metadata/image_metadata.json`): A JSON list where each entry contains image details (`image_id`, dimensions, path) and lists of all words and n-grams within it, including their exact pixel bounding boxes.
- Queries (`./synthetic_data/metadata/queries.json`): A JSON list containing all generated queries. Each query includes `query_id`, `image_id` (the ground truth source image), `query_text`, `query_region_norm` (normalized target area), ground truth pixel/normalized bounding boxes for the text, and the `generation_type` indicating how the query region was created relative to the target text.

## 3. Search Algorithm Implementation (`search_engine/`)

The core search engine implements the algorithm outlined in the project description.

**Modules:**

- **`utils.py`:** Contains shared helper functions:
  - `normalize_bbox`: Converts pixel coordinates to normalized [0-100] percentages.
  - `calculate_iou`: Computes Intersection over Union between two normalized bounding boxes.
  - `get_bbox_center`: Calculates the center coordinates of a normalized bounding box.
  - `calculate_proximity_score`: Computes a spatial proximity score (based on `exp(-k * distance)`) between two center points.
  - `generate_ngrams_from_text`: Extracts n-grams from query text.
  - `parse_region_string`: Parses user-friendly region strings (e.g., `"top: 10-30, left: 50"`) into normalized `[t, l, b, r]` bounding boxes, handling various formats and defaults.
- **`indexer.py`:**
  - `build_index`: Reads the `image_metadata.json`, iterates through images and their pre-calculated n-grams, normalizes the n-gram pixel bounding boxes using image dimensions, and builds the core `inverted_index`.
  - **Inverted Index:** A `defaultdict(list)` mapping n-gram strings to a list of tuples `(image_id: str, normalized_bbox: List[float])`. This allows efficient lookup of all locations for a given n-gram.
  - `save_index` / `load_index`: Uses `pickle` to save the built index to disk (`output/index.pkl` by default) and load it efficiently for searching, avoiding rebuilding.
- **`query_parser.py`:**
  - `parse_query`: Takes raw user query text and an optional region string. Uses `utils` to generate query n-grams and parse the region string into a normalized bounding box.
- **`searcher.py`:**
  - `search_images`:
    - Takes parsed query n-grams, the normalized query bounding box, the loaded `inverted_index`, and an optional `is_baseline_search` flag.
    - Iterates through query n-grams, looking them up in the index.
    - For each found occurrence `(image_id, ngram_bbox_norm)`:
      - Calculates the `spatial_score_component`. If `is_baseline_search` is True or the query is non-spatial (full image), this is 1.0. Otherwise, it's a weighted sum: `(config.IOU_WEIGHT * iou) + (config.PROXIMITY_WEIGHT * proximity_score)`, where IoU and proximity score (distance between centers) are calculated using `utils`.
      - Calculates the final `score_contribution = spatial_score_component * ngram_length`.
      - Accumulates scores per `image_id` in a `defaultdict(float)`.
  - `rank_results`: Sorts the accumulated `image_scores` dictionary in descending order.

**Main Interfaces:**

- **`main_search.py`:** Provides a command-line interface (CLI) to run searches. Accepts query text, optional region string (`-r`), index path (`-i`), number of results (`-n`), and a flag to build the index (`--build`).
- **`visualize_search.py`:** Provides a Tkinter-based GUI for interactive searching and result visualization (detailed below).

## 4. Evaluation (`evaluate.py`)

A dedicated script provides quantitative evaluation of the search engine's performance against the synthetic dataset.

**Methodology:**

- **Ground Truth:** Uses the generated `queries.json`. Each query entry implicitly defines the single "correct" or "relevant" result as the `image_id` from which it was generated.
- **Baseline Comparison:** Compares the "Current Config" (using spatial scoring defined by `IOU_WEIGHT` and `PROXIMITY_WEIGHT` in `config.py`) against a "Baseline" configuration (non-spatial search, where `searcher.search_images` is called with `is_baseline_search=True`, effectively ignoring location).
- **Metrics:**
  - **Mean Average Precision (MAP@k):** Calculates Average Precision (AP) for each query. Since only one result is relevant, `AP = 1 / rank` if the target image is found at rank `i` (within top `k`), otherwise `AP = 0`. MAP is the mean of these AP scores across all queries. It measures ranking quality.
  - **Mean Precision@k (P@k):** Calculates the fraction of the top `k` results that are the single relevant target image for each query (`0.0` or `1.0/k`). Mean P@k averages this across all queries. It measures recall within the top `k`.
- **Statistical Significance:**
  - Performs a **Wilcoxon signed-rank test** (using `scipy.stats.wilcoxon`) on the paired differences between the AP scores of the Current Config and the Baseline for each query.
  - Tests the alternative hypothesis that the Current Config's AP scores are significantly _greater_ than the Baseline's.
  - Reports the p-value. A p-value < 0.05 indicates that the observed improvement in MAP for the Current Config is statistically significant (unlikely due to random chance alone on this dataset).
- **Output:** Presents the MAP@k and P@k scores for both configurations side-by-side, reports the p-value from the statistical test, and provides textual interpretation of the results and their significance. Includes `tqdm` for progress monitoring.

## 5. Visualization (`visualize_search.py`)

To aid understanding and debugging of the spatial search behavior, a GUI tool was developed using Tkinter and Pillow.

**Features:**

- **Input:** Allows users to enter query text and specify query regions using start/end percentages for top and left boundaries.
- **Search Execution:** Runs the search using the loaded index and the backend `search_engine` modules.
- **Results Display:** Presents results in a scrollable grid. It displays the Top 15, Middle 15, and Last 15 ranked results (handling cases with fewer total results) with section headers indicating rank ranges.
- **Overlays:** For each displayed result image:
  - Draws a semi-transparent light blue rectangle representing the user's specified **query region**.
  - Finds all occurrences of the query n-grams within that specific image.
  - Draws semi-transparent bounding boxes around **each found n-gram occurrence**.
  - **Colors** the n-gram bounding boxes based on their **IoU** with the query region (Red for 0 overlap, Yellow for ~0.5, Green for 1.0), allowing visual assessment of spatial relevance according to IoU.
- **Information:** Displays the Rank, Image ID, and calculated Score for each result.

## 6. Conclusion

This project successfully implemented a spatially-aware text search engine. Key achievements include: a flexible synthetic data generator enabling controlled experiments; a modular search engine implementation incorporating configurable spatial scoring (weighted IoU and proximity); command-line and GUI interfaces; and a rigorous evaluation framework comparing spatial search against a non-spatial baseline using MAP, P@k, and statistical significance testing. The visualization tool provides valuable insight into the interplay between text matching, spatial regions, and ranking scores.

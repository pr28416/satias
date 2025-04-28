# A Spatially-Aware Search Engine for Textual Content in Images

**Pranav Ramesh, Mohamed Zidan Cassim, Giovanni D'Antonio**

[_View GitHub repository_](https://github.com/pr28416/satias)

## 0. Abstract

_Standard image search engines often treat text within images as secondary metadata or ignore its spatial location. This limits users' ability to find images based on text appearing in specific visual areas. We present a spatially-aware textual image search engine designed to address this limitation. Our approach utilizes an inverted index mapping text n-grams to their normalized bounding box coordinates within images. Queries consist of text and an optional spatial region. Relevance scoring combines spatial factors (Intersection over Union - IoU and proximity) with n-gram length, weighted according to configurable parameters. To facilitate development and evaluation, we developed a pipeline for generating synthetic datasets with controlled text placement and ground truth. We evaluated our system against non-spatial baselines (keyword-only and n-gram-only) using Mean Average Precision (MAP) and Precision@k (P@k) on this synthetic data. Results demonstrate statistically significant improvements in ranking quality for both n-gram usage over keywords (MAP 0.21 vs 0.03) and spatial awareness over n-grams alone (MAP 0.67 vs 0.21), validating the effectiveness of incorporating both n-grams and spatial context. A visualization tool was also developed to aid in understanding search results._

---

## 1. Introduction

### 1.1 Problem Statement

Images frequently contain rich textual information, such as signs, labels, headlines, logos, or embedded text in documents and screenshots. Traditional image search systems primarily focus on visual features or global textual tags, often failing to leverage the specific content and location of text within the image. Users cannot easily query for images containing specific text within a particular visual region (e.g., "find photos with 'SALE' in the top-right corner" or "show screenshots where 'error message' appears near the bottom"). This lack of spatial awareness limits the precision and utility of text-based image retrieval.

### 1.2 Motivation

The ability to search for text within specific spatial regions of images unlocks numerous applications. Examples include:

- **Document Analysis:** Finding specific sections or figures in scanned documents or presentations based on headings or captions in known layout areas.
- **Retail/E-commerce:** Locating product images where a price or discount tag appears in a particular location relative to the product.
- **Scene Understanding:** Identifying street signs, shop names, or specific labels within photographs of complex scenes.
- **UI/UX Research:** Analyzing screenshots to find instances where specific labels or error messages appear in certain interface elements.
- **Accessibility:** Enabling visually impaired users to query the location of text within an image.

To illustrate the practical utility more concretely, consider a case study in **Automated Data Entry from Scanned Documents**. Imagine processing a high volume of scanned invoices or receipts for accounting. While Optical Character Recognition (OCR) can extract all text from an image, accurately identifying the _semantic role_ of specific text snippets (e.g., distinguishing the "Total Amount" from line item prices, or the "Invoice Date" from a "Payment Due Date") is a significant challenge due to the wide variety of document layouts. A traditional keyword search for terms like "Total" or for date patterns might yield multiple ambiguous candidates scattered across the document.

This is where spatially-aware search offers a distinct advantage. By leveraging common layout conventions, the system can target queries to specific regions. For example, a query searching for text matching a monetary amount pattern (e.g., `\\$\\d+\\.\\d{2}`) _specifically within the normalized coordinates corresponding to the bottom-right quadrant_ of the document is highly likely to isolate the final **Total Amount**. Similarly, querying for date-like text primarily within the _top-right quadrant_ can reliably extract the **Invoice Date**. This targeted spatial querying drastically reduces ambiguity compared to context-agnostic text search, significantly simplifying the development of robust automated data entry pipelines without requiring complex template matching or sophisticated layout analysis models for every document variant. It demonstrates how spatial awareness can act as a powerful heuristic for semantic disambiguation in structured documents.

Existing methods often rely on whole-image tags or complex scene understanding models that may not precisely capture localized text queries. A dedicated system focusing on spatial text search promises higher precision and user control for these tasks.

### 1.3 Proposed Solution & Contributions

To address the limitations of traditional methods, we propose and implement **SATIAS (Spatially-Aware Textual Image Search)**, a search engine designed to retrieve images based not only on _what_ text they contain but also _where_ that text is located. The core idea is to move beyond simple keyword matching by creating an index that explicitly links textual content (represented as n-grams, typically sequences of 1 to 3 words) to its precise spatial location within each image. This is achieved by building an inverted index where keys are text n-grams and values are lists of occurrences, each storing the `image_id` and the n-gram's **normalized bounding box** coordinates (percentages of image width/height) to ensure scale and aspect-ratio invariance. User queries can specify both a `query_text`, which is parsed into n-grams, and an optional target **spatial region**, also represented as a normalized bounding box. Candidate images containing matching n-grams are retrieved via the index, and each potential match is evaluated using a novel, configurable scoring mechanism. This scoring combines **spatial relevance**—calculated as a weighted sum of **Intersection over Union (IoU)** for overlap and **centroid proximity** for nearness between the query region and the n-gram box—with **textual relevance**, where matches involving longer n-grams contribute more significantly. The relative importance of IoU versus proximity can be tuned via configuration weights. Finally, scores are aggregated per image, and the results are ranked to provide the user with images where the desired text appears in the specified location.

This paper details the design, implementation, and rigorous evaluation of the SATIAS system. Our primary contributions include: (1) the **novel algorithm design** itself, particularly the use of normalized coordinates for indexing and the hybrid spatial scoring function combining weighted IoU and proximity; (2) a robust and parallelized **synthetic data generation pipeline** (`data_generation/`) that creates large datasets with precise ground-truth bounding boxes and targeted queries, crucial for controlled offline evaluation independent of OCR errors; (3) a modular Python **system implementation** (`search_engine/`) encompassing indexing, flexible query parsing, spatial calculations, and search logic; (4) a **rigorous quantitative evaluation** framework (`evaluate.py`) using MAP@k and P@k metrics to compare SATIAS against keyword-only and n-gram-only baselines on the synthetic data, including statistical significance testing (Wilcoxon signed-rank test); and (5) an **interactive visualization tool** (`visualize_search.py`) with a GUI that allows users to execute searches and inspect results with overlays showing query regions and color-coded n-gram bounding boxes based on IoU, aiding qualitative analysis and debugging.

## 2. Prior Work

The challenge of searching for textual content within images, potentially constrained by location, has been explored from various perspectives. Our work draws upon foundational concepts while offering a specific, geometrically focused solution.

### 2.1 Foundational and Explicit Spatial Methods

**Foundational Text-in-Image Search:** Early research, such as that by Manmatha et al. (UMass CIIR, 2000) [4], focused on the fundamental problems of detecting, extracting (via OCR), and indexing text found within images for keyword-based retrieval. These systems laid the groundwork but typically treated text as document-level metadata or lacked mechanisms for precise spatial querying. Our system builds on this by explicitly indexing the _location_ of extracted text (n-grams) and enabling queries against these locations.

**Spatial-Semantic Approaches:** More recent work has integrated spatial reasoning with semantic understanding. Mai et al. (CVPR 2017) [2] proposed a spatial-semantic image search framework where users define semantic layouts on a canvas, and a CNN synthesizes corresponding visual features for retrieval. This differs from our approach, which focuses narrowly on matching the precise geometric location (bounding box) of specific text n-grams provided in the query, rather than interpreting broader semantic layouts.

### 2.2.1 The Rise of MLLMs in Spatial Grounding (Recent Trends)

Recent years (particularly 2024-2025) have seen a significant shift towards utilizing Multimodal Large Language Models (MLLMs) for tasks involving spatial grounding. These models, such as KOSMOS-2 [9] and Groma [10], aim to integrate visual perception, language comprehension, and spatial reasoning within unified architectures, often trained on web-scale datasets [9].

**Approach:** Instead of explicit geometric indexing and scoring like our system, MLLMs typically handle spatial information implicitly through learned mechanisms:

- **Location Tokens:** Continuous bounding box coordinates are often discretized into special tokens integrated into the language model's vocabulary [9].
- **Attention Mechanisms:** Cross-modal attention allows the model to learn correlations between text and image regions.
- **Learned Embeddings:** Joint embedding spaces align visual regions and textual descriptions, implicitly encoding spatial relationships.

**Comparison to Our System:**

- **Semantic Understanding:** MLLMs possess strong semantic capabilities derived from their underlying LLMs, enabling them to handle synonyms, paraphrasing, and complex natural language queries describing spatial relations (e.g., "the book to the left of the lamp") [9, 10]. Our system relies on exact n-gram matching and lacks this semantic flexibility.
- **Interpretability vs. Complexity:** While powerful, the reasoning process of MLLMs is often opaque ("black box"). Our system, using explicit IoU and proximity calculations, offers greater interpretability and direct control via tunable weights.
- **Data & Training Needs:** Training state-of-the-art MLLMs for grounding requires massive datasets (like the GRIT dataset used for KOSMOS-2 [9]) and significant computational resources for pre-training and fine-tuning [10]. Our indexing approach, while requiring OCR, may have different data dependencies, primarily needing the images themselves rather than extensive grounded text-region pairs for initial training.

This trend towards MLLMs highlights a different paradigm for spatial understanding, trading explicit geometric control for learned semantic richness and query flexibility, albeit with associated challenges in interpretability and data requirements.

### 2.3 Enabling Technologies

**Text Spotting:** Accurate detection and bounding box generation are critical prerequisites for any text-in-image search system. The field has advanced to handle arbitrary text shapes using segmentation, contour embedding, Bezier curves (ABCNet), Mask R-CNN, or sequential deformation. However, bounding box inaccuracy remains a challenge for real-world geometric scoring.

**Indexing:** Scalability requires efficient indexing structures. While our approach uses an in-memory inverted index, spatial databases traditionally use R-Trees/Quadtrees, often combined with inverted indexes in hybrid structures. Recent Learned Sparse Retrieval (LSR) methods (e.g., STAIR [7], arXiv:2303.07740 [6], arXiv:2402.17535 [8]) map dense embeddings to sparse lexical vectors compatible with inverted indexes, offering a promising direction for scalable multimodal retrieval.

### 2.4 Contributions

Our work occupies a niche focused on precise, spatially constrained retrieval of specific text n-grams. Compared to the prior work, our contributions are:

1. The use of an efficient inverted index mapping n-grams directly to normalized bounding boxes
2. A tunable spatial scoring function explicitly combining geometric overlap (IoU) and centroid proximity, offering direct control over spatial relevance criteria
3. A dedicated synthetic data generation pipeline and evaluation methodology designed to rigorously assess the performance of spatial text localization, isolating it from OCR errors and providing targeted spatial query scenarios

Our approach provides a simple, interpretable method for precise geometric localization of exact text n-grams within rectangular regions. Its strengths are direct geometric control and the synthetic data pipeline for evaluation. Key limitations include dependence on OCR accuracy, lack of semantic understanding (unlike VSE/attention models), limited query expressiveness (compared to canvas/trace/relational queries), and scalability issues addressed by spatial/hybrid/LSR indexing. It represents a valuable baseline but stands apart from dominant deep learning trends emphasizing semantics and learned alignments.

### 2.5 Comparison of Approaches

Table 1 provides a comprehensive comparison of various spatially-aware image-text retrieval approaches, highlighting the distinctive positioning of our system among existing methods.

**Table 1: Comparison of Spatially-Aware Image-Text Retrieval Approaches**

| Approach Category          | Key Papers/Examples                       | Query Input                                | Spatial Representation/Handling                                      | Indexing Method                           | Scoring/Matching Mechanism                                                           | Key Strengths                                                                      | Key Weaknesses                                                                         |
| -------------------------- | ----------------------------------------- | ------------------------------------------ | -------------------------------------------------------------------- | ----------------------------------------- | ------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------- |
| Foundational Text-in-Image | Manmatha et al. (2000) [4]                | Keywords / Word Image                      | Typically Ignored / Implicit in Word Spotting                        | Inverted Index / Visual Features          | Keyword Match / Visual Similarity (Word Spotting)                                    | Established text extraction/search                                                 | No explicit spatial querying; OCR dependent                                            |
| Explicit Spatial-Semantic  | Mai et al. (2017) [2]                     | Concept text-boxes on 2D canvas            | User-defined spatial layout on canvas                                | Visual Feature Index (e.g., k-NN)         | Similarity of synthesized visual features representing query canvas layout           | Flexible semantic layout specification                                             | Less precise for exact text; relies on feature synthesis                               |
| Deep Alignment (Fragments) | Karpathy et al. (2014) [3]                | Sentence                                   | Implicit alignment of object regions & sentence fragments            | Learned Embeddings (Regions/Phrases)      | Learned similarity/alignment score in joint embedding space (max-margin objective)   | Fine-grained semantic alignment; bidirectional retrieval                           | No explicit geometric query; aligns objects, not specific text instances geometrically |
| General VSE / Attention    | VSE++ [5], SCAN, CAAN                     | Text Query                                 | Implicit via learned embeddings / Attention weights on regions/words | Learned Embeddings (Global/Local)         | Learned similarity (e.g., cosine) in embedding space; Attention-weighted aggregation | Strong semantic matching; handles paraphrasing; fine-grained alignment (attention) | Typically no explicit geometric query input; complexity; interpretability issues       |
| Multimodal Spatial Query   | Changpinyo et al. (2021) [1]              | Spoken Language + Mouse Traces on Canvas   | Interpreted from mouse traces via learned model (Transformer)        | Learned Embeddings (Image/Text+Trace)     | Learned similarity score (e.g., dot product) between query & image embeddings        | Natural spatial input; handles general layout queries                              | Complex model; focuses on general image retrieval, not specific text location          |
| Relational / Grounding     | Scene Graph methods                       | Text describing spatial relations          | Scene Graphs / Relational Features / Spatial Loss functions          | Scene Graph Index / Learned Embeddings    | Graph Matching / Learned similarity considering relations                            | Handles complex spatial relationship queries                                       | High complexity; often requires structured representations or specific losses          |
| Learned Sparse Retrieval   | STAIR [7], Cao et al. [6], Bai et al. [8] | Text Query                                 | Typically Not Explicit                                               | Sparse Vector Inverted Indexes            | Efficient index-based retrieval with sparse vectors                                  | Scalability; Compatibility with traditional IR systems                             | Often lacks explicit spatial understanding                                             |
| Our Approach               | (This work)                               | Text N-grams + Optional Rectangular Region | Normalized Bounding Boxes                                            | In-memory Inverted Index (N-gram -> List) | Explicit Geometric Score (Weighted IoU + Proximity) + N-gram Length Weighting        | Simple; Interpretable; Direct geometric control; Precise text localization         | OCR dependent; No semantics; Limited query; Scalability; Simple geometry               |
| MLLM Grounding             | KOSMOS-2 [9], Groma [10]                  | Natural Language (complex relations)       | Learned (Location Tokens, Attention, Embeddings)                     | End-to-End MLLM                           | Implicitly Learned via Attention/Embeddings                                          | Semantic Flexibility; Complex Queries; Generalization; SOTA Grounding Performance  | Interpretability (Black Box); High Data/Compute Needs; Potential Hallucination         |

## 3. Methodology

Our system comprises two main phases: offline indexing and online query processing/search.

### 3.1 Core Algorithm Overview

The system first preprocesses a collection of images (or uses pre-computed metadata in our synthetic case) to build an inverted index. This index maps text n-grams to a list of all locations (image ID and normalized bounding box) where they appear. During online search, a user query (text + optional region) is processed. N-grams are extracted from the query text. The inverted index is used to retrieve candidate image locations matching these n-grams. Each match is scored based on n-gram length and spatial relevance relative to the query region. Scores are aggregated per image, and results are ranked.

### 3.2 Indexing Phase (`indexer.py`)

**Objective:** Create an efficient lookup structure for n-gram occurrences and their spatial locations.

**Process:**

1.  **Input:** Image metadata, typically derived from OCR output (or our `image_metadata.json` containing pre-calculated words, n-grams, and their pixel bounding boxes).
2.  **Normalization:** For each n-gram in each image, its pixel bounding box `[t, l, b, r]` is converted to normalized percentage coordinates `[norm_t, norm_l, norm_b, norm_r]` using the image's width and height (`utils.normalize_bbox`). This ensures that spatial comparisons are independent of image resolution and aspect ratio. Normalization is crucial for consistent spatial querying across diverse image sources.
3.  **Inverted Index Construction (`build_index`):** A Python `defaultdict(list)` is used. The keys are the n-gram text strings. The values are lists containing tuples `(image_id: str, normalized_bbox: List[float])`. As n-grams are processed, their `(image_id, normalized_bbox)` tuple is appended to the list associated with the n-gram text. `defaultdict` simplifies the append operation.
4.  **Persistence (`save_index`, `load_index`):** The built index is saved to disk using Python's `pickle` module for efficient serialization and loading, avoiding the need to rebuild the index for every search session. The default location is `output/index.pkl`.

### 3.3 Query Processing (`query_parser.py`)

**Objective:** Convert user input into a format suitable for searching the index.

**Process (`parse_query`):**

1.  **Text Parsing:** The input `query_text` is split into words, and all n-grams (from `config.NGRAM_MIN` to `config.NGRAM_MAX` words) are generated (`utils.generate_ngrams_from_text`).
2.  **Region Parsing:** An optional `query_region_str` (e.g., `"top: 10-30, left: 50-70"`) is parsed by `utils.parse_region_string`. This function uses regular expressions to interpret various formats specifying top, left, bottom, and right boundaries (or ranges for top/left) as percentages. It performs validation and defaults to the full image `[0.0, 0.0, 100.0, 100.0]` if the string is missing, invalid, or cannot be parsed.
3.  **Output:** Returns the list of query n-grams and the single normalized query bounding box `query_bbox_norm`.

### 3.4 Search and Ranking (`searcher.py`)

**Objective:** Retrieve and rank images based on textual and spatial relevance.

**Process (`search_images`):**

1.  **Initialization:** An `image_scores = defaultdict(float)` is created to accumulate scores.
2.  **N-gram Lookup:** Iterate through each `query_ngram`. Look it up in the `inverted_index`.
3.  **Occurrence Scoring:** If the `query_ngram` is found, iterate through its list of locations `(image_id, ngram_bbox_norm)`. For each occurrence:
    - **Baseline Check:** If performing a baseline (non-spatial) search (`is_baseline_search=True`), the `spatial_score_component` is set to `1.0`.
    - **Non-Spatial Query Check:** If the `query_bbox_norm` is the default full image `[0.0, 0.0, 100.0, 100.0]`, the `spatial_score_component` is set to `1.0`.
    - **Spatial Scoring:** Otherwise (for a specific spatial query):
      - Calculate Intersection over Union: `iou = utils.calculate_iou(query_bbox_norm, ngram_bbox_norm)`. This measures the fractional overlap area.
      - Calculate Center Proximity: `query_center = utils.get_bbox_center(query_bbox_norm)`, `ngram_center = utils.get_bbox_center(ngram_bbox_norm)`. `proximity_score = utils.calculate_proximity_score(query_center, ngram_center)` using an exponential decay function `exp(-k * distance)`. This measures center-to-center closeness.
      - Combine Weighted Scores: `spatial_score_component = (config.IOU_WEIGHT * iou) + (config.PROXIMITY_WEIGHT * proximity_score)`. The weights (`IOU_WEIGHT`, `PROXIMITY_WEIGHT` in `config.py`) allow tuning the relative importance of overlap vs. closeness. This hybrid approach addresses limitations of using only IoU (which ignores nearby matches) or only proximity (which ignores overlap/size).
    - **N-gram Length Weighting:** Calculate `ngram_length = len(ngram.split())`. Longer n-gram matches are considered more significant.
    - **Score Contribution:** `score_contribution = spatial_score_component * ngram_length`.
    - **Accumulation:** Add the `score_contribution` to `image_scores[image_id]`.
4.  **Ranking (`rank_results`):** Sort the `image_scores` dictionary by the accumulated score (value) in descending order.
5.  **Output:** Return the sorted list of `(image_id, score)` tuples.

### 3.5 Synthetic Dataset Generation (`data_generation/`)

**Motivation:** Generating a synthetic dataset allowed for:

- **Control:** Precisely controlling the text content, layout, and repetition.
- **Ground Truth:** Obtaining perfect pixel-level bounding boxes for every word and n-gram, eliminating OCR errors as a confounding variable during algorithm development.
- **Targeted Queries:** Automatically generating queries with known ground truth target images and specific spatial relationships (overlap, proximity, etc.) to test different scoring scenarios systematically.
- **Scalability:** Efficiently generating large datasets (thousands of images, tens of thousands of queries) using parallel processing.

**Process:**

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

## 4. Evaluation

We conducted a quantitative evaluation (`evaluate.py`) to assess the performance of the spatially-aware search engine compared to relevant non-spatial baselines, using the generated synthetic dataset.

### 4.1 Evaluation Setup

- **Dataset:** The evaluation used the `queries.json` file generated by the synthetic data pipeline. For the reported results, this contained 50,000 queries derived from 2000 generated images.
- **Ground Truth Relevance:** For each query, the single "relevant" image was defined as the `image_id` specified in the query record (i.e., the image the query's target n-gram was originally sampled from). All other images were considered non-relevant for that query.
- **Configurations Compared:** Three configurations were compared:
  - **Spatial N-gram:** The full spatially-aware algorithm (`search_mode="spatial"`), combining spatial scoring (IoU + proximity with weights from `config.py`, e.g., 0.5/0.5) and n-gram length weighting.
  - **N-gram Baseline:** A non-spatial search using n-grams (`search_mode="ngram_text_only"`). This ignores the query region and uses a `spatial_score_component` of 1.0, effectively ranking based only on text match (n-gram presence) and `ngram_length`. This isolates the effect of n-grams compared to simple keywords.
  - **Keyword Baseline:** A rudimentary non-spatial search (`search_mode="keyword_only"`). This breaks the query text into unique words and scores images based simply on the count of matching words found anywhere in the image, ignoring n-grams and location. This serves as a fundamental baseline.
- **Cutoff:** Metrics were calculated using `k=10`.

### 4.2 Evaluation Metrics

Standard information retrieval metrics were chosen:

- **Mean Average Precision (MAP@k):** MAP is well-suited for evaluating the overall quality of a ranked list, especially when the order matters. It averages the Average Precision (AP) across all queries. AP for a single query rewards retrieving relevant items at higher ranks. In our single-relevant-item scenario, AP simplifies to `1 / rank` if the target image is found at rank `i <= k`, and 0 otherwise. High MAP indicates the system consistently ranks the correct item near the top.
- **Mean Precision@k (P@k):** P@k measures the proportion of relevant items found within the top `k` results, averaged across all queries. It reflects the system's ability to retrieve relevant items within the first page of results (recall within the top k).

_Note: In our specific evaluation setup, where each query has exactly one ground-truth relevant image, the P@k score for a single query is 1/k if the relevant image is ranked within the top k, and 0 otherwise. Therefore, the maximum achievable average P@k score across all queries is 1/k (e.g., 0.1 for k=10). This context is important when interpreting the reported P@k values._

### 4.3 Statistical Analysis

To determine if the observed performance differences between the configurations were statistically meaningful, we employed statistical significance testing.

- **Test:** The **Wilcoxon signed-rank test** was chosen. It is a non-parametric test suitable for comparing paired differences (the AP score for the Current Config vs. the Baseline on the _same_ query). It does not assume a normal distribution of the differences, which is appropriate for AP scores.
- **Tests:** Two **Wilcoxon signed-rank tests** were performed. This non-parametric test is suitable for comparing paired differences in AP scores on the same query without assuming normality.
- **Hypotheses:** We tested two one-sided alternative hypotheses:
  1. That the AP scores from the **Spatial N-gram** configuration are significantly _greater_ than those from the **N-gram Baseline**.
  2. That the AP scores from the **N-gram Baseline** are significantly _greater_ than those from the **Keyword Baseline**.
- **Significance Level:** A standard alpha level of `p < 0.05` was used.

### 4.4 Results

The evaluation script (`evaluate.py`) was run on the dataset (N=50000 queries) with `k=10`. The results were:

| Metric                                     | Spatial N-gram (IOU=0.5, Prox=0.5) | N-gram Baseline | Keyword Baseline |
| :----------------------------------------- | :--------------------------------- | :-------------- | :--------------- |
| Mean Average Precision (MAP@10) (Max: 1.0) | **0.6711**                         | 0.2110          | 0.0294           |
| Mean Precision@k (P@10) (Max: 0.1)         | **0.0795**                         | 0.0324          | 0.0054           |

**Statistical Significance (Wilcoxon Tests comparing AP scores):**

- Test 1 (Spatial N-gram vs. N-gram Baseline):
  - p-value = 0.0000
  - Result: The improvement of Spatial N-gram over N-gram Baseline is **STATISTICALLY SIGNIFICANT** (p < 0.05).
- Test 2 (N-gram Baseline vs. Keyword Baseline):
  - p-value = 0.0000
  - Result: The improvement of N-gram Baseline over Keyword Baseline is **STATISTICALLY SIGNIFICANT** (p < 0.05).

_(Note: Specific values are based on the latest run with 2000 images / 50k queries; re-running evaluation might yield slightly different numbers due to randomization in data generation if not using a fixed seed)._

### 4.5 Discussion/Analysis of Results

The three-way comparison provides clear insights into the contributions of both n-grams and spatial awareness.

- **Keyword Baseline Performance:** The Keyword Baseline performed poorly (MAP=0.0294, P@10=0.0054), indicating that simple keyword matching is insufficient for this task, likely due to common words appearing in many images.
- **N-gram Baseline Improvement:** The N-gram Baseline (MAP=0.2110, P@10=0.0324) significantly outperformed the Keyword Baseline (p < 0.0001). This demonstrates the substantial benefit of using multi-word n-grams and weighting by their length for textual relevance ranking, even without spatial information. Longer, more specific phrases help discriminate better than single words.
- **Spatial N-gram Advantage:** The Spatial N-gram configuration (MAP=0.6711, P@10=0.0795) dramatically outperformed the N-gram Baseline (p < 0.0001). This highlights the critical importance of the spatial scoring component (IoU and proximity). While the N-gram Baseline improves upon basic keyword search, incorporating spatial context allows the system to much more accurately rank the image where the text appears in the _correct location_.
- **P@k Analysis:** Comparing P@10 scores further clarifies the performance gains. The Keyword Baseline rarely found the target in the top 10 (P@10 ≈ 0.005, or 0.5% recall@10). The N-gram baseline improved this significantly (P@10 ≈ 0.032, or 3.2% recall@10), showing n-grams help surface the correct item more often. However, the Spatial N-gram configuration achieved a much higher recall within the top 10 (P@10 ≈ 0.080, or 8.0% recall@10). While the MAP metric shows an even larger relative improvement (reflecting much better ranking _within_ the top 10), the P@10 results confirm that spatial awareness also substantially increases the likelihood of retrieving the correct item within the top 10 results compared to the non-spatial methods.

In conclusion, both n-gram matching and spatial awareness provide statistically significant and substantial improvements over simpler approaches. N-grams enhance textual relevance, while spatial scoring is crucial for accurately ranking results based on location, leading to the best overall performance.

## 5. Visualization Tool (`visualize_search.py`)

To complement the quantitative evaluation, an interactive GUI tool was developed using Tkinter and Pillow. This tool allows users to:

- Enter query text and specify spatial regions using percentage inputs.
- Execute searches using the implemented backend.
- View ranked results (Top, Middle, and Last sections) in a scrollable grid.
- Inspect individual result images with overlays showing:
  _ The specified query region (semi-transparent blue).
  _ Bounding boxes around all found occurrences of the query n-grams within that image. \* Color-coding of n-gram boxes based on their IoU with the query region (Red=0 to Green=1), providing immediate visual feedback on spatial relevance according to overlap.
  This tool proved invaluable for debugging the region parsing, understanding the scoring behavior (IoU vs. proximity), and visually verifying search results.

## 6. Conclusion and Future Work

### 6.1 Summary of Findings

We successfully designed, implemented, and evaluated a spatially-aware textual image search engine. By indexing text n-grams with their normalized spatial coordinates and employing a scoring function that weights both spatial overlap (IoU) and proximity, the system demonstrates a statistically significant improvement in ranking quality (MAP) compared to relevant non-spatial baselines on a large synthetic dataset. The developed synthetic data pipeline and visualization tool were crucial for iterative development and analysis.

### 6.2 Limitations

The current work relies on synthetic data with perfect text bounding boxes. Real-world application would require integrating an actual OCR engine (like Tesseract via pytesseract, as originally planned), which introduces challenges like OCR errors, inaccurate bounding boxes, and confidence scoring. The current relevance model in the evaluation is binary and based on a single ground-truth source image per query. Real-world evaluation would need human-annotated graded relevance. The current scoring weights (IOU=0.5, Proximity=0.5) were chosen as a balance but may not be optimal for all use cases.

Furthermore, the current system has several other limitations:

- **Index Scalability:** The entire inverted index is loaded into memory. This approach may not scale efficiently to extremely large datasets due to memory constraints, requiring investigation into disk-based or distributed indexing strategies for production use.
- **Lack of Semantic Understanding:** Search relies on exact n-gram matching. It cannot handle synonyms, paraphrasing, or related concepts (e.g., finding "huge sale" when searching for "big discount"), limiting robustness when query wording differs from the in-image text.
- **Simple Spatial Query Language:** Queries are restricted to a single rectangular region. More complex spatial queries (e.g., relative positioning like "text A near text B", non-rectangular shapes) are not supported.

While effective for its targeted task, these limitations are notable when compared to the capabilities of recent MLLM-based approaches [9, 10], which excel at semantic understanding and handling complex natural language queries.

However, a key advantage of our system lies in its **Interpretability and Control**. Unlike the often opaque reasoning of large MLLMs, our scoring is directly tied to explicit geometric calculations (IoU, proximity), making results easier to understand, debug, and tune via weighting parameters.

### 6.3 Future Directions

- **Real-World Integration:** Integrate Tesseract/pytesseract for OCR, incorporate OCR confidence scores into indexing/ranking, and handle noisy/inaccurate bounding boxes.
- **Real-World Evaluation:** Collect or utilize existing datasets with real images and human annotations for spatial text queries. Evaluate performance using graded metrics like Normalized Discounted Cumulative Gain (nDCG) in addition to MAP/P@k.
  _Using nDCG effectively would require moving beyond the current synthetic data's binary, single-relevant-item ground truth; it necessitates gathering human relevance judgments, potentially with multiple relevance levels (e.g., perfect, good, fair, bad), for queries against real-world images, rather than the binary scoring system we've currently adopted._
- **Advanced Scoring Models:** Explore more sophisticated scoring functions, potentially incorporating semantic text similarity, visual context features, or adaptive weighting schemes.
- **Spatial Indexing:** For extremely large datasets, investigate dedicated spatial indexing structures (e.g., R-trees) alongside the inverted index to potentially accelerate spatial filtering, although the current approach filters post-retrieval.
- **User Interface Enhancements:** Improve the GUI visualizer, potentially allowing users to draw query boxes directly on an image.
- **Hybrid Systems:** Explore hybrid architectures combining the strengths of this system (precision, interpretability for exact geometric queries) with MLLMs [9, 10]. This could involve using our system for fast candidate generation, followed by MLLM-based re-ranking for semantic refinement or handling more complex spatial relations.
- **Semantic Augmentation:** Investigate methods to incorporate semantic understanding, perhaps by augmenting n-gram matching with text embeddings to handle synonyms or related concepts, bridging the gap towards MLLM capabilities.
- **Advanced Evaluation for Grounding:** Incorporate grounding-specific evaluation metrics, inspired by recent work like SMuDGE [11], especially for real-world evaluation. This would allow measuring not just if the correct image is retrieved, but how accurately the spatial constraint was met within that image.
- **Complex Spatial Queries:** Extend the query language to support more complex spatial relationships (e.g., relative positioning, non-rectangular areas), drawing inspiration from the capabilities of MLLMs or other spatial reasoning frameworks.

## References

1. Changpinyo, S., Sharma, P., Ding, N., & Soricut, R. (2021). Telling the What While Pointing to the Where: Multimodal Queries for Image Retrieval. In _Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)_ (pp. 1528-1538). [https://openaccess.thecvf.com/content/ICCV2021/papers/Changpinyo_Telling_the_What_While_Pointing_to_the_Where_Multimodal_Queries_ICCV_2021_paper.pdf](https://openaccess.thecvf.com/content/ICCV2021/papers/Changpinyo_Telling_the_What_While_Pointing_to_the_Where_Multimodal_Queries_ICCV_2021_paper.pdf)
2. Mai, L., Zhang, H., & Feng, Z. (2017). Spatial-Semantic Image Search. In _Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)_ (pp. 5589-5598). [https://openaccess.thecvf.com/content_cvpr_2017/papers/Mai_Spatial-Semantic_Image_Search_CVPR_2017_paper.pdf](https://openaccess.thecvf.com/content_cvpr_2017/papers/Mai_Spatial-Semantic_Image_Search_CVPR_2017_paper.pdf)
3. Karpathy, A., & Fei-Fei, L. (2014). Deep Fragment Embeddings for Bidirectional Image Sentence Mapping. _arXiv preprint arXiv:1406.5679_. [https://arxiv.org/pdf/1406.5679](https://arxiv.org/pdf/1406.5679)
4. Manmatha, R., Rath, T. M., & Feng, F. (2000). Searching Text in Images. _CIIR Technical Report_. University of Massachusetts Amherst. [https://ciir-publications.cs.umass.edu/getpdf.php?id=317](https://ciir-publications.cs.umass.edu/getpdf.php?id=317)
5. Faghri, F., Fleet, D. J., Kiros, J. R., & Fidler, S. (2017). VSE++: Improving Visual-Semantic Embeddings with Hard Negatives. _arXiv preprint arXiv:1707.05612_. [https://arxiv.org/pdf/1707.05612](https://arxiv.org/pdf/1707.05612)
6. Cao, M., Bai, Y., Wang, J., Cao, Z., Nie, L., & Zhang, M. (2023). Efficient Image-Text Retrieval via Keyword-Guided Pre-Screening. _arXiv preprint arXiv:2303.07740_. [https://arxiv.org/pdf/2303.07740](https://arxiv.org/pdf/2303.07740)
7. Chen, Z., Zhu, Y., Zhang, W., Joty, S. R., & Bing, L. (2023). STAIR: Learning Sparse Text and Image Representation in Grounded Tokens. _ICLR 2023 Workshop_. [https://openreview.net/forum?id=HXUdnYIe8r](https://openreview.net/forum?id=HXUdnYIe8r)
8. Bai, Y., Yu, Z., Xu, X., Yang, X., Wang, X., & Qin, B. (2024). Efficient Text-Image Sparse Retrieval via Bernoulli Random Variables Controlled Query Expansion. _arXiv preprint arXiv:2402.17535_. [https://arxiv.org/pdf/2402.17535](https://arxiv.org/pdf/2402.17535)
9. Huang, Z., Lv, F., Bai, W., Wang, X., Liu, J., Yang, H., ... & Wei, F. (2024). Grounding Multimodal Large Language Models to the World. _International Conference on Learning Representations (ICLR)_. [https://openreview.net/pdf/0ea36b222b82ac76c018c9aa7a47f9f978c705b2.pdf](https://openreview.net/pdf/0ea36b222b82ac76c018c9aa7a47f9f978c705b2.pdf) (Referenced as KOSMOS-2)
10. Yin, Z., Chen, C., Savva, M., & Sung, F. (2024). Groma: Grounded Multimodal Large Language Model with Localized Visual Tokenization. _European Conference on Computer Vision (ECCV)_. (Code: [https://github.com/FoundationVision/Groma](https://github.com/FoundationVision/Groma))
11. Nguyen, H. D., Bull, A. N., & Nair, V. (2025). Where is this coming from? Making groundedness count in the evaluation of Document VQA models. _arXiv preprint arXiv:2503.19120_. [https://arxiv.org/html/2503.19120v1](https://arxiv.org/html/2503.19120v1) (Referenced as SMuDGE framework)

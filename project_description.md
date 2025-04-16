# **Final Project Proposal: Spatially-Aware Textual Image Search**

## **Project Description**

This project proposes the development and evaluation of a spatially-aware textual image search engine. The system will leverage Optical Character Recognition (OCR) with word-level region detection (specifically utilizing pytesseract which wraps the Tesseract OCR engine) to index text present within images, along with its precise location and confidence score. Unlike traditional image search that relies primarily on visual content or simple keyword matching across the entire image, this system will enable users to search for specific text appearing within user-defined spatial regions of images.

The core idea is to enhance the precision and relevance of image search by incorporating the spatial context of textual information. Users will be able to formulate queries that specify both the text they are looking for and the area within the image where they expect to find it (e.g., "find 'price' in the bottom right corner," "show images where 'SOLD OUT' is near a product image").

The project will involve:

1. **Image Pre-processing (Indexing):** Processing a dataset of images using pytesseract to extract individual words, their bounding box coordinates, and OCR confidence scores. Low-confidence words will be filtered out. Bounding boxes will be normalized. N-grams (sequences of 1 to 3 words) will be generated, their normalized bounding boxes calculated, and an inverted index mapping n-grams to their locations ((image\_id, bbox)) will be built.  
2. **Query Processing:** Allowing users to input a textual query and optionally define a target spatial region using percentage offsets from the top and left of the image.  
3. **Search and Ranking Algorithm:** Implementing an algorithm that retrieves images based on the presence of query n-grams. The ranking score for each image will be based on the spatial overlap (Intersection over Union \- IoU) between the n-gram's bounding box and the query region, weighted by the length of the n-gram.  
4. **Evaluation:** Creating a ground truth dataset by running the search algorithm for specific queries and having human annotators assign graded relevance scores to the top-ranked images based on both textual and spatial relevance. The system's performance will be evaluated using Precision@k and Mean Average Precision (MAP) calculated from these annotations.

The successful completion of this project will demonstrate the effectiveness of incorporating spatial awareness into textual image search, leading to a more precise and user-centric way to discover information within images.

## **Algorithm**

The algorithm consists of two main phases: offline pre-processing/indexing and online query handling/search/ranking.

**I. Pre-processing (Image Indexing using Pytesseract):**

This phase is performed once for the image dataset to build the search index.

1. **Initialization:** Create an empty data structure (e.g., a dictionary) to serve as the inverted\_index. Define the maximum n-gram length N (e.g., N=3). Define an OCR confidence threshold (e.g., min\_confidence \= 60 on a 0-100 scale).  
2. **Iterate Through Images:** For each image in the dataset:  
   * Get the image's unique image\_id and its dimensions (width, height).  
   * **Run OCR:** Use pytesseract (specifically, a function like image\_to\_data with output\_type=Output.DICT) to extract information for each detected word in the image. This typically includes the word's text, its bounding box coordinates (e.g., left, top, width, height), and a confidence score.  
   * **Filter & Normalize Words:** Create a list of valid words for the current image. For each word detected by pytesseract:  
     * Check if its confidence score meets or exceeds min\_confidence.  
     * If the confidence is sufficient and the word text is not empty:  
       * Calculate the word's bounding box in \[top, left, bottom, right\] format from the pytesseract output.  
       * Normalize the bounding box coordinates to percentages (0-100) relative to the image's height and width:  
         * norm\_top \= (top / image\_height) \* 100  
         * norm\_left \= (left / image\_width) \* 100  
         * norm\_bottom \= ((top \+ height) / image\_height) \* 100  
         * norm\_right \= ((left \+ width) / image\_width) \* 100  
       * Store the word's text and its normalized bounding box \[norm\_top, norm\_left, norm\_bottom, norm\_right\]. Keep track of the order of words.  
   * **Generate and Index N-grams:** Using the filtered and ordered list of words and their normalized bounding boxes for the current image:  
     * Iterate through all possible n-gram lengths from 1 up to N.  
     * For each length n, iterate through all possible starting word positions i.  
     * Construct the n-gram text by joining the n words starting from position i.  
     * Calculate the bounding box for this n-gram by finding the minimum top, minimum left, maximum bottom, and maximum right coordinates among the normalized bounding boxes of the n constituent words (i.e., the union of the word boxes).  
     * **Add to Index:** In the inverted\_index, find the entry for the ngram text. If it doesn't exist, create a new empty list. Append a tuple (image\_id, ngram\_normalized\_bbox) to this list.  
3. **Final Index:** After processing all images, the inverted\_index contains the mapping from each detected n-gram (that met confidence thresholds) to a list of all locations (image and normalized bounding box) where it appears.

**II. Query Handling, Search, and Ranking:**

This phase is performed each time a user submits a search query.

1. **Parse Query:**  
   * Receive the user's query\_text and optional query\_region\_str (e.g., "top: 10-30, left: 60-80").  
   * Split the query\_text into words.  
   * Generate all query n-grams from the query words, for lengths 1 up to N (e.g., N=3).  
   * Parse the query\_region\_str. If it's provided, convert it into a normalized bounding box format query\_bbox\_norm \= \[min\_top, min\_left, max\_bottom, max\_right\] (e.g., \[10, 60, 30, 80\]). If no region string is provided, use the default full image region query\_bbox\_norm \= \[0, 0, 100, 100\].  
2. **Initialize Scores:** Create an empty dictionary image\_scores to store the accumulated relevance score for each candidate image.  
3. **Retrieve and Score Matches:**  
   * Iterate through each ngram generated from the query\_text.  
   * Check if the ngram exists as a key in the inverted\_index.  
   * If it exists:  
     * Retrieve the list of occurrences: list\_of\_locations \= inverted\_index\[ngram\].  
     * For each occurrence (image\_id, ngram\_bbox\_norm) in list\_of\_locations:  
       * **Determine Spatial Score (IoU Component):** Check if the query is non-spatial (i.e., if query\_bbox\_norm is \[0, 0, 100, 100\]).  
         * If the query is non-spatial, set the spatial score component IoU \= 1.  
         * Otherwise (if a specific spatial region was provided), calculate the geometric Intersection over Union (IoU) between query\_bbox\_norm and ngram\_bbox\_norm. IoU is calculated as Area\_of\_Overlap / Area\_of\_Union, resulting in a value between 0 and 1\.  
       * **Calculate N-gram Weight:** Determine the length of the current ngram in terms of the number of words it contains (ngram\_length).  
       * **Calculate Contribution:** Calculate the score contribution for this specific match: score\_contribution \= IoU \* ngram\_length.  
       * **Accumulate Image Score:** Add this contribution to the total score for the image\_id in the image\_scores dictionary. If the image\_id is not yet in the dictionary, initialize its score to 0 before adding. (image\_scores\[image\_id\] \= image\_scores.get(image\_id, 0\) \+ score\_contribution).  
4. **Rank Results:** Sort the image\_scores dictionary in descending order based on the accumulated scores.  
5. **Return Results:** Return the list of image\_ids according to their rank.

## **Evaluation Method**

**Ground Truth Dataset Creation:**

1. Select a diverse subset of images from the indexed dataset (e.g., 30-50 images).  
2. Define a set of representative queries (e.g., 15-20), each including specific query text and a target spatial region (e.g., query: "price", region: "bottom: 80-100, right: 70-100"). Include some queries without spatial regions.  
3. Run the SEARCH function (implementing the algorithm above) for each query on the selected image subset, retrieving the top K results (e.g., K=10).  
4. Have at least two human annotators independently assign a graded relevance score (0-4) to each of the top K images retrieved for each query. The scoring should be based on both the presence/accuracy of the text *and* its location relative to the specified target region, using a scale like this:  
   * **4 \- Highly Relevant:** The exact query text (or very close semantic match) is prominently located and clearly *within* the specified target region.  
   * **3 \- Relevant:** The query text is present and significantly *overlaps* with the target region, or a key part of the query text is within the region.  
   * **2 \- Partially Relevant:** The query text is present, but only marginally overlaps or is very near the target region. Or, the text is relevant but slightly different (e.g., synonym), and spatially somewhat correct.  
   * **1 \- Marginally Relevant:** The query text is found in the image but is clearly *outside* the target region, OR text somewhat related to the query is within the region.  
   * **0 \- Not Relevant:** The query text is absent from the image, or completely unrelated text is found in the target region.  
   * *Note:* For queries without a spatial region, relevance depends only on the presence and prominence of the text anywhere in the image.  
5. Store the annotations in a structured format (e.g., a Python dictionary) mapping each query to a dictionary of image IDs and a list of their assigned relevance scores (one score per annotator):

ground\_truth\_annotations \= {  
    "query1\_text\_regionA": {"image\_id\_a": \[4, 3\], "image\_id\_b": \[2, 3\], ...},  
    "query2\_text\_only": {"image\_id\_c": \[4, 4\], "image\_id\_d": \[0, 0\], ...},  
    ...  
}

**Evaluation Metrics:**

Calculate standard information retrieval metrics using the ground truth annotations. Define a relevance threshold (e.g., threshold \= 3, meaning scores of 3 and 4 are considered relevant).

1. **Precision@k (P@k):** For each query, calculate the proportion of the top k retrieved images that are relevant (average score \>= threshold). Average P@k across all queries.  
   * P@k \= (Number of relevant results in top k) / k  
2. **Mean Average Precision (MAP):** Calculate the Average Precision (AP) for each query. AP rewards systems that rank relevant documents higher. It's the average of precision values obtained after each relevant document is retrieved. MAP is the mean of AP scores across all queries.  
   * AP \= (Sum of (P@i \* rel(i)) for i=1 to k) / (Number of relevant documents)  
     (where rel(i) is 1 if the item at rank i is relevant, 0 otherwise; P@i is the precision at rank i).  
   * MAP \= (Sum of AP for all queries) / (Number of queries)

These metrics will provide a quantitative measure of the system's ability to retrieve textually and spatially relevant images and rank them appropriately.
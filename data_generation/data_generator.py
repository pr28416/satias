import os
import random
import math
from PIL import Image, ImageDraw, ImageFont
from faker import Faker
import config

# Initialize Faker
fake = Faker()


# --- Step 2: Sentence Pool Generation ---
def generate_sentence_pool(num_sentences=config.NUM_SENTENCES_IN_POOL):
    """Generates a pool of unique sentences using Faker."""
    sentences = set()
    while len(sentences) < num_sentences:
        sentences.add(fake.sentence(nb_words=random.randint(5, 15)))
    return list(sentences)


# --- Helper: Font Loading ---
def load_font(font_path=config.FONT_PATH, font_size=config.FONT_SIZE):
    """Loads the font, falling back to Pillow's default if path is None or invalid."""
    if font_path:
        try:
            return ImageFont.truetype(font_path, font_size)
        except IOError:
            print(f"Warning: Font not found at {font_path}. Using default font.")
            return ImageFont.load_default()  # Adjust size later if needed
    else:
        # Using load_default() doesn't directly support size,
        # but it's better than failing. Text size will depend on the default font.
        # For more control, specifying a valid FONT_PATH is recommended.
        print(f"Warning: No font path specified. Using default font.")
        return ImageFont.load_default()


# --- Step 3: Image Generation & Word Metadata ---
def generate_image_and_metadata(image_id, sentence_pool):
    """Generates a single image with text and returns its metadata."""
    img = Image.new("RGB", (config.W, config.H), color="white")
    draw = ImageDraw.Draw(img)
    font = load_font()

    # Basic font metrics (may be less accurate for default font)
    try:
        # Use textbbox which is more reliable for ascent/descent
        # Get height of a standard character like 'Mg'
        bbox = draw.textbbox((0, 0), "Mg", font=font)
        line_height = bbox[3] - bbox[1]
        effective_line_height = line_height * config.LINE_SPACING_FACTOR
    except AttributeError:  # Fallback for older Pillow or basic default font
        # This might be less accurate
        ascent, descent = font.getmetrics()
        line_height = ascent + descent
        effective_line_height = line_height * config.LINE_SPACING_FACTOR
        print(
            "Warning: Using legacy font.getmetrics(). Line spacing might be less accurate."
        )

    # Select sentences and form text block
    num_sentences_to_use = random.randint(
        config.MIN_SENTENCES_PER_IMAGE, config.MAX_SENTENCES_PER_IMAGE
    )
    selected_sentences = random.sample(sentence_pool, num_sentences_to_use)
    text_block = " ".join(selected_sentences)

    # --- Inject Test Phrases (Optional) ---
    words = list(text_block.split())  # Start with words from random sentences
    if (
        config.TEST_PHRASES
        and random.random() < config.TEST_PHRASE_INJECTION_PROBABILITY
    ):
        num_phrases_to_inject = random.randint(1, config.MAX_TEST_PHRASES_TO_INJECT)
        phrases_to_inject = random.sample(config.TEST_PHRASES, num_phrases_to_inject)

        for phrase in phrases_to_inject:
            num_insertions = random.randint(
                config.MIN_INSERTIONS_PER_PHRASE, config.MAX_INSERTIONS_PER_PHRASE
            )
            phrase_words = phrase.split()
            for _ in range(num_insertions):
                if not words:  # Avoid inserting into empty list
                    words.extend(phrase_words)
                else:
                    insert_index = random.randint(0, len(words))
                    # Insert words one by one to maintain order
                    for i, word in enumerate(phrase_words):
                        words.insert(insert_index + i, word)

    # Shuffle slightly to break up obvious sentence structure if phrases were added
    # (Optional, can be commented out if structure is preferred)
    # random.shuffle(words)

    word_data = []
    x, y = config.MARGIN, config.MARGIN

    for word in words:
        if not word:
            continue

        try:
            # Get bounding box *before* drawing
            # textbbox returns [left, top, right, bottom] relative to the given (x,y) top-left corner
            word_bbox = draw.textbbox((x, y), word, font=font)
            word_width = word_bbox[2] - word_bbox[0]
            word_height = word_bbox[3] - word_bbox[1]  # Use bbox height

        except Exception as e:
            print(f"Error getting textbbox for word '{word}': {e}. Skipping word.")
            continue

        # Check for line wrap
        if word_bbox[2] > config.W - config.MARGIN:
            x = config.MARGIN
            y += effective_line_height
            # Recalculate bbox at new position
            try:
                word_bbox = draw.textbbox((x, y), word, font=font)
                word_width = word_bbox[2] - word_bbox[0]
                word_height = word_bbox[3] - word_bbox[1]
            except Exception as e:
                print(
                    f"Error getting textbbox for word '{word}' after wrap: {e}. Skipping word."
                )
                continue

        # # Check if word fits vertically (REMOVED to allow text bleed)
        # if word_bbox[3] > config.H - config.MARGIN:
        #     break # Stop adding words if it overflows vertically

        # Draw the word
        draw.text((x, y), word, fill="black", font=font)

        # Store word data with precise bbox [top, left, bottom, right]
        # Note: word_bbox is [left, top, right, bottom]
        pixel_bbox = [word_bbox[1], word_bbox[0], word_bbox[3], word_bbox[2]]
        word_data.append(
            {"text": word, "bbox_pixels": pixel_bbox}  # [top, left, bottom, right]
        )

        # Update x for the next word
        x += word_width + config.WORD_SPACING

    # Save the image
    image_path = os.path.join(config.IMAGE_DIR, f"{image_id}.png")
    img.save(image_path)

    metadata = {
        "image_id": image_id,
        "width": config.W,
        "height": config.H,
        "image_path": image_path,
        "word_data": word_data,
    }
    return metadata


# --- Step 4: N-gram Calculation ---
def calculate_ngrams(word_data, n_min=config.NGRAM_MIN, n_max=config.NGRAM_MAX):
    """Calculates n-grams and their union bounding boxes from word data."""
    ngrams_data = []
    num_words = len(word_data)

    for n in range(n_min, n_max + 1):
        for i in range(num_words - n + 1):
            ngram_words = word_data[i : i + n]
            ngram_text = " ".join([wd["text"] for wd in ngram_words])

            # Calculate union bounding box
            min_top = min(wd["bbox_pixels"][0] for wd in ngram_words)
            min_left = min(wd["bbox_pixels"][1] for wd in ngram_words)
            max_bottom = max(wd["bbox_pixels"][2] for wd in ngram_words)
            max_right = max(wd["bbox_pixels"][3] for wd in ngram_words)

            union_bbox_pixels = [min_top, min_left, max_bottom, max_right]

            ngrams_data.append(
                {
                    "text": ngram_text,
                    "bbox_pixels": union_bbox_pixels,
                    "words": ngram_words,  # Keep refs to original words if needed
                }
            )
    return ngrams_data


# --- Step 5: Query Generation ---


# Helper: Calculate IoU
def calculate_iou(boxA, boxB):
    """Calculates Intersection over Union for two bounding boxes.
    Boxes are expected in [top, left, bottom, right] format (normalized 0-100)."""
    # Determine the coordinates of the intersection rectangle
    top_int = max(boxA[0], boxB[0])
    left_int = max(boxA[1], boxB[1])
    bottom_int = min(boxA[2], boxB[2])
    right_int = min(boxA[3], boxB[3])

    # Compute the area of intersection rectangle
    # Ensure the intersection is valid (positive width and height)
    interArea = max(0, right_int - left_int) * max(0, bottom_int - top_int)
    if interArea == 0:
        return 0.0

    # Compute the area of both bounding boxes
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    # Compute the area of the union
    unionArea = boxAArea + boxBArea - interArea

    # Compute the Intersection over Union
    iou = interArea / unionArea
    return iou


# Helper: Normalize Bbox
def normalize_bbox(bbox_pixels, width, height):
    """Converts pixel bbox [t, l, b, r] to normalized [0-100]."""
    t, l, b, r = bbox_pixels
    return [
        (t / height) * 100.0,
        (l / width) * 100.0,
        (b / height) * 100.0,
        (r / width) * 100.0,
    ]


# Helper: Clamp coordinates
def clamp(value, min_val=0.0, max_val=100.0):
    return max(min_val, min(value, max_val))


# Helper: Generate perturbed bbox for overlap/nearby/distant
def generate_perturbed_bbox(target_norm, strategy, params):
    """Generates a new normalized bbox based on the target and strategy."""
    t, l, b, r = target_norm
    h = b - t
    w = r - l

    if h <= 0 or w <= 0:  # Handle degenerate cases
        return [0.0, 0.0, 10.0, 10.0]  # Return small default box

    q_t, q_l, q_b, q_r = t, l, b, r  # Start with target

    if strategy == "High IoU Overlap" or strategy == "Low IoU Overlap":
        scale_min, scale_max = params["scale"]
        offset_min, offset_max = params["offset"]

        scale_h = random.uniform(scale_min, scale_max)
        scale_w = random.uniform(scale_min, scale_max)
        offset_y = random.uniform(offset_min, offset_max) * h
        offset_x = random.uniform(offset_min, offset_max) * w

        center_y = t + h / 2
        center_x = l + w / 2

        new_h = h * scale_h
        new_w = w * scale_w

        q_t = center_y - new_h / 2 + offset_y
        q_l = center_x - new_w / 2 + offset_x
        q_b = q_t + new_h
        q_r = q_l + new_w

    elif strategy == "Nearby" or strategy == "Distant":
        offset_factor_min, offset_factor_max = params["offset_factor"]
        offset_factor = random.uniform(offset_factor_min, offset_factor_max)

        # Choose a random direction (0=up, 1=down, 2=left, 3=right, 4-7=diagonals)
        direction = random.randint(0, 7)
        dist_y = offset_factor * h
        dist_x = offset_factor * w

        if direction == 0:  # Up
            q_t, q_b = t - dist_y, b - dist_y
        elif direction == 1:  # Down
            q_t, q_b = t + dist_y, b + dist_y
        elif direction == 2:  # Left
            q_l, q_r = l - dist_x, r - dist_x
        elif direction == 3:  # Right
            q_l, q_r = l + dist_x, r + dist_x
        elif direction == 4:  # Up-Left
            q_t, q_b = t - dist_y, b - dist_y
            q_l, q_r = l - dist_x, r - dist_x
        elif direction == 5:  # Up-Right
            q_t, q_b = t - dist_y, b - dist_y
            q_l, q_r = l + dist_x, r + dist_x
        elif direction == 6:  # Down-Left
            q_t, q_b = t + dist_y, b + dist_y
            q_l, q_r = l - dist_x, r - dist_x
        else:  # Down-Right
            q_t, q_b = t + dist_y, b + dist_y
            q_l, q_r = l + dist_x, r + dist_x

    # Clamp all coordinates to be within [0, 100]
    q_t = clamp(q_t)
    q_l = clamp(q_l)
    q_b = clamp(q_b)
    q_r = clamp(q_r)

    # Ensure bottom > top and right > left after clamping
    if q_b <= q_t:
        q_b = q_t + 1.0  # Min height
    if q_r <= q_l:
        q_r = q_l + 1.0  # Min width
    q_b = clamp(q_b)
    q_r = clamp(q_r)

    return [q_t, q_l, q_b, q_r]


def generate_queries_for_image(image_id, image_width, image_height, ngrams_data):
    """Generates a list of queries for a single image based on its n-grams."""
    if not ngrams_data:
        return []

    queries = []
    query_types = list(config.QUERY_TYPE_DISTRIBUTION.keys())
    query_weights = list(config.QUERY_TYPE_DISTRIBUTION.values())

    # Pre-calculate normalized bboxes for all ngrams
    for ng in ngrams_data:
        ng["bbox_norm"] = normalize_bbox(ng["bbox_pixels"], image_width, image_height)

    for q_idx in range(config.NUM_QUERIES_PER_IMAGE):
        # Select a random target n-gram
        target_ngram = random.choice(ngrams_data)
        target_bbox_norm = target_ngram["bbox_norm"]

        # Choose query type based on distribution
        query_type = random.choices(query_types, weights=query_weights, k=1)[0]

        query_region_norm = None
        if query_type == "No Region":
            query_region_norm = [0.0, 0.0, 100.0, 100.0]
        elif query_type == "Exact Match":
            query_region_norm = target_bbox_norm[:]  # Make a copy
        elif query_type == "High IoU Overlap":
            query_region_norm = generate_perturbed_bbox(
                target_bbox_norm, query_type, config.HIGH_IOU_PARAMS
            )
            # Optional: Re-check IoU and regenerate if not high enough? For simplicity, we assume generation aims correctly.
        elif query_type == "Low IoU Overlap":
            query_region_norm = generate_perturbed_bbox(
                target_bbox_norm, query_type, config.LOW_IOU_PARAMS
            )
            # Optional: Re-check IoU and regenerate if not low enough / too high?
        elif query_type == "Nearby":
            query_region_norm = generate_perturbed_bbox(
                target_bbox_norm, query_type, config.NEARBY_PARAMS
            )
        elif query_type == "Distant":
            query_region_norm = generate_perturbed_bbox(
                target_bbox_norm, query_type, config.DISTANT_PARAMS
            )

        if query_region_norm is None:  # Should not happen with current types
            print(
                f"Warning: query_region_norm is None for type {query_type}. Defaulting to full image."
            )
            query_region_norm = [0.0, 0.0, 100.0, 100.0]

        query = {
            "query_id": f"q_{image_id}_{q_idx}",
            "image_id": image_id,
            "query_text": target_ngram["text"],
            "query_region_norm": query_region_norm,  # [t,l,b,r] percentages, or [0,0,100,100]
            "target_ngram_bbox_pixels": target_ngram[
                "bbox_pixels"
            ],  # Ground truth location in pixels
            "target_ngram_bbox_norm": target_bbox_norm,  # Ground truth location normalized
            "generation_type": query_type,  # String identifier
        }
        queries.append(query)

    return queries

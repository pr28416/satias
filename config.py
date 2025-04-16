import os

# --- Image Generation ---
W = 640  # Image Width (pixels)
H = 360  # Image Height (pixels)
NUM_IMAGES = 50  # Number of images to generate
# Attempt to use a common default font, adjust if not found on your system
# On macOS, common paths include "/System/Library/Fonts/Helvetica.ttc"
# On Linux, common paths include "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
# On Windows, common paths include "C:/Windows/Fonts/arial.ttf"
# If None, Pillow's default will be used, which might be basic.
FONT_PATH = "/System/Library/Fonts/Supplemental/Arial.ttf"  # Or specify a path like "/System/Library/Fonts/Helvetica.ttc"
FONT_SIZE = 16
MARGIN = 10  # Pixels
WORD_SPACING = 5  # Pixels between words on the same line
LINE_SPACING_FACTOR = 1.4  # Multiplier for line height

# --- Text Generation ---
NUM_SENTENCES_IN_POOL = 200  # Size of the pool to sample from
MIN_SENTENCES_PER_IMAGE = 15
MAX_SENTENCES_PER_IMAGE = 25
# Add configuration for injecting specific test phrases
TEST_PHRASES = ["special offer", "limited time", "click here now", "important update"]
TEST_PHRASE_INJECTION_PROBABILITY = 0.3  # 30% chance to inject phrases into an image
MAX_TEST_PHRASES_TO_INJECT = 2  # Max different phrases to inject per image
MIN_INSERTIONS_PER_PHRASE = 2  # Min times to insert each chosen phrase
MAX_INSERTIONS_PER_PHRASE = 4  # Max times to insert each chosen phrase

# Configuration for replacing words with more notable ones
NOTABLE_WORDS = [
    # Examples (expand with more diverse/relevant words if needed)
    "algorithm",
    "database",
    "interface",
    "protocol",
    "parameter",
    "variable",
    "computation",
    "visualization",
    "heuristic",
    "optimization",
    "framework",
    "component",
    "architecture",
    "deployment",
    "integration",
    "validation",
    "galaxy",
    "nebula",
    "supernova",
    "asteroid",
    "comet",
    "orbit",
    "molecule",
    "enzyme",
    "chromosome",
    "protein",
    "synthesis",
    "catalyst",
    "symphony",
    "concerto",
    "crescendo",
    "melody",
    "harmony",
    "rhythm",
    "metaphor",
    "alliteration",
    "onomatopoeia",
    "hyperbole",
    "paradox",
    "symbolism",
    "extraordinary",
    "magnificent",
    "serendipity",
    "ephemeral",
    "quintessential",
    "ubiquitous",
]
NOTABLE_WORD_REPLACE_PROBABILITY = 0.5  # 50% chance to perform replacements in an image
MAX_NOTABLE_WORD_TYPES_TO_USE = 5  # Max different notable words to use per image
MIN_REPLACEMENTS_PER_TYPE = 1  # Min times to replace with each chosen notable word
MAX_REPLACEMENTS_PER_TYPE = 3  # Max times to replace with each chosen notable word

# --- N-gram Calculation ---
NGRAM_MIN = 1
NGRAM_MAX = 3

# --- Query Generation ---
NUM_QUERIES_PER_IMAGE = 10
# Target distribution for query types (should sum roughly to 1.0)
QUERY_TYPE_DISTRIBUTION = {
    "No Region": 0.20,
    "Exact Match": 0.20,
    "High IoU Overlap": 0.20,  # Target IoU > 0.7
    "Low IoU Overlap": 0.20,  # Target 0.1 < IoU < 0.4
    "Nearby": 0.10,  # Target IoU = 0, close proximity
    "Distant": 0.10,  # Target IoU = 0, far proximity
}
# Parameters for spatial query generation relative to target bbox
# Format: [min_scale, max_scale] for size change, [min_offset, max_offset] for position change (percentage of target dimension)
HIGH_IOU_PARAMS = {"scale": [0.8, 1.2], "offset": [-0.1, 0.1]}
LOW_IOU_PARAMS = {
    "scale": [0.5, 1.5],
    "offset": [-0.3, 0.3],
}  # Wider range, aiming for lower IoU
NEARBY_PARAMS = {
    "offset_factor": [1.1, 1.5]
}  # How many times its own size away (min/max)
DISTANT_PARAMS = {"offset_factor": [3.0, 6.0]}  # Further away

# --- Scoring ---
PROXIMITY_BONUS_FACTOR = 0.1  # Multiplier for proximity score when IoU is 0
PROXIMITY_DISTANCE_SCALE = 0.05  # Scaling factor 'k' for exp(-k * distance)
# --- NEW: Weights for combining IoU and Proximity ---
IOU_WEIGHT = 0.5  # Weight for IoU score component
PROXIMITY_WEIGHT = 0.5  # Weight for proximity score component

# --- Output ---
OUTPUT_DIR = "./synthetic_data"
IMAGE_DIR = os.path.join(OUTPUT_DIR, "images")
METADATA_DIR = os.path.join(OUTPUT_DIR, "metadata")
IMAGE_METADATA_FILE = os.path.join(METADATA_DIR, "image_metadata.json")
QUERIES_FILE = os.path.join(METADATA_DIR, "queries.json")

# --- Ensure distribution sums close to 1.0 ---
assert (
    abs(sum(QUERY_TYPE_DISTRIBUTION.values()) - 1.0) < 1e-9
), "Query type distribution must sum to 1.0"

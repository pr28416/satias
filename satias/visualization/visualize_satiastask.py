import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np
from scipy.ndimage import gaussian_filter

# --- Metadata for synth_000 ---
image_path = "writeup/synth_000.png"
width, height = 640, 360

# Only the relevant words for the n-gram
word_data = [
    {"text": "star", "bbox_pixels": [157.6, 151, 168.6, 175]},
    {"text": "rest.", "bbox_pixels": [157.6, 180, 168.6, 208]},
    {"text": "Point", "bbox_pixels": [157.6, 213, 168.6, 245]},
]

# Target n-gram for the query
ngram_bbox = [157.6, 151, 168.6, 245]  # "star rest. Point"

# Query region (normalized to [0, 100], convert to pixel coordinates)
query_region_norm = [43.77777777777778, 100.0, 46.833333333333336, 100.0]
query_region_pixels = [
    (query_region_norm[1] / 100) * width,  # left
    (query_region_norm[0] / 100) * height,  # top
    ((query_region_norm[3] - query_region_norm[1]) / 100) * width,  # width
    ((query_region_norm[2] - query_region_norm[0]) / 100) * height,  # height
]

fig, ax = plt.subplots(figsize=(10, 6))
img = Image.open(image_path)
ax.imshow(img)

# Draw the n-gram bounding box (answer)
t, l, b, r = ngram_bbox
ngram_rect = patches.Rectangle(
    (l, t),
    r - l,
    b - t,
    linewidth=3,
    edgecolor="green",
    facecolor="lime",
    alpha=0.3,
    zorder=2,
)
ax.add_patch(ngram_rect)

# Add a heat overlay centered on the green n-gram box (with padding)
pad_x = 30  # pixels of padding around the n-gram
pad_y = 20
heat_left = max(l - pad_x, 0)
heat_top = max(t - pad_y, 0)
heat_right = min(r + pad_x, width)
heat_bottom = min(b + pad_y, height)
heatmap = np.zeros((height, width))
heatmap[int(heat_top):int(heat_bottom), int(heat_left):int(heat_right)] = 1.0
heatmap = gaussian_filter(heatmap, sigma=20)
ax.imshow(
    np.dstack((np.ones_like(heatmap), np.zeros_like(heatmap), np.zeros_like(heatmap), 0.4 * heatmap)),
    extent=[0, width, height, 0],
    zorder=2.5,
)

# Draw word bounding boxes for the n-gram (keep the blue rectangles, but no labels)
for word in word_data:
    t, l, b, r = word["bbox_pixels"]
    rect = patches.Rectangle(
        (l, t),
        r - l,
        b - t,
        linewidth=2,
        edgecolor="blue",
        facecolor="none",
        alpha=0.9,
        zorder=4,
    )
    ax.add_patch(rect)

# Add a single blue label for the n-gram, bottom right, and indicate it's the query and location
# Calculate approximate percentage location for the query (top, left)
query_top_pct = int((ngram_bbox[0] / height) * 100)
query_left_pct = int((ngram_bbox[1] / width) * 100)
ngram_phrase = f'Query: "star rest. Point" (top: {query_top_pct}%, left: {query_left_pct}%)'
ax.text(
    width - 10,  # 10 pixels from the right edge
    height - 10, # 10 pixels from the bottom edge
    ngram_phrase,
    color="blue",
    fontsize=20,
    va="bottom",
    ha="right",
    weight="bold",
    bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=1),
    zorder=10,
)

# Remove axes and make everything big and readable
ax.axis("off")
ax.set_title(
    "SATIAS Process: Example Query and Answer", fontsize=22, weight="bold", pad=18
)

plt.tight_layout()
plt.savefig("writeup/satias_example.png", dpi=200)
plt.show()

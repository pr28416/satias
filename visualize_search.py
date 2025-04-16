import tkinter as tk
from tkinter import ttk, messagebox, Canvas, Frame, Scrollbar
import os
import sys
from PIL import Image, ImageTk, ImageDraw
from collections import defaultdict
from typing import List, Tuple, Optional, Dict, Any

# --- Adjust path to import from parent directory --- START
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
# --- Adjust path --- END

import config
from search_engine import indexer, query_parser, searcher, utils
from search_engine.indexer import InvertedIndexType

# --- Constants ---
THUMBNAIL_WIDTH = 320
RESULTS_PER_ROW = 3
QUERY_REGION_COLOR = (100, 100, 255, 100)  # Light Blue semi-transparent RGBA
# NGRAM_BOX_COLOR_MAP = ... # Could define a complex map, or calculate dynamically
NGRAM_BOX_ALPHA = 128  # Transparency for ngram boxes


class SearchVisualizerApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Search Result Visualizer")
        self.root.geometry("1100x750")  # Adjust initial size as needed
        self.root.columnconfigure(0, weight=1)  # Allow content to expand horizontally
        self.root.rowconfigure(1, weight=1)  # Allow results frame to expand vertically

        self.inverted_index: Optional[InvertedIndexType] = None

        # List to keep PhotoImage references
        self.image_references: List[ImageTk.PhotoImage] = []

        # --- Input Frame ---
        input_frame = ttk.Frame(root, padding="10")
        input_frame.pack(side=tk.TOP, fill=tk.X, pady=(0, 5))  # Add bottom padding
        # Configure resizing behavior for the input frame grid
        input_frame.columnconfigure(1, weight=1)
        input_frame.columnconfigure(3, weight=1)

        ttk.Label(input_frame, text="Query Text:").grid(
            row=0, column=0, padx=5, pady=5, sticky=tk.W
        )
        self.query_text_entry = ttk.Entry(input_frame, width=50)
        self.query_text_entry.grid(
            row=0, column=1, columnspan=3, padx=5, pady=5, sticky=tk.EW
        )

        ttk.Label(input_frame, text="Top Range (%):").grid(
            row=1, column=0, padx=5, pady=5, sticky=tk.W
        )
        self.top_start_entry = ttk.Entry(input_frame, width=5)
        self.top_start_entry.grid(
            row=1, column=1, padx=5, pady=5, sticky=tk.W
        )  # Sticky W
        ttk.Label(input_frame, text="-").grid(row=1, column=2, padx=1)
        self.top_end_entry = ttk.Entry(input_frame, width=5)
        self.top_end_entry.grid(
            row=1, column=3, padx=5, pady=5, sticky=tk.W
        )  # Sticky W

        ttk.Label(input_frame, text="Left Range (%):").grid(
            row=2, column=0, padx=5, pady=5, sticky=tk.W
        )
        self.left_start_entry = ttk.Entry(input_frame, width=5)
        self.left_start_entry.grid(
            row=2, column=1, padx=5, pady=5, sticky=tk.W
        )  # Sticky W
        ttk.Label(input_frame, text="-").grid(row=2, column=2, padx=1)
        self.left_end_entry = ttk.Entry(input_frame, width=5)
        self.left_end_entry.grid(
            row=2, column=3, padx=5, pady=5, sticky=tk.W
        )  # Sticky W

        self.search_button = ttk.Button(
            input_frame, text="Search", command=self.perform_search
        )
        self.search_button.grid(
            row=0, column=4, rowspan=3, padx=10, pady=5, sticky=tk.NS
        )

        # --- Status Bar --- (Optional)
        self.status_var = tk.StringVar()
        status_bar = ttk.Label(
            root,
            textvariable=self.status_var,
            relief=tk.SUNKEN,
            anchor=tk.W,
            padding="5",
        )
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        # --- Results Frame (Scrollable) ---
        results_outer_frame = ttk.Frame(root, padding="10")
        results_outer_frame.pack(
            side=tk.TOP, fill=tk.BOTH, expand=True
        )  # Changed side to TOP
        # Configure resizing behavior for the outer frame
        results_outer_frame.columnconfigure(0, weight=1)
        results_outer_frame.rowconfigure(0, weight=1)

        self.canvas = Canvas(
            results_outer_frame, borderwidth=0, highlightthickness=0
        )  # Removed highlight
        self.results_frame = ttk.Frame(self.canvas)  # Frame inside canvas
        vsb = ttk.Scrollbar(
            results_outer_frame, orient="vertical", command=self.canvas.yview
        )
        hsb = ttk.Scrollbar(
            results_outer_frame, orient="horizontal", command=self.canvas.xview
        )
        self.canvas.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

        # Use grid for better control within results_outer_frame
        self.canvas.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")

        self.canvas_window = self.canvas.create_window(
            (0, 0), window=self.results_frame, anchor="nw"
        )  # Add frame to canvas

        self.results_frame.bind(
            "<Configure>", self.on_frame_configure
        )  # Update scroll region
        self.canvas.bind(
            "<Configure>", self.on_canvas_configure
        )  # Resize canvas window

        # --- Load Index ---
        self.load_search_index()

    def on_frame_configure(self, event: Any = None) -> None:
        """Reset the scroll region to encompass the inner frame"""
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def on_canvas_configure(self, event: Any) -> None:
        """Reset the canvas window to encompass the inner frame when canvas resizes."""
        canvas_width = event.width
        self.canvas.itemconfig(self.canvas_window, width=canvas_width)

    def set_status(self, message: str) -> None:
        self.status_var.set(message)
        self.root.update_idletasks()

    def load_search_index(self) -> None:
        self.set_status(f"Loading index from {indexer.DEFAULT_INDEX_PATH}...")
        self.inverted_index = indexer.load_index()
        if self.inverted_index is None:
            self.set_status(
                "Error: Index not found or failed to load. Build index first."
            )
            messagebox.showerror(
                "Index Error",
                f"Could not load index file: {indexer.DEFAULT_INDEX_PATH}\n\nPlease build the index by running:\npython search_engine/indexer.py",
            )
            self.search_button.config(state=tk.DISABLED)
        else:
            self.set_status("Index loaded successfully. Ready to search.")
            self.search_button.config(state=tk.NORMAL)

    def parse_percent_range(
        self, start_entry: ttk.Entry, end_entry: ttk.Entry
    ) -> Tuple[Optional[float], Optional[float]]:
        """Parses start/end entries, validates, returns (start, end) or (None, None)."""
        start_str = start_entry.get().strip()
        end_str = end_entry.get().strip()
        start, end = None, None

        try:
            if start_str:
                start = float(start_str)
                if not (0 <= start <= 100):
                    messagebox.showerror(
                        "Input Error",
                        f"Start value '{start_str}' must be between 0 and 100.",
                    )
                    return None, None
            if end_str:
                end = float(end_str)
                if not (0 <= end <= 100):
                    messagebox.showerror(
                        "Input Error",
                        f"End value '{end_str}' must be between 0 and 100.",
                    )
                    return None, None

            if start is not None and end is not None and start > end:
                messagebox.showerror(
                    "Input Error",
                    f"Start value '{start}' cannot be greater than end value '{end}'.",
                )
                return None, None

            # If only end is provided, assume start is 0
            if end_str and not start_str:
                start = 0.0
            # If only start is provided, assume end is 100
            if start_str and not end_str:
                end = 100.0

            return start, end
        except ValueError:
            val_err = (
                start_str if not start_str.replace(".", "", 1).isdigit() else end_str
            )
            messagebox.showerror(
                "Input Error",
                f"Invalid number entered: '{val_err}'. Please enter percentages (0-100).",
            )
            return None, None

    def construct_region_string(
        self,
        top_range: Tuple[Optional[float], Optional[float]],
        left_range: Tuple[Optional[float], Optional[float]],
    ) -> Optional[str]:
        """Constructs the region string for the parser from validated ranges."""
        parts = []
        t_start, t_end = top_range
        l_start, l_end = left_range

        # Only add if both start and end are defined (after parse_percent_range defaults)
        if t_start is not None and t_end is not None:
            parts.append(f"top: {t_start:.1f}-{t_end:.1f}")
        # else: Error should have been caught by parse_percent_range returning None tuple

        if l_start is not None and l_end is not None:
            parts.append(f"left: {l_start:.1f}-{l_end:.1f}")
        # else: Error should have been caught

        return ", ".join(parts) if parts else None

    def get_color_for_iou(self, iou: float) -> Tuple[int, int, int]:
        """Maps IoU (0.0 to 1.0) to a Red-Yellow-Green color tuple (RGB)."""
        iou = max(0.0, min(1.0, iou))  # Clamp iou to [0, 1]
        # Simple linear interpolation: Red (iou=0) -> Yellow (iou=0.5) -> Green (iou=1)
        if iou < 0.5:
            # Red to Yellow (Increase Green)
            red = 255
            green = int(255 * (iou * 2))
            blue = 0
        else:
            # Yellow to Green (Decrease Red)
            red = int(255 * (1 - (iou - 0.5) * 2))
            green = 255
            blue = 0
        return (red, green, blue)

    def create_visualization(
        self,
        image_path: str,
        query_bbox_norm: List[float],
        found_details: List[Dict[str, Any]],
    ) -> Optional[ImageTk.PhotoImage]:
        """Loads image, draws query region and detected ngram boxes, returns PhotoImage."""
        try:
            img = Image.open(image_path).convert("RGBA")  # Use RGBA for transparency
            img_w, img_h = img.size

            # Create overlay layer
            overlay = Image.new(
                "RGBA", img.size, (255, 255, 255, 0)
            )  # Transparent overlay
            draw = ImageDraw.Draw(overlay)

            # --- Draw Query Region ---
            q_t, q_l, q_b, q_r = query_bbox_norm
            # Check if it's the default full image region
            is_default_region = (
                q_t == 0.0 and q_l == 0.0 and q_b == 100.0 and q_r == 100.0
            )
            if not is_default_region:
                query_pix_coords = [
                    int(q_l * img_w / 100.0),
                    int(q_t * img_h / 100.0),
                    int(q_r * img_w / 100.0),
                    int(q_b * img_h / 100.0),
                ]  # Pillow uses (left, top, right, bottom)
                draw.rectangle(query_pix_coords, fill=QUERY_REGION_COLOR)

            # --- Draw N-gram Boxes ---
            for detail in found_details:
                ng_bbox_norm = detail["bbox_norm"]
                iou = detail["iou"]
                ng_t, ng_l, ng_b, ng_r = ng_bbox_norm
                ngram_pix_coords = [
                    int(ng_l * img_w / 100.0),
                    int(ng_t * img_h / 100.0),
                    int(ng_r * img_w / 100.0),
                    int(ng_b * img_h / 100.0),
                ]
                # Color based on IoU
                rgb_color = self.get_color_for_iou(iou)
                rgba_fill_color = rgb_color + (NGRAM_BOX_ALPHA,)  # Add alpha
                # Draw the rectangle regardless of IoU; color indicates overlap
                draw.rectangle(
                    ngram_pix_coords,
                    outline=rgb_color,  # Color shows IoU (red=0)
                    fill=rgba_fill_color,  # Semi-transparent fill
                    width=2,  # Thicker outline
                )

                # Optional: Draw IoU value - might be too cluttered
                # draw.text((ngram_pix_coords[0], ngram_pix_coords[1]), f"{iou:.2f}", fill="black")

            # Combine image and overlay
            img = Image.alpha_composite(img, overlay)
            img = img.convert("RGB")  # Convert back for PhotoImage if needed

            # --- Resize for display ---
            img.thumbnail(
                (THUMBNAIL_WIDTH, THUMBNAIL_WIDTH * img_h // img_w),
                Image.Resampling.LANCZOS,
            )

            return ImageTk.PhotoImage(img)

        except FileNotFoundError:
            print(f"Error: Image file not found at {image_path}")
            # Return a placeholder? For now, return None
            return None
        except Exception as e:
            print(f"Error creating visualization for {image_path}: {e}")
            return None

    def perform_search(self) -> None:
        if self.inverted_index is None:
            messagebox.showerror("Error", "Index is not loaded.")
            return

        query_text = self.query_text_entry.get().strip()
        if not query_text:
            messagebox.showwarning("Input Missing", "Please enter query text.")
            return

        # Validate and get ranges
        top_range = self.parse_percent_range(self.top_start_entry, self.top_end_entry)
        if top_range == (None, None) and (
            self.top_start_entry.get().strip() or self.top_end_entry.get().strip()
        ):
            return  # Error occurred

        left_range = self.parse_percent_range(
            self.left_start_entry, self.left_end_entry
        )
        if left_range == (None, None) and (
            self.left_start_entry.get().strip() or self.left_end_entry.get().strip()
        ):
            return  # Error occurred

        region_str = self.construct_region_string(top_range, left_range)
        print(f"[DEBUG GUI] Constructed region string: {region_str}")  # DEBUG PRINT 1

        self.set_status(
            f"Parsing query and region '{region_str if region_str else 'Full Image'}'..."
        )
        query_ngrams, query_bbox_norm = query_parser.parse_query(query_text, region_str)
        print(f"[DEBUG GUI] Parsed query bbox norm: {query_bbox_norm}")  # DEBUG PRINT 2

        if not query_ngrams:
            self.set_status("Query parsing resulted in no n-grams.")
            messagebox.showwarning(
                "Parsing Error", "Could not generate any n-grams from the query text."
            )
            return

        self.set_status(f"Searching index for {len(query_ngrams)} n-grams...")
        image_scores = searcher.search_images(
            query_ngrams, query_bbox_norm, self.inverted_index
        )
        ranked_results = searcher.rank_results(image_scores)
        self.set_status(
            f"Search complete. Found {len(ranked_results)} results. Displaying top..."
        )

        # --- Clear previous results ---
        for widget in self.results_frame.winfo_children():
            widget.destroy()

        # Clear old image references
        self.image_references.clear()

        if not ranked_results:
            ttk.Label(self.results_frame, text="No results found.").grid(
                row=0, column=0, padx=10, pady=10
            )
            self.on_frame_configure()  # Update scroll region even if empty
            return

        # --- Gather details and display results ---
        num_results_to_display = 15  # Or make this configurable
        row, col = 0, 0
        for i, (image_id, score) in enumerate(ranked_results[:num_results_to_display]):
            # Gather details needed for visualization
            found_details: List[Dict[str, Any]] = []
            # --- Gather details for ALL query ngrams found in this ranked image ---
            for ngram in query_ngrams:
                if ngram in self.inverted_index:
                    for found_id, found_bbox_norm in self.inverted_index[ngram]:
                        if found_id == image_id:
                            # Calculate IoU for this specific occurrence
                            iou = utils.calculate_iou(query_bbox_norm, found_bbox_norm)
                            # Append details regardless of IoU value
                            found_details.append(
                                {
                                    "text": ngram,
                                    "bbox_norm": found_bbox_norm,
                                    "iou": iou,  # Will be 0 if no overlap
                                }
                            )

            image_path = os.path.join(config.IMAGE_DIR, f"{image_id}.png")
            print(
                f"[DEBUG GUI] Visualizing image {image_id} with query_bbox_norm: {query_bbox_norm}"
            )  # DEBUG PRINT 3
            vis_photo = self.create_visualization(
                image_path, query_bbox_norm, found_details
            )

            # Create frame for this result
            result_cell = ttk.Frame(
                self.results_frame, padding="10"
            )  # Increased padding, removed border
            result_cell.grid(
                row=row, column=col, padx=10, pady=10, sticky=tk.NSEW
            )  # Increased padding

            if vis_photo:
                img_label = ttk.Label(result_cell, image=vis_photo)
                # Keep reference to avoid garbage collection
                self.image_references.append(vis_photo)
                img_label.pack(pady=5)
            else:
                ttk.Label(result_cell, text="[Image Error]").pack(pady=5)

            # Add Rank Label
            ttk.Label(
                result_cell, text=f"Rank: {i + 1}", font=("TkDefaultFont", 10, "bold")
            ).pack()
            ttk.Label(result_cell, text=f"ID: {image_id}").pack()
            ttk.Label(result_cell, text=f"Score: {score:.4f}").pack()

            # Update grid position
            col += 1
            if col >= RESULTS_PER_ROW:
                col = 0
                row += 1

        # Make result cells resize width with window
        for c in range(RESULTS_PER_ROW):
            self.results_frame.columnconfigure(c, weight=1)

        # Update scroll region after adding all widgets
        self.root.update_idletasks()
        self.on_frame_configure()


# --- Main ---
if __name__ == "__main__":
    root = tk.Tk()
    app = SearchVisualizerApp(root)
    root.mainloop()

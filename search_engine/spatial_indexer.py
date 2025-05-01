"""
Spatial Indexer Module for SATIAS

This module implements various spatial indexing structures to improve
search efficiency and potentially accuracy in the SATIAS system.

Implemented structures:
1. R-tree based indexing (using rtree package)
2. Quadtree based indexing
3. Grid-based spatial indexing
"""

import sys
import time
import json
import pickle
import os
from collections import defaultdict
from typing import List, Dict, Tuple, Any, Optional, Set, DefaultDict, Union

# Adjust path to import modules from project root
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import from project modules  
import config
from search_engine import utils
from search_engine import query_parser

# Try importing rtree and set a flag
try:
    from rtree import index
    RTREE_AVAILABLE = True
except ImportError:
    RTREE_AVAILABLE = False
    index = None  # To satisfy linters

# Define type hints for clarity
StandardIndexType = Dict[str, List[Tuple[str, List[float]]]]
BBoxType = List[float]  # [top, left, bottom, right]
# Define node types for spatial indices
QuadNodeType = Tuple[int, int]  # (row, col) in [0,1] for each dimension
GridCellType = Tuple[int, int]  # (row, col) in [0,grid_size-1] range

# Type definitions for different index structures
QuadTreeIndexType = DefaultDict[str, DefaultDict[QuadNodeType, DefaultDict[str, List[BBoxType]]]]
GridIndexType = DefaultDict[str, DefaultDict[GridCellType, DefaultDict[str, List[BBoxType]]]]
MetadataItemType = Dict[str, Any]

# Helper functions for defaultdict factories (pickle-friendly)
def _create_image_bbox_dict():
    return defaultdict(list)

def _create_node_image_dict():
    return defaultdict(_create_image_bbox_dict)

def _create_ngram_node_dict():
    return defaultdict(_create_node_image_dict)

class SpatialIndexer:
    """Base class for spatial indexing implementations."""
    
    def __init__(self, index_type: str = "standard", iou_weight: float = config.IOU_WEIGHT, proximity_weight: float = config.PROXIMITY_WEIGHT):
        """Initialize the spatial indexer.
        
        Args:
            index_type: Type of spatial index to use ('standard', 'rtree', 'quadtree', 'grid')
            iou_weight: Weight for IoU score in final scoring
            proximity_weight: Weight for proximity score in final scoring
        """
        self.index_type: str = index_type
        self.iou_weight: float = iou_weight
        self.proximity_weight: float = proximity_weight
        self.index: Optional[StandardIndexType] = None # Base ngram index {ngram -> [(img_id, bbox), ...]}
        self.index_size: int = 0
        
        # Attributes specific to index types
        if RTREE_AVAILABLE:
            self.rtree_index: Optional[index.Index] = None
            # Store mapping from rtree item id to (img_id, ngram, bbox_norm)
            self.rtree_data: Dict[int, Tuple[str, str, BBoxType]] = {}
        self.quad_indices: Optional[QuadTreeIndexType] = None
        self.grid_indices: Optional[GridIndexType] = None
        self.grid_size: int = 10 # Default grid size
        
        # Build time tracking
        self.build_time: float = 0.0
        
    def build_index(self, metadata_path: str = config.IMAGE_METADATA_FILE) -> Optional[StandardIndexType]:
        """Builds the specified spatial index.
        
        Args:
            metadata_path: Path to the image metadata JSON file.
            
        Returns:
            The built index structure, or None if error.
        """
        # Load the image metadata directly
        try:
            with open(metadata_path, "r") as f:
                all_image_metadata: List[MetadataItemType] = json.load(f)
        except FileNotFoundError:
            return None
        except json.JSONDecodeError:
            return None
            
        # 1. Always build the standard index first, as others might rely on it
        self.index = self._build_standard_index(all_image_metadata)
        if not self.index:
             return None
             
        # 2. Build the specific spatial index structure if not 'standard'
        if self.index_type == "rtree":
            if not RTREE_AVAILABLE:
                # Still return the standard index
                pass
            else:
                # R-tree build uses data derived during standard index build
                self._build_rtree_index(all_image_metadata) 
        elif self.index_type == "quadtree":
            self.quad_indices = self._build_quadtree_index(all_image_metadata)
            if not self.quad_indices:
                # Still return the standard index
                pass
        elif self.index_type == "grid":
            self.grid_indices = self._build_grid_index(all_image_metadata)
            if not self.grid_indices:
                # Still return the standard index
                pass
        elif self.index_type != "standard":
            # Still return the standard index
            pass
        # Always return the standard index dictionary
        return self.index
    
    def _build_standard_index(self, all_image_metadata: List[MetadataItemType]) -> StandardIndexType:
        """Build a standard inverted index (same as original)."""
        inverted_index: StandardIndexType = defaultdict(list)
        images_processed = 0
        ngrams_indexed = 0
        
        for image_meta in all_image_metadata:
            image_id = image_meta.get("image_id")
            width = image_meta.get("width")
            height = image_meta.get("height")
            ngrams = image_meta.get("ngrams", [])
            
            if not all([image_id, width, height]):
                continue
                
            for ngram_data in ngrams:
                ngram_text = ngram_data.get("text")
                bbox_pixels = ngram_data.get("bbox_pixels")
                
                if not ngram_text or bbox_pixels is None:
                    continue
                    
                # Normalize bbox
                normalized_bbox = utils.normalize_bbox(bbox_pixels, width, height)
                
                # Add to inverted index
                inverted_index[ngram_text].append((image_id, normalized_bbox))
                ngrams_indexed += 1
                
            images_processed += 1
            
        self.index_size = sys.getsizeof(pickle.dumps(inverted_index))
        return inverted_index
    
    def _build_rtree_index(self, all_image_metadata: List[MetadataItemType]) -> None:
        """Builds an R-tree index over all n-gram bounding boxes.
           This method assumes self.index (standard index) is already populated.
        """
        if not RTREE_AVAILABLE:
            return
        if not self.index:
             return

        # Items for R-tree: list of (id, bbox_tuple, obj) tuples
        items_for_rtree = []
        # Dictionary to map rtree item ids to associated data
        self.rtree_data = {}
        rtree_item_id = 0
        bbox_count = 0

        # Iterate through the already built standard index to get items for R-tree
        for ngram, entries in self.index.items():
            for img_id, bbox_norm in entries:
                 # R-tree expects (left, bottom, right, top) -> or (left, top, right, bottom)
                 # bbox_norm is [top, left, bottom, right]
                 rtree_bbox_tuple = (bbox_norm[1], bbox_norm[0], bbox_norm[3], bbox_norm[2]) 
                 
                 # Add item for R-tree construction: (id, bbox_tuple, obj=None)
                 items_for_rtree.append((rtree_item_id, rtree_bbox_tuple, None))
                 
                 # Store associated data mapped by rtree_item_id
                 self.rtree_data[rtree_item_id] = (img_id, ngram, bbox_norm)
                 
                 rtree_item_id += 1
                 bbox_count += 1

        if not items_for_rtree:
            self.rtree_index = None
            return
            
        # Create the R-tree index using the collected items
        p = index.Property()
        p.dimension = 2 # 2D spatial data
        try:
            # Pass the list directly instead of using a generator
            self.rtree_index = index.Index(items_for_rtree, properties=p)
            
            # Update total size (Add R-tree index size + R-tree data map size)
            rtree_index_size = sys.getsizeof(self.rtree_index) if self.rtree_index else 0
            rtree_data_size = sys.getsizeof(pickle.dumps(self.rtree_data))
            self.index_size += (rtree_index_size + rtree_data_size)
        except Exception as e:
            self.rtree_index = None
            self.rtree_data = {}
        
    def _build_quadtree_index(self, all_image_metadata: List[MetadataItemType]) -> QuadTreeIndexType:
        """Build a Quadtree-based spatial index.
        
        Since we don't have a direct quadtree library, this is a simple implementation.
        Index structure: {ngram -> {quadrant -> {image_id -> [bbox, ...]}}}
        Quadrants: NW (0,0), NE (0,1), SW (1,0), SE (1,1)
        """
        # Use named functions for defaultdict factories
        quad_indices: QuadTreeIndexType = _create_ngram_node_dict()
        
        images_processed = 0
        ngrams_indexed = 0
        
        for image_data in all_image_metadata:
            image_id = image_data.get("image_id")
            width = image_data.get("width")
            height = image_data.get("height")
            ngrams = image_data.get("ngrams", [])
            
            if not all([image_id, width, height]):
                continue
                
            images_processed += 1
            
            for ngram_data in ngrams:
                ngram_text = ngram_data.get("text")  # Use 'text' key
                bbox_pixels = ngram_data.get("bbox_pixels")
                
                if not ngram_text or bbox_pixels is None:
                    continue
                    
                # Normalize bbox
                bbox_norm = utils.normalize_bbox(bbox_pixels, width, height)
                if not bbox_norm:
                    continue
                
                # Determine which quadrant(s) this bbox belongs to
                quadrants = self._get_quadtree_nodes(bbox_norm)
                
                for quadrant in quadrants:
                    # Store the bbox in appropriate quadrant
                    quad_indices[ngram_text][quadrant][image_id].append(bbox_norm)
                
                ngrams_indexed += 1
        
        # Convert back to regular dict for pickling if needed, though pickle should handle defaultdict with named functions
        self.index_size = sys.getsizeof(pickle.dumps(dict(quad_indices))) 
        return quad_indices

    def _build_grid_index(self, all_image_metadata: List[MetadataItemType], grid_size: int = 10) -> GridIndexType:
        """Build a grid-based spatial index.
        
        Index structure: {ngram -> {grid_cell -> {image_id -> [bbox, ...]}}}
        
        Args:
            all_image_metadata: List of image metadata dictionaries
            grid_size: Number of cells in each dimension (default 10x10 grid)
        """
        self.grid_size = grid_size
        
        # Use named functions for defaultdict factories
        grid_indices: GridIndexType = _create_ngram_node_dict()
        
        images_processed = 0
        ngrams_indexed = 0
        
        for image_data in all_image_metadata:
            image_id = image_data.get("image_id")
            width = image_data.get("width")
            height = image_data.get("height")
            ngrams = image_data.get("ngrams", [])
            
            if not all([image_id, width, height]):
                continue
                
            images_processed += 1
            
            for ngram_data in ngrams:
                ngram_text = ngram_data.get("text") # Use 'text' key
                bbox_pixels = ngram_data.get("bbox_pixels")
                
                if not ngram_text or bbox_pixels is None:
                    continue
                    
                # Normalize bbox
                bbox_norm = utils.normalize_bbox(bbox_pixels, width, height)
                if not bbox_norm:
                    continue
                
                # Determine which grid cell(s) this bbox belongs to
                cells = self._get_grid_cells(bbox_norm)
                
                for cell in cells:
                    # Store the bbox in appropriate cell
                    grid_indices[ngram_text][cell][image_id].append(bbox_norm)
                
                ngrams_indexed += 1
        
        # Convert back to regular dict for pickling if needed
        self.index_size = sys.getsizeof(pickle.dumps(dict(grid_indices))) 
        return grid_indices
        
    def _search_rtree(self, query_ngrams: List[str], query_bbox_norm: List[float]) -> Dict[str, float]:
        """Search using R-tree for spatial filtering, then score."""
        # Checks already performed in search()
        if not self.rtree_index or not self.rtree_data:
             return {}

        # Use R-tree to find spatially relevant bounding box IDs
        # Query bbox format for rtree: (left, top, right, bottom)
        left = query_bbox_norm[1]
        top = query_bbox_norm[0]
        right = query_bbox_norm[3]
        bottom = query_bbox_norm[2]
        epsilon = 1e-3
        # Expand degenerate rectangles
        if left == right:
            left = max(0.0, left - epsilon)
            right = min(100.0, right + epsilon)
        if top == bottom:
            top = max(0.0, top - epsilon)
            bottom = min(100.0, bottom + epsilon)
        query_rect = (left, top, right, bottom)
        
        intersecting_item_ids = []
        try:
            # Get IDs of items intersecting the query rectangle
            intersecting_item_ids = list(self.rtree_index.intersection(query_rect))
        except Exception as e:
            return self._search_standard(query_ngrams, query_bbox_norm)
            
        if not intersecting_item_ids:
            # No spatial overlap found by R-tree
            return {}
        
        # Filter these spatial matches by the query n-grams and score
        image_scores: Dict[str, float] = defaultdict(float)
        match_counts: Dict[str, int] = defaultdict(int)
        query_ngram_set = set(query_ngrams)
        
        for item_id in intersecting_item_ids:
            # Retrieve the data associated with this R-tree item ID
            if item_id not in self.rtree_data:
                 # This shouldn't happen if built correctly, but good to check
                 continue
             
            img_id, ngram, bbox = self.rtree_data[item_id]
            
            # Check if the ngram of this spatial match is in our query n-grams
            if ngram in query_ngram_set:
                # Calculate spatial score
                iou = utils.calculate_iou(query_bbox_norm, bbox)
                proximity = utils.calculate_proximity_score(query_bbox_norm, bbox)
                score = (self.iou_weight * iou) + (self.proximity_weight * proximity)
                
                # Accumulate score and count matches per image
                image_scores[img_id] += score
                match_counts[img_id] += 1
        
        # Normalize scores
        final_scores = {}
        for img_id, total_score in image_scores.items():
            count = match_counts[img_id]
            if count > 0:
                final_scores[img_id] = total_score / count
                
        return final_scores
        
    def _search_standard(self, query_ngrams: List[str], query_bbox_norm: List[float]) -> Dict[str, float]:
        """Standard search implementation (same as original)."""
        # Check if index was built
        if not self.index:
            return {}
        
        # Directly use the standard inverted index format from build_standard_index
        # which is {ngram -> [(image_id, bbox), ...]}
        image_scores: Dict[str, float] = defaultdict(float)
        
        for ngram in query_ngrams:
            if ngram not in self.index:
                continue
                
            # The index structure is a list of (image_id, bbox) tuples
            entries = self.index[ngram]
            
            # Track how many matches we found for this n-gram
            matches_found = 0
            
            for entry in entries:
                # Extract image_id and bbox from the entry
                if len(entry) != 2:  # Ensure entry has expected format
                    continue
                    
                img_id, bbox = entry
                
                # Calculate IoU (Intersection over Union)
                iou = utils.calculate_iou(query_bbox_norm, bbox)
                
                # Calculate proximity
                proximity = utils.calculate_proximity_score(query_bbox_norm, bbox)
                
                # Weighted score (using config weights)
                score = (self.iou_weight * iou) + (self.proximity_weight * proximity)
                
                # Accumulate score for this image
                image_scores[img_id] += score
                matches_found += 1
            
            if matches_found > 0:
                pass
            
        # Normalize scores by number of matches
        # This is a simplified version for initial testing
        return dict(image_scores)
        
    def _search_quadtree(self, query_ngrams: List[str], query_bbox_norm: List[float]) -> Dict[str, float]:
        """Search using quadtree spatial index."""
        # Check if index was built
        if not self.index or not self.quad_indices:
            return {}
            
        # Get image IDs from standard index for the query n-grams
        candidate_image_ids: Set[str] = set()
        for ngram in query_ngrams:
            if ngram in self.index:
                # For list-based index structure, extract image IDs from tuples
                for img_id, _ in self.index[ngram]:
                    candidate_image_ids.add(img_id)
                
        if not candidate_image_ids:
            return {}
        
        # Find quadtree nodes for query bbox
        query_nodes = self._get_quadtree_nodes(query_bbox_norm)
        
        if not query_nodes:
            # Fall back to standard search if no quadtree nodes
            return self._search_standard(query_ngrams, query_bbox_norm)
        
        # Find candidate matches from quadtree
        candidates: Dict[str, List[List[float]]] = defaultdict(list)
        
        for ngram in query_ngrams:
            if ngram not in self.quad_indices:
                continue
                
            for node in query_nodes:
                if node not in self.quad_indices[ngram]:
                    continue
                    
                for img_id, bboxes in self.quad_indices[ngram][node].items():
                    if img_id in candidate_image_ids:
                        candidates[img_id].extend(bboxes)
        
        if not candidates:
            return self._search_standard(query_ngrams, query_bbox_norm)
            
        # Score candidates
        image_scores: Dict[str, float] = {}
        match_counts: Dict[str, int] = defaultdict(int)
        
        for img_id, bboxes in candidates.items():
            total_score: float = 0.0
            
            for bbox in bboxes:
                # Calculate IoU
                iou = utils.calculate_iou(query_bbox_norm, bbox)
                
                # Calculate proximity
                proximity = utils.calculate_proximity_score(query_bbox_norm, bbox)
                
                # Weighted score
                score = (self.iou_weight * iou) + (self.proximity_weight * proximity)
                total_score += score
                match_counts[img_id] += 1
            
            # Only add score if we found matches
            if match_counts[img_id] > 0:
                image_scores[img_id] = total_score / match_counts[img_id]
        
        return image_scores
        
    def _search_grid(self, query_ngrams: List[str], query_bbox_norm: List[float]) -> Dict[str, float]:
        """Search using grid-based spatial index."""
        # Check if index was built
        if not self.index or not self.grid_indices:
            return {}
            
        # Get image IDs from standard index for the query n-grams
        candidate_image_ids: Set[str] = set()
        for ngram in query_ngrams:
            if ngram in self.index:
                # For list-based index structure, extract image IDs from tuples
                for img_id, _ in self.index[ngram]:
                    candidate_image_ids.add(img_id)
                
        if not candidate_image_ids:
            return {}
        
        # Find grid cells for query bbox
        query_cells = self._get_grid_cells(query_bbox_norm)
        
        if not query_cells:
            # Fall back to standard search if no grid cells
            return self._search_standard(query_ngrams, query_bbox_norm)
        
        # Find candidate matches from grid
        candidates: Dict[str, List[List[float]]] = defaultdict(list)
        
        for ngram in query_ngrams:
            if ngram not in self.grid_indices:
                continue
                
            for cell in query_cells:
                if cell not in self.grid_indices[ngram]:
                    continue
                    
                for img_id, bboxes in self.grid_indices[ngram][cell].items():
                    if img_id in candidate_image_ids:
                        candidates[img_id].extend(bboxes)
        
        if not candidates:
            return self._search_standard(query_ngrams, query_bbox_norm)
            
        # Score candidates
        image_scores: Dict[str, float] = {}
        match_counts: Dict[str, int] = defaultdict(int)
        
        for img_id, bboxes in candidates.items():
            total_score: float = 0.0
            
            for bbox in bboxes:
                # Calculate IoU
                iou = utils.calculate_iou(query_bbox_norm, bbox)
                
                # Calculate proximity
                proximity = utils.calculate_proximity_score(query_bbox_norm, bbox)
                
                # Weighted score
                score = (self.iou_weight * iou) + (self.proximity_weight * proximity)
                total_score += score
                match_counts[img_id] += 1
            
            # Only add score if we found matches
            if match_counts[img_id] > 0:
                image_scores[img_id] = total_score / match_counts[img_id]
        
        return image_scores

    def save_index(self, output_path: str) -> None:
        """Save the index to a file."""
        if self.index is None:
            return
            
        try:
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "wb") as f:
                pickle.dump((self.index_type, self.index), f)
        except Exception as e:
            print(f"Error saving index: {e}")
    
    def load_index(self, input_path: str) -> bool:
        """Load the index from a file."""
        try:
            with open(input_path, "rb") as f:
                self.index_type, self.index = pickle.load(f)
            return True
        except Exception as e:
            print(f"Error loading index: {e}")
            return False
            
    def search(self, query_ngrams: List[str], query_bbox_norm: List[float]) -> DefaultDict[str, float]:
        """Search the index for matches to the query.
        
        Args:
            query_ngrams: List of query n-grams
            query_bbox_norm: Normalized query bounding box [top, left, bottom, right]
            
        Returns:
            Dictionary mapping image_id to score
        """
        if self.index is None:
            return defaultdict(float)
            
        # Dispatch to appropriate search method
        if self.index_type == "standard":
            return self._search_standard(query_ngrams, query_bbox_norm)
        elif self.index_type == "rtree":
            return self._search_rtree(query_ngrams, query_bbox_norm)
        elif self.index_type == "quadtree":
            return self._search_quadtree(query_ngrams, query_bbox_norm)
        elif self.index_type == "grid":
            return self._search_grid(query_ngrams, query_bbox_norm)
        else:
            return defaultdict(float)
    
    def _get_quadtree_nodes(self, bbox_norm: List[float]) -> List[Tuple[int, int]]:
        """Determine the quadtree node(s) a bounding box belongs to.
        
        Args:
            bbox_norm: Normalized bounding box [top, left, bottom, right]
            
        Returns:
            List of quadtree node tuples (row, col) where each dimension is 0 or 1
        """
        # Simple implementation: divide the space into 4 quadrants
        # and return nodes based on center point
        center_x = (bbox_norm[1] + bbox_norm[3]) / 2  # Center X coordinate
        center_y = (bbox_norm[0] + bbox_norm[2]) / 2  # Center Y coordinate
        
        # Determine quadrant: (0,0) top-left, (0,1) top-right, (1,0) bottom-left, (1,1) bottom-right
        # Use 50.0 threshold as coordinates are normalized to 0-100
        row = 1 if center_y >= 50.0 else 0
        col = 1 if center_x >= 50.0 else 0
        
        # Return as list of tuples (could be extended to return multiple overlapping nodes)
        return [(row, col)]
    
    def _get_grid_cells(self, bbox_norm: List[float]) -> List[Tuple[int, int]]:
        """Determine the grid cell(s) a bounding box belongs to.
        
        Args:
            bbox_norm: Normalized bounding box [top, left, bottom, right]
            
        Returns:
            List of grid cell tuples (row, col) in range [0, grid_size-1]
        """
        # Get center point of the bbox
        center_x = (bbox_norm[1] + bbox_norm[3]) / 2  # Center X coordinate
        center_y = (bbox_norm[0] + bbox_norm[2]) / 2  # Center Y coordinate
        
        # Convert to grid coordinates (0 to grid_size-1)
        # Use 50.0 threshold as coordinates are normalized to 0-100
        col = min(int(center_x / (100.0 / self.grid_size)), self.grid_size - 1)
        row = min(int(center_y / (100.0 / self.grid_size)), self.grid_size - 1)
        
        # Return as list of tuples (could be extended to return multiple overlapping cells)
        return [(row, col)]

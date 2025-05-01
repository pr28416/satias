#!/usr/bin/env python3
"""
Benchmark different spatial indexing methods for SATIAS.

This script evaluates various spatial indexing structures in terms of:
1. Index building time
2. Index size
3. Query execution time
4. Retrieval accuracy (MAP, P@k)

Usage:
    python benchmark_spatial_indices.py
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import List, Dict, Tuple, Any, Optional

# --- Adjust path to import from parent directory ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import config
from search_engine import utils, query_parser
from search_engine.spatial_indexer import SpatialIndexer

# Check for rtree package
try:
    import rtree
    RTREE_AVAILABLE = True
except ImportError:
    RTREE_AVAILABLE = False
    print("rtree package not available. R-tree indexing will be disabled.")
    print("To install: pip install rtree")

# Default paths
DEFAULT_QUERIES_PATH = os.path.join(config.OUTPUT_DIR, "metadata", "queries.json")
OUTPUT_DIR = os.path.join(parent_dir, "output", "benchmark")


def load_queries(queries_path: str) -> List[Dict]:
    """Load test queries."""
    try:
        with open(queries_path, "r") as f:
            queries = json.load(f)
        print(f"Loaded {len(queries)} queries from {queries_path}")
        return queries
    except Exception as e:
        print(f"Error loading queries: {e}")
        return []


def run_benchmark(index_types: List[str], 
                  metadata_path: str = config.IMAGE_METADATA_FILE,
                  queries_path: str = DEFAULT_QUERIES_PATH,
                  num_queries: int = 100) -> Dict[str, Dict[str, Any]]:
    """Run benchmarks for different index types.
    
    Args:
        index_types: List of index types to benchmark
        metadata_path: Path to image metadata
        queries_path: Path to test queries
        num_queries: Number of queries to test (limit for faster testing)
        
    Returns:
        Dictionary with benchmark results
    """
    # Create results structure
    results: Dict[str, Dict[str, Any]] = {}
    
    # Load test queries
    all_queries = load_queries(queries_path)
    if not all_queries:
        print("Error loading queries. Exiting.")
        return results
        
    # Limit number of queries for testing if specified
    test_queries = all_queries[:num_queries] if num_queries else all_queries
    print(f"Using {len(test_queries)} queries for benchmarking")
    
    # Benchmark each index type
    for index_type in index_types:
        print(f"\n{'='*20} Benchmarking {index_type} indexing {'='*20}")
        
        if index_type == "rtree" and not RTREE_AVAILABLE:
            print("Skipping R-tree benchmark (package not available)")
            continue
            
        results[index_type] = {
            "build_time": 0,
            "query_times": [],
            "index_size": 0,
            "ap_scores": [],
            "precision_at_k": defaultdict(list)
        }
        
        # Initialize indexer
        indexer = SpatialIndexer(index_type=index_type)
        
        # Build and time index construction
        print(f"Building {index_type} index...")
        start_time = time.time()
        index = indexer.build_index(metadata_path)
        build_time = time.time() - start_time
        
        if index is None:
            print(f"Error building {index_type} index")
            continue
            
        results[index_type]["build_time"] = build_time
        results[index_type]["index_size"] = indexer.index_size
        
        print(f"Index built in {build_time:.2f} seconds")
        print(f"Index size: {indexer.index_size / (1024*1024):.2f} MB")
        
        # Run test queries
        print(f"Running {len(test_queries)} test queries...")
        
        total_processed = 0
        successful_queries = 0
        
        for i, query in enumerate(test_queries):
            # Parse query
            query_text = query.get("query_text", "")
            query_region_norm = query.get("query_region_norm", [])
            target_id = query.get("image_id", "")
            
            if not query_text or not target_id:
                print(f"Warning: Skipping query {i+1} due to missing data")
                continue
                
            # Parse query - adapt to match the structure
            query_ngrams, _ = query_parser.parse_query(query_text)
            query_bbox = query_region_norm if query_region_norm else [0, 0, 100, 100]  # Default if missing
            
            if not query_ngrams:
                print(f"Warning: No n-grams found for query {i+1}")
                continue
            
            # Time the query execution
            start_time = time.time()
            image_scores = indexer.search(query_ngrams, query_bbox)
            query_time = time.time() - start_time
            
            # Record query time
            results[index_type]["query_times"].append(query_time)
            
            # Check if we got any results
            if not image_scores:
                print(f"Warning: Query {i+1} returned no results")
                continue
            
            # Convert to ranked list
            ranked_ids = [img_id for img_id, _ in 
                         sorted(image_scores.items(), key=lambda x: x[1], reverse=True)]
            
            if not ranked_ids:
                print(f"Warning: No ranked IDs for query {i+1}")
                continue
                
            # Calculate Average Precision
            ap = calculate_average_precision(ranked_ids, target_id)
            results[index_type]["ap_scores"].append(ap)
            
            # Calculate Precision@k for different k values
            for k in [1, 5, 10]:
                pk = calculate_precision_at_k(ranked_ids, target_id, k)
                results[index_type]["precision_at_k"][k].append(pk)
                
            successful_queries += 1
                
            # Show progress
            if (i+1) % 5 == 0 or i+1 == len(test_queries):
                print(f"Processed {i+1}/{len(test_queries)} queries, successful: {successful_queries}")
                
            total_processed += 1
        
        # Calculate summary metrics - only if we have at least one successful query
        print(f"\nFinal stats: {successful_queries}/{total_processed} successful queries")
        
        if successful_queries > 0:
            if results[index_type]["query_times"]:
                mean_qt = float(np.mean(results[index_type]["query_times"]))
                results[index_type]["mean_query_time"] = mean_qt
                print(f"Mean query time: {mean_qt*1000:.2f} ms")
                
            if results[index_type]["ap_scores"]:
                mean_ap = float(np.mean(results[index_type]["ap_scores"]))
                results[index_type]["mean_ap"] = mean_ap
                print(f"Mean AP: {mean_ap:.4f}")
                
            for k in sorted(results[index_type]["precision_at_k"].keys()):
                if results[index_type]["precision_at_k"][k]:
                    mean_pk = float(np.mean(results[index_type]["precision_at_k"][k]))
                    mean_pk_key = f"mean_p@{k}"
                    results[index_type][mean_pk_key] = mean_pk
                    print(f"Mean P@{k}: {mean_pk:.4f}")
                
        # Debug output to check what's in the results
        print(f"\nDebug: Results for {index_type}:")
        print(f"  Build time: {results[index_type].get('build_time', 'Not set')}")
        print(f"  Mean query time: {results[index_type].get('mean_query_time', 'Not set')}")
        print(f"  Mean AP: {results[index_type].get('mean_ap', 'Not set')}")
        for k in sorted(results[index_type].get("precision_at_k", {}).keys()):
            mean_pk_key = f"mean_p@{k}"
            print(f"  Mean P@{k}: {results[index_type].get(mean_pk_key, 'Not set')}")
    
    # Overall debug output before plotting
    print("\nFinal results structure before plotting:")
    for index_type in results:
        print(f"{index_type}: {list(results[index_type].keys())}")
    
    return results


def calculate_average_precision(ranked_ids: List[str], target_id: str) -> float:
    """Calculate Average Precision (AP) for a single query."""
    if not ranked_ids or target_id not in ranked_ids:
        return 0.0
        
    # Find position of target (0-indexed)
    target_pos = ranked_ids.index(target_id)
    
    # AP = 1/(rank+1) where rank is the position (0-indexed)
    return 1.0 / (target_pos + 1)


def calculate_precision_at_k(ranked_ids: List[str], target_id: str, k: int) -> float:
    """Calculate Precision@k for a single query."""
    if not ranked_ids or k <= 0:
        return 0.0
        
    # Check if target is in top-k results
    top_k = ranked_ids[:k]
    if target_id in top_k:
        return 1.0 / k  # Found in top-k, so precision is 1/k
    
    return 0.0


def plot_results(results: Dict[str, Dict[str, Any]], output_dir: str) -> None:
    """Generate plots of benchmark results."""
    if not results:
        print("No results to plot")
        return
        
    # Create output directory if needed
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract index types with valid results
    index_types = []
    for idx in results:
        if "mean_ap" in results[idx]:
            index_types.append(idx)
    
    if not index_types:
        print("No valid results for plotting (no mean_ap values found)")
        # Debug which keys are available in each result
        for idx in results:
            print(f"{idx} has keys: {list(results[idx].keys())}")
        return
    
    # Prepare data for plotting
    build_times = [results[idx]["build_time"] for idx in index_types]
    query_times = [results[idx]["mean_query_time"] * 1000 for idx in index_types]  # ms
    index_sizes = [results[idx]["index_size"] / (1024*1024) for idx in index_types]  # MB
    mean_aps = [results[idx]["mean_ap"] for idx in index_types]
    
    # Precision@k values (extract all k values that exist)
    k_values = []
    for idx in index_types:
        for k in results[idx]["precision_at_k"]:
            if k not in k_values:
                k_values.append(k)
    k_values.sort()
    
    precision_at_k = {}
    for k in k_values:
        mean_pk_key = f"mean_p@{k}"
        precision_at_k[k] = [results[idx].get(mean_pk_key, 0) for idx in index_types]
    
    # Create figure with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Build Time
    axes[0, 0].bar(index_types, build_times)
    axes[0, 0].set_title('Index Build Time (seconds)')
    axes[0, 0].set_ylabel('Seconds')
    for i, v in enumerate(build_times):
        axes[0, 0].text(i, v + 0.1, f'{v:.2f}s', ha='center')
    
    # 2. Query Time
    axes[0, 1].bar(index_types, query_times)
    axes[0, 1].set_title('Mean Query Time (milliseconds)')
    axes[0, 1].set_ylabel('Milliseconds')
    for i, v in enumerate(query_times):
        axes[0, 1].text(i, v + 0.1, f'{v:.2f}ms', ha='center')
    
    # 3. Mean AP and Precision@k
    ax3 = axes[1, 0]
    width = 0.8 / (1 + len(k_values))
    x = np.arange(len(index_types))
    
    # Plot Mean AP
    ax3.bar(x, mean_aps, width, label='MAP')
    
    # Plot P@k for each k
    for i, k in enumerate(k_values):
        pos = x + width * (i + 1)
        ax3.bar(pos, precision_at_k[k], width, label=f'P@{k}')
    
    ax3.set_title('Retrieval Accuracy')
    ax3.set_xticks(x + width * (len(k_values) / 2))
    ax3.set_xticklabels(index_types)
    ax3.set_ylabel('Score')
    ax3.legend()
    
    # 4. Index Size
    axes[1, 1].bar(index_types, index_sizes)
    axes[1, 1].set_title('Index Size (MB)')
    axes[1, 1].set_ylabel('Size (MB)')
    for i, v in enumerate(index_sizes):
        axes[1, 1].text(i, v + 0.1, f'{v:.2f}MB', ha='center')
    
    # Adjust layout and save
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'benchmark_results.pdf')
    plt.savefig(plot_path)
    plt.savefig(os.path.join(output_dir, 'benchmark_results.png'))
    print(f"Plots saved to {plot_path}")
    
    # Create a table for the paper
    create_latex_table(results, index_types, k_values, os.path.join(output_dir, 'benchmark_table.tex'))


def create_latex_table(results: Dict[str, Dict[str, Any]], 
                       index_types: List[str], 
                       k_values: List[int],
                       output_path: str) -> None:
    """Create a LaTeX table of benchmark results for the paper."""
    with open(output_path, 'w') as f:
        f.write(r'''\begin{table}[htbp]
    \centering
    \caption{Performance Comparison of Different Spatial Indexing Methods}
    \begin{tabular}{|l|''')
        
        # Header row with metrics
        f.write(r'c|c|c|c|')  # Build time, Query time, Size, MAP
        for k in k_values:
            f.write(r'c|')  # P@k
        f.write(r'}' + '\n')
        
        f.write(r'''\hline
    \textbf{Method} & \textbf{Build Time (s)} & \textbf{Query Time (ms)} & \textbf{Size (MB)} & 
    \textbf{MAP} ''')
        
        for k in k_values:
            f.write(f'& \\textbf{{P@{k}}} ')
        f.write(r'\\' + '\n')
        
        f.write(r'\hline' + '\n')
        
        # Data rows
        for idx in index_types:
            row = f'    {idx.capitalize()} & '
            row += f'{results[idx]["build_time"]:.2f} & '
            row += f'{results[idx]["mean_query_time"]*1000:.2f} & '
            row += f'{results[idx]["index_size"]/(1024*1024):.2f} & '
            row += f'{results[idx]["mean_ap"]:.4f} '
            
            for k in k_values:
                mean_pk_key = f"mean_p@{k}"
                row += f'& {results[idx].get(mean_pk_key, 0):.4f} '
            
            row += r'\\'
            f.write(row + '\n')
            
        # Footer
        f.write(r'''\hline
    \end{tabular}
    \label{tab:spatial-index-comparison}
\end{table}''')
    
    print(f"LaTeX table saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark different spatial indexing methods for SATIAS"
    )
    parser.add_argument(
        "--metadata", 
        type=str, 
        default=config.IMAGE_METADATA_FILE,
        help="Path to the image metadata file"
    )
    parser.add_argument(
        "--queries", 
        type=str, 
        default=DEFAULT_QUERIES_PATH,
        help="Path to the test queries file"
    )
    parser.add_argument(
        "--num-queries", 
        type=int, 
        default=500,
        help="Number of queries to test (for faster benchmarking)"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default=OUTPUT_DIR,
        help="Directory to save benchmark results"
    )
    args = parser.parse_args()
    
    # List of index types to benchmark
    index_types = ["standard"]
    if RTREE_AVAILABLE:
        index_types.append("rtree")
    index_types.extend(["quadtree", "grid"])
    
    # Run the benchmark
    results = run_benchmark(
        index_types=index_types,
        metadata_path=args.metadata,
        queries_path=args.queries,
        num_queries=args.num_queries
    )
    
    # Plot the results
    plot_results(results, args.output_dir)


if __name__ == "__main__":
    main()

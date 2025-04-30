#!/usr/bin/env python3
"""
Plot sensitivity analysis results for SATIAS.
This script reads the JSON output from evaluate.py and generates plots
showing the impact of different IoU/Proximity weight combinations on performance metrics.
"""

import argparse
import json
import os
import matplotlib.pyplot as plt
import numpy as np


def load_results(input_file):
    """Load sensitivity analysis results from JSON file."""
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Results file {input_file} not found")
    
    with open(input_file, 'r') as f:
        results = json.load(f)
    
    return results


def plot_sensitivity(results, output_dir, file_format="pdf"):
    """
    Plot sensitivity analysis results.
    
    Args:
        results: Dictionary loaded from JSON with sensitivity analysis data
        output_dir: Directory to save the output plots
        file_format: Output file format (pdf, png, etc.)
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract data for plotting
    iou_weights = [config[0] for config in results["weight_configs"]]
    metrics = results["metrics"]
    
    # Sort data by IoU weight
    sorted_indices = np.argsort(iou_weights)
    iou_weights = [iou_weights[i] for i in sorted_indices]
    metrics = [metrics[i] for i in sorted_indices]
    
    # Extract MAP and P@k values
    map_values = [metric["map"] for metric in metrics]
    p_at_1 = [metric["p@k"].get("1", 0) for metric in metrics]
    p_at_5 = [metric["p@k"].get("5", 0) for metric in metrics]
    p_at_10 = [metric["p@k"].get("10", 0) for metric in metrics]
    
    # Find the optimal configuration (highest MAP)
    optimal_idx = np.argmax(map_values)
    optimal_iou = iou_weights[optimal_idx]
    
    # Create figure with two subplots (MAP and P@k)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: MAP vs IoU Weight
    ax1.plot(iou_weights, map_values, 'o-', linewidth=2, markersize=8)
    ax1.axvline(x=optimal_iou, color='red', linestyle='--', alpha=0.5)
    ax1.set_xlabel('IoU Weight (1-Proximity Weight)')
    ax1.set_ylabel('Mean Average Precision (MAP)')
    ax1.set_title('MAP vs IoU/Proximity Weight Balance')
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Annotate optimal point
    ax1.plot(optimal_iou, map_values[optimal_idx], 'ro', markersize=10)
    ax1.annotate(f'Optimal: ({optimal_iou:.2f}, {map_values[optimal_idx]:.4f})',
                xy=(optimal_iou, map_values[optimal_idx]),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=10)
    
    # Plot 2: P@k vs IoU Weight
    ax2.plot(iou_weights, p_at_1, 'o-', label='P@1', linewidth=2)
    ax2.plot(iou_weights, p_at_5, 's-', label='P@5', linewidth=2)
    ax2.plot(iou_weights, p_at_10, '^-', label='P@10', linewidth=2)
    ax2.axvline(x=optimal_iou, color='red', linestyle='--', alpha=0.5)
    ax2.set_xlabel('IoU Weight (1-Proximity Weight)')
    ax2.set_ylabel('Precision@k')
    ax2.set_title('Precision@k vs IoU/Proximity Weight Balance')
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend()
    
    # Set x-axis ticks to match the evaluated weights
    for ax in [ax1, ax2]:
        ax.set_xticks(iou_weights)
        ax.set_xticklabels([f'{w:.2f}' for w in iou_weights])

    # Adjust layout
    plt.tight_layout()
    
    # Save plot
    output_path = os.path.join(output_dir, f'sensitivity_analysis.{file_format}')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    
    # Create LaTeX figure code for easy inclusion in the paper
    latex_path = os.path.join(output_dir, 'sensitivity_analysis_latex.txt')
    with open(latex_path, 'w') as f:
        f.write(r'''\begin{figure}[htbp]
    \centering
    \includegraphics[width=\textwidth]{figures/sensitivity_analysis.pdf}
    \caption{Sensitivity analysis of IoU and Proximity weight balance in SATIAS. The left plot shows Mean Average Precision (MAP) as a function of IoU weight (with Proximity weight = 1-IoU weight). The right plot shows Precision@k metrics. The optimal performance is achieved with an IoU-weighted balance (IoU: 0.75, Proximity: 0.25), indicating that while both spatial overlap and center proximity are important factors, the degree of overlap (IoU) has slightly more impact on retrieval performance than proximity.}
    \label{fig:sensitivity}
\end{figure}''')
    print(f"LaTeX figure code saved to: {latex_path}")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Plot sensitivity analysis results from SATIAS evaluation"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="evaluation_results.json",
        help="Path to the evaluation results JSON file (default: evaluation_results.json)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="writeup/figures",
        help="Directory to save the output plots (default: writeup/figures)",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="pdf",
        choices=["pdf", "png", "jpg", "svg"],
        help="Output file format (default: pdf)",
    )
    args = parser.parse_args()
    
    # Load results
    results = load_results(args.input)
    
    # Generate plot
    plot_sensitivity(results, args.output_dir, args.format)


if __name__ == "__main__":
    main()

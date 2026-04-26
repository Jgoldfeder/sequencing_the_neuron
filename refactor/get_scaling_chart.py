#!/usr/bin/env python3
"""
Generate a scaling chart from experiment results.

Usage:
    python get_scaling_chart.py <experiment_name>

Output: A table where columns are configurations (layer tuples),
rows are K values (num_samples), and cells show mae, max_ae, and pass/fail.
"""

import argparse
import json
import os
import glob
from collections import defaultdict
from typing import Dict, List, Any, Optional


def parse_run_name(run_name: str) -> Dict[str, Any]:
    """Parse a run directory name to extract parameters."""
    # Format: {layers}_outer-iterations-{oi}_samples-{ns}_epochs-{ne}_dataset-{d}_activation-{a}_seed-{s}_{comment}
    parts = run_name.split("_")

    result = {}

    # Find layers (everything before "outer-iterations")
    layers_parts = []
    i = 0
    while i < len(parts) and not parts[i].startswith("outer-iterations"):
        layers_parts.append(parts[i])
        i += 1
    result["layers"] = "-".join(layers_parts)

    # Parse remaining key-value pairs
    remaining = "_".join(parts[i:])

    for kv in ["outer-iterations", "samples", "epochs", "dataset", "activation", "seed"]:
        if f"{kv}-" in remaining:
            start = remaining.find(f"{kv}-") + len(kv) + 1
            end = remaining.find("_", start)
            if end == -1:
                end = len(remaining)
            result[kv.replace("-", "_")] = remaining[start:end]

    return result


def load_experiment_results(experiment_name: str) -> Dict[str, Dict[int, Dict]]:
    """
    Load all results from an experiment.

    Returns:
        Dict mapping config_key -> {K -> results_dict}
    """
    base_dir = f"./experiments/{experiment_name}/"
    models_dir = base_dir + "models/"

    if not os.path.exists(models_dir):
        print(f"Error: Models directory not found: {models_dir}")
        return {}

    # Find all results.json files
    results_files = glob.glob(models_dir + "*/fnn/results.json")

    # Group results by configuration (layers + dataset)
    results_by_config: Dict[str, Dict[int, Dict]] = defaultdict(dict)

    for results_file in results_files:
        try:
            with open(results_file, "r") as f:
                data = json.load(f)

            args = data.get("args", {})
            layers = args.get("layers", [])
            dataset = args.get("dataset", "unknown")
            num_samples = args.get("num_samples", 0)

            # Create config key
            if isinstance(layers, list):
                layers_str = "-".join(str(l) for l in layers)
            else:
                layers_str = str(layers)

            config_key = f"{layers_str}|{dataset}"

            # Store results
            results_by_config[config_key][num_samples] = {
                "mae": data["best_student"]["mae"],
                "max_ae": data["best_student"]["max_ae"],
                "mse": data["best_student"]["mse"],
            }

        except Exception as e:
            print(f"Warning: Could not load {results_file}: {e}")

    return dict(results_by_config)


def load_threshold(experiment_name: str) -> float:
    """Load the max_ae threshold from scaling_analysis_results.json."""
    results_file = f"./experiments/{experiment_name}/scaling_analysis_results.json"
    if os.path.exists(results_file):
        with open(results_file, "r") as f:
            data = json.load(f)
        return data.get("config", {}).get("max_ae_threshold", 0.001)
    return 0.001  # Default


def format_cell(mae: float, max_ae: float, passed: bool) -> str:
    """Format a cell for display."""
    status = "✓" if passed else "✗"
    return f"mae={mae:.4f} max={max_ae:.4f} {status}"


def print_chart(results: Dict[str, Dict[int, Dict]], threshold: float):
    """Print the scaling chart."""
    if not results:
        print("No results found.")
        return

    # Get all unique K values across all configs
    all_k_values = set()
    for config_results in results.values():
        all_k_values.update(config_results.keys())
    k_values = sorted(all_k_values)

    # Get all config keys
    config_keys = sorted(results.keys())

    # Calculate column widths
    header_width = max(len(ck.split("|")[0]) for ck in config_keys) + 2
    cell_width = 35
    k_col_width = 12

    # Print header
    print("\n" + "=" * 80)
    print(f"SCALING CHART (threshold: max_ae < {threshold})")
    print("=" * 80)

    # Print column headers (configs)
    header_row = f"{'K':<{k_col_width}}"
    for ck in config_keys:
        layers, dataset = ck.split("|")
        header_row += f" | {layers:<{cell_width}}"
    print(header_row)
    print("-" * len(header_row))

    # Print rows (K values)
    for k in k_values:
        row = f"{k:<{k_col_width}}"
        for ck in config_keys:
            if k in results[ck]:
                r = results[ck][k]
                passed = r["max_ae"] < threshold
                cell = format_cell(r["mae"], r["max_ae"], passed)
            else:
                cell = "-"
            row += f" | {cell:<{cell_width}}"
        print(row)

    print("=" * len(header_row))

    # Print summary
    print("\nSUMMARY:")
    print("-" * 60)
    for ck in config_keys:
        layers, dataset = ck.split("|")
        config_results = results[ck]

        # Find minimum K that passed
        passing_ks = [k for k, r in config_results.items() if r["max_ae"] < threshold]
        if passing_ks:
            min_passing_k = min(passing_ks)
            best_result = config_results[min_passing_k]
            print(f"{layers} ({dataset}): SOLVED at K={min_passing_k}, max_ae={best_result['max_ae']:.6f}")
        else:
            # Find best attempt
            if config_results:
                best_k = min(config_results.keys(), key=lambda k: config_results[k]["max_ae"])
                best_result = config_results[best_k]
                print(f"{layers} ({dataset}): UNSOLVED, best max_ae={best_result['max_ae']:.6f} at K={best_k}")
            else:
                print(f"{layers} ({dataset}): NO DATA")


def export_csv(results: Dict[str, Dict[int, Dict]], threshold: float, output_file: str):
    """Export results to CSV."""
    if not results:
        return

    all_k_values = set()
    for config_results in results.values():
        all_k_values.update(config_results.keys())
    k_values = sorted(all_k_values)
    config_keys = sorted(results.keys())

    with open(output_file, "w") as f:
        # Header
        header = ["K"]
        for ck in config_keys:
            layers, dataset = ck.split("|")
            header.extend([f"{layers}_mae", f"{layers}_max_ae", f"{layers}_pass"])
        f.write(",".join(header) + "\n")

        # Rows
        for k in k_values:
            row = [str(k)]
            for ck in config_keys:
                if k in results[ck]:
                    r = results[ck][k]
                    passed = "1" if r["max_ae"] < threshold else "0"
                    row.extend([f"{r['mae']:.6f}", f"{r['max_ae']:.6f}", passed])
                else:
                    row.extend(["", "", ""])
            f.write(",".join(row) + "\n")

    print(f"\nCSV exported to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Generate scaling chart from experiment results")
    parser.add_argument("experiment_name", type=str, help="Name of the experiment")
    parser.add_argument("--csv", type=str, help="Export to CSV file", default=None)
    parser.add_argument("--threshold", type=float, help="Override max_ae threshold", default=None)
    args = parser.parse_args()

    # Load results
    results = load_experiment_results(args.experiment_name)

    if not results:
        print(f"No results found for experiment: {args.experiment_name}")
        return

    # Get threshold
    threshold = args.threshold if args.threshold else load_threshold(args.experiment_name)

    # Print chart
    print_chart(results, threshold)

    # Export CSV if requested
    if args.csv:
        export_csv(results, threshold, args.csv)
    else:
        # Default CSV export
        csv_file = f"./experiments/{args.experiment_name}/scaling_chart.csv"
        export_csv(results, threshold, csv_file)


if __name__ == "__main__":
    main()

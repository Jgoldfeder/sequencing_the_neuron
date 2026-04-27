#!/usr/bin/env python3
"""
Scaling Analysis Script

Runs FNN experiments with increasing num_samples until max_ae falls below threshold.
Tuples are sorted easiest-to-hardest, so failed K values carry over to harder configs.

Usage:
    python scaling_analysis.py configs/scaling_exp_width_001.yaml
"""

import argparse
import subprocess
import sys
import json
import os
import yaml
from dataclasses import dataclass
from typing import List, Dict, Any, Optional


@dataclass
class Config:
    """Experiment configuration loaded from YAML."""
    experiment_name: str
    activation: str
    num_epochs: int
    outer_iterations: int
    seed: int
    max_ae_threshold: float
    initial_k: int
    max_k: int
    configurations: List[Dict[str, Any]]  # List of {layers: [...], dataset: "..."}
    num_gpus: int = 1
    population_size: int = 10

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls(
            experiment_name=data["experiment_name"],
            activation=data["activation"],
            num_epochs=data["num_epochs"],
            outer_iterations=data["outer_iterations"],
            seed=data["seed"],
            max_ae_threshold=data["max_ae_threshold"],
            initial_k=data["initial_k"],
            max_k=data["max_k"],
            configurations=data["configurations"],
            num_gpus=data.get("num_gpus", 1),
            population_size=data.get("population_size", 10),
        )


@dataclass
class RunResult:
    """Result from a single run."""
    success: bool
    max_ae: Optional[float]
    num_samples: int
    layers: List[int]
    dataset: str
    seed: int
    error_msg: Optional[str] = None


def get_results_json_path(cfg: Config, layers: List[int], dataset: str, num_samples: int, comment: str) -> str:
    """Compute the path to results.json matching main.py's naming convention."""
    layers_str = "-".join(str(l) for l in layers)
    name = f"{layers_str}_outer-iterations-{cfg.outer_iterations}_samples-{num_samples}_epochs-{cfg.num_epochs}_dataset-{dataset}_activation-{cfg.activation}_seed-{cfg.seed}_{comment}"
    base_dir = f"./experiments/{cfg.experiment_name}/" if cfg.experiment_name else "./"
    models_path = f"{base_dir}models/{name}/fnn/"
    return models_path + "results.json"


def run_experiment(cfg: Config, layers: List[int], dataset: str, num_samples: int) -> RunResult:
    """Run a single experiment with the given configuration."""

    comment = f"scaling_K{num_samples}"
    results_json_path = get_results_json_path(cfg, layers, dataset, num_samples, comment)

    # Check if results already exist (skip completed runs)
    if os.path.exists(results_json_path):
        try:
            with open(results_json_path, "r") as f:
                results_data = json.load(f)
            max_ae = results_data["best_student"]["max_ae"]
            print(f"\n[SKIP] Already completed: layers={layers}, K={num_samples}, max_ae={max_ae:.6f}")
            return RunResult(
                success=(max_ae < cfg.max_ae_threshold),
                max_ae=max_ae,
                num_samples=num_samples,
                layers=layers,
                dataset=dataset,
                seed=cfg.seed
            )
        except (json.JSONDecodeError, KeyError):
            print(f"\n[RERUN] Corrupted results file, re-running: {results_json_path}")

    cmd = [
        sys.executable, "main.py",
        "--model_type", "fnn",
        "--layers", *[str(l) for l in layers],
        "--dataset", dataset,
        "--seed", str(cfg.seed),
        "--activation", cfg.activation,
        "--num_epochs", str(cfg.num_epochs),
        "--outer_iterations", str(cfg.outer_iterations),
        "--num_samples", str(num_samples),
        "--comment", comment,
        "--experiment_name", cfg.experiment_name,
        "--num_gpus", str(cfg.num_gpus),
        "--population_size", str(cfg.population_size),
    ]

    print(f"\n{'='*60}")
    print(f"Running: layers={layers}, dataset={dataset}, seed={cfg.seed}, K={num_samples}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True
        )

        # Print stderr for visibility (main.py prints progress there)
        if result.stderr:
            print(result.stderr)

        # Read results from JSON file
        if not os.path.exists(results_json_path):
            return RunResult(
                success=False,
                max_ae=None,
                num_samples=num_samples,
                layers=layers,
                dataset=dataset,
                seed=cfg.seed,
                error_msg=f"Results file not found: {results_json_path}"
            )

        with open(results_json_path, "r") as f:
            results_data = json.load(f)

        max_ae = results_data["best_student"]["max_ae"]

        print(f"Result: max_ae = {max_ae:.6f} (threshold: {cfg.max_ae_threshold})")

        return RunResult(
            success=(max_ae < cfg.max_ae_threshold),
            max_ae=max_ae,
            num_samples=num_samples,
            layers=layers,
            dataset=dataset,
            seed=cfg.seed
        )

    except Exception as e:
        return RunResult(
            success=False,
            max_ae=None,
            num_samples=num_samples,
            layers=layers,
            dataset=dataset,
            seed=cfg.seed,
            error_msg=str(e)
        )


def main():
    parser = argparse.ArgumentParser(description="Run scaling analysis experiments")
    parser.add_argument("config", type=str, help="Path to YAML config file")
    args = parser.parse_args()

    # Load config
    cfg = Config.from_yaml(args.config)

    print("="*60)
    print("SCALING ANALYSIS")
    print("="*60)
    print(f"Config file: {args.config}")
    print(f"Experiment: {cfg.experiment_name}")
    print(f"Output dir: ./experiments/{cfg.experiment_name}/")
    print(f"Shared params: activation={cfg.activation}, epochs={cfg.num_epochs}, outer_iter={cfg.outer_iterations}, seed={cfg.seed}")
    print(f"Success threshold: max_ae < {cfg.max_ae_threshold}")
    print(f"Initial K: {cfg.initial_k}")
    print(f"Configurations to test: {len(cfg.configurations)}")
    print("="*60)

    # Track the minimum K that has failed across all configs
    # Since configs are sorted easiest->hardest, if K failed on an easier config,
    # it will likely fail on harder ones too
    min_successful_k_so_far = cfg.initial_k  # Start trying from this K

    results_summary = []

    for config_idx, config_item in enumerate(cfg.configurations):
        layers = config_item["layers"]
        dataset = config_item["dataset"]

        print(f"\n{'#'*60}")
        print(f"# Configuration {config_idx + 1}/{len(cfg.configurations)}")
        print(f"# Layers: {layers}, Dataset: {dataset}")
        print(f"# Starting K: {min_successful_k_so_far} (based on previous results)")
        print(f"{'#'*60}")

        current_k = min_successful_k_so_far
        solved = False
        final_result = None

        while current_k <= cfg.max_k:
            result = run_experiment(cfg, layers, dataset, current_k)
            final_result = result

            if result.success:
                print(f"\n*** SUCCESS: Solved with K={current_k}, max_ae={result.max_ae:.6f} ***")
                solved = True
                break
            else:
                if result.max_ae is not None:
                    print(f"FAILED: max_ae={result.max_ae:.6f} >= {cfg.max_ae_threshold}, doubling K")
                else:
                    print(f"FAILED: {result.error_msg}, doubling K")

                # Update the minimum K for future configs
                min_successful_k_so_far = max(min_successful_k_so_far, current_k * 2)
                current_k *= 2

        if not solved:
            print(f"\n*** GAVE UP: Could not solve with K up to {cfg.max_k} ***")

        results_summary.append({
            "config_idx": config_idx,
            "layers": layers,
            "dataset": dataset,
            "solved": solved,
            "final_k": final_result.num_samples if final_result else None,
            "final_max_ae": final_result.max_ae if final_result else None
        })

    # Print summary
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    print(f"{'Config':<8} {'Layers':<25} {'Dataset':<12} {'Solved':<8} {'K':<10} {'max_ae':<12}")
    print("-"*80)

    for r in results_summary:
        layers_str = str(r['layers'])
        max_ae_str = f"{r['final_max_ae']:.6f}" if r['final_max_ae'] is not None else "N/A"
        k_str = str(r['final_k']) if r['final_k'] is not None else "N/A"
        print(f"{r['config_idx']+1:<8} {layers_str:<25} {r['dataset']:<12} {str(r['solved']):<8} {k_str:<10} {max_ae_str:<12}")

    print("="*60)

    # Save results to file under experiment directory
    base_dir = f"./experiments/{cfg.experiment_name}/" if cfg.experiment_name else "./"
    os.makedirs(base_dir, exist_ok=True)
    results_file = base_dir + "scaling_analysis_results.json"
    with open(results_file, "w") as f:
        json.dump({
            "config": {
                "experiment_name": cfg.experiment_name,
                "activation": cfg.activation,
                "num_epochs": cfg.num_epochs,
                "outer_iterations": cfg.outer_iterations,
                "seed": cfg.seed,
                "max_ae_threshold": cfg.max_ae_threshold,
                "initial_k": cfg.initial_k,
            },
            "results": results_summary
        }, f, indent=2)
    print(f"Results saved to {results_file}")


if __name__ == "__main__":
    main()

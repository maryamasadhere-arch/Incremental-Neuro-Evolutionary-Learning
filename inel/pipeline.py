"""Orchestrates the full experiment: data prep, all three conditions,
metrics, and (optionally) report figures. See cli.py for the entry point."""

from __future__ import annotations

import json
import time
from pathlib import Path

from .config import ExperimentConfig
from .data import load_tasks
from .metrics import compute_metrics, print_summary
from .models.backprop import run_baseline
from .models.ea import run_ea2007
from .models.neat import run_neat


def load_or_run(name: str, fn, path: Path, *args):
    if path.exists():
        print(f"  Loading cached {name} results...")
        with open(path) as f:
            return json.load(f)
    t0 = time.time()
    results = fn(*args)
    print(f"  {name} done in {time.time() - t0:.0f}s")
    with open(path, "w") as f:
        json.dump(results, f)
    return results


def run_experiment(cfg: ExperimentConfig, data_dir: Path, results_dir: Path,
                    make_plots: bool = True) -> dict:
    data_dir.mkdir(exist_ok=True, parents=True)
    results_dir.mkdir(exist_ok=True, parents=True)

    print("=" * 55)
    print("INCREMENTAL NEURO-EVOLUTIONARY LEARNING")
    print(f"Preset: {cfg.name}")
    print("=" * 55)

    print(f"\n[1/5] Preparing Split-MNIST benchmark ({'synthetic' if cfg.data.synthetic else 'real MNIST'})...")
    d = load_tasks(data_dir, cfg.data, cfg.n_tasks)
    tasks = d["tasks"]
    n_in = d["n_dim"]
    print(f"  {len(tasks)} tasks ready, {n_in}-dimensional inputs")

    print("\n[2/5] Running Condition 1: Backpropagation Baseline...")
    base = load_or_run("Baseline", run_baseline, results_dir / "baseline.json",
                        tasks, n_in, cfg.baseline, cfg.n_runs)

    print("\n[3/5] Running Condition 2: 2007 EA Replication...")
    ea = load_or_run("2007 EA", run_ea2007, results_dir / "ea2007.json",
                      tasks, n_in, cfg.ea, cfg.n_runs)

    print("\n[4/5] Running Condition 3: NEAT Extension...")
    neat = load_or_run("NEAT", run_neat, results_dir / "neat.json",
                        tasks, n_in, cfg.neat, cfg.n_runs)

    print("\n[5/5] Computing metrics (RA, FR, FT, EC)...")
    mb = compute_metrics(base, "Baseline (Backprop)", N=cfg.n_tasks,
                          ec_prior_threshold=cfg.ec_prior_threshold,
                          ec_current_threshold=cfg.ec_current_threshold)
    me = compute_metrics(ea, "2007 EA (Replication)", N=cfg.n_tasks,
                          ec_prior_threshold=cfg.ec_prior_threshold,
                          ec_current_threshold=cfg.ec_current_threshold)
    mn = compute_metrics(neat, "NEAT (Extension)", N=cfg.n_tasks,
                          ec_prior_threshold=cfg.ec_prior_threshold,
                          ec_current_threshold=cfg.ec_current_threshold)

    print_summary([mb, me, mn])

    metrics = {"baseline": mb, "ea2007": me, "neat": mn}
    with open(results_dir / "all_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    if make_plots:
        try:
            from .report import make_all_figures
            make_all_figures(base, ea, neat, metrics, results_dir / "figures")
        except ImportError:
            print("\n  (matplotlib not installed - skipping figures; "
                  "`pip install matplotlib` to enable them)")

    print("\n" + "=" * 55)
    print("ALL DONE. Results saved to", results_dir)
    print("=" * 55)
    for p in sorted(results_dir.glob("*.json")):
        print(f"  {p.name}")

    return {"baseline_runs": base, "ea2007_runs": ea, "neat_runs": neat, "metrics": metrics}

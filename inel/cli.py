"""Command-line entry point.

Examples:
    python main.py                  # dev-scale run (real Split-MNIST, few minutes)
    python main.py --quick          # offline smoke test (synthetic data, seconds)
    python main.py --full           # paper-scale run per report Table 3.1 (slow)
    python main.py --runs 5         # override the number of independent runs
    python main.py --no-plots       # skip matplotlib figure generation
"""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

from .config import PRESETS
from .pipeline import run_experiment


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="main.py",
        description="Incremental Neuro-Evolutionary Learning: baseline vs. "
                     "2007 EA replication vs. NEAT extension on Split-MNIST.",
    )
    scale = p.add_mutually_exclusive_group()
    scale.add_argument("--quick", action="store_true",
                        help="tiny, fully offline smoke test (synthetic data)")
    scale.add_argument("--full", action="store_true",
                        help="paper-scale configuration (report Table 3.1); slow")
    p.add_argument("--runs", type=int, default=None,
                    help="override the number of independent runs")
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--results-dir", type=Path, default=Path("results"))
    p.add_argument("--no-plots", action="store_true",
                    help="skip matplotlib figure generation")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    preset_name = "full" if args.full else "quick" if args.quick else "dev"
    cfg = PRESETS[preset_name]()
    if args.runs is not None:
        cfg = dataclasses.replace(cfg, n_runs=args.runs)

    run_experiment(cfg, data_dir=args.data_dir, results_dir=args.results_dir,
                    make_plots=not args.no_plots)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

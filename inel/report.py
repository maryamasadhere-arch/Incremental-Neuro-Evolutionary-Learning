"""
Comparative figures for Objective O6. Requires matplotlib (optional
dependency - the pipeline degrades gracefully without it, see pipeline.py).

Colors are a fixed categorical assignment (one series = one color,
everywhere, never re-cycled) using the first three slots of the project's
validated categorical palette: blue/orange/aqua clear every CVD and
normal-vision separation check pairwise, so Baseline/2007 EA/NEAT stay
distinguishable to colorblind readers across all three figures.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COLORS = {"Baseline": "#2a78d6", "2007 EA": "#eb6834", "NEAT": "#1baf7a"}
CHANCE = 50.0


def _mean_curve(runs: list[list[list[float]]], task_i: int, n_tasks: int) -> list[float | None]:
    """Mean accuracy (%) on `task_i` at each episode, across runs; None
    before the task has been introduced or evaluated."""
    out = []
    for ep in range(n_tasks):
        vals = [r[ep][task_i] * 100 for r in runs if task_i < len(r[ep])]
        out.append(sum(vals) / len(vals) if vals else None)
    return out


def plot_retention_curves(base, ea, neat, n_tasks: int, out: Path) -> None:
    fig, axes = plt.subplots(1, n_tasks, figsize=(3.1 * n_tasks, 3.2), sharey=True)
    if n_tasks == 1:
        axes = [axes]
    episodes = np.arange(1, n_tasks + 1)
    for ti, ax in enumerate(axes):
        for label, runs in (("Baseline", base), ("2007 EA", ea), ("NEAT", neat)):
            curve = _mean_curve(runs, ti, n_tasks)
            xs = [e for e, v in zip(episodes, curve) if v is not None]
            ys = [v for v in curve if v is not None]
            ax.plot(xs, ys, marker="o", ms=4, lw=2, color=COLORS[label], label=label)
        ax.axhline(CHANCE, color="#9a9a94", lw=1, ls="--")
        ax.set_title(f"Task {ti + 1}", fontsize=10)
        ax.set_xlabel("Training episode")
        ax.set_xticks(episodes)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Test accuracy (%)")
    axes[0].set_ylim(40, 102)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 1.06))
    fig.suptitle("Retention accuracy per task across the training sequence", y=1.15, fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_metric_bars(metrics: dict, out: Path) -> None:
    names = ["Baseline", "2007 EA", "NEAT"]
    keys = [m["name"] for m in metrics.values()]
    order = [metrics["baseline"], metrics["ea2007"], metrics["neat"]]

    fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.2))
    specs = [("RA_mean", "Retention Accuracy (%)", 0, 105),
             ("FR_mean", "Forgetting Rate (%)", 0, None),
             ("EC", "Evolvability Ceiling (/5 tasks)", 0, 5.5)]
    for ax, (key, title, ymin, ymax) in zip(axes, specs):
        vals = [m[key] for m in order]
        bars = ax.bar(names, vals, color=[COLORS[n] for n in names], width=0.6)
        ax.set_title(title, fontsize=10)
        ax.set_ylim(ymin, ymax)
        ax.spines[["top", "right"]].set_visible(False)
        ax.bar_label(bars, fmt="%.1f", padding=2, fontsize=9)
    fig.suptitle("Comparative metrics across conditions (Objective O6)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_ra_per_task(metrics: dict, n_tasks: int, out: Path) -> None:
    names = ["Baseline", "2007 EA", "NEAT"]
    order = [metrics["baseline"], metrics["ea2007"], metrics["neat"]]
    x = np.arange(n_tasks)
    width = 0.25

    fig, ax = plt.subplots(figsize=(7, 3.5))
    for i, (name, m) in enumerate(zip(names, order)):
        ax.bar(x + (i - 1) * width, m["RA"], width, label=name, color=COLORS[name])
    ax.axhline(CHANCE, color="#9a9a94", lw=1, ls="--")
    ax.set_xticks(x, [f"Task {i + 1}" for i in range(n_tasks)])
    ax.set_ylabel("Retention accuracy (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Retention accuracy by task (after full sequence)", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def make_all_figures(base, ea, neat, metrics: dict, out_dir: Path) -> None:
    out_dir.mkdir(exist_ok=True, parents=True)
    n_tasks = len(metrics["baseline"]["RA"])
    plot_retention_curves(base, ea, neat, n_tasks, out_dir / "retention_curves.png")
    plot_metric_bars(metrics, out_dir / "metric_comparison.png")
    plot_ra_per_task(metrics, n_tasks, out_dir / "ra_per_task.png")
    print(f"  Figures written to {out_dir}/")

"""
Condition 2 - replication of the 2007 incremental evolutionary model
(report Sec. 3.4, Objective O3).

Fixed-topology feedforward network (n_in -> H -> 1) evolved with a
(mu + lambda) evolutionary strategy and Gaussian mutation. Incremental
learning is achieved purely through the carry-over mechanism: the k
fittest genomes from task i are seeded into task i+1's initial population
and protected from being displaced except by strictly fitter individuals
(mu+lambda selection keeps them unless something else outperforms them).
"""

from __future__ import annotations

import math

import numpy as np

from ..activations import relu, sig
from ..config import EAConfig


def genome_len(n_in: int, n_h: int) -> int:
    return (n_in + 1) * n_h + (n_h + 1)


def eval_population(pop: np.ndarray, X: np.ndarray, y: np.ndarray, n_in: int, n_h: int) -> np.ndarray:
    """Vectorised fitness evaluation of an entire population."""
    W1 = pop[:, :n_in * n_h].reshape(-1, n_in, n_h)
    b1 = pop[:, n_in * n_h:(n_in + 1) * n_h]
    W2 = pop[:, (n_in + 1) * n_h:(n_in + 1) * n_h + n_h]
    b2 = pop[:, -1]
    H = relu(np.einsum("ni,mij->mnj", X, W1) + b1[:, None, :])
    out = sig(np.einsum("mnj,mj->mn", H, W2) + b2[:, None])
    return np.mean((out >= 0.5) == (y[None, :] >= 0.5), axis=1)


def eval_one(g: np.ndarray, X: np.ndarray, y: np.ndarray, n_in: int, n_h: int) -> float:
    W1 = g[:n_in * n_h].reshape(n_in, n_h)
    b1 = g[n_in * n_h:(n_in + 1) * n_h]
    W2 = g[(n_in + 1) * n_h:(n_in + 1) * n_h + n_h]
    b2 = g[-1]
    out = sig(relu(X @ W1 + b1) @ W2 + b2)
    return float(np.mean((out >= 0.5) == (y >= 0.5)))


def mutate(parents: np.ndarray, n_offspring: int, sigma: float, rng: np.random.Generator | None = None) -> np.ndarray:
    """Gaussian mutation (report Sec. 3.4.2): each offspring is produced from
    a single randomly-chosen parent by perturbing every weight with
    independent N(0, sigma^2) noise, i.e. w' = w + N(0, sigma^2). No
    crossover/recombination between two parents is used here - matching the
    2007 model being replicated, which is a (mu+lambda) Evolution Strategy
    (mutation-only reproduction), not a genetic algorithm."""
    draw = np.random.randint if rng is None else rng.integers
    randn = np.random.randn if rng is None else rng.standard_normal
    parent_idx = draw(0, len(parents), n_offspring)
    noise = (randn(n_offspring, parents.shape[1]) * sigma).astype(np.float32)
    return parents[parent_idx] + noise


def ea_task(task_X: np.ndarray, task_y: np.ndarray, carryover: list[np.ndarray],
            seed: int, n_in: int, cfg: EAConfig) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    """(mu+lambda)-ES on one task episode. See module docstring for the
    carry-over mechanism."""
    np.random.seed(seed)
    n_h = cfg.hidden_units
    G = genome_len(n_in, n_h)
    lim = 1.0 / math.sqrt(n_in)
    rows = [np.array(g, dtype=np.float32) for g in carryover]
    while len(rows) < cfg.mu:
        rows.append(np.random.uniform(-lim, lim, G).astype(np.float32))
    pop = np.array(rows[:cfg.mu])

    fits = eval_population(pop, task_X, task_y, n_in, n_h)
    order = np.argsort(-fits)
    pop, fits = pop[order], fits[order]
    n_par = max(1, cfg.mu // 3)

    for gen in range(cfg.n_gen):
        offs = mutate(pop[:n_par], cfg.lam, cfg.sigma)
        of = eval_population(offs, task_X, task_y, n_in, n_h)
        all_p = np.vstack([pop, offs])
        all_f = np.concatenate([fits, of])
        order = np.argsort(-all_f)
        pop, fits = all_p[order[:cfg.mu]], all_f[order[:cfg.mu]]
        if gen == cfg.n_gen - 1:
            print(f"      final best={fits[0] * 100:.1f}% mean={fits.mean() * 100:.1f}%")
    return pop, fits, list(pop[:cfg.carryover_k])


def run_ea2007(tasks: list[dict], n_in: int, cfg: EAConfig, n_runs: int) -> list[list[list[float]]]:
    print("\n" + "=" * 55)
    print("CONDITION 2: 2007 EA REPLICATION")
    print("=" * 55)
    all_runs = []
    for run in range(n_runs):
        np.random.seed(run * 777 + 1)
        mat, co, bgs = [], [], {}
        print(f"Run {run + 1}/{n_runs}")
        for ep, task in enumerate(tasks):
            print(f"  Ep{ep + 1} {task['label']}...", flush=True)
            X = np.array(task["train_x"], dtype=np.float32)
            y = np.array(task["train_y"], dtype=np.float32)
            n = min(cfg.fitness_eval_n, len(X))
            idx = np.random.choice(len(X), n, replace=False)
            pop, fits, co = ea_task(X[idx], y[idx], co, seed=run * 777 + ep, n_in=n_in, cfg=cfg)
            bgs[ep] = pop[0]
            ep_accs = []
            for ti in range(ep + 1):
                tX = np.array(tasks[ti]["test_x"], dtype=np.float32)
                ty = np.array(tasks[ti]["test_y"], dtype=np.float32)
                ep_accs.append(round(eval_one(bgs[ti], tX, ty, n_in, cfg.hidden_units), 4))
            mat.append(ep_accs)
            print(f"  -> {[f'{a * 100:.1f}%' for a in ep_accs]}")
        all_runs.append(mat)
    return all_runs

"""
Condition 1 - backpropagation baseline (report Sec. 3.3, Objective O2).

A feedforward network of arbitrary depth (default 784-256-128-1, matching
the report), trained sequentially task-by-task via mini-batch SGD with
momentum. This condition exists to *demonstrate* catastrophic forgetting,
not to avoid it: no continual-learning safeguards are applied.
"""

from __future__ import annotations

import math
import random

import numpy as np

from ..activations import relu, sig
from ..config import BaselineConfig


class BackpropNet:
    def __init__(self, n_in: int, cfg: BaselineConfig, seed: int = 0):
        r = random.Random(seed)
        sizes = [n_in, *cfg.hidden_layers, 1]
        self.sizes = sizes
        self.W: list[np.ndarray] = []
        self.b: list[np.ndarray] = []
        self.vW: list[np.ndarray] = []
        self.vb: list[np.ndarray] = []
        for fan_in, fan_out in zip(sizes[:-1], sizes[1:]):
            lim = math.sqrt(6 / (fan_in + fan_out))
            W = np.array([[r.uniform(-lim, lim) for _ in range(fan_out)]
                          for _ in range(fan_in)], dtype=np.float32)
            self.W.append(W)
            self.b.append(np.zeros(fan_out, dtype=np.float32))
            self.vW.append(np.zeros_like(W))
            self.vb.append(np.zeros(fan_out, dtype=np.float32))
        self.lr = np.float32(cfg.lr)
        self.mom = np.float32(cfg.momentum)
        self.batch_size = cfg.batch_size

    def forward(self, X: np.ndarray) -> list[np.ndarray]:
        """Returns activations [X, h1, h2, ..., out] for every layer."""
        acts = [X]
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = acts[-1] @ W + b
            is_last = i == len(self.W) - 1
            acts.append(sig(z) if is_last else relu(z))
        return acts

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        out = self.forward(X)[-1].ravel()
        return float(np.mean((out >= 0.5) == (y >= 0.5)))

    def train_epoch(self, X: np.ndarray, y: np.ndarray, batch: int | None = None) -> None:
        batch = batch or self.batch_size
        idx = np.random.permutation(len(X))
        for i in range(0, len(X), batch):
            b_idx = idx[i:i + batch]
            Xb, yb = X[b_idx], y[b_idx]
            acts = self.forward(Xb)
            n = len(b_idx)

            delta = acts[-1].ravel() - yb  # BCE + sigmoid derivative
            delta = delta[:, None]
            for l in range(len(self.W) - 1, -1, -1):
                a_prev = acts[l]
                dW = a_prev.T @ delta / n
                db = delta.mean(axis=0)
                if l > 0:
                    mask = (acts[l] > 0).astype(np.float32)  # ReLU'(z_l) == ReLU'(a_l) for a_l=relu(z_l)
                    delta = (delta @ self.W[l].T) * mask
                self.vW[l] = self.mom * self.vW[l] - self.lr * dW
                self.vb[l] = self.mom * self.vb[l] - self.lr * db
                self.W[l] += self.vW[l]
                self.b[l] += self.vb[l]


def run_baseline(tasks: list[dict], n_in: int, cfg: BaselineConfig, n_runs: int) -> list[list[list[float]]]:
    print("\n" + "=" * 55)
    print("CONDITION 1: BACKPROPAGATION BASELINE")
    print("=" * 55)
    all_runs = []
    for run in range(n_runs):
        np.random.seed(run * 111)
        net = BackpropNet(n_in, cfg, seed=run)
        mat = []
        print(f"Run {run + 1}/{n_runs}")
        for ep, task in enumerate(tasks):
            X = np.array(task["train_x"], dtype=np.float32)
            y = np.array(task["train_y"], dtype=np.float32)
            for _ in range(cfg.epochs_per_task):
                net.train_epoch(X, y)
            ep_accs = []
            for ti in range(ep + 1):
                tX = np.array(tasks[ti]["test_x"], dtype=np.float32)
                ty = np.array(tasks[ti]["test_y"], dtype=np.float32)
                ep_accs.append(round(net.accuracy(tX, ty), 4))
            mat.append(ep_accs)
            print(f"  Ep{ep + 1}: {[f'{a * 100:.1f}%' for a in ep_accs]}")
        all_runs.append(mat)
    return all_runs

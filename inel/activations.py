"""Shared activation functions."""

import numpy as np


def sig(x: np.ndarray) -> np.ndarray:
    # exp() overflows well before 500 for float32 (~88) even though it's
    # safe for float64 (~700); clip per-dtype so large-magnitude inputs
    # saturate to 0/1 silently instead of raising a RuntimeWarning.
    bound = 80.0 if np.asarray(x).dtype == np.float32 else 500.0
    return 1.0 / (1.0 + np.exp(-np.clip(x, -bound, bound)))


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)

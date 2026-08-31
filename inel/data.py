"""
Split-MNIST benchmark construction (report Sec. 2.4, 3.2).

Two data sources are supported:

  * Real MNIST, downloaded from a public mirror and split into 5 sequential
    binary tasks (0v1, 2v3, 4v5, 6v7, 8v9), exactly as specified in the
    report. Optionally random-projected to a lower dimensionality to keep
    the "dev" preset fast (see config.py).

  * A synthetic linearly-separable benchmark with the same task/shape
    contract, used by the "quick" preset and the test suite so that both
    can run fully offline in well under a second.

Network access uses a properly verified TLS context (the original
prototype disabled certificate verification, which is unnecessary here and
is exactly the kind of MITM exposure that shouldn't ship) and retries
transient failures with backoff before giving up with a clear error.
"""

from __future__ import annotations

import array
import gzip
import pickle
import random
import struct
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

from .config import DataConfig

MNIST_BASE_URL = "https://raw.githubusercontent.com/fgnt/mnist/master/"
MNIST_FILES = {
    "train-images": "train-images-idx3-ubyte.gz",
    "train-labels": "train-labels-idx1-ubyte.gz",
    "test-images": "t10k-images-idx3-ubyte.gz",
    "test-labels": "t10k-labels-idx1-ubyte.gz",
}
DIGIT_PAIRS = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]


class DataUnavailableError(RuntimeError):
    """Raised when MNIST cannot be downloaded and no cached copy exists."""


def download_mnist(data_dir: Path, retries: int = 3, timeout: float = 60.0) -> None:
    data_dir.mkdir(exist_ok=True, parents=True)
    for name, fname in MNIST_FILES.items():
        out = data_dir / name
        if out.exists():
            continue
        url = MNIST_BASE_URL + fname
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        last_err: Exception | None = None
        for attempt in range(1, retries + 1):
            try:
                print(f"  Downloading {name} (attempt {attempt}/{retries})...")
                with urllib.request.urlopen(req, timeout=timeout) as r:
                    raw = gzip.decompress(r.read())
                out.write_bytes(raw)
                print(f"  Saved {len(raw):,} bytes")
                break
            except (urllib.error.URLError, TimeoutError, OSError) as e:
                last_err = e
                if attempt < retries:
                    time.sleep(2 ** attempt)
        else:
            raise DataUnavailableError(
                f"Could not download {name} from {url} after {retries} attempts: "
                f"{last_err}. Pass DataConfig(synthetic=True) to run without MNIST."
            ) from last_err


def read_images(path: Path) -> list[list[float]]:
    with open(path, "rb") as f:
        _, n, r, c = struct.unpack(">IIII", f.read(16))
        data = array.array("B", f.read())
    sz = r * c
    return [[data[i * sz + j] / 255.0 for j in range(sz)] for i in range(n)]


def read_labels(path: Path) -> list[int]:
    with open(path, "rb") as f:
        struct.unpack(">II", f.read(8))
        return list(array.array("B", f.read()))


def _project(x_list: list[list[float]], proj: np.ndarray) -> list[list[float]]:
    X = np.array(x_list, dtype=np.float32)
    return (X @ proj).tolist()


def build_split_mnist(data_dir: Path, cfg: DataConfig) -> dict:
    """Build the 5 real Split-MNIST binary tasks, caching the result."""
    cache = data_dir / f"tasks_real_{cfg.project_dims or cfg.n_input_dims}.pkl"
    if cache.exists():
        print("  Loading cached Split-MNIST...")
        with open(cache, "rb") as f:
            return pickle.load(f)

    download_mnist(data_dir)

    print("  Reading raw MNIST...")
    tr_x = read_images(data_dir / "train-images")
    tr_y = read_labels(data_dir / "train-labels")
    te_x = read_images(data_dir / "test-images")
    te_y = read_labels(data_dir / "test-labels")

    n_dim = cfg.project_dims or cfg.n_input_dims
    proj = None
    if cfg.project_dims:
        rng = random.Random(cfg.seed)
        scale = 1.0 / (cfg.project_dims ** 0.5)
        proj = np.array(
            [[rng.gauss(0, scale) for _ in range(cfg.project_dims)] for _ in range(784)],
            dtype=np.float32,
        )

    def project(x_list):
        return _project(x_list, proj) if proj is not None else x_list

    tasks = []
    for a, b in DIGIT_PAIRS:
        tr_pool = [(tr_x[i], 0.0) for i in range(len(tr_x)) if tr_y[i] == a] + \
                  [(tr_x[i], 1.0) for i in range(len(tr_x)) if tr_y[i] == b]
        te_pool = [(te_x[i], 0.0) for i in range(len(te_x)) if te_y[i] == a] + \
                  [(te_x[i], 1.0) for i in range(len(te_x)) if te_y[i] == b]
        rng = random.Random(cfg.seed)
        rng.shuffle(tr_pool)
        rng.shuffle(te_pool)

        tr_x2 = project([s[0] for s in tr_pool])
        tr_y2 = [s[1] for s in tr_pool]
        te_x2 = project([s[0] for s in te_pool])
        te_y2 = [s[1] for s in te_pool]

        tasks.append({
            "label": f"Task({a}v{b})",
            "train_x": tr_x2, "train_y": tr_y2,
            "test_x": te_x2, "test_y": te_y2,
        })
        print(f"  Task({a}v{b}): {len(tr_x2)} train, {len(te_x2)} test")

    d = {"tasks": tasks, "n_dim": n_dim}
    with open(cache, "wb") as f:
        pickle.dump(d, f)
    return d


def build_synthetic_tasks(cfg: DataConfig, n_tasks: int = 5) -> dict:
    """Offline, linearly-separable stand-in for Split-MNIST (quick/test use)."""
    rng = np.random.RandomState(cfg.seed)
    n_dim = cfg.n_input_dims
    tasks = []
    for t in range(n_tasks):
        w = rng.randn(n_dim)
        w /= np.linalg.norm(w)

        def make_split(n):
            X = rng.randn(n, n_dim).astype(np.float32)
            y = (X @ w > 0).astype(np.float32)
            X += rng.normal(0, 0.05, X.shape).astype(np.float32)  # label noise via jitter
            return X.tolist(), y.tolist()

        tr_x, tr_y = make_split(cfg.synthetic_n_train)
        te_x, te_y = make_split(cfg.synthetic_n_test)
        tasks.append({
            "label": f"SyntheticTask{t}",
            "train_x": tr_x, "train_y": tr_y,
            "test_x": te_x, "test_y": te_y,
        })
    return {"tasks": tasks, "n_dim": n_dim}


def load_tasks(data_dir: Path, cfg: DataConfig, n_tasks: int = 5) -> dict:
    if cfg.synthetic:
        return build_synthetic_tasks(cfg, n_tasks)
    return build_split_mnist(data_dir, cfg)

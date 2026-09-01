import numpy as np

from inel.config import BaselineConfig
from inel.models.backprop import BackpropNet


def _linearly_separable(n=200, d=6, seed=0):
    rng = np.random.RandomState(seed)
    w = rng.randn(d)
    X = rng.randn(n, d).astype(np.float32)
    y = (X @ w > 0).astype(np.float32)
    return X, y


def test_forward_shapes_multilayer():
    cfg = BaselineConfig(hidden_layers=(5, 3))
    net = BackpropNet(n_in=4, cfg=cfg, seed=0)
    X = np.random.randn(10, 4).astype(np.float32)
    acts = net.forward(X)
    assert [a.shape[1] for a in acts] == [4, 5, 3, 1]


def test_accuracy_is_a_fraction():
    cfg = BaselineConfig(hidden_layers=(4,))
    net = BackpropNet(n_in=6, cfg=cfg, seed=0)
    X, y = _linearly_separable()
    acc = net.accuracy(X, y)
    assert 0.0 <= acc <= 1.0


def test_training_improves_accuracy_on_separable_data():
    np.random.seed(0)
    cfg = BaselineConfig(hidden_layers=(8,), epochs_per_task=1, lr=0.1, batch_size=32)
    net = BackpropNet(n_in=6, cfg=cfg, seed=1)
    X, y = _linearly_separable(n=400)
    before = net.accuracy(X, y)
    for _ in range(20):
        net.train_epoch(X, y)
    after = net.accuracy(X, y)
    assert after > before
    assert after > 0.8

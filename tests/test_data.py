from inel.config import DataConfig
from inel.data import build_synthetic_tasks, load_tasks


def test_synthetic_tasks_shape_and_determinism():
    cfg = DataConfig(n_input_dims=10, synthetic=True, synthetic_n_train=50,
                      synthetic_n_test=20, seed=7)
    d1 = build_synthetic_tasks(cfg, n_tasks=3)
    d2 = build_synthetic_tasks(cfg, n_tasks=3)

    assert d1["n_dim"] == 10
    assert len(d1["tasks"]) == 3
    t0 = d1["tasks"][0]
    assert len(t0["train_x"]) == 50
    assert len(t0["train_x"][0]) == 10
    assert len(t0["test_x"]) == 20
    assert set(t0["train_y"]) <= {0.0, 1.0}
    assert d1 == d2  # same seed -> identical data, no network access needed


def test_load_tasks_routes_to_synthetic_when_configured(tmp_path):
    cfg = DataConfig(n_input_dims=8, synthetic=True, synthetic_n_train=20, synthetic_n_test=10)
    d = load_tasks(tmp_path, cfg, n_tasks=2)
    assert len(d["tasks"]) == 2
    assert not any(tmp_path.iterdir())  # nothing written to disk for synthetic data

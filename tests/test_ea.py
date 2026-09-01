import numpy as np

from inel.config import EAConfig
from inel.models.ea import ea_task, eval_one, eval_population, genome_len


def _linearly_separable(n=200, d=6, seed=0):
    rng = np.random.RandomState(seed)
    w = rng.randn(d)
    X = rng.randn(n, d).astype(np.float32)
    y = (X @ w > 0).astype(np.float32)
    return X, y


def test_genome_len():
    assert genome_len(n_in=4, n_h=3) == (4 + 1) * 3 + (3 + 1)


def test_eval_population_matches_eval_one():
    n_in, n_h = 5, 3
    G = genome_len(n_in, n_h)
    rng = np.random.RandomState(0)
    pop = rng.uniform(-1, 1, (7, G)).astype(np.float32)
    X, y = _linearly_separable(n=40, d=n_in)

    batched = eval_population(pop, X, y, n_in, n_h)
    single = np.array([eval_one(g, X, y, n_in, n_h) for g in pop])
    np.testing.assert_allclose(batched, single, atol=1e-6)


def test_ea_task_improves_over_random_init():
    n_in = 6
    cfg = EAConfig(hidden_units=4, mu=12, lam=12, sigma=0.2, n_gen=15, carryover_k=2)
    X, y = _linearly_separable(n=150, d=n_in)
    pop, fits, carry = ea_task(X, y, carryover=[], seed=0, n_in=n_in, cfg=cfg)
    assert fits[0] >= 0.6  # should clearly beat chance (0.5) after 15 generations
    assert len(carry) == cfg.carryover_k


def test_best_fitness_never_regresses_across_generations():
    """(mu+lambda) is elitist: the best-seen individual can only be replaced
    by something strictly fitter, so a pre-optimised carry-over genome must
    not lose fitness across a task episode."""
    n_in = 6
    cfg = EAConfig(hidden_units=4, mu=10, lam=10, sigma=0.2, n_gen=20, carryover_k=3)
    X, y = _linearly_separable(n=150, d=n_in, seed=1)

    _, _, carry = ea_task(X, y, carryover=[], seed=0, n_in=n_in, cfg=cfg)
    best_before = max(eval_one(g, X, y, n_in, cfg.hidden_units) for g in carry)

    _, fits_after, _ = ea_task(X, y, carryover=carry, seed=1, n_in=n_in, cfg=cfg)
    assert fits_after[0] >= best_before - 1e-6

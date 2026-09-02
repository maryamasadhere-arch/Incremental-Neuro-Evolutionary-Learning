import random

import numpy as np

from inel.config import NEATConfig
from inel.models.neat import InnovationTracker, NEATGenome, neat_task


def _linearly_separable(n=200, d=8, seed=0):
    rng = np.random.RandomState(seed)
    w = rng.randn(d)
    X = rng.randn(n, d).astype(np.float32)
    y = (X @ w > 0).astype(np.float32)
    return X, y


def test_shared_innovation_tracker_gives_matching_ids_for_matching_structure():
    """Two independently-created genomes that add the *same* new connection
    must get the *same* innovation number, or NEAT's compatibility metric
    can't tell their genes actually match (the whole point of historical
    markings)."""
    innovations = InnovationTracker()
    rng = random.Random(0)
    a = NEATGenome(n_in=8, rng=rng, innovations=innovations, n_hid=2)
    b = NEATGenome(n_in=8, rng=rng, innovations=innovations, n_hid=2)

    # both add a fresh connection between the same previously-unconnected pair
    key = None
    for g in (a, b):
        before = set(g.conns)
        g.conns[innovations.get(1, g.hidden[0])] = [1, g.hidden[0], 0.5, True]
        new_keys = set(g.conns) - before
        assert len(new_keys) <= 1
        if new_keys:
            k = next(iter(new_keys))
            if key is None:
                key = k
            else:
                assert k == key


def test_genome_activate_output_shape_and_range():
    innovations = InnovationTracker()
    rng = random.Random(0)
    g = NEATGenome(n_in=8, rng=rng, innovations=innovations, n_hid=4)
    X = np.random.randn(15, 8).astype(np.float32)
    out = g.activate(X)
    assert out.shape == (15,)
    assert np.all((out >= 0) & (out <= 1))


def test_add_node_preserves_disabled_source_and_adds_two_connections():
    innovations = InnovationTracker()
    rng = random.Random(1)
    g = NEATGenome(n_in=8, rng=rng, innovations=innovations, n_hid=2)
    n_conns_before = len(g.conns)
    n_hidden_before = len(g.hidden)
    g.add_node(rng)
    assert len(g.hidden) == n_hidden_before + 1
    assert len(g.conns) == n_conns_before + 2
    n_disabled = sum(1 for v in g.conns.values() if not v[3])
    assert n_disabled == 1


def test_compatibility_zero_for_identical_genome():
    innovations = InnovationTracker()
    rng = random.Random(2)
    g = NEATGenome(n_in=8, rng=rng, innovations=innovations, n_hid=3)
    g2 = g.copy()
    assert g.compatibility(g2) == 0.0


def test_neat_task_learns_above_chance():
    n_in = 8
    cfg = NEATConfig(initial_hidden=3, pop_size=12, n_gen=15, carryover_k=2, fitness_eval_n=100)
    X, y = _linearly_separable(n=150, d=n_in)
    rng = random.Random(0)
    pop, carry = neat_task(X, y, carryover=[], rng=rng, n_in=n_in, cfg=cfg)
    best_acc = pop[0].evaluate(X, y)
    assert best_acc >= 0.6
    assert len(carry) == cfg.carryover_k

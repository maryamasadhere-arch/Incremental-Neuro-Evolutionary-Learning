"""
Condition 3 - NEAT extension (report Sec. 3.5, Objective O5).

NeuroEvolution of Augmenting Topologies (Stanley & Miikkulainen, 2002):
genomes evolve both connection weights and network topology, historical
markings (innovation numbers) enable meaningful crossover-free structural
comparison, and speciation with explicit fitness sharing protects novel
topologies from premature elimination. The same carry-over mechanism used
in the 2007 EA condition is applied here so the two conditions are
comparable under Objective O6.

Innovation numbers must be assigned consistently across the *whole
population* within a task episode - two genomes that independently evolve
the same structural change need the same innovation number, or
`NEATGenome.compatibility` cannot tell that their genes actually match.
This is done via an explicit `InnovationTracker` passed into every genome,
rather than hidden global/class state, so a tracker's lifetime is scoped
to exactly one task episode (as the report's reset-per-episode semantics
require) and is trivially testable in isolation.

Note on "minimal start" (report Sec. 3.5.1): true NEAT begins with zero
hidden nodes. This implementation starts with a small sparse hidden layer
(``NEATConfig.initial_hidden``) as a practical concession to fitness-
evaluation cost at 784-dimensional input - fully input-connected zero-
hidden genomes would need many generations of add-node mutations before
any hidden representation exists at all. This is a deliberate, documented
deviation, not an oversight.
"""

from __future__ import annotations

import random

import numpy as np

from ..activations import relu, sig
from ..config import NEATConfig


class InnovationTracker:
    """Shared historical-marking registry for one task episode's population."""

    def __init__(self) -> None:
        self._map: dict[tuple[int, int], int] = {}
        self._count = 0

    def get(self, src: int, tgt: int) -> int:
        k = (src, tgt)
        if k not in self._map:
            self._count += 1
            self._map[k] = self._count
        return self._map[k]


class NEATGenome:
    """NEAT genome with variable topology for a fixed input dimensionality."""

    def __init__(self, n_in: int, rng: random.Random | None, innovations: InnovationTracker,
                 n_hid: int = 8, _skip_init: bool = False):
        self.n_in = n_in
        self.innovations = innovations
        self.out = n_in
        self.next_nid = n_in + 1
        self.hidden: list[int] = []
        self.conns: dict[int, list] = {}
        self.fitness = 0.0
        if _skip_init:
            return
        for _ in range(n_hid):
            nid = self.next_nid
            self.next_nid += 1
            self.hidden.append(nid)
        # Sparse initial connections (every 4th input -> hidden)
        for h in self.hidden:
            for i in range(0, n_in, 4):
                self.conns[innovations.get(i, h)] = [i, h, rng.gauss(0, 0.3), True]
            self.conns[innovations.get(-1, h)] = [-1, h, rng.gauss(0, 0.1), True]
            self.conns[innovations.get(h, self.out)] = [h, self.out, rng.gauss(0, 0.3), True]
        self.conns[innovations.get(-1, self.out)] = [-1, self.out, rng.gauss(0, 0.1), True]

    def copy(self) -> "NEATGenome":
        g = NEATGenome(self.n_in, None, self.innovations, _skip_init=True)
        g.out = self.out
        g.next_nid = self.next_nid
        g.hidden = list(self.hidden)
        g.conns = {k: list(v) for k, v in self.conns.items()}
        g.fitness = self.fitness
        return g

    def add_node(self, rng: random.Random) -> None:
        en = [k for k, v in self.conns.items() if v[3]]
        if not en:
            return
        k = rng.choice(en)
        s, t, w, _ = self.conns[k]
        self.conns[k][3] = False
        nid = self.next_nid
        self.next_nid += 1
        self.hidden.append(nid)
        self.conns[self.innovations.get(s, nid)] = [s, nid, 1.0, True]
        self.conns[self.innovations.get(nid, t)] = [nid, t, w, True]

    def add_connection(self, rng: random.Random) -> None:
        srcs = list(range(0, self.n_in, 2)) + [-1] + self.hidden
        tgts = self.hidden + [self.out]
        ex = {(v[0], v[1]) for v in self.conns.values()}
        for _ in range(20):
            s, t = rng.choice(srcs), rng.choice(tgts)
            if (s, t) not in ex:
                self.conns[self.innovations.get(s, t)] = [s, t, rng.gauss(0, 0.3), True]
                return

    def mutate(self, rng: random.Random, cfg: NEATConfig) -> "NEATGenome":
        g = self.copy()
        for k in g.conns:
            if rng.random() < 0.9:
                g.conns[k][2] += rng.gauss(0, cfg.weight_sigma) if rng.random() < 0.9 \
                    else rng.gauss(0, 1.0)
        if rng.random() < cfg.p_add_node:
            g.add_node(rng)
        if rng.random() < cfg.p_add_connection:
            g.add_connection(rng)
        return g

    def compatibility(self, other: "NEATGenome", c1: float = 1.0, c2: float = 0.4) -> float:
        k1, k2 = set(self.conns), set(other.conns)
        match = [abs(self.conns[k][2] - other.conns[k][2]) for k in k1 & k2]
        N = max(len(k1), len(k2), 1)
        return c1 * len(k1 ^ k2) / N + c2 * (sum(match) / max(len(match), 1))

    def activate(self, X: np.ndarray) -> np.ndarray:
        n = X.shape[0]
        vals = {i: X[:, i] for i in range(self.n_in)}
        vals[-1] = np.ones(n)
        for node in self.hidden + [self.out]:
            inc = [(v[2], v[0]) for v in self.conns.values()
                   if v[1] == node and v[3] and v[0] in vals]
            if inc:
                z = sum(w * vals[s] for w, s in inc)
                vals[node] = sig(z) if node == self.out else relu(z)
            elif node not in vals:
                vals[node] = np.zeros(n)
        return vals.get(self.out, np.zeros(n))

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> float:
        return float(np.mean((self.activate(X) >= 0.5) == (y >= 0.5)))


def neat_task(task_X: np.ndarray, task_y: np.ndarray, carryover: list[NEATGenome],
              rng: random.Random, n_in: int, cfg: NEATConfig) -> tuple[list[NEATGenome], list[NEATGenome]]:
    """NEAT evolution on one task episode. Innovation numbers reset here,
    scoped to this episode's population, per report Sec. 3.5.1."""
    innovations = InnovationTracker()
    pop = [g.copy() for g in (carryover or [])]
    for g in pop:
        g.innovations = innovations  # carried-over genomes join this episode's registry
    while len(pop) < cfg.pop_size:
        pop.append(NEATGenome(n_in, rng, innovations, n_hid=cfg.initial_hidden))

    fx = fy = None
    for gen in range(cfg.n_gen):
        n = min(cfg.fitness_eval_n, len(task_X))
        idx = np.random.choice(len(task_X), n, replace=False)
        fx, fy = task_X[idx], task_y[idx]
        for g in pop:
            g.fitness = g.evaluate(fx, fy)

        species: list[list[NEATGenome]] = []
        for g in pop:
            placed = False
            for sp in species:
                if g.compatibility(sp[0]) < cfg.species_threshold:
                    sp.append(g)
                    placed = True
                    break
            if not placed:
                species.append([g])

        for sp in species:
            n_sp = len(sp)
            for g in sp:
                g.fitness /= n_sp

        new_pop = []
        for sp in species:
            sp.sort(key=lambda g: -g.fitness)
            new_pop.append(sp[0].copy())
        while len(new_pop) < cfg.pop_size:
            sp = rng.choice(species)
            parent = rng.choice(sp[:max(1, len(sp) // 2)])
            new_pop.append(parent.mutate(rng, cfg))
        pop = new_pop[:cfg.pop_size]

        if gen == cfg.n_gen - 1:
            for g in pop:
                g.fitness = g.evaluate(fx, fy)
            pop.sort(key=lambda g: -g.fitness)
            print(f"      final best={pop[0].fitness * 100:.1f}% "
                  f"species={len(species)} hidden_nodes={len(pop[0].hidden)}")

    pop.sort(key=lambda g: -g.fitness)
    return pop, pop[:cfg.carryover_k]


def run_neat(tasks: list[dict], n_in: int, cfg: NEATConfig, n_runs: int) -> list[list[list[float]]]:
    print("\n" + "=" * 55)
    print("CONDITION 3: NEAT EXTENSION")
    print("=" * 55)
    all_runs = []
    for run in range(n_runs):
        np.random.seed(run * 555 + 3)
        rng = random.Random(run * 555 + 3)
        mat, carryover, bgs = [], [], {}
        print(f"Run {run + 1}/{n_runs}")
        for ep, task in enumerate(tasks):
            print(f"  Ep{ep + 1} {task['label']}...", flush=True)
            X = np.array(task["train_x"], dtype=np.float32)
            y = np.array(task["train_y"], dtype=np.float32)
            pop, carryover = neat_task(X, y, carryover, rng, n_in, cfg)
            bgs[ep] = pop[0]
            ep_accs = []
            for ti in range(ep + 1):
                tX = np.array(tasks[ti]["test_x"], dtype=np.float32)
                ty = np.array(tasks[ti]["test_y"], dtype=np.float32)
                ep_accs.append(round(bgs[ti].evaluate(tX, ty), 4))
            mat.append(ep_accs)
            print(f"  -> {[f'{a * 100:.1f}%' for a in ep_accs]}")
        all_runs.append(mat)
    return all_runs

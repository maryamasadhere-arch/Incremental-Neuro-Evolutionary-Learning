"""
Experiment configuration.

Three presets are provided:

  quick   - tiny synthetic data, single run, small populations. Used by the
            test suite and by `python main.py --quick` for a fast sanity
            check of the whole pipeline (seconds, no network access).
  dev     - the scale used during day-to-day development: real Split-MNIST
            projected to a lower dimensionality, moderate population/
            generation counts, 3 runs. Finishes in a few minutes.
  full    - the paper-scale configuration, matching Table 3.1 of the project
            report as closely as a pure-Python/NumPy implementation allows
            (raw 784-dim pixels, 10 runs, population/generation counts from
            the report). This is computationally heavy - expect a long
            run - and is intended to be started deliberately, e.g. in the
            background, rather than run interactively every time.

Every field that corresponds to a cell in Table 3.1 is annotated with the
report value it targets. Where this implementation deviates (necessarily,
because the report describes a Java implementation and does not fully
specify every hyperparameter, e.g. the 2007 model's hidden-layer size), the
deviation is called out explicitly rather than left implicit.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class DataConfig:
    """Split-MNIST benchmark configuration (report Sec. 3.2)."""

    n_input_dims: int = 784        # report: raw 28x28 pixels flattened
    project_dims: int | None = None  # if set, randomly project 784 -> this many dims
    synthetic: bool = False        # bypass MNIST entirely (tests / offline use)
    synthetic_n_train: int = 200
    synthetic_n_test: int = 60
    seed: int = 42


@dataclass(frozen=True)
class BaselineConfig:
    """Condition 1 - backprop baseline (report Sec. 3.3, Table 3.1)."""

    hidden_layers: tuple[int, ...] = (256, 128)  # report: 784-256-128-1
    epochs_per_task: int = 10                    # report: 10 epochs
    batch_size: int = 32                         # report: 32
    lr: float = 0.01                             # report: eta = 0.01
    momentum: float = 0.9                        # report: beta = 0.9


@dataclass(frozen=True)
class EAConfig:
    """Condition 2 - 2007 EA replication (report Sec. 3.4, Table 3.1)."""

    hidden_units: int = 20        # report leaves H unspecified ("per paper");
                                   # not recoverable from the report alone -
                                   # chosen to keep genome size tractable.
    mu: int = 50                  # report: population size 50
    lam: int = 50                 # report: offspring 50
    sigma: float = 0.1            # report: mutation step size 0.1
    n_gen: int = 100              # report: 100 generations/task
    carryover_k: int = 10         # report: k = 10
    fitness_eval_n: int = 500     # report: 500 samples/task


@dataclass(frozen=True)
class NEATConfig:
    """Condition 3 - NEAT extension (report Sec. 3.5, Table 3.1)."""

    initial_hidden: int = 8       # report: "minimal start" - a small sparse
                                   # hidden layer is used as the practical
                                   # minimal topology (see NEAT module notes)
    pop_size: int = 150           # report: population size 150
    n_gen: int = 200              # report: 200 generations/task
    weight_sigma: float = 0.05    # report: mutation step size 0.05
    p_add_node: float = 0.03      # report: structural prob. 0.03
    p_add_connection: float = 0.05  # report: structural prob. 0.05
    species_threshold: float = 3.0  # report: delta_t = 3.0
    carryover_k: int = 10          # report: k = 10 (held equal to EA condition)
    fitness_eval_n: int = 500      # report: 500 samples/task


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    data: DataConfig
    baseline: BaselineConfig
    ea: EAConfig
    neat: NEATConfig
    n_runs: int = 10               # report: 10 independent runs (Sec. 3.7)
    n_tasks: int = 5                # Split-MNIST: 5 sequential binary tasks
    ec_prior_threshold: float = 0.70   # report Sec. 2.5: RA > 70% on prior tasks
    ec_current_threshold: float = 0.85  # report Sec. 2.5: acc > 85% on current task


def quick_config() -> ExperimentConfig:
    """Fast, fully offline configuration for tests and smoke runs."""
    return ExperimentConfig(
        name="quick",
        data=DataConfig(n_input_dims=16, project_dims=None, synthetic=True,
                         synthetic_n_train=120, synthetic_n_test=40),
        baseline=BaselineConfig(hidden_layers=(8,), epochs_per_task=2, batch_size=16),
        ea=EAConfig(hidden_units=6, mu=8, lam=8, n_gen=4, carryover_k=2, fitness_eval_n=60),
        neat=NEATConfig(initial_hidden=2, pop_size=8, n_gen=4, carryover_k=2, fitness_eval_n=60),
        n_runs=1,
    )


def dev_config() -> ExperimentConfig:
    """Development-scale configuration: real Split-MNIST, modest compute budget."""
    return ExperimentConfig(
        name="dev",
        data=DataConfig(n_input_dims=784, project_dims=64, synthetic=False),
        baseline=BaselineConfig(hidden_layers=(32,), epochs_per_task=10, batch_size=64),
        ea=EAConfig(hidden_units=32, mu=80, lam=80, sigma=0.1, n_gen=60,
                    carryover_k=10, fitness_eval_n=1500),
        neat=NEATConfig(initial_hidden=8, pop_size=80, n_gen=60, weight_sigma=0.08,
                         p_add_node=0.04, p_add_connection=0.05,
                         species_threshold=3.0, carryover_k=10, fitness_eval_n=400),
        n_runs=3,
    )


def full_config() -> ExperimentConfig:
    """Paper-scale configuration matching report Table 3.1. Slow - see README."""
    return ExperimentConfig(
        name="full",
        data=DataConfig(n_input_dims=784, project_dims=None, synthetic=False),
        baseline=BaselineConfig(hidden_layers=(256, 128), epochs_per_task=10, batch_size=32,
                                 lr=0.01, momentum=0.9),
        ea=EAConfig(hidden_units=20, mu=50, lam=50, sigma=0.1, n_gen=100,
                    carryover_k=10, fitness_eval_n=500),
        neat=NEATConfig(initial_hidden=8, pop_size=150, n_gen=200, weight_sigma=0.05,
                         p_add_node=0.03, p_add_connection=0.05,
                         species_threshold=3.0, carryover_k=10, fitness_eval_n=500),
        n_runs=10,
    )


PRESETS = {"quick": quick_config, "dev": dev_config, "full": full_config}

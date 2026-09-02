# Incremental Neuro-Evolutionary Learning


Does evolving a *population* of networks resist catastrophic forgetting better
than gradient descent on a single one? This project compares three learning
conditions on the Split-MNIST continual-learning benchmark, in **two
independent implementations**:

* **Python** (`inel/`, root of this repo) — the primary implementation used
  for day-to-day development and the results below. Fast to iterate on,
  fully tested, uses NumPy for the dense numeric hot paths.
* **Java + native C** (`java/`) — satisfies Objective O3's explicit
  requirement ("implement... in Java").

Both implementations run the same three conditions:

1. **Baseline** — a standard backprop feedforward network, trained sequentially.
   Demonstrates catastrophic forgetting.
2. **2007 EA** — a replication of a fixed-topology `(mu+lambda)` evolutionary
   algorithm with an incremental carry-over mechanism (Channon, 2007).
3. **NEAT** — the same incremental framework with the fixed-topology EA
   replaced by NeuroEvolution of Augmenting Topologies (Stanley & Miikkulainen,
   2002), which evolves network structure as well as weights.

All three are evaluated on identical Split-MNIST task sequences using four
metrics: Retention Accuracy (RA), Forgetting Rate (FR), Forward Transfer (FT),
and Evolvability Ceiling (EC).

## Objectives → where they live

| # | Objective | Python | Java / C |
|---|---|---|---|
| O2 | Backprop baseline + forgetting measurement | `inel/models/backprop.py` | `java/.../inel/BackpropNet.java` |
| O3 | 2007 EA replication | `inel/models/ea.py` | `java/.../inel/ea/EA.java` + native kernel `java/src/main/c/fitness_native.c` |
| O4 | Four-metric evaluation | `inel/metrics.py` | `java/.../inel/Metrics.java` |
| O5 | NEAT extension | `inel/models/neat.py` | `java/.../inel/neat/{NeatGenome,Neat}.java` |
| O6 | Comparative analysis | `inel/pipeline.py` | `java/.../inel/Pipeline.java` (CSV output) |

## Quick start (Python)

```bash
pip install -r requirements.txt
python main.py --quick          # offline smoke test, seconds, no network
python main.py                  # dev-scale run, real Split-MNIST, ~2-3 minutes
python main.py --full           # paper-scale run — slow, see below
```

Each run writes `results/{baseline,ea2007,neat}.json` (raw per-run accuracy
matrices), `results/all_metrics.json` (RA/FR/FT/EC per condition), and, if
matplotlib is installed, `results/figures/*.png`. Results are cached — delete
the relevant JSON file (or the whole `results/` directory) to force a re-run.

Run the test suite with:

```bash
pytest
```

## Architecture

```
inel/
  config.py          dataclasses for every hyperparameter, three presets
  data.py             Split-MNIST download/build + synthetic offline fallback
  activations.py      sigmoid / ReLU
  models/
    backprop.py        Condition 1
    ea.py               Condition 2
    neat.py             Condition 3
  metrics.py          RA / FR / FT / EC
  report.py           comparative figures (optional, needs matplotlib)
  pipeline.py         orchestrates data → 3 conditions → metrics → figures
  cli.py              argparse entry point
main.py               thin wrapper around inel.cli.main
tests/                pytest suite, fully offline (synthetic data + tiny configs)
java/                 second implementation (see java/README.md)
```

The original single-file prototype (`main.py`, ~570 lines) has been split
along these lines so each condition, the metrics, and the data pipeline can
be tested, configured, and reasoned about independently — the algorithms
themselves are preserved, not rewritten.

## Configuration presets (`inel/config.py`)

| Preset | Data | Scale | Runtime | Use for |
|---|---|---|---|---|
| `--quick` | synthetic, offline | tiny (pop ≤ 12, ≤ 15 generations, 1 run) | seconds | tests, CI, sanity checks |
| *(default)* | real Split-MNIST, projected to 64 dims | pop 80, 60 generations, 3 runs | ~2-3 min | day-to-day development |
| `--full` | real Split-MNIST, raw 784 dims | pop 50/150, 100/200 generations, 10 runs | long — run it deliberately | full paper-scale reproduction |

NEAT's "minimal start": true NEAT begins with zero hidden nodes; this
implementation starts from a small sparse hidden layer
(`NEATConfig.initial_hidden`) because at 784-dimensional input, a
zero-hidden genome would need many generations of pure structural mutation
before any hidden representation exists at all to evaluate. See the
docstring in `inel/models/neat.py`.

## Java + native C

```bash
cd java
export JAVA_HOME=/path/to/jdk-17-or-newer
./build_native.sh          # optional: builds target/native/libfitness.{so,dylib}
mvn test                   # 21 tests
mvn package                # target/inel.jar
java -Djava.library.path=target/native -jar target/inel.jar --quick
java -Djava.library.path=target/native -jar target/inel.jar          # dev-scale, real Split-MNIST
```

See `java/README.md` for the full breakdown of this implementation.

## Results

**Python, dev-scale** (3 runs, projected 64-dim Split-MNIST):

| Metric | Baseline | 2007 EA | NEAT |
|---|---|---|---|
| Mean RA (%) | 60.6 | 96.2 | 89.4 |
| Mean FR (%) | 47.8 | 0.0 | 0.0 |
| Mean FT (%) | 48.6 | 45.4 | 37.2 |
| EC (/5) | 1.0 | 5.0 | 1.0 |

**Python, paper-scale `--full`** (10 runs, raw 784-dim Split-MNIST; took
~5 hours, almost entirely in the NEAT condition):

| Metric | Baseline | 2007 EA | NEAT |
|---|---|---|---|
| Mean RA (%) | 57.1 | 96.0 | 97.0 |
| Mean FR (%) | 53.4 | 0.0 | 0.0 |
| Mean FT (%) | 49.7 | 45.2 | 46.4 |
| EC (/5) | 2.0 | 5.0 | 5.0 |

**Java, dev-scale** (3 runs, real Split-MNIST, `data/` shared read-only with
the Python side since both read the same raw IDX files):

| Metric | Baseline | 2007 EA | NEAT |
|---|---|---|---|
| Mean RA (%) | 59.1 | 95.5 | 91.0 |
| Mean FR (%) | 49.8 | 0.0 | 0.0 |
| Mean FT (%) | 48.7 | 44.4 | 39.1 |
| EC (/5) | 1.0 | 5.0 | 4.0 |

All three runs show the same pattern: the backprop baseline forgets
earlier tasks (FR 48-53%), while both evolutionary conditions retain them
(FR 0.0%). Exact numbers vary run to run and between the two languages
(different RNGs) — re-run to reproduce.

## Reproducibility

Every run seeds NumPy and Python's `random` deterministically per run index
(see `models/*/run_*`), so a given preset and run count reproduces the same
sequence of results. Real Split-MNIST data is cached under `data/` after the
first download; delete the cache to force a re-download.

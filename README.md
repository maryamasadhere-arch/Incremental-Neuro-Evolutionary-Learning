# Incremental Neuro-Evolutionary Learning


Does evolving a *population* of networks resist catastrophic forgetting better
than gradient descent on a single one? This project compares three learning
conditions on the Split-MNIST continual-learning benchmark:

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

| # | Objective | Implementation |
|---|---|---|
| O1 | Literature review | `docs/Farooq_Report_Ch1_Ch2_Ch3.docx` (project report, not code) |
| O2 | Backprop baseline + forgetting measurement | `inel/models/backprop.py` |
| O3 | 2007 EA replication | `inel/models/ea.py` |
| O4 | Four-metric evaluation | `inel/metrics.py` |
| O5 | NEAT extension | `inel/models/neat.py` |
| O6 | Comparative analysis | `inel/pipeline.py`, `inel/report.py` |

## Quick start

```bash
pip install -r requirements.txt
python main.py --quick          # offline smoke test, seconds, no network
python main.py                  # dev-scale run, real Split-MNIST, ~2-3 minutes
python main.py --full           # paper-scale run (Table 3.1) — slow, see below
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
| `--full` | real Split-MNIST, raw 784 dims | pop 50/150, 100/200 generations, 10 runs | long — run it deliberately | paper-scale reproduction of report Table 3.1 |

`--full` targets the report's Table 3.1 hyperparameters directly (see
`config.py` docstrings for the exact cell each field maps to). Two deviations
from the report are unavoidable and are called out explicitly in the code
rather than left implicit:

* **Language.** The report specifies a Java implementation; this is a Python/
  NumPy port. The algorithms (fixed-topology `(mu+lambda)`-ES with carry-over;
  NEAT with innovation numbers, speciation, and fitness sharing) are
  implemented as specified — only the host language differs.
* **2007 EA hidden-layer size.** The report leaves this unspecified beyond
  "matching the original 2007 paper as closely as possible," which isn't
  recoverable from the report text alone. `EAConfig.hidden_units` documents
  this and defaults to a value chosen to keep genome size tractable.
* **NEAT's "minimal start."** True NEAT begins with zero hidden nodes; this
  implementation starts from a small sparse hidden layer
  (`NEATConfig.initial_hidden`) because at 784-dimensional input, a
  zero-hidden genome would need many generations of pure structural mutation
  before any hidden representation exists at all to evaluate. See the
  docstring in `inel/models/neat.py`.

## Bugs fixed during the rewrite

The prototype's core algorithms were sound, but three issues were fixed
along the way (all covered by regression tests in `tests/`):

1. **Evolvability Ceiling used one threshold instead of two, and never
   stopped at the first failure.** The report (Sec. 2.5) defines EC as the
   longest sequence for which RA stays above 70% on *prior* tasks **and**
   accuracy exceeds 85% on the *current* task — two different thresholds.
   The original code checked `>= 0.70` uniformly across all tasks including
   the current one, and kept overwriting the ceiling on every later episode
   that happened to pass again, rather than stopping at the first failure.
   Fixed in `metrics.py::compute_metrics`; see `tests/test_metrics.py` for
   the exact before/after behaviour this changes.
2. **MNIST download disabled TLS certificate verification**
   (`ctx.verify_mode = ssl.CERT_NONE`), an unnecessary exposure to
   man-in-the-middle tampering of the downloaded dataset. `inel/data.py` now
   downloads over a normal, fully-verified HTTPS context and retries
   transient network failures with backoff instead of failing outright.
3. **`sig()` overflowed for float32 inputs**, raising a `RuntimeWarning` (the
   EA population arrays are float32, but the clip bound was tuned for
   float64). The result was still numerically correct (saturates to 0), but
   the warning is silenced properly in `inel/activations.py` by clipping to
   a dtype-appropriate bound instead of suppressing the message.

## Example dev-scale result

A representative `python main.py` run (3 runs, real Split-MNIST):

| Metric | Baseline | 2007 EA | NEAT |
|---|---|---|---|
| Mean RA (%) | 60.6 | 96.2 | 89.4 |
| Mean FR (%) | 47.8 | 0.0 | 0.0 |
| Mean FT (%) | 48.6 | 45.4 | 37.2 |
| EC (/5) | 1.0 | 5.0 | 1.0 |

This reproduces the qualitative claim under test: both evolutionary
conditions eliminate forgetting (FR ≈ 0%) that the backprop baseline clearly
exhibits (FR ≈ 48%). The EC gap between the two evolutionary conditions
(5/5 vs 1/5) is a genuine — not a metric-bug — finding: NEAT's evolving
topology more often needs extra generations to exceed the 85% *current-task*
bar on a freshly-introduced task, even though it retains prior tasks just as
well once learned. Exact numbers vary run to run; re-run `python main.py` to
reproduce.

## Reproducibility

Every run seeds NumPy and Python's `random` deterministically per run index
(see `models/*/run_*`), so a given preset and run count reproduces the same
sequence of results. Real Split-MNIST data is cached under `data/` after the
first download; delete the cache to force a re-download.

# Incremental Neuro-Evolutionary Learning


Does evolving a *population* of networks resist catastrophic forgetting better
than gradient descent on a single one? This project compares three learning
conditions on the Split-MNIST continual-learning benchmark, in **two
independent implementations**:

* **Python** (`inel/`, root of this repo) — the primary implementation used
  for day-to-day development and the results below. Fast to iterate on,
  fully tested, uses NumPy for the dense numeric hot paths.
* **Java + native C** (`java/`) — satisfies Objective O3's explicit
  requirement ("implement... in Java") and report Sec. 3.6 ("implemented in
  Java (SE 17)... ExecutorService... custom Java logger class... CSV
  files"), with the EA condition's population fitness evaluation offloaded
  to a small JNI native C kernel — the concrete answer to Sec. 3.9's own
  observation that Java has "a performance overhead relative to optimised
  C++ ... implementations" for exactly this kind of dense numeric loop. See
  [`java/README.md`](java/README.md) for build/run instructions and how it
  maps to the report.

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
| O1 | Literature review | `docs/Farooq_Report_Ch1_Ch2_Ch3.docx` (project report, not code) | — |
| O2 | Backprop baseline + forgetting measurement | `inel/models/backprop.py` | `java/.../inel/BackpropNet.java` |
| O3 | 2007 EA replication **(report specifies Java)** | `inel/models/ea.py` | `java/.../inel/ea/EA.java` + native kernel `java/src/main/c/fitness_native.c` |
| O4 | Four-metric evaluation | `inel/metrics.py` | `java/.../inel/Metrics.java` |
| O5 | NEAT extension | `inel/models/neat.py` | `java/.../inel/neat/{NeatGenome,Neat}.java` |
| O6 | Comparative analysis | `inel/pipeline.py`, `inel/report.py` | `java/.../inel/Pipeline.java` (CSV output) |

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
rather than left implicit (the Python implementation's language choice is
no longer one of them — see the Java/C section below):

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

## Java + native C implementation (`java/`)

The report's Objective O3 explicitly says "implement... in Java", and
Sec. 3.6 specifies Java SE 17, `ExecutorService`-parallelised fitness
evaluation (called out by name for NEAT), and CSV-based result logging via
"a custom Java logger class". `java/` satisfies this literally — a
from-scratch second implementation of all three conditions, not a
transpilation of the Python one, validated the same way: 21 JUnit tests
(including the identical Evolvability Ceiling regression tests as the
Python suite) plus a real run against actual Split-MNIST data.

Where Sec. 3.9 observes that Java carries "a performance overhead relative
to optimised C++ ... implementations" for dense numeric evolutionary loops,
`java/src/main/c/fitness_native.c` is that C implementation: a JNI-callable
kernel that evaluates an entire EA population's classification accuracy in
one native call (the same batched forward-pass-plus-accuracy operation
`inel/models/ea.py` vectorises with NumPy's `einsum` on the Python side).
It's wired up as an optional accelerator — `inel.ea.NativeFitness`
transparently falls back to an identical pure-Java implementation if the
native library hasn't been built or fails to load, so the Java build never
hard-depends on a C toolchain being present; `EATest.nativeAndJavaFitnessAgree`
asserts the two paths produce bit-identical results whenever the native
library *is* available.

```bash
cd java
export JAVA_HOME=/path/to/jdk-17-or-newer
./build_native.sh          # optional: builds target/native/libfitness.{so,dylib}
mvn test                   # 21 tests
mvn package                # target/inel.jar
java -Djava.library.path=target/native -jar target/inel.jar --quick
java -Djava.library.path=target/native -jar target/inel.jar          # dev-scale, real Split-MNIST
```

NEAT's population-level parallelism uses a `java.util.concurrent.ExecutorService`
fixed thread pool sized to `Runtime.getRuntime().availableProcessors()`,
exactly as Sec. 3.6 describes. Results are written as CSV (not JSON, unlike
the Python side) to `results-java/` — per-run accuracy matrices plus
per-generation best/mean fitness (and species count, for NEAT) — matching
the report's specified logging format directly. See `java/README.md` for
the full breakdown.

**A real dev-scale run** (3 runs, real Split-MNIST, `data/` shared read-only
with the Python side since both read the same raw IDX files) reproduces the
same core finding independently, with numbers close to but not identical to
the Python run — expected, since the two use unrelated RNG streams:

| Metric | Baseline | 2007 EA | NEAT |
|---|---|---|---|
| Mean RA (%) | 59.1 | 95.5 | 91.0 |
| Mean FR (%) | 49.8 | 0.0 | 0.0 |
| Mean FT (%) | 48.7 | 44.4 | 39.1 |
| EC (/5) | 1.0 | 5.0 | 4.0 |

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

## Paper-scale (`--full`) result

`python main.py --full` (10 runs, raw 784-dim Split-MNIST, report Table 3.1
hyperparameters) has been run to completion — this is not an untested
code path. It took roughly 5 hours in total, almost entirely in the NEAT
condition (population 150 × 200 generations × 5 tasks × 10 runs):

| Metric | Baseline | 2007 EA | NEAT |
|---|---|---|---|
| Mean RA (%) | 57.1 | 96.0 | 97.0 |
| Mean FR (%) | 53.4 | 0.0 | 0.0 |
| Mean FT (%) | 49.7 | 45.2 | 46.4 |
| EC (/5) | 2.0 | 5.0 | 5.0 |

The same qualitative finding holds at full scale, more decisively than at
dev-scale: the baseline's forgetting is if anything *worse* at full
resolution (FR 53.4% vs. 47.8%, since a higher-capacity 784-256-128-1
network has more room to overwrite earlier-task weights), while both
evolutionary conditions still show zero measured forgetting. Notably,
NEAT's Evolvability Ceiling reaches the full 5/5 at this scale (unlike the
dev-scale run's 1/5) — with the report's full population size (150, vs.
80 at dev-scale) and generation budget (200, vs. 60), NEAT has enough
search budget to reliably clear the 85% current-task bar even on a freshly
introduced task, closing the gap with the 2007 EA that dev-scale runs
undersell. Full artifacts (`baseline.json`, `ea2007.json`, `neat.json`,
`all_metrics.json`, `figures/*.png`) are reproducible by re-running
`python main.py --full` (data is cached; expect several hours, dominated by
the NEAT condition).

## Reproducibility

Every run seeds NumPy and Python's `random` deterministically per run index
(see `models/*/run_*`), so a given preset and run count reproduces the same
sequence of results. Real Split-MNIST data is cached under `data/` after the
first download; delete the cache to force a re-download.

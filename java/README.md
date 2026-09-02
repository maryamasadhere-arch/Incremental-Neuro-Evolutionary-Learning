# Incremental Neuro-Evolutionary Learning — Java implementation


## Build & run

```bash
export JAVA_HOME=/path/to/jdk-17-or-newer   # JDK 21 also works; 

./build_native.sh      # optional: compiles the native C fitness kernel
                        # (gcc/clang required). Skip this and everything
                        # still works via the pure-Java fallback.

mvn test                # 21 JUnit tests
mvn package             # -> target/inel.jar

java -Djava.library.path=target/native -jar target/inel.jar --quick
java -Djava.library.path=target/native -jar target/inel.jar
java -Djava.library.path=target/native -jar target/inel.jar --full --runs 5
```

`-Djava.library.path=target/native` is only needed if you built the native
library and want it picked up automatically by `NativeFitness`'s
classloader-relative search; omitting it (or not running `build_native.sh`
at all) is fine — `NativeFitness.evalPopulation` falls back to a pure-Java
implementation with identical semantics.



## Architecture

```
java/
  pom.xml
  build_native.sh                 compiles src/main/c -> target/native/libfitness.{so,dylib}
  src/main/c/
    fitness_native.c              JNI native EA population fitness kernel
  src/main/java/inel/
    Config.java                   records mirroring inel/config.py's presets exactly
    Task.java                     one Split-MNIST task episode
    Mnist.java                    IDX download/read + Split-MNIST split + synthetic fallback
    Activations.java              sigmoid / relu
    Fmt.java                      shared console-formatting helper
    CsvLogger.java                the "custom Java logger class" 
    BackpropNet.java              Condition 1 (O2)
    Metrics.java                  RA / FR / FT / EC (O4) - same corrected EC logic as Python
    Pipeline.java                 orchestrates data -> 3 conditions -> metrics -> CSV
    Main.java                     CLI entry point
    ea/
      NativeFitness.java          JNI bindings + pure-Java fallback
      EA.java                     Condition 2 (O3): (mu+lambda)-ES with carry-over
    neat/
      InnovationTracker.java      shared historical-marking registry (per task episode)
      ConnGene.java                one NEAT connection gene
      NeatGenome.java              genome: mutation, speciation distance, activation
      Neat.java                    Condition 3 (O5): speciation + ExecutorService parallel eval
  src/test/java/inel/             JUnit 5 tests mirroring tests/test_*.py, incl. EC regressions
```

## Design notes

**Why a native C kernel for EA, not NEAT.** The EA condition's population
is a fixed-topology dense computation (every individual has the exact same
matrix shapes) - a natural fit for a branch-free native loop, mirroring
what `inel/models/ea.py` does with NumPy's vectorised `einsum` in the
Python implementation. NEAT's population has *ragged, per-genome* topology
(variable node/connection counts, hash-map-based graph traversal per
genome), which doesn't lend itself to the same kind of batched native
kernel without a much larger effort (a real payoff would need a custom
sparse-graph IR compiled per genome). NEAT is parallelised instead across
CPU cores via `ExecutorService`, 

**Native kernel is unoptimised on purpose.** `fitness_native.c` is a
straightforward triple-nested loop, not a BLAS-backed or SIMD-vectorised
matrix multiply. It is still faster than an interpreted-language
equivalent, but it is *not* expected to beat NumPy's einsum (which calls
into a tuned BLAS implementation) on wall-clock time for the same
population/generation counts - the dev-scale Java run above takes longer
in total than the equivalent Python run, almost entirely in the
NEAT condition, which is identical Java-vs-Java work regardless of the
native kernel. Further optimisation (OpenMP, cache blocking, linking
against a real BLAS) is a natural next step if throughput matters more
than the current "prove the language/native-interop path end-to-end"
scope.

**RNG streams are not cross-language-reproducible.** `java.util.Random`
and NumPy's Mersenne Twister are different algorithms; a given seed does
not produce the same sequence in both languages. Each implementation is
internally deterministic (same seed -> same result, within that language),
and both are validated to reproduce the *same qualitative finding*
(catastrophic forgetting in the baseline, ~0% forgetting in both
evolutionary conditions) - but exact numbers will never match bit-for-bit
between the two, and that's expected, not a bug.

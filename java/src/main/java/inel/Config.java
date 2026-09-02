package inel;

/**
 * Experiment configuration records, mirroring inel/config.py exactly (see
 * that file's docstring for the full report-Table-3.1 mapping and
 * documented deviations). Three presets: quick (offline synthetic smoke
 * test), dev (real Split-MNIST, projected dims, fast iteration), full
 * (paper-scale, report Table 3.1).
 */
public final class Config {

    public record DataConfig(
            int nInputDims,
            int projectDims,   // 0 means "no projection, use raw dims"
            boolean synthetic,
            int syntheticNTrain,
            int syntheticNTest,
            long seed
    ) {}

    public record BaselineConfig(
            int[] hiddenLayers,
            int epochsPerTask,
            int batchSize,
            double lr,
            double momentum
    ) {}

    public record EAConfig(
            int hiddenUnits,
            int mu,
            int lambda,
            double sigma,
            int nGen,
            int carryoverK,
            int fitnessEvalN
    ) {}

    public record NEATConfig(
            int initialHidden,
            int popSize,
            int nGen,
            double weightSigma,
            double pAddNode,
            double pAddConnection,
            double speciesThreshold,
            int carryoverK,
            int fitnessEvalN
    ) {}

    public record ExperimentConfig(
            String name,
            DataConfig data,
            BaselineConfig baseline,
            EAConfig ea,
            NEATConfig neat,
            int nRuns,
            int nTasks,
            double ecPriorThreshold,
            double ecCurrentThreshold
    ) {
        public ExperimentConfig withRuns(int runs) {
            return new ExperimentConfig(name, data, baseline, ea, neat, runs, nTasks,
                    ecPriorThreshold, ecCurrentThreshold);
        }
    }

    public static ExperimentConfig quick() {
        return new ExperimentConfig(
                "quick",
                new DataConfig(16, 0, true, 120, 40, 42),
                new BaselineConfig(new int[]{8}, 2, 16, 0.01, 0.9),
                new EAConfig(6, 8, 8, 0.1, 4, 2, 60),
                new NEATConfig(2, 8, 4, 0.08, 0.04, 0.05, 3.0, 2, 60),
                1, 5, 0.70, 0.85
        );
    }

    public static ExperimentConfig dev() {
        return new ExperimentConfig(
                "dev",
                new DataConfig(784, 64, false, 0, 0, 42),
                new BaselineConfig(new int[]{32}, 10, 64, 0.01, 0.9),
                new EAConfig(32, 80, 80, 0.1, 60, 10, 1500),
                new NEATConfig(8, 80, 60, 0.08, 0.04, 0.05, 3.0, 10, 400),
                3, 5, 0.70, 0.85
        );
    }

    public static ExperimentConfig full() {
        return new ExperimentConfig(
                "full",
                new DataConfig(784, 0, false, 0, 0, 42),
                new BaselineConfig(new int[]{256, 128}, 10, 32, 0.01, 0.9),
                new EAConfig(20, 50, 50, 0.1, 100, 10, 500),
                new NEATConfig(8, 150, 200, 0.05, 0.03, 0.05, 3.0, 10, 500),
                10, 5, 0.70, 0.85
        );
    }

    public static ExperimentConfig preset(String name) {
        return switch (name) {
            case "quick" -> quick();
            case "dev" -> dev();
            case "full" -> full();
            default -> throw new IllegalArgumentException("unknown preset: " + name);
        };
    }

    private Config() {}
}

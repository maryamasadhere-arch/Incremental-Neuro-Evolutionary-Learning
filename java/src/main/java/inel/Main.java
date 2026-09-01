package inel;

import java.nio.file.Path;

/**
 * CLI entry point, mirroring inel/cli.py's options.
 *
 * Usage:
 *   java -jar target/inel.jar --quick
 *   java -jar target/inel.jar
 *   java -jar target/inel.jar --full --runs 5
 */
public final class Main {
    public static void main(String[] args) throws Exception {
        String preset = "dev";
        int runsOverride = -1;
        Path dataDir = Path.of("data");
        Path resultsDir = Path.of("results-java");

        for (int i = 0; i < args.length; i++) {
            switch (args[i]) {
                case "--quick" -> preset = "quick";
                case "--full" -> preset = "full";
                case "--runs" -> runsOverride = Integer.parseInt(args[++i]);
                case "--data-dir" -> dataDir = Path.of(args[++i]);
                case "--results-dir" -> resultsDir = Path.of(args[++i]);
                case "--help", "-h" -> {
                    printHelp();
                    return;
                }
                default -> {
                    System.err.println("Unknown argument: " + args[i]);
                    printHelp();
                    System.exit(1);
                }
            }
        }

        Config.ExperimentConfig cfg = Config.preset(preset);
        if (runsOverride > 0) cfg = cfg.withRuns(runsOverride);

        Pipeline.run(cfg, dataDir, resultsDir);
    }

    private static void printHelp() {
        System.out.println("""
                Incremental Neuro-Evolutionary Learning (Java)
                  --quick              tiny, fully offline smoke test (synthetic data)
                  --full               paper-scale configuration (report Table 3.1); slow
                  --runs N             override the number of independent runs
                  --data-dir PATH      (default: data)
                  --results-dir PATH   (default: results-java)
                """);
    }
}

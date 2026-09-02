package inel;

import inel.ea.EA;
import inel.neat.Neat;

import java.io.IOException;
import java.nio.file.*;
import java.util.List;

/**
 * Orchestrates the full experiment: data prep, all three conditions,
 * metrics, CSV export. Results are written as CSV (report Sec. 3.6: "All
 * experimental results are logged to CSV files via a custom Java logger
 * class"), not JSON - CSV is the report's specified format for the Java
 * implementation, unlike the Python pipeline's JSON (a choice made for
 * that implementation, not a report requirement).
 */
public final class Pipeline {

    public static void run(Config.ExperimentConfig cfg, Path dataDir, Path resultsDir) throws IOException {
        Files.createDirectories(dataDir);
        Files.createDirectories(resultsDir);

        System.out.println("=".repeat(55));
        System.out.println("INCREMENTAL NEURO-EVOLUTIONARY LEARNING (Java/" + (
                inel.ea.NativeFitness.isNativeAvailable() ? "native C fitness kernel" : "pure-Java fallback") + ")");
        System.out.println("Preset: " + cfg.name());
        System.out.println("=".repeat(55));

        System.out.println("\n[1/5] Preparing Split-MNIST benchmark (" +
                (cfg.data().synthetic() ? "synthetic" : "real MNIST") + ")...");
        Task[] tasks = Mnist.loadTasks(dataDir, cfg.data(), cfg.nTasks());
        int nIn = cfg.data().synthetic() ? cfg.data().nInputDims()
                : (cfg.data().projectDims() > 0 ? cfg.data().projectDims() : 784);
        System.out.println("  " + tasks.length + " tasks ready, " + nIn + "-dimensional inputs");

        System.out.println("\n[2/5] Running Condition 1: Backpropagation Baseline...");
        List<List<List<Double>>> baseRuns = BackpropNet.runBaseline(tasks, nIn, cfg.baseline(), cfg.nRuns());
        writeAccuracyCsv(resultsDir.resolve("baseline_accuracy.csv"), baseRuns);

        System.out.println("\n[3/5] Running Condition 2: 2007 EA Replication...");
        try (CsvLogger eaLog = new CsvLogger(resultsDir.resolve("ea_generations.csv"),
                "condition,run,episode,generation,best_fitness,mean_fitness")) {
            List<List<List<Double>>> eaRuns = EA.runEa2007(tasks, nIn, cfg.ea(), cfg.nRuns(), eaLog);
            writeAccuracyCsv(resultsDir.resolve("ea2007_accuracy.csv"), eaRuns);

            System.out.println("\n[4/5] Running Condition 3: NEAT Extension...");
            try (CsvLogger neatLog = new CsvLogger(resultsDir.resolve("neat_generations.csv"),
                    "condition,run,episode,generation,best_fitness,mean_fitness,species_count")) {
                List<List<List<Double>>> neatRuns = Neat.runNeat(tasks, nIn, cfg.neat(), cfg.nRuns(), neatLog);
                writeAccuracyCsv(resultsDir.resolve("neat_accuracy.csv"), neatRuns);

                System.out.println("\n[5/5] Computing metrics (RA, FR, FT, EC)...");
                Metrics mb = Metrics.compute(baseRuns, "Baseline (Backprop)", cfg.nTasks(),
                        cfg.ecPriorThreshold(), cfg.ecCurrentThreshold());
                Metrics me = Metrics.compute(eaRuns, "2007 EA (Replication)", cfg.nTasks(),
                        cfg.ecPriorThreshold(), cfg.ecCurrentThreshold());
                Metrics mn = Metrics.compute(neatRuns, "NEAT (Extension)", cfg.nTasks(),
                        cfg.ecPriorThreshold(), cfg.ecCurrentThreshold());
                Metrics.printSummary(mb, me, mn);
                writeMetricsCsv(resultsDir.resolve("metrics.csv"), mb, me, mn);
            }
        }

        System.out.println("\n" + "=".repeat(55));
        System.out.println("ALL DONE. Results saved to " + resultsDir);
        System.out.println("=".repeat(55));
    }

    private static void writeAccuracyCsv(Path path, List<List<List<Double>>> runs) throws IOException {
        try (CsvLogger log = new CsvLogger(path, "run,episode,task_index,accuracy")) {
            for (int r = 0; r < runs.size(); r++) {
                List<List<Double>> episodes = runs.get(r);
                for (int ep = 0; ep < episodes.size(); ep++) {
                    List<Double> accs = episodes.get(ep);
                    for (int ti = 0; ti < accs.size(); ti++) {
                        log.row(r, ep, ti, accs.get(ti));
                    }
                }
            }
        }
    }

    private static void writeMetricsCsv(Path path, Metrics mb, Metrics me, Metrics mn) throws IOException {
        try (CsvLogger log = new CsvLogger(path, "condition,RA_mean,FR_mean,FT_mean,EC")) {
            for (Metrics m : new Metrics[]{mb, me, mn}) {
                log.row(m.name(), m.raMean(), m.frMean(), m.ftMean(), m.ec());
            }
        }
    }

    private Pipeline() {}
}

package inel.ea;

import inel.Config;
import inel.CsvLogger;
import inel.Task;

import java.util.*;

/**
 * Condition 2 - replication of the 2007 incremental evolutionary model
 * (report Sec. 3.4, Objective O3). Fixed-topology (nIn -> nH -> 1) network
 * evolved with a (mu+lambda) evolutionary strategy and Gaussian mutation.
 * Incremental learning comes purely from the carry-over mechanism: the k
 * fittest genomes from task i seed task i+1's population and are protected
 * by (mu+lambda) elitist selection. Matches inel/models/ea.py exactly,
 * with population fitness evaluation delegated to NativeFitness (native
 * C kernel when available, pure-Java fallback otherwise).
 */
public final class EA {

    public static int genomeLen(int nIn, int nH) {
        return (nIn + 1) * nH + (nH + 1);
    }

    public record TaskResult(double[][] population, double[] fitness, List<double[]> carryover) {}

    public static double[] evalOne(double[] genome, double[][] X, double[] y, int nIn, int nH) {
        double[] fits = new double[1];
        NativeFitness.evalPopulation(genome, 1, nIn, nH, flatten(X), X.length, y, fits);
        return fits;
    }

    /**
     * Gaussian mutation (report Sec. 3.4.2): each offspring is produced from
     * a single randomly-chosen parent (drawn from the fittest {@code nPar}
     * of {@code pop}) by perturbing every weight with independent
     * N(0, sigma^2) noise, i.e. w' = w + N(0, sigma^2). No crossover /
     * recombination between two parents is used here - matching the 2007
     * model being replicated, which is a (mu+lambda) Evolution Strategy
     * (mutation-only reproduction), not a genetic algorithm.
     */
    public static double[][] mutate(double[][] pop, int nPar, int nOffspring, double sigma, Random rng) {
        int genomeLen = pop[0].length;
        double[][] offspring = new double[nOffspring][genomeLen];
        for (int o = 0; o < nOffspring; o++) {
            int parent = rng.nextInt(nPar);
            for (int k = 0; k < genomeLen; k++) {
                offspring[o][k] = pop[parent][k] + rng.nextGaussian() * sigma;
            }
        }
        return offspring;
    }

    public static TaskResult eaTask(double[][] taskX, double[] taskY, List<double[]> carryover,
                                     long seed, int nIn, Config.EAConfig cfg) {
        return eaTask(taskX, taskY, carryover, seed, nIn, cfg, null, 0, 0);
    }

    public static TaskResult eaTask(double[][] taskX, double[] taskY, List<double[]> carryover,
                                     long seed, int nIn, Config.EAConfig cfg,
                                     CsvLogger logger, int run, int episode) {
        Random rng = new Random(seed);
        int nH = cfg.hiddenUnits();
        int G = genomeLen(nIn, nH);
        double lim = 1.0 / Math.sqrt(nIn);

        List<double[]> rows = new ArrayList<>(carryover);
        while (rows.size() < cfg.mu()) {
            double[] g = new double[G];
            for (int i = 0; i < G; i++) g[i] = (rng.nextDouble() * 2 - 1) * lim;
            rows.add(g);
        }
        double[][] pop = rows.subList(0, cfg.mu()).toArray(new double[0][]);
        double[] fits = evalPop(pop, taskX, taskY, nIn, nH);
        int[] order = argsortDesc(fits);
        pop = reorderRows(pop, order);
        fits = reorderVals(fits, order);
        int nPar = Math.max(1, cfg.mu() / 3);

        for (int gen = 0; gen < cfg.nGen(); gen++) {
            double[][] offs = mutate(pop, nPar, cfg.lambda(), cfg.sigma(), rng);
            double[] of = evalPop(offs, taskX, taskY, nIn, nH);

            double[][] allPop = new double[pop.length + offs.length][];
            System.arraycopy(pop, 0, allPop, 0, pop.length);
            System.arraycopy(offs, 0, allPop, pop.length, offs.length);
            double[] allFits = new double[fits.length + of.length];
            System.arraycopy(fits, 0, allFits, 0, fits.length);
            System.arraycopy(of, 0, allFits, fits.length, of.length);

            int[] ord2 = argsortDesc(allFits);
            double[][] newPop = new double[cfg.mu()][];
            double[] newFits = new double[cfg.mu()];
            for (int i = 0; i < cfg.mu(); i++) {
                newPop[i] = allPop[ord2[i]];
                newFits[i] = allFits[ord2[i]];
            }
            pop = newPop;
            fits = newFits;

            if (logger != null) logger.row("EA", run, episode, gen, fits[0], mean(fits));
            if (gen == cfg.nGen() - 1) {
                System.out.printf("      final best=%.1f%% mean=%.1f%%%n", fits[0] * 100, mean(fits) * 100);
            }
        }

        List<double[]> carry = new ArrayList<>();
        for (int i = 0; i < Math.min(cfg.carryoverK(), pop.length); i++) carry.add(pop[i]);
        return new TaskResult(pop, fits, carry);
    }

    public static List<List<List<Double>>> runEa2007(Task[] tasks, int nIn, Config.EAConfig cfg, int nRuns,
                                                       CsvLogger logger) {
        System.out.println("\n" + "=".repeat(55));
        System.out.println("CONDITION 2: 2007 EA REPLICATION");
        System.out.println("=".repeat(55));
        List<List<List<Double>>> allRuns = new ArrayList<>();
        for (int run = 0; run < nRuns; run++) {
            Random rng = new Random((long) run * 777 + 1);
            List<double[]> carry = new ArrayList<>();
            Map<Integer, double[]> bgs = new HashMap<>();
            List<List<Double>> mat = new ArrayList<>();
            System.out.println("Run " + (run + 1) + "/" + nRuns);
            for (int ep = 0; ep < tasks.length; ep++) {
                System.out.println("  Ep" + (ep + 1) + " " + tasks[ep].label() + "...");
                double[][] X = tasks[ep].trainX();
                double[] y = tasks[ep].trainY();
                int n = Math.min(cfg.fitnessEvalN(), X.length);
                int[] idx = sampleWithoutReplacement(X.length, n, rng);
                double[][] Xs = subsetRows(X, idx);
                double[] ys = subsetVals(y, idx);

                TaskResult r = eaTask(Xs, ys, carry, (long) run * 777 + ep, nIn, cfg, logger, run, ep);
                carry = r.carryover();
                bgs.put(ep, r.population()[0]);

                List<Double> epAccs = new ArrayList<>();
                for (int ti = 0; ti <= ep; ti++) {
                    double acc = evalOne(bgs.get(ti), tasks[ti].testX(), tasks[ti].testY(), nIn, cfg.hiddenUnits())[0];
                    epAccs.add(acc);
                }
                mat.add(epAccs);
                System.out.println("  -> " + inel.Fmt.pct(epAccs));
            }
            allRuns.add(mat);
        }
        return allRuns;
    }

    // ---- small numeric helpers (no external deps, matching numpy semantics used in ea.py) ----

    private static double[] evalPop(double[][] pop, double[][] X, double[] y, int nIn, int nH) {
        double[] fits = new double[pop.length];
        NativeFitness.evalPopulation(flatten(pop), pop.length, nIn, nH, flatten(X), X.length, y, fits);
        return fits;
    }

    private static double[] flatten(double[][] m) {
        int rows = m.length, cols = m[0].length;
        double[] out = new double[rows * cols];
        for (int i = 0; i < rows; i++) System.arraycopy(m[i], 0, out, i * cols, cols);
        return out;
    }

    private static int[] argsortDesc(double[] v) {
        Integer[] idx = new Integer[v.length];
        for (int i = 0; i < v.length; i++) idx[i] = i;
        Arrays.sort(idx, (a, b) -> Double.compare(v[b], v[a]));
        int[] out = new int[v.length];
        for (int i = 0; i < v.length; i++) out[i] = idx[i];
        return out;
    }

    private static double[][] reorderRows(double[][] m, int[] order) {
        double[][] out = new double[order.length][];
        for (int i = 0; i < order.length; i++) out[i] = m[order[i]];
        return out;
    }

    private static double[] reorderVals(double[] v, int[] order) {
        double[] out = new double[order.length];
        for (int i = 0; i < order.length; i++) out[i] = v[order[i]];
        return out;
    }

    private static double mean(double[] v) {
        double s = 0;
        for (double x : v) s += x;
        return s / v.length;
    }

    private static int[] sampleWithoutReplacement(int total, int k, Random rng) {
        int[] pool = new int[total];
        for (int i = 0; i < total; i++) pool[i] = i;
        for (int i = 0; i < k; i++) {
            int j = i + rng.nextInt(total - i);
            int t = pool[i]; pool[i] = pool[j]; pool[j] = t;
        }
        return Arrays.copyOf(pool, k);
    }

    private static double[][] subsetRows(double[][] m, int[] idx) {
        double[][] out = new double[idx.length][];
        for (int i = 0; i < idx.length; i++) out[i] = m[idx[i]];
        return out;
    }

    private static double[] subsetVals(double[] v, int[] idx) {
        double[] out = new double[idx.length];
        for (int i = 0; i < idx.length; i++) out[i] = v[idx[i]];
        return out;
    }

    private EA() {}
}

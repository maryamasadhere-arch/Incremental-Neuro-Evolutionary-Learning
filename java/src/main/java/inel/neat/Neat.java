package inel.neat;

import inel.Config;
import inel.CsvLogger;
import inel.Fmt;
import inel.Task;

import java.util.*;
import java.util.concurrent.*;

/**
 * Condition 3 - NEAT extension (report Sec. 3.5, Objective O5). Evolves
 * both weights and topology; historical markings, speciation with explicit
 * fitness sharing, and the same carry-over mechanism as the 2007 EA
 * condition. Matches inel/models/neat.py exactly.
 *
 * Report Sec. 3.6 explicitly specifies Java's ExecutorService "particularly
 * [for] the fitness evaluation of large NEAT populations" - that is done
 * here, splitting each generation's population fitness evaluation across a
 * fixed thread pool sized to the available processor cores.
 */
public final class Neat {

    public record TaskResult(List<NeatGenome> population, List<NeatGenome> carryover) {}

    public static TaskResult neatTask(double[][] taskX, double[] taskY, List<NeatGenome> carryover,
                                       Random rng, int nIn, Config.NEATConfig cfg) {
        return neatTask(taskX, taskY, carryover, rng, nIn, cfg, null, 0, 0);
    }

    public static TaskResult neatTask(double[][] taskX, double[] taskY, List<NeatGenome> carryover,
                                       Random rng, int nIn, Config.NEATConfig cfg,
                                       CsvLogger logger, int run, int episode) {
        InnovationTracker innovations = new InnovationTracker();
        List<NeatGenome> pop = new ArrayList<>();
        for (NeatGenome g : carryover) pop.add(g.withInnovations(innovations));
        while (pop.size() < cfg.popSize()) {
            pop.add(new NeatGenome(nIn, rng, innovations, cfg.initialHidden()));
        }

        ExecutorService pool = Executors.newFixedThreadPool(
                Math.max(1, Runtime.getRuntime().availableProcessors()));
        try {
            double[][] fx = null;
            double[] fy = null;
            for (int gen = 0; gen < cfg.nGen(); gen++) {
                int n = Math.min(cfg.fitnessEvalN(), taskX.length);
                int[] idx = sampleWithoutReplacement(taskX.length, n, rng);
                fx = subsetRows(taskX, idx);
                fy = subsetVals(taskY, idx);
                evaluateAllParallel(pop, fx, fy, pool);
                double rawBest = pop.stream().mapToDouble(g -> g.fitness).max().orElse(0);
                double rawMean = pop.stream().mapToDouble(g -> g.fitness).average().orElse(0);

                List<List<NeatGenome>> species = speciate(pop, cfg.speciesThreshold());
                for (List<NeatGenome> sp : species) {
                    int size = sp.size();
                    for (NeatGenome g : sp) g.fitness /= size;
                }

                List<NeatGenome> newPop = new ArrayList<>();
                for (List<NeatGenome> sp : species) {
                    sp.sort((a, b) -> Double.compare(b.fitness, a.fitness));
                    newPop.add(sp.get(0).copy());
                }
                while (newPop.size() < cfg.popSize()) {
                    List<NeatGenome> sp = species.get(rng.nextInt(species.size()));
                    int half = Math.max(1, sp.size() / 2);
                    NeatGenome parent = sp.get(rng.nextInt(half));
                    newPop.add(parent.mutate(rng, cfg));
                }
                pop = newPop.size() > cfg.popSize() ? newPop.subList(0, cfg.popSize()) : newPop;

                if (logger != null) {
                    // raw (pre fitness-sharing) accuracy - directly comparable across
                    // generations, unlike the shared value used internally for selection
                    logger.row("NEAT", run, episode, gen, rawBest, rawMean, species.size());
                }
                if (gen == cfg.nGen() - 1) {
                    evaluateAllParallel(pop, fx, fy, pool);
                    pop.sort((a, b) -> Double.compare(b.fitness, a.fitness));
                    System.out.printf("      final best=%.1f%% species=%d hidden_nodes=%d%n",
                            pop.get(0).fitness * 100, species.size(), pop.get(0).hidden.size());
                }
            }
        } finally {
            pool.shutdown();
        }

        pop.sort((a, b) -> Double.compare(b.fitness, a.fitness));
        List<NeatGenome> carry = new ArrayList<>();
        for (int i = 0; i < Math.min(cfg.carryoverK(), pop.size()); i++) carry.add(pop.get(i));
        return new TaskResult(pop, carry);
    }

    public static List<List<List<Double>>> runNeat(Task[] tasks, int nIn, Config.NEATConfig cfg, int nRuns,
                                                     CsvLogger logger) {
        System.out.println("\n" + "=".repeat(55));
        System.out.println("CONDITION 3: NEAT EXTENSION");
        System.out.println("=".repeat(55));
        List<List<List<Double>>> allRuns = new ArrayList<>();
        for (int run = 0; run < nRuns; run++) {
            Random rng = new Random((long) run * 555 + 3);
            List<NeatGenome> carry = new ArrayList<>();
            Map<Integer, NeatGenome> bgs = new HashMap<>();
            List<List<Double>> mat = new ArrayList<>();
            System.out.println("Run " + (run + 1) + "/" + nRuns);
            for (int ep = 0; ep < tasks.length; ep++) {
                System.out.println("  Ep" + (ep + 1) + " " + tasks[ep].label() + "...");
                TaskResult r = neatTask(tasks[ep].trainX(), tasks[ep].trainY(), carry, rng, nIn, cfg, logger, run, ep);
                carry = r.carryover();
                bgs.put(ep, r.population().get(0));

                List<Double> epAccs = new ArrayList<>();
                for (int ti = 0; ti <= ep; ti++) {
                    epAccs.add(bgs.get(ti).evaluate(tasks[ti].testX(), tasks[ti].testY()));
                }
                mat.add(epAccs);
                System.out.println("  -> " + Fmt.pct(epAccs));
            }
            allRuns.add(mat);
        }
        return allRuns;
    }

    private static void evaluateAllParallel(List<NeatGenome> pop, double[][] fx, double[] fy, ExecutorService pool) {
        List<Future<?>> futures = new ArrayList<>(pop.size());
        for (NeatGenome g : pop) {
            futures.add(pool.submit(() -> g.fitness = g.evaluate(fx, fy)));
        }
        for (Future<?> f : futures) {
            try {
                f.get();
            } catch (InterruptedException | ExecutionException e) {
                throw new RuntimeException(e);
            }
        }
    }

    private static List<List<NeatGenome>> speciate(List<NeatGenome> pop, double threshold) {
        List<List<NeatGenome>> species = new ArrayList<>();
        for (NeatGenome g : pop) {
            boolean placed = false;
            for (List<NeatGenome> sp : species) {
                if (g.compatibility(sp.get(0)) < threshold) {
                    sp.add(g);
                    placed = true;
                    break;
                }
            }
            if (!placed) {
                List<NeatGenome> sp = new ArrayList<>();
                sp.add(g);
                species.add(sp);
            }
        }
        return species;
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

    private Neat() {}
}

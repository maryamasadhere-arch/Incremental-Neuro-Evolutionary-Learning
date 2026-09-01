package inel.ea;

import inel.Config;
import org.junit.jupiter.api.Test;
import java.util.*;
import static org.junit.jupiter.api.Assertions.*;

class EATest {

    private static double[][] X;
    private static double[] y;

    private static void makeLinearlySeparable(int n, int d, long seed) {
        Random rng = new Random(seed);
        double[] w = new double[d];
        for (int i = 0; i < d; i++) w[i] = rng.nextGaussian();
        X = new double[n][d];
        y = new double[n];
        for (int i = 0; i < n; i++) {
            double dot = 0;
            for (int j = 0; j < d; j++) {
                X[i][j] = rng.nextGaussian();
                dot += X[i][j] * w[j];
            }
            y[i] = dot > 0 ? 1.0 : 0.0;
        }
    }

    @Test
    void genomeLenMatchesFormula() {
        assertEquals((4 + 1) * 3 + (3 + 1), EA.genomeLen(4, 3));
    }

    @Test
    void nativeAndJavaFitnessAgree() {
        int nIn = 5, nH = 3, popSize = 6;
        makeLinearlySeparable(30, nIn, 7);
        int G = EA.genomeLen(nIn, nH);
        Random rng = new Random(0);
        double[] pop = new double[popSize * G];
        for (int i = 0; i < pop.length; i++) pop[i] = rng.nextGaussian();
        double[] flatX = new double[X.length * nIn];
        for (int i = 0; i < X.length; i++) System.arraycopy(X[i], 0, flatX, i * nIn, nIn);

        double[] outJava = new double[popSize];
        NativeFitness.evalPopulationJava(pop, popSize, nIn, nH, flatX, X.length, y, outJava);

        assumeTrueOrSkip(NativeFitness.isNativeAvailable());
        double[] outNative = new double[popSize];
        NativeFitness.evalPopulationNative(pop, popSize, nIn, nH, flatX, X.length, y, outNative);
        assertArrayEquals(outJava, outNative, 1e-9);
    }

    private static void assumeTrueOrSkip(boolean cond) {
        org.junit.jupiter.api.Assumptions.assumeTrue(cond, "native fitness library not built; skipping parity check");
    }

    @Test
    void eaTaskImprovesOverRandomInit() {
        makeLinearlySeparable(150, 6, 1);
        Config.EAConfig cfg = new Config.EAConfig(4, 12, 12, 0.2, 15, 2, 150);
        EA.TaskResult r = EA.eaTask(X, y, List.of(), 0, 6, cfg);
        assertTrue(r.fitness()[0] >= 0.6, "expected clear improvement over chance, got " + r.fitness()[0]);
        assertEquals(2, r.carryover().size());
    }

    @Test
    void bestFitnessNeverRegressesAcrossEpisodes() {
        makeLinearlySeparable(150, 6, 1);
        Config.EAConfig cfg = new Config.EAConfig(4, 10, 10, 0.2, 20, 3, 150);
        EA.TaskResult r1 = EA.eaTask(X, y, List.of(), 0, 6, cfg);
        double bestBefore = 0;
        for (double[] g : r1.carryover()) {
            bestBefore = Math.max(bestBefore, EA.evalOne(g, X, y, 6, cfg.hiddenUnits())[0]);
        }
        EA.TaskResult r2 = EA.eaTask(X, y, r1.carryover(), 1, 6, cfg);
        assertTrue(r2.fitness()[0] >= bestBefore - 1e-9);
    }
}

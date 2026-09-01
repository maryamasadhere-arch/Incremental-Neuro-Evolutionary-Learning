package inel.neat;

import inel.Config;
import org.junit.jupiter.api.Test;
import java.util.*;
import static org.junit.jupiter.api.Assertions.*;

class NeatTest {

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
    void sharedInnovationTrackerGivesMatchingIdsForMatchingStructure() {
        InnovationTracker innovations = new InnovationTracker();
        Random rng = new Random(0);
        NeatGenome a = new NeatGenome(8, rng, innovations, 2);
        NeatGenome b = new NeatGenome(8, rng, innovations, 2);

        int hiddenA = a.hidden.get(0);
        int hiddenB = b.hidden.get(0);
        assertEquals(hiddenA, hiddenB, "fresh genomes number their hidden nodes independently, starting from nIn+1");

        int keyA = innovations.get(1, hiddenA);
        int keyB = innovations.get(1, hiddenB);
        assertEquals(keyA, keyB, "the same (src,tgt) pair must get the same innovation number");
    }

    @Test
    void activateOutputIsInZeroOneRange() {
        InnovationTracker innovations = new InnovationTracker();
        NeatGenome g = new NeatGenome(8, new Random(0), innovations, 4);
        double[][] X = new double[15][8];
        Random rng = new Random(1);
        for (double[] row : X) for (int i = 0; i < row.length; i++) row[i] = rng.nextGaussian();
        double[] out = g.activate(X);
        assertEquals(15, out.length);
        for (double v : out) assertTrue(v >= 0 && v <= 1);
    }

    @Test
    void addNodePreservesDisabledSourceAndAddsTwoConnections() {
        InnovationTracker innovations = new InnovationTracker();
        NeatGenome g = new NeatGenome(8, new Random(1), innovations, 2);
        int connsBefore = g.conns.size();
        int hiddenBefore = g.hidden.size();
        g.addNode(new Random(1));
        assertEquals(hiddenBefore + 1, g.hidden.size());
        assertEquals(connsBefore + 2, g.conns.size());
        long disabled = g.conns.values().stream().filter(c -> !c.enabled).count();
        assertEquals(1, disabled);
    }

    @Test
    void compatibilityZeroForIdenticalGenome() {
        InnovationTracker innovations = new InnovationTracker();
        NeatGenome g = new NeatGenome(8, new Random(2), innovations, 3);
        NeatGenome g2 = g.copy();
        assertEquals(0.0, g.compatibility(g2));
    }

    @Test
    void neatTaskLearnsAboveChance() {
        makeLinearlySeparable(150, 8, 0);
        Config.NEATConfig cfg = new Config.NEATConfig(3, 12, 15, 0.08, 0.04, 0.05, 3.0, 2, 100);
        Neat.TaskResult r = Neat.neatTask(X, y, List.of(), new Random(0), 8, cfg);
        double bestAcc = r.population().get(0).evaluate(X, y);
        assertTrue(bestAcc >= 0.6, "expected clear improvement over chance, got " + bestAcc);
        assertEquals(2, r.carryover().size());
    }
}

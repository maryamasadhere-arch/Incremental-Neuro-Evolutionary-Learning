package inel;

import org.junit.jupiter.api.Test;
import java.util.Random;
import static org.junit.jupiter.api.Assertions.*;

class BackpropNetTest {

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
    void accuracyIsAFraction() {
        makeLinearlySeparable(50, 6, 0);
        BackpropNet net = new BackpropNet(6, new Config.BaselineConfig(new int[]{4}, 1, 16, 0.01, 0.9), 0);
        double acc = net.accuracy(X, y);
        assertTrue(acc >= 0.0 && acc <= 1.0);
    }

    @Test
    void trainingImprovesAccuracyOnSeparableData() {
        makeLinearlySeparable(400, 6, 0);
        Config.BaselineConfig cfg = new Config.BaselineConfig(new int[]{8}, 1, 32, 0.1, 0.9);
        BackpropNet net = new BackpropNet(6, cfg, 1);
        double before = net.accuracy(X, y);
        for (int i = 0; i < 20; i++) net.trainEpoch(X, y);
        double after = net.accuracy(X, y);
        assertTrue(after > before, "expected accuracy to improve: before=" + before + " after=" + after);
        assertTrue(after > 0.8, "expected strong fit on separable data, got " + after);
    }

    @Test
    void multiLayerNetworkTrainsWithoutError() {
        makeLinearlySeparable(200, 5, 2);
        Config.BaselineConfig cfg = new Config.BaselineConfig(new int[]{6, 4}, 1, 16, 0.05, 0.9);
        BackpropNet net = new BackpropNet(5, cfg, 2);
        for (int i = 0; i < 10; i++) net.trainEpoch(X, y);
        double acc = net.accuracy(X, y);
        assertTrue(acc >= 0.0 && acc <= 1.0);
    }
}

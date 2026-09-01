package inel;

import java.util.*;

/**
 * Condition 1 - backpropagation baseline (report Sec. 3.3, Objective O2).
 * Feedforward network of arbitrary depth, trained sequentially task-by-task
 * via mini-batch SGD with momentum. Matches inel/models/backprop.py exactly
 * (same Xavier init, same manual backprop derivation).
 */
public final class BackpropNet {

    private final int[] sizes;
    private final double[][][] W; // W[l] is fanIn x fanOut
    private final double[][] b;
    private final double[][][] vW;
    private final double[][] vb;
    private final double lr, momentum;
    private final int batchSize;
    private final Random rng;

    public BackpropNet(int nIn, Config.BaselineConfig cfg, long seed) {
        int[] hidden = cfg.hiddenLayers();
        sizes = new int[hidden.length + 2];
        sizes[0] = nIn;
        for (int i = 0; i < hidden.length; i++) sizes[i + 1] = hidden[i];
        sizes[sizes.length - 1] = 1;

        Random init = new Random(seed);
        int L = sizes.length - 1;
        W = new double[L][][];
        b = new double[L][];
        vW = new double[L][][];
        vb = new double[L][];
        for (int l = 0; l < L; l++) {
            int fanIn = sizes[l], fanOut = sizes[l + 1];
            double lim = Math.sqrt(6.0 / (fanIn + fanOut));
            W[l] = new double[fanIn][fanOut];
            for (int i = 0; i < fanIn; i++)
                for (int j = 0; j < fanOut; j++)
                    W[l][i][j] = (init.nextDouble() * 2 - 1) * lim;
            b[l] = new double[fanOut];
            vW[l] = new double[fanIn][fanOut];
            vb[l] = new double[fanOut];
        }
        this.lr = cfg.lr();
        this.momentum = cfg.momentum();
        this.batchSize = cfg.batchSize();
        this.rng = new Random(seed + 1);
    }

    /** Forward pass; returns activations[0..L] where activations[0]=input, activations[L]=sigmoid output. */
    private double[][] forward(double[][] X) {
        int L = W.length;
        double[][] acts = X;
        double[][][] all = new double[L + 1][][];
        all[0] = X;
        for (int l = 0; l < L; l++) {
            int n = acts.length, fanOut = sizes[l + 1];
            double[][] next = new double[n][fanOut];
            boolean isLast = l == L - 1;
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < fanOut; j++) {
                    double z = b[l][j];
                    for (int k = 0; k < acts[i].length; k++) z += acts[i][k] * W[l][k][j];
                    next[i][j] = isLast ? Activations.sigmoid(z) : Activations.relu(z);
                }
            }
            all[l + 1] = next;
            acts = next;
        }
        this.lastActs = all;
        return acts;
    }

    private double[][][] lastActs; // stashed by forward() for use in trainEpoch's backward pass

    public double accuracy(double[][] X, double[] y) {
        double[][] out = forward(X);
        int correct = 0;
        for (int i = 0; i < X.length; i++) {
            boolean pred = out[i][0] >= 0.5;
            boolean actual = y[i] >= 0.5;
            if (pred == actual) correct++;
        }
        return (double) correct / X.length;
    }

    public void trainEpoch(double[][] X, double[] y) {
        int n = X.length;
        Integer[] idx = new Integer[n];
        for (int i = 0; i < n; i++) idx[i] = i;
        Collections.shuffle(Arrays.asList(idx), rng);

        for (int start = 0; start < n; start += batchSize) {
            int end = Math.min(start + batchSize, n);
            int bs = end - start;
            double[][] Xb = new double[bs][];
            double[] yb = new double[bs];
            for (int i = 0; i < bs; i++) {
                Xb[i] = X[idx[start + i]];
                yb[i] = y[idx[start + i]];
            }
            forward(Xb); // populates lastActs
            backwardAndUpdate(lastActs, yb, bs);
        }
    }

    private void backwardAndUpdate(double[][][] acts, double[] yb, int bs) {
        int L = W.length;
        double[][] delta = new double[bs][sizes[L]]; // dL/dz at the current layer
        for (int i = 0; i < bs; i++) delta[i][0] = acts[L][i][0] - yb[i]; // BCE+sigmoid derivative

        for (int l = L - 1; l >= 0; l--) {
            double[][] aPrev = acts[l];
            int fanIn = sizes[l], fanOut = sizes[l + 1];
            double[][] dW = new double[fanIn][fanOut];
            double[] db = new double[fanOut];
            for (int i = 0; i < bs; i++) {
                for (int j = 0; j < fanOut; j++) {
                    db[j] += delta[i][j] / bs;
                    for (int k = 0; k < fanIn; k++) dW[k][j] += aPrev[i][k] * delta[i][j] / bs;
                }
            }

            double[][] nextDelta = null;
            if (l > 0) {
                nextDelta = new double[bs][fanIn];
                for (int i = 0; i < bs; i++) {
                    for (int k = 0; k < fanIn; k++) {
                        double s = 0;
                        for (int j = 0; j < fanOut; j++) s += delta[i][j] * W[l][k][j];
                        // ReLU'(z_l) == 1 iff a_l > 0, since a_l = relu(z_l)
                        nextDelta[i][k] = acts[l][i][k] > 0 ? s : 0.0;
                    }
                }
            }

            for (int k = 0; k < fanIn; k++) {
                for (int j = 0; j < fanOut; j++) {
                    vW[l][k][j] = momentum * vW[l][k][j] - lr * dW[k][j];
                    W[l][k][j] += vW[l][k][j];
                }
            }
            for (int j = 0; j < fanOut; j++) {
                vb[l][j] = momentum * vb[l][j] - lr * db[j];
                b[l][j] += vb[l][j];
            }

            if (nextDelta != null) delta = nextDelta;
        }
    }

    /** Condition 1 orchestration (report Sec. 3.3, Objective O2): sequential
     * training over all tasks, evaluating retained accuracy on every task
     * seen so far after each episode. Matches inel/models/backprop.py::run_baseline. */
    public static List<List<List<Double>>> runBaseline(Task[] tasks, int nIn, Config.BaselineConfig cfg, int nRuns) {
        System.out.println("\n" + "=".repeat(55));
        System.out.println("CONDITION 1: BACKPROPAGATION BASELINE");
        System.out.println("=".repeat(55));
        List<List<List<Double>>> allRuns = new ArrayList<>();
        for (int run = 0; run < nRuns; run++) {
            BackpropNet net = new BackpropNet(nIn, cfg, run);
            List<List<Double>> mat = new ArrayList<>();
            System.out.println("Run " + (run + 1) + "/" + nRuns);
            for (int ep = 0; ep < tasks.length; ep++) {
                for (int e = 0; e < cfg.epochsPerTask(); e++) {
                    net.trainEpoch(tasks[ep].trainX(), tasks[ep].trainY());
                }
                List<Double> epAccs = new ArrayList<>();
                for (int ti = 0; ti <= ep; ti++) {
                    epAccs.add(net.accuracy(tasks[ti].testX(), tasks[ti].testY()));
                }
                mat.add(epAccs);
                System.out.println("  Ep" + (ep + 1) + ": " + Fmt.pct(epAccs));
            }
            allRuns.add(mat);
        }
        return allRuns;
    }
}

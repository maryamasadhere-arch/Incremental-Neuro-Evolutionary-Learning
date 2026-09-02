package inel.neat;

import inel.Activations;
import inel.Config;

import java.util.*;

/**
 * NEAT genome with variable topology for a fixed input dimensionality
 * (Stanley & Miikkulainen, 2002). Matches inel/models/neat.py::NEATGenome
 * exactly, including the same "minimal start" deviation documented there
 * (a small sparse hidden layer rather than a true zero-hidden-node start,
 * because at 784-dimensional input a zero-hidden genome would need many
 * generations of pure structural mutation before any hidden representation
 * exists at all to evaluate).
 */
public final class NeatGenome {
    public final int nIn;
    public final int out;
    public int nextNid;
    public List<Integer> hidden = new ArrayList<>();
    public Map<Integer, ConnGene> conns = new LinkedHashMap<>();
    public double fitness = 0.0;
    private final InnovationTracker innovations;

    /** Fresh random genome with a small sparse hidden layer. */
    public NeatGenome(int nIn, Random rng, InnovationTracker innovations, int nHid) {
        this.nIn = nIn;
        this.innovations = innovations;
        this.out = nIn;
        this.nextNid = nIn + 1;
        for (int h = 0; h < nHid; h++) {
            hidden.add(nextNid++);
        }
        for (int h : hidden) {
            for (int i = 0; i < nIn; i += 4) {
                conns.put(innovations.get(i, h), new ConnGene(i, h, rng.nextGaussian() * 0.3, true));
            }
            conns.put(innovations.get(-1, h), new ConnGene(-1, h, rng.nextGaussian() * 0.1, true));
            conns.put(innovations.get(h, out), new ConnGene(h, out, rng.nextGaussian() * 0.3, true));
        }
        conns.put(innovations.get(-1, out), new ConnGene(-1, out, rng.nextGaussian() * 0.1, true));
    }

    /** Bare genome for copy(); caller populates fields. */
    private NeatGenome(int nIn, int out, InnovationTracker innovations) {
        this.nIn = nIn;
        this.out = out;
        this.innovations = innovations;
    }

    public NeatGenome copy() {
        NeatGenome g = new NeatGenome(nIn, out, innovations);
        g.nextNid = nextNid;
        g.hidden = new ArrayList<>(hidden);
        g.conns = new LinkedHashMap<>();
        for (var e : conns.entrySet()) g.conns.put(e.getKey(), e.getValue().copy());
        g.fitness = fitness;
        return g;
    }

    /** Reattach this (typically carried-over) genome to a new episode's innovation registry. */
    public NeatGenome withInnovations(InnovationTracker newTracker) {
        NeatGenome g = new NeatGenome(nIn, out, newTracker);
        g.nextNid = nextNid;
        g.hidden = new ArrayList<>(hidden);
        g.conns = new LinkedHashMap<>();
        for (var e : conns.entrySet()) g.conns.put(e.getKey(), e.getValue().copy());
        g.fitness = fitness;
        return g;
    }

    public void addNode(Random rng) {
        List<Integer> enabledKeys = new ArrayList<>();
        for (var e : conns.entrySet()) if (e.getValue().enabled) enabledKeys.add(e.getKey());
        if (enabledKeys.isEmpty()) return;
        int k = enabledKeys.get(rng.nextInt(enabledKeys.size()));
        ConnGene c = conns.get(k);
        int s = c.src, t = c.tgt;
        double w = c.weight;
        c.enabled = false;
        int nid = nextNid++;
        hidden.add(nid);
        conns.put(innovations.get(s, nid), new ConnGene(s, nid, 1.0, true));
        conns.put(innovations.get(nid, t), new ConnGene(nid, t, w, true));
    }

    public void addConnection(Random rng) {
        List<Integer> srcs = new ArrayList<>();
        for (int i = 0; i < nIn; i += 2) srcs.add(i);
        srcs.add(-1);
        srcs.addAll(hidden);
        List<Integer> tgts = new ArrayList<>(hidden);
        tgts.add(out);

        Set<Long> existing = new HashSet<>();
        for (var c : conns.values()) existing.add(pairKey(c.src, c.tgt));

        for (int attempt = 0; attempt < 20; attempt++) {
            int s = srcs.get(rng.nextInt(srcs.size()));
            int t = tgts.get(rng.nextInt(tgts.size()));
            long pk = pairKey(s, t);
            if (!existing.contains(pk)) {
                conns.put(innovations.get(s, t), new ConnGene(s, t, rng.nextGaussian() * 0.3, true));
                return;
            }
        }
    }

    private static long pairKey(int a, int b) {
        return ((long) a << 32) ^ (b & 0xffffffffL);
    }

    public NeatGenome mutate(Random rng, Config.NEATConfig cfg) {
        NeatGenome g = copy();
        for (var c : g.conns.values()) {
            if (rng.nextDouble() < 0.9) {
                c.weight += rng.nextDouble() < 0.9 ? rng.nextGaussian() * cfg.weightSigma() : rng.nextGaussian();
            }
        }
        if (rng.nextDouble() < cfg.pAddNode()) g.addNode(rng);
        if (rng.nextDouble() < cfg.pAddConnection()) g.addConnection(rng);
        return g;
    }

    public double compatibility(NeatGenome other) {
        return compatibility(other, 1.0, 0.4);
    }

    public double compatibility(NeatGenome other, double c1, double c2) {
        Set<Integer> k1 = conns.keySet(), k2 = other.conns.keySet();
        Set<Integer> union = new HashSet<>(k1);
        union.addAll(k2);
        Set<Integer> inter = new HashSet<>(k1);
        inter.retainAll(k2);
        Set<Integer> symDiff = new HashSet<>(union);
        symDiff.removeAll(inter);

        double sumAbsDiff = 0;
        for (int k : inter) sumAbsDiff += Math.abs(conns.get(k).weight - other.conns.get(k).weight);
        int N = Math.max(Math.max(k1.size(), k2.size()), 1);
        double matchTerm = inter.isEmpty() ? 0.0 : sumAbsDiff / inter.size();
        return c1 * symDiff.size() / (double) N + c2 * matchTerm;
    }

    public double[] activate(double[][] X) {
        int n = X.length;
        Map<Integer, double[]> vals = new HashMap<>();
        for (int i = 0; i < nIn; i++) {
            double[] col = new double[n];
            for (int r = 0; r < n; r++) col[r] = X[r][i];
            vals.put(i, col);
        }
        double[] ones = new double[n];
        Arrays.fill(ones, 1.0);
        vals.put(-1, ones);

        List<Integer> order = new ArrayList<>(hidden);
        order.add(out);
        for (int node : order) {
            double[] z = new double[n];
            boolean hasIncoming = false;
            for (var c : conns.values()) {
                if (c.tgt == node && c.enabled && vals.containsKey(c.src)) {
                    hasIncoming = true;
                    double[] src = vals.get(c.src);
                    for (int r = 0; r < n; r++) z[r] += c.weight * src[r];
                }
            }
            if (hasIncoming) {
                double[] activated = new double[n];
                boolean isOut = node == out;
                for (int r = 0; r < n; r++) activated[r] = isOut ? Activations.sigmoid(z[r]) : Activations.relu(z[r]);
                vals.put(node, activated);
            } else if (!vals.containsKey(node)) {
                vals.put(node, new double[n]);
            }
        }
        return vals.getOrDefault(out, new double[n]);
    }

    public double evaluate(double[][] X, double[] y) {
        double[] out = activate(X);
        int correct = 0;
        for (int i = 0; i < X.length; i++) {
            boolean pred = out[i] >= 0.5;
            boolean actual = y[i] >= 0.5;
            if (pred == actual) correct++;
        }
        return (double) correct / X.length;
    }
}

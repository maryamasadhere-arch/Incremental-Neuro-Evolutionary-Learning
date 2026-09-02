package inel.neat;

/** A single NEAT connection gene: source node id (-1 = bias), target node id, weight, enabled flag. */
public final class ConnGene {
    public int src, tgt;
    public double weight;
    public boolean enabled;

    public ConnGene(int src, int tgt, double weight, boolean enabled) {
        this.src = src;
        this.tgt = tgt;
        this.weight = weight;
        this.enabled = enabled;
    }

    public ConnGene copy() {
        return new ConnGene(src, tgt, weight, enabled);
    }
}

package inel;

/** Shared activation functions, matching inel/activations.py. */
public final class Activations {

    public static double sigmoid(double x) {
        double bound = 500.0; // Java doubles are 64-bit; no float32 overflow concern here
        double c = Math.max(-bound, Math.min(bound, x));
        return 1.0 / (1.0 + Math.exp(-c));
    }

    public static double relu(double x) {
        return Math.max(0.0, x);
    }

    private Activations() {}
}

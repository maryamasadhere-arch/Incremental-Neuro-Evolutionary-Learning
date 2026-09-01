package inel.ea;

import java.nio.file.*;

/**
 * The "use C efficiently" piece: batched fitness evaluation of an entire
 * fixed-topology (nIn -> nH -> 1) population against a batch of samples is
 * a dense numeric hot loop with no branching or dynamic structure - exactly
 * the kind of code a native kernel earns its keep on (this is the same
 * operation inel/models/ea.py vectorises with a NumPy einsum). The native
 * library is optional: if it isn't built or fails to load, evalPopulation
 * transparently falls back to the pure-Java implementation below, which is
 * semantically identical (see EATest for a parity test between the two).
 */
public final class NativeFitness {
    private static volatile boolean nativeAvailable = false;

    static {
        try {
            String libName = System.mapLibraryName("fitness");
            for (Path candidate : new Path[]{
                    Paths.get("target", "native", libName),
                    Paths.get("java", "target", "native", libName),
            }) {
                if (Files.exists(candidate)) {
                    System.load(candidate.toAbsolutePath().toString());
                    nativeAvailable = true;
                    break;
                }
            }
        } catch (Throwable t) {
            nativeAvailable = false;
        }
    }

    public static boolean isNativeAvailable() {
        return nativeAvailable;
    }

    /**
     * Genome layout per individual (matches inel/models/ea.py::eval_population):
     * [0, nIn*nH)                  W1, row-major (nIn, nH): W1[k][j] at k*nH+j
     * [nIn*nH, (nIn+1)*nH)         b1[j] at nIn*nH+j
     * [(nIn+1)*nH, (nIn+1)*nH+nH)  W2[j] at (nIn+1)*nH+j
     * (nIn+1)*nH+nH                b2 (single value)
     */
    public static native void evalPopulationNative(double[] flatPop, int popSize, int nIn, int nH,
                                                     double[] flatX, int n, double[] y, double[] outFits);

    public static void evalPopulationJava(double[] flatPop, int popSize, int nIn, int nH,
                                           double[] flatX, int n, double[] y, double[] outFits) {
        int genomeLen = (nIn + 1) * nH + (nH + 1);
        double[] h = new double[nH];
        for (int p = 0; p < popSize; p++) {
            long base = (long) p * genomeLen;
            int correct = 0;
            for (int i = 0; i < n; i++) {
                long xBase = (long) i * nIn;
                for (int j = 0; j < nH; j++) {
                    double z = flatPop[(int) (base + (long) nIn * nH + j)];
                    for (int k = 0; k < nIn; k++) {
                        z += flatX[(int) (xBase + k)] * flatPop[(int) (base + (long) k * nH + j)];
                    }
                    h[j] = Math.max(0.0, z);
                }
                double z2 = flatPop[(int) (base + (long) (nIn + 1) * nH + nH)];
                for (int j = 0; j < nH; j++) z2 += h[j] * flatPop[(int) (base + (long) (nIn + 1) * nH + j)];
                double c = Math.max(-500.0, Math.min(500.0, z2));
                double out = 1.0 / (1.0 + Math.exp(-c));
                boolean pred = out >= 0.5;
                boolean actual = y[i] >= 0.5;
                if (pred == actual) correct++;
            }
            outFits[p] = (double) correct / n;
        }
    }

    public static void evalPopulation(double[] flatPop, int popSize, int nIn, int nH,
                                       double[] flatX, int n, double[] y, double[] outFits) {
        if (nativeAvailable) {
            evalPopulationNative(flatPop, popSize, nIn, nH, flatX, n, y, outFits);
        } else {
            evalPopulationJava(flatPop, popSize, nIn, nH, flatX, n, y, outFits);
        }
    }

    private NativeFitness() {}
}

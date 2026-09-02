/*
 * Native fitness kernel for inel.ea.NativeFitness.evalPopulationNative.
 *
 * Evaluates classification accuracy of an entire (nIn -> nH -> 1) fixed-
 * topology population against a batch of samples. This is the same dense,
 * branch-free numeric loop that inel/models/ea.py offloads to NumPy's
 * vectorised einsum on the Python side; here it's the "use C efficiently"
 * hot path referenced in the project report's Sec. 3.9 discussion of
 * Java's performance overhead relative to C++.
 *
 * Genome layout per individual (must match NativeFitness.java's doc comment
 * and inel/models/ea.py::eval_population exactly):
 *   [0, nIn*nH)                 W1, row-major (nIn, nH): W1[k][j] at k*nH+j
 *   [nIn*nH, (nIn+1)*nH)        b1[j] at nIn*nH+j
 *   [(nIn+1)*nH, (nIn+1)*nH+nH) W2[j] at (nIn+1)*nH+j
 *   (nIn+1)*nH+nH               b2 (single value)
 */
#include <jni.h>
#include <stdlib.h>
#include <math.h>

JNIEXPORT void JNICALL Java_inel_ea_NativeFitness_evalPopulationNative(
        JNIEnv *env, jclass cls,
        jdoubleArray flatPopArr, jint popSize, jint nIn, jint nH,
        jdoubleArray flatXArr, jint n, jdoubleArray yArr, jdoubleArray outFitsArr) {

    jdouble *pop = (*env)->GetDoubleArrayElements(env, flatPopArr, NULL);
    jdouble *X = (*env)->GetDoubleArrayElements(env, flatXArr, NULL);
    jdouble *y = (*env)->GetDoubleArrayElements(env, yArr, NULL);
    jdouble *fits = (*env)->GetDoubleArrayElements(env, outFitsArr, NULL);

    if (pop == NULL || X == NULL || y == NULL || fits == NULL) {
        if (pop) (*env)->ReleaseDoubleArrayElements(env, flatPopArr, pop, JNI_ABORT);
        if (X) (*env)->ReleaseDoubleArrayElements(env, flatXArr, X, JNI_ABORT);
        if (y) (*env)->ReleaseDoubleArrayElements(env, yArr, y, JNI_ABORT);
        if (fits) (*env)->ReleaseDoubleArrayElements(env, outFitsArr, fits, JNI_ABORT);
        return;
    }

    const long genomeLen = (long) (nIn + 1) * nH + (nH + 1);
    double *h = (double *) malloc(sizeof(double) * (size_t) nH);

    for (int p = 0; p < popSize; p++) {
        long base = (long) p * genomeLen;
        int correct = 0;
        for (int i = 0; i < n; i++) {
            long xBase = (long) i * nIn;
            for (int j = 0; j < nH; j++) {
                double z = pop[base + (long) nIn * nH + j];
                for (int k = 0; k < nIn; k++) {
                    z += X[xBase + k] * pop[base + (long) k * nH + j];
                }
                h[j] = z > 0.0 ? z : 0.0;
            }
            double z2 = pop[base + (long) (nIn + 1) * nH + nH];
            for (int j = 0; j < nH; j++) {
                z2 += h[j] * pop[base + (long) (nIn + 1) * nH + j];
            }
            double c = z2 < -500.0 ? -500.0 : (z2 > 500.0 ? 500.0 : z2);
            double out = 1.0 / (1.0 + exp(-c));
            int pred = out >= 0.5;
            int actual = y[i] >= 0.5;
            if (pred == actual) correct++;
        }
        fits[p] = (double) correct / (double) n;
    }

    free(h);
    (*env)->ReleaseDoubleArrayElements(env, flatPopArr, pop, JNI_ABORT);
    (*env)->ReleaseDoubleArrayElements(env, flatXArr, X, JNI_ABORT);
    (*env)->ReleaseDoubleArrayElements(env, yArr, y, JNI_ABORT);
    (*env)->ReleaseDoubleArrayElements(env, outFitsArr, fits, 0);
}

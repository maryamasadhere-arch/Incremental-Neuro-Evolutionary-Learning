package inel;

/** One Split-MNIST binary-classification task episode. */
public record Task(String label, double[][] trainX, double[] trainY,
                    double[][] testX, double[] testY) {}

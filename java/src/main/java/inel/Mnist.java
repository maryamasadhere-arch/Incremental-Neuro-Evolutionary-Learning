package inel;

import java.io.*;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.file.*;
import java.time.Duration;
import java.util.*;
import java.util.zip.GZIPInputStream;

/**
 * Split-MNIST benchmark construction (report Sec. 2.4, 3.2), matching
 * inel/data.py exactly. Downloads (or reuses a cache shared with the
 * Python implementation) real MNIST and splits it into 5 sequential
 * binary tasks: 0v1, 2v3, 4v5, 6v7, 8v9. A synthetic offline fallback
 * (linearly separable per task) is provided for quick/test use.
 */
public final class Mnist {

    private static final String BASE_URL = "https://raw.githubusercontent.com/fgnt/mnist/master/";
    private static final Map<String, String> FILES = Map.of(
            "train-images", "train-images-idx3-ubyte.gz",
            "train-labels", "train-labels-idx1-ubyte.gz",
            "test-images", "t10k-images-idx3-ubyte.gz",
            "test-labels", "t10k-labels-idx1-ubyte.gz"
    );
    private static final int[][] DIGIT_PAIRS = {{0, 1}, {2, 3}, {4, 5}, {6, 7}, {8, 9}};

    public static void download(Path dataDir, int retries) throws IOException {
        Files.createDirectories(dataDir);
        HttpClient client = HttpClient.newBuilder().connectTimeout(Duration.ofSeconds(30)).build();
        for (var entry : FILES.entrySet()) {
            Path out = dataDir.resolve(entry.getKey());
            if (Files.exists(out)) continue;
            String url = BASE_URL + entry.getValue();
            Exception last = null;
            for (int attempt = 1; attempt <= retries; attempt++) {
                try {
                    System.out.println("  Downloading " + entry.getKey() + " (attempt " + attempt + "/" + retries + ")...");
                    HttpRequest req = HttpRequest.newBuilder(URI.create(url))
                            .header("User-Agent", "Mozilla/5.0")
                            .timeout(Duration.ofSeconds(60))
                            .build();
                    HttpResponse<byte[]> resp = client.send(req, HttpResponse.BodyHandlers.ofByteArray());
                    byte[] raw;
                    try (var gz = new GZIPInputStream(new ByteArrayInputStream(resp.body()))) {
                        raw = gz.readAllBytes();
                    }
                    Files.write(out, raw);
                    System.out.println("  Saved " + raw.length + " bytes");
                    last = null;
                    break;
                } catch (Exception e) {
                    last = e;
                    if (attempt < retries) {
                        try { Thread.sleep((long) Math.pow(2, attempt) * 1000L); } catch (InterruptedException ignored) {}
                    }
                }
            }
            if (last != null) {
                throw new IOException("Could not download " + entry.getKey() + " after " + retries + " attempts", last);
            }
        }
    }

    public static double[][] readImages(Path path) throws IOException {
        try (DataInputStream in = new DataInputStream(new BufferedInputStream(Files.newInputStream(path)))) {
            in.readInt(); // magic
            int n = in.readInt();
            int rows = in.readInt();
            int cols = in.readInt();
            int sz = rows * cols;
            byte[] buf = in.readAllBytes();
            double[][] out = new double[n][sz];
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < sz; j++) {
                    out[i][j] = (buf[i * sz + j] & 0xFF) / 255.0;
                }
            }
            return out;
        }
    }

    public static int[] readLabels(Path path) throws IOException {
        try (DataInputStream in = new DataInputStream(new BufferedInputStream(Files.newInputStream(path)))) {
            in.readInt(); // magic
            int n = in.readInt();
            byte[] buf = in.readAllBytes();
            int[] out = new int[n];
            for (int i = 0; i < n; i++) out[i] = buf[i] & 0xFF;
            return out;
        }
    }

    private static double[][] project(double[][] x, double[][] proj) {
        if (proj == null) return x;
        int n = x.length, dOut = proj[0].length, dIn = proj.length;
        double[][] out = new double[n][dOut];
        for (int i = 0; i < n; i++) {
            for (int k = 0; k < dOut; k++) {
                double s = 0;
                for (int j = 0; j < dIn; j++) s += x[i][j] * proj[j][k];
                out[i][k] = s;
            }
        }
        return out;
    }

    public static Task[] buildSplitMnist(Path dataDir, Config.DataConfig cfg) throws IOException {
        download(dataDir, 3);
        double[][] trX = readImages(dataDir.resolve("train-images"));
        int[] trY = readLabels(dataDir.resolve("train-labels"));
        double[][] teX = readImages(dataDir.resolve("test-images"));
        int[] teY = readLabels(dataDir.resolve("test-labels"));

        double[][] proj = null;
        if (cfg.projectDims() > 0) {
            Random rng = new Random(cfg.seed());
            double scale = 1.0 / Math.sqrt(cfg.projectDims());
            proj = new double[784][cfg.projectDims()];
            for (int i = 0; i < 784; i++)
                for (int j = 0; j < cfg.projectDims(); j++)
                    proj[i][j] = rng.nextGaussian() * scale;
        }

        Task[] tasks = new Task[DIGIT_PAIRS.length];
        for (int t = 0; t < DIGIT_PAIRS.length; t++) {
            int a = DIGIT_PAIRS[t][0], b = DIGIT_PAIRS[t][1];
            List<Integer> trIdx = new ArrayList<>(), teIdx = new ArrayList<>();
            List<Double> trLab = new ArrayList<>(), teLab = new ArrayList<>();
            for (int i = 0; i < trY.length; i++) {
                if (trY[i] == a) { trIdx.add(i); trLab.add(0.0); }
                else if (trY[i] == b) { trIdx.add(i); trLab.add(1.0); }
            }
            for (int i = 0; i < teY.length; i++) {
                if (teY[i] == a) { teIdx.add(i); teLab.add(0.0); }
                else if (teY[i] == b) { teIdx.add(i); teLab.add(1.0); }
            }
            shuffleTogether(trIdx, trLab, new Random(cfg.seed()));
            shuffleTogether(teIdx, teLab, new Random(cfg.seed()));

            double[][] trainX = project(gather(trX, trIdx), proj);
            double[][] testX = project(gather(teX, teIdx), proj);
            tasks[t] = new Task("Task(" + a + "v" + b + ")", trainX, toArray(trLab), testX, toArray(teLab));
            System.out.println("  Task(" + a + "v" + b + "): " + trainX.length + " train, " + testX.length + " test");
        }
        return tasks;
    }

    public static Task[] buildSyntheticTasks(Config.DataConfig cfg, int nTasks) {
        Random rng = new Random(cfg.seed());
        int d = cfg.nInputDims();
        Task[] tasks = new Task[nTasks];
        for (int t = 0; t < nTasks; t++) {
            double[] w = new double[d];
            double norm = 0;
            for (int i = 0; i < d; i++) { w[i] = rng.nextGaussian(); norm += w[i] * w[i]; }
            norm = Math.sqrt(norm);
            for (int i = 0; i < d; i++) w[i] /= norm;

            double[][] trainX = new double[cfg.syntheticNTrain()][d];
            double[] trainY = new double[cfg.syntheticNTrain()];
            fillSeparable(trainX, trainY, w, rng);
            double[][] testX = new double[cfg.syntheticNTest()][d];
            double[] testY = new double[cfg.syntheticNTest()];
            fillSeparable(testX, testY, w, rng);
            tasks[t] = new Task("SyntheticTask" + t, trainX, trainY, testX, testY);
        }
        return tasks;
    }

    private static void fillSeparable(double[][] X, double[] y, double[] w, Random rng) {
        int d = w.length;
        for (int i = 0; i < X.length; i++) {
            double dot = 0;
            for (int j = 0; j < d; j++) {
                X[i][j] = rng.nextGaussian();
                dot += X[i][j] * w[j];
            }
            y[i] = dot > 0 ? 1.0 : 0.0;
            for (int j = 0; j < d; j++) X[i][j] += rng.nextGaussian() * 0.05;
        }
    }

    public static Task[] loadTasks(Path dataDir, Config.DataConfig cfg, int nTasks) throws IOException {
        if (cfg.synthetic()) return buildSyntheticTasks(cfg, nTasks);
        return buildSplitMnist(dataDir, cfg);
    }

    private static double[][] gather(double[][] src, List<Integer> idx) {
        double[][] out = new double[idx.size()][];
        for (int i = 0; i < idx.size(); i++) out[i] = src[idx.get(i)];
        return out;
    }

    private static double[] toArray(List<Double> l) {
        double[] out = new double[l.size()];
        for (int i = 0; i < l.size(); i++) out[i] = l.get(i);
        return out;
    }

    private static void shuffleTogether(List<Integer> idx, List<Double> lab, Random rng) {
        for (int i = idx.size() - 1; i > 0; i--) {
            int j = rng.nextInt(i + 1);
            Integer ti = idx.get(i); idx.set(i, idx.get(j)); idx.set(j, ti);
            Double tl = lab.get(i); lab.set(i, lab.get(j)); lab.set(j, tl);
        }
    }

    private Mnist() {}
}

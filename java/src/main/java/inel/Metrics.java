package inel;

import java.util.*;

/**
 * Evaluation metrics (report Sec. 2.5, Objectives O4/O6): Retention
 * Accuracy (RA), Forgetting Rate (FR), Forward Transfer (FT), Evolvability
 * Ceiling (EC). Ports inel/metrics.py::compute_metrics exactly, including
 * the corrected EC definition (two distinct thresholds - 70% on prior
 * tasks, 85% on the current task - and stopping at the first failed
 * episode rather than the last passing one). See inel/metrics.py's
 * docstring/comments for the full rationale; kept identical here so the
 * Java and Python implementations can never silently diverge on this.
 *
 * `runs` is a list of per-run accuracy matrices: runs.get(r).get(ep).get(ti)
 * is the test accuracy on task ti after training episode ep, for ti <= ep.
 */
public record Metrics(String name,
                       List<Double> ra, double raMean,
                       List<Double> fr, double frMean,
                       List<Double> ft, double ftMean,
                       double ec, List<Integer> ecRuns) {

    public static Metrics compute(List<List<List<Double>>> runs, String name, int N,
                                   double ecPriorThreshold, double ecCurrentThreshold) {
        int n = runs.size();

        List<Double> ra = new ArrayList<>();
        for (int ti = 0; ti < N; ti++) {
            double sum = 0;
            int cnt = 0;
            for (List<List<Double>> run : runs) {
                List<Double> last = run.get(run.size() - 1);
                if (ti < last.size()) { sum += last.get(ti); cnt++; }
            }
            ra.add(round1(sum / cnt * 100));
        }
        double raMean = round1(ra.stream().mapToDouble(Double::doubleValue).sum() / N);

        List<Double> fr = new ArrayList<>();
        for (int ti = 0; ti < N - 1; ti++) {
            double sum = 0;
            for (List<List<Double>> run : runs) {
                double mx = Double.NEGATIVE_INFINITY;
                for (int ep = ti; ep < Math.min(N, run.size()); ep++) {
                    if (ti < run.get(ep).size()) mx = Math.max(mx, run.get(ep).get(ti));
                }
                List<Double> last = run.get(run.size() - 1);
                double fn = ti < last.size() ? last.get(ti) : 0.0;
                sum += Math.max(0, mx - fn);
            }
            fr.add(round1(sum / n * 100));
        }
        double frMean = fr.isEmpty() ? 0.0 : round1(fr.stream().mapToDouble(Double::doubleValue).sum() / fr.size());

        List<Double> ft = new ArrayList<>();
        for (int ti = 1; ti < N; ti++) {
            double sum = 0;
            int cnt = 0;
            for (List<List<Double>> run : runs) {
                if (ti < run.size() && ti < run.get(ti).size()) {
                    sum += run.get(ti).get(ti) * 100 - 50;
                    cnt++;
                }
            }
            if (cnt > 0) ft.add(round1(sum / cnt));
        }
        double ftMean = ft.isEmpty() ? 0.0 : round1(ft.stream().mapToDouble(Double::doubleValue).sum() / ft.size());

        List<Integer> ecRuns = new ArrayList<>();
        for (List<List<Double>> run : runs) {
            int s = 0;
            for (int ep = 0; ep < N; ep++) {
                List<Double> accs = run.get(ep);
                boolean priorOk = true;
                for (int i = 0; i < accs.size() - 1; i++) {
                    if (accs.get(i) < ecPriorThreshold) { priorOk = false; break; }
                }
                boolean currentOk = accs.get(accs.size() - 1) >= ecCurrentThreshold;
                if (priorOk && currentOk) {
                    s = ep + 1;
                } else {
                    break;
                }
            }
            ecRuns.add(s);
        }
        double ecMean = round1(ecRuns.stream().mapToInt(Integer::intValue).sum() / (double) n);

        return new Metrics(name, ra, raMean, fr, frMean, ft, ftMean, ecMean, ecRuns);
    }

    private static double round1(double v) {
        return Math.round(v * 10.0) / 10.0;
    }

    public static void printSummary(Metrics baseline, Metrics ea, Metrics neat) {
        System.out.println("\n" + "=".repeat(65));
        System.out.println("COMPARATIVE RESULTS  -  Objective O6 (Java implementation)");
        System.out.println("=".repeat(65));
        System.out.printf("%-32s%11s%11s%11s%n", "Metric", "Baseline", "2007 EA", "NEAT");
        System.out.println("-".repeat(65));
        System.out.printf("  %-30s%11s%11s%11s%n", "Mean RA (%)", baseline.raMean, ea.raMean, neat.raMean);
        System.out.printf("  %-30s%11s%11s%11s%n", "Mean FR (%)", baseline.frMean, ea.frMean, neat.frMean);
        System.out.printf("  %-30s%11s%11s%11s%n", "Mean FT (%)", baseline.ftMean, ea.ftMean, neat.ftMean);
        System.out.printf("  %-30s%11s%11s%11s%n", "EC (/5)", baseline.ec, ea.ec, neat.ec);
        System.out.println("-".repeat(65));
        System.out.printf("%n  EA vs Baseline RA gain:   %+.1f pp%n", ea.raMean - baseline.raMean);
        System.out.printf("  NEAT vs Baseline RA gain: %+.1f pp%n", neat.raMean - baseline.raMean);
        System.out.printf("  EA FR / NEAT FR: %s%% / %s%%%n", ea.frMean, neat.frMean);
        System.out.printf("  EA EC / NEAT EC: %s/5 / %s/5%n", ea.ec, neat.ec);
    }
}

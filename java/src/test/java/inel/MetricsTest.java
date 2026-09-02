package inel;

import org.junit.jupiter.api.Test;
import java.util.*;
import static org.junit.jupiter.api.Assertions.*;

/** Mirrors tests/test_metrics.py exactly, including the EC-fix regression tests. */
class MetricsTest {

    private static List<List<List<Double>>> singleRun(double[][] episodeRows) {
        List<List<Double>> episodes = new ArrayList<>();
        for (double[] row : episodeRows) {
            List<Double> accs = new ArrayList<>();
            for (double a : row) accs.add(a);
            episodes.add(accs);
        }
        return List.of(episodes);
    }

    @Test
    void raIsFinalEpisodeAccuracyPerTask() {
        var runs = singleRun(new double[][]{
                {0.95},
                {0.60, 0.96},
                {0.55, 0.70, 0.97},
        });
        Metrics m = Metrics.compute(runs, "x", 3, 0.70, 0.85);
        assertEquals(List.of(55.0, 70.0, 97.0), m.ra());
    }

    @Test
    void frExcludesFinalTaskAndUsesPeakMinusFinal() {
        var runs = singleRun(new double[][]{
                {0.95},
                {0.60, 0.80},
                {0.55, 0.70, 0.97},
        });
        Metrics m = Metrics.compute(runs, "x", 3, 0.70, 0.85);
        assertEquals(List.of(40.0, 10.0), m.fr());
    }

    @Test
    void ecRequiresSeparatePriorAndCurrentThresholds() {
        var runs = singleRun(new double[][]{{0.80}});
        Metrics m = Metrics.compute(runs, "x", 1, 0.70, 0.85);
        assertEquals(0.0, m.ec());
    }

    @Test
    void ecPassesWhenCurrentTaskClearsTheHigherBar() {
        var runs = singleRun(new double[][]{{0.90}});
        Metrics m = Metrics.compute(runs, "x", 1, 0.70, 0.85);
        assertEquals(1.0, m.ec());
    }

    @Test
    void ecStopsAtFirstFailureNotLastSuccess() {
        var runs = singleRun(new double[][]{
                {0.90},
                {0.50, 0.90},
                {0.95, 0.95, 0.90},
        });
        Metrics m = Metrics.compute(runs, "x", 3, 0.70, 0.85);
        assertEquals(1.0, m.ec());
    }

    @Test
    void ftExcludesFirstTaskAndMeasuresRelativeToChance() {
        var runs = singleRun(new double[][]{
                {0.95},
                {0.60, 0.80},
        });
        Metrics m = Metrics.compute(runs, "x", 2, 0.70, 0.85);
        assertEquals(List.of(30.0), m.ft());
    }
}

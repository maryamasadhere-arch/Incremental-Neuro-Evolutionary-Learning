package inel.neat;

import java.util.HashMap;
import java.util.Map;

/**
 * Shared historical-marking registry for one task episode's population.
 * Innovation numbers must be assigned consistently across the whole
 * population within an episode - two genomes that independently evolve the
 * same structural change need the same innovation number, or
 * NeatGenome.compatibility cannot tell that their genes actually match.
 */
public final class InnovationTracker {
    private final Map<Long, Integer> map = new HashMap<>();
    private int count = 0;

    private static long key(int src, int tgt) {
        return ((long) src << 32) ^ (tgt & 0xffffffffL);
    }

    public synchronized int get(int src, int tgt) {
        long k = key(src, tgt);
        return map.computeIfAbsent(k, kk -> ++count);
    }
}

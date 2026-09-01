package inel;

import java.util.List;

/** Small shared formatting helper used by each condition's console logging. */
public final class Fmt {
    public static String pct(List<Double> accs) {
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < accs.size(); i++) {
            if (i > 0) sb.append(", ");
            sb.append(String.format("%.1f%%", accs.get(i) * 100));
        }
        return sb.append("]").toString();
    }

    private Fmt() {}
}

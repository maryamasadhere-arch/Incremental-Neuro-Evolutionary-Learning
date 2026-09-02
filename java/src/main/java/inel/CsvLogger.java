package inel;

import java.io.*;
import java.nio.file.*;

/**
 * Custom CSV logger (report Sec. 3.6: "All experimental results are logged
 * to CSV files via a custom Java logger class"). Used per-generation by the
 * EA and NEAT conditions to record best/mean fitness and (NEAT only)
 * species count, matching the report's description.
 */
public final class CsvLogger implements Closeable {
    private final PrintWriter out;

    public CsvLogger(Path path, String header) throws IOException {
        if (path.getParent() != null) Files.createDirectories(path.getParent());
        out = new PrintWriter(new BufferedWriter(new FileWriter(path.toFile())));
        out.println(header);
    }

    public synchronized void row(Object... fields) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < fields.length; i++) {
            if (i > 0) sb.append(',');
            sb.append(fields[i]);
        }
        out.println(sb);
        out.flush();
    }

    @Override
    public void close() {
        out.close();
    }
}

package dps_llm.config;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import java.util.Objects;

/**
 * Minimalist .env file parser for loading configuration from environment files.
 * <p>
 * This utility class reads key-value pairs from .env files to supply API credentials
 * and configuration without committing sensitive data to version control. It supports
 * basic .env syntax including comments and blank lines.
 * </p>
 * <p>
 * Key responsibilities:
 * <ul>
 *   <li>Parse .env files in KEY=VALUE format</li>
 *   <li>Skip comments (lines starting with #) and blank lines</li>
 *   <li>Return an immutable map of configuration values</li>
 *   <li>Handle missing files gracefully by returning an empty map</li>
 * </ul>
 * </p>
 * 
 * @author Najam
 */
public final class DotEnvLoader {

    private DotEnvLoader() {
    }

    /**
     * Loads configuration key-value pairs from a .env file.
     * <p>
     * Lines starting with # are treated as comments. Blank lines are ignored.
     * Each configuration line must be in the format KEY=VALUE.
     * </p>
     * 
     * @param path the path to the .env file
     * @return a map of configuration keys to values, or empty map if file doesn't exist
     * @throws IOException if file reading fails
     * @throws NullPointerException if path is null
     */
    public static Map<String, String> load(Path path) throws IOException {
        Objects.requireNonNull(path, "path must not be null");
        if (!Files.exists(path)) {
            return Collections.emptyMap();
        }
        Map<String, String> values = new HashMap<>();
        for (String rawLine : Files.readAllLines(path, StandardCharsets.UTF_8)) {
            String line = rawLine.trim();
            if (line.isEmpty() || line.startsWith("#")) {
                continue;
            }
            int equalsIndex = line.indexOf('=');
            if (equalsIndex <= 0) {
                continue;
            }
            String key = line.substring(0, equalsIndex).trim();
            String value = line.substring(equalsIndex + 1).trim();
            values.put(key, value);
        }
        return values;
    }
}

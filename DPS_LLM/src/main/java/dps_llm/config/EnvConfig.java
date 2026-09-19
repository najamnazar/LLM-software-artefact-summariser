package dps_llm.config;

import java.util.Map;

/**
 * Reads required settings from the shared {@code .env}, with OS environment variables as an
 * override path and no compiled-in fallback values.
 * <p>
 * Every accessor here either returns the configured value or throws
 * {@link ConfigurationException}. There is deliberately no {@code getOrDefault} variant for
 * run-shaping settings: a default in Java is a second, invisible copy of a value that is supposed
 * to live in {@code .env}, and the two have silently disagreed before. An unparseable value is
 * treated the same way as a missing one rather than being swallowed with a console notice.
 * </p>
 *
 * @author Najam
 */
public final class EnvConfig {

    private EnvConfig() {
    }

    /**
     * Returns a setting, or null when it is absent or blank in both .env and the OS environment.
     *
     * @param dotEnv configuration map loaded from .env, may be null
     * @param key the setting name
     * @return the trimmed value, or null
     */
    public static String value(Map<String, String> dotEnv, String key) {
        String raw = dotEnv == null ? null : dotEnv.get(key);
        if (raw == null || raw.isBlank()) {
            raw = System.getenv(key);
        }
        return raw == null || raw.isBlank() ? null : raw.trim();
    }

    /**
     * Returns a required setting.
     *
     * @param dotEnv configuration map loaded from .env
     * @param key the setting name
     * @return the configured value
     * @throws ConfigurationException if the setting is absent or blank
     */
    public static String require(Map<String, String> dotEnv, String key) {
        String value = value(dotEnv, key);
        if (value == null) {
            throw new ConfigurationException(key + " is not set. Add it to the shared .env; "
                    + "this pipeline carries no built-in fallback for it.");
        }
        return value;
    }

    /**
     * Returns a required integer setting.
     *
     * @param dotEnv configuration map loaded from .env
     * @param key the setting name
     * @return the configured value
     * @throws ConfigurationException if the setting is absent, blank or not an integer
     */
    public static int requireInt(Map<String, String> dotEnv, String key) {
        return parseInt(key, require(dotEnv, key));
    }

    /**
     * Returns a required integer setting, accepting a superseded key name for older .env files.
     *
     * @param dotEnv configuration map loaded from .env
     * @param key the current setting name
     * @param legacyKey the superseded setting name, consulted only when key is absent
     * @return the configured value
     * @throws ConfigurationException if neither key is set, or the value is not an integer
     */
    public static int requireInt(Map<String, String> dotEnv, String key, String legacyKey) {
        String raw = value(dotEnv, key);
        if (raw != null) {
            return parseInt(key, raw);
        }
        String legacyRaw = value(dotEnv, legacyKey);
        if (legacyRaw != null) {
            System.out.println("Using deprecated " + legacyKey + "; rename it to " + key + " in .env.");
            return parseInt(legacyKey, legacyRaw);
        }
        throw new ConfigurationException(key + " is not set. Add it to the shared .env; "
                + "this pipeline carries no built-in fallback for it.");
    }

    /**
     * Returns a required floating-point setting.
     *
     * @param dotEnv configuration map loaded from .env
     * @param key the setting name
     * @return the configured value
     * @throws ConfigurationException if the setting is absent, blank or not a number
     */
    public static double requireDouble(Map<String, String> dotEnv, String key) {
        String raw = require(dotEnv, key);
        try {
            return Double.parseDouble(raw);
        } catch (NumberFormatException ex) {
            throw new ConfigurationException(key + " must be a number in .env, but is: " + raw);
        }
    }

    private static int parseInt(String key, String raw) {
        try {
            return Integer.parseInt(raw);
        } catch (NumberFormatException ex) {
            throw new ConfigurationException(key + " must be an integer in .env, but is: " + raw);
        }
    }
}

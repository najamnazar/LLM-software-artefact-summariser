package dps_llm.config;

import java.util.Map;

/**
 * How many fields, constructors, methods and pattern insights of a class reach the LLM prompt.
 * <p>
 * These four caps used to be {@code private static final int} constants inside
 * {@code ClassFeatureExtractor}, so the only record of them was the Java source: they appeared in
 * neither {@code .env} nor {@code prompts.json}, and changing one meant a rebuild. They are
 * configuration, not code, and they matter to the three-way comparison — DPS_NLG and DPS_SWUM
 * describe every method of a class, so any cap here gives the LLM strictly less to work with than
 * the two baselines see.
 * </p>
 * <p>
 * The keys are required rather than defaulted: a fallback here would be a second copy of a value
 * that belongs in {@code .env}, and this pipeline has twice run with a literal that had quietly
 * drifted from the configured one. {@code .env} documents the original 6/4/6/6.
 * </p>
 *
 * @author Najam
 */
public final class FeatureLimits {

    // Najam: the DEFAULT_* constants that first carried these values out of ClassFeatureExtractor
    // are gone. Keeping them here would have left the same four literals in the code, one package
    // further away from where they are used, still able to disagree with .env without anyone
    // noticing. The keys below are required; .env documents 6/4/6/6.
    // public static final int DEFAULT_FIELD_LIMIT = 6;
    // public static final int DEFAULT_CONSTRUCTOR_LIMIT = 4;
    // public static final int DEFAULT_METHOD_LIMIT = 6;
    // public static final int DEFAULT_PATTERN_LIMIT = 6;

    public static final String FIELD_LIMIT_KEY = "LLM_FEATURE_FIELD_LIMIT";
    public static final String CONSTRUCTOR_LIMIT_KEY = "LLM_FEATURE_CONSTRUCTOR_LIMIT";
    public static final String METHOD_LIMIT_KEY = "LLM_FEATURE_METHOD_LIMIT";
    public static final String PATTERN_LIMIT_KEY = "LLM_FEATURE_PATTERN_LIMIT";

    private final int fieldLimit;
    private final int constructorLimit;
    private final int methodLimit;
    private final int patternLimit;

    /**
     * @param fieldLimit maximum field signatures in the prompt; 0 or less means no cap
     * @param constructorLimit maximum constructor signatures; 0 or less means no cap
     * @param methodLimit maximum method summaries; 0 or less means no cap
     * @param patternLimit maximum design-pattern insight lines; 0 or less means no cap
     */
    public FeatureLimits(int fieldLimit, int constructorLimit, int methodLimit, int patternLimit) {
        this.fieldLimit = fieldLimit;
        this.constructorLimit = constructorLimit;
        this.methodLimit = methodLimit;
        this.patternLimit = patternLimit;
    }

    /**
     * Reads the caps from the shared {@code .env}.
     *
     * @param dotEnv configuration map loaded from .env
     * @return the configured limits
     * @throws ConfigurationException if any of the four keys is absent or not an integer
     */
    public static FeatureLimits fromEnv(Map<String, String> dotEnv) {
        return new FeatureLimits(
                EnvConfig.requireInt(dotEnv, FIELD_LIMIT_KEY),
                EnvConfig.requireInt(dotEnv, CONSTRUCTOR_LIMIT_KEY),
                EnvConfig.requireInt(dotEnv, METHOD_LIMIT_KEY),
                EnvConfig.requireInt(dotEnv, PATTERN_LIMIT_KEY));
    }

    public int getFieldLimit() {
        return fieldLimit;
    }

    public int getConstructorLimit() {
        return constructorLimit;
    }

    public int getMethodLimit() {
        return methodLimit;
    }

    public int getPatternLimit() {
        return patternLimit;
    }

    @Override
    public String toString() {
        return "fields=" + describe(fieldLimit)
                + ", constructors=" + describe(constructorLimit)
                + ", methods=" + describe(methodLimit)
                + ", patternInsights=" + describe(patternLimit);
    }

    private String describe(int limit) {
        return limit <= 0 ? "all" : String.valueOf(limit);
    }
}

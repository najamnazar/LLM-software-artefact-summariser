package dps_llm.config;

/**
 * Raised when a required setting is missing from {@code .env} or cannot be parsed.
 * <p>
 * Settings that control what is sent to the model — token budget, temperature, prompt feature
 * limits — are read from {@code .env} with no compiled-in fallback. A literal default in the code
 * duplicates the documented value and drifts from it: the token budget once fell back to 256 while
 * {@code .env} said 512, and the temperature fell back to 0.2 while {@code .env} said 0, in both
 * cases without a word on the console. Failing loudly on an absent key removes that whole class of
 * silent divergence.
 * </p>
 *
 * @author Najam
 */
public class ConfigurationException extends RuntimeException {

    private static final long serialVersionUID = 1L;

    public ConfigurationException(String message) {
        super(message);
    }
}

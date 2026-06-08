package dps_llm.client;

/**
 * Exception thrown when LLM API requests fail or cannot be fulfilled.
 * <p>
 * This exception encapsulates errors that occur during communication with the remote
 * LLM endpoint, including network failures, authentication errors, rate limiting,
 * and invalid API responses.
 * </p>
 * 
 * @author Najam
 */
public class LlmClientException extends Exception {

    /**
     * Constructs a new exception with the specified detail message.
     * 
     * @param message the detail message explaining the error
     */
    public LlmClientException(String message) {
        super(message);
    }

    /**
     * Constructs a new exception with the specified detail message and cause.
     * 
     * @param message the detail message explaining the error
     * @param cause the underlying cause of this exception
     */
    public LlmClientException(String message, Throwable cause) {
        super(message, cause);
    }
}

package dps_llm.summary;

/**
 * Thrown when a run stops because the LLM is failing continuously rather than for one bad class.
 * <p>
 * A single failed class is recorded and skipped so the rest of the corpus still gets summarised.
 * That is right for one class and wrong for a session-wide problem: on 2026-09-19 an expired
 * rate limit returned HTTP 429 for every request, and the per-class recovery dutifully walked the
 * corpus at roughly 105 seconds of backoff each, destroying the previous CSV on the way without
 * any prospect of producing a new one. Past LLM_MAX_CONSECUTIVE_FAILURES failures in a row the run
 * gives up instead, and the output file is left as it was.
 * </p>
 *
 * @author Najam
 */
public class RunAbortedException extends RuntimeException {

    private static final long serialVersionUID = 1L;

    /**
     * @param message what stopped the run, including the failure that triggered it
     */
    public RunAbortedException(String message) {
        super(message);
    }
}

package dps_llm.summary;

import dps_llm.client.LlmClient;
import dps_llm.config.FeatureLimits;
import dps_llm.client.LlmClientException;
import dps_llm.model.ClassFeatureSnapshot;
import dps_llm.prompt.LlmPromptBuilder;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;

/**
 * Coordinates the LLM-based summarization process for parsed projects.
 * <p>
 * This service orchestrates feature extraction, prompt construction, and remote LLM
 * API calls to generate natural language summaries of Java classes. It integrates
 * design pattern information and handles the complete summarization workflow.
 * </p>
 * <p>
 * Key responsibilities:
 * <ul>
 *   <li>Coordinate feature extraction for all classes in a project</li>
 *   <li>Build pattern-aware insights from DPS analysis</li>
 *   <li>Generate prompts and call the LLM API</li>
 *   <li>Write summaries to CSV output</li>
 *   <li>Track and report processing statistics</li>
 * </ul>
 * </p>
 * 
 * @author Najam
 */
@SuppressWarnings({"rawtypes", "unchecked"})
public class LlmSummaryService {

    private final ClassFeatureExtractor extractor;
    // private final LlmPromptBuilder promptBuilder = new LlmPromptBuilder(); // Original default prompt builder kept for reference
    private final LlmPromptBuilder promptBuilder; // Allows configuring prompt alias per run
    private final LlmClient llmClient;
    /** Consecutive failures that stop the run; 0 disables the check. From LLM_MAX_CONSECUTIVE_FAILURES. */
    private final int maxConsecutiveFailures;
    /** Runs across projects, not just within one: a rate limit does not respect directory boundaries. */
    private int consecutiveFailures;

    // Najam: these two convenience constructors are gone along with FeatureLimits.defaults(). Both
    // only existed to fill in the prompt feature limits with the values compiled into
    // ClassFeatureExtractor, and neither had a live caller. The limits now come from .env.
    // public LlmSummaryService(LlmClient llmClient) {
    //     this(llmClient, new LlmPromptBuilder(), FeatureLimits.defaults());
    // }
    // public LlmSummaryService(LlmClient llmClient, LlmPromptBuilder promptBuilder) {
    //     this(llmClient, promptBuilder, FeatureLimits.defaults());
    // }

    /**
     * Constructs a new summary service with a specific prompt builder and feature limits.
     * The limits control how much of each class reaches the prompt and are resolved from .env; this
     * overload exists so they are no longer compiled into ClassFeatureExtractor.
     *
     * @param llmClient the LLM client for making API requests
     * @param promptBuilder the prompt builder configured for the desired alias
     * @param featureLimits caps on fields, constructors, methods and pattern insights
     * @param maxConsecutiveFailures failures in a row that abort the run; 0 disables the check
     */
    public LlmSummaryService(LlmClient llmClient, LlmPromptBuilder promptBuilder, FeatureLimits featureLimits,
                             int maxConsecutiveFailures) {
        if (llmClient == null) {
            throw new IllegalArgumentException("llmClient must not be null");
        }
        if (promptBuilder == null) {
            throw new IllegalArgumentException("promptBuilder must not be null");
        }
        if (featureLimits == null) {
            throw new IllegalArgumentException("featureLimits must not be null");
        }
        this.llmClient = llmClient;
        this.promptBuilder = promptBuilder; // Custom prompt builder injected for multi-prompt execution
        this.extractor = new ClassFeatureExtractor(featureLimits);
        this.maxConsecutiveFailures = Math.max(0, maxConsecutiveFailures);
    }

    /**
     * Generates summaries for all classes in a parsed project.
     * <p>
     * Processes each class in the project, extracts features, generates prompts,
     * calls the LLM API, and writes summaries to the CSV file.
     * </p>
     * 
     * @param parsedProject the parsed project data from DPS
     * @param projectKey the project identifier key
     * @param projectDisplayName the display name for the project
     * @param writer the CSV writer for output
     * @return statistics about the summarization process
     * @throws IOException if writing to the CSV file fails
     */
    public SummaryStats generateSummaries(HashMap<String, Object> parsedProject,
                                          String projectKey,
                                          String projectDisplayName,
                                          LlmSummaryWriter writer) throws IOException, LlmClientException {
        if (parsedProject == null) {
            throw new IllegalArgumentException("parsedProject must not be null");
        }
        if (projectKey == null) {
            throw new IllegalArgumentException("projectKey must not be null");
        }
        if (projectDisplayName == null) {
            throw new IllegalArgumentException("projectDisplayName must not be null");
        }
        if (writer == null) {
            throw new IllegalArgumentException("writer must not be null");
        }

        Object projectObject = parsedProject.get(projectKey);
        if (!(projectObject instanceof Map)) {
            System.err.println("Unexpected project payload for " + projectKey + ". Skipping.");
            return SummaryStats.empty();
        }
        Map<String, HashMap> projectFileMap = (Map<String, HashMap>) projectObject;

        Map<String, List<String>> patternInsights = buildPatternInsights(
                parsedProject.get("summary_NLG"),
                parsedProject.get("design_pattern"));

        int processedClasses = 0;
        int successfulSummaries = 0;
        int skippedClasses = 0;
        int failedSummaries = 0;
        // Identities, not just counts. A count tells you a class is missing from the CSV
        // but not which one, so a short run could not be distinguished from a correct one
        // without diffing the output against the corpus by hand.
        List<MissingClass> missing = new ArrayList<>();

        for (Map.Entry<String, HashMap> entry : projectFileMap.entrySet()) {
            String className = entry.getKey();
            HashMap classData = entry.getValue();
            List<String> insights = patternInsights.getOrDefault(className, List.of());

            Optional<ClassFeatureSnapshot> snapshotOpt = extractor.extract(projectDisplayName, className, classData, insights);
            if (snapshotOpt.isEmpty()) {
                System.out.printf("  Skipping %s/%s: insufficient feature data.%n", projectDisplayName, className);
                skippedClasses++;
                missing.add(new MissingClass(projectDisplayName, className, "skipped", "insufficient feature data"));
                continue;
            }

            processedClasses++;
            ClassFeatureSnapshot snapshot = snapshotOpt.get();
            String userPrompt = promptBuilder.buildUserPrompt(snapshot);

            Optional<String> summary;
            try {
                summary = llmClient.createSummary(promptBuilder.getSystemPrompt(), userPrompt);
            } catch (LlmClientException e) {
                // Previously this propagated out of the method, which abandoned every
                // remaining class in the project directory: one exhausted retry budget on a
                // single class silently cost the whole folder. Record it and keep going, so
                // the loss is bounded to the class that actually failed and is reported.
                System.err.printf("  LLM error for %s/%s: %s%n", projectDisplayName, className, e.getMessage());
                failedSummaries++;
                missing.add(new MissingClass(projectDisplayName, className, "failed",
                        e.getMessage() == null ? e.getClass().getSimpleName() : e.getMessage()));
                recordFailure(projectDisplayName, className,
                        e.getMessage() == null ? e.getClass().getSimpleName() : e.getMessage());
                continue;
            }

            if (summary.isEmpty()) {
                System.err.println("LLM returned no content for " + className + " in project " + projectDisplayName);
                failedSummaries++;
                missing.add(new MissingClass(projectDisplayName, className, "failed", "LLM returned no content"));
                recordFailure(projectDisplayName, className, "LLM returned no content");
                continue;
            }

            writer.writeRow(projectDisplayName, snapshot.getSourceFile(), summary.get());
            consecutiveFailures = 0; // One success clears the streak: the problem was that class, not the session.
            successfulSummaries++;
            System.out.printf("  Generated LLM summary for %s/%s%n", projectDisplayName, className);
        }

        return new SummaryStats(processedClasses, successfulSummaries, skippedClasses, failedSummaries, missing);
    }

    /**
     * Counts a failure and stops the run once they stop looking class-specific.
     * <p>
     * The streak spans projects and is cleared by any success. A long run of failures means the
     * session is broken -- a rate limit, an exhausted balance, a revoked key -- and continuing only
     * spends wall-clock time on backoff while the output file stays truncated.
     * </p>
     *
     * @param projectDisplayName the project whose class failed
     * @param className the class that failed
     * @param reason the failure as reported by the client
     * @throws RunAbortedException once the streak reaches the configured limit
     */
    private void recordFailure(String projectDisplayName, String className, String reason) {
        consecutiveFailures++;
        if (maxConsecutiveFailures > 0 && consecutiveFailures >= maxConsecutiveFailures) {
            throw new RunAbortedException(String.format(
                    "%d consecutive failures, last at %s/%s: %s. Stopping rather than walking the rest of the "
                    + "corpus -- this looks like a session-wide problem (rate limit, credit balance, API key), "
                    + "not one bad class. Raise LLM_MAX_CONSECUTIVE_FAILURES in .env to allow more.",
                    consecutiveFailures, projectDisplayName, className, reason));
        }
    }

    private Map<String, List<String>> buildPatternInsights(Object summaryNlgObj, Object designPatternObj) {
        Map<String, List<String>> perClass = new LinkedHashMap<>();
        if (summaryNlgObj instanceof Map) {
            Map<?, ?> patternMap = (Map<?, ?>) summaryNlgObj;
            for (Map.Entry<?, ?> patternEntry : patternMap.entrySet()) {
                String patternName = String.valueOf(patternEntry.getKey());
                Object value = patternEntry.getValue();
                if (!(value instanceof Map)) {
                    continue;
                }
                Map<?, ?> classMap = (Map<?, ?>) value;
                for (Map.Entry<?, ?> classEntry : classMap.entrySet()) {
                    String className = String.valueOf(classEntry.getKey());
                    Object sentencesObj = classEntry.getValue();
                    List<String> sentences = new ArrayList<>();
                    if (sentencesObj instanceof Iterable) {
                        for (Object sentence : (Iterable<?>) sentencesObj) {
                            if (sentence != null) {
                                sentences.add(patternName + ": " + sentence.toString());
                            }
                        }
                    }
                    if (!sentences.isEmpty()) {
                        perClass.computeIfAbsent(className, key -> new ArrayList<>()).addAll(sentences);
                    }
                }
            }
        }

        // Fallback: if no detailed summaries, still record pattern membership.
        if (designPatternObj instanceof Iterable) {
            for (Object patternNode : (Iterable<?>) designPatternObj) {
                if (!(patternNode instanceof Map)) {
                    continue;
                }
                Map<?, ?> patternMap = (Map<?, ?>) patternNode;
                for (Map.Entry<?, ?> entry : patternMap.entrySet()) {
                    String patternName = String.valueOf(entry.getKey());
                    Set<String> classNames = extractClassNames(entry.getValue());
                    for (String className : classNames) {
                        perClass.computeIfAbsent(className, key -> new ArrayList<>())
                                .add(patternName + " pattern detected via static analysis.");
                    }
                }
            }
        }

        return perClass;
    }

    private Set<String> extractClassNames(Object node) {
        Set<String> result = new LinkedHashSet<>();
        if (node instanceof Map) {
            Map<?, ?> map = (Map<?, ?>) node;
            for (Map.Entry<?, ?> entry : map.entrySet()) {
                Object key = entry.getKey();
                if (key instanceof String && looksLikeClassName((String) key)) {
                    result.add((String) key);
                }
                result.addAll(extractClassNames(entry.getValue()));
            }
        } else if (node instanceof Iterable) {
            for (Object element : (Iterable<?>) node) {
                if (element instanceof String && looksLikeClassName((String) element)) {
                    result.add((String) element);
                } else {
                    result.addAll(extractClassNames(element));
                }
            }
        }
        return result;
    }

    private boolean looksLikeClassName(String value) {
        if (value == null || value.isEmpty()) {
            return false;
        }
        char first = value.charAt(0);
        return Character.isUpperCase(first) && value.length() > 1;
    }

    /**
     * One class that produced no summary, with enough detail to act on it.
     *
     * @param project the project identifier (e.g. "AbdurRKhalid/Observer")
     * @param className the class that produced no summary
     * @param outcome either "skipped" (never sent to the API) or "failed" (sent, no usable reply)
     * @param reason human-readable explanation
     */
    public record MissingClass(String project, String className, String outcome, String reason) {
    }

    public static final class SummaryStats {
        private static final SummaryStats EMPTY = new SummaryStats(0, 0, 0, 0, List.of());

        private final int processedClasses;
        private final int successfulSummaries;
        private final int skippedClasses;
        private final int failedSummaries;
        private final List<MissingClass> missingClasses;

        SummaryStats(int processedClasses, int successfulSummaries, int skippedClasses, int failedSummaries,
                     List<MissingClass> missingClasses) {
            this.processedClasses = processedClasses;
            this.successfulSummaries = successfulSummaries;
            this.skippedClasses = skippedClasses;
            this.failedSummaries = failedSummaries;
            this.missingClasses = List.copyOf(missingClasses);
        }

        public static SummaryStats empty() {
            return EMPTY;
        }

        public int getProcessedClasses() {
            return processedClasses;
        }

        public int getSuccessfulSummaries() {
            return successfulSummaries;
        }

        public int getSkippedClasses() {
            return skippedClasses;
        }

        public int getFailedSummaries() {
            return failedSummaries;
        }

        /** Every class in this project that produced no summary row. */
        public List<MissingClass> getMissingClasses() {
            return missingClasses;
        }

        public boolean hasResults() {
            return processedClasses > 0 || successfulSummaries > 0 || skippedClasses > 0 || failedSummaries > 0;
        }
    }
}

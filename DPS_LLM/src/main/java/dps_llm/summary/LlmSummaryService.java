package dps_llm.summary;

import dps_llm.client.LlmClient;
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

    private final ClassFeatureExtractor extractor = new ClassFeatureExtractor();
    // private final LlmPromptBuilder promptBuilder = new LlmPromptBuilder(); // Original default prompt builder kept for reference
    private final LlmPromptBuilder promptBuilder; // Allows configuring prompt alias per run
    private final LlmClient llmClient;

    /**
     * Constructs a new summary service with the specified LLM client.
     * 
     * @param llmClient the LLM client for making API requests
     * @throws IllegalArgumentException if llmClient is null
     */
    public LlmSummaryService(LlmClient llmClient) {
        if (llmClient == null) {
            throw new IllegalArgumentException("llmClient must not be null");
        }
        this.llmClient = llmClient;
        this.promptBuilder = new LlmPromptBuilder(); // Default constructor preserved for backwards compatibility
    }

    /**
     * Constructs a new summary service with a specific prompt builder.
     * This overload enables running the pipeline with different prompt aliases without mutating the original behavior.
     *
     * @param llmClient the LLM client for making API requests
     * @param promptBuilder the prompt builder configured for the desired alias
     */
    public LlmSummaryService(LlmClient llmClient, LlmPromptBuilder promptBuilder) {
        if (llmClient == null) {
            throw new IllegalArgumentException("llmClient must not be null");
        }
        if (promptBuilder == null) {
            throw new IllegalArgumentException("promptBuilder must not be null");
        }
        this.llmClient = llmClient;
        this.promptBuilder = promptBuilder; // Custom prompt builder injected for multi-prompt execution
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

        for (Map.Entry<String, HashMap> entry : projectFileMap.entrySet()) {
            String className = entry.getKey();
            HashMap classData = entry.getValue();
            List<String> insights = patternInsights.getOrDefault(className, List.of());

            Optional<ClassFeatureSnapshot> snapshotOpt = extractor.extract(projectDisplayName, className, classData, insights);
            if (snapshotOpt.isEmpty()) {
                System.out.printf("  Skipping %s/%s: insufficient feature data.%n", projectDisplayName, className);
                skippedClasses++;
                continue;
            }

            processedClasses++;
            ClassFeatureSnapshot snapshot = snapshotOpt.get();
            String userPrompt = promptBuilder.buildUserPrompt(snapshot);
            Optional<String> summary = llmClient.createSummary(promptBuilder.getSystemPrompt(), userPrompt);
            
            if (summary.isEmpty()) {
                System.err.println("LLM returned no content for " + className + " in project " + projectDisplayName);
                failedSummaries++;
                continue;
            }

            writer.writeRow(projectDisplayName, snapshot.getSourceFile(), summary.get());
            successfulSummaries++;
            System.out.printf("  Generated LLM summary for %s/%s%n", projectDisplayName, className);
        }

        return new SummaryStats(processedClasses, successfulSummaries, skippedClasses, failedSummaries);
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

    public static final class SummaryStats {
        private static final SummaryStats EMPTY = new SummaryStats(0, 0, 0, 0);

        private final int processedClasses;
        private final int successfulSummaries;
        private final int skippedClasses;
        private final int failedSummaries;

        SummaryStats(int processedClasses, int successfulSummaries, int skippedClasses, int failedSummaries) {
            this.processedClasses = processedClasses;
            this.successfulSummaries = successfulSummaries;
            this.skippedClasses = skippedClasses;
            this.failedSummaries = failedSummaries;
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

        public boolean hasResults() {
            return processedClasses > 0 || successfulSummaries > 0 || skippedClasses > 0 || failedSummaries > 0;
        }
    }
}

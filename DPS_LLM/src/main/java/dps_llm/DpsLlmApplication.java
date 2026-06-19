package dps_llm;

import com.fasterxml.jackson.core.util.DefaultIndenter;
import com.fasterxml.jackson.core.util.DefaultPrettyPrinter;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.ObjectWriter;
import common.projectparser.ParseProject;
import dps_llm.client.LlmClient;
import dps_llm.client.LlmClientException;
import dps_llm.config.DotEnvLoader;
import dps_llm.prompt.LlmPromptBuilder;
import dps_llm.summary.LlmSummaryService;
import dps_llm.summary.LlmSummaryService.SummaryStats;
import dps_llm.summary.LlmSummaryWriter;

import java.io.File;
import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Main application entry point for generating LLM-based code summaries.
 * <p>
 * This class orchestrates the LLM-based summarization pipeline, which processes Java projects
 * from the input directory, extracts structural features, and generates natural language summaries
 * using a remote Large Language Model API (OpenRouter). The application reads configuration from
 * a .env file, processes projects recursively, and outputs both JSON feature data and CSV summaries.
 * </p>
 * <p>
 * Key responsibilities:
 * <ul>
 *   <li>Load and validate API configuration from .env file or environment variables</li>
 *   <li>Recursively discover Java project directories in the input folder</li>
 *   <li>Parse and extract code features from each project</li>
 *   <li>Generate LLM-based summaries via remote API calls</li>
 *   <li>Write structured output to JSON and CSV files</li>
 *   <li>Track and report processing statistics</li>
 * </ul>
 * </p>
 * 
 * @author Najam
 */
public class DpsLlmApplication {

    private static final String DEFAULT_LLM_SUMMARY_PATH = "output/summary-output/LLM_SUMMARY.csv"; // Fallback path used only when model cannot be identified
    // private static final String DEFAULT_LLM_SUMMARY_PATH = "output/summary-output/llm_summaries_nonconcise.csv"; // Non-concise path retained for quick reactivation when required
    private static final String DEFAULT_PROMPT_ALIAS = "SENIOR_ANALYST_50_WORDS"; // Fallback prompt alias when no overrides are provided
    // private static final String DEFAULT_PROMPT_ALIAS = "SENIOR_ANALYST_50_WORDS_NON_CONCISE"; // Alternate alias kept for runs that require non-concise summaries
    private static final String PROMPT_ALIAS_CONFIG_KEY = "LLM_PROMPT_ALIASES"; // Config key that enables multi-prompt execution from prompts.json
    private static final String PROMPT_ALIAS_PROPERTY = "llm.prompt.aliases"; // JVM property alternative for specifying prompt aliases

    /**
     * Application entry point.
     * <p>
     * Initializes and runs the LLM summarization pipeline. Catches and reports any fatal errors
     * that occur during execution, exiting with status code 1 on failure.
     * </p>
     * 
    * @param args command line arguments; the first non-flag value can override the summary model
     */
    public static void main(String[] args) {
        try {
            new DpsLlmApplication().run(args);
        } catch (IOException e) {
            System.err.println("Fatal IO error: " + e.getMessage());
            e.printStackTrace();
            System.exit(1);
        } catch (Exception e) {
            System.err.println("Unexpected error: " + e.getMessage());
            e.printStackTrace();
            System.exit(1);
        }
    }

    /**
     * Executes the main LLM summarization workflow.
     * <p>
     * This method orchestrates the complete processing pipeline:
     * <ol>
     *   <li>Creates required output directories</li>
     *   <li>Loads API configuration from .env file</li>
     *   <li>Initializes the LLM client with configuration parameters</li>
     *   <li>Discovers all Java projects in the input directory</li>
     *   <li>Processes each project to generate summaries</li>
     *   <li>Writes results to CSV and JSON files</li>
     *   <li>Reports final statistics</li>
     * </ol>
     * </p>
     * 
     * @throws IOException if directory creation, file I/O, or API communication fails
     */
    private void run(String[] args) throws IOException {
        ParseProject parseProject = new ParseProject();
        createDirectories();

        // Ensure duplicate tracking starts clean for this run
        ParseProject.resetDuplicateTracking();

        Map<String, String> dotEnv = DotEnvLoader.load(Path.of(".env"));
        if (dotEnv.isEmpty()) {
            System.out.println("No .env file found or it was empty. Falling back to OS environment variables.");
        }
        String apiKey = resolveValue(dotEnv, "OPENROUTER_API_KEY");
        if (apiKey == null) {
            System.out.println("OPENROUTER_API_KEY not set. Populate .env or export an environment variable before running the LLM pipeline.");
            return;
        }

        String apiUrl = resolveValue(dotEnv, "OPENROUTER_API_URL");
        if (apiUrl == null) {
            System.out.println("OPENROUTER_API_URL not set. Populate .env or export an environment variable before running the LLM pipeline.");
            return;
        }

        // Keep the legacy OPENROUTER_MODEL lookup commented for traceability, but resolve the model from the
        // per-model .env entries so the selected LLM stays configuration-driven.
        // String model = resolveValue(dotEnv, "OPENROUTER_MODEL");
        // Prefer an explicit command-line model override, then fall back to the per-model .env entries.
        String cliModelOverride = resolveCliModelOverride(args);
        String model = cliModelOverride != null ? cliModelOverride : firstConfiguredValue(
                resolveValue(dotEnv, "DEEPSEEK_MODEL"),
                resolveValue(dotEnv, "MISTRAL_MODEL"),
                resolveValue(dotEnv, "GPT_MODEL"),
                resolveValue(dotEnv, "CLAUDE_MODEL"),
                resolveValue(dotEnv, "QWEN_MODEL")
                //resolveValue(dotEnv, "GEMINI_MODEL")
        );
        if (model == null) {
            System.out.println("No summary model found in .env. Set DEEPSEEK_MODEL, MISTRAL_MODEL, GPT_MODEL, CLAUDE_MODEL, QWEN_MODEL, or GEMINI_MODEL before running the LLM pipeline.");
            return;
        }

        // OPENROUTER_MAX_TOKENS is deprecated in .env (renamed to OPENROUTER_MAX_COMPLETION_TOKENS to match the OpenRouter API field);
        // reading it here caused max tokens to silently fall back to 256 instead of the intended 512, consuming Gemini's
        // output budget before the summary could be written.
        // int maxTokens = resolveInt(dotEnv, "OPENROUTER_MAX_TOKENS", 256);

        // Read OPENROUTER_MAX_COMPLETION_TOKENS first; if absent, fall back to the legacy OPENROUTER_MAX_TOKENS key so
        // older .env files still work, then default to 512 matching the value documented in .env.
        int maxTokens = resolveInt(dotEnv, "OPENROUTER_MAX_COMPLETION_TOKENS",
                resolveInt(dotEnv, "OPENROUTER_MAX_TOKENS", 512));
        // Hardcoded default of 0.2 did not match OPENROUTER_TEMPERATURE=0 in .env; if the key were ever absent the
        // pipeline would silently run at a higher temperature than intended, producing less deterministic summaries.
        // double temperature = resolveDouble(dotEnv, "OPENROUTER_TEMPERATURE", 0.2);

        // Default is now 0.0, matching the value documented in .env so a missing key never introduces temperature drift.
        double temperature = resolveDouble(dotEnv, "OPENROUTER_TEMPERATURE", 0.0);
        String referer = resolveValue(dotEnv, "OPENROUTER_HTTP_REFERER");
        String title = resolveValue(dotEnv, "OPENROUTER_TITLE");

        LlmClient client = new LlmClient(apiUrl, apiKey, model, maxTokens, temperature, referer, title);
        // LlmSummaryService summaryService = new LlmSummaryService(client); // Single-prompt execution retained for reference

        File inputRoot = new File("input");
        if (!inputRoot.exists() || !inputRoot.isDirectory()) {
            System.out.println("No projects found in input directory.");
            return;
        }

        List<File> projectDirs = findAllProjectDirectories(inputRoot);
        if (projectDirs.isEmpty()) {
            System.out.println("No projects found in input directory.");
            return;
        }

        projectDirs.sort(Comparator.comparing(File::getPath));

        int projectLimit = resolveProjectLimit(dotEnv);
        if (projectLimit > 0 && projectLimit < projectDirs.size()) {
            System.out.printf("LLM_PROJECT_LIMIT=%d set. Processing first %d projects only.%n", projectLimit, projectLimit);
            projectDirs = projectDirs.subList(0, projectLimit);
        }

        ObjectWriter jsonWriter = new ObjectMapper()
                .writer(new DefaultPrettyPrinter().withObjectIndenter(new DefaultIndenter("\t", "\n")));

        int projectsAttempted = 0;
        // int projectsWithSummaries = 0;
        // int totalClasses = 0;
        // int totalSuccesses = 0;
        // int totalFailures = 0;
        // int totalSkipped = 0;
        Map<String, SummaryAccumulator> perPromptTotals = new LinkedHashMap<>(); // Tracks per-alias aggregates for multi-prompt runs

        // Resolve output CSV path: explicit override wins; otherwise derive from the active model
        // so each model writes to its own file (e.g. LLM_DEEPSEEK_SUMMARY.csv, LLM_GPT_SUMMARY.csv).
        String configuredOutput = resolveValue(dotEnv, "LLM_SUMMARY_PATH");
        if (configuredOutput == null) {
            String sysProp = System.getProperty("llm.summary.path");
            configuredOutput = firstNonBlank(sysProp);
        }
        String outputCsvPath = configuredOutput != null ? configuredOutput
                : buildModelOutputPath(resolveModelLabel(dotEnv, model));

        /*
         * Previous multi-prompt execution retained for reference. Uncomment to regenerate the
         * 20/40/60/80 word variants.
            System.out.println("Using summary model: " + model);
         *
         * try (PromptRunContext run20 = new PromptRunContext("SENIOR_ANALYST_20_WORDS", "output/summary-output/llm_summaries_20.csv", client);
         *      PromptRunContext run40 = new PromptRunContext("SENIOR_ANALYST_40_WORDS", "output/summary-output/llm_summaries_40.csv", client);
         *      PromptRunContext run60 = new PromptRunContext("SENIOR_ANALYST_60_WORDS", "output/summary-output/llm_summaries_60.csv", client);
         *      PromptRunContext run80 = new PromptRunContext("SENIOR_ANALYST_80_WORDS", "output/summary-output/llm_summaries_80.csv", client)) {
         *
         *     List<PromptRunContext> promptRuns = List.of(run20, run40, run60, run80);
         *     // ... (see git history prior to 50-word prompt switch)
         * }
         */

        // try (PromptRunContext run50 = new PromptRunContext("SENIOR_ANALYST_50_WORDS", outputCsvPath, client)) {
        //     List<PromptRunContext> promptRuns = List.of(run50);
        //
        //     for (File projectDir : projectDirs) {
        //         projectsAttempted++;
        //         String relativePath = inputRoot.toPath().relativize(projectDir.toPath()).toString().replace("\\", "/");
        //         String projectIdentifier = sanitizeProjectIdentifier(relativePath);
        //         try {
        //             Map<String, SummaryStats> statsByPrompt = processProject(projectDir, relativePath, projectIdentifier, parseProject, jsonWriter, promptRuns);
        //             for (PromptRunContext context : promptRuns) {
        //                 SummaryStats stats = statsByPrompt.getOrDefault(context.alias, SummaryStats.empty());
        //                 SummaryAccumulator accumulator = perPromptTotals.computeIfAbsent(context.alias, key -> new SummaryAccumulator(context.alias, context.outputCsvPath));
        //                 accumulator.record(stats);
        //             }
        //         } catch (LlmClientException e) {
        //             System.err.println("  LLM error while processing " + relativePath + ": " + e.getMessage());
        //             for (PromptRunContext context : promptRuns) {
        //                 SummaryAccumulator accumulator = perPromptTotals.computeIfAbsent(context.alias, key -> new SummaryAccumulator(context.alias, context.outputCsvPath));
        //                 accumulator.recordFailure();
        //             }
        //         } catch (IOException e) {
        //             System.err.println("  IO error while processing " + relativePath + ": " + e.getMessage());
        //             for (PromptRunContext context : promptRuns) {
        //                 SummaryAccumulator accumulator = perPromptTotals.computeIfAbsent(context.alias, key -> new SummaryAccumulator(context.alias, context.outputCsvPath));
        //                 accumulator.recordFailure();
        //             }
        //         }
        //     }
        // }
        List<PromptRunContext> promptRuns = createPromptRunContexts(dotEnv, client, outputCsvPath); // Build contexts from configuration so any prompt alias can run without code edits
        try {
            for (File projectDir : projectDirs) {
                projectsAttempted++;
                String relativePath = inputRoot.toPath().relativize(projectDir.toPath()).toString().replace("\\", "/");
                String projectIdentifier = sanitizeProjectIdentifier(relativePath);
                try {
                    Map<String, SummaryStats> statsByPrompt = processProject(projectDir, relativePath, projectIdentifier, parseProject, jsonWriter, promptRuns);
                    for (PromptRunContext context : promptRuns) {
                        SummaryStats stats = statsByPrompt.getOrDefault(context.alias, SummaryStats.empty());
                        SummaryAccumulator accumulator = perPromptTotals.computeIfAbsent(context.alias, key -> new SummaryAccumulator(context.alias, context.outputCsvPath));
                        accumulator.record(stats);
                    }
                } catch (LlmClientException e) {
                    System.err.println("  LLM error while processing " + relativePath + ": " + e.getMessage());
                    for (PromptRunContext context : promptRuns) {
                        SummaryAccumulator accumulator = perPromptTotals.computeIfAbsent(context.alias, key -> new SummaryAccumulator(context.alias, context.outputCsvPath));
                        accumulator.recordFailure();
                    }
                } catch (IOException e) {
                    System.err.println("  IO error while processing " + relativePath + ": " + e.getMessage());
                    for (PromptRunContext context : promptRuns) {
                        SummaryAccumulator accumulator = perPromptTotals.computeIfAbsent(context.alias, key -> new SummaryAccumulator(context.alias, context.outputCsvPath));
                        accumulator.recordFailure();
                    }
                }
            }
        } finally {
            closePromptRuns(promptRuns); // Ensure writers flush/close even when exceptions occur mid-run
        }

        System.out.printf("%nLLM summarisation complete for %d projects:%n", projectsAttempted);
        perPromptTotals.forEach((alias, totals) -> {
            System.out.printf("  Prompt %s -> output %s%n", alias, totals.getOutputPath());
            System.out.printf("    Projects attempted: %d%n", totals.getProjectsAttempted());
            System.out.printf("    Projects with summaries: %d%n", totals.getProjectsWithSummaries());
            System.out.printf("    Classes processed: %d%n", totals.getTotalClassesProcessed());
            System.out.printf("    Summaries generated: %d%n", totals.getTotalSummariesGenerated());
            System.out.printf("    Failed summaries: %d%n", totals.getTotalFailures());
            System.out.printf("    Classes skipped: %d%n", totals.getTotalSkipped());
        });
        System.out.println("Default single-output path retained for compatibility: " + outputCsvPath);
    }

    /**
     * Processes a single Java project to generate LLM-based summaries.
     * <p>
     * Parses the project structure, extracts features, generates summaries via LLM,
     * and writes output to both JSON and CSV files.
     * </p>
     * 
     * @param projectDir the project directory to process
     * @param relativePath the relative path from the input root
     * @param projectIdentifier sanitized identifier for file naming
     * @param parseProject the project parser instance
    * @param jsonWriter Jackson writer for JSON output
    * @param promptRuns configured prompt executions for this run
    * @return statistics about the processing results keyed by prompt alias
     */
    // private SummaryStats processProject(File projectDir,
    //                                     String relativePath,
    //                                     String projectIdentifier,
    //                                     ParseProject parseProject,
    //                                     ObjectWriter jsonWriter,
    //                                     LlmSummaryService summaryService,
    //                                     LlmSummaryWriter csvWriter) throws IOException, LlmClientException {
    //     if (projectDir == null) {
    //         throw new IllegalArgumentException("projectDir must not be null");
    //     }
    //     if (relativePath == null) {
    //         throw new IllegalArgumentException("relativePath must not be null");
    //     }
    //     if (projectIdentifier == null) {
    //         throw new IllegalArgumentException("projectIdentifier must not be null");
    //     }
    //     if (parseProject == null) {
    //         throw new IllegalArgumentException("parseProject must not be null");
    //     }
    //     if (jsonWriter == null) {
    //         throw new IllegalArgumentException("jsonWriter must not be null");
    //     }
    //     if (summaryService == null) {
    //         throw new IllegalArgumentException("summaryService must not be null");
    //     }
    //     if (csvWriter == null) {
    //         throw new IllegalArgumentException("csvWriter must not be null");
    //     }
    //
    //     System.out.println();
    //     System.out.println(relativePath);
    //
    //     HashMap<String, Object> parsedProject = parseProject.parseProject(projectDir, relativePath, false);
    //     if (parsedProject == null || parsedProject.isEmpty()) {
    //         System.out.println("  No parseable classes found.");
    //         return SummaryStats.empty();
    //     }
    //
    //     jsonWriter.writeValue(new File("output/json-output/llm/" + projectIdentifier + ".json"), parsedProject);
    //     SummaryStats stats = summaryService.generateSummaries(parsedProject, projectDir.getName(), projectIdentifier, csvWriter);
    //     if (stats.hasResults()) {
    //         System.out.printf("  Project summary: %d classes processed, %d summaries generated, %d failed, %d skipped.%n",
    //                 stats.getProcessedClasses(),
    //                 stats.getSuccessfulSummaries(),
    //                 stats.getFailedSummaries(),
    //                 stats.getSkippedClasses());
    //     } else {
    //         System.out.println("  No eligible classes for LLM summarisation.");
    //     }
    //     return stats;
    // }

    private Map<String, SummaryStats> processProject(File projectDir,
                                                     String relativePath,
                                                     String projectIdentifier,
                                                     ParseProject parseProject,
                                                     ObjectWriter jsonWriter,
                                                     List<PromptRunContext> promptRuns) throws IOException, LlmClientException {
        if (projectDir == null) {
            throw new IllegalArgumentException("projectDir must not be null");
        }
        if (relativePath == null) {
            throw new IllegalArgumentException("relativePath must not be null");
        }
        if (projectIdentifier == null) {
            throw new IllegalArgumentException("projectIdentifier must not be null");
        }
        if (parseProject == null) {
            throw new IllegalArgumentException("parseProject must not be null");
        }
        if (jsonWriter == null) {
            throw new IllegalArgumentException("jsonWriter must not be null");
        }
        if (promptRuns == null || promptRuns.isEmpty()) {
            throw new IllegalArgumentException("promptRuns must not be null or empty");
        }

        System.out.println();
        System.out.println(relativePath);
        
        HashMap<String, Object> parsedProject = parseProject.parseProject(projectDir, relativePath, false);
        Map<String, SummaryStats> results = new LinkedHashMap<>();
        if (parsedProject == null || parsedProject.isEmpty()) {
            System.out.println("  No parseable classes found.");
            for (PromptRunContext context : promptRuns) {
                results.put(context.alias, SummaryStats.empty());
                System.out.printf("  [%s] No eligible classes for LLM summarisation.%n", context.alias);
            }
            return results;
        }

        jsonWriter.writeValue(new File("output/json-output/llm/" + projectIdentifier + ".json"), parsedProject);

        for (PromptRunContext context : promptRuns) {
            // Bug fix: ParseProject.parseProject() stores data under the full relative path key
            // (e.g., "AbdurRKhalid/AbstractFactory") since the Bug 2 fix — projectDir.getName()
            // returned only the leaf name ("AbstractFactory") and caused parsedProject.get() to
            // return null, silently skipping all classes. relativePath matches the stored key.
            SummaryStats stats = context.summaryService.generateSummaries(parsedProject, relativePath, projectIdentifier, context.writer);
            results.put(context.alias, stats);
            if (stats.hasResults()) {
                System.out.printf("  [%s] Project summary: %d classes processed, %d summaries generated, %d failed, %d skipped.%n",
                        context.alias,
                        stats.getProcessedClasses(),
                        stats.getSuccessfulSummaries(),
                        stats.getFailedSummaries(),
                        stats.getSkippedClasses());
            } else {
                System.out.printf("  [%s] No eligible classes for LLM summarisation.%n", context.alias);
            }
        }
        return results;
    }

    /**
     * Creates all required output directories if they don't exist.
     * 
     * @throws IOException if directory creation fails
     */
    private void createDirectories() throws IOException {
        String[] directories = {"output", "output/json-output", "output/json-output/llm", "output/summary-output", "reference"};
        for (String dir : directories) {
            File file = new File(dir);
            if (!file.exists() && !file.mkdirs()) {
                throw new IOException("Unable to create directory: " + dir);
            }
        }
    }

    /**
     * Recursively discovers all directories containing Java files.
     * 
     * @param root the root directory to search
     * @return list of directories containing .java files
     */
    private List<File> findAllProjectDirectories(File root) {
        List<File> projectDirs = new ArrayList<>();
        findProjectDirectoriesRecursive(root, projectDirs);
        return projectDirs;
    }

    /**
     * Recursively searches for directories containing Java files.
     * 
     * @param directory the directory to search
     * @param projectDirs accumulator list for discovered project directories
     */
    private void findProjectDirectoriesRecursive(File directory, List<File> projectDirs) {
        if (!directory.isDirectory()) {
            return;
        }

        File[] javaFiles = directory.listFiles((dir, name) -> name.endsWith(".java"));
        boolean hasJavaFiles = javaFiles != null && javaFiles.length > 0;

        if (hasJavaFiles) {
            projectDirs.add(directory);
        }

        File[] subdirs = directory.listFiles(File::isDirectory);
        if (subdirs != null) {
            for (File subdir : subdirs) {
                findProjectDirectoriesRecursive(subdir, projectDirs);
            }
        }
    }

    /**
     * Sanitizes a relative path to create a valid filename identifier.
     * 
     * @param relativePath the relative path to sanitize
     * @return sanitized identifier with path separators replaced by underscores
     */
    private String sanitizeProjectIdentifier(String relativePath) {
        return relativePath.replace("/", "_").replace("\\", "_");
    }

    /**
     * Derives a short model label (DEEPSEEK, GPT, CLAUDE, MISTRAL, GEMINI) by matching
     * the resolved model string against the per-model .env keys.  Returns "LLM" if no
     * key matches (e.g. a CLI override with an unrecognised provider string).
     */
    private String resolveModelLabel(Map<String, String> dotEnv, String resolvedModel) {
        String[][] candidates = {
            {"DEEPSEEK_MODEL", "DEEPSEEK"},
            {"MISTRAL_MODEL",  "MISTRAL"},
            {"GPT_MODEL",      "GPT"},
            {"CLAUDE_MODEL",   "CLAUDE"},
            {"QWEN_MODEL",     "QWEN"},
            {"GEMINI_MODEL",   "GEMINI"}
        };
        for (String[] pair : candidates) {
            if (resolvedModel.equals(resolveValue(dotEnv, pair[0]))) {
                return pair[1];
            }
        }
        return "LLM";
    }

    /** Builds the model-specific CSV output path, e.g. output/summary-output/LLM_DEEPSEEK_SUMMARY.csv */
    private String buildModelOutputPath(String modelLabel) {
        return "output/summary-output/LLM_" + modelLabel + "_SUMMARY.csv";
    }

    /**
     * Resolves the project limit from configuration.
     *
     * @param dotEnv configuration map from .env file
     * @return maximum number of projects to process, or 0 for unlimited
     */
    private int resolveProjectLimit(Map<String, String> dotEnv) {
        String raw = resolveValue(dotEnv, "LLM_PROJECT_LIMIT");
        if (raw == null) {
            return 0;
        }
        try {
            int candidate = Integer.parseInt(raw);
            return candidate < 0 ? 0 : candidate;
        } catch (NumberFormatException ex) {
            System.out.println("Ignoring invalid LLM_PROJECT_LIMIT value: " + raw);
            return 0;
        }
    }

    /**
     * Returns the first non-blank value from a trimmed string.
     * 
     * @param value the string to check
     * @return trimmed non-empty string, or null if blank
     */
    private String firstNonBlank(String value) {
        if (value == null) {
            return null;
        }
        String trimmed = value.trim();
        return trimmed.isEmpty() ? null : trimmed;
    }

    /**
     * Returns the first non-blank value from a list of candidate strings.
     *
     * @param values the strings to inspect in priority order
     * @return the first trimmed non-empty string, or null if none are provided
     */
    private String firstConfiguredValue(String... values) {
        if (values == null) {
            return null;
        }
        for (String value : values) {
            String resolved = firstNonBlank(value);
            if (resolved != null) {
                return resolved;
            }
        }
        return null;
    }

    /**
     * Returns the first non-flag command-line argument, treating it as a model override.
     *
     * @param args the raw command-line arguments
     * @return the first positional argument, or null if none is provided
     */
    private String resolveCliModelOverride(String[] args) {
        if (args == null) {
            return null;
        }
        for (String arg : args) {
            String candidate = firstNonBlank(arg);
            if (candidate == null) {
                continue;
            }
            if (candidate.startsWith("-")) {
                continue;
            }
            return candidate;
        }
        return null;
    }

    /**
     * Resolves a configuration value from .env or environment variables.
     * <p>
     * Checks .env file first, then falls back to OS environment variables.
     * </p>
     * 
     * @param dotEnv configuration map from .env file
     * @param key the configuration key to look up
     * @return the resolved value, or null if not found
     */
    private String resolveValue(Map<String, String> dotEnv, String key) {
        String override = dotEnv == null ? null : dotEnv.get(key);
        String value = firstNonBlank(override);
        if (value != null) {
            return value;
        }
        return firstNonBlank(System.getenv(key));
    }

    /**
     * Resolves an integer configuration value with a default fallback.
     * 
     * @param dotEnv configuration map from .env file
     * @param key the configuration key to look up
     * @param defaultValue the default value if key not found or invalid
     * @return the resolved integer value
     */
    private int resolveInt(Map<String, String> dotEnv, String key, int defaultValue) {
        String raw = resolveValue(dotEnv, key);
        if (raw == null) {
            return defaultValue;
        }
        try {
            return Integer.parseInt(raw);
        } catch (NumberFormatException ex) {
            System.out.println("Ignoring invalid " + key + " value: " + raw);
            return defaultValue;
        }
    }

    /**
     * Resolves a double configuration value with a default fallback.
     * 
     * @param dotEnv configuration map from .env file
     * @param key the configuration key to look up
     * @param defaultValue the default value if key not found or invalid
     * @return the resolved double value
     */
    private double resolveDouble(Map<String, String> dotEnv, String key, double defaultValue) {
        String raw = resolveValue(dotEnv, key);
        if (raw == null) {
            return defaultValue;
        }
        try {
            return Double.parseDouble(raw);
        } catch (NumberFormatException ex) {
            System.out.println("Ignoring invalid " + key + " value: " + raw);
            return defaultValue;
        }
    }

    /**
     * Builds prompt run contexts from configuration so any alias in prompts.json can be executed without editing code.
     * Accepts semicolon- or comma-separated entries in the form alias or alias=outputPath via .env/env/system properties.
     */
    private List<PromptRunContext> createPromptRunContexts(Map<String, String> dotEnv,
                                                           LlmClient client,
                                                           String defaultOutputPath) throws IOException {
        List<PromptRunContext> contexts = new ArrayList<>();
        String config = resolvePromptAliasConfig(dotEnv);
        boolean defaultConsumed = false;
        String safeDefault = firstNonBlank(defaultOutputPath) == null ? DEFAULT_LLM_SUMMARY_PATH : defaultOutputPath;
        try {
            if (config == null || config.isBlank()) {
                contexts.add(new PromptRunContext(DEFAULT_PROMPT_ALIAS, safeDefault, client));
                return contexts;
            }

            String[] entries = config.split("[;,]");
            for (String entry : entries) {
                if (entry == null) {
                    continue;
                }
                String trimmed = entry.trim();
                if (trimmed.isEmpty()) {
                    continue;
                }

                String alias = trimmed;
                String customOutput = null;
                int delimiterIndex = trimmed.indexOf('=');
                if (delimiterIndex < 0) {
                    delimiterIndex = trimmed.indexOf(':');
                }
                if (delimiterIndex >= 0) {
                    alias = trimmed.substring(0, delimiterIndex).trim();
                    customOutput = trimmed.substring(delimiterIndex + 1).trim();
                }
                if (alias.isEmpty()) {
                    continue;
                }

                String resolvedOutput = firstNonBlank(customOutput);
                if (resolvedOutput == null) {
                    if (!defaultConsumed) {
                        resolvedOutput = safeDefault;
                        defaultConsumed = true;
                    } else {
                        resolvedOutput = buildDefaultOutputPath(alias);
                    }
                }
                contexts.add(new PromptRunContext(alias, resolvedOutput, client));
            }
        } catch (IOException | RuntimeException ex) {
            closePromptRuns(contexts);
            throw ex;
        }

        if (contexts.isEmpty()) {
            contexts.add(new PromptRunContext(DEFAULT_PROMPT_ALIAS, safeDefault, client));
        }
        return contexts;
    }

    /**
     * Retrieves the raw prompt alias configuration from .env, environment variables, or JVM properties.
     */
    private String resolvePromptAliasConfig(Map<String, String> dotEnv) {
        String configured = resolveValue(dotEnv, PROMPT_ALIAS_CONFIG_KEY);
        if (configured != null) {
            return configured;
        }
        return firstNonBlank(System.getProperty(PROMPT_ALIAS_PROPERTY));
    }

    /**
     * Derives a deterministic CSV filename for a prompt alias so each run records summaries separately.
     */
    private String buildDefaultOutputPath(String alias) {
        if (alias == null || alias.isBlank()) {
            return DEFAULT_LLM_SUMMARY_PATH;
        }
        if (DEFAULT_PROMPT_ALIAS.equals(alias)) {
            return DEFAULT_LLM_SUMMARY_PATH;
        }
        String sanitized = alias.toLowerCase().replaceAll("[^a-z0-9]+", "_");
        if (sanitized.isBlank()) {
            sanitized = "custom";
        }
        while (sanitized.contains("__")) {
            sanitized = sanitized.replace("__", "_");
        }
        return "output/summary-output/llm_summaries_" + sanitized + ".csv";
    }

    /**
     * Closes all PromptRunContext instances, logging (but not rethrowing) IO issues so shutdown remains graceful.
     */
    private void closePromptRuns(List<PromptRunContext> promptRuns) {
        if (promptRuns == null) {
            return;
        }
        for (PromptRunContext context : promptRuns) {
            if (context == null) {
                continue;
            }
            try {
                context.close();
            } catch (IOException ioe) {
                System.err.println("Failed to close writer for prompt " + context.alias + ": " + ioe.getMessage());
            }
        }
    }

    private static final class PromptRunContext implements AutoCloseable {
        private final String alias;
        private final String outputCsvPath;
        private final LlmSummaryService summaryService;
        private final LlmSummaryWriter writer;

        PromptRunContext(String alias, String outputCsvPath, LlmClient client) throws IOException {
            if (alias == null || alias.trim().isEmpty()) {
                throw new IllegalArgumentException("Prompt alias must not be blank");
            }
            if (outputCsvPath == null || outputCsvPath.trim().isEmpty()) {
                throw new IllegalArgumentException("Output path must not be blank");
            }
            if (client == null) {
                throw new IllegalArgumentException("LlmClient must not be null");
            }
            this.alias = alias;
            this.outputCsvPath = outputCsvPath;
            // this.summaryService = new LlmSummaryService(client); // Original single-prompt builder retained for reference
            this.summaryService = new LlmSummaryService(client, new LlmPromptBuilder(alias)); // Inject prompt-specific builder for multi-run execution
            this.writer = new LlmSummaryWriter(outputCsvPath);
        }

        @Override
        public void close() throws IOException {
            writer.close();
        }
    }

    private static final class SummaryAccumulator {
        private final String outputPath;
        private int projectsAttempted;
        private int projectsWithSummaries;
        private int totalClassesProcessed;
        private int totalSummariesGenerated;
        private int totalFailures;
        private int totalSkipped;

        SummaryAccumulator(String alias, String outputPath) {
            this.outputPath = outputPath;
        }

        void record(SummaryStats stats) {
            projectsAttempted++;
            if (stats != null) {
                totalClassesProcessed += stats.getProcessedClasses();
                totalSummariesGenerated += stats.getSuccessfulSummaries();
                totalFailures += stats.getFailedSummaries();
                totalSkipped += stats.getSkippedClasses();
                if (stats.getSuccessfulSummaries() > 0) {
                    projectsWithSummaries++;
                }
            }
        }

        String getOutputPath() {
            return outputPath;
        }

        void recordFailure() {
            projectsAttempted++;
            totalFailures++;
        }

        int getProjectsWithSummaries() {
            return projectsWithSummaries;
        }

        int getProjectsAttempted() {
            return projectsAttempted;
        }

        int getTotalClassesProcessed() {
            return totalClassesProcessed;
        }

        int getTotalSummariesGenerated() {
            return totalSummariesGenerated;
        }

        int getTotalFailures() {
            return totalFailures;
        }

        int getTotalSkipped() {
            return totalSkipped;
        }
    }
}

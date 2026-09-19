package dps_llm;

import com.fasterxml.jackson.core.util.DefaultIndenter;
import com.fasterxml.jackson.core.util.DefaultPrettyPrinter;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.ObjectWriter;
import common.projectparser.ParseProject;
import common.projectparser.ProjectJsonStore;
import dps_llm.client.LlmClient;
import dps_llm.client.LlmClientException;
import dps_llm.config.ConfigurationException;
import dps_llm.config.DotEnvLoader;
import dps_llm.config.EnvConfig;
import dps_llm.config.FeatureLimits;
import dps_llm.prompt.LlmPromptBuilder;
import dps_llm.prompt.PromptManager;
import dps_llm.summary.LlmSummaryService;
import dps_llm.summary.LlmSummaryService.MissingClass;
import dps_llm.summary.LlmSummaryService.SummaryStats;
import dps_llm.summary.LlmSummaryWriter;
import dps_llm.summary.RunAbortedException;

import java.io.BufferedWriter;
import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
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
    // Najam: both 50-word prompts are now selectable at run time, so neither alias is compiled in.
    // Switching between them used to mean editing this line, recompiling, and remembering to move the
    // previous output aside -- the same edit-to-configure pattern as the .env literals. The default is
    // llm_summarization.system_prompt.alias in prompts.json; the shorthands that select either one are
    // llm_summarization.cli_prompt_variants. Run: java dps_llm.DpsLlmApplication <variant> <MODEL>
    // private static final String DEFAULT_PROMPT_ALIAS = "SENIOR_ANALYST_50_WORDS"; // Fallback prompt alias when no overrides are provided
    // private static final String DEFAULT_PROMPT_ALIAS = "SENIOR_ANALYST_50_WORDS_NON_CONCISE"; // Alternate alias kept for runs that require non-concise summaries
    private static final String PROMPT_ALIAS_CONFIG_KEY = "LLM_PROMPT_ALIASES"; // Config key that enables multi-prompt execution from prompts.json
    private static final String PROMPT_ALIAS_PROPERTY = "llm.prompt.aliases"; // JVM property alternative for specifying prompt aliases
    // Najam: removed. LLM_SUMMARY_MAX_CHARS is required from .env; a compiled-in copy of the value is
    // exactly the drift this pipeline has already been bitten by twice.
    // private static final int DEFAULT_LLM_SUMMARY_MAX_CHARS = 600;
    /** run() returns this when a required setting is absent, as opposed to a count of missing summaries. */
    private static final int CONFIGURATION_ERROR = -1;
    /**
     * The summarisation models this pipeline knows, as .env key -> label. One roster, used to name the
     * active model, to accept a label on the command line, and to list the choices in the usage text,
     * so .env keys that are not summarisation models (RANK_SUMMARIES_MODEL) are never offered as one.
     */
    private static final String[][] MODEL_LABEL_KEYS = {
        {"MISTRAL_MODEL",  "MISTRAL"},
        {"GPT_MODEL",      "GPT"},
        {"CLAUDE_MODEL",   "CLAUDE"},
        {"QWEN_MODEL",     "QWEN"}
    };
    private static final String LLM_JSON_OUTPUT_DIR = "output/json-output/llm"; // DPS_LLM's own JSON representation, written then read back for prompting

    /** Reader for the JSON representation this pipeline writes; restores collection types on read. */
    private final ObjectMapper jsonObjectMapper = new ObjectMapper();

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
            int missing = new DpsLlmApplication().run(args);
            if (missing == CONFIGURATION_ERROR) {
                // run() has already explained which setting is missing; fail the process so a
                // scripted pipeline stops rather than continuing with no summaries at all.
                System.exit(1);
            }
            if (missing > 0) {
                // A run that silently produced 149 of 150 summaries looks identical to a
                // complete one from the shell. Exit non-zero so a scripted pipeline stops
                // instead of carrying an incomplete CSV into evaluation.
                System.err.printf("%nRun incomplete: %d class(es) produced no summary. "
                        + "See the .missing.csv manifest next to each output CSV.%n", missing);
                System.exit(1);
            }
        } catch (ConfigurationException e) {
            // A missing or malformed .env entry is a setup problem, not a crash; report it plainly
            // rather than as an unexpected error with a stack trace.
            System.err.println("Configuration error: " + e.getMessage());
            System.exit(1);
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
    private int run(String[] args) throws IOException {
        ParseProject parseProject = new ParseProject();
        createDirectories();

        // Ensure duplicate tracking starts clean for this run
        ParseProject.resetDuplicateTracking();

        Map<String, String> dotEnv = DotEnvLoader.load(Path.of("..", ".env"));
        if (dotEnv.isEmpty()) {
            System.out.println("No .env file found or it was empty. Falling back to OS environment variables.");
        }
        // Parsed before the credential checks so that --help, and a mistyped model or prompt, are
        // answered without requiring an API key to be configured first.
        // CliOptions carries both selections, so one command names them: `Concise QWEN`.
        CliOptions cli = parseCliArguments(args, dotEnv);
        if (cli == null) {
            return CONFIGURATION_ERROR;
        }

        String apiKey = resolveValue(dotEnv, "OPENROUTER_API_KEY");
        if (apiKey == null) {
            System.out.println("OPENROUTER_API_KEY not set. Populate .env or export an environment variable before running the LLM pipeline.");
            // run() returns int; a bare return here did not compile.
            return CONFIGURATION_ERROR;
        }

        String apiUrl = resolveValue(dotEnv, "OPENROUTER_API_URL");
        if (apiUrl == null) {
            System.out.println("OPENROUTER_API_URL not set. Populate .env or export an environment variable before running the LLM pipeline.");
            // run() returns int; a bare return here did not compile.
            return CONFIGURATION_ERROR;
        }

        // Keep the legacy OPENROUTER_MODEL lookup commented for traceability, but resolve the model from the
        // per-model .env entries so the selected LLM stays configuration-driven.
        // String model = resolveValue(dotEnv, "OPENROUTER_MODEL");
        // Prefer an explicit command-line model override, then fall back to the per-model .env entries.
        String model = cli.model != null ? cli.model : firstConfiguredValue(
                //resolveValue(dotEnv, "DEEPSEEK_MODEL"),
                resolveValue(dotEnv, "MISTRAL_MODEL"),
                resolveValue(dotEnv, "GPT_MODEL"),
                resolveValue(dotEnv, "CLAUDE_MODEL"),
                resolveValue(dotEnv, "QWEN_MODEL")
                //resolveValue(dotEnv, "GEMINI_MODEL")
        );
        if (model == null) {
            System.out.println("No summary model found in .env. Set DEEPSEEK_MODEL, MISTRAL_MODEL, GPT_MODEL, CLAUDE_MODEL, QWEN_MODEL, or GEMINI_MODEL before running the LLM pipeline.");
            // run() returns int; a bare return here did not compile.
            return CONFIGURATION_ERROR;
        }

        // OPENROUTER_MAX_TOKENS is deprecated in .env (renamed to OPENROUTER_MAX_COMPLETION_TOKENS to match
        // the OpenRouter API field); the legacy name is still accepted so older .env files keep working.
        //
        // Both of these were read with a literal default, and both literals had already drifted from .env
        // at least once: the token budget fell back to 256 while .env said 512, and the temperature fell
        // back to 0.2 while .env said 0. Correcting the literals to match .env only postpones the problem,
        // because the value then exists in two places that nothing keeps in step. The keys are required
        // instead, so .env is the single source and a missing or malformed entry stops the run.
        // int maxTokens = resolveInt(dotEnv, "OPENROUTER_MAX_TOKENS", 256);
        // int maxTokens = resolveInt(dotEnv, "OPENROUTER_MAX_COMPLETION_TOKENS", resolveInt(dotEnv, "OPENROUTER_MAX_TOKENS", 512));
        // double temperature = resolveDouble(dotEnv, "OPENROUTER_TEMPERATURE", 0.2);
        // double temperature = resolveDouble(dotEnv, "OPENROUTER_TEMPERATURE", 0.0);
        int maxTokens = EnvConfig.requireInt(dotEnv, "OPENROUTER_MAX_COMPLETION_TOKENS", "OPENROUTER_MAX_TOKENS");
        double temperature = EnvConfig.requireDouble(dotEnv, "OPENROUTER_TEMPERATURE");
        String referer = resolveValue(dotEnv, "OPENROUTER_HTTP_REFERER");
        String title = resolveValue(dotEnv, "OPENROUTER_TITLE");

        String modelLabel = resolveModelLabel(dotEnv, model);
        boolean disableReasoning = isReasoningDisabled(dotEnv, modelLabel);
        if (disableReasoning) {
            System.out.printf("Reasoning disabled for %s (%s) per LLM_NO_REASONING_MODELS.%n", modelLabel, model);
        }

        // Pacing beats retrying against a rate limit: OpenRouter ties an account's request rate to its
        // credit balance, so a low balance returns 429 no matter how long the client waits afterwards.
        long requestIntervalMillis = EnvConfig.requireInt(dotEnv, "LLM_REQUEST_INTERVAL_MS");
        if (requestIntervalMillis > 0) {
            System.out.printf("Pacing requests at one per %dms (LLM_REQUEST_INTERVAL_MS).%n", requestIntervalMillis);
        }
        LlmClient client = new LlmClient(apiUrl, apiKey, model, maxTokens, temperature, referer, title,
                disableReasoning, requestIntervalMillis);

        // State the two choices that decide what this run produces. Which prompt was active used to be
        // invisible at run time -- it was whichever alias was uncommented when the classes were built.
        System.out.printf("Model: %s (%s). Prompt: %s.%n", modelLabel, model,
                cli.promptAlias != null ? cli.promptAlias + " (command line)" : "from configuration");
        // LlmSummaryService summaryService = new LlmSummaryService(client); // Single-prompt execution retained for reference

        File inputRoot = new File("input/dataset");
        if (!inputRoot.exists() || !inputRoot.isDirectory()) {
            System.out.println("No projects found in input directory.");
            return 0;
        }

        List<File> projectDirs = findAllProjectDirectories(inputRoot);
        if (projectDirs.isEmpty()) {
            System.out.println("No projects found in input directory.");
            return 0;
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
        // The prompt variant is part of the filename, not just the model: a non-concise run used to
        // land on top of the concise CSV of the same model unless LLM_SUMMARY_PATH was set by hand.
        String outputCsvPath = configuredOutput != null ? configuredOutput
                : buildModelOutputPath(modelLabel, cli.outputSuffix);

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
        List<PromptRunContext> promptRuns = createPromptRunContexts(dotEnv, client, outputCsvPath, cli.promptAlias); // Build contexts from configuration so any prompt alias can run without code edits
        boolean aborted = false;
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
                        accumulator.recordFailure(relativePath, "LLM error: " + e.getMessage());
                    }
                } catch (IOException e) {
                    System.err.println("  IO error while processing " + relativePath + ": " + e.getMessage());
                    for (PromptRunContext context : promptRuns) {
                        SummaryAccumulator accumulator = perPromptTotals.computeIfAbsent(context.alias, key -> new SummaryAccumulator(context.alias, context.outputCsvPath));
                        accumulator.recordFailure(relativePath, "IO error: " + e.getMessage());
                    }
                }
            }
        } catch (RunAbortedException e) {
            // Not a per-class failure: the session itself is broken, so stop and keep the old CSV.
            System.err.println();
            System.err.println("Run aborted: " + e.getMessage());
            aborted = true;
        } finally {
            // Commit only a run that finished and has rows. Anything else leaves the previous CSV in place.
            closePromptRuns(promptRuns, !aborted); // Ensure writers flush/close even when exceptions occur mid-run
        }

        // Read before reporting: the writers are closed above but still hold their counts.
        Map<String, Integer> truncationsByAlias = new LinkedHashMap<>();
        for (PromptRunContext context : promptRuns) {
            truncationsByAlias.put(context.alias, context.writer.getTruncatedSummaries());
        }

        System.out.printf("%nLLM summarisation complete for %d projects:%n", projectsAttempted);
        int totalMissing = 0;
        for (Map.Entry<String, SummaryAccumulator> promptEntry : perPromptTotals.entrySet()) {
            SummaryAccumulator totals = promptEntry.getValue();
            System.out.printf("  Prompt %s -> output %s%n", promptEntry.getKey(), totals.getOutputPath());
            System.out.printf("    Projects attempted: %d%n", totals.getProjectsAttempted());
            System.out.printf("    Projects with summaries: %d%n", totals.getProjectsWithSummaries());
            System.out.printf("    Classes processed: %d%n", totals.getTotalClassesProcessed());
            System.out.printf("    Summaries generated: %d%n", totals.getTotalSummariesGenerated());
            System.out.printf("    Failed summaries: %d%n", totals.getTotalFailures());
            System.out.printf("    Classes skipped: %d%n", totals.getTotalSkipped());

            int truncated = truncationsByAlias.getOrDefault(promptEntry.getKey(), 0);
            if (truncated > 0) {
                System.err.printf("    WARNING: %d summary/summaries exceeded LLM_SUMMARY_MAX_CHARS "
                        + "and were truncated; their scores understate the model.%n", truncated);
            }

            List<MissingClass> missing = totals.getMissingClasses();
            if (missing.isEmpty()) {
                continue;
            }
            totalMissing += missing.size();
            Path manifest = writeMissingManifest(totals.getOutputPath(), missing);
            System.err.printf("%n    WARNING: %d class(es) produced no summary row in %s.%n",
                    missing.size(), totals.getOutputPath());
            for (MissingClass entry : missing) {
                System.err.printf("      [%s] %s/%s — %s%n",
                        entry.outcome(), entry.project(), entry.className(), entry.reason());
            }
            System.err.printf("    Manifest written to: %s%n", manifest);
        }
        System.out.println("Default single-output path retained for compatibility: " + outputCsvPath);
        return totalMissing;
    }

    /**
     * Writes the list of classes that produced no summary beside the summary CSV.
     * <p>
     * The counts printed at the end of a run were the only trace of a class that did not make
     * it into the output, so a CSV with 149 of 150 rows was indistinguishable from a complete
     * one once the console scrolled away. The manifest names each one and why, which is what a
     * targeted re-run needs.
     * </p>
     *
     * @param outputCsvPath the summary CSV this manifest accompanies
     * @param missing the classes that produced no summary row
     * @return the path the manifest was written to
     * @throws IOException if the manifest cannot be written
     */
    private Path writeMissingManifest(String outputCsvPath, List<MissingClass> missing) throws IOException {
        String base = outputCsvPath.endsWith(".csv")
                ? outputCsvPath.substring(0, outputCsvPath.length() - 4)
                : outputCsvPath;
        Path manifest = Path.of(base + ".missing.csv");
        Path parent = manifest.getParent();
        if (parent != null) {
            Files.createDirectories(parent);
        }

        try (BufferedWriter writer = Files.newBufferedWriter(manifest, StandardCharsets.UTF_8)) {
            writer.write("Project,Class,Outcome,Reason\n");
            for (MissingClass entry : missing) {
                writer.write(String.format("\"%s\",\"%s\",\"%s\",\"%s\"%n",
                        csvEscape(entry.project()),
                        csvEscape(entry.className()),
                        csvEscape(entry.outcome()),
                        csvEscape(entry.reason())));
            }
        }
        return manifest;
    }

    /** Escapes a value for the quoted CSV fields written by {@link #writeMissingManifest}. */
    private static String csvEscape(String value) {
        if (value == null) {
            return "";
        }
        return value.replace("\"", "\"\"").replace("\r", " ").replace("\n", " ");
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
        
        // The third argument used to be a generateNlgSummary flag. ParseProject no longer summarises.
        // HashMap<String, Object> parsedProject = parseProject.parseProject(projectDir, relativePath, false);
        HashMap<String, Object> parsedProject = parseProject.parseProject(projectDir, relativePath);
        Map<String, SummaryStats> results = new LinkedHashMap<>();
        if (parsedProject == null || parsedProject.isEmpty()) {
            System.out.println("  No parseable classes found.");
            for (PromptRunContext context : promptRuns) {
                results.put(context.alias, SummaryStats.empty());
                System.out.printf("  [%s] No eligible classes for LLM summarisation.%n", context.alias);
            }
            return results;
        }

        // DPS_LLM owns output/json-output/llm. The file used to be written here and never read again —
        // the prompts were built from the in-memory parse, so the JSON was a write-only by-product.
        // It is now the input to prompt construction, matching DPS_NLG and DPS_SWUM: parse, write the
        // JSON representation, then summarise from that representation.
        File projectJson = new File(LLM_JSON_OUTPUT_DIR + "/" + projectIdentifier + ".json");
        ProjectJsonStore.write(jsonWriter, projectJson, parsedProject);
        parsedProject = ProjectJsonStore.read(jsonObjectMapper, projectJson);

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
        String[] directories = {"output", "output/json-output", LLM_JSON_OUTPUT_DIR, "output/summary-output", "reference"};
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
        for (String[] pair : MODEL_LABEL_KEYS) {
            if (resolvedModel.equals(resolveValue(dotEnv, pair[0]))) {
                return pair[1];
            }
        }
        return "LLM";
    }

    /**
     * Whether the active model's internal chain of thought must be switched off.
     * <p>
     * Driven by LLM_NO_REASONING_MODELS in .env, a comma-separated list of the same labels
     * {@link #resolveModelLabel} produces. Keeping the roster in configuration means adding
     * a hybrid reasoning model never requires editing this class.
     * </p>
     *
     * @param dotEnv configuration map from .env file
     * @param modelLabel the resolved label for the active model, e.g. "QWEN"
     * @return true when reasoning should be disabled for this model
     */
    private boolean isReasoningDisabled(Map<String, String> dotEnv, String modelLabel) {
        String configured = resolveValue(dotEnv, "LLM_NO_REASONING_MODELS");
        if (configured == null || configured.isBlank()) {
            return false;
        }
        for (String entry : configured.split("[;,]")) {
            if (entry.trim().equalsIgnoreCase(modelLabel)) {
                return true;
            }
        }
        return false;
    }

    /**
     * Builds the CSV output path for a model and prompt variant, e.g.
     * output/summary-output/LLM_QWEN_SUMMARY.csv, or LLM_QWEN_NC_SUMMARY.csv for the
     * non-concise prompt. The suffix comes from cli_prompt_variants in prompts.json.
     *
     * @param modelLabel the resolved model label, e.g. "QWEN"
     * @param promptSuffix the variant's output_suffix, empty for the default prompt
     * @return the CSV path for this model/prompt combination
     */
    private String buildModelOutputPath(String modelLabel, String promptSuffix) {
        return "output/summary-output/LLM_" + modelLabel + (promptSuffix == null ? "" : promptSuffix) + "_SUMMARY.csv";
    }

    /**
     * The alias a run uses when neither argv nor configuration names a prompt: whatever
     * llm_summarization.system_prompt declares in prompts.json.
     *
     * @return the default prompt alias
     */
    private String defaultPromptAlias() {
        return PromptManager.getInstance().getDefaultPromptAlias();
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
     * What the command line selected for this run: which prompt, and which model.
     * <p>
     * Both are optional and order-independent, so {@code Concise QWEN} and {@code QWEN Concise}
     * are the same run. A null field means "not named on the command line"; the caller then falls
     * back to configuration.
     * </p>
     */
    private static final class CliOptions {
        /** Full prompt alias, or null when argv named no prompt. */
        private final String promptAlias;
        /** The variant's CSV infix, e.g. "_NC"; empty when the default prompt is in use. */
        private final String outputSuffix;
        /** OpenRouter model id, or null when argv named no model. */
        private final String model;

        CliOptions(String promptAlias, String outputSuffix, String model) {
            this.promptAlias = promptAlias;
            this.outputSuffix = outputSuffix == null ? "" : outputSuffix;
            this.model = model;
        }
    }

    /**
     * Parses the command line into a prompt choice and a model choice.
     * <p>
     * Each positional argument is classified by what it names rather than by its position:
     * </p>
     * <ol>
     *   <li>a shorthand from cli_prompt_variants in prompts.json (Concise, NonConcise);</li>
     *   <li>a prompt alias spelled in full (SENIOR_ANALYST_50_WORDS_NON_CONCISE);</li>
     *   <li>a model label backed by a .env key, so QWEN resolves through QWEN_MODEL;</li>
     *   <li>a raw OpenRouter model id, recognised by its provider slash (qwen/qwen3.7-plus).</li>
     * </ol>
     * <p>
     * Anything else is rejected with the valid choices listed, rather than being passed to
     * OpenRouter as a model name and failing one HTTP call into the run. Case is ignored for
     * labels and shorthands. Flags (leading "-") are skipped, except -h/--help which prints usage.
     * </p>
     *
     * @param args the raw command-line arguments
     * @param dotEnv configuration map, consulted for &lt;LABEL&gt;_MODEL keys
     * @return the parsed selection, or null when an argument was not understood
     */
    private CliOptions parseCliArguments(String[] args, Map<String, String> dotEnv) {
        String promptAlias = null;
        String outputSuffix = "";
        String model = null;
        if (args == null) {
            return new CliOptions(null, "", null);
        }
        PromptManager prompts = PromptManager.getInstance();

        for (String arg : args) {
            String token = firstNonBlank(arg);
            if (token == null) {
                continue;
            }
            if (token.startsWith("-")) {
                if (token.equals("-h") || token.equals("--help")) {
                    printUsage(dotEnv);
                    return null;
                }
                continue;
            }

            PromptManager.PromptVariant variant = prompts.resolveCliPromptVariant(token);
            if (variant == null && prompts.hasPrompt(token)) {
                // An alias spelled in full still gets its variant's CSV suffix when one is declared.
                variant = prompts.findVariantByAlias(token);
                if (variant == null) {
                    if (promptAlias != null) {
                        System.err.printf("Two prompts named on the command line: %s and %s.%n", promptAlias, token);
                        return null;
                    }
                    promptAlias = token;
                    continue;
                }
            }
            if (variant != null) {
                if (promptAlias != null) {
                    System.err.printf("Two prompts named on the command line: %s and %s.%n", promptAlias, variant.getAlias());
                    return null;
                }
                promptAlias = variant.getAlias();
                outputSuffix = variant.getOutputSuffix();
                continue;
            }

            String resolvedModel = resolveModelForLabel(dotEnv, token);
            if (resolvedModel == null && token.contains("/")) {
                resolvedModel = token; // Raw provider/model id, as accepted before labels were supported
            }
            if (resolvedModel != null) {
                if (model != null) {
                    System.err.printf("Two models named on the command line: %s and %s.%n", model, resolvedModel);
                    return null;
                }
                model = resolvedModel;
                continue;
            }

            System.err.println("Unrecognised argument: " + token);
            printUsage(dotEnv);
            return null;
        }
        return new CliOptions(promptAlias, outputSuffix, model);
    }

    /** Prints how to invoke the pipeline, with the prompts and models this configuration actually offers. */
    private void printUsage(Map<String, String> dotEnv) {
        PromptManager prompts = PromptManager.getInstance();
        System.out.println("Usage: java dps_llm.DpsLlmApplication [prompt] [model]");
        System.out.println("  Both arguments are optional and may appear in either order.");
        System.out.println("  Prompts: " + String.join(", ", prompts.getCliPromptVariantNames())
                + " (or a full alias); default " + prompts.getDefaultPromptAlias());
        System.out.println("  Models:  " + String.join(", ", availableModelLabels(dotEnv))
                + " (or a provider/model id)");
        System.out.println("  Example: java dps_llm.DpsLlmApplication Concise QWEN");
    }

    /**
     * Lists the model labels this .env supports, derived from its &lt;LABEL&gt;_MODEL keys so a
     * newly added model appears in the usage text without a code change.
     *
     * @param dotEnv configuration map from .env file
     * @return the available labels, sorted
     */
    private List<String> availableModelLabels(Map<String, String> dotEnv) {
        List<String> labels = new ArrayList<>();
        for (String[] pair : MODEL_LABEL_KEYS) {
            if (resolveValue(dotEnv, pair[0]) != null) {
                labels.add(pair[1]);
            }
        }
        return labels;
    }

    /**
     * Resolves a model label typed on the command line through its .env key, so QWEN becomes
     * whatever QWEN_MODEL currently names. Returns null for a label this .env does not configure,
     * which keeps an unconfigured model a startup error rather than a failed HTTP call.
     *
     * @param dotEnv configuration map from .env file
     * @param token the label as typed, case-insensitive
     * @return the configured model id, or null
     */
    private String resolveModelForLabel(Map<String, String> dotEnv, String token) {
        for (String[] pair : MODEL_LABEL_KEYS) {
            if (pair[1].equalsIgnoreCase(token)) {
                return resolveValue(dotEnv, pair[0]);
            }
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

    // Najam: resolveInt and resolveDouble are removed. Their whole purpose was to substitute a literal
    // when a key was missing or unparseable — the silent-default mechanism that let the token budget
    // run at 256 while .env said 512, and the temperature at 0.2 while .env said 0. Run-shaping
    // settings are read through EnvConfig.require*, which fails with the key name instead.
    // private int resolveInt(Map<String, String> dotEnv, String key, int defaultValue) { ... }
    // private double resolveDouble(Map<String, String> dotEnv, String key, double defaultValue) { ... }

    /**
     * Builds prompt run contexts from configuration so any alias in prompts.json can be executed without editing code.
     * Accepts semicolon- or comma-separated entries in the form alias or alias=outputPath via .env/env/system properties.
     */
    private List<PromptRunContext> createPromptRunContexts(Map<String, String> dotEnv,
                                                           LlmClient client,
                                                           String defaultOutputPath,
                                                           String cliPromptAlias) throws IOException {
        List<PromptRunContext> contexts = new ArrayList<>();
        // An alias named on the command line is the whole run: it beats LLM_PROMPT_ALIASES, which
        // otherwise silently turns `Concise QWEN` into whatever multi-prompt list .env happens to hold.
        String config = cliPromptAlias != null ? null : resolvePromptAliasConfig(dotEnv);
        if (cliPromptAlias != null && resolvePromptAliasConfig(dotEnv) != null) {
            System.out.printf("Command-line prompt %s overrides %s from configuration.%n",
                    cliPromptAlias, PROMPT_ALIAS_CONFIG_KEY);
        }
        boolean defaultConsumed = false;
        String safeDefault = firstNonBlank(defaultOutputPath) == null ? DEFAULT_LLM_SUMMARY_PATH : defaultOutputPath;
        // Summary length cap comes from .env so it can be raised without a rebuild; the
        // limit used to be a literal 600 buried in the writer and truncation was silent.
        // A DEFAULT_LLM_SUMMARY_MAX_CHARS fallback would reintroduce the same literal one level up,
        // so the key is required instead.
        // int maxSummaryChars = resolveInt(dotEnv, "LLM_SUMMARY_MAX_CHARS", DEFAULT_LLM_SUMMARY_MAX_CHARS);
        int maxSummaryChars = EnvConfig.requireInt(dotEnv, "LLM_SUMMARY_MAX_CHARS");
        // How many failures in a row mean the session is broken rather than the class. See
        // RunAbortedException for what this was added after.
        int maxConsecutiveFailures = EnvConfig.requireInt(dotEnv, "LLM_MAX_CONSECUTIVE_FAILURES");
        // How much of each class reaches the prompt. Previously four constants inside
        // ClassFeatureExtractor with no record in .env or prompts.json.
        FeatureLimits featureLimits = FeatureLimits.fromEnv(dotEnv);
        System.out.println("LLM prompt feature limits (" + FeatureLimits.FIELD_LIMIT_KEY + " etc.): " + featureLimits);
        try {
            if (config == null || config.isBlank()) {
                String alias = cliPromptAlias != null ? cliPromptAlias : defaultPromptAlias();
                contexts.add(new PromptRunContext(alias, safeDefault, client, maxSummaryChars, featureLimits, maxConsecutiveFailures));
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
                contexts.add(new PromptRunContext(alias, resolvedOutput, client, maxSummaryChars, featureLimits, maxConsecutiveFailures));
            }
        } catch (IOException | RuntimeException ex) {
            closePromptRuns(contexts, false); // Setup failed: never publish a CSV from a run that did not start
            throw ex;
        }

        if (contexts.isEmpty()) {
            contexts.add(new PromptRunContext(defaultPromptAlias(), safeDefault, client, maxSummaryChars, featureLimits, maxConsecutiveFailures));
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
        if (alias.equals(defaultPromptAlias())) {
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
    private void closePromptRuns(List<PromptRunContext> promptRuns, boolean commit) {
        if (promptRuns == null) {
            return;
        }
        for (PromptRunContext context : promptRuns) {
            if (context == null) {
                continue;
            }
            try {
                // A run with no rows has nothing to publish; committing it would replace a complete
                // CSV with a header and call that success.
                if (commit && context.writer.getRowsWritten() > 0) {
                    context.writer.commit();
                    System.out.printf("  Wrote %d row(s) to %s%n", context.writer.getRowsWritten(), context.outputCsvPath);
                }
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

        PromptRunContext(String alias, String outputCsvPath, LlmClient client, int maxSummaryChars,
                         FeatureLimits featureLimits, int maxConsecutiveFailures) throws IOException {
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
            if (featureLimits == null) {
                throw new IllegalArgumentException("featureLimits must not be null");
            }
            // this.summaryService = new LlmSummaryService(client); // Original single-prompt builder retained for reference
            // this.summaryService = new LlmSummaryService(client, new LlmPromptBuilder(alias)); // Limits were compiled into ClassFeatureExtractor
            this.summaryService = new LlmSummaryService(client, new LlmPromptBuilder(alias), featureLimits,
                    maxConsecutiveFailures); // Inject prompt-specific builder and .env-driven feature limits
            this.writer = new LlmSummaryWriter(outputCsvPath, maxSummaryChars);
        }

        @Override
        public void close() throws IOException {
            writer.close();
        }
    }

    private static final class SummaryAccumulator {
        private final String outputPath;
        private final List<MissingClass> missingClasses = new ArrayList<>();
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
                missingClasses.addAll(stats.getMissingClasses());
                if (stats.getSuccessfulSummaries() > 0) {
                    projectsWithSummaries++;
                }
            }
        }

        String getOutputPath() {
            return outputPath;
        }

        void recordFailure(String projectPath, String reason) {
            projectsAttempted++;
            totalFailures++;
            // A project that threw before any class was summarised loses every class in it,
            // so record the directory itself rather than leaving only an incremented counter.
            missingClasses.add(new MissingClass(projectPath, "(entire project)", "failed", reason));
        }

        List<MissingClass> getMissingClasses() {
            return missingClasses;
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

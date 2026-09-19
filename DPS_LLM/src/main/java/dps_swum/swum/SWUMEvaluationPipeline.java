package dps_swum.swum;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.ObjectWriter;

import common.projectparser.ParseProject;
import common.projectparser.ProjectJsonStore;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;

/**
 * Orchestrates SWUM processing for all available DPS summaries.
 * <p>
 * This pipeline manages the complete workflow for processing multiple JSON files
 * through the SWUM summarizer. It handles directory setup, batch processing,
 * progress tracking, and summary generation. Note that Java-based evaluation
 * has been removed; downstream analysis is performed by the Python evaluation script.
 * </p>
 * <p>
 * Key responsibilities:
 * <ul>
 *   <li>Create and verify output directory structure</li>
 *   <li>Discover and process all JSON files in the input directory</li>
 *   <li>Track processing progress and statistics</li>
 *   <li>Generate both JSON and CSV output</li>
 *   <li>Create processing summary reports</li>
 * </ul>
 * </p>
 * 
 * @author Najam
 */
public class SWUMEvaluationPipeline {

    // DPS_SWUM used to read DPS_NLG's JSON directory, which meant SWUM could not run unless DPS_NLG
    // had run first and SWUM's "own" representation was really NLG's. SWUM now parses the corpus
    // itself and writes its own JSON representation, then applies SWUM grammar to that file.
    // private static final String INPUT_DIR = "output/json-output/nlg";
    private static final String SOURCE_DIR = "input/dataset";
    private static final String SWUM_OUTPUT_DIR = "output/json-output/swum";
    /** SWUM's own JSON representation of the corpus lives beside its grammar output. */
    private static final String INPUT_DIR = SWUM_OUTPUT_DIR;
    private static final String SUMMARY_OUTPUT_DIR = "evaluation-results";

    private final SWUMSummarizer swumSummarizer;
    private final ObjectMapper objectMapper;

    /**
     * Constructs a new SWUM evaluation pipeline.
     * <p>
     * Initializes the SWUM summarizer and JSON object mapper.
     * </p>
     */
    public SWUMEvaluationPipeline() {
        this.swumSummarizer = new SWUMSummarizer();
        this.objectMapper = new ObjectMapper();
    }

    /**
     * Runs the complete SWUM processing pipeline.
     * <p>
     * This method orchestrates:
     * <ol>
     *   <li>Output directory creation</li>
     *   <li>Batch processing of all JSON files</li>
     *   <li>CSV summary generation</li>
     *   <li>Processing summary report creation</li>
     * </ol>
     * </p>
     */
    public void runCompletePipeline() {
        try {
            System.out.println("=== SWUM Processing Pipeline ===");

            createOutputDirectories();

            System.out.println("\nStep 1: Parsing the corpus into SWUM's own JSON representation...");
            int parsedCount = parseCorpusToJson();
            System.out.println("Parsed " + parsedCount + " projects into " + SWUM_OUTPUT_DIR);

            System.out.println("\nStep 2: Applying SWUM grammar to that JSON representation...");
            int processedCount = processAllFilesWithSWUM();

            System.out.println("\nStep 3: Creating processing summary...");
            generateProcessingSummary(processedCount);

            System.out.println("\n=== Pipeline Complete ===");
            System.out.println("SWUM JSON representation and grammar output written to: " + SWUM_OUTPUT_DIR);
            System.out.println("SWUM CSV summaries written to: output/summary-output/swum_summaries.csv");
            System.out.println("Processing summary written to: " + SUMMARY_OUTPUT_DIR);
        } catch (Exception e) {
            System.err.println("Pipeline failed: " + e.getMessage());
            e.printStackTrace();
        }
    }

    /**
     * Processes a single DPS JSON file and emits the corresponding SWUM output.
     */
    public void processSingleFile(String inputFilePath, String outputFilePath) throws IOException {
        System.out.println("Processing: " + inputFilePath);

        JsonNode originalJson = objectMapper.readTree(new File(inputFilePath));
        JsonNode swumJson = swumSummarizer.processProjectJson(originalJson);

        try (FileWriter writer = new FileWriter(outputFilePath)) {
            objectMapper.writerWithDefaultPrettyPrinter().writeValue(writer, swumJson);
        }

        System.out.println("SWUM output saved to: " + outputFilePath);
    }

    /**
     * Parses the source corpus and writes SWUM's own JSON representation of it.
     * <p>
     * The representation is the same structural feature set the other pipelines extract — SWUM
     * differs in what it does with that representation, not in how it reads the code — but it is
     * produced and owned by this pipeline, so DPS_SWUM no longer requires DPS_NLG to have run.
     * </p>
     *
     * @return the number of projects written
     */
    private int parseCorpusToJson() throws IOException {
        File sourceDir = new File(SOURCE_DIR);
        if (!sourceDir.exists() || !sourceDir.isDirectory()) {
            throw new IOException("Source corpus not found: " + sourceDir.getAbsolutePath());
        }

        List<File> projectDirs = new ArrayList<>();
        findProjectDirectoriesRecursive(sourceDir, projectDirs);
        projectDirs.sort(Comparator.comparing(File::getAbsolutePath));

        ParseProject parseProject = new ParseProject();
        ParseProject.resetDuplicateTracking();
        ObjectWriter writer = ProjectJsonStore.newWriter();

        int written = 0;
        for (File projectDir : projectDirs) {
            String relativePath = sourceDir.toPath().relativize(projectDir.toPath())
                    .toString().replace("\\", "/");
            try {
                HashMap<String, Object> parsedProject = parseProject.parseProject(projectDir, relativePath);
                if (parsedProject.isEmpty()) {
                    System.out.println("\t" + relativePath + ": empty");
                    continue;
                }
                String jsonFileName = relativePath.replace("/", "_").replace("\\", "_");
                ProjectJsonStore.write(writer, new File(SWUM_OUTPUT_DIR, jsonFileName + ".json"), parsedProject);
                written++;
            } catch (Exception e) {
                System.err.println("\tError parsing " + relativePath + ": " + e.getMessage());
            }
        }
        return written;
    }

    /** Collects every directory that directly contains at least one .java file. */
    private void findProjectDirectoriesRecursive(File directory, List<File> projectDirs) {
        if (!directory.isDirectory()) {
            return;
        }
        File[] javaFiles = directory.listFiles((dir, name) -> name.endsWith(".java"));
        if (javaFiles != null && javaFiles.length > 0) {
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
     * Processes every JSON file in the DPS output directory.
     *
     * @return number of projects that were processed successfully
     */
    private int processAllFilesWithSWUM() throws IOException {
        File inputDir = new File(INPUT_DIR);
        File outputDir = new File(SWUM_OUTPUT_DIR);

        if (!inputDir.exists() || !inputDir.isDirectory()) {
            throw new IOException("SWUM JSON directory not found: " + inputDir.getAbsolutePath());
        }

        File[] jsonFiles = inputDir.listFiles((dir, name) -> 
            name.endsWith(".json") && !name.endsWith("_swum.json"));
        if (jsonFiles == null || jsonFiles.length == 0) {
            throw new IOException("No JSON files found in " + inputDir.getAbsolutePath());
        }

        Arrays.sort(jsonFiles, Comparator.comparing(File::getName));

        int processed = 0;
        for (File jsonFile : jsonFiles) {
            try {
                String outputFileName = jsonFile.getName().replace(".json", "_swum.json");
                File outputFile = new File(outputDir, outputFileName);
                String projectName = jsonFile.getName().replace(".json", "");

                swumSummarizer.processJsonFileAndWriteCsv(
                    jsonFile.getAbsolutePath(),
                    outputFile.getAbsolutePath(),
                    projectName
                );
                processed++;

                System.out.printf("Progress: %d/%d files processed%n", processed, jsonFiles.length);
            } catch (Exception ex) {
                System.err.println("Error processing " + jsonFile.getName() + ": " + ex.getMessage());
            }
        }

        SWUMSummarizer.closeCsvWriter();

        System.out.printf("SWUM processing complete: %d/%d files processed successfully%n", processed, jsonFiles.length);
        System.out.println("SWUM CSV summaries saved to: output/summary-output/swum_summaries.csv");
        return processed;
    }

    /**
     * Writes a short text summary to guide analysts toward the Python evaluation script.
     */
    private void generateProcessingSummary(int processedCount) throws IOException {
        File swumDir = new File(SWUM_OUTPUT_DIR);
        File[] sampleOutputs = swumDir.listFiles((dir, name) -> name.endsWith("_swum.json"));

        Files.createDirectories(Paths.get(SUMMARY_OUTPUT_DIR));
        File summaryFile = new File(SUMMARY_OUTPUT_DIR, "pipeline_summary.txt");

        try (FileWriter writer = new FileWriter(summaryFile)) {
            writer.write("=== SWUM Processing Summary ===\n\n");
            writer.write(String.format("Projects processed: %d%n", processedCount));
            writer.write(String.format("SWUM output directory: %s%n%n", swumDir.getAbsolutePath()));

            writer.write("Sample outputs:\n");
            if (sampleOutputs != null && sampleOutputs.length > 0) {
                Arrays.sort(sampleOutputs, Comparator.comparing(File::getName));
                int limit = Math.min(sampleOutputs.length, 5);
                for (int i = 0; i < limit; i++) {
                    writer.write("- " + sampleOutputs[i].getName() + System.lineSeparator());
                }
                if (sampleOutputs.length > limit) {
                    writer.write(String.format("- ... (%d more)%n", sampleOutputs.length - limit));
                }
            } else {
                writer.write("- No SWUM outputs were generated.\n");
            }

            writer.write("\nNext steps:\n");
            writer.write("- Review generated SWUM summaries under output/json-output/ (*_swum.json files).\n");
            writer.write("- SWUM CSV summaries available at output/summary-output/swum_summaries.csv\n");
            writer.write("- Run python/evaluate_summaries.py to compute similarity metrics if required.\n");
        }

        System.out.println("Processing summary saved to: " + summaryFile.getAbsolutePath());
    }

    private void createOutputDirectories() throws IOException {
        Files.createDirectories(Paths.get(SWUM_OUTPUT_DIR));
        Files.createDirectories(Paths.get(SUMMARY_OUTPUT_DIR));
    }

    public static void main(String[] args) {
        SWUMEvaluationPipeline pipeline = new SWUMEvaluationPipeline();

        if (args.length == 0) {
            pipeline.runCompletePipeline();
        } else if (args.length == 2) {
            try {
                pipeline.processSingleFile(args[0], args[1]);
            } catch (IOException e) {
                System.err.println("Error processing file: " + e.getMessage());
                System.exit(1);
            }
        } else {
            System.out.println("Usage:");
            System.out.println("  java SWUMEvaluationPipeline                    # Run complete pipeline");
            System.out.println("  java SWUMEvaluationPipeline <input> <output>   # Process single file");
            System.exit(1);
        }
    }

    public static void listAvailableFiles() {
        File inputDir = new File(INPUT_DIR);
        if (!inputDir.exists()) {
            System.out.println("Input directory not found: " + INPUT_DIR);
            return;
        }

        File[] jsonFiles = inputDir.listFiles((dir, name) -> 
            name.endsWith(".json") && !name.endsWith("_swum.json"));
        if (jsonFiles == null || jsonFiles.length == 0) {
            System.out.println("No JSON files found in " + INPUT_DIR);
            return;
        }

        Arrays.sort(jsonFiles, Comparator.comparing(File::getName));
        System.out.println("Available JSON files:");
        for (File file : jsonFiles) {
            System.out.println("  " + file.getName());
        }
    }
}


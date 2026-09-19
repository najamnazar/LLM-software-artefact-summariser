package dps_nlg;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

import java.util.Arrays;
import java.util.Comparator;

import com.fasterxml.jackson.databind.ObjectWriter;

import common.projectparser.ParseProject;
import common.projectparser.ProjectJsonStore;
import dps_nlg.summarygenerator.Summarise;

/**
 * Main application entry point for NLG-based design pattern summarization.
 * <p>
 * This application processes Java projects using Natural Language Generation (NLG)
 * techniques with the SimpleNLG library to produce human-readable summaries of classes,
 * methods, and design pattern implementations. It recursively discovers Java projects,
 * parses their structure, and generates both JSON output and CSV summaries.
 * </p>
 * <p>
 * Key responsibilities:
 * <ul>
 *   <li>Discover all Java project directories recursively</li>
 *   <li>Parse project structures using the DPS parser</li>
 *   <li>Generate NLG-based summaries for each project</li>
 *   <li>Write structured JSON output and CSV summary files</li>
 *   <li>Track duplicate files and processing statistics</li>
 * </ul>
 * </p>
 * 
 * @author Najam
 */
public class Application {

    /** DPS_NLG owns this directory: phase 1 writes it, phase 2 reads it back. */
    private static final String JSON_OUTPUT_DIR = "output/json-output/nlg";

    /**
     * Main entry point for the NLG summarization application.
     * <p>
     * Initiates the complete processing workflow including directory setup,
     * project discovery, parsing, and summary generation.
     * </p>
     * 
     * @param args command line arguments (currently unused)
     */
    public static void main(String[] args) {
        try {
            runApplication();
        } catch (IOException e) {
            System.err.println("Fatal error during application execution: " + e.getMessage());
            e.printStackTrace();
            System.exit(1);
        } catch (Exception e) {
            System.err.println("Unexpected error: " + e.getMessage());
            e.printStackTrace();
            System.exit(1);
        }
    }
    
    private static void runApplication() throws IOException {
        ParseProject parseProject = new ParseProject();

        // Create output and reference Directory if non-existent
        createDirectories();
        
        // Reset duplicate tracking
        ParseProject.resetDuplicateTracking();

        // Recursively find all directories containing Java files
        File inputDir = new File("input/dataset");
        if (!inputDir.exists() || !inputDir.isDirectory()) {
            throw new IOException("Input directory not found or is not a directory");
        }
        
        List<File> projectDirs = findAllProjectDirectories(inputDir);
        System.out.println("Found " + projectDirs.size() + " project directories to process\n");

        // Bug 6 fix: ObjectWriter was constructed inside processProject(), creating a new ObjectMapper
        // and writer instance for every project in the batch. Constructed once here and passed in.
        ObjectWriter writer = ProjectJsonStore.newWriter();

        // Phase 1: parse each project and write its JSON representation. Nothing is summarised here.
        System.out.println("=== Phase 1: parsing projects into " + JSON_OUTPUT_DIR + " ===");
        int written = 0;
        for (File project : projectDirs) {
            if (processProject(project, parseProject, inputDir, writer)) {
                written++;
            }
        }
        System.out.println("\nPhase 1 complete: " + written + " project JSON files written.");

        // Phase 2: read those JSON files back and generate the NLG summaries from them. Summarisation
        // used to run inside ParseProject on the in-memory parse, which meant the JSON was written
        // after the fact and never read. NLG now consumes the same representation SWUM consumes.
        System.out.println("\n=== Phase 2: generating NLG summaries from " + JSON_OUTPUT_DIR + " ===");
        int summarised = summariseAllJson(writer);

        // Close the CSV writer to finalize the summary file
        Summarise.closeCsvWriter();

        // Report duplicate statistics
        int skippedCount = ParseProject.getSkippedDuplicatesCount();
        System.out.println("\nAll projects processed. CSV summary file has been generated.");
        System.out.println("Summarised " + summarised + " project JSON files into output/summary-output/nlg_summaries.csv");
        if (skippedCount > 0) {
            System.out.println("Skipped " + skippedCount + " duplicate files (same name and content).");
        }
    }

    /**
     * Reads every project JSON file written by phase 1 and generates its NLG summaries.
     *
     * @param writer the shared writer used to store the summaries back into each JSON file
     * @return the number of project files summarised successfully
     */
    private static int summariseAllJson(ObjectWriter writer) {
        File jsonDir = new File(JSON_OUTPUT_DIR);
        File[] jsonFiles = jsonDir.listFiles((dir, name) -> name.endsWith(".json"));
        if (jsonFiles == null || jsonFiles.length == 0) {
            System.err.println("No JSON files found in " + jsonDir.getAbsolutePath() + "; nothing to summarise.");
            return 0;
        }
        Arrays.sort(jsonFiles, Comparator.comparing(File::getName));

        NlgJsonSummariser summariser = new NlgJsonSummariser();
        int summarised = 0;
        for (File jsonFile : jsonFiles) {
            try {
                int rows = summariser.summariseJsonFile(jsonFile, writer);
                System.out.println("\t" + jsonFile.getName() + " -> " + rows + " class summaries");
                summarised++;
            } catch (Exception e) {
                System.err.println("\tError summarising " + jsonFile.getName() + ": " + e.getMessage());
                e.printStackTrace();
            }
        }
        return summarised;
    }
    
    /**
     * Recursively find all directories that contain Java files
     */
    private static List<File> findAllProjectDirectories(File directory) {
        List<File> projectDirs = new ArrayList<>();
        findProjectDirectoriesRecursive(directory, projectDirs);
        return projectDirs;
    }
    
    private static void findProjectDirectoriesRecursive(File directory, List<File> projectDirs) {
        if (!directory.isDirectory()) {
            return;
        }
        
        // Check if this directory contains any Java files
        File[] javaFiles = directory.listFiles((dir, name) -> name.endsWith(".java"));
        boolean hasJavaFiles = javaFiles != null && javaFiles.length > 0;
        
        if (hasJavaFiles) {
            projectDirs.add(directory);
        }
        
        // Recurse into subdirectories
        File[] subdirs = directory.listFiles(File::isDirectory);
        if (subdirs != null) {
            for (File subdir : subdirs) {
                findProjectDirectoriesRecursive(subdir, projectDirs);
            }
        }
    }
    
    private static void createDirectories() throws IOException {
        String[] directories = {"output", "output/json-output", JSON_OUTPUT_DIR, "output/summary-output", "reference"};
        
        for (String dirPath : directories) {
            File dir = new File(dirPath);
            if (!dir.exists() && !dir.mkdirs()) {
                throw new IOException("Failed to create directory: " + dirPath);
            }
        }
    }
    
    // Bug 6 fix: added ObjectWriter parameter so the caller (runApplication) constructs it once and
    // reuses it across all projects rather than creating a new ObjectMapper per project.
    // private static void processProject(File project, ParseProject parseProject, File inputDir) throws IOException {
    private static boolean processProject(File project, ParseProject parseProject, File inputDir, ObjectWriter writer) throws IOException {
        // Calculate relative path from input directory
        String relativePath = inputDir.toPath().relativize(project.toPath()).toString().replace("\\", "/");
        System.out.println("\n" + relativePath);
        HashMap<String, Object> parsedProject;

        try {
            // Each directory in input folder is parsed with its relative path
            // The third argument used to be a generateNlgSummary flag; summarisation is now phase 2.
            // parsedProject = parseProject.parseProject(project, relativePath, true);
            parsedProject = parseProject.parseProject(project, relativePath);
        } catch (Exception e) {
            System.err.println("\tError during project " + relativePath + ": " + e.getMessage());
            e.printStackTrace();
            return false; // Continue with next project instead of throwing
        }

        // Bug 6 fix: writer is now received as a parameter; removed per-project instantiation.
        // ObjectWriter writer = new ObjectMapper()
        //         .writer(new DefaultPrettyPrinter().withObjectIndenter(new DefaultIndenter("\t", "\n")));

        if (parsedProject.isEmpty()) {
            System.out.println("\tEmpty");
            return false;
        }

        // Use relative path for JSON filename (sanitize for filesystem)
        String jsonFileName = relativePath.replace("/", "_").replace("\\", "_");
        ProjectJsonStore.write(writer, new File(JSON_OUTPUT_DIR + "/" + jsonFileName + ".json"), parsedProject);
        return true;
    }
}


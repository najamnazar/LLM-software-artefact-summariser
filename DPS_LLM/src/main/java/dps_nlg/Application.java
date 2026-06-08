package dps_nlg;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

import com.fasterxml.jackson.core.util.DefaultIndenter;
import com.fasterxml.jackson.core.util.DefaultPrettyPrinter;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.ObjectWriter;

import common.projectparser.ParseProject;
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
        File inputDir = new File("input");
        if (!inputDir.exists() || !inputDir.isDirectory()) {
            throw new IOException("Input directory not found or is not a directory");
        }
        
        List<File> projectDirs = findAllProjectDirectories(inputDir);
        System.out.println("Found " + projectDirs.size() + " project directories to process\n");

        // Bug 6 fix: ObjectWriter was constructed inside processProject(), creating a new ObjectMapper
        // and writer instance for every project in the batch. Constructed once here and passed in.
        ObjectWriter writer = new ObjectMapper()
                .writer(new DefaultPrettyPrinter().withObjectIndenter(new DefaultIndenter("\t", "\n")));

        for (File project : projectDirs) {
            processProject(project, parseProject, inputDir, writer);
        }
        
        // Close the CSV writer to finalize the summary file
        Summarise.closeCsvWriter();
        
        // Report duplicate statistics
        int skippedCount = ParseProject.getSkippedDuplicatesCount();
        System.out.println("\nAll projects processed. CSV summary file has been generated.");
        if (skippedCount > 0) {
            System.out.println("Skipped " + skippedCount + " duplicate files (same name and content).");
        }
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
        String[] directories = {"output", "output/json-output", "output/json-output/nlg", "output/summary-output", "reference"};
        
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
    private static void processProject(File project, ParseProject parseProject, File inputDir, ObjectWriter writer) throws IOException {
        // Calculate relative path from input directory
        String relativePath = inputDir.toPath().relativize(project.toPath()).toString().replace("\\", "/");
        System.out.println("\n" + relativePath);
        HashMap<String, Object> parsedProject;

        try {
            // Each directory in input folder is parsed with its relative path
            parsedProject = parseProject.parseProject(project, relativePath, true);
        } catch (Exception e) {
            System.err.println("\tError during project " + relativePath + ": " + e.getMessage());
            e.printStackTrace();
            return; // Continue with next project instead of throwing
        }

        // Bug 6 fix: writer is now received as a parameter; removed per-project instantiation.
        // ObjectWriter writer = new ObjectMapper()
        //         .writer(new DefaultPrettyPrinter().withObjectIndenter(new DefaultIndenter("\t", "\n")));

        if (parsedProject.isEmpty()) {
            System.out.println("\tEmpty");
            return;
        }
        
        // Use relative path for JSON filename (sanitize for filesystem)
        String jsonFileName = relativePath.replace("/", "_").replace("\\", "_");
        writer.writeValue(new File("output/json-output/nlg/" + jsonFileName + ".json"), parsedProject);
    }
}


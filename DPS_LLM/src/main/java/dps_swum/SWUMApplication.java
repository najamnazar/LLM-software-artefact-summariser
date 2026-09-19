package dps_swum;

import java.io.File;
import java.io.IOException;

import dps_swum.swum.SWUMEvaluationPipeline;

/**
 * Entry point for the SWUM-based design pattern summarization application.
 * <p>
 * This application parses the source corpus into its own JSON representation and then applies
 * Software Word Usage Model (SWUM) grammar to that representation to generate code summaries.
 * SWUM analyzes identifier names and code structure to produce natural language descriptions.
 * </p>
 * <p>
 * Key responsibilities:
 * <ul>
 *   <li>Verify that the source corpus exists</li>
 *   <li>Create SWUM output directories</li>
 *   <li>Initialize and run the SWUM evaluation pipeline</li>
 *   <li>Handle errors and report status</li>
 * </ul>
 * </p>
 * 
 * @author Najam
 */
public class SWUMApplication {

    /**
     * Main entry point for the SWUM summarization application.
     * <p>
     * Checks for required input directories, creates output directories,
     * and runs the complete SWUM processing pipeline.
     * </p>
     * 
     * @param args command line arguments (currently unused)
     * @throws IOException if file I/O operations fail
     */
    public static void main(String[] args) throws IOException {
        System.out.println("Starting SWUM-based Design Pattern Summarizer...");
        
        // SWUM parses the corpus itself, so the only precondition is the source corpus. It used to
        // require output/json-output to exist because it read DPS_NLG's JSON files.
        // File jsonOutputDir = new File("output/json-output");
        File sourceDir = new File("input/dataset");
        if (!sourceDir.exists() || !sourceDir.isDirectory()) {
            System.err.println("Error: source corpus not found at: " + sourceDir.getAbsolutePath());
            System.err.println("DPS_SWUM parses input/dataset directly; it no longer depends on DPS_NLG output.");
            return;
        }

        File swumOutputDir = new File("output/json-output/swum");
        if (!swumOutputDir.exists()) {
            swumOutputDir.mkdirs();
        }
        
        try {
            // Run SWUM evaluation pipeline
            SWUMEvaluationPipeline pipeline = new SWUMEvaluationPipeline();
            pipeline.runCompletePipeline();
            
            System.out.println("\nSWUM processing completed successfully!");
            System.out.println("SWUM JSON output available in: " + swumOutputDir.getAbsolutePath());
            System.out.println("SWUM CSV summaries available in: output/summary-output/swum_summaries.csv");
            
        } catch (Exception e) {
            System.err.println("Error during SWUM processing: " + e.getMessage());
            e.printStackTrace();
        }
    }
}


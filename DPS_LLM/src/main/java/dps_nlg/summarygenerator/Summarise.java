package dps_nlg.summarygenerator;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.*;

import org.apache.commons.collections4.MultiValuedMap;

import simplenlg.framework.NLGFactory;
import simplenlg.lexicon.Lexicon;
import simplenlg.realiser.english.Realiser;
import common.utils.ProjectPathFormatter;
import common.utils.Utils;

/**
 * Main summarization orchestrator using Natural Language Generation.
 * <p>
 * This class coordinates the generation of natural language summaries for Java projects
 * by combining class/interface descriptions, method summaries, and design pattern information.
 * It uses the SimpleNLG library to generate grammatically correct English descriptions
 * and writes output to both console and CSV files.
 * </p>
 * <p>
 * Key responsibilities:
 * <ul>
 *   <li>Coordinate NLG-based summary generation for projects</li>
 *   <li>Integrate class, method, and design pattern descriptions</li>
 *   <li>Write summaries to CSV format with proper escaping</li>
 *   <li>Manage static CSV writer for accumulating results</li>
 *   <li>Format summaries using ProjectPathFormatter</li>
 * </ul>
 * </p>
 * 
 * @author Najam
 */
public class Summarise {
    
    // Static CSV writer to accumulate all summaries
    private static FileWriter csvWriter = null;
    private static boolean csvHeaderWritten = false;
    
    /**
     * Generates summaries for all files in a project.
     * <p>
     * Processes each Java file to create comprehensive summaries that include
     * class descriptions, method information, and design pattern context.
     * Summaries are written to a CSV file with project metadata.
     * </p>
     * 
     * @param fileDetails map of file names to their parsed class data
     * @param designPatternDetails list of identified design patterns
     * @param summary multi-valued map for organizing pattern-specific summaries
     * @param projectPath the project path identifier
     * @return consolidated project summary text
     * @throws IOException if CSV writing fails
     */
    public String summarise(HashMap<String, HashMap> fileDetails,
            ArrayList<HashMap> designPatternDetails,
            HashMap<String, MultiValuedMap<String, String>> summary, String projectPath) throws IOException {

        ClassInterfaceSummariser classInterfaceSummariser = new ClassInterfaceSummariser();
        DesignPatternSummarise designPatternSummarise = new DesignPatternSummarise();
        MethodSummariser methodSummariser = new MethodSummariser();

        Lexicon lexicon = Lexicon.getDefaultLexicon();
        NLGFactory nlgFactory = new NLGFactory(lexicon);
        Realiser realiser = new Realiser(lexicon);

        String projectSummary = "";

        // Initialize CSV file if not already done
        initializeCsvWriter();

        // if the project has a design pattern, include the multivalue map to store values
        if (!designPatternDetails.isEmpty()) {
            designPatternSummarise.summarise(fileDetails, designPatternDetails, summary);
        }
        
        // Process each file individually and write separate CSV rows
        for (Map.Entry<String, HashMap> fileEntry : fileDetails.entrySet()) {
            String file = fileEntry.getKey();
            String fileSummary = "";
            
            // Check if this file has any design pattern summaries
            boolean hasDesignPatterns = false;
            for (String designPattern : summary.keySet()) {
                if (summary.get(designPattern).containsKey(file) && 
                    !summary.get(designPattern).get(file).isEmpty()) {
                    hasDesignPatterns = true;
                    break;
                }
            }
            
            if (hasDesignPatterns) {
                // Generate summary for files with design patterns
                for (String designPattern : summary.keySet()) {
                    if (summary.get(designPattern).containsKey(file) && 
                        !summary.get(designPattern).get(file).isEmpty()) {
                        
                        HashSet<String> fileSummarySet = new HashSet<>();
                        for (String summary_text : summary.get(designPattern).get(file))
                            fileSummarySet.add(summary_text);

                        // generate class detail description, put summary as a parameter so that
                        // design pattern details shall be included.
                        ArrayList classDetails = Utils.getClassOrInterfaceDetails(fileEntry.getValue());
                        if (classDetails.size() == 0) {
                            continue;
                        }
                        HashMap classDetail = (HashMap) Utils.getClassOrInterfaceDetails(fileEntry.getValue()).get(0);
                        String classDescription = classInterfaceSummariser.generateClassDescription(nlgFactory,
                                realiser, classDetail, fileSummarySet);
                        // generate method description, as well as method usage description, merge into
                        // method summary
                        ArrayList<HashMap> methodDetails = Utils.getMethodDetails(fileEntry.getValue());
                        if (methodDetails.size() != 0) {
                            String methodDescription = methodSummariser.generateMethodsSummary(nlgFactory, realiser,
                                    methodDetails, file);
                            String methodUsageDescription = methodSummariser.generateMethodDescription(nlgFactory, realiser,
                                    methodDetails);
                            String methodSummary = methodDescription + " " + methodUsageDescription;
                            classDescription += " " + methodSummary;
                        }
                        
                        if (!fileSummary.isEmpty()) {
                            fileSummary += " ";
                        }
                        fileSummary += designPattern + ": " + classDescription;
                    }
                }
            } else {
                // Generate summary for files without design patterns
                ArrayList classDetails = Utils.getClassOrInterfaceDetails(fileEntry.getValue());
                if (classDetails.size() > 0) {
                    HashMap classDetail = (HashMap) classDetails.get(0);
                    String classDescription = classInterfaceSummariser.generateClassDescription(nlgFactory,
                            realiser, classDetail, new HashSet<>());
                    
                    ArrayList<HashMap> methodDetails = Utils.getMethodDetails(fileEntry.getValue());
                    if (methodDetails.size() != 0) {
                        String methodDescription = methodSummariser.generateMethodsSummary(nlgFactory, realiser,
                                methodDetails, file);
                        String methodUsageDescription = methodSummariser.generateMethodDescription(nlgFactory, realiser,
                                methodDetails);
                        String methodSummary = methodDescription + " " + methodUsageDescription;
                        classDescription += " " + methodSummary;
                    }
                    
                    fileSummary = classDescription;
                }
            }
            
            // Write individual file summary to CSV
            if (!fileSummary.isEmpty()) {
                // Convert class name to Java filename (e.g., "VideoConversionFacade" -> "VideoConversionFacade.java")
                String javaFilename = file + ".java";
                writeToCsv(projectPath, javaFilename, fileSummary);
                projectSummary += javaFilename + ": " + fileSummary + "\n";
            }
        }

        return projectSummary;
    }

    /**
     * Initialize the CSV writer for summary output
     */
    private static void initializeCsvWriter() throws IOException {
        if (csvWriter == null) {
            // Ensure summary-output directory exists
            File summaryOutputDir = new File("output/summary-output");
            if (!summaryOutputDir.exists()) {
                summaryOutputDir.mkdirs();
            }
            
            File csvFile = new File("output/summary-output/nlg_summaries.csv");
            csvWriter = new FileWriter(csvFile, false); // false = overwrite existing file
            
            // Write CSV header
            if (!csvHeaderWritten) {
                csvWriter.write("Project,Folder Name,File Name,Summary\n");
                csvHeaderWritten = true;
            }
            
            System.out.println("Initialized CSV summary file: " + csvFile.getAbsolutePath());
        }
    }

    /**
     * Write a single project summary to the CSV file
     */
    private static void writeToCsv(String projectPath, String fileName, String summary) throws IOException {
        if (csvWriter == null) {
            initializeCsvWriter();
        }

        // Clean up the summary for CSV format
        String cleanSummary = summary.replace("\"", "\"\""); // Escape quotes
        cleanSummary = cleanSummary.replace("\r\n", " ").replace("\n", " ").replace("\r", " "); // Remove newlines
        cleanSummary = cleanSummary.trim();

        // Limit summary length to prevent CSV issues
        if (cleanSummary.length() > 1000) {
            cleanSummary = cleanSummary.substring(0, 1000) + "...";
        }

        ProjectPathFormatter.Parts parts = ProjectPathFormatter.split(projectPath);
        String projectName = escape(parts.projectName());
        String folderName = escape(parts.folderName());
        String cleanFileName = escape(fileName);

        csvWriter.write(String.format("\"%s\",\"%s\",\"%s\",\"%s\"\n",
                projectName,
                folderName,
                cleanFileName,
                cleanSummary));
        csvWriter.flush(); // Ensure data is written immediately
    }

    private static String escape(String value) {
        if (value == null) {
            return "";
        }
        return value.replace("\"", "\"\"").trim();
    }

    /**
     * Close the CSV writer - call this when all processing is complete
     */
    public static void closeCsvWriter() throws IOException {
        if (csvWriter != null) {
            csvWriter.close();
            csvWriter = null;
            csvHeaderWritten = false;
            System.out.println("CSV summary file closed successfully.");
        }
    }
}


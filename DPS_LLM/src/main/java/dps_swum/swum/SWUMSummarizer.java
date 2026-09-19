package dps_swum.swum;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import dps_swum.swum.context.MethodPatternContext;
import dps_swum.swum.context.PatternContext;
import dps_swum.swum.context.PatternContextExtractor;
import dps_swum.swum.grammar.SWUMGrammarParser;
import dps_swum.swum.model.SWUMStructure;

import common.utils.ProjectPathFormatter;
import common.designpatternidentifier.CheckPattern;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.*;

/**
 * Main SWUM summarizer that processes JSON output and generates code summaries.
 * <p>
 * This class implements the Software Word Usage Model (SWUM) technique to generate
 * natural language summaries of Java code. It processes JSON files produced by the
 * DPS parser, applies SWUM grammar rules to method and class names, and generates
 * structured summaries that incorporate design pattern context.
 * </p>
 * <p>
 * Key responsibilities:
 * <ul>
 *   <li>Parse JSON files containing code structure data</li>
 *   <li>Apply SWUM grammar rules to identifier names</li>
 *   <li>Extract and integrate design pattern information</li>
 *   <li>Generate method-level and class-level summaries</li>
 *   <li>Write summaries to both JSON and CSV formats</li>
 *   <li>Track statistics on actions, objects, and patterns</li>
 * </ul>
 * </p>
 * 
 * @author Najam
 */
public class SWUMSummarizer {
    
    private SWUMGrammarParser parser;
    private ObjectMapper objectMapper;
    
    // Static CSV writer for accumulating all SWUM summaries
    private static FileWriter csvWriter = null;
    private static boolean csvHeaderWritten = false;
    
    /**
     * Constructs a new SWUM summarizer with default configuration.
     * <p>
     * Initializes the SWUM grammar parser and JSON object mapper.
     * </p>
     */
    public SWUMSummarizer() {
        this.parser = new SWUMGrammarParser();
        this.objectMapper = new ObjectMapper();
    }
    
    /**
     * Processes a single JSON file and generates SWUM summaries.
     * <p>
     * Reads the JSON file, extracts class and method information, applies SWUM
     * grammar rules, and writes the results to a new JSON file with SWUM annotations.
     * </p>
     * 
     * @param inputPath path to the input JSON file
     * @param outputPath path to the output JSON file
     * @throws IOException if file I/O operations fail
     */
    public void processJsonFile(String inputPath, String outputPath) throws IOException {
        File inputFile = new File(inputPath);
        JsonNode rootNode = objectMapper.readTree(inputFile);

        // SWUM-fix: build fileDetails from AST features in the NLG JSON and run SWUM's own
        // structural pattern detection — independent of NLG's pre-computed design_pattern field.
        // Old: pattern context was read from rootNode.get("design_pattern") written by NLG.
        HashMap<String, HashMap> swumFileDetails = buildFileDetailsFromJson(rootNode);
        ArrayList<HashMap> swumDetectedPatterns = runStructuralPatternDetection(swumFileDetails);
        final JsonNode swumPatternsNode = objectMapper.valueToTree(swumDetectedPatterns);

        // Create output structure
        ObjectNode swumOutput = objectMapper.createObjectNode();
        
        String projectName = extractProjectName(rootNode);
        swumOutput.put("project_name", projectName);
        swumOutput.put("swum_version", "1.0");
        swumOutput.put("generated_timestamp", System.currentTimeMillis());
        
        // Process each class/interface in the JSON
        ObjectNode classesSummaries = objectMapper.createObjectNode();
        
        // Find the project container (the key that contains all the classes)
        rootNode.fields().forEachRemaining(entry -> {
            String key = entry.getKey();
            JsonNode value = entry.getValue();
            
            // Skip system fields
            // Bug 3 fix (companion change): ParseProject now writes a "project_identifier" provenance
            // field into every JSON file. Without this guard, SWUM would treat it as a project
            // container node and attempt to iterate its string value as a map of classes.
            // if (key.equals("final_summary") || key.equals("design_pattern") || key.equals("summary_NLG")) {
            if (key.equals("final_summary") || key.equals("design_pattern") || key.equals("summary_NLG") || key.equals("project_identifier")) {
                return;
            }
            
            // This key should contain all the classes
            if (value.isObject()) {
                value.fields().forEachRemaining(classEntry -> {
                    String className = classEntry.getKey();
                    JsonNode classData = classEntry.getValue();
                    
                    if (classData.isObject()) {
                        try {
                            // SWUM-fix: pass swumPatternsNode (SWUM's own detection) instead of rootNode
                            // Old: ObjectNode classSummary = processClass(className, classData, rootNode);
                            ObjectNode classSummary = processClass(className, classData, swumPatternsNode);
                            classesSummaries.set(className, classSummary);
                        } catch (Exception e) {
                            System.err.println("Error processing class " + className + ": " + e.getMessage());
                        }
                    }
                });
            }
        });
        
        swumOutput.set("class_summaries", classesSummaries);
        
        // Generate overall project summary using SWUM
        // SWUM-fix: use swumPatternsNode (SWUM's own detection) instead of rootNode
        // Old: String projectSummary = generateProjectSummary(classesSummaries, rootNode);
        String projectSummary = generateProjectSummary(classesSummaries, swumPatternsNode);
        swumOutput.put("swum_project_summary", projectSummary);

        // SWUM-fix: store SWUM's own detected patterns; no longer copies NLG's design_pattern
        swumOutput.set("swum_design_pattern", swumPatternsNode);
        // Old: copied NLG's design_pattern into SWUM output — replaced by independent detection
        // JsonNode designPatterns = rootNode.get("design_pattern");
        // if (designPatterns != null) {
        //     swumOutput.set("design_patterns", designPatterns);
        // }
        
        // Write output
        try (FileWriter writer = new FileWriter(outputPath)) {
            objectMapper.writerWithDefaultPrettyPrinter().writeValue(writer, swumOutput);
        }
        
        System.out.println("SWUM summary generated: " + outputPath);
    }
    
    /**
     * Initializes the CSV writer for SWUM summary output
     */
    private static void initializeCsvWriter() throws IOException {
        if (csvWriter == null) {
            // Ensure summary-output directory exists
            File summaryOutputDir = new File("output/summary-output");
            if (!summaryOutputDir.exists()) {
                summaryOutputDir.mkdirs();
            }
            
            File csvFile = new File("output/summary-output/swum_summaries.csv");
            csvWriter = new FileWriter(csvFile, false); // false = overwrite existing file
            
            // Write CSV header
            if (!csvHeaderWritten) {
                csvWriter.write("Project,Folder Name,File Name,Summary\n");
                csvHeaderWritten = true;
            }
            
            System.out.println("Initialized SWUM CSV summary file: " + csvFile.getAbsolutePath());
        }
    }
    
    /**
     * Writes a single class summary to the CSV file
     */
    public static void writeToCsv(String projectName, String fileName, String summary) throws IOException {
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

        ProjectPathFormatter.Parts parts = ProjectPathFormatter.split(projectName);
        String escapedProject = escape(parts.projectName());
        String escapedFolder = escape(parts.folderName());
        String escapedFile = escape(fileName);

        csvWriter.write(String.format("\"%s\",\"%s\",\"%s\",\"%s\"\n",
                escapedProject,
                escapedFolder,
                escapedFile,
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
     * Closes the CSV writer - call this when all processing is complete
     */
    public static void closeCsvWriter() throws IOException {
        if (csvWriter != null) {
            csvWriter.close();
            csvWriter = null;
            csvHeaderWritten = false;
            System.out.println("SWUM CSV summary file closed successfully.");
        }
    }
    
    /**
     * Processes a JSON file and writes class summaries to CSV
     */
    public void processJsonFileAndWriteCsv(String inputPath, String outputPath, String projectName) throws IOException {
        // Process the JSON file normally
        processJsonFile(inputPath, outputPath);
        
        // Read the generated SWUM JSON
        File outputFile = new File(outputPath);
        if (outputFile.exists()) {
            JsonNode swumJson = objectMapper.readTree(outputFile);
            
            // Extract and write class summaries to CSV
            JsonNode classSummaries = swumJson.get("class_summaries");
            if (classSummaries != null && classSummaries.isObject()) {
                classSummaries.fields().forEachRemaining(entry -> {
                    String className = entry.getKey();
                    JsonNode classData = entry.getValue();

                    // Najam: the CSV used to export swum_class_summary alone and drop method_summaries
                    // entirely, so it carried roughly a quarter of the text SWUM had generated. That is
                    // why SWUM summaries looked so much shorter than the NLG ones, whose CSV rows do
                    // include method-level description. The method sentences SWUM already produced are
                    // now part of the exported summary.
                    // String classSummary = classData.has("swum_class_summary") ?
                    //     classData.get("swum_class_summary").asText() : "No summary available";
                    String classSummary = composeCsvSummary(classData);

                    try {
                        writeToCsv(projectName, className + ".java", classSummary);
                    } catch (IOException e) {
                        System.err.println("Error writing CSV for " + className + ": " + e.getMessage());
                    }
                });
            }
        }
    }
    
    /**
     * Builds the CSV summary for one class from its SWUM output.
     * <p>
     * The class-level sentence comes first, followed by the SWUM sentence for each method, with the
     * subject "This method" replaced by the method name so the sentences remain distinguishable once
     * concatenated. Sentences already produced at class level (the pattern-role sentence is repeated
     * in every method summary) are emitted once.
     * </p>
     *
     * @param classData one entry of the {@code class_summaries} object
     * @return the summary text for the CSV row
     */
    private String composeCsvSummary(JsonNode classData) {
        LinkedHashSet<String> sentences = new LinkedHashSet<>();

        if (classData.has("swum_class_summary")) {
            addSentences(sentences, classData.get("swum_class_summary").asText());
        }

        JsonNode methodSummaries = classData.get("method_summaries");
        if (methodSummaries != null && methodSummaries.isObject()) {
            methodSummaries.fields().forEachRemaining(methodEntry -> {
                JsonNode methodData = methodEntry.getValue();
                if (methodData == null || !methodData.has("swum_summary")) {
                    return;
                }
                String methodSummary = methodData.get("swum_summary").asText();
                // "This method retrieves ..." -> "The getRadius method retrieves ..."
                String named = methodSummary.startsWith("This method ")
                        ? "The " + methodEntry.getKey() + " method "
                                + methodSummary.substring("This method ".length())
                        : methodSummary;
                addSentences(sentences, named);
            });
        }

        if (sentences.isEmpty()) {
            return "No summary available";
        }
        return String.join(" ", sentences);
    }

    /** Splits generated text into sentences, keeping the terminating period on each. */
    private void addSentences(LinkedHashSet<String> sentences, String text) {
        if (text == null || text.isBlank()) {
            return;
        }
        for (String part : text.split("(?<=\\.)\\s+")) {
            String sentence = part.trim();
            if (!sentence.isEmpty()) {
                sentences.add(sentence);
            }
        }
    }

    /**
     * Processes all JSON files and writes summaries to CSV
     */
    public void processAllFilesWithCsv(String inputDir, String outputDir) throws IOException {
        File inputDirectory = new File(inputDir);
        File outputDirectory = new File(outputDir);
        
        if (!outputDirectory.exists()) {
            outputDirectory.mkdirs();
        }
        
        File[] jsonFiles = inputDirectory.listFiles((dir, name) -> name.endsWith(".json"));
        
        if (jsonFiles == null) {
            System.err.println("No JSON files found in " + inputDir);
            return;
        }
        
        int processed = 0;
        for (File jsonFile : jsonFiles) {
            try {
                String outputFileName = jsonFile.getName().replace(".json", "_swum.json");
                String outputPath = new File(outputDirectory, outputFileName).getAbsolutePath();
                String projectName = jsonFile.getName().replace(".json", "");
                
                processJsonFileAndWriteCsv(jsonFile.getAbsolutePath(), outputPath, projectName);
                processed++;
            } catch (Exception e) {
                System.err.println("Error processing " + jsonFile.getName() + ": " + e.getMessage());
                e.printStackTrace();
            }
        }
        
        System.out.println("Processed " + processed + " JSON files with SWUM summarizer and wrote to CSV");
        
        // Close CSV writer after all files are processed
        closeCsvWriter();
    }
    
    /**
     * Processes a single class and generates SWUM summaries for its methods.
     * SWUM-fix: parameter renamed from rootNode to swumPatternsNode — now receives SWUM's own
     * detected patterns (not NLG's rootNode) so pattern context is independently derived.
     */
    // Old: private ObjectNode processClass(String className, JsonNode classData, JsonNode rootNode) {
    //     PatternContext patternContext = PatternContextExtractor.extractContextForClass(className, rootNode);
    private ObjectNode processClass(String className, JsonNode classData, JsonNode swumPatternsNode) {
        ObjectNode classSummary = objectMapper.createObjectNode();

        // SWUM-fix: use SWUM's own detected patterns for context — not NLG's design_pattern field
        PatternContext patternContext = PatternContextExtractor.extractContextForClassFromSwum(className, swumPatternsNode);
        
        // Extract class information
        JsonNode classDetails = classData.get("CLASSORINTERFACEDETAIL");
        String classType = "class";
        if (classDetails != null && classDetails.isArray() && classDetails.size() > 0) {
            JsonNode firstDetail = classDetails.get(0);
            if (firstDetail.has("ISINTERFACEORNOT") && firstDetail.get("ISINTERFACEORNOT").asBoolean()) {
                classType = "interface";
            }
        }
        
        classSummary.put("class_name", className);
        classSummary.put("class_type", classType);
        
        // Process methods
        JsonNode methodDetails = classData.get("METHODDETAIL");
        ObjectNode methodSummaries = objectMapper.createObjectNode();
        List<String> methodNames = new ArrayList<>();
        
        if (methodDetails != null && methodDetails.isArray()) {
            for (JsonNode method : methodDetails) {
                String methodName = method.get("METHODNAME").asText();
                methodNames.add(methodName);
                
                ObjectNode methodSummary = processMethod(methodName, className, method, classData, patternContext);
                methodSummaries.set(methodName, methodSummary);
            }
        }
        
        classSummary.set("method_summaries", methodSummaries);
        
        // Generate class-level SWUM summary with pattern context
        String designPattern = extractDesignPattern(className);
        String classSummaryText = parser.generateClassSummaryWithContext(className, methodNames, designPattern, patternContext);
        classSummary.put("swum_class_summary", classSummaryText);
        
        return classSummary;
    }
    
    /**
     * Processes a single method and generates SWUM summary
     */
    private ObjectNode processMethod(String methodName, String className, JsonNode methodData, 
                                     JsonNode classData, PatternContext patternContext) {
        ObjectNode methodSummary = objectMapper.createObjectNode();
        
        // Extract method-level pattern context
        MethodPatternContext methodPatternContext = PatternContextExtractor.extractMethodContext(
            className, methodName, classData, patternContext
        );
        
        // Extract method information
        String returnType = methodData.has("METHODRETURNTYPE") ? 
            methodData.get("METHODRETURNTYPE").asText() : "void";
        
        List<String> parameters = new ArrayList<>();
        JsonNode paramNodes = methodData.get("METHODPARAMETER");
        if (paramNodes != null && paramNodes.isArray()) {
            for (JsonNode param : paramNodes) {
                if (param.has("PARAMETERTYPE")) {
                    parameters.add(param.get("PARAMETERTYPE").asText());
                }
            }
        }
        
        // Generate SWUM summary for method with pattern context
        String designPattern = extractDesignPattern(className);
        String swumSummary = parser.generateMethodSummaryWithContext(
            methodName, className, parameters, returnType, designPattern, 
            patternContext, methodPatternContext
        );
        
        methodSummary.put("method_name", methodName);
        methodSummary.put("return_type", returnType);
        methodSummary.put("parameters", String.join(", ", parameters));
        methodSummary.put("swum_summary", swumSummary);
        
        // Add structural information
        SWUMStructure structure = parser.parseMethodName(methodName, className);
        structure.setPatternContext(patternContext);
        structure.setMethodPatternContext(methodPatternContext);
        
        methodSummary.put("swum_actions", String.join(", ", structure.getActions()));
        methodSummary.put("swum_objects", String.join(", ", structure.getObjects()));
        
        if (structure.getParseTree() != null) {
            methodSummary.put("swum_parse_tree", structure.getParseTree().toTreeString());
        }
        
        return methodSummary;
    }
    
    /**
     * Generates overall project summary using SWUM analysis.
     * SWUM-fix: parameter renamed from rootData to swumPatternsNode — reads SWUM's own detected
     * patterns directly instead of NLG's design_pattern field from rootData.
     */
    // Old: private String generateProjectSummary(ObjectNode classSummaries, JsonNode rootData) {
    //     JsonNode designPatterns = rootData.get("design_pattern");
    private String generateProjectSummary(ObjectNode classSummaries, JsonNode swumPatternsNode) {
        StringBuilder summary = new StringBuilder();

        // SWUM-fix: read from swumPatternsNode (already the patterns array) instead of rootData.get("design_pattern")
        Set<String> detectedPatterns = new HashSet<>();
        JsonNode designPatterns = swumPatternsNode;
        if (designPatterns != null && designPatterns.isArray()) {
            for (JsonNode pattern : designPatterns) {
                pattern.fieldNames().forEachRemaining(detectedPatterns::add);
            }
        }
        
        // Count classes and methods
        int classCount = 0;
        int methodCount = 0;
        Set<String> allActions = new HashSet<>();
        Set<String> allObjects = new HashSet<>();
        
        classSummaries.fields().forEachRemaining(classEntry -> {
            JsonNode classData = classEntry.getValue();
            JsonNode methodSummaries = classData.get("method_summaries");
            
            if (methodSummaries != null) {
                methodSummaries.fields().forEachRemaining(methodEntry -> {
                    JsonNode methodData = methodEntry.getValue();
                    
                    // Extract actions and objects
                    if (methodData.has("swum_actions")) {
                        String actions = methodData.get("swum_actions").asText();
                        if (!actions.isEmpty()) {
                            allActions.addAll(Arrays.asList(actions.split(", ")));
                        }
                    }
                    
                    if (methodData.has("swum_objects")) {
                        String objects = methodData.get("swum_objects").asText();
                        if (!objects.isEmpty()) {
                            allObjects.addAll(Arrays.asList(objects.split(", ")));
                        }
                    }
                });
            }
        });
        
        // Count methods separately to avoid lambda variable issue
        final int[] methodCountArray = {0};
        classSummaries.fields().forEachRemaining(entry -> {
            JsonNode classData = entry.getValue();
            JsonNode methods = classData.get("method_summaries");
            if (methods != null) {
                methodCountArray[0] += methods.size();
            }
        });
        methodCount = methodCountArray[0];
        classCount = classSummaries.size();
        
        // Build summary
        summary.append("This software project contains ").append(classCount).append(" classes");
        if (methodCount > 0) {
            summary.append(" with ").append(methodCount).append(" methods");
        }
        summary.append(". ");
        
        if (!detectedPatterns.isEmpty()) {
            summary.append("The system implements ");
            if (detectedPatterns.size() == 1) {
                summary.append("the ").append(detectedPatterns.iterator().next()).append(" design pattern");
            } else {
                summary.append("multiple design patterns including ");
                summary.append(String.join(", ", detectedPatterns));
            }
            summary.append(". ");
        }
        
        if (!allActions.isEmpty()) {
            List<String> topActions = allActions.stream()
                .filter(action -> !action.isEmpty())
                .sorted()
                .limit(5)
                .collect(java.util.stream.Collectors.toList());
            if (!topActions.isEmpty()) {
                summary.append("Primary operations include ");
                summary.append(String.join(", ", topActions));
                summary.append(". ");
            }
        }
        
        if (!allObjects.isEmpty()) {
            List<String> topObjects = allObjects.stream()
                .filter(object -> !object.isEmpty())
                .sorted()
                .limit(5)
                .collect(java.util.stream.Collectors.toList());
            if (!topObjects.isEmpty()) {
                summary.append("Key objects manipulated are ");
                summary.append(String.join(", ", topObjects));
                summary.append(".");
            }
        }
        
        return summary.toString().trim();
    }
    
    /**
     * Extracts project name from JSON data
     */
    private String extractProjectName(JsonNode rootNode) {
        Iterator<String> fieldNames = rootNode.fieldNames();
        while (fieldNames.hasNext()) {
            String fieldName = fieldNames.next();
            if (!fieldName.equals("final_summary") && !fieldName.equals("design_pattern") && !fieldName.equals("summary_NLG")) {
                return fieldName;
            }
        }
        return "Unknown Project";
    }
    
    /**
     * Extracts design pattern information for a class
     */
    private String extractDesignPattern(String className) {
        String lowerClassName = className.toLowerCase();
        
        if (lowerClassName.contains("factory")) return "Factory";
        if (lowerClassName.contains("builder")) return "Builder";
        if (lowerClassName.contains("singleton")) return "Singleton";
        if (lowerClassName.contains("observer")) return "Observer";
        if (lowerClassName.contains("adapter")) return "Adapter";
        if (lowerClassName.contains("decorator")) return "Decorator";
        if (lowerClassName.contains("facade")) return "Facade";
        if (lowerClassName.contains("visitor")) return "Visitor";
        if (lowerClassName.contains("memento")) return "Memento";
        if (lowerClassName.contains("strategy")) return "Strategy";
        
        return null;
    }
    
    /**
     * Converts the AST-based class data from an NLG JSON into the HashMap<String, HashMap>
     * structure expected by the structural pattern detectors in CheckPattern.
     * This is the same representation the NLG pipeline builds from JavaParser — SWUM reads
     * the serialised form back from disk and reconstructs it for independent detection.
     */
    @SuppressWarnings("unchecked")
    private HashMap<String, HashMap> buildFileDetailsFromJson(JsonNode rootNode) {
        HashMap<String, HashMap> fileDetails = new HashMap<>();
        rootNode.fields().forEachRemaining(entry -> {
            String key = entry.getKey();
            JsonNode value = entry.getValue();
            // Skip NLG-generated system fields — only the project container holds class AST data
            if (key.equals("final_summary") || key.equals("design_pattern")
                    || key.equals("summary_NLG") || key.equals("project_identifier")) {
                return;
            }
            if (value.isObject()) {
                value.fields().forEachRemaining(classEntry -> {
                    String className = classEntry.getKey();
                    JsonNode classData = classEntry.getValue();
                    if (classData.isObject()) {
                        // Jackson deserialises JSON objects as LinkedHashMap (subclass of HashMap),
                        // arrays as ArrayList — both are compatible with Utils cast expectations.
                        HashMap classMap = objectMapper.convertValue(classData, HashMap.class);
                        fileDetails.put(className, classMap);
                    }
                });
            }
        });
        return fileDetails;
    }

    /**
     * Runs SWUM's own structural pattern detection using the AST features extracted from
     * the NLG JSON. This is independent of NLG's pre-computed design_pattern output.
     */
    @SuppressWarnings("unchecked")
    private ArrayList<HashMap> runStructuralPatternDetection(HashMap<String, HashMap> fileDetails) {
        ArrayList<HashMap> swumPatterns = new ArrayList<>();
        new CheckPattern().extractDesignPattern(fileDetails, swumPatterns);
        return swumPatterns;
    }

    /**
     * Processes all JSON files in the output directory
     */
    public void processAllFiles(String inputDir, String outputDir) {
        File inputDirectory = new File(inputDir);
        File outputDirectory = new File(outputDir);
        
        if (!outputDirectory.exists()) {
            outputDirectory.mkdirs();
        }
        
        File[] jsonFiles = inputDirectory.listFiles((dir, name) -> name.endsWith(".json"));
        
        if (jsonFiles == null) {
            System.err.println("No JSON files found in " + inputDir);
            return;
        }
        
        int processed = 0;
        for (File jsonFile : jsonFiles) {
            try {
                String outputFileName = jsonFile.getName().replace(".json", "_swum.json");
                String outputPath = new File(outputDirectory, outputFileName).getAbsolutePath();
                
                processJsonFile(jsonFile.getAbsolutePath(), outputPath);
                processed++;
            } catch (Exception e) {
                System.err.println("Error processing " + jsonFile.getName() + ": " + e.getMessage());
                e.printStackTrace();
            }
        }
        
        System.out.println("Processed " + processed + " JSON files with SWUM summarizer");
    }
    
    /**
     * Processes a complete project JSON file and adds SWUM summaries
     */
    public JsonNode processProjectJson(JsonNode originalJson) {
        ObjectNode result = objectMapper.createObjectNode();
        
        // Copy original data
        result.setAll((ObjectNode) originalJson);
        
        // Add SWUM processing timestamp
        result.put("swum_processing_timestamp", System.currentTimeMillis());
        result.put("swum_version", "1.0");
        
        // Process project-level summary
        String projectName = originalJson.has("project_name") ? 
            originalJson.get("project_name").asText() : "UnknownProject";
        
        String projectSummary = generateProjectSummary(originalJson);
        result.put("swum_project_summary", projectSummary);
        
        // Process individual files if available
        if (originalJson.has("files")) {
            ObjectNode swumFiles = objectMapper.createObjectNode();
            JsonNode files = originalJson.get("files");
            
            files.fields().forEachRemaining(entry -> {
                String fileName = entry.getKey();
                JsonNode fileData = entry.getValue();
                
                ObjectNode swumFileData = objectMapper.createObjectNode();
                swumFileData.setAll((ObjectNode) fileData);
                
                // Generate SWUM summary for the file
                String swumSummary = generateFileSummary(fileName, fileData);
                swumFileData.put("swum_summary", swumSummary);
                
                // Process methods if available
                if (fileData.has("methods")) {
                    ObjectNode swumMethods = objectMapper.createObjectNode();
                    JsonNode methods = fileData.get("methods");
                    
                    methods.fields().forEachRemaining(methodEntry -> {
                        String methodName = methodEntry.getKey();
                        JsonNode methodData = methodEntry.getValue();
                        
                        ObjectNode swumMethodData = objectMapper.createObjectNode();
                        swumMethodData.setAll((ObjectNode) methodData);
                        
                        String methodSummary = generateMethodSummary(methodName);
                        swumMethodData.put("swum_summary", methodSummary);
                        
                        swumMethods.set(methodName, swumMethodData);
                    });
                    
                    swumFileData.set("swum_methods", swumMethods);
                }
                
                swumFiles.set(fileName, swumFileData);
            });
            
            result.set("swum_files", swumFiles);
        }
        
        // Add SWUM statistics
        ObjectNode swumStats = objectMapper.createObjectNode();
        swumStats.put("total_files_processed", result.has("files") ? result.get("files").size() : 0);
        swumStats.put("grammar_rules_applied", parser.getAppliedRulesCount());
        swumStats.put("vocabulary_coverage", calculateVocabularyCoverage(originalJson));
        
        result.set("swum_statistics", swumStats);
        
        return result;
    }
    
    /**
     * Generates a SWUM summary for an entire project
     */
    private String generateProjectSummary(JsonNode projectJson) {
        String projectName = projectJson.has("project_name") ? 
            projectJson.get("project_name").asText() : "Project";
        
        // Extract patterns if available
        StringBuilder summary = new StringBuilder();
        
        if (projectJson.has("detected_patterns")) {
            JsonNode patterns = projectJson.get("detected_patterns");
            if (patterns.isArray() && patterns.size() > 0) {
                summary.append("This ").append(cleanIdentifier(projectName)).append(" implements ");
                
                List<String> patternNames = new ArrayList<>();
                for (JsonNode pattern : patterns) {
                    if (pattern.has("pattern")) {
                        patternNames.add(pattern.get("pattern").asText());
                    }
                }
                
                if (!patternNames.isEmpty()) {
                    summary.append(String.join(", ", patternNames)).append(" design pattern");
                    if (patternNames.size() > 1) summary.append("s");
                }
            }
        }
        
        // Add file count information
        if (projectJson.has("files")) {
            int fileCount = projectJson.get("files").size();
            if (summary.length() > 0) {
                summary.append(" across ");
            } else {
                summary.append("This project contains ");
            }
            summary.append(fileCount).append(" source file");
            if (fileCount > 1) summary.append("s");
        }
        
        // Add final summary if available
        if (projectJson.has("final_summary")) {
            String finalSummary = projectJson.get("final_summary").asText();
            if (!finalSummary.trim().isEmpty()) {
                if (summary.length() > 0) {
                    summary.append(". ");
                }
                // Use SWUM to process the final summary
                summary.append(generateSummaryFromText(finalSummary));
            }
        }
        
        if (summary.length() == 0) {
            summary.append("This ").append(cleanIdentifier(projectName)).append(" represents a software implementation");
        }
        
        return summary.toString();
    }
    
    /**
     * Generates a SWUM summary for a single file
     */
    private String generateFileSummary(String fileName, JsonNode fileData) {
        String cleanFileName = cleanIdentifier(fileName.replaceAll("\\.(java|py|cpp|cs)$", ""));
        
        StringBuilder summary = new StringBuilder();
        summary.append("The ").append(cleanFileName).append(" component");
        
        // Check for class information
        if (fileData.has("class_name")) {
            String className = fileData.get("class_name").asText();
            String classSummary = generateClassSummary(className);
            summary.append(" ").append(classSummary.toLowerCase());
        }
        
        // Add method count
        if (fileData.has("methods")) {
            int methodCount = fileData.get("methods").size();
            summary.append(" and contains ").append(methodCount).append(" method");
            if (methodCount > 1) summary.append("s");
        }
        
        // Add pattern information if available
        if (fileData.has("pattern")) {
            String pattern = fileData.get("pattern").asText();
            summary.append(" implementing ").append(pattern).append(" pattern");
        }
        
        return summary.toString();
    }
    
    /**
     * Generates SWUM summary from existing text
     */
    private String generateSummaryFromText(String text) {
        // Extract key action words and objects from the text
        String[] words = text.toLowerCase().split("\\s+");
        List<String> actionWords = new ArrayList<>();
        List<String> objectWords = new ArrayList<>();
        
        for (String word : words) {
            String cleanWord = word.replaceAll("[^a-zA-Z]", "");
            if (parser.getActionVerbs().contains(cleanWord)) {
                actionWords.add(cleanWord);
            } else if (parser.getObjectNouns().contains(cleanWord)) {
                objectWords.add(cleanWord);
            }
        }
        
        // Construct SWUM-style summary
        StringBuilder swumSummary = new StringBuilder();
        if (!actionWords.isEmpty() && !objectWords.isEmpty()) {
            swumSummary.append("This system ");
            swumSummary.append(actionWords.get(0));
            swumSummary.append("s ");
            swumSummary.append(objectWords.get(0));
            if (objectWords.size() > 1) {
                swumSummary.append(" and ").append(objectWords.get(1));
            }
        } else {
            // Fallback to simplified version
            swumSummary.append(text);
        }
        
        return swumSummary.toString();
    }
    
    /**
     * Calculates vocabulary coverage for SWUM processing
     */
    private double calculateVocabularyCoverage(JsonNode projectJson) {
        Set<String> allWords = new HashSet<>();
        Set<String> recognizedWords = new HashSet<>();
        
        // Extract words from project
        extractWordsFromJson(projectJson, allWords);
        
        // Check recognition
        for (String word : allWords) {
            String cleanWord = word.toLowerCase().replaceAll("[^a-zA-Z]", "");
            if (parser.getActionVerbs().contains(cleanWord) || 
                parser.getObjectNouns().contains(cleanWord) ||
                parser.getPatternTerms().contains(cleanWord)) {
                recognizedWords.add(cleanWord);
            }
        }
        
        return allWords.isEmpty() ? 0.0 : (double) recognizedWords.size() / allWords.size();
    }
    
    /**
     * Recursively extracts words from JSON structure
     */
    private void extractWordsFromJson(JsonNode node, Set<String> words) {
        if (node.isTextual()) {
            String text = node.asText();
            String[] textWords = text.split("\\W+");
            for (String word : textWords) {
                if (!word.trim().isEmpty()) {
                    words.add(word.trim());
                }
            }
        } else if (node.isObject()) {
            node.fields().forEachRemaining(entry -> {
                words.add(entry.getKey());
                extractWordsFromJson(entry.getValue(), words);
            });
        } else if (node.isArray()) {
            for (JsonNode arrayElement : node) {
                extractWordsFromJson(arrayElement, words);
            }
        }
    }
    
    /**
     * Cleans identifier names by converting camelCase to readable text
     */
    private String cleanIdentifier(String identifier) {
        if (identifier == null || identifier.trim().isEmpty()) {
            return identifier;
        }
        
        // Split camelCase and convert to lowercase with spaces
        String result = identifier.replaceAll("([a-z])([A-Z])", "$1 $2").toLowerCase();
        
        // Clean up extra spaces
        return result.replaceAll("\\s+", " ").trim();
    }
    
    /**
     * Generates a simple method summary from method name only
     */
    public String generateMethodSummary(String methodName) {
        if (methodName == null || methodName.trim().isEmpty()) {
            return "Unknown method";
        }
        
        // Use SWUM grammar parser to create summary
        SWUMStructure structure = parser.parseMethodName(methodName, "UnknownClass");
        
        StringBuilder summary = new StringBuilder();
        if (!structure.getActions().isEmpty()) {
            summary.append("This method ");
            summary.append(structure.getActions().get(0));
            if (!structure.getObjects().isEmpty()) {
                summary.append(" ");
                summary.append(structure.getObjects().get(0));
            }
        } else {
            summary.append("The ").append(cleanIdentifier(methodName)).append(" method");
        }
        
        return summary.toString();
    }
    
    /**
     * Generates a simple class summary from class name only
     */
    public String generateClassSummary(String className) {
        if (className == null || className.trim().isEmpty()) {
            return "Unknown class";
        }
        
        String cleanName = cleanIdentifier(className);
        
        // Check for common patterns in class name
        String lowerName = className.toLowerCase();
        if (lowerName.contains("factory")) {
            return "Creates and manages " + cleanName.replace("factory", "").trim() + " instances";
        } else if (lowerName.contains("builder")) {
            return "Constructs " + cleanName.replace("builder", "").trim() + " objects step by step";
        } else if (lowerName.contains("adapter")) {
            return "Adapts " + cleanName.replace("adapter", "").trim() + " interface";
        } else if (lowerName.contains("decorator")) {
            return "Decorates " + cleanName.replace("decorator", "").trim() + " with additional functionality";
        } else if (lowerName.contains("facade")) {
            return "Provides simplified interface to " + cleanName.replace("facade", "").trim() + " subsystem";
        } else if (lowerName.contains("observer")) {
            return "Observes changes in " + cleanName.replace("observer", "").trim() + " state";
        } else if (lowerName.contains("visitor")) {
            return "Visits " + cleanName.replace("visitor", "").trim() + " elements";
        } else if (lowerName.contains("singleton")) {
            return "Ensures single instance of " + cleanName.replace("singleton", "").trim();
        } else if (lowerName.contains("memento")) {
            return "Captures " + cleanName.replace("memento", "").trim() + " state";
        } else {
            return "The " + cleanName + " class";
        }
    }
}


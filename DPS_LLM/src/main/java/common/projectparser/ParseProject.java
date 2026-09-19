package common.projectparser;

import com.github.javaparser.StaticJavaParser;
import com.github.javaparser.ParserConfiguration.LanguageLevel;
import com.github.javaparser.ast.CompilationUnit;
import com.github.javaparser.ast.body.MethodDeclaration;
import com.github.javaparser.ast.expr.MethodCallExpr;
import com.github.javaparser.ast.visitor.VoidVisitorAdapter;
import com.github.javaparser.resolution.declarations.ResolvedMethodDeclaration;
import com.github.javaparser.symbolsolver.JavaSymbolSolver;
import com.github.javaparser.symbolsolver.javaparsermodel.declarations.JavaParserMethodDeclaration;

import common.designpatternidentifier.CheckPattern;

import common.utils.*;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.*;
import java.util.regex.Pattern;

public class ParseProject {

    // reference: java callgraph
    // 需要跳过的pattern列表
    private final List<Pattern> skipPatterns = new ArrayList<>();
    
    // Track processed files to skip duplicates (filename + content hash)
    // Bug 1 fix: was static, so the map persisted across all projects in a batch run. Two different
    // projects with a file of the same name and identical content caused the second project's file
    // to be silently skipped, producing an incomplete JSON for that project. Moved to a local
    // variable inside parseProject() so deduplication is scoped per-project only.
    // private static final HashMap<String, String> processedFiles = new HashMap<>();
    private static int skippedDuplicates = 0;
    
    /**
     * Compute MD5 hash of file content to detect duplicates
     */
    private String computeFileHash(File file) throws IOException {
        try {
            MessageDigest md = MessageDigest.getInstance("MD5");
            try (FileInputStream fis = new FileInputStream(file)) {
                byte[] buffer = new byte[8192];
                int bytesRead;
                while ((bytesRead = fis.read(buffer)) != -1) {
                    md.update(buffer, 0, bytesRead);
                }
            }
            byte[] hashBytes = md.digest();
            StringBuilder sb = new StringBuilder();
            for (byte b : hashBytes) {
                sb.append(String.format("%02x", b));
            }
            return sb.toString();
        } catch (NoSuchAlgorithmException e) {
            throw new IOException("MD5 algorithm not available", e);
        }
    }
    
    /**
     * Get count of skipped duplicate files
     */
    public static int getSkippedDuplicatesCount() {
        return skippedDuplicates;
    }
    
    /**
     * Reset duplicate tracking (call at start of batch processing)
     */
    public static void resetDuplicateTracking() {
        // Bug 1 fix: processedFiles is now local to parseProject(); there is no class-level map to clear.
        // processedFiles.clear();
        skippedDuplicates = 0;
    }

    public HashMap<String, Object> parseProject(File directory) throws FileNotFoundException, IOException {
        return parseProject(directory, directory.getName());
    }

    // Najam: the boolean generateNlgSummary parameter is gone. ParseProject is the shared feature
    // extractor for DPS_NLG, DPS_SWUM and DPS_LLM, and it used to run the NLG summariser inline for
    // one of those three. That made NLG's summaries a side effect of parsing, so NLG never read the
    // JSON representation it emitted, and it tied the common parser to the dps_nlg package. Each
    // pipeline now writes the JSON and summarises from it.
    // public HashMap<String, Object> parseProject(File directory, boolean generateNlgSummary) ...
    // public HashMap<String, Object> parseProject(File directory, String projectIdentifier, boolean generateNlgSummary) ...
    public HashMap<String, Object> parseProject(File directory, String projectIdentifier) throws FileNotFoundException, IOException {

        // Bug 1 fix: processedFiles is now a local variable so each call to parseProject() gets its
        // own fresh deduplication context. Previously the static map accumulated entries across all
        // projects in a batch, causing a file from ProjectB to be silently skipped if ProjectA had
        // already processed a file with the same name and identical content.
        HashMap<String, String> processedFiles = new HashMap<>();

        ArrayList<File> fileArrayList = new ArrayList<>();

        // referenced from Java Callgraph
        ArrayList<String> srcPathList = new ArrayList<>();
        ArrayList<String> libPathList = new ArrayList<>();

        // names of all files in directory added to fileArrayList (list not tree)
        // srcPathList and libPathList consist of abs paths of src and lib folders
        fetchFiles(directory, fileArrayList, srcPathList, libPathList);

        // Configure symbol solver for better type resolution
        System.out.println("Configuring symbol resolver with " + srcPathList.size() + " source paths and " + libPathList.size() + " library paths");
        // referenced from Java Callgraph
        JavaSymbolSolver symbolSolver = SymbolSolverFactory.getJavaSymbolSolver(srcPathList, libPathList);
        StaticJavaParser.getParserConfiguration().setSymbolResolver(symbolSolver);
        StaticJavaParser.getParserConfiguration().setLanguageLevel(LanguageLevel.BLEEDING_EDGE);

        // referenced from Java callgraph
        // 获取src目录中的全部java文件，并进行解析
        HashMap<String, ArrayList<String>> callerCallees = new HashMap<>();

        HashMap<String, HashMap> parsedFile = new HashMap<>();
        CheckPattern checkPattern = new CheckPattern();

        ArrayList designPatternArrayList = new ArrayList<>();

        // Empty placeholders keeping the JSON schema stable across all three pipelines; DPS_NLG
        // overwrites them once it has summarised the JSON, DPS_SWUM and DPS_LLM leave them empty.
        HashMap<String, HashMap<String, HashSet<String>>> summaryMap = new HashMap<String, HashMap<String, HashSet<String>>>();
        String finalSummary = "";

        // go through all files under the project
        for (File file : fileArrayList) {
            // Check for duplicates (same filename + same content)
            String fileName = file.getName();
            String fileHash = computeFileHash(file);
            String fileKey = fileName + "|" + fileHash;

            if (processedFiles.containsKey(fileKey)) {
                skippedDuplicates++;
                System.out.println("\tSkipping duplicate: " + fileName + " (already processed from " + processedFiles.get(fileKey) + ")");
                continue;
            }

            // Mark this file as processed
            processedFiles.put(fileKey, directory.getName());

            HashMap<String, ArrayList> fileDetails = new HashMap<>();
            CompilationUnit compilationUnit = null;
            try {
                compilationUnit = parseFileToCompilationUnit(file);
            } catch (Exception e) {
                System.out.println("WARNING: Exception during parsing file: " + file.getName() + " - " + e.getClass().getSimpleName() + ": " + e.getMessage());
            } catch (Error e) {
                System.out.println("ERROR: Error during parsing file: " + file.getName() + " - " + e.getClass().getSimpleName() + ": " + e.getMessage());
            }

            if (compilationUnit != null) {
                try {
                    // File parsed successfully - extract detailed information
                    MethodsExtr methodsExtr = new MethodsExtr();
                    FieldExtr fieldExtr = new FieldExtr();
                    ConstructorExtr constructorExtr = new ConstructorExtr();
                    VariableExtr variableExtr = new VariableExtr();
                    ClassOrInterfaceExtr classOrInterfaceExtr = new ClassOrInterfaceExtr();

                    fileDetails.put("FIELDDETAIL", fieldExtr.getFieldInfo(compilationUnit));
                    fileDetails.put("CONSTRUCTORDETAIL", constructorExtr.getConstructorInfo(compilationUnit));
                    fileDetails.put("VARIABLEDETAIL", variableExtr.getVariableInfo(compilationUnit));
                    fileDetails.put("METHODDETAIL", methodsExtr.getMethodInfo(compilationUnit));
                    fileDetails.put("CLASSORINTERFACEDETAIL", classOrInterfaceExtr.getClassInterfaceInfo(compilationUnit));
                    extract(compilationUnit, callerCallees, skipPatterns);
                } catch (Exception e) {
                    System.out.println("WARNING: Exception during symbol resolution or extraction for file: " + file.getName() + " - " + e.getClass().getSimpleName() + ": " + e.getMessage());
                    // If extraction fails, still include the file with empty details
                    fileDetails.put("FIELDDETAIL", new ArrayList<>());
                    fileDetails.put("CONSTRUCTORDETAIL", new ArrayList<>());
                    fileDetails.put("VARIABLEDETAIL", new ArrayList<>());
                    fileDetails.put("METHODDETAIL", new ArrayList<>());
                    fileDetails.put("CLASSORINTERFACEDETAIL", new ArrayList<>());
                } catch (Error e) {
                    System.out.println("ERROR: Error during symbol resolution or extraction for file: " + file.getName() + " - " + e.getClass().getSimpleName() + ": " + e.getMessage());
                    fileDetails.put("FIELDDETAIL", new ArrayList<>());
                    fileDetails.put("CONSTRUCTORDETAIL", new ArrayList<>());
                    fileDetails.put("VARIABLEDETAIL", new ArrayList<>());
                    fileDetails.put("METHODDETAIL", new ArrayList<>());
                    fileDetails.put("CLASSORINTERFACEDETAIL", new ArrayList<>());
                }
            } else {
                // File couldn't be parsed - create empty details but still include in summary
                fileDetails.put("FIELDDETAIL", new ArrayList<>());
                fileDetails.put("CONSTRUCTORDETAIL", new ArrayList<>());
                fileDetails.put("VARIABLEDETAIL", new ArrayList<>());
                fileDetails.put("METHODDETAIL", new ArrayList<>());
                fileDetails.put("CLASSORINTERFACEDETAIL", new ArrayList<>());
                // Note: Can't extract call graph info for unparseable files
            }

            // Always add file to parsedFile map for summary generation
            parsedFile.put(Utils.getBaseName(file.getName()), fileDetails);
        }

        // merge the features with the callgraph
        HashMap<String, Object> parsedProject = new HashMap<>();
        // Bug 4 fix: extractCallgraphResults mutates parsedFile in-place (it adds OUTGOINGMETHOD,
        // INCOMINGMETHOD, and NUMBEROFINCOMINGMETHODS directly into the nested maps) and then returns
        // the same reference, not a new object. The old name "extractedCallGraph" implied a separate
        // derived structure and made the mutation invisible at the call site.
        // HashMap extractedCallGraph = extractCallgraphResults(parsedFile, callerCallees);
        HashMap enrichedParsedFile = extractCallgraphResults(parsedFile, callerCallees);

        // Only return empty if no files were processed at all
        if (parsedFile.isEmpty())
            return new HashMap<>();

        // Extract design patterns only if call graph information is available
        // Bug 4 fix: updated to use renamed variable.
        // if (!extractedCallGraph.isEmpty()) {
        //     checkPattern.extractDesignPattern(extractedCallGraph, designPatternArrayList);
        // }
        if (!enrichedParsedFile.isEmpty()) {
            checkPattern.extractDesignPattern(enrichedParsedFile, designPatternArrayList);
        }

        // Decide which data to summarise/store: prefer call graph enriched data if available
        // Bug 4 fix: enrichedParsedFile is the same reference as parsedFile (see extractCallgraphResults),
        // so isEmpty() here is always false given the guard above; updated to use renamed variable.
        // HashMap dataToStore = extractedCallGraph.isEmpty() ? parsedFile : extractedCallGraph;
        HashMap dataToStore = enrichedParsedFile.isEmpty() ? parsedFile : enrichedParsedFile;

        // Summarisation no longer happens here. DPS_NLG reads the JSON this method produces and runs
        // Summarise over it (see dps_nlg.NlgJsonSummariser), which is how DPS_SWUM and DPS_LLM already
        // worked. summary_NLG and final_summary are still written so the JSON schema is unchanged for
        // every consumer; DPS_NLG fills them in on its second pass.
        // Old inline call:
        // if (generateNlgSummary && summarise != null) {
        //     finalSummary = summarise.summarise(dataToStore, designPatternArrayList, summaries, projectIdentifier);
        // }

        // Bug 2 fix: was directory.getName() (leaf folder name only), which is ambiguous when two
        // different projects share the same folder name but differ in their parent path. Using
        // projectIdentifier (the full relative path passed by the caller) makes the key unique
        // across the dataset. SWUM is not affected because it iterates all non-system keys by name.
        // parsedProject.put(directory.getName(), dataToStore);
        parsedProject.put(projectIdentifier, dataToStore);
        // Bug 3 fix: store the project identifier as an explicit named field so any consumer reading
        // the JSON can determine provenance without relying solely on the filename or key iteration.
        parsedProject.put("project_identifier", projectIdentifier);
        parsedProject.put("design_pattern", designPatternArrayList);
        parsedProject.put("summary_NLG", summaryMap);
        parsedProject.put("final_summary", finalSummary);

        // return the result, which contains all files of the project, stored in the
        // hashmap, the key is file name, the value is the details.
        return parsedProject;
    }

    /**
     * Helper method to parse file to CompilationUnit with proper exception handling
     * @param file The file to parse
     * @return CompilationUnit or null if parsing fails
     */
    private CompilationUnit parseFileToCompilationUnit(File file) {
        try {
            // Bug 5 fix: removed hardcoded debug logging targeting specific filenames. Any file that
            // needs targeted debugging should use a general-purpose logger or a debugger breakpoint
            // rather than file-name conditions baked into production code.
            // Special logging for the problematic record files
            // if (file.getName().equals("ElfWeapon.java") || file.getName().equals("OrcWeapon.java")) {
            //     System.out.println("DEBUG: Attempting to parse record file: " + file.getAbsolutePath());
            // }
            CompilationUnit cu = StaticJavaParser.parse(file);
            // if (file.getName().equals("ElfWeapon.java") || file.getName().equals("OrcWeapon.java")) {
            //     System.out.println("DEBUG: Successfully parsed " + file.getName() + " - Types found: " + cu.getTypes().size());
            // }
            return cu;
        } catch (Exception e) {
            System.out.println("WARNING: Skipping file due to parse exception: " + file.getName() + " - " + e.getClass().getSimpleName() + ": " + e.getMessage());
            // Bug 5 fix: removed hardcoded stack-trace print for specific filenames.
            // if (file.getName().equals("ElfWeapon.java") || file.getName().equals("OrcWeapon.java")) {
            //     e.printStackTrace();
            // }
            return null;
        } catch (Error e) {
            System.out.println("ERROR: Skipping file due to parse error: " + file.getName() + " - " + e.getClass().getSimpleName() + ": " + e.getMessage());
            // Bug 5 fix: removed hardcoded stack-trace print for specific filenames.
            // if (file.getName().equals("ElfWeapon.java") || file.getName().equals("OrcWeapon.java")) {
            //     e.printStackTrace();
            // }
            return null;
        }
    }

    private HashMap<String, HashMap> extractCallgraphResults(HashMap<String, HashMap> parsedFile,
            HashMap<String, ArrayList<String>> callerCallees) {
        Set<String> classNames = parsedFile.keySet();

        for (HashMap.Entry mapElement : callerCallees.entrySet()) {
            String caller = (String) mapElement.getKey();
            ArrayList<String> callees = callerCallees.get(caller);

            String callerClass = extractCallgraphClass(caller);
            String callerMethodName = extractCallgraphMethodName(caller);

            if (classNames.contains(callerClass)) {
                HashMap<String, ArrayList> parsedCallerClass = parsedFile.get(callerClass);
                ArrayList<HashMap> parsedCalledMethods = Utils.getMethodDetails(parsedCallerClass);
                for (HashMap parsedCalledMethod : parsedCalledMethods) {

                    // need parameter comparison also
                    if (Utils.getMethodName(parsedCalledMethod).equals(callerMethodName)) {
                        for (String callee : callees) {

                            String calleeClass = extractCallgraphClass(callee);
                            String calleeMethodName = extractCallgraphMethodName(callee);

                            HashMap<String, String> newOutgoing = new HashMap<>();
                            newOutgoing.put("CALLEECLASS", calleeClass);
                            newOutgoing.put("CALLEEMETHODNAME", calleeMethodName);

                            Utils.getOutgoingMethod(parsedCalledMethod).add(newOutgoing);

                            // add incoming method for the method in caller class
                            HashMap<String, ArrayList> parsedCalleeClass = parsedFile.get(calleeClass);
                            if (parsedCalleeClass == null) {
                                continue;
                            }
                            ArrayList<HashMap> parsedCallingMethods = Utils.getMethodDetails(parsedCalleeClass);

                            for (HashMap parsedCallingMethod : parsedCallingMethods) {

                                if (Utils.getMethodName(parsedCallingMethod).equals(calleeMethodName)) {

                                    HashMap<String, String> newIncoming = new HashMap<>();
                                    newIncoming.put("CALLEDCLASS", callerClass);
                                    newIncoming.put("CALLEDMETHODNAME", callerMethodName);

                                    Utils.getIncomingMethod(parsedCallingMethod).add(newIncoming);

                                    // Update number of incoming calls
                                    parsedCallingMethod.put("NUMBEROFINCOMINGMETHODS",
                                            Utils.getIncomingMethod(parsedCallingMethod).size());
                                }
                            }
                        }
                    }
                }
            }
        }
        return parsedFile;
    }

    private String extractCallgraphClass(String caller) {
        String filteredCaller = caller.replaceAll("\\(.*\\)", "");
        return Utils.splitByDot(filteredCaller, 2);
    }

    private String extractCallgraphMethodName(String caller) {
        String filteredCaller = caller.replaceAll("\\(.*\\)", "");
        return Utils.splitByDot(filteredCaller, 1);
    }

    private void fetchFiles(File dir, ArrayList<File> fileList, ArrayList<String> srcPathList,
            ArrayList<String> libPathList) {
        if (dir.getName().equals("src")) {
            srcPathList.add(dir.getAbsolutePath());
        }

        if (dir.getName().equals("lib")) {
            libPathList.add(dir.getAbsolutePath());
        }

        if (dir.isDirectory()) {
            // Najam, 2026-06-04: listFiles() returns null on I/O error or permission failure; calling for-each
            // directly on the null result throws NPE. Captured to a local and guarded before iterating.
            // for (File file1 : dir.listFiles()) {
            //     fetchFiles(file1, fileList, srcPathList, libPathList);
            // }
            File[] children = dir.listFiles();
            if (children != null) {
                for (File file1 : children) {
                    fetchFiles(file1, fileList, srcPathList, libPathList);
                }
            }
        } else if (Utils.getExtension(dir).equals("java")) {
            fileList.add(dir);
            
            // Add the parent directory as a source path for symbol resolution
            // This helps resolve symbols in flat project structures without "src" directories
            String parentPath = dir.getParent();
            if (parentPath != null && !srcPathList.contains(parentPath)) {
                srcPathList.add(parentPath);
            }
        }

    }

    // referenced from Java Callgraph
    private void extract(CompilationUnit compilationUnit, HashMap<String, ArrayList<String>> callerCallees,
            List<Pattern> skipPatterns) {

        // 获取到方法声明，并进行遍历
        List<MethodDeclaration> all = compilationUnit.findAll(MethodDeclaration.class);
        for (MethodDeclaration methodDeclaration : all) {
            ArrayList<String> curCallees = new ArrayList<>();

            // 对每个方法声明内容进行遍历，查找内部调用的其他方法
            methodDeclaration.accept(new MethodCallVisitor(skipPatterns), curCallees);
            String caller = getQualifiedSignature(methodDeclaration);
            assert caller != null;

            // // 如果map中还没有key，则添加key
            if (!callerCallees.containsKey(caller) && !Utils.shouldSkip(caller, skipPatterns)) {
                callerCallees.put(caller, new ArrayList<>());
            }

            if (!Utils.shouldSkip(caller, skipPatterns)) {
                callerCallees.get(caller).addAll(curCallees);
            }

        }
    }

    // 遍历源码文件时，只关注方法调用的Visitor， 然后提取存放到第二个参数collector中
    private static class MethodCallVisitor extends VoidVisitorAdapter<List<String>> {

        private List<Pattern> skipPatterns = new ArrayList<>();

        public MethodCallVisitor(List<Pattern> skipPatterns) {
            if (skipPatterns != null) {
                this.skipPatterns = skipPatterns;

            }
        }

        @Override
        public void visit(MethodCallExpr n, List<String> collector) {
            // 提取方法调用
            String signature = ParseProject.getResolvedMethodSignature(n);
            if (signature != null && !Utils.shouldSkip(signature, skipPatterns)) {
                ResolvedMethodDeclaration resolvedMethodDeclaration;
                try {
                    resolvedMethodDeclaration = n.resolve();
                    if (resolvedMethodDeclaration instanceof JavaParserMethodDeclaration) {
                        collector.add(signature);
                    }
                } catch (Exception e) {
                    // Continue execution - just log the issue
                }
            }
            // Don't forget to call super, it may find more method calls inside the
            // arguments of this method call, for example.
            super.visit(n, collector);
        }
    }

    /**
     * Helper method to get qualified signature with fallback handling
     * @param methodDeclaration The method declaration
     * @return qualified signature or simple signature as fallback
     */
    private String getQualifiedSignature(MethodDeclaration methodDeclaration) {
        try {
            return methodDeclaration.resolve().getQualifiedSignature();
        } catch (Exception e) {
            String fallback = methodDeclaration.getSignature().asString();
            System.out.println("Use " + fallback + " instead of qualified signature, cause: " + e.getMessage());
            return fallback;
        }
    }

    /**
     * Helper method to get resolved method signature with error handling
     * @param methodCall The method call expression
     * @return qualified signature or null if resolution fails
     */
    private static String getResolvedMethodSignature(MethodCallExpr methodCall) {
        try {
            return methodCall.resolve().getQualifiedSignature();
        } catch (Exception e) {
            System.out.print("Line ");
            // Najam, 2026-06-04: getRange() returns Optional<Range>; calling .get() without isPresent() throws
            // NoSuchElementException when range is absent, turning a logged warning into an unhandled crash.
            // System.out.print(methodCall.getRange().get().begin.line);
            System.out.print(methodCall.getRange().map(r -> r.begin.line).orElse(-1));
            System.out.print(", ");
            System.out.print(
                    methodCall.getNameAsString() + methodCall.getArguments()
                            .toString().replace("[", "(").replace("]", ")"));
            System.out.print(" cannot resolve some symbol, because ");
            System.out.println(e.getMessage());
            return null;
        }
    }

}

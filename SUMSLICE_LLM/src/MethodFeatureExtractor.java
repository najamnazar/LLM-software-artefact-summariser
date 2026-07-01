import org.eclipse.jdt.core.dom.AST;
import org.eclipse.jdt.core.dom.ASTParser;
import org.eclipse.jdt.core.dom.ASTVisitor;
import org.eclipse.jdt.core.dom.ClassInstanceCreation;
import org.eclipse.jdt.core.dom.CompilationUnit;
import org.eclipse.jdt.core.dom.ConstructorInvocation;
import org.eclipse.jdt.core.dom.Expression;
import org.eclipse.jdt.core.dom.MethodDeclaration;
import org.eclipse.jdt.core.dom.MethodInvocation;
import org.eclipse.jdt.core.dom.QualifiedName;
import org.eclipse.jdt.core.dom.SimpleName;
import org.eclipse.jdt.core.dom.SingleVariableDeclaration;
import org.eclipse.jdt.core.dom.SuperMethodInvocation;

import java.io.BufferedWriter;
import java.io.IOException;
import java.io.Reader;
import java.nio.charset.StandardCharsets;
import java.nio.file.FileVisitResult;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.SimpleFileVisitor;
import java.nio.file.attribute.BasicFileAttributes;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashSet;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;

public class MethodFeatureExtractor {

    public static void main(String[] args) throws Exception {
        Path corpusDir = args.length > 0 ? Paths.get(args[0]) : Paths.get("input", "corpus", "nanoxml");
        Path outputDir = args.length > 1 ? Paths.get(args[1]) : Paths.get("output");
        String projectName = corpusDir.toAbsolutePath().normalize().getFileName().toString();
        Path outputJsonPath = outputDir.resolve(projectName + "-methods.json");
        Path swumOutPath = outputDir.resolve(projectName + ".out");
        Path callGraphPath = outputDir.resolve(projectName + "-graph.txt");

        if (!Files.isDirectory(corpusDir)) {
            throw new IllegalArgumentException("Corpus directory not found: " + corpusDir.toAbsolutePath());
        }
        Files.createDirectories(outputDir);

        List<Path> javaFiles = findJavaFiles(corpusDir);
        if (javaFiles.isEmpty()) {
            throw new IllegalStateException("No Java source files found under: " + corpusDir.toAbsolutePath());
        }

        List<ParsedFile> parsedFiles = parseCompilationUnits(javaFiles);
        AnalysisResult analysis = analyze(parsedFiles);
        List<MethodInfo> methods = analysis.methods;

        writeSwumOut(swumOutPath, methods);
        writeCallGraph(callGraphPath, methods);

        inferUseTypes(methods);

        int totalCalls = methods.stream().mapToInt(m -> m.calls.size()).sum();
        int totalCalled = methods.stream().mapToInt(m -> m.calledBy.size()).sum();
        int avgCalls = methods.isEmpty() ? 0 : totalCalls / methods.size();
        int avgCalled = methods.isEmpty() ? 0 : totalCalled / methods.size();

        JsonWriter jsonWriter = new JsonWriter();
        try (BufferedWriter writer = Files.newBufferedWriter(outputJsonPath, StandardCharsets.UTF_8)) {
            jsonWriter.writeMethodList(writer, methods, avgCalled, avgCalls);
        }

        System.out.println("Generated methods: " + methods.size());
        System.out.println("SWUM file: " + swumOutPath.toAbsolutePath());
        System.out.println("Callgraph file: " + callGraphPath.toAbsolutePath());
        System.out.println("JSON file: " + outputJsonPath.toAbsolutePath());
    }

    private static List<Path> findJavaFiles(Path root) throws IOException {
        List<Path> javaFiles = new ArrayList<>();
        Files.walkFileTree(root, new SimpleFileVisitor<>() {
            @Override
            public FileVisitResult visitFile(Path file, BasicFileAttributes attrs) {
                if (file.getFileName().toString().endsWith(".java")) {
                    javaFiles.add(file);
                }
                return FileVisitResult.CONTINUE;
            }
        });
        javaFiles.sort(Comparator.comparing(Path::toString));
        return javaFiles;
    }

    private static List<ParsedFile> parseCompilationUnits(List<Path> javaFiles) throws IOException {
        List<ParsedFile> parsedFiles = new ArrayList<>();

        for (Path javaFile : javaFiles) {
            String source = readFully(javaFile);
            String className = classNameFromFile(javaFile);

            ASTParser parser = ASTParser.newParser(AST.JLS3);
            parser.setKind(ASTParser.K_COMPILATION_UNIT);
            parser.setResolveBindings(false);
            parser.setSource(source.toCharArray());

            CompilationUnit cu = (CompilationUnit) parser.createAST(null);
            parsedFiles.add(new ParsedFile(javaFile, className, source, cu));
        }

        return parsedFiles;
    }

    private static AnalysisResult analyze(List<ParsedFile> files) {
        List<MethodInfo> methods = new ArrayList<>();
        Map<String, List<MethodInfo>> byClassAndName = new HashMap<>();
        Map<String, List<MethodInfo>> byNameAndArity = new HashMap<>();
        Map<ParsedFile, Map<Integer, MethodInfo>> methodByStart = new HashMap<>();

        final int[] nextId = {0};
        for (ParsedFile file : files) {
            Map<Integer, MethodInfo> startMap = new HashMap<>();
            file.compilationUnit.accept(new ASTVisitor() {
                @Override
                public void endVisit(MethodDeclaration node) {
                    MethodInfo info = new MethodInfo();
                    info.id = nextId[0];
                    nextId[0]++;
                    info.className = file.className;
                    info.methodName = node.getName().getIdentifier();
                    info.constructor = node.isConstructor();
                    info.returnType = node.isConstructor() ? file.className : (node.getReturnType2() == null ? "void" : node.getReturnType2().toString());
                    info.methodBody = node.getBody() == null ? "" : node.getBody().toString();

                    List<String> argTypes = new ArrayList<>();
                    for (Object parameter : node.parameters()) {
                        SingleVariableDeclaration arg = (SingleVariableDeclaration) parameter;
                        String type = arg.getType().toString();
                        argTypes.add(type);
                        info.argumentTypes.add(type);
                    }

                    SwumRecord swum = SwumRecord.fromMethodName(
                            info.id,
                            info.className,
                            info.methodName,
                            info.returnType,
                            argTypes,
                            info.constructor);

                    info.swumId = swum.id;
                    info.parseId = swum.parseId;
                    info.swumVerb = swum.action;
                    info.swumObject = swum.theme;

                    methods.add(info);
                    startMap.put(node.getStartPosition(), info);

                    String classNameKey = classAndMethodKey(info.className, info.methodName, info.argumentTypes.size(), info.constructor);
                    byClassAndName.computeIfAbsent(classNameKey, k -> new ArrayList<>()).add(info);

                    String nameArityKey = methodAndArityKey(info.methodName, info.argumentTypes.size(), info.constructor);
                    byNameAndArity.computeIfAbsent(nameArityKey, k -> new ArrayList<>()).add(info);
                }
            });
            methodByStart.put(file, startMap);
        }

        for (ParsedFile file : files) {
            Map<Integer, MethodInfo> startMap = methodByStart.get(file);
            file.compilationUnit.accept(new ASTVisitor() {
                private MethodInfo current;

                @Override
                public boolean visit(MethodDeclaration node) {
                    current = startMap.get(node.getStartPosition());
                    return true;
                }

                @Override
                public void endVisit(MethodDeclaration node) {
                    current = null;
                }

                @Override
                public boolean visit(MethodInvocation node) {
                    if (current == null) {
                        return true;
                    }

                    List<MethodInfo> targets = resolveMethodTargets(
                            current.className,
                            node.getExpression(),
                            node.getName().getIdentifier(),
                            node.arguments().size(),
                            false,
                            byClassAndName,
                            byNameAndArity);
                    for (MethodInfo target : targets) {
                        addEdge(current, target);
                    }
                    return true;
                }

                @Override
                public boolean visit(SuperMethodInvocation node) {
                    if (current == null) {
                        return true;
                    }

                    List<MethodInfo> targets = resolveMethodTargets(
                            current.className,
                            null,
                            node.getName().getIdentifier(),
                            node.arguments().size(),
                            false,
                            byClassAndName,
                            byNameAndArity);
                    for (MethodInfo target : targets) {
                        addEdge(current, target);
                    }
                    return true;
                }

                @Override
                public boolean visit(ConstructorInvocation node) {
                    if (current == null) {
                        return true;
                    }

                    List<MethodInfo> targets = resolveMethodTargets(
                            current.className,
                            null,
                            current.className,
                            node.arguments().size(),
                            true,
                            byClassAndName,
                            byNameAndArity);
                    for (MethodInfo target : targets) {
                        addEdge(current, target);
                    }
                    return true;
                }

                @Override
                public boolean visit(ClassInstanceCreation node) {
                    if (current == null) {
                        return true;
                    }

                    String typeName = node.getType() == null ? "" : node.getType().toString();
                    String simpleType = simpleTypeName(typeName);
                    List<MethodInfo> targets = resolveMethodTargets(
                            current.className,
                            null,
                            simpleType,
                            node.arguments().size(),
                            true,
                            byClassAndName,
                            byNameAndArity);
                    for (MethodInfo target : targets) {
                        addEdge(current, target);
                    }
                    return true;
                }
            });
        }

        return new AnalysisResult(methods);
    }

    private static void addEdge(MethodInfo from, MethodInfo to) {
        if (from.id == to.id) {
            return;
        }
        from.addCall(to.id);
        to.addCalledBy(from.id);
    }

    private static List<MethodInfo> resolveMethodTargets(
            String callerClass,
            Expression expression,
            String methodName,
            int argCount,
            boolean constructor,
            Map<String, List<MethodInfo>> byClassAndName,
            Map<String, List<MethodInfo>> byNameAndArity) {

        List<MethodInfo> results = new ArrayList<>();
        String explicitClass = classFromExpression(expression);

        if (explicitClass != null && !explicitClass.isBlank()) {
            String key = classAndMethodKey(explicitClass, methodName, argCount, constructor);
            List<MethodInfo> exact = byClassAndName.getOrDefault(key, List.of());
            if (!exact.isEmpty()) {
                results.addAll(exact);
                return results;
            }
        }

        String localKey = classAndMethodKey(callerClass, methodName, argCount, constructor);
        List<MethodInfo> local = byClassAndName.getOrDefault(localKey, List.of());
        if (!local.isEmpty()) {
            results.addAll(local);
            return results;
        }

        String globalKey = methodAndArityKey(methodName, argCount, constructor);
        List<MethodInfo> global = byNameAndArity.getOrDefault(globalKey, List.of());
        if (global.size() == 1) {
            results.add(global.get(0));
        }
        return results;
    }

    private static String classFromExpression(Expression expression) {
        if (expression == null) {
            return null;
        }
        if (expression instanceof SimpleName) {
            String token = ((SimpleName) expression).getIdentifier();
            if (!token.isEmpty() && Character.isUpperCase(token.charAt(0))) {
                return token;
            }
        }
        if (expression instanceof QualifiedName) {
            String token = ((QualifiedName) expression).getName().getIdentifier();
            if (!token.isEmpty() && Character.isUpperCase(token.charAt(0))) {
                return token;
            }
        }
        return null;
    }

    private static String classAndMethodKey(String className, String methodName, int arity, boolean constructor) {
        return className.toLowerCase(Locale.ROOT) + "#" + methodName.toLowerCase(Locale.ROOT) + "#" + arity + "#" + constructor;
    }

    private static String methodAndArityKey(String methodName, int arity, boolean constructor) {
        return methodName.toLowerCase(Locale.ROOT) + "#" + arity + "#" + constructor;
    }

    private static String simpleTypeName(String type) {
        if (type == null || type.isBlank()) {
            return "";
        }
        String cleaned = type.replaceAll("<.*>", "");
        int dot = cleaned.lastIndexOf('.');
        if (dot >= 0 && dot < cleaned.length() - 1) {
            return cleaned.substring(dot + 1);
        }
        return cleaned;
    }

    private static void writeSwumOut(Path swumOutPath, List<MethodInfo> methods) throws IOException {

        try (BufferedWriter writer = Files.newBufferedWriter(swumOutPath, StandardCharsets.UTF_8)) {
            for (MethodInfo method : methods) {
                SwumRecord record = SwumRecord.fromMethodName(
                        method.id,
                        method.className,
                        method.methodName,
                        method.returnType,
                        method.argumentTypes,
                        method.constructor);

                writer.write(record.toOutLine());
                writer.newLine();
            }
        }
    }

    private static void writeCallGraph(Path callGraphPath, List<MethodInfo> methods) throws IOException {
        Map<Integer, MethodInfo> byId = new HashMap<>();
        for (MethodInfo method : methods) {
            byId.put(method.id, method);
        }

        Set<String> lines = new HashSet<>();
        for (MethodInfo from : methods) {
            for (Integer toId : from.calls) {
                MethodInfo to = byId.get(toId);
                if (to == null) {
                    continue;
                }
                lines.add(from.parseId + " " + to.parseId);
            }
        }

        List<String> sorted = new ArrayList<>(lines);
        sorted.sort(String::compareTo);

        try (BufferedWriter writer = Files.newBufferedWriter(callGraphPath, StandardCharsets.UTF_8)) {
            for (String line : sorted) {
                writer.write(line);
                writer.newLine();
            }
        }
    }

    private static void inferUseTypes(List<MethodInfo> methods) {
        Map<Integer, MethodInfo> byId = new HashMap<>();
        for (MethodInfo method : methods) {
            byId.put(method.id, method);
        }

        for (MethodInfo target : methods) {
            for (int callerId : target.calledBy) {
                MethodInfo caller = byId.get(callerId);
                if (caller == null || caller.methodBody == null || caller.methodBody.isBlank()) {
                    continue;
                }

                String[] lines = caller.methodBody.split("\\R");
                for (String rawLine : lines) {
                    String line = rawLine.trim();
                    if (!line.contains(target.methodName)) {
                        continue;
                    }

                    target.useExample = line;
                    String lower = line.toLowerCase(Locale.ROOT);
                    if (lower.startsWith("if") || lower.startsWith("else")) {
                        target.useType = "conditional";
                    } else if (lower.startsWith("for") || lower.startsWith("while")) {
                        target.useType = "iteration";
                    } else if (line.contains("=") && !line.contains("==")) {
                        target.useType = "assignment";
                    } else {
                        target.useType = "procedural";
                    }
                    break;
                }

                if (!"unknown".equals(target.useType)) {
                    break;
                }
            }
        }
    }

    private static String classNameFromFile(Path path) {
        String filename = path.getFileName().toString();
        int dot = filename.lastIndexOf('.');
        return dot > 0 ? filename.substring(0, dot) : filename;
    }

    private static String readFully(Path file) throws IOException {
        StringBuilder sb = new StringBuilder();
        try (Reader reader = Files.newBufferedReader(file, StandardCharsets.UTF_8)) {
            char[] buffer = new char[4096];
            int read;
            while ((read = reader.read(buffer)) != -1) {
                sb.append(buffer, 0, read);
            }
        }
        return sb.toString();
    }

    private static final class ParsedFile {
        private final Path path;
        private final String className;
        private final String source;
        private final CompilationUnit compilationUnit;

        private ParsedFile(Path path, String className, String source, CompilationUnit compilationUnit) {
            this.path = path;
            this.className = className;
            this.source = source;
            this.compilationUnit = compilationUnit;
        }
    }

    private static final class AnalysisResult {
        private final List<MethodInfo> methods;

        private AnalysisResult(List<MethodInfo> methods) {
            this.methods = methods;
        }
    }

}

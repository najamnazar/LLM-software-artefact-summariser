import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.time.Duration;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class LlmSummaryGenerator {

    private static final String DEFAULT_SYSTEM_PROMPT =
        "You are documenting Java methods for developers. Write a concise natural-language summary in 1-2 sentences. Mention intent first, then usage context if available. Avoid filler and do not include bullet points.";
    private static final List<String> DEFAULT_USER_PROMPT_SECTIONS = List.of(
        "id: {method.id}",
        "class: {method.className}",
        "name: {method.name}",
        "returnType: {method.returnType}",
        "swumVerb: {method.swumVerb}",
        "swumObject: {method.swumObject}",
        "calledByCount: {method.calledCount}",
        "callsCount: {method.callsCount}",
        "useType: {method.useType}",
        "useExample: {method.useExample}");

    public static void main(String[] args) throws Exception {
        if (args.length >= 2 && args[0].equalsIgnoreCase("ALL")) {
            String modelKey     = args[1].toUpperCase();
            Path allOutputDir   = args.length > 2 ? Paths.get(args[2]) : Paths.get("output", "all");
            Path envPath        = args.length > 3 ? Paths.get(args[3]) : Paths.get("..", ".env");
            int maxMethods      = args.length > 4 ? Integer.parseInt(args[4]) : -1;
            Path promptPath     = args.length > 5 ? Paths.get(args[5]) : Paths.get("..", "resources", "prompts.json");
            Path groundTruthDir = args.length > 6 ? Paths.get(args[6]) : Paths.get("input", "ground-truth");
            runAll(modelKey, allOutputDir, envPath, maxMethods, promptPath, groundTruthDir);
        } else {
            Path jsonPath    = args.length > 0 ? Paths.get(args[0]) : Paths.get("output", "nanoxml-methods.json");
            Path outputPath  = args.length > 1 ? Paths.get(args[1]) : Paths.get("output", "nanoxml-method-summaries.json");
            Path envPath     = args.length > 2 ? Paths.get(args[2]) : Paths.get("..", ".env");
            int maxMethods   = args.length > 3 ? Integer.parseInt(args[3]) : -1;
            Path promptPath  = args.length > 4 ? Paths.get(args[4]) : Paths.get("..", "resources", "prompts.json");
            if (args.length < 6) {
                throw new IllegalArgumentException(
                    "Model key required as 6th argument. Example: MISTRAL, GPT, CLAUDE, QWEN");
            }
            String modelKey      = args[5].toUpperCase();
            Path groundTruthPath = args.length > 6 ? Paths.get(args[6]) : null;
            runSingle(jsonPath, outputPath, envPath, maxMethods, promptPath, modelKey,
                      loadGroundTruthKeys(groundTruthPath));
        }
    }

    private static void runAll(String modelKey, Path allOutputDir, Path envPath, int maxMethods, Path promptPath, Path groundTruthDir) throws Exception {
        if (!Files.isDirectory(allOutputDir)) {
            throw new IllegalArgumentException("All-projects directory not found: " + allOutputDir.toAbsolutePath());
        }

        List<Path> projectDirs = new ArrayList<>();
        try (var stream = Files.list(allOutputDir)) {
            stream.filter(Files::isDirectory).sorted().forEach(projectDirs::add);
        }
        if (projectDirs.isEmpty()) {
            throw new IllegalStateException("No project subdirectories found in: " + allOutputDir.toAbsolutePath());
        }

        StringBuilder names = new StringBuilder();
        for (Path d : projectDirs) {
            if (names.length() > 0) names.append(", ");
            names.append(d.getFileName());
        }
        System.out.println("Projects found: " + names);
        System.out.println("Model key: " + modelKey);

        List<MethodSummary> allSummaries = new ArrayList<>();
        String modelUsed = null;

        for (Path projectDir : projectDirs) {
            String projectName = projectDir.getFileName().toString();

            List<Path> methodFiles = new ArrayList<>();
            try (var stream = Files.list(projectDir)) {
                stream.filter(p -> p.getFileName().toString().endsWith("-methods.json"))
                      .sorted()
                      .forEach(methodFiles::add);
            }
            if (methodFiles.isEmpty()) {
                System.out.println("\nSkipping " + projectName + ": no *-methods.json found");
                continue;
            }

            Set<String> allowedKeys = null;
            if (groundTruthDir != null) {
                Path gtFile = findGroundTruthFile(groundTruthDir, projectName);
                if (gtFile != null) {
                    allowedKeys = loadGroundTruthKeys(gtFile);
                    System.out.println("Ground-truth: " + gtFile.getFileName() + " (" + allowedKeys.size() + " methods)");
                } else {
                    System.out.println("[WARN] No ground-truth file matched for project: " + projectName);
                }
            }

            Path jsonPath = methodFiles.get(0);

            System.out.println("\n=== Project: " + projectName + " ===");
            GenerationResult result = generateSummaries(jsonPath, envPath, maxMethods, promptPath, modelKey, allowedKeys, projectName);
            allSummaries.addAll(result.summaries);
            modelUsed = result.model;

            Path combinedOutput = combinedOutputPath(allOutputDir, modelKey);
            Files.createDirectories(combinedOutput.getParent());
            Files.writeString(combinedOutput, toJson(allSummaries, modelUsed), StandardCharsets.UTF_8);
            System.out.println("Checkpoint saved: " + combinedOutput.toAbsolutePath() + " (" + allSummaries.size() + " summaries so far)");
        }

        if (allSummaries.isEmpty()) {
            throw new IllegalStateException("No summaries generated across all projects.");
        }

        System.out.println("\nCombined summary file: " + combinedOutputPath(allOutputDir, modelKey).toAbsolutePath());
        System.out.println("Total summaries: " + allSummaries.size());
    }

    private static Path combinedOutputPath(Path allOutputDir, String modelKey) {
        Path outputRoot = allOutputDir.toAbsolutePath().normalize().getParent();
        return (outputRoot != null ? outputRoot : Paths.get("."))
                .resolve("SUMSLICE_" + modelKey.toUpperCase() + "_SUMMARY.json");
    }

    private static void runSingle(Path jsonPath, Path outputPath, Path envPath, int maxMethods, Path promptPath, String modelKey, Set<String> allowedKeys) throws Exception {
        GenerationResult result = generateSummaries(jsonPath, envPath, maxMethods, promptPath, modelKey, allowedKeys, null);

        Files.createDirectories(outputPath.getParent());
        Files.writeString(outputPath, toJson(result.summaries, result.model), StandardCharsets.UTF_8);
        System.out.println("Summary file: " + outputPath.toAbsolutePath());
        System.out.println("Model used: " + result.model);
        System.out.println("Prompt file: " + promptPath.toAbsolutePath());
    }

    private static GenerationResult generateSummaries(Path jsonPath, Path envPath, int maxMethods, Path promptPath, String modelKey, Set<String> allowedKeys, String projectName) throws Exception {
        if (!Files.exists(jsonPath)) {
            throw new IllegalArgumentException("Method JSON not found: " + jsonPath.toAbsolutePath());
        }
        if (!Files.exists(envPath)) {
            throw new IllegalArgumentException("Env file not found: " + envPath.toAbsolutePath());
        }
        if (!Files.exists(promptPath)) {
            throw new IllegalArgumentException("Prompt JSON not found: " + promptPath.toAbsolutePath());
        }

        PromptConfig promptConfig = loadPromptConfig(promptPath);

        Map<String, String> env = loadEnv(envPath);
        String apiKey = required(env, "OPENROUTER_API_KEY");
        String apiUrl = required(env, "OPENROUTER_API_URL");
        String model = required(env, modelKey + "_MODEL");
        int maxTokens = parseInt(env.getOrDefault("OPENROUTER_MAX_COMPLETION_TOKENS",
                                  env.getOrDefault("OPENROUTER_MAX_TOKENS", "512")), 512);
        double temperature = parseDouble(required(env, "OPENROUTER_TEMPERATURE"), 0.0);

        String json = Files.readString(jsonPath, StandardCharsets.UTF_8);
        List<MethodDoc> methods = parseMethods(json);
        if (methods.isEmpty()) {
            throw new IllegalStateException("No methods found in JSON representation.");
        }

        if (allowedKeys != null && !allowedKeys.isEmpty()) {
            List<MethodDoc> filtered = new ArrayList<>();
            for (MethodDoc m : methods) {
                if (allowedKeys.contains(groundTruthKey(m.className, m.name))) {
                    filtered.add(m);
                }
            }
            System.out.println("Ground-truth filter: " + filtered.size() + "/" + methods.size() + " methods selected (matched by class+name).");
            methods = filtered;
        }

        if (maxMethods > 0 && methods.size() > maxMethods) {
            methods = methods.subList(0, maxMethods);
        }

        HttpClient client = HttpClient.newBuilder()
                .connectTimeout(Duration.ofSeconds(30))
                .build();

        List<MethodSummary> summaries = new ArrayList<>();
        int index = 0;
        for (MethodDoc method : methods) {
            index++;
            String prompt = buildUserPrompt(method, promptConfig.userPromptSections);
            String response = callChatCompletion(
                    client,
                    apiUrl,
                    apiKey,
                    model,
                    promptConfig.systemPrompt,
                    prompt,
                    maxTokens,
                    temperature);
            String summary = extractAssistantContent(response);
            if (summary == null || summary.isBlank()) {
                summary = "Summary generation failed for this method.";
            }

            summaries.add(new MethodSummary(method.id, method.name, method.className, summary.trim(), projectName));
            System.out.println("Summarized " + index + "/" + methods.size() + ": " + method.className + "." + method.name);
        }

        return new GenerationResult(model, summaries);
    }

    private static Set<String> loadGroundTruthKeys(Path gtFile) throws IOException {
        if (gtFile == null || !Files.exists(gtFile)) {
            return null;
        }
        String json = Files.readString(gtFile, StandardCharsets.UTF_8);
        int arrKey = json.indexOf("\"summaries\"");
        if (arrKey < 0) {
            return null;
        }
        int arrStart = json.indexOf('[', arrKey);
        if (arrStart < 0) {
            return null;
        }
        int arrEnd = findMatchingBracket(json, arrStart, '[', ']');
        if (arrEnd < 0) {
            return null;
        }

        String arrayText = json.substring(arrStart + 1, arrEnd);
        Set<String> keys = new HashSet<>();
        for (String object : splitTopLevelObjects(arrayText)) {
            String methodName = extractString(object, "methodName");
            String className = extractString(object, "className");
            if (methodName != null && className != null) {
                keys.add(groundTruthKey(className, methodName));
            }
        }
        return keys.isEmpty() ? null : keys;
    }

    private static String groundTruthKey(String className, String methodName) {
        return className + "#" + methodName;
    }

    private static Path findGroundTruthFile(Path groundTruthDir, String projectName) throws IOException {
        if (!Files.isDirectory(groundTruthDir)) {
            return null;
        }
        String projectLower = projectName.toLowerCase(Locale.ROOT);
        List<Path> candidates = new ArrayList<>();
        try (var stream = Files.list(groundTruthDir)) {
            stream.filter(p -> p.getFileName().toString().endsWith("-example-summaries.json"))
                  .forEach(candidates::add);
        }
        for (Path candidate : candidates) {
            String prefix = candidate.getFileName().toString()
                    .replace("-example-summaries.json", "")
                    .toLowerCase(Locale.ROOT);
            if (projectLower.startsWith(prefix) || prefix.startsWith(projectLower)) {
                return candidate;
            }
        }
        return null;
    }

    private static String buildUserPrompt(MethodDoc method, List<String> sections) {
        StringBuilder sb = new StringBuilder();
        sb.append("Method metadata:\n");
        for (String section : sections) {
            if (section == null || section.isBlank()) {
                continue;
            }
            sb.append(applyMethodTemplate(section, method)).append('\n');
        }
        return sb.toString();
    }

    private static String callChatCompletion(
            HttpClient client,
            String apiUrl,
            String apiKey,
            String model,
            String systemPrompt,
            String prompt,
            int maxTokens,
            double temperature) throws IOException, InterruptedException {

        String payload = "{" +
                "\"model\":\"" + escapeJson(model) + "\"," +
                "\"messages\":[" +
                "{\"role\":\"system\",\"content\":\"" + escapeJson(systemPrompt) + "\"}," +
                "{\"role\":\"user\",\"content\":\"" + escapeJson(prompt) + "\"}" +
                "]," +
                "\"max_tokens\":" + maxTokens + "," +
                "\"temperature\":" + String.format(Locale.ROOT, "%.2f", temperature) +
                "}";

        HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create(apiUrl))
                .timeout(Duration.ofSeconds(90))
                .header("Authorization", "Bearer " + apiKey)
                .header("Content-Type", "application/json")
                .POST(HttpRequest.BodyPublishers.ofString(payload, StandardCharsets.UTF_8))
                .build();

        HttpResponse<String> response = sendWithRetry(client, request, 5);
        if (response.statusCode() >= 400) {
            throw new IOException("LLM API error (" + response.statusCode() + "): " + response.body());
        }
        return response.body();
    }

    private static HttpResponse<String> sendWithRetry(HttpClient client, HttpRequest request, int maxAttempts)
            throws IOException, InterruptedException {
        IOException lastError = null;
        for (int attempt = 1; attempt <= maxAttempts; attempt++) {
            try {
                return client.send(request, HttpResponse.BodyHandlers.ofString(StandardCharsets.UTF_8));
            } catch (IOException e) {
                lastError = e;
            } catch (java.nio.channels.UnresolvedAddressException e) {
                lastError = new IOException("DNS resolution failed: " + e.getMessage(), e);
            }
            if (attempt < maxAttempts) {
                long backoffMs = 2000L * attempt;
                System.out.println("[WARN] Network error on attempt " + attempt + "/" + maxAttempts
                        + ": " + lastError.getMessage() + " - retrying in " + (backoffMs / 1000) + "s");
                Thread.sleep(backoffMs);
            }
        }
        throw lastError;
    }

    private static String extractAssistantContent(String responseJson) {
        Pattern p = Pattern.compile("\\\"content\\\"\\s*:\\s*\\\"((?:\\\\\\\"|\\\\\\\\|\\\\n|\\\\r|\\\\t|[^\\\"])*)\\\"");
        Matcher m = p.matcher(responseJson);
        if (!m.find()) {
            return null;
        }
        return unescapeJson(m.group(1));
    }

    private static List<MethodDoc> parseMethods(String json) {
        int methodsKey = json.indexOf("\"method\"");
        if (methodsKey < 0) {
            return List.of();
        }
        int arrStart = json.indexOf('[', methodsKey);
        if (arrStart < 0) {
            return List.of();
        }
        int arrEnd = findMatchingBracket(json, arrStart, '[', ']');
        if (arrEnd < 0) {
            return List.of();
        }

        String arrayText = json.substring(arrStart + 1, arrEnd);
        List<String> objects = splitTopLevelObjects(arrayText);
        List<MethodDoc> methods = new ArrayList<>();
        for (String object : objects) {
            MethodDoc doc = parseMethodObject(object);
            if (doc != null) {
                methods.add(doc);
            }
        }
        return methods;
    }

    private static MethodDoc parseMethodObject(String object) {
        Integer id = extractInt(object, "id");
        String name = extractString(object, "name");
        String className = extractString(object, "class");
        String returnType = extractString(object, "returntype");
        String swumVerb = extractNestedString(object, "swum", "verb");
        String swumObject = extractNestedString(object, "swum", "object");
        String useType = extractNestedString(object, "use", "type");
        String useExample = extractNestedString(object, "use", "example");
        int calledCount = extractArrayCount(object, "called");
        int callsCount = extractArrayCount(object, "calls");

        if (id == null || name == null || className == null) {
            return null;
        }

        MethodDoc doc = new MethodDoc();
        doc.id = id;
        doc.name = name;
        doc.className = className;
        doc.returnType = returnType == null ? "void" : returnType;
        doc.swumVerb = swumVerb == null ? "unknown" : swumVerb;
        doc.swumObject = swumObject == null ? "unknown" : swumObject;
        doc.useType = useType == null ? "unknown" : useType;
        doc.useExample = useExample == null ? "unknown" : useExample;
        doc.calledCount = calledCount;
        doc.callsCount = callsCount;
        return doc;
    }

    private static Integer extractInt(String src, String key) {
        Pattern p = Pattern.compile("\\\"" + Pattern.quote(key) + "\\\"\\s*:\\s*(\\d+)");
        Matcher m = p.matcher(src);
        if (!m.find()) {
            return null;
        }
        return Integer.parseInt(m.group(1));
    }

    private static String extractString(String src, String key) {
        Pattern p = Pattern.compile("\\\"" + Pattern.quote(key) + "\\\"\\s*:\\s*\\\"((?:\\\\\\\"|\\\\\\\\|\\\\n|\\\\r|\\\\t|[^\\\"])*)\\\"");
        Matcher m = p.matcher(src);
        if (!m.find()) {
            return null;
        }
        return unescapeJson(m.group(1));
    }

    private static String extractNestedString(String src, String objectKey, String fieldKey) {
        Pattern container = Pattern.compile("\\\"" + Pattern.quote(objectKey) + "\\\"\\s*:\\s*\\{(.*?)\\}", Pattern.DOTALL);
        Matcher cm = container.matcher(src);
        if (!cm.find()) {
            return null;
        }
        return extractString(cm.group(1), fieldKey);
    }

    private static int extractArrayCount(String src, String key) {
        Pattern p = Pattern.compile("\\\"" + Pattern.quote(key) + "\\\"\\s*:\\s*\\[(.*?)\\]", Pattern.DOTALL);
        Matcher m = p.matcher(src);
        if (!m.find()) {
            return 0;
        }
        String inner = m.group(1).trim();
        if (inner.isEmpty()) {
            return 0;
        }
        int count = 1;
        for (int i = 0; i < inner.length(); i++) {
            if (inner.charAt(i) == ',') {
                count++;
            }
        }
        return count;
    }

    private static List<String> splitTopLevelObjects(String arrayText) {
        List<String> out = new ArrayList<>();
        int depth = 0;
        int start = -1;
        boolean inString = false;
        boolean escape = false;

        for (int i = 0; i < arrayText.length(); i++) {
            char ch = arrayText.charAt(i);
            if (inString) {
                if (escape) {
                    escape = false;
                } else if (ch == '\\') {
                    escape = true;
                } else if (ch == '"') {
                    inString = false;
                }
                continue;
            }

            if (ch == '"') {
                inString = true;
                continue;
            }

            if (ch == '{') {
                if (depth == 0) {
                    start = i;
                }
                depth++;
            } else if (ch == '}') {
                depth--;
                if (depth == 0 && start >= 0) {
                    out.add(arrayText.substring(start, i + 1));
                    start = -1;
                }
            }
        }

        return out;
    }

    private static int findMatchingBracket(String text, int openIndex, char openChar, char closeChar) {
        int depth = 0;
        boolean inString = false;
        boolean escape = false;

        for (int i = openIndex; i < text.length(); i++) {
            char ch = text.charAt(i);
            if (inString) {
                if (escape) {
                    escape = false;
                } else if (ch == '\\') {
                    escape = true;
                } else if (ch == '"') {
                    inString = false;
                }
                continue;
            }

            if (ch == '"') {
                inString = true;
                continue;
            }

            if (ch == openChar) {
                depth++;
            } else if (ch == closeChar) {
                depth--;
                if (depth == 0) {
                    return i;
                }
            }
        }

        return -1;
    }

    private static PromptConfig loadPromptConfig(Path promptPath) throws IOException {
        String promptJson = Files.readString(promptPath, StandardCharsets.UTF_8);

        // prompts.json is shared across projects; scope extraction to this project's
        // namespace so sibling projects' "content"/"sections" keys are never matched.
        String scoped = extractObjectRegion(promptJson, "sumslice_llm");
        scoped = extractObjectRegion(scoped != null ? scoped : promptJson, "sumslice-llm");
        if (scoped == null) {
            scoped = promptJson;
        }

        String systemPrompt = extractString(scoped, "content");
        if (systemPrompt == null || systemPrompt.isBlank()) {
            systemPrompt = DEFAULT_SYSTEM_PROMPT;
        }

        List<String> sections = extractStringArray(scoped, "sections");
        if (sections.isEmpty()) {
            sections = DEFAULT_USER_PROMPT_SECTIONS;
        }

        return new PromptConfig(systemPrompt, sections);
    }

    private static String extractObjectRegion(String src, String key) {
        if (src == null) {
            return null;
        }
        int keyIndex = src.indexOf("\"" + key + "\"");
        if (keyIndex < 0) {
            return null;
        }
        int openIndex = src.indexOf('{', keyIndex);
        if (openIndex < 0) {
            return null;
        }
        int closeIndex = findMatchingBracket(src, openIndex, '{', '}');
        if (closeIndex < 0) {
            return null;
        }
        return src.substring(openIndex + 1, closeIndex);
    }

    private static List<String> extractStringArray(String src, String key) {
        Pattern p = Pattern.compile("\\\"" + Pattern.quote(key) + "\\\"\\s*:\\s*\\[(.*?)\\]", Pattern.DOTALL);
        Matcher m = p.matcher(src);
        if (!m.find()) {
            return List.of();
        }

        String inner = m.group(1);
        List<String> out = new ArrayList<>();
        Pattern quoted = Pattern.compile("\\\"((?:\\\\\\\"|\\\\\\\\|\\\\n|\\\\r|\\\\t|[^\\\"])*)\\\"");
        Matcher qm = quoted.matcher(inner);
        while (qm.find()) {
            out.add(unescapeJson(qm.group(1)));
        }
        return out;
    }

    private static String applyMethodTemplate(String template, MethodDoc method) {
        return template
                .replace("{method.id}", String.valueOf(method.id))
                .replace("{method.className}", safe(method.className))
                .replace("{method.name}", safe(method.name))
                .replace("{method.returnType}", safe(method.returnType))
                .replace("{method.swumVerb}", safe(method.swumVerb))
                .replace("{method.swumObject}", safe(method.swumObject))
                .replace("{method.calledCount}", String.valueOf(method.calledCount))
                .replace("{method.callsCount}", String.valueOf(method.callsCount))
                .replace("{method.useType}", safe(method.useType))
                .replace("{method.useExample}", safe(method.useExample))
                .replace("<method.id>", String.valueOf(method.id))
                .replace("<method.className>", safe(method.className))
                .replace("<method.name>", safe(method.name))
                .replace("<method.returnType>", safe(method.returnType))
                .replace("<method.swumVerb>", safe(method.swumVerb))
                .replace("<method.swumObject>", safe(method.swumObject))
                .replace("<method.calledCount>", String.valueOf(method.calledCount))
                .replace("<method.callsCount>", String.valueOf(method.callsCount))
                .replace("<method.useType>", safe(method.useType))
                .replace("<method.useExample>", safe(method.useExample));
    }

    private static String safe(String value) {
        return value == null ? "unknown" : value;
    }

    private static String toJson(List<MethodSummary> summaries, String model) {
        StringBuilder sb = new StringBuilder();
        sb.append("{\n  \"model\": \"").append(escapeJson(model)).append("\",\n");
        sb.append("  \"summaries\": [\n");
        for (int i = 0; i < summaries.size(); i++) {
            MethodSummary s = summaries.get(i);
            sb.append("    {\n");
            sb.append("      \"id\": ").append(s.id).append(",\n");
            if (s.project != null) {
                sb.append("      \"project\": \"").append(escapeJson(s.project)).append("\",\n");
            }
            sb.append("      \"name\": \"").append(escapeJson(s.name)).append("\",\n");
            sb.append("      \"class\": \"").append(escapeJson(s.className)).append("\",\n");
            sb.append("      \"summary\": \"").append(escapeJson(s.summary)).append("\"\n");
            sb.append("    }");
            if (i < summaries.size() - 1) {
                sb.append(',');
            }
            sb.append('\n');
        }
        sb.append("  ]\n}\n");
        return sb.toString();
    }

    private static Map<String, String> loadEnv(Path envPath) throws IOException {
        Map<String, String> values = new HashMap<>();
        List<String> lines = Files.readAllLines(envPath, StandardCharsets.UTF_8);
        for (String raw : lines) {
            String line = raw.trim();
            if (line.isEmpty() || line.startsWith("#")) {
                continue;
            }
            int eq = line.indexOf('=');
            if (eq <= 0) {
                continue;
            }
            String key = line.substring(0, eq).trim();
            String value = line.substring(eq + 1).trim();
            values.put(key, value);
        }
        return values;
    }

    private static String required(Map<String, String> env, String key) {
        String value = env.get(key);
        if (value == null || value.isBlank()) {
            throw new IllegalStateException("Missing required env key: " + key);
        }
        return value;
    }

    private static int parseInt(String value, int fallback) {
        try {
            return Integer.parseInt(value.trim());
        } catch (Exception ex) {
            return fallback;
        }
    }

    private static double parseDouble(String value, double fallback) {
        try {
            return Double.parseDouble(value.trim());
        } catch (Exception ex) {
            return fallback;
        }
    }

    private static String escapeJson(String value) {
        if (value == null) {
            return "";
        }
        return value
                .replace("\\", "\\\\")
                .replace("\"", "\\\"")
                .replace("\n", "\\n")
                .replace("\r", "\\r")
                .replace("\t", "\\t");
    }

    private static String unescapeJson(String value) {
        if (value == null) {
            return "";
        }
        StringBuilder sb = new StringBuilder();
        boolean escaping = false;
        for (int i = 0; i < value.length(); i++) {
            char ch = value.charAt(i);
            if (!escaping) {
                if (ch == '\\') {
                    escaping = true;
                } else {
                    sb.append(ch);
                }
            } else {
                switch (ch) {
                    case 'n':
                        sb.append('\n');
                        break;
                    case 'r':
                        sb.append('\r');
                        break;
                    case 't':
                        sb.append('\t');
                        break;
                    case '"':
                        sb.append('"');
                        break;
                    case '\\':
                        sb.append('\\');
                        break;
                    default:
                        sb.append(ch);
                }
                escaping = false;
            }
        }
        return sb.toString();
    }

    private static final class MethodDoc {
        private int id;
        private String name;
        private String className;
        private String returnType;
        private String swumVerb;
        private String swumObject;
        private String useType;
        private String useExample;
        private int calledCount;
        private int callsCount;
    }

    private static final class MethodSummary {
        private final int id;
        private final String name;
        private final String className;
        private final String summary;
        private final String project;

        private MethodSummary(int id, String name, String className, String summary, String project) {
            this.id = id;
            this.name = name;
            this.className = className;
            this.summary = summary;
            this.project = project;
        }
    }

    private static final class GenerationResult {
        private final String model;
        private final List<MethodSummary> summaries;

        private GenerationResult(String model, List<MethodSummary> summaries) {
            this.model = model;
            this.summaries = summaries;
        }
    }

    private static final class PromptConfig {
        private final String systemPrompt;
        private final List<String> userPromptSections;

        private PromptConfig(String systemPrompt, List<String> userPromptSections) {
            this.systemPrompt = systemPrompt;
            this.userPromptSections = userPromptSections;
        }
    }
}

package sumslicellm;

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
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Step 3. Generate one summary per sample method with an LLM, replacing
 * SumSlice's lexicalisation, aggregation and realisation. The model receives
 * the SumSlice content-determination facts for that method together with its
 * real declaration in the project source (input/dataset/selected_methods.jsonl,
 * written by {@link GenerateToolSummaries}).
 *
 * Usage: java sumslicellm.LlmSummaryGenerator MODEL [MODEL ...] [--fresh] [--limit N]
 *        MODEL = GPT | QWEN | CLAUDE | MISTRAL | ALL
 *
 * Configuration (.env in the repository root, or environment variables):
 *   OPENROUTER_API_KEY, OPENROUTER_API_URL, OPENROUTER_TEMPERATURE,
 *   OPENROUTER_MAX_COMPLETION_TOKENS, &lt;MODEL&gt;_MODEL
 * Prompt: ../resources/prompts.json, section "sumslice_llm" -> "sumslice-llm",
 * shared with PR_LLM and DPS_LLM.
 *
 * Output: output/SUMSLICE_&lt;MODEL&gt;_SUMMARY.jsonl
 *   {project, method_id, class, name, system, model, summary}
 * The run is resumable: methods with a non-empty summary are kept, others are
 * (re)generated. Every placeholder in the template must be filled, otherwise
 * the run stops before any API call.
 */
public final class LlmSummaryGenerator {

    static final List<String> ALL_MODELS = List.of("GPT", "QWEN", "CLAUDE", "MISTRAL");
    private static final Pattern PLACEHOLDER = Pattern.compile("\\{([A-Za-z_][A-Za-z0-9_.]*)\\}");

    /**
     * Models whose internal chain of thought must be switched off.
     *
     * qwen3.7-plus is a hybrid reasoning model that thinks before answering.
     * Under this pipeline's token budget it spent the whole allowance on that
     * chain of thought and hit finish_reason='length' before emitting any
     * answer, so every item came back with an empty completion:
     *
     *     completion_tokens: 512, reasoning_tokens: 512, content: ""
     *
     * Raising max_tokens for Qwen alone was rejected: it would give one model
     * extended reasoning the other three do not get, confounding the ranking
     * and Friedman tests. Disabling reasoning has all four models answer
     * directly under the same budget. PR_LLM resolves this the same way
     * (see PR_LLM/python/pr_llm_summariser.py, NO_REASONING_LABELS).
     */
    static final List<String> NO_REASONING = List.of("QWEN");

    /** Sends one chat request and returns the assistant text. */
    interface ChatClient {
        String complete(String model, String system, String user, boolean disableReasoning)
                throws IOException, InterruptedException;
    }

    public static void main(String[] args) throws Exception {
        List<String> models = new ArrayList<>();
        boolean fresh = false;
        int limit = -1;
        for (int i = 0; i < args.length; i++) {
            if (args[i].equals("--fresh")) {
                fresh = true;
            } else if (args[i].equals("--limit")) {
                limit = Integer.parseInt(args[++i]);
            } else if (args[i].equalsIgnoreCase("ALL")) {
                models.addAll(ALL_MODELS);
            } else {
                String m = args[i].toUpperCase(Locale.ROOT);
                if (!ALL_MODELS.contains(m)) {
                    throw new IllegalArgumentException("Unknown model " + args[i] + "; use " + ALL_MODELS + " or ALL");
                }
                models.add(m);
            }
        }
        if (models.isEmpty()) {
            throw new IllegalArgumentException("Usage: LlmSummaryGenerator MODEL [MODEL ...] [--fresh] [--limit N]");
        }

        Map<String, String> env = Env.load(Paths.get("..", ".env"));
        String apiUrl = env.getOrDefault("OPENROUTER_API_URL", "https://openrouter.ai/api/v1/chat/completions");
        String apiKey = Env.required(env, "OPENROUTER_API_KEY");
        double temperature = Double.parseDouble(env.getOrDefault("OPENROUTER_TEMPERATURE", "0.0"));
        int maxTokens = Integer.parseInt(env.getOrDefault("OPENROUTER_MAX_COMPLETION_TOKENS", "512"));
        ChatClient client = new OpenRouterClient(apiUrl, apiKey, temperature, maxTokens);

        int failures = 0;
        for (String key : models) {
            String modelId = Env.required(env, key + "_MODEL");
            failures += run(key, modelId, client, fresh, limit,
                    Paths.get("input", "dataset", "selected_methods.jsonl"),
                    Paths.get("..", "resources", "prompts.json"),
                    Paths.get("output", "SUMSLICE_" + key + "_SUMMARY.jsonl"));
        }
        if (failures > 0) {
            System.err.println(failures + " summaries are empty; rerun to retry them.");
            System.exit(2);
        }
    }

    static int run(String key, String modelId, ChatClient client, boolean fresh, int limit,
                   Path inputPath, Path promptPath, Path outputPath) throws Exception {
        Map<String, Object> prompts = Json.obj(Json.obj(Json.readObject(promptPath).get("sumslice_llm")).get("sumslice-llm"));
        String system = Json.str(Json.obj(prompts.get("system_prompt")), "content");
        List<Object> sections = Json.arr(Json.obj(prompts.get("user_prompt_template")).get("sections"));
        String template = String.join("\n", sections.stream().map(Object::toString).toArray(String[]::new));

        List<Map<String, Object>> inputs = Json.readJsonl(inputPath);
        if (inputs.isEmpty()) {
            throw new IllegalStateException("No inputs in " + inputPath + "; run GenerateToolSummaries first.");
        }
        if (limit > 0 && inputs.size() > limit) {
            inputs = inputs.subList(0, limit);
        }
        // build every prompt first so a template problem stops the run before any API call
        List<String> userPrompts = new ArrayList<>();
        for (Map<String, Object> rec : inputs) {
            userPrompts.add(render(template, promptValues(rec)));
        }

        Map<String, Map<String, Object>> existing = new LinkedHashMap<>();
        if (!fresh) {
            for (Map<String, Object> r : Json.readJsonl(outputPath)) {
                existing.put(keyOf(r), r);
            }
        }

        List<Map<String, Object>> out = new ArrayList<>();
        int failed = 0;
        for (int i = 0; i < inputs.size(); i++) {
            Map<String, Object> rec = inputs.get(i);
            Map<String, Object> prev = existing.get(keyOf(rec));
            if (prev != null && prev.get("summary") != null && !prev.get("summary").toString().isBlank()) {
                out.add(prev);
                continue;
            }
            String summary = "";
            try {
                summary = client.complete(modelId, system, userPrompts.get(i),
                        NO_REASONING.contains(key)).trim();
            } catch (Exception e) {
                System.err.printf("[%s] %s.%s failed: %s%n", key, rec.get("class"), rec.get("name"), e.getMessage());
            }
            if (summary.isEmpty()) {
                failed++;
            }
            Map<String, Object> row = new LinkedHashMap<>();
            row.put("project", rec.get("project"));
            row.put("method_id", rec.get("method_id"));
            row.put("class", rec.get("class"));
            row.put("name", rec.get("name"));
            row.put("system", key);
            row.put("model", modelId);
            row.put("summary", summary);
            out.add(row);
            Json.writeJsonl(outputPath, out);   // checkpoint after every call
            System.out.printf("[%s] %d/%d %s %s.%s%n", key, i + 1, inputs.size(),
                    rec.get("project"), rec.get("class"), rec.get("name"));
            System.out.println(summary.isEmpty() ? "    (empty)" : indent(summary));
            System.out.println();
        }
        Json.writeJsonl(outputPath, out);
        System.out.printf("[%s] %d summaries -> %s (%d empty)%n", key, out.size(), outputPath, failed);
        return failed;
    }

    static String keyOf(Map<String, Object> r) {
        return r.get("project") + "::" + r.get("method_id");
    }

    /** Indents a summary by four spaces so it reads as a block under its method. */
    static String indent(String text) {
        return "    " + text.replace("\n", "\n    ");
    }

    /**
     * Values for the {@code {method.*}} placeholders of the shared prompt.
     * Every field is filled: the SumSlice facts as the tool determined them,
     * and the declaration read from the project source. A fact the original
     * omits (no usage example, no caller) and source that could not be
     * resolved both become an explicit "none" / "not available" rather than an
     * empty string, so the model is never handed a blank field.
     */
    static Map<String, String> promptValues(Map<String, Object> f) {
        Map<String, String> v = new LinkedHashMap<>();
        v.put("project", Json.str(f, "project"));
        v.put("method.id", num(f, "method_id"));
        v.put("method.className", Json.str(f, "class"));
        v.put("method.name", Json.str(f, "name"));
        v.put("method.classExtends", or(Json.str(f, "class_extends"), "none"));
        v.put("method.classImplements", or(Json.str(f, "class_implements"), "none"));
        v.put("method.parameters", or(Json.str(f, "parameters"), "none"));
        v.put("method.returnType", or(Json.str(f, "return_type"), "void"));
        v.put("method.swumVerb", or(Json.str(f, "swum_verb"), "none"));
        v.put("method.swumObject", or(Json.str(f, "swum_object"), "none"));
        v.put("method.body", or(Json.str(f, "body"), "not available in the project source"));

        List<String> callers = new ArrayList<>();
        for (Object o : Json.arr(f.get("output_used_by"))) {
            Map<String, Object> u = Json.obj(o);
            callers.add(Json.str(u, "verb") + " the " + Json.str(u, "object"));
        }
        v.put("method.topCallers", callers.isEmpty() ? "none" : String.join("; ", callers));
        v.put("method.calledCount", num(f, "called_count"));
        v.put("method.callsCount", num(f, "calls_count"));
        v.put("method.importanceLabel", or(Json.str(f, "importance"), "unknown"));
        v.put("method.useType", or(Json.str(f, "use_type"), "none"));
        v.put("method.useExample", or(Json.str(f, "use_example"), "none"));
        return v;
    }

    private static String or(String value, String fallback) {
        return value == null || value.isBlank() ? fallback : value;
    }

    /**
     * A numeric fact. A missing key means the input file was written by an
     * older GenerateToolSummaries, which is worth saying plainly rather than
     * failing later with a null pointer.
     */
    private static String num(Map<String, Object> f, String key) {
        Object value = f.get(key);
        if (!(value instanceof Number)) {
            throw new IllegalStateException("Field '" + key + "' is missing from the LLM input; "
                    + "rerun GenerateToolSummaries (step 2) to rebuild input/dataset/selected_methods.jsonl.");
        }
        return String.valueOf(((Number) value).longValue());
    }

    static String render(String template, Map<String, String> values) {
        Matcher m = PLACEHOLDER.matcher(template);
        StringBuffer sb = new StringBuffer();
        while (m.find()) {
            String value = values.get(m.group(1));
            if (value == null) {
                throw new IllegalStateException("Prompt placeholder {" + m.group(1) + "} has no value. Known: " + values.keySet());
            }
            m.appendReplacement(sb, Matcher.quoteReplacement(value));
        }
        m.appendTail(sb);
        return sb.toString();
    }

    /** OpenRouter (OpenAI-compatible) chat client with retries. */
    static final class OpenRouterClient implements ChatClient {
        private final HttpClient http = HttpClient.newBuilder().connectTimeout(Duration.ofSeconds(30)).build();
        private final String url;
        private final String key;
        private final double temperature;
        private final int maxTokens;

        OpenRouterClient(String url, String key, double temperature, int maxTokens) {
            this.url = url;
            this.key = key;
            this.temperature = temperature;
            this.maxTokens = maxTokens;
        }

        @Override
        public String complete(String model, String system, String user, boolean disableReasoning)
                throws IOException, InterruptedException {
            Map<String, Object> body = new LinkedHashMap<>();
            body.put("model", model);
            List<Object> messages = new ArrayList<>();
            messages.add(Map.of("role", "system", "content", system));
            messages.add(Map.of("role", "user", "content", user));
            body.put("messages", messages);
            body.put("temperature", temperature);
            body.put("max_tokens", maxTokens);
            if (disableReasoning) {
                body.put("reasoning", Map.of("enabled", false));
            }
            HttpRequest request = HttpRequest.newBuilder(URI.create(url))
                    .timeout(Duration.ofSeconds(120))
                    .header("Authorization", "Bearer " + key)
                    .header("Content-Type", "application/json")
                    .POST(HttpRequest.BodyPublishers.ofString(Json.write(body), StandardCharsets.UTF_8))
                    .build();
            IOException last = null;
            for (int attempt = 1; attempt <= 5; attempt++) {
                try {
                    HttpResponse<String> resp = http.send(request, HttpResponse.BodyHandlers.ofString(StandardCharsets.UTF_8));
                    if (resp.statusCode() == 429 || resp.statusCode() >= 500) {
                        last = new IOException("HTTP " + resp.statusCode() + ": " + resp.body());
                    } else if (resp.statusCode() >= 400) {
                        throw new IOException("HTTP " + resp.statusCode() + ": " + resp.body());
                    } else {
                        Map<String, Object> json = Json.obj(Json.parse(resp.body()));
                        List<Object> choices = Json.arr(json.get("choices"));
                        if (choices.isEmpty()) {
                            last = new IOException("no choices in response: " + resp.body());
                        } else {
                            Object content = Json.obj(Json.obj(choices.get(0)).get("message")).get("content");
                            if (content != null && !content.toString().isBlank()) {
                                return content.toString();
                            }
                            last = new IOException("empty completion");
                        }
                    }
                } catch (IOException e) {
                    if (e.getMessage() != null && e.getMessage().startsWith("HTTP 4")) {
                        throw e;
                    }
                    last = e;
                }
                Thread.sleep(2000L * attempt);
            }
            throw last;
        }
    }

    /** Reads KEY=VALUE lines from .env; environment variables take precedence. */
    static final class Env {
        static Map<String, String> load(Path path) throws IOException {
            Map<String, String> env = new LinkedHashMap<>();
            if (Files.exists(path)) {
                for (String line : Files.readAllLines(path, StandardCharsets.UTF_8)) {
                    String t = line.trim();
                    int eq = t.indexOf('=');
                    if (t.isEmpty() || t.startsWith("#") || eq < 0) {
                        continue;
                    }
                    String v = t.substring(eq + 1).trim();
                    if (v.length() >= 2 && (v.startsWith("\"") && v.endsWith("\"") || v.startsWith("'") && v.endsWith("'"))) {
                        v = v.substring(1, v.length() - 1);
                    }
                    env.put(t.substring(0, eq).trim(), v);
                }
            }
            env.putAll(System.getenv());
            return env;
        }

        static String required(Map<String, String> env, String key) {
            String v = env.get(key);
            if (v == null || v.isBlank()) {
                throw new IllegalStateException(key + " is not set (.env or environment)");
            }
            return v;
        }
    }
}

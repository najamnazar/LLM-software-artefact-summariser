package dps_llm.client;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;

/**
 * HTTP client for communicating with LLM APIs via the OpenRouter platform.
 * <p>
 * This client provides a simplified interface for making chat completion requests
 * to Large Language Models through OpenRouter's unified API. It handles request
 * construction, HTTP communication, and response parsing.
 * </p>
 * <p>
 * Key responsibilities:
 * <ul>
 *   <li>Construct properly formatted JSON payloads for chat completion requests</li>
 *   <li>Manage HTTP connections with timeout and retry logic</li>
 *   <li>Parse and extract text responses from LLM API responses</li>
 *   <li>Handle authentication and custom headers (referer, title)</li>
 * </ul>
 * </p>
 * 
 * @author Najam
 */
public class LlmClient {

    private final String apiUrl;
    private final String apiKey;
    private final String model;
    private final int maxOutputTokens;
    private final double temperature;
    private final String httpReferer;
    private final String title;
    private final boolean disableReasoning;
    /** Minimum gap between request starts, from LLM_REQUEST_INTERVAL_MS. 0 sends as fast as possible. */
    private final long minRequestIntervalMillis;
    /** When the previous request was sent, for {@link #awaitRequestSlot()}. */
    private long lastRequestStartMillis;

    private final HttpClient httpClient;
    private final ObjectMapper mapper = new ObjectMapper();

    /**
     * Constructs a new LLM client with the specified configuration.
     * 
     * @param apiUrl the base URL for the LLM API endpoint
     * @param apiKey the API key for authentication
    * @param model the model identifier resolved from the .env configuration
     * @param maxOutputTokens maximum number of tokens in the response
     * @param temperature sampling temperature (0.0-1.0) for response generation
     * @param httpReferer optional HTTP referer header value
     * @param title optional title header for request identification
     */
    // Najam: no live caller, and it could only supply a request interval by inventing one. Retired
    // rather than given a literal, for the same reason the .env fallbacks were removed.
    // public LlmClient(String apiUrl, String apiKey, String model, int maxOutputTokens,
    //                  double temperature, String httpReferer, String title) {
    //     this(apiUrl, apiKey, model, maxOutputTokens, temperature, httpReferer, title, false);
    // }

    /**
     * Constructs a new LLM client, optionally switching off the model's internal reasoning.
     *
     * @param apiUrl the base URL for the LLM API endpoint
     * @param apiKey the API key for authentication
     * @param model the model identifier resolved from the .env configuration
     * @param maxOutputTokens maximum number of tokens in the response
     * @param temperature sampling temperature (0.0-1.0) for response generation
     * @param httpReferer optional HTTP referer header value
     * @param title optional title header for request identification
     * @param disableReasoning true for a hybrid reasoning model, see {@link #buildPayload}
     * @param minRequestIntervalMillis minimum gap between requests, from LLM_REQUEST_INTERVAL_MS
     */
    public LlmClient(String apiUrl,
                     String apiKey,
                     String model,
                     int maxOutputTokens,
                     double temperature,
                     String httpReferer,
                     String title,
                     boolean disableReasoning,
                     long minRequestIntervalMillis) {
        if (apiUrl == null || apiUrl.isBlank()) {
            throw new IllegalArgumentException("apiUrl must not be null or blank");
        }
        if (apiKey == null || apiKey.isBlank()) {
            throw new IllegalArgumentException("apiKey must not be null or blank");
        }
        if (model == null || model.isBlank()) {
            throw new IllegalArgumentException("model must not be null or blank");
        }
        if (maxOutputTokens <= 0) {
            throw new IllegalArgumentException("maxOutputTokens must be positive");
        }
        if (temperature < 0.0 || temperature > 2.0) {
            throw new IllegalArgumentException("temperature must be between 0.0 and 2.0");
        }
        
        this.apiUrl = apiUrl;
        this.apiKey = apiKey;
        this.model = model;
        this.maxOutputTokens = maxOutputTokens;
        this.temperature = temperature;
        this.httpReferer = httpReferer;
        this.title = title;
        this.disableReasoning = disableReasoning;
        this.minRequestIntervalMillis = Math.max(0L, minRequestIntervalMillis);
        this.httpClient = HttpClient.newBuilder()
                .connectTimeout(Duration.ofSeconds(30))
                .build();
    }

    /**
     * Generates a summary by sending prompts to the LLM API.
     * <p>
     * Constructs a chat completion request with system and user messages,
     * sends it to the configured API endpoint, and extracts the generated text.
     * </p>
     * 
     * @param systemPrompt the system-level instruction that sets the behavior/role of the LLM
     * @param userPrompt the user message containing the content to summarize
     * @return an Optional containing the generated summary, or empty if no content returned
     * @throws LlmClientException if the API request fails or returns an error
     */
    // Backoff delays (seconds) between successive retries; length also determines max retry count.
    private static final int[] RETRY_DELAY_SECONDS = {15, 30, 60};

    /**
     * Whether an HTTP status is worth retrying.
     * <p>
     * 429 means rate limited. Any 5xx means the provider or the upstream model is briefly
     * unavailable — OpenRouter returns 502 and 503 routinely when it fails over between
     * providers, and a retry almost always succeeds. Every other 4xx reflects a problem with
     * the request itself and would fail identically however many times it is resent.
     * </p>
     *
     * @param statusCode the HTTP status returned by the API
     * @return true when resending the identical request is worth attempting
     */
    private static boolean isRetryableStatus(int statusCode) {
        return statusCode == 429 || statusCode >= 500;
    }

    /**
     * Sends a request, retrying transient failures with the configured backoff.
     * <p>
     * Only HTTP 429 used to be retried, so a single 502 from the provider or one dropped
     * connection permanently lost that class's summary: the exception unwound to the caller,
     * the run carried on with the next project, and the class was simply absent from the
     * output CSV with nothing to indicate it should have been there. Transport-level
     * IOExceptions (connection reset, read timeout) are now retried on the same schedule.
     * </p>
     *
     * @param request the fully built request to send
     * @return the final response, which may still carry a retryable status if every attempt failed
     * @throws IOException if the last attempt failed at the transport level
     * @throws InterruptedException if the backoff sleep is interrupted
     */
    /**
     * Waits until the configured interval has elapsed since the previous request started.
     * <p>
     * Najam: retrying was the only defence against HTTP 429, and it is the wrong one. A retry reacts
     * after the limit has already been hit, at 15/30/60 seconds a time; pacing keeps the run under
     * the limit so the 429 never happens. OpenRouter ties an account's request rate to its credit
     * balance, so a low balance throttles hard no matter how patiently the client retries.
     * </p>
     *
     * @throws InterruptedException if the wait is interrupted
     */
    private synchronized void awaitRequestSlot() throws InterruptedException {
        if (minRequestIntervalMillis <= 0) {
            return;
        }
        long waitFor = lastRequestStartMillis + minRequestIntervalMillis - System.currentTimeMillis();
        if (waitFor > 0) {
            Thread.sleep(waitFor);
        }
        lastRequestStartMillis = System.currentTimeMillis();
    }

    private HttpResponse<String> sendWithRetries(HttpRequest request)
            throws IOException, InterruptedException {
        HttpResponse<String> response = null;
        IOException lastTransportError = null;

        for (int attempt = 0; attempt <= RETRY_DELAY_SECONDS.length; attempt++) {
            if (attempt > 0) {
                int delay = RETRY_DELAY_SECONDS[attempt - 1];
                String cause = lastTransportError != null
                        ? lastTransportError.getClass().getSimpleName()
                        : "HTTP " + response.statusCode();
                System.out.printf("  Transient failure (%s). Waiting %ds before retry %d/%d...%n",
                        cause, delay, attempt, RETRY_DELAY_SECONDS.length);
                Thread.sleep(delay * 1000L);
            }

            try {
                awaitRequestSlot(); // Pace every attempt, retries included
                response = httpClient.send(request, HttpResponse.BodyHandlers.ofString());
                lastTransportError = null;
                if (!isRetryableStatus(response.statusCode())) {
                    return response;
                }
            } catch (IOException e) {
                response = null;
                lastTransportError = e;
            }
        }

        if (lastTransportError != null) {
            throw lastTransportError;
        }
        // Retries exhausted on a retryable status; the caller turns this into an error.
        return response;
    }

    public Optional<String> createSummary(String systemPrompt, String userPrompt) throws LlmClientException {
        if (systemPrompt == null) {
            throw new IllegalArgumentException("systemPrompt must not be null");
        }
        if (userPrompt == null) {
            throw new IllegalArgumentException("userPrompt must not be null");
        }

        try {
            String payload = buildPayload(systemPrompt, userPrompt);
            // Log the selected model before each request so one-off runs can verify the target model.
            System.out.println("Calling OpenRouter model: " + model);
            HttpRequest.Builder requestBuilder = HttpRequest.newBuilder()
                    .uri(URI.create(apiUrl))
                    .timeout(Duration.ofSeconds(60))
                    .header("Content-Type", "application/json")
                    .header("Authorization", "Bearer " + apiKey)
                    .POST(HttpRequest.BodyPublishers.ofString(payload));

            if (httpReferer != null && !httpReferer.isBlank()) {
                requestBuilder.header("HTTP-Referer", httpReferer);
            }
            if (title != null && !title.isBlank()) {
                requestBuilder.header("X-Title", title);
            }

            HttpRequest request = requestBuilder.build();
            HttpResponse<String> response = sendWithRetries(request);

            if (response.statusCode() < 200 || response.statusCode() >= 300) {
                throw new LlmClientException(String.format("LLM request failed (%d): %s", response.statusCode(), response.body()));
            }

            return extractMessage(response.body());
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new LlmClientException("LLM request interrupted", e);
        } catch (IOException e) {
            throw new LlmClientException("LLM request error: " + (e.getMessage() != null ? e.getMessage() : e.getClass().getSimpleName()), e);
        }
    }

    /**
     * Constructs the JSON payload for an LLM API request.
     * 
     * @param systemPrompt the system message
     * @param userPrompt the user message
     * @return JSON string representation of the request payload
     * @throws IOException if JSON serialization fails
     */
    private String buildPayload(String systemPrompt, String userPrompt) throws IOException {
        Map<String, Object> json = new HashMap<>();
        json.put("model", model);
        json.put("max_tokens", maxOutputTokens);
        json.put("temperature", temperature);

        // Hybrid reasoning models think before answering, and those reasoning tokens are
        // billed against max_tokens. qwen3.7-plus spent the entire budget on its chain of
        // thought and returned empty content with finish_reason=length for every class, so
        // the run produced no summaries at all. Raising the budget for Qwen alone was
        // rejected because it would give one model extended reasoning the other three do
        // not get, confounding the ranking and Friedman tests this corpus is built for.
        // The roster of affected models is LLM_NO_REASONING_MODELS in .env. PR_LLM and
        // SUMSLICE_LLM resolve this the same way (NO_REASONING_LABELS / NO_REASONING).
        if (disableReasoning) {
            json.put("reasoning", Map.of("enabled", false));
        }

        List<Map<String, String>> messages = new ArrayList<>();
        messages.add(createMessage("system", systemPrompt));
        messages.add(createMessage("user", userPrompt));
        json.put("messages", messages);

        return mapper.writeValueAsString(json);
    }

    /**
     * Creates a chat message map with role and content.
     * 
     * @param role the message role ("system", "user", or "assistant")
     * @param content the message content
     * @return a map representing the message
     */
    private Map<String, String> createMessage(String role, String content) {
        Map<String, String> message = new HashMap<>();
        message.put("role", role);
        message.put("content", content);
        return message;
    }

    /**
     * Extracts the text content from an LLM API response.
     * 
     * @param body the raw JSON response body
     * @return an Optional containing the extracted text, or empty if not found
     * @throws IOException if JSON parsing fails
     */
    private Optional<String> extractMessage(String body) throws IOException {
        JsonNode root = mapper.readTree(body);
        JsonNode choices = root.path("choices");
        if (!choices.isArray() || choices.isEmpty()) {
            return Optional.empty();
        }

        JsonNode choice = choices.get(0);
        JsonNode messageNode = choice.path("message");
        String content = messageNode.has("content") ? messageNode.path("content").asText().trim() : "";

        // DeepSeek returns the literal string "null" for near-empty prompts (sparse class data);
        // treat it as no content so it counts as a failed summary rather than polluting the CSV.
        if (content.isEmpty() || content.equalsIgnoreCase("null")) {
            warnIfReasoningExhaustedBudget(root, choice);
            return Optional.empty();
        }
        return Optional.of(content);
    }

    /**
     * Explains an empty completion when it was caused by reasoning consuming the token budget.
     * <p>
     * An empty answer from a hybrid reasoning model is indistinguishable from any other empty
     * answer in the output CSV, and diagnosing it previously meant inspecting a raw response
     * by hand. The usage block names the cause precisely, so it is reported here.
     * </p>
     */
    private void warnIfReasoningExhaustedBudget(JsonNode root, JsonNode choice) {
        if (!"length".equals(choice.path("finish_reason").asText())) {
            return;
        }
        int reasoningTokens = root.path("usage")
                .path("completion_tokens_details")
                .path("reasoning_tokens")
                .asInt(0);
        if (reasoningTokens <= 0 || disableReasoning) {
            return;
        }
        System.err.printf(
                "  WARNING: %s returned no content after spending %d of %d tokens on internal "
                        + "reasoning. Add its label to LLM_NO_REASONING_MODELS in .env rather than "
                        + "raising OPENROUTER_MAX_COMPLETION_TOKENS, which must stay equal across models.%n",
                model, reasoningTokens, maxOutputTokens);
    }
}

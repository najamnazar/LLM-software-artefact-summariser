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
    public LlmClient(String apiUrl,
                     String apiKey,
                     String model,
                     int maxOutputTokens,
                     double temperature,
                     String httpReferer,
                     String title) {
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

            HttpResponse<String> response = httpClient.send(requestBuilder.build(), HttpResponse.BodyHandlers.ofString());
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

        JsonNode messageNode = choices.get(0).path("message");
        if (!messageNode.has("content")) {
            return Optional.empty();
        }
        String content = messageNode.path("content").asText().trim();
        return content.isEmpty() ? Optional.empty() : Optional.of(content);
    }
}

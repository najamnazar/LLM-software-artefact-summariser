package dps_llm.prompt;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.IOException;
import java.io.InputStream;
import java.util.HashMap;
import java.util.Map;

/**
 * Manages loading and retrieval of prompts from the prompts.json configuration file.
 * <p>
 * This class provides centralized access to all LLM prompts used throughout the application.
 * Prompts are loaded once from the JSON configuration file and cached in memory for
 * efficient retrieval by alias.
 * </p>
 * <p>
 * Usage example:
 * <pre>
 * PromptManager manager = PromptManager.getInstance();
 * String systemPrompt = manager.getPrompt("SENIOR_ANALYST_CONCISE");
 * </pre>
 * </p>
 * 
 * @author Najam
 */
public class PromptManager {
    
    private static PromptManager instance;
    private final Map<String, String> promptCache;
    private final ObjectMapper objectMapper;
    
    /**
     * Private constructor to enforce singleton pattern.
     * Loads prompts from prompts.json on initialization.
     */
    private PromptManager() {
        this.objectMapper = new ObjectMapper();
        this.promptCache = new HashMap<>();
        loadPrompts();
    }
    
    /**
     * Gets the singleton instance of PromptManager.
     * 
     * @return the singleton instance
     */
    public static synchronized PromptManager getInstance() {
        if (instance == null) {
            instance = new PromptManager();
        }
        return instance;
    }
    
    /**
     * Loads all prompts from the prompts.json configuration file.
     * Prompts are indexed by their alias for quick retrieval.
     */
    private void loadPrompts() {
        try (InputStream is = getClass().getClassLoader().getResourceAsStream("prompts.json")) {
            if (is == null) {
                throw new IllegalStateException("prompts.json not found in resources");
            }
            
            JsonNode root = objectMapper.readTree(is);
            
            // Load LLM summarization prompts
            loadCategoryPrompts(root, "llm_summarization");
            // Najam, 2026-06-04: These four categories do not exist in prompts.json; each call no-ops silently
            // due to the null check in loadCategoryPrompts, but their presence misled PromptConfigurationExample
            // into referencing aliases (CODE_REVIEWER, TEST_ENGINEER, etc.) that were never defined.
            // Commented until the categories and their aliases are actually added to prompts.json.
            // loadCategoryPrompts(root, "code_review");
            // loadCategoryPrompts(root, "refactoring");
            // loadCategoryPrompts(root, "test_generation");
            // loadCategoryPrompts(root, "documentation");
            
        } catch (IOException e) {
            throw new IllegalStateException("Failed to load prompts.json", e);
        }
    }
    
    /**
     * Loads prompts from a specific category in the JSON configuration.
     * 
     * @param root the root JSON node
     * @param category the category name (e.g., "llm_summarization")
     */
    private void loadCategoryPrompts(JsonNode root, String category) {
        JsonNode categoryNode = root.get(category);
        if (categoryNode == null) {
            return;
        }
        
        // Load system_prompt
        JsonNode systemPromptNode = categoryNode.get("system_prompt");
        if (systemPromptNode != null && systemPromptNode.has("alias")) {
            String alias = systemPromptNode.get("alias").asText();
            String content = systemPromptNode.get("content").asText();
            promptCache.put(alias, content);
        }
        
        // Load alternative_prompts
        JsonNode alternativesNode = categoryNode.get("alternative_prompts");
        if (alternativesNode != null) {
            alternativesNode.fields().forEachRemaining(entry -> {
                JsonNode promptNode = entry.getValue();
                if (promptNode.has("alias")) {
                    String alias = promptNode.get("alias").asText();
                    String content = promptNode.get("content").asText();
                    promptCache.put(alias, content);
                }
            });
        }
    }
    
    /**
     * Retrieves a prompt by its alias.
     * 
     * @param alias the prompt alias (e.g., "SENIOR_ANALYST_CONCISE")
     * @return the prompt content
     * @throws IllegalArgumentException if the alias is not found
     */
    public String getPrompt(String alias) {
        if (alias == null || alias.trim().isEmpty()) {
            throw new IllegalArgumentException("Prompt alias cannot be null or empty");
        }
        
        String prompt = promptCache.get(alias);
        if (prompt == null) {
            throw new IllegalArgumentException("Prompt not found for alias: " + alias);
        }
        
        return prompt;
    }
    
    /**
     * Checks if a prompt exists for the given alias.
     * 
     * @param alias the prompt alias to check
     * @return true if the prompt exists, false otherwise
     */
    public boolean hasPrompt(String alias) {
        return promptCache.containsKey(alias);
    }
    
    /**
     * Gets all available prompt aliases.
     * 
     * @return a set of all prompt aliases
     */
    public java.util.Set<String> getAvailableAliases() {
        return new java.util.HashSet<>(promptCache.keySet());
    }
    
    /**
     * Reloads prompts from the configuration file.
     * Useful for hot-reloading during development.
     */
    public synchronized void reload() {
        promptCache.clear();
        loadPrompts();
    }
}

package dps_llm.prompt;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
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
    /** Section templates for the user prompt, read from dps_llm.llm_summarization.user_prompt_template. */
    private List<String> userPromptSections = List.of();
    /** Default word limit from the same block; null when prompts.json does not specify one. */
    private Integer userPromptWordLimit;
    /** Alias of llm_summarization.system_prompt: what a run uses when nothing selects a prompt. */
    private String defaultPromptAlias;
    /** Command-line shorthands from llm_summarization.cli_prompt_variants, keyed by normalised token. */
    private Map<String, PromptVariant> cliPromptVariants = Map.of();
    
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

            // prompts.json is shared across projects; scope lookups to this project's namespace.
            JsonNode dpsRoot = root.get("dps_llm");
            if (dpsRoot == null) {
                throw new IllegalStateException("dps_llm section not found in prompts.json");
            }

            // Load LLM summarization prompts
            loadCategoryPrompts(dpsRoot, "llm_summarization");
            // Najam: the user_prompt_template block existed in prompts.json but nothing ever read it.
            // The user prompt was assembled by a hardcoded StringBuilder, so editing prompts.json had
            // no effect on what the model actually received. It is loaded here and rendered by
            // LlmPromptBuilder.
            loadUserPromptTemplate(dpsRoot.get("llm_summarization"));
            // Najam: the choice between the concise and non-concise 50-word prompts used to be made by
            // editing DEFAULT_PROMPT_ALIAS and recompiling, with the unused alias left as a comment.
            // The shorthands live beside the prompts they name so a run can select either from argv.
            loadCliPromptVariants(dpsRoot.get("llm_summarization"));
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
            if ("llm_summarization".equals(category)) {
                defaultPromptAlias = alias;
            }
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
     * Loads the user prompt section templates and their default word limit.
     *
     * @param categoryNode the llm_summarization node, may be null
     */
    private void loadUserPromptTemplate(JsonNode categoryNode) {
        if (categoryNode == null) {
            return;
        }
        JsonNode templateNode = categoryNode.get("user_prompt_template");
        if (templateNode == null) {
            return;
        }

        JsonNode sectionsNode = templateNode.get("sections");
        if (sectionsNode != null && sectionsNode.isArray()) {
            List<String> sections = new ArrayList<>();
            for (JsonNode section : sectionsNode) {
                sections.add(section.asText());
            }
            userPromptSections = Collections.unmodifiableList(sections);
        }

        JsonNode wordLimitNode = templateNode.get("word_limit");
        if (wordLimitNode != null && wordLimitNode.isInt()) {
            userPromptWordLimit = wordLimitNode.asInt();
        }
    }

    /**
     * Loads the command-line prompt shorthands, e.g. concise -> SENIOR_ANALYST_50_WORDS.
     * <p>
     * Entries without an alias (such as the explanatory {@code _doc} note) are skipped, so
     * prompts.json can document the block in place.
     * </p>
     *
     * @param categoryNode the llm_summarization node, may be null
     */
    private void loadCliPromptVariants(JsonNode categoryNode) {
        if (categoryNode == null) {
            return;
        }
        JsonNode variantsNode = categoryNode.get("cli_prompt_variants");
        if (variantsNode == null) {
            return;
        }
        Map<String, PromptVariant> variants = new LinkedHashMap<>();
        variantsNode.fields().forEachRemaining(entry -> {
            JsonNode node = entry.getValue();
            if (node == null || !node.has("alias")) {
                return;
            }
            String alias = node.get("alias").asText();
            String suffix = node.has("output_suffix") ? node.get("output_suffix").asText() : "";
            variants.put(normaliseVariantKey(entry.getKey()), new PromptVariant(entry.getKey(), alias, suffix));
        });
        cliPromptVariants = Collections.unmodifiableMap(variants);
    }

    /** Lower-cases a shorthand and drops separators so concise, Concise and NON_CONCISE all match. */
    private static String normaliseVariantKey(String token) {
        return token == null ? "" : token.toLowerCase().replaceAll("[^a-z0-9]", "");
    }

    /**
     * Resolves a command-line shorthand to the prompt it names.
     *
     * @param token the argument as typed, e.g. "Concise"
     * @return the matching variant, or null when the token names no variant
     */
    public PromptVariant resolveCliPromptVariant(String token) {
        if (token == null || token.isBlank()) {
            return null;
        }
        return cliPromptVariants.get(normaliseVariantKey(token));
    }

    /**
     * Returns the variant whose alias is the given one, so an alias typed in full still
     * carries its output suffix.
     *
     * @param alias a full prompt alias
     * @return the matching variant, or null when no variant declares that alias
     */
    public PromptVariant findVariantByAlias(String alias) {
        if (alias == null) {
            return null;
        }
        for (PromptVariant variant : cliPromptVariants.values()) {
            if (variant.getAlias().equals(alias)) {
                return variant;
            }
        }
        return null;
    }

    /**
     * Returns the shorthands as declared in prompts.json, for usage messages.
     *
     * @return the declared variant names, in file order
     */
    public List<String> getCliPromptVariantNames() {
        List<String> names = new ArrayList<>();
        for (PromptVariant variant : cliPromptVariants.values()) {
            names.add(variant.getName());
        }
        return Collections.unmodifiableList(names);
    }

    /**
     * Returns the alias of llm_summarization.system_prompt, used when nothing selects a prompt.
     *
     * @return the default alias, or null when prompts.json declares no system prompt
     */
    public String getDefaultPromptAlias() {
        return defaultPromptAlias;
    }

    /** A command-line prompt shorthand: the name typed, the alias it selects, and its CSV suffix. */
    public static final class PromptVariant {
        private final String name;
        private final String alias;
        private final String outputSuffix;

        PromptVariant(String name, String alias, String outputSuffix) {
            this.name = name;
            this.alias = alias;
            this.outputSuffix = outputSuffix == null ? "" : outputSuffix;
        }

        /** @return the shorthand as spelled in prompts.json, e.g. "nonconcise" */
        public String getName() {
            return name;
        }

        /** @return the prompt alias this shorthand selects */
        public String getAlias() {
            return alias;
        }

        /** @return the infix placed after the model label in the CSV name, e.g. "_NC" */
        public String getOutputSuffix() {
            return outputSuffix;
        }
    }

    /**
     * Returns the user prompt section templates in the order they appear in prompts.json.
     *
     * @return the section templates, empty when prompts.json defines none
     */
    public List<String> getUserPromptSections() {
        return userPromptSections;
    }

    /**
     * Returns the default summary word limit declared in prompts.json.
     *
     * @return the configured limit, or null when prompts.json does not declare one
     */
    public Integer getUserPromptWordLimit() {
        return userPromptWordLimit;
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
        userPromptSections = List.of();
        userPromptWordLimit = null;
        defaultPromptAlias = null;
        cliPromptVariants = Map.of();
        loadPrompts();
    }
}

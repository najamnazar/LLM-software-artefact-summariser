package dps_llm.prompt;

import dps_llm.model.ClassFeatureSnapshot;

import java.util.List;
import java.util.StringJoiner;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Builds structured prompts for LLM-based code summarization.
 * <p>
 * This class constructs carefully formatted system and user prompts that guide the LLM
 * to generate concise, factual summaries of Java classes. The prompts include class
 * structure, relationships, design patterns, and behavioral context.
 * </p>
 * <p>
 * Key responsibilities:
 * <ul>
 *   <li>Generate consistent system prompts that define LLM behavior</li>
 *   <li>Format class features into structured user prompts</li>
 *   <li>Include relevant context like inheritance, methods, and patterns</li>
 *   <li>Enforce summary length and style constraints via prompting</li>
 *   <li>The summaries should be concise, factual, and objective</li>
 * </ul>
 * </p>
 * 
 * @author Najam
 */
public class LlmPromptBuilder {

    private static final Pattern WORD_LIMIT_PATTERN = Pattern.compile("(\\d+)_WORDS");
    private final PromptManager promptManager;
    private final String systemPromptAlias;
    // private final int wordLimit = 75; // Original fixed limit retained for reference
    private final int wordLimit; // Word limit inferred from prompt alias to keep summaries aligned with expectation
    
    /**
     * Constructs a new LlmPromptBuilder with default prompt alias.
     * Uses "SENIOR_ANALYST_50_WORDS" as the default system prompt.
     * Najam, 2026-06-04: Corrected Javadoc alias from "SENIOR_ANALYST_CONCISE" which does not exist in
     * prompts.json and would throw IllegalArgumentException if used directly via PromptManager.getPrompt().
     */
    public LlmPromptBuilder() {
        this("SENIOR_ANALYST_50_WORDS");
    }
    
    /**
     * Constructs a new LlmPromptBuilder with a specific prompt alias.
     * 
     * @param systemPromptAlias the alias of the system prompt to use
     */
    public LlmPromptBuilder(String systemPromptAlias) {
        this.promptManager = PromptManager.getInstance();
        this.systemPromptAlias = systemPromptAlias;
        this.wordLimit = resolveWordLimit(systemPromptAlias); // Capture numeric limit embedded in alias (e.g., 20/40/60/80)
    }

    /**
     * Returns the system prompt that configures LLM behavior.
     * <p>
     * The system prompt instructs the LLM to act as a senior software analyst
     * who writes concise, factual summaries. It enforces constraints on length,
     * tone, and content accuracy.
     * </p>
     * 
     * @return the system-level instruction prompt
     */
    public String getSystemPrompt() {
        return promptManager.getPrompt(systemPromptAlias);
    }

    /**
     * Builds a structured user prompt from a class feature snapshot.
     * <p>
     * The prompt includes:
     * <ul>
     *   <li>Project and class identification</li>
     *   <li>Class modifiers and type (class/interface)</li>
     *   <li>Inheritance and interface implementation</li>
     *   <li>Sample fields, constructors, and methods</li>
     *   <li>Method interaction patterns</li>
     *   <li>Design pattern insights</li>
     *   <li>Explicit task instructions</li>
     * </ul>
     * </p>
     * 
     * @param snapshot the class feature snapshot to convert into a prompt
     * @return formatted user prompt text
     * @throws IllegalArgumentException if snapshot is null
     */
    public String buildUserPrompt(ClassFeatureSnapshot snapshot) {
        if (snapshot == null) {
            throw new IllegalArgumentException("snapshot must not be null");
        }
        StringBuilder builder = new StringBuilder();
        builder.append("Project: ").append(snapshot.getProjectName()).append('\n');
        builder.append("Class: ").append(snapshot.getClassName()).append(" (" + snapshot.getClassKind());
        if (!snapshot.getModifiers().isEmpty()) {
            builder.append(", modifiers: ").append(String.join(" ", snapshot.getModifiers()));
        }
        builder.append(")\n");

        appendList(builder, "Extends", snapshot.getExtendsTypes());
        appendList(builder, "Implements", snapshot.getImplementsTypes());

        appendSection(builder, "Fields", snapshot.getFieldSignatures(), snapshot.getTotalFieldCount());
        appendSection(builder, "Constructors", snapshot.getConstructorSignatures(), snapshot.getTotalConstructorCount());
        appendSection(builder, "Methods", snapshot.getMethodSummaries(), snapshot.getTotalMethodCount());

        if (!snapshot.getInteractionNotes().isEmpty()) {
            builder.append("Interactions: ");
            builder.append(String.join(" | ", snapshot.getInteractionNotes()));
            builder.append('\n');
        }

        if (!snapshot.getPatternInsights().isEmpty()) {
            builder.append("Design pattern insights:\n");
            for (String insight : snapshot.getPatternInsights()) {
                builder.append("- ").append(insight).append('\n');
            }
        } else {
            builder.append("Design pattern insights: none captured in static analysis.\n");
        }

        // builder.append("\nTask: Produce a single-paragraph summary (<=75 words) that stresses the class responsibility, its collaborators, and the provided design-pattern context. Avoid repeating raw bullet text."); // Original instruction retained
        builder.append("\nTask: Produce a single-paragraph summary (<=").append(wordLimit)
            .append(" words) that stresses the class responsibility, its collaborators, and the provided design-pattern context. Avoid repeating raw bullet text.");
        return builder.toString();
    }

    /**
     * Appends a labeled list of values to the prompt builder.
     * 
     * @param builder the string builder to append to
     * @param label the label for the list (e.g., "Extends", "Implements")
     * @param values the list of values to append
     */
    private void appendList(StringBuilder builder, String label, List<String> values) {
        if (values == null || values.isEmpty()) {
            return;
        }
        builder.append(label).append(':').append(' ');
        builder.append(String.join(", ", values)).append('\n');
    }

    /**
     * Appends a section with samples and total count to the prompt.
     * 
     * @param builder the string builder to append to
     * @param label the section label (e.g., "Fields", "Methods")
     * @param samples the sample items to display
     * @param totalCount the total count of items in this category
     */
    private void appendSection(StringBuilder builder, String label, List<String> samples, int totalCount) {
        if (samples == null || samples.isEmpty()) {
            return;
        }
        StringJoiner joiner = new StringJoiner("; ");
        for (String sample : samples) {
            joiner.add(sample);
        }
        builder.append(label).append(" (total ").append(totalCount).append("): ").append(joiner).append('\n');
    }

    private int resolveWordLimit(String alias) {
        if (alias == null) {
            return 75;
        }
        Matcher matcher = WORD_LIMIT_PATTERN.matcher(alias);
        if (matcher.find()) {
            try {
                return Integer.parseInt(matcher.group(1));
            } catch (NumberFormatException ignored) {
                // Fall through to default when parsing fails
            }
        }
        return 75;
    }
}

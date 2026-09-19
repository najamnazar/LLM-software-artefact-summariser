package dps_llm.prompt;

import dps_llm.model.ClassFeatureSnapshot;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
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
    /** Matches {name} and {?name} in a section template from prompts.json. */
    private static final Pattern PLACEHOLDER_PATTERN = Pattern.compile("\\{(\\??)([a-z_]+)\\}");
    // Najam: FALLBACK_WORD_LIMIT is removed. prompts.json declares user_prompt_template.word_limit,
    // and a literal here silently shadowed it for any alias that encodes no number -- the same drift
    // the .env literals caused. An unresolvable limit is now a configuration error.
    // private static final int FALLBACK_WORD_LIMIT = 75;
    private final PromptManager promptManager;
    private final String systemPromptAlias;
    // private final int wordLimit = 75; // Original fixed limit retained for reference
    private final Integer wordLimit; // Word limit inferred from prompt alias to keep summaries aligned with expectation
    
    /**
     * Constructs a new LlmPromptBuilder using the prompt prompts.json declares as its system prompt.
     * <p>
     * Najam, 2026-06-04: Corrected Javadoc alias from "SENIOR_ANALYST_CONCISE" which does not exist in
     * prompts.json and would throw IllegalArgumentException if used directly via PromptManager.getPrompt().
     * The alias is no longer written here at all: a literal copy of it would go stale the moment
     * llm_summarization.system_prompt changed.
     * </p>
     */
    // public LlmPromptBuilder() { this("SENIOR_ANALYST_50_WORDS"); }
    public LlmPromptBuilder() {
        this(PromptManager.getInstance().getDefaultPromptAlias());
    }
    
    /**
     * Constructs a new LlmPromptBuilder with a specific prompt alias.
     * 
     * @param systemPromptAlias the alias of the system prompt to use
     */
    public LlmPromptBuilder(String systemPromptAlias) {
        this.promptManager = PromptManager.getInstance();
        this.systemPromptAlias = systemPromptAlias;
        this.wordLimit = resolveWordLimit(systemPromptAlias); // Alias-embedded limit (e.g., 20/40/60/80), else prompts.json
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
        // Najam: this method used to assemble the prompt from a hardcoded sequence of appends while
        // prompts.json carried a user_prompt_template block that nothing read. The section list in
        // prompts.json is now the definition of the prompt; the code only supplies the values.
        // There is deliberately no code fallback: one existed briefly, and it meant that dropping
        // `sections` from prompts.json would silently send the model a different prompt than the file
        // describes. The constructor rejects a missing section list instead.
        List<String> sections = promptManager.getUserPromptSections();

        Map<String, String> values = resolvePlaceholders(snapshot);
        StringBuilder builder = new StringBuilder();
        for (String section : sections) {
            String rendered = renderSection(section, values);
            if (rendered != null) {
                builder.append(rendered).append('\n');
            }
        }
        // The final section is the task line, which carries no trailing newline of its own.
        if (builder.length() > 0 && builder.charAt(builder.length() - 1) == '\n') {
            builder.setLength(builder.length() - 1);
        }
        return builder.toString();
    }

    /**
     * Supplies the value of every placeholder a section template may reference.
     *
     * @param snapshot the class feature snapshot being described
     * @return placeholder name to rendered value; an empty value means "absent"
     */
    private Map<String, String> resolvePlaceholders(ClassFeatureSnapshot snapshot) {
        Map<String, String> values = new LinkedHashMap<>();
        values.put("project_name", snapshot.getProjectName());
        values.put("class_name", snapshot.getClassName());
        values.put("class_kind", snapshot.getClassKind());
        // Pre-formatted with its separator so an absent modifier list leaves no dangling comma.
        values.put("modifiers", snapshot.getModifiers().isEmpty()
                ? "" : ", modifiers: " + String.join(" ", snapshot.getModifiers()));
        values.put("extends_types", String.join(", ", snapshot.getExtendsTypes()));
        values.put("implements_types", String.join(", ", snapshot.getImplementsTypes()));
        values.put("field_count", String.valueOf(snapshot.getTotalFieldCount()));
        values.put("field_signatures", joinSamples(snapshot.getFieldSignatures()));
        values.put("constructor_count", String.valueOf(snapshot.getTotalConstructorCount()));
        values.put("constructor_signatures", joinSamples(snapshot.getConstructorSignatures()));
        values.put("method_count", String.valueOf(snapshot.getTotalMethodCount()));
        values.put("method_summaries", joinSamples(snapshot.getMethodSummaries()));
        values.put("interaction_notes", String.join(" | ", snapshot.getInteractionNotes()));
        values.put("pattern_insights", formatPatternInsights(snapshot.getPatternInsights()));
        // {?word_limit} is optional, so an unspecified limit renders as empty and keeps the task line.
        values.put("word_limit", wordLimit == null ? "" : String.valueOf(wordLimit));
        return values;
    }

    /**
     * Renders one section template.
     * <p>
     * {@code {name}} is required: when its value is empty the whole section is omitted, which is how
     * Extends, Implements, Fields, Constructors, Methods and Interactions disappear for classes that
     * have none. {@code {?name}} is optional and never suppresses its section, which is how counts,
     * modifiers, the pattern-insight block and the word limit behave.
     * </p>
     *
     * @param template one entry of user_prompt_template.sections
     * @param values placeholder values from {@link #resolvePlaceholders}
     * @return the rendered line, or null when the section should be omitted
     */
    private String renderSection(String template, Map<String, String> values) {
        Matcher matcher = PLACEHOLDER_PATTERN.matcher(template);
        StringBuilder rendered = new StringBuilder();
        while (matcher.find()) {
            boolean optional = !matcher.group(1).isEmpty();
            String value = values.get(matcher.group(2));
            if (value == null) {
                value = "";
            }
            if (!optional && value.isEmpty()) {
                return null;
            }
            matcher.appendReplacement(rendered, Matcher.quoteReplacement(value));
        }
        matcher.appendTail(rendered);
        return rendered.toString();
    }

    private String joinSamples(List<String> samples) {
        if (samples == null || samples.isEmpty()) {
            return "";
        }
        StringJoiner joiner = new StringJoiner("; ");
        for (String sample : samples) {
            joiner.add(sample);
        }
        return joiner.toString();
    }

    /**
     * Formats the design-pattern insights as the bullet block that follows the section label, or the
     * explicit "none" note when static analysis captured nothing for this class.
     */
    private String formatPatternInsights(List<String> insights) {
        if (insights == null || insights.isEmpty()) {
            return " none captured in static analysis.";
        }
        StringBuilder block = new StringBuilder();
        for (String insight : insights) {
            block.append('\n').append("- ").append(insight);
        }
        return block.toString();
    }

    // Najam: the code-assembled prompt and its two helpers are retained below as a record only.
    // They duplicated every line of user_prompt_template.sections in Java, so the wording existed in
    // two places and only one of them was reviewable as configuration. Verified byte-identical to the
    // rendered template over all 150 classes before removal.
//     /**
//      * The original hardcoded prompt assembly, kept as the fallback for a prompts.json that declares
//      * no user_prompt_template.sections.
//      *
//      * @param snapshot the class feature snapshot to convert into a prompt
//      * @return formatted user prompt text
//      */
//     private String buildUserPromptFromCode(ClassFeatureSnapshot snapshot) {
//         StringBuilder builder = new StringBuilder();
//         builder.append("Project: ").append(snapshot.getProjectName()).append('\n');
//         builder.append("Class: ").append(snapshot.getClassName()).append(" (" + snapshot.getClassKind());
//         if (!snapshot.getModifiers().isEmpty()) {
//             builder.append(", modifiers: ").append(String.join(" ", snapshot.getModifiers()));
//         }
//         builder.append(")\n");
//
//         appendList(builder, "Extends", snapshot.getExtendsTypes());
//         appendList(builder, "Implements", snapshot.getImplementsTypes());
//
//         appendSection(builder, "Fields", snapshot.getFieldSignatures(), snapshot.getTotalFieldCount());
//         appendSection(builder, "Constructors", snapshot.getConstructorSignatures(), snapshot.getTotalConstructorCount());
//         appendSection(builder, "Methods", snapshot.getMethodSummaries(), snapshot.getTotalMethodCount());
//
//         if (!snapshot.getInteractionNotes().isEmpty()) {
//             builder.append("Interactions: ");
//             builder.append(String.join(" | ", snapshot.getInteractionNotes()));
//             builder.append('\n');
//         }
//
//         if (!snapshot.getPatternInsights().isEmpty()) {
//             builder.append("Design pattern insights:\n");
//             for (String insight : snapshot.getPatternInsights()) {
//                 builder.append("- ").append(insight).append('\n');
//             }
//         } else {
//             builder.append("Design pattern insights: none captured in static analysis.\n");
//         }
//
//         // builder.append("\nTask: Produce a single-paragraph summary (<=75 words) that stresses the class responsibility, its collaborators, and the provided design-pattern context. Avoid repeating raw bullet text."); // Original instruction retained
//         builder.append("\nTask: Produce a single-paragraph summary (<=").append(wordLimit)
//             .append(" words) that stresses the class responsibility, its collaborators, and the provided design-pattern context. Avoid repeating raw bullet text.");
//         return builder.toString();
//     }
//
//     /**
//      * Appends a labeled list of values to the prompt builder.
//      *
//      * @param builder the string builder to append to
//      * @param label the label for the list (e.g., "Extends", "Implements")
//      * @param values the list of values to append
//      */
//     private void appendList(StringBuilder builder, String label, List<String> values) {
//         if (values == null || values.isEmpty()) {
//             return;
//         }
//         builder.append(label).append(':').append(' ');
//         builder.append(String.join(", ", values)).append('\n');
//     }
//
//     /**
//      * Appends a section with samples and total count to the prompt.
//      *
//      * @param builder the string builder to append to
//      * @param label the section label (e.g., "Fields", "Methods")
//      * @param samples the sample items to display
//      * @param totalCount the total count of items in this category
//      */
//     private void appendSection(StringBuilder builder, String label, List<String> samples, int totalCount) {
//         if (samples == null || samples.isEmpty()) {
//             return;
//         }
//         StringJoiner joiner = new StringJoiner("; ");
//         for (String sample : samples) {
//             joiner.add(sample);
//         }
//         builder.append(label).append(" (total ").append(totalCount).append("): ").append(joiner).append('\n');
//     }
//

    /**
     * Resolves the summary word limit for this run.
     * <p>
     * The alias wins because it names the limit explicitly (SENIOR_ANALYST_50_WORDS), and the system
     * prompt that goes with it states the same number. prompts.json supplies the limit for aliases
     * that encode no number.
     * </p>
     *
     * @param alias the system prompt alias for this run
     * @return the word limit for the task line, or null when neither source specifies one
     */
    private Integer resolveWordLimit(String alias) {
        if (alias != null) {
            Matcher matcher = WORD_LIMIT_PATTERN.matcher(alias);
            if (matcher.find()) {
                return Integer.parseInt(matcher.group(1));
            }
        }
        return promptManager.getUserPromptWordLimit();
    }
}

package common.projectparser;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.ObjectWriter;
import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.core.util.DefaultIndenter;
import com.fasterxml.jackson.core.util.DefaultPrettyPrinter;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Reads and writes the JSON representation of a parsed project.
 * <p>
 * The JSON file is the hand-off point between parsing and summarisation. DPS_NLG, DPS_SWUM and
 * DPS_LLM each write their own copy of this representation and then generate their summaries by
 * reading it back, so every generator consumes the same documented artefact rather than an
 * in-memory structure that only exists inside one process.
 * </p>
 * <p>
 * Round-tripping through JSON loses Java collection identity: every JSON array deserialises to an
 * {@link ArrayList}, but the design-pattern detectors in {@code common.designpatternidentifier}
 * store some role values as {@link HashSet} and cast them back on read. {@link #read} therefore
 * restores the declared collection type for the design-pattern block before handing it to a
 * summariser. The mapping below mirrors the detectors: change one and change the other.
 * </p>
 *
 * @author Najam
 */
public final class ProjectJsonStore {

    /**
     * Role keys whose value the detectors write as a {@code HashSet<String>} and cast back to
     * {@code HashSet} when summarising:
     * target/adaptee (AdapterPattern), concrete_memento (MementoPattern),
     * concrete_element (VisitorPattern), concrete_observer (ObserverPattern).
     */
    private static final Set<String> SET_VALUED_ROLES = Set.of(
            "target", "adaptee", "concrete_memento", "concrete_element", "concrete_observer");

    /**
     * Role keys whose value is a {@code HashMap<String, HashSet<String>>}: publisher
     * (ObserverPattern), caretaker/originator (MementoPattern), visitor (VisitorPattern).
     * Every role not listed here or in {@link #SET_VALUED_ROLES} is list- or map-valued and is
     * left exactly as Jackson produced it (concrete_factory, concrete_product, abstract_product,
     * concrete_decorator, concrete_component, component).
     */
    private static final Set<String> MAP_OF_SET_VALUED_ROLES = Set.of(
            "publisher", "caretaker", "originator", "visitor");

    private ProjectJsonStore() {
    }

    /** Creates the writer used for every project JSON file, so all three pipelines format alike. */
    public static ObjectWriter newWriter() {
        return new ObjectMapper()
                .writer(new DefaultPrettyPrinter().withObjectIndenter(new DefaultIndenter("\t", "\n")));
    }

    /**
     * Writes a parsed project to {@code file}, creating parent directories as needed.
     *
     * @param writer the shared writer from {@link #newWriter()}
     * @param file the destination JSON file
     * @param parsedProject the parsed project payload
     * @throws IOException if the file cannot be written
     */
    public static void write(ObjectWriter writer, File file, Map<String, Object> parsedProject) throws IOException {
        if (writer == null) {
            throw new IllegalArgumentException("writer must not be null");
        }
        if (file == null) {
            throw new IllegalArgumentException("file must not be null");
        }
        if (parsedProject == null) {
            throw new IllegalArgumentException("parsedProject must not be null");
        }
        File parent = file.getParentFile();
        if (parent != null && !parent.exists() && !parent.mkdirs() && !parent.exists()) {
            throw new IOException("Failed to create output directory: " + parent.getAbsolutePath());
        }
        writer.writeValue(file, parsedProject);
    }

    /**
     * Reads a project JSON file back into the structure the summarisers expect.
     * <p>
     * The design-pattern block is normalised in place so that role values the detectors declare as
     * sets are sets again; without this a summariser would throw ClassCastException on the
     * ArrayList that Jackson produces for every JSON array.
     * </p>
     *
     * @param mapper the object mapper to read with
     * @param file the JSON file previously written by {@link #write}
     * @return the parsed project payload
     * @throws IOException if the file cannot be read or parsed
     */
    public static HashMap<String, Object> read(ObjectMapper mapper, File file) throws IOException {
        if (mapper == null) {
            throw new IllegalArgumentException("mapper must not be null");
        }
        if (file == null) {
            throw new IllegalArgumentException("file must not be null");
        }
        HashMap<String, Object> parsedProject =
                mapper.readValue(file, new TypeReference<HashMap<String, Object>>() { });
        parsedProject.put("design_pattern", normaliseDesignPatterns(parsedProject.get("design_pattern")));
        return parsedProject;
    }

    /**
     * Restores the collection types the design-pattern summarisers cast to.
     *
     * @param designPatternObj the raw {@code design_pattern} value straight out of Jackson
     * @return the same data with set-valued roles converted back to {@link HashSet}
     */
    @SuppressWarnings({"rawtypes", "unchecked"})
    public static ArrayList<HashMap> normaliseDesignPatterns(Object designPatternObj) {
        ArrayList<HashMap> normalised = new ArrayList<>();
        if (!(designPatternObj instanceof List)) {
            return normalised;
        }
        for (Object patternEntryObj : (List<?>) designPatternObj) {
            if (!(patternEntryObj instanceof Map)) {
                continue;
            }
            // { patternName -> { className -> { roleKey -> value } } }
            HashMap<String, Object> patternEntry = new HashMap<>((Map<String, Object>) patternEntryObj);
            for (Map.Entry<String, Object> pattern : patternEntry.entrySet()) {
                if (!(pattern.getValue() instanceof Map)) {
                    continue;
                }
                Map<String, Object> classMap = (Map<String, Object>) pattern.getValue();
                for (Map.Entry<String, Object> classEntry : classMap.entrySet()) {
                    if (!(classEntry.getValue() instanceof Map)) {
                        continue;
                    }
                    normaliseRoles((Map<String, Object>) classEntry.getValue());
                }
            }
            normalised.add(patternEntry);
        }
        return normalised;
    }

    @SuppressWarnings("unchecked")
    private static void normaliseRoles(Map<String, Object> roles) {
        for (Map.Entry<String, Object> role : roles.entrySet()) {
            String roleKey = role.getKey();
            Object value = role.getValue();
            if (SET_VALUED_ROLES.contains(roleKey) && value instanceof List) {
                role.setValue(toStringSet((List<?>) value));
            } else if (MAP_OF_SET_VALUED_ROLES.contains(roleKey) && value instanceof Map) {
                Map<String, Object> inner = (Map<String, Object>) value;
                for (Map.Entry<String, Object> innerEntry : inner.entrySet()) {
                    if (innerEntry.getValue() instanceof List) {
                        innerEntry.setValue(toStringSet((List<?>) innerEntry.getValue()));
                    }
                }
            }
        }
    }

    private static HashSet<String> toStringSet(List<?> values) {
        HashSet<String> set = new HashSet<>();
        for (Object value : values) {
            if (value != null) {
                set.add(String.valueOf(value));
            }
        }
        return set;
    }
}

package dps_llm.model;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Immutable snapshot of structural features extracted from a Java class.
 * <p>
 * This class encapsulates all relevant structural information about a Java class
 * that is used to generate LLM prompts and summaries. It includes class metadata,
 * inheritance relationships, field/method signatures, and design pattern insights.
 * </p>
 * <p>
 * Key responsibilities:
 * <ul>
 *   <li>Store class structural information in an immutable form</li>
 *   <li>Provide defensive copies of all collections</li>
 *   <li>Include both complete and truncated views of class members</li>
 *   <li>Capture design pattern context and inter-class relationships</li>
 * </ul>
 * </p>
 * 
 * @author Najam
 */
public final class ClassFeatureSnapshot {

    private final String projectName;
    private final String className;
    private final String sourceFile;
    private final String classKind;
    private final List<String> modifiers;
    private final List<String> extendsTypes;
    private final List<String> implementsTypes;
    private final List<String> fieldSignatures;
    private final int totalFieldCount;
    private final List<String> constructorSignatures;
    private final int totalConstructorCount;
    private final List<String> methodSummaries;
    private final int totalMethodCount;
    private final List<String> interactionNotes;
    private final List<String> patternInsights;

    /**
     * Constructs a new class feature snapshot with the specified structural information.
     * <p>
     * All collection parameters are defensively copied to ensure immutability.
     * </p>
     * 
     * @param projectName the project name this class belongs to
     * @param className the simple name of the class
     * @param sourceFile the source file name
     * @param classKind either "class" or "interface"
     * @param modifiers list of class modifiers (public, abstract, final, etc.)
     * @param extendsTypes list of parent class types
     * @param implementsTypes list of implemented interface types
     * @param fieldSignatures sample field signatures (may be truncated)
     * @param totalFieldCount total number of fields in the class
     * @param constructorSignatures sample constructor signatures (may be truncated)
     * @param totalConstructorCount total number of constructors
     * @param methodSummaries sample method summaries (may be truncated)
     * @param totalMethodCount total number of methods
     * @param interactionNotes notes about method call interactions
     * @param patternInsights design pattern insights from static analysis
     */
    public ClassFeatureSnapshot(
            String projectName,
            String className,
            String sourceFile,
            String classKind,
            List<String> modifiers,
            List<String> extendsTypes,
            List<String> implementsTypes,
            List<String> fieldSignatures,
            int totalFieldCount,
            List<String> constructorSignatures,
            int totalConstructorCount,
            List<String> methodSummaries,
            int totalMethodCount,
            List<String> interactionNotes,
            List<String> patternInsights) {
        if (projectName == null || projectName.isBlank()) {
            throw new IllegalArgumentException("projectName must not be null or blank");
        }
        if (className == null || className.isBlank()) {
            throw new IllegalArgumentException("className must not be null or blank");
        }
        if (sourceFile == null || sourceFile.isBlank()) {
            throw new IllegalArgumentException("sourceFile must not be null or blank");
        }
        if (classKind == null || classKind.isBlank()) {
            throw new IllegalArgumentException("classKind must not be null or blank");
        }
        if (totalFieldCount < 0) {
            throw new IllegalArgumentException("totalFieldCount must not be negative");
        }
        if (totalConstructorCount < 0) {
            throw new IllegalArgumentException("totalConstructorCount must not be negative");
        }
        if (totalMethodCount < 0) {
            throw new IllegalArgumentException("totalMethodCount must not be negative");
        }
        
        this.projectName = projectName;
        this.className = className;
        this.sourceFile = sourceFile;
        this.classKind = classKind;
        this.modifiers = copy(modifiers);
        this.extendsTypes = copy(extendsTypes);
        this.implementsTypes = copy(implementsTypes);
        this.fieldSignatures = copy(fieldSignatures);
        this.totalFieldCount = totalFieldCount;
        this.constructorSignatures = copy(constructorSignatures);
        this.totalConstructorCount = totalConstructorCount;
        this.methodSummaries = copy(methodSummaries);
        this.totalMethodCount = totalMethodCount;
        this.interactionNotes = copy(interactionNotes);
        this.patternInsights = copy(patternInsights);
    }

    private List<String> copy(List<String> source) {
        if (source == null || source.isEmpty()) {
            return Collections.emptyList();
        }
        return Collections.unmodifiableList(new ArrayList<>(source));
    }

    public String getProjectName() {
        return projectName;
    }

    public String getClassName() {
        return className;
    }

    public String getSourceFile() {
        return sourceFile;
    }

    public String getClassKind() {
        return classKind;
    }

    public List<String> getModifiers() {
        return modifiers;
    }

    public List<String> getExtendsTypes() {
        return extendsTypes;
    }

    public List<String> getImplementsTypes() {
        return implementsTypes;
    }

    public List<String> getFieldSignatures() {
        return fieldSignatures;
    }

    public int getTotalFieldCount() {
        return totalFieldCount;
    }

    public List<String> getConstructorSignatures() {
        return constructorSignatures;
    }

    public int getTotalConstructorCount() {
        return totalConstructorCount;
    }

    public List<String> getMethodSummaries() {
        return methodSummaries;
    }

    public int getTotalMethodCount() {
        return totalMethodCount;
    }

    public List<String> getInteractionNotes() {
        return interactionNotes;
    }

    public List<String> getPatternInsights() {
        return patternInsights;
    }
}

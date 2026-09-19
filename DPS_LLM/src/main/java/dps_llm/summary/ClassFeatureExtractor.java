package dps_llm.summary;

import common.utils.Utils;
import dps_llm.config.FeatureLimits;
import dps_llm.model.ClassFeatureSnapshot;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.stream.Collectors;

/**
 * Extracts and transforms raw DPS parsing output into compact feature snapshots.
 * <p>
 * This class processes the HashMap-based class data from the DPS parser and converts it
 * into structured ClassFeatureSnapshot objects that are easier to use for prompt generation.
 * It handles truncation of large feature sets and formats signatures for readability.
 * </p>
 * <p>
 * Key responsibilities:
 * <ul>
 *   <li>Transform raw parse data into structured snapshots</li>
 *   <li>Truncate large feature collections to configurable limits</li>
 *   <li>Format field, constructor, and method signatures</li>
 *   <li>Extract method interaction patterns (incoming/outgoing calls)</li>
 *   <li>Integrate design pattern insights</li>
 * </ul>
 * </p>
 * 
 * @author Najam
 */
@SuppressWarnings({"rawtypes", "unchecked"})
public class ClassFeatureExtractor {

    // Najam: these four caps were compiled-in constants, recorded nowhere but this file — not in
    // .env, not in prompts.json — so the amount of each class the model actually saw could not be
    // inspected or changed without a rebuild. They now arrive as configuration; FeatureLimits
    // defaults to the same 6/4/6/6 so existing runs are reproduced exactly.
    // private static final int FIELD_LIMIT = 6;
    // private static final int CONSTRUCTOR_LIMIT = 4;
    // private static final int METHOD_LIMIT = 6;
    // private static final int PATTERN_LIMIT = 6;

    private final FeatureLimits limits;

    // Najam: the no-argument constructor is gone with FeatureLimits.defaults(). It existed only to
    // supply the compiled-in 6/4/6/6, which is the copy of .env this change is removing. Callers
    // pass the limits resolved from .env.
    // public ClassFeatureExtractor() {
    //     this(FeatureLimits.defaults());
    // }

    /**
     * @param limits how much of each class reaches the prompt, resolved from .env
     */
    public ClassFeatureExtractor(FeatureLimits limits) {
        if (limits == null) {
            throw new IllegalArgumentException("limits must not be null");
        }
        this.limits = limits;
    }

    /**
     * Extracts a feature snapshot from raw class data.
     * <p>
     * Processes the parsed class data and pattern insights to create a compact,
     * structured snapshot suitable for LLM prompt generation.
     * </p>
     * 
     * @param projectName the project name
     * @param className the class name
     * @param classData the raw parsed class data from DPS
     * @param patternInsights design pattern insights for this class
     * @return an Optional containing the snapshot, or empty if extraction failed
     */
    public Optional<ClassFeatureSnapshot> extract(String projectName,
                                                  String className,
                                                  HashMap classData,
                                                  List<String> patternInsights) {
        if (projectName == null || projectName.isBlank()) {
            throw new IllegalArgumentException("projectName must not be null or blank");
        }
        if (className == null || className.isBlank()) {
            throw new IllegalArgumentException("className must not be null or blank");
        }
        if (classData == null) {
            return Optional.empty();
        }

        ArrayList<HashMap> classDetails = Utils.getClassOrInterfaceDetails(classData);
        if (classDetails == null || classDetails.isEmpty()) {
            return Optional.empty();
        }

        HashMap targetClassDetail = resolveClassDetail(classDetails, className);
        if (targetClassDetail == null) {
            return Optional.empty();
        }

        String kind = Utils.isInterfaceOrNot(targetClassDetail) ? "interface" : "class";
        List<String> modifiers = safeList(Utils.getClassModifierType(targetClassDetail));
        List<String> extendsTypes = safeList(Utils.getExtendsFrom(targetClassDetail));
        List<String> implementsTypes = safeList(Utils.getImplementsFrom(targetClassDetail));

        ArrayList<HashMap> fieldDetails = Utils.getFieldDetails(classData);
        ArrayList<HashMap> constructorDetails = Utils.getConstructorDetails(classData);
        ArrayList<HashMap> methodDetails = Utils.getMethodDetails(classData);

        List<String> fieldSignatures = truncate(formatFields(fieldDetails), limits.getFieldLimit());
        List<String> constructorSignatures = truncate(formatConstructors(constructorDetails, className), limits.getConstructorLimit());
        List<String> methodSummaries = truncate(formatMethods(methodDetails), limits.getMethodLimit());

        List<String> interactionNotes = buildInteractionNotes(methodDetails);
        List<String> limitedPatternInsights = truncate(patternInsights == null ? List.of() : patternInsights, limits.getPatternLimit());

        ClassFeatureSnapshot snapshot = new ClassFeatureSnapshot(
                projectName,
                className,
                className + ".java",
                kind,
                modifiers,
                extendsTypes,
                implementsTypes,
                fieldSignatures,
                fieldDetails == null ? 0 : fieldDetails.size(),
                constructorSignatures,
                constructorDetails == null ? 0 : constructorDetails.size(),
                methodSummaries,
                methodDetails == null ? 0 : methodDetails.size(),
                interactionNotes,
                limitedPatternInsights
        );
        return Optional.of(snapshot);
    }

    private HashMap resolveClassDetail(List<HashMap> classDetails, String className) {
        if (classDetails == null || classDetails.isEmpty()) {
            return null;
        }
        if (className == null) {
            return classDetails.get(0);
        }
        for (HashMap detail : classDetails) {
            Object name = detail.get("CLASSNAME");
            if (className.equals(name)) {
                return detail;
            }
        }
        return classDetails.get(0);
    }

    private List<String> formatFields(List<HashMap> fieldDetails) {
        if (fieldDetails == null || fieldDetails.isEmpty()) {
            return List.of();
        }
        List<String> signatures = new ArrayList<>();
        for (HashMap detail : fieldDetails) {
            String type = asString(detail.get("FIELDDATATYPE"));
            List<String> modifiers = safeList((List<String>) detail.get("FIELDMODIFIERTYPE"));
            List<String> declarations = safeList((List<String>) detail.get("FIELDDECLARATION"));
            if (type.isEmpty() && declarations.isEmpty()) {
                continue;
            }
            String name = declarations.isEmpty() ? "<unnamed>" : declarations.get(0);
            String signature = buildSignature(modifiers, type, name);
            signatures.add(signature);
        }
        return signatures;
    }

    private List<String> formatConstructors(List<HashMap> constructorDetails, String className) {
        if (constructorDetails == null || constructorDetails.isEmpty()) {
            return List.of();
        }
        List<String> signatures = new ArrayList<>();
        for (HashMap detail : constructorDetails) {
            String name = className == null || className.isEmpty() ? "<constructor>" : className;
            List<String> modifiers = safeList((List<String>) detail.get("CONSTRUCTORMODIFIER"));
            List<HashMap> params = safeList((List<HashMap>) detail.get("CONSTRUCTORPARAMETER"));
            String paramList = params.stream()
                    .map(this::formatParameter)
                    .collect(Collectors.joining(", "));
            String signature = buildSignature(modifiers, name + "(" + paramList + ")");
            signatures.add(signature);
        }
        return signatures;
    }

    private List<String> formatMethods(List<HashMap> methodDetails) {
        if (methodDetails == null || methodDetails.isEmpty()) {
            return List.of();
        }
        List<String> methods = new ArrayList<>();
        for (HashMap detail : methodDetails) {
            String name = Utils.getMethodName(detail);
            String returnType = Utils.getMethodReturnType(detail);
            List<String> modifiers = safeList(Utils.getMethodModifierType(detail));
            List<HashMap> params = safeList(Utils.getMethodParameters(detail));
            String paramList = params.stream().map(this::formatParameter).collect(Collectors.joining(", "));
            boolean overrides = Utils.isMethodOverride(detail);
            StringBuilder builder = new StringBuilder();
            if (!modifiers.isEmpty()) {
                builder.append(String.join(" ", modifiers)).append(' ');
            }
            builder.append(returnType == null || returnType.isEmpty() ? "void" : returnType);
            builder.append(' ').append(name).append('(').append(paramList).append(')');
            if (overrides) {
                builder.append(" [overrides]");
            }
            methods.add(builder.toString().trim());
        }
        return methods;
    }

    private List<String> buildInteractionNotes(List<HashMap> methodDetails) {
        if (methodDetails == null || methodDetails.isEmpty()) {
            return List.of();
        }
        Set<String> incoming = new LinkedHashSet<>();
        Set<String> outgoing = new LinkedHashSet<>();
        for (HashMap detail : methodDetails) {
            List<HashMap> incomingList = safeList(Utils.getIncomingMethod(detail));
            for (HashMap incomingEntry : incomingList) {
                String targetClass = Utils.getIncomingMethodClass(incomingEntry);
                String targetMethod = Utils.getIncomingMethodName(incomingEntry);
                if (!targetClass.isEmpty() && !targetMethod.isEmpty()) {
                    incoming.add(targetClass + "." + targetMethod);
                }
            }
            List<HashMap> outgoingList = safeList(Utils.getOutgoingMethod(detail));
            for (HashMap outgoingEntry : outgoingList) {
                String targetClass = Utils.getOutgoingMethodClass(outgoingEntry);
                String targetMethod = Utils.getOutgoingMethodName(outgoingEntry);
                if (!targetClass.isEmpty() && !targetMethod.isEmpty()) {
                    outgoing.add(targetClass + "." + targetMethod);
                }
            }
        }
        List<String> notes = new ArrayList<>();
        if (!incoming.isEmpty()) {
            notes.add("Incoming calls from: " + truncateJoin(incoming, 6));
        }
        if (!outgoing.isEmpty()) {
            notes.add("Outgoing calls to: " + truncateJoin(outgoing, 6));
        }
        return notes;
    }

    private String truncateJoin(Set<String> values, int limit) {
        List<String> list = new ArrayList<>(values);
        boolean truncated = list.size() > limit;
        if (truncated) {
            list = list.subList(0, limit);
        }
        String joined = String.join(", ", list);
        return truncated ? joined + ", ..." : joined;
    }

    private String formatParameter(HashMap parameter) {
        if (parameter == null) {
            return "?";
        }
        String type = Utils.getParameterType(parameter);
        String name = Utils.getParameterName(parameter);
        if (type.isEmpty()) {
            return name;
        }
        if (name.isEmpty()) {
            return type;
        }
        return type + " " + name;
    }

    private <T> List<T> safeList(List<T> data) {
        if (data == null) {
            return List.of();
        }
        return data;
    }

    private String asString(Object value) {
        return value == null ? "" : String.valueOf(value);
    }

    private List<String> truncate(List<String> source, int limit) {
        if (source == null || source.isEmpty()) {
            return List.of();
        }
        // A limit of 0 or less means "send everything", so the LLM can be given the same complete
        // class that the NLG and SWUM baselines describe.
        if (limit <= 0 || source.size() <= limit) {
            return List.copyOf(source);
        }
        List<String> truncated = new ArrayList<>(source.subList(0, limit));
        truncated.add("...");
        return truncated;
    }

    private String buildSignature(List<String> modifiers, String type, String name) {
        StringBuilder builder = new StringBuilder();
        if (modifiers != null && !modifiers.isEmpty()) {
            builder.append(String.join(" ", modifiers)).append(' ');
        }
        if (type != null && !type.isEmpty()) {
            builder.append(type).append(' ');
        }
        builder.append(name);
        return builder.toString().trim();
    }

    private String buildSignature(List<String> modifiers, String descriptor) {
        StringBuilder builder = new StringBuilder();
        if (modifiers != null && !modifiers.isEmpty()) {
            builder.append(String.join(" ", modifiers)).append(' ');
        }
        builder.append(descriptor);
        return builder.toString().trim();
    }
}

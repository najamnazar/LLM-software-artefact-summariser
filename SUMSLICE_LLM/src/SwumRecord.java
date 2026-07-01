import java.util.ArrayList;
import java.util.List;

public class SwumRecord {
    public int numericId;
    public String type;
    public String className;
    public String methodName;
    public String outputType;
    public String action;
    public String theme;
    public String id;
    public String parseId;

    public static SwumRecord fromMethodName(
            int numericId,
            String className,
            String methodName,
            String returnType,
            List<String> argumentTypes,
            boolean constructor) {
        SwumRecord record = new SwumRecord();
        record.numericId = numericId;
        record.className = className;
        record.methodName = methodName;
        record.outputType = constructor ? className : normalizeType(returnType);
        record.type = classify(methodName, constructor);

        List<String> words = splitCamelCase(methodName);
        record.action = detectAction(words, constructor);
        record.theme = detectTheme(words, record.action);

        String argsPart = String.join("-", normalizeTypes(argumentTypes));
        String output = constructor ? className : normalizeType(returnType);
        record.id = className + "_" + output + "_" + methodName + "__" + argsPart + "__";
        record.parseId = className.toLowerCase() + ":" + methodName;

        return record;
    }

    public String toOutLine() {
        StringBuilder sb = new StringBuilder();
        sb.append(type).append("::");
        sb.append(action).append(" (V)");
        if (theme != null && !theme.isBlank()) {
            sb.append(" | ").append(theme).append(" (N)");
        }
        sb.append(" ++ :: ").append(outputType).append(" (N)");
        sb.append(" :: ").append(className.toLowerCase()).append(" (N)");
        sb.append(" ::").append(id);
        return sb.toString();
    }

    private static String classify(String methodName, boolean constructor) {
        if (constructor) {
            return "CONSTRUCTOR";
        }
        if (methodName.startsWith("get") || methodName.startsWith("set") || methodName.startsWith("is")) {
            return "SPECIAL";
        }
        return "BASE_VERB";
    }

    private static String detectAction(List<String> words, boolean constructor) {
        if (constructor) {
            return "create";
        }
        if (words.isEmpty()) {
            return "do";
        }
        return words.get(0).toLowerCase();
    }

    private static String detectTheme(List<String> words, String action) {
        if (words.size() <= 1) {
            return "value";
        }
        List<String> themeWords = new ArrayList<>();
        for (int i = 1; i < words.size(); i++) {
            String lowered = words.get(i).toLowerCase();
            if (!lowered.equals(action)) {
                themeWords.add(lowered);
            }
        }
        if (themeWords.isEmpty()) {
            return "value";
        }
        return String.join(" ", themeWords);
    }

    private static String normalizeType(String type) {
        if (type == null || type.isBlank()) {
            return "void";
        }
        return type.replace("[]", "Array").replace("<", "").replace(">", "").replace("?", "").trim();
    }

    private static List<String> normalizeTypes(List<String> types) {
        List<String> normalized = new ArrayList<>();
        for (String type : types) {
            normalized.add(normalizeType(type));
        }
        return normalized;
    }

    private static List<String> splitCamelCase(String input) {
        List<String> words = new ArrayList<>();
        if (input == null || input.isBlank()) {
            return words;
        }
        String split = input.replaceAll("([a-z])([A-Z])", "$1 $2").replace('_', ' ');
        for (String token : split.split("\\s+")) {
            if (!token.isBlank()) {
                words.add(token);
            }
        }
        return words;
    }
}

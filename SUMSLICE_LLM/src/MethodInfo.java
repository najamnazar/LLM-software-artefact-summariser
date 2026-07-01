import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;

public class MethodInfo {
    private static final int MAX_CALL_METHODS = 10;

    public int id;
    public String swumId;
    public String parseId;
    public String methodName;
    public String className;
    public String returnType = "void";
    public String swumVerb = "unknown";
    public String swumObject = "unknown";
    public String useType = "unknown";
    public String useExample = "unknown";
    public String methodBody;
    public boolean constructor;
    public final List<String> argumentTypes = new ArrayList<>();

    public final Set<Integer> calls = new LinkedHashSet<>();
    public final Set<Integer> calledBy = new LinkedHashSet<>();

    public void addCall(int targetId) {
        calls.add(targetId);
    }

    public void addCalledBy(int callerId) {
        calledBy.add(callerId);
    }

    public List<Integer> getTopCalls() {
        return takeTop(calls, MAX_CALL_METHODS);
    }

    public List<Integer> getTopCalledBy() {
        return takeTop(calledBy, MAX_CALL_METHODS);
    }

    private List<Integer> takeTop(Set<Integer> source, int max) {
        List<Integer> list = new ArrayList<>(source);
        if (list.size() <= max) {
            return list;
        }
        return list.subList(0, max);
    }
}

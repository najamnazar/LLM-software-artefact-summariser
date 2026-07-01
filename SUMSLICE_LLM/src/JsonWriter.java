import java.io.IOException;
import java.io.Writer;
import java.util.Iterator;
import java.util.List;

public class JsonWriter {

    public void writeMethodList(Writer writer, List<MethodInfo> methods, int avgCalled, int avgCalls) throws IOException {
        writer.write("{\n");
        writer.write("  \"method_list\": {\n");
        writer.write("    \"averages\": {\n");
        writer.write("      \"called\": " + avgCalled + ",\n");
        writer.write("      \"calls\": " + avgCalls + "\n");
        writer.write("    },\n");
        writer.write("    \"method\": [\n");

        for (int i = 0; i < methods.size(); i++) {
            MethodInfo method = methods.get(i);
            writeMethod(writer, method);
            if (i < methods.size() - 1) {
                writer.write(",");
            }
            writer.write("\n");
        }

        writer.write("    ]\n");
        writer.write("  }\n");
        writer.write("}\n");
    }

    private void writeMethod(Writer writer, MethodInfo method) throws IOException {
        writer.write("      {\n");
        writer.write("        \"id\": " + method.id + ",\n");
        writer.write("        \"name\": \"" + escape(method.methodName) + "\",\n");
        writer.write("        \"class\": \"" + escape(method.className) + "\",\n");
        writer.write("        \"returntype\": \"" + escape(method.returnType) + "\",\n");
        writer.write("        \"called\": ");
        writeIntArray(writer, method.getTopCalledBy());
        writer.write(",\n");
        writer.write("        \"calls\": ");
        writeIntArray(writer, method.getTopCalls());
        writer.write(",\n");
        writer.write("        \"use\": {\n");
        writer.write("          \"type\": \"" + escape(method.useType) + "\",\n");
        writer.write("          \"example\": \"" + escape(method.useExample) + "\"\n");
        writer.write("        },\n");
        writer.write("        \"swum\": {\n");
        writer.write("          \"object\": \"" + escape(method.swumObject) + "\",\n");
        writer.write("          \"verb\": \"" + escape(method.swumVerb) + "\"\n");
        writer.write("        }\n");
        writer.write("      }");
    }

    private void writeIntArray(Writer writer, List<Integer> values) throws IOException {
        writer.write("[");
        Iterator<Integer> it = values.iterator();
        while (it.hasNext()) {
            writer.write(String.valueOf(it.next()));
            if (it.hasNext()) {
                writer.write(", ");
            }
        }
        writer.write("]");
    }

    private String escape(String value) {
        if (value == null) {
            return "";
        }
        return value
                .replace("\\", "\\\\")
                .replace("\"", "\\\"")
                .replace("\n", "\\n")
                .replace("\r", "\\r")
                .replace("\t", "\\t");
    }
}

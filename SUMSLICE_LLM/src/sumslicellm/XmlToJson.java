package sumslicellm;

import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.w3c.dom.Node;
import org.w3c.dom.NodeList;

import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/**
 * Step 1. Convert the original SumSlice input files (conf/&lt;project&gt;-example.xml)
 * into JSON, one file per project, without changing any value.
 *
 * The XML is read straight from the original SumSlice release
 * (../ORIGINAL_SUMSLICE/sumslice/conf), so this project keeps no second copy of
 * it; only the JSON form is stored. The filter on *-example.xml selects the six
 * project files and skips conf's default-lexicon.xml, example.xml and
 * example2.xml.
 *
 * Usage: java sumslicellm.XmlToJson [xmlDir] [jsonDir]
 *        defaults: ../ORIGINAL_SUMSLICE/sumslice/conf  input/dataset/methods
 *
 * Output: input/dataset/methods/&lt;project&gt;.json
 * {
 *   "project": "nanoXML", "source_xml": "nanoXML-example.xml",
 *   "averages": {"called": 1, "calls": 1, "pagerank": 0.0035},
 *   "methods": [ {"id", "name", "class", "returntype", "called": [ids],
 *                 "calls": [ids], "use": {"type", "example"},
 *                 "pagerank", "swum": {"object", "verb"}} ... ]
 * }
 * All entries are kept exactly as in the XML (including duplicates and id 0);
 * cleaning happens when the JSON is loaded (see ProjectData).
 */
public final class XmlToJson {

    public static void main(String[] args) throws Exception {
        Path xmlDir = Paths.get(args.length > 0 ? args[0] : "../ORIGINAL_SUMSLICE/sumslice/conf");
        Path jsonDir = Paths.get(args.length > 1 ? args[1] : "input/dataset/methods");
        List<Path> files;
        try (Stream<Path> s = Files.list(xmlDir)) {
            files = s.filter(p -> p.getFileName().toString().endsWith("-example.xml"))
                     .sorted().collect(Collectors.toList());
        }
        if (files.isEmpty()) {
            throw new IllegalStateException("No *-example.xml files in " + xmlDir.toAbsolutePath());
        }
        DocumentBuilder builder = DocumentBuilderFactory.newInstance().newDocumentBuilder();
        for (Path xml : files) {
            String project = xml.getFileName().toString().replace("-example.xml", "");
            Document doc = builder.parse(xml.toFile());
            Element root = doc.getDocumentElement();

            Map<String, Object> out = new LinkedHashMap<>();
            out.put("project", project);
            out.put("source_xml", xml.getFileName().toString());
            Element avg = firstChild(root, "averages");
            Map<String, Object> averages = new LinkedHashMap<>();
            averages.put("called", Long.parseLong(text(avg, "called").trim()));
            averages.put("calls", Long.parseLong(text(avg, "calls").trim()));
            averages.put("pagerank", Double.parseDouble(text(avg, "pagerank").trim()));
            out.put("averages", averages);

            List<Object> methods = new ArrayList<>();
            for (Element m : children(root, "method")) {
                Map<String, Object> rec = new LinkedHashMap<>();
                rec.put("id", Long.parseLong(text(m, "id").trim()));
                rec.put("name", text(m, "name"));
                rec.put("class", text(m, "class"));
                rec.put("returntype", text(m, "returntype"));
                rec.put("called", ids(firstChild(m, "called")));
                rec.put("calls", ids(firstChild(m, "calls")));
                Element use = firstChild(m, "use");
                if (use != null) {
                    Map<String, Object> u = new LinkedHashMap<>();
                    u.put("type", text(use, "type"));
                    u.put("example", text(use, "example"));
                    rec.put("use", u);
                }
                String pr = text(m, "pagerank");
                rec.put("pagerank", pr == null ? null : Double.parseDouble(pr.trim()));
                Element swum = firstChild(m, "swum");
                if (swum != null) {
                    Map<String, Object> sw = new LinkedHashMap<>();
                    sw.put("object", text(swum, "object"));
                    sw.put("verb", text(swum, "verb"));
                    rec.put("swum", sw);
                }
                methods.add(rec);
            }
            out.put("methods", methods);
            Path target = jsonDir.resolve(project + ".json");
            Json.writeFile(target, out);
            System.out.printf("%-10s %6d method entries -> %s%n", project, methods.size(), target);
        }
    }

    private static List<Element> children(Element parent, String tag) {
        List<Element> out = new ArrayList<>();
        if (parent == null) {
            return out;
        }
        NodeList nodes = parent.getChildNodes();
        for (int i = 0; i < nodes.getLength(); i++) {
            Node n = nodes.item(i);
            if (n instanceof Element && ((Element) n).getTagName().equals(tag)) {
                out.add((Element) n);
            }
        }
        return out;
    }

    private static Element firstChild(Element parent, String tag) {
        List<Element> c = children(parent, tag);
        return c.isEmpty() ? null : c.get(0);
    }

    /** Text content exactly as SumSlice's Xml.content() returns it (not trimmed). */
    private static String text(Element parent, String tag) {
        Element e = firstChild(parent, tag);
        return e == null ? null : e.getTextContent();
    }

    private static List<Object> ids(Element list) {
        List<Object> out = new ArrayList<>();
        for (Element id : children(list, "id")) {
            out.add(Long.parseLong(id.getTextContent().trim()));
        }
        return out;
    }
}

package sumslicellm;

import sumslicellm.ProjectData.MethodRecord;
import sumslicellm.messages.CalledMessage;
import sumslicellm.messages.ImportanceMessage;
import sumslicellm.messages.OutputUsedMessage;
import sumslicellm.messages.QuickSummaryMessage;
import sumslicellm.messages.ReturnMessage;
import sumslicellm.messages.UseMessage;

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
 * Step 2. Run SumSlice on every project, select the sample, and write both the
 * tool summaries and the matching LLM input.
 *
 * For each project the complete method list is planned and realised (so call
 * counts, callers and averages come from the whole project). The TOP_N methods
 * with the highest PageRank (ties: lower id first) form the sample.
 *
 * Usage: java sumslicellm.GenerateToolSummaries [jsonDir] [lexicon] [topN] [datasetDir]
 *        defaults: input/dataset/methods
 *                  ../ORIGINAL_SUMSLICE/sumslice/conf/default-lexicon.xml
 *                  25  input/dataset/projects
 *
 * The tool summaries are produced from the SumSlice messages alone, exactly as
 * the original does. The LLM input additionally carries the method's real
 * source code, read from the project sources under {@code datasetDir}.
 *
 * Outputs
 *   input/tool_summaries/SUMSLICE_TOOL_SUMMARY.jsonl
 *                                         tool (reference) summaries, one per sample method
 *   input/dataset/selected_methods.jsonl  the SumSlice messages of each sample method
 *                                         plus its declaration in the project source
 *                                         (the LLM input)
 *   input/dataset/selection_report.json   cleaning and selection details per project
 *   input/tool_summaries/all/&lt;project&gt;.jsonl
 *                                         tool summaries for every method (for reference)
 */
public final class GenerateToolSummaries {

    public static void main(String[] args) throws Exception {
        Path jsonDir = Paths.get(args.length > 0 ? args[0] : "input/dataset/methods");
        Path lexicon = Paths.get(args.length > 1 ? args[1] : "../ORIGINAL_SUMSLICE/sumslice/conf/default-lexicon.xml");
        int topN = args.length > 2 ? Integer.parseInt(args[2]) : 25;
        Path datasetDir = Paths.get(args.length > 3 ? args[3] : "input/dataset/projects");
        SourceIndex.Cache sources = new SourceIndex.Cache(datasetDir);

        List<Path> files;
        try (Stream<Path> s = Files.list(jsonDir)) {
            files = s.filter(p -> p.toString().endsWith(".json")).sorted().collect(Collectors.toList());
        }
        if (files.isEmpty()) {
            throw new IllegalStateException("No project JSON in " + jsonDir.toAbsolutePath() + "; run XmlToJson first.");
        }

        List<Map<String, Object>> toolRows = new ArrayList<>();
        List<Map<String, Object>> inputRows = new ArrayList<>();
        Map<String, Object> report = new LinkedHashMap<>();
        report.put("top_n", topN);
        report.put("selection", "highest PageRank per project; ties broken by lower method id");

        for (Path file : files) {
            ProjectData data = ProjectData.load(file);
            long t0 = System.currentTimeMillis();

            DocumentPlanner dp = new DocumentPlanner(data);
            dp.generateMessages();
            dp.createDocumentPlan();
            MicroPlanner mp = new MicroPlanner(lexicon.toString());
            mp.lexicalize(dp.getDocumentPlan());
            mp.aggregate();

            List<Map<String, Object>> all = new ArrayList<>();
            for (MethodRecord m : data.methods) {
                String summary = mp.realiseParagraph((int) m.id);
                all.add(row(data, m, -1, summary));
            }
            Json.writeJsonl(Paths.get("input", "tool_summaries", "all", data.project + ".jsonl"), all);

            SourceIndex index = sources.get(data.project);
            List<MethodRecord> ranked = data.rankedByPagerank();
            List<Map<String, Object>> selected = new ArrayList<>();
            int withSource = 0;
            int ambiguous = 0;
            for (int r = 0; r < ranked.size() && selected.size() < topN; r++) {
                MethodRecord m = ranked.get(r);
                String summary = mp.realiseParagraph((int) m.id);
                if (summary == null || summary.isBlank()) {
                    continue;   // cannot happen with SumSlice's messages, kept as a guard
                }
                toolRows.add(row(data, m, r + 1, summary));
                Map<String, Object> f = facts(data, dp, m, r + 1);
                attachSource(f, index, m);
                if (Boolean.TRUE.equals(f.get("source_found"))) {
                    withSource++;
                }
                if (((Number) f.get("source_overloads")).intValue() > 1) {
                    ambiguous++;
                }
                inputRows.add(f);
                selected.add(Map.of("method_id", m.id, "rank", r + 1));
            }

            Map<String, Object> proj = new LinkedHashMap<>(data.report);
            proj.put("selected", selected.size());
            proj.put("source_dir", index.root().toString());
            proj.put("source_files_indexed", index.fileCount());
            proj.put("selected_with_source", withSource);
            proj.put("selected_with_overloads", ambiguous);
            proj.put("pagerank_of_last_selected", ranked.get(Math.min(topN, ranked.size()) - 1).pagerank);
            proj.put("seconds", (System.currentTimeMillis() - t0) / 1000.0);
            report.put(data.project, proj);
            System.out.printf("%-10s methods %5d  selected %d  source %d/%d (%d overloaded)  pagerank: %s  (%.1fs)%n",
                    data.project, data.methods.size(), selected.size(), withSource, selected.size(),
                    ambiguous, data.pagerankSource, (System.currentTimeMillis() - t0) / 1000.0);
        }

        Json.writeJsonl(Paths.get("input", "tool_summaries", "SUMSLICE_TOOL_SUMMARY.jsonl"), toolRows);
        Json.writeJsonl(Paths.get("input", "dataset", "selected_methods.jsonl"), inputRows);
        Json.writeFile(Paths.get("input", "dataset", "selection_report.json"), report);
        System.out.println("Wrote " + toolRows.size() + " tool summaries and LLM inputs.");
    }

    private static Map<String, Object> row(ProjectData data, MethodRecord m, int rank, String summary) {
        Map<String, Object> r = new LinkedHashMap<>();
        r.put("project", data.project);
        r.put("method_id", m.id);
        r.put("class", m.className);
        r.put("name", m.name);
        if (rank > 0) {
            r.put("pagerank_rank", rank);
        }
        r.put("system", "TOOL");
        r.put("model", "SumSlice (McBurney & McMillan), ported");
        r.put("summary", summary);
        return r;
    }

    /** The SumSlice messages of one method, as structured facts for the LLM. */
    static Map<String, Object> facts(ProjectData data, DocumentPlanner dp, MethodRecord m, int rank) {
        int id = (int) m.id;
        Map<String, Object> f = new LinkedHashMap<>();
        f.put("project", data.project);
        f.put("method_id", m.id);
        f.put("class", m.className);
        f.put("name", m.name);
        f.put("pagerank_rank", rank);

        QuickSummaryMessage qs = (QuickSummaryMessage) dp.getMessage(id, QuickSummaryMessage.class);
        f.put("swum_verb", qs == null ? null : qs.getVerb());
        f.put("swum_object", qs == null ? null : qs.getObject());

        ReturnMessage rm = (ReturnMessage) dp.getMessage(id, ReturnMessage.class);
        f.put("return_type", rm == null ? null : rm.getReturnType());

        // the original drops "output used" phrases whose verb or object is "null"
        List<Object> usedBy = new ArrayList<>();
        for (Object o : dp.getMessages(id, OutputUsedMessage.class)) {
            OutputUsedMessage oum = (OutputUsedMessage) o;
            if (!"null".equals(oum.getVP()) && !"null".equals(oum.getNP())) {
                Map<String, Object> u = new LinkedHashMap<>();
                u.put("verb", oum.getVP());
                u.put("object", oum.getNP());
                usedBy.add(u);
            }
        }
        f.put("output_used_by", usedBy);

        CalledMessage cm = (CalledMessage) dp.getMessage(id, CalledMessage.class);
        f.put("called_count", cm == null ? 0 : cm.getCalledCount());
        f.put("calls_count", m.calls.size());

        ImportanceMessage im = (ImportanceMessage) dp.getMessage(id, ImportanceMessage.class);
        f.put("pagerank", im.getPagerank());
        f.put("avg_pagerank", im.getAvgPagerank());
        f.put("importance", importanceLevel(im));

        // the original drops the use sentence when the example is "unknown"
        UseMessage um = (UseMessage) dp.getMessage(id, UseMessage.class);
        boolean useKnown = um != null && !um.getExample().equals("unknown");
        f.put("use_type", useKnown ? um.getType() : null);
        f.put("use_example", useKnown ? um.getExample() : null);
        return f;
    }

    /**
     * Adds the method's declaration in the project source to its facts. The
     * class header comes from the same file. Methods that cannot be resolved
     * keep explicit "not found" values, so the prompt is still complete and
     * the gap is visible in the input file.
     */
    static void attachSource(Map<String, Object> f, SourceIndex index, MethodRecord m) throws Exception {
        SourceIndex.ClassSource cls = index.classHeader(m.className);
        f.put("class_extends", cls.found ? cls.extendsType : null);
        f.put("class_implements", cls.found ? cls.implementsTypes : null);

        SourceIndex.MethodSource src = index.method(m.className, m.name, m.returnType);
        f.put("source_found", src.found);
        f.put("source_file", src.found ? src.file : cls.file);
        f.put("source_overloads", src.overloads);
        f.put("parameters", src.found ? src.parameters : null);
        f.put("signature", src.found ? src.signature : null);
        f.put("body", src.found ? src.body : null);
    }

    /** Same thresholds as MicroPlanner.handleMessage(ImportanceMessage). */
    static String importanceLevel(ImportanceMessage im) {
        if (im.getPagerank() > 1.50 * im.getAvgPagerank()) {
            return "far more important than average";
        } else if (im.getPagerank() > im.getAvgPagerank()) {
            return "slightly more important than average";
        }
        return "less important than average";
    }
}

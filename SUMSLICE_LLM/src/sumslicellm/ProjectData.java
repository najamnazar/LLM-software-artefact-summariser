package sumslicellm;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

/**
 * One project's method list, read from input/dataset/methods/&lt;project&gt;.json (converted
 * one-to-one from the original SumSlice conf/&lt;project&gt;-example.xml).
 *
 * Cleaning applied on load (recorded in {@link #report}):
 *  - exact duplicate method entries (same id, identical content) are kept once;
 *  - ids shared by different methods are dropped. In every original XML this is
 *    id 0, the catch-all for methods the original tooling could not parse;
 *  - call/caller references to dropped or unknown ids are removed.
 *
 * PageRank: if every method in the file has PageRank 0 (jajuk and jhotdraw in
 * the original release), PageRank is recomputed on the file's own call graph
 * (damping 0.85, 100 iterations) and the average is recomputed accordingly.
 */
public final class ProjectData {

    public static final class MethodRecord {
        public final long id;
        public final String name;
        public final String className;
        public final String returnType;
        public final List<Long> called;   // callers, in file order
        public final List<Long> calls;    // callees, in file order
        public final String useType;
        public final String useExample;
        public double pagerank;
        public final String swumVerb;
        public final String swumObject;

        MethodRecord(Map<String, Object> m) {
            id = Json.lng(m, "id");
            name = Json.str(m, "name");
            className = Json.str(m, "class");
            returnType = Json.str(m, "returntype");
            called = longs(m.get("called"));
            calls = longs(m.get("calls"));
            Map<String, Object> use = m.get("use") == null ? Map.of() : Json.obj(m.get("use"));
            useType = Json.str(use, "type");
            useExample = Json.str(use, "example");
            pagerank = Json.dbl(m, "pagerank");
            Map<String, Object> swum = m.get("swum") == null ? Map.of() : Json.obj(m.get("swum"));
            swumVerb = Json.str(swum, "verb");
            swumObject = Json.str(swum, "object");
        }

        private static List<Long> longs(Object v) {
            List<Long> out = new ArrayList<>();
            for (Object o : Json.arr(v)) {
                out.add(((Number) o).longValue());
            }
            return out;
        }
    }

    public final String project;
    public final int avgCalled;
    public final int avgCalls;
    public final double avgPagerank;
    public final String pagerankSource;
    public final List<MethodRecord> methods;          // cleaned, file order
    public final Map<Long, MethodRecord> byId;
    public final Map<String, Object> report = new LinkedHashMap<>();

    private ProjectData(String project, int avgCalled, int avgCalls, double avgPagerank,
                        String pagerankSource, List<MethodRecord> methods) {
        this.project = project;
        this.avgCalled = avgCalled;
        this.avgCalls = avgCalls;
        this.avgPagerank = avgPagerank;
        this.pagerankSource = pagerankSource;
        this.methods = Collections.unmodifiableList(methods);
        Map<Long, MethodRecord> map = new LinkedHashMap<>();
        for (MethodRecord m : methods) {
            map.put(m.id, m);
        }
        this.byId = Collections.unmodifiableMap(map);
    }

    public static ProjectData load(Path jsonFile) throws IOException {
        Map<String, Object> root = Json.readObject(jsonFile);
        String project = Json.str(root, "project");
        Map<String, Object> averages = Json.obj(root.get("averages"));
        List<Object> raw = Json.arr(root.get("methods"));

        // group entries by id to detect duplicates and collisions
        Map<Long, List<Map<String, Object>>> byId = new LinkedHashMap<>();
        for (Object o : raw) {
            Map<String, Object> m = Json.obj(o);
            byId.computeIfAbsent(Json.lng(m, "id"), k -> new ArrayList<>()).add(m);
        }
        List<MethodRecord> methods = new ArrayList<>();
        int exactDuplicates = 0;
        Map<Long, Integer> collisions = new LinkedHashMap<>();
        for (Map.Entry<Long, List<Map<String, Object>>> e : byId.entrySet()) {
            List<Map<String, Object>> entries = e.getValue();
            Set<String> distinct = new LinkedHashSet<>();
            for (Map<String, Object> m : entries) {
                distinct.add(Json.write(m));
            }
            if (distinct.size() > 1) {
                collisions.put(e.getKey(), entries.size());
                continue;
            }
            exactDuplicates += entries.size() - 1;
            methods.add(new MethodRecord(entries.get(0)));
        }

        // drop references to ids that are not kept
        Set<Long> kept = new LinkedHashSet<>();
        for (MethodRecord m : methods) {
            kept.add(m.id);
        }
        int droppedRefs = 0;
        for (MethodRecord m : methods) {
            int before = m.called.size() + m.calls.size();
            m.called.removeIf(x -> !kept.contains(x));
            m.calls.removeIf(x -> !kept.contains(x));
            droppedRefs += before - m.called.size() - m.calls.size();
        }

        double avgPr = Json.dbl(averages, "pagerank");
        String prSource = "original";
        boolean allZero = methods.stream().allMatch(m -> m.pagerank == 0.0);
        if (allZero) {
            Map<Long, Double> pr = pageRank(methods, 0.85, 100);
            double total = 0;
            for (MethodRecord m : methods) {
                m.pagerank = pr.get(m.id);
                total += m.pagerank;
            }
            avgPr = total / methods.size();
            prSource = "recomputed from call graph (original values all 0)";
        }

        ProjectData data = new ProjectData(project,
                (int) Json.lng(averages, "called"), (int) Json.lng(averages, "calls"),
                avgPr, prSource, methods);
        data.report.put("raw_entries", raw.size());
        data.report.put("methods_kept", methods.size());
        data.report.put("exact_duplicates_merged", exactDuplicates);
        Map<String, Object> coll = new LinkedHashMap<>();
        collisions.forEach((k, v) -> coll.put(String.valueOf(k), v));
        data.report.put("colliding_ids_dropped", coll);
        data.report.put("dangling_call_refs_removed", droppedRefs);
        data.report.put("pagerank_source", prSource);
        data.report.put("avg_pagerank", avgPr);
        return data;
    }

    /** Standard PageRank over caller -> callee edges, dangling mass spread uniformly. */
    static Map<Long, Double> pageRank(List<MethodRecord> methods, double damping, int iterations) {
        int n = methods.size();
        Map<Long, Integer> index = new HashMap<>();
        for (int i = 0; i < n; i++) {
            index.put(methods.get(i).id, i);
        }
        int[][] out = new int[n][];
        for (int i = 0; i < n; i++) {
            out[i] = methods.get(i).calls.stream().distinct()
                    .map(index::get).filter(Objects::nonNull).mapToInt(Integer::intValue).toArray();
        }
        double[] pr = new double[n];
        java.util.Arrays.fill(pr, 1.0 / n);
        for (int it = 0; it < iterations; it++) {
            double[] next = new double[n];
            double dangling = 0;
            for (int i = 0; i < n; i++) {
                if (out[i].length == 0) {
                    dangling += pr[i];
                } else {
                    double share = damping * pr[i] / out[i].length;
                    for (int j : out[i]) {
                        next[j] += share;
                    }
                }
            }
            double base = (1 - damping) / n + damping * dangling / n;
            for (int i = 0; i < n; i++) {
                next[i] += base;
            }
            pr = next;
        }
        Map<Long, Double> result = new HashMap<>();
        for (int i = 0; i < n; i++) {
            result.put(methods.get(i).id, pr[i]);
        }
        return result;
    }

    /** Methods ordered by PageRank (descending), ties broken by id (ascending). */
    public List<MethodRecord> rankedByPagerank() {
        List<MethodRecord> sorted = new ArrayList<>(methods);
        sorted.sort((a, b) -> {
            int c = Double.compare(b.pagerank, a.pagerank);
            return c != 0 ? c : Long.compare(a.id, b.id);
        });
        return sorted;
    }
}

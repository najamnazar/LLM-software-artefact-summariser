package dps_nlg;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.ObjectWriter;

import org.apache.commons.collections4.MultiValuedMap;

import common.projectparser.ProjectJsonStore;
import dps_nlg.summarygenerator.Summarise;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;

/**
 * Generates NLG summaries from a project's JSON representation.
 * <p>
 * This is the second phase of the DPS_NLG pipeline. The first phase parses the source and writes
 * {@code output/json-output/nlg/&lt;project&gt;.json}; this phase reads that file back and runs the
 * SimpleNLG summariser over it. Summarisation used to run inside {@code ParseProject} on the
 * in-memory parse result, so the JSON was a record of work already done rather than its input, and
 * DPS_NLG was the only pipeline that did not consume its own JSON representation. DPS_SWUM reads
 * JSON and applies SWUM grammar to it; DPS_NLG now reads JSON and applies NLG templates to it.
 * </p>
 * <p>
 * The summaries are written back into the same JSON file under {@code summary_NLG} and
 * {@code final_summary}, so the finished artefact carries both the features and the summaries
 * derived from them.
 * </p>
 *
 * @author Najam
 */
public class NlgJsonSummariser {

    private final ObjectMapper objectMapper = new ObjectMapper();

    /**
     * Summarises one project JSON file and writes the summaries back into it.
     *
     * @param jsonFile the project JSON file written by the parse phase
     * @param writer the shared pretty-printing writer used to rewrite the file
     * @return number of classes the summariser emitted a CSV row for
     * @throws IOException if the file cannot be read, summarised or rewritten
     */
    @SuppressWarnings({"rawtypes", "unchecked"})
    public int summariseJsonFile(File jsonFile, ObjectWriter writer) throws IOException {
        if (jsonFile == null) {
            throw new IllegalArgumentException("jsonFile must not be null");
        }
        if (writer == null) {
            throw new IllegalArgumentException("writer must not be null");
        }

        HashMap<String, Object> parsedProject = ProjectJsonStore.read(objectMapper, jsonFile);

        Object identifierObj = parsedProject.get("project_identifier");
        if (identifierObj == null) {
            System.err.println("\tNo project_identifier in " + jsonFile.getName() + "; skipping.");
            return 0;
        }
        String projectIdentifier = String.valueOf(identifierObj);

        Object projectObj = parsedProject.get(projectIdentifier);
        if (!(projectObj instanceof Map)) {
            System.err.println("\tNo class payload under \"" + projectIdentifier + "\" in "
                    + jsonFile.getName() + "; skipping.");
            return 0;
        }
        HashMap<String, HashMap> fileDetails = new HashMap<>((Map<String, HashMap>) projectObj);

        // ProjectJsonStore.read has already restored the collection types the pattern summarisers
        // cast to, so this data behaves exactly like the in-memory parse result did.
        ArrayList<HashMap> designPatterns =
                (ArrayList<HashMap>) parsedProject.get("design_pattern");
        if (designPatterns == null) {
            designPatterns = new ArrayList<>();
        }

        HashMap<String, MultiValuedMap<String, String>> summaries = new HashMap<>();
        String finalSummary = new Summarise().summarise(fileDetails, designPatterns, summaries, projectIdentifier);

        parsedProject.put("summary_NLG", toSummaryMap(summaries));
        parsedProject.put("final_summary", finalSummary);
        ProjectJsonStore.write(writer, jsonFile, parsedProject);

        // Every non-empty line of the project summary corresponds to one CSV row.
        return finalSummary.isEmpty() ? 0 : finalSummary.split("\n").length;
    }

    /**
     * Converts the summariser's multi-valued map into the nested map/set form stored under
     * {@code summary_NLG}. Moved here verbatim from ParseProject when summarisation moved out of
     * the parser.
     */
    private HashMap<String, HashMap<String, HashSet<String>>> toSummaryMap(
            HashMap<String, MultiValuedMap<String, String>> summaries) {
        HashMap<String, HashMap<String, HashSet<String>>> summaryMap = new HashMap<>();
        for (String designPattern : summaries.keySet()) {
            summaryMap.put(designPattern, new HashMap<>());
            for (String classString : summaries.get(designPattern).keySet()) {
                HashSet<String> summarySet = new HashSet<>();
                for (String summary : summaries.get(designPattern).get(classString)) {
                    summarySet.add(summary);
                }
                summaryMap.get(designPattern).put(classString, summarySet);
            }
        }
        return summaryMap;
    }
}

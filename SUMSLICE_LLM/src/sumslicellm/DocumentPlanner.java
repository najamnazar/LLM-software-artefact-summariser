package sumslicellm;

/*
 * Derived from the released SumSlice source of P. W. McBurney and
 * C. McMillan (2013), bundled in this repository at
 *   ORIGINAL_SUMSLICE/sumslice/sumslice/DocumentPlanner.java.
 *
 * Changed: reads the JSON method list (ProjectData) instead of the XML config
 * file; messages kept in insertion order; deterministic caller tie-breaking;
 * NO_CALLER = -1 instead of 0. The message logic itself is unchanged. The
 * class comment below lists each difference and why it was needed.
 */

import sumslicellm.messages.*;
import sumslicellm.ProjectData.MethodRecord;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.Vector;

/**
 * Content determination and document structuring of SumSlice
 * (original: Collin McMillan, 2013-03-12), reading the JSON method list instead
 * of the XML config file.
 *
 * The message logic is unchanged. Differences to the original, all needed for
 * a correct and repeatable run:
 *  1. Messages are kept in insertion order (the original used a HashSet, so the
 *     order of the two "output used" sentences depended on the JVM's hashing).
 *  2. Callers are examined in the order listed in the input, so ties in
 *     PageRank are always broken the same way (first listed caller wins).
 *  3. When no caller qualifies, the original returned id 0 and then looked up
 *     the summary of whichever method had id 0 (the catch-all for unparsed
 *     methods). Here "no caller" is -1 and produces no message.
 */
public class DocumentPlanner {

    public static final int NO_CALLER = -1;

    private final ProjectData data;
    private final Set<Integer> methodList = new LinkedHashSet<>();
    private final Set<Message> messages = new LinkedHashSet<>();
    private final Map<Integer, List<Message>> messagesById = new HashMap<>();
    private final Vector<Message> documentPlan = new Vector<>();

    public DocumentPlanner(ProjectData data) {
        this.data = data;
    }

    private void add(Message m) {
        messages.add(m);
        messagesById.computeIfAbsent(m.getId(), k -> new ArrayList<>()).add(m);
    }

    /** "Content Determination": create the unordered messages. */
    public int generateMessages() {
        double avgPagerank = data.avgPagerank;

        for (MethodRecord method : data.methods) {
            int mid = (int) method.id;
            String mname = method.name;
            methodList.add(mid);

            if (method.returnType != null && !method.returnType.equals("void")) {
                ReturnMessage rm = new ReturnMessage();
                rm.setId(mid);
                rm.setMethod(mname);
                rm.setReturnType(method.returnType);
                add(rm);
            }

            // importance is measured with PageRank
            ImportanceMessage im = new ImportanceMessage();
            im.setId(mid);
            im.setMethod(mname);
            im.setPagerank(method.pagerank);
            im.setAvgPagerank(avgPagerank);
            add(im);

            if (method.swumVerb != null && method.swumObject != null) {
                QuickSummaryMessage sm = new QuickSummaryMessage();
                sm.setId(mid);
                sm.setMethod(mname);
                sm.setVerb(method.swumVerb);
                sm.setObject(method.swumObject);
                add(sm);
            }

            if (method.useType != null && method.useExample != null) {
                UseMessage um = new UseMessage();
                um.setId(mid);
                um.setMethod(mname);
                um.setType(method.useType);
                um.setExample(method.useExample);
                add(um);
            }
        }

        // second pass: messages that need all methods to be known
        for (MethodRecord method : data.methods) {
            int mid = (int) method.id;
            String mname = method.name;

            Set<Integer> callerset = new LinkedHashSet<>();
            int calledcount = 0;
            for (Long id : method.called) {
                calledcount++;
                callerset.add(id.intValue());
            }

            int callerone = getCallerOne(callerset);
            int callertwo = getCallerTwo(callerset, callerone);

            CalledMessage cm = new CalledMessage();
            cm.setId(mid);
            cm.setMethod(mname);
            cm.setCalledCount(calledcount);
            cm.setCallerSet(callerset);
            cm.setCallerOne(callerone);
            cm.setCallerTwo(callertwo);
            add(cm);

            for (int caller : new int[]{callerone, callertwo}) {
                QuickSummaryMessage sm = caller == NO_CALLER ? null : getQuickSummaryMessage(caller);
                if (sm != null) {
                    OutputUsedMessage oum = new OutputUsedMessage();
                    oum.setId(mid);
                    oum.setMethod(mname);
                    oum.setVP(sm.getVerb());
                    oum.setNP(sm.getObject());
                    add(oum);
                }
            }
        }
        return messages.size();
    }

    private QuickSummaryMessage getQuickSummaryMessage(int id) {
        return (QuickSummaryMessage) getMessage(id, QuickSummaryMessage.class);
    }

    private double pagerankOf(int id) {
        ImportanceMessage im = (ImportanceMessage) getMessage(id, ImportanceMessage.class);
        return im == null ? -1 : im.getPagerank();
    }

    /** The most important caller (highest PageRank, must be above 0). */
    private int getCallerOne(Set<Integer> callerset) {
        double highPr = 0;
        int highId = NO_CALLER;
        for (int id : callerset) {
            double pr = pagerankOf(id);
            if (highPr < pr) {
                highPr = pr;
                highId = id;
            }
        }
        return highId;
    }

    /** The second most important caller. */
    private int getCallerTwo(Set<Integer> callerset, int callerone) {
        double highPr = 0;
        int highId = NO_CALLER;
        for (int id : callerset) {
            double pr = pagerankOf(id);
            if (highPr < pr && id != callerone) {
                highPr = pr;
                highId = id;
            }
        }
        return highId;
    }

    public Set<Message> getMessages() {
        return messages;
    }

    /** "Document Structuring": order messages per method. */
    public void createDocumentPlan() {
        for (Integer id : methodList) {
            Message sm = getMessage(id, QuickSummaryMessage.class);
            Message rm = getMessage(id, ReturnMessage.class);
            List<Message> om = getMessages(id, OutputUsedMessage.class);
            Message cm = getMessage(id, CalledMessage.class);
            Message im = getMessage(id, ImportanceMessage.class);
            Message um = getMessage(id, UseMessage.class);

            if (sm != null) documentPlan.add(sm);
            if (rm != null) documentPlan.add(rm);
            documentPlan.addAll(om);
            if (cm != null) documentPlan.add(cm);
            if (im != null) documentPlan.add(im);
            if (um != null) documentPlan.add(um);
        }
    }

    public List<Message> getMessages(int id, Class<?> type) {
        List<Message> out = new ArrayList<>();
        for (Message m : messagesById.getOrDefault(id, List.of())) {
            if (type.isInstance(m)) {
                out.add(m);
            }
        }
        return out;
    }

    public Message getMessage(int id, Class<?> type) {
        for (Message m : messagesById.getOrDefault(id, List.of())) {
            if (type.isInstance(m)) {
                return m;
            }
        }
        return null;
    }

    public Vector<Message> getDocumentPlan() {
        return documentPlan;
    }
}

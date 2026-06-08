package common.designpatternidentifier;

import java.util.*;

import org.apache.commons.collections4.MultiValuedMap;

import dps_nlg.summarygenerator.messages.DesignPatternClassMessage;
import dps_nlg.summarygenerator.messages.DesignPatternMethodMessage;
import common.utils.Utils;

// Adapter is a structural design pattern that allows objects with incompatible interfaces to collaborate.
public class AdapterPattern extends DesignPatterns {
    public AdapterPattern() {
        super("adapter");
    }

    @Override
    public HashMap checkPattern(HashMap<String, HashMap> fileDetails) {
        // System.out.print("Pattern: \t");
        // System.out.println(patternName);

        HashMap output = new HashMap<>();
        HashMap adapters = new HashMap<>();

        for (Map.Entry<String, HashMap> targetEntry : fileDetails.entrySet()) {

            String target = targetEntry.getKey();
            for (Map.Entry<String, HashMap> adapterEntry : fileDetails.entrySet()) {
                for (HashMap adapterClassDetail : Utils.getClassOrInterfaceDetails(adapterEntry.getValue())) {
                    if (!Utils.getImplementsFrom(adapterClassDetail).contains(target) && !Utils
                            .getExtendsFrom(adapterClassDetail).contains(target))
                        continue;
                    String adapter = adapterEntry.getKey();

                    // Check if atleast one method is common
                    boolean methodMatch = false;
                    for (HashMap adapterMethodDetail : Utils.getMethodDetails(adapterEntry.getValue())) {
                        for (HashMap targetMethodDetail : Utils.getMethodDetails(targetEntry.getValue())) {
                            // Bug 3 fix: refactoring.guru does not mandate matching method visibility.
                            // Interface methods are stored with empty modifier list [] while implementing
                            // class methods store ['public']; comparing them always fails. Removed.
                            if (Utils.getMethodName(adapterMethodDetail)
                                    .equals(Utils.getMethodName(targetMethodDetail))
                                    && Utils.getMethodReturnType(adapterMethodDetail)
                                            .equals(Utils.getMethodReturnType(targetMethodDetail))
                                    // && Utils.getMethodModifierType(adapterMethodDetail)
                                    //         .equals(Utils.getMethodModifierType(targetMethodDetail))
                                    ) {
                                methodMatch = true;
                                break;
                            }
                        }
                        if (methodMatch)
                            break;
                    }

                    if (!methodMatch)
                        continue;

                    // Object Adapter
                    for (HashMap fieldDetail : Utils.getFieldDetails(adapterEntry.getValue())) {

                        String adaptee = Utils.getFieldDataType(fieldDetail);
                        if (fileDetails.containsKey(adaptee) && !adaptee.equals(target)) {

                            adapters.putIfAbsent(adapter, new HashMap<>());
                            ((HashMap) adapters.get(adapter)).putIfAbsent("target", new HashSet<>());
                            ((HashSet) ((HashMap) adapters.get(adapter)).get("target")).add(target);
                            ((HashMap) adapters.get(adapter)).putIfAbsent("adaptee", new HashSet<>());
                            ((HashSet) ((HashMap) adapters.get(adapter)).get("adaptee")).add(adaptee);
                        }
                    }

                    // Class Adapter
                    ArrayList<String> possibleAdaptees = Utils.getExtendsFrom(adapterClassDetail);
                    possibleAdaptees.addAll(Utils.getImplementsFrom(adapterClassDetail));
                    for (String adaptee : possibleAdaptees) {
                        if (fileDetails.containsKey(adaptee) && !adaptee.equals(target)) {

                            // Adaptee and Target are implemented/extended by Adapter.
                            // Adapter overrides atleast one method of Target

                            adapters.putIfAbsent(adapter, new HashMap<>());
                            ((HashMap) adapters.get(adapter)).putIfAbsent("target", new HashSet<>());
                            ((HashSet) ((HashMap) adapters.get(adapter)).get("target")).add(target);
                            ((HashMap) adapters.get(adapter)).putIfAbsent("adaptee", new HashSet<>());
                            ((HashSet) ((HashMap) adapters.get(adapter)).get("adaptee")).add(adaptee);
                        }
                    }
                }
            }
        }

        if (!adapters.isEmpty())
            output.put(patternName, adapters);
        return output;
    }

    @Override
    public void summarise(HashMap<String, HashMap> fileDetails, HashMap designPatternDetails,
            MultiValuedMap<String, String> summary) {
        for (Map.Entry<String, HashMap> adapterDetails : ((HashMap<String, HashMap>) designPatternDetails
                .getOrDefault(patternName, new HashMap<>())).entrySet()) {

            String adapter = adapterDetails.getKey();
            HashSet<String> targets = (HashSet) adapterDetails.getValue().getOrDefault("target", new HashSet<>());
            HashSet<String> adaptees = (HashSet) adapterDetails.getValue().getOrDefault("adaptee", new HashSet<>());

            // generate target sentence
            for (String adaptee : adaptees) {
                for (String target : targets) {

                    ArrayList<DesignPatternMethodMessage> mmal = new ArrayList<>();
                    DesignPatternClassMessage cm = new DesignPatternClassMessage(target, "target");
                    cm.setRelatedClassDesignPattern(patternNameAsText);
                    cm.setRelatedClassName(adapter);

                    // could be target?
                    HashMap classDetail = fileDetails.getOrDefault(target, new HashMap<>());
                    for (HashMap methodDetail : Utils.getMethodDetails(classDetail)) {

                        ArrayList<String> incomingMethodArrayList = Utils
                                .getMethodNameFromMatchingIncomingMethod(methodDetail, adapter, adaptee);

                        String incomingMethodNames = String.join(", ", incomingMethodArrayList);

                        DesignPatternMethodMessage mm = new DesignPatternMethodMessage();
                        mm.setMethodVerb("adapt");
                        mm.setTarget(target);
                        mm.setMethodName(incomingMethodNames);

                        mmal.add(mm);
                    }

                    String targetSentence = sentenceGenerator.generateSentence(cm, mmal, null);
                    summary.put(target, targetSentence);
                }

                // generate adaptee sentence
                String targetsAsText = String.join(", ", targets);
                ArrayList<DesignPatternMethodMessage> adapteeMmal = new ArrayList<>();
                DesignPatternClassMessage adapteeCm = new DesignPatternClassMessage(adaptee, "adaptee");
                adapteeCm.setRelatedClassDesignPattern("adapter");
                adapteeCm.setRelatedClassName(adapter);

                HashMap adapteeDetails = fileDetails.getOrDefault(adaptee, new HashMap<>());
                for (HashMap methodDetail : Utils.getMethodDetails(adapteeDetails)) {
                    ArrayList<String> incomingMethodArrayList = Utils
                            .getMethodNameFromMatchingIncomingMethod(methodDetail, adapter, adaptee);

                    String incomingMethodNames = String.join(", ", incomingMethodArrayList);

                    DesignPatternMethodMessage mm = new DesignPatternMethodMessage();
                    mm.setMethodVerb("adapt");
                    mm.setTarget(targetsAsText);
                    mm.setMethodName(incomingMethodNames);

                    adapteeMmal.add(mm);
                }

                String adapteeSentence = sentenceGenerator.generateSentence(adapteeCm, adapteeMmal, null);
                summary.put(adaptee, adapteeSentence);

                // generate adapter sentence
                ArrayList<DesignPatternMethodMessage> adapterMmal = new ArrayList<>();
                DesignPatternClassMessage adapterCm = new DesignPatternClassMessage(adapter, "adapter");

                HashMap adapterDetail = fileDetails.getOrDefault(adapter, new HashMap<>());
                for (HashMap methodDetail : Utils.getMethodDetails(adapterDetail)) {
                    ArrayList<String> incomingMethodArrayList = Utils
                            .getMethodNameFromMatchingIncomingMethod(methodDetail, adapter, adaptee);

                    String incomingMethodNames = String.join(", ", incomingMethodArrayList);

                    DesignPatternMethodMessage adapterMm = new DesignPatternMethodMessage();
                    adapterMm.setMethodVerb("adapt");
                    adapterMm.setTarget(targetsAsText);
                    adapterMm.setMethodName(incomingMethodNames);

                    adapterMmal.add(adapterMm);
                }
                String adapterSentence = sentenceGenerator.generateSentence(adapterCm, adapterMmal, null);
                summary.put(adapter, adapterSentence);
            }
        }
    }
}


package common.designpatternidentifier;

import java.util.*;

import org.apache.commons.collections4.MultiValuedMap;

import dps_nlg.summarygenerator.messages.DesignPatternClassMessage;
import dps_nlg.summarygenerator.messages.DesignPatternInheritClassMessage;
import dps_nlg.summarygenerator.messages.DesignPatternMethodMessage;
import common.utils.Utils;

// Visitor is a behavioral design pattern that lets you separate algorithms from the objects on which they operate.
public class VisitorPattern extends DesignPatterns {
    public VisitorPattern() {
        super("visitor");
    }

    @Override
    public HashMap checkPattern(HashMap<String, HashMap> fileDetails) {

        HashMap output = new HashMap<>();
        HashMap visitors = new HashMap<>();

        for (Map.Entry<String, HashMap> visitorEntry : fileDetails.entrySet()) {

            boolean isInterfaceOrNot = false;
            for (HashMap classDetail : Utils.getClassOrInterfaceDetails(visitorEntry.getValue())) {
                if (Utils.isInterfaceOrNot(classDetail)) {
                    isInterfaceOrNot = true;
                    break;
                }
            }

            if (!isInterfaceOrNot)
                continue;
            String visitor = visitorEntry.getKey();

            // Visitor must have method that accepts parameter of type Element
            HashSet<String> concreteVisitors = new HashSet<>();
            for (Map.Entry<String, HashMap> concreteVisitorEntry : fileDetails.entrySet()) {

                boolean doesImplement = false;
                for (HashMap classDetail : Utils.getClassOrInterfaceDetails(concreteVisitorEntry.getValue())) {
                    if (Utils.getImplementsFrom(classDetail).contains(visitor)) {
                        doesImplement = true;
                        break;
                    }
                }

                if (!doesImplement)
                    continue;

                for (HashMap methodDetail : Utils.getMethodDetails(visitorEntry.getValue())) {
                    for (HashMap concreteMethodDetail : Utils.getMethodDetails(concreteVisitorEntry.getValue())) {
                        if (Utils.isMethodOverride(concreteMethodDetail)
                                && Utils.getMethodName(methodDetail)
                                        .equals(Utils.getMethodName(concreteMethodDetail))) {
                            concreteVisitors.add(concreteVisitorEntry.getKey());
                            break;
                        }
                    }
                }

            }

            if (concreteVisitors.size() == 0)
                continue;

            for (Map.Entry<String, HashMap> elementEntry : fileDetails.entrySet()) {

                // Bug 4 fix: refactoring.guru uses "Element Interface" structurally, not as a Java
                // keyword. Abstract classes and concrete classes with accept(Visitor) serve the same
                // structural role. The method-parameter check below is sufficient gating.
                // boolean isInterfaceOrNotElement = false;
                // for (HashMap classDetail : Utils.getClassOrInterfaceDetails(elementEntry.getValue())) {
                //     if (Utils.isInterfaceOrNot(classDetail)) {
                //         isInterfaceOrNotElement = true;
                //         break;
                //     }
                // }
                // if (!isInterfaceOrNotElement)
                //     continue;

                String element = elementEntry.getKey();
                for (HashMap methodDetail : Utils.getMethodDetails(elementEntry.getValue())) {
                    for (HashMap parameter : (ArrayList<HashMap>) Utils.getMethodParameters(methodDetail)) {
                        if (Utils.getParameterType(parameter).equals(visitor)) {
                            visitors.putIfAbsent(element, new HashMap<>());
                            break;
                        }
                    }
                }

                if (!visitors.containsKey(element))
                    continue;

                for (Map.Entry<String, HashMap> concreteElementEntry : fileDetails.entrySet()) {
                    for (HashMap classDetail : Utils.getClassOrInterfaceDetails(concreteElementEntry.getValue())) {
                        if (Utils.getImplementsFrom(classDetail).contains(element)) {
                            ((HashMap) visitors.get(element)).putIfAbsent("concrete_element", new HashSet<>());
                            ((HashSet) ((HashMap) visitors.get(element)).get("concrete_element"))
                                    .add(concreteElementEntry.getKey());
                            break;
                        }
                    }
                }

                ((HashMap) visitors.get(element)).putIfAbsent("visitor", new HashMap<>());
                ((HashMap) ((HashMap) visitors.get(element)).get("visitor"))
                        .putIfAbsent(visitor, concreteVisitors);
            }
        }
        if (!visitors.isEmpty())
            output.put(patternName, visitors);
        return output;
    }

    @Override
    public void summarise(HashMap<String, HashMap> fileDetails, HashMap designPatternDetails,
            MultiValuedMap<String, String> summary) {
        for (Map.Entry<String, HashMap> visitorEntry : ((HashMap<String, HashMap>) designPatternDetails
                .getOrDefault(patternName, new HashMap<>())).entrySet()) {

            String element = visitorEntry.getKey();

            String elementSentence = "";
            HashSet<String> concreteElements = (HashSet<String>) visitorEntry.getValue().getOrDefault("concrete_element", 
                    new HashSet<>());
            HashMap<String, HashSet> visitors = (HashMap<String, HashSet>) visitorEntry.getValue().getOrDefault("visitor",
                    new HashMap<>());
            HashSet baseVisitors = new HashSet<>();

            for (String concreteElement : concreteElements) {

                HashMap classDetail = fileDetails.getOrDefault(concreteElement, new HashMap<>());
                HashMap parentClassDetail = fileDetails.getOrDefault(element, new HashMap<>());

                ArrayList<DesignPatternMethodMessage> cemmal = new ArrayList<>();
                DesignPatternMethodMessage mm = new DesignPatternMethodMessage();
                ArrayList<String> overrideMethodArray = new ArrayList<>();
                HashSet<String> concreteElementVisitors = new HashSet<>();

                overrideMethodArray.addAll(Utils.checkMethodOverride(classDetail, parentClassDetail,
                        " method of " + element));

                mm.setOverrideMethod(overrideMethodArray);
                mm.setMethodVerb("override");
                cemmal.add(mm);

                // find all visitors that the element has
                mm = new DesignPatternMethodMessage();
                for (HashMap methodDetail : Utils.getMethodDetails(classDetail)) {
                    ArrayList<String> methodParameterClasses = Utils.getMethodParameterAsText(methodDetail);

                    for (String methodParameterClass : methodParameterClasses) {
                        if (visitors.keySet().contains(methodParameterClass)) {

                            baseVisitors.add(methodParameterClass);
                            concreteElementVisitors.add(methodParameterClass);
                            ArrayList<String> methodParameters = Utils.getMethodParameterAsText(methodDetail);
                            String methodName = Utils.getMethodName(methodDetail);
                            mm.setMethodVerb("take");
                            mm.setParameter(methodParameters);
                            mm.setTarget(methodName);

                            cemmal.add(mm);
                        }
                    }
                }
                // generate concrete element sentence
                String relatedVisitorClasses = String.join(", ", concreteElementVisitors);

                DesignPatternClassMessage cecm = new DesignPatternClassMessage(concreteElement, "element");
                cecm.setRelatedClassDesignPattern("visitor");
                cecm.setRelatedClassName(relatedVisitorClasses);

                // Reason for change:
                // concrete element messages were prepared but never emitted into summary, which caused
                // many concrete element classes to fall back to generic non-pattern text.
                // Legacy behavior kept below for traceability (do not remove changed code).
                // String relatedVisitorClasses = String.join(", ", baseVisitors);
                // DesignPatternClassMessage cecm = new DesignPatternClassMessage(concreteElement, "element");
                // cecm.setRelatedClassDesignPattern("visitor");
                // cecm.setRelatedClassName(relatedVisitorClasses);

                String concreteElementSentence = sentenceGenerator.generateSentence(cecm, cemmal, null);
                summary.put(concreteElement, concreteElementSentence);
            }

            for (Map.Entry<String, HashSet> visitorDetails : visitors.entrySet()) {

                // Reason for change:
                // visitorEntry.getKey() is the element key, not the visitor key. Using it here mislabels
                // visitor sentences and prevents correct role summaries.
                // Legacy line kept for traceability (do not remove changed code):
                // String visitor = visitorEntry.getKey();
                String visitor = visitorDetails.getKey();
                for (String concreteVisitor : (HashSet<String>) visitorDetails.getValue()) {
                    ArrayList<DesignPatternMethodMessage> cvmmal = new ArrayList<>();
                    DesignPatternMethodMessage mm = new DesignPatternMethodMessage();
                    ArrayList<String> overrideMethodArray = new ArrayList<>();

                    HashMap classDetail = fileDetails.getOrDefault(concreteVisitor, new HashMap<>());
                    HashMap parentClassDetail = fileDetails.getOrDefault(visitor, new HashMap<>());
                    overrideMethodArray
                            .addAll(Utils.checkMethodOverride(classDetail, parentClassDetail, " method of " + element));

                    mm.setOverrideMethod(overrideMethodArray);
                    mm.setMethodVerb("override");
                    cvmmal.add(mm);

                    // find all elements that the concrete visitor has
                    for (HashMap methodDetail : Utils.getMethodDetails(classDetail)) {
                        mm = new DesignPatternMethodMessage();
                        ArrayList<String> methodParameterClasses = Utils
                                .getMethodParameterAsText(methodDetail);
                        String currentMethodName = Utils.getMethodName(methodDetail);

                        String[] currentMethodNameArray = Utils.splitByCamelCase(currentMethodName);
                        if (currentMethodNameArray.length > 1) {
                            ArrayList<String> methodActionArray = new ArrayList<>();

                            // first word of the method name is set to be the verb, the rest is set to be
                            // the actions
                            methodActionArray.addAll(Arrays.asList(currentMethodNameArray).subList(1,
                                    currentMethodNameArray.length));
                            String methodAction = String.join(" ", methodActionArray);

                            mm.setMethodVerb(currentMethodNameArray[0]);
                            mm.setMethodAction(methodAction);
                        } else {
                            mm.setMethodVerb(currentMethodNameArray[0]);
                            mm.setParameter(methodParameterClasses);
                        }
                        cvmmal.add(mm);
                    }

                    // generate concrete visitor sentence
                    DesignPatternClassMessage cvcm = new DesignPatternClassMessage(concreteVisitor, "visitor");
                    cvcm.setRelatedClassDesignPattern("element");
                    cvcm.setRelatedClassName(element);

                    String concreteVisitorSentence = sentenceGenerator.generateSentence(cvcm, cvmmal, null);

                    summary.put(concreteVisitor, concreteVisitorSentence);
                }

                // generate visitor sentence
                String concreteVisitorClasses = String.join(", ", (HashSet<String>) visitorDetails.getValue());

                DesignPatternClassMessage vcm = new DesignPatternClassMessage(visitor, patternNameAsText);
                vcm.setRelatedClassName(element);
                vcm.setRelatedClassDesignPattern("element");
                ArrayList<DesignPatternMethodMessage> vmmal = new ArrayList<>();
                DesignPatternMethodMessage mm = new DesignPatternMethodMessage();

                // find all elements that the visitor accepts
                HashMap classDetail = fileDetails.getOrDefault(visitor, new HashMap<>());
                for (HashMap methodDetail : Utils.getMethodDetails(classDetail)) {
                    mm = new DesignPatternMethodMessage();
                    ArrayList<String> methodParameterClasses = Utils
                            .getMethodParameterAsText(methodDetail);
                    String currentMethodName = Utils.getMethodName(methodDetail);

                    String[] currentMethodNameArray = Utils.splitByCamelCase(currentMethodName);
                    if (currentMethodNameArray.length > 1) {
                        ArrayList<String> methodActionArray = new ArrayList<>();

                        // first word of the method name is set to be the verb, the rest is set to be
                        // the actions
                        methodActionArray.addAll(
                                Arrays.asList(currentMethodNameArray).subList(1, currentMethodNameArray.length));
                        String methodAction = String.join(" ", methodActionArray);

                        mm.setMethodVerb(currentMethodNameArray[0]);
                        mm.setMethodAction(methodAction);
                    } else {
                        mm.setMethodVerb(currentMethodNameArray[0]);
                        mm.setParameter(methodParameterClasses);
                    }
                    vmmal.add(mm);
                }

                DesignPatternInheritClassMessage vicm = new DesignPatternInheritClassMessage();
                vicm.setInheritClass(concreteVisitorClasses);

                String visitorSentence = sentenceGenerator.generateSentence(vcm, vmmal, vicm);
                summary.put(visitor, visitorSentence);

                // generate base element sentence
                String concreteElementClasses = String.join(", ", concreteElements);

                DesignPatternClassMessage ecm = new DesignPatternClassMessage(element, "element");
                ecm.setRelatedClassName(visitor);
                ecm.setRelatedClassDesignPattern("visitor");
                ArrayList<DesignPatternMethodMessage> emmal = new ArrayList<>();

                classDetail = fileDetails.getOrDefault(element, new HashMap<>());
                for (HashMap methodDetail : Utils.getMethodDetails(classDetail)) {
                    mm = new DesignPatternMethodMessage();
                    ArrayList<String> methodParameterClasses = Utils
                            .getMethodParameterAsText(methodDetail);
                    String currentMethodName = Utils.getMethodName(methodDetail);

                    String[] currentMethodNameArray = Utils.splitByCamelCase(currentMethodName);
                    if (currentMethodNameArray.length > 1) {
                        ArrayList<String> methodActionArray = new ArrayList<>();

                        // first word of the method name is set to be the verb, the rest is set to be
                        // the actions
                        methodActionArray.addAll(
                                Arrays.asList(currentMethodNameArray).subList(1, currentMethodNameArray.length));
                        String methodAction = String.join(" ", methodActionArray);

                        mm.setMethodVerb(currentMethodNameArray[0]);
                        mm.setMethodAction(methodAction);
                    } else {
                        mm.setMethodVerb(currentMethodNameArray[0]);
                        mm.setParameter(methodParameterClasses);
                    }
                    emmal.add(mm);
                }

                DesignPatternInheritClassMessage eicm = new DesignPatternInheritClassMessage();
                eicm.setInheritClass(concreteElementClasses);

                elementSentence = sentenceGenerator.generateSentence(ecm, emmal, eicm);
            }

            summary.put(element, elementSentence);
        }
    }
}


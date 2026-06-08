package common.designpatternidentifier;

import java.util.*;

import org.apache.commons.collections4.MultiValuedMap;

import dps_nlg.summarygenerator.messages.DesignPatternClassMessage;
import dps_nlg.summarygenerator.messages.DesignPatternInheritClassMessage;
import dps_nlg.summarygenerator.messages.DesignPatternMethodMessage;
import common.utils.Utils;

// Memento is a behavioral design pattern that lets you save and restore the previous state of an object without revealing the details of its implementation
public class MementoPattern extends DesignPatterns {
    public MementoPattern() {
        super("memento");
    }

    @Override
    public HashMap checkPattern(HashMap<String, HashMap> fileDetails) {

        HashMap mementos = new HashMap<>();

        // originator has method returning type Memento
        for (Map.Entry<String, HashMap> originatorEntry : fileDetails.entrySet()) {

            String originator = originatorEntry.getKey();
            for (HashMap methodDetail : Utils.getMethodDetails(originatorEntry.getValue())) {

                // originator must have a method returning type Memento
                String methodReturnType = Utils.getMethodReturnType(methodDetail);
                if (!fileDetails.containsKey(methodReturnType))
                    continue;

                HashMap mementoClassDetail = fileDetails.getOrDefault(methodReturnType, new HashMap<>());
                if (!hasConcreteTypeInformation(mementoClassDetail)) {
                    // Skip interfaces or classes we cannot describe (prevents false positives on skinny APIs)
                    continue;
                }

                if (!hasConcreteTypeInformation(originatorEntry.getValue())) {
                    continue;
                }

                if (!usesTypeAsMethodParameter(originatorEntry.getValue(), methodReturnType)) {
                    // A true originator must also restore state from the same memento type.
                    continue;
                }

                // No Memento methods can have parameters
                boolean hasParameters = false;
                for (HashMap mementoMethodDetail : Utils.getMethodDetails(mementoClassDetail)) {
                    if (Utils.getMethodParameters(mementoMethodDetail).size() > 0) {
                        hasParameters = true;
                        break;
                    }
                }
                if (hasParameters)
                    continue;

                // All fields in Memento must match all fields in originator
                ArrayList<String> mementoFields = new ArrayList<>();
                for (HashMap fieldDetail : Utils.getFieldDetails(mementoClassDetail)) {
                    mementoFields.add(Utils.getFieldDataType(fieldDetail));
                }
                ArrayList<String> originatorFields = new ArrayList<>();
                for (HashMap fieldDetail : Utils.getFieldDetails(originatorEntry.getValue())) {
                    originatorFields.add(Utils.getFieldDataType(fieldDetail));
                }

                if (mementoFields.isEmpty() || originatorFields.isEmpty())
                    continue;

                // Bug 5 fix: refactoring.guru states memento "stores copies of originator's state
                // fields" but does not mandate identical Java types. StringBuilder (originator) vs
                // String (memento) stores the same logical state in a different representation.
                // Relaxed to count-only; type identity check removed.
                // if (!(mementoFields.size() == originatorFields.size()
                //         && mementoFields.containsAll(originatorFields)))
                //     continue;
                if (mementoFields.size() != originatorFields.size())
                    continue;

                String memento = methodReturnType;
                mementos.putIfAbsent(memento, new HashMap<>());
                ((HashMap) mementos.get(memento)).putIfAbsent("originator", new HashMap<>());
                ((HashMap) mementos.get(memento)).putIfAbsent("concrete_memento", new HashSet());
                ((HashMap) mementos.get(memento)).putIfAbsent("caretaker", new HashMap<>());

                ((HashMap) ((HashMap) mementos.get(memento)).get("originator")).putIfAbsent(originator,
                        new HashSet<>());

                for (Map.Entry<String, HashMap> fileEntry : fileDetails.entrySet()) {

                    // caretaker stores Memento
                    for (HashMap fieldDetail : Utils.getFieldDetails(fileEntry.getValue())) {
                        if (Utils.getFieldDataType(fieldDetail).contains(memento)) {
                            ((HashMap) ((HashMap) mementos.get(memento)).get("caretaker")).putIfAbsent(
                                    fileEntry.getKey(), new HashSet());

                        }
                    }

                    // Find concrete memento and concrete originator
                    for (HashMap classDetail : Utils.getClassOrInterfaceDetails(fileEntry.getValue())) {
                        if (Utils.getImplementsFrom(classDetail).contains(memento)
                                || Utils.getExtendsFrom(classDetail).contains(memento)) {
                            ((HashSet) ((HashMap) mementos.get(memento)).get("concrete_memento"))
                                    .add(fileEntry.getKey());
                        }

                        if (Utils.getImplementsFrom(classDetail).contains(originator)
                                || Utils.getExtendsFrom(classDetail).contains(originator)) {
                            ((HashSet) ((HashMap) ((HashMap) mementos.get(memento)).get("originator")).get(originator))
                                    .add(fileEntry.getKey());
                        }
                    }
                }

                // Find concrete caretaker
                for (Map.Entry<String, HashSet> caretaker : ((HashMap<String, HashSet>) ((HashMap) mementos
                        .get(memento)).get("caretaker")).entrySet()) {
                    for (Map.Entry<String, HashMap> fileEntry : fileDetails.entrySet()) {
                        for (HashMap classDetail : Utils.getClassOrInterfaceDetails(fileEntry.getValue())) {
                            if (Utils.getImplementsFrom(classDetail).contains(caretaker.getKey())
                                    || Utils.getExtendsFrom(classDetail).contains(caretaker.getKey())) {
                                caretaker.getValue().add(fileEntry.getKey());
                            }
                        }

                    }
                }

            }
        }

        if (!mementos.isEmpty()) {
            for (String memento : (Set<String>) mementos.keySet()) {

                // Remove "originators" that implement originators or are caretakers
                HashSet<String> removeList = new HashSet<>();
                for (String originator : (Set<String>) (((HashMap) ((HashMap) mementos.get(memento)).get("originator")))
                        .keySet()) {
                    HashMap originatorDetails = fileDetails.get(originator);
                    for (HashMap originatorClassDetail : Utils.getClassOrInterfaceDetails(originatorDetails)) {
                        if (Utils.getImplementsFrom(originatorClassDetail).size() > 0
                                || Utils.getExtendsFrom(originatorClassDetail).size() > 0) {
                            removeList.add(originator);
                            break;
                        }
                    }
                }
                for (String removeString : removeList) {
                    (((HashMap) ((HashMap) mementos.get(memento)).get("originator"))).remove(removeString);
                }
            }
        }
        return createPatternResult(mementos);
    }

    @Override
    public void summarise(HashMap<String, HashMap> fileDetails, HashMap designPatternDetails,
            MultiValuedMap<String, String> summary) {

        for (Map.Entry<String, HashMap> mementoEntry : ((HashMap<String, HashMap>) designPatternDetails
                .getOrDefault(patternName, new HashMap<>())).entrySet()) {

            String memento = mementoEntry.getKey();
            HashSet<String> concreteMementos = ((HashSet) mementoEntry.getValue().getOrDefault("concrete_memento", new HashSet<>()));
            if (concreteMementos.size() != 0) {
                for (String concreteMemento : concreteMementos) {

                    DesignPatternClassMessage cmcm = new DesignPatternClassMessage(concreteMemento, patternNameAsText);
                    DesignPatternMethodMessage mm = new DesignPatternMethodMessage();
                    ArrayList<DesignPatternMethodMessage> cmmmal = new ArrayList<>();

                    HashMap classDetail = fileDetails.getOrDefault(concreteMemento, new HashMap<>());
                    for (HashMap methodDetail : Utils.getMethodDetails(classDetail)) {

                        String methodName = Utils.getMethodName(methodDetail);
                        ArrayList<String> methodParameters = Utils.getMethodParameterAsText(methodDetail);
                        String[] currentMethodNameArray = Utils.splitByCamelCase(methodName);
                        if (currentMethodNameArray.length > 1) {
                            ArrayList<String> methodActionArray = new ArrayList<>();

                            // first word of the method name is set to be the verb, the rest is set to be
                            // the actions
                            methodActionArray.addAll(Arrays.asList(currentMethodNameArray).subList(1,
                                    currentMethodNameArray.length));
                            String methodAction = String.join(" ", methodActionArray);

                            mm.setMethodVerb(currentMethodNameArray[0]);
                            mm.setMethodAction(methodAction);
                            mm.setParameter(methodParameters);
                        } else {
                            mm.setMethodVerb(currentMethodNameArray[0]);
                            mm.setParameter(methodParameters);
                        }
                        cmmmal.add(mm);
                    }

                    String concreteMementoSentence = sentenceGenerator.generateSentence(cmcm, cmmmal, null);
                    summary.put(concreteMemento, concreteMementoSentence);
                }
            }

            for (Map.Entry<String, HashSet> caretakerEntry : ((HashMap<String, HashSet>) mementoEntry.getValue()
                    .getOrDefault("caretaker", new HashMap<>())).entrySet()) {

                String caretaker = caretakerEntry.getKey();

                // generate concrete caretaker sentence
                HashSet<String> concretecaretakers = caretakerEntry.getValue();
                if (concretecaretakers.size() != 0) {
                    for (String concretecaretaker : concretecaretakers) {

                        DesignPatternMethodMessage mm = new DesignPatternMethodMessage();
                        ArrayList<DesignPatternMethodMessage> cctmmal = new ArrayList<>();
                        DesignPatternClassMessage cctcm = new DesignPatternClassMessage(concretecaretaker,
                                "caretaker");
                        cctcm.setRelatedClassDesignPattern(patternNameAsText);
                        cctcm.setRelatedClassName(memento);

                        HashMap classDetail = fileDetails.getOrDefault(concretecaretaker, new HashMap<>());
                        for (HashMap methodDetail : Utils.getMethodDetails(classDetail)) {

                            mm = new DesignPatternMethodMessage();
                            String methodName = Utils.getMethodName(methodDetail);
                            ArrayList<String> methodParameters = Utils.getMethodParameterAsText(methodDetail);
                            String[] currentMethodNameArray = Utils.splitByCamelCase(methodName);
                            if (currentMethodNameArray.length > 1) {
                                ArrayList<String> methodActionArray = new ArrayList<>();

                                // first word of the method name is set to be the verb, the rest is set to be
                                // the actions
                                methodActionArray.addAll(Arrays.asList(currentMethodNameArray).subList(1,
                                        currentMethodNameArray.length));
                                String methodAction = String.join(" ", methodActionArray);

                                mm.setMethodVerb(currentMethodNameArray[0]);
                                mm.setMethodAction(methodAction);
                                mm.setParameter(methodParameters);
                            } else {
                                mm.setMethodVerb(currentMethodNameArray[0]);
                                mm.setParameter(methodParameters);
                            }
                            cctmmal.add(mm);
                        }

                        String concretecaretakerSentence = sentenceGenerator.generateSentence(cctcm, cctmmal, null);
                        summary.put(concretecaretaker, concretecaretakerSentence);
                    }
                }

                // generate caretaker sentence
                String concretecaretakerClasses = String.join(", ", concretecaretakers);

                DesignPatternMethodMessage mm = new DesignPatternMethodMessage();
                ArrayList<DesignPatternMethodMessage> ctmmal = new ArrayList<>();
                DesignPatternClassMessage ctcm = new DesignPatternClassMessage(caretaker, "caretaker");
                ctcm.setRelatedClassDesignPattern(patternNameAsText);
                ctcm.setRelatedClassName(memento);

                HashMap classDetail = fileDetails.getOrDefault(caretaker, new HashMap<>());
                for (HashMap methodDetail : Utils.getMethodDetails(classDetail)) {

                    mm = new DesignPatternMethodMessage();
                    String methodName = Utils.getMethodName(methodDetail);
                    ArrayList<String> methodParameters = Utils.getMethodParameterAsText(methodDetail);

                    String[] currentMethodNameArray = Utils.splitByCamelCase(methodName);
                    if (currentMethodNameArray.length > 1) {
                        ArrayList<String> methodActionArray = new ArrayList<>();

                        // first word of the method name is set to be the verb, the rest is set to be
                        // the actions
                        methodActionArray.addAll(
                                Arrays.asList(currentMethodNameArray).subList(1, currentMethodNameArray.length));
                        String methodAction = String.join(" ", methodActionArray);

                        mm.setMethodVerb(currentMethodNameArray[0]);
                        mm.setMethodAction(methodAction);
                        mm.setParameter(methodParameters);
                    } else {
                        mm.setMethodVerb(currentMethodNameArray[0]);
                        mm.setParameter(methodParameters);
                    }
                    ctmmal.add(mm);
                }

                String caretakerSentence = sentenceGenerator.generateSentence(ctcm, ctmmal, null);
                if (concretecaretakerClasses.length() != 0) {
                    DesignPatternInheritClassMessage cicm = new DesignPatternInheritClassMessage();
                    caretakerSentence = sentenceGenerator.generateSentence(ctcm, ctmmal, cicm);
                }
                summary.put(caretaker, caretakerSentence);
            }

            for (Map.Entry<String, HashSet> originatorEntry : ((HashMap<String, HashSet>) mementoEntry.getValue()
                    .getOrDefault("originator", new HashMap<>())).entrySet()) {

                String originator = originatorEntry.getKey();

                // generate concrete originator sentence
                HashSet<String> concreteoriginators = originatorEntry.getValue();
                if (concreteoriginators.size() != 0) {
                    for (String concreteoriginator : concreteoriginators) {

                        DesignPatternMethodMessage mm = new DesignPatternMethodMessage();
                        ArrayList<DesignPatternMethodMessage> commal = new ArrayList<>();
                        DesignPatternClassMessage cocm = new DesignPatternClassMessage(
                                concreteoriginator,
                                "originator");
                        cocm.setRelatedClassDesignPattern(patternNameAsText);
                        cocm.setRelatedClassName(memento);

                        HashMap classDetail = fileDetails.getOrDefault(concreteoriginator, new HashMap<>());
                        for (HashMap methodDetail : Utils.getMethodDetails(classDetail)) {

                            mm = new DesignPatternMethodMessage();
                            String MethodNameByParameterType = Utils.getMethodNameFromMatchingParameterType(
                                    methodDetail, memento);
                            String MethodNameByReturnVal = Utils.getMethodNameFromMatchingReturnType(
                                    methodDetail, memento);

                            if (!MethodNameByParameterType.equals("")) {
                                ArrayList<String> methodParameters = Utils.getMethodParameterAsText(methodDetail);

                                String[] currentMethodNameArray = Utils.splitByCamelCase(MethodNameByParameterType);
                                if (currentMethodNameArray.length > 1) {
                                    ArrayList<String> methodActionArray = new ArrayList<>();

                                    // first word of the method name is set to be the verb, the rest is set to be
                                    // the actions
                                    methodActionArray.addAll(Arrays.asList(currentMethodNameArray).subList(1,
                                            currentMethodNameArray.length));
                                    String methodAction = String.join(" ", methodActionArray);

                                    mm.setMethodVerb(currentMethodNameArray[0]);
                                    mm.setMethodAction(methodAction);
                                    mm.setParameter(methodParameters);
                                } else {
                                    mm.setMethodVerb(currentMethodNameArray[0]);
                                    mm.setParameter(methodParameters);
                                }
                                commal.add(mm);
                            }

                            if (!MethodNameByReturnVal.equals("")) {

                                String methodReturnVal = Utils.getMethodReturnType(methodDetail);

                                String[] currentMethodNameArray = Utils.splitByCamelCase(MethodNameByReturnVal);
                                if (currentMethodNameArray.length > 1) {
                                    ArrayList<String> methodActionArray = new ArrayList<>();

                                    // first word of the method name is set to be the verb, the rest is set to be
                                    // the actions
                                    methodActionArray.addAll(Arrays.asList(currentMethodNameArray).subList(1,
                                            currentMethodNameArray.length));
                                    String methodAction = String.join(" ", methodActionArray);

                                    mm.setMethodVerb(currentMethodNameArray[0]);
                                    mm.setMethodAction(methodAction);
                                    mm.setMethodReturn(methodReturnVal);
                                } else {
                                    mm.setMethodVerb(currentMethodNameArray[0]);
                                    mm.setMethodReturn(methodReturnVal);
                                }
                                commal.add(mm);
                            }
                        }

                        String concreteoriginatorSentence = sentenceGenerator.generateSentence(cocm, commal, null);
                        summary.put(concreteoriginator, concreteoriginatorSentence);
                    }
                }

                // generate originator sentence
                String concreteoriginatorClasses = String.join(", ", concreteoriginators);

                DesignPatternMethodMessage mm = new DesignPatternMethodMessage();
                ArrayList<DesignPatternMethodMessage> ommal = new ArrayList<>();
                DesignPatternClassMessage ocm = new DesignPatternClassMessage(originator, "originator");
                ocm.setRelatedClassDesignPattern(patternNameAsText);
                ocm.setRelatedClassName(memento);

                HashMap classDetail = fileDetails.getOrDefault(originator, new HashMap<>());
                for (HashMap methodDetail : Utils.getMethodDetails(classDetail)) {

                    mm = new DesignPatternMethodMessage();
                    String MethodNameByParameterType = Utils.getMethodNameFromMatchingParameterType(methodDetail,
                            memento);
                    String MethodNameByReturnVal = Utils.getMethodNameFromMatchingReturnType(methodDetail, memento);

                    if (!MethodNameByParameterType.equals("")) {
                        ArrayList<String> methodParameters = Utils.getMethodParameterAsText(methodDetail);

                        String[] currentMethodNameArray = Utils.splitByCamelCase(MethodNameByParameterType);
                        if (currentMethodNameArray.length > 1) {
                            ArrayList<String> methodActionArray = new ArrayList<>();

                            // first word of the method name is set to be the verb, the rest is set to be
                            // the actions
                            methodActionArray.addAll(Arrays.asList(currentMethodNameArray).subList(1,
                                    currentMethodNameArray.length));
                            String methodAction = String.join(" ", methodActionArray);

                            mm.setMethodVerb(currentMethodNameArray[0]);
                            mm.setMethodAction(methodAction);
                            mm.setParameter(methodParameters);
                        } else {
                            mm.setMethodVerb(currentMethodNameArray[0]);
                            mm.setParameter(methodParameters);
                        }
                        ommal.add(mm);
                    }

                    if (!MethodNameByReturnVal.equals("")) {

                        String methodReturnVal = Utils.getMethodReturnType(methodDetail);

                        String[] currentMethodNameArray = Utils.splitByCamelCase(MethodNameByReturnVal);
                        if (currentMethodNameArray.length > 1) {
                            ArrayList<String> methodActionArray = new ArrayList<>();

                            // first word of the method name is set to be the verb, the rest is set to be
                            // the actions
                            methodActionArray.addAll(Arrays.asList(currentMethodNameArray).subList(1,
                                    currentMethodNameArray.length));
                            String methodAction = String.join(" ", methodActionArray);

                            mm.setMethodVerb(currentMethodNameArray[0]);
                            mm.setMethodAction(methodAction);
                            mm.setMethodReturn(methodReturnVal);
                        } else {
                            mm.setMethodVerb(currentMethodNameArray[0]);
                            mm.setMethodReturn(methodReturnVal);
                        }
                        ommal.add(mm);
                    }
                }

                String originatorSentence = sentenceGenerator.generateSentence(ocm, ommal, null);
                if (concreteoriginatorClasses.length() != 0) {
                    DesignPatternInheritClassMessage cicm = new DesignPatternInheritClassMessage();
                    originatorSentence = sentenceGenerator.generateSentence(ocm, ommal, cicm);
                }
                summary.put(originator, originatorSentence);
            }

            String concreteMementosAsText = String.join(", ", concreteMementos);

            DesignPatternClassMessage mcm = new DesignPatternClassMessage(memento, patternNameAsText);
            DesignPatternMethodMessage mm = new DesignPatternMethodMessage();
            ArrayList<DesignPatternMethodMessage> mmmal = new ArrayList<>();

            HashMap classDetail = fileDetails.getOrDefault(memento, new HashMap<>());
            for (HashMap methodDetail : Utils.getMethodDetails(classDetail)) {
                String methodName = Utils.getMethodName(methodDetail);
                ArrayList<String> methodParameters = Utils.getMethodParameterAsText(methodDetail);

                String[] currentMethodNameArray = Utils.splitByCamelCase(methodName);
                if (currentMethodNameArray.length > 1) {
                    ArrayList<String> methodActionArray = new ArrayList<>();

                    // first word of the method name is set to be the verb, the rest is set to be
                    // the actions
                    methodActionArray.addAll(
                            Arrays.asList(currentMethodNameArray).subList(1, currentMethodNameArray.length));
                    String methodAction = String.join(" ", methodActionArray);

                    mm.setMethodVerb(currentMethodNameArray[0]);
                    mm.setMethodAction(methodAction);
                    mm.setParameter(methodParameters);
                } else {
                    mm.setMethodVerb(currentMethodNameArray[0]);
                    mm.setParameter(methodParameters);
                }
                mmmal.add(mm);
            }

            if (concreteMementos.size() != 0) {
                DesignPatternInheritClassMessage micm = new DesignPatternInheritClassMessage();
                micm.setInheritClass(concreteMementosAsText);

                String mementoSentence = sentenceGenerator.generateSentence(mcm, mmmal, micm);
                summary.put(memento, mementoSentence);
            } else {
                String mementoSentence = sentenceGenerator.generateSentence(mcm, mmmal, null);
                summary.put(memento, mementoSentence);
            }
        }
    }

    private boolean hasConcreteTypeInformation(HashMap classDetailContainer) {
        ArrayList<HashMap> classDetails = Utils.getClassOrInterfaceDetails(classDetailContainer);
        if (classDetails.isEmpty()) {
            return false;
        }

        Object isInterfaceObj = classDetails.get(0).get("ISINTERFACEORNOT");
        if (isInterfaceObj instanceof Boolean && (Boolean) isInterfaceObj) {
            return false;
        }

        return true;
    }

    private boolean usesTypeAsMethodParameter(HashMap classDetailContainer, String typeName) {
        if (classDetailContainer == null || typeName == null || typeName.isEmpty()) {
            return false;
        }

        for (HashMap methodDetail : Utils.getMethodDetails(classDetailContainer)) {
            for (HashMap parameter : Utils.getMethodParameters(methodDetail)) {
                if (typeName.equals(Utils.getParameterType(parameter))) {
                    return true;
                }
            }
        }

        return false;
    }
}


package common.designpatternidentifier;

import java.util.*;

import org.apache.commons.collections4.MultiValuedMap;

import dps_nlg.summarygenerator.SentenceGenerator;
import dps_nlg.summarygenerator.messages.DesignPatternClassMessage;
import dps_nlg.summarygenerator.messages.DesignPatternMethodMessage;
import common.utils.Utils;

// Singleton is a creational design pattern that lets you ensure that a 
// class has only one instance, while providing a global access point to this instance.
public class SingletonPattern extends DesignPatterns {
    public SingletonPattern() {
        super("singleton");
    }

    // Singleton:
    // All constructors must be private
    // Static creation method acting as a constructor and returns instance
    // Static private field storing instance
    @Override
    public HashMap checkPattern(HashMap<String, HashMap> fileDetails) {

        ArrayList<String> matchedClasses = new ArrayList<>();

        // For every file
        for (Map.Entry<String, HashMap> fileEntry : fileDetails.entrySet()) {

            // CN
            String classInterfaceName = fileEntry.getKey();
            // Must have at least one static field that contains object of same type as class.
            // Reason for change:
            // Some dataset examples implement Singleton with a non-private static instance field
            // (for example: "public static Singleton instance"). The previous strict condition
            // produced false negatives and prevented Singleton role summaries from being generated.
            boolean hasClassInstanceAnyVisibility = false;
            boolean hasClassInstanceStrictPrivate = false;

            // FDT, FMT
            for (HashMap fieldDetail : Utils.getFieldDetails(fileEntry.getValue())) {
                if (Utils.getFieldDataType(fieldDetail).equals(classInterfaceName)
                        && Utils.getFieldModifierType(fieldDetail).contains("static")) {
                    hasClassInstanceAnyVisibility = true;
                    if (Utils.getFieldModifierType(fieldDetail).contains("private")) {
                        hasClassInstanceStrictPrivate = true;
                    }
                    break;
                }
            }

            // Legacy strict gate kept for traceability (do not remove changed code):
            // if (!hasClassInstanceStrictPrivate)
            //     continue;

            // Updated gate: accept Singleton candidates with static self-instance field regardless
            // of visibility, while still tracking strict-private cases via hasClassInstanceStrictPrivate.
            if (!hasClassInstanceAnyVisibility)
                continue;

            // All constructors must be private. Static is optional
            boolean hasAllConstructorsPrivate = true;

            // CM
            for (HashMap constructorDetail : Utils.getConstructorDetails(fileEntry.getValue())) {
                if (!Utils.getConstructorModifier(constructorDetail).contains("private")) {
                    hasAllConstructorsPrivate = false;
                    break;
                }
            }

            if (!hasAllConstructorsPrivate)
                continue;

            // Method that creates instance must be public static
            // Must return instance of class
            boolean hasCreationMethod = false;

            // MMT, MRT
            for (HashMap methodDetail : Utils.getMethodDetails(fileEntry.getValue())) {
                if (Utils.getMethodModifierType(methodDetail).contains("public")
                        && Utils.getMethodModifierType(methodDetail).contains("static")
                        && Utils.getMethodReturnType(methodDetail).equals(classInterfaceName)) {
                    hasCreationMethod = true;
                    break;
                }
            }

            if (!hasCreationMethod)
                continue;

            matchedClasses.add(classInterfaceName);
        }

        return createPatternResult(matchedClasses);
    }

    @Override
    public void summarise(HashMap<String, HashMap> fileDetails, HashMap designPatternDetails,
            MultiValuedMap<String, String> summary) {
        for (String singleton : (ArrayList<String>) designPatternDetails.getOrDefault(patternName, new ArrayList<>())) {

            DesignPatternClassMessage cm = new DesignPatternClassMessage(singleton, patternNameAsText);
            ArrayList<DesignPatternMethodMessage> mmal = new ArrayList<>();

            HashMap classDetails = fileDetails.getOrDefault(singleton, new HashMap<>());
            for (HashMap methodDetail : Utils.getMethodDetails(classDetails)) {

                DesignPatternMethodMessage mm = new DesignPatternMethodMessage();
                String methodReturnType = Utils.getMethodReturnType(methodDetail);
                String methodName = Utils.getMethodName(methodDetail);
                ArrayList<Object> methodModifierType = new ArrayList<>();
                methodModifierType.addAll(Utils.getMethodModifierType(methodDetail));
                ArrayList<String> incomingMethod = Utils.getIncomingMethodAsText(methodDetail);

                if (methodReturnType.equals(singleton)
                        && methodModifierType.contains("public")
                        && methodModifierType.contains("static")) {
                    mm.setMethodReturn(methodReturnType);
                    mm.setMethodModifier(methodModifierType);
                    mm.setIncomingMethod(incomingMethod);

                    String[] methodNameArray = Utils.splitByCamelCase(methodName);
                    if (methodNameArray.length > 1) {
                        ArrayList<String> methodActionArray = new ArrayList<>();

                        // first word of the method name is set to be the verb, the rest is set to be
                        // the actions
                        methodActionArray.addAll(
                                Arrays.asList(methodNameArray).subList(1, methodNameArray.length));
                        String methodAction = String.join(" ", methodActionArray);

                        mm.setMethodVerb(methodNameArray[0]);
                        mm.setMethodAction(methodAction);
                    } else {
                        mm.setMethodVerb(methodNameArray[0]);
                    }
                    mmal.add(mm);
                }
            }

            SentenceGenerator singletonSentenceGenerator = new SentenceGenerator();
            String singletonSentence = singletonSentenceGenerator.generateSentence(cm, mmal, null);
            summary.put(singleton, singletonSentence);
        }
    }
}


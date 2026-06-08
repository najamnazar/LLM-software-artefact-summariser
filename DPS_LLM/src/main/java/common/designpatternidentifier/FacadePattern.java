package common.designpatternidentifier;

import java.util.*;

import org.apache.commons.collections4.MultiValuedMap;

import dps_nlg.summarygenerator.messages.DesignPatternClassMessage;
import dps_nlg.summarygenerator.messages.DesignPatternMethodMessage;
import common.utils.Utils;

// Facade is a structural design pattern that provides a simplified interface to a library, a framework, or any other complex set of classes.
public class FacadePattern extends DesignPatterns {
    public FacadePattern() {
        super("facade");
    }

    @Override
    public HashMap checkPattern(HashMap<String, HashMap> fileDetails) {
        HashMap output = new HashMap<>();
        HashMap possibleFacades = new HashMap<>();

        for (Map.Entry<String, HashMap> fileEntry : fileDetails.entrySet()) {

            // Bug 2 fix: refactoring.guru defines Facade as coordinating subsystem objects, which
            // requires creating/using those objects — typically as local variables (VARIABLEDETAIL).
            // The old check incorrectly excluded the Facade class itself. Removed.
            // if (!Utils.getVariableDetails(fileEntry.getValue()).isEmpty())
            //     continue;

            for (HashMap methodDetail : Utils.getMethodDetails(fileEntry.getValue())) {
                for (HashMap outgoingMethod : (ArrayList<HashMap>) Utils.getOutgoingMethod(methodDetail)) {

                    String outgoingMethodClass = Utils.getOutgoingMethodClass(outgoingMethod);
                    if (fileDetails.containsKey(outgoingMethodClass)
                            && !outgoingMethodClass.equals(fileEntry.getKey())) {

                        if (!possibleFacades.containsKey(fileEntry.getKey()))
                            possibleFacades.put(fileEntry.getKey(), new HashSet<String>());
                        ((Set) possibleFacades.get(fileEntry.getKey())).add(outgoingMethodClass);
                    }
                }
            }

            // Needs to call/create multiple classes
            HashMap<String, Integer> classesCalled = new HashMap<>();
            for (HashMap methodDetail : Utils.getMethodDetails(fileEntry.getValue())) {

                String outgoingMethodClass = Utils.getOutgoingMethodClass(methodDetail);
                if (fileDetails.containsKey(outgoingMethodClass) && !outgoingMethodClass.equals(fileEntry.getKey()))
                    classesCalled.compute(Utils.getOutgoingMethodClass(methodDetail),
                            (key, value) -> (value == null) ? 1 : value + 1);
            }

            if (classesCalled.size() > 1) {
                if (possibleFacades.containsKey(fileEntry.getKey()))
                    possibleFacades.put(fileEntry.getKey(), new HashSet<String>());
                for (String classCalled : classesCalled.keySet())
                    ((Set) possibleFacades.get(fileEntry.getKey())).add(classCalled);
            }
        }

        HashMap facades = new HashMap<>();
        for (Map.Entry<String, Set<String>> facadeDetails : ((HashMap<String, Set<String>>) possibleFacades)
                .entrySet()) {
            String facade = facadeDetails.getKey();
            ArrayList<String> exclusiveSubsystems = new ArrayList<>();
            for (String subsystem : facadeDetails.getValue()) {

                HashMap subsystemDetails = fileDetails.get(subsystem);
                boolean isExclusiveSubsystem = true;
                for (HashMap methodDetail : Utils.getMethodDetails(subsystemDetails)) {
                    for (HashMap incomingMethod : (ArrayList<HashMap>) Utils.getIncomingMethod(methodDetail)) {
                        if (!Utils.getIncomingMethodClass(incomingMethod).equals(facade)) {
                            isExclusiveSubsystem = false;
                            break;
                        }
                    }
                }
                if (isExclusiveSubsystem)
                    exclusiveSubsystems.add(subsystem);
            }
            if (exclusiveSubsystems.size() > 0)
                facades.put(facade, exclusiveSubsystems);
        }

        if (!facades.isEmpty()) {
            output.put(patternName, facades);
        }

        return output;

    }

    @Override
    public void summarise(HashMap<String, HashMap> fileDetails, HashMap designPatternDetails,
            MultiValuedMap<String, String> summary) {
        for (Map.Entry<String, ArrayList<String>> facadeDetails : ((HashMap<String, ArrayList<String>>) designPatternDetails
                .getOrDefault(patternName, new HashMap<>())).entrySet()) {

            // generate subsystem sentence
            String facade = facadeDetails.getKey();
            for (String subsystem : facadeDetails.getValue()) {

                HashMap classDetail = fileDetails.getOrDefault(subsystem, new HashMap<>());

                DesignPatternClassMessage sscm = new DesignPatternClassMessage(subsystem, "subsystem");
                sscm.setRelatedClassName(facade);
                sscm.setRelatedClassDesignPattern("facade");

                DesignPatternMethodMessage mm = new DesignPatternMethodMessage();
                ArrayList<DesignPatternMethodMessage> ssmmal = new ArrayList<>();

                // which ... method is called by ... method of ...
                for (HashMap methodDetail : Utils.getMethodDetails(classDetail)) {

                    mm = new DesignPatternMethodMessage();
                    ArrayList<String> incomingMethods = Utils.getIncomingMethodAsText(methodDetail);

                    if (incomingMethods.size() == 0)
                        continue;

                    String methodName = Utils.getMethodName(methodDetail);
                    mm.setCalledMethod(methodName);
                    mm.setMethodVerb("call");
                    mm.setCallerMethodArr(incomingMethods);
                    ssmmal.add(mm);
                }

                String subsystemSentence = sentenceGenerator.generateSentence(sscm, ssmmal, null);
                summary.put(subsystem, subsystemSentence);
            }

            // generate facade sentence
            String subsystemClasses = String.join(", ", facadeDetails.getValue());
            DesignPatternClassMessage fcm = new DesignPatternClassMessage(facade, patternNameAsText);
            fcm.setRelatedClassName(subsystemClasses);
            fcm.setRelatedClassDesignPattern("subsystem");

            HashMap facadeClassDetail = fileDetails.getOrDefault(facade, new HashMap<>());
            DesignPatternMethodMessage mm = new DesignPatternMethodMessage();
            ArrayList<DesignPatternMethodMessage> fmmal = new ArrayList<>();
            for (HashMap methodDetail : Utils.getMethodDetails(facadeClassDetail)) {

                mm = new DesignPatternMethodMessage();
                ArrayList<String> outgoingMethods = Utils.getOutgoingMethodAsText(methodDetail);

                if (outgoingMethods.size() == 0)
                    continue;

                String methodName = Utils.getMethodName(methodDetail);
                mm.setCallerMethod(methodName);
                mm.setMethodVerb("call");
                mm.setCalledMethodArr(outgoingMethods);
                fmmal.add(mm);
            }
            String facadeSentence = sentenceGenerator.generateSentence(fcm, fmmal, null);
            summary.put(facade, facadeSentence);
        }
    }
}


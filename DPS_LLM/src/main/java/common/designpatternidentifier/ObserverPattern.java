package common.designpatternidentifier;

import java.util.*;

import org.apache.commons.collections4.MultiValuedMap;

import dps_nlg.summarygenerator.messages.DesignPatternClassMessage;
import dps_nlg.summarygenerator.messages.DesignPatternInheritClassMessage;
import dps_nlg.summarygenerator.messages.DesignPatternMethodMessage;
import common.utils.Utils;

// Observer is a behavioral design pattern that lets you define a subscription mechanism to notify multiple objects about any events that happen to the object they’re observing.
public class ObserverPattern extends DesignPatterns {
    public ObserverPattern() {
        super("observer");
    }

    @Override
    public HashMap checkPattern(HashMap<String, HashMap> fileDetails) {

        HashMap output = new HashMap<>();
        HashMap observers = new HashMap<>();

        for (Map.Entry<String, HashMap> publisherEntry : fileDetails.entrySet()) {

            // Publisher needs to have method that has Subscriber as parameter
            String publisher = publisherEntry.getKey();

            for (HashMap methodDetail : Utils.getMethodDetails(publisherEntry.getValue())) {
                for (HashMap parameter : (ArrayList<HashMap>) Utils.getMethodParameters(methodDetail)) {

                    if (!fileDetails.containsKey(Utils.getParameterType(parameter)))
                        continue;

                    // Publisher stores array/list of Subscribers
                    String subscriber = Utils.getParameterType(parameter);
                    boolean hasField = false;
                    for (HashMap fieldDetail : Utils.getFieldDetails(publisherEntry.getValue())) {
                        if (Utils.getFieldDataType(fieldDetail).contains(subscriber))
                            hasField = true;
                    }

                    if (!hasField)
                        continue;

                    HashSet<String> concreteObservers = new HashSet<>();
                    HashSet<String> concretePublishers = new HashSet<>();
                    for (Map.Entry<String, HashMap> concreteEntry : fileDetails.entrySet()) {
                        for (HashMap classDetail : Utils.getClassOrInterfaceDetails(concreteEntry.getValue())) {
                            if (Utils.getImplementsFrom(classDetail).contains(subscriber)
                                    || Utils.getExtendsFrom(classDetail).contains(subscriber)) {
                                concreteObservers.add(concreteEntry.getKey());
                            }
                            if (Utils.getImplementsFrom(classDetail).contains(publisher)
                                    || Utils.getExtendsFrom(classDetail).contains(publisher)) {
                                concretePublishers.add(concreteEntry.getKey());
                            }

                        }
                    }

                    observers.putIfAbsent(subscriber, new HashMap<>());
                    ((HashMap) observers.get(subscriber)).putIfAbsent("publisher", new HashMap<>());
                    ((HashMap) ((HashMap) observers.get(subscriber)).get("publisher")).putIfAbsent(publisher,
                            new HashSet<>());
                    ((HashSet) ((HashMap) ((HashMap) observers.get(subscriber)).get("publisher")).get(publisher))
                            .addAll(concretePublishers);
                    ((HashMap) observers.get(subscriber)).putIfAbsent("concrete_observer", new HashSet<>());
                    ((HashSet) ((HashMap) observers.get(subscriber)).get("concrete_observer"))
                            .addAll(concreteObservers);

                    // Check if publisher itself is a concrete publisher
                    for (HashMap classDetail : Utils.getClassOrInterfaceDetails(publisherEntry.getValue())) {
                        ArrayList<String> parentClassList = Utils.getExtendsFrom(classDetail);
                        parentClassList.addAll(Utils.getImplementsFrom(classDetail));
                        for (String parent : parentClassList) {
                            if (!fileDetails.containsKey(parent))
                                continue;
                            HashMap parentDetails = fileDetails.get(parent);
                            for (HashMap parentMethodDetail : Utils.getMethodDetails(parentDetails)) {
                                for (HashMap parentParameter : (ArrayList<HashMap>) Utils.getMethodParameters(
                                        parentMethodDetail)) {
                                    if (Utils.getParameterType(parentParameter).equals(subscriber)) {
                                        ((HashMap) ((HashMap) observers.get(subscriber)).get("publisher")).putIfAbsent(
                                                parent, new HashSet<>());
                                        ((HashSet) ((HashMap) ((HashMap) observers.get(subscriber)).get("publisher"))
                                                .get(parent)).add(publisher);
                                    }
                                }
                            }
                        }
                    }

                }
            }
        }

        if (!observers.isEmpty())
            output.put(patternName, observers);
        return output;
    }

    @Override
    public void summarise(HashMap<String, HashMap> fileDetails, HashMap designPatternDetails,
            MultiValuedMap<String, String> summary) {
        for (Map.Entry<String, HashMap> observerDetails : ((HashMap<String, HashMap>) designPatternDetails
                .getOrDefault(patternName, new HashMap<>())).entrySet()) {

            String observer = observerDetails.getKey();
            String publisher = "";
            String concretePublishers = "";

            // generate publisher sentence
            for (Map.Entry<String, HashSet> publisherEntry : ((HashMap<String, HashSet>) observerDetails.getValue()
                    .getOrDefault("publisher", new HashMap<>())).entrySet()) {

                publisher = publisherEntry.getKey();
                for (String concretePublisher : (HashSet<String>) publisherEntry.getValue()) {

                    DesignPatternMethodMessage mm = new DesignPatternMethodMessage();
                    ArrayList<DesignPatternMethodMessage> csmmal = new ArrayList<>();
                    DesignPatternClassMessage cscm = new DesignPatternClassMessage(concretePublisher, "publisher");
                    cscm.setRelatedClassDesignPattern(patternNameAsText);
                    cscm.setRelatedClassName(observer);

                    HashMap classDetail = fileDetails.getOrDefault(concretePublisher, new HashMap<>());
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
                        csmmal.add(mm);
                    }

                    String concretePublisherSentence = sentenceGenerator.generateSentence(cscm, csmmal, null);
                    summary.put(concretePublisher, concretePublisherSentence);
                }

                // generate concrete publisher sentence
                concretePublishers = String.join(", ", (HashSet<String>) publisherEntry.getValue());

                DesignPatternMethodMessage mm = new DesignPatternMethodMessage();
                ArrayList<DesignPatternMethodMessage> smmal = new ArrayList<>();
                DesignPatternClassMessage scm = new DesignPatternClassMessage(publisher, "publisher");
                scm.setRelatedClassDesignPattern(patternNameAsText);
                scm.setRelatedClassName(observer);

                HashMap classDetail = fileDetails.getOrDefault(publisher, new HashMap<>());
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
                    smmal.add(mm);
                }

                DesignPatternInheritClassMessage sicm = new DesignPatternInheritClassMessage();
                sicm.setInheritClass(concretePublishers);

                String publisherSentence = sentenceGenerator.generateSentence(scm, smmal, sicm);
                summary.put(publisher, publisherSentence);
            }

            // generate concrete observer sentence
            for (String concreteObserver : (HashSet<String>) observerDetails.getValue().getOrDefault("concrete_observer", new HashSet<>())) {

                DesignPatternClassMessage cocm = new DesignPatternClassMessage(concreteObserver, patternNameAsText);
                DesignPatternMethodMessage mm = new DesignPatternMethodMessage();
                ArrayList<DesignPatternMethodMessage> commal = new ArrayList<>();

                HashMap classDetail = fileDetails.getOrDefault(concreteObserver, new HashMap<>());
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
                    commal.add(mm);
                }

                String concreteObserverSentence = sentenceGenerator.generateSentence(cocm, commal, null);
                summary.put(concreteObserver, concreteObserverSentence);
            }

            // generate observer sentence
            String concreteObservers = String.join(", ",
                    (HashSet<String>) observerDetails.getValue().getOrDefault("concrete_observer", new HashSet<>()));

            DesignPatternMethodMessage mm = new DesignPatternMethodMessage();
            ArrayList<DesignPatternMethodMessage> ommal = new ArrayList<>();
            DesignPatternClassMessage ocm = new DesignPatternClassMessage(observer, patternNameAsText);
            ocm.setRelatedClassName(publisher + ", " + concretePublishers);
            ocm.setRelatedClassDesignPattern("publisher");

            HashMap classDetail = fileDetails.getOrDefault(observer, new HashMap<>());
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
                ommal.add(mm);
            }

            DesignPatternInheritClassMessage oicm = new DesignPatternInheritClassMessage();
            oicm.setInheritClass(concreteObservers);

            String observerSentence = sentenceGenerator.generateSentence(ocm, ommal, oicm);
            summary.put(observer, observerSentence);
        }
    }
}


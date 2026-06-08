package common.designpatternidentifier;

import java.util.*;

import org.apache.commons.collections4.MultiValuedMap;

import dps_nlg.summarygenerator.messages.DesignPatternClassMessage;
import dps_nlg.summarygenerator.messages.DesignPatternInheritClassMessage;
import dps_nlg.summarygenerator.messages.DesignPatternMethodMessage;
import common.utils.Utils;

// Decorator is a structural design pattern that lets you attach new behaviors
// to objects by placing these objects inside special wrapper objects that
// contain the behaviors.
public class DecoratorPattern extends DesignPatterns {
    public DecoratorPattern() {
        super("decorator");
    }

    @Override
    public HashMap checkPattern(HashMap<String, HashMap> fileDetails) {
        HashMap output = new HashMap<>();
        HashMap decorators = new HashMap<>();

        // Find component (has to be an interface)
        for (Map.Entry<String, HashMap> fileEntry : fileDetails.entrySet()) {

            boolean hasInterface = false;
            for (HashMap classDetail : Utils.getClassOrInterfaceDetails(fileEntry.getValue())) {
                if (Utils.isInterfaceOrNot(classDetail)) {
                    hasInterface = true;
                    break;
                }
            }

            // Component has to be an interface
            if (!hasInterface)
                continue;
            String component = fileEntry.getKey();

            // Find concrete components and decorator
            ArrayList<String> possibleConcreteComponents = new ArrayList<String>();
            HashMap<String, HashMap> baseDecorator = new HashMap();
            for (Map.Entry<String, HashMap> possibleConcreteComponentOrDecorator : fileDetails.entrySet()) {

                // Both decorators and concrete components must implement component
                for (HashMap classDetail : Utils
                        .getClassOrInterfaceDetails(possibleConcreteComponentOrDecorator.getValue())) {

                    if (Utils.getImplementsFrom(classDetail).contains(component)) {
                        possibleConcreteComponents.add(possibleConcreteComponentOrDecorator.getKey());

                        // Decorator must have a non-public field of type Component
                        boolean hasField = false;
                        for (HashMap fieldDetail : Utils
                                .getFieldDetails(possibleConcreteComponentOrDecorator.getValue())) {
                            if (Utils.getFieldDataType(fieldDetail).equals(component)
                                    && !Utils.getFieldModifierType(fieldDetail).contains("public")) {
                                hasField = true;
                                break;
                            }
                        }

                        if (!hasField)
                            continue;

                        // Decorator must have a constructor with parameter containing Component
                        boolean hasConstructor = false;
                        for (HashMap constructorDetail : Utils
                                .getConstructorDetails(possibleConcreteComponentOrDecorator.getValue())) {

                            for (HashMap constructorParameter : (ArrayList<HashMap>) Utils
                                    .getConstructorParameters(constructorDetail))
                                // Bug 1 fix: refactoring.guru does not mandate public constructor visibility for
                                // Decorator. protected is valid for abstract base decorators (e.g.
                                // DataSourceDecorator). Accept public or protected.
                                // if (Utils.getParameterType(constructorParameter).equals(component)
                                //         && Utils.getConstructorModifier(constructorDetail).contains("public")) {
                                if (Utils.getParameterType(constructorParameter).equals(component)
                                        && (Utils.getConstructorModifier(constructorDetail).contains("public")
                                                || Utils.getConstructorModifier(constructorDetail).contains("protected"))) {
                                    hasConstructor = true;
                                    break;
                                }
                        }

                        if (!hasConstructor)
                            continue;
                        String decorator = possibleConcreteComponentOrDecorator.getKey();

                        // Find concrete decorators based on current found decorator
                        ArrayList<String> concreteDecorators = new ArrayList<>();
                        for (Map.Entry<String, HashMap> possibleConcreteDecorator : fileDetails.entrySet()) {

                            for (HashMap concreteDecoratorClassDetail : Utils
                                    .getClassOrInterfaceDetails(possibleConcreteDecorator.getValue())) {
                                if (Utils.getExtendsFrom(
                                        concreteDecoratorClassDetail).contains(decorator)) {
                                    concreteDecorators.add(possibleConcreteDecorator.getKey());
                                    break;
                                }
                            }
                        }

                        baseDecorator.put(decorator, new HashMap<>());
                        ((HashMap) baseDecorator.get(decorator)).put("concrete_decorator", concreteDecorators);
                        break;
                    }
                }

            }

            // Find all method names in decorators and match them to their respective
            // decorators
            HashMap<String, String> decoratorMethodNames = new HashMap<>();
            for (String decorator : baseDecorator.keySet()) {
                HashMap decoratorDetails = fileDetails.get(decorator);
                for (HashMap methodDetail : Utils.getMethodDetails(decoratorDetails)) {
                    decoratorMethodNames.put(Utils.getMethodName(methodDetail), decorator);
                }
            }

            // Find concrete components using method names
            // Methods must be overriden to decorate
            // Concrete component can not be a decorator
            ArrayList<String> concreteComponents = new ArrayList<>();
            Set<String> usedDecorators = new HashSet<>();
            for (String possibleConcreteComponent : possibleConcreteComponents) {
                if (baseDecorator.containsKey(possibleConcreteComponent))
                    continue;

                HashMap possibleConcreteComponentDetails = fileDetails.get(possibleConcreteComponent);
                for (HashMap methodDetail : Utils.getMethodDetails(possibleConcreteComponentDetails)) {
                    if (decoratorMethodNames.containsKey(Utils.getMethodName(methodDetail))) {
                        concreteComponents.add(possibleConcreteComponent);
                        usedDecorators.add(decoratorMethodNames.get(Utils.getMethodName(methodDetail)));
                    }
                }
            }

            HashMap<String, HashMap> componentMap = new HashMap<>();
            componentMap.put(component, new HashMap<>());
            componentMap.get(component).put("concrete_component", concreteComponents);

            for (String usedDecorator : usedDecorators) {
                decorators.put(usedDecorator, baseDecorator.get(usedDecorator));
                ((HashMap) baseDecorator.get(usedDecorator)).put("component", componentMap);
            }
        }

        if (!decorators.isEmpty())
            output.put(patternName, decorators);
        return output;
    }

    @Override
    public void summarise(HashMap<String, HashMap> fileDetails, HashMap designPatternDetails,
            MultiValuedMap<String, String> summary) {

        for (Map.Entry<String, HashMap> patternDetails : ((HashMap<String, HashMap>) designPatternDetails
                .getOrDefault(patternName, new HashMap<>())).entrySet()) {

            String decorator = patternDetails.getKey();
            String component = "";

            for (String concreteDecorator : (ArrayList<String>) patternDetails.getValue().getOrDefault("concrete_decorator", new ArrayList<>())) {
                HashMap classDetail = fileDetails.getOrDefault(concreteDecorator, new HashMap<>());
                HashMap parentClassDetail = fileDetails.getOrDefault(decorator, new HashMap<>());

                ArrayList<DesignPatternMethodMessage> mmal = new ArrayList<>();
                ArrayList<String> overrideMethodArray = new ArrayList<>();
                DesignPatternMethodMessage mm = new DesignPatternMethodMessage();
                DesignPatternClassMessage cm = new DesignPatternClassMessage(concreteDecorator, "decorator");
                cm.setRelatedClassName(decorator);
                cm.setRelatedClassDesignPattern(patternNameAsText);

                overrideMethodArray
                        .addAll(Utils.checkMethodOverride(classDetail, parentClassDetail, " method of " + decorator));
                if (overrideMethodArray.size() != 0) {
                    mm.setOverrideMethod(overrideMethodArray);
                    mm.setMethodVerb("override");
                    mmal.add(mm);
                }

                String concreteDecoratorSentence = sentenceGenerator.generateSentence(cm, mmal, null);
                summary.put(concreteDecorator, concreteDecoratorSentence);
            }

            // generate concrete component sentence
            HashMap<String, HashMap> componentDetails = (HashMap) patternDetails.getValue().getOrDefault("component", new HashMap<>());
            for (Map.Entry<String, HashMap> componentDetail : componentDetails.entrySet()) {
                component = componentDetail.getKey();
                for (String concreteComponent : (ArrayList<String>) componentDetail.getValue()
                        .getOrDefault("concrete_component", new ArrayList<>())) {
                    HashMap classDetail = fileDetails.getOrDefault(concreteComponent, new HashMap<>());
                    HashMap parentClassDetail = fileDetails.getOrDefault(decorator, new HashMap<>());

                    ArrayList<DesignPatternMethodMessage> mmal = new ArrayList<>();
                    ArrayList<String> overrideMethodArray = new ArrayList<>();
                    DesignPatternMethodMessage mm = new DesignPatternMethodMessage();
                    DesignPatternClassMessage cm = new DesignPatternClassMessage(concreteComponent, "component");
                    cm.setRelatedClassName(decorator);
                    cm.setRelatedClassDesignPattern(patternNameAsText);

                    overrideMethodArray.addAll(Utils.checkMethodOverride(classDetail, parentClassDetail, component));
                    if (overrideMethodArray.size() != 0) {
                        mm.setOverrideMethod(overrideMethodArray);
                        mm.setMethodVerb("override");
                        mmal.add(mm);
                    }

                    String concreteComponentSentence = sentenceGenerator.generateSentence(cm, mmal, null);
                    summary.put(concreteComponent, concreteComponentSentence);
                }

                // generate component sentence
                String concreteComponents = String.join(", ", (ArrayList<String>) componentDetail.getValue()
                        .getOrDefault("concrete_component", new ArrayList<>()));
                DesignPatternClassMessage ccm = new DesignPatternClassMessage(component, "component");
                ccm.setRelatedClassName(decorator);

                DesignPatternInheritClassMessage cicm = new DesignPatternInheritClassMessage();
                cicm.setInheritClass(concreteComponents);

                String componentSentence = sentenceGenerator.generateSentence(ccm, null, cicm);
                summary.put(component, componentSentence);
            }

            // generate decorator sentences
            String concreteDecorators = String.join(", ",
                    (ArrayList<String>) patternDetails.getValue().getOrDefault("concrete_decorator", new ArrayList<>()));

            DesignPatternClassMessage dcm = new DesignPatternClassMessage(decorator, patternNameAsText);
            dcm.setRelatedClassDesignPattern("component");
            dcm.setRelatedClassName(component);

            DesignPatternInheritClassMessage dicm = new DesignPatternInheritClassMessage();
            dicm.setInheritClass(concreteDecorators);

            String decoratorSentence = sentenceGenerator.generateSentence(dcm, null, dicm);
            summary.put(decorator, decoratorSentence);
        }
    }
}


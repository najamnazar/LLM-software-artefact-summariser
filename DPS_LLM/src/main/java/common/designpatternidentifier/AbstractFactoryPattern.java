package common.designpatternidentifier;

import java.util.*;

import org.apache.commons.collections4.MultiValuedMap;

import dps_nlg.summarygenerator.messages.DesignPatternClassMessage;
import dps_nlg.summarygenerator.messages.DesignPatternInheritClassMessage;
import dps_nlg.summarygenerator.messages.DesignPatternMethodMessage;
import common.utils.Utils;

// Abstract Factory is a creational design pattern that lets
// you produce families of related objects without specifying their concrete
// classes.
public class AbstractFactoryPattern extends DesignPatterns {

    public AbstractFactoryPattern() {
        super("abstract_factory");
    }

    // Abstract Factory:
    // Has an interface - abstract factory
    // Abstract Products created by Abstract Factory
    // Concrete Factories implement Abstract Factory
    // Concrete Products implement/extend Abstract Products
    @Override
    public HashMap checkPattern(HashMap<String, HashMap> fileDetails) {
        // System.out.print("Pattern: \t");
        // System.out.println(patternName);

        HashMap abstractFactories = new HashMap<>();

        // Find interfaces
        for (Map.Entry<String, HashMap> fileEntry : fileDetails.entrySet()) {

            // Only an interface can be an abstract factory
            boolean isInterfaceOrNot = false;

            // ION
            for (HashMap classDetail : Utils.getClassOrInterfaceDetails(fileEntry.getValue())) {
                if (Utils.isInterfaceOrNot(classDetail)) {
                    isInterfaceOrNot = true;
                    break;
                }
            }

            if (!isInterfaceOrNot)
                continue;
            String abstractFactory = fileEntry.getKey();

            // Find abstract products using abstract factory
            ArrayList<String> abstractProducts = new ArrayList<>();
            for (HashMap methodDetail : Utils.getMethodDetails(fileEntry.getValue())) {

                // Abstract Product type is "created" by Abstract Factory
                String methodReturnType = Utils.getMethodReturnType(methodDetail);
                if (fileDetails.keySet().contains(methodReturnType))
                    abstractProducts.add(methodReturnType);
            }

            if (abstractProducts.size() == 0)
                continue;

            // Find concrete factory and concrete products
            // Find abstract products that have implementations
            ArrayList<String> concreteFactories = new ArrayList<>();
            HashMap<String, ArrayList> abstractProductsWithExistingConcreteProducts = new HashMap<String, ArrayList>();
            for (Map.Entry<String, HashMap> possibleConcrete : fileDetails.entrySet()) {
                for (HashMap classDetail : Utils.getClassOrInterfaceDetails(possibleConcrete.getValue())) {

                    // Concrete Factory implements Abstract Factory
                    if (Utils.getImplementsFrom(classDetail).contains(abstractFactory))
                        concreteFactories.add(possibleConcrete.getKey());

                    // Concrete Product implements Abstract Product
                    for (String abstractProduct : abstractProducts)
                        if (Utils.getImplementsFrom(classDetail).contains(abstractProduct)
                                || Utils.getExtendsFrom(classDetail).contains(abstractProduct)) {
                            if (!abstractProductsWithExistingConcreteProducts.containsKey(abstractProduct))
                                abstractProductsWithExistingConcreteProducts.put(abstractProduct,
                                        new ArrayList<String>());
                            abstractProductsWithExistingConcreteProducts.get(abstractProduct)
                                    .add(possibleConcrete.getKey());
                        }
                }
            }

            abstractFactories.put(abstractFactory, new HashMap());
            ((HashMap) abstractFactories.get(abstractFactory)).put("abstract_product",
                    abstractProductsWithExistingConcreteProducts);
            ((HashMap) abstractFactories.get(abstractFactory)).put("concrete_factory", concreteFactories);
        }
        
        return createPatternResult(abstractFactories);
    }

    @Override
    public void summarise(HashMap<String, HashMap> fileDetails, HashMap designPatternDetails,
            MultiValuedMap<String, String> summary) {

        // Details of abstract factory
        for (Map.Entry<String, HashMap> abstractFactoryDetails : ((HashMap<String, HashMap>) designPatternDetails
                .getOrDefault(patternName, new HashMap<>())).entrySet()) {
            String abstractFactory = abstractFactoryDetails.getKey();
            ArrayList<String> concreteFactoryList = (ArrayList) abstractFactoryDetails.getValue()
                    .getOrDefault("concrete_factory", new ArrayList<>());
            HashMap<String, ArrayList<String>> abstractProductMap = (HashMap<String, ArrayList<String>>) abstractFactoryDetails
                    .getValue().getOrDefault("abstract_product", new HashMap<>());

            // Details of abstract products
            for (Map.Entry<String, ArrayList<String>> abstractProductDetails : abstractProductMap.entrySet()) {
                String abstractProduct = abstractProductDetails.getKey();
                ArrayList<String> concreteProductList = abstractProductDetails.getValue();
                for (String concreteProduct : concreteProductList) {

                    HashMap classDetail = (HashMap) fileDetails.getOrDefault(concreteProduct, new HashMap<>());
                    HashMap parentClassDetail = (HashMap) fileDetails.getOrDefault(abstractProduct, new HashMap<>());

                    DesignPatternClassMessage cm = new DesignPatternClassMessage(concreteProduct,
                            patternNameAsText);
                    ArrayList<DesignPatternMethodMessage> mmal = new ArrayList<>();
                    ArrayList<String> overrideMethodArray = new ArrayList<>();
                    DesignPatternMethodMessage mm = new DesignPatternMethodMessage();

                    overrideMethodArray.addAll(Utils.checkMethodOverride(classDetail, parentClassDetail,
                            " method of " + abstractProduct));

                    mm.setOverrideMethod(overrideMethodArray);
                    mm.setMethodVerb("override");
                    mmal.add(mm);

                    cm.setClassName(concreteProduct);
                    cm.setDesignPattern("product");
                    cm.setRelatedClassName(abstractFactory);

                    String concreteProductSentence = sentenceGenerator.generateSentence(cm, mmal, null);
                    summary.put(concreteProduct, concreteProductSentence);
                }

                DesignPatternClassMessage apcm = new DesignPatternClassMessage(abstractProduct,
                        Utils.convertToPlainText("abstract_product"));
                apcm.setRelatedClassName(abstractProduct);

                String concreteProductsAsText = String.join(", ", concreteProductList);
                DesignPatternInheritClassMessage apicm = new DesignPatternInheritClassMessage();
                apicm.setInheritClass(concreteProductsAsText);

                String abstractProductSentence = sentenceGenerator.generateSentence(apcm, null, apicm);
                summary.put(abstractProduct, abstractProductSentence);
            }

            String abstractProductsAsText = String.join(", ",
                    new ArrayList(abstractProductMap.keySet()));

            // Concrete Factory
            for (String concreteFactory : concreteFactoryList) {

                HashMap classDetail = (HashMap) fileDetails.getOrDefault(concreteFactory, new HashMap<>());
                HashMap parentClassDetail = (HashMap) fileDetails.getOrDefault(abstractFactory, new HashMap<>());

                ArrayList<DesignPatternMethodMessage> mmal = new ArrayList<>();
                ArrayList<String> overrideMethodArray = new ArrayList<>();
                DesignPatternMethodMessage mm = new DesignPatternMethodMessage();
                DesignPatternClassMessage cm = new DesignPatternClassMessage(concreteFactory,
                        patternNameAsText);

                overrideMethodArray.addAll(Utils.checkMethodOverride(classDetail, parentClassDetail,
                        " method of " + abstractFactory));

                mm.setOverrideMethod(overrideMethodArray);
                mm.setMethodVerb("override");
                mmal.add(mm);

                cm.setClassName(concreteFactory);
                cm.setDesignPattern("factory");
                cm.setRelatedClassName(abstractFactory);

                String concreteFactorySentence = sentenceGenerator.generateSentence(cm, mmal, null);
                summary.put(concreteFactory, concreteFactorySentence);
            }

            String concreteFactoriesAsText = String.join(", ", concreteFactoryList);

            // For abstract factory
            DesignPatternClassMessage afcm = new DesignPatternClassMessage(abstractFactory,
                    patternNameAsText);
            afcm.setRelatedClassName(abstractProductsAsText);

            DesignPatternInheritClassMessage aficm = new DesignPatternInheritClassMessage();
            aficm.setInheritClass(concreteFactoriesAsText);

            String abstractFactorySentence = sentenceGenerator.generateSentence(afcm, null, aficm);
            summary.put(abstractFactory, abstractFactorySentence);
        }
    }

}

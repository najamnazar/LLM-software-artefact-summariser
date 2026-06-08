package dps_nlg.summarygenerator;

import simplenlg.framework.CoordinatedPhraseElement;
import simplenlg.framework.NLGFactory;
import simplenlg.phrasespec.*;
import simplenlg.realiser.english.Realiser;
import common.utils.Utils;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;

/**
 * Generates natural language descriptions for Java classes and interfaces.
 * <p>
 * This class uses the SimpleNLG framework to create human-readable descriptions
 * of class structures, including their modifiers, inheritance relationships,
 * and implemented interfaces. It also incorporates design pattern information
 * into the generated descriptions.
 * </p>
 * <p>
 * Key responsibilities:
 * <ul>
 *   <li>Generate complete class descriptions in natural language</li>
 *   <li>Describe inheritance hierarchies (extends relationships)</li>
 *   <li>Describe interface implementations</li>
 *   <li>Incorporate design pattern roles into descriptions</li>
 *   <li>Handle both classes and interfaces appropriately</li>
 * </ul>
 * </p>
 * 
 * @author Najam
 */
public class ClassInterfaceSummariser {
    /**
     * Generates a complete natural language description of a Java class or interface.
     * <p>
     * Creates a grammatically correct sentence describing the class/interface type,
     * modifiers, inheritance relationships, and design pattern roles. The description
     * integrates information from the parsed class details and detected design patterns.
     * </p>
     * 
     * @param nlgFactory the SimpleNLG factory for creating linguistic elements
     * @param realiser the SimpleNLG realiser for converting phrases to text
     * @param classDetail a map containing parsed class metadata (modifiers, name, extends, implements)
     * @param designPatternDescriptionCollect a set of design pattern role descriptions for this class
     * @return a complete natural language description of the class or interface
     */
    public String generateClassDescription(NLGFactory nlgFactory, Realiser realiser, HashMap classDetail,
            HashSet<String> designPatternDescriptionCollect) {

        // retrieve the details from the json file
        String classModifier = retrieveModifiers(classDetail);
        String className = Utils.getClassName(classDetail);
        String classExtends = retrieveExtends(classDetail);
        String classImplements = retrieveImplements(classDetail);
        boolean isInterfaceOrNot = Utils.isInterfaceOrNot(classDetail);

        SPhraseSpec classDescription = nlgFactory.createClause();
        VPPhraseSpec verbBe = nlgFactory.createVerbPhrase("be");

        NPPhraseSpec classType;

        // if it is an interface
        if (isInterfaceOrNot) {
            classType = nlgFactory.createNounPhrase("interface");
        } else {
            classType = nlgFactory.createNounPhrase("class");
        }

        // add details to the sentence
        classType.addPreModifier(classModifier);
        classType.setDeterminer("a");
        classDescription.setSubject("It");
        classDescription.setVerb(verbBe);
        classDescription.setObject(classType);

        CoordinatedPhraseElement implementsAndExtends = nlgFactory.createCoordinatedPhrase();

        // if the class is extended from another class, add the base class
        if (!classExtends.equals("")) {
            SPhraseSpec classExtendsPhrase = nlgFactory.createClause();

            VPPhraseSpec verbExtend = nlgFactory.createVerbPhrase("extend");
            NPPhraseSpec objectExtend = nlgFactory.createNounPhrase(classExtends);

            classExtendsPhrase.setVerb(verbExtend);
            classExtendsPhrase.setObject(objectExtend);
            implementsAndExtends.addCoordinate(classExtendsPhrase);
        }

        // if the class implements an interface, add the interface details
        if (!classImplements.equals("")) {
            SPhraseSpec classImplementsPhrase = nlgFactory.createClause();

            VPPhraseSpec verbImplement = nlgFactory.createVerbPhrase("implement");
            NPPhraseSpec objectImplement = nlgFactory.createNounPhrase(classImplements);

            classImplementsPhrase.setVerb(verbImplement);
            classImplementsPhrase.setObject(objectImplement);
            implementsAndExtends.addCoordinate(classImplementsPhrase);
        }

        classDescription.addComplement(implementsAndExtends);

        String classDescriptionSentence = realiser.realiseSentence(classDescription);

        // add design pattern description
        String designPatternDescriptions = String.join(" ", designPatternDescriptionCollect);
        // Bug fix: when no role was detected, previously said "X does not have any design pattern."
        // That is a false negative — CheckPattern may simply not have detected the role (e.g., concrete
        // elements in Visitor, or whole patterns like Singleton/Decorator that CheckPattern misses).
        // Returning only the structural description makes no claim either way, which is accurate for
        // both genuinely role-less classes (Test.java) and participants whose role wasn't detected.
        // if (designPatternDescriptions.equals("")) {
        //     designPatternDescriptions = className + " does not have any design pattern. ";
        // }
        // return designPatternDescriptions + " " + classDescriptionSentence;
        if (designPatternDescriptions.equals("")) {
            return classDescriptionSentence;
        }

        return designPatternDescriptions + " " + classDescriptionSentence;
    }

    /**
     * Retrieves the base class name from which this class extends.
     * 
     * @param classDetail a map containing parsed class metadata
     * @return the name of the base class, or empty string if none
     */
    private String retrieveExtends(HashMap classDetail) {
        ArrayList<String> classExtendsArray = Utils.getExtendsFrom(classDetail);
        if (classExtendsArray.size() != 0)
            return classExtendsArray.get(0);
        return "";
    }

    /**
     * Retrieves and concatenates all class modifiers (e.g., public, final, abstract).
     * 
     * @param classDetail a map containing parsed class metadata
     * @return a space-separated string of modifiers
     */
    private String retrieveModifiers(HashMap classDetail) {
        StringBuilder modifiers = new StringBuilder();
        for (String modifier : Utils.getClassModifierType(classDetail))
            modifiers.append(modifier).append(" ");
        return modifiers.toString();
    }

    /**
     * Retrieves and formats the interfaces implemented by this class.
     * <p>
     * Formats multiple interfaces with proper grammar: single interface as-is,
     * two interfaces with "and", three or more with commas and "and" before the last.
     * </p>
     * 
     * @param classDetail a map containing parsed class metadata
     * @return a grammatically correct string listing implemented interfaces, or empty string if none
     */
    private String retrieveImplements(HashMap classDetail) {
        StringBuilder interfaces = new StringBuilder();
        ArrayList<String> interfaceArray = Utils.getImplementsFrom(classDetail);

        if (interfaceArray.size() == 1) {
            return interfaceArray.get(0);
        } else if (interfaceArray.size() == 2) {
            interfaces = new StringBuilder(interfaceArray.get(0) + " and " + interfaceArray.get(1));
            return interfaces.toString();
        } else if (interfaceArray.size() > 2) {
            interfaces.append(interfaceArray.get(0));
            for (int i = 1; i < interfaceArray.size() - 1; i++) {
                interfaces.append(", ").append(interfaceArray.get(i));
            }
            interfaces.append(" and ").append(interfaceArray.get(interfaceArray.size() - 1));
            return interfaces.toString();
        }
        return "";
    }
}


package dps_nlg.summarygenerator;

import simplenlg.features.Feature;
import simplenlg.framework.CoordinatedPhraseElement;
import simplenlg.framework.NLGElement;
import simplenlg.framework.NLGFactory;
import simplenlg.lexicon.Lexicon;
import simplenlg.phrasespec.NPPhraseSpec;
import simplenlg.phrasespec.PPPhraseSpec;
import simplenlg.phrasespec.SPhraseSpec;
import simplenlg.phrasespec.VPPhraseSpec;
import simplenlg.realiser.english.Realiser;
import dps_nlg.summarygenerator.messages.DesignPatternClassMessage;
import dps_nlg.summarygenerator.messages.DesignPatternInheritClassMessage;
import dps_nlg.summarygenerator.messages.DesignPatternMethodMessage;

import java.util.ArrayList;

/**
 * Generates natural language sentences from design pattern messages.
 * <p>
 * This class uses the SimpleNLG library to transform structured design pattern
 * information into grammatically correct English sentences. It processes class-level,
 * method-level, and inheritance messages to create comprehensive textual descriptions
 * of design pattern implementations.
 * </p>
 * <p>
 * Key responsibilities:
 * <ul>
 *   <li>Generate sentences describing class roles in design patterns</li>
 *   <li>Create method descriptions with actions and relationships</li>
 *   <li>Describe inheritance relationships in pattern context</li>
 *   <li>Merge multiple phrases into coordinated sentences</li>
 *   <li>Use SimpleNLG for grammatically correct output</li>
 * </ul>
 * </p>
 * 
 * @author Najam
 */
public class SentenceGenerator {
    private Lexicon lexicon = Lexicon.getDefaultLexicon();
    private NLGFactory nlgFactory = new NLGFactory(lexicon);
    private Realiser realiser = new Realiser(lexicon);

    public SPhraseSpec handleDesignPatternClass(DesignPatternClassMessage cm) {
        String className = cm.getClassName();
        String designPattern = cm.getDesignPattern();
        String relatedClassName = cm.getRelatedClassName();
        String relatedClassDesignPattern = cm.getRelatedClassDesignPattern();

        SPhraseSpec designPatternDescription = nlgFactory.createClause();

        NPPhraseSpec subject = nlgFactory.createNounPhrase(className);
        VPPhraseSpec verbAct = nlgFactory.createVerbPhrase("act");
        PPPhraseSpec ppAs = nlgFactory.createPrepositionPhrase("as");
        NPPhraseSpec object = nlgFactory.createNounPhrase(designPattern);

        PPPhraseSpec ppFor = nlgFactory.createPrepositionPhrase("for");

        object.setDeterminer("a");
        ppAs.addComplement(object);

        if (relatedClassName != null) {
            NPPhraseSpec relatedClass = nlgFactory.createNounPhrase(relatedClassName);

            if (relatedClassDesignPattern != null) {
                relatedClass.addPreModifier(relatedClassDesignPattern);
            }
            ppFor.addComplement(relatedClass);
            object.addPostModifier(ppFor);
        }

        designPatternDescription.setSubject(subject);
        designPatternDescription.setVerb(verbAct);
        designPatternDescription.setObject(ppAs);

        return designPatternDescription;
    }

    public ArrayList<SPhraseSpec> handleDesignPatternMethod(ArrayList<DesignPatternMethodMessage> mmal) {
        ArrayList<SPhraseSpec> designPatternMethodSPhraseArray = new ArrayList<>();

        for (DesignPatternMethodMessage mm : mmal) {

            SPhraseSpec designPatternMethodDescription = nlgFactory.createClause();
            PPPhraseSpec ppTo = nlgFactory.createPrepositionPhrase("to");

            if (mm.getMethodVerb() != null) {
                VPPhraseSpec methodVerb = nlgFactory.createVerbPhrase(mm.getMethodVerb());
                designPatternMethodDescription.setVerb(methodVerb);
            }

            if (mm.getMethodReturn() != null) {
                NPPhraseSpec methodReturn = nlgFactory.createNounPhrase(mm.getMethodReturn());
                designPatternMethodDescription.setObject(methodReturn);
            }

            if (mm.getMethodAction() != null) {
                NPPhraseSpec object = nlgFactory.createNounPhrase(mm.getMethodAction());
                object.setPreModifier(mm.getMethodReturn());
                object.setDeterminer("a");

                object.addComplement(ppTo);
                designPatternMethodDescription.setObject(object);
            }

            if (mm.getOverrideMethod() != null) {
                VPPhraseSpec methodVerb = nlgFactory.createVerbPhrase(mm.getMethodVerb());
                String object = String.join(", ", mm.getOverrideMethod());
                designPatternMethodDescription.setVerb(methodVerb);
                designPatternMethodDescription.setObject(object);
            }

            if (mm.getIncomingMethod() != null) {
                String complementObjectStr = String.join(", ", mm.getIncomingMethod());
                NPPhraseSpec complementObject = nlgFactory.createNounPhrase(complementObjectStr);
                ppTo.addComplement(complementObject);
            }

            if (mm.getMethodName() != null) {
                VPPhraseSpec methodVerb = nlgFactory.createVerbPhrase(mm.getMethodVerb());
                NPPhraseSpec methodName = nlgFactory.createNounPhrase(mm.getMethodName());

                designPatternMethodDescription.setVerb(methodVerb);
                designPatternMethodDescription.setObject(methodName);
            }

            if (mm.getTarget() != null) {
                VPPhraseSpec methodVerb = nlgFactory.createVerbPhrase(mm.getMethodVerb());
                NPPhraseSpec object = nlgFactory.createNounPhrase();
                if (mm.getMethodName() != null) {
                    object = nlgFactory.createNounPhrase(mm.getMethodName());
                }

                if (mm.getParameter() != null) {
                    String parameters = String.join(", ", mm.getParameter());
                    object = nlgFactory.createNounPhrase(parameters);
                }

                NPPhraseSpec complementObject = nlgFactory.createNounPhrase(mm.getTarget());

                methodVerb.setPostModifier(object);
                complementObject.setPreModifier(ppTo);

                designPatternMethodDescription.setVerb(methodVerb);
                designPatternMethodDescription.setObject(complementObject);
            }

            if (mm.getParameter() != null) {
                VPPhraseSpec methodVerb = nlgFactory.createVerbPhrase(mm.getMethodVerb());
                String parameters = String.join(", ", mm.getParameter());

                NPPhraseSpec object = nlgFactory.createNounPhrase(parameters);

                if (mm.getMethodAction() != null) {
                    NPPhraseSpec complementObject = nlgFactory.createNounPhrase(mm.getMethodAction());
                    object.setPostModifier(complementObject);
                }

                designPatternMethodDescription.setVerb(methodVerb);
                designPatternMethodDescription.setObject(object);
            }

            if (mm.getCalledMethod() != null && mm.getCallerMethodArr() != null) {
                VPPhraseSpec methodVerb = nlgFactory.createVerbPhrase(mm.getMethodVerb());

                String callerMethods = String.join(", ", mm.getCallerMethodArr());
                String calledMethod = mm.getCalledMethod();

                NPPhraseSpec subject = nlgFactory.createNounPhrase(callerMethods);

                designPatternMethodDescription.setVerb(methodVerb);
                methodVerb.addPreModifier(subject);
                designPatternMethodDescription.setObject(calledMethod);

            }

            if (mm.getCalledMethodArr() != null && mm.getCallerMethod() != null) {
                VPPhraseSpec methodVerb = nlgFactory.createVerbPhrase(mm.getMethodVerb());

                String calledMethods = String.join(", ", mm.getCalledMethodArr());
                String callerMethod = mm.getCallerMethod();

                NPPhraseSpec object = nlgFactory.createNounPhrase(callerMethod);

                designPatternMethodDescription.setVerb(methodVerb);
                methodVerb.addPreModifier(object);
                designPatternMethodDescription.setObject(calledMethods);
            }
            designPatternMethodSPhraseArray.add(designPatternMethodDescription);
        }

        return designPatternMethodSPhraseArray;
    }

    public SPhraseSpec handleDesignPatternInheritClassMessage(DesignPatternInheritClassMessage icm) {
        String inheritClass = icm.getInheritClass();
        SPhraseSpec designPatternInheritClassDescription = nlgFactory.createClause();

        NPPhraseSpec object = nlgFactory.createNounPhrase(inheritClass);
        VPPhraseSpec verbInherit = nlgFactory.createVerbPhrase("inherit");

        verbInherit.addPostModifier("by");
        verbInherit.setFeature(Feature.PASSIVE, true);

        designPatternInheritClassDescription.setVerb(verbInherit);
        designPatternInheritClassDescription.setObject(object);

        return designPatternInheritClassDescription;
    }

    public String generateSentence(DesignPatternClassMessage cm, ArrayList<DesignPatternMethodMessage> mmal,
            DesignPatternInheritClassMessage icm) {
        SPhraseSpec designPatternDescription;
        ArrayList<SPhraseSpec> designPatternMethodSPhraseArray = new ArrayList<>();
        SPhraseSpec designPatternInheritClassDescription;

        designPatternDescription = handleDesignPatternClass(cm);

        if (mmal != null) {
            designPatternMethodSPhraseArray = handleDesignPatternMethod(mmal);
        }

        NLGElement newPhrase = designPatternDescription;

        if (icm != null) {
            designPatternInheritClassDescription = handleDesignPatternInheritClassMessage(icm);
            newPhrase = mergeSentence(newPhrase, designPatternInheritClassDescription, ", which");
        }

        for (SPhraseSpec designPatternMethodSPhrase : designPatternMethodSPhraseArray) {
            newPhrase = mergeSentence(newPhrase, designPatternMethodSPhrase, ", which");
        }

        String sentence = realiser.realiseSentence(newPhrase);
        return sentence;
    }

    private NLGElement mergeSentence(NLGElement nlgElement, SPhraseSpec phraseSpec, String conjunctionWord) {
        SPhraseSpec clause = nlgFactory.createClause();
        clause.setVerb(phraseSpec.getVerb());
        clause.setObject(phraseSpec.getObject());

        CoordinatedPhraseElement coordinatedPhraseElement = nlgFactory.createCoordinatedPhrase();
        coordinatedPhraseElement.addCoordinate(nlgElement);
        coordinatedPhraseElement.addCoordinate(clause);
        coordinatedPhraseElement.setConjunction(conjunctionWord);

        return coordinatedPhraseElement;
    }
}


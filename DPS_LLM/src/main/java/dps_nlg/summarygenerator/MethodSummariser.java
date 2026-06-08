package dps_nlg.summarygenerator;

import simplenlg.features.Feature;
import simplenlg.features.NumberAgreement;
import simplenlg.framework.CoordinatedPhraseElement;
import simplenlg.framework.DocumentElement;
import simplenlg.framework.NLGElement;
import simplenlg.framework.NLGFactory;
import simplenlg.phrasespec.NPPhraseSpec;
import simplenlg.phrasespec.PPPhraseSpec;
import simplenlg.phrasespec.SPhraseSpec;
import simplenlg.phrasespec.VPPhraseSpec;
import simplenlg.realiser.english.Realiser;
import common.utils.Utils;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;

/**
 * Generates natural language summaries for Java methods.
 * <p>
 * This class analyzes method call relationships and creates human-readable descriptions
 * using SimpleNLG. It processes incoming/outgoing method calls to explain how methods
 * interact within a codebase, generating both detailed usage descriptions and concise
 * summaries.
 * </p>
 * <p>
 * Key responsibilities:
 * <ul>
 *   <li>Generate method usage descriptions showing call relationships</li>
 *   <li>Create summaries listing all methods in a class</li>
 *   <li>Process incoming and outgoing method call information</li>
 *   <li>Format method signatures with return types</li>
 *   <li>Use SimpleNLG for grammatically correct output</li>
 * </ul>
 * </p>
 * 
 * @author Najam
 */
public class MethodSummariser {
    public String generateMethodDescription(NLGFactory nlgFactory, Realiser realiser,
            ArrayList<HashMap> methodDetails) {

        List<DocumentElement> methodDescriptions = new ArrayList<DocumentElement>();

        for (HashMap method : methodDetails) {
            NLGElement methodUsageDescription = generateMethodUsage(nlgFactory, realiser, method);
            DocumentElement methodDescriptionSentence = nlgFactory.createSentence(methodUsageDescription);
            methodDescriptions.add(methodDescriptionSentence);
        }
        DocumentElement methodDescriptionParagraph = nlgFactory.createParagraph(methodDescriptions);
        String methodDescription = realiser.realise(methodDescriptionParagraph).getRealisation();

        return methodDescription;
    }

    // summarise the usage of methods, look into the incoming/outgoing methods, as
    // well as the classes that these method belongs to
    public NLGElement generateMethodUsage(NLGFactory nlgFactory, Realiser realiser, HashMap methodDetail) {

        NPPhraseSpec methodName = nlgFactory.createNounPhrase();
        methodName.setNoun(Utils.getMethodName(methodDetail).toString());

        SPhraseSpec methodOutgoing = nlgFactory.createClause();
        VPPhraseSpec verbCall = nlgFactory.createVerbPhrase("call");
        methodOutgoing.setSubject(methodName);
        methodOutgoing.setVerb(verbCall);

        /* Outgoing methods */
        CoordinatedPhraseElement outgoingList = nlgFactory.createCoordinatedPhrase();

        ArrayList<HashMap> outgoingMethodsArray = Utils.getOutgoingMethod(methodDetail);
        HashSet<HashMap> outgoingMethodsSet = new HashSet<>(outgoingMethodsArray);
        Integer numOutgoing = outgoingMethodsSet.size();
        NPPhraseSpec outgoingCount = nlgFactory.createNounPhrase();
        outgoingCount.setNoun("method");

        if (numOutgoing == 0) {
            outgoingCount.setFeature(Feature.NUMBER, NumberAgreement.PLURAL);
            outgoingCount.addPreModifier("no");
        } else if (numOutgoing > 1) {
            outgoingCount.setFeature(Feature.NUMBER, NumberAgreement.PLURAL);
            outgoingCount.addPreModifier(numOutgoing.toString());
            outgoingList.addPreModifier(":");

        } else {
            outgoingCount.addPreModifier("only one");
            outgoingList.addPreModifier(":");
        }

        /* actual list of methods */
        for (HashMap outgoing : outgoingMethodsSet) {
            NPPhraseSpec outgoingName = nlgFactory.createNounPhrase();

            String outgoingMethodName = Utils.getOutgoingMethodName(outgoing).toString();
            String outgoingClassName = Utils.getOutgoingMethodClass(outgoing).toString();

            outgoingName.setNoun(outgoingMethodName + " method of class " + outgoingClassName);
            outgoingList.addCoordinate(outgoingName);
        }

        outgoingCount.addPostModifier(outgoingList);
        methodOutgoing.setObject(outgoingCount);

        /* coordinate clause: Incoming methods */
        SPhraseSpec methodIncoming = nlgFactory.createClause();
        methodIncoming.setObject(methodName);
        methodIncoming.setVerb(verbCall);

        CoordinatedPhraseElement incomingList = nlgFactory.createCoordinatedPhrase();

        ArrayList<HashMap> incomingMethodsArray = Utils.getIncomingMethod(methodDetail);
        HashSet<HashMap> incomingMethodsSet = new HashSet<>(incomingMethodsArray);
        Integer numIncoming = incomingMethodsSet.size();
        NPPhraseSpec incomingCount = nlgFactory.createNounPhrase();
        incomingCount.setNoun("method");

        if (numIncoming == 0) {
            incomingCount.setFeature(Feature.NUMBER, NumberAgreement.PLURAL);
            incomingCount.addPreModifier("no");
        } else if (numIncoming > 1) {
            incomingCount.setFeature(Feature.NUMBER, NumberAgreement.PLURAL);
            incomingCount.addPreModifier(numIncoming.toString());
            incomingList.addPreModifier("(");
            incomingList.addPostModifier(")");
        } else {
            incomingCount.addPreModifier("only one");
            incomingList.addPreModifier("(");
            incomingList.addPostModifier(")");
        }

        /* actual list of methods */
        for (HashMap incoming : incomingMethodsSet) {
            NPPhraseSpec incomingName = nlgFactory.createNounPhrase();

            String incomingMethodName = Utils.getIncomingMethodName(incoming).toString();
            String incomingClassName = Utils.getIncomingMethodClass(incoming).toString();

            incomingName.setNoun(incomingMethodName + " method of class " + incomingClassName);
            incomingList.addCoordinate(incomingName);
        }
        incomingCount.addPostModifier(incomingList);
        methodIncoming.setSubject(incomingCount);

        CoordinatedPhraseElement methodUsage = nlgFactory.createCoordinatedPhrase();
        methodUsage.addCoordinate(methodIncoming);
        methodUsage.addCoordinate(methodOutgoing);

        return methodUsage;
    }

    /* generates a short summary of all the methods of a class */
    public String generateMethodsSummary(NLGFactory nlgFactory, Realiser realiser, ArrayList<HashMap> methodDetails,
            String className) {

        /* Method summary */
        SPhraseSpec methodSummary = nlgFactory.createClause();
        VPPhraseSpec verbBe = nlgFactory.createVerbPhrase("be");
        NPPhraseSpec methodCount = nlgFactory.createNounPhrase();
        CoordinatedPhraseElement methodList = nlgFactory.createCoordinatedPhrase();

        /* Subject: the ... methods of ... */
        Integer numMethods = methodDetails.size();

        methodCount.setDeterminer("the");
        methodCount.setNoun("method");

        if (numMethods > 1) {
            methodCount.setFeature(Feature.NUMBER, NumberAgreement.PLURAL);
            methodCount.addPreModifier(numMethods.toString());
        } else {
            methodCount.addPreModifier("only");
        }

        PPPhraseSpec genitiveClass = nlgFactory.createPrepositionPhrase("of");
        genitiveClass.addComplement(className);
        methodCount.addComplement(genitiveClass);

        for (HashMap methodDetail : methodDetails) {
            NPPhraseSpec methodName = nlgFactory.createNounPhrase();

            methodName.setNoun(Utils.getMethodName(methodDetail).toString());
            methodName.addPostModifier("(" + Utils.getMethodReturnType(methodDetail) + ")");

            methodList.addCoordinate(methodName);

        }
        methodSummary.setSubject(methodCount);
        methodSummary.setVerb(verbBe);
        methodSummary.setObject(methodList);
        String methodSummaryOutput = realiser.realiseSentence(methodSummary);

        return methodSummaryOutput;
    }

}


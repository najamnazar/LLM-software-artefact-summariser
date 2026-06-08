package dps_nlg.summarygenerator;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

import org.apache.commons.collections4.MultiValuedMap;
import org.apache.commons.collections4.multimap.HashSetValuedHashMap;

import common.designpatternidentifier.*;

/**
 * Simplified design pattern summarizer that uses the centralized PatternRegistry.
 * 
 * @author Najam
 */
public class DesignPatternSummarise {
    private HashMap<String, DesignPatterns> patternList;

    public DesignPatternSummarise() {
        patternList = PatternRegistry.getPatternMap();
    }

    public void summarise(HashMap<String, HashMap> fileDetails,
            ArrayList<HashMap> designPatternDetails, HashMap<String, MultiValuedMap<String, String>> summary) {
        for (HashMap<String, Object> identifiedDesignPattern : designPatternDetails)
            for (Map.Entry<String, Object> designPatternEntry : identifiedDesignPattern.entrySet()) {
                summary.put(designPatternEntry.getKey(), new HashSetValuedHashMap<String, String>());
                patternList.get(designPatternEntry.getKey()).summarise(fileDetails,
                        identifiedDesignPattern, summary.get(designPatternEntry.getKey()));
            }
    }
}


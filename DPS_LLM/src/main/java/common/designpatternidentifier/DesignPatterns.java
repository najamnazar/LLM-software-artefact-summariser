package common.designpatternidentifier;

import java.util.*;
import org.apache.commons.collections4.MultiValuedMap;

import dps_nlg.summarygenerator.SentenceGenerator;
import common.utils.Utils;

public abstract class DesignPatterns {
    protected String patternName;
    protected String patternNameAsText;

    protected SentenceGenerator sentenceGenerator = new SentenceGenerator();

    public DesignPatterns(String patternName) {
        this.patternName = patternName;
        this.patternNameAsText = Utils.convertToPlainText(patternName);
    }

    public String getPatternName() {
        return patternName;
    }

    public abstract HashMap checkPattern(HashMap<String, HashMap> fileDetails);

    public abstract void summarise(HashMap<String, HashMap> fileDetails,
            HashMap designPatternDetails, MultiValuedMap<String, String> summary);
    
    /**
     * Helper method to create result HashMap for checkPattern methods.
     * Eliminates duplication across all pattern classes.
     */
    protected HashMap<String, Object> createPatternResult(Object patternData) {
        HashMap<String, Object> result = new HashMap<>();
        if (patternData != null) {
            // Check if the data structure is empty based on its type
            boolean isEmpty = false;
            if (patternData instanceof Collection) {
                isEmpty = ((Collection<?>) patternData).isEmpty();
            } else if (patternData instanceof Map) {
                isEmpty = ((Map<?, ?>) patternData).isEmpty();
            }
            
            if (!isEmpty) {
                result.put(patternName, patternData);
            }
        }
        return result;
    }
}


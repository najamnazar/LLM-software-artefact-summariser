package common.designpatternidentifier;

import java.util.*;

/**
 * Simplified pattern checker that uses the centralized PatternRegistry.
 */
public class CheckPattern {
    
    /**
     * Extracts design patterns from file details using all registered patterns.
     */
    @SuppressWarnings("unchecked")
    public void extractDesignPattern(HashMap fileDetails, ArrayList designPatternArrayList) {
        for (DesignPatterns pattern : PatternRegistry.getAllPatterns()) {
            HashMap output = pattern.checkPattern(fileDetails);
            if (!output.isEmpty()) {
                designPatternArrayList.add(output);
            }
        }
    }
}


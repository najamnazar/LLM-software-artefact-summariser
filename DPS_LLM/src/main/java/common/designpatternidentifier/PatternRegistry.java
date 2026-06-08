package common.designpatternidentifier;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

/**
 * Centralized registry for all design pattern instances.
 * Eliminates duplication between CheckPattern and DesignPatternSummarise.
 */
public class PatternRegistry {
    private static final List<DesignPatterns> PATTERN_INSTANCES = createPatternInstances();
    
    /**
     * Creates all design pattern instances in one place.
     * This is the single source of truth for available patterns.
     */
    private static List<DesignPatterns> createPatternInstances() {
        List<DesignPatterns> patterns = new ArrayList<>();
        patterns.add(new SingletonPattern());
        patterns.add(new FactoryPattern());
        patterns.add(new AbstractFactoryPattern());
        patterns.add(new AdapterPattern());
        patterns.add(new DecoratorPattern());
        patterns.add(new VisitorPattern());
        patterns.add(new FacadePattern());
        patterns.add(new ObserverPattern());
        patterns.add(new MementoPattern());
        return patterns;
    }
    
    /**
     * Returns a list of all pattern instances for checking patterns.
     */
    public static List<DesignPatterns> getAllPatterns() {
        return new ArrayList<>(PATTERN_INSTANCES);
    }
    
    /**
     * Returns a map of pattern name to pattern instance for summarization.
     */
    public static HashMap<String, DesignPatterns> getPatternMap() {
        HashMap<String, DesignPatterns> patternMap = new HashMap<>();
        for (DesignPatterns pattern : PATTERN_INSTANCES) {
            patternMap.put(pattern.getPatternName(), pattern);
        }
        return patternMap;
    }
}

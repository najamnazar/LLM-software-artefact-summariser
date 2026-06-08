package dps_swum.swum.context;

/**
 * Holds design pattern context information specific to a method.
 * <p>
 * This class links method behavior to design pattern responsibilities, enabling
 * pattern-aware summary generation. It captures how a specific method participates
 * in design pattern implementations and what role it plays.
 * </p>
 * <p>
 * Key responsibilities:
 * <ul>
 *   <li>Store method-level pattern information</li>
 *   <li>Link methods to their pattern behaviors</li>
 *   <li>Track related classes in pattern relationships</li>
 *   <li>Generate natural language descriptions of pattern behavior</li>
 * </ul>
 * </p>
 * 
 * @author Najam
 */
public class MethodPatternContext {
    
    private String methodName;
    private String pattern;
    private String patternBehavior;
    private String relatedClass;
    
    /**
     * Constructs a new method pattern context for the specified method.
     * 
     * @param methodName the name of the method
     */
    public MethodPatternContext(String methodName) {
        this.methodName = methodName;
    }
    
    public String getMethodName() {
        return methodName;
    }
    
    public String getPattern() {
        return pattern;
    }
    
    public void setPattern(String pattern) {
        this.pattern = pattern;
    }
    
    public String getPatternBehavior() {
        return patternBehavior;
    }
    
    public void setPatternBehavior(String patternBehavior) {
        this.patternBehavior = patternBehavior;
    }
    
    public String getRelatedClass() {
        return relatedClass;
    }
    
    public void setRelatedClass(String relatedClass) {
        this.relatedClass = relatedClass;
    }
    
    /**
     * Checks if this context has pattern-specific information.
     * 
     * @return true if both pattern name and behavior are set
     */
    public boolean hasPatternContext() {
        return pattern != null && patternBehavior != null;
    }
    
    /**
     * Generates a natural language description of the method's pattern behavior.
     * <p>
     * Creates a sentence describing what this method does in the context of
     * its design pattern role.
     * </p>
     * 
     * @return behavior description, or null if no pattern context available
     */
    public String generateBehaviorDescription() {
        if (!hasPatternContext()) {
            return null;
        }
        
        StringBuilder desc = new StringBuilder();
        desc.append("This method ").append(patternBehavior);
        
        if (relatedClass != null) {
            desc.append(" for ").append(relatedClass);
        }
        
        desc.append(" as part of the ").append(pattern).append(" pattern");
        
        return desc.toString();
    }
}

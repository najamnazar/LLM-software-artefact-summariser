package dps_nlg.summarygenerator.messages;

/**
 * Represents design pattern role information for a Java class.
 * <p>
 * This message object encapsulates the relationship between a class and its
 * design pattern role, including relationships with other classes that participate
 * in the same pattern. It is used during NLG summary generation to describe
 * how classes interact within design pattern implementations.
 * </p>
 * <p>
 * Key information stored:
 * <ul>
 *   <li>Class name and its design pattern role</li>
 *   <li>Related class names and their pattern roles</li>
 *   <li>Pattern relationships and collaborations</li>
 * </ul>
 * </p>
 * 
 * @author Najam
 */
public class DesignPatternClassMessage extends Message {
    private String className;
    private String designPattern;
    private String relatedClassName;
    private String relatedClassDesignPattern;

    /**
     * Constructs a new design pattern class message.
     * 
     * @param className the name of the class
     * @param designPattern the design pattern role of this class
     */
    public DesignPatternClassMessage(String className, String designPattern) {
        this.className = className;
        this.designPattern = designPattern;
    }

    /**
     * Gets the class name.
     * 
     * @return the class name
     */
    public String getClassName() {
        return className;
    }

    /**
     * Sets the class name.
     * 
     * @param className the class name
     */
    public void setClassName(String className) {
        this.className = className;
    }

    /**
     * Gets the design pattern role of this class.
     * 
     * @return the design pattern role
     */
    public String getDesignPattern() {
        return designPattern;
    }

    /**
     * Sets the design pattern role of this class.
     * 
     * @param designPattern the design pattern role
     */
    public void setDesignPattern(String designPattern) {
        this.designPattern = designPattern;
    }

    /**
     * Gets the name of a related class in the design pattern.
     * 
     * @return the related class name
     */
    public String getRelatedClassName() {
        return relatedClassName;
    }

    /**
     * Sets the name of a related class in the design pattern.
     * 
     * @param relatedClassName the related class name
     */
    public void setRelatedClassName(String relatedClassName) {
        this.relatedClassName = relatedClassName;
    }

    /**
     * Gets the design pattern role of the related class.
     * 
     * @return the design pattern role of the related class
     */
    public String getRelatedClassDesignPattern() {
        return relatedClassDesignPattern;
    }

    /**
     * Sets the design pattern role of the related class.
     * 
     * @param relatedClassDesignPattern the design pattern role of the related class
     */
    public void setRelatedClassDesignPattern(String relatedClassDesignPattern) {
        this.relatedClassDesignPattern = relatedClassDesignPattern;
    }
}


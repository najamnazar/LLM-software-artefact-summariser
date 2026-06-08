package dps_nlg.summarygenerator.messages;

/**
 * Represents inheritance relationship information in design pattern analysis.
 * <p>
 * This message object captures the inheritance hierarchy by storing information
 * about base classes (superclasses) that a class inherits from. It is used during
 * NLG summarization to describe class inheritance relationships in natural language.
 * </p>
 * <p>
 * In object-oriented design patterns, inheritance plays a crucial role in defining
 * class relationships and polymorphic behavior. This class helps communicate those
 * relationships to the summary generator.
 * </p>
 * 
 * @author Najam
 */
public class DesignPatternInheritClassMessage extends Message {
    private String inheritClass;

    /**
     * Gets the name of the inherited (parent) class.
     * 
     * @return the name of the base class
     */
    public String getInheritClass() {
        return inheritClass;
    }

    /**
     * Sets the name of the inherited (parent) class.
     * 
     * @param inheritClass the name of the base class
     */
    public void setInheritClass(String inheritClass) {
        this.inheritClass = inheritClass;
    }
}


package dps_swum.swum.context;

import java.util.*;

/**
 * Holds comprehensive design pattern context information for a class.
 * <p>
 * This class maintains all pattern-related information for a specific class,
 * including the patterns it participates in, its roles within those patterns,
 * textual descriptions from NLG analysis, and relationships with other classes
 * in the pattern structure.
 * </p>
 * <p>
 * Key responsibilities:
 * <ul>
 *   <li>Store multiple pattern memberships (a class can be in multiple patterns)</li>
 *   <li>Map each pattern to the class's role within that pattern</li>
 *   <li>Maintain NLG-generated descriptions of pattern behavior</li>
 *   <li>Track relationships with other classes in pattern implementations</li>
 *   <li>Generate natural language descriptions of roles and relationships</li>
 * </ul>
 * </p>
 * 
 * @author Najam
 */
public class PatternContext {
    
    private String className;
    private Set<String> patterns;
    private Map<String, String> patternRoles; // pattern -> role mapping
    private Map<String, List<String>> patternDescriptions; // pattern -> descriptions
    private Map<String, String> relatedClasses; // className -> relationship type
    
    /**
     * Constructs a new pattern context for the specified class.
     * 
     * @param className the class name this context applies to
     */
    public PatternContext(String className) {
        this.className = className;
        this.patterns = new HashSet<>();
        this.patternRoles = new HashMap<>();
        this.patternDescriptions = new HashMap<>();
        this.relatedClasses = new HashMap<>();
    }
    
    public String getClassName() {
        return className;
    }
    
    public Set<String> getPatterns() {
        return patterns;
    }
    
    public void addPattern(String pattern) {
        this.patterns.add(pattern);
    }
    
    public void addPatternRole(String pattern, String role) {
        this.patternRoles.put(pattern, role);
        this.patterns.add(pattern);
    }
    
    public String getPatternRole(String pattern) {
        return patternRoles.get(pattern);
    }
    
    public Map<String, String> getAllPatternRoles() {
        return patternRoles;
    }
    
    public void addPatternDescription(String pattern, String description) {
        patternDescriptions.putIfAbsent(pattern, new ArrayList<>());
        patternDescriptions.get(pattern).add(description);
    }
    
    public List<String> getPatternDescriptions(String pattern) {
        return patternDescriptions.getOrDefault(pattern, new ArrayList<>());
    }
    
    public void addRelatedClass(String relatedClassName, String relationshipType) {
        this.relatedClasses.put(relatedClassName, relationshipType);
    }
    
    public Map<String, String> getRelatedClasses() {
        return relatedClasses;
    }
    
    /**
     * Checks if this context has any pattern-related information.
     * 
     * @return true if the class participates in at least one pattern
     */
    public boolean hasPatternContext() {
        return !patterns.isEmpty();
    }
    
    /**
     * Generates a natural language description of the class's pattern roles.
     * <p>
     * Creates a formatted sentence describing all pattern roles this class plays,
     * e.g., "acts as a publisher in the observer pattern and a factory in the factory pattern".
     * </p>
     * 
     * @return role description, or null if no roles are defined
     */
    public String generateRoleDescription() {
        if (patternRoles.isEmpty()) {
            return null;
        }
        
        StringBuilder desc = new StringBuilder();
        List<Map.Entry<String, String>> entries = new ArrayList<>(patternRoles.entrySet());
        
        for (int i = 0; i < entries.size(); i++) {
            Map.Entry<String, String> entry = entries.get(i);
            String pattern = entry.getKey();
            String role = entry.getValue();
            
            if (i > 0) {
                if (i == entries.size() - 1) {
                    desc.append(" and ");
                } else {
                    desc.append(", ");
                }
            }
            
            desc.append("acts as ").append(formatArticle(role)).append(" ")
                .append(formatRole(role)).append(" in the ").append(pattern).append(" pattern");
        }
        
        return desc.toString();
    }
    
    /**
     * Generates a description of relationships with other classes in patterns.
     * <p>
     * Creates formatted text describing how this class relates to other classes
     * in pattern implementations (e.g., "publishes to ObserverA and observes SubjectB").
     * Limits output to the first 3 relationships to avoid overly long descriptions.
     * </p>
     * 
     * @return relationship description, or null if no relationships exist
     */
    public String generateRelationshipDescription() {
        if (relatedClasses.isEmpty()) {
            return null;
        }
        
        StringBuilder desc = new StringBuilder();
        List<Map.Entry<String, String>> entries = new ArrayList<>(relatedClasses.entrySet());
        
        for (int i = 0; i < Math.min(entries.size(), 3); i++) {
            Map.Entry<String, String> entry = entries.get(i);
            String relatedClass = entry.getKey();
            String relationship = entry.getValue();
            
            if (i > 0) {
                if (i == entries.size() - 1 || i == 2) {
                    desc.append(" and ");
                } else {
                    desc.append(", ");
                }
            }
            
            desc.append(formatRelationship(relationship)).append(" ")
                .append(relatedClass);
        }
        
        if (entries.size() > 3) {
            desc.append(" and ").append(entries.size() - 3).append(" other classes");
        }
        
        return desc.toString();
    }
    
    private String formatArticle(String role) {
        String lowerRole = role.toLowerCase();
        if (lowerRole.startsWith("a") || lowerRole.startsWith("e") || 
            lowerRole.startsWith("i") || lowerRole.startsWith("o") || 
            lowerRole.startsWith("u")) {
            return "an";
        }
        return "a";
    }
    
    private String formatRole(String role) {
        // Replace underscores with spaces
        return role.replace("_", " ");
    }
    
    private String formatRelationship(String relationship) {
        switch (relationship.toLowerCase()) {
            case "publisher":
                return "publishes to";
            case "subscriber":
            case "observer":
                return "observes";
            case "factory":
                return "is created by factory";
            case "product":
                return "is a product of";
            case "concrete_observer":
                return "implements observer";
            case "concrete_product":
                return "implements product";
            case "caretaker":
                return "manages memento for";
            case "originator":
                return "creates memento for";
            default:
                return "is related to";
        }
    }
}

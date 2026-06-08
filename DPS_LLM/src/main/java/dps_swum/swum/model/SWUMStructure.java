package dps_swum.swum.model;

import dps_swum.swum.context.PatternContext;
import dps_swum.swum.context.MethodPatternContext;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

/**
 * Represents a complete SWUM parse structure for a method or class.
 * <p>
 * This class encapsulates the results of SWUM analysis on a code identifier,
 * including the parse tree, extracted semantic components (actions and objects),
 * parameter and return type information, and design pattern context. It provides
 * methods for generating natural language summaries with proper English grammar.
 * </p>
 * <p>
 * Key responsibilities:
 * <ul>
 *   <li>Store SWUM parse tree and semantic components</li>
 *   <li>Maintain method/class metadata (parameters, return types, modifiers)</li>
 *   <li>Integrate design pattern context information</li>
 *   <li>Generate grammatically correct natural language summaries</li>
 *   <li>Handle pattern-aware summary generation</li>
 * </ul>
 * </p>
 * 
 * @author Najam
 */
public class SWUMStructure {
    
    private String methodName;
    private String className;
    private SWUMNode parseTree;
    private List<String> parameters;
    private String returnType;
    private List<String> actions;
    private List<String> objects;
    private String designPattern;
    
    // Pattern context fields
    private PatternContext patternContext;
    private MethodPatternContext methodPatternContext;
    
    /**
     * Constructs an empty SWUM structure.
     * <p>
     * Initializes empty collections for parameters, actions, and objects.
     * </p>
     */
    public SWUMStructure() {
        this.parameters = new ArrayList<>();
        this.actions = new ArrayList<>();
        this.objects = new ArrayList<>();
    }
    
    /**
     * Constructs a SWUM structure for a specific method and class.
     * 
     * @param methodName the method name to analyze
     * @param className the class name containing the method
     */
    public SWUMStructure(String methodName, String className) {
        this();
        this.methodName = methodName;
        this.className = className;
    }
    
    // Getters and setters
    public String getMethodName() { return methodName; }
    public void setMethodName(String methodName) { this.methodName = methodName; }
    
    public String getClassName() { return className; }
    public void setClassName(String className) { this.className = className; }
    
    public SWUMNode getParseTree() { return parseTree; }
    public void setParseTree(SWUMNode parseTree) { this.parseTree = parseTree; }
    
    public List<String> getParameters() { return parameters; }
    public void setParameters(List<String> parameters) { this.parameters = parameters; }
    public void addParameter(String parameter) { this.parameters.add(parameter); }
    
    public String getReturnType() { return returnType; }
    public void setReturnType(String returnType) { this.returnType = returnType; }
    
    public List<String> getActions() { return actions; }
    public void setActions(List<String> actions) { this.actions = actions; }
    public void addAction(String action) { this.actions.add(action); }
    
    public List<String> getObjects() { return objects; }
    public void setObjects(List<String> objects) { this.objects = objects; }
    public void addObject(String object) { this.objects.add(object); }
    
    public String getDesignPattern() { return designPattern; }
    public void setDesignPattern(String designPattern) { this.designPattern = designPattern; }
    
    public PatternContext getPatternContext() { return patternContext; }
    public void setPatternContext(PatternContext patternContext) { this.patternContext = patternContext; }
    
    public MethodPatternContext getMethodPatternContext() { return methodPatternContext; }
    public void setMethodPatternContext(MethodPatternContext methodPatternContext) { 
        this.methodPatternContext = methodPatternContext; 
    }
    
    /**
     * Generates a natural language summary using SWUM grammar rules with proper English grammar
     */
    public String generateSummary() {
        if (parseTree == null) {
            return generateBasicSummary();
        }
        
        StringBuilder summary = new StringBuilder();
        
        // Generate grammatically correct summary based on parse tree structure
        if (methodName != null) {
            summary.append(generateMethodSummaryWithGrammar());
        } else if (className != null) {
            summary.append(generateClassSummaryWithGrammar());
        } else {
            // Fallback to parse tree yield
            List<String> yield = parseTree.getYield();
            if (!yield.isEmpty()) {
                summary.append(String.join(" ", yield));
            }
        }
        
        return summary.toString().trim();
    }
    
    /**
     * Generates a grammatically correct method summary using subject-verb-object structure
     */
    private String generateMethodSummaryWithGrammar() {
        StringBuilder summary = new StringBuilder();
        
        // Check if method has pattern-specific behavior
        if (methodPatternContext != null && methodPatternContext.hasPatternContext()) {
            String behaviorDesc = methodPatternContext.generateBehaviorDescription();
            if (behaviorDesc != null) {
                summary.append(behaviorDesc);
            }
        }
        
        // If no pattern behavior, generate standard summary
        if (summary.length() == 0) {
            // Subject: "This method" or "The <methodName> method"
            summary.append("This method");
            
            // Verb: Extract primary action or use "performs"
            String verb = extractPrimaryVerb();
            if (verb != null) {
                summary.append(" ").append(conjugateVerb(verb));
            } else {
                summary.append(" performs");
            }
            
            // Object: Extract primary objects or use generic description
            if (!objects.isEmpty()) {
                summary.append(" ");
                if (objects.size() == 1) {
                    summary.append(addArticle(objects.get(0)));
                } else {
                    summary.append(String.join(", ", objects));
                }
            } else if (!actions.isEmpty() && actions.size() > 1) {
                summary.append(" operations");
            }
            
            // Add context about parameters
            if (!parameters.isEmpty()) {
                summary.append(" using ");
                if (parameters.size() == 1) {
                    summary.append("a parameter");
                } else {
                    summary.append(parameters.size()).append(" parameters");
                }
            }
            
            // Add return type information
            if (returnType != null && !returnType.equals("void")) {
                summary.append(" and returns ").append(formatType(returnType));
            }
            
            // Add design pattern context from pattern context if available
            if (patternContext != null && patternContext.hasPatternContext()) {
                String roleDesc = patternContext.generateRoleDescription();
                if (roleDesc != null) {
                    summary.append(". The ").append(className).append(" class ").append(roleDesc);
                }
            } else if (designPattern != null) {
                summary.append(" as part of the ").append(designPattern).append(" pattern");
            }
        }
        
        summary.append(".");
        return summary.toString();
    }
    
    /**
     * Generates a grammatically correct class summary
     */
    private String generateClassSummaryWithGrammar() {
        StringBuilder summary = new StringBuilder();
        
        // Subject: "The <className> class"
        summary.append("The ").append(className).append(" class");
        
        // Add pattern role information first (more specific than general pattern)
        if (patternContext != null && patternContext.hasPatternContext()) {
            String roleDesc = patternContext.generateRoleDescription();
            if (roleDesc != null) {
                summary.append(" ").append(roleDesc);
                
                // Add related class information
                String relationshipDesc = patternContext.generateRelationshipDescription();
                if (relationshipDesc != null) {
                    summary.append(". It ").append(relationshipDesc);
                }
            }
        } else if (designPattern != null) {
            // Fallback to simple design pattern mention
            summary.append(" is part of the ").append(designPattern).append(" pattern");
        }
        
        // Add functional description based on actions/objects
        if (!actions.isEmpty() || !objects.isEmpty()) {
            if (patternContext != null && patternContext.hasPatternContext()) {
                summary.append(". It");
            } else if (designPattern != null) {
                summary.append(" and");
            }
            
            // Verb and object based on actions
            if (!actions.isEmpty()) {
                summary.append(" ");
                if (actions.size() == 1) {
                    summary.append(conjugateVerb(actions.get(0)));
                } else {
                    summary.append("provides functionality to ").append(actions.get(0).toLowerCase());
                    if (actions.size() > 1) {
                        summary.append(", ").append(actions.get(1).toLowerCase());
                        if (actions.size() > 2) {
                            summary.append(" and perform other operations");
                        }
                    }
                }
            } else {
                summary.append(" provides functionality");
            }
            
            // Add object information
            if (!objects.isEmpty()) {
                summary.append(" for ");
                if (objects.size() == 1) {
                    summary.append(objects.get(0).toLowerCase());
                } else {
                    summary.append(String.join(", ", objects));
                }
            }
        }
        
        summary.append(".");
        return summary.toString();
    }
    
    /**
     * Generates a basic summary when no parse tree is available
     */
    private String generateBasicSummary() {
        StringBuilder summary = new StringBuilder();
        
        if (className != null) {
            summary.append("The ").append(className).append(" class");

            boolean hasPredicate = false;
            // Add pattern role information first (similar to grammar-based summary)
            if (patternContext != null && patternContext.hasPatternContext()) {
                String roleDesc = patternContext.generateRoleDescription();
                if (roleDesc != null) {
                    summary.append(" ").append(roleDesc);
                    hasPredicate = true;
                }
            } else if (designPattern != null) {
                summary.append(" is part of the ").append(designPattern).append(" pattern");
                hasPredicate = true;
            }

            if (!actions.isEmpty()) {
                // Use semantic grouping if there are many actions (10+ suggests complex class)
                if (actions.size() >= 10) {
                    String groupedSummary = generateSemanticActionSummary(actions);
                    summary.append(" ").append(groupedSummary);
                } else {
                    if (hasPredicate) {
                        summary.append(" and provides methods to ");
                    } else {
                        summary.append(" provides methods to ");
                    }
                    summary.append(String.join(", ", actions).toLowerCase());
                }
            }
            
            if (!objects.isEmpty()) {
                summary.append(" for managing ").append(String.join(", ", objects).toLowerCase());
            }
            
            summary.append(".");
        } else if (methodName != null) {
            summary.append("The ").append(methodName).append(" method");
            
            if (!actions.isEmpty()) {
                summary.append(" ").append(conjugateVerb(actions.get(0)));
            }
            
            if (!objects.isEmpty()) {
                summary.append(" ").append(addArticle(objects.get(0)));
            }
            
            summary.append(".");
        }
        
        return summary.toString().trim();
    }
    
    /**
     * Extracts the primary verb from actions
     */
    private String extractPrimaryVerb() {
        if (actions.isEmpty()) {
            return null;
        }
        return actions.get(0);
    }
    
    /**
     * Groups actions into semantic categories and generates a summary
     * for classes with many methods (10+)
     */
    private String generateSemanticActionSummary(List<String> actions) {
        // Count occurrences of each action
        Map<String, Integer> actionCounts = new LinkedHashMap<>();
        for (String action : actions) {
            String lowerAction = action.toLowerCase();
            actionCounts.put(lowerAction, actionCounts.getOrDefault(lowerAction, 0) + 1);
        }
        
        // Categorize actions into semantic groups
        Map<String, Integer> categories = new LinkedHashMap<>();
        int totalMethods = actions.size();
        
        for (Map.Entry<String, Integer> entry : actionCounts.entrySet()) {
            String action = entry.getKey();
            int count = entry.getValue();
            
            // Map actions to semantic categories
            String category = categorizeAction(action);
            categories.put(category, categories.getOrDefault(category, 0) + count);
        }
        
        // Build summary based on categories
        StringBuilder summary = new StringBuilder();
        summary.append("provides ");
        
        // Get top categories
        List<Map.Entry<String, Integer>> sortedCategories = categories.entrySet().stream()
            .sorted((e1, e2) -> e2.getValue().compareTo(e1.getValue()))
            .collect(Collectors.toList());
        
        if (sortedCategories.size() == 1) {
            Map.Entry<String, Integer> category = sortedCategories.get(0);
            summary.append(category.getValue()).append(" methods for ")
                   .append(category.getKey());
        } else if (sortedCategories.size() == 2) {
            summary.append(sortedCategories.get(0).getValue()).append(" methods for ")
                   .append(sortedCategories.get(0).getKey())
                   .append(" and ").append(sortedCategories.get(1).getValue())
                   .append(" for ").append(sortedCategories.get(1).getKey());
        } else if (sortedCategories.size() >= 3) {
            summary.append(totalMethods).append(" methods including ")
                   .append(sortedCategories.get(0).getKey()).append(", ")
                   .append(sortedCategories.get(1).getKey());
            if (sortedCategories.size() > 3) {
                summary.append(", ").append(sortedCategories.get(2).getKey())
                       .append(" and other operations");
            } else {
                summary.append(" and ").append(sortedCategories.get(2).getKey());
            }
        }
        
        return summary.toString();
    }
    
    /**
     * Maps an action verb to a semantic category
     */
    private String categorizeAction(String action) {
        // Retrieval operations
        if (action.matches("get|fetch|retrieve|obtain|find|read|load")) {
            return "retrieval";
        }
        
        // Storage/modification operations
        if (action.matches("set|put|store|save|write|update|modify|edit|change")) {
            return "storage and modification";
        }
        
        // Creation operations
        if (action.matches("create|make|build|construct|generate|produce|add|insert|append")) {
            return "creation";
        }
        
        // Deletion operations
        if (action.matches("delete|remove|clear|clean|drop|destroy")) {
            return "deletion";
        }
        
        // Validation/verification operations
        if (action.matches("check|verify|validate|test|ensure|confirm")) {
            return "validation";
        }
        
        // Registration/lifecycle operations (common in singleton/factory patterns)
        if (action.matches("register|unregister|initialize|start|stop|close|open")) {
            return "registration and lifecycle";
        }
        
        // Processing operations
        if (action.matches("process|handle|execute|run|perform|invoke")) {
            return "processing";
        }
        
        // Notification/communication operations
        if (action.matches("notify|send|receive|publish|subscribe|broadcast")) {
            return "notification";
        }
        
        // Default: use the action itself
        return action + " operations";
    }
    
    /**
     * Conjugates a verb to third person singular present tense
     */
    private String conjugateVerb(String verb) {
        if (verb == null || verb.isEmpty()) {
            return "processes";
        }
        
        String lowerVerb = verb.toLowerCase();
        
        // Special cases
        if (lowerVerb.equals("get")) return "retrieves";
        if (lowerVerb.equals("set")) return "sets";
        if (lowerVerb.equals("add")) return "adds";
        if (lowerVerb.equals("remove")) return "removes";
        if (lowerVerb.equals("create")) return "creates";
        if (lowerVerb.equals("delete")) return "deletes";
        if (lowerVerb.equals("update")) return "updates";
        if (lowerVerb.equals("find")) return "finds";
        if (lowerVerb.equals("build")) return "builds";
        if (lowerVerb.equals("process")) return "processes";
        if (lowerVerb.equals("handle")) return "handles";
        if (lowerVerb.equals("make")) return "makes";
        if (lowerVerb.equals("check")) return "checks";
        if (lowerVerb.equals("validate")) return "validates";
        if (lowerVerb.equals("execute")) return "executes";
        if (lowerVerb.equals("notify")) return "notifies";
        
        // Default: add 's' to the end
        if (lowerVerb.endsWith("s") || lowerVerb.endsWith("x") || 
            lowerVerb.endsWith("ch") || lowerVerb.endsWith("sh")) {
            return lowerVerb + "es";
        } else if (lowerVerb.endsWith("y") && lowerVerb.length() > 1 && 
                   !isVowel(lowerVerb.charAt(lowerVerb.length() - 2))) {
            return lowerVerb.substring(0, lowerVerb.length() - 1) + "ies";
        } else {
            return lowerVerb + "s";
        }
    }
    
    /**
     * Adds appropriate article (a/an/the) to a noun
     */
    private String addArticle(String noun) {
        if (noun == null || noun.isEmpty()) {
            return noun;
        }
        
        String lowerNoun = noun.toLowerCase();
        char firstChar = lowerNoun.charAt(0);
        
        // Use "an" before vowels, "a" before consonants
        if (isVowel(firstChar)) {
            return "an " + noun.toLowerCase();
        } else {
            return "a " + noun.toLowerCase();
        }
    }
    
    /**
     * Checks if a character is a vowel
     */
    private boolean isVowel(char c) {
        return c == 'a' || c == 'e' || c == 'i' || c == 'o' || c == 'u';
    }
    
    /**
     * Formats a type name for display
     */
    private String formatType(String type) {
        if (type == null) {
            return "a value";
        }
        
        String lowerType = type.toLowerCase();
        if (lowerType.contains("string")) {
            return "a String";
        } else if (lowerType.contains("int") || lowerType.contains("long") || 
                   lowerType.contains("double") || lowerType.contains("float")) {
            return "a number";
        } else if (lowerType.contains("bool")) {
            return "a boolean";
        } else if (lowerType.contains("list") || lowerType.contains("array")) {
            return "a collection";
        } else {
            return "an object";
        }
    }
    
    /**
     * Returns a structured representation for JSON serialization
     */
    public String toStructuredString() {
        StringBuilder sb = new StringBuilder();
        sb.append("SWUM Structure:\n");
        sb.append("  Method: ").append(methodName != null ? methodName : "N/A").append("\n");
        sb.append("  Class: ").append(className != null ? className : "N/A").append("\n");
        sb.append("  Actions: ").append(actions).append("\n");
        sb.append("  Objects: ").append(objects).append("\n");
        sb.append("  Parameters: ").append(parameters).append("\n");
        sb.append("  Return Type: ").append(returnType != null ? returnType : "void").append("\n");
        sb.append("  Design Pattern: ").append(designPattern != null ? designPattern : "None").append("\n");
        if (parseTree != null) {
            sb.append("  Parse Tree: ").append(parseTree.toTreeString()).append("\n");
        }
        sb.append("  Summary: ").append(generateSummary());
        return sb.toString();
    }
}


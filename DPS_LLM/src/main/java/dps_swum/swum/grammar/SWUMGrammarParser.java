package dps_swum.swum.grammar;

import dps_swum.swum.model.SWUMNode;
import dps_swum.swum.model.SWUMNode.NodeType;
import dps_swum.swum.model.SWUMStructure;
import dps_swum.swum.context.PatternContext;
import dps_swum.swum.context.MethodPatternContext;

import java.util.*;
import java.util.regex.Pattern;

/**
 * SWUM grammar parser that converts code identifiers into structured representations.
 * <p>
 * This parser implements the Software Word Usage Model (SWUM) technique for analyzing
 * method names and class names. It splits camelCase identifiers, identifies action verbs
 * and object nouns, builds parse trees, and integrates design pattern context to generate
 * natural language summaries following software naming conventions.
 * </p>
 * <p>
 * Key responsibilities:
 * <ul>
 *   <li>Maintain vocabularies of software action verbs and object nouns</li>
 *   <li>Parse camelCase and snake_case identifiers into word sequences</li>
 *   <li>Build SWUM parse trees representing grammatical structure</li>
 *   <li>Extract semantic components (actions, objects) from identifiers</li>
 *   <li>Identify design patterns from naming conventions</li>
 *   <li>Generate context-aware natural language summaries</li>
 *   <li>Track grammar rule application statistics</li>
 * </ul>
 * </p>
 * 
 * @author Najam
 */
public class SWUMGrammarParser {
    
    // Common software action verbs
    private static final Set<String> ACTION_VERBS = new HashSet<>(Arrays.asList(
        "create", "make", "build", "construct", "generate", "produce",
        "get", "fetch", "retrieve", "obtain", "acquire", "find",
        "set", "put", "place", "assign", "store", "save",
        "add", "insert", "append", "include", "push",
        "remove", "delete", "clear", "clean", "drop",
        "update", "modify", "change", "edit", "alter",
        "process", "handle", "manage", "execute", "run",
        "check", "verify", "validate", "test", "ensure",
        "convert", "transform", "parse", "format", "render",
        "open", "close", "start", "stop", "begin", "end",
        "connect", "disconnect", "bind", "unbind", "attach",
        "send", "receive", "transmit", "broadcast", "notify",
        "load", "unload", "read", "write", "copy", "move",
        "encode", "decode", "compress", "decompress",
        "restore", "undo", "redo", "reset", "initialize", "init",
        "publish", "subscribe", "unsubscribe", "register", "unregister",
        "display", "print", "log", "show", "hide",
        "serialize", "deserialize", "compare", "sort", "filter", "search",
        "accept", "visit", "deploy", "launch", "trigger", "invoke"
    ));
    
    // Common software object nouns
    // Excluded: "set" (also an action verb), "class"/"interface"/"string" (Java keywords),
    // "factory"/"builder"/"adapter" (design pattern roles, not domain objects)
    private static final Set<String> OBJECT_NOUNS = new HashSet<>(Arrays.asList(
        "file", "document", "data", "information", "content",
        "user", "customer", "client", "server", "service",
        "connection", "session", "request", "response", "message",
        "object", "instance", "element", "item", "component",
        "list", "array", "collection", "map",
        "text", "number", "value",
        "event",
        "window", "dialog", "panel", "button", "field",
        "database", "table", "record", "row", "column"
    ));
    
    // Design pattern related terms
    private static final Set<String> PATTERN_TERMS = new HashSet<>(Arrays.asList(
        "factory", "builder", "singleton", "observer", "adapter",
        "decorator", "facade", "visitor", "memento", "strategy",
        "command", "state", "template", "proxy", "bridge"
    ));
    
    // Camel case splitting pattern
    private static final Pattern CAMEL_CASE = Pattern.compile("(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])");
    
    // Instance fields for tracking parser state
    private int appliedRulesCount = 0;
    
    public SWUMGrammarParser() {
        this.appliedRulesCount = 0;
    }
    
    /**
     * Gets the count of grammar rules applied during parsing
     */
    public int getAppliedRulesCount() {
        return appliedRulesCount;
    }
    
    /**
     * Resets the applied rules counter
     */
    public void resetRulesCount() {
        this.appliedRulesCount = 0;
    }
    
    /**
     * Gets the action verbs vocabulary
     */
    public Set<String> getActionVerbs() {
        return new HashSet<>(ACTION_VERBS);
    }
    
    /**
     * Gets the object nouns vocabulary
     */
    public Set<String> getObjectNouns() {
        return new HashSet<>(OBJECT_NOUNS);
    }
    
    /**
     * Gets the pattern terms vocabulary
     */
    public Set<String> getPatternTerms() {
        return new HashSet<>(PATTERN_TERMS);
    }
    
    /**
     * Parses a method name using SWUM grammar rules
     */
    public SWUMStructure parseMethodName(String methodName, String className) {
        appliedRulesCount++; // Increment for each parse operation
        
        SWUMStructure structure = new SWUMStructure(methodName, className);
        
        // Split camelCase method name
        List<String> words = splitIdentifier(methodName);
        
        // Create parse tree
        SWUMNode root = createParseTree(words);
        structure.setParseTree(root);
        
        // Extract semantic components
        extractSemanticComponents(words, structure);
        
        return structure;
    }
    
    /**
     * Parses a class name and its methods
     */
    public SWUMStructure parseClass(String className, List<String> methodNames) {
        SWUMStructure structure = new SWUMStructure(null, className);
        
        List<String> classWords = splitIdentifier(className);
        
        // Identify design patterns from class name
        identifyDesignPattern(classWords, structure);
        
        // Extract object information from class name
        for (String word : classWords) {
            if (OBJECT_NOUNS.contains(word.toLowerCase())) {
                structure.addObject(word);
            }
        }
        
        // Analyze methods for common actions
        for (String methodName : methodNames) {
            List<String> methodWords = splitIdentifier(methodName);
            for (String word : methodWords) {
                if (ACTION_VERBS.contains(word.toLowerCase())) {
                    structure.addAction(word);
                }
            }
        }
        
        return structure;
    }
    
    /**
     * Creates a SWUM parse tree from a list of words
     */
    private SWUMNode createParseTree(List<String> words) {
        if (words.isEmpty()) {
            return new SWUMNode(NodeType.TERMINAL);
        }
        
        SWUMNode root = new SWUMNode(NodeType.VERB_PHRASE);
        
        // Find the main action verb (usually first action verb in the name)
        int verbIndex = -1;
        for (int i = 0; i < words.size(); i++) {
            if (ACTION_VERBS.contains(words.get(i).toLowerCase())) {
                verbIndex = i;
                break;
            }
        }
        
        if (verbIndex >= 0) {
            // Create verb phrase structure: [Subject] Verb [Object] [Modifier]
            
            // Add subject (words before verb)
            if (verbIndex > 0) {
                SWUMNode subject = new SWUMNode(NodeType.SUBJECT);
                for (int i = 0; i < verbIndex; i++) {
                    SWUMNode word = new SWUMNode(NodeType.TERMINAL, words.get(i));
                    subject.addChild(word);
                }
                root.addChild(subject);
            }
            
            // Add main verb
            SWUMNode verb = new SWUMNode(NodeType.TERMINAL, words.get(verbIndex));
            root.addChild(verb);
            
            // Add object (words after verb)
            if (verbIndex + 1 < words.size()) {
                SWUMNode object = new SWUMNode(NodeType.OBJECT);
                for (int i = verbIndex + 1; i < words.size(); i++) {
                    SWUMNode word = new SWUMNode(NodeType.TERMINAL, words.get(i));
                    object.addChild(word);
                }
                root.addChild(object);
            }
        } else {
            // No clear verb, treat as noun phrase
            root.setType(NodeType.NOUN_PHRASE);
            for (String word : words) {
                SWUMNode terminal = new SWUMNode(NodeType.TERMINAL, word);
                root.addChild(terminal);
            }
        }
        
        return root;
    }
    
    /**
     * Extracts semantic components from words
     */
    private void extractSemanticComponents(List<String> words, SWUMStructure structure) {
        for (String word : words) {
            String lowerWord = word.toLowerCase();
            
            if (ACTION_VERBS.contains(lowerWord)) {
                structure.addAction(word);
            }
            
            if (OBJECT_NOUNS.contains(lowerWord)) {
                structure.addObject(word);
            }
        }
    }
    
    /**
     * Identifies design patterns from class or method names
     */
    private void identifyDesignPattern(List<String> words, SWUMStructure structure) {
        for (String word : words) {
            String lowerWord = word.toLowerCase();
            if (PATTERN_TERMS.contains(lowerWord)) {
                structure.setDesignPattern(word);
                break;
            }
        }
    }
    
    /**
     * Splits camelCase or snake_case identifiers into individual words
     */
    private List<String> splitIdentifier(String identifier) {
        if (identifier == null || identifier.isEmpty()) {
            return new ArrayList<>();
        }
        
        // Handle snake_case
        identifier = identifier.replace("_", " ");
        
        // Handle camelCase
        String[] parts = CAMEL_CASE.split(identifier);
        
        List<String> words = new ArrayList<>();
        for (String part : parts) {
            if (!part.trim().isEmpty()) {
                words.add(part.trim());
            }
        }
        
        return words;
    }
    
    /**
     * Generates a complete SWUM summary for a method with context
     */
    public String generateMethodSummary(String methodName, String className, 
                                      List<String> parameters, String returnType,
                                      String designPattern) {
        SWUMStructure structure = parseMethodName(methodName, className);
        
        if (parameters != null) {
            structure.setParameters(parameters);
        }
        structure.setReturnType(returnType);
        structure.setDesignPattern(designPattern);
        
        return structure.generateSummary();
    }
    
    /**
     * Generates a SWUM summary for a class
     */
    public String generateClassSummary(String className, List<String> methodNames, String designPattern) {
        SWUMStructure structure = parseClass(className, methodNames);
        structure.setDesignPattern(designPattern);
        
        return structure.generateSummary();
    }
    
    /**
     * Generates a complete SWUM summary for a method with pattern context
     */
    public String generateMethodSummaryWithContext(String methodName, String className, 
                                                  List<String> parameters, String returnType,
                                                  String designPattern, PatternContext patternContext,
                                                  MethodPatternContext methodPatternContext) {
        SWUMStructure structure = parseMethodName(methodName, className);
        
        if (parameters != null) {
            structure.setParameters(parameters);
        }
        structure.setReturnType(returnType);
        structure.setDesignPattern(designPattern);
        structure.setPatternContext(patternContext);
        structure.setMethodPatternContext(methodPatternContext);
        
        return structure.generateSummary();
    }
    
    /**
     * Generates a SWUM summary for a class with pattern context
     */
    public String generateClassSummaryWithContext(String className, List<String> methodNames, 
                                                 String designPattern, PatternContext patternContext) {
        SWUMStructure structure = parseClass(className, methodNames);
        structure.setDesignPattern(designPattern);
        structure.setPatternContext(patternContext);
        
        return structure.generateSummary();
    }
}



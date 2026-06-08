package dps_swum.swum.context;

import com.fasterxml.jackson.databind.JsonNode;
import java.util.*;

/**
 * Extracts design pattern context from DPS JSON output.
 * <p>
 * This class analyzes the DPS JSON structure to identify pattern roles,
 * relationships between classes, and method-level pattern behaviors. It processes
 * both the design_pattern section (structural analysis) and summary_NLG section
 * (textual descriptions) to build comprehensive pattern context objects.
 * </p>
 * <p>
 * Key responsibilities:
 * <ul>
 *   <li>Parse design pattern information from JSON</li>
 *   <li>Identify class roles in design patterns</li>
 *   <li>Extract relationships between classes in patterns</li>
 *   <li>Determine method-level pattern behaviors</li>
 *   <li>Match classes to pattern roles via name analysis and NLG descriptions</li>
 * </ul>
 * </p>
 * 
 * @author Najam
 */
public class PatternContextExtractor {
    
    /**
     * Extracts pattern context for a specific class from JSON data.
     * <p>
     * Analyzes both the design_pattern and summary_NLG sections to build
     * a complete picture of the class's role in design patterns.
     * </p>
     *
     * @param className the class name to extract context for
     * @param rootNode the root JSON node containing DPS output
     * @return pattern context object with extracted information
     */
    public static PatternContext extractContextForClass(String className, JsonNode rootNode) {
        PatternContext context = new PatternContext(className);

        // Extract from design_pattern section
        JsonNode designPatterns = rootNode.get("design_pattern");
        if (designPatterns != null && designPatterns.isArray()) {
            for (JsonNode patternNode : designPatterns) {
                processDesignPattern(className, patternNode, context);
            }
        }

        // Extract from summary_NLG section for more detailed descriptions
        JsonNode summaryNLG = rootNode.get("summary_NLG");
        if (summaryNLG != null && summaryNLG.isObject()) {
            processSummaryNLG(className, summaryNLG, context);
        }

        return context;
    }

    /**
     * SWUM-fix: extracts pattern context using SWUM's own detected patterns rather than
     * reading NLG's pre-computed design_pattern or summary_NLG fields.
     * swumPatternsNode is the JsonNode form of the ArrayList produced by
     * CheckPattern.extractDesignPattern() — same structural format as design_pattern array
     * in the NLG JSON, but derived independently from AST features by SWUM.
     *
     * @param className       the class name to extract context for
     * @param swumPatternsNode SWUM's own pattern detection result (array of pattern maps)
     * @return pattern context built from SWUM's independent structural analysis
     */
    public static PatternContext extractContextForClassFromSwum(String className, JsonNode swumPatternsNode) {
        PatternContext context = new PatternContext(className);

        // Process SWUM-detected patterns — same traversal as the NLG design_pattern array
        if (swumPatternsNode != null && swumPatternsNode.isArray()) {
            for (JsonNode patternNode : swumPatternsNode) {
                processDesignPattern(className, patternNode, context);
            }
        }

        // SWUM-fix: summary_NLG is NOT used here — SWUM derives context from code features only,
        // not from NLG's textual descriptions.
        // Old (NLG-dependent path, kept for reference):
        // JsonNode summaryNLG = rootNode.get("summary_NLG");
        // if (summaryNLG != null && summaryNLG.isObject()) {
        //     processSummaryNLG(className, summaryNLG, context);
        // }

        return context;
    }
    
    /**
     * Processes design_pattern JSON node to extract pattern roles and relationships
     */
    private static void processDesignPattern(String className, JsonNode patternNode, PatternContext context) {
        patternNode.fields().forEachRemaining(patternEntry -> {
            String patternName = patternEntry.getKey();
            JsonNode patternDetails = patternEntry.getValue();
            
            // Traverse the pattern structure to find this class
            findClassInPattern(className, patternName, patternDetails, context, new ArrayList<>());
        });
    }
    
    /**
     * Recursively finds the class in the pattern structure and extracts its role.
     *
     * SWUM-fix (two bugs corrected, both pre-existing but masked by summary_NLG in old path):
     *
     * Bug A — top-level key with empty pathRoles: classes that ARE the primary actor in a
     * pattern (Adapter, BaseDecorator, Facade, Memento, etc.) appear as direct top-level keys
     * under the pattern name. At that point pathRoles is empty so no role was assigned.
     * Fix: call getTopLevelRole(patternName) to assign the implicit role.
     *
     * Bug B — text nodes inside arrays: concrete visitors/elements stored as text values in
     * arrays (e.g. ["ManagerVisitor","BossVisitor"]) are not objects or sub-arrays, so the
     * old traversal silently skipped them. Fix: handle the isTextual() case.
     */
    private static void findClassInPattern(String className, String patternName,
                                          JsonNode node, PatternContext context,
                                          List<String> pathRoles) {
        if (node.isObject()) {
            node.fields().forEachRemaining(entry -> {
                String key = entry.getKey();
                JsonNode value = entry.getValue();

                // Check if this key is a role (publisher, subscriber, factory, product, etc.)
                if (isPatternRole(key)) {
                    List<String> newPath = new ArrayList<>(pathRoles);
                    newPath.add(key);

                    if (value.isObject()) {
                        // Role contains nested structure
                        findClassInPattern(className, patternName, value, context, newPath);
                    } else if (value.isArray()) {
                        // Role contains array of classes
                        for (JsonNode classNode : value) {
                            if (classNode.isTextual() && classNode.asText().equals(className)) {
                                context.addPatternRole(patternName, key);
                                context.addPattern(patternName);
                            }
                        }
                    }
                } else if (key.equals(className)) {
                    // Found the class as a key
                    if (!pathRoles.isEmpty()) {
                        context.addPatternRole(patternName, pathRoles.get(pathRoles.size() - 1));
                        context.addPattern(patternName);
                    } else {
                        // SWUM-fix Bug A: class is a direct top-level key under the pattern root;
                        // pathRoles is empty so we infer the implicit role from the pattern name.
                        // Old: no role was assigned here, falling back to summary_NLG text parsing.
                        String impliedRole = getTopLevelRole(patternName);
                        if (impliedRole != null) {
                            context.addPatternRole(patternName, impliedRole);
                        }
                        context.addPattern(patternName);
                    }

                    // Extract relationships from the value
                    if (value.isObject()) {
                        value.fields().forEachRemaining(relEntry -> {
                            String relType = relEntry.getKey();
                            JsonNode relValue = relEntry.getValue();

                            if (relValue.isObject()) {
                                relValue.fields().forEachRemaining(relClassEntry -> {
                                    context.addRelatedClass(relClassEntry.getKey(), relType);
                                });
                            } else if (relValue.isArray()) {
                                for (JsonNode relClass : relValue) {
                                    if (relClass.isTextual()) {
                                        context.addRelatedClass(relClass.asText(), relType);
                                    }
                                }
                            }
                        });
                    }
                } else {
                    // Continue searching
                    findClassInPattern(className, patternName, value, context, pathRoles);
                }
            });
        } else if (node.isArray()) {
            for (JsonNode arrayItem : node) {
                findClassInPattern(className, patternName, arrayItem, context, pathRoles);
            }
        } else if (node.isTextual()) {
            // SWUM-fix Bug B: class name stored as a plain text value inside an array under a
            // role path (e.g. concrete visitors in ["ManagerVisitor","BossVisitor",...]).
            // Old: text nodes were silently skipped; only objects and arrays were traversed.
            if (node.asText().equals(className) && !pathRoles.isEmpty()) {
                context.addPatternRole(patternName, pathRoles.get(pathRoles.size() - 1));
                context.addPattern(patternName);
            }
        }
    }

    /**
     * Returns the implicit role for a class that appears as a direct top-level key under the
     * pattern name in the design_pattern JSON structure.
     * Each pattern stores its primary actor as the top-level key:
     *   memento   → the memento class itself
     *   visitor   → the element interface/class (elements hold the accept method)
     *   decorator → the base decorator class
     *   adapter   → the adapter class
     *   facade    → the facade class
     *   observer  → the publisher (subject) class
     */
    private static String getTopLevelRole(String patternName) {
        switch (patternName.toLowerCase()) {
            case "memento":          return "memento";
            case "visitor":          return "element";
            case "decorator":        return "decorator";
            case "adapter":          return "adapter";
            case "facade":           return "facade";
            case "observer":         return "publisher";
            case "abstract_factory": return "factory";
            case "factory":          return "factory";
            case "singleton":        return "singleton";
            case "strategy":         return "strategy";
            case "builder":          return "builder";
            default:                 return null;
        }
    }
    
    /**
     * Processes summary_NLG to extract pattern descriptions
     */
    private static void processSummaryNLG(String className, JsonNode summaryNLG, PatternContext context) {
        summaryNLG.fields().forEachRemaining(patternEntry -> {
            String patternName = patternEntry.getKey();
            JsonNode patternSummaries = patternEntry.getValue();
            
            if (patternSummaries.has(className)) {
                JsonNode classSummaries = patternSummaries.get(className);
                if (classSummaries.isArray()) {
                    for (JsonNode summary : classSummaries) {
                        if (summary.isTextual()) {
                            String summaryText = summary.asText();
                            context.addPatternDescription(patternName, summaryText);
                            
                            // Extract role from description (e.g., "acts as a publisher")
                            extractRoleFromDescription(summaryText, patternName, context);
                        }
                    }
                }
            }
        });
    }
    
    /**
     * Extracts pattern role from natural language description
     */
    private static void extractRoleFromDescription(String description, String patternName, PatternContext context) {
        String lowerDesc = description.toLowerCase();
        
        // Observer pattern roles
        if (lowerDesc.contains("acts as a publisher") || lowerDesc.contains("acts as an publisher")) {
            context.addPatternRole(patternName, "publisher");
        } else if (lowerDesc.contains("acts as a subscriber") || lowerDesc.contains("acts as an subscriber") 
                || lowerDesc.contains("acts as an observer") || lowerDesc.contains("acts as a observer")) {
            context.addPatternRole(patternName, "subscriber");
        }
        // Factory pattern roles
        else if (lowerDesc.contains("acts as a factory") || lowerDesc.contains("acts as an factory")) {
            context.addPatternRole(patternName, "factory");
        } else if (lowerDesc.contains("acts as a product") || lowerDesc.contains("acts as an product")) {
            context.addPatternRole(patternName, "product");
        }
        // Singleton pattern
        else if (lowerDesc.contains("acts as a singleton") || lowerDesc.contains("acts as an singleton")) {
            context.addPatternRole(patternName, "singleton");
        }
        // Decorator pattern roles
        else if (lowerDesc.contains("acts as a decorator") || lowerDesc.contains("acts as an decorator")) {
            context.addPatternRole(patternName, "decorator");
        } else if (lowerDesc.contains("acts as a component") || lowerDesc.contains("acts as an component")) {
            context.addPatternRole(patternName, "component");
        }
        // Adapter pattern roles
        else if (lowerDesc.contains("acts as an adapter") || lowerDesc.contains("acts as a adapter")) {
            context.addPatternRole(patternName, "adapter");
        } else if (lowerDesc.contains("acts as an adaptee") || lowerDesc.contains("acts as a adaptee")) {
            context.addPatternRole(patternName, "adaptee");
        }
        // Facade pattern
        else if (lowerDesc.contains("acts as a facade") || lowerDesc.contains("acts as an facade")) {
            context.addPatternRole(patternName, "facade");
        }
        // Visitor pattern roles
        else if (lowerDesc.contains("acts as a visitor") || lowerDesc.contains("acts as an visitor")) {
            context.addPatternRole(patternName, "visitor");
        } else if (lowerDesc.contains("acts as an element") || lowerDesc.contains("acts as a element")) {
            context.addPatternRole(patternName, "element");
        }
        // Memento pattern roles
        else if (lowerDesc.contains("acts as an originator") || lowerDesc.contains("acts as a originator")) {
            context.addPatternRole(patternName, "originator");
        } else if (lowerDesc.contains("acts as a memento") || lowerDesc.contains("acts as an memento")) {
            context.addPatternRole(patternName, "memento");
        } else if (lowerDesc.contains("acts as a caretaker") || lowerDesc.contains("acts as an caretaker")) {
            context.addPatternRole(patternName, "caretaker");
        }
        // Abstract Factory roles
        else if (lowerDesc.contains("acts as an abstract product") || lowerDesc.contains("acts as a abstract product")) {
            context.addPatternRole(patternName, "abstract_product");
        }
    }
    
    /**
     * Checks if a string represents a pattern role
     */
    private static boolean isPatternRole(String key) {
        Set<String> knownRoles = new HashSet<>(Arrays.asList(
            "publisher", "subscriber", "observer", "concrete_observer",
            "factory", "product", "concrete_product", "abstract_product",
            "singleton",
            "decorator", "component", "concrete_component", "concrete_decorator",
            "adapter", "adaptee", "target",
            "facade", "subsystem",
            "visitor", "concrete_visitor", "element", "concrete_element",
            "originator", "memento", "caretaker", "concrete_memento"
        ));
        return knownRoles.contains(key.toLowerCase());
    }
    
    /**
     * Extracts method-level pattern context based on method name and class role.
     * <p>
     * Analyzes method names in the context of the class's pattern role to identify
     * pattern-specific behaviors (e.g., subscribe/notify in Observer pattern,
     * create in Factory pattern).
     * </p>
     * 
     * @param className the class containing the method
     * @param methodName the method name to analyze
     * @param classDetails the JSON node with class details
     * @param classContext the pattern context of the containing class
     * @return method pattern context with identified behaviors
     */
    public static MethodPatternContext extractMethodContext(String className, String methodName, 
                                                           JsonNode classDetails, PatternContext classContext) {
        MethodPatternContext methodContext = new MethodPatternContext(methodName);
        
        // Check if method is related to pattern behavior based on name and pattern role
        if (!classContext.getPatterns().isEmpty()) {
            String lowerMethodName = methodName.toLowerCase();
            
            for (String pattern : classContext.getPatterns()) {
                String role = classContext.getPatternRole(pattern);
                
                // Observer pattern methods
                if (pattern.equals("observer")) {
                    if (role != null && role.contains("publisher")) {
                        if (lowerMethodName.contains("unsubscribe") || lowerMethodName.contains("detach")
                                || lowerMethodName.contains("remove")) {
                            methodContext.setPatternBehavior("unsubscribes observers");
                            methodContext.setPattern(pattern);
                        } else if (lowerMethodName.contains("subscribe") || lowerMethodName.contains("attach")
                                || lowerMethodName.contains("register") || lowerMethodName.contains("add")) {
                            methodContext.setPatternBehavior("subscribes observers");
                            methodContext.setPattern(pattern);
                        } else if (lowerMethodName.contains("notify") || lowerMethodName.contains("update")) {
                            methodContext.setPatternBehavior("notifies all subscribers");
                            methodContext.setPattern(pattern);
                        }
                    } else if (role != null && (role.contains("subscriber") || role.contains("observer"))) {
                        if (lowerMethodName.contains("update") || lowerMethodName.contains("notify")) {
                            methodContext.setPatternBehavior("receives updates from publisher");
                            methodContext.setPattern(pattern);
                        }
                    }
                }
                // Factory pattern methods
                else if (pattern.contains("factory")) {
                    if (role != null && role.contains("factory")) {
                        if (lowerMethodName.contains("create") || lowerMethodName.contains("make") 
                                || lowerMethodName.contains("get") || lowerMethodName.contains("build")) {
                            methodContext.setPatternBehavior("creates product objects");
                            methodContext.setPattern(pattern);
                        }
                    }
                }
                // Singleton pattern methods
                else if (pattern.equals("singleton")) {
                    if (lowerMethodName.contains("getinstance") || lowerMethodName.contains("get_instance") 
                            || lowerMethodName.equals("getinstance")) {
                        methodContext.setPatternBehavior("returns the singleton instance");
                        methodContext.setPattern(pattern);
                    }
                }
                // Visitor pattern methods
                else if (pattern.equals("visitor")) {
                    if (role != null && role.contains("element")) {
                        if (lowerMethodName.contains("accept")) {
                            methodContext.setPatternBehavior("accepts visitor");
                            methodContext.setPattern(pattern);
                        }
                    } else if (role != null && role.contains("visitor")) {
                        if (lowerMethodName.contains("visit")) {
                            methodContext.setPatternBehavior("visits element");
                            methodContext.setPattern(pattern);
                        }
                    }
                }
            }
        }
        
        return methodContext;
    }
}

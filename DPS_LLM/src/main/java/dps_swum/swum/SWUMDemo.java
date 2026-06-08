package dps_swum.swum;

import dps_swum.swum.grammar.SWUMGrammarParser;
import dps_swum.swum.model.SWUMStructure;

/**
 * Demonstration class for testing SWUM evaluation capabilities.
 * <p>
 * This class provides a simple command-line demo that showcases the SWUM grammar
 * parsing and summarization features. It tests SWUM on sample method and class names
 * to illustrate how the system analyzes and summarizes code identifiers.
 * </p>
 * <p>
 * Key functionalities:
 * <ul>
 *   <li>Test SWUM grammar parsing on sample method names</li>
 *   <li>Demonstrate method and class summary generation</li>
 *   <li>Display parsed actions and objects from identifier names</li>
 *   <li>Show grammar rule application statistics</li>
 * </ul>
 * </p>
 * 
 * @author Najam
 */
public class SWUMDemo {
    
    /**
     * Runs the SWUM demonstration tests.
     * <p>
     * Executes grammar parsing and summarization tests on sample code identifiers
     * and displays the results to the console.
     * </p>
     * 
     * @param args command line arguments (currently unused)
     */
    public static void main(String[] args) {
        System.out.println("=== SWUM (Software Word Usage Model) Demonstration ===\n");
        
        // Test SWUM grammar parsing
        testSWUMGrammarParsing();
        
        // Test SWUM summarization
        testSWUMSummarization();
        
        // Test evaluation metrics
        //testEvaluationMetrics();
        
        System.out.println("\n=== SWUM Demo Complete ===");
        System.out.println("To run the full evaluation pipeline:");
        System.out.println("java -cp target/classes swum.SWUMEvaluationPipeline");
    }
    
    /**
     * Test SWUM grammar parsing functionality
     */
    private static void testSWUMGrammarParsing() {
        System.out.println("1. Testing SWUM Grammar Parsing:");
        
        try {
            SWUMGrammarParser parser = new SWUMGrammarParser();
            
            String[] testMethods = {
                "getUserAccountFromDatabase",
                "createFactoryInstance", 
                "validateUserInput",
                "processPaymentTransaction",
                "sendEmailNotification"
            };
            
            for (String methodName : testMethods) {
                SWUMStructure structure = parser.parseMethodName(methodName, "TestClass");
                System.out.printf("  Method: %-25s -> Actions: %s, Objects: %s\n", 
                    methodName, 
                    String.join(", ", structure.getActions()),
                    String.join(", ", structure.getObjects())
                );
            }
            
            System.out.printf("  Grammar rules applied: %d\n", parser.getAppliedRulesCount());
            
        } catch (Exception e) {
            System.err.println("  Error in grammar parsing: " + e.getMessage());
        }
        
        System.out.println();
    }
    
    /**
     * Test SWUM summarization
     */
    private static void testSWUMSummarization() {
        System.out.println("2. Testing SWUM Summarization:");
        
        try {
            SWUMSummarizer summarizer = new SWUMSummarizer();
            
            String[] testMethods = {
                "createUserAccount",
                "validatePassword", 
                "processOrder",
                "sendNotification"
            };
            
            String[] testClasses = {
                "UserAccountFactory",
                "PaymentProcessor",
                "EmailAdapter", 
                "OrderBuilder"
            };
            
            System.out.println("  Method Summaries:");
            for (String method : testMethods) {
                String summary = summarizer.generateMethodSummary(method);
                System.out.printf("    %-20s -> %s\n", method, summary);
            }
            
            System.out.println("\n  Class Summaries:");
            for (String className : testClasses) {
                String summary = summarizer.generateClassSummary(className);
                System.out.printf("    %-20s -> %s\n", className, summary);
            }
            
        } catch (Exception e) {
            System.err.println("  Error in summarization: " + e.getMessage());
        }
        
        System.out.println();
    }
}


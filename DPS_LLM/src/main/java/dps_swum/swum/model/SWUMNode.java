package dps_swum.swum.model;

import java.util.ArrayList;
import java.util.List;

/**
 * Represents a node in the SWUM parse tree structure.
 * <p>
 * Each node represents a grammatical component (verb phrase, noun phrase, etc.)
 * in the parsed structure of a method or class name. Nodes can have children
 * forming a tree structure, or be terminal nodes representing individual words.
 * </p>
 * <p>
 * Key responsibilities:
 * <ul>
 *   <li>Store node type and associated word/phrase</li>
 *   <li>Maintain parent-child relationships in the parse tree</li>
 *   <li>Support part-of-speech tagging</li>
 *   <li>Generate string representations of the tree structure</li>
 *   <li>Extract terminal word sequences (yields)</li>
 * </ul>
 * </p>
 * 
 * @author Najam
 */
public class SWUMNode {
    
    /**
     * Enumeration of SWUM node types representing grammatical constructs.
     * <p>
     * These types correspond to syntactic categories used in SWUM grammar:
     * <ul>
     *   <li>VERB_PHRASE - Action or operation</li>
     *   <li>NOUN_PHRASE - Object or entity</li>
     *   <li>PREPOSITION - Relationship indicator</li>
     *   <li>DETERMINER - Article (the, a, an)</li>
     *   <li>SUBJECT - Subject of an action</li>
     *   <li>OBJECT - Object of an action</li>
     *   <li>MODIFIER - Adjective or adverb</li>
     *   <li>CONJUNCTION - Connecting words (and, or, but)</li>
     *   <li>TERMINAL - Leaf node containing actual word</li>
     * </ul>
     * </p>
     */
    public enum NodeType {
        VERB_PHRASE,     // VP - Action or operation
        NOUN_PHRASE,     // NP - Object or entity
        PREPOSITION,     // P - Relationship
        DETERMINER,      // D - Article (the, a, an)
        SUBJECT,         // S - Subject of action
        OBJECT,          // O - Object of action
        MODIFIER,        // M - Adjective or adverb
        CONJUNCTION,     // C - And, or, but
        TERMINAL         // Terminal word
    }
    
    private NodeType type;
    private String word;
    private List<SWUMNode> children;
    private SWUMNode parent;
    private String pos; // Part of speech
    
    /**
     * Constructs a SWUM node with the specified type and word.
     * 
     * @param type the grammatical type of this node
     * @param word the word or phrase associated with this node
     */
    public SWUMNode(NodeType type, String word) {
        this.type = type;
        this.word = word;
        this.children = new ArrayList<>();
        this.pos = "";
    }
    
    /**
     * Constructs a SWUM node with the specified type and empty word.
     * 
     * @param type the grammatical type of this node
     */
    public SWUMNode(NodeType type) {
        this(type, "");
    }
    
    // Getters and setters
    public NodeType getType() { return type; }
    public void setType(NodeType type) { this.type = type; }
    
    public String getWord() { return word; }
    public void setWord(String word) { this.word = word; }
    
    public List<SWUMNode> getChildren() { return children; }
    public void addChild(SWUMNode child) { 
        this.children.add(child); 
        child.setParent(this);
    }
    
    public SWUMNode getParent() { return parent; }
    public void setParent(SWUMNode parent) { this.parent = parent; }
    
    public String getPos() { return pos; }
    public void setPos(String pos) { this.pos = pos; }
    
    public boolean isTerminal() {
        return type == NodeType.TERMINAL;
    }
    
    public boolean hasChildren() {
        return !children.isEmpty();
    }
    
    /**
     * Returns the yield (terminal words) of this subtree
     */
    public List<String> getYield() {
        List<String> yield = new ArrayList<>();
        if (isTerminal()) {
            if (!word.isEmpty()) {
                yield.add(word);
            }
        } else {
            for (SWUMNode child : children) {
                yield.addAll(child.getYield());
            }
        }
        return yield;
    }
    
    /**
     * Returns a string representation of the tree structure
     */
    public String toTreeString() {
        if (isTerminal()) {
            return word.isEmpty() ? type.toString() : word;
        }
        
        StringBuilder sb = new StringBuilder();
        sb.append("(").append(type);
        if (!word.isEmpty()) {
            sb.append(" ").append(word);
        }
        for (SWUMNode child : children) {
            sb.append(" ").append(child.toTreeString());
        }
        sb.append(")");
        return sb.toString();
    }
    
    @Override
    public String toString() {
        return String.join(" ", getYield());
    }
}


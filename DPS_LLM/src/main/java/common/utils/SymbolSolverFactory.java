package common.utils;

import com.github.javaparser.symbolsolver.JavaSymbolSolver;
import com.github.javaparser.symbolsolver.resolution.typesolvers.CombinedTypeSolver;
import com.github.javaparser.symbolsolver.resolution.typesolvers.JarTypeSolver;
import com.github.javaparser.symbolsolver.resolution.typesolvers.JavaParserTypeSolver;
import com.github.javaparser.symbolsolver.resolution.typesolvers.ReflectionTypeSolver;

import java.io.*;
import java.util.ArrayList;
import java.util.List;

/**
 * Factory class for creating and configuring JavaParser symbol solvers.
 * <p>
 * This class provides a centralized mechanism for creating JavaSymbolSolver instances
 * with appropriate type resolvers for analyzing Java projects. It combines multiple
 * type resolution strategies (reflection, source code, JAR files) to enable
 * comprehensive symbol resolution during code analysis.
 * </p>
 * <p>
 * The symbol solver is essential for resolving:
 * <ul>
 *   <li>Type information across project boundaries</li>
 *   <li>Inheritance relationships and interface implementations</li>
 *   <li>Method calls and field references</li>
 *   <li>Dependencies on external libraries</li>
 * </ul>
 * </p>
 * <p>
 * This implementation is shared across NLG, SWUM, and other analysis modules
 * to ensure consistent symbol resolution behavior throughout the project.
 * </p>
 * <p>
 * Referenced from Java Callgraph project.
 * </p>
 * 
 * @author allen
 * @author Najam
 */
public class SymbolSolverFactory {
    
    /**
     * Creates a configured JavaSymbolSolver combining multiple type resolution strategies.
     * <p>
     * The solver is configured with the following priority order:
     * <ol>
     *   <li>ReflectionTypeSolver - for JDK classes (most reliable)</li>
     *   <li>JavaParserTypeSolver - for project source code paths</li>
     *   <li>JarTypeSolver - for external JAR libraries</li>
     * </ol>
     * </p>
     * <p>
     * The method gracefully handles missing or invalid paths and provides
     * diagnostic output to help troubleshoot symbol resolution issues.
     * </p>
     * 
     * @param srcPaths list of source code directory paths
     * @param libPaths list of library directory paths (containing JARs)
     * @return configured JavaSymbolSolver instance
     * @throws IOException if JAR file access fails
     */
    public static JavaSymbolSolver getJavaSymbolSolver(List<String> srcPaths, List<String> libPaths) throws IOException {
        if (srcPaths == null) {
            srcPaths = new ArrayList<>();
        }
        if (libPaths == null) {
            libPaths = new ArrayList<>();
        }
        
        CombinedTypeSolver combinedTypeSolver = new CombinedTypeSolver();
        
        // Add reflection type solver first for JDK classes (most reliable)
        ReflectionTypeSolver reflectionTypeSolver = new ReflectionTypeSolver(false);
        combinedTypeSolver.add(reflectionTypeSolver);
        
        // Add Java parser type solvers with diagnostic output
        List<JavaParserTypeSolver> javaParserTypeSolvers = makeJavaParserTypeSolvers(srcPaths);
        int sourcePathCount = 0;
        for (JavaParserTypeSolver solver : javaParserTypeSolvers) {
            combinedTypeSolver.add(solver);
            sourcePathCount++;
        }
        if (sourcePathCount > 0) {
            System.out.println("  Added " + sourcePathCount + " source path(s) for symbol resolution");
        }
        
        // Add jar type solvers
        List<JarTypeSolver> jarTypeSolvers = makeJarTypeSolvers(libPaths);
        if (jarTypeSolvers.size() > 0) {
            for (JarTypeSolver solver : jarTypeSolvers) {
                combinedTypeSolver.add(solver);
            }
            System.out.println("  Added " + jarTypeSolvers.size() + " JAR file(s) for symbol resolution");
        }
        
        // Create and return the symbol solver
        JavaSymbolSolver symbolSolver = new JavaSymbolSolver(combinedTypeSolver);
        return symbolSolver;
    }

    /**
     * Creates JAR type solvers for all JAR files found in the specified library paths.
     * 
     * @param libPaths list of library directory paths
     * @return list of configured JarTypeSolver instances
     * @throws IOException if JAR file cannot be read
     */
    private static List<JarTypeSolver> makeJarTypeSolvers(List<String> libPaths) throws IOException {
        List<String> jarPaths = Utils.getFilesBySuffixInPaths("jar", libPaths);
        List<JarTypeSolver> jarTypeSolvers = new ArrayList<>(jarPaths.size());
        for (String jarPath : jarPaths) {
            jarTypeSolvers.add(new JarTypeSolver(jarPath));
        }
        return jarTypeSolvers;
    }

    /**
     * Creates Java parser type solvers for all valid source directories.
     * 
     * @param srcPaths list of source code directory paths
     * @return list of configured JavaParserTypeSolver instances
     */
    private static List<JavaParserTypeSolver> makeJavaParserTypeSolvers(List<String> srcPaths) {
        List<JavaParserTypeSolver> javaParserTypeSolvers = new ArrayList<>();
        for (String srcPath : srcPaths) {
            if (srcPath == null || srcPath.isEmpty()) {
                continue;
            }
            File srcDir = new File(srcPath);
            if (srcDir.exists() && srcDir.isDirectory()) {
                JavaParserTypeSolver typeSolver = new JavaParserTypeSolver(srcDir);
                javaParserTypeSolvers.add(typeSolver);
            }
        }
        return javaParserTypeSolvers;
    }

    /**
     * Convenience method to create a symbol solver for single source and library paths.
     * 
     * @param srcPath single source code directory path
     * @param libPath single library directory path
     * @return configured JavaSymbolSolver instance
     * @throws IOException if JAR file access fails
     */
    public static JavaSymbolSolver getJavaSymbolSolver(String srcPath, String libPath) throws IOException {
        return getJavaSymbolSolver(Utils.makeListFromOneElement(srcPath), Utils.makeListFromOneElement(libPath));
    }
}

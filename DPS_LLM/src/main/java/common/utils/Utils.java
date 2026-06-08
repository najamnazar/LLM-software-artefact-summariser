package common.utils;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

import com.github.javaparser.ast.NodeList;

/**
 * Shared utility class providing helper methods for file operations, string manipulation,
 * and data extraction from parsed code structures.
 * <p>
 * This class contains static utility methods used across multiple modules (NLG, SWUM, LLM)
 * for common operations like file handling, name parsing, and extracting specific details
 * from the HashMap-based class representations produced by JavaParser.
 * </p>
 * <p>
 * Key functionalities:
 * <ul>
 *   <li>File extension and basename extraction</li>
 *   <li>String manipulation (camelCase splitting, special character removal)</li>
 *   <li>File system operations (recursive file finding)</li>
 *   <li>Data extraction from parsed class structures</li>
 *   <li>Method override checking and parameter matching</li>
 * </ul>
 * </p>
 * 
 * @author Najam
 */
public class Utils {
    
    /**
     * Gets the file extension from a file.
     * 
     * @param f the file to extract extension from
     * @return the file extension in lowercase, or empty string if none
     */
    public static String getExtension(File f) {
        if (f == null) {
            return "";
        }
        String s = f.getName();
        int i = s.lastIndexOf('.');

        if (i > 0 && i < s.length() - 1) {
            return s.substring(i + 1).toLowerCase();
        }
        return "";
    }

    /**
     * Extracts the base name of a file (without extension).
     * 
     * @param fileName the file name to process
     * @return the file name without extension
     */
    public static String getBaseName(String fileName) {
        if (fileName == null || fileName.isEmpty()) {
            return "";
        }
        int index = fileName.lastIndexOf('.');
        if (index == -1) {
            return fileName;
        } else {
            return fileName.substring(0, index);
        }
    }

    /**
     * Splits a string by dot separator and returns the specified part.
     * 
     * @param parsedString the string to split
     * @param part which part of the split string to be returned (1 for method and 2 for class name when used for caller, 1 for class when used for param)
     * @return the requested part of the string
     */
    public static String splitByDot(String parsedString, int part) {
        if (parsedString == null || !parsedString.contains(".")) {
            return parsedString;
        }
        String[] parsedStringList = parsedString.split("\\.", -1);
        int index = parsedStringList.length - part;
        return (index >= 0 && index < parsedStringList.length) ? parsedStringList[index] : parsedString;
    }

    /**
     * Removes special characters from a string, keeping only letters.
     * 
     * @param parsedString the string to clean
     * @return string with only alphabetic characters
     */
    public static String removeSpecialCharacter(String parsedString) {
        if (parsedString == null) {
            return "";
        }
        return parsedString.replaceAll("[^a-zA-Z]+", "");
    }

    /**
     * Checks if a string should be skipped based on skip patterns.
     * 
     * @param s the string to check
     * @param skipPatterns list of regex patterns to match against
     * @return true if the string matches any skip pattern
     */
    public static boolean shouldSkip(String s, List<Pattern> skipPatterns) {
        if (s == null || skipPatterns == null) {
            return false;
        }
        for (Pattern skipPattern : skipPatterns) {
            if (skipPattern.matcher(s).matches()) {
                return true;
            }
        }
        return false;
    }

    /**
     * Recursively finds all files with the specified suffix in a directory.
     * 
     * @param suffix the file suffix to search for (e.g., ".java")
     * @param path the directory path to search
     * @return list of file paths matching the suffix
     * @throws IOException if file system traversal fails
     */
    public static List<String> getFilesBySuffixInPath(String suffix, String path) throws IOException {
        if (suffix == null || path == null) {
            return new ArrayList<>();
        }
        return Files.find(Paths.get(path), Integer.MAX_VALUE, (filePath, fileAttr) -> fileAttr.isRegularFile())
                .filter(f -> f.toString().toLowerCase().endsWith(suffix.toLowerCase()))
                .map(f -> f.toString())
                .collect(Collectors.toList());
    }

    /**
     * Recursively finds all files with the specified suffix in multiple directories.
     * 
     * @param suffix the file suffix to search for
     * @param paths list of directory paths to search
     * @return list of file paths matching the suffix
     * @throws IOException if file system traversal fails
     */
    public static List<String> getFilesBySuffixInPaths(String suffix, List<String> paths) throws IOException {
        if (paths == null) {
            return new ArrayList<>();
        }
        List<String> files = new ArrayList<>();
        for (String path : paths) {
            files.addAll(getFilesBySuffixInPath(suffix, path));
        }
        return files;
    }

    /**
     * Creates a list from a single element.
     * 
     * @param <T> the type of the element
     * @param object the element to wrap in a list
     * @return list containing the element, or empty list if element is null
     */
    public static <T> List<T> makeListFromOneElement(T object) {
        ArrayList<T> list = new ArrayList<>();
        if (object != null) {
            list.add(object);
        }
        return list;
    }

    /**
     * Converts a JavaParser NodeList to an ArrayList of strings.
     * 
     * @param nodeList the NodeList to convert
     * @return ArrayList containing string representations of nodes
     */
    public static ArrayList<String> nodeListToArrayList(NodeList nodeList) {
        ArrayList<String> arrayList = new ArrayList<>();
        if (nodeList != null) {
            for (Object node : nodeList) {
                if (node != null) {
                    arrayList.add(node.toString().strip());
                }
            }
        }
        return arrayList;
    }

    /**
     * Converts underscores to spaces in text.
     * 
     * @param text the text to convert
     * @return text with underscores replaced by spaces
     */
    public static String convertToPlainText(String text) {
        if (text == null) {
            return "";
        }
        if (text.contains("_")) {
            text = text.replace("_", " ");
        }
        return text;
    }

    /**
     * Splits a camelCase string into separate words.
     * 
     * @param text the camelCase text to split
     * @return array of words
     */
    public static String[] splitByCamelCase(String text) {
        if (text == null || text.isEmpty()) {
            return new String[0];
        }
        String regex = "(?<=\\p{Ll})(?=\\p{Lu})";
        return text.split(regex);
    }

    /**
     * Extracts project details from file details map.
     * 
     * @param fileDetails map of file details
     * @return project details HashMap
     */
    public static HashMap getProjectDetails(HashMap<String, HashMap> fileDetails) {
        if (fileDetails == null) {
            return new HashMap<>();
        }
        for (Map.Entry<String, HashMap> fileEntry : fileDetails.entrySet()) {
            if (!fileEntry.getKey().equals("design_pattern")) {
                return fileEntry.getValue();
            }
        }
        return new HashMap<>();
    }

    // Najam, 2026-06-04: The ternary only guarded against a null map; when the key is absent or its value is
    // explicitly null, HashMap.get() returns null and the cast propagates that null to callers. Callers such as
    // extractCallgraphResults() iterate the result directly, causing NPE. Each method now falls back to an empty
    // list so all callers receive a safe, non-null collection regardless of map contents.
    // public static ArrayList<HashMap> getMethodDetails(HashMap classDetails) {
    //     return classDetails == null ? new ArrayList<>() : (ArrayList<HashMap>) classDetails.get("METHODDETAIL");
    // }
    public static ArrayList<HashMap> getMethodDetails(HashMap classDetails) {
        if (classDetails == null) return new ArrayList<>();
        ArrayList<HashMap> result = (ArrayList<HashMap>) classDetails.get("METHODDETAIL");
        return result != null ? result : new ArrayList<>();
    }

    // Najam, 2026-06-04: Same null-value guard as getMethodDetails above.
    // public static ArrayList<HashMap> getClassOrInterfaceDetails(HashMap classDetails) {
    //     return classDetails == null ? new ArrayList<>() : (ArrayList<HashMap>) classDetails.get("CLASSORINTERFACEDETAIL");
    // }
    public static ArrayList<HashMap> getClassOrInterfaceDetails(HashMap classDetails) {
        if (classDetails == null) return new ArrayList<>();
        ArrayList<HashMap> result = (ArrayList<HashMap>) classDetails.get("CLASSORINTERFACEDETAIL");
        return result != null ? result : new ArrayList<>();
    }

    // Najam, 2026-06-04: Same null-value guard as getMethodDetails above.
    // public static ArrayList<HashMap> getFieldDetails(HashMap classDetails) {
    //     return classDetails == null ? new ArrayList<>() : (ArrayList<HashMap>) classDetails.get("FIELDDETAIL");
    // }
    public static ArrayList<HashMap> getFieldDetails(HashMap classDetails) {
        if (classDetails == null) return new ArrayList<>();
        ArrayList<HashMap> result = (ArrayList<HashMap>) classDetails.get("FIELDDETAIL");
        return result != null ? result : new ArrayList<>();
    }

    // Najam, 2026-06-04: Same null-value guard as getMethodDetails above.
    // public static ArrayList<HashMap> getConstructorDetails(HashMap classDetails) {
    //     return classDetails == null ? new ArrayList<>() : (ArrayList<HashMap>) classDetails.get("CONSTRUCTORDETAIL");
    // }
    public static ArrayList<HashMap> getConstructorDetails(HashMap classDetails) {
        if (classDetails == null) return new ArrayList<>();
        ArrayList<HashMap> result = (ArrayList<HashMap>) classDetails.get("CONSTRUCTORDETAIL");
        return result != null ? result : new ArrayList<>();
    }

    // Najam, 2026-06-04: Same null-value guard as getMethodDetails above.
    // public static ArrayList<HashMap> getVariableDetails(HashMap classDetails) {
    //     return classDetails == null ? new ArrayList<>() : (ArrayList<HashMap>) classDetails.get("VARIABLEDETAIL");
    // }
    public static ArrayList<HashMap> getVariableDetails(HashMap classDetails) {
        if (classDetails == null) return new ArrayList<>();
        ArrayList<HashMap> result = (ArrayList<HashMap>) classDetails.get("VARIABLEDETAIL");
        return result != null ? result : new ArrayList<>();
    }

    // Najam, 2026-06-04: Same null-value guard as getMethodDetails — map value can be null when key is absent.
    // public static ArrayList<String> getImplementsFrom(HashMap classDetail) {
    //     return classDetail == null ? new ArrayList<>() : (ArrayList<String>) classDetail.get("IMPLEMENTSFROM");
    // }
    public static ArrayList<String> getImplementsFrom(HashMap classDetail) {
        if (classDetail == null) return new ArrayList<>();
        ArrayList<String> result = (ArrayList<String>) classDetail.get("IMPLEMENTSFROM");
        return result != null ? result : new ArrayList<>();
    }

    // Najam, 2026-06-04: Same null-value guard as getMethodDetails — map value can be null when key is absent.
    // public static ArrayList<String> getExtendsFrom(HashMap classDetail) {
    //     return classDetail == null ? new ArrayList<>() : (ArrayList<String>) classDetail.get("EXTENDSFROM");
    // }
    public static ArrayList<String> getExtendsFrom(HashMap classDetail) {
        if (classDetail == null) return new ArrayList<>();
        ArrayList<String> result = (ArrayList<String>) classDetail.get("EXTENDSFROM");
        return result != null ? result : new ArrayList<>();
    }

    public static String getClassName(HashMap classDetail) {
        return classDetail == null ? "" : (String) classDetail.get("CLASSNAME");
    }

    public static String getMethodReturnType(HashMap methodDetail) {
        return methodDetail == null ? "" : (String) methodDetail.get("METHODRETURNTYPE");
    }

    public static String getMethodName(HashMap methodDetail) {
        return methodDetail == null ? "" : (String) methodDetail.get("METHODNAME");
    }

    public static ArrayList<String> getMethodParameterAsText(HashMap methodDetail) {
        ArrayList<String> resultArray = new ArrayList<>();
        if (methodDetail == null) {
            return resultArray;
        }
        Object params = methodDetail.get("METHODPARAMETER");
        if (params instanceof ArrayList) {
            for (HashMap methodParameter : (ArrayList<HashMap>) params) {
                String methodParameterType = (String) methodParameter.get("PARAMETERTYPE");
                String methodParameterName = (String) methodParameter.get("PARAMETERNAME");
                String resultString = methodParameterType + " parameter of " + methodParameterName;
                resultArray.add(resultString);
            }
        }
        return resultArray;
    }

    public static ArrayList<HashMap> getParameters(NodeList parameters) {
        ArrayList<HashMap> resultArray = new ArrayList<>();
        if (parameters == null) {
            return resultArray;
        }
        for (Object parameter : parameters) {
            String[] parameterString = parameter.toString().split(" ");
            if (parameterString.length < 2) {
                continue;
            }
            HashMap resultMap = new HashMap<>();
            resultMap.put("PARAMETERTYPE", parameterString[0]);
            resultMap.put("PARAMETERNAME", parameterString[1]);
            resultArray.add(resultMap);
        }
        return resultArray;
    }

    public static boolean isMethodOverride(HashMap methodDetail) {
        if (methodDetail == null) {
            return false;
        }
        ArrayList<String> methodOverride = (ArrayList) methodDetail.get("METHODOVERRIDE");
        return methodOverride != null && methodOverride.contains("@Override");
    }

    public static boolean isInterfaceOrNot(HashMap classDetail) {
        if (classDetail == null) {
            return false;
        }
        Object isInterface = classDetail.get("ISINTERFACEORNOT");
        return isInterface instanceof Boolean && (Boolean) isInterface;
    }

    public static String getFieldDataType(HashMap fieldDetail) {
        return fieldDetail == null ? "" : (String) fieldDetail.get("FIELDDATATYPE");
    }

    // Najam, 2026-06-04: Same null-value guard as getMethodDetails — map value can be null when key is absent.
    // public static ArrayList<String> getFieldModifierType(HashMap fieldDetail) {
    //     return fieldDetail == null ? new ArrayList<>() : (ArrayList<String>) fieldDetail.get("FIELDMODIFIERTYPE");
    // }
    public static ArrayList<String> getFieldModifierType(HashMap fieldDetail) {
        if (fieldDetail == null) return new ArrayList<>();
        ArrayList<String> result = (ArrayList<String>) fieldDetail.get("FIELDMODIFIERTYPE");
        return result != null ? result : new ArrayList<>();
    }

    // Najam, 2026-06-04: Same null-value guard as getMethodDetails — map value can be null when key is absent.
    // public static ArrayList<String> getMethodModifierType(HashMap methodDetail) {
    //     return methodDetail == null ? new ArrayList<>() : (ArrayList<String>) methodDetail.get("METHODMODIFIERTYPE");
    // }
    public static ArrayList<String> getMethodModifierType(HashMap methodDetail) {
        if (methodDetail == null) return new ArrayList<>();
        ArrayList<String> result = (ArrayList<String>) methodDetail.get("METHODMODIFIERTYPE");
        return result != null ? result : new ArrayList<>();
    }

    // Najam, 2026-06-04: Same null-value guard as getMethodDetails — map value can be null when key is absent.
    // public static ArrayList<String> getConstructorModifier(HashMap constructorDetail) {
    //     return constructorDetail == null ? new ArrayList<>() : (ArrayList<String>) constructorDetail.get("CONSTRUCTORMODIFIER");
    // }
    public static ArrayList<String> getConstructorModifier(HashMap constructorDetail) {
        if (constructorDetail == null) return new ArrayList<>();
        ArrayList<String> result = (ArrayList<String>) constructorDetail.get("CONSTRUCTORMODIFIER");
        return result != null ? result : new ArrayList<>();
    }

    // Najam, 2026-06-04: Same null-value guard as getMethodDetails — map value can be null when key is absent.
    // public static ArrayList<String> getClassModifierType(HashMap classDetail) {
    //     return classDetail == null ? new ArrayList<>() : (ArrayList<String>) classDetail.get("CLASSMODIFIERTYPE");
    // }
    public static ArrayList<String> getClassModifierType(HashMap classDetail) {
        if (classDetail == null) return new ArrayList<>();
        ArrayList<String> result = (ArrayList<String>) classDetail.get("CLASSMODIFIERTYPE");
        return result != null ? result : new ArrayList<>();
    }

    // Najam, 2026-06-04: Same null-value guard as getMethodDetails — map value can be null when key is absent.
    // public static ArrayList<HashMap> getConstructorParameters(HashMap constructorDetail) {
    //     return constructorDetail == null ? new ArrayList<>() : (ArrayList<HashMap>) constructorDetail.get("CONSTRUCTORPARAMETER");
    // }
    public static ArrayList<HashMap> getConstructorParameters(HashMap constructorDetail) {
        if (constructorDetail == null) return new ArrayList<>();
        ArrayList<HashMap> result = (ArrayList<HashMap>) constructorDetail.get("CONSTRUCTORPARAMETER");
        return result != null ? result : new ArrayList<>();
    }

    public static ArrayList<String> getIncomingMethodAsText(HashMap methodDetail) {
        ArrayList<String> resultArray = new ArrayList<>();
        if (methodDetail == null) {
            return resultArray;
        }
        Object incoming = methodDetail.get("INCOMINGMETHOD");
        if (incoming instanceof ArrayList) {
            for (HashMap incomingMethod : (ArrayList<HashMap>) incoming) {
                String incomingMethodClass = (String) incomingMethod.get("CALLEDCLASS");
                String incomingMethodName = (String) incomingMethod.get("CALLEDMETHODNAME");
                String resultString = incomingMethodName + " method of " + incomingMethodClass;
                if (!resultArray.contains(resultString)) {
                    resultArray.add(resultString);
                }
            }
        }
        return resultArray;
    }

    public static ArrayList<String> getOutgoingMethodAsText(HashMap methodDetail) {
        ArrayList<String> resultArray = new ArrayList<>();
        if (methodDetail == null) {
            return resultArray;
        }
        Object outgoing = methodDetail.get("OUTGOINGMETHOD");
        if (outgoing instanceof ArrayList) {
            for (HashMap outgoingMethod : (ArrayList<HashMap>) outgoing) {
                String outgoingMethodClass = (String) outgoingMethod.get("CALLEECLASS");
                String outgoingMethodName = (String) outgoingMethod.get("CALLEEMETHODNAME");
                String resultString = outgoingMethodName + " method of " + outgoingMethodClass;
                if (!resultArray.contains(resultString)) {
                    resultArray.add(resultString);
                }
            }
        }
        return resultArray;
    }

    // Najam, 2026-06-04: Same null-value guard as getMethodDetails — map value can be null when key is absent.
    // public static ArrayList<HashMap> getMethodParameters(HashMap methodDetail) {
    //     return methodDetail == null ? new ArrayList<>() : (ArrayList<HashMap>) methodDetail.get("METHODPARAMETER");
    // }
    public static ArrayList<HashMap> getMethodParameters(HashMap methodDetail) {
        if (methodDetail == null) return new ArrayList<>();
        ArrayList<HashMap> result = (ArrayList<HashMap>) methodDetail.get("METHODPARAMETER");
        return result != null ? result : new ArrayList<>();
    }

    /**
     * Compares methods of parent and current class to identify overridden methods.
     * Checks if name, parameters and return type are matching and overridden.
     * 
     * @param currentClass the current class details
     * @param parentClass the parent class details
     * @param methodOfParentString suffix to append to method names
     * @return list of overridden method descriptions
     */
    public static ArrayList<String> checkMethodOverride(HashMap currentClass, HashMap parentClass,
            String methodOfParentString) {
        ArrayList<String> overrideMethodArray = new ArrayList<>();
        if (currentClass == null || parentClass == null) {
            return overrideMethodArray;
        }
        
        for (HashMap methodDetail : Utils.getMethodDetails(currentClass)) {
            String currentMethodReturnType = Utils.getMethodReturnType(methodDetail);
            String currentMethodName = Utils.getMethodName(methodDetail);
            ArrayList<String> currentMethodParametersAsText = Utils.getMethodParameterAsText(methodDetail);

            if (!Utils.isMethodOverride(methodDetail)) {
                continue;
            }

            for (HashMap parentMethodDetail : Utils.getMethodDetails(parentClass)) {
                String parentMethodReturnType = Utils.getMethodReturnType(parentMethodDetail);
                String parentMethodName = Utils.getMethodName(parentMethodDetail);
                ArrayList<String> parentMethodParametersAsText = Utils.getMethodParameterAsText(parentMethodDetail);

                if (currentMethodName.equals(parentMethodName)
                        && currentMethodParametersAsText.equals(parentMethodParametersAsText)
                        && currentMethodReturnType.equals(parentMethodReturnType)) {
                    overrideMethodArray.add(parentMethodName + methodOfParentString);
                }
            }
        }
        return overrideMethodArray;
    }

    // Najam, 2026-06-04: Same null-value guard as getMethodDetails — map value can be null when key is absent.
    // public static ArrayList<HashMap> getOutgoingMethod(HashMap methodDetail) {
    //     return methodDetail == null ? new ArrayList<>() : (ArrayList<HashMap>) methodDetail.get("OUTGOINGMETHOD");
    // }
    public static ArrayList<HashMap> getOutgoingMethod(HashMap methodDetail) {
        if (methodDetail == null) return new ArrayList<>();
        ArrayList<HashMap> result = (ArrayList<HashMap>) methodDetail.get("OUTGOINGMETHOD");
        return result != null ? result : new ArrayList<>();
    }

    // Najam, 2026-06-04: Same null-value guard as getMethodDetails — map value can be null when key is absent.
    // public static ArrayList<HashMap> getIncomingMethod(HashMap methodDetail) {
    //     return methodDetail == null ? new ArrayList<>() : (ArrayList<HashMap>) methodDetail.get("INCOMINGMETHOD");
    // }
    public static ArrayList<HashMap> getIncomingMethod(HashMap methodDetail) {
        if (methodDetail == null) return new ArrayList<>();
        ArrayList<HashMap> result = (ArrayList<HashMap>) methodDetail.get("INCOMINGMETHOD");
        return result != null ? result : new ArrayList<>();
    }

    public static String getIncomingMethodClass(HashMap incomingMethod) {
        return incomingMethod == null ? "" : (String) incomingMethod.getOrDefault("CALLEDCLASS", "");
    }

    public static String getIncomingMethodName(HashMap incomingMethod) {
        return incomingMethod == null ? "" : (String) incomingMethod.getOrDefault("CALLEDMETHODNAME", "");
    }

    public static String getOutgoingMethodClass(HashMap outgoingMethod) {
        return outgoingMethod == null ? "" : (String) outgoingMethod.getOrDefault("CALLEECLASS", "");
    }

    public static String getOutgoingMethodName(HashMap outgoingMethod) {
        return outgoingMethod == null ? "" : (String) outgoingMethod.getOrDefault("CALLEEMETHODNAME", "");
    }

    public static String getParameterType(HashMap parameters) {
        return parameters == null ? "" : (String) parameters.getOrDefault("PARAMETERTYPE", "");
    }

    public static String getParameterName(HashMap parameters) {
        return parameters == null ? "" : (String) parameters.getOrDefault("PARAMETERNAME", "");
    }

    public static String getMethodNameFromMatchingParameterType(HashMap methodDetail, String className) {
        if (methodDetail == null || className == null) {
            return "";
        }
        for (HashMap parameter : Utils.getMethodParameters(methodDetail)) {
            if (className.equals(Utils.getParameterType(parameter))) {
                return Utils.getMethodName(methodDetail);
            }
        }
        return "";
    }

    public static String getMethodNameFromMatchingReturnType(HashMap methodDetail, String className) {
        if (methodDetail == null || className == null) {
            return "";
        }
        if (className.equals(Utils.getMethodReturnType(methodDetail))) {
            return Utils.getMethodName(methodDetail);
        }
        return "";
    }

    public static ArrayList<String> getMethodNameFromMatchingIncomingMethod(HashMap methodDetail, String className,
            String originClass) {
        ArrayList<String> resultArrayList = new ArrayList<>();
        if (methodDetail == null || className == null || originClass == null) {
            return resultArrayList;
        }
        
        for (HashMap incomingMethod : Utils.getIncomingMethod(methodDetail)) {
            if (className.equals(Utils.getIncomingMethodClass(incomingMethod))) {
                resultArrayList.add(Utils.getMethodName(methodDetail) + " method of " + originClass);
            }
        }
        return resultArrayList;
    }
}

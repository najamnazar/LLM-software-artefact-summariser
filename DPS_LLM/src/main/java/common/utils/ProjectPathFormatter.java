package common.utils;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * Utility class for transforming raw project identifiers into user-friendly names.
 * <p>
 * This class provides helpers for parsing and formatting project identifiers that flow
 * through the summarization pipelines. It handles various separator conventions
 * (slashes, underscores) and normalizes them into structured project and folder names.
 * </p>
 * <p>
 * Key responsibilities:
 * <ul>
 *   <li>Split composite project identifiers into parts</li>
 *   <li>Normalize path separators (/, \, _)</li>
 *   <li>Format folder hierarchies with readable separators</li>
 *   <li>Handle edge cases (null, blank, malformed identifiers)</li>
 * </ul>
 * </p>
 * 
 * @author Najam
 */
public final class ProjectPathFormatter {

    private ProjectPathFormatter() {
        // Utility class - prevent instantiation
    }

    /**
     * Immutable container for parsed project path components.
     */
    public static final class Parts {
        private final String projectName;
        private final String folderName;

        /**
         * Constructs a new Parts instance with the specified names.
         * 
         * @param projectName the project name (null-safe, trimmed)
         * @param folderName the folder path (null-safe, trimmed)
         */
        public Parts(String projectName, String folderName) {
            this.projectName = Objects.requireNonNullElse(projectName, "").trim();
            this.folderName = Objects.requireNonNullElse(folderName, "").trim();
        }

        /**
         * Returns the project name.
         * 
         * @return project name, never null
         */
        public String projectName() {
            return projectName;
        }

        /**
         * Returns the folder name/path.
         * 
         * @return folder name, never null
         */
        public String folderName() {
            return folderName;
        }
    }

    /**
     * Splits a raw project identifier into project name and folder path components.
     * <p>
     * The method handles various separator conventions:
     * <ul>
     *   <li>Forward slashes (/) and backslashes (\) as primary separators</li>
     *   <li>Underscores (_) as fallback separators</li>
     *   <li>Multiple underscores as directory level indicators</li>
     * </ul>
     * </p>
     * <p>
     * Example transformations:
     * <ul>
     *   <li>"ProjectA/folder1/folder2" → projectName: "ProjectA", folderName: "folder1 / folder2"</li>
     *   <li>"ProjectB_subfolder_nested" → projectName: "ProjectB", folderName: "subfolder / nested"</li>
     *   <li>"SimpleProject" → projectName: "SimpleProject", folderName: ""</li>
     * </ul>
     * </p>
     * 
     * @param rawIdentifier the raw project identifier to parse
     * @return Parts object containing project and folder names
     */
    public static Parts split(String rawIdentifier) {
        if (rawIdentifier == null || rawIdentifier.isBlank()) {
            return new Parts("", "");
        }

        String normalized = rawIdentifier.trim().replace('\\', '/');
        String projectName = normalized;
        String folderPortion = "";

        // First try to split by slash
        int slashIndex = normalized.indexOf('/');
        if (slashIndex >= 0) {
            projectName = normalized.substring(0, slashIndex).trim();
            folderPortion = normalized.substring(slashIndex + 1).trim();
        } else {
            // Fall back to underscore as separator
            int underscoreIndex = normalized.indexOf('_');
            if (underscoreIndex >= 0) {
                projectName = normalized.substring(0, underscoreIndex).trim();
                folderPortion = normalized.substring(underscoreIndex + 1).trim();
            }
        }

        if (folderPortion.isEmpty()) {
            return new Parts(projectName, "");
        }

        // Normalize all separators to forward slashes
        folderPortion = folderPortion.replace('\\', '/');
        // Treat multiple underscores as directory separators when no slashes exist
        folderPortion = folderPortion.replaceAll("_+", "/");

        // Split into segments and clean each one
        String[] rawSegments = folderPortion.split("/");
        List<String> cleanedSegments = new ArrayList<>(rawSegments.length);
        for (String segment : rawSegments) {
            if (segment == null) {
                continue;
            }
            String cleaned = segment.trim();
            if (cleaned.isEmpty()) {
                continue;
            }
            // Collapse repeated whitespace inside the segment
            cleaned = cleaned.replaceAll("\\s+", " ");
            cleanedSegments.add(cleaned);
        }

        String folderName = String.join(" / ", cleanedSegments);
        return new Parts(projectName, folderName);
    }
}

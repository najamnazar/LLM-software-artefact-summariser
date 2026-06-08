package dps_llm.summary;

import java.io.BufferedWriter;
import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;

import common.utils.ProjectPathFormatter;

/**
 * CSV writer for LLM-generated summaries.
 * <p>
 * This class manages the output CSV file that stores generated summaries alongside
 * project and file metadata. It handles CSV formatting, escaping, and ensures
 * thread-safe write operations.
 * </p>
 * <p>
 * Key responsibilities:
 * <ul>
 *   <li>Create and initialize the CSV output file with headers</li>
 *   <li>Write summary rows with proper CSV escaping</li>
 *   <li>Format project paths using ProjectPathFormatter</li>
 *   <li>Truncate overly long summaries to prevent CSV issues</li>
 *   <li>Ensure thread-safe write operations</li>
 * </ul>
 * </p>
 * 
 * @author Najam
 */
public class LlmSummaryWriter implements AutoCloseable {

    private final BufferedWriter writer;

    /**
     * Constructs a new CSV writer and initializes the output file.
     * <p>
     * Creates the output directory if it doesn't exist and writes the CSV header row.
     * </p>
     * 
     * @param outputPath the path to the output CSV file
     * @throws IOException if directory creation or file writing fails
     * @throws IllegalArgumentException if outputPath is null or blank
     */
    public LlmSummaryWriter(String outputPath) throws IOException {
        if (outputPath == null || outputPath.isBlank()) {
            throw new IllegalArgumentException("outputPath must not be null or blank");
        }
        File file = new File(outputPath);
        File parent = file.getParentFile();
        if (parent != null && !parent.exists()) {
            if (!parent.mkdirs() && !parent.exists()) {
                throw new IOException("Failed to create output directory: " + parent.getAbsolutePath());
            }
        }
        
        this.writer = Files.newBufferedWriter(file.toPath(), StandardCharsets.UTF_8);
        writer.write("Project,Folder Name,File Name,Summary\n");
    }

    /**
     * Writes a single summary row to the CSV file.
     * <p>
     * The method is synchronized to ensure thread-safe writes. Long summaries
     * are automatically truncated to 600 characters.
     * </p>
     * 
     * @param projectIdentifier the project identifier (formatted as "ProjectName/FolderName")
     * @param filename the source file name
     * @param summary the generated summary text
     * @throws IOException if writing fails
     * @throws IllegalArgumentException if projectIdentifier or filename is null
     */
    public synchronized void writeRow(String projectIdentifier, String filename, String summary) throws IOException {
        if (projectIdentifier == null) {
            throw new IllegalArgumentException("projectIdentifier must not be null");
        }
        if (filename == null) {
            throw new IllegalArgumentException("filename must not be null");
        }
        String safeSummary = clean(summary);
        ProjectPathFormatter.Parts parts = ProjectPathFormatter.split(projectIdentifier);
        writer.write(String.format("\"%s\",\"%s\",\"%s\",\"%s\"%n",
                escape(parts.projectName()),
                escape(parts.folderName()),
                escape(filename),
                safeSummary));
        writer.flush();
    }

    // Najam, 2026-06-04: Escaping quotes ("→"") before truncating could split a doubled-quote pair ("") at the
    // 600-char boundary, leaving a lone " that breaks CSV field parsing downstream. Truncate on raw text first,
    // then escape so the 600-char limit is applied before any expansion occurs.
    // private String clean(String text) {
    //     if (text == null) {
    //         return "";
    //     }
    //     String cleaned = text.replace("\"", "\"\"")
    //             .replace("\r", " ")
    //             .replace("\n", " ")
    //             .trim();
    //     if (cleaned.length() > 600) {
    //         cleaned = cleaned.substring(0, 600) + "...";
    //     }
    //     return cleaned;
    // }
    private String clean(String text) {
        if (text == null) {
            return "";
        }
        String raw = text.replace("\r", " ")
                .replace("\n", " ")
                .trim();
        if (raw.length() > 600) {
            raw = raw.substring(0, 600) + "...";
        }
        return raw.replace("\"", "\"\"");
    }

    private String escape(String value) {
        if (value == null) {
            return "";
        }
        return value.replace("\"", "\"\"");
    }

    /**
     * Closes the CSV writer and flushes any remaining data.
     * 
     * @throws IOException if closing the writer fails
     */
    @Override
    public void close() throws IOException {
        writer.close();
    }
}

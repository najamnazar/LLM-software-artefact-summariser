package dps_llm.summary;

import java.io.BufferedWriter;
import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.StandardCopyOption;

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
 *   <li>Write to a temporary file and replace the real one only on {@link #commit()}</li>
 *   <li>Create and initialize the CSV output file with headers</li>
 *   <li>Write summary rows with proper CSV escaping</li>
 *   <li>Format project paths using ProjectPathFormatter</li>
 *   <li>Truncate overly long summaries to prevent CSV issues, reporting each one</li>
 *   <li>Ensure thread-safe write operations</li>
 * </ul>
 * </p>
 * 
 * @author Najam
 */
public class LlmSummaryWriter implements AutoCloseable {

    private final BufferedWriter writer;
    private final int maxSummaryChars;
    private int truncatedSummaries;
    /** Where the summaries are meant to land. Untouched until {@link #commit()}. */
    private final File finalFile;
    /** Where rows are actually written until then. */
    private final File partialFile;
    private int rowsWritten;
    private boolean committed;

    /**
     * Constructs a new CSV writer and initializes the output file.
     * <p>
     * Creates the output directory if it doesn't exist and writes the CSV header row.
     * </p>
     * 
     * Najam: the writer used to open the destination directly, which truncated the previous results
     * the moment the run started. A run that then failed -- every request rate-limited, a key
     * rejected, an interrupt -- left nothing behind and no way back, because this tree is not under
     * version control. Rows now accumulate in a .partial file and replace the real one only when
     * {@link #commit()} is called, so a failed run costs nothing that already existed.
     *
     * @param outputPath the path to the output CSV file
     * @param maxSummaryChars longest summary written before truncation, from LLM_SUMMARY_MAX_CHARS
     * @throws IOException if directory creation or file writing fails
     * @throws IllegalArgumentException if outputPath is null or blank, or maxSummaryChars is not positive
     */
    public LlmSummaryWriter(String outputPath, int maxSummaryChars) throws IOException {
        if (outputPath == null || outputPath.isBlank()) {
            throw new IllegalArgumentException("outputPath must not be null or blank");
        }
        if (maxSummaryChars <= 0) {
            throw new IllegalArgumentException("maxSummaryChars must be positive");
        }
        this.maxSummaryChars = maxSummaryChars;
        File file = new File(outputPath);
        File parent = file.getParentFile();
        if (parent != null && !parent.exists()) {
            if (!parent.mkdirs() && !parent.exists()) {
                throw new IOException("Failed to create output directory: " + parent.getAbsolutePath());
            }
        }
        
        this.finalFile = file;
        this.partialFile = new File(outputPath + ".partial");
        // this.writer = Files.newBufferedWriter(file.toPath(), StandardCharsets.UTF_8); // Truncated the previous run's output at startup
        this.writer = Files.newBufferedWriter(partialFile.toPath(), StandardCharsets.UTF_8);
        writer.write("Project,Folder Name,File Name,Summary\n");
    }

    /**
     * Writes a single summary row to the CSV file.
     * <p>
     * The method is synchronized to ensure thread-safe writes. Summaries longer than
     * the configured limit are truncated, and each truncation is reported on stderr and
     * counted, so a shortened summary can never pass unnoticed into the evaluation.
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
        String safeSummary = clean(summary, projectIdentifier, filename);
        ProjectPathFormatter.Parts parts = ProjectPathFormatter.split(projectIdentifier);
        writer.write(String.format("\"%s\",\"%s\",\"%s\",\"%s\"%n",
                escape(parts.projectName()),
                escape(parts.folderName()),
                escape(filename),
                safeSummary));
        writer.flush();
        rowsWritten++;
    }

    // Najam, 2026-06-04: Escaping quotes ("→"") before truncating could split a doubled-quote pair ("") at the
    // character limit, leaving a lone " that breaks CSV field parsing downstream. Truncate on raw text first,
    // then escape so the limit is applied before any expansion occurs.
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
    private String clean(String text, String projectIdentifier, String filename) {
        if (text == null) {
            return "";
        }
        String raw = text.replace("\r", " ")
                .replace("\n", " ")
                .trim();
        if (raw.length() > maxSummaryChars) {
            // Truncation used to be silent, so a shortened summary reached the evaluation
            // looking like the model's complete answer and quietly depressed its scores.
            truncatedSummaries++;
            System.err.printf("  WARNING: truncated summary for %s/%s from %d to %d characters "
                            + "(LLM_SUMMARY_MAX_CHARS). Raise the limit to keep it whole.%n",
                    projectIdentifier, filename, raw.length(), maxSummaryChars);
            raw = raw.substring(0, maxSummaryChars) + "...";
        }
        return raw.replace("\"", "\"\"");
    }

    /** How many rows have been written so far. */
    public synchronized int getRowsWritten() {
        return rowsWritten;
    }

    /**
     * Replaces the real output file with what this run produced.
     * <p>
     * Called only when a run finished and actually has rows. Until then the previous CSV is intact,
     * which is the whole point: an aborted or wholly failed run must not be able to destroy results
     * it could not replace.
     * </p>
     *
     * @throws IOException if the file cannot be closed or moved into place
     */
    public synchronized void commit() throws IOException {
        if (committed) {
            return;
        }
        writer.close();
        Files.move(partialFile.toPath(), finalFile.toPath(), StandardCopyOption.REPLACE_EXISTING);
        committed = true;
    }

    /** How many summaries this writer had to shorten. */
    public int getTruncatedSummaries() {
        return truncatedSummaries;
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
    public synchronized void close() throws IOException {
        if (committed) {
            return; // commit() already closed the stream and moved the file into place
        }
        writer.close();
        // Say plainly what was and was not kept, so a failed run is never mistaken for a completed one.
        System.err.printf("  Not committed: %s keeps its previous contents. This run's %d row(s) are in %s.%n",
                finalFile.getPath(), rowsWritten, partialFile.getPath());
    }
}

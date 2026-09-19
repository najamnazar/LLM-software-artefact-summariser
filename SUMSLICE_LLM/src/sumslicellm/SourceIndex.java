package sumslicellm;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.charset.CharacterCodingException;
import java.nio.charset.CodingErrorAction;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/**
 * Resolves a SumSlice method record to its declaration in the original project
 * source, so the LLM can be given the real code and not only SumSlice's facts.
 *
 * The original SumSlice input names a method by class and method name only
 * (conf/&lt;project&gt;-example.xml has no parameter list), so a class with
 * several overloads of the same name cannot be resolved to one declaration.
 * In that case every overload is returned, each under its own signature, and
 * {@link MethodSource#overloads} records how many there were; nothing is
 * silently guessed.
 *
 * Java is scanned directly rather than parsed with a compiler front end: the
 * datasets are Java 1.4-1.6 sources that a modern parser rejects, and only the
 * declaration text is needed. Comments and string literals are masked out
 * before any brace or parenthesis is counted, so text inside them cannot
 * affect the result.
 */
public final class SourceIndex {

    /** One method's declaration as it appears in the project source. */
    public static final class MethodSource {
        public final boolean found;
        public final String parameters;
        public final String returnType;
        public final String signature;
        public final String body;
        public final int overloads;
        public final String file;

        MethodSource(boolean found, String parameters, String returnType, String signature,
                     String body, int overloads, String file) {
            this.found = found;
            this.parameters = parameters;
            this.returnType = returnType;
            this.signature = signature;
            this.body = body;
            this.overloads = overloads;
            this.file = file;
        }

        static MethodSource missing() {
            return new MethodSource(false, null, null, null, null, 0, null);
        }
    }

    /** A class declaration's own header information. */
    public static final class ClassSource {
        public final boolean found;
        public final String extendsType;
        public final String implementsTypes;
        public final String file;

        ClassSource(boolean found, String extendsType, String implementsTypes, String file) {
            this.found = found;
            this.extendsType = extendsType;
            this.implementsTypes = implementsTypes;
            this.file = file;
        }
    }

    private static final Set<String> KEYWORDS = Set.of(
            "return", "new", "throw", "throws", "case", "else", "do", "while", "if", "for",
            "switch", "synchronized", "catch", "assert", "instanceof", "yield", "break",
            "continue", "default", "super", "this");

    private static final Pattern PREFIX_OK = Pattern.compile("[A-Za-z0-9_$<>\\[\\],.@\\s]+");

    private final Path datasetRoot;
    private final Path root;
    /** simple class name -> source files declaring a top-level type of that name */
    private final Map<String, List<Path>> classes = new TreeMap<>();
    private final Map<Path, String> sourceCache = new HashMap<>();

    private SourceIndex(Path datasetRoot, Path root) {
        this.datasetRoot = datasetRoot;
        this.root = root;
    }

    /**
     * Indexes the source tree of one project under {@code datasetRoot}. The
     * directory is the one whose name starts with the project name, ignoring
     * case (jajuk -> jajuk-src-1.10.5, jedit -> jEdit, nanoXML -> nanoxml).
     */
    public static SourceIndex forProject(Path datasetRoot, String project) throws IOException {
        if (!Files.isDirectory(datasetRoot)) {
            throw new IllegalStateException("Dataset directory not found: " + datasetRoot.toAbsolutePath());
        }
        String want = project.toLowerCase(Locale.ROOT);
        List<Path> hits;
        try (Stream<Path> s = Files.list(datasetRoot)) {
            hits = s.filter(Files::isDirectory)
                    .filter(p -> p.getFileName().toString().toLowerCase(Locale.ROOT).startsWith(want))
                    .sorted()
                    .collect(Collectors.toList());
        }
        if (hits.size() != 1) {
            throw new IllegalStateException("Expected exactly one source directory for project '" + project
                    + "' under " + datasetRoot.toAbsolutePath() + ", found " + hits);
        }
        SourceIndex index = new SourceIndex(datasetRoot, hits.get(0));
        index.scan();
        return index;
    }

    private void scan() throws IOException {
        try (Stream<Path> s = Files.walk(root)) {
            s.filter(Files::isRegularFile)
                    .filter(p -> p.getFileName().toString().endsWith(".java"))
                    .sorted()
                    .forEach(p -> {
                        String n = p.getFileName().toString();
                        classes.computeIfAbsent(n.substring(0, n.length() - 5), k -> new ArrayList<>()).add(p);
                    });
        }
    }

    public Path root() {
        return root;
    }

    public int fileCount() {
        return classes.values().stream().mapToInt(List::size).sum();
    }

    /** Decodes a source file: UTF-8 where valid, ISO-8859-1 otherwise (jhotdraw, jajuk). */
    static String decode(Path file) throws IOException {
        byte[] bytes = Files.readAllBytes(file);
        try {
            return StandardCharsets.UTF_8.newDecoder()
                    .onMalformedInput(CodingErrorAction.REPORT)
                    .onUnmappableCharacter(CodingErrorAction.REPORT)
                    .decode(ByteBuffer.wrap(bytes))
                    .toString();
        } catch (CharacterCodingException e) {
            return new String(bytes, StandardCharsets.ISO_8859_1);
        }
    }

    private String source(Path file) throws IOException {
        String s = sourceCache.get(file);
        if (s == null) {
            s = decode(file);
            sourceCache.put(file, s);
        }
        return s;
    }

    private Path fileFor(String className) {
        List<Path> hits = classes.get(className);
        return hits == null || hits.size() != 1 ? null : hits.get(0);
    }

    /**
     * Replaces every comment and string/character literal with spaces, keeping
     * offsets and line breaks, so braces and parentheses can be counted safely.
     */
    static char[] mask(String s) {
        char[] m = s.toCharArray();
        int n = s.length();
        int i = 0;
        while (i < n) {
            char c = s.charAt(i);
            if (c == '/' && i + 1 < n && s.charAt(i + 1) == '/') {
                while (i < n && s.charAt(i) != '\n') {
                    m[i++] = ' ';
                }
            } else if (c == '/' && i + 1 < n && s.charAt(i + 1) == '*') {
                m[i++] = ' ';
                m[i++] = ' ';
                while (i < n && !(s.charAt(i) == '*' && i + 1 < n && s.charAt(i + 1) == '/')) {
                    if (s.charAt(i) != '\n') {
                        m[i] = ' ';
                    }
                    i++;
                }
                if (i < n) {
                    m[i++] = ' ';
                    if (i < n) {
                        m[i++] = ' ';
                    }
                }
            } else if (c == '"' || c == '\'') {
                char quote = c;
                m[i++] = ' ';
                while (i < n && s.charAt(i) != quote) {
                    if (s.charAt(i) == '\\') {
                        m[i++] = ' ';
                        if (i < n) {
                            if (s.charAt(i) != '\n') {
                                m[i] = ' ';
                            }
                            i++;
                        }
                        continue;
                    }
                    if (s.charAt(i) != '\n') {
                        m[i] = ' ';
                    }
                    i++;
                }
                if (i < n) {
                    m[i++] = ' ';
                }
            } else {
                i++;
            }
        }
        return m;
    }

    /** The {@code extends} and {@code implements} clauses of a class declaration. */
    public ClassSource classHeader(String className) throws IOException {
        Path file = fileFor(className);
        if (file == null) {
            return new ClassSource(false, null, null, null);
        }
        String src = source(file);
        String masked = new String(mask(src));
        Matcher m = Pattern.compile("\\b(?:class|interface|enum)\\s+" + Pattern.quote(className) + "\\b")
                .matcher(masked);
        if (!m.find()) {
            return new ClassSource(false, null, null, rel(file));
        }
        int brace = masked.indexOf('{', m.end());
        if (brace < 0) {
            return new ClassSource(false, null, null, rel(file));
        }
        String header = squash(src.substring(m.end(), brace));
        String ext = clause(header, "extends");
        String impl = clause(header, "implements");
        return new ClassSource(true, ext, impl, rel(file));
    }

    private static String clause(String header, String keyword) {
        Matcher m = Pattern.compile("\\b" + keyword + "\\s+(.*)").matcher(header);
        if (!m.find()) {
            return null;
        }
        String rest = m.group(1);
        for (String stop : new String[]{" extends ", " implements "}) {
            int at = rest.indexOf(stop);
            if (at >= 0) {
                rest = rest.substring(0, at);
            }
        }
        rest = rest.trim();
        return rest.isEmpty() ? null : rest;
    }

    /**
     * Finds the declaration(s) of {@code name} in {@code className}. When the
     * class declares several overloads, {@code returnType} (the value SumSlice
     * recorded, compared without case) is used to narrow them down; if that
     * still leaves more than one, all of them are returned together.
     */
    public MethodSource method(String className, String name, String returnType) throws IOException {
        Path file = fileFor(className);
        if (file == null) {
            return MethodSource.missing();
        }
        String src = source(file);
        char[] maskedChars = mask(src);
        String masked = new String(maskedChars);

        List<Decl> decls = new ArrayList<>();
        Matcher m = Pattern.compile("(?<![A-Za-z0-9_$.])" + Pattern.quote(name) + "\\s*\\(").matcher(masked);
        while (m.find()) {
            Decl d = declarationAt(src, masked, name, m.start());
            if (d != null) {
                decls.add(d);
            }
        }
        if (decls.isEmpty()) {
            return MethodSource.missing();
        }

        int total = decls.size();
        if (total > 1 && returnType != null && !returnType.isBlank()) {
            List<Decl> narrowed = decls.stream()
                    .filter(d -> d.returnType != null && d.returnType.equalsIgnoreCase(returnType))
                    .collect(Collectors.toList());
            if (narrowed.size() == 1) {
                decls = narrowed;
            }
        }
        if (decls.size() == 1) {
            Decl d = decls.get(0);
            return new MethodSource(true, d.parameters, d.returnType, d.signature, d.body, total, rel(file));
        }

        // still ambiguous: hand over every overload rather than guess one
        StringBuilder body = new StringBuilder();
        List<String> params = new ArrayList<>();
        Set<String> types = new LinkedHashSet<>();
        for (int i = 0; i < decls.size(); i++) {
            Decl d = decls.get(i);
            params.add(d.parameters);
            if (d.returnType != null) {
                types.add(d.returnType);
            }
            if (i > 0) {
                body.append("\n\n");
            }
            body.append("// overload ").append(i + 1).append(" of ").append(decls.size())
                    .append(": ").append(d.signature).append('\n').append(d.body);
        }
        return new MethodSource(true, String.join(" | ", params), String.join(" | ", types),
                decls.get(0).signature, body.toString(), total, rel(file));
    }

    private static final class Decl {
        final String parameters;
        final String returnType;
        final String signature;
        final String body;

        Decl(String parameters, String returnType, String signature, String body) {
            this.parameters = parameters;
            this.returnType = returnType;
            this.signature = signature;
            this.body = body;
        }
    }

    /**
     * Confirms that the {@code name (} at {@code nameStart} really starts a
     * method or constructor declaration and, if so, reads it.
     *
     * A declaration is recognised by what stands between the end of the
     * previous member or statement and the name: the modifiers and the return
     * type, which contain no operator, no call and no control keyword. This
     * separates {@code public void run(} from {@code return run(},
     * {@code x = run(}, {@code if (run(} and {@code obj.run(}.
     */
    private static Decl declarationAt(String src, String masked, String name, int nameStart) {
        int from = nameStart;
        while (from > 0 && ";{}".indexOf(masked.charAt(from - 1)) < 0) {
            from--;
        }
        String prefix = squash(masked.substring(from, nameStart));
        if (prefix.isEmpty() || prefix.indexOf('=') >= 0 || prefix.indexOf('(') >= 0
                || prefix.indexOf(')') >= 0 || !PREFIX_OK.matcher(prefix).matches()) {
            return null;
        }
        List<String> tokens = new ArrayList<>();
        for (String t : prefix.split("\\s+")) {
            if (!t.isEmpty() && !t.startsWith("@")) {
                tokens.add(t);
            }
        }
        if (tokens.isEmpty()) {
            return null;
        }
        for (String t : tokens) {
            if (KEYWORDS.contains(t)) {
                return null;
            }
        }

        int open = masked.indexOf('(', nameStart);
        int close = matching(masked, open, '(', ')');
        if (close < 0) {
            return null;
        }
        int i = close + 1;
        while (i < masked.length() && Character.isWhitespace(masked.charAt(i))) {
            i++;
        }
        // an optional throws clause may sit between the parameters and the body
        if (masked.startsWith("throws", i)) {
            int brace = masked.indexOf('{', i);
            int semi = masked.indexOf(';', i);
            i = brace >= 0 && (semi < 0 || brace < semi) ? brace : semi;
            if (i < 0) {
                return null;
            }
        }
        if (i >= masked.length()) {
            return null;
        }

        String parameters = squash(src.substring(open + 1, close));
        String returnType = returnTypeOf(tokens, name);
        String signature = squash(prefix + " " + name + "(" + parameters + ")");
        char next = masked.charAt(i);
        if (next == ';') {
            return new Decl(parameters, returnType, signature,
                    "// declared without a body (abstract method or interface declaration)");
        }
        if (next != '{') {
            return null;
        }
        int end = matching(masked, i, '{', '}');
        if (end < 0) {
            return null;
        }
        return new Decl(parameters, returnType, signature, src.substring(i, end + 1));
    }

    /**
     * The declared return type: the last token of the modifier list, unless the
     * declaration is a constructor (no type) or the token is itself a modifier.
     */
    private static String returnTypeOf(List<String> tokens, String name) {
        String last = tokens.get(tokens.size() - 1);
        Set<String> modifiers = Set.of("public", "protected", "private", "static", "final",
                "abstract", "synchronized", "native", "strictfp", "transient", "volatile", "default");
        return modifiers.contains(last) ? null : last;
    }

    /** Index of the closer matching the opener at {@code open}, or -1. */
    private static int matching(String masked, int open, char opener, char closer) {
        if (open < 0 || open >= masked.length() || masked.charAt(open) != opener) {
            return -1;
        }
        int depth = 0;
        for (int i = open; i < masked.length(); i++) {
            char c = masked.charAt(i);
            if (c == opener) {
                depth++;
            } else if (c == closer) {
                if (--depth == 0) {
                    return i;
                }
            }
        }
        return -1;
    }

    private static String squash(String s) {
        return s.replaceAll("\\s+", " ").trim();
    }

    /** The file's path relative to the dataset root, for the record in the input file. */
    private String rel(Path file) {
        try {
            return datasetRoot.relativize(file).toString();
        } catch (IllegalArgumentException e) {
            return file.toString();
        }
    }

    /** Caches one index per project. */
    public static final class Cache {
        private final Path datasetRoot;
        private final Map<String, SourceIndex> byProject = new LinkedHashMap<>();

        public Cache(Path datasetRoot) {
            this.datasetRoot = datasetRoot;
        }

        public SourceIndex get(String project) throws IOException {
            SourceIndex index = byProject.get(project);
            if (index == null) {
                index = forProject(datasetRoot, project);
                byProject.put(project, index);
            }
            return index;
        }

        public Map<String, SourceIndex> all() {
            return Collections.unmodifiableMap(byProject);
        }
    }
}

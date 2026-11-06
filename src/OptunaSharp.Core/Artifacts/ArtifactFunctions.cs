using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using NumSharp;
using OptunaSharp.Collections;
using OptunaSharp.Internal;
using OptunaSharp.Storages;

namespace OptunaSharp.Artifacts;

/// <summary>
///     Meta information for an artifact.
///     Port of <c>optuna/artifacts/_upload.py: ArtifactMeta</c>.
/// </summary>
/// <param name="ArtifactId">The identifier of the artifact.</param>
/// <param name="Filename">The artifact file name used for the upload.</param>
/// <param name="Mimetype">
///     A MIME type of the artifact. If not specified, the MIME type is guessed from the file
///     extension.
/// </param>
/// <param name="Encoding">
///     An encoding of the artifact, suitable for use as a Content-Encoding header (e.g. gzip).
///     If not specified, the encoding is guessed from the file extension.
/// </param>
public sealed record ArtifactMeta(string ArtifactId, string Filename, string Mimetype, string? Encoding);

/// <summary>
///     Module-level artifact functions: <c>upload_artifact</c>, <c>download_artifact</c>,
///     <c>get_all_artifact_meta</c>.
/// </summary>
public static class ArtifactFunctions
{
    /// <summary>Reserved system-attribute prefix for serialized artifact metadata.</summary>
    internal const string ArtifactsAttrPrefix = "artifacts:";
    /// <summary>Fallback MIME type used when no filename mapping or explicit type is available.</summary>
    internal const string DefaultMimeType = "application/octet-stream";

    /// <summary>
    ///     Upload an artifact to the artifact store (Python <c>upload_artifact</c>).
    /// </summary>
    /// <param name="artifactStore">An artifact store.</param>
    /// <param name="filePath">A path to the file to be uploaded.</param>
    /// <param name="studyOrTrial">A <see cref="Trial" />, <see cref="FrozenTrial" /> or <see cref="Study" />.</param>
    /// <param name="storage">A storage object. Required only if <paramref name="studyOrTrial" /> is a <see cref="FrozenTrial" />.</param>
    /// <param name="mimetype">A MIME type of the artifact; guessed from the file extension when omitted.</param>
    /// <param name="encoding">A Content-Encoding of the artifact; guessed from the file extension when omitted.</param>
    /// <returns>An artifact ID.</returns>
    /// <exception cref="NumSharp.ValueError">No storage can be resolved for the supplied object.</exception>
    /// <exception cref="NumSharp.TypeError">The object is neither Trial, FrozenTrial, nor Study.</exception>
    /// <exception cref="OptunaSharp.UpdateFinishedTrialError">Metadata is written to a finished trial.</exception>
    /// <exception cref="System.IO.IOException">The local input file cannot be read.</exception>
    /// <exception cref="System.UnauthorizedAccessException">Read access to the input file is denied.</exception>
    /// <exception cref="System.Exception">The artifact or metadata backend fails.</exception>
    /// <remarks>Metadata is stored before the content is uploaded. An upload failure can leave metadata pointing to absent or partial content; the operation is not transactional.</remarks>
    public static string UploadArtifact(
        IArtifactStore artifactStore,
        string filePath,
        object studyOrTrial,
        BaseStorage? storage = null,
        string? mimetype = null,
        string? encoding = null)
    {
        string filename = Path.GetFileName(filePath);

        if (studyOrTrial is Trial trialWithStorage && storage is null)
            storage = trialWithStorage.Storage;
        else if (studyOrTrial is Study studyWithStorage && storage is null)
            storage = studyWithStorage.Storage;

        if (storage is null)
            throw new ValueError("storage is required for FrozenTrial.");

        string artifactId = Guid.NewGuid().ToString();
        var (guessMimetype, guessEncoding) = MimeTypes.GuessType(filename);
        // Python's `mimetype or guess_mimetype or DEFAULT_MIME_TYPE` and `encoding or guess_encoding`: an empty
        // string is false there, so "" falls back to the guess exactly like null does (QA-25).
        var artifact = new ArtifactMeta(
            ArtifactId: artifactId,
            Filename: filename,
            Mimetype: !string.IsNullOrEmpty(mimetype) ? mimetype
                : !string.IsNullOrEmpty(guessMimetype) ? guessMimetype
                : DefaultMimeType,
            Encoding: !string.IsNullOrEmpty(encoding) ? encoding : guessEncoding);

        // Metadata and bytes live in separate stores. Publishing metadata first preserves Optuna's
        // ordering, including its nontransactional failure case if the following upload fails.
        string attrKey = ArtifactsAttrPrefix + artifactId;
        string metaJson = PyJson.Dumps(new OrderedDict<string, object?>
        {
            ["artifact_id"] = artifact.ArtifactId,
            ["filename"] = artifact.Filename,
            ["mimetype"] = artifact.Mimetype,
            ["encoding"] = artifact.Encoding,
        });
        switch (studyOrTrial)
        {
            case Trial trial:
                storage.SetTrialSystemAttr(trial.TrialId, attrKey, metaJson);
                break;
            case FrozenTrial frozenTrial:
                storage.SetTrialSystemAttr(frozenTrial.TrialId, attrKey, metaJson);
                break;
            case Study study:
                storage.SetStudySystemAttr(study.StudyId, attrKey, metaJson);
                break;
            default:
                throw new TypeError(
                    $"study_or_trial must be a Trial, FrozenTrial or Study, but got {studyOrTrial.GetType().Name}.");
        }

        using (var f = new FileStream(filePath, FileMode.Open, FileAccess.Read))
        {
            artifactStore.Write(artifactId, f);
        }
        return artifactId;
    }

    /// <summary>
    ///     Download an artifact from the artifact store (Python <c>download_artifact</c>).
    /// </summary>
    /// <param name="artifactStore">An artifact store.</param>
    /// <param name="filePath">A path to save the downloaded artifact.</param>
    /// <param name="artifactId">The identifier of the artifact to download.</param>
    /// <exception cref="System.IO.IOException">The destination already exists or file copying fails.</exception>
    /// <exception cref="OptunaSharp.Artifacts.ArtifactNotFound">The requested artifact does not exist.</exception>
    /// <exception cref="System.UnauthorizedAccessException">The destination cannot be written.</exception>
    /// <remarks>Both streams are disposed on failure, but a partially written destination file is retained.</remarks>
    public static void DownloadArtifact(IArtifactStore artifactStore, string filePath, string artifactId)
    {
        if (File.Exists(filePath))
            throw new IOException($"File already exists: {filePath}");

        using var reader = artifactStore.OpenReader(artifactId);
        using var writer = new FileStream(filePath, FileMode.CreateNew, FileAccess.Write);
        reader.CopyTo(writer);
    }

    /// <summary>
    ///     List the associated artifact information of the provided trial or study
    ///     (Python <c>get_all_artifact_meta</c>). If a <see cref="Study" /> is provided, the
    ///     artifacts uploaded to the study are returned, not those of all its trials.
    /// </summary>
    /// <param name="storage">Storage backend holding study and trial records.</param>
    /// <param name="studyOrTrial">Trial, FrozenTrial, or Study whose artifact metadata is requested.</param>
    /// <returns>Metadata attached directly to the supplied study or trial.</returns>
    /// <exception cref="NumSharp.ValueError">No storage can be resolved, or stored metadata is not valid JSON.</exception>
    /// <exception cref="NumSharp.TypeError">The object is neither Trial, FrozenTrial, nor Study.</exception>
    /// <exception cref="System.InvalidCastException">Stored artifact metadata has an incompatible shape.</exception>
    /// <exception cref="System.Collections.Generic.KeyNotFoundException">A required metadata field is missing.</exception>
    public static List<ArtifactMeta> GetAllArtifactMeta(object studyOrTrial, BaseStorage? storage = null)
    {
        if (studyOrTrial is Trial trialWithStorage && storage is null)
            storage = trialWithStorage.Storage;
        else if (studyOrTrial is Study studyWithStorage && storage is null)
            storage = studyWithStorage.Storage;

        if (storage is null)
            throw new ValueError("storage is required for FrozenTrial.");

        IReadOnlyDictionary<string, object?> systemAttrs = studyOrTrial switch
        {
            Trial trial => storage.GetTrialSystemAttrs(trial.TrialId),
            FrozenTrial frozenTrial => storage.GetTrialSystemAttrs(frozenTrial.TrialId),
            Study study => storage.GetStudySystemAttrs(study.StudyId),
            _ => throw new TypeError(
                $"study_or_trial must be a Trial, FrozenTrial or Study, but got {studyOrTrial.GetType().Name}."),
        };

        var artifactMetaList = new List<ArtifactMeta>();
        foreach (var (attrKey, attrJsonString) in systemAttrs)
        {
            if (!attrKey.StartsWith(ArtifactsAttrPrefix, StringComparison.Ordinal))
                continue;

            var attrContent = (OrderedDict<string, object?>)PyJson.Loads((string)attrJsonString!)!;
            artifactMetaList.Add(new ArtifactMeta(
                ArtifactId: (string)attrContent["artifact_id"]!,
                Filename: (string)attrContent["filename"]!,
                Mimetype: (string)attrContent["mimetype"]!,
                Encoding: (string?)attrContent["encoding"]));
        }

        return artifactMetaList;
    }
}

/// <summary>
///     Port of CPython 3.12's <c>mimetypes.MimeTypes.guess_type(url, strict=True)</c> over its <b>built-in</b> tables
///     (<c>_suffix_map_default</c>, <c>_encodings_map_default</c>, <c>_types_map_default</c>; identical in CPython
///     3.12.10 and 3.12.12): the (MIME type, content encoding) <c>upload_artifact</c> stores for a file name.
/// </summary>
/// <remarks>
///     <para>Owner decision (QA-25, question Q2 of the 2026-11-02 parity QA run): follow CPython's built-in table only.
///     The module-level <c>mimetypes.guess_type</c> Optuna calls also reads the platform's files on first use: on Windows
///     the registry's <c>Content Type</c> values (<c>.csv</c> becomes <c>application/vnd.ms-excel</c>, <c>.zip</c>
///     <c>application/x-zip-compressed</c>), on Linux and macOS <c>/etc/mime.types</c> and its siblings. What Python
///     stores therefore depends on the machine; the port stores the same type everywhere, equal to Python wherever the
///     machine adds no MIME files (containers, CI images). Footgun: the same file name uploaded from a desktop Python
///     and from the port can carry different MIME types.</para>
///     <para>The tables are CPython 3.12.10's. Another CPython answers differently for the extensions its table adds or
///     lacks (3.12.0 has no <c>.md</c>, <c>.markdown</c> or <c>.rst</c>; 3.13 adds more).</para>
///     <para>Not ported, because the input is always a file's base name: <c>os.path.splitdrive</c> (a base name has no
///     drive) and urlsplit's netloc checks (a base name has no <c>/</c>, so a <c>//netloc</c> cannot occur; one is still
///     stripped, but bracketed IPv6 and NFKC netloc validation, which raise <c>ValueError</c> in Python, are not run).</para>
/// </remarks>
internal static class MimeTypes
{
    /// <summary>
    ///     CPython's <c>_suffix_map_default</c>: compound suffixes rewritten to their two-suffix form before the encoding
    ///     is read. Looked up with the suffix lowercased, as 3.12's <c>guess_type</c> does (<c>A.TGZ</c> is a gzip tar).
    /// </summary>
    private static readonly Dictionary<string, string> SuffixMap = new(StringComparer.Ordinal)
    {
        [".svgz"] = ".svg.gz",
        [".tgz"] = ".tar.gz",
        [".taz"] = ".tar.gz",
        [".tz"] = ".tar.gz",
        [".tbz2"] = ".tar.bz2",
        [".txz"] = ".tar.xz",
    };

    /// <summary>
    ///     CPython's <c>_encodings_map_default</c>. Looked up case-sensitively, as in Python: <c>a.GZ</c> has no encoding,
    ///     and only one encoding suffix is ever stripped (<c>a.txt.gz.gz</c> is <c>(None, 'gzip')</c>).
    /// </summary>
    private static readonly Dictionary<string, string> EncodingsMap = new(StringComparer.Ordinal)
    {
        [".gz"] = "gzip",
        [".Z"] = "compress",
        [".bz2"] = "bzip2",
        [".xz"] = "xz",
        [".br"] = "br",
    };

    /// <summary>
    ///     CPython's strict <c>_types_map_default</c> (152 entries, in CPython's order), keyed by lowercase extension.
    ///     <c>strict=True</c>, Optuna's call, never consults the non-standard <c>common_types</c>.
    /// </summary>
    private static readonly Dictionary<string, string> TypesMap = new(StringComparer.Ordinal)
    {
        [".js"] = "text/javascript",
        [".mjs"] = "text/javascript",
        [".json"] = "application/json",
        [".webmanifest"] = "application/manifest+json",
        [".doc"] = "application/msword",
        [".dot"] = "application/msword",
        [".wiz"] = "application/msword",
        [".nq"] = "application/n-quads",
        [".nt"] = "application/n-triples",
        [".bin"] = "application/octet-stream",
        [".a"] = "application/octet-stream",
        [".dll"] = "application/octet-stream",
        [".exe"] = "application/octet-stream",
        [".o"] = "application/octet-stream",
        [".obj"] = "application/octet-stream",
        [".so"] = "application/octet-stream",
        [".oda"] = "application/oda",
        [".pdf"] = "application/pdf",
        [".p7c"] = "application/pkcs7-mime",
        [".ps"] = "application/postscript",
        [".ai"] = "application/postscript",
        [".eps"] = "application/postscript",
        [".trig"] = "application/trig",
        [".m3u"] = "application/vnd.apple.mpegurl",
        [".m3u8"] = "application/vnd.apple.mpegurl",
        [".xls"] = "application/vnd.ms-excel",
        [".xlb"] = "application/vnd.ms-excel",
        [".ppt"] = "application/vnd.ms-powerpoint",
        [".pot"] = "application/vnd.ms-powerpoint",
        [".ppa"] = "application/vnd.ms-powerpoint",
        [".pps"] = "application/vnd.ms-powerpoint",
        [".pwz"] = "application/vnd.ms-powerpoint",
        [".wasm"] = "application/wasm",
        [".bcpio"] = "application/x-bcpio",
        [".cpio"] = "application/x-cpio",
        [".csh"] = "application/x-csh",
        [".dvi"] = "application/x-dvi",
        [".gtar"] = "application/x-gtar",
        [".hdf"] = "application/x-hdf",
        [".h5"] = "application/x-hdf5",
        [".latex"] = "application/x-latex",
        [".mif"] = "application/x-mif",
        [".cdf"] = "application/x-netcdf",
        [".nc"] = "application/x-netcdf",
        [".p12"] = "application/x-pkcs12",
        [".pfx"] = "application/x-pkcs12",
        [".ram"] = "application/x-pn-realaudio",
        [".pyc"] = "application/x-python-code",
        [".pyo"] = "application/x-python-code",
        [".sh"] = "application/x-sh",
        [".shar"] = "application/x-shar",
        [".swf"] = "application/x-shockwave-flash",
        [".sv4cpio"] = "application/x-sv4cpio",
        [".sv4crc"] = "application/x-sv4crc",
        [".tar"] = "application/x-tar",
        [".tcl"] = "application/x-tcl",
        [".tex"] = "application/x-tex",
        [".texi"] = "application/x-texinfo",
        [".texinfo"] = "application/x-texinfo",
        [".roff"] = "application/x-troff",
        [".t"] = "application/x-troff",
        [".tr"] = "application/x-troff",
        [".man"] = "application/x-troff-man",
        [".me"] = "application/x-troff-me",
        [".ms"] = "application/x-troff-ms",
        [".ustar"] = "application/x-ustar",
        [".src"] = "application/x-wais-source",
        [".xsl"] = "application/xml",
        [".rdf"] = "application/xml",
        [".wsdl"] = "application/xml",
        [".xpdl"] = "application/xml",
        [".zip"] = "application/zip",
        [".3gp"] = "audio/3gpp",
        [".3gpp"] = "audio/3gpp",
        [".3g2"] = "audio/3gpp2",
        [".3gpp2"] = "audio/3gpp2",
        [".aac"] = "audio/aac",
        [".adts"] = "audio/aac",
        [".loas"] = "audio/aac",
        [".ass"] = "audio/aac",
        [".au"] = "audio/basic",
        [".snd"] = "audio/basic",
        [".mp3"] = "audio/mpeg",
        [".mp2"] = "audio/mpeg",
        [".opus"] = "audio/opus",
        [".aif"] = "audio/x-aiff",
        [".aifc"] = "audio/x-aiff",
        [".aiff"] = "audio/x-aiff",
        [".ra"] = "audio/x-pn-realaudio",
        [".wav"] = "audio/x-wav",
        [".avif"] = "image/avif",
        [".bmp"] = "image/bmp",
        [".gif"] = "image/gif",
        [".ief"] = "image/ief",
        [".jpg"] = "image/jpeg",
        [".jpe"] = "image/jpeg",
        [".jpeg"] = "image/jpeg",
        [".heic"] = "image/heic",
        [".heif"] = "image/heif",
        [".png"] = "image/png",
        [".svg"] = "image/svg+xml",
        [".tiff"] = "image/tiff",
        [".tif"] = "image/tiff",
        [".ico"] = "image/vnd.microsoft.icon",
        [".ras"] = "image/x-cmu-raster",
        [".pnm"] = "image/x-portable-anymap",
        [".pbm"] = "image/x-portable-bitmap",
        [".pgm"] = "image/x-portable-graymap",
        [".ppm"] = "image/x-portable-pixmap",
        [".rgb"] = "image/x-rgb",
        [".xbm"] = "image/x-xbitmap",
        [".xpm"] = "image/x-xpixmap",
        [".xwd"] = "image/x-xwindowdump",
        [".eml"] = "message/rfc822",
        [".mht"] = "message/rfc822",
        [".mhtml"] = "message/rfc822",
        [".nws"] = "message/rfc822",
        [".css"] = "text/css",
        [".csv"] = "text/csv",
        [".html"] = "text/html",
        [".htm"] = "text/html",
        [".md"] = "text/markdown",
        [".markdown"] = "text/markdown",
        [".n3"] = "text/n3",
        [".txt"] = "text/plain",
        [".bat"] = "text/plain",
        [".c"] = "text/plain",
        [".h"] = "text/plain",
        [".ksh"] = "text/plain",
        [".pl"] = "text/plain",
        [".srt"] = "text/plain",
        [".rtx"] = "text/richtext",
        [".tsv"] = "text/tab-separated-values",
        [".vtt"] = "text/vtt",
        [".py"] = "text/x-python",
        [".rst"] = "text/x-rst",
        [".etx"] = "text/x-setext",
        [".sgm"] = "text/x-sgml",
        [".sgml"] = "text/x-sgml",
        [".vcf"] = "text/x-vcard",
        [".xml"] = "text/xml",
        [".mp4"] = "video/mp4",
        [".mpeg"] = "video/mpeg",
        [".m1v"] = "video/mpeg",
        [".mpa"] = "video/mpeg",
        [".mpe"] = "video/mpeg",
        [".mpg"] = "video/mpeg",
        [".mov"] = "video/quicktime",
        [".qt"] = "video/quicktime",
        [".webm"] = "video/webm",
        [".avi"] = "video/x-msvideo",
        [".movie"] = "video/x-sgi-movie",
    };

    /// <summary>urllib.parse's <c>uses_params</c>: the schemes whose last path segment loses its <c>;params</c>.</summary>
    private static readonly HashSet<string> UsesParams = new(StringComparer.Ordinal)
    {
        "", "ftp", "hdl", "prospero", "http", "imap", "https", "shttp", "rtsp", "rtsps", "rtspu", "sip", "sips", "mms", "sftp",
        "tel",
    };

    /// <summary>
    ///     Guesses (MIME type, content encoding) from a file's base name exactly as CPython 3.12's
    ///     <c>MimeTypes().guess_type(name)</c> does over the built-in tables.
    /// </summary>
    /// <param name="filename">The file's base name; only its text is inspected, no file is opened.</param>
    /// <returns>
    ///     The guessed MIME type (null when the suffix is unknown) and encoding (null when there is none), independently.
    /// </returns>
    /// <remarks>
    ///     Python parses the name as a URL first: a name with a scheme of two or more characters before a colon (Linux
    ///     allows <c>http:a.txt</c>) is guessed from the URL's path, and a <c>data:</c> name from its media type. The
    ///     lowercasing uses <see cref="string.ToLowerInvariant" /> where Python uses <c>str.lower</c>; every table key is
    ///     ASCII, and a character whose full lowercase differs between the two (<c>İ</c>) keeps a non-ASCII mark in both,
    ///     so no lookup can differ.
    /// </remarks>
    internal static (string? Mimetype, string? Encoding) GuessType(string filename)
    {
        string url = filename;
        string? scheme = SplitScheme(filename, out string path);
        // A one-letter scheme is a drive letter to Python (`len(p.scheme) > 1`), so the name is used as given.
        if (scheme is { Length: > 1 })
            url = path;
        else
            scheme = null;

        if (scheme == "data")
            return (DataUrlType(url), null);

        var (root, ext) = SplitExt(url);
        // `while (ext_lower := ext.lower()) in self.suffix_map`: every replacement ends in an encoding suffix, which
        // is not itself in the map, so the loop ends after one rewrite.
        while (SuffixMap.TryGetValue(ext.ToLowerInvariant(), out string? replacement))
            (root, ext) = SplitExt(root + replacement);

        string? encoding = null;
        if (EncodingsMap.TryGetValue(ext, out string? suffixEncoding))
        {
            encoding = suffixEncoding;
            (root, ext) = SplitExt(root);
        }

        return (TypesMap.TryGetValue(ext.ToLowerInvariant(), out string? type) ? type : null, encoding);
    }

    /// <summary>
    ///     The scheme and path <c>urllib.parse.urlparse</c> finds in <paramref name="url" />, the part of it
    ///     <c>guess_type</c> uses.
    /// </summary>
    /// <param name="url">The text to parse.</param>
    /// <param name="path">The URL's path when a scheme is found (netloc, query, fragment and <c>;params</c> removed); otherwise empty.</param>
    /// <returns>The lowercased scheme, or null when the text has none.</returns>
    private static string? SplitScheme(string url, out string path)
    {
        path = "";
        // urlsplit strips leading C0 controls and spaces and deletes tab, CR and LF before it looks for the scheme.
        string cleaned = url.TrimStart(C0ControlOrSpace).Replace("\t", "").Replace("\r", "").Replace("\n", "");
        int colon = cleaned.IndexOf(':');
        if (colon <= 0 || !char.IsAsciiLetter(cleaned[0]))
            return null;
        for (int i = 0; i < colon; i++)
        {
            char c = cleaned[i];
            if (!char.IsAsciiLetterOrDigit(c) && c != '+' && c != '-' && c != '.')
                return null;
        }

        string scheme = cleaned.Substring(0, colon).ToLowerInvariant();
        string rest = cleaned.Substring(colon + 1);
        if (rest.StartsWith("//", StringComparison.Ordinal))
        {
            // _splitnetloc: the netloc runs to the first '/', '?' or '#' after the slashes.
            int end = rest.Length;
            foreach (char delimiter in "/?#")
            {
                int at = rest.IndexOf(delimiter, 2);
                if (at >= 0)
                    end = Math.Min(end, at);
            }
            rest = rest.Substring(end);
        }

        int hash = rest.IndexOf('#');
        if (hash >= 0)
            rest = rest.Substring(0, hash);
        int question = rest.IndexOf('?');
        if (question >= 0)
            rest = rest.Substring(0, question);
        if (UsesParams.Contains(scheme) && rest.Contains(';'))
        {
            // _splitparams: only the last path segment's parameters are removed.
            int slash = rest.LastIndexOf('/');
            int semicolon = slash >= 0 ? rest.IndexOf(';', slash) : rest.IndexOf(';');
            if (semicolon >= 0)
                rest = rest.Substring(0, semicolon);
        }

        path = rest;
        return scheme;
    }

    /// <summary>The media type of a <c>data:</c> URL's path (<c>mediatype [;base64] , data</c>), as <c>guess_type</c> reads it.</summary>
    /// <param name="path">The URL's path, the part after <c>data:</c>.</param>
    /// <returns>The media type, <c>text/plain</c> when it has a parameter or no subtype, or null without a comma.</returns>
    private static string? DataUrlType(string path)
    {
        int comma = path.IndexOf(',');
        if (comma < 0)
            return null;   // a bad data URL
        int semicolon = path.IndexOf(';', 0, comma);
        string type = semicolon >= 0 ? path.Substring(0, semicolon) : path.Substring(0, comma);
        return type.Contains('=') || !type.Contains('/') ? "text/plain" : type;
    }

    /// <summary>
    ///     <c>posixpath.splitext</c>: the root and the extension after the last dot of the last <c>/</c>-separated
    ///     component, where leading dots do not start an extension (<c>.bashrc</c> has none).
    /// </summary>
    /// <param name="p">The path to split.</param>
    /// <returns>The root and the extension, which keeps its dot; the extension is empty when there is none.</returns>
    private static (string Root, string Ext) SplitExt(string p)
    {
        int sepIndex = p.LastIndexOf('/');
        int dotIndex = p.LastIndexOf('.');
        if (dotIndex > sepIndex)
        {
            for (int filenameIndex = sepIndex + 1; filenameIndex < dotIndex; filenameIndex++)
            {
                if (p[filenameIndex] != '.')
                    return (p.Substring(0, dotIndex), p.Substring(dotIndex));
            }
        }

        return (p, "");
    }

    /// <summary>urllib.parse's <c>_WHATWG_C0_CONTROL_OR_SPACE</c>: U+0000 to U+001F and the space.</summary>
    private static readonly char[] C0ControlOrSpace = Enumerable.Range(0, 0x20).Select(i => (char)i).Append(' ').ToArray();
}

using System;
using System.IO;
using System.Linq;
using AwesomeAssertions;
using OptunaSharp.Artifacts;

namespace OptunaSharp.Tests.QA;

/// <summary>
///     Parity bugs in the artifact functions, found by the 2026-11-02 parity QA run
///     (<c>docs/qa/PARITY_QA_2026-11-02.md</c>). Expected values are what Optuna 5.0.0 stores on CPython 3.12
///     with <c>mimetypes</c>' built-in table. A test marked <c>[OpenBugs]</c> fails until its divergence is fixed; the
///     attribute is removed with the fix, and the test then guards against a regression. QA-25 is fixed (2026-11-06):
///     <c>MimeTypes</c> ports CPython 3.12's built-in tables and lookup, the owner's choice for question Q2.
/// </summary>
[TestClass]
public class ArtifactParityBugTests
{
    /// <summary>
    ///     Uploads an empty file with the given name to a scratch store and returns the metadata Optuna-compatible
    ///     readers would see.
    /// </summary>
    /// <param name="fileName">The file name whose extension drives the MIME guess; the file is created empty.</param>
    /// <param name="mimetype">An explicit MIME type, or null to let the upload guess one.</param>
    /// <param name="encoding">An explicit content encoding, or null to let the upload guess one.</param>
    /// <returns>The stored <see cref="ArtifactMeta" /> of the uploaded artifact.</returns>
    /// <exception cref="IOException">The scratch folder cannot be created, written or deleted.</exception>
    /// <remarks>
    ///     Everything lives in a fresh folder under the temp directory that is deleted afterwards, so parallel
    ///     tests never share files.
    /// </remarks>
    private static ArtifactMeta Upload(string fileName, string? mimetype = null, string? encoding = null)
    {
        string dir = Path.Combine(Path.GetTempPath(), "optunasharp-qa-artifacts-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        try
        {
            string file = Path.Combine(dir, fileName);
            File.WriteAllBytes(file, Array.Empty<byte>());
            Directory.CreateDirectory(Path.Combine(dir, "store"));
            var store = new FileSystemArtifactStore(Path.Combine(dir, "store"));
            var study = Optuna.CreateStudy();
            ArtifactFunctions.UploadArtifact(store, file, study, mimetype: mimetype, encoding: encoding);
            return ArtifactFunctions.GetAllArtifactMeta(study).Single();
        }
        finally
        {
            // Remove the scratch file and the store copy whether or not the upload succeeded.
            Directory.Delete(dir, recursive: true);
        }
    }

    /// <summary>
    ///     QA-25 (fixed 2026-11-06): the guessed MIME type and encoding must be CPython's <c>mimetypes.guess_type</c>;
    ///     the port's old hand table mapped types Python does not know, missed ones it does, and stripped encodings
    ///     differently.
    /// </summary>
    /// <remarks>
    ///     CPython 3.12's built-in table (what <c>mimetypes.MimeTypes()</c> knows before any system file or the
    ///     Windows registry is read): <c>.yaml</c> and <c>.webp</c> are unknown, so Optuna stores the default
    ///     <c>application/octet-stream</c>; <c>.tgz</c> goes through <c>suffix_map</c> to
    ///     <c>('application/x-tar', 'gzip')</c>; <c>.tsv</c> is <c>text/tab-separated-values</c>; the encodings map
    ///     is case-sensitive (<c>a.GZ</c> has no encoding); only one encoding suffix is stripped
    ///     (<c>a.txt.gz.gz</c> is <c>(None, 'gzip')</c>). The module-level <c>guess_type</c> Optuna calls also reads
    ///     platform files (on Windows <c>.csv</c> becomes <c>application/vnd.ms-excel</c>); the owner chose the
    ///     built-in table (Q2), so the port answers the same on every machine.
    /// </remarks>
    [TestMethod]
    public void UploadArtifact_GuessedMimeTypes_MatchCPythonMimetypes()
    {
        Upload("a.yaml").Mimetype.Should().Be("application/octet-stream");
        Upload("a.webp").Mimetype.Should().Be("application/octet-stream");
        var tgz = Upload("model.tgz");
        (tgz.Mimetype, tgz.Encoding).Should().Be(("application/x-tar", "gzip"));
        Upload("a.tsv").Mimetype.Should().Be("text/tab-separated-values");
        var doubleGz = Upload("a.txt.gz.gz");
        (doubleGz.Mimetype, doubleGz.Encoding).Should().Be(("application/octet-stream", "gzip"));
    }

    /// <summary>
    ///     QA-25 (explicit values, fixed 2026-11-06): an empty-string MIME type or encoding must fall back to the guess,
    ///     as Python's <c>mimetype or guess_mimetype or DEFAULT_MIME_TYPE</c> does; the port stored the empty string.
    /// </summary>
    /// <remarks>
    ///     Python treats <c>""</c> as false in the <c>or</c> chain; the port's old <c>??</c> only replaced null.
    /// </remarks>
    [TestMethod]
    public void UploadArtifact_EmptyMimeTypeAndEncoding_FallBackLikePython()
    {
        var meta = Upload("model.tar.gz", mimetype: "", encoding: "");

        meta.Mimetype.Should().Be("application/x-tar");
        meta.Encoding.Should().Be("gzip");
    }

    /// <summary>
    ///     QA-25: <c>MimeTypes.GuessType</c> answers exactly what CPython 3.12.12's <c>mimetypes.MimeTypes().guess_type</c>
    ///     (the built-in tables only) answered for each name, generated on 2026-11-06.
    /// </summary>
    /// <param name="name">The base name to guess from.</param>
    /// <param name="mimetype">CPython's MIME type, or null when it has none.</param>
    /// <param name="encoding">CPython's encoding, or null when it has none.</param>
    /// <remarks>
    ///     The rows cover every branch of the lookup: case-insensitive type hits, extensions only the old hand table knew
    ///     (Python has no type, so <c>upload_artifact</c> stores <c>application/octet-stream</c>), the lowercased
    ///     <c>suffix_map</c> loop, case-sensitive encodings with one suffix stripped, <c>posixpath.splitext</c>'s leading
    ///     dots, and the URL parsing <c>guess_type</c> runs first (a scheme of two or more letters, fragments,
    ///     <c>;params</c>, <c>data:</c> URLs, a one-letter "scheme" treated as a drive).
    /// </remarks>
    [TestMethod]
    [DataRow("data.csv", "text/csv", null)]
    [DataRow("archive.zip", "application/zip", null)]
    [DataRow("Report.PDF", "application/pdf", null)]
    [DataRow("notes.TXT", "text/plain", null)]
    [DataRow("model.h5", "application/x-hdf5", null)]
    [DataRow("run.sh", "application/x-sh", null)]
    [DataRow("icon.ico", "image/vnd.microsoft.icon", null)]
    [DataRow("table.tsv", "text/tab-separated-values", null)]
    [DataRow("readme.md", "text/markdown", null)]
    [DataRow("guide.rst", "text/x-rst", null)]
    [DataRow("clip.webm", "video/webm", null)]
    [DataRow("photo.jpe", "image/jpeg", null)]
    [DataRow("config.yaml", null, null)]
    [DataRow("config.yml", null, null)]
    [DataRow("image.webp", null, null)]
    [DataRow("weights.pkl", null, null)]
    [DataRow("array.npy", null, null)]
    [DataRow("backup.tgz", "application/x-tar", "gzip")]
    [DataRow("BACKUP.TGZ", "application/x-tar", "gzip")]
    [DataRow("plot.svgz", "image/svg+xml", "gzip")]
    [DataRow("old.taz", "application/x-tar", "gzip")]
    [DataRow("src.tbz2", "application/x-tar", "bzip2")]
    [DataRow("logs.txz", "application/x-tar", "xz")]
    [DataRow("notes.txt.gz", "text/plain", "gzip")]
    [DataRow("NOTES.TXT.GZ", null, null)]
    [DataRow("a.txt.gz.gz", null, "gzip")]
    [DataRow("a.Z", null, "compress")]
    [DataRow("a.z", null, null)]
    [DataRow("dump.sql.bz2", null, "bzip2")]
    [DataRow("page.html.br", "text/html", "br")]
    [DataRow("x.tar.xz", "application/x-tar", "xz")]
    [DataRow(".bashrc", null, null)]
    [DataRow("..gz", null, null)]
    [DataRow("a.", null, null)]
    [DataRow("noext", null, null)]
    [DataRow("http:a.txt", "text/plain", null)]
    [DataRow("ab:c#d.txt", null, null)]
    [DataRow("http:a.txt;v=1", "text/plain", null)]
    [DataRow("data:text/csv;base64,eA==", "text/csv", null)]
    [DataRow("data:abc", null, null)]
    [DataRow("data:,x", "text/plain", null)]
    [DataRow("c:x.txt", "text/plain", null)]
    public void GuessType_MatchesCPython312BuiltInTable(string name, string? mimetype, string? encoding)
    {
        MimeTypes.GuessType(name).Should().Be((mimetype, encoding));
    }
}

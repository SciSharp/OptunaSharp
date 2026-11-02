using System;
using System.IO;
using System.Linq;
using AwesomeAssertions;
using OptunaSharp.Artifacts;

namespace OptunaSharp.Tests.QA;

/// <summary>
///     Open parity bugs in the artifact functions, found by the 2026-11-02 parity QA run
///     (<c>docs/qa/PARITY_QA_2026-11-02.md</c>). Expected values are what Optuna 5.0.0 stores on CPython 3.12
///     with <c>mimetypes</c>' built-in table; each test fails today and passes once fixed, at which point its
///     <c>[OpenBugs]</c> attribute should be removed.
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
    ///     QA-25: the guessed MIME type and encoding must be CPython's <c>mimetypes.guess_type</c>; the port's
    ///     hand table maps types Python does not know, misses ones it does, and strips encodings differently.
    /// </summary>
    /// <remarks>
    ///     CPython 3.12's built-in table (what <c>mimetypes.MimeTypes()</c> knows before any system file or the
    ///     Windows registry is read): <c>.yaml</c> and <c>.webp</c> are unknown, so Optuna stores the default
    ///     <c>application/octet-stream</c>; <c>.tgz</c> goes through <c>suffix_map</c> to
    ///     <c>('application/x-tar', 'gzip')</c>; <c>.tsv</c> is <c>text/tab-separated-values</c>; the encodings map
    ///     is case-sensitive (<c>a.GZ</c> has no encoding); only one encoding suffix is stripped
    ///     (<c>a.txt.gz.gz</c> is <c>(None, 'gzip')</c>). Note that the module-level <c>guess_type</c> Optuna
    ///     calls also reads platform files (on Windows <c>.csv</c> becomes <c>application/vnd.ms-excel</c>), so
    ///     which table the port should follow is an owner question (Q2 in the QA journal); these cases agree
    ///     between the built-in table and this machine's registry.
    /// </remarks>
    [TestMethod]
    [OpenBugs]
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
    ///     QA-25 (explicit values): an empty-string MIME type or encoding must fall back to the guess, as Python's
    ///     <c>mimetype or guess_mimetype or DEFAULT_MIME_TYPE</c> does; the port stores the empty string.
    /// </summary>
    /// <remarks>
    ///     Python treats <c>""</c> as false in the <c>or</c> chain; the port's <c>??</c> only replaces null.
    /// </remarks>
    [TestMethod]
    [OpenBugs]
    public void UploadArtifact_EmptyMimeTypeAndEncoding_FallBackLikePython()
    {
        var meta = Upload("model.tar.gz", mimetype: "", encoding: "");

        meta.Mimetype.Should().Be("application/x-tar");
        meta.Encoding.Should().Be("gzip");
    }
}

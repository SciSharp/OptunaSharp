using System;
using System.IO;
using AwesomeAssertions;
using OptunaSharp.Artifacts;

namespace OptunaSharp.Tests.QA;

/// <summary>
///     Open parity bugs in the artifact functions found by the 2026-11-10 parity QA run
///     (<c>docs/qa/PARITY_QA_2026-11-10.md</c>). A test marked <c>[OpenBugs]</c> fails until its divergence is fixed.
/// </summary>
[TestClass]
public class Run2ArtifactParityBugTests
{
    /// <summary>
    ///     QA-78: <c>download_artifact</c> refuses any existing path with <c>os.path.exists(file_path)</c>, which is true for
    ///     a directory too, so Optuna raises <c>FileExistsError("File already exists: …")</c>. The port checks
    ///     <c>File.Exists</c>, false for a directory, then opens the directory as a file and fails with
    ///     <c>UnauthorizedAccessException("Access to the path '…' is denied.")</c> on Windows. The port's file case
    ///     already raises <c>IOException("File already exists: …")</c>, its mapping of <c>FileExistsError</c>.
    ///     <b>Fixed 2026-11-12:</b> the existence check covers directories too, as <c>os.path.exists</c> does; since
    ///     NumSharp 0.81.0 both cases raise Python's own class, <c>NumSharp.FileExistsError</c> (an <c>IOException</c>).
    /// </summary>
    [TestMethod]
    public void DownloadOntoAnExistingDirectory_RaisesFileExistsLikeOptuna()
    {
        string root = Path.Combine(Path.GetTempPath(), "optunasharp-qa78-" + Guid.NewGuid().ToString("N"));
        try
        {
            Directory.CreateDirectory(Path.Combine(root, "store"));
            string target = Directory.CreateDirectory(Path.Combine(root, "target")).FullName;
            var store = new FileSystemArtifactStore(Path.Combine(root, "store"));
            string source = Path.Combine(root, "a.txt");
            File.WriteAllText(source, "x");
            string artifactId = ArtifactFunctions.UploadArtifact(store, source, Optuna.CreateStudy());

            Action download = () => ArtifactFunctions.DownloadArtifact(store, target, artifactId);

            download.Should().ThrowExactly<NumSharp.FileExistsError>().WithMessage("File already exists: " + target);
        }
        finally
        {
            // Best effort: the temp folder is cleaned later if a handle is still open.
            try
            {
                Directory.Delete(root, recursive: true);
            }
            catch (IOException)
            {
            }
        }
    }
}

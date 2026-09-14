using System;
using System.IO;
using AwesomeAssertions;
using OptunaSharp.Artifacts;
using OptunaSharp.Samplers;
using OptunaSharp.Storages;

namespace OptunaSharp.Tests.Interop;

/// <summary>
///     Live-Optuna cross-language parity for the artifacts recipe (<c>recipes/artifacts.md</c>).
///     OptunaSharp uploads a file to a <see cref="FileSystemArtifactStore" /> and records its meta on a
///     SQLite-backed trial; then <b>real Optuna</b>, opening the same database and the same artifact
///     directory, lists that meta and downloads the artifact — the retrieved bytes, id and filename must
///     match what OptunaSharp wrote. This proves OptunaSharp's artifact layout (<c>base_path/&lt;id&gt;</c>)
///     and meta system-attribute (<c>artifacts:&lt;id&gt;</c>, the four-field JSON) are byte-compatible with
///     Optuna's, and it also checks OptunaSharp's own upload→download round-trip.
/// </summary>
[TestClass]
public class RecipeArtifactsParityTests : OptunaInteropTestBase
{
    private string _dir = null!;

    /// <summary>Creates a unique working directory for this test's store, database and downloads.</summary>
    [TestInitialize]
    public void ArtInit()
    {
        _dir = Path.Combine(Path.GetTempPath(), $"optunasharp-artrec-{Guid.NewGuid():N}");
        Directory.CreateDirectory(_dir);
    }

    /// <summary>Recursively removes the working directory.</summary>
    [TestCleanup]
    public void ArtCleanup()
    {
        try
        {
            Directory.Delete(_dir, recursive: true);
        }
        catch (IOException)
        {
        }
    }

    private string P(string name) => Path.Combine(_dir, name).Replace('\\', '/');

    /// <summary>
    ///     Artifacts recipe: an OptunaSharp-uploaded artifact (file bytes + trial meta) must be readable and
    ///     downloadable by real Optuna from the same store and database, byte-for-byte.
    /// </summary>
    [TestMethod]
    [TestCategory("RequiresPython")]
    public void Artifacts_UploadDownload_CrossLanguage_Parity()
    {
        PythonSession.EnsureOrInconclusive();

        string baseDir = P("artifacts");
        Directory.CreateDirectory(baseDir);
        string srcPath = P("model.txt");
        string dbUrl = $"sqlite:///{P("study.db")}";
        const string content = "hello artifact 42\nsecond line\n";
        File.WriteAllText(srcPath, content);

        string uploadedId;
        using (var rdb = new RDBStorage(dbUrl))
        {
            var study = Optuna.CreateStudy(storage: rdb, studyName: "art", sampler: new RandomSampler(0));
            var store = new FileSystemArtifactStore(baseDir);
            string? id = null;
            study.Optimize(t =>
            {
                double x = t.SuggestFloat("x", -10, 10);
                id = ArtifactFunctions.UploadArtifact(store, srcPath, t);   // returns the artifact id
                t.SetUserAttr("artifact_id", id);
                return (x - 2) * (x - 2);
            }, nTrials: 1);
            uploadedId = id!;

            // OptunaSharp's own upload -> download round-trip is byte-exact.
            string csDownload = P("cs_dl.txt");
            ArtifactFunctions.DownloadArtifact(store, csDownload, uploadedId);
            File.ReadAllText(csDownload).Should().Be(content, "OptunaSharp upload/download round-trips");
        }

        // Real Optuna opens the SAME database + artifact store, lists the meta, and downloads the file.
        string pyDownload = P("py_dl.txt");
        PyExec(
            "from optuna.artifacts import FileSystemArtifactStore, download_artifact, get_all_artifact_meta\n" +
            $"study = optuna.load_study(study_name='art', storage=r'{dbUrl}')\n" +
            $"store = FileSystemArtifactStore(base_path=r'{baseDir}')\n" +
            "meta0 = get_all_artifact_meta(study.trials[0], storage=study._storage)[0]\n" +
            $"download_artifact(artifact_store=store, file_path=r'{pyDownload}', artifact_id=meta0.artifact_id)\n" +
            "py_id = meta0.artifact_id\n" +
            "py_fn = meta0.filename\n");

        PyStr("py_id").Should().Be(uploadedId, "Optuna reads the artifact id OptunaSharp recorded");
        PyStr("py_fn").Should().Be("model.txt", "Optuna reads the filename OptunaSharp recorded");
        File.ReadAllText(pyDownload).Should().Be(content, "Optuna downloads exactly the bytes OptunaSharp uploaded");
    }
}

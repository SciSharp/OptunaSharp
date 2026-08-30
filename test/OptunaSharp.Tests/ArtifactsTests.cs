using System;
using System.IO;
using System.Linq;
using System.Text;
using AwesomeAssertions;
using OptunaSharp.Artifacts;
using OptunaSharp.Samplers;

namespace OptunaSharp.Tests;

/// <summary>
///     Artifacts (plan 1.4): the filesystem store, id-escape guard, backoff middleware,
///     upload/download round trip through study/trial system attrs, and meta listing.
/// </summary>
[TestClass]
public class ArtifactsTests
{
    private string _dir = null!;

    [TestInitialize]
    public void SetUp()
    {
        _dir = Path.Combine(Path.GetTempPath(), $"optunasharp-artifacts-{Guid.NewGuid():N}");
        Directory.CreateDirectory(_dir);
    }

    [TestCleanup]
    public void TearDown()
    {
        try
        {
            Directory.Delete(_dir, recursive: true);
        }
        catch (IOException)
        {
        }
    }

    [TestMethod]
    public void FileSystemStore_Write_Read_Remove()
    {
        var store = new FileSystemArtifactStore(_dir);
        var payload = Encoding.UTF8.GetBytes("hello artifacts");

        using (var ms = new MemoryStream(payload))
            store.Write("id-1", ms);

        using (var reader = store.OpenReader("id-1"))
        using (var buffer = new MemoryStream())
        {
            reader.CopyTo(buffer);
            buffer.ToArray().Should().Equal(payload);
        }

        store.Remove("id-1");
        var reopen = () => store.OpenReader("id-1");
        reopen.Should().Throw<ArtifactNotFound>().WithMessage("not found");
        var removeAgain = () => store.Remove("id-1");
        removeAgain.Should().Throw<ArtifactNotFound>();
    }

    [TestMethod]
    public void FileSystemStore_Rejects_Escaping_Ids()
    {
        var store = new FileSystemArtifactStore(_dir);
        var absolute = () => store.OpenReader(Path.Combine(Path.GetTempPath(), "abs"));
        absolute.Should().Throw<NumSharp.ValueError>();
        var traversal = () => store.OpenReader("../escape");
        traversal.Should().Throw<NumSharp.ValueError>();
    }

    [TestMethod]
    public void UploadArtifact_Records_Meta_On_Trial_And_Stores_Bytes()
    {
        var store = new FileSystemArtifactStore(_dir);
        var study = Optuna.CreateStudy(sampler: new RandomSampler(1));
        string filePath = Path.Combine(_dir, "model.txt");
        File.WriteAllText(filePath, "weights");

        string? artifactId = null;
        study.Optimize(t =>
        {
            artifactId = ArtifactFunctions.UploadArtifact(store, filePath, t);
            return t.SuggestFloat("x", 0, 1);
        }, nTrials: 1);

        artifactId.Should().NotBeNull();
        var metas = ArtifactFunctions.GetAllArtifactMeta(study.Trials[0], storage: study.Storage);
        metas.Should().ContainSingle();
        metas[0].ArtifactId.Should().Be(artifactId);
        metas[0].Filename.Should().Be("model.txt");
        metas[0].Mimetype.Should().Be("text/plain");
        metas[0].Encoding.Should().BeNull();

        string downloadPath = Path.Combine(_dir, "downloaded.txt");
        ArtifactFunctions.DownloadArtifact(store, downloadPath, artifactId!);
        File.ReadAllText(downloadPath).Should().Be("weights");

        var again = () => ArtifactFunctions.DownloadArtifact(store, downloadPath, artifactId!);
        again.Should().Throw<IOException>().WithMessage($"File already exists: {downloadPath}");
    }

    [TestMethod]
    public void UploadArtifact_On_Study_Uses_Study_SystemAttrs()
    {
        var store = new FileSystemArtifactStore(_dir);
        var study = Optuna.CreateStudy(sampler: new RandomSampler(1));
        string filePath = Path.Combine(_dir, "report.tar.gz");
        File.WriteAllBytes(filePath, new byte[] { 1, 2, 3 });

        ArtifactFunctions.UploadArtifact(store, filePath, study);

        var metas = ArtifactFunctions.GetAllArtifactMeta(study);
        metas.Should().ContainSingle();
        metas[0].Filename.Should().Be("report.tar.gz");
        metas[0].Mimetype.Should().Be("application/x-tar", "encoding suffixes strip like Python's mimetypes");
        metas[0].Encoding.Should().Be("gzip");
    }

    [TestMethod]
    public void Backoff_Passes_Through_And_Preserves_NotFound()
    {
        var store = new Backoff(new FileSystemArtifactStore(_dir), maxRetries: 2, minDelay: 0.001, maxDelay: 0.002);
        using (var ms = new MemoryStream(new byte[] { 42 }))
            store.Write("via-backoff", ms);
        using (var reader = store.OpenReader("via-backoff"))
            reader.ReadByte().Should().Be(42);

        var missing = () => store.OpenReader("missing");
        missing.Should().Throw<ArtifactNotFound>("ArtifactNotFound must not be retried");
        store.Remove("via-backoff");
    }
}

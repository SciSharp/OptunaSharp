using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using AwesomeAssertions;
using Microsoft.Data.Sqlite;
using NumSharp;
using OptunaSharp.Storages;

namespace OptunaSharp.Tests.Interop;

/// <summary>
///     Review item H7, live: <see cref="StorageFunctions.GetStorage(string)" /> must treat a storage string the way
///     Python's <c>optuna.storages.get_storage</c> does. Each string goes through both, and the outcome is compared
///     as <c>kind|detail</c>: what was returned (a cache over an RDB storage) or which error was raised with which
///     message. Before the fix every string threw <see cref="NotSupportedException" /> in C#.
/// </summary>
/// <remarks>
///     <para>The strings are the ones whose outcome does not depend on the machine: SQLite URLs, which open on both
///     sides, and the refusals. PostgreSQL and MySQL URLs are left out because Python's outcome depends on which
///     driver modules are installed and C#'s on a reachable server; the EF Core storage suite covers them.</para>
///     <para>Kinds map as follows. C#'s <see cref="CachedStorage" /> is Python's <c>_CachedStorage</c>. C#'s plain
///     <see cref="ArgumentException" /> is SQLAlchemy's <c>ArgumentError</c>, or its <c>NoSuchModuleError</c>
///     subclass when the message is the plugin error. C#'s <see cref="NotSupportedException" /> is Optuna's
///     <c>ImportError</c>, compared up to the end of Optuna's text because the port adds which package to
///     reference.</para>
/// </remarks>
[TestClass]
public class StorageUrlParityTests : OptunaInteropTestBase
{
    private const string ImportErrorMessage =
        "Failed to import DB access module for the specified storage URL. Please install appropriate one.";

    private readonly List<string> _files = new();

    /// <summary>Deletes every SQLite file the test created, on either side.</summary>
    [TestCleanup]
    public void DeleteFiles()
    {
        // Pooled connections keep files open on Windows after their storage is disposed.
        SqliteConnection.ClearAllPools();
        foreach (string file in _files.SelectMany(f => Directory.GetFiles(Path.GetDirectoryName(f)!, Path.GetFileName(f) + "*")))
        {
            try
            {
                File.Delete(file);
            }
            catch (IOException)
            {
                // A Python handle can still be closing; the temp folder is cleaned by the OS eventually.
            }
        }
    }

    /// <summary>A new temporary SQLite path with forward slashes, recorded for cleanup.</summary>
    /// <param name="tag">A label that keeps the Python and C# files apart.</param>
    /// <returns>The path, ready to follow <c>sqlite:///</c>.</returns>
    private string NewDbPath(string tag)
    {
        string path = Path.Combine(Path.GetTempPath(), $"optunasharp-url-{tag}-{Guid.NewGuid():N}.db");
        _files.Add(path);
        return path.Replace('\\', '/');
    }

    /// <summary>
    ///     The storage strings: <c>{db}</c> stands for a fresh database path, a different one on each side so the
    ///     two never share a file.
    /// </summary>
    private static readonly string[] Cases =
    {
        "sqlite:///{db}",
        "sqlite+pysqlite:///{db}",
        "sqlite:///:memory:",
        "study.db",
        "",
        "C:/runs/study.db",
        "my-db://host/db",
        "foo://user@host/db",
        "foo+bar://host/db",
        "SQLITE:///{db}",
        "postgres://user:pw@127.0.0.1:1/optuna",
        "Redis://localhost:6379",
        "redis://localhost:6379",
        "rediss://localhost:6379",
        "oracle://user:pw@127.0.0.1:1/optuna",
        "mssql+pyodbc://user:pw@127.0.0.1:1/optuna",
    };

    /// <summary>The C# outcome of <see cref="StorageFunctions.GetStorage(string)" /> for one string, in Python's terms.</summary>
    /// <param name="url">The storage string.</param>
    /// <returns><c>kind|detail</c>, comparable with the Python side's line.</returns>
    private static string CSharpOutcome(string url)
    {
        try
        {
            BaseStorage storage = StorageFunctions.GetStorage(url);
            if (storage is not CachedStorage cached)
                return $"OK|{storage.GetType().Name}";
            // The cache does not own the storage, as in Python: dispose it to release the file.
            (cached.Backend as IDisposable)?.Dispose();
            return $"OK|_CachedStorage({cached.Backend.GetType().Name})";
        }
        catch (ValueError e)
        {
            return $"ValueError|{e.Message}";
        }
        catch (NotSupportedException e)
        {
            return e.Message.StartsWith(ImportErrorMessage, StringComparison.Ordinal)
                ? $"ImportError|{ImportErrorMessage}"
                : $"NotSupportedException|{e.Message}";
        }
        catch (ArgumentException e) when (e.GetType() == typeof(ArgumentException))
        {
            return e.Message.StartsWith("Can't load plugin: ", StringComparison.Ordinal)
                ? $"NoSuchModuleError|{e.Message}"
                : $"ArgumentError|{e.Message}";
        }
    }

    /// <summary>
    ///     Every storage string gets the same outcome from <c>optuna.storages.get_storage</c> and from
    ///     <see cref="StorageFunctions.GetStorage(string)" />: the same returned storage shape, or the same error
    ///     kind with the same message.
    /// </summary>
    [TestMethod]
    [TestCategory("RequiresPython")]
    public void GetStorage_TreatsEveryStorageStringLikePython()
    {
        PythonSession.EnsureOrInconclusive();

        string[] pyUrls = Cases.Select(c => c.Replace("{db}", NewDbPath("py"), StringComparison.Ordinal)).ToArray();
        string[] csUrls = Cases.Select(c => c.Replace("{db}", NewDbPath("cs"), StringComparison.Ordinal)).ToArray();
        string[] csOutcomes = csUrls.Select(CSharpOutcome).ToArray();

        // The strings cross as one newline-joined CLR string, so no Python source escaping is involved.
        PySet("_oss_urls", string.Join("\n", pyUrls));
        PyExec(
            "def _oss_outcome(url):\n" +
            "    try:\n" +
            "        s = optuna.storages.get_storage(url)\n" +
            "    except Exception as e:\n" +
            "        return f'{type(e).__name__}|{(str(e).splitlines() or [\"\"])[0]}'\n" +
            "    inner = getattr(s, '_backend', None)\n" +
            "    if inner is None:\n" +
            "        return f'OK|{type(s).__name__}'\n" +
            "    inner.engine.dispose()\n" +
            "    return f'OK|{type(s).__name__}({type(inner).__name__})'\n" +
            "_oss_result = '\\n'.join(_oss_outcome(u) for u in _oss_urls.split('\\n'))\n");
        string[] pyOutcomes = PyStr("_oss_result").Split('\n');

        csOutcomes.Should().Equal(pyOutcomes, "every storage string must get Python's outcome");
        csOutcomes.Take(3).Should().AllBe("OK|_CachedStorage(RDBStorage)", "the SQLite URLs must open on both sides");
    }
}

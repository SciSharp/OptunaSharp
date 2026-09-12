using System;
using System.IO;
using AwesomeAssertions;
using OptunaSharp.Storages;

namespace OptunaSharp.Tests.Interop;

/// <summary>
///     The differential gate for the SQLite schema-migration chain (Phase 4.1's deferred alembic
///     upgrade): OptunaSharp's <see cref="RDBStorage.Upgrade" /> must transform an old database into
///     <b>byte-for-byte the same file</b> that Python Optuna's <c>alembic upgrade head</c> produces.
///     For every committed per-version asset (<c>refs/optuna/.../test_upgrade_assets/{version}.db</c>)
///     each side upgrades its own copy and the two results are compared table-by-table — column
///     layout (<c>PRAGMA table_info</c>), indexes, foreign keys and every row — the only excused
///     divergence being <c>version_info.library_version</c> (OptunaSharp self-identifies there, as it
///     does on a fresh database).
///
///     <para>A second test proves the reverse direction end-to-end: Python opens and reads a database
///     OptunaSharp migrated.</para>
///
///     <para>Self-skips (Inconclusive) when Python/optuna is absent, like every <c>RequiresPython</c>
///     test; <c>Microsoft.Data.Sqlite</c> bundles the engine, so no external service is needed.</para>
/// </summary>
[TestClass]
public class SqliteRdbUpgradeInteropParityTests : OptunaInteropTestBase
{
    private readonly System.Collections.Generic.List<string> _tempFiles = new();

    [TestCleanup]
    public void UpgradeCleanup()
    {
        foreach (var f in _tempFiles)
        {
            try
            {
                if (File.Exists(f))
                    File.Delete(f);
            }
            catch (IOException)
            {
            }
        }
    }

    [TestMethod]
    [TestCategory("RequiresPython")]
    [DataRow("0.9.0.a")]
    [DataRow("1.2.0.a")]
    [DataRow("1.3.0.a")]
    [DataRow("2.4.0.a")]
    [DataRow("2.6.0.a")]
    [DataRow("3.0.0.a")]
    [DataRow("3.0.0.b")]
    [DataRow("3.0.0.c")]
    [DataRow("3.0.0.d")]
    [DataRow("3.2.0.a")]
    public void Upgrade_MatchesLivePython_SchemaAndData(string version)
    {
        PythonSession.EnsureOrInconclusive();

        string asset = Path.Combine(UpgradeAssetsDir(), $"{version}.db");

        // OptunaSharp upgrades its copy.
        string csPath = TempDb(version, "cs");
        File.Copy(asset, csPath);
        using (var storage = new RDBStorage(CsUrl(csPath), skipCompatibilityCheck: true, skipTableCreation: true))
        {
            storage.GetCurrentVersion().Should().Be($"v{version}");
            storage.Upgrade();
            storage.GetCurrentVersion().Should().Be("v3.2.0.a");
        }

        // Python upgrades its own copy of the same asset.
        string pyPath = TempDb(version, "py");
        PyExec(
            "import shutil, warnings\n" +
            $"shutil.copyfile(r'{Fwd(asset)}', r'{Fwd(pyPath)}')\n" +
            "from optuna.storages._rdb.storage import RDBStorage as _RDB\n" +
            $"_st = _RDB('sqlite:///{Fwd(pyPath)}', skip_compatibility_check=True, skip_table_creation=True)\n" +
            "with warnings.catch_warnings():\n" +
            "    warnings.simplefilter('ignore')\n" +
            "    _st.upgrade()\n" +
            "_st.engine.dispose()\n");

        // Compare the two migrated files, in Python (it has both), structurally + every row.
        PyExec(DumpAndDiffSource);
        PyExec($"_oss_diff = _oss_upgrade_diff(r'{Fwd(csPath)}', r'{Fwd(pyPath)}')");
        string diff = PyStr("_oss_diff");
        diff.Should().BeEmpty(
            $"OptunaSharp's upgrade of {version}.db must match Python's alembic upgrade byte-for-byte, but:\n{diff}");
    }

    [TestMethod]
    [TestCategory("RequiresPython")]
    [DataRow("0.9.0.a")]
    [DataRow("2.4.0.a")]
    [DataRow("3.0.0.d")]
    public void Upgrade_PythonReadsOptunaSharpMigratedFile(string version)
    {
        PythonSession.EnsureOrInconclusive();

        string asset = Path.Combine(UpgradeAssetsDir(), $"{version}.db");
        string csPath = TempDb(version, "csread");
        File.Copy(asset, csPath);

        using (var storage = new RDBStorage(CsUrl(csPath), skipCompatibilityCheck: true, skipTableCreation: true))
            storage.Upgrade();

        // OptunaSharp reopens the migrated file with the normal constructor (compat check must pass).
        using var reopened = new RDBStorage(CsUrl(csPath));
        var csState = StudySnapshot.Capture(Optuna.LoadStudy("single", reopened));

        // Python opens the same OptunaSharp-migrated file and reads the same study.
        PyExec($"study = optuna.load_study(study_name='single', storage=r'sqlite:///{Fwd(csPath)}')");
        StudySnapshot.AssertParity(PyStudyState("study"), csState, DatetimeComparison.NullnessOnly);
    }

    // ------------------------------------------------------------------ helpers

    private string TempDb(string version, string tag)
    {
        string path = Path.Combine(Path.GetTempPath(), $"optunasharp-upg-{tag}-{version}-{Guid.NewGuid():N}.db");
        _tempFiles.Add(path);
        return path;
    }

    private static string Fwd(string path) => path.Replace('\\', '/');
    private static string CsUrl(string path) => "sqlite:///" + Fwd(path);

    private static string UpgradeAssetsDir()
    {
        string? dir = AppContext.BaseDirectory;
        while (dir is not null && !File.Exists(Path.Combine(dir, "SciSharp.OptunaSharp.sln")))
            dir = Path.GetDirectoryName(dir);
        if (dir is null)
            throw new DirectoryNotFoundException("Could not locate the repository root (SciSharp.OptunaSharp.sln).");
        return Path.Combine(
            dir, "refs", "optuna", "tests", "storages_tests", "rdb_tests", "test_upgrade_assets");
    }

    // Python: dump a SQLite database's structure (table_info / index_list+info / foreign_key_list) and
    // every row (sorted), then diff two such dumps. version_info.library_version is dropped — OptunaSharp
    // writes its own library string there, the one documented divergence from Python.
    private const string DumpAndDiffSource =
        "def _oss_dump_db(db):\n" +
        "    import sqlite3\n" +
        "    con = sqlite3.connect(db); cur = con.cursor()\n" +
        "    out = []\n" +
        "    cur.execute(\"SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name\")\n" +
        "    for t in [r[0] for r in cur.fetchall()]:\n" +
        "        cur.execute('PRAGMA table_info(\"%s\")' % t)\n" +
        "        out.append('TABLE %s: %s' % (t, [(r[1], r[2], r[3], r[5]) for r in cur.fetchall()]))\n" +
        "        cur.execute('PRAGMA index_list(\"%s\")' % t)\n" +
        "        idxs = []\n" +
        "        for r in cur.fetchall():\n" +
        "            c2 = con.cursor(); c2.execute('PRAGMA index_info(\"%s\")' % r[1])\n" +
        "            idxs.append((r[1], r[2], r[3], [c[2] for c in c2.fetchall()]))\n" +
        "        out.append('  IDX %s: %s' % (t, sorted(idxs)))\n" +
        "        cur.execute('PRAGMA foreign_key_list(\"%s\")' % t)\n" +
        "        out.append('  FK %s: %s' % (t, sorted((r[2], r[3], r[4]) for r in cur.fetchall())))\n" +
        "        cur.execute('SELECT * FROM \"%s\"' % t)\n" +
        "        names = [d[0] for d in cur.description]\n" +
        "        rows = cur.fetchall()\n" +
        "        if t == 'version_info':\n" +
        "            li = names.index('library_version')\n" +
        "            rows = [tuple(v for i, v in enumerate(r) if i != li) for r in rows]\n" +
        "        out.append('  DATA %s (%d): %s' % (t, len(rows), sorted(repr(r) for r in rows)))\n" +
        "    con.close()\n" +
        "    return '\\n'.join(out)\n" +
        "def _oss_upgrade_diff(cs, py):\n" +
        "    import difflib\n" +
        "    dc = _oss_dump_db(cs); dp = _oss_dump_db(py)\n" +
        "    if dc == dp:\n" +
        "        return ''\n" +
        "    return '\\n'.join(difflib.unified_diff(dp.split('\\n'), dc.split('\\n'), 'python', 'csharp', lineterm=''))\n";
}

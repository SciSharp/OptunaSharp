using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using AwesomeAssertions;
using Microsoft.Data.Sqlite;
using OptunaSharp;
using OptunaSharp.Storages;

namespace OptunaSharp.Tests.Storage;

/// <summary>
///     Open parity bugs in the SQLite RDB storage found by the 2026-11-10 parity QA run
///     (<c>docs/qa/PARITY_QA_2026-11-10.md</c>). Each test fails today and passes once fixed, at which point its
///     <c>OpenBugs</c> category should be removed.
/// </summary>
[TestClass]
public class Run2SqliteParityBugTests
{
    private readonly List<string> _tempFiles = new();

    /// <summary>Deletes the SQLite files the tests created; a file still held by the connection pool is left.</summary>
    [TestCleanup]
    public void Cleanup()
    {
        // Release pooled connections first so Windows lets the files go.
        SqliteConnection.ClearAllPools();
        foreach (var f in _tempFiles)
        {
            try
            {
                if (File.Exists(f))
                    File.Delete(f);
            }
            catch (IOException)
            {
                // The pool may still hold the file on Windows; the temp folder is cleaned later.
            }
        }
    }

    /// <summary>
    ///     QA-59: the cached refresh asks SQLite for every unfinished trial below the last finished one by id
    ///     (<c>trial_id IN (...)</c>, one bound variable per trial). Past SQLite's variable limit (32,766 in the
    ///     <c>e_sqlite3</c> build Microsoft.Data.Sqlite ships) the read throws <c>SqliteException: 'too many SQL
    ///     variables'</c> and the study can no longer be read. Optuna catches that <c>OperationalError</c>, logs
    ///     <c>Caught an error from sqlalchemy: … Falling back to a slower alternative.</c> and reads every trial of the
    ///     study instead (<c>RDBStorage._get_trials</c>).
    /// </summary>
    /// <remarks>
    ///     <para>
    ///         Built here the way a long distributed run gets there: 33,000 trials left RUNNING (crashed workers, never
    ///         failed by a heartbeat) below a COMPLETE one. The first read has no watermark yet and succeeds; the second
    ///         sends the 33,000 running ids. Measured: Optuna 5.0.0 reads 33,003 trials both times (its Anaconda SQLite
    ///         3.51.0 allows 250,000 variables, so it does not even need the fallback; a 32,766-variable build takes the
    ///         fallback with the warning); the port's second read throws.
    ///     </para>
    ///     <para>
    ///         The EF Core storage builds the same <c>IN</c> list (PostgreSQL allows 65,535 parameters, MySQL a packet's
    ///         worth) without a fallback either; it is not covered here.
    ///     </para>
    /// </remarks>
    [TestMethod]
    [TestCategory("OpenBugs")]
    public void CachedRefresh_TooManyRunningTrialsBelowTheWatermark_FallsBackLikeOptuna()
    {
        string path = Path.Combine(Path.GetTempPath(), $"optunasharp-rdb-qa59-{Guid.NewGuid():N}.db");
        _tempFiles.Add(path);
        string url = "sqlite:///" + path.Replace('\\', '/');

        var creator = Optuna.CreateStudy(studyName: "many", storage: new RDBStorage(url));
        creator.Tell(creator.Ask(), 1.0);

        // 33,000 RUNNING rows and one COMPLETE row above them, written in one transaction: through the API every trial
        // would be its own synchronous commit, minutes for a test.
        const int running = 33_000;
        using (var connection = new SqliteConnection("Data Source=" + path))
        {
            connection.Open();
            long studyId;
            using (var find = connection.CreateCommand())
            {
                find.CommandText = "SELECT study_id FROM studies WHERE study_name = 'many'";
                studyId = (long)find.ExecuteScalar()!;
            }
            string now = DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss.ffffff", CultureInfo.InvariantCulture);
            using var tx = connection.BeginTransaction();
            using (var insert = connection.CreateCommand())
            {
                insert.Transaction = tx;
                insert.CommandText =
                    "INSERT INTO trials (number, study_id, state, datetime_start, datetime_complete) VALUES ($n, $s, $state, $t, $c)";
                var number = insert.Parameters.Add("$n", SqliteType.Integer);
                insert.Parameters.AddWithValue("$s", studyId);
                var state = insert.Parameters.Add("$state", SqliteType.Text);
                insert.Parameters.AddWithValue("$t", now);
                var complete = insert.Parameters.Add("$c", SqliteType.Text);
                for (int i = 0; i <= running; i++)
                {
                    number.Value = i + 1;
                    bool last = i == running;
                    state.Value = last ? "COMPLETE" : "RUNNING";
                    complete.Value = last ? now : DBNull.Value;
                    insert.ExecuteNonQuery();
                }
            }
            using (var value = connection.CreateCommand())
            {
                value.Transaction = tx;
                value.CommandText =
                    "INSERT INTO trial_values (trial_id, objective, value, value_type) VALUES (last_insert_rowid(), 0, 2.0, 'FINITE')";
                value.ExecuteNonQuery();
            }
            tx.Commit();
        }

        var study = Optuna.LoadStudy("many", new RDBStorage(url));
        study.Trials.Should().HaveCount(running + 2, "the first read has no watermark yet");
        // The second read sends every running id below the COMPLETE trial; Optuna answers it (or falls back).
        study.Trials.Should().HaveCount(running + 2);
    }

    /// <summary>
    ///     QA-71: on SQLite, Optuna reads a study's and a trial's user and system attributes in key order, the port in
    ///     insertion order. Optuna's queries have no <c>ORDER BY</c> (<c>StudyUserAttributeModel.where_study_id</c>, the
    ///     <c>selectinload</c> of a trial's attributes), and SQLite answers <c>WHERE study_id = ?</c> /
    ///     <c>WHERE trial_id IN (...)</c> through the unique <c>(study_id, key)</c> / <c>(trial_id, key)</c> autoindex
    ///     (<c>EXPLAIN QUERY PLAN</c>: <c>SEARCH ... USING INDEX sqlite_autoindex_..._1</c>), so the rows come sorted by
    ///     key. The port's queries order by the attribute id.
    /// </summary>
    /// <remarks>
    ///     Measured with Optuna 5.0.0 / SQLAlchemy 2.0.45 / SQLite 3.51: study user attributes set as <c>zz, b, m</c> read
    ///     back <c>b, m, zz</c> through <c>study.user_attrs</c> and <c>load_study(...).user_attrs</c>; system attributes
    ///     <c>q, d</c> as <c>d, q</c>; trial user attributes <c>k, a</c> as <c>a, k</c> through <c>study.trials</c>,
    ///     <c>get_trial</c> and <c>best_trial</c>. <c>get_all_studies()</c> scans the whole attribute table (no
    ///     <c>WHERE</c>, <c>SCAN study_user_attributes</c>), so its summaries keep insertion order (<c>zz, b, m</c>), as
    ///     the port's do. The in-memory and journal storages keep insertion order in both. Parameters and intermediate
    ///     values already match (Optuna sorts parameters by <c>param_id</c>; intermediate values come by step through
    ///     the same kind of index, as in the port). Only key order differs; no sampler depends on it (they read system
    ///     attributes by key, CMA-ES sorts its keys).
    /// </remarks>
    [TestMethod]
    [TestCategory("OpenBugs")]
    public void AttributeReads_ComeBackInKeyOrderLikeOptunasIndexScan()
    {
        string path = Path.Combine(Path.GetTempPath(), $"optunasharp-rdb-qa71-{Guid.NewGuid():N}.db");
        _tempFiles.Add(path);
        var storage = new RDBStorage("sqlite:///" + path.Replace('\\', '/'));
        var study = Optuna.CreateStudy(studyName: "attrs", storage: storage);
        foreach (var key in new[] { "zz", "b", "m" })
            study.SetUserAttr(key, 1L);
        int studyId = storage.GetStudyIdFromName("attrs");
        storage.SetStudySystemAttr(studyId, "q", 1L);
        storage.SetStudySystemAttr(studyId, "d", 1L);
        var trial = study.Ask();
        trial.SetUserAttr("k", 1L);
        trial.SetUserAttr("a", 2L);
        study.Tell(trial, 1.0);

        study.UserAttrs.Keys.Should().Equal(new[] { "b", "m", "zz" }, "Optuna's study.user_attrs comes through the (study_id, key) index");
        storage.GetStudySystemAttrs(studyId).Keys.Should().Equal("d", "q");
        study.Trials[0].UserAttrs.Keys.Should().Equal(new[] { "a", "k" }, "Optuna's trial reads come through the (trial_id, key) index");
        storage.GetAllStudies()[0].UserAttrs.Keys.Should().Equal(new[] { "zz", "b", "m" }, "get_all_studies scans the table in insertion order");
    }

    /// <summary>
    ///     QA-75: opening a database whose schema predates the head revision without <c>skip_compatibility_check</c>
    ///     raises Optuna's <c>RuntimeError</c> naming the runtime version as <c>version.__version__</c>, i.e.
    ///     <c>5.0.0</c>; the port interpolates <c>OptunaSharpVersion.OptunaCompatibilityTarget</c>, which is
    ///     <c>"5.0"</c>, so the verbatim message differs (both the SQLite and the EF Core storage use it).
    /// </summary>
    /// <remarks>
    ///     Measured on copies of every Optuna upgrade-test asset (<c>refs/optuna/tests/storages_tests/rdb_tests/
    ///     test_upgrade_assets</c>): the nine pre-3.2 databases give the same message on both sides except
    ///     <c>5.0.0</c> / <c>5.0</c>; with <c>skip_compatibility_check=True</c> both open them and report the same
    ///     current / head revisions and studies.
    /// </remarks>
    [TestMethod]
    [TestCategory("OpenBugs")]
    public void OldSchemaDatabase_CompatibilityErrorNamesOptunasFullVersion()
    {
        string asset = Path.Combine(RepoRoot(), "refs", "optuna", "tests", "storages_tests", "rdb_tests",
            "test_upgrade_assets", "3.0.0.a.db");
        if (!File.Exists(asset))
            Assert.Inconclusive("refs/optuna submodule not initialized (`git submodule update --init refs/optuna`).");
        string path = Path.Combine(Path.GetTempPath(), $"optunasharp-rdb-qa75-{Guid.NewGuid():N}.db");
        File.Copy(asset, path);
        _tempFiles.Add(path);

        Action open = () => new RDBStorage("sqlite:///" + path.Replace('\\', '/'));

        open.Should().Throw<NumSharp.RuntimeError>().WithMessage(
            "The runtime optuna version 5.0.0 is no longer compatible with the table schema (set up by optuna 3.0.0). " +
            "Please execute `$ optuna storage upgrade --storage $STORAGE_URL` for upgrading the storage.");
    }

    /// <summary>Returns the repository root, two levels above this source file, wherever the test binaries were built.</summary>
    /// <param name="thisFile">Filled in by the compiler with this file's path.</param>
    /// <returns>The absolute repository root.</returns>
    private static string RepoRoot([System.Runtime.CompilerServices.CallerFilePath] string? thisFile = null)
        => Path.GetFullPath(Path.Combine(Path.GetDirectoryName(thisFile)!, "..", ".."));
}

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
}

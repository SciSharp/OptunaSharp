using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using AwesomeAssertions;
using OptunaSharp;
using OptunaSharp.Collections;
using OptunaSharp.Storages;

namespace OptunaSharp.Tests.Storage;

/// <summary>
///     Open parity bugs in the SQLite RDB storage, found by the 2026-11-02 parity QA run
///     (<c>docs/qa/PARITY_QA_2026-11-02.md</c>, D53: a study with special values written by each language into
///     SQLite and a journal file and read back by both). Each test fails today and passes once fixed, at which point
///     its <c>OpenBugs</c> category should be removed.
/// </summary>
[TestClass]
public class SqliteRdbParityBugTests
{
    private readonly List<string> _tempFiles = new();

    /// <summary>Deletes the SQLite files the tests created; a file still held by the connection pool is left.</summary>
    [TestCleanup]
    public void Cleanup()
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
                // The pool may still hold the file on Windows; the temp folder is cleaned later.
            }
        }
    }

    /// <summary>
    ///     QA-38: user attributes read back from SQLite must come in Optuna's order, which is the key order of the
    ///     <c>(trial_id, key)</c> unique index; the port orders them by row id, i.e. insertion order.
    /// </summary>
    /// <remarks>
    ///     Optuna's <c>get_all_trials</c> loads attributes with <c>selectinload</c>, a query without <c>ORDER BY</c>;
    ///     SQLite answers it through <c>sqlite_autoindex_trial_user_attributes_1 (trial_id, key)</c>, so a trial whose
    ///     attributes were set as <c>err, e, o</c> reads back as <c>{'e': …, 'err': …, 'o': …}</c> in Python and as
    ///     <c>err, e, o</c> in the port (<c>LoadTrialAttrs</c> adds <c>ORDER BY trial_user_attribute_id</c>). The values
    ///     are equal; the dict order shows in user code, JSON output and plot hovertexts, and <c>copy_study</c>
    ///     re-inserts in read order. Parameters are not affected (Optuna sorts them by <c>param_id</c> itself), and a
    ///     journal storage keeps insertion order in both languages.
    /// </remarks>
    [TestMethod]
    [TestCategory("OpenBugs")]
    public void TrialUserAttrs_ReadBackInIndexOrderLikeOptuna()
    {
        string path = Path.Combine(Path.GetTempPath(), $"optunasharp-rdb-qa-{Guid.NewGuid():N}.db");
        _tempFiles.Add(path);
        string url = "sqlite:///" + path.Replace('\\', '/');
        using (var storage = new RDBStorage(url))
        {
            var study = Optuna.CreateStudy(studyName: "s", storage: storage);
            study.AddTrial(FrozenTrial.CreateTrial(value: 1.0,
                userAttrs: new OrderedDict<string, object?> { ["err"] = 1L, ["e"] = 2L, ["o"] = 3L }));
        }

        using var reopened = new RDBStorage(url);
        var trial = Optuna.LoadStudy(studyName: "s", storage: reopened).Trials.Single();

        trial.UserAttrs.Keys.Should().Equal("e", "err", "o");
    }
}

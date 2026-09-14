using System;
using System.IO;
using AwesomeAssertions;
using OptunaSharp.Samplers;
using OptunaSharp.Storages;
using OptunaSharp.Storages.Journal;

namespace OptunaSharp.Tests.Interop;

/// <summary>
///     Live-Optuna cross-language parity for the storage recipes (<c>recipes/rdb.md</c>,
///     <c>recipes/journal-storage.md</c>). OptunaSharp writes a persistent study; real Optuna opens the
///     <b>same file</b> and its view is asserted to match OptunaSharp's, proving the recipes' central
///     claim — "Python Optuna can open the same file". This is the strongest form of storage parity: it
///     exercises the byte-for-byte schema/format compatibility, not merely same-seed reproduction.
/// </summary>
[TestClass]
public class RecipeStorageParityTests : OptunaInteropTestBase
{
    private string _path = null!;

    /// <summary>Allocates a unique temp-file stem for this test (extensions appended per storage kind).</summary>
    [TestInitialize]
    public void PathInit()
        => _path = Path.Combine(Path.GetTempPath(), $"optunasharp-recipe-{Guid.NewGuid():N}").Replace('\\', '/');

    /// <summary>Deletes the temp study file and any sidecar (journal/lock/WAL) files it spawned.</summary>
    [TestCleanup]
    public void PathCleanup()
    {
        var dir = Path.GetDirectoryName(_path)!;
        foreach (var f in Directory.GetFiles(dir, Path.GetFileName(_path) + "*"))
        {
            try
            {
                File.Delete(f);
            }
            catch (IOException)
            {
            }
        }
    }

    /// <summary>The recipes' quadratic objective, deterministic under a seeded sampler.</summary>
    /// <param name="t">The trial being evaluated.</param>
    /// <returns>The squared error from 2.</returns>
    private static double QuadraticObjective(Trial t)
    {
        double x = t.SuggestFloat("x", -10, 10);
        return (x - 2) * (x - 2);
    }

    // ---------------------------------------------------------------- rdb.md

    /// <summary>
    ///     Saving/Resuming with RDB: OptunaSharp creates a SQLite study, resumes it (<c>loadIfExists</c>
    ///     appends 3 more trials to the persisted 3), and real Optuna opening the same <c>.db</c> must see
    ///     the identical 6-trial study — including the study user-attributes surfaced through
    ///     <c>get_all_study_summaries</c>.
    /// </summary>
    /// <param name="seed">RandomSampler seed used while writing the study.</param>
    [TestMethod]
    [TestCategory("RequiresPython")]
    [DataRow(0)]
    [DataRow(42)]
    public void Rdb_SaveResumeAndSummaries_CrossLanguage_Parity(int seed)
    {
        PythonSession.EnsureOrInconclusive();
        string url = $"sqlite:///{_path}.db";

        // Save: create + optimize 3, then dispose to flush the database file.
        using (var rdb = new RDBStorage(url))
        {
            var s = Optuna.CreateStudy(storage: rdb, studyName: "example-study", sampler: new RandomSampler(seed));
            s.SetUserAttr("dataset", "MNIST");
            s.SetUserAttr("contributors", new object?[] { "Akiba", "Sano" });
            s.Optimize(QuadraticObjective, nTrials: 3);
        }

        // Resume: reopen with loadIfExists and append 3 more trials -> 6 total.
        StudySnapshot csState;
        object? csSummaryDataset;
        using (var rdb = new RDBStorage(url))
        {
            var s = Optuna.CreateStudy(
                storage: rdb, studyName: "example-study", loadIfExists: true, sampler: new RandomSampler(seed));
            s.Optimize(QuadraticObjective, nTrials: 3);
            s.Trials.Count.Should().Be(6, "resume must append to the 3 persisted trials");
            csState = StudySnapshot.Capture(s);
            csSummaryDataset = Optuna.GetAllStudySummaries(rdb)[0].UserAttrs["dataset"];
        }

        // Cross-language: Python opens the same file and must see the identical study.
        PyExec($"study = optuna.load_study(study_name='example-study', storage=r'{url}')");
        StudySnapshot.AssertParity(PyStudyState("study"), csState, DatetimeComparison.NullnessOnly);

        // Study user-attributes survive into get_all_study_summaries on both runtimes.
        PyExec($"summ_ds = optuna.get_all_study_summaries(r'{url}')[0].user_attrs['dataset']\n");
        csSummaryDataset.Should().Be(PyStr("summ_ds"));
    }

    // ---------------------------------------------------------------- journal-storage.md

    /// <summary>
    ///     Journal-file storage: OptunaSharp writes a study through <c>JournalFileBackend</c>, and real
    ///     Optuna opening the same line-delimited-JSON log must replay the identical study — the recipe's
    ///     "byte-compatible with Python Optuna's JournalFileBackend" claim.
    /// </summary>
    /// <param name="seed">RandomSampler seed used while writing the journal.</param>
    [TestMethod]
    [TestCategory("RequiresPython")]
    [DataRow(0)]
    [DataRow(42)]
    public void Journal_Persist_CrossLanguage_Parity(int seed)
    {
        PythonSession.EnsureOrInconclusive();
        string logPath = $"{_path}.log";

        var storage = new JournalStorage(new JournalFileBackend(logPath, lockObj: new JournalFileOpenLock(logPath)));
        var s = Optuna.CreateStudy(storage: storage, studyName: "example-study", sampler: new RandomSampler(seed));
        s.SetUserAttr("writer", "csharp");
        s.Optimize(QuadraticObjective, nTrials: 4);
        var csState = StudySnapshot.Capture(s);

        // Python opens the SAME journal file (the format is cross-language) and must see the same study.
        PyExec(
            "study = optuna.load_study(study_name='example-study', storage=" +
            PyJournalStorageExpr(logPath) + ")\n");
        StudySnapshot.AssertParity(PyStudyState("study"), csState, DatetimeComparison.NullnessOnly);
    }
}

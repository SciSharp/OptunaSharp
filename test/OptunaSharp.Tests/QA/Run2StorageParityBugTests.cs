using System.IO;
using AwesomeAssertions;
using OptunaSharp.Samplers;
using OptunaSharp.Storages;
using OptunaSharp.Storages.Journal;

namespace OptunaSharp.Tests.QA;

/// <summary>
///     Storage parity bugs found by the 2026-11-10 parity QA run (<c>docs/qa/PARITY_QA_2026-11-10.md</c>). Expected
///     values are what Optuna 5.0.0 produces on CPython 3.12. A test marked <c>[OpenBugs]</c> fails until its divergence is
///     fixed; the attribute is removed with the fix, and the test then guards against a regression.
/// </summary>
[TestClass]
public class Run2StorageParityBugTests
{
    /// <summary>
    ///     Runs one trial that reads its own stored record with <c>get_trials(deepcopy=False)</c> after its first
    ///     suggestion, then suggests again, sets a user attribute, reports an intermediate value and sets a constraint.
    /// </summary>
    /// <param name="storage">The storage under test, or <see langword="null" /> for the default in-memory storage.</param>
    /// <returns>The record obtained before the later writes, as the caller holds it after the trial finished.</returns>
    /// <remarks>The record is read while the trial is RUNNING, the only time a write can still reach it.</remarks>
    private static FrozenTrial HeldRecordAfterLaterWrites(BaseStorage? storage)
    {
        var study = Optuna.CreateStudy(storage: storage, sampler: new RandomSampler(0));
        FrozenTrial? held = null;
        study.Optimize(t =>
        {
            t.SuggestFloat("x", 0, 1);
            held = study.GetTrials(deepcopy: false)[^1];
            t.SuggestFloat("y", 0, 1);
            t.SetUserAttr("k", 1L);
            t.Report(0.5, 0);
            t.SetConstraint("c", 0.0);
            return 1.0;
        }, nTrials: 1);
        return held!;
    }

    /// <summary>
    ///     QA-58: in Optuna, a <c>FrozenTrial</c> a caller obtained with <c>deepcopy=False</c> while its trial runs keeps
    ///     receiving that trial's later user attributes, intermediate values and system attributes (but not its later
    ///     parameters), on the in-memory and the journal storage. The port's record never changes.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///         Why Python behaves so: <c>Trial</c> keeps the record the storage returned when the trial was created
    ///         (<c>_cached_frozen_trial</c>) and, after each write, mutates that record's dictionary in place
    ///         (<c>user_attrs[key] = value</c>, <c>intermediate_values[step] = value</c>, the constraint's system
    ///         attribute). The storages copy a record shallowly on every write and replace only the dictionary being
    ///         written (<c>set_trial_param</c> copies <c>params</c> and <c>distributions</c>, <c>set_trial_user_attr</c>
    ///         copies <c>user_attrs</c>, ...), so every later record still shares the dictionaries no write has replaced
    ///         yet with the trial's cached record. A record read after the first suggestion therefore shares the cached
    ///         record's <c>user_attrs</c>, <c>intermediate_values</c> and <c>system_attrs</c> objects and sees the
    ///         trial's in-place updates to them; its <c>params</c> were replaced by that suggestion and stay as they were.
    ///         The SQLite storage hands out fresh objects and shows nothing, in both languages.
    ///     </para>
    ///     <para>
    ///         Optuna 5.0.0, measured (in-memory and journal alike): before the writes <c>params=['x'] ui=[] iv=[]
    ///         sys=[]</c>, after them <c>params=['x'] ui=['k'] iv=[0] sys=['constraints:c']</c>. The port: unchanged
    ///         (<c>InMemoryStorage</c> copies all five dictionaries before writing to a handed-out record, plan 13 S4, and
    ///         the port's <c>Trial</c> keeps a detached copy since <c>3a75811</c>). An aliasing artifact of Python's
    ///         copy-on-write rather than a designed contract, so LOW; the owner decides whether to copy it.
    ///     </para>
    /// </remarks>
    [TestMethod]
    [OpenBugs]
    public void HeldRunningRecord_SeesTheTrialsLaterAttributeWrites_LikeOptuna()
    {
        var memory = HeldRecordAfterLaterWrites(null);
        memory.Params.Keys.Should().Equal("x");
        memory.UserAttrs.Keys.Should().Equal("k");
        memory.IntermediateValues.Keys.Should().Equal(0L);
        memory.SystemAttrs.Keys.Should().Equal("constraints:c");

        string file = Path.Combine(Path.GetTempPath(), "optunasharp-qa58-" + System.Guid.NewGuid().ToString("N") + ".log");
        try
        {
            var journal = HeldRecordAfterLaterWrites(new JournalStorage(new JournalFileBackend(file)));
            journal.Params.Keys.Should().Equal("x");
            journal.UserAttrs.Keys.Should().Equal("k");
            journal.IntermediateValues.Keys.Should().Equal(0L);
            journal.SystemAttrs.Keys.Should().Equal("constraints:c");
        }
        finally
        {
            // The journal file and its lock directory are scratch state of this test only.
            if (File.Exists(file))
                File.Delete(file);
        }
    }

    /// <summary>
    ///     QA-66: two storage-level exceptions differ from Optuna's. <c>get_trial_param</c> for a parameter the trial
    ///     does not have raises Python's <c>KeyError</c> (the record's own dictionary lookup) on the in-memory and the
    ///     journal storage; the port raises <see cref="System.Collections.Generic.KeyNotFoundException" />, which a
    ///     <c>catch (KeyError)</c> written against the port's other storage errors does not catch. And the storages raise
    ///     <c>DuplicatedStudyError</c> without a message (<c>raise DuplicatedStudyError</c>), so <c>str(e)</c> is empty;
    ///     the port's carries .NET's default text.
    /// </summary>
    /// <remarks>
    ///     Found by a storage API script (every <c>BaseStorage</c> method with valid and invalid ids, finished trials,
    ///     incompatible distributions) on memory / journal / SQLite in both languages: every other result and message
    ///     matched, storage by storage (<c>No study with study_id 999 exists.</c> in memory, <c>Record does not
    ///     exist.</c> in the journal and SQLite, the journal's bare <c>KeyError: 999</c>, the journal still returning a
    ///     deleted study's trial). Users reach these only by calling the storage directly: <c>create_study</c> replaces
    ///     the empty <c>DuplicatedStudyError</c> with its own message, which the port copies. NIT.
    /// </remarks>
    [TestMethod]
    [OpenBugs]
    public void StorageLevelErrors_MissingParamAndDuplicatedStudy_MatchOptuna()
    {
        var storage = new InMemoryStorage();
        int studyId = storage.CreateNewStudy(new[] { StudyDirection.Minimize }, "a");
        int trialId = storage.CreateNewTrial(studyId);

        System.Action missingParam = () => storage.GetTrialParam(trialId, "missing");
        missingParam.Should().Throw<NumSharp.KeyError>("Optuna raises KeyError('missing')");

        System.Action duplicate = () => storage.CreateNewStudy(new[] { StudyDirection.Minimize }, "a");
        duplicate.Should().Throw<DuplicatedStudyError>().Which.Message.Should().BeEmpty("str(DuplicatedStudyError()) is ''");
    }

    /// <summary>
    ///     Creates studies <c>a</c>, <c>b</c>, <c>c</c>, deletes <c>b</c>, creates <c>d</c>, deletes <c>a</c>, then creates
    ///     <c>e</c> and <c>f</c>, and lists the studies the way <c>get_all_study_names</c> does.
    /// </summary>
    /// <param name="storage">The storage under test.</param>
    /// <returns>The study names in the storage's listing order.</returns>
    /// <remarks>Each deletion frees a slot that a later creation can take, which is where the orders part.</remarks>
    private static System.Collections.Generic.List<string> NamesAfterDeletions(BaseStorage storage)
    {
        foreach (string name in new[] { "a", "b", "c" })
            Optuna.CreateStudy(storage: storage, studyName: name);
        Optuna.DeleteStudy("b", storage);
        Optuna.CreateStudy(storage: storage, studyName: "d");
        Optuna.DeleteStudy("a", storage);
        Optuna.CreateStudy(storage: storage, studyName: "e");
        Optuna.CreateStudy(storage: storage, studyName: "f");
        return Optuna.GetAllStudyNames(storage);
    }

    /// <summary>
    ///     QA-67: after a study is deleted, the in-memory and journal storages list studies in a different order from
    ///     Optuna. Python keeps studies in a <c>dict</c>, which appends every new key after the existing ones; the port
    ///     keeps them in a <see cref="System.Collections.Generic.Dictionary{TKey, TValue}" />, which reuses a removed
    ///     entry's slot for the next insertion, so a study created after a deletion is listed where the deleted one was.
    /// </summary>
    /// <remarks>
    ///     Optuna 5.0.0 (in-memory and journal): <c>['c', 'd', 'e', 'f']</c>; the port: <c>[e, d, c, f]</c>. The SQLite
    ///     storage orders by study id in both. The order reaches <c>get_all_study_names</c>,
    ///     <c>get_all_study_summaries</c>, the CLI's <c>studies</c> command on a journal file, and the study list in
    ///     <c>load_study(study_name=None)</c>'s error message. The containers are <c>InMemoryStorage._studies</c> and
    ///     <c>JournalStorageReplayResult._studies</c>; an insertion-ordered dictionary (the repository's
    ///     <c>OrderedDict</c>) would keep Python's order.
    /// </remarks>
    [TestMethod]
    [OpenBugs]
    public void StudyListing_AfterDeletions_KeepsCreationOrderLikePythonDicts()
    {
        NamesAfterDeletions(new InMemoryStorage()).Should().Equal("c", "d", "e", "f");

        string file = Path.Combine(Path.GetTempPath(), "optunasharp-qa67-" + System.Guid.NewGuid().ToString("N") + ".log");
        try
        {
            NamesAfterDeletions(new JournalStorage(new JournalFileBackend(file))).Should().Equal("c", "d", "e", "f");
        }
        finally
        {
            // The journal file is scratch state of this test only.
            if (File.Exists(file))
                File.Delete(file);
        }
    }
}

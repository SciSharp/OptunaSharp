using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using AwesomeAssertions;
using NumSharp;
using OptunaSharp.Collections;
using OptunaSharp.Distributions;
using OptunaSharp.Internal;
using OptunaSharp.Storages;
using OptunaSharp.Storages.Journal;

namespace OptunaSharp.Tests.Storages;

/// <summary>
///     Plan 13 (docs/plans/13-rustuna-derived-performance.md), item S8, the owner's choice of 2026-11-09: bulk attribute
///     writes under Rustuna's names (<c>SetTrialUserAttrs</c>, <c>SetStudyUserAttrs</c>, ...) and Optuna's
///     (<c>SetUserAttr(IReadOnlyDictionary)</c> overloads). A batch leaves exactly what one single write per entry leaves;
///     the built-in storages write it in one step and check first, so an unknown or finished target gets nothing; a
///     journal batch is ordinary single-key records, which Python's journal reader requires.
/// </summary>
[TestClass]
public class BulkAttributeWritesTests
{
    private static readonly StudyDirection[] Minimize = { StudyDirection.Minimize };

    private readonly List<string> _paths = new();

    /// <summary>Deletes the journal files the tests made; a file still held open is left for the temp cleanup.</summary>
    [TestCleanup]
    public void TearDown()
    {
        foreach (string path in _paths)
        {
            foreach (string file in Directory.GetFiles(Path.GetDirectoryName(path)!, Path.GetFileName(path) + "*"))
            {
                try
                {
                    File.Delete(file);
                }
                catch (IOException)
                {
                }
            }
        }
    }

    /// <summary>Attribute values of every JSON kind, in an order the batch must keep.</summary>
    /// <returns>The attributes.</returns>
    private static OrderedDict<string, object?> SampleAttrs() => new()
    {
        ["seed"] = 7L,
        ["ratio"] = 0.25,
        ["name"] = "run-a",
        ["flag"] = true,
        ["missing"] = null,
        ["list"] = new List<object?> { 1L, "two", 3.5 },
        ["nested"] = new OrderedDict<string, object?> { ["k"] = new List<object?> { false } },
    };

    /// <summary>A fresh storage of the named kind: <c>memory</c>, <c>journal</c> (a file), or <c>default</c> (the base loop).</summary>
    /// <param name="kind">The storage kind.</param>
    /// <returns>The storage.</returns>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="kind" /> is not a known kind.</exception>
    private BaseStorage Open(string kind)
    {
        switch (kind)
        {
            case "memory":
                return new InMemoryStorage();
            case "journal":
            {
                string path = Path.Combine(Path.GetTempPath(), $"optunasharp-bulk-{Guid.NewGuid():N}.log");
                _paths.Add(path);
                return new JournalStorage(new JournalFileBackend(path));
            }
            case "default":
                return new DelegatingStorage(new InMemoryStorage());
            default:
                throw new ArgumentOutOfRangeException(nameof(kind), kind, "Unknown storage kind.");
        }
    }

    // ------------------------------------------------------------------ same result as single writes

    /// <summary>
    ///     For trial user and system attributes and study user and system attributes, a batch over attributes that
    ///     partly overwrite earlier ones leaves the same keys, in the same order, with the same values, as one single
    ///     write per entry.
    /// </summary>
    /// <param name="kind">The storage kind.</param>
    [TestMethod]
    [DataRow("memory")]
    [DataRow("journal")]
    [DataRow("default")]
    public void ABatch_LeavesWhatOneSingleWritePerEntryLeaves(string kind)
    {
        BaseStorage bulk = Open(kind), single = Open(kind);
        int bulkStudy = bulk.CreateNewStudy(Minimize, "s"), singleStudy = single.CreateNewStudy(Minimize, "s");
        int bulkTrial = bulk.CreateNewTrial(bulkStudy), singleTrial = single.CreateNewTrial(singleStudy);
        foreach (BaseStorage storage in new[] { bulk, single })
        {
            int trialId = storage == bulk ? bulkTrial : singleTrial;
            int studyId = storage == bulk ? bulkStudy : singleStudy;
            storage.SetTrialUserAttr(trialId, "name", "earlier");
            storage.SetTrialSystemAttr(trialId, "seed", 1L);
            storage.SetStudyUserAttr(studyId, "flag", false);
            storage.SetStudySystemAttr(studyId, "ratio", 9.0);
        }

        OrderedDict<string, object?> attrs = SampleAttrs();
        bulk.SetTrialUserAttrs(bulkTrial, attrs);
        bulk.SetTrialSystemAttrs(bulkTrial, attrs);
        bulk.SetStudyUserAttrs(bulkStudy, attrs);
        bulk.SetStudySystemAttrs(bulkStudy, attrs);
        foreach (var (key, value) in attrs)
        {
            single.SetTrialUserAttr(singleTrial, key, value);
            single.SetTrialSystemAttr(singleTrial, key, value);
            single.SetStudyUserAttr(singleStudy, key, value);
            single.SetStudySystemAttr(singleStudy, key, value);
        }

        Describe(bulk.GetTrial(bulkTrial).UserAttrs).Should().Be(Describe(single.GetTrial(singleTrial).UserAttrs));
        Describe(bulk.GetTrial(bulkTrial).SystemAttrs).Should().Be(Describe(single.GetTrial(singleTrial).SystemAttrs));
        Describe(bulk.GetStudyUserAttrs(bulkStudy)).Should().Be(Describe(single.GetStudyUserAttrs(singleStudy)));
        Describe(bulk.GetStudySystemAttrs(bulkStudy)).Should().Be(Describe(single.GetStudySystemAttrs(singleStudy)));
        bulk.GetTrial(bulkTrial).UserAttrs.Keys.Should().Equal(
            "name", "seed", "ratio", "flag", "missing", "list", "nested");
    }

    /// <summary>
    ///     The Optuna-name overloads take the same path: <c>SetUserAttr(IReadOnlyDictionary)</c> on a trial and a study,
    ///     and <c>SetTrialUserAttr(int, IReadOnlyDictionary)</c> and its three siblings on a storage.
    /// </summary>
    [TestMethod]
    public void TheOptunaNameOverloads_WriteTheBatch()
    {
        var storage = new InMemoryStorage();
        Study study = Optuna.CreateStudy(storage: storage);
        Trial trial = study.Ask();
        var attrs = new Dictionary<string, object?> { ["a"] = 1L, ["b"] = "two" };

        trial.SetUserAttr(attrs);
        study.SetUserAttr(attrs);
        storage.SetTrialSystemAttr(trial.TrialId, attrs);
        storage.SetStudySystemAttr(study.StudyId, attrs);

        trial.UserAttrs.Should().Contain(attrs);
        study.UserAttrs.Should().Contain(attrs);
        storage.GetTrial(trial.TrialId).UserAttrs.Should().Contain(attrs);
        storage.GetTrial(trial.TrialId).SystemAttrs.Should().Contain(attrs);
        storage.GetStudySystemAttrs(study.StudyId).Should().Contain(attrs);

        storage.SetTrialUserAttr(trial.TrialId, new Dictionary<string, object?> { ["c"] = 3L });
        storage.SetStudyUserAttr(study.StudyId, new Dictionary<string, object?> { ["d"] = 4L });
        storage.GetTrial(trial.TrialId).UserAttrs["c"].Should().Be(3L);
        storage.GetStudyUserAttrs(study.StudyId)["d"].Should().Be(4L);
    }

    /// <summary>
    ///     <see cref="Trial.SetUserAttrs" /> updates the trial's own view at once, and a <see cref="FrozenTrial" /> and a
    ///     <see cref="FixedTrial" /> take the per-entry default; the Optuna-name overload works on every trial kind.
    /// </summary>
    [TestMethod]
    public void EveryTrialKind_TakesABatch()
    {
        Study study = Optuna.CreateStudy();
        Trial trial = study.Ask();
        trial.SetUserAttrs(SampleAttrs());
        Describe(trial.UserAttrs).Should().Be(Describe(SampleAttrs()));
        study.Tell(trial, 1.0);
        Describe(study.Trials[0].UserAttrs).Should().Be(Describe(SampleAttrs()));

        FrozenTrial frozen = FrozenTrial.CreateTrial(state: TrialState.Complete, value: 1.0);
        frozen.SetUserAttr(new Dictionary<string, object?> { ["x"] = 1L });
        frozen.UserAttrs["x"].Should().Be(1L);

        var fixedTrial = new FixedTrial(new OrderedDict<string, object?>());
        fixedTrial.SetUserAttrs(new Dictionary<string, object?> { ["y"] = 2L });
        fixedTrial.UserAttrs["y"].Should().Be(2L);
    }

    // ------------------------------------------------------------------ nothing written on refusal

    /// <summary>
    ///     A finished trial refuses a batch with the single write's exception and message, and gets nothing written; on
    ///     a journal not even a record is appended. An unknown trial and an unknown study refuse with a key error.
    /// </summary>
    /// <param name="kind">The storage kind.</param>
    [TestMethod]
    [DataRow("memory")]
    [DataRow("journal")]
    public void AFinishedOrUnknownTarget_RefusesTheBatch_AndGetsNothingWritten(string kind)
    {
        BaseStorage storage = Open(kind);
        int studyId = storage.CreateNewStudy(Minimize, "refuse");
        int trialId = storage.CreateNewTrial(studyId);
        storage.SetTrialUserAttr(trialId, "kept", 1L);
        storage.SetTrialStateValues(trialId, TrialState.Complete, new[] { 1.0 });
        long linesBefore = JournalLines(storage);

        Action single = () => storage.SetTrialUserAttr(trialId, "a", 1L);
        string message = single.Should().Throw<UpdateFinishedTrialError>().Which.Message;
        long linesAfterSingle = JournalLines(storage);

        Action batch = () => storage.SetTrialUserAttrs(trialId, SampleAttrs());
        Action systemBatch = () => storage.SetTrialSystemAttrs(trialId, SampleAttrs());
        batch.Should().Throw<UpdateFinishedTrialError>().WithMessage(message);
        systemBatch.Should().Throw<UpdateFinishedTrialError>().WithMessage(message);
        storage.GetTrial(trialId).UserAttrs.Keys.Should().Equal("kept");
        JournalLines(storage).Should().Be(linesAfterSingle, "a refused batch appends no record");
        _ = linesBefore;

        Action unknownTrial = () => storage.SetTrialUserAttrs(424242, SampleAttrs());
        Action unknownStudy = () => storage.SetStudyUserAttrs(424242, SampleAttrs());
        unknownTrial.Should().Throw<KeyError>();
        unknownStudy.Should().Throw<KeyError>();
        JournalLines(storage).Should().Be(linesAfterSingle);
    }

    /// <summary>An empty batch writes and checks nothing; a null one is refused before anything happens.</summary>
    /// <param name="kind">The storage kind.</param>
    [TestMethod]
    [DataRow("memory")]
    [DataRow("journal")]
    [DataRow("default")]
    public void AnEmptyBatch_DoesNothing_AndANullOneIsRefused(string kind)
    {
        BaseStorage storage = Open(kind);
        int studyId = storage.CreateNewStudy(Minimize, "empty");
        int trialId = storage.CreateNewTrial(studyId);
        long lines = JournalLines(storage);

        storage.SetTrialUserAttrs(trialId, new Dictionary<string, object?>());
        storage.SetStudyUserAttrs(studyId, new Dictionary<string, object?>());
        storage.SetTrialUserAttrs(424242, new Dictionary<string, object?>());  // nothing to write, nothing checked
        JournalLines(storage).Should().Be(lines);

        Action nullTrial = () => storage.SetTrialUserAttrs(trialId, null!);
        Action nullStudy = () => storage.SetStudySystemAttrs(studyId, null!);
        nullTrial.Should().Throw<ArgumentNullException>();
        nullStudy.Should().Throw<ArgumentNullException>();
    }

    // ------------------------------------------------------------------ one step

    /// <summary>
    ///     In memory, a batch of ten attributes on a trial a reader holds copies the record once, then writes every entry
    ///     into the copy; the reader's record keeps its contents.
    /// </summary>
    [TestMethod]
    public void InMemory_ABatchCopiesAHeldRecordOnce()
    {
        var storage = new InMemoryStorage();
        int studyId = storage.CreateNewStudy(Minimize, "copy");
        int trialId = storage.CreateNewTrial(studyId);
        FrozenTrial held = storage.GetTrial(trialId);
        var attrs = Enumerable.Range(0, 10).ToDictionary(i => $"k{i}", i => (object?)(long)i);

        storage.SetTrialUserAttrs(trialId, attrs);

        storage.WriteCopyCount.Should().Be(1);
        held.UserAttrs.Should().BeEmpty();
        storage.GetTrial(trialId).UserAttrs.Should().HaveCount(10);
    }

    /// <summary>
    ///     On a journal, a batch is one record per entry, each the single-key record a single write appends (Python's
    ///     reader asserts one key per record), all appended in one call; another storage replaying the file sees the same
    ///     attributes.
    /// </summary>
    [TestMethod]
    public void Journal_ABatchIsOrdinarySingleKeyRecords_AppendedTogether()
    {
        var storage = (JournalStorage)Open("journal");
        string path = _paths[^1];
        int studyId = storage.CreateNewStudy(Minimize, "records");
        int trialId = storage.CreateNewTrial(studyId);
        string[] before = File.ReadAllLines(path);

        storage.SetTrialUserAttrs(trialId, SampleAttrs());

        string[] added = File.ReadAllLines(path).Skip(before.Length).ToArray();
        added.Should().HaveCount(SampleAttrs().Count);
        var keys = new List<string>();
        foreach (string line in added)
        {
            var record = (OrderedDict<string, object?>)PyJson.Loads(line)!;
            Convert.ToInt32(record["op_code"]).Should().Be((int)JournalOperation.SetTrialUserAttr);
            Convert.ToInt32(record["trial_id"]).Should().Be(trialId);
            var userAttr = (OrderedDict<string, object?>)record["user_attr"]!;
            userAttr.Should().HaveCount(1, "Python's replay asserts len(log['user_attr']) == 1");
            keys.Add(userAttr.Keys.Single());
        }
        keys.Should().Equal(SampleAttrs().Keys);

        var replay = new JournalStorage(new JournalFileBackend(path));
        Describe(replay.GetTrial(trialId).UserAttrs).Should().Be(Describe(SampleAttrs()));
    }

    /// <summary>
    ///     If another process finishes the trial between the batch's check and its append, the replay rejects every record
    ///     of the batch, one per synchronization. The batch write consumes them all and raises once; later, unrelated
    ///     calls of the storage do not raise for the batch's leftovers.
    /// </summary>
    [TestMethod]
    public void Journal_ABatchRejectedAfterItsCheck_IsDrained_SoLaterCallsDoNotRaise()
    {
        var writer = (JournalStorage)Open("journal");
        string path = _paths[^1];
        int studyId = writer.CreateNewStudy(Minimize, "drain");
        int trialId = writer.CreateNewTrial(studyId);
        int otherTrial = writer.CreateNewTrial(studyId);

        // Another worker finishes the trial; the writer has not synchronized since.
        var otherProcess = new JournalStorage(new JournalFileBackend(path));
        otherProcess.SetTrialStateValues(trialId, TrialState.Fail);

        // The batch as WriteAttributeBatch builds it after a check that passed on the writer's stale state.
        string workerId = writer.WorkerIdForTests;
        var logs = SampleAttrs().Select(kv => new OrderedDict<string, object?>
        {
            ["op_code"] = (int)JournalOperation.SetTrialUserAttr,
            ["worker_id"] = workerId,
            ["trial_id"] = trialId,
            ["user_attr"] = new OrderedDict<string, object?> { [kv.Key] = kv.Value },
        }).ToList();

        Action batch = () => writer.AppendBatchAndSyncForTests(logs);
        batch.Should().Throw<UpdateFinishedTrialError>();

        // Every leftover was consumed: these would raise for the batch's remaining records otherwise.
        Action read = () => writer.GetTrial(otherTrial);
        read.Should().NotThrow();
        writer.SetTrialUserAttrs(otherTrial, new Dictionary<string, object?> { ["fine"] = 1L });
        writer.GetTrial(otherTrial).UserAttrs["fine"].Should().Be(1L);
        writer.GetTrial(trialId).UserAttrs.Should().BeEmpty();
    }

    /// <summary>
    ///     The base implementation, which storages without an override (the gRPC client, third-party storages) use, makes
    ///     exactly one single-entry call per entry, in the dictionary's order.
    /// </summary>
    [TestMethod]
    public void TheBaseImplementation_CallsTheSingleWriteOncePerEntry_InOrder()
    {
        var storage = new DelegatingStorage(new InMemoryStorage());
        int studyId = storage.CreateNewStudy(Minimize, "default");
        int trialId = storage.CreateNewTrial(studyId);

        storage.SetTrialUserAttrs(trialId, SampleAttrs());
        storage.SetStudySystemAttrs(studyId, new Dictionary<string, object?> { ["p"] = 1L, ["q"] = 2L });

        storage.Calls.Should().Equal(
            SampleAttrs().Keys.Select(k => $"trial-user:{k}").Concat(new[] { "study-system:p", "study-system:q" }));
    }

    // ------------------------------------------------------------------ helpers

    /// <summary>Describes attributes as JSON in their order, so dictionaries of nested values compare by content.</summary>
    /// <param name="attrs">The attributes.</param>
    /// <returns>The description.</returns>
    private static string Describe(IReadOnlyDictionary<string, object?> attrs)
        => string.Join("; ", attrs.Select(kv => $"{kv.Key}={PyJson.Dumps(kv.Value)}"));

    /// <summary>The number of lines of a journal storage's file, or -1 for other storages.</summary>
    /// <param name="storage">The storage.</param>
    /// <returns>The line count.</returns>
    private long JournalLines(BaseStorage storage)
        => storage is JournalStorage ? File.ReadAllLines(_paths[^1]).LongLength : -1;

    /// <summary>
    ///     A storage that forwards every member to an in-memory storage and overrides none of the bulk writes, so they run
    ///     <see cref="BaseStorage" />'s per-entry default; it records the single attribute writes it receives.
    /// </summary>
    private sealed class DelegatingStorage : BaseStorage
    {
        private readonly InMemoryStorage _inner;

        /// <summary>Wraps <paramref name="inner" />.</summary>
        /// <param name="inner">The storage every call goes to.</param>
        internal DelegatingStorage(InMemoryStorage inner) => _inner = inner;

        /// <summary>The single attribute writes received, as <c>kind:key</c>.</summary>
        internal List<string> Calls { get; } = new();

        /// <inheritdoc />
        public override int CreateNewStudy(IReadOnlyList<StudyDirection> directions, string? studyName = null)
            => _inner.CreateNewStudy(directions, studyName);

        /// <inheritdoc />
        public override void DeleteStudy(int studyId) => _inner.DeleteStudy(studyId);

        /// <inheritdoc />
        public override void SetStudyUserAttr(int studyId, string key, object? value)
        {
            Calls.Add($"study-user:{key}");
            _inner.SetStudyUserAttr(studyId, key, value);
        }

        /// <inheritdoc />
        public override void SetStudySystemAttr(int studyId, string key, object? value)
        {
            Calls.Add($"study-system:{key}");
            _inner.SetStudySystemAttr(studyId, key, value);
        }

        /// <inheritdoc />
        public override int GetStudyIdFromName(string studyName) => _inner.GetStudyIdFromName(studyName);

        /// <inheritdoc />
        public override string GetStudyNameFromId(int studyId) => _inner.GetStudyNameFromId(studyId);

        /// <inheritdoc />
        public override List<StudyDirection> GetStudyDirections(int studyId) => _inner.GetStudyDirections(studyId);

        /// <inheritdoc />
        public override OrderedDict<string, object?> GetStudyUserAttrs(int studyId) => _inner.GetStudyUserAttrs(studyId);

        /// <inheritdoc />
        public override OrderedDict<string, object?> GetStudySystemAttrs(int studyId) => _inner.GetStudySystemAttrs(studyId);

        /// <inheritdoc />
        public override List<FrozenStudy> GetAllStudies() => _inner.GetAllStudies();

        /// <inheritdoc />
        public override int CreateNewTrial(int studyId, FrozenTrial? templateTrial = null)
            => _inner.CreateNewTrial(studyId, templateTrial);

        /// <inheritdoc />
        public override void SetTrialParam(int trialId, string paramName, double paramValueInternal, BaseDistribution distribution)
            => _inner.SetTrialParam(trialId, paramName, paramValueInternal, distribution);

        /// <inheritdoc />
        public override bool SetTrialStateValues(int trialId, TrialState state, IReadOnlyList<double>? values = null)
            => _inner.SetTrialStateValues(trialId, state, values);

        /// <inheritdoc />
        public override void SetTrialIntermediateValue(int trialId, long step, double intermediateValue)
            => _inner.SetTrialIntermediateValue(trialId, step, intermediateValue);

        /// <inheritdoc />
        public override void SetTrialUserAttr(int trialId, string key, object? value)
        {
            Calls.Add($"trial-user:{key}");
            _inner.SetTrialUserAttr(trialId, key, value);
        }

        /// <inheritdoc />
        public override void SetTrialSystemAttr(int trialId, string key, object? value)
        {
            Calls.Add($"trial-system:{key}");
            _inner.SetTrialSystemAttr(trialId, key, value);
        }

        /// <inheritdoc />
        public override FrozenTrial GetTrial(int trialId) => _inner.GetTrial(trialId);

        /// <inheritdoc />
        public override List<FrozenTrial> GetAllTrials(int studyId, bool deepcopy = true, IReadOnlyCollection<TrialState>? states = null)
            => _inner.GetAllTrials(studyId, deepcopy, states);
    }
}

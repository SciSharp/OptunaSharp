using System;
using System.Collections.Generic;
using System.Linq;
using AwesomeAssertions;
using NumSharp;
using OptunaSharp.Collections;
using OptunaSharp.Distributions;
using OptunaSharp.Samplers;
using OptunaSharp.Storages;

namespace OptunaSharp.Tests.Storages;

/// <summary>
///     Plan 13 (docs/plans/13-rustuna-derived-performance.md), phase 2. Item S4: <see cref="InMemoryStorage" /> writes a
///     running trial's record in place while no reader holds it, and copies it first otherwise, so a record any read
///     returned never changes under its reader (Optuna's copy-on-write contract). Item S7: bitwise-identical float and
///     integer distributions and the parameter names are shared across a study's trials, never a categorical, a -0.0
///     bound or a distribution of another runtime type.
/// </summary>
[TestClass]
public class InMemoryStorageInPlaceWritesTests
{
    private static readonly StudyDirection[] Minimize = { StudyDirection.Minimize };

    // ------------------------------------------------------------------ S4: snapshots never change

    /// <summary>
    ///     Every way a caller can receive a stored record (or one of its dictionaries) is read, and then each kind of write
    ///     lands twice in a row: a parameter, an intermediate value, a user attribute, a system attribute. What the reader
    ///     holds stays exactly as it was read through every write, while a fresh read sees every write.
    /// </summary>
    /// <param name="readPath">The read under test; see <see cref="Snapshot" /> for the list.</param>
    [TestMethod]
    [DataRow("get-trial")]
    [DataRow("all-trials")]
    [DataRow("all-trials-running")]
    [DataRow("all-trials-waiting")]
    [DataRow("view-indexer")]
    [DataRow("view-enumerator")]
    [DataRow("trial-params")]
    [DataRow("trial-user-attrs")]
    [DataRow("trial-system-attrs")]
    [DataRow("best-trial-of-another")]
    public void EveryReadPath_KeepsItsSnapshot_ThroughEveryKindOfWrite(string readPath)
    {
        var storage = new InMemoryStorage();
        int studyId = storage.CreateNewStudy(Minimize, "snapshots");
        // A completed trial first, so the best-trial path has something to return and the view is longer than one.
        int done = storage.CreateNewTrial(studyId);
        storage.SetTrialParam(done, "x", 0.5, new FloatDistribution(0, 1));
        storage.SetTrialStateValues(done, TrialState.Complete, new[] { 1.0 });

        int trialId = readPath == "all-trials-waiting"
            ? storage.CreateNewTrial(studyId, FrozenTrial.CreateTrial(state: TrialState.Waiting))
            : storage.CreateNewTrial(studyId);
        storage.SetTrialParam(trialId, "x", 0.25, new FloatDistribution(0, 1));
        storage.SetTrialUserAttr(trialId, "seed", 1L);

        var held = new List<(Func<string> Fingerprint, string Expected)>();
        var writes = new (string Name, Action<int> Write)[]
        {
            ("param", k => storage.SetTrialParam(trialId, $"p{k}", 0.125 * k, new FloatDistribution(0, 10))),
            ("intermediate", k => storage.SetTrialIntermediateValue(trialId, k, k * 1.5)),
            ("user attr", k => storage.SetTrialUserAttr(trialId, $"u{k}", k)),
            ("system attr", k => storage.SetTrialSystemAttr(trialId, $"s{k}", $"v{k}")),
        };

        int step = 0;
        foreach (var (name, write) in writes)
        {
            Func<string> fingerprint = Snapshot(storage, studyId, trialId, done, readPath);
            held.Add((fingerprint, fingerprint()));
            // Two in a row: the first copies (the reader holds the record), the second goes in place on the copy.
            write(step++);
            write(step++);

            foreach (var (heldFingerprint, expected) in held)
                heldFingerprint().Should().Be(expected, $"a {readPath} snapshot must not see the {name} writes after it");
        }

        // Every write landed on the stored record.
        FrozenTrial current = storage.GetTrial(trialId);
        current.Params.Keys.Should().Contain(new[] { "x", "p0", "p1" });
        current.IntermediateValues.Keys.Should().Equal(2L, 3L);
        current.UserAttrs.Keys.Should().Equal("seed", "u4", "u5");
        current.SystemAttrs.Keys.Should().Contain(new[] { "s6", "s7" });
    }

    /// <summary>
    ///     Reads a stored record, or one of its dictionaries, through <paramref name="readPath" /> and returns a function
    ///     that fingerprints what the reader holds, so a later call shows whether it changed.
    /// </summary>
    /// <param name="storage">The storage under test.</param>
    /// <param name="studyId">The study holding both trials.</param>
    /// <param name="trialId">The running (or waiting) trial being written.</param>
    /// <param name="doneId">A completed trial of the same study.</param>
    /// <param name="readPath">
    ///     <c>get-trial</c>, <c>all-trials</c> (no state filter), <c>all-trials-running</c>, <c>all-trials-waiting</c>,
    ///     <c>view-indexer</c>, <c>view-enumerator</c>, <c>trial-params</c>, <c>trial-user-attrs</c>,
    ///     <c>trial-system-attrs</c>, or <c>best-trial-of-another</c> (the completed trial, held while the other is
    ///     written, together with the running record read through <c>GetTrial</c>).
    /// </param>
    /// <returns>A function returning the fingerprint of what was read.</returns>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="readPath" /> is not one of the listed reads.</exception>
    private static Func<string> Snapshot(InMemoryStorage storage, int studyId, int trialId, int doneId, string readPath)
    {
        switch (readPath)
        {
            case "get-trial":
                return FingerprintOf(storage.GetTrial(trialId));
            case "all-trials":
                return FingerprintOf(storage.GetAllTrials(studyId, deepcopy: false).Single(t => t.TrialId == trialId));
            case "all-trials-running":
                return FingerprintOf(storage.GetAllTrials(studyId, deepcopy: false, states: new[] { TrialState.Running }).Single());
            case "all-trials-waiting":
                return FingerprintOf(storage.GetAllTrials(studyId, deepcopy: false, states: new[] { TrialState.Waiting }).Single());
            case "view-indexer":
                return FingerprintOf(storage.GetTrialsPrefixView(studyId)[1]);
            case "view-enumerator":
                return FingerprintOf(storage.GetTrialsPrefixView(studyId).Single(t => t.TrialId == trialId));
            case "trial-params":
                return FingerprintOf(storage.GetTrialParams(trialId));
            case "trial-user-attrs":
                return FingerprintOf(storage.GetTrialUserAttrs(trialId));
            case "trial-system-attrs":
                return FingerprintOf(storage.GetTrialSystemAttrs(trialId));
            case "best-trial-of-another":
            {
                Func<string> best = FingerprintOf(storage.GetBestTrial(studyId));
                storage.GetBestTrial(studyId).TrialId.Should().Be(doneId);
                Func<string> running = FingerprintOf(storage.GetTrial(trialId));
                return () => best() + "|" + running();
            }
            default:
                throw new ArgumentOutOfRangeException(nameof(readPath), readPath, "Unknown read path.");
        }
    }

    /// <summary>Fingerprints a record by its Python repr, which prints every dictionary, its state and values.</summary>
    /// <param name="trial">The record a reader holds.</param>
    /// <returns>A function returning the record's repr at the time of the call.</returns>
    private static Func<string> FingerprintOf(FrozenTrial trial) => () => trial.ToString();

    /// <summary>Fingerprints a dictionary a reader holds, its keys and values in order.</summary>
    /// <param name="dictionary">The dictionary a read returned.</param>
    /// <returns>A function returning the dictionary's entries at the time of the call.</returns>
    private static Func<string> FingerprintOf(IReadOnlyDictionary<string, object?> dictionary)
        => () => string.Join(";", dictionary.Select(kv => $"{kv.Key}={kv.Value}"));

    /// <summary>
    ///     A deep copy taken by the caller holds nothing of the storage's, so the stored record stays writable in place;
    ///     the copy itself never sees a later write.
    /// </summary>
    [TestMethod]
    public void DeepCopiedReads_LeaveTheStoredRecordWritableInPlace_AndNeverChange()
    {
        var storage = new InMemoryStorage();
        int studyId = storage.CreateNewStudy(Minimize, "deep");
        int trialId = storage.CreateNewTrial(studyId);
        storage.SetTrialParam(trialId, "x", 0.25, new FloatDistribution(0, 1));

        FrozenTrial copy = storage.GetAllTrials(studyId, deepcopy: true).Single();
        string before = copy.ToString();
        storage.SetTrialParam(trialId, "y", 0.5, new FloatDistribution(0, 1));
        storage.SetTrialUserAttr(trialId, "k", 1L);

        copy.ToString().Should().Be(before);
        storage.WriteCopyCount.Should().Be(0, "a deep copy holds none of the stored record's dictionaries");
    }

    // ------------------------------------------------------------------ S4: in place when nobody holds the record

    /// <summary>
    ///     A running trial that nobody read is written in place however many writes land, and a read that does not hand out
    ///     the stored record (a deep copy, the internal-repr parameter, a state filter that skips the trial, making a view)
    ///     keeps it that way. Each read that hands the record out makes exactly the next write copy it, once; that includes
    ///     a count, which Optuna's base storage takes through <c>GetAllTrials(deepcopy: false)</c>.
    /// </summary>
    [TestMethod]
    public void WritesNobodyReadsBetween_GoInPlace_AndEachHandOutCostsOneCopy()
    {
        var storage = new InMemoryStorage();
        int studyId = storage.CreateNewStudy(Minimize, "in-place");
        int trialId = storage.CreateNewTrial(studyId);

        // A trial's suggestions and attributes, nobody reading between: Optuna copied the record for each of these.
        for (int i = 0; i < 20; i++)
            storage.SetTrialParam(trialId, $"x{i}", 0.5, new FloatDistribution(0, 1));
        storage.SetTrialUserAttr(trialId, "u", 1L);
        storage.SetTrialSystemAttr(trialId, "s", 1L);
        storage.SetTrialIntermediateValue(trialId, 0, 0.5);
        storage.WriteCopyCount.Should().Be(0);

        // Reads that do not hand the stored record out.
        _ = storage.GetAllTrials(studyId, deepcopy: true);
        _ = storage.GetTrialParam(trialId, "x0");
        _ = storage.GetAllTrials(studyId, deepcopy: false, states: new[] { TrialState.Complete });
        _ = storage.GetTrialsPrefixView(studyId);  // a view hands a record out when it is read, not when it is made
        WriteThrice(storage, trialId, "a");
        storage.WriteCopyCount.Should().Be(0);

        // Each read that hands it out: exactly one copy on the next write, then in place again.
        int expected = 0;
        foreach (Action read in new Action[]
                 {
                     () => storage.GetTrial(trialId),
                     () => storage.GetAllTrials(studyId, deepcopy: false),
                     () => storage.GetAllTrials(studyId, deepcopy: false, states: new[] { TrialState.Running }),
                     () => _ = storage.GetTrialsPrefixView(studyId)[0],
                     () => _ = storage.GetTrialsPrefixView(studyId).ToList(),
                     () => storage.GetTrialParams(trialId),
                     () => storage.GetTrialUserAttrs(trialId),
                     () => storage.GetTrialSystemAttrs(trialId),
                     () => storage.GetNTrials(studyId),
                 })
        {
            read();
            WriteThrice(storage, trialId, $"r{expected}");
            storage.WriteCopyCount.Should().Be(++expected);
        }
    }

    /// <summary>Writes a parameter, a user attribute and a system attribute named after <paramref name="tag" />.</summary>
    /// <param name="storage">The storage under test.</param>
    /// <param name="trialId">The running trial to write.</param>
    /// <param name="tag">Makes the names unique per call.</param>
    private static void WriteThrice(InMemoryStorage storage, int trialId, string tag)
    {
        storage.SetTrialParam(trialId, $"p_{tag}", 0.5, new FloatDistribution(0, 1));
        storage.SetTrialUserAttr(trialId, $"u_{tag}", 1L);
        storage.SetTrialSystemAttr(trialId, $"s_{tag}", 1L);
    }

    /// <summary>
    ///     A state change publishes a new record that shares the old one's dictionaries, so its first write copies; a trial
    ///     created from a template owns deep copies, so its writes go in place and never reach the template.
    /// </summary>
    [TestMethod]
    public void AStateChange_MakesTheNextWriteCopy_AndATemplatedTrialNeverWritesItsTemplate()
    {
        var storage = new InMemoryStorage();
        int studyId = storage.CreateNewStudy(Minimize, "states");
        var template = FrozenTrial.CreateTrial(state: TrialState.Waiting);
        template.UserAttrsDict["from"] = "template";
        string templateBefore = template.ToString();

        int trialId = storage.CreateNewTrial(studyId, template);
        storage.SetTrialUserAttr(trialId, "k", 1L);
        storage.SetTrialParam(trialId, "x", 0.5, new FloatDistribution(0, 1));
        storage.WriteCopyCount.Should().Be(0);
        template.ToString().Should().Be(templateBefore, "the stored record owns copies of the template's dictionaries");

        FrozenTrial waiting = storage.GetAllTrials(studyId, deepcopy: false, states: new[] { TrialState.Waiting }).Single();
        storage.SetTrialStateValues(trialId, TrialState.Running).Should().BeTrue();
        storage.SetTrialUserAttr(trialId, "after", 2L);
        storage.SetTrialUserAttr(trialId, "after2", 3L);
        storage.WriteCopyCount.Should().Be(1, "the running record shares the waiting record's dictionaries, once");
        waiting.UserAttrs.Keys.Should().Equal("from", "k");
        waiting.State.Should().Be(TrialState.Waiting);
    }

    /// <summary>
    ///     The view keeps its documented liveness: each read returns the trial's current record, whether the write after
    ///     the previous read went in place or through a copy; a finished record is returned as stored, without a copy.
    /// </summary>
    [TestMethod]
    public void TheView_ReturnsTheCurrentRecordOnEveryRead_AndFinishedRecordsAsStored()
    {
        var storage = new InMemoryStorage();
        int studyId = storage.CreateNewStudy(Minimize, "view");
        int trialId = storage.CreateNewTrial(studyId);
        IReadOnlyList<FrozenTrial> view = storage.GetTrialsPrefixView(studyId);

        storage.SetTrialParam(trialId, "x", 0.25, new FloatDistribution(0, 1));
        view[0].Params.Keys.Should().Equal("x");  // written in place before this first read
        storage.SetTrialParam(trialId, "y", 0.25, new FloatDistribution(0, 1));
        view[0].Params.Keys.Should().Equal("x", "y");  // written through a copy after the read
        view.Single().Params.Keys.Should().Equal("x", "y");

        storage.SetTrialStateValues(trialId, TrialState.Complete, new[] { 2.0 });
        FrozenTrial finished = view[0];
        finished.State.Should().Be(TrialState.Complete);
        view[0].Should().BeSameAs(finished, "a finished record is never written again, so it is returned as stored");
        storage.GetTrial(trialId).Should().BeSameAs(finished);
    }

    /// <summary>A finished trial still refuses every write, so its record is never changed in place.</summary>
    [TestMethod]
    public void AFinishedTrial_RefusesWrites_SoItsRecordNeverChanges()
    {
        var storage = new InMemoryStorage();
        int studyId = storage.CreateNewStudy(Minimize, "finished");
        int trialId = storage.CreateNewTrial(studyId);
        storage.SetTrialStateValues(trialId, TrialState.Complete, new[] { 1.0 });
        FrozenTrial record = storage.GetTrialsPrefixView(studyId)[0];
        string before = record.ToString();

        Action param = () => storage.SetTrialParam(trialId, "x", 0.5, new FloatDistribution(0, 1));
        Action attr = () => storage.SetTrialUserAttr(trialId, "k", 1L);
        Action report = () => storage.SetTrialIntermediateValue(trialId, 0, 1.0);
        param.Should().Throw<UpdateFinishedTrialError>();
        attr.Should().Throw<UpdateFinishedTrialError>();
        report.Should().Throw<UpdateFinishedTrialError>();
        record.ToString().Should().Be(before);
        storage.WriteCopyCount.Should().Be(0);
    }

    /// <summary>
    ///     A study driven through ask/suggest/tell costs about one copy per trial instead of one per write: the trial's
    ///     own read when it starts is the only hand-out between its suggestions.
    /// </summary>
    [TestMethod]
    public void AnAskTellLoop_CopiesAboutOncePerTrial_NotOncePerWrite()
    {
        var storage = new InMemoryStorage();
        Study study = Optuna.CreateStudy(storage: storage, sampler: new RandomSampler(3));
        const int trials = 25, parameters = 12;
        for (int n = 0; n < trials; n++)
        {
            Trial trial = study.Ask();
            double sum = 0;
            for (int i = 0; i < parameters; i++)
                sum += trial.SuggestFloat($"x{i}", -1, 1);
            trial.SetUserAttr("sum", sum);
            study.Tell(trial, sum * sum);
        }

        // Optuna copies per write: 25 x 13. One copy per trial is the trial's own read at Ask.
        storage.WriteCopyCount.Should().BeLessThanOrEqualTo(trials);
        study.Trials.Should().HaveCount(trials);
        study.Trials.Should().OnlyContain(t => t.Params.Count == parameters && t.UserAttrs.ContainsKey("sum"));
    }

    // ------------------------------------------------------------------ S7: shared distributions and names

    /// <summary>
    ///     Float and integer distributions identical to the bit are shared across the study's trials, so the second
    ///     trial's record holds the first trial's instance; a compatible but different one keeps the caller's instance.
    /// </summary>
    [TestMethod]
    public void BitwiseIdenticalFloatAndIntDistributions_AreShared_AndOthersKeepTheCallersInstance()
    {
        var storage = new InMemoryStorage();
        int studyId = storage.CreateNewStudy(Minimize, "shared");
        int first = storage.CreateNewTrial(studyId);
        storage.SetTrialParam(first, "f", 0.5, new FloatDistribution(0, 1));
        storage.SetTrialParam(first, "s", 0.5, new FloatDistribution(0, 1, step: 0.25));
        storage.SetTrialParam(first, "i", 3, new IntDistribution(1, 9));

        int second = storage.CreateNewTrial(studyId);
        storage.SetTrialParam(second, "f", 0.25, new FloatDistribution(0, 1));
        storage.SetTrialParam(second, "s", 0.25, new FloatDistribution(0, 1, step: 0.25));
        storage.SetTrialParam(second, "i", 5, new IntDistribution(1, 9));

        FrozenTrial a = storage.GetTrial(first), b = storage.GetTrial(second);
        b.Distributions["f"].Should().BeSameAs(a.Distributions["f"]);
        b.Distributions["s"].Should().BeSameAs(a.Distributions["s"]);
        b.Distributions["i"].Should().BeSameAs(a.Distributions["i"]);
        b.Params["f"].Should().Be(0.25);
        b.Params["i"].Should().Be(5L);

        // Compatible (same kind and log flag) but not identical: the trial keeps what the caller passed.
        var wider = new FloatDistribution(0, 2);
        var stepped = new IntDistribution(1, 9, step: 2);
        int third = storage.CreateNewTrial(studyId);
        storage.SetTrialParam(third, "f", 1.5, wider);
        storage.SetTrialParam(third, "i", 3, stepped);
        FrozenTrial c = storage.GetTrial(third);
        c.Distributions["f"].Should().BeSameAs(wider);
        c.Distributions["i"].Should().BeSameAs(stepped);
    }

    /// <summary>
    ///     A -0.0 bound equals 0.0 but prints differently, so a trial given one keeps its own instance and its repr shows
    ///     Python's <c>-0.0</c>.
    /// </summary>
    [TestMethod]
    public void ANegativeZeroBound_KeepsTheCallersInstance_AndItsRepr()
    {
        var storage = new InMemoryStorage();
        int studyId = storage.CreateNewStudy(Minimize, "negative-zero");
        int first = storage.CreateNewTrial(studyId);
        storage.SetTrialParam(first, "x", 0.5, new FloatDistribution(0.0, 1));

        var negativeZero = new FloatDistribution(-0.0, 1);
        int second = storage.CreateNewTrial(studyId);
        storage.SetTrialParam(second, "x", 0.5, negativeZero);

        FrozenTrial b = storage.GetTrial(second);
        b.Distributions["x"].Should().BeSameAs(negativeZero);
        b.Distributions["x"].ToString().Should().Contain("low=-0.0");
        storage.GetTrial(first).Distributions["x"].ToString().Should().Contain("low=0.0");
    }

    /// <summary>
    ///     Categorical distributions are never shared: equal choices may be different objects (Python calls 1 and 1.0
    ///     equal), and each trial must report the choice object of the distribution it was given.
    /// </summary>
    [TestMethod]
    public void CategoricalDistributions_AreNeverShared_SoEachTrialReportsItsOwnChoices()
    {
        var storage = new InMemoryStorage();
        int studyId = storage.CreateNewStudy(Minimize, "categorical");
        var ints = new CategoricalDistribution(new CategoricalChoice[] { 1, 2 });
        var sameInts = new CategoricalDistribution(new CategoricalChoice[] { 1, 2 });
        var doubles = new CategoricalDistribution(new CategoricalChoice[] { 1.0, 2.0 });

        int first = storage.CreateNewTrial(studyId);
        storage.SetTrialParam(first, "c", 0, ints);
        int second = storage.CreateNewTrial(studyId);
        storage.SetTrialParam(second, "c", 0, sameInts);
        int third = storage.CreateNewTrial(studyId);
        storage.SetTrialParam(third, "c", 0, doubles);  // compatible: Python equality

        storage.GetTrial(second).Distributions["c"].Should().BeSameAs(sameInts);
        storage.GetTrial(third).Distributions["c"].Should().BeSameAs(doubles);
        storage.GetTrial(first).Params["c"].Should().Be(1L);
        storage.GetTrial(third).Params["c"].Should().BeOfType<double>().And.Be(1.0);
    }

    /// <summary>
    ///     Parameter names built per suggestion are stored once per study: every trial's dictionaries key on the first
    ///     trial's string instance, and the names still compare and print as given.
    /// </summary>
    [TestMethod]
    public void ParameterNamesBuiltPerSuggestion_AreStoredOncePerStudy()
    {
        var storage = new InMemoryStorage();
        int studyId = storage.CreateNewStudy(Minimize, "names");
        var trialIds = new List<int>();
        for (int n = 0; n < 3; n++)
        {
            int trialId = storage.CreateNewTrial(studyId);
            trialIds.Add(trialId);
            // A new string instance per call, as $"x{i}" in an objective makes one.
            storage.SetTrialParam(trialId, new string("lr".ToCharArray()), 0.5, new FloatDistribution(0, 1));
        }

        string key = storage.GetTrial(trialIds[0]).Params.Keys.Single();
        foreach (int trialId in trialIds)
        {
            FrozenTrial trial = storage.GetTrial(trialId);
            trial.Params.Keys.Single().Should().BeSameAs(key);
            trial.Distributions.Keys.Single().Should().BeSameAs(key);
            trial.Params.Keys.Single().Should().Be("lr");
        }
    }

    /// <summary>
    ///     The rule behind S7, case by case: the same instance, or the same runtime type with identical bits; never a
    ///     categorical pair, a deprecated subclass against its base, two kinds, or bounds that differ only in the sign of
    ///     zero.
    /// </summary>
    [TestMethod]
    public void IsBitwiseIdentical_TheSameInstanceOrTheSameTypeAndBits_NeverCategoricals()
    {
        var categorical = new CategoricalDistribution(new CategoricalChoice[] { "a", "b" });
        Distribution.IsBitwiseIdentical(categorical, categorical).Should().BeTrue();
        Distribution.IsBitwiseIdentical(categorical, new CategoricalDistribution(new CategoricalChoice[] { "a", "b" }))
            .Should().BeFalse();

        Distribution.IsBitwiseIdentical(new FloatDistribution(0, 1), new FloatDistribution(0, 1)).Should().BeTrue();
        Distribution.IsBitwiseIdentical(new FloatDistribution(1e-3, 1, log: true), new FloatDistribution(1e-3, 1, log: true))
            .Should().BeTrue();
        Distribution.IsBitwiseIdentical(new FloatDistribution(0, 1, step: 0.1), new FloatDistribution(0, 1, step: 0.1))
            .Should().BeTrue();
        Distribution.IsBitwiseIdentical(new FloatDistribution(0, 1), new FloatDistribution(-0.0, 1)).Should().BeFalse();
        Distribution.IsBitwiseIdentical(new FloatDistribution(0, 1), new FloatDistribution(0, 1, step: 0.5)).Should().BeFalse();
        Distribution.IsBitwiseIdentical(new FloatDistribution(0, 1, step: 0.5), new FloatDistribution(0, 1, step: 0.25))
            .Should().BeFalse();
        Distribution.IsBitwiseIdentical(new FloatDistribution(0.1, 1), new FloatDistribution(0.1, 1, log: true)).Should().BeFalse();
        Distribution.IsBitwiseIdentical(new FloatDistribution(0, 1), new FloatDistribution(0, 2)).Should().BeFalse();
#pragma warning disable CS0618 // the deprecated family is the point of this case
        Distribution.IsBitwiseIdentical(new FloatDistribution(0, 1), new UniformDistribution(0, 1)).Should().BeFalse();
#pragma warning restore CS0618

        Distribution.IsBitwiseIdentical(new IntDistribution(1, 9), new IntDistribution(1, 9)).Should().BeTrue();
        Distribution.IsBitwiseIdentical(new IntDistribution(1, 9), new IntDistribution(1, 9, step: 2)).Should().BeFalse();
        Distribution.IsBitwiseIdentical(new IntDistribution(1, 9), new IntDistribution(1, 9, log: true)).Should().BeFalse();
        Distribution.IsBitwiseIdentical(new IntDistribution(0, 1), new FloatDistribution(0, 1)).Should().BeFalse();
    }

    /// <summary>
    ///     A parameter's compatibility is still checked before anything is shared: another kind or another log flag is
    ///     refused with Optuna's message, exactly as before.
    /// </summary>
    [TestMethod]
    public void IncompatibleDistributions_AreStillRefused_WithOptunasMessages()
    {
        var storage = new InMemoryStorage();
        int studyId = storage.CreateNewStudy(Minimize, "compat");
        int first = storage.CreateNewTrial(studyId);
        storage.SetTrialParam(first, "x", 0.5, new FloatDistribution(0.1, 1));
        int second = storage.CreateNewTrial(studyId);

        Action kind = () => storage.SetTrialParam(second, "x", 1, new IntDistribution(0, 1));
        Action log = () => storage.SetTrialParam(second, "x", 0.5, new FloatDistribution(0.1, 1, log: true));
        kind.Should().Throw<ValueError>().WithMessage("Cannot set different distribution kind to the same parameter name.");
        log.Should().Throw<ValueError>().WithMessage("Cannot set different log configuration to the same parameter name.");
        storage.GetTrial(second).Params.Should().BeEmpty();
    }
}

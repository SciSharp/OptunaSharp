using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading;
using System.Threading.Tasks;
using AwesomeAssertions;
using OptunaSharp.Collections;
using OptunaSharp.Distributions;
using OptunaSharp.Internal;
using OptunaSharp.Logging;
using OptunaSharp.Pruners;
using OptunaSharp.Samplers;
using OptunaSharp.Storages;

namespace OptunaSharp.Tests.Storages;

/// <summary>
///     Plan 13 (docs/plans/13-rustuna-derived-performance.md), item S12: <c>InMemoryStorage(compactFinishedTrials: true)</c>
///     stores each finished trial compactly (parameters as internal values against a schema the study's trials share,
///     attributes as arrays with long ASCII strings as bytes, nothing for empty dictionaries) while every API returns the
///     same contents. The tests compare a compact storage against Optuna's representation value by value (types and bits),
///     prove the read-only views refuse writes and the mutable accessors keep writes, pin the fallbacks that keep a
///     dictionary when a value would not read back identical, and run every sampler family twice (compact and not) to the
///     same seeded trials, with the records still compact afterwards.
/// </summary>
[TestClass]
public class CompactFinishedTrialsTests
{
    private static readonly StudyDirection[] Minimize = { StudyDirection.Minimize };

    // ------------------------------------------------------------------ fixtures

    /// <summary>
    ///     A study's worth of trials written through the storage API with every kind of value: each float, integer and
    ///     categorical flavor (a -0.0 value, log and stepped ranges, choices of every JSON type), attributes with short,
    ///     long ASCII, long odd-length lowercase hexadecimal, uppercase hexadecimal and non-ASCII strings, nested
    ///     containers, intermediate values (and a trial without), a failed and a pruned trial.
    /// </summary>
    /// <param name="compact">Whether the storage compacts finished trials.</param>
    /// <returns>The storage and the study it holds.</returns>
    private static (InMemoryStorage Storage, int StudyId) WriteEveryKind(bool compact)
    {
        var storage = new InMemoryStorage(compactFinishedTrials: compact);
        int studyId = storage.CreateNewStudy(Minimize, "every-kind");
        var categorical = new CategoricalDistribution(
            new CategoricalChoice[] { "a", CategoricalChoice.None, true, 7L, 2.5, "z" });
        for (int k = 0; k < 4; k++)
        {
            int trialId = storage.CreateNewTrial(studyId);
            storage.SetTrialParam(trialId, "x", 0.25 * k - 0.5, new FloatDistribution(-1, 1));
            storage.SetTrialParam(trialId, "neg0", -0.0, new FloatDistribution(-1, 1));
            storage.SetTrialParam(trialId, "lx", 0.01 * (k + 1), new FloatDistribution(1e-3, 1, log: true));
            storage.SetTrialParam(trialId, "sx", 0.25 * k, new FloatDistribution(0, 1, step: 0.25));
            storage.SetTrialParam(trialId, "n", k - 2, new IntDistribution(-3, 3));
            storage.SetTrialParam(trialId, "ln", 1 << k, new IntDistribution(1, 64, log: true));
            storage.SetTrialParam(trialId, "sn", 2 * k, new IntDistribution(0, 10, step: 2));
            // A new categorical distribution per trial, as every SuggestCategorical builds one.
            var perTrialCategorical = new CategoricalDistribution(
                new CategoricalChoice[] { "a", CategoricalChoice.None, true, 7L, 2.5, "z" });
            storage.SetTrialParam(trialId, "c", k, k % 2 == 0 ? categorical : perTrialCategorical);
            storage.SetTrialParam(trialId, "c5", 5, perTrialCategorical);

            storage.SetTrialUserAttr(trialId, "short", "s");
            storage.SetTrialUserAttr(trialId, "ascii", new string('q', 100) + k);
            storage.SetTrialUserAttr(trialId, "hex", LowerHex(65 + k));
            storage.SetTrialUserAttr(trialId, "upper-hex", LowerHex(80).ToUpperInvariant());
            storage.SetTrialUserAttr(trialId, "unicode", new string('é', 70));
            storage.SetTrialUserAttr(trialId, "nested", new List<object?> { 1L, "two", 3.5, null, true });
            storage.SetTrialUserAttr(trialId, "map", new OrderedDict<string, object?> { ["a"] = 1L, ["b"] = -0.0 });
            storage.SetTrialUserAttr(trialId, "none", null);
            storage.SetTrialSystemAttr(trialId, "gen", (long)k);
            storage.SetTrialSystemAttr(trialId, "chunk:0", LowerHex(2045));
            storage.SetTrialSystemAttr(trialId, "constraints", new List<object?> { 0.5 - k, -1.0 });
            if (k != 1)
            {
                storage.SetTrialIntermediateValue(trialId, 0, 1.5 * k);
                storage.SetTrialIntermediateValue(trialId, 7, -0.0);
                storage.SetTrialIntermediateValue(trialId, long.MaxValue, double.NegativeInfinity);
            }
            TrialState state = k switch { 2 => TrialState.Fail, 3 => TrialState.Pruned, _ => TrialState.Complete };
            storage.SetTrialStateValues(trialId, state, state == TrialState.Complete ? new[] { 1.0 / (k + 1) } : null);
        }
        return (storage, studyId);
    }

    /// <summary>A lowercase hexadecimal string of the given length (odd lengths included).</summary>
    /// <param name="length">Number of digits.</param>
    /// <returns>Digits cycling through <c>0-9a-f</c>.</returns>
    private static string LowerHex(int length)
    {
        var sb = new StringBuilder(length);
        for (int i = 0; i < length; i++)
            sb.Append("0123456789abcdef"[(i * 7 + 3) % 16]);
        return sb.ToString();
    }

    /// <summary>
    ///     Everything a reader can learn from a trial except object identity and the clock: values with their runtime
    ///     types and bits, entries in order, the distributions' reprs and the trial's repr without its datetimes.
    /// </summary>
    /// <param name="trial">The trial to describe; read through the read-only accessors only.</param>
    /// <returns>A string equal for two trials exactly when no read can tell them apart.</returns>
    private static string Fingerprint(FrozenTrial trial)
    {
        var sb = new StringBuilder();
        sb.Append(trial.Number).Append('|').Append(trial.State).Append("|values=");
        sb.Append(trial.Values is null
            ? "None"
            : string.Join(",", trial.Values.Select(v => BitConverter.DoubleToInt64Bits(v))));
        foreach (var (name, value) in trial.Params)
            sb.Append("|p:").Append(name).Append('=').Append(Describe(value));
        foreach (var (name, distribution) in trial.Distributions)
            sb.Append("|d:").Append(name).Append('=').Append(distribution.GetType().Name).Append(distribution);
        foreach (var (key, value) in trial.UserAttrs)
            sb.Append("|u:").Append(key).Append('=').Append(Describe(value));
        foreach (var (key, value) in trial.SystemAttrs)
            sb.Append("|s:").Append(key).Append('=').Append(Describe(value));
        foreach (var (step, value) in trial.IntermediateValuesForRead)
            sb.Append("|i:").Append(step).Append('=').Append(BitConverter.DoubleToInt64Bits(value));
        sb.Append("|last=").Append(trial.LastStep);
        sb.Append("|constraints=").Append(string.Join(",", trial.Constraints.Select(kv => $"{kv.Key}:{BitConverter.DoubleToInt64Bits(kv.Value)}")));
        sb.Append("|repr=").Append(WithoutDatetimes(trial.ToString()));
        return sb.ToString();
    }

    /// <summary>A value's runtime type and Python repr, with doubles by their bits so -0.0 and 0.0 differ.</summary>
    /// <param name="value">The value to describe.</param>
    /// <returns>Type name and representation.</returns>
    private static string Describe(object? value) => value switch
    {
        null => "None",
        double d => $"double:{BitConverter.DoubleToInt64Bits(d)}",
        _ => $"{value.GetType().Name}:{PythonRepr.Repr(value)}",
    };

    /// <summary>A trial repr with its two datetimes blanked, the one thing two storages cannot agree on.</summary>
    /// <param name="repr">The repr from <see cref="FrozenTrial.ToString" />.</param>
    /// <returns>The repr with <c>datetime_start</c> and <c>datetime_complete</c> values removed.</returns>
    private static string WithoutDatetimes(string repr)
        => Regex.Replace(repr, @"datetime_(start|complete)=('[^']*'|None)", "datetime_$1=?");

    // ------------------------------------------------------------------ the switch and what it touches

    /// <summary>The default storage keeps Optuna's representation: no record is ever compact, every dictionary an OrderedDict.</summary>
    [TestMethod]
    public void Default_IsOff_AndKeepsEveryDictionary()
    {
        new InMemoryStorage().CompactFinishedTrials.Should().BeFalse();
        new InMemoryStorage(compactFinishedTrials: false).CompactFinishedTrials.Should().BeFalse();
        new InMemoryStorage(compactFinishedTrials: true).CompactFinishedTrials.Should().BeTrue();

        var (storage, studyId) = WriteEveryKind(compact: false);
        foreach (FrozenTrial trial in storage.GetAllTrials(studyId, deepcopy: false))
        {
            trial.IsCompact.Should().BeFalse();
            trial.Params.Should().BeOfType<OrderedDict<string, object?>>();
            trial.Distributions.Should().BeOfType<OrderedDict<string, BaseDistribution>>();
            trial.UserAttrs.Should().BeOfType<OrderedDict<string, object?>>();
            trial.SystemAttrs.Should().BeOfType<OrderedDict<string, object?>>();
        }
    }

    /// <summary>
    ///     Only finished records are compacted: a running and a waiting trial keep their dictionaries (they are still
    ///     written), each finished state (complete, failed, pruned) is compact, and a trial added already finished is
    ///     compacted at creation.
    /// </summary>
    [TestMethod]
    public void OnlyFinishedRecords_AreCompacted()
    {
        var storage = new InMemoryStorage(compactFinishedTrials: true);
        var study = Optuna.CreateStudy(storage: storage, sampler: new RandomSampler(seed: 1));
        study.EnqueueTrial(new OrderedDict<string, object?> { ["x"] = 0.5 });
        Trial running = study.Ask();   // pops the enqueued trial
        running.SuggestFloat("x", 0, 1);
        study.EnqueueTrial(new OrderedDict<string, object?> { ["x"] = 0.25 });
        study.AddTrial(FrozenTrial.CreateTrial(
            value: 3.0,
            @params: new OrderedDict<string, object?> { ["x"] = 0.75 },
            distributions: new OrderedDict<string, BaseDistribution> { ["x"] = new FloatDistribution(0, 1) },
            userAttrs: new OrderedDict<string, object?> { ["added"] = true }));
        study.Optimize(t => t.SuggestFloat("x", 0, 1), nTrials: 3);

        var records = storage.GetAllTrials(study.StudyId, deepcopy: false);
        foreach (FrozenTrial record in records)
            record.IsCompact.Should().Be(record.State.IsFinished(), $"trial {record.Number} is {record.State}");
        records.Count(r => r.State == TrialState.Running).Should().Be(1);
        records.Single(r => r.UserAttrs.ContainsKey("added")).Params["x"].Should().Be(0.75);
    }

    // ------------------------------------------------------------------ contents: identical to Optuna's representation

    /// <summary>
    ///     Every kind of value reads back from a compact record exactly as from Optuna's representation: same runtime
    ///     types, same bits (-0.0 stays -0.0), same order, same reprs and constraints, the same <c>LastStep</c>; and the
    ///     compact storage did compact every finished record, with the long strings stored as bytes (hexadecimal two to a
    ///     byte, odd length included) and the uppercase-hex and non-ASCII strings left as strings.
    /// </summary>
    [TestMethod]
    public void EveryKindOfValue_ReadsBackIdentical()
    {
        var (plain, plainStudy) = WriteEveryKind(compact: false);
        var (compact, compactStudy) = WriteEveryKind(compact: true);
        var expected = plain.GetAllTrials(plainStudy, deepcopy: false);
        var actual = compact.GetAllTrials(compactStudy, deepcopy: false);

        actual.Select(Fingerprint).Should().Equal(expected.Select(Fingerprint));
        actual.Should().OnlyContain(t => t.IsCompact);
        actual.Select(t => t.Params).Should().AllBeOfType<CompactParams>();

        var userAttrs = (CompactAttrs)actual[0].UserAttrs;
        userAttrs.StoredValue("ascii").Should().BeOfType<CompactString>().Which.IsHex.Should().BeFalse();
        var hex = userAttrs.StoredValue("hex").Should().BeOfType<CompactString>().Which;
        hex.IsHex.Should().BeTrue();
        hex.StoredBytes.Should().Be(32, "65 digits are 32 bytes and one spare digit");
        userAttrs.StoredValue("upper-hex").Should().BeOfType<CompactString>().Which.IsHex.Should().BeFalse();
        userAttrs.StoredValue("unicode").Should().BeOfType<string>();
        userAttrs.StoredValue("short").Should().BeOfType<string>();
        ((CompactAttrs)actual[0].SystemAttrs).StoredValue("chunk:0").Should().BeOfType<CompactString>()
            .Which.StoredBytes.Should().Be(1022, "a 2,045-digit CMA-ES chunk is 1,022 bytes and one spare digit");
        actual[1].IntermediateValuesForRead.Should().BeEmpty();
        actual[1].LastStep.Should().BeNull();

        // The storage's own internal-value read returns the stored value of a compact trial without converting it.
        foreach (FrozenTrial trial in expected)
        {
            foreach (string name in trial.Params.Keys)
                BitConverter.DoubleToInt64Bits(compact.GetTrialParam(trial.TrialId, name))
                    .Should().Be(BitConverter.DoubleToInt64Bits(plain.GetTrialParam(trial.TrialId, name)), name);
            Assert.ThrowsExactly<KeyNotFoundException>(() => plain.GetTrialParam(trial.TrialId, "missing"));
            Assert.ThrowsExactly<KeyNotFoundException>(() => compact.GetTrialParam(trial.TrialId, "missing"));
        }
    }

    /// <summary>
    ///     A compact record equals the same trial in Optuna's representation, both ways, and prints the same repr: the
    ///     equality and repr paths read the views, not the dictionaries.
    /// </summary>
    [TestMethod]
    public void Equals_AndRepr_MatchOptunasRepresentation()
    {
        var (plain, plainStudy) = WriteEveryKind(compact: false);
        var (compact, compactStudy) = WriteEveryKind(compact: true);
        var expected = plain.GetAllTrials(plainStudy, deepcopy: false);
        var actual = compact.GetAllTrials(compactStudy, deepcopy: false);
        for (int i = 0; i < actual.Count; i++)
        {
            FrozenTrial twin = expected[i].DeepCopy();
            twin.SetDatetimeStart(actual[i].DatetimeStart);
            twin.DatetimeComplete = actual[i].DatetimeComplete;
            twin.TrialId = actual[i].TrialId;

            actual[i].Equals(twin).Should().BeTrue($"trial {i} compact equals the dictionary form");
            twin.Equals(actual[i]).Should().BeTrue($"trial {i} dictionary form equals the compact one");
            actual[i].ToString().Should().Be(twin.ToString());
            actual[i].IsCompact.Should().BeTrue("comparing and printing read the views only");
        }
    }

    /// <summary>
    ///     A deep copy of a compact record holds real dictionaries with the same contents and leaves the record compact;
    ///     a shallow copy first gives the record dictionaries of its own and shares them, as Python's <c>copy.copy</c>
    ///     shares a trial's dicts.
    /// </summary>
    [TestMethod]
    public void Copies_DeepHoldDictionaries_ShallowShareTheRecordsOwn()
    {
        var (compact, studyId) = WriteEveryKind(compact: true);
        FrozenTrial record = compact.GetAllTrials(studyId, deepcopy: false)[0];

        FrozenTrial deep = record.DeepCopy();
        deep.IsCompact.Should().BeFalse();
        deep.Params.Should().BeOfType<OrderedDict<string, object?>>();
        Fingerprint(deep).Should().Be(Fingerprint(record));
        record.IsCompact.Should().BeTrue();
        compact.GetAllTrials(studyId, deepcopy: true).Select(Fingerprint)
            .Should().Equal(compact.GetAllTrials(studyId, deepcopy: false).Select(Fingerprint));

        string before = Fingerprint(record);
        FrozenTrial shallow = record.ShallowCopy();
        record.IsCompact.Should().BeFalse("a shallow copy shares real dictionaries");
        shallow.ParamsDict.Should().BeSameAs(record.ParamsDict);
        shallow.DistributionsDict.Should().BeSameAs(record.DistributionsDict);
        shallow.UserAttrsDict.Should().BeSameAs(record.UserAttrsDict);
        shallow.SystemAttrsDict.Should().BeSameAs(record.SystemAttrsDict);
        shallow.IntermediateValues.Should().BeSameAs(record.IntermediateValues);
        Fingerprint(record).Should().Be(before);
        shallow.ParamsDict["x"] = 0.125;
        record.Params["x"].Should().Be(0.125, "the copy and the record share the dictionary");
    }

    // ------------------------------------------------------------------ the views and the mutable accessors

    /// <summary>
    ///     The read-only views behave as dictionaries for reads (indexer, lookups, ordered keys and values, the
    ///     non-generic interface PythonRepr and PyJson use, CopyTo) and refuse every write through
    ///     <see cref="IDictionary" />; a missing key throws the same exception, with the same message, as an
    ///     <see cref="OrderedDict{TKey,TValue}" />.
    /// </summary>
    [TestMethod]
    public void Views_ReadLikeDictionaries_AndRefuseWrites()
    {
        var (plain, plainStudy) = WriteEveryKind(compact: false);
        var (compact, compactStudy) = WriteEveryKind(compact: true);
        FrozenTrial expected = plain.GetAllTrials(plainStudy, deepcopy: false)[0];
        FrozenTrial actual = compact.GetAllTrials(compactStudy, deepcopy: false)[0];

        void CheckView<TValue>(IReadOnlyDictionary<string, TValue> view, IReadOnlyDictionary<string, TValue> reference)
        {
            view.Should().NotBeAssignableTo<OrderedDict<string, TValue>>();
            view.Count.Should().Be(reference.Count);
            view.Keys.Should().Equal(reference.Keys);
            view.Values.Select(v => Describe(v)).Should().Equal(reference.Values.Select(v => Describe(v)));
            foreach (string key in reference.Keys)
            {
                view.ContainsKey(key).Should().BeTrue();
                view.TryGetValue(key, out TValue? value).Should().BeTrue();
                Describe(value).Should().Be(Describe(reference[key]));
                Describe(view[key]).Should().Be(Describe(reference[key]));
            }
            view.ContainsKey("missing").Should().BeFalse();
            view.TryGetValue("missing", out _).Should().BeFalse();
            string expectedMessage = Assert.ThrowsExactly<KeyNotFoundException>(() => _ = reference["missing"]).Message;
            Assert.ThrowsExactly<KeyNotFoundException>(() => _ = view["missing"]).Message.Should().Be(expectedMessage);
            Assert.ThrowsExactly<ArgumentNullException>(() => view.ContainsKey(null!));

            var nonGeneric = (IDictionary)view;
            var referenceNonGeneric = (IDictionary)reference;
            nonGeneric.IsReadOnly.Should().BeTrue();
            nonGeneric.IsFixedSize.Should().BeTrue();
            nonGeneric.Count.Should().Be(reference.Count);
            nonGeneric.Keys.Cast<string>().Should().Equal(referenceNonGeneric.Keys.Cast<string>());
            nonGeneric.Values.Cast<object?>().Select(Describe).Should().Equal(referenceNonGeneric.Values.Cast<object?>().Select(Describe));
            nonGeneric.Contains(reference.Keys.First()).Should().BeTrue();
            nonGeneric.Contains(42).Should().BeFalse("a key of another type is absent, as in a dictionary");
            nonGeneric["missing"].Should().BeNull();
            nonGeneric[42].Should().BeNull();
            var entries = new List<string>();
            IDictionaryEnumerator enumerator = nonGeneric.GetEnumerator();
            while (enumerator.MoveNext())
                entries.Add($"{enumerator.Key}={Describe(enumerator.Value)}={Describe(enumerator.Entry.Value)}");
            entries.Should().Equal(reference.Select(kv => $"{kv.Key}={Describe(kv.Value)}={Describe(kv.Value)}"));
            enumerator.MoveNext().Should().BeFalse();
            enumerator.Reset();
            enumerator.MoveNext().Should().BeTrue();
            enumerator.Key.Should().Be(reference.Keys.First());

            var pairs = new KeyValuePair<string, TValue>[reference.Count + 1];
            nonGeneric.CopyTo(pairs, 1);
            pairs.Skip(1).Select(kv => kv.Key).Should().Equal(reference.Keys);
            var dictionaryEntries = new DictionaryEntry[reference.Count];
            nonGeneric.CopyTo(dictionaryEntries, 0);
            dictionaryEntries.Select(e => (string)e.Key).Should().Equal(reference.Keys);
            var objects = new object?[reference.Count];
            nonGeneric.CopyTo(objects, 0);
            objects.Should().AllBeOfType<KeyValuePair<string, TValue>>();
            Assert.ThrowsExactly<ArgumentException>(() => nonGeneric.CopyTo(new int[reference.Count], 0));
            Assert.ThrowsExactly<ArgumentException>(() => nonGeneric.CopyTo(new object?[reference.Count - 1], 0));
            Assert.ThrowsExactly<ArgumentOutOfRangeException>(() => nonGeneric.CopyTo(objects, -1));
            nonGeneric.IsSynchronized.Should().BeFalse();
            nonGeneric.SyncRoot.Should().BeSameAs(view);

            Assert.ThrowsExactly<NotSupportedException>(() => nonGeneric["k"] = 1);
            Assert.ThrowsExactly<NotSupportedException>(() => nonGeneric.Add("k", 1));
            Assert.ThrowsExactly<NotSupportedException>(() => nonGeneric.Remove(reference.Keys.First()));
            Assert.ThrowsExactly<NotSupportedException>(() => nonGeneric.Clear());

            PythonRepr.Repr(view).Should().Be(PythonRepr.Repr(reference));
        }

        CheckView(actual.Params, expected.Params);
        CheckView(actual.Distributions, expected.Distributions);
        CheckView(actual.UserAttrs, expected.UserAttrs);
        CheckView(actual.SystemAttrs, expected.SystemAttrs);
        PyJson.Dumps(actual.UserAttrs).Should().Be(PyJson.Dumps(expected.UserAttrs));
        PyJson.Dumps(actual.Params).Should().Be(PyJson.Dumps(expected.Params));
        actual.IsCompact.Should().BeTrue("every read above went through the views");
    }

    /// <summary>
    ///     Each mutable accessor gives a compact record a real dictionary of its own, returns that same dictionary on every
    ///     later read, and keeps a change made through it; the other dictionaries stay compact until their own accessor
    ///     is read. <see cref="FrozenTrial.SetUserAttr" /> and <see cref="FrozenTrial.SetConstraint" /> write through them.
    /// </summary>
    [TestMethod]
    public void MutableAccessors_GiveTheRecordItsOwnDictionary_AndKeepWrites()
    {
        var (compact, studyId) = WriteEveryKind(compact: true);
        var records = compact.GetAllTrials(studyId, deepcopy: false);
        FrozenTrial trial = records[0];
        string before = Fingerprint(trial);

        OrderedDict<string, object?> @params = trial.ParamsDict;
        Fingerprint(trial).Should().Be(before, "turning a dictionary back changes no content");
        trial.ParamsDict.Should().BeSameAs(@params);
        trial.Params.Should().BeSameAs(@params);
        trial.Distributions.Should().BeOfType<DistributionsView>("only the parameters were asked for");
        @params["x"] = 0.875;
        trial.Params["x"].Should().Be(0.875);

        OrderedDict<string, BaseDistribution> distributions = trial.DistributionsDict;
        trial.DistributionsDict.Should().BeSameAs(distributions);
        distributions["y"] = new FloatDistribution(0, 2);
        trial.Distributions["y"].Should().Be(new FloatDistribution(0, 2));

        OrderedDict<string, object?> userAttrs = trial.UserAttrsDict;
        trial.UserAttrsDict.Should().BeSameAs(userAttrs);
        userAttrs["ascii"].Should().Be(new string('q', 100) + 0, "a compacted string comes back as the string");
        trial.SetUserAttr("added", 1L);
        trial.UserAttrs["added"].Should().Be(1L);

        OrderedDict<string, object?> systemAttrs = trial.SystemAttrsDict;
        trial.SystemAttrsDict.Should().BeSameAs(systemAttrs);
        trial.SetConstraint("c", -2.0);
        trial.SystemAttrs.Should().ContainKey("constraints:c");

        FrozenTrial noReports = records[1];
        noReports.IntermediateValuesForRead.Should().BeEmpty();
        noReports.IsCompact.Should().BeTrue();
        OrderedDict<long, double> intermediate = noReports.IntermediateValues;
        noReports.IntermediateValues.Should().BeSameAs(intermediate);
        intermediate[3] = 1.0;
        noReports.IntermediateValues[3].Should().Be(1.0);
        noReports.LastStep.Should().Be(3);

        trial.IsCompact.Should().BeFalse();
        records[2].IsCompact.Should().BeTrue("no other record was touched");
    }

    /// <summary>
    ///     Many threads asking the same compact record for a mutable dictionary at once all get the one dictionary the
    ///     record keeps (the compare-exchange in <c>FrozenTrial.Decompact</c>), for every accessor, over many records.
    /// </summary>
    [TestMethod]
    public void MutableAccessors_UnderContention_AllReturnTheOneDictionary()
    {
        var storage = new InMemoryStorage(compactFinishedTrials: true);
        var study = Optuna.CreateStudy(storage: storage, sampler: new RandomSampler(seed: 3));
        study.Optimize(t =>
        {
            t.SetUserAttr("u", 1L);
            return t.SuggestFloat("x", 0, 1) + t.SuggestInt("n", 0, 3);
        }, nTrials: 64);
        var records = storage.GetAllTrials(study.StudyId, deepcopy: false);
        records.Should().OnlyContain(r => r.IsCompact);

        const int threads = 8;
        foreach (FrozenTrial record in records)
        {
            var seen = new object[threads][];
            using var start = new Barrier(threads);
            Parallel.For(0, threads, new ParallelOptions { MaxDegreeOfParallelism = threads }, i =>
            {
                start.SignalAndWait();
                seen[i] = new object[]
                {
                    record.ParamsDict, record.DistributionsDict, record.UserAttrsDict, record.SystemAttrsDict,
                    record.IntermediateValues,
                };
            });
            for (int i = 1; i < threads; i++)
                for (int field = 0; field < seen[0].Length; field++)
                    seen[i][field].Should().BeSameAs(seen[0][field], $"trial {record.Number}, field {field}, thread {i}");
            ((object)record.ParamsDict).Should().BeSameAs(seen[0][0]);
            ((object)record.IntermediateValues).Should().BeSameAs(seen[0][4]);
        }
    }

    // ------------------------------------------------------------------ fallbacks: keep the dictionary when not exact

    /// <summary>
    ///     A trial whose parameters would not read back identical keeps its parameter dictionary (while its attributes are
    ///     still compacted): an <c>int</c> under a float distribution (Optuna keeps it an <c>int</c>), an integer beyond
    ///     2^53, a categorical choice Python-equal to an earlier one (<c>1.0</c> after <c>1</c>), a value its distribution
    ///     cannot convert, and names listed in another order than the distributions.
    /// </summary>
    /// <param name="case">Which inexact trial to add.</param>
    [TestMethod]
    [DataRow("int-under-float")]
    [DataRow("beyond-2^53")]
    [DataRow("python-equal-choice")]
    [DataRow("unconvertible")]
    [DataRow("order")]
    public void InexactParameters_KeepTheirDictionary(string @case)
    {
        var storage = new InMemoryStorage(compactFinishedTrials: true);
        int studyId = storage.CreateNewStudy(Minimize, "inexact");
        OrderedDict<string, object?> @params;
        OrderedDict<string, BaseDistribution> distributions;
        switch (@case)
        {
            case "int-under-float":
                @params = new() { ["x"] = 1 };
                distributions = new() { ["x"] = new FloatDistribution(0, 2) };
                break;
            case "beyond-2^53":
                @params = new() { ["n"] = (1L << 53) + 1 };
                distributions = new() { ["n"] = new IntDistribution(0, long.MaxValue / 2) };
                break;
            case "python-equal-choice":
                @params = new() { ["c"] = 1.0 };
                distributions = new() { ["c"] = new CategoricalDistribution(new CategoricalChoice[] { 1L, 1.0 }) };
                break;
            case "unconvertible":
                @params = new() { ["x"] = "not a number" };
                distributions = new() { ["x"] = new FloatDistribution(0, 2) };
                break;
            default:
                @params = new() { ["x"] = 0.5, ["y"] = 0.25 };
                distributions = new() { ["y"] = new FloatDistribution(0, 1), ["x"] = new FloatDistribution(0, 1) };
                break;
        }
        // Built directly (CreateTrial would reject the unconvertible value); the storage adds it already finished.
        var template = new FrozenTrial(
            number: -1, state: TrialState.Complete, value: 1.0, datetimeStart: DateTime.Now,
            datetimeComplete: DateTime.Now, @params: @params, distributions: distributions,
            userAttrs: new OrderedDict<string, object?> { ["u"] = new string('a', 80) },
            systemAttrs: new OrderedDict<string, object?>(), intermediateValues: new OrderedDict<long, double>(),
            trialId: -1);
        int trialId = storage.CreateNewTrial(studyId, template);

        FrozenTrial record = storage.GetTrial(trialId);
        record.Params.Should().BeOfType<OrderedDict<string, object?>>();
        record.Distributions.Should().BeOfType<OrderedDict<string, BaseDistribution>>();
        record.UserAttrs.Should().BeOfType<CompactAttrs>("the attributes compact independently");
        record.IsCompact.Should().BeTrue();
        foreach (var (name, value) in @params)
            Describe(record.Params[name]).Should().Be(Describe(value), "the stored value, type and all");
        record.Params.Keys.Should().Equal(@params.Keys);
    }

    /// <summary>
    ///     Trials of one search space share one schema, so one distributions view, even with a new categorical
    ///     distribution per trial; a dynamic space keeps one schema per shape; past <see cref="ParamSchemaCache.MaxSchemas" />
    ///     shapes, new shapes keep their dictionaries.
    /// </summary>
    [TestMethod]
    public void Schemas_AreSharedPerShape_AndBounded()
    {
        var storage = new InMemoryStorage(compactFinishedTrials: true);
        var study = Optuna.CreateStudy(storage: storage, sampler: new RandomSampler(seed: 5));
        study.Optimize(t =>
        {
            double x = t.SuggestFloat("x", 0, 1);
            CategoricalChoice c = t.SuggestCategorical("c", new CategoricalChoice[] { "a", "b", 3L });
            // A conditional parameter: two shapes.
            if (c.Equals((CategoricalChoice)"a"))
                t.SuggestInt("only-a", 0, 5);
            return x;
        }, nTrials: 40);
        var records = storage.GetAllTrials(study.StudyId, deepcopy: false);
        records.Should().OnlyContain(r => r.Params is CompactParams);
        var views = records.Select(r => r.Distributions).Distinct(ReferenceEqualityComparer.Instance).ToList();
        views.Should().HaveCount(2, "one schema per shape, shared across every trial of that shape");
        records.GroupBy(r => r.Params.ContainsKey("only-a"))
            .Should().OnlyContain(g => g.Select(r => r.Distributions).Distinct(ReferenceEqualityComparer.Instance).Count() == 1);

        var cache = new ParamSchemaCache();
        var distribution = new FloatDistribution(0, 1);
        for (int i = 0; i < ParamSchemaCache.MaxSchemas + 10; i++)
        {
            ParamSchema? schema = cache.FindOrAdd(
                new OrderedDict<string, object?> { [$"p{i}"] = 0.5 },
                new OrderedDict<string, BaseDistribution> { [$"p{i}"] = distribution });
            if (i < ParamSchemaCache.MaxSchemas)
                schema.Should().NotBeNull();
            else
                schema.Should().BeNull("the cache is full");
        }
        cache.Count.Should().Be(ParamSchemaCache.MaxSchemas);
        cache.FindOrAdd(new OrderedDict<string, object?> { ["p3"] = 0.1 },
                new OrderedDict<string, BaseDistribution> { ["p3"] = new FloatDistribution(0, 1) })
            .Should().NotBeNull("a known shape is still found when the cache is full");
    }

    /// <summary>
    ///     The value and distribution identity rules: bits for floating point (-0.0 is not 0.0), type for numbers (1 is
    ///     not 1.0), characters for strings, the same instance for anything else; categoricals by identical choices,
    ///     float and integer distributions bitwise, a categorical subclass by reference only; hashes agree with matches.
    /// </summary>
    [TestMethod]
    public void IdentityRules_MatchWhatAReaderCanSee()
    {
        DistributionIdentity.IsSameValue(0.0, -0.0).Should().BeFalse();
        DistributionIdentity.IsSameValue(1.5, 1.5).Should().BeTrue();
        DistributionIdentity.IsSameValue(1L, 1.0).Should().BeFalse();
        DistributionIdentity.IsSameValue(1L, 1).Should().BeFalse("long and int are different types");
        DistributionIdentity.IsSameValue(2f, 2f).Should().BeTrue();
        DistributionIdentity.IsSameValue(true, true).Should().BeTrue();
        DistributionIdentity.IsSameValue("ab", new string(new[] { 'a', 'b' })).Should().BeTrue();
        DistributionIdentity.IsSameValue(null, null).Should().BeTrue();
        DistributionIdentity.IsSameValue(null, "x").Should().BeFalse();
        DistributionIdentity.IsSameValue(1.0m, 1.00m).Should().BeFalse("decimals print their scale");
        DistributionIdentity.IsSameValue(new List<object?>(), new List<object?>()).Should().BeFalse();

        CategoricalDistribution Categorical(params CategoricalChoice[] choices) => new(choices);
        var pairs = new (BaseDistribution A, BaseDistribution B, bool Interchangeable)[]
        {
            (Categorical("a", 2L, 2.5), Categorical("a", 2L, 2.5), true),
            (Categorical("a", CategoricalChoice.None), Categorical("a", CategoricalChoice.None), true),
            (Categorical(1L), Categorical(1.0), false),
            (Categorical("a"), Categorical("a", "b"), false),
            (new FloatDistribution(0, 1), new FloatDistribution(0, 1), true),
            (new FloatDistribution(-0.0, 1), new FloatDistribution(0, 1), false),
            (new IntDistribution(0, 9, step: 3), new IntDistribution(0, 9, step: 3), true),
            (new IntDistribution(0, 9), new FloatDistribution(0, 9), false),
            (new EnumLikeCategorical("a"), new EnumLikeCategorical("a"), false),
        };
        foreach (var (a, b, interchangeable) in pairs)
        {
            DistributionIdentity.AreInterchangeable(a, b).Should().Be(interchangeable, $"{a} vs {b}");
            DistributionIdentity.AreInterchangeable(b, a).Should().Be(interchangeable, $"{b} vs {a}");
            if (interchangeable)
                DistributionIdentity.HashOf(a).Should().Be(DistributionIdentity.HashOf(b));
        }
        var same = new EnumLikeCategorical("a");
        DistributionIdentity.AreInterchangeable(same, same).Should().BeTrue();
    }

    /// <summary>
    ///     The fast path's positions: a reader's distribution that converts as the schema's does gets the stored
    ///     internal value's position, a missing name <see cref="ParamSchema.Absent" />, and a Python-equal but not
    ///     identical categorical <see cref="ParamSchema.NotInterchangeable" />; at every position the stored value is
    ///     exactly what the reader's conversion of the external value returns.
    /// </summary>
    [TestMethod]
    public void SchemaPositions_ReturnTheStoredValueOnlyWhereItIsExact()
    {
        var x = new FloatDistribution(0, 1);
        var c = new CategoricalDistribution(new CategoricalChoice[] { 1L, 2L });
        var @params = new OrderedDict<string, object?> { ["x"] = 0.25, ["c"] = 2L };
        var cache = new ParamSchemaCache();
        ParamSchema schema = cache.FindOrAdd(@params, new OrderedDict<string, BaseDistribution> { ["x"] = x, ["c"] = c })!;
        CompactParams compact = CompactParams.TryCreate(schema, @params)!;

        var readers = new BaseDistribution[]
        {
            new CategoricalDistribution(new CategoricalChoice[] { 1L, 2L }), new FloatDistribution(0, 1),
            new FloatDistribution(0, 1), new CategoricalDistribution(new CategoricalChoice[] { 1.0, 2.0 }),
        };
        int[] positions = schema.PositionsFor(new[] { "c", "x", "missing", "c" }, readers);
        positions.Should().Equal(1, 0, ParamSchema.Absent, ParamSchema.NotInterchangeable);
        compact.InternalAt(positions[0]).Should().Be(readers[0].ToInternalRepr(compact["c"]));
        compact.InternalAt(positions[1]).Should().Be(readers[1].ToInternalRepr(compact["x"]));
        readers[3].ToInternalRepr(compact["c"]).Should().Be(1.0, "the slow path still converts the Python-equal choice");
        compact.TryGetInternal("c", out double stored).Should().BeTrue();
        stored.Should().Be(1.0);
        compact.TryGetInternal("missing", out _).Should().BeFalse();
    }

    /// <summary>A categorical subclass: only the same instance may stand for it.</summary>
    private sealed class EnumLikeCategorical : CategoricalDistribution
    {
        /// <summary>Builds the subclass over the given choices.</summary>
        /// <param name="choices">The choices.</param>
        public EnumLikeCategorical(params CategoricalChoice[] choices) : base(choices)
        {
        }
    }

    /// <summary>
    ///     Strings: shorter than the minimum or non-ASCII stay strings; ASCII becomes one byte a character; lowercase
    ///     hexadecimal two digits a byte at even and odd lengths; uppercase hexadecimal is only ASCII; every stored form
    ///     reads back as a new string with the same characters.
    /// </summary>
    [TestMethod]
    public void CompactStrings_RoundTripExactly()
    {
        CompactString.TryCreate(new string('a', CompactString.MinLength - 1)).Should().BeNull();
        CompactString.TryCreate(new string('é', 100)).Should().BeNull();
        foreach (string text in new[]
                 {
                     new string('a', CompactString.MinLength), LowerHex(64), LowerHex(65), LowerHex(2045),
                     LowerHex(64).ToUpperInvariant(), "0123456789" + LowerHex(70) + " tail", new string('0', 99),
                 })
        {
            CompactString compact = CompactString.TryCreate(text)!;
            compact.Should().NotBeNull(text);
            string back = compact.Materialize();
            back.Should().Be(text);
            back.Should().NotBeSameAs(text);
            bool hex = text.All(c => c is (>= '0' and <= '9') or (>= 'a' and <= 'f'));
            compact.IsHex.Should().Be(hex, text);
            compact.StoredBytes.Should().Be(hex ? text.Length / 2 : text.Length);
        }
    }

    // ------------------------------------------------------------------ samplers: same trials, records stay compact

    /// <summary>
    ///     Each sampler family runs the same seeded study on Optuna's representation and on the compact storage: every
    ///     trial's parameters, values, attributes, intermediate values and repr are identical, and every finished record
    ///     is still compact at the end, so no sampler read gave a record its dictionaries back.
    /// </summary>
    /// <param name="kind">The sampler (and pruner or constraint) under test.</param>
    [TestMethod]
    [DataRow("random")]
    [DataRow("tpe")]
    [DataRow("tpe-independent")]
    [DataRow("tpe-group")]
    [DataRow("tpe-equal-choices")]
    [DataRow("tpe-pruned")]
    [DataRow("motpe-constrained")]
    [DataRow("cmaes")]
    [DataRow("cmaes-separable")]
    [DataRow("nsgaii-constrained")]
    [DataRow("nsgaiii")]
    [DataRow("qmc")]
    [DataRow("gp")]
    [DataRow("bruteforce")]
    [DataRow("grid")]
    public void Samplers_GiveTheSameTrials_AndLeaveRecordsCompact(string kind)
    {
        List<string> plain = RunStudy(kind, compact: false, out _);
        List<string> compact = RunStudy(kind, compact: true, out List<FrozenTrial> records);

        compact.Should().Equal(plain);
        records.Where(r => r.State.IsFinished()).Should().OnlyContain(r => r.IsCompact && r.Params is CompactParams,
            "every sampler reads finished records through the read-only accessors");
        records.Should().Contain(r => r.State == TrialState.Complete);
    }

    /// <summary>Runs one seeded study of a sampler family and describes its trials.</summary>
    /// <param name="kind">The sampler (and pruner or constraint) to use.</param>
    /// <param name="compact">Whether the storage compacts finished trials.</param>
    /// <param name="records">The storage's records after the run.</param>
    /// <returns>One fingerprint per trial, in number order.</returns>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="kind" /> names no sampler family.</exception>
    /// <exception cref="TrialPruned">Never escapes: the pruned objective throws it inside the optimize loop, which records the trial as pruned.</exception>
    private static List<string> RunStudy(string kind, bool compact, out List<FrozenTrial> records)
    {
        var storage = new InMemoryStorage(compactFinishedTrials: compact);
        Func<FrozenTrial, IEnumerable<double>> constraints = t => new[] { (double)t.Params["x"]! - 1.0 };
        bool twoObjectives = kind is "motpe-constrained" or "nsgaii-constrained" or "nsgaiii";
        BaseSampler sampler = kind switch
        {
            "random" => new RandomSampler(seed: 11),
            "tpe" => new TPESampler(seed: 11, multivariate: true, nStartupTrials: 5),
            "tpe-independent" => new TPESampler(seed: 11, nStartupTrials: 5),
            "tpe-group" => new TPESampler(seed: 11, multivariate: true, group: true, nStartupTrials: 5),
            "tpe-equal-choices" => new TPESampler(seed: 11, multivariate: true, nStartupTrials: 5),
            "tpe-pruned" => new TPESampler(seed: 11, nStartupTrials: 5),
            "motpe-constrained" => new TPESampler(seed: 11, nStartupTrials: 5, constraintsFunc: constraints),
            "cmaes" => new CmaEsSampler(seed: 11),
            "cmaes-separable" => new CmaEsSampler(seed: 11, useSeparableCma: true),
            "nsgaii-constrained" => new NSGAIISampler(populationSize: 8, seed: 11, constraintsFunc: constraints),
            "nsgaiii" => new NSGAIIISampler(populationSize: 8, seed: 11),
            "qmc" => new QMCSampler(seed: 11),
            "gp" => new GPSampler(seed: 11, nStartupTrials: 4),
            "bruteforce" => new BruteForceSampler(seed: 11),
            "grid" => new GridSampler(new Dictionary<string, IReadOnlyList<CategoricalChoice>>
            {
                ["x"] = new CategoricalChoice[] { -1.0, 0.0, 2.5 },
                ["c"] = new CategoricalChoice[] { "a", 3L },
            }, seed: 11),
            _ => throw new ArgumentOutOfRangeException(nameof(kind)),
        };
        BasePruner? pruner = kind == "tpe-pruned" ? new MedianPruner(nStartupTrials: 2) : null;
        var study = Optuna.CreateStudy(
            storage: storage, sampler: sampler, pruner: pruner,
            directions: twoObjectives ? new[] { StudyDirection.Minimize, StudyDirection.Maximize } : Minimize);
        int nTrials = kind switch
        {
            "gp" => 10,
            "grid" => 6,
            "bruteforce" => 12,
            "tpe" or "tpe-independent" or "tpe-group" or "tpe-equal-choices" or "tpe-pruned" or "motpe-constrained" => 25,
            _ => 40,
        };

        double[] Evaluate(Trial trial)
        {
            double x;
            CategoricalChoice c;
            if (kind == "bruteforce")
            {
                x = trial.SuggestFloat("x", -1, 1, step: 0.5);
                trial.SuggestInt("n", 0, 1);
                c = trial.SuggestCategorical("c", new CategoricalChoice[] { "a", true });
            }
            else if (kind == "grid")
            {
                x = trial.SuggestFloat("x", -1, 3);
                c = trial.SuggestCategorical("c", new CategoricalChoice[] { "a", 3L });
            }
            else
            {
                x = trial.SuggestFloat("x", -5, 5);
                if (kind != "gp")
                {
                    trial.SuggestFloat("lx", 1e-3, 1e3, log: true);
                    trial.SuggestFloat("sx", -1, 1, step: 0.25);
                    trial.SuggestInt("ln", 1, 64, log: true);
                }
                trial.SuggestInt("n", -3, 3);
                c = trial.SuggestCategorical("c", new CategoricalChoice[] { "a", "b", CategoricalChoice.None, 7L, 2.5 });
                if (kind == "tpe-group" && c.Equals((CategoricalChoice)"a"))
                    trial.SuggestFloat("only-a", 0, 1);
                // Python-equal choices that are not identical (1 and 1.0): the search space keeps the first trial's
                // distribution, which reads the other variant's compact trials through their external values.
                if (kind == "tpe-equal-choices")
                {
                    trial.SuggestCategorical("e", trial.Number % 2 == 0
                        ? new CategoricalChoice[] { 1L, 2L }
                        : new CategoricalChoice[] { 1.0, 2.0 });
                }
            }
            trial.SetUserAttr("blob", new string('q', 90) + trial.Number);
            trial.SetUserAttr("hex", LowerHex(65 + trial.Number % 3));
            trial.SetUserAttr("nested", new List<object?> { (long)trial.Number, "two", 3.5 });
            double value = x * x + (c.Value is string ? 0 : 1);
            if (kind == "tpe-pruned")
            {
                for (int step = 0; step < 4; step++)
                {
                    trial.Report(value + 1.0 / (step + 1), step);
                    if (trial.ShouldPrune())
                        throw new TrialPruned();
                }
            }
            return twoObjectives ? new[] { value, -x } : new[] { value };
        }

        study.Optimize(t => (IReadOnlyList<double>)Evaluate(t), nTrials: nTrials);
        records = storage.GetAllTrials(study.StudyId, deepcopy: false);
        // Fingerprints read only through the read-only accessors, so they cannot undo compaction themselves.
        return records.Select(Fingerprint).ToList();
    }

    /// <summary>
    ///     A CMA-ES study stores its optimizer state as 2,045-digit lowercase hexadecimal chunks in system attributes;
    ///     the compact storage keeps them as bytes (two digits a byte), the sampler restores from them, and a second
    ///     process-free reload of the same storage continues to the same trials.
    /// </summary>
    [TestMethod]
    public void CmaEsOptimizerState_IsStoredAsBytes_AndRestores()
    {
        static List<string> Run(InMemoryStorage storage, int seed)
        {
            var study = Optuna.CreateStudy(storage: storage, sampler: new CmaEsSampler(seed: seed), studyName: "cma",
                loadIfExists: true);
            study.Optimize(t => Enumerable.Range(0, 6).Sum(i => Math.Pow(t.SuggestFloat($"x{i}", -3, 3) - 1, 2)), nTrials: 40);
            return storage.GetAllTrials(study.StudyId, deepcopy: false).Select(Fingerprint).ToList();
        }

        var plain = new InMemoryStorage();
        var compact = new InMemoryStorage(compactFinishedTrials: true);
        Run(compact, 2).Should().Equal(Run(plain, 2));
        // A new sampler over the same storage restores the optimizer from the stored chunks.
        Run(compact, 3).Should().Equal(Run(plain, 3));

        var holders = compact.GetAllTrials(0, deepcopy: false)
            .Where(r => r.SystemAttrs.Keys.Any(k => k.StartsWith("cma:optimizer", StringComparison.Ordinal)))
            .ToList();
        holders.Should().NotBeEmpty();
        foreach (FrozenTrial holder in holders)
        {
            var attrs = (CompactAttrs)holder.SystemAttrs;
            foreach (string key in holder.SystemAttrs.Keys.Where(k => k.StartsWith("cma:optimizer", StringComparison.Ordinal)))
                attrs.StoredValue(key).Should().BeOfType<CompactString>().Which.IsHex.Should().BeTrue(key);
        }
    }

    // ------------------------------------------------------------------ concurrency

    /// <summary>
    ///     Four workers optimize one compact study while other threads read every record through both kinds of
    ///     accessor: no read fails or sees a partial record, every finished record that only the read-only accessors
    ///     touched is still compact, and each record's contents never change between reads.
    /// </summary>
    [TestMethod]
    public void ParallelWorkers_AndReaders_SeeConsistentRecords()
    {
        var storage = new InMemoryStorage(compactFinishedTrials: true);
        var study = Optuna.CreateStudy(storage: storage, sampler: new TPESampler(seed: 7, nStartupTrials: 5));
        using var stop = new CancellationTokenSource();
        var failures = new List<Exception>();
        var firstSeen = new System.Collections.Concurrent.ConcurrentDictionary<int, string>();
        var readers = Enumerable.Range(0, 3).Select(r => Task.Run(() =>
        {
            try
            {
                while (!stop.IsCancellationRequested)
                {
                    foreach (FrozenTrial record in study.GetTrials(deepcopy: false))
                    {
                        if (!record.State.IsFinished())
                            continue;
                        string fingerprint = Fingerprint(record);
                        string first = firstSeen.GetOrAdd(record.Number, fingerprint);
                        if (first != fingerprint)
                            throw new InvalidOperationException($"trial {record.Number} changed between reads");
                        // One reader also takes the mutable dictionaries of every third record.
                        if (r == 0 && record.Number % 3 == 0)
                            _ = record.ParamsDict.Count + record.UserAttrsDict.Count;
                    }
                }
            }
            catch (Exception exception)
            {
                lock (failures)
                    failures.Add(exception);
            }
        })).ToArray();

        study.Optimize(t =>
        {
            t.SetUserAttr("blob", new string('w', 70));
            return Math.Pow(t.SuggestFloat("x", -2, 2), 2) + t.SuggestInt("n", 0, 3);
        }, nTrials: 60, nJobs: 4);
        stop.Cancel();
        Task.WaitAll(readers);

        failures.Should().BeEmpty();
        var records = storage.GetAllTrials(study.StudyId, deepcopy: false);
        records.Should().HaveCount(60).And.OnlyContain(r => r.State == TrialState.Complete);
        records.Where(r => r.Number % 3 != 0).Should().OnlyContain(r => r.IsCompact);
        foreach (FrozenTrial record in records)
            if (firstSeen.TryGetValue(record.Number, out string? seen))
                Fingerprint(record).Should().Be(seen);
    }

    // ------------------------------------------------------------------ memory

    /// <summary>
    ///     What the switch is for: a finished 40-parameter random-search study retains at most a quarter of the memory
    ///     in the compact storage (measured about 0.76 KB against 6.8 KB a trial at 1,000 trials, 0.65 KB against
    ///     6.7 KB at 5,000).
    /// </summary>
    [TestMethod]
    [DoNotParallelize]
    public void CompactStorage_RetainsAFractionOfTheMemory()
    {
        int verbosity = OptunaLogging.GetVerbosity();
        // Each finished trial logs a line with all its parameters at INFO, and the test host keeps every line it
        // captures: about 2.3 KB a trial, more than the compact record itself, in both measurements.
        OptunaLogging.SetVerbosity(LogLevel.Warning);
        try
        {
            // A first study pays for the statics and the JIT of both paths outside the measurements.
            RetainedBytes(compact: true, trials: 200);
            long plain = RetainedBytes(compact: false, trials: 1000);
            long compact = RetainedBytes(compact: true, trials: 1000);
            compact.Should().BeLessThan(plain / 4, $"compact {compact / 1000.0:F0} KB against {plain / 1000.0:F0} KB");
        }
        finally
        {
            OptunaLogging.SetVerbosity(verbosity);
        }
    }

    /// <summary>The heap a finished 40-parameter random-search study keeps alive.</summary>
    /// <param name="compact">Whether the storage compacts finished trials.</param>
    /// <param name="trials">Number of trials to run.</param>
    /// <returns>Bytes retained by the storage and the study after a full collection.</returns>
    [MethodImpl(MethodImplOptions.NoInlining)]
    private static long RetainedBytes(bool compact, int trials)
    {
        string[] names = Enumerable.Range(0, 40).Select(i => "x" + i).ToArray();
        long before = GC.GetTotalMemory(forceFullCollection: true);
        var storage = new InMemoryStorage(compactFinishedTrials: compact);
        var study = Optuna.CreateStudy(storage: storage, sampler: new RandomSampler(seed: 0));
        study.Optimize(t => names.Sum(n => t.SuggestFloat(n, -10, 10)), nTrials: trials);
        long after = GC.GetTotalMemory(forceFullCollection: true);
        GC.KeepAlive(storage);
        GC.KeepAlive(study);
        return after - before;
    }
}

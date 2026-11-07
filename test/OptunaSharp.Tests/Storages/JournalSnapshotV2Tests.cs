using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Text;
using AwesomeAssertions;
using NumSharp;
using OptunaSharp.Collections;
using OptunaSharp.Distributions;
using OptunaSharp.Internal;
using OptunaSharp.Storages.Journal;
using OptunaSharp.Tests.Internal;

namespace OptunaSharp.Tests.Storages;

/// <summary>
///     Holds the binary v2 journal snapshot (plan 10, item 2.7) to the v1 JSON snapshot it replaced: for every replay
///     state, restoring the v2 bytes gives exactly what restoring the v1 bytes gives, which is the state itself.
/// </summary>
/// <remarks>
///     <para>
///         The owner chose the binary format on the condition of "very high parity test coverage". The states here
///         come from random journals of every record kind (studies with one or several directions, deleted studies,
///         trials of every state with no, one or several values, NaN and infinite values, every distribution type
///         including the deprecated ones, intermediate values at 64-bit steps, attributes of every JSON kind, records
///         another worker had rejected), replayed by <see cref="JournalStorageReplayResult.ApplyLogs" /> as a real
///         journal would be. Equality is checked three ways: v1's bytes for the restored state equal v1's bytes for
///         the original (everything v1 captures), every field of every study and trial compared strictly (runtime
///         types, key order, double bits), and the v2 bytes stable across a round trip.
///     </para>
///     <para>
///         The remaining cases cover the format's edges: v1 snapshots still restoring, corrupt or truncated v2 bytes
///         restoring nothing instead of throwing, attribute values a record cannot carry taking v1's normalization,
///         the v1 fallback keeping v1's exception, and distribution objects shared as the owner chose.
///     </para>
/// </remarks>
[TestClass]
public class JournalSnapshotV2Tests
{
    /// <summary>The worker id of every generated record: another process's, so a rejected record is skipped silently.</summary>
    private const string OtherWorker = "other-process-0";

    // ------------------------------------------------------------------ random journals

    /// <summary>
    ///     Builds a replay state from a random journal: random records of every operation, serialized to JSON and parsed
    ///     back as a backend would hand them over.
    /// </summary>
    /// <param name="seed">The generator's seed.</param>
    /// <param name="records">How many records to generate.</param>
    /// <returns>The replayed state.</returns>
    internal static JournalStorageReplayResult RandomState(int seed, int records)
    {
        var rng = new Random(seed);
        var state = new JournalStorageReplayResult("this-process-");
        var logs = new List<OrderedDict<string, object?>>();
        int studies = 0;
        int trials = 0;
        for (int i = 0; i < records; i++)
        {
            OrderedDict<string, object?> log = RandomRecord(rng, ref studies, ref trials);
            // Through JSON text, as every backend delivers records: the values are then exactly PyJson's object model.
            logs.Add((OrderedDict<string, object?>)PyJsonUtf8.Loads(Encoding.UTF8.GetBytes(PyJson.Dumps(log)))!);
        }
        state.ApplyLogs(logs);
        return state;
    }

    /// <summary>One random journal record, of the kinds Optuna's <c>JournalStorage</c> writes.</summary>
    /// <param name="rng">The generator.</param>
    /// <param name="studies">The studies created so far; a create-study record adds one.</param>
    /// <param name="trials">The trials created so far; a create-trial record adds one.</param>
    /// <returns>The record.</returns>
    private static OrderedDict<string, object?> RandomRecord(Random rng, ref int studies, ref int trials)
    {
        var log = new OrderedDict<string, object?> { ["worker_id"] = OtherWorker };
        int kind = studies == 0 ? 0 : trials == 0 && rng.Next(3) > 0 ? 4 : rng.Next(40);
        // Mostly trial records, as in a real journal; the rarer kinds still appear in every seed's few hundred records.
        int trialId = trials == 0 ? 0 : rng.Next(trials);
        int studyId = rng.Next(studies + 1);
        switch (kind)
        {
            case 0:
                log["op_code"] = 0L;
                log["study_name"] = "study-" + studies + RandomText(rng);
                log["directions"] = Enumerable.Range(0, 1 + rng.Next(3)).Select(_ => (object?)(long)(1 + rng.Next(2))).ToList();
                studies++;
                break;
            case 1:
                log["op_code"] = 1L;
                log["study_id"] = (long)studyId;
                break;
            case 2:
            case 3:
                log["op_code"] = kind == 2 ? 2L : 3L;
                log["study_id"] = (long)studyId;
                log[kind == 2 ? "user_attr" : "system_attr"] = new OrderedDict<string, object?>
                {
                    [RandomText(rng)] = PyJsonUtf8Tests.RandomValue(rng, depth: 0),
                };
                break;
            case 4:
            case 5:
            case 6:
            case 7:
            case 8:
            case 9:
                log["op_code"] = 4L;
                log["study_id"] = (long)rng.Next(studies);
                FillCreateTrial(rng, log);
                trials++;
                break;
            case 10:
            case 11:
            case 12:
            case 13:
            case 14:
            case 15:
            case 16:
            case 17:
            {
                log["op_code"] = 5L;
                log["trial_id"] = (long)trialId;
                log["param_name"] = "p" + rng.Next(6);
                (string json, double internalValue) = RandomDistribution(rng);
                log["param_value_internal"] = internalValue;
                log["distribution"] = json;
                break;
            }
            case 18:
            case 19:
            case 20:
            case 21:
            case 22:
            case 23:
            {
                log["op_code"] = 6L;
                log["trial_id"] = (long)trialId;
                // TrialState: 0 RUNNING (needs a start), 1 COMPLETE, 2 PRUNED, 3 FAIL (need a completion), 4 WAITING.
                long stateCode = rng.Next(5);
                log["state"] = stateCode;
                log["values"] = rng.Next(3) == 0 ? null : RandomValues(rng);
                if (stateCode == 0)
                    log["datetime_start"] = RandomIso(rng);
                else if (stateCode != 4)
                    log["datetime_complete"] = RandomIso(rng);
                break;
            }
            case 24:
            case 25:
            case 26:
            case 27:
            case 28:
            case 29:
                log["op_code"] = 7L;
                log["trial_id"] = (long)trialId;
                log["step"] = RandomStep(rng);
                log["intermediate_value"] = RandomDouble(rng);
                break;
            case 30:
            case 31:
            case 32:
            case 33:
            case 34:
                log["op_code"] = 8L;
                log["trial_id"] = (long)trialId;
                log["user_attr"] = new OrderedDict<string, object?> { [RandomText(rng)] = PyJsonUtf8Tests.RandomValue(rng, depth: 0) };
                break;
            default:
                log["op_code"] = 9L;
                log["trial_id"] = (long)trialId;
                log["system_attr"] = new OrderedDict<string, object?> { [RandomText(rng)] = PyJsonUtf8Tests.RandomValue(rng, depth: 0) };
                break;
        }
        return log;
    }

    /// <summary>Fills a create-trial record: a random state, datetimes, parameters, attributes and intermediate values.</summary>
    /// <param name="rng">The generator.</param>
    /// <param name="log">The record, holding its operation code and study id.</param>
    private static void FillCreateTrial(Random rng, OrderedDict<string, object?> log)
    {
        // TrialState: 0 RUNNING, 1 COMPLETE, 2 PRUNED, 3 FAIL, 4 WAITING (no start yet).
        long stateCode = rng.Next(5);
        log["state"] = stateCode;
        log["datetime_start"] = stateCode == 4 ? null : RandomIso(rng);
        log["datetime_complete"] = stateCode is 1 or 2 or 3 ? RandomIso(rng) : null;
        var distributions = new OrderedDict<string, object?>();
        var @params = new OrderedDict<string, object?>();
        int count = rng.Next(5);
        for (int i = 0; i < count; i++)
        {
            string name = "p" + rng.Next(6) + (rng.Next(8) == 0 ? "é\U0001F600" : "");
            (string json, double internalValue) = RandomDistribution(rng);
            distributions[name] = json;
            @params[name] = internalValue;
        }
        log["distributions"] = distributions;
        log["params"] = @params;
        log["user_attrs"] = RandomAttributes(rng);
        log["system_attrs"] = RandomAttributes(rng);
        var intermediate = new OrderedDict<string, object?>();
        int steps = rng.Next(4);
        for (int i = 0; i < steps; i++)
            intermediate[RandomStep(rng).ToString(CultureInfo.InvariantCulture)] = RandomDouble(rng);
        log["intermediate_values"] = intermediate;
        if (stateCode is 1 or 2 && rng.Next(4) > 0)
        {
            List<object?> values = RandomValues(rng);
            if (values.Count == 1 && rng.Next(2) == 0)
                log["value"] = values[0];
            else
                log["values"] = values;
        }
    }

    /// <summary>A few random attributes of every JSON kind.</summary>
    /// <param name="rng">The generator.</param>
    /// <returns>The attributes.</returns>
    private static OrderedDict<string, object?> RandomAttributes(Random rng)
    {
        var attrs = new OrderedDict<string, object?>();
        int n = rng.Next(4);
        for (int i = 0; i < n; i++)
            attrs[RandomText(rng)] = PyJsonUtf8Tests.RandomValue(rng, depth: 0);
        return attrs;
    }

    /// <summary>One to three objective values, often NaN, infinite, negative zero or subnormal.</summary>
    /// <param name="rng">The generator.</param>
    /// <returns>The values.</returns>
    private static List<object?> RandomValues(Random rng)
        => Enumerable.Range(0, 1 + rng.Next(3)).Select(_ => (object?)RandomDouble(rng)).ToList();

    /// <summary>A double with the edge cases JSON round trips must keep.</summary>
    /// <param name="rng">The generator.</param>
    /// <returns>The double.</returns>
    private static double RandomDouble(Random rng) => rng.Next(9) switch
    {
        0 => double.NaN,
        1 => double.PositiveInfinity,
        2 => double.NegativeInfinity,
        3 => -0.0,
        4 => double.Epsilon * rng.Next(1, 100),
        5 => rng.NextDouble() * Math.Pow(10, rng.Next(-300, 300)),
        _ => rng.NextDouble() * 200 - 100,
    };

    /// <summary>A reporting step, 64-bit and sometimes negative, as Python's unbounded ints allow.</summary>
    /// <param name="rng">The generator.</param>
    /// <returns>The step.</returns>
    private static long RandomStep(Random rng) => rng.Next(4) switch
    {
        0 => rng.NextInt64(),
        1 => -rng.Next(1000),
        _ => rng.Next(1000),
    };

    /// <summary>
    ///     A random instant from 1970 to 2100 at microsecond precision, in the UTC form Optuna writes, often next to the
    ///     European and Israeli daylight-saving changes, whose local wall times are ambiguous or skipped.
    /// </summary>
    /// <param name="rng">The generator.</param>
    /// <returns>The ISO text.</returns>
    private static string RandomIso(Random rng)
    {
        DateTime utc;
        if (rng.Next(3) == 0)
        {
            // The last Sundays of March and October, around 00:00 to 02:00 UTC: the changes of most of Europe and Israel.
            int year = 1980 + rng.Next(100);
            int month = rng.Next(2) == 0 ? 3 : 10;
            var lastSunday = new DateTime(year, month, DateTime.DaysInMonth(year, month), 0, 0, 0, DateTimeKind.Utc);
            while (lastSunday.DayOfWeek != DayOfWeek.Sunday)
                lastSunday = lastSunday.AddDays(-1);
            utc = lastSunday.AddDays(-1).AddTicks(rng.NextInt64(TimeSpan.TicksPerDay * 2));
        }
        else
        {
            utc = new DateTime(1970, 1, 1, 0, 0, 0, DateTimeKind.Utc).AddTicks(rng.NextInt64(TimeSpan.TicksPerDay * 365L * 130));
        }
        utc = utc.AddTicks(-(utc.Ticks % 10));
        return JournalDatetime.ToIsoUtc(utc);
    }

    /// <summary>A short random text, sometimes with non-ASCII, non-BMP and lone-surrogate characters.</summary>
    /// <param name="rng">The generator.</param>
    /// <returns>The text.</returns>
    private static string RandomText(Random rng) => rng.Next(6) switch
    {
        0 => "ké€" + rng.Next(5),
        1 => "\ud800k" + rng.Next(5),
        2 => "emoji\U0001F600" + rng.Next(5),
        _ => "k" + rng.Next(8),
    };

    /// <summary>
    ///     A random distribution's JSON, as a record carries it, with a valid internal value: every current type with
    ///     its options, every deprecated type, and the abbreviated form <see cref="Distribution.FromJson" /> also reads.
    /// </summary>
    /// <param name="rng">The generator.</param>
    /// <returns>The JSON and an internal value inside it.</returns>
    private static (string Json, double Internal) RandomDistribution(Random rng)
    {
        switch (rng.Next(10))
        {
            case 0:
                return (Distribution.ToJson(new FloatDistribution(-1.5, 2.5)), rng.NextDouble() * 4 - 1.5);
            case 1:
                return (Distribution.ToJson(new FloatDistribution(1e-5, 10, log: true)), 1e-5 + rng.NextDouble());
            case 2:
                return (Distribution.ToJson(new FloatDistribution(0, 1, step: 0.25)), rng.Next(5) * 0.25);
            case 3:
                return (Distribution.ToJson(new IntDistribution(-3, 30, step: 3)), -3 + 3 * rng.Next(12));
            case 4:
                return (Distribution.ToJson(new IntDistribution(1, 1024, log: true)), 1 + rng.Next(1024));
            case 5:
            {
                var choices = new object?[] { null, true, 3L, 2.5, "aé\U0001F600", double.NaN, -0.0 };
                return (Distribution.ToJson(new CategoricalDistribution(choices.Select(CategoricalChoice.From))), rng.Next(choices.Length));
            }
            case 6:
                return ("{\"name\": \"UniformDistribution\", \"attributes\": {\"low\": 0.0, \"high\": 3.0}}", rng.NextDouble() * 3);
            case 7:
                return ("{\"name\": \"DiscreteUniformDistribution\", \"attributes\": {\"low\": 0.0, \"high\": 1.0, \"q\": 0.5}}", rng.Next(3) * 0.5);
            case 8:
                return ("{\"name\": \"IntLogUniformDistribution\", \"attributes\": {\"low\": 1, \"high\": 64, \"step\": 1}}", 1 + rng.Next(64));
            default:
                return ("{\"type\": \"int\", \"low\": 0, \"high\": 9}", rng.Next(10));
        }
    }

    // ------------------------------------------------------------------ equality

    /// <summary>
    ///     Asserts two replay states equal in everything a snapshot carries: v1's bytes for both, then every study and
    ///     trial field compared strictly.
    /// </summary>
    /// <param name="expected">The reference state.</param>
    /// <param name="actual">The state under test.</param>
    /// <param name="because">What is being compared, for the failure message.</param>
    internal static void AssertSameState(JournalStorageReplayResult expected, JournalStorageReplayResult actual, string because)
    {
        actual.LogNumberRead.Should().Be(expected.LogNumberRead, because);
        actual.SerializeV1().Should().Equal(expected.SerializeV1(), "v1 captures the whole state: {0}", because);

        List<FrozenStudy> expectedStudies = expected.GetAllStudies();
        List<FrozenStudy> actualStudies = actual.GetAllStudies();
        actualStudies.Select(s => s.StudyId).Should().Equal(expectedStudies.Select(s => s.StudyId), because);
        for (int i = 0; i < expectedStudies.Count; i++)
        {
            FrozenStudy e = expectedStudies[i];
            FrozenStudy a = actualStudies[i];
            a.StudyName.Should().Be(e.StudyName, because);
            a.Directions.Should().Equal(e.Directions, because);
            PyJsonUtf8Tests.Same(e.UserAttrs, a.UserAttrs).Should().BeTrue("study user attrs: {0}", because);
            PyJsonUtf8Tests.Same(e.SystemAttrs, a.SystemAttrs).Should().BeTrue("study system attrs: {0}", because);
            actual.GetStudyTrialIds(a.StudyId).Should().Equal(expected.GetStudyTrialIds(e.StudyId), because);
        }

        for (int trialId = 0; ; trialId++)
        {
            FrozenTrial? e = expected.TryGetTrial(trialId);
            FrozenTrial? a = actual.TryGetTrial(trialId);
            if (e is null)
            {
                a.Should().BeNull(because);
                break;
            }
            a.Should().NotBeNull(because);
            AssertSameTrial(e, a!, $"trial {trialId}: {because}");
        }
    }

    /// <summary>Asserts two trials equal field by field: runtime types, key order and double bits included.</summary>
    /// <param name="e">The reference trial.</param>
    /// <param name="a">The trial under test.</param>
    /// <param name="because">What is being compared.</param>
    private static void AssertSameTrial(FrozenTrial e, FrozenTrial a, string because)
    {
        a.TrialId.Should().Be(e.TrialId, because);
        a.Number.Should().Be(e.Number, because);
        a.State.Should().Be(e.State, because);
        if (e.Values is null)
        {
            a.Values.Should().BeNull(because);
        }
        else
        {
            a.Values.Should().NotBeNull(because);
            a.Values!.Select(BitConverter.DoubleToInt64Bits).Should().Equal(e.Values.Select(BitConverter.DoubleToInt64Bits), because);
        }
        a.DatetimeStart.Should().Be(e.DatetimeStart, because);
        a.DatetimeStart?.Kind.Should().Be(e.DatetimeStart!.Value.Kind, because);
        a.DatetimeComplete.Should().Be(e.DatetimeComplete, because);
        a.DatetimeComplete?.Kind.Should().Be(e.DatetimeComplete!.Value.Kind, because);

        a.DistributionsDict.Keys.Should().Equal(e.DistributionsDict.Keys, because);
        foreach (var (name, distribution) in e.DistributionsDict)
        {
            a.DistributionsDict[name].GetType().Should().Be(distribution.GetType(), because);
            Distribution.ToJson(a.DistributionsDict[name]).Should().Be(Distribution.ToJson(distribution), because);
        }

        a.ParamsDict.Keys.Should().Equal(e.ParamsDict.Keys, because);
        foreach (var (name, value) in e.ParamsDict)
        {
            object? other = a.ParamsDict[name];
            (other?.GetType()).Should().Be(value?.GetType(), "param {0}: {1}", name, because);
            if (value is double d)
                BitConverter.DoubleToInt64Bits((double)other!).Should().Be(BitConverter.DoubleToInt64Bits(d), because);
            else
                other.Should().Be(value, because);
        }

        PyJsonUtf8Tests.Same(e.UserAttrsDict, a.UserAttrsDict).Should().BeTrue("trial user attrs: {0}", because);
        PyJsonUtf8Tests.Same(e.SystemAttrsDict, a.SystemAttrsDict).Should().BeTrue("trial system attrs: {0}", because);
        a.IntermediateValues.Keys.Should().Equal(e.IntermediateValues.Keys, because);
        a.IntermediateValues.Values.Select(BitConverter.DoubleToInt64Bits)
            .Should().Equal(e.IntermediateValues.Values.Select(BitConverter.DoubleToInt64Bits), because);
    }

    // ------------------------------------------------------------------ round trips

    /// <summary>
    ///     For random journals, restoring the v2 snapshot gives the original state and exactly what restoring the v1
    ///     snapshot gives, and the v2 bytes are the same after a round trip. Seeded, so a failure reproduces.
    /// </summary>
    /// <param name="seed">The generator's seed.</param>
    [TestMethod]
    [DataRow(1)]
    [DataRow(2)]
    [DataRow(3)]
    [DataRow(4)]
    [DataRow(5)]
    [DataRow(6)]
    [DataRow(7)]
    [DataRow(8)]
    public void RandomJournals_V2RestoresWhatV1Restores(int seed)
    {
        JournalStorageReplayResult original = RandomState(seed, records: 600);
        byte[] v2 = original.Serialize();
        JournalStorageReplayResult.IsSnapshotV2(v2).Should().BeTrue("the storage now saves v2");

        JournalStorageReplayResult? fromV2 = JournalStorageReplayResult.TryDeserialize(v2);
        JournalStorageReplayResult? fromV1 = JournalStorageReplayResult.TryDeserialize(original.SerializeV1());
        fromV2.Should().NotBeNull();
        fromV1.Should().NotBeNull();

        AssertSameState(fromV1!, fromV2!, $"seed {seed}: v2 restore against v1 restore");
        AssertSameState(original, fromV2!, $"seed {seed}: v2 restore against the original");
        fromV2!.Serialize().Should().Equal(v2, "a restored state saves the same bytes");
    }

    /// <summary>A state with no study at all, and one with a study but no trial, round-trip too.</summary>
    [TestMethod]
    public void EmptyStates_RoundTrip()
    {
        var empty = new JournalStorageReplayResult("p-");
        AssertSameState(empty, JournalStorageReplayResult.TryDeserialize(empty.Serialize())!, "no study");

        var oneStudy = new JournalStorageReplayResult("p-");
        oneStudy.ApplyLogs(new[]
        {
            new OrderedDict<string, object?> { ["op_code"] = 0L, ["worker_id"] = OtherWorker, ["study_name"] = "s", ["directions"] = new List<object?> { 1L } },
        });
        AssertSameState(oneStudy, JournalStorageReplayResult.TryDeserialize(oneStudy.Serialize())!, "one study, no trial");
    }

    /// <summary>
    ///     A deleted study leaves a gap the next study's dictionary slot fills, so the studies enumerate out of creation
    ///     order; both formats keep that order, as v1 always did.
    /// </summary>
    [TestMethod]
    public void DeletedStudies_KeepTheirEnumerationOrder()
    {
        var state = new JournalStorageReplayResult("p-");
        OrderedDict<string, object?> Create(string name)
            => new() { ["op_code"] = 0L, ["worker_id"] = OtherWorker, ["study_name"] = name, ["directions"] = new List<object?> { 1L } };
        state.ApplyLogs(new[]
        {
            Create("a"), Create("b"), Create("c"),
            new OrderedDict<string, object?> { ["op_code"] = 1L, ["worker_id"] = OtherWorker, ["study_id"] = 1L },
            Create("d"),
        });

        JournalStorageReplayResult restored = JournalStorageReplayResult.TryDeserialize(state.Serialize())!;
        restored.GetAllStudies().Select(s => s.StudyName).Should().Equal(state.GetAllStudies().Select(s => s.StudyName));
        AssertSameState(state, restored, "after a deletion");
    }

    /// <summary>
    ///     Restored trials share one object per shareable distribution, as replaying the records does (the owner's
    ///     "Share one object"), and the records replayed after a restore get those same objects; a deprecated
    ///     distribution gets its own object in every trial.
    /// </summary>
    [TestMethod]
    public void RestoredDistributions_AreSharedAsReplayShares()
    {
        string floatJson = Distribution.ToJson(new FloatDistribution(0, 1));
        const string uniformJson = "{\"name\": \"UniformDistribution\", \"attributes\": {\"low\": 0.0, \"high\": 3.0}}";
        var logs = new List<OrderedDict<string, object?>>
        {
            new() { ["op_code"] = 0L, ["worker_id"] = OtherWorker, ["study_name"] = "s", ["directions"] = new List<object?> { 1L } },
        };
        for (int t = 0; t < 3; t++)
        {
            logs.Add(new()
            {
                ["op_code"] = 4L, ["worker_id"] = OtherWorker, ["study_id"] = 0L, ["state"] = 1L, ["value"] = 1.0,
                ["datetime_start"] = "2026-11-07T00:00:00.000000+00:00", ["datetime_complete"] = "2026-11-07T00:00:01.000000+00:00",
                ["distributions"] = new OrderedDict<string, object?> { ["x"] = floatJson, ["u"] = uniformJson },
                ["params"] = new OrderedDict<string, object?> { ["x"] = 0.5, ["u"] = 1.5 },
            });
        }
        var state = new JournalStorageReplayResult("p-");
        state.ApplyLogs(logs);
        state.GetTrial(0).DistributionsDict["x"].Should().BeSameAs(state.GetTrial(2).DistributionsDict["x"], "replay shares it");
        state.GetTrial(0).DistributionsDict["u"].Should().NotBeSameAs(state.GetTrial(2).DistributionsDict["u"], "a deprecated type is never shared");

        JournalStorageReplayResult restored = JournalStorageReplayResult.TryDeserialize(state.Serialize())!;
        restored.GetTrial(0).DistributionsDict["x"].Should().BeSameAs(restored.GetTrial(2).DistributionsDict["x"]);
        restored.GetTrial(0).DistributionsDict["u"].Should().NotBeSameAs(restored.GetTrial(2).DistributionsDict["u"]);

        restored.ApplyLogs(new[] { logs[1] });
        restored.GetTrial(3).DistributionsDict["x"].Should().BeSameAs(restored.GetTrial(0).DistributionsDict["x"],
            "a record replayed after the restore gets the restored object");
    }

    /// <summary>
    ///     The distribution lookup keeps at most <see cref="JournalDistributionCache.Capacity" /> texts: past that, a new
    ///     text is parsed per use (a new, equal object each time), and the texts kept before still share theirs.
    /// </summary>
    [TestMethod]
    public void DistributionCache_StopsSharingPastItsCapacity_ButStaysEqual()
    {
        var cache = new JournalDistributionCache();
        string First(int i) => Distribution.ToJson(new FloatDistribution(0, 1 + i));
        for (int i = 0; i < JournalDistributionCache.Capacity; i++)
            cache.FromJson(First(i));

        cache.FromJson(First(0)).Should().BeSameAs(cache.FromJson(First(0)), "a text kept before the limit stays shared");
        string past = Distribution.ToJson(new IntDistribution(0, 7));
        BaseDistribution one = cache.FromJson(past);
        BaseDistribution two = cache.FromJson(past);
        two.Should().NotBeSameAs(one, "a text past the limit is parsed per use");
        two.Should().Be(one);
    }

    /// <summary>
    ///     A v1 snapshot, as every older OptunaSharp version saves it, still restores, through the same dispatch, to the
    ///     state it was taken from.
    /// </summary>
    /// <param name="seed">The generator's seed.</param>
    [TestMethod]
    [DataRow(21)]
    [DataRow(22)]
    public void V1Snapshots_StillRestore(int seed)
    {
        JournalStorageReplayResult original = RandomState(seed, records: 400);
        byte[] v1 = original.SerializeV1();
        v1[0].Should().Be((byte)'{');
        JournalStorageReplayResult.IsSnapshotV2(v1).Should().BeFalse();
        AssertSameState(original, JournalStorageReplayResult.TryDeserialize(v1)!, $"seed {seed}: v1 restore");
    }

    // ------------------------------------------------------------------ corrupt bytes

    /// <summary>
    ///     Every truncation of a v2 snapshot restores nothing (the storage then replays the journal) and never throws;
    ///     so does a wrong version, nonzero flags, or a byte appended at the end.
    /// </summary>
    [TestMethod]
    public void TruncatedOrForeignV2Bytes_RestoreNothing()
    {
        byte[] v2 = RandomState(31, records: 120).Serialize();
        for (int length = 0; length < v2.Length; length++)
            JournalStorageReplayResult.TryDeserialize(v2.AsSpan(0, length).ToArray()).Should().BeNull("{0} of {1} bytes", length, v2.Length);

        byte[] wrongVersion = (byte[])v2.Clone();
        wrongVersion[4] = 3;
        JournalStorageReplayResult.TryDeserialize(wrongVersion).Should().BeNull();
        byte[] flags = (byte[])v2.Clone();
        flags[6] = 1;
        JournalStorageReplayResult.TryDeserialize(flags).Should().BeNull();
        JournalStorageReplayResult.TryDeserialize(v2.Concat(new byte[] { 0 }).ToArray()).Should().BeNull();
    }

    /// <summary>
    ///     Random byte changes in a v2 snapshot either restore a state, when they only hit a value's payload, or restore
    ///     nothing: a restore never throws and never asks for an absurd allocation.
    /// </summary>
    /// <param name="seed">The generator's seed.</param>
    [TestMethod]
    [DataRow(41)]
    [DataRow(42)]
    public void CorruptV2Bytes_NeverThrow(int seed)
    {
        byte[] v2 = RandomState(seed, records: 150).Serialize();
        var rng = new Random(seed);
        for (int i = 0; i < 3000; i++)
        {
            byte[] corrupt = (byte[])v2.Clone();
            int edits = 1 + rng.Next(4);
            for (int e = 0; e < edits; e++)
            {
                int at = 8 + rng.Next(corrupt.Length - 8);
                corrupt[at] = rng.Next(3) switch { 0 => 0xFF, 1 => 0x80, _ => (byte)rng.Next(256) };
            }
            var act = () => JournalStorageReplayResult.TryDeserialize(corrupt);
            act.Should().NotThrow();
        }
    }

    // ------------------------------------------------------------------ values a record cannot carry

    /// <summary>
    ///     Attribute values that never come from a journal record (an <see cref="int" />, an enum, a decimal, a float, a
    ///     plain dictionary) restore from v2 as v1 restores them: through JSON, to PyJson's object model.
    /// </summary>
    [TestMethod]
    public void NonRecordValues_RestoreAsV1RestoresThem()
    {
        JournalStorageReplayResult state = RandomState(51, records: 50);
        FrozenStudy study = state.GetAllStudies().First();
        study.UserAttrs = new OrderedDict<string, object?>
        {
            ["int"] = 7,
            ["enum"] = TrialState.Complete,
            ["decimal"] = 0.1m,
            ["float"] = 1.5f,
            ["dict"] = new Dictionary<string, object?> { ["a"] = 1, ["b"] = new[] { 1, 2 } },
            ["choice"] = CategoricalChoice.From("c"),
            ["nan"] = double.NaN,
        };

        JournalStorageReplayResult fromV2 = JournalStorageReplayResult.TryDeserialize(state.Serialize())!;
        JournalStorageReplayResult fromV1 = JournalStorageReplayResult.TryDeserialize(state.SerializeV1())!;
        AssertSameState(fromV1, fromV2, "normalized values");
        fromV2.GetAllStudies().First().UserAttrs["int"].Should().Be(7L);
    }

    /// <summary>
    ///     A value neither format can write keeps v1's outcome: the v2 writer gives up, the v1 fallback raises PyJson's
    ///     <see cref="TypeError" />, exactly as saving did before v2.
    /// </summary>
    [TestMethod]
    public void UnwritableValues_KeepV1sException()
    {
        JournalStorageReplayResult state = RandomState(52, records: 50);
        state.GetAllStudies().First().UserAttrs = new OrderedDict<string, object?> { ["bad"] = new object() };

        var v2 = () => state.Serialize();
        var v1 = () => state.SerializeV1();
        string expected = v1.Should().Throw<TypeError>().Which.Message;
        v2.Should().Throw<TypeError>().WithMessage(expected);
    }

    /// <summary>
    ///     An attribute nested to PyJson's depth limit, the deepest a record can carry, saves and restores in v2, as it
    ///     parsed from its record.
    /// </summary>
    [TestMethod]
    public void AttributesAtTheDepthLimit_RoundTrip()
    {
        object? deep = 1L;
        for (int i = 0; i < PyJson.MaxDepth; i++)
            deep = new List<object?> { deep };
        JournalStorageReplayResult state = RandomState(53, records: 20);
        state.GetAllStudies().First().UserAttrs = new OrderedDict<string, object?> { ["deep"] = deep };

        byte[] v2 = state.Serialize();
        JournalStorageReplayResult.IsSnapshotV2(v2).Should().BeTrue();
        JournalStorageReplayResult restored = JournalStorageReplayResult.TryDeserialize(v2)!;
        PyJsonUtf8Tests.Same(deep, restored.GetAllStudies().First().UserAttrs["deep"]).Should().BeTrue();
    }

    // ------------------------------------------------------------------ through the storage

    /// <summary>
    ///     Through <see cref="JournalStorage" />: a storage that saves v2 snapshots every few trials, and a new storage
    ///     that restores the latest one and replays only the records after it, see the same trials as a storage that
    ///     replays the whole journal.
    /// </summary>
    [TestMethod]
    public void JournalStorage_RestoresV2_AndSeesWhatAFullReplaySees()
    {
        var backend = new SnapshotMemoryJournal();
        var writer = new JournalStorage(backend, snapshotInterval: 7);
        // Two objectives (Optuna refuses Report there), then one objective that reports intermediate values.
        var multi = Optuna.CreateStudy(storage: writer, sampler: new global::OptunaSharp.Samplers.RandomSampler(seed: 3),
            directions: new[] { StudyDirection.Minimize, StudyDirection.Maximize });
        multi.Optimize(t =>
        {
            t.SetUserAttr("u", new List<object?> { t.Number, "é" });
            double x = t.SuggestFloat("x", -2, 2);
            long k = t.SuggestInt("k", 1, 100, log: true);
            object? c = t.SuggestCategorical("c", new object?[] { "a", null, 2.5 }.Select(CategoricalChoice.From)).Value;
            return new[] { x * x + k, c is null ? double.NaN : x };
        }, nTrials: 40);
        var single = Optuna.CreateStudy(storage: writer, sampler: new global::OptunaSharp.Samplers.RandomSampler(seed: 4));
        single.Optimize(t =>
        {
            double x = t.SuggestFloat("x", 1e-3, 1, log: true);
            for (int step = 0; step < 3; step++)
                t.Report(x * step, step: step);
            return x;
        }, nTrials: 25);

        backend.Snapshot.Should().NotBeNull();
        JournalStorageReplayResult.IsSnapshotV2(backend.Snapshot!).Should().BeTrue();

        var restored = new JournalStorage(backend);
        var replayedOnly = new JournalStorage(new SnapshotMemoryJournal(backend.Records));
        foreach (Study study in new[] { multi, single })
        {
            var a = restored.GetAllTrials(restored.GetStudyIdFromName(study.StudyName));
            var b = replayedOnly.GetAllTrials(replayedOnly.GetStudyIdFromName(study.StudyName));
            a.Should().HaveCount(b.Count).And.HaveCount(study.Trials.Count);
            for (int i = 0; i < a.Count; i++)
                AssertSameTrial(b[i], a[i], $"{study.StudyName} trial {i} through the storage");
        }
    }

    /// <summary>An in-memory journal backend with a snapshot slot, keeping records as JSON text as Redis does.</summary>
    private sealed class SnapshotMemoryJournal : BaseJournalBackend, IBaseJournalSnapshot
    {
        /// <summary>Starts empty.</summary>
        internal SnapshotMemoryJournal()
        {
        }

        /// <summary>Starts with another journal's records and no snapshot, for a replay of everything.</summary>
        /// <param name="records">The records' JSON text.</param>
        internal SnapshotMemoryJournal(List<string> records)
        {
            Records.AddRange(records);
        }

        /// <summary>The records' JSON text, in log order.</summary>
        internal List<string> Records { get; } = new();

        /// <summary>The latest snapshot, or <c>null</c>.</summary>
        internal byte[]? Snapshot { get; private set; }

        /// <summary>Returns the records from <paramref name="logNumberFrom" /> on, parsed as a backend parses them.</summary>
        /// <param name="logNumberFrom">The first log number.</param>
        /// <returns>The records.</returns>
        public override IEnumerable<OrderedDict<string, object?>> ReadLogs(int logNumberFrom)
            => Records.Skip(logNumberFrom).Select(r => (OrderedDict<string, object?>)PyJsonUtf8.Loads(Encoding.UTF8.GetBytes(r))!).ToList();

        /// <summary>Appends records as JSON text.</summary>
        /// <param name="logs">The records.</param>
        public override void AppendLogs(IReadOnlyList<OrderedDict<string, object?>> logs)
            => Records.AddRange(logs.Select(l => PyJson.Dumps(l)));

        /// <summary>Keeps the latest snapshot.</summary>
        /// <param name="snapshot">The snapshot.</param>
        public void SaveSnapshot(byte[] snapshot) => Snapshot = snapshot;

        /// <summary>Returns the latest snapshot.</summary>
        /// <returns>The snapshot, or <c>null</c>.</returns>
        public byte[]? LoadSnapshot() => Snapshot;
    }
}

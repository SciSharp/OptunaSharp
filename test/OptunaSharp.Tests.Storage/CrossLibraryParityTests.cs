using System.Collections.Generic;
using System.Linq;
using AwesomeAssertions;
using OptunaSharp;
using OptunaSharp.Collections;
using OptunaSharp.Samplers;
using OptunaSharp.Storages.Journal;
using StackExchange.Redis;

namespace OptunaSharp.Tests.Storage;

/// <summary>
///     The capstone: <b>1-to-1 parity across all three client libraries over the same Redis</b>. A
///     study/log written by any of StackExchange / NRedisStack / ServiceStack must be readable by every
///     one of them (all 9 writer×reader combinations), and the stored bytes must be identical no matter
///     which library wrote them. This is what proves the three backends are genuinely interchangeable
///     (and, with the interop suite's Python↔C# test, byte-compatible with Python Optuna too).
/// </summary>
[TestClass]
[TestCategory("Integration")]
public class CrossLibraryParityTests
{
    // A varied sample exercising every JSON leaf type, so byte-equality is a real assertion.
    private static readonly OrderedDict<string, object?>[] Sample =
    {
        new() { ["op"] = 0L, ["worker"] = "w-a", ["x"] = 1.5, ["flag"] = true, ["n"] = null, ["items"] = new List<object?> { 1L, 2L } },
        new() { ["op"] = 4L, ["worker"] = "w-b", ["nested"] = new OrderedDict<string, object?> { ["k"] = "v" } },
        new() { ["op"] = 6L, ["worker"] = "w-c" },
    };

    /// <summary>Every library writes the SAME logs → the stored counter + log bytes are byte-identical.</summary>
    [TestMethod]
    [TestCategory("Integration")]
    public void AllLibraries_ProduceByteIdenticalStoredLogs()
    {
        RedisSession.EnsureReachable();
        var db = RedisSession.RawConnection().GetDatabase();

        string? refCounter = null;
        string?[]? refLogs = null;
        string? refWriter = null;

        foreach (var (name, create) in RedisSession.Backends)
        {
            string prefix = RedisSession.UniquePrefix();
            create(prefix, false).AppendLogs(Sample);

            string? counter = db.StringGet(RedisJournalProtocol.LogNumberKey(prefix));
            string?[] logs = Enumerable.Range(0, Sample.Length)
                .Select(i => (string?)db.StringGet(RedisJournalProtocol.LogKey(prefix, i)))
                .ToArray();

            if (refLogs is null)
            {
                refLogs = logs;
                refCounter = counter;
                refWriter = name;
            }
            else
            {
                counter.Should().Be(refCounter, "{0} counter bytes must match {1}", name, refWriter);
                logs.Should().Equal(refLogs, "{0} stored log bytes must be identical to {1}", name, refWriter);
            }
        }
    }

    /// <summary>Every (writer, reader) pair — all 9 combinations — round-trips the logs.</summary>
    [TestMethod]
    [TestCategory("Integration")]
    public void EveryWriter_IsReadableByEveryReader()
    {
        RedisSession.EnsureReachable();
        foreach (var (writerName, writerCreate) in RedisSession.Backends)
        {
            string prefix = RedisSession.UniquePrefix();
            writerCreate(prefix, false).AppendLogs(Sample);

            foreach (var (readerName, readerCreate) in RedisSession.Backends)
            {
                var got = readerCreate(prefix, false).ReadLogs(0).ToList();
                got.Should().HaveCount(Sample.Length, "{0} must read {1}'s logs", readerName, writerName);
                for (int i = 0; i < Sample.Length; i++)
                    got[i]["op"].Should().Be(Sample[i]["op"], "{0} reading {1}, log {2}", readerName, writerName, i);
                got[^1]["worker"].Should().Be("w-c");
            }
        }
    }

    /// <summary>The full "read-write-read": a Study written by each library replays through each library.</summary>
    [TestMethod]
    [TestCategory("Integration")]
    public void Study_WrittenByEachLibrary_ReplaysThroughEachLibrary()
    {
        RedisSession.EnsureReachable();
        foreach (var (writerName, writerCreate) in RedisSession.Backends)
        {
            string prefix = RedisSession.UniquePrefix();
            var writeStudy = Optuna.CreateStudy(
                storage: new JournalStorage(writerCreate(prefix, false)),
                studyName: "xlib", sampler: new RandomSampler(seed: 11), direction: StudyDirection.Minimize);
            writeStudy.Optimize(t => { double x = t.SuggestFloat("x", -5, 5); return x * x; }, nTrials: 8);
            int count = writeStudy.GetTrials().Count;
            double best = writeStudy.BestValue;

            foreach (var (readerName, readerCreate) in RedisSession.Backends)
            {
                var readStudy = Optuna.CreateStudy(
                    storage: new JournalStorage(readerCreate(prefix, false)),
                    studyName: "xlib", loadIfExists: true);
                readStudy.GetTrials().Count.Should().Be(count, "{0} replays {1}'s study", readerName, writerName);
                readStudy.BestValue.Should().Be(best, "{0} best == {1} best", readerName, writerName);
            }
        }
    }
}

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using AwesomeAssertions;
using OptunaSharp.Collections;
using OptunaSharp.Internal;
using OptunaSharp.Storages.Journal;

namespace OptunaSharp.Tests.Storages;

/// <summary>
///     Plan 14: <see cref="JournalFileBackend" /> keeps the start offset of the furthest line read and one checkpoint every
///     <see cref="JournalFileBackend.OffsetCheckpointInterval" /> lines instead of Python's offset per line. A read must
///     still yield exactly the records of a full read from the file's start, from any position, on a backend that has read
///     nothing, everything, or part of the file, while other readers share it; and an undecodable line must still never
///     become a place a later read starts after.
/// </summary>
[TestClass]
public class JournalOffsetCheckpointTests
{
    private string _path = null!;

    /// <summary>Creates a fresh journal path in the temporary folder.</summary>
    [TestInitialize]
    public void SetUp()
        => _path = Path.Combine(Path.GetTempPath(), $"optunasharp-offsets-{Guid.NewGuid():N}.log");

    /// <summary>Deletes the journal and its lock files.</summary>
    [TestCleanup]
    public void TearDown()
    {
        foreach (var f in Directory.GetFiles(Path.GetDirectoryName(_path)!, Path.GetFileName(_path) + "*"))
        {
            try
            {
                File.Delete(f);
            }
            catch (IOException)
            {
                // Best effort: a lock file another test process still holds is cleaned up by the next run.
            }
        }
    }

    /// <summary>
    ///     Appends <paramref name="count" /> records of varied lengths (so line starts are not a regular grid), numbered from
    ///     <paramref name="first" />, in batches like a study's writes.
    /// </summary>
    /// <param name="backend">The backend to append through.</param>
    /// <param name="first">The number of the first record.</param>
    /// <param name="count">How many records to append.</param>
    private static void Append(JournalFileBackend backend, int first, int count)
    {
        var batch = new List<OrderedDict<string, object?>>();
        for (int i = first; i < first + count; i++)
        {
            batch.Add(new OrderedDict<string, object?>
            {
                ["op_code"] = 5L,
                ["i"] = (long)i,
                ["pad"] = new string('x', (i * 7919) % 97),
                ["v"] = i * 0.1,
            });
            if (batch.Count == 37)
            {
                backend.AppendLogs(batch);
                batch.Clear();
            }
        }
        if (batch.Count > 0)
            backend.AppendLogs(batch);
    }

    /// <summary>Every record a read yields, as JSON text, to compare reads by value.</summary>
    /// <param name="backend">The backend to read through.</param>
    /// <param name="from">The first log number to read.</param>
    /// <returns>The records' JSON, in order.</returns>
    private static List<string> Read(JournalFileBackend backend, int from)
        => backend.ReadLogs(from).Select(log => PyJson.Dumps(log)).ToList();

    /// <summary>
    ///     A read from any position equals the matching slice of a full read, on a backend that has read nothing (cold) and
    ///     on one that has read the whole file (warm); and the warm backend kept only the sparse offsets.
    /// </summary>
    [TestMethod]
    public void ReadLogs_FromAnyPosition_EqualsTheSliceOfAFullRead_ColdAndWarm()
    {
        int interval = JournalFileBackend.OffsetCheckpointInterval;
        int total = 2 * interval + 500;
        var writer = new JournalFileBackend(_path, new JournalFileOpenLock(_path));
        Append(writer, 0, total);
        List<string> reference = Read(new JournalFileBackend(_path, new JournalFileOpenLock(_path)), 0);
        reference.Should().HaveCount(total);

        var warm = new JournalFileBackend(_path, new JournalFileOpenLock(_path));
        Read(warm, 0).Should().Equal(reference);
        warm.OffsetCheckpointCount.Should().Be(3, "checkpoints at 0, 4096 and 8192 only");

        int[] froms = { 0, 1, 2, interval - 1, interval, interval + 1, 5000, 2 * interval - 1, 2 * interval, 2 * interval + 1, total - 1, total, total + 5 };
        foreach (int from in froms)
        {
            List<string> expected = from >= total ? new List<string>() : reference.GetRange(from, total - from);
            Read(warm, from).Should().Equal(expected, $"warm read from {from}");
            Read(new JournalFileBackend(_path, new JournalFileOpenLock(_path)), from).Should().Equal(expected, $"cold read from {from}");
        }
    }

    /// <summary>
    ///     Two storages sharing one backend: one read stops half way, the file grows, another read reaches the new end, and a
    ///     read resuming from the first one's position (behind the furthest) starts at a checkpoint and yields exactly the
    ///     records after it.
    /// </summary>
    [TestMethod]
    public void SharedBackend_AReaderBehindTheFurthest_ResumesExactly()
    {
        int interval = JournalFileBackend.OffsetCheckpointInterval;
        var backend = new JournalFileBackend(_path, new JournalFileOpenLock(_path));
        Append(backend, 0, interval + 2000);
        // A reader that stops after 5,000 records, as a sync of a storage that was behind would.
        var firstPart = backend.ReadLogs(0).Take(5000).Select(log => PyJson.Dumps(log)).ToList();
        firstPart.Should().HaveCount(5000);

        Append(backend, interval + 2000, interval + 300);
        int total = 2 * interval + 2300;
        List<string> reference = Read(new JournalFileBackend(_path, new JournalFileOpenLock(_path)), 0);
        reference.Should().HaveCount(total);
        Read(backend, 0).Should().Equal(reference, "a full read through the shared backend");

        Read(backend, 5000).Should().Equal(reference.GetRange(5000, total - 5000), "the behind reader resumes at its own position");
        firstPart.Should().Equal(reference.GetRange(0, 5000));
    }

    /// <summary>
    ///     Concurrent readers of one backend, from scattered positions, while the file is static: every read equals its
    ///     slice of the reference, and the bookkeeping stays consistent (one checkpoint per interval).
    /// </summary>
    [TestMethod]
    public void ConcurrentReaders_FromScatteredPositions_AllReadExactly()
    {
        int interval = JournalFileBackend.OffsetCheckpointInterval;
        int total = 3 * interval + 77;
        Append(new JournalFileBackend(_path, new JournalFileOpenLock(_path)), 0, total);
        List<string> reference = Read(new JournalFileBackend(_path, new JournalFileOpenLock(_path)), 0);

        var shared = new JournalFileBackend(_path, new JournalFileOpenLock(_path));
        var froms = new[] { 0, 9000, 4096, 100, 12000, 8191, 3, 6000, 12364, 4097, 2, 11111 };
        Parallel.ForEach(froms, new ParallelOptions { MaxDegreeOfParallelism = 6 }, from =>
        {
            List<string> expected = from >= total ? new List<string>() : reference.GetRange(from, total - from);
            Read(shared, from).Should().Equal(expected, $"read from {from}");
        });
        shared.OffsetCheckpointCount.Should().Be(4, "checkpoints at 0, 4096, 8192 and 12288");
    }

    /// <summary>
    ///     A record that does not decode, followed by another record, fails the read with Python's error; its successor
    ///     never becomes a start position, so a later read from it, or from anywhere before it, fails the same way instead of
    ///     skipping it.
    /// </summary>
    [TestMethod]
    public void UndecodableRecord_FollowedByAnother_FailsEveryReadThatReachesIt()
    {
        int interval = JournalFileBackend.OffsetCheckpointInterval;
        var backend = new JournalFileBackend(_path, new JournalFileOpenLock(_path));
        Append(backend, 0, interval + 10);
        File.AppendAllText(_path, "{\"op_code\": 5, broken\n", Encoding.UTF8);
        Append(backend, interval + 11, 20);

        Action full = () => Read(backend, 0);
        full.Should().Throw<NumSharp.ValueError>();
        Action fromBefore = () => Read(backend, interval + 5);
        fromBefore.Should().Throw<NumSharp.ValueError>("the read must reach the bad record again, not start after it");
        Action fromIt = () => Read(backend, interval + 10);
        fromIt.Should().Throw<NumSharp.ValueError>();
        // Stops after one record, before the bad one: the record before it still reads.
        backend.ReadLogs(interval + 9).Take(1).Count().Should().Be(1);
    }

    /// <summary>
    ///     A last line still being written (no newline yet) is left for the next read, as in Python; once completed, the next
    ///     read from the furthest position yields it.
    /// </summary>
    [TestMethod]
    public void PartialLastLine_IsLeftForTheNextRead()
    {
        var backend = new JournalFileBackend(_path, new JournalFileOpenLock(_path));
        Append(backend, 0, 10);
        File.AppendAllText(_path, "{\"op_code\": 5, \"i\": 10", Encoding.UTF8);
        Read(backend, 0).Should().HaveCount(10);
        File.AppendAllText(_path, "}\n", Encoding.UTF8);
        var rest = Read(backend, 10);
        rest.Should().ContainSingle().Which.Should().Contain("\"i\": 10");
    }
}

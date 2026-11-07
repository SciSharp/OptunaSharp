using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using AwesomeAssertions;
using OptunaSharp.Collections;
using OptunaSharp.Internal;
using OptunaSharp.Storages.Journal;
using Python.Runtime;

namespace OptunaSharp.Tests.Interop;

/// <summary>
///     Plan 10, item 2.7 against live Optuna: journals Python writes, with every value kind JSON carries, parse through
///     OptunaSharp's UTF-8 record reader exactly as <see cref="PyJson" /> parses them, and replay, directly or through a
///     v1 or binary v2 snapshot, into the study Python itself sees.
/// </summary>
/// <remarks>
///     <para>
///         The owner's condition for the item was very high parity coverage. Python writes the records here, in both
///         forms its backends use: the file journal's compact separators and the Redis journal's default ones. The
///         values cover NaN, both infinities, -0.0, subnormals, the largest and smallest doubles, integers past 64 bits,
///         unicode escapes, surrogate pairs, a lone surrogate, control characters and nested lists and dictionaries, in
///         study and trial attributes, objective values and intermediate values at steps past 32 bits. The trials cover
///         every state, a single and a multi-objective study, and every distribution type Optuna 5 writes.
///     </para>
///     <para>
///         Each record must also take the reader's fast path (not the fallback to PyJson), so the parity holds for the
///         code that actually runs.
///     </para>
/// </remarks>
[TestClass]
public class JournalCodecParityInteropTests : OptunaInteropTestBase
{
    /// <summary>
    ///     Python source of <c>_oss_write_every_kind(storage)</c>: two studies over <c>storage</c> holding every value
    ///     kind, every trial state and every distribution type.
    /// </summary>
    private const string WriteEveryKindSource = """
        def _oss_values():
            return {
                'none': None, 'true': True, 'false': False,
                'zero': 0, 'negative': -7, 'int64_max': 2**63 - 1, 'past_int64': 2**70,
                'tenth': 0.1, 'negative_zero': -0.0, 'nan': float('nan'),
                'inf': float('inf'), 'negative_inf': float('-inf'),
                'subnormal': 5e-324, 'smallest_normal': 2.2250738585072014e-308, 'largest': 1.7976931348623157e308,
                'text': 'plain', 'latin': 'café', 'euro': '€', 'emoji': '\U0001F600', 'lone': '\ud800x',
                'controls': '\x00\x01\x1f\n\t"\\/',
                'list': [1, 2.5, 'x', None, [True, {'k': []}]],
                'dict': {'nested': {'deeper': [1, {'deepest': 'v'}]}, 'empty': {}},
            }

        def _oss_write_every_kind(storage):
            mo = optuna.create_study(study_name='kinds-mo', storage=storage, directions=['minimize', 'maximize'])
            for key, value in _oss_values().items():
                mo.set_user_attr(key, value)
            mo._storage.set_study_system_attr(mo._study_id, 'sys', _oss_values())
            outcomes = [[0.1, -0.0], [float('inf'), 5e-324], None, [1.7976931348623157e308, -2.5], None]
            for i, outcome in enumerate(outcomes):
                t = mo.ask()
                t.suggest_float('x', -1, 1)
                t.suggest_int('k', 1, 64, log=True)
                t.suggest_categorical('c', [None, True, 1, 2.5, 'a', 'café\U0001F600'])
                t.suggest_float('s', 0, 1, step=0.25)
                t.suggest_float('l', 1e-5, 1, log=True)
                t.suggest_int('n', -3, 30, step=3)
                t.set_user_attr('values', _oss_values())
                mo._storage.set_trial_system_attr(t._trial_id, 'sys', [-1.0, 0.0, 2.5, float('nan')])
                if outcome is not None:
                    mo.tell(t, outcome)
                elif i == 2:
                    mo.tell(t, state=optuna.trial.TrialState.FAIL)
                # i == 4 stays RUNNING
            mo.enqueue_trial({'x': 0.5, 'k': 3})

            so = optuna.create_study(study_name='kinds-so', storage=storage)
            so.set_user_attr('nested', _oss_values())
            for i in range(5):
                t = so.ask()
                x = t.suggest_float('x', 0, 1)
                for step in (0, 2**40, 2**62):
                    t.report(x * step if step else float('-inf'), step)
                if i == 0:
                    so.tell(t, state=optuna.trial.TrialState.PRUNED)
                elif i == 1:
                    so.tell(t, state=optuna.trial.TrialState.FAIL)
                elif i < 4:
                    so.tell(t, [float('-inf'), 5e-324][i - 2])
            so.enqueue_trial({'x': 0.25})
            return mo, so

        def _oss_redis_form(lines):
            # The Redis journal writes json.dumps(log) with Python's default separators.
            return [json.dumps(json.loads(line)) for line in lines]
        """;

    private string _path = null!;

    /// <summary>A fresh journal file path for each case.</summary>
    [TestInitialize]
    public void JournalInit()
        => _path = Path.Combine(Path.GetTempPath(), $"optunasharp-codec-{Guid.NewGuid():N}.log").Replace('\\', '/');

    /// <summary>Deletes the case's journal and its lock files.</summary>
    [TestCleanup]
    public void JournalCleanup()
    {
        foreach (string f in Directory.GetFiles(Path.GetDirectoryName(_path)!, Path.GetFileName(_path) + "*"))
        {
            try
            {
                File.Delete(f);
            }
            catch (IOException)
            {
                // A lock file still held by a finalizing handle; the temp folder keeps it.
            }
        }
    }

    /// <summary>
    ///     Python writes both studies into a journal file; returns its lines (file form, then Redis form) and Python's
    ///     own view of both studies.
    /// </summary>
    /// <returns>The compact lines, the default-separator lines, and Python's states of the two studies.</returns>
    private (List<byte[]> File, List<byte[]> Redis, StudySnapshot Mo, StudySnapshot So) PythonWritesEveryKind()
    {
        PyExec(WriteEveryKindSource);
        PyExec($"mo, so = _oss_write_every_kind({PyJournalStorageExpr(_path)})\n" +
               $"_oss_lines = open(r'{_path}', encoding='utf-8').read().splitlines()\n" +
               "_oss_redis_lines = _oss_redis_form(_oss_lines)\n");
        StudySnapshot mo = PyStudyState("mo");
        StudySnapshot so = PyStudyState("so");

        List<byte[]> fileLines = File.ReadAllText(_path, Encoding.UTF8)
            .Split('\n', StringSplitOptions.RemoveEmptyEntries)
            .Select(Encoding.UTF8.GetBytes)
            .ToList();
        // The Redis-form lines cross as one JSON array of strings; Python's lines are ASCII, so nothing changes on the way.
        List<byte[]> redisLines = ((List<object?>)PyJson.Loads(PyStr("json.dumps(_oss_redis_lines)"))!)
            .Select(line => Encoding.UTF8.GetBytes((string)line!))
            .ToList();
        return (fileLines, redisLines, mo, so);
    }

    /// <summary>
    ///     Every record Python writes, in the file journal's form and the Redis journal's, parses through the UTF-8
    ///     reader's fast path to exactly <see cref="PyJson" />'s value.
    /// </summary>
    [TestMethod]
    [TestCategory("RequiresPython")]
    public void PythonRecordsOfEveryKind_ParseExactlyAsPyJson()
    {
        var (fileLines, redisLines, _, _) = PythonWritesEveryKind();
        fileLines.Should().HaveCountGreaterThan(40, "both studies wrote their records");
        redisLines.Should().HaveCount(fileLines.Count);

        foreach (byte[] line in fileLines.Concat(redisLines))
        {
            object? expected = PyJson.Loads(Encoding.UTF8.GetString(line));
            PyJsonUtf8Reader.TryParse(line, out object? fast).Should().BeTrue("Python's record must take the fast path: {0}", Encoding.UTF8.GetString(line));
            Same(expected, fast).Should().BeTrue("the fast reader must give PyJson's value for {0}", Encoding.UTF8.GetString(line));
            Same(expected, PyJsonUtf8.Loads(line)).Should().BeTrue();
        }
    }

    /// <summary>
    ///     OptunaSharp replays the Python-written journal (file form) into the studies Python sees, field by field, with
    ///     the UTF-8 reader and the shared distributions in the path.
    /// </summary>
    [TestMethod]
    [TestCategory("RequiresPython")]
    public void PythonJournalOfEveryKind_ReplaysIntoPythonsStudies()
    {
        var (_, _, mo, so) = PythonWritesEveryKind();
        var storage = new JournalStorage(new JournalFileBackend(_path, lockObj: new JournalFileOpenLock(_path)));
        StudySnapshot.AssertParity(mo, StudySnapshot.Capture(Optuna.LoadStudy("kinds-mo", storage)), DatetimeComparison.Instant);
        StudySnapshot.AssertParity(so, StudySnapshot.Capture(Optuna.LoadStudy("kinds-so", storage)), DatetimeComparison.Instant);
    }

    /// <summary>
    ///     The Python-written journal (Redis form) replayed, saved as a snapshot (binary v2, and v1 for comparison) and
    ///     restored by a new storage gives the studies Python sees: the restore is as faithful as the replay.
    /// </summary>
    /// <param name="v1">Whether to snapshot in the older v1 JSON format instead of v2.</param>
    [TestMethod]
    [TestCategory("RequiresPython")]
    [DataRow(false)]
    [DataRow(true)]
    public void PythonJournalOfEveryKind_RestoresFromASnapshotIntoPythonsStudies(bool v1)
    {
        var (_, redisLines, mo, so) = PythonWritesEveryKind();
        var replay = new JournalStorageReplayResult("interop-");
        replay.ApplyLogs(redisLines.Select(l => (OrderedDict<string, object?>)PyJsonUtf8.Loads(l)!).ToList());
        byte[] snapshot = v1 ? replay.SerializeV1() : replay.Serialize();
        JournalStorageReplayResult.IsSnapshotV2(snapshot).Should().Be(!v1);

        // The storage restores the snapshot, which already covers every record, so it replays none of them.
        var backend = new SnapshotOnlyBackend(redisLines, snapshot);
        var storage = new JournalStorage(backend);
        backend.ReadsFrom.Should().OnlyContain(from => from == redisLines.Count, "the snapshot covered every record");
        StudySnapshot.AssertParity(mo, StudySnapshot.Capture(Optuna.LoadStudy("kinds-mo", storage)), DatetimeComparison.Instant);
        StudySnapshot.AssertParity(so, StudySnapshot.Capture(Optuna.LoadStudy("kinds-so", storage)), DatetimeComparison.Instant);
    }

    /// <summary>
    ///     Strict equality over PyJson's object model: same runtime types, same key order, doubles compared by their
    ///     bits.
    /// </summary>
    /// <param name="a">The first value.</param>
    /// <param name="b">The second value.</param>
    /// <returns>Whether the two are the same value in every observable respect but object identity.</returns>
    /// <exception cref="InvalidOperationException">A value outside PyJson's object model.</exception>
    private static bool Same(object? a, object? b)
    {
        switch (a)
        {
            case null:
                return b is null;
            case bool ab:
                return b is bool bb && ab == bb;
            case long al:
                return b is long bl && al == bl;
            case double ad:
                return b is double bd && BitConverter.DoubleToInt64Bits(ad) == BitConverter.DoubleToInt64Bits(bd);
            case string astr:
                return b is string bstr && string.Equals(astr, bstr, StringComparison.Ordinal);
            case List<object?> alist:
                return b is List<object?> blist && alist.Count == blist.Count && alist.Zip(blist).All(p => Same(p.First, p.Second));
            case OrderedDict<string, object?> adict:
                return b is OrderedDict<string, object?> bdict && adict.Count == bdict.Count
                       && adict.Zip(bdict).All(p => p.First.Key == p.Second.Key && Same(p.First.Value, p.Second.Value));
            default:
                throw new InvalidOperationException($"Not PyJson's object model: {a.GetType()}");
        }
    }

    /// <summary>
    ///     A journal backend over fixed records with a snapshot slot, recording where each read starts, so a test can
    ///     see that a restore skipped the records its snapshot covers.
    /// </summary>
    private sealed class SnapshotOnlyBackend : BaseJournalBackend, IBaseJournalSnapshot
    {
        private readonly List<byte[]> _records;
        private byte[]? _snapshot;

        /// <summary>Holds the records and the snapshot to hand out.</summary>
        /// <param name="records">The records' UTF-8 JSON.</param>
        /// <param name="snapshot">The snapshot.</param>
        internal SnapshotOnlyBackend(List<byte[]> records, byte[] snapshot)
        {
            _records = records;
            _snapshot = snapshot;
        }

        /// <summary>The first log number of every read, in order.</summary>
        internal List<int> ReadsFrom { get; } = new();

        /// <summary>Returns the records from <paramref name="logNumberFrom" /> on, through the UTF-8 reader.</summary>
        /// <param name="logNumberFrom">The first log number.</param>
        /// <returns>The records.</returns>
        public override IEnumerable<OrderedDict<string, object?>> ReadLogs(int logNumberFrom)
        {
            ReadsFrom.Add(logNumberFrom);
            return _records.Skip(logNumberFrom).Select(r => (OrderedDict<string, object?>)PyJsonUtf8.Loads(r)!).ToList();
        }

        /// <summary>Refuses writes: the cases only read.</summary>
        /// <param name="logs">The records.</param>
        /// <exception cref="NotSupportedException">Always.</exception>
        public override void AppendLogs(IReadOnlyList<OrderedDict<string, object?>> logs)
            => throw new NotSupportedException("This backend only replays a fixed journal.");

        /// <summary>Keeps a snapshot.</summary>
        /// <param name="snapshot">The snapshot.</param>
        public void SaveSnapshot(byte[] snapshot) => _snapshot = snapshot;

        /// <summary>Returns the snapshot.</summary>
        /// <returns>The snapshot.</returns>
        public byte[]? LoadSnapshot() => _snapshot;
    }
}

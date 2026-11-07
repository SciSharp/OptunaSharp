namespace OptunaSharp.Storages.Journal;

/// <summary>
///     The on-the-wire Redis journal protocol shared by every client backend: the key layout, the
///     append and read Lua scripts, and the snapshot keys and intervals. Sharing these constants (rather
///     than an implementation abstraction) is exactly what keeps the three independent client backends
///     — StackExchange / NRedisStack / ServiceStack — byte-identical against the same Redis, and
///     byte-compatible with Python Optuna's <c>JournalRedisBackend</c>.
/// </summary>
/// <remarks>
///     <para>
///         Everything Python Optuna reads or writes is unchanged: the counter and record keys, the record bytes and
///         the zero-based numbering. Two additions are OptunaSharp's own and invisible to Python workers on the same
///         journal (plan 10, item 2.2): <see cref="ReadLogsScript" />, which fetches the counter and a batch of records
///         in one round trip instead of one round trip each, and <see cref="OwnSnapshotKey" />, a snapshot key of its
///         own so the two languages stop overwriting each other's snapshot.
///     </para>
/// </remarks>
public static class RedisJournalProtocol
{
    /// <summary>The integer log-number counter key: <c>{prefix}:log_number</c>.</summary>
    /// <param name="prefix">The journal key namespace, shared by its counter, log records, and snapshot.</param>
    /// <returns>The namespaced Redis journal-counter key.</returns>
    public static string LogNumberKey(string prefix) => $"{prefix}:log_number";

    /// <summary>The key of log <paramref name="logNumber" />: <c>{prefix}:log:{n}</c>.</summary>
    /// <param name="prefix">The journal key namespace, shared by its counter, log records, and snapshot.</param>
    /// <param name="logNumber">The zero-based sequence number of the journal entry.</param>
    /// <returns>The namespaced Redis key for the requested journal record.</returns>
    public static string LogKey(string prefix, long logNumber) => $"{prefix}:log:{logNumber}";

    /// <summary>
    ///     Python Optuna's snapshot key, <c>{prefix}:snapshot</c>, where Python stores a pickle of its replay state.
    ///     OptunaSharp no longer writes it; it looks there only while its own key is empty, for a snapshot of its own
    ///     an older version saved there (see <see cref="OwnSnapshotKey" />).
    /// </summary>
    /// <param name="prefix">The journal key namespace, shared by its counter, log records, and snapshot.</param>
    /// <returns>The namespaced Redis snapshot key Python Optuna uses.</returns>
    public static string SnapshotKey(string prefix) => $"{prefix}:snapshot";

    /// <summary>
    ///     OptunaSharp's own snapshot key, <c>{prefix}:snapshot:optunasharp</c>, which holds its JSON replay state.
    /// </summary>
    /// <param name="prefix">The journal key namespace, shared by its counter, log records, and snapshots.</param>
    /// <returns>The namespaced Redis key of OptunaSharp's snapshot.</returns>
    /// <remarks>
    ///     Python pickles its replay state and OptunaSharp writes JSON. While both used <see cref="SnapshotKey" />, each
    ///     language's new workers on a mixed fleet found the other's bytes, logged
    ///     <c>Failed to restore `JournalStorageReplayResult`.</c>, and replayed the whole journal. With a key each, a
    ///     cold start of either language restores its own latest snapshot and reads only the records after it, and
    ///     Python's key is again Python's alone (the owner's decision, 2026-11-07).
    /// </remarks>
    public static string OwnSnapshotKey(string prefix) => $"{prefix}:snapshot:optunasharp";

    /// <summary>
    ///     Whether <paramref name="snapshot" /> can be one of OptunaSharp's own: its JSON starts with <c>{</c>, where a
    ///     Python pickle (protocol 2 or later) starts with the byte <c>0x80</c>.
    /// </summary>
    /// <param name="snapshot">Bytes read from a snapshot key.</param>
    /// <returns><c>true</c> when the bytes may be an OptunaSharp snapshot; the journal storage still checks its format.</returns>
    /// <remarks>
    ///     A backend applies this to the first byte of <see cref="SnapshotKey" /> (<c>GETRANGE 0 0</c>) before it reads
    ///     the whole value, and again to the whole value: a pickle there is Python's, and handing it on would only make
    ///     the storage log a failed restore before it replays the journal anyway.
    /// </remarks>
    public static bool IsOptunaSharpSnapshot(byte[] snapshot) => snapshot.Length > 0 && snapshot[0] == (byte)'{';

    /// <summary>
    ///     The snapshot interval the backends give a journal storage on a standalone server
    ///     (<c>IJournalSnapshotInterval.DefaultSnapshotInterval</c>): one OptunaSharp snapshot every 1000 study or trial
    ///     ids.
    /// </summary>
    /// <remarks>
    ///     On a standalone server, <see cref="ReadLogsScript" /> replays a thousand records per round trip, so a snapshot
    ///     saves a cold start little and costs its writer a serialization of the whole study. Measured on 2026-11-07 at
    ///     20,000 two-parameter trials: every trial written 6% slower than with no snapshot, against 48% at Optuna's
    ///     interval of 100, while a cold start took 1.22 s against 1.19 s at 100 and 1.30 s with no snapshot. The owner
    ///     chose 1000 over never snapshotting, so that a faster snapshot codec (plan 10, item 2.7) pays off without a
    ///     new default.
    /// </remarks>
    public const int StandaloneSnapshotInterval = 1000;

    /// <summary>
    ///     The snapshot interval the backends give a journal storage in cluster mode: Optuna's 100.
    /// </summary>
    /// <remarks>
    ///     In cluster mode a reader fetches every record with its own <c>GET</c>, as Python does, so the records after a
    ///     snapshot cost a round trip each. Measured on 2026-11-07 at 20,000 trials: a cold start took 1.17 s with a
    ///     snapshot every 100 ids, 4.03 s with one every 1000, and 53.8 s with none.
    /// </remarks>
    public const int ClusterSnapshotInterval = 100;

    /// <summary>The initial counter value written by SETNX before the first append (first INCR makes it 0).</summary>
    public const long InitialLogNumber = -1;

    /// <summary>
    ///     The append-log Lua script: Optuna's, with the counter's <c>SETNX</c> moved inside it. <c>numkeys = 0</c>, so
    ///     <c>ARGV[1]</c> = the prefix and <c>ARGV[2]</c> = <c>json.dumps(log)</c>; the script derives the counter and log
    ///     keys itself, which is why it is used only on a standalone server: a cluster cannot slot-route keys computed
    ///     inside a script.
    /// </summary>
    /// <remarks>
    ///     Python sends <c>SETNX {prefix}:log_number -1</c> as a separate command before every append, so the first
    ///     <c>INCR</c> numbers the first record 0. Doing it inside the script costs no round trip, and keeps the
    ///     numbering right even if the journal's keys are deleted while a backend is alive. Keys, numbering and record
    ///     bytes stay Python's, so Python workers on the same journal see no difference (plan 10, item 2.2). The
    ///     <c>-1</c> is <see cref="InitialLogNumber" />.
    /// </remarks>
    public const string AppendLogScript =
        "redis.call('setnx', string.format('%s:log_number', ARGV[1]), -1) " +
        "local i = redis.call('incr', string.format('%s:log_number', ARGV[1])) " +
        "redis.call('set', string.format('%s:log:%d', ARGV[1], i), ARGV[2])";

    /// <summary>
    ///     The most records one <see cref="ReadLogsScript" /> call returns. A reader catching up on a long journal
    ///     calls the script again for the next batch.
    /// </summary>
    /// <remarks>
    ///     Redis runs a script on its single command thread, so every other client of the server waits while it runs.
    ///     The cap keeps each call short: without it, a worker joining a journal of a million records with no usable
    ///     snapshot would read every key inside one script and build a reply of hundreds of megabytes. A worker that
    ///     is up to date reads only the records since its last sync, which is one call.
    /// </remarks>
    public const int ReadLogsBatchLimit = 1000;

    /// <summary>
    ///     A read-only Lua script that returns the counter and up to <see cref="ReadLogsBatchLimit" /> records in one
    ///     round trip, where Python sends one <c>GET</c> for the counter and one per record. <c>numkeys = 0</c>;
    ///     <c>ARGV[1]</c> = the prefix, <c>ARGV[2]</c> = the first log number to read, <c>ARGV[3]</c> = the most
    ///     records to return, <c>ARGV[4]</c> = the highest log number to read, or an empty string for the counter's
    ///     current value. Standalone servers only, like <see cref="AppendLogScript" />.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///         The reply is an empty array when the counter does not exist yet; otherwise the counter's value, then the
    ///         records in order. The script stops at the first record that does not exist: the counter can lead a
    ///         record only when a cluster-mode writer shares the prefix between its <c>INCR</c> and its <c>SET</c>, and
    ///         the backend then waits for that record as Python's reader does.
    ///     </para>
    ///     <para>
    ///         A reader passes the counter value of its first call as <c>ARGV[4]</c> of the following calls, so a read
    ///         ends where Python's would, at the counter value it saw first.
    ///     </para>
    /// </remarks>
    public const string ReadLogsScript =
        "local n = redis.call('get', string.format('%s:log_number', ARGV[1])) " +
        "if not n then return {} end " +
        "local first = tonumber(ARGV[2]) " +
        "local last = tonumber(n) " +
        "if ARGV[4] ~= '' then last = math.min(last, tonumber(ARGV[4])) end " +
        "last = math.min(last, first + tonumber(ARGV[3]) - 1) " +
        "local out = { n } " +
        "for i = first, last do " +
        "local log = redis.call('get', string.format('%s:log:%d', ARGV[1], i)) " +
        "if not log then break end " +
        "out[#out + 1] = log " +
        "end " +
        "return out";
}

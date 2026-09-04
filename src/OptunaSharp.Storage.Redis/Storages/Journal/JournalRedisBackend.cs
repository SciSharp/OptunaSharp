using System;
using System.Collections.Generic;
using System.Threading;
using NumSharp;                     // Python-style exceptions (ValueError) — same as JournalFileBackend.
using OptunaSharp.Collections;      // OrderedDict<,> — the log-dict object model.
using OptunaSharp.Internal;         // PyJson (byte-compatible json), OptunaWarnings/WarningCategory.
using StackExchange.Redis;

namespace OptunaSharp.Storages.Journal;

/// <summary>
///     Redis storage class for Journal log backend.
///     Port of <c>optuna/storages/journal/_redis.py: JournalRedisBackend</c>
///     (experimental since Optuna 3.1.0).
///
///     Ships in the satellite package <c>OptunaSharp.Storage.Redis</c> (Core stays pure managed;
///     anything needing a foreign runtime is a separate NuGet — the <c>NumSharp.Interop.OpenBLAS</c>
///     pattern). The on-wire encoding is byte-compatible with Python Optuna's: every log is stored
///     as <c>json.dumps(log)</c> (the DEFAULT <c>(", ", ": ")</c> separators — note the file backend
///     uses the compact <c>(",", ":")</c> form instead, so the two are deliberately different) under
///     the same <c>{prefix}:log:{n}</c> keys and the same <c>{prefix}:log_number</c> counter, so a
///     study persisted to Redis by either side can be read by the other.
/// </summary>
/// <remarks>
///     <para>
///         <b>use_cluster.</b> When <c>false</c> (default) it is assumed a standalone Redis server is
///         used and that a write is atomic; the append is one Lua <c>EVAL</c> that increments the
///         counter and sets the log in a single call, giving the preserved logs consistency. When
///         <c>true</c> the Redis cluster is used and a write is not ensured atomic — the counter is
///         incremented and the log set as two separate commands, so the preserved logs can be
///         inconsistent under network errors and may cause errors. (Verbatim from Optuna's docs.)
///     </para>
///     <para>
///         <b>prefix.</b> Prefix of the preserved key of logs. Useful when multiple users work on one
///         Redis server. With the default empty prefix the keys are literally <c>:log_number</c>,
///         <c>:log:{n}</c> and <c>:snapshot</c> — identical to Python.
///     </para>
/// </remarks>
public class JournalRedisBackend : BaseJournalBackend, IBaseJournalSnapshot
{
    // The append-log Lua script, character-for-character as Optuna's (kept identical for diffing).
    // numkeys = 0, so ARGV[1] = prefix and ARGV[2] = the json.dumps(log). The script derives the
    // counter/log keys itself — which is exactly why it is only used on a standalone server (a
    // cluster cannot slot-route keys computed inside the script; hence the use_cluster branch).
    private const string AppendLogScript =
        "local i = redis.call('incr', string.format('%s:log_number', ARGV[1])) " +
        "redis.call('set', string.format('%s:log:%d', ARGV[1], i), ARGV[2])";

    private readonly string _url;
    private readonly bool _useCluster;
    private readonly string _prefix;
    private readonly IConnectionMultiplexer _connection;

    /// <param name="url">
    ///     URL of the redis storage, password and db are optional (ie: <c>redis://localhost:6379</c>).
    ///     <c>redis://</c>, <c>rediss://</c> (TLS) and <c>unix://</c> schemes are accepted, with an
    ///     optional <c>[username]:[password]@</c> and <c>/db</c> (or <c>?db=</c>).
    /// </param>
    /// <param name="useCluster">
    ///     Flag whether you use the Redis cluster. If <c>false</c> (default), it is assumed you use
    ///     the standalone Redis server and ensured that a write operation is atomic. This provides the
    ///     consistency of the preserved logs. If <c>true</c>, it is assumed that you use the Redis
    ///     cluster and not ensured that a write operation is atomic. This means the preserved logs can
    ///     be inconsistent due to network errors, and may cause errors.
    /// </param>
    /// <param name="prefix">
    ///     Prefix of the preserved key of logs. This is useful when multiple users work on one Redis
    ///     server.
    /// </param>
    public JournalRedisBackend(string url, bool useCluster = false, string prefix = "")
        : this(url, ConnectionMultiplexer.Connect(RedisUrl.BuildConfiguration(url)), useCluster, prefix)
    {
    }

    /// <summary>
    ///     Advanced/testing constructor that binds to an already-built multiplexer (e.g. one shared
    ///     across an application, or a fake for unit tests) instead of connecting from the URL. The
    ///     <paramref name="url" /> is retained only for identification.
    /// </summary>
    internal JournalRedisBackend(string url, IConnectionMultiplexer connection, bool useCluster, string prefix)
    {
        // @experimental_class("3.1.0"): warn on construction, exactly as Optuna's decorator does.
        OptunaWarnings.Warn(
            "JournalRedisBackend is experimental (supported from v3.1.0). The interface can change in the future.",
            WarningCategory.ExperimentalWarning);

        _url = url;
        _connection = connection;
        _useCluster = useCluster;
        _prefix = prefix;
    }

    private IDatabase Db => _connection.GetDatabase();

    public override IEnumerable<OrderedDict<string, object?>> ReadLogs(int logNumberFrom)
    {
        RedisValue maxLogNumberBytes = Db.StringGet($"{_prefix}:log_number");
        if (maxLogNumberBytes.IsNull)
            yield break;
        long maxLogNumber = (long)maxLogNumberBytes;

        for (long logNumber = logNumberFrom; logNumber <= maxLogNumber; logNumber++)
        {
            double sleepSecs = 0.1;
            RedisValue log;
            while (true)
            {
                log = Db.StringGet(KeyLogId(logNumber));
                if (!log.IsNull)
                    break;
                Thread.Sleep(TimeSpan.FromSeconds(sleepSecs));
                sleepSecs = Math.Min(sleepSecs * 2, 10);
            }

            // Optuna: `try: yield json.loads(log) except JSONDecodeError as err: if log_number !=
            // max_log_number: raise err`. A corrupt/partial LAST log is a writer caught mid-append
            // (its counter is bumped before the value is set) — tolerate it by stopping; any earlier
            // decode error is a real corruption and re-raised. (yield can't sit inside try/catch in a
            // C# iterator, so parse into a local and yield after.)
            OrderedDict<string, object?>? parsed = null;
            bool decoded = true;
            try
            {
                parsed = (OrderedDict<string, object?>)PyJson.Loads(log.ToString())!;
            }
            catch (ValueError)
            {
                if (logNumber != maxLogNumber)
                    throw;
                decoded = false;
            }

            if (!decoded)
                yield break;
            yield return parsed!;
        }
    }

    public override void AppendLogs(IReadOnlyList<OrderedDict<string, object?>> logs)
    {
        IDatabase db = Db;
        // Initialize the counter to -1 if it does not yet exist (the first incr makes it 0).
        db.StringSet($"{_prefix}:log_number", -1, when: When.NotExists);
        foreach (var log in logs)
        {
            if (!_useCluster)
            {
                // Atomic incr-then-set in one round trip (standalone-server assumption).
                db.ScriptEvaluate(
                    AppendLogScript,
                    keys: Array.Empty<RedisKey>(),
                    values: new RedisValue[] { _prefix, PyJson.Dumps(log) });
            }
            else
            {
                // Cluster: two non-atomic commands (a script cannot slot-route derived keys).
                long logNumber = db.StringIncrement($"{_prefix}:log_number", 1);
                db.StringSet(KeyLogId(logNumber), PyJson.Dumps(log));
            }
        }
    }

    public void SaveSnapshot(byte[] snapshot)
    {
        Db.StringSet($"{_prefix}:snapshot", snapshot);
    }

    public byte[]? LoadSnapshot()
    {
        RedisValue snapshotBytes = Db.StringGet($"{_prefix}:snapshot");
        return snapshotBytes.IsNull ? null : (byte[]?)snapshotBytes;
    }

    private string KeyLogId(long logNumber) => $"{_prefix}:log:{logNumber}";
}

/// <summary>
///     Deprecated alias of <see cref="JournalRedisBackend" />.
///     Port of <c>optuna/storages/journal/_redis.py: JournalRedisStorage</c>.
/// </summary>
[Obsolete("Deprecated in v4.0.0. This feature will be removed in v6.0.0. Use JournalRedisBackend instead.")]
public class JournalRedisStorage : JournalRedisBackend
{
    public JournalRedisStorage(string url, bool useCluster = false, string prefix = "")
        : base(url, useCluster, prefix)
    {
    }
}

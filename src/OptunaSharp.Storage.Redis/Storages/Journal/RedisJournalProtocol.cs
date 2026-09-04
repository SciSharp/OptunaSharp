namespace OptunaSharp.Storages.Journal;

/// <summary>
///     The on-the-wire Redis journal protocol shared by every client backend: the key layout, the
///     atomic append Lua script, and the experimental-warning text. Sharing these constants (rather
///     than an implementation abstraction) is exactly what keeps the three independent client backends
///     — StackExchange / NRedisStack / ServiceStack — byte-identical against the same Redis, and
///     byte-compatible with Python Optuna's <c>JournalRedisBackend</c>.
/// </summary>
public static class RedisJournalProtocol
{
    /// <summary>The integer log-number counter key: <c>{prefix}:log_number</c>.</summary>
    public static string LogNumberKey(string prefix) => $"{prefix}:log_number";

    /// <summary>The key of log <paramref name="logNumber" />: <c>{prefix}:log:{n}</c>.</summary>
    public static string LogKey(string prefix, long logNumber) => $"{prefix}:log:{logNumber}";

    /// <summary>The snapshot blob key: <c>{prefix}:snapshot</c>.</summary>
    public static string SnapshotKey(string prefix) => $"{prefix}:snapshot";

    /// <summary>The initial counter value written by SETNX before the first append (first INCR makes it 0).</summary>
    public const long InitialLogNumber = -1;

    /// <summary>
    ///     Optuna's append-log Lua script, character-for-character. <c>numkeys = 0</c>, so
    ///     <c>ARGV[1]</c> = the prefix and <c>ARGV[2]</c> = <c>json.dumps(log)</c>; the script derives
    ///     the counter/log keys itself (which is why it is only used on a standalone server — a cluster
    ///     cannot slot-route keys computed inside a script).
    /// </summary>
    public const string AppendLogScript =
        "local i = redis.call('incr', string.format('%s:log_number', ARGV[1])) " +
        "redis.call('set', string.format('%s:log:%d', ARGV[1], i), ARGV[2])";

    /// <summary>
    ///     The Optuna <c>@experimental_class("3.1.0")</c> warning message for a given backend type
    ///     name (each client backend passes its own class name).
    /// </summary>
    public static string ExperimentalWarning(string typeName) =>
        $"{typeName} is experimental (supported from v3.1.0). The interface can change in the future.";
}

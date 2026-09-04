using System;
using System.Collections.Generic;
using System.Net.Sockets;
using System.Threading;
using NumSharp;                     // ValueError
using OptunaSharp.Collections;      // OrderedDict<,>
using OptunaSharp.Internal;         // PyJson, OptunaWarnings / WarningCategory
using StackExchange.Redis;

namespace OptunaSharp.Storages.Journal;

/// <summary>
///     Redis journal-log backend on <b>StackExchange.Redis</b> — the de-facto-standard .NET Redis
///     client. A port of Optuna's <c>optuna/storages/journal/_redis.py: JournalRedisBackend</c>
///     (experimental since Optuna 3.1.0).
///
///     One of three interchangeable client backends (see also
///     <c>NRedisStackJournalRedisBackend</c> / <c>ServiceStackJournalRedisBackend</c>). All three use
///     the shared <see cref="RedisJournalProtocol" /> key layout + Core's <c>PyJson</c>, so a study is
///     byte-identical across them and byte-compatible with Python Optuna over the same Redis — only the
///     client calls (here, StackExchange.Redis' <see cref="IDatabase" />) differ.
/// </summary>
public class StackExchangeJournalRedisBackend : BaseJournalBackend, IBaseJournalSnapshot
{
    private readonly string _url;
    private readonly bool _useCluster;
    private readonly string _prefix;
    private readonly IConnectionMultiplexer _connection;

    /// <param name="url">URL of the redis storage (ie: <c>redis://localhost:6379</c>); <c>redis://</c>,
    ///   <c>rediss://</c> (TLS), <c>unix://</c> accepted, password/db optional.</param>
    /// <param name="useCluster">If <c>false</c> (default), a standalone server + one atomic Lua
    ///   <c>EVAL</c>; if <c>true</c>, a cluster + non-atomic <c>INCR</c>+<c>SET</c>.</param>
    /// <param name="prefix">Prefix of the preserved keys — useful when many users share one server.</param>
    public StackExchangeJournalRedisBackend(string url, bool useCluster = false, string prefix = "")
        : this(url, ConnectionMultiplexer.Connect(ToConfiguration(RedisUrl.Parse(url))), useCluster, prefix)
    {
    }

    /// <summary>Advanced/testing constructor binding to an already-built multiplexer.</summary>
    internal StackExchangeJournalRedisBackend(string url, IConnectionMultiplexer connection, bool useCluster, string prefix)
    {
        OptunaWarnings.Warn(
            RedisJournalProtocol.ExperimentalWarning(nameof(StackExchangeJournalRedisBackend)),
            WarningCategory.ExperimentalWarning);
        _url = url;
        _connection = connection;
        _useCluster = useCluster;
        _prefix = prefix;
    }

    private IDatabase Db => _connection.GetDatabase();

    public override IEnumerable<OrderedDict<string, object?>> ReadLogs(int logNumberFrom)
    {
        RedisValue maxLogNumberBytes = Db.StringGet(RedisJournalProtocol.LogNumberKey(_prefix));
        if (maxLogNumberBytes.IsNull)
            yield break;
        long maxLogNumber = (long)maxLogNumberBytes;

        for (long logNumber = logNumberFrom; logNumber <= maxLogNumber; logNumber++)
        {
            double sleepSecs = 0.1;
            RedisValue log;
            while (true)
            {
                log = Db.StringGet(RedisJournalProtocol.LogKey(_prefix, logNumber));
                if (!log.IsNull)
                    break;
                Thread.Sleep(TimeSpan.FromSeconds(sleepSecs));
                sleepSecs = Math.Min(sleepSecs * 2, 10);
            }

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
        db.StringSet(RedisJournalProtocol.LogNumberKey(_prefix), RedisJournalProtocol.InitialLogNumber, when: When.NotExists);
        foreach (var log in logs)
        {
            if (!_useCluster)
            {
                db.ScriptEvaluate(
                    RedisJournalProtocol.AppendLogScript,
                    keys: Array.Empty<RedisKey>(),
                    values: new RedisValue[] { _prefix, PyJson.Dumps(log) });
            }
            else
            {
                long logNumber = db.StringIncrement(RedisJournalProtocol.LogNumberKey(_prefix), 1);
                db.StringSet(RedisJournalProtocol.LogKey(_prefix, logNumber), PyJson.Dumps(log));
            }
        }
    }

    public void SaveSnapshot(byte[] snapshot)
        => Db.StringSet(RedisJournalProtocol.SnapshotKey(_prefix), snapshot);

    public byte[]? LoadSnapshot()
    {
        RedisValue snapshotBytes = Db.StringGet(RedisJournalProtocol.SnapshotKey(_prefix));
        return snapshotBytes.IsNull ? null : (byte[]?)snapshotBytes;
    }

    /// <summary>Maps the neutral <see cref="RedisEndpoint" /> onto StackExchange.Redis' options.</summary>
    internal static ConfigurationOptions ToConfiguration(RedisEndpoint ep)
    {
        var o = new ConfigurationOptions { AbortOnConnectFail = false };  // redis-py connects lazily
        if (ep.UnixSocketPath is not null)
        {
            o.EndPoints.Add(new UnixDomainSocketEndPoint(ep.UnixSocketPath));
        }
        else
        {
            o.EndPoints.Add(ep.Host, ep.Port);
            o.Ssl = ep.Ssl;
            if (ep.Ssl)
                o.SslHost = ep.Host;
        }
        if (ep.User is not null)
            o.User = ep.User;
        if (ep.Password is not null)
            o.Password = ep.Password;
        if (ep.Db is not null)
            o.DefaultDatabase = ep.Db;
        return o;
    }
}

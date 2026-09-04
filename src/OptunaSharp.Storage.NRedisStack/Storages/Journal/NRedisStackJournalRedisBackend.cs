using System;
using System.Collections.Generic;
using System.Net.Sockets;
using System.Threading;
using NumSharp;                     // ValueError
using OptunaSharp.Collections;      // OrderedDict<,>
using OptunaSharp.Internal;         // PyJson, OptunaWarnings / WarningCategory
using StackExchange.Redis;          // NRedisStack is a superset of StackExchange.Redis (brought transitively)

namespace OptunaSharp.Storages.Journal;

/// <summary>
///     Redis journal-log backend on <b>NRedisStack</b> — Redis's official .NET client. NRedisStack
///     layers the Redis Stack module commands (JSON / Search / TimeSeries / Bloom) on top of
///     StackExchange.Redis; a journal backend needs only plain GET/SET/INCR/EVAL, which NRedisStack
///     serves through the underlying <see cref="IDatabase" />. So this backend is byte-identical to
///     <c>StackExchangeJournalRedisBackend</c> by construction (they share <see cref="RedisJournalProtocol" />
///     + Core's <c>PyJson</c>) — it differs only in the client package you depend on. Storing the logs
///     as RedisJSON documents (NRedisStack's headline feature) is deliberately NOT used: that would
///     change the stored bytes and break interoperability with Python Optuna and the sibling backends.
/// </summary>
public class NRedisStackJournalRedisBackend : BaseJournalBackend, IBaseJournalSnapshot
{
    private readonly string _url;
    private readonly bool _useCluster;
    private readonly string _prefix;
    private readonly IConnectionMultiplexer _connection;

    /// <param name="url">URL of the redis storage (ie: <c>redis://localhost:6379</c>).</param>
    /// <param name="useCluster">If <c>false</c> (default), standalone + atomic Lua <c>EVAL</c>; if
    ///   <c>true</c>, cluster + non-atomic <c>INCR</c>+<c>SET</c>.</param>
    /// <param name="prefix">Prefix of the preserved keys.</param>
    public NRedisStackJournalRedisBackend(string url, bool useCluster = false, string prefix = "")
        : this(url, ConnectionMultiplexer.Connect(ToConfiguration(RedisUrl.Parse(url))), useCluster, prefix)
    {
    }

    /// <summary>Advanced/testing constructor binding to an already-built multiplexer.</summary>
    internal NRedisStackJournalRedisBackend(string url, IConnectionMultiplexer connection, bool useCluster, string prefix)
    {
        OptunaWarnings.Warn(
            RedisJournalProtocol.ExperimentalWarning(nameof(NRedisStackJournalRedisBackend)),
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
        var o = new ConfigurationOptions { AbortOnConnectFail = false };
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

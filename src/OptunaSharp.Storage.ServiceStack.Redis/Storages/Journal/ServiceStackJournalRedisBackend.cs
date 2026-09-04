using System;
using System.Collections.Generic;
using System.Globalization;
using NumSharp;                     // ValueError
using OptunaSharp.Collections;      // OrderedDict<,>
using OptunaSharp.Internal;         // PyJson, OptunaWarnings / WarningCategory
using ServiceStack.Redis;
using SsRedisEndpoint = ServiceStack.Redis.RedisEndpoint;

namespace OptunaSharp.Storages.Journal;

/// <summary>
///     Redis journal-log backend on <b>ServiceStack.Redis</b>. A port of Optuna's
///     <c>optuna/storages/journal/_redis.py: JournalRedisBackend</c> (experimental since Optuna 3.1.0),
///     one of three interchangeable client backends. All three use the shared
///     <see cref="RedisJournalProtocol" /> key layout + Core's <c>PyJson</c>, so a study is
///     byte-identical across them and byte-compatible with Python Optuna over the same Redis — here the
///     client is ServiceStack.Redis' <see cref="IRedisClient" /> (pooled via an
///     <see cref="IRedisClientsManager" />; raw string/EVAL/byte commands, no typed serialization).
///
///     <para><b>License:</b> ServiceStack.Redis is commercially licensed (its free tier is
///     request-quota-limited). Prefer the MIT <c>StackExchange</c>/<c>NRedisStack</c> backends unless
///     your app already standardizes on ServiceStack.</para>
/// </summary>
public class ServiceStackJournalRedisBackend : BaseJournalBackend, IBaseJournalSnapshot
{
    private readonly string _url;
    private readonly bool _useCluster;
    private readonly string _prefix;
    private readonly IRedisClientsManager _manager;

    /// <param name="url">URL of the redis storage (ie: <c>redis://localhost:6379</c>); <c>redis://</c>
    ///   and <c>rediss://</c> (TLS) are accepted (ServiceStack.Redis does not support unix sockets).</param>
    /// <param name="useCluster">If <c>false</c> (default), standalone + atomic Lua <c>EVAL</c>; if
    ///   <c>true</c>, cluster + non-atomic <c>INCR</c>+<c>SET</c>.</param>
    /// <param name="prefix">Prefix of the preserved keys.</param>
    public ServiceStackJournalRedisBackend(string url, bool useCluster = false, string prefix = "")
        : this(url, CreateManager(RedisUrl.Parse(url)), useCluster, prefix)
    {
    }

    /// <summary>Advanced/testing constructor binding to an already-built client manager.</summary>
    internal ServiceStackJournalRedisBackend(string url, IRedisClientsManager manager, bool useCluster, string prefix)
    {
        OptunaWarnings.Warn(
            RedisJournalProtocol.ExperimentalWarning(nameof(ServiceStackJournalRedisBackend)),
            WarningCategory.ExperimentalWarning);
        _url = url;
        _manager = manager;
        _useCluster = useCluster;
        _prefix = prefix;
    }

    public override IEnumerable<OrderedDict<string, object?>> ReadLogs(int logNumberFrom)
    {
        string? maxStr = GetValue(RedisJournalProtocol.LogNumberKey(_prefix));
        if (maxStr is null)
            yield break;
        long maxLogNumber = long.Parse(maxStr, CultureInfo.InvariantCulture);

        for (long logNumber = logNumberFrom; logNumber <= maxLogNumber; logNumber++)
        {
            double sleepSecs = 0.1;
            string? log;
            while (true)
            {
                log = GetValue(RedisJournalProtocol.LogKey(_prefix, logNumber));
                if (log is not null)
                    break;
                System.Threading.Thread.Sleep(TimeSpan.FromSeconds(sleepSecs));
                sleepSecs = Math.Min(sleepSecs * 2, 10);
            }

            OrderedDict<string, object?>? parsed = null;
            bool decoded = true;
            try
            {
                parsed = (OrderedDict<string, object?>)PyJson.Loads(log)!;
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
        using IRedisClient client = _manager.GetClient();
        client.SetValueIfNotExists(
            RedisJournalProtocol.LogNumberKey(_prefix),
            RedisJournalProtocol.InitialLogNumber.ToString(CultureInfo.InvariantCulture));

        foreach (var log in logs)
        {
            string json = PyJson.Dumps(log);
            if (!_useCluster)
            {
                // Atomic incr-then-set in one round trip: EVAL script 0 <prefix> <json> — the shared
                // Lua uses ARGV[1]=prefix, ARGV[2]=json (numkeys = 0).
                client.ExecLua(RedisJournalProtocol.AppendLogScript, Array.Empty<string>(), new[] { _prefix, json });
            }
            else
            {
                long logNumber = client.IncrementValue(RedisJournalProtocol.LogNumberKey(_prefix));
                client.SetValue(RedisJournalProtocol.LogKey(_prefix, logNumber), json);
            }
        }
    }

    public void SaveSnapshot(byte[] snapshot)
    {
        using IRedisClient client = _manager.GetClient();
        ((IRedisNativeClient)client).Set(RedisJournalProtocol.SnapshotKey(_prefix), snapshot);
    }

    public byte[]? LoadSnapshot()
    {
        using IRedisClient client = _manager.GetClient();
        // ServiceStack's native GET returns null for a missing key — same as the other backends'
        // RedisValue.IsNull, so an absent snapshot is null and a present one is its exact bytes.
        return ((IRedisNativeClient)client).Get(RedisJournalProtocol.SnapshotKey(_prefix));
    }

    private string? GetValue(string key)
    {
        using IRedisClient client = _manager.GetClient();
        return client.GetValue(key);   // null when the key does not exist
    }

    /// <summary>Builds a pooled ServiceStack client manager from the neutral <see cref="RedisEndpoint" />.</summary>
    internal static IRedisClientsManager CreateManager(RedisEndpoint ep)
    {
        if (ep.UnixSocketPath is not null)
            throw new ValueError("ServiceStack.Redis does not support unix domain sockets; use redis:// or rediss://.");

        var sep = new SsRedisEndpoint(ep.Host, ep.Port <= 0 ? 6379 : ep.Port)
        {
            Ssl = ep.Ssl,
            Db = ep.Db ?? 0,
            Password = ep.Password,
            Username = ep.User,
        };
        return new RedisManagerPool(sep.ToString());
    }
}

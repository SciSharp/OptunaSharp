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
///
///     <para>Beyond Optuna's URL-only constructor, this backend lets you drive StackExchange.Redis
///     on its own terms — pass a fully-built <see cref="ConfigurationOptions" />, hand in an existing
///     <see cref="IConnectionMultiplexer" /> (a connection you pool / share / manage through DI), or
///     keep the URL and mutate the derived options through a callback. Every StackExchange.Redis knob
///     is reachable; the on-wire journal format is fixed by <see cref="RedisJournalProtocol" />
///     regardless of how you connect, so byte-compatibility holds however you configure the client.</para>
/// </summary>
public class StackExchangeJournalRedisBackend : BaseJournalBackend, IBaseJournalSnapshot
{
    private readonly string _url;
    private readonly bool _useCluster;
    private readonly string _prefix;
    private readonly IConnectionMultiplexer _connection;

    // ---------------------------------------------------------------- construction
    //
    // Four ways in, from "just give it a URL" (Optuna-identical) to "here is the client I already
    // built". The three richer overloads let you use StackExchange.Redis exactly as you would on its
    // own — every ConfigurationOptions knob (timeouts, ConnectRetry, ReconnectRetryPolicy, ClientName,
    // ChannelPrefix, SSL protocols/certificates, CommandMap, …) is yours.

    /// <summary>Connects from a <c>redis-py</c>-style URL — the Optuna-identical entry point.</summary>
    /// <param name="url">URL of the redis storage (ie: <c>redis://localhost:6379</c>); <c>redis://</c>,
    ///   <c>rediss://</c> (TLS), <c>unix://</c> accepted, password/db optional. Only the components
    ///   <c>redis-py</c> puts on the URL are read; for StackExchange.Redis-specific options use one of
    ///   the other constructors.</param>
    /// <param name="useCluster">If <c>false</c> (default), a standalone server + one atomic Lua
    ///   <c>EVAL</c>; if <c>true</c>, a cluster + non-atomic <c>INCR</c>+<c>SET</c>.</param>
    /// <param name="prefix">Prefix of the preserved keys — useful when many users share one server.</param>
    public StackExchangeJournalRedisBackend(string url, bool useCluster = false, string prefix = "")
        : this(ConnectionMultiplexer.Connect(ToConfiguration(RedisUrl.Parse(url))), url, useCluster, prefix)
    {
    }

    /// <summary>
    ///     Connects from a <c>redis-py</c>-style URL, then hands you the derived
    ///     <see cref="ConfigurationOptions" /> to adjust before the connection opens — the easy way to
    ///     keep the familiar URL yet set knobs it cannot express (<c>ConnectTimeout</c>,
    ///     <c>SyncTimeout</c>, <c>ConnectRetry</c>, <c>ClientName</c>, TLS certificates, …).
    /// </summary>
    /// <param name="url">The <c>redis-py</c>-style base URL (see the URL-only constructor).</param>
    /// <param name="configure">Invoked on the <see cref="ConfigurationOptions" /> parsed from
    ///   <paramref name="url" /> immediately before connecting; mutate it in place.</param>
    /// <param name="useCluster">Append strategy (see the URL-only constructor).</param>
    /// <param name="prefix">Key prefix (see the URL-only constructor).</param>
    public StackExchangeJournalRedisBackend(string url, Action<ConfigurationOptions> configure,
        bool useCluster = false, string prefix = "")
        : this(ConnectionMultiplexer.Connect(Configure(RedisUrl.Parse(url), configure)), url, useCluster, prefix)
    {
    }

    /// <summary>
    ///     Connects from a fully-specified StackExchange.Redis <see cref="ConfigurationOptions" /> — build
    ///     the connection entirely in StackExchange.Redis' own terms (multiple endpoints, a
    ///     <c>CommandMap</c>, TLS settings, and so on) instead of through a URL.
    /// </summary>
    /// <param name="options">The connection options; passed straight to <see cref="ConnectionMultiplexer" />.</param>
    /// <param name="useCluster">Append strategy (see the URL-only constructor).</param>
    /// <param name="prefix">Key prefix (see the URL-only constructor).</param>
    public StackExchangeJournalRedisBackend(ConfigurationOptions options, bool useCluster = false, string prefix = "")
        : this(ConnectionMultiplexer.Connect(options ?? throw new ArgumentNullException(nameof(options))),
               "", useCluster, prefix)
    {
    }

    /// <summary>
    ///     Binds to a StackExchange.Redis client you already built and connected — the full escape hatch.
    ///     Everything about the connection is yours (pooling, a multiplexer shared across your app,
    ///     DI-managed lifetime); the backend only issues journal commands on it. <b>You own the
    ///     multiplexer's lifetime</b> — the backend never disposes a connection it did not create.
    /// </summary>
    /// <param name="connection">A connected <see cref="IConnectionMultiplexer" />.</param>
    /// <param name="useCluster">Append strategy (see the URL-only constructor).</param>
    /// <param name="prefix">Key prefix (see the URL-only constructor).</param>
    public StackExchangeJournalRedisBackend(IConnectionMultiplexer connection, bool useCluster = false, string prefix = "")
        : this(connection ?? throw new ArgumentNullException(nameof(connection)), "", useCluster, prefix)
    {
    }

    /// <summary>Shared core: every public constructor funnels here (so all raise the experimental warning).</summary>
    private StackExchangeJournalRedisBackend(IConnectionMultiplexer connection, string url, bool useCluster, string prefix)
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
        // StackExchange.Redis defaults AbortOnConnectFail to true (Connect throws if no server is up);
        // we set false so construction succeeds without a live server, matching redis-py's lazy connect
        // (Optuna's `redis.Redis.from_url(url)` does not dial until the first command).
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

    /// <summary>Builds the URL's options, then applies the caller's native <paramref name="configure" /> tweak.</summary>
    internal static ConfigurationOptions Configure(RedisEndpoint ep, Action<ConfigurationOptions> configure)
    {
        if (configure is null)
            throw new ArgumentNullException(nameof(configure));
        ConfigurationOptions o = ToConfiguration(ep);
        configure(o);
        return o;
    }
}

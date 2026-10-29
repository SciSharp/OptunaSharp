using System;
using System.Collections.Generic;
using System.Net.Sockets;
using System.Threading;
using System.Threading.Tasks;
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
///
///     <para>Beyond Optuna's URL-only constructor, this backend lets you drive the client on its own
///     terms — pass a fully-built <see cref="ConfigurationOptions" />, hand in an existing
///     <see cref="IConnectionMultiplexer" /> (a connection you pool / share / manage through DI, and on
///     which you may already be using NRedisStack's module APIs), or keep the URL and mutate the derived
///     options through a callback. The on-wire journal format is fixed by <see cref="RedisJournalProtocol" />
///     regardless of how you connect, so byte-compatibility holds however you configure the client.</para>
/// </summary>
public class NRedisStackJournalRedisBackend : BaseJournalBackend, IBaseJournalSnapshot, IDisposable, IAsyncDisposable
{
    private readonly string _url;
    private readonly bool _useCluster;
    private readonly string _prefix;
    private readonly IConnectionMultiplexer _connection;

    // True when a constructor opened _connection from a URL or options, so Dispose/DisposeAsync must close it;
    // false for an injected multiplexer, which stays the caller's. Before Dispose existed, every scoped or transient
    // DI registration leaked one open Redis connection per scope (40 scopes left 40 connections).
    private readonly bool _ownsConnection;
    private int _disposed;

    // ---------------------------------------------------------------- construction
    //
    // Four ways in, from "just give it a URL" (Optuna-identical) to "here is the client I already
    // built". The three richer overloads let you use the client exactly as you would on its own —
    // every ConfigurationOptions knob (timeouts, ConnectRetry, ReconnectRetryPolicy, ClientName,
    // ChannelPrefix, SSL protocols/certificates, CommandMap, …) is yours, and the injected-connection
    // overload accepts a multiplexer you may already be using with NRedisStack's module APIs.

    /// <summary>Connects from a <c>redis-py</c>-style URL — the Optuna-identical entry point.</summary>
    /// <param name="url">URL of the redis storage (ie: <c>redis://localhost:6379</c>). It is read exactly
    ///   as <c>redis-py</c>'s <c>Redis.from_url</c> reads it, query options included (<c>socket_timeout</c>,
    ///   <c>client_name</c>, <c>protocol</c>, <c>ssl_cert_reqs</c>, <c>ssl_ca_certs</c>, …; see
    ///   <see cref="RedisUrl" />); for client-specific options use one of the other constructors.</param>
    /// <param name="useCluster">If <c>false</c> (default), standalone + atomic Lua <c>EVAL</c>; if
    ///   <c>true</c>, cluster + non-atomic <c>INCR</c>+<c>SET</c>.</param>
    /// <param name="prefix">Prefix of the preserved keys.</param>
    /// <exception cref="NumSharp.ValueError">A URL redis-py rejects, with redis-py's message (see <see cref="RedisUrl.Parse" />).</exception>
    /// <exception cref="NumSharp.TypeError">A query argument redis-py's connection class does not take, with redis-py's message.</exception>
    /// <exception cref="OverflowException">A timeout or TLS version too large for Python, with Python's message.</exception>
    /// <exception cref="NotSupportedException">A query argument no .NET client can honor, or white-space-only credentials.</exception>
    /// <exception cref="System.IO.FileNotFoundException">A TLS certificate, key or authority file that does not exist.</exception>
    /// <exception cref="System.Security.Cryptography.CryptographicException">A malformed TLS certificate, key or authority file.</exception>
    /// <remarks>The backend owns the connection it opens here: <see cref="Dispose" /> closes it.</remarks>
    public NRedisStackJournalRedisBackend(string url, bool useCluster = false, string prefix = "")
        : this(ConnectionMultiplexer.Connect(ToConfiguration(RedisUrl.Parse(url))), url, useCluster, prefix, ownsConnection: true)
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
    /// <remarks>The backend owns the connection it opens here: <see cref="Dispose" /> closes it.</remarks>
    public NRedisStackJournalRedisBackend(string url, Action<ConfigurationOptions> configure,
        bool useCluster = false, string prefix = "")
        : this(ConnectionMultiplexer.Connect(Configure(RedisUrl.Parse(url), configure)), url, useCluster, prefix, ownsConnection: true)
    {
    }

    /// <summary>
    ///     Connects from a fully-specified <see cref="ConfigurationOptions" /> — build the connection
    ///     entirely in the client's own terms (multiple endpoints, a <c>CommandMap</c>, TLS settings, and
    ///     so on) instead of through a URL.
    /// </summary>
    /// <param name="options">The connection options; passed straight to <see cref="ConnectionMultiplexer" />.</param>
    /// <param name="useCluster">Append strategy (see the URL-only constructor).</param>
    /// <param name="prefix">Key prefix (see the URL-only constructor).</param>
    /// <exception cref="System.ArgumentNullException"><paramref name="options" /> is null.</exception>
    /// <remarks>The backend owns the connection it opens here: <see cref="Dispose" /> closes it.</remarks>
    public NRedisStackJournalRedisBackend(ConfigurationOptions options, bool useCluster = false, string prefix = "")
        : this(ConnectionMultiplexer.Connect(options ?? throw new ArgumentNullException(nameof(options))),
               "", useCluster, prefix, ownsConnection: true)
    {
    }

    /// <summary>
    ///     Binds to a client you already built and connected — the full escape hatch. Everything about
    ///     the connection is yours (pooling, a multiplexer shared across your app and its NRedisStack
    ///     module calls, DI-managed lifetime); the backend only issues journal commands on it. <b>You own
    ///     the multiplexer's lifetime</b> — the backend never disposes a connection it did not create.
    /// </summary>
    /// <param name="connection">A connected <see cref="IConnectionMultiplexer" />.</param>
    /// <param name="useCluster">Append strategy (see the URL-only constructor).</param>
    /// <param name="prefix">Key prefix (see the URL-only constructor).</param>
    /// <exception cref="System.ArgumentNullException"><paramref name="connection" /> is null.</exception>
    /// <remarks><see cref="Dispose" /> leaves an injected multiplexer open: its lifetime stays yours.</remarks>
    public NRedisStackJournalRedisBackend(IConnectionMultiplexer connection, bool useCluster = false, string prefix = "")
        : this(connection ?? throw new ArgumentNullException(nameof(connection)), "", useCluster, prefix, ownsConnection: false)
    {
    }

    /// <summary>Binds the shared Redis connection, key namespace, and append strategy for all constructor paths.</summary>
    /// <param name="connection">The database or Redis connection used by the backend.</param>
    /// <param name="url">The database connection URL to resolve.</param>
    /// <param name="useCluster">Whether to use separate counter and log writes for a Redis cluster rather than the standalone Lua script.</param>
    /// <param name="prefix">The journal key namespace, shared by its counter, log records, and snapshot.</param>
    /// <param name="ownsConnection">Whether this backend opened <paramref name="connection" /> and must close it on <see cref="Dispose" /> or <see cref="DisposeAsync" />.</param>
    private NRedisStackJournalRedisBackend(IConnectionMultiplexer connection, string url, bool useCluster, string prefix, bool ownsConnection)
    {
        // Optuna decorates JournalRedisBackend with @experimental_class("3.1.0"), which raises an
        // ExperimentalWarning on construction. OptunaSharp targets the Optuna 5.x surface and treats
        // it as stable, so that warning is intentionally omitted (deliberate divergence).
        _url = url;
        _connection = connection;
        _useCluster = useCluster;
        _prefix = prefix;
        _ownsConnection = ownsConnection;
    }

    /// <summary>
    ///     Closes the Redis connection this backend opened from a URL or options; an injected multiplexer is left
    ///     open. Calling it again does nothing. Any later journal operation on this backend fails with
    ///     <see cref="ObjectDisposedException" /> from StackExchange.Redis.
    /// </summary>
    /// <remarks>
    ///     A DI container calls this when the scope (or, for a singleton, the container) that created the backend is
    ///     disposed synchronously, so scoped and transient registrations no longer leak a connection each. A
    ///     <see cref="JournalStorage" /> over this backend must not be used afterwards. The close runs on the calling
    ///     thread, which waits while the multiplexer flushes and shuts its sockets; in async code prefer
    ///     <see cref="DisposeAsync" />.
    /// </remarks>
    public void Dispose()
    {
        // Only the first call closes, whichever of Dispose and DisposeAsync it is; DI may dispose a forwarded
        // registration and the backend itself.
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
            return;
        if (_ownsConnection)
            _connection.Dispose();
        GC.SuppressFinalize(this);
    }

    /// <summary>
    ///     Closes the Redis connection this backend opened from a URL or options without blocking a thread; an
    ///     injected multiplexer is left open. This is the asynchronous form of <see cref="Dispose" />, used by
    ///     <c>await using</c> and by DI scopes that are disposed asynchronously (ASP.NET Core request scopes,
    ///     <c>AsyncServiceScope</c>).
    /// </summary>
    /// <returns>A task that completes once the owned connection is closed, or at once when there is nothing to close.</returns>
    /// <remarks>
    ///     Shares <see cref="Dispose" />'s once-only flag: whichever runs first closes the connection, and the other
    ///     then does nothing. The synchronous close holds the calling thread until the multiplexer's sockets are shut;
    ///     in an async host that is a thread-pool thread, the resource StackExchange.Redis (under NRedisStack) completes
    ///     calls on. A DI container calls this method when it disposes a scope asynchronously, and
    ///     <see cref="Dispose" /> when it disposes one synchronously; both paths are supported.
    /// </remarks>
    public async ValueTask DisposeAsync()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
            return;
        if (_ownsConnection)
            await _connection.DisposeAsync().ConfigureAwait(false);
        GC.SuppressFinalize(this);
    }

    private IDatabase Db => _connection.GetDatabase();

    /// <inheritdoc />
    /// <exception cref="NumSharp.ValueError">A nonfinal journal record contains malformed JSON.</exception>
    public override IEnumerable<OrderedDict<string, object?>> ReadLogs(int logNumberFrom)
    {
        RedisValue maxLogNumberBytes = Db.StringGet(RedisJournalProtocol.LogNumberKey(_prefix));
        if (maxLogNumberBytes.IsNull)
            yield break;
        long maxLogNumber = (long)maxLogNumberBytes;

        for (long logNumber = logNumberFrom; logNumber <= maxLogNumber; logNumber++)
        {
            // A cluster writer reserves a sequence number before publishing its record. Waiting
            // preserves replay order when the counter becomes visible before the corresponding log.
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
                // Defer only an incomplete tail record; older malformed records are persistent
                // corruption and must not be skipped during journal replay.
                if (logNumber != maxLogNumber)
                    throw;
                decoded = false;
            }

            if (!decoded)
                yield break;
            yield return parsed!;
        }
    }

    /// <inheritdoc />
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

    /// <summary>Replaces the Redis journal snapshot with the supplied opaque serialized bytes.</summary>
    /// <param name="snapshot">The serialized journal-state snapshot to store as opaque bytes.</param>
    public void SaveSnapshot(byte[] snapshot)
        => Db.StringSet(RedisJournalProtocol.SnapshotKey(_prefix), snapshot);

    /// <summary>Reads the latest Redis journal snapshot, if one has been saved.</summary>
    /// <returns>The saved snapshot bytes, or null when the snapshot key is absent.</returns>
    public byte[]? LoadSnapshot()
    {
        RedisValue snapshotBytes = Db.StringGet(RedisJournalProtocol.SnapshotKey(_prefix));
        return snapshotBytes.IsNull ? null : (byte[]?)snapshotBytes;
    }

    /// <summary>
    ///     Maps the neutral <see cref="RedisEndpoint" /> onto StackExchange.Redis' options (the client NRedisStack builds
    ///     on), so the connection behaves as redis-py's would for the same URL: its 5-second command and connect
    ///     timeouts, its client name and protocol, and its TLS checks (the system's certificate authorities plus the
    ///     URL's, the host name check unless the URL turns it off, a client certificate, a minimum TLS version, and no
    ///     revocation check, which Python never does and StackExchange.Redis does by default).
    /// </summary>
    /// <param name="ep">The client-neutral, parsed Redis endpoint.</param>
    /// <returns>Native StackExchange.Redis options representing the parsed endpoint.</returns>
    /// <exception cref="NotSupportedException">
    ///     A user name or password made only of white space: StackExchange.Redis skips authentication for it, where
    ///     redis-py sends it as it is.
    /// </exception>
    /// <exception cref="NumSharp.TypeError">An <c>ssl_keyfile</c> without an <c>ssl_certfile</c>; Python's message.</exception>
    /// <exception cref="System.IO.FileNotFoundException">A certificate, key or certificate-authority file that does not exist.</exception>
    /// <exception cref="System.Security.Cryptography.CryptographicException">A malformed certificate, key or authority file, a key that does not match, or a wrong key password.</exception>
    internal static ConfigurationOptions ToConfiguration(RedisEndpoint ep)
    {
        // StackExchange.Redis authenticates only for a user name or password that is not white space, where redis-py
        // authenticates for any non-empty one; refuse the credentials the two would send differently.
        if ((ep.User is not null && string.IsNullOrWhiteSpace(ep.User)) || (ep.User is null && ep.Password is not null && string.IsNullOrWhiteSpace(ep.Password)))
            throw new NotSupportedException(
                "StackExchange.Redis does not authenticate with a user name or password made only of white space, which " +
                "redis-py sends as it is; use the configure callback to set ConfigurationOptions.User/Password explicitly.");
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
            {
                o.SslHost = ep.Host;
                o.CheckCertificateRevocation = false;
                o.SslProtocols = RedisTls.Protocols(ep);
                if (RedisTls.Validation(ep) is { } validate)
                    o.CertificateValidation += validate;
                if (RedisTls.ClientCertificate(ep) is { } certificate)
                    o.CertificateSelection += (_, _, _, _, _) => certificate;
            }
        }
        if (ep.User is not null)
            o.User = ep.User;
        if (ep.Password is not null)
            o.Password = ep.Password;
        if (ep.Db is not null)
            o.DefaultDatabase = ep.Db;
        // Always explicit: StackExchange.Redis would otherwise derive the connect timeout from the command timeout.
        o.SyncTimeout = RedisEndpoint.ToMilliseconds(ep.SocketTimeout);
        o.AsyncTimeout = o.SyncTimeout;
        o.ConnectTimeout = RedisEndpoint.ToMilliseconds(ep.SocketConnectTimeout);
        if (ep.ClientName is not null)
            o.ClientName = ep.ClientName;
        if (ep.LibraryName is not null)
            o.LibraryName = ep.LibraryName;
        if (ep.Protocol is { } protocol)
            o.Protocol = protocol == 3 ? RedisProtocol.Resp3 : RedisProtocol.Resp2;
        return o;
    }

    /// <summary>Builds the URL's options, then applies the caller's native <paramref name="configure" /> tweak.</summary>
    /// <param name="ep">The client-neutral, parsed Redis endpoint.</param>
    /// <param name="configure">The callback that customizes native connection options before opening the connection.</param>
    /// <returns>The native Redis options after the callback has customized them.</returns>
    /// <exception cref="System.ArgumentNullException"><paramref name="configure" /> is null.</exception>
    internal static ConfigurationOptions Configure(RedisEndpoint ep, Action<ConfigurationOptions> configure)
    {
        if (configure is null)
            throw new ArgumentNullException(nameof(configure));
        ConfigurationOptions o = ToConfiguration(ep);
        configure(o);
        return o;
    }
}

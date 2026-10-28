using System;
using System.Collections.Generic;
using System.Globalization;
using System.Threading.Tasks;
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
///     <para>Beyond Optuna's URL-only constructor, this backend lets you drive ServiceStack.Redis on
///     its own terms — hand in an <see cref="IRedisClientsManager" /> you already built (any manager
///     type, pool config and lifetime — shared across your app or DI-managed; the full-control path),
///     pass a ServiceStack.Redis <see cref="SsRedisEndpoint" />, or keep the URL and mutate the derived
///     endpoint through a callback. The on-wire journal format is fixed by
///     <see cref="RedisJournalProtocol" /> regardless of how you connect, so byte-compatibility holds
///     however you configure the client.</para>
///
///     <para>ServiceStack specifics for the endpoint/callback paths (they build a
///     <see cref="RedisManagerPool" /> from the endpoint's connection string): that string reliably
///     carries routing and auth (host, port, <c>Ssl</c>, <c>Db</c>, <c>Password</c>, <c>Username</c>),
///     but ServiceStack's pooled managers configure timeouts/retry through the process-wide
///     <c>RedisConfig</c> defaults rather than a per-endpoint <c>ConnectTimeout</c>, and the endpoint's
///     <c>NamespacePrefix</c> must be left unset (use this backend's <c>prefix</c> parameter — a
///     namespace prefix would diverge from the shared key layout). For anything more, inject a
///     fully-built <see cref="IRedisClientsManager" />.</para>
///
///     <para><b>License:</b> ServiceStack.Redis is commercially licensed (its free tier is
///     request-quota-limited). Prefer the MIT <c>StackExchange</c>/<c>NRedisStack</c> backends unless
///     your app already standardizes on ServiceStack.</para>
/// </summary>
public class ServiceStackJournalRedisBackend : BaseJournalBackend, IBaseJournalSnapshot, IDisposable, IAsyncDisposable
{
    private readonly string _url;
    private readonly bool _useCluster;
    private readonly string _prefix;
    private readonly IRedisClientsManager _manager;

    // True when a constructor built _manager (a RedisManagerPool) from a URL or endpoint, so Dispose/DisposeAsync must
    // close its pooled connections; false for an injected manager, which stays the caller's.
    private readonly bool _ownsManager;
    private int _disposed;

    // ---------------------------------------------------------------- construction
    //
    // Four ways in, from "just give it a URL" (Optuna-identical) to "here is the client manager I
    // already built". The IRedisClientsManager overload is the full-control path — your choice of manager
    // (RedisManagerPool / PooledRedisClientManager / BasicRedisClientManager) with any pool config, and
    // any timeouts/retry via RedisConfig. The endpoint/callback overloads are a convenience over
    // RedisManagerPool: they carry the endpoint's routing/auth (host/port/Ssl/Db/Password/Username), but
    // ServiceStack's pooled managers take timeouts from RedisConfig, not the per-endpoint value (see the
    // type-level remarks).

    /// <summary>Connects from a <c>redis-py</c>-style URL — the Optuna-identical entry point.</summary>
    /// <param name="url">URL of the redis storage (ie: <c>redis://localhost:6379</c>); <c>redis://</c>
    ///   and <c>rediss://</c> (TLS) are accepted (ServiceStack.Redis does not support unix sockets). Only
    ///   the components <c>redis-py</c> puts on the URL are read; for ServiceStack.Redis-specific options
    ///   use one of the other constructors.</param>
    /// <param name="useCluster">If <c>false</c> (default), standalone + atomic Lua <c>EVAL</c>; if
    ///   <c>true</c>, cluster + non-atomic <c>INCR</c>+<c>SET</c>.</param>
    /// <param name="prefix">Prefix of the preserved keys.</param>
    /// <remarks>The backend owns the client pool it builds here: <see cref="Dispose" /> closes it.</remarks>
    public ServiceStackJournalRedisBackend(string url, bool useCluster = false, string prefix = "")
        : this(CreateManager(RedisUrl.Parse(url)), url, useCluster, prefix, ownsManager: true)
    {
    }

    /// <summary>
    ///     Connects from a <c>redis-py</c>-style URL, then hands you the derived ServiceStack.Redis
    ///     <see cref="SsRedisEndpoint" /> to adjust before the client manager is built — the easy way to
    ///     keep the familiar URL yet set endpoint settings it cannot express (<c>Ssl</c>, <c>Db</c>,
    ///     <c>Client</c>, …). See the type-level remarks for what the pooled manager applies (routing/auth)
    ///     versus what comes from <c>RedisConfig</c> (timeouts), and why <c>NamespacePrefix</c> must be
    ///     left unset.
    /// </summary>
    /// <param name="url">The <c>redis-py</c>-style base URL (see the URL-only constructor).</param>
    /// <param name="configure">Invoked on the <see cref="SsRedisEndpoint" /> parsed from
    ///   <paramref name="url" /> immediately before the client manager is built; mutate it in place.</param>
    /// <param name="useCluster">Append strategy (see the URL-only constructor).</param>
    /// <param name="prefix">Key prefix (see the URL-only constructor).</param>
    /// <remarks>The backend owns the client pool it builds here: <see cref="Dispose" /> closes it.</remarks>
    public ServiceStackJournalRedisBackend(string url, Action<SsRedisEndpoint> configure,
        bool useCluster = false, string prefix = "")
        : this(CreateManager(RedisUrl.Parse(url), configure), url, useCluster, prefix, ownsManager: true)
    {
    }

    /// <summary>
    ///     Builds a pooled client from a ServiceStack.Redis <see cref="SsRedisEndpoint" /> you specify in
    ///     full — every endpoint knob is yours. (Uses a <see cref="RedisManagerPool" />; for a different
    ///     manager type or pool config, use the <see cref="IRedisClientsManager" /> constructor.)
    /// </summary>
    /// <param name="endpoint">A ServiceStack.Redis endpoint.</param>
    /// <param name="useCluster">Append strategy (see the URL-only constructor).</param>
    /// <param name="prefix">Key prefix (see the URL-only constructor).</param>
    /// <exception cref="System.ArgumentNullException"><paramref name="endpoint" /> is null.</exception>
    /// <remarks>The backend owns the client pool it builds here: <see cref="Dispose" /> closes it.</remarks>
    public ServiceStackJournalRedisBackend(SsRedisEndpoint endpoint, bool useCluster = false, string prefix = "")
        : this(PoolFor(endpoint ?? throw new ArgumentNullException(nameof(endpoint))), "", useCluster, prefix, ownsManager: true)
    {
    }

    /// <summary>
    ///     Binds to a ServiceStack.Redis <see cref="IRedisClientsManager" /> you already built — the full
    ///     escape hatch. Manager type, pool size, redis config and lifetime are all yours (a pool shared
    ///     across your app, DI-managed, …); the backend only borrows clients from it to issue journal
    ///     commands. <b>You own the manager's lifetime</b> — the backend never disposes a manager it did
    ///     not create.
    /// </summary>
    /// <param name="manager">A ServiceStack.Redis client manager (e.g. <see cref="RedisManagerPool" />,
    ///   <c>PooledRedisClientManager</c>, <c>BasicRedisClientManager</c>).</param>
    /// <param name="useCluster">Append strategy (see the URL-only constructor).</param>
    /// <param name="prefix">Key prefix (see the URL-only constructor).</param>
    /// <exception cref="System.ArgumentNullException"><paramref name="manager" /> is null.</exception>
    /// <remarks><see cref="Dispose" /> leaves an injected manager open: its lifetime stays yours.</remarks>
    public ServiceStackJournalRedisBackend(IRedisClientsManager manager, bool useCluster = false, string prefix = "")
        : this(manager ?? throw new ArgumentNullException(nameof(manager)), "", useCluster, prefix, ownsManager: false)
    {
    }

    /// <summary>Binds the shared Redis connection, key namespace, and append strategy for all constructor paths.</summary>
    /// <param name="manager">The Redis connection manager retained for journal operations.</param>
    /// <param name="url">The database connection URL to resolve.</param>
    /// <param name="useCluster">Whether to use separate counter and log writes for a Redis cluster rather than the standalone Lua script.</param>
    /// <param name="prefix">The journal key namespace, shared by its counter, log records, and snapshot.</param>
    /// <param name="ownsManager">Whether this backend built <paramref name="manager" /> and must close it on <see cref="Dispose" /> or <see cref="DisposeAsync" />.</param>
    private ServiceStackJournalRedisBackend(IRedisClientsManager manager, string url, bool useCluster, string prefix, bool ownsManager)
    {
        // Optuna decorates JournalRedisBackend with @experimental_class("3.1.0"), which raises an
        // ExperimentalWarning on construction. OptunaSharp targets the Optuna 5.x surface and treats
        // it as stable, so that warning is intentionally omitted (deliberate divergence).
        _url = url;
        _manager = manager;
        _useCluster = useCluster;
        _prefix = prefix;
        _ownsManager = ownsManager;
    }

    /// <summary>
    ///     Closes the client pool this backend built from a URL or endpoint; an injected manager is left open.
    ///     Calling it again does nothing. A journal operation after disposal fails in ServiceStack.Redis.
    /// </summary>
    /// <remarks>
    ///     A DI container calls this when the scope (or, for a singleton, the container) that created the backend is
    ///     disposed synchronously, so pooled connections close with the scope instead of waiting for garbage
    ///     collection. A <see cref="JournalStorage" /> over this backend must not be used afterwards. In async code
    ///     prefer <see cref="DisposeAsync" />.
    /// </remarks>
    public void Dispose()
    {
        // Only the first call closes, whichever of Dispose and DisposeAsync it is; DI may dispose a forwarded
        // registration and the backend itself.
        if (System.Threading.Interlocked.Exchange(ref _disposed, 1) != 0)
            return;
        if (_ownsManager)
            _manager.Dispose();
        GC.SuppressFinalize(this);
    }

    /// <summary>
    ///     Closes the client pool this backend built from a URL or endpoint through ServiceStack's asynchronous close;
    ///     an injected manager is left open. This is the asynchronous form of <see cref="Dispose" />, used by
    ///     <c>await using</c> and by DI scopes that are disposed asynchronously (ASP.NET Core request scopes,
    ///     <c>AsyncServiceScope</c>).
    /// </summary>
    /// <returns>A task that completes once the owned pool is closed, or at once when there is nothing to close.</returns>
    /// <remarks>
    ///     Shares <see cref="Dispose" />'s once-only flag: whichever runs first closes the pool, and the other then does
    ///     nothing. A DI container calls this method when it disposes a scope asynchronously, and
    ///     <see cref="Dispose" /> when it disposes one synchronously; both paths are supported.
    /// </remarks>
    public async ValueTask DisposeAsync()
    {
        if (System.Threading.Interlocked.Exchange(ref _disposed, 1) != 0)
            return;
        if (_ownsManager)
        {
            // The pool this backend builds (RedisManagerPool) implements IAsyncDisposable, but the interface it is held
            // through (IRedisClientsManager) does not, so the instance is tested. A manager without an async close is
            // closed synchronously instead, so disposal never silently skips it.
            if (_manager is IAsyncDisposable asyncManager)
                await asyncManager.DisposeAsync().ConfigureAwait(false);
            else
                _manager.Dispose();
        }
        GC.SuppressFinalize(this);
    }

    /// <inheritdoc />
    /// <exception cref="NumSharp.ValueError">A nonfinal journal record contains malformed JSON.</exception>
    public override IEnumerable<OrderedDict<string, object?>> ReadLogs(int logNumberFrom)
    {
        string? maxStr = GetValue(RedisJournalProtocol.LogNumberKey(_prefix));
        if (maxStr is null)
            yield break;
        long maxLogNumber = long.Parse(maxStr, CultureInfo.InvariantCulture);

        for (long logNumber = logNumberFrom; logNumber <= maxLogNumber; logNumber++)
        {
            // Cluster append reserves a sequence number before writing its record. A visible counter
            // can therefore lead the record; wait for that writer so replay never skips an operation.
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
                // Only the final record may still be incomplete; corruption earlier in the journal
                // cannot be deferred without changing the replayed sequence of storage operations.
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

    /// <summary>Replaces the Redis journal snapshot with the supplied opaque serialized bytes.</summary>
    /// <param name="snapshot">The serialized journal-state snapshot to store as opaque bytes.</param>
    public void SaveSnapshot(byte[] snapshot)
    {
        using IRedisClient client = _manager.GetClient();
        ((IRedisNativeClient)client).Set(RedisJournalProtocol.SnapshotKey(_prefix), snapshot);
    }

    /// <summary>Reads the latest Redis journal snapshot, if one has been saved.</summary>
    /// <returns>The saved snapshot bytes, or null when the snapshot key is absent.</returns>
    public byte[]? LoadSnapshot()
    {
        using IRedisClient client = _manager.GetClient();
        // ServiceStack's native GET returns null for a missing key — same as the other backends'
        // RedisValue.IsNull, so an absent snapshot is null and a present one is its exact bytes.
        return ((IRedisNativeClient)client).Get(RedisJournalProtocol.SnapshotKey(_prefix));
    }

    /// <summary>Reads the string value of a Redis key through a borrowed pooled connection.</summary>
    /// <param name="key">The attribute name or exact storage key to access.</param>
    /// <returns>The Redis value, or null when the key is absent.</returns>
    private string? GetValue(string key)
    {
        using IRedisClient client = _manager.GetClient();
        return client.GetValue(key);   // null when the key does not exist
    }

    /// <summary>Builds a pooled ServiceStack client manager from the neutral <see cref="RedisEndpoint" />.</summary>
    /// <param name="ep">The client-neutral, parsed Redis endpoint.</param>
    /// <returns>A ServiceStack client pool configured for the parsed endpoint.</returns>
    internal static IRedisClientsManager CreateManager(RedisEndpoint ep)
        => PoolFor(ToServiceStackEndpoint(ep));

    /// <summary>
    ///     Maps the neutral endpoint onto ServiceStack.Redis' own, applies the caller's native
    ///     <paramref name="configure" /> tweak, then builds the pool.
    /// </summary>
    /// <param name="ep">The client-neutral, parsed Redis endpoint.</param>
    /// <param name="configure">The callback that customizes native connection options before opening the connection.</param>
    /// <returns>A ServiceStack client pool configured for the parsed endpoint.</returns>
    /// <exception cref="System.ArgumentNullException"><paramref name="configure" /> is null.</exception>
    internal static IRedisClientsManager CreateManager(RedisEndpoint ep, Action<SsRedisEndpoint> configure)
    {
        if (configure is null)
            throw new ArgumentNullException(nameof(configure));
        SsRedisEndpoint sep = ToServiceStackEndpoint(ep);
        configure(sep);
        return PoolFor(sep);
    }

    /// <summary>Maps the neutral <see cref="RedisEndpoint" /> onto ServiceStack.Redis' own endpoint type.</summary>
    /// <param name="ep">The client-neutral, parsed Redis endpoint.</param>
    /// <returns>The ServiceStack endpoint matching the parsed TCP/TLS connection settings.</returns>
    /// <exception cref="NumSharp.ValueError">The URL selects a Unix-domain socket, which this backend does not support.</exception>
    private static SsRedisEndpoint ToServiceStackEndpoint(RedisEndpoint ep)
    {
        if (ep.UnixSocketPath is not null)
            throw new ValueError("ServiceStack.Redis does not support unix domain sockets; use redis:// or rediss://.");

        return new SsRedisEndpoint(ep.Host, ep.Port <= 0 ? 6379 : ep.Port)
        {
            Ssl = ep.Ssl,
            Db = ep.Db ?? 0,
            Password = ep.Password,
            Username = ep.User,
        };
    }

    /// <summary>
    ///     Builds the pooled client manager for a ServiceStack.Redis endpoint. <see cref="RedisManagerPool" />
    ///     takes host <em>strings</em> (there is no <see cref="SsRedisEndpoint" />-object constructor), so
    ///     the endpoint round-trips through its own connection-string form (<c>ToString()</c>). That string
    ///     carries the endpoint's routing/auth (host, port, <c>Ssl</c>, <c>Db</c>, <c>Password</c>,
    ///     <c>Username</c>); timeouts/retry are taken from the process-wide <c>RedisConfig</c> defaults, not
    ///     the per-endpoint value (a ServiceStack pooled-manager behavior). See the type-level remarks.
    /// </summary>
    /// <param name="sep">The configured ServiceStack Redis endpoint.</param>
    /// <returns>The newly created ServiceStack client pool.</returns>
    private static IRedisClientsManager PoolFor(SsRedisEndpoint sep)
        => new RedisManagerPool(sep.ToString());
}

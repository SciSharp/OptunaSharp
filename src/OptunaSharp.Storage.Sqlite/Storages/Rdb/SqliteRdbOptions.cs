using System;
using System.Collections.Generic;
using Microsoft.Data.Sqlite;

namespace OptunaSharp.Storages;

/// <summary>
///     SQLite journal mode (<c>PRAGMA journal_mode</c>). <see cref="Wal" /> (write-ahead logging) is
///     the usual choice for better concurrency; the SQLite default is <see cref="Delete" />.
/// </summary>
public enum SqliteJournalMode
{
    /// <summary>Deletes the rollback journal after committing.</summary>
    Delete,
    /// <summary>Truncates the rollback journal after committing.</summary>
    Truncate,
    /// <summary>Invalidates the rollback-journal header while retaining its file.</summary>
    Persist,
    /// <summary>Keeps the rollback journal in memory.</summary>
    Memory,
    /// <summary>Uses a write-ahead log rather than a rollback journal.</summary>
    Wal,
    /// <summary>Disables rollback-journal creation.</summary>
    Off,
}

/// <summary>Durability vs. speed trade-off (<c>PRAGMA synchronous</c>).</summary>
public enum SqliteSynchronous
{
    /// <summary>Disables synchronization of database writes to durable storage.</summary>
    Off,
    /// <summary>Uses SQLite's normal synchronization policy for the selected journal mode.</summary>
    Normal,
    /// <summary>Requests full synchronization of critical database writes.</summary>
    Full,
    /// <summary>Adds directory synchronization where supported by the journal mode.</summary>
    Extra,
}

/// <summary>Where temporary tables and indices are stored (<c>PRAGMA temp_store</c>).</summary>
public enum SqliteTempStore
{
    /// <summary>Uses the SQLite build's default temporary-storage policy.</summary>
    Default,
    /// <summary>Stores temporary tables and indexes in files.</summary>
    File,
    /// <summary>Stores temporary tables and indexes in memory.</summary>
    Memory,
}

/// <summary>Auto-vacuum behavior (<c>PRAGMA auto_vacuum</c>). Only settable on a fresh database.</summary>
public enum SqliteAutoVacuum
{
    /// <summary>Leaves free pages available for reuse until an explicit vacuum.</summary>
    None,
    /// <summary>Reclaims free pages at each commit.</summary>
    Full,
    /// <summary>Records reclaimable pages for explicit incremental-vacuum operations.</summary>
    Incremental,
}

/// <summary>
///     Full, "golden-standard" configuration surface for <see cref="RDBStorage" /> — beyond Optuna's
///     own <c>RDBStorage(url, engine_kwargs=…)</c> API. It bundles the <see cref="RDBStorage" />
///     (Optuna) knobs with the <em>entire</em> SQLite configuration surface: every
///     <see cref="SqliteConnectionStringBuilder" /> option and the load-bearing runtime
///     <c>PRAGMA</c>s, plus two escape hatches (<see cref="ConfigureConnectionString" /> and
///     <see cref="ConfigureConnection" />) for anything not modelled here.
///
///     <para>Every property is opt-in: a <c>null</c> value leaves SQLite's own default in place. The
///     one exception is <see cref="BusyTimeoutMs" />, which defaults to <c>30000</c> so concurrent
///     writers wait rather than fail immediately — set it to <c>null</c> to leave the SQLite default.</para>
///
///     <para>Connection-string options apply only when OptunaSharp builds the connection (the
///     <c>url</c> and <see cref="SqliteConnectionStringBuilder" /> constructors); they are ignored
///     when you inject your own <see cref="SqliteConnection" />. The <c>PRAGMA</c> options and both
///     escape hatches apply in every case.</para>
/// </summary>
public sealed class SqliteRdbOptions
{
    // ------------------------------------------------------------------ RDBStorage (Optuna) knobs

    /// <summary>Skip the alembic schema-compatibility check (Python <c>skip_compatibility_check</c>).</summary>
    public bool SkipCompatibilityCheck { get; set; }

    /// <summary>Skip creating the tables — assume they already exist (Python <c>skip_table_creation</c>).</summary>
    public bool SkipTableCreation { get; set; }

    /// <summary>Heartbeat recording interval in seconds (experimental). Must be positive when set.</summary>
    public int? HeartbeatInterval { get; set; }

    /// <summary>
    ///     Grace period in seconds before a running trial is failed from its last heartbeat.
    ///     Defaults to <c>2 * HeartbeatInterval</c> when <c>null</c>. Must be positive when set.
    /// </summary>
    public int? GracePeriod { get; set; }

    /// <summary>Callback invoked after each heartbeat-stale trial is failed.</summary>
    public Action<Study, FrozenTrial>? HeartbeatStaleTrialCallback { get; set; }

    /// <summary>
    ///     Deprecated alias of <see cref="HeartbeatStaleTrialCallback" /> (Python
    ///     <c>failed_trial_callback</c>, deprecated in v4.9.0, removed in v6.0.0). Specifying both this
    ///     and <see cref="HeartbeatStaleTrialCallback" /> is an error; specifying this alone is used as
    ///     the effective callback and emits a deprecation warning. Use
    ///     <see cref="HeartbeatStaleTrialCallback" /> instead.
    /// </summary>
    public Action<Study, FrozenTrial>? FailedTrialCallback { get; set; }

    // ------------------------------------------------------------------ connection-string options

    /// <summary>
    ///     Open mode (<c>Mode=</c>). Defaults to <see cref="SqliteOpenMode.ReadWriteCreate" />, or
    ///     <see cref="SqliteOpenMode.Memory" /> for a <c>:memory:</c> data source.
    /// </summary>
    public SqliteOpenMode? Mode { get; set; }

    /// <summary>Cache mode (<c>Cache=</c>) — e.g. <see cref="SqliteCacheMode.Shared" /> for a shared in-memory database.</summary>
    public SqliteCacheMode? Cache { get; set; }

    /// <summary>Connection pooling (<c>Pooling=</c>).</summary>
    public bool? Pooling { get; set; }

    /// <summary>Encryption key (<c>Password=</c>) — requires an encryption-capable SQLite provider (e.g. SQLCipher).</summary>
    public string? Password { get; set; }

    /// <summary>Recursive triggers connection-string flag (<c>Recursive Triggers=</c>).</summary>
    public bool? RecursiveTriggers { get; set; }

    /// <summary>
    ///     Default <em>command</em> timeout in seconds (<c>Default Timeout=</c>) — how long a command
    ///     retries on a busy database. Distinct from <see cref="BusyTimeoutMs" /> (the SQLite-level
    ///     busy handler).
    /// </summary>
    /// <remarks>
    ///     A storage URL's <c>?timeout=</c> sets this too (rounded up to whole seconds, at least one) when it is
    ///     left <see langword="null" /> here: Microsoft.Data.Sqlite retries a busy command until this timeout on top
    ///     of SQLite's busy handler, so the URL's wait would otherwise stretch to the 30-second default. A value set
    ///     here wins over the URL, as <c>connect_args</c> win over the URL query in SQLAlchemy.
    /// </remarks>
    public int? DefaultTimeoutSeconds { get; set; }

    /// <summary>
    ///     Final tweak of the <see cref="SqliteConnectionStringBuilder" /> before the connection is
    ///     opened — the escape hatch for any connection-string option not modelled above. Runs after
    ///     all the typed connection-string properties.
    /// </summary>
    public Action<SqliteConnectionStringBuilder>? ConfigureConnectionString { get; set; }

    // ------------------------------------------------------------------ PRAGMA options

    /// <summary><c>PRAGMA journal_mode</c> — e.g. <see cref="SqliteJournalMode.Wal" />.</summary>
    public SqliteJournalMode? JournalMode { get; set; }

    /// <summary><c>PRAGMA synchronous</c>.</summary>
    public SqliteSynchronous? Synchronous { get; set; }

    /// <summary>
    ///     <c>PRAGMA foreign_keys</c> — enforce foreign keys at the DB level. <b>Defaults to
    ///     <c>false</c></b> to match Optuna's Python behavior (foreign-key enforcement off; cascades
    ///     happen in application code, as does <see cref="RDBStorage" />'s <c>DeleteStudy</c>) — and
    ///     to stay deterministic regardless of the bundled SQLite build's own default (the
    ///     <c>e_sqlite3</c> binary ships with foreign keys <em>on</em>). Turning it on is safe with
    ///     <see cref="RDBStorage" />'s explicit, deepest-first cascade. Set to <c>null</c> to leave
    ///     the connection's existing setting untouched.
    /// </summary>
    public bool? ForeignKeys { get; set; } = false;

    /// <summary>
    ///     <c>PRAGMA busy_timeout</c> in milliseconds — how long a locked database is retried before
    ///     giving up. Defaults to <c>30000</c>; set to <c>null</c> for the SQLite default (0).
    /// </summary>
    /// <remarks>
    ///     A storage URL's <c>?timeout=</c> (seconds, as Python's <c>sqlite3.connect(timeout=…)</c> takes it)
    ///     replaces the default, unless this property was assigned: an explicit value wins over the URL, as
    ///     <c>connect_args</c> win over the URL query in SQLAlchemy. Python's own default is 5 seconds; OptunaSharp
    ///     keeps 30 so that concurrent writers wait rather than fail.
    /// </remarks>
    public int? BusyTimeoutMs
    {
        get => _busyTimeoutMs;
        set
        {
            _busyTimeoutMs = value;
            BusyTimeoutMsSpecified = true;
        }
    }

    private int? _busyTimeoutMs = 30_000;

    /// <summary>
    ///     Whether <see cref="BusyTimeoutMs" /> was assigned (even to its default), which makes it take precedence
    ///     over a storage URL's <c>?timeout=</c>.
    /// </summary>
    internal bool BusyTimeoutMsSpecified { get; private set; }

    /// <summary><c>PRAGMA cache_size</c> (pages if positive, kibibytes if negative).</summary>
    public int? CacheSize { get; set; }

    /// <summary><c>PRAGMA page_size</c> in bytes — only takes effect on a fresh database (before any table).</summary>
    public int? PageSize { get; set; }

    /// <summary><c>PRAGMA mmap_size</c> in bytes — the memory-mapped I/O window.</summary>
    public long? MmapSize { get; set; }

    /// <summary><c>PRAGMA temp_store</c>.</summary>
    public SqliteTempStore? TempStore { get; set; }

    /// <summary><c>PRAGMA auto_vacuum</c> — only settable on a fresh database.</summary>
    public SqliteAutoVacuum? AutoVacuum { get; set; }

    /// <summary><c>PRAGMA secure_delete</c>.</summary>
    public bool? SecureDelete { get; set; }

    /// <summary>
    ///     Extra <c>PRAGMA</c> bodies applied verbatim after the typed ones, e.g.
    ///     <c>"wal_autocheckpoint=1000"</c> (each is run as <c>PRAGMA &lt;body&gt;</c>). The escape
    ///     hatch for any PRAGMA not modelled above.
    /// </summary>
    public IReadOnlyList<string>? AdditionalPragmas { get; set; }

    /// <summary>
    ///     Arbitrary configuration of the open <see cref="SqliteConnection" /> — the ultimate escape
    ///     hatch, run after every PRAGMA above and before any schema work. Use for registering
    ///     collations/functions, running non-PRAGMA setup SQL, etc.
    /// </summary>
    public Action<SqliteConnection>? ConfigureConnection { get; set; }
}

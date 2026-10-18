using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Data.Common;
using System.Linq;
using System.Text;
using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Storage;
using MySqlConnector;
using Npgsql;
using Npgsql.NameTranslation;
using NumSharp;
using OptunaSharp.Collections;
using OptunaSharp.Distributions;
using OptunaSharp.Internal;
using OptunaSharp.Logging;
using OptunaSharp.Storages;

namespace OptunaSharp.Storages.EFCore;

/// <summary>
///     RDB storage on PostgreSQL and MySQL, over Entity Framework Core. Port of
///     <c>optuna/storages/_rdb/storage.py: RDBStorage</c> for server databases — the EF Core sibling
///     of <c>OptunaSharp.Storage.Sqlite</c>'s hand-SQL <c>RDBStorage</c>. It maps Optuna's SQLAlchemy
///     models one-to-one (see <see cref="OptunaRdbContext" />) against the exact schema SQLAlchemy
///     generates (created by <see cref="RdbSchema" />), with the <c>alembic_version</c> =
///     <c>v3.2.0.a</c> + <c>schema_version</c> = 12 pins, so Python Optuna and OptunaSharp can share
///     the same PostgreSQL/MySQL database.
///
///     <para>Like Optuna, this backend is meant to be wrapped in a <see cref="CachedStorage" />: it
///     implements <see cref="ICachedStorageBackend" /> and
///     <see cref="StorageFunctions.GetStorage(BaseStorage?)" /> wraps it automatically (Python's
///     <c>get_storage</c> returns <c>_CachedStorage(RDBStorage(...))</c>).</para>
///
///     <para><b>Concurrency.</b> A short-lived <see cref="OptunaRdbContext" /> is created per operation
///     (EF Core contexts are not thread-safe); the read-modify-write paths Optuna guards with
///     <c>SELECT … FOR UPDATE</c> take the same row lock (PostgreSQL/MySQL support it, unlike SQLite),
///     and duplicate-key races are ignored exactly as Optuna's <c>ignore_integrity_error</c> does.</para>
/// </summary>
/// <example>
///     <code>
///     using var storage = new RDBStorage("postgresql://user:pass@localhost:5432/optuna");
///     var study = Optuna.CreateStudy(storage: storage);
///     study.Optimize(t => Math.Pow(t.SuggestFloat("x", -100, 100), 2), nTrials: 10);
///     </code>
/// </example>
public sealed class RDBStorage : BaseStorage, ICachedStorageBackend, IDisposable
{
    private static readonly Logger _logger = OptunaLogging.GetLogger("optuna.storages._rdb.storage");

    private RdbProvider _provider;
    private DbContextOptions<OptunaRdbContext> _ctxOptions = null!;
    private NpgsqlDataSource? _dataSource;
    private bool _ownsDataSource;

    private int? _heartbeatInterval;
    private int? _gracePeriod;
    private Action<Study, FrozenTrial>? _heartbeatStaleTrialCallback;

    /// <summary>The storage URL/connection string this storage was opened with.</summary>
    public string Url { get; private set; } = "";

    /// <summary>The RDB provider (PostgreSQL or MySQL) resolved from the URL.</summary>
    public RdbProvider Provider => _provider;

    /// <summary>
    ///     Create an RDB storage from a SQLAlchemy-style URL — the Optuna-faithful constructor
    ///     (<c>optuna.storages.RDBStorage(url, …)</c>). The scheme selects the provider:
    ///     <c>postgresql[+driver]://…</c> → PostgreSQL (Npgsql), <c>mysql[+driver]://…</c> → MySQL
    ///     (Pomelo). SQLite URLs are rejected (use <c>OptunaSharp.Storage.Sqlite</c>).
    /// </summary>
    /// <param name="url">The database connection URL to resolve.</param>
    /// <param name="skipCompatibilityCheck">Whether to skip checking the stored Alembic revision against the supported schema.</param>
    /// <param name="skipTableCreation">Whether to assume schema tables already exist.</param>
    /// <param name="heartbeatInterval">Heartbeat recording interval in seconds, or null to disable recording; positive when supplied.</param>
    /// <param name="gracePeriod">Seconds after the latest heartbeat before a running trial is stale; defaults to twice the heartbeat interval.</param>
    /// <param name="heartbeatStaleTrialCallback">Callback invoked after a stale trial is failed, or null for no callback.</param>
    /// <param name="failedTrialCallback">Deprecated alias of the stale-trial callback; cannot be supplied together with its replacement.</param>
    public RDBStorage(
        string url,
        bool skipCompatibilityCheck = false,
        bool skipTableCreation = false,
        int? heartbeatInterval = null,
        int? gracePeriod = null,
        Action<Study, FrozenTrial>? heartbeatStaleTrialCallback = null,
        Action<Study, FrozenTrial>? failedTrialCallback = null)
        : this(url, new EFCoreRdbOptions
        {
            SkipCompatibilityCheck = skipCompatibilityCheck,
            SkipTableCreation = skipTableCreation,
            HeartbeatInterval = heartbeatInterval,
            GracePeriod = gracePeriod,
            HeartbeatStaleTrialCallback = heartbeatStaleTrialCallback,
            FailedTrialCallback = failedTrialCallback,
        })
    {
    }

    /// <summary>Create an RDB storage from a URL with the full <see cref="EFCoreRdbOptions" /> surface.</summary>
    /// <param name="url">The database connection URL to resolve.</param>
    /// <param name="options">The storage and provider configuration applied before schema initialization.</param>
    public RDBStorage(string url, EFCoreRdbOptions options)
    {
        ArgumentNullException.ThrowIfNull(url);
        ArgumentNullException.ThrowIfNull(options);
        ValidateOptions(options);
        var (provider, builder) = RdbUrl.Parse(FillStorageUrlTemplate(url));
        // ConfigureConnectionString is applied once, in InitFromConnectionString, for every path.
        InitFromConnectionString(provider, builder.ConnectionString, options, url);
    }

    /// <summary>
    ///     Create an RDB storage from an explicit provider + native ADO.NET connection string (BYO
    ///     connection string). The "feel like the client, not Optuna" form.
    /// </summary>
    /// <param name="provider">The relational database dialect whose connection and SQL rules apply.</param>
    /// <param name="connectionString">The native provider connection string.</param>
    /// <param name="options">The storage and provider configuration applied before schema initialization.</param>
    public RDBStorage(RdbProvider provider, string connectionString, EFCoreRdbOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(connectionString);
        options ??= new EFCoreRdbOptions();
        ValidateOptions(options);
        InitFromConnectionString(provider, connectionString, options, connectionString);
    }

    /// <summary>
    ///     Create a PostgreSQL RDB storage from a native <see cref="NpgsqlConnectionStringBuilder" /> —
    ///     the golden-standard "feel like Npgsql" form. Every Npgsql connection-string knob (pooling,
    ///     SSL, timeouts, …) is yours to set on the builder.
    /// </summary>
    /// <param name="connectionStringBuilder">The native provider connection-string builder, customized before the connection opens.</param>
    /// <param name="options">The storage and provider configuration applied before schema initialization.</param>
    public RDBStorage(NpgsqlConnectionStringBuilder connectionStringBuilder, EFCoreRdbOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(connectionStringBuilder);
        options ??= new EFCoreRdbOptions();
        ValidateOptions(options);
        string cs = connectionStringBuilder.ConnectionString;
        InitFromConnectionString(RdbProvider.PostgreSql, cs, options, cs);
    }

    /// <summary>
    ///     Create a MySQL RDB storage from a native <see cref="MySqlConnectionStringBuilder" /> — the
    ///     golden-standard "feel like MySqlConnector" form.
    /// </summary>
    /// <param name="connectionStringBuilder">The native provider connection-string builder, customized before the connection opens.</param>
    /// <param name="options">The storage and provider configuration applied before schema initialization.</param>
    public RDBStorage(MySqlConnectionStringBuilder connectionStringBuilder, EFCoreRdbOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(connectionStringBuilder);
        options ??= new EFCoreRdbOptions();
        ValidateOptions(options);
        string cs = connectionStringBuilder.ConnectionString;
        InitFromConnectionString(RdbProvider.MySql, cs, options, cs);
    }

    /// <summary>
    ///     Create a PostgreSQL RDB storage over a <see cref="NpgsqlDataSource" /> you built and own —
    ///     the fullest "feel like Npgsql" form (your own pooling, logging, type mappings, …). The data
    ///     source <b>must have Optuna's enum types mapped</b>; build it with
    ///     <see cref="MapOptunaEnums(NpgsqlDataSourceBuilder)" />:
    ///     <code>
    ///     var dsb = new NpgsqlDataSourceBuilder(connString);
    ///     RDBStorage.MapOptunaEnums(dsb);
    ///     dsb.UseLoggerFactory(myLoggerFactory);      // …any Npgsql configuration you like
    ///     using var storage = new RDBStorage(dsb.Build(), ownsDataSource: true);
    ///     </code>
    ///     <para><b>.NET 8 footgun:</b> reuse ONE data source for every storage on the same database. Npgsql's
    ///     EF Core 8 provider builds a separate EF service provider per data-source instance and EF Core
    ///     fails every new context in the process once more than twenty exist, so building a fresh data
    ///     source per storage breaks a long-running .NET 8 process (EF Core 9 / net10.0 is unaffected). The
    ///     URL and connection-string constructors already share one per connection string.</para>
    /// </summary>
    /// <param name="ownsDataSource">
    ///     When <c>true</c>, <see cref="Dispose" /> disposes the data source; when <c>false</c>
    ///     (default), the caller retains ownership.
    /// </param>
    /// <param name="dataSource">The PostgreSQL data source providing pooled connections and enum mappings.</param>
    /// <param name="options">The storage and provider configuration applied before schema initialization.</param>
    public RDBStorage(NpgsqlDataSource dataSource, EFCoreRdbOptions? options = null, bool ownsDataSource = false)
    {
        ArgumentNullException.ThrowIfNull(dataSource);
        options ??= new EFCoreRdbOptions();
        ValidateOptions(options);

        _provider = RdbProvider.PostgreSql;
        Url = dataSource.ConnectionString;
        SetHeartbeat(options);
        _dataSource = dataSource;
        _ownsDataSource = ownsDataSource;
        _ctxOptions = BuildPostgresOptions(dataSource, options);
        Initialize(options);
    }

    // ------------------------------------------------------------------ bootstrap

    /// <summary>Resolves heartbeat configuration and the deprecated callback alias.</summary>
    /// <param name="o">The configured storage options; null-valued settings leave provider defaults unchanged.</param>
    private void SetHeartbeat(EFCoreRdbOptions o)
    {
        _heartbeatInterval = o.HeartbeatInterval;
        _gracePeriod = o.GracePeriod;
        // Python: `heartbeat_stale_trial_callback or failed_trial_callback` — the deprecated alias is
        // used only when the primary is unset (ValidateOptions rejects specifying both).
        _heartbeatStaleTrialCallback = o.HeartbeatStaleTrialCallback ?? o.FailedTrialCallback;
    }

    /// <summary>Builds provider resources from a native connection string and initializes the storage.</summary>
    /// <param name="provider">The relational database dialect whose connection and SQL rules apply.</param>
    /// <param name="connString">The native provider connection string.</param>
    /// <param name="options">The storage and provider configuration applied before schema initialization.</param>
    /// <param name="url">The database connection URL to resolve.</param>
    private void InitFromConnectionString(RdbProvider provider, string connString, EFCoreRdbOptions options, string url)
    {
        _provider = provider;
        Url = url;
        SetHeartbeat(options);

        if (provider == RdbProvider.PostgreSql)
        {
            var csb = new NpgsqlConnectionStringBuilder(connString);
            options.ConfigureConnectionString?.Invoke(csb);
            // One data source per connection string, shared — NOT one per storage: Npgsql's EF Core 8
            // provider gives every data-source instance its own EF service provider and EF Core fails the
            // whole process past ~20 of them (see SharedPostgresDataSource). Shared, so never ours to dispose.
            _dataSource = SharedPostgresDataSource(csb.ConnectionString);
            _ownsDataSource = false;
            _ctxOptions = BuildPostgresOptions(_dataSource, options);
        }
        else
        {
            // MySQL / MariaDB. AllowUserVariables + UseAffectedRows are Pomelo's recommended flags so
            // SaveChanges sees the rows it matched, not just the rows it changed.
            var csb = new MySqlConnectionStringBuilder(connString)
            {
                AllowUserVariables = true,
                UseAffectedRows = false,
            };
            options.ConfigureConnectionString?.Invoke(csb);
            _ctxOptions = BuildMySqlOptions(csb.ConnectionString, options);
        }

        Initialize(options);
    }

    /// <summary>Builds EF Core options with PostgreSQL enum mappings and user configuration.</summary>
    /// <param name="dataSource">The PostgreSQL data source providing pooled connections and enum mappings.</param>
    /// <param name="options">The storage and provider configuration applied before schema initialization.</param>
    /// <returns>Configured PostgreSQL EF Core options shared by operation-scoped contexts.</returns>
    private static DbContextOptions<OptunaRdbContext> BuildPostgresOptions(NpgsqlDataSource dataSource, EFCoreRdbOptions options)
    {
        var ob = new DbContextOptionsBuilder<OptunaRdbContext>();
        ob.UseNpgsql(dataSource, npg =>
        {
#if !NET8_0
            // EF Core 9 (net10.0): register the native enum types on the EF provider via the v9
            // options-builder MapEnum. On net8.0 (EF Core 8) that API does not exist; the model's
            // HasPostgresEnum (OptunaRdbContext.OnModelCreating) fills that role, while the data-source
            // MapEnum (MapOptunaEnums, applied to _dataSource on both TFMs) performs the value conversion.
            MapPostgresEnums(npg);
#endif
            options.ConfigureNpgsql?.Invoke(npg);
        });
        options.ConfigureDbContext?.Invoke(ob);
        return ob.Options;
    }

    /// <summary>Builds EF Core options for MySQL, detecting its server version when none is supplied.</summary>
    /// <param name="connString">The native provider connection string.</param>
    /// <param name="options">The storage and provider configuration applied before schema initialization.</param>
    /// <returns>Configured MySQL EF Core options shared by operation-scoped contexts.</returns>
    private static DbContextOptions<OptunaRdbContext> BuildMySqlOptions(string connString, EFCoreRdbOptions options)
    {
        var ob = new DbContextOptionsBuilder<OptunaRdbContext>();
        ServerVersion version = options.MySqlServerVersion ?? ServerVersion.AutoDetect(connString);
        ob.UseMySql(connString, version, my => options.ConfigureMySql?.Invoke(my));
        options.ConfigureDbContext?.Invoke(ob);
        return ob.Options;
    }

    // Port of the heartbeat validation + warnings block of `RDBStorage.__init__`
    // (optuna/storages/_rdb/storage.py) — the exact order matters (the experimental warning fires
    // before the grace-period check; the deprecation warning fires after the "specify only one" check).
    /// <summary>Rejects invalid heartbeat settings and conflicting callback aliases before opening storage.</summary>
    /// <param name="options">The storage and provider configuration applied before schema initialization.</param>
    /// <exception cref="NumSharp.ValueError">A heartbeat interval or grace period is nonpositive, or both callback aliases are supplied.</exception>
    private static void ValidateOptions(EFCoreRdbOptions options)
    {
        if (options.HeartbeatInterval is not null)
        {
            if (options.HeartbeatInterval <= 0)
                throw new ValueError("The value of `heartbeat_interval` should be a positive integer.");
            // Optuna raises an ExperimentalWarning (warn_experimental_argument) for heartbeat_interval.
            // OptunaSharp targets the Optuna 5.x surface and treats it as stable, so it is omitted.
        }
        if (options.GracePeriod is not null and <= 0)
            throw new ValueError("The value of `grace_period` should be a positive integer.");
        if (options.HeartbeatStaleTrialCallback is not null && options.FailedTrialCallback is not null)
            throw new ValueError(
                "Specify only one of `heartbeat_stale_trial_callback` and `failed_trial_callback`.");
        if (options.FailedTrialCallback is not null)
            OptunaWarnings.Warn(
                "`failed_trial_callback` has been deprecated in v4.9.0. This feature will be removed in " +
                "v6.0.0. See https://github.com/optuna/optuna/releases/tag/v4.9.0. " +
                "Use `heartbeat_stale_trial_callback` instead.",
                WarningCategory.FutureWarning);
    }

    /// <summary>Substitutes the supported schema version into an Optuna storage URL template.</summary>
    /// <param name="template">A storage URL that may contain the schema-version placeholder.</param>
    /// <returns>The URL with the schema-version placeholder replaced.</returns>
    private static string FillStorageUrlTemplate(string template)
        => template.Replace("{SCHEMA_VERSION}", RdbModels.SchemaVersion.ToString(RdbModels.Inv));

    private static readonly NpgsqlNullNameTranslator _verbatim = new();

    // The process-wide PostgreSQL data sources of the URL / connection-string constructors, one per final
    // connection string. Lazy so concurrent first opens of the same database build exactly one.
    private static readonly ConcurrentDictionary<string, Lazy<NpgsqlDataSource>> _sharedDataSources =
        new(StringComparer.Ordinal);

    /// <summary>
    ///     The data source every URL / connection-string PostgreSQL storage on <paramref name="connectionString" />
    ///     shares for the life of the process — built on first use with Optuna's enum types mapped
    ///     (<see cref="MapOptunaEnums" />), never disposed, like the per-connection-string pools of classic
    ///     ADO.NET and of MySqlConnector. Idle physical connections are still closed by Npgsql's idle pruning.
    /// </summary>
    /// <remarks>
    ///     Why shared instead of one per storage: Npgsql's EF Core 8 provider (the net8.0 build) gives every
    ///     distinct <see cref="NpgsqlDataSource" /> <i>instance</i> its own EF internal service provider
    ///     (<c>ShouldUseSameServiceProvider</c> compares it by reference), EF caches those providers for the
    ///     life of the process, and once more than twenty exist EF Core throws
    ///     <c>ManyServiceProvidersCreatedWarning</c> as an <see cref="InvalidOperationException" /> from every
    ///     new context — PostgreSQL and MySQL alike. Per-storage data sources therefore capped a .NET 8 process
    ///     at about twenty PostgreSQL storages over its whole lifetime (disposing them did not help). Sharing
    ///     bounds providers by the number of distinct databases. EF Core 9's provider (net10.0) no longer keys
    ///     on the instance, but the sharing applies on both targets so storage lifetime does not depend on the
    ///     target framework.
    /// </remarks>
    /// <param name="connectionString">The final connection string, after <see cref="EFCoreRdbOptions.ConfigureConnectionString" />.</param>
    /// <returns>The shared data source; callers must not dispose it.</returns>
    /// <exception cref="System.ArgumentException">The connection string is invalid (the failure is cached for this exact string).</exception>
    private static NpgsqlDataSource SharedPostgresDataSource(string connectionString)
        => _sharedDataSources.GetOrAdd(
            connectionString,
            cs => new Lazy<NpgsqlDataSource>(() =>
            {
                var dsb = new NpgsqlDataSourceBuilder(cs);
                MapOptunaEnums(dsb);
                return dsb.Build();
            })).Value;

    /// <summary>
    ///     Map Optuna's four native PostgreSQL enum types (<c>studydirection</c>, <c>trialstate</c>,
    ///     <c>trialvaluetype</c>, <c>trialintermediatevaluetype</c>) onto an
    ///     <see cref="NpgsqlDataSourceBuilder" />, using the enum labels' verbatim names. Call this when
    ///     building a data source to pass to <see cref="RDBStorage(NpgsqlDataSource, EFCoreRdbOptions?, bool)" />.
    /// </summary>
    /// <param name="b">The provider builder on which Optuna database enum mappings are registered.</param>
    public static void MapOptunaEnums(NpgsqlDataSourceBuilder b)
    {
        ArgumentNullException.ThrowIfNull(b);
        b.MapEnum<DbStudyDirection>("studydirection", _verbatim);
        b.MapEnum<DbTrialState>("trialstate", _verbatim);
        b.MapEnum<DbTrialValueType>("trialvaluetype", _verbatim);
        b.MapEnum<DbTrialIntermediateValueType>("trialintermediatevaluetype", _verbatim);
    }

#if !NET8_0
    /// <summary>
    ///     Registers the Optuna schema enum types on the PostgreSQL EF Core provider (net10.0 / EF Core 9).
    ///     Uses the v9-only <c>NpgsqlDbContextOptionsBuilder.MapEnum</c>; on net8.0 (EF Core 8) this API does
    ///     not exist and the model's <c>HasPostgresEnum</c> (in <see cref="OptunaRdbContext.OnModelCreating" />)
    ///     performs the equivalent EF-provider registration instead — see the <c>#if NET8_0</c> there.
    /// </summary>
    /// <param name="b">The provider builder on which Optuna database enum mappings are registered.</param>
    private static void MapPostgresEnums(Npgsql.EntityFrameworkCore.PostgreSQL.Infrastructure.NpgsqlDbContextOptionsBuilder b)
    {
        b.MapEnum<DbStudyDirection>("studydirection", nameTranslator: _verbatim);
        b.MapEnum<DbTrialState>("trialstate", nameTranslator: _verbatim);
        b.MapEnum<DbTrialValueType>("trialvaluetype", nameTranslator: _verbatim);
        b.MapEnum<DbTrialIntermediateValueType>("trialintermediatevaluetype", nameTranslator: _verbatim);
    }
#endif

    /// <summary>Creates an operation-scoped EF Core context using the immutable storage configuration.</summary>
    /// <returns>A new context that the caller must dispose after the operation.</returns>
    private OptunaRdbContext NewContext() => new(_ctxOptions);

    /// <summary>
    ///     Run a mutating operation on a fresh short-lived context under EF Core's configured
    ///     <see cref="IExecutionStrategy" />. The body opens and commits its own transaction; wrapping
    ///     it in the strategy is what lets a caller enable <c>EnableRetryOnFailure()</c> without hitting
    ///     "the configured execution strategy does not support user-initiated transactions" — and the
    ///     change tracker is cleared before each attempt so a retry starts clean. With the default
    ///     (non-retrying) strategy this runs the body exactly once.
    /// </summary>
    /// <param name="body">The operation executed in the current context or scope; its exceptions propagate to the caller.</param>
    /// <returns>The callback's result after the transaction commits successfully.</returns>
    /// <typeparam name="T">The managed result returned by the transaction callback.</typeparam>
    private T Mutate<T>(Func<OptunaRdbContext, T> body)
    {
        using var ctx = NewContext();
        return ctx.Database.CreateExecutionStrategy().Execute(() =>
        {
            ctx.ChangeTracker.Clear();
            return body(ctx);
        });
    }

    /// <summary>Runs a database mutation with the configured execution strategy and transaction ownership.</summary>
    /// <param name="body">The operation executed in the current context or scope; its exceptions propagate to the caller.</param>
    private void Mutate(Action<OptunaRdbContext> body) => Mutate<object?>(ctx =>
    {
        body(ctx);
        return null;
    });

    /// <summary>Opens and configures the storage resources, then initializes or verifies the Optuna schema.</summary>
    /// <param name="options">The storage and provider configuration applied before schema initialization.</param>
    private void Initialize(EFCoreRdbOptions options)
    {
        using var ctx = NewContext();
        if (!options.SkipTableCreation)
            RdbSchema.EnsureCreated(ctx, _provider);

        // Npgsql resolves the native ENUM type OIDs (studydirection, …) from the catalog on first
        // connection and caches them per data source. When this storage just created the types (or they
        // were created after the data source connected), that cache is stale — reload it so the enum
        // columns map, instead of failing with "the data type name 'studydirection' isn't present".
#if NET8_0
        // Npgsql 8 (net8.0) exposes no NpgsqlDataSource.ReloadTypes() (added in Npgsql 9); reload the
        // catalog by opening a short-lived connection from the data source and reloading types on it,
        // which refreshes the data source's cached type mappings just the same.
        if (_dataSource is not null)
        {
            using var reloadConn = _dataSource.OpenConnection();
            reloadConn.ReloadTypes();
        }
#else
        _dataSource?.ReloadTypes();
#endif

        InitVersionInfo(ctx);
        InitAlembic(ctx);
        if (!options.SkipCompatibilityCheck)
            CheckTableSchemaCompatibility(ctx);
    }

    /// <summary>Creates the informational schema/library version record if it is absent.</summary>
    /// <param name="ctx">The operation-scoped EF Core context; the caller owns its lifetime and transaction.</param>
    private void InitVersionInfo(OptunaRdbContext ctx)
    {
        if (ctx.VersionInfos.Any())
            return;
        ctx.VersionInfos.Add(new VersionInfoEntity
        {
            VersionInfoId = 1,
            SchemaVersion = RdbModels.SchemaVersion,
            LibraryVersion = RdbModels.LibraryVersion,
        });
        try
        {
            ctx.SaveChanges();
        }
        catch (DbUpdateException) when (ctx.VersionInfos.AsNoTracking().Any())
        {
            // Another writer inserted the singleton row first (the version_info_id=1 CHECK/PK).
            ctx.ChangeTracker.Clear();
        }
    }

    /// <summary>Initializes the Alembic migration marker for a newly created or legacy database.</summary>
    /// <param name="ctx">The operation-scoped EF Core context; the caller owns its lifetime and transaction.</param>
    private void InitAlembic(OptunaRdbContext ctx)
    {
        if (ctx.AlembicVersions.Any())
            return;
        ctx.AlembicVersions.Add(new AlembicVersionEntity { VersionNum = RdbModels.HeadVersion });
        try
        {
            ctx.SaveChanges();
        }
        catch (DbUpdateException) when (ctx.AlembicVersions.AsNoTracking().Any())
        {
            ctx.ChangeTracker.Clear();
        }
    }

    /// <summary>Checks that the database migration revision is compatible with this storage implementation.</summary>
    /// <param name="ctx">The operation-scoped EF Core context; the caller owns its lifetime and transaction.</param>
    /// <exception cref="NumSharp.RuntimeError">The Alembic marker is absent or the schema revision is incompatible.</exception>
    private static void CheckTableSchemaCompatibility(OptunaRdbContext ctx)
    {
        string? libraryVersion = ctx.VersionInfos.AsNoTracking().Select(v => v.LibraryVersion).FirstOrDefault();
        string current = ctx.AlembicVersions.AsNoTracking().Select(a => a.VersionNum).FirstOrDefault()
            ?? throw new RuntimeError("The alembic_version table is empty.");
        if (current == RdbModels.HeadVersion)
            return;

        var message = new StringBuilder(
            $"The runtime optuna version {OptunaSharpVersion.OptunaCompatibilityTarget} is no longer " +
            "compatible with the table schema " +
            $"(set up by optuna {libraryVersion}). ");
        message.Append(RdbModels.KnownVersions.Contains(current)
            ? "Please execute `$ optuna storage upgrade --storage $STORAGE_URL` for upgrading the storage."
            : "Please try updating optuna to the latest version by `$ pip install -U optuna`.");
        throw new RuntimeError(message.ToString());
    }

    // ------------------------------------------------------------------ study manipulation

    /// <inheritdoc />
    /// <exception cref="DuplicatedStudyError">Another study already uses the requested name.</exception>
    public override int CreateNewStudy(IReadOnlyList<StudyDirection> directions, string? studyName = null)
    {
        string? name = studyName;
        int studyId = Mutate(ctx =>
        {
            name ??= CreateUniqueStudyName(ctx);
            using var tx = ctx.Database.BeginTransaction();
            var study = new StudyEntity { StudyName = name };
            ctx.Studies.Add(study);
            try
            {
                ctx.SaveChanges();
                for (int objective = 0; objective < directions.Count; objective++)
                {
                    ctx.StudyDirections.Add(new StudyDirectionEntity
                    {
                        StudyId = study.StudyId,
                        Objective = objective,
                        Direction = RdbModels.DirectionToDb(directions[objective]),
                    });
                }
                ctx.SaveChanges();
                tx.Commit();
            }
            catch (DbUpdateException e) when (IsUniqueViolation(e))
            {
                tx.Rollback();
                throw new DuplicatedStudyError();
            }
            return study.StudyId;
        });

        _logger.Info($"A new study created in RDB with name: {name}");
        return studyId;
    }

    /// <summary>Generates a study name and checks the database until the name is unused.</summary>
    /// <param name="ctx">The operation-scoped EF Core context; the caller owns its lifetime and transaction.</param>
    /// <returns>A generated name that is not currently present in the studies table.</returns>
    private static string CreateUniqueStudyName(OptunaRdbContext ctx)
    {
        while (true)
        {
            string studyName = DefaultStudyNamePrefix + Guid.NewGuid();
            if (!ctx.Studies.Any(s => s.StudyName == studyName))
                return studyName;
        }
    }

    /// <inheritdoc />
    public override void DeleteStudy(int studyId) => Mutate(ctx =>
    {
        using var tx = ctx.Database.BeginTransaction();
        EnsureStudyExists(ctx, studyId);

        // Foreign keys are enforced by these servers, so children are deleted deepest-first (Optuna
        // cascades in the ORM layer). Executed as set-based deletes over the trial scope.
        ctx.Database.ExecuteSqlRaw(
            "DELETE FROM trial_params WHERE trial_id IN (SELECT trial_id FROM trials WHERE study_id = {0})", studyId);
        ctx.Database.ExecuteSqlRaw(
            "DELETE FROM trial_values WHERE trial_id IN (SELECT trial_id FROM trials WHERE study_id = {0})", studyId);
        ctx.Database.ExecuteSqlRaw(
            "DELETE FROM trial_intermediate_values WHERE trial_id IN (SELECT trial_id FROM trials WHERE study_id = {0})", studyId);
        ctx.Database.ExecuteSqlRaw(
            "DELETE FROM trial_user_attributes WHERE trial_id IN (SELECT trial_id FROM trials WHERE study_id = {0})", studyId);
        ctx.Database.ExecuteSqlRaw(
            "DELETE FROM trial_system_attributes WHERE trial_id IN (SELECT trial_id FROM trials WHERE study_id = {0})", studyId);
        ctx.Database.ExecuteSqlRaw(
            "DELETE FROM trial_heartbeats WHERE trial_id IN (SELECT trial_id FROM trials WHERE study_id = {0})", studyId);
        ctx.Database.ExecuteSqlRaw("DELETE FROM trials WHERE study_id = {0}", studyId);
        ctx.Database.ExecuteSqlRaw("DELETE FROM study_directions WHERE study_id = {0}", studyId);
        ctx.Database.ExecuteSqlRaw("DELETE FROM study_user_attributes WHERE study_id = {0}", studyId);
        ctx.Database.ExecuteSqlRaw("DELETE FROM study_system_attributes WHERE study_id = {0}", studyId);
        ctx.Database.ExecuteSqlRaw("DELETE FROM studies WHERE study_id = {0}", studyId);
        tx.Commit();
    });

    /// <inheritdoc />
    public override void SetStudyUserAttr(int studyId, string key, object? value)
        => SetStudyAttr(isUser: true, studyId, key, value);

    /// <inheritdoc />
    public override void SetStudySystemAttr(int studyId, string key, object? value)
        => SetStudyAttr(isUser: false, studyId, key, value);

    /// <summary>Upserts a JSON-serializable attribute after verifying that its study exists.</summary>
    /// <param name="isUser">Whether to use user-attribute records instead of system-attribute records.</param>
    /// <param name="studyId">The storage-wide study identifier, not its display name.</param>
    /// <param name="key">The attribute name or exact storage key to access.</param>
    /// <param name="value">The value to serialize, persist, or convert to the requested representation.</param>
    private void SetStudyAttr(bool isUser, int studyId, string key, object? value) => Mutate(ctx =>
    {
        using var tx = ctx.Database.BeginTransaction();
        EnsureStudyExists(ctx, studyId);
        string json = PyJson.Dumps(value);

        if (isUser)
        {
            var row = ctx.StudyUserAttributes.FirstOrDefault(a => a.StudyId == studyId && a.Key == key);
            if (row is null)
                ctx.StudyUserAttributes.Add(new StudyUserAttributeEntity { StudyId = studyId, Key = key, ValueJson = json });
            else
                row.ValueJson = json;
        }
        else
        {
            var row = ctx.StudySystemAttributes.FirstOrDefault(a => a.StudyId == studyId && a.Key == key);
            if (row is null)
                ctx.StudySystemAttributes.Add(new StudySystemAttributeEntity { StudyId = studyId, Key = key, ValueJson = json });
            else
                row.ValueJson = json;
        }
        ctx.SaveChanges();
        tx.Commit();
    });

    /// <inheritdoc />
    /// <exception cref="NumSharp.KeyError">No study has the requested name.</exception>
    public override int GetStudyIdFromName(string studyName)
    {
        using var ctx = NewContext();
        var id = ctx.Studies.AsNoTracking()
            .Where(s => s.StudyName == studyName).Select(s => (int?)s.StudyId).FirstOrDefault();
        if (id is null)
            throw new KeyError(RdbModels.NotFoundMsg);
        return id.Value;
    }

    /// <inheritdoc />
    /// <exception cref="NumSharp.KeyError">The study identifier does not exist.</exception>
    public override string GetStudyNameFromId(int studyId)
    {
        using var ctx = NewContext();
        var name = ctx.Studies.AsNoTracking()
            .Where(s => s.StudyId == studyId).Select(s => s.StudyName).FirstOrDefault();
        if (name is null)
            throw new KeyError(RdbModels.NotFoundMsg);
        return name;
    }

    /// <inheritdoc />
    public override List<StudyDirection> GetStudyDirections(int studyId)
    {
        using var ctx = NewContext();
        EnsureStudyExists(ctx, studyId);
        return ctx.StudyDirections.AsNoTracking()
            .Where(d => d.StudyId == studyId).OrderBy(d => d.Objective)
            .Select(d => d.Direction).AsEnumerable()
            .Select(RdbModels.DirectionFromDb).ToList();
    }

    /// <inheritdoc />
    public override OrderedDict<string, object?> GetStudyUserAttrs(int studyId)
    {
        using var ctx = NewContext();
        EnsureStudyExists(ctx, studyId);
        var attrs = new OrderedDict<string, object?>();
        foreach (var row in ctx.StudyUserAttributes.AsNoTracking()
                     .Where(a => a.StudyId == studyId).OrderBy(a => a.StudyUserAttributeId))
            attrs[row.Key!] = PyJson.Loads(row.ValueJson!);
        return attrs;
    }

    /// <inheritdoc />
    public override OrderedDict<string, object?> GetStudySystemAttrs(int studyId)
    {
        using var ctx = NewContext();
        EnsureStudyExists(ctx, studyId);
        var attrs = new OrderedDict<string, object?>();
        foreach (var row in ctx.StudySystemAttributes.AsNoTracking()
                     .Where(a => a.StudyId == studyId).OrderBy(a => a.StudySystemAttributeId))
            attrs[row.Key!] = PyJson.Loads(row.ValueJson!);
        return attrs;
    }

    /// <inheritdoc />
    public override List<FrozenStudy> GetAllStudies()
    {
        using var ctx = NewContext();
        var studyRows = ctx.Studies.AsNoTracking().OrderBy(s => s.StudyId)
            .Select(s => new { s.StudyId, s.StudyName }).ToList();

        var directions = new Dictionary<int, List<StudyDirection>>();
        foreach (var d in ctx.StudyDirections.AsNoTracking().OrderBy(x => x.StudyId).ThenBy(x => x.Objective)
                     .Select(x => new { x.StudyId, x.Direction }).AsEnumerable())
        {
            if (!directions.TryGetValue(d.StudyId, out var list))
                directions[d.StudyId] = list = new List<StudyDirection>();
            list.Add(RdbModels.DirectionFromDb(d.Direction));
        }

        var userAttrs = LoadAllStudyAttrs(ctx, isUser: true);
        var systemAttrs = LoadAllStudyAttrs(ctx, isUser: false);

        var result = new List<FrozenStudy>(studyRows.Count);
        foreach (var s in studyRows)
        {
            result.Add(new FrozenStudy(
                studyName: s.StudyName,
                direction: null,
                directions: directions.TryGetValue(s.StudyId, out var dd) ? dd : new List<StudyDirection>(),
                userAttrs: userAttrs.TryGetValue(s.StudyId, out var u) ? u : new OrderedDict<string, object?>(),
                systemAttrs: systemAttrs.TryGetValue(s.StudyId, out var sy) ? sy : new OrderedDict<string, object?>(),
                studyId: s.StudyId));
        }
        return result;
    }

    /// <summary>Loads decoded study attributes in one pass, grouped by storage-wide study identifier.</summary>
    /// <param name="ctx">The operation-scoped EF Core context; the caller owns its lifetime and transaction.</param>
    /// <param name="isUser">Whether to use user-attribute records instead of system-attribute records.</param>
    /// <returns>A dictionary mapping study identifiers to decoded attribute dictionaries.</returns>
    private static Dictionary<int, OrderedDict<string, object?>> LoadAllStudyAttrs(OptunaRdbContext ctx, bool isUser)
    {
        var byStudy = new Dictionary<int, OrderedDict<string, object?>>();
        IEnumerable<(int? StudyId, string? Key, string? Json)> rows = isUser
            ? ctx.StudyUserAttributes.AsNoTracking().OrderBy(a => a.StudyUserAttributeId)
                .Select(a => new ValueTuple<int?, string?, string?>(a.StudyId, a.Key, a.ValueJson)).AsEnumerable()
            : ctx.StudySystemAttributes.AsNoTracking().OrderBy(a => a.StudySystemAttributeId)
                .Select(a => new ValueTuple<int?, string?, string?>(a.StudyId, a.Key, a.ValueJson)).AsEnumerable();
        foreach (var (studyId, key, json) in rows)
        {
            int sid = studyId!.Value;
            if (!byStudy.TryGetValue(sid, out var dict))
                byStudy[sid] = dict = new OrderedDict<string, object?>();
            dict[key!] = PyJson.Loads(json!);
        }
        return byStudy;
    }

    // ------------------------------------------------------------------ trial creation

    /// <inheritdoc />
    public override int CreateNewTrial(int studyId, FrozenTrial? templateTrial = null)
        => CreateNewTrialFull(studyId, templateTrial).TrialId;

    /// <summary>Creates a trial and returns its complete stored snapshot, optionally copied from a template.</summary>
    /// <param name="studyId">The storage-wide study identifier, not its display name.</param>
    /// <param name="templateTrial">An optional trial snapshot whose attributes, parameters, values, and state initialize the new record.</param>
    /// <returns>The newly stored trial including its assigned identifier and study-local number.</returns>
    public FrozenTrial CreateNewTrialFull(int studyId, FrozenTrial? templateTrial = null) => Mutate(ctx =>
    {
        using var tx = ctx.Database.BeginTransaction();
        EnsureStudyExists(ctx, studyId);

        var trial = new TrialEntity { StudyId = studyId, State = RdbModels.StateToDb(TrialState.Running) };
        if (templateTrial is null)
            trial.DatetimeStart = RdbModels.ToDbUtc(DateTime.Now);
        else
        {
            trial.DatetimeStart = templateTrial.DatetimeStart is { } s ? RdbModels.ToDbUtc(s) : null;
            trial.DatetimeComplete = templateTrial.DatetimeComplete is { } c ? RdbModels.ToDbUtc(c) : null;
        }
        ctx.Trials.Add(trial);
        ctx.SaveChanges();
        int trialId = trial.TrialId;

        if (templateTrial is not null)
        {
            var values = templateTrial.Values;
            if (values is { Count: > 1 })
            {
                for (int objective = 0; objective < values.Count; objective++)
                    AddTrialValue(ctx, trialId, objective, values[objective]);
            }
            else if (templateTrial.Value is not null)
            {
                AddTrialValue(ctx, trialId, 0, templateTrial.Value.Value);
            }

            foreach (var (paramName, paramValue) in templateTrial.Params)
            {
                var distribution = templateTrial.Distributions[paramName];
                double internalRepr = distribution.ToInternalRepr(paramValue);
                ctx.TrialParams.Add(new TrialParamEntity
                {
                    TrialId = trialId,
                    ParamName = paramName,
                    ParamValue = internalRepr,
                    DistributionJson = Distribution.ToJson(distribution),
                });
            }

            foreach (var (key, value) in templateTrial.UserAttrs)
                ctx.TrialUserAttributes.Add(new TrialUserAttributeEntity { TrialId = trialId, Key = key, ValueJson = PyJson.Dumps(value) });
            foreach (var (key, value) in templateTrial.SystemAttrs)
                ctx.TrialSystemAttributes.Add(new TrialSystemAttributeEntity { TrialId = trialId, Key = key, ValueJson = PyJson.Dumps(value) });
            foreach (var (step, intermediateValue) in templateTrial.IntermediateValues)
                AddTrialIntermediateValue(ctx, trialId, step, intermediateValue);

            trial.State = RdbModels.StateToDb(templateTrial.State);
            ctx.SaveChanges();
        }

        // number = count of the study's trials created before this one (Python `count_past_trials`).
        int number = ctx.Trials.Count(t => t.StudyId == studyId && t.TrialId < trialId);
        trial.Number = number;
        ctx.SaveChanges();

        DateTime? datetimeStart = RdbModels.FromDbUtc(trial.DatetimeStart);
        tx.Commit();

        if (templateTrial is not null)
        {
            var frozen = templateTrial.DeepCopy();
            frozen.SetNumber(number);
            frozen.SetDatetimeStart(datetimeStart);
            frozen.TrialId = trialId;
            return frozen;
        }

        return new FrozenTrial(
            number: number,
            state: TrialState.Running,
            value: null,
            datetimeStart: datetimeStart,
            datetimeComplete: null,
            @params: new OrderedDict<string, object?>(),
            distributions: new OrderedDict<string, BaseDistribution>(),
            userAttrs: new OrderedDict<string, object?>(),
            systemAttrs: new OrderedDict<string, object?>(),
            intermediateValues: new OrderedDict<int, double>(),
            trialId: trialId);
    });

    // ------------------------------------------------------------------ trial params

    /// <inheritdoc />
    public override void SetTrialParam(
        int trialId, string paramName, double paramValueInternal, BaseDistribution distribution)
        => SetTrialParamWithCachedDistribution(trialId, paramName, paramValueInternal, distribution, null);

    /// <summary>Stores a parameter after checking trial mutability and distribution compatibility.</summary>
    /// <param name="trialId">The storage-wide trial identifier, distinct from its study-local number.</param>
    /// <param name="paramName">The case-sensitive parameter name.</param>
    /// <param name="paramValueInternal">The numeric internal representation defined by the parameter distribution.</param>
    /// <param name="distribution">The distribution describing the parameter and its internal representation.</param>
    /// <param name="cachedDistribution">A previously validated distribution for this parameter, or null when storage must check compatibility.</param>
    public void SetTrialParamWithCachedDistribution(
        int trialId,
        string paramName,
        double paramValueInternal,
        BaseDistribution distribution,
        BaseDistribution? cachedDistribution) => Mutate(ctx =>
    {
        using var tx = ctx.Database.BeginTransaction();
        var trial = FindTrialForUpdate(ctx, trialId);
        CheckTrialIsUpdatableOrRaise(trial);

        BaseDistribution? previousDistribution = cachedDistribution;
        if (previousDistribution is null)
        {
            string? previousJson = (
                from tp in ctx.TrialParams
                join t in ctx.Trials on tp.TrialId equals t.TrialId
                where t.StudyId == trial.StudyId && tp.ParamName == paramName
                select tp.DistributionJson).FirstOrDefault();
            if (previousJson is not null)
                previousDistribution = Distribution.FromJson(previousJson);
        }

        if (previousDistribution is not null)
            Distribution.CheckDistributionCompatibility(previousDistribution, distribution);

        // Optuna inserts (not upserts) under ignore_integrity_error=True: a duplicate
        // (trial_id, param_name) is silently ignored, so the first write for a parameter wins.
        ctx.TrialParams.Add(new TrialParamEntity
        {
            TrialId = trialId,
            ParamName = paramName,
            ParamValue = paramValueInternal,
            DistributionJson = Distribution.ToJson(distribution),
        });
        try
        {
            ctx.SaveChanges();
            tx.Commit();
        }
        catch (DbUpdateException e) when (IsUniqueViolation(e))
        {
            tx.Rollback();
        }
    });

    /// <inheritdoc />
    /// <exception cref="NumSharp.KeyError">The trial or requested parameter does not exist.</exception>
    public override double GetTrialParam(int trialId, string paramName)
    {
        using var ctx = NewContext();
        EnsureTrialExists(ctx, trialId);
        var value = ctx.TrialParams.AsNoTracking()
            .Where(p => p.TrialId == trialId && p.ParamName == paramName)
            .Select(p => p.ParamValue).FirstOrDefault();
        if (value is null)
            throw new KeyError(RdbModels.NotFoundMsg);
        return value.Value;
    }

    // ------------------------------------------------------------------ trial state / values

    /// <inheritdoc />
    public override bool SetTrialStateValues(int trialId, TrialState state, IReadOnlyList<double>? values = null) => Mutate(ctx =>
    {
        using var tx = ctx.Database.BeginTransaction();
        try
        {
            var trial = FindTrialForUpdate(ctx, trialId);
            CheckTrialIsUpdatableOrRaise(trial);

            if (values is not null)
            {
                for (int objective = 0; objective < values.Count; objective++)
                    SetTrialValueUpsert(ctx, trialId, objective, values[objective]);
            }

            if (state == TrialState.Running && RdbModels.StateFromDb(trial.State) != TrialState.Waiting)
            {
                tx.Rollback();
                return false;
            }

            trial.State = RdbModels.StateToDb(state);
            if (state == TrialState.Running)
                trial.DatetimeStart = RdbModels.ToDbUtc(DateTime.Now);
            if (state.IsFinished())
                trial.DatetimeComplete = RdbModels.ToDbUtc(DateTime.Now);

            ctx.SaveChanges();
            tx.Commit();
        }
        catch (DbUpdateException e) when (IsUniqueViolation(e))
        {
            tx.Rollback();
            return false;
        }
        return true;
    });

    /// <summary>Adds a tagged objective-value entity to the current context without saving it.</summary>
    /// <param name="ctx">The operation-scoped EF Core context; the caller owns its lifetime and transaction.</param>
    /// <param name="trialId">The storage-wide trial identifier, distinct from its study-local number.</param>
    /// <param name="objective">The zero-based objective index.</param>
    /// <param name="value">The value to serialize, persist, or convert to the requested representation.</param>
    private static void AddTrialValue(OptunaRdbContext ctx, int trialId, int objective, double value)
    {
        var (stored, type) = RdbModels.ValueToStoredRepr(value);
        ctx.TrialValues.Add(new TrialValueEntity
        {
            TrialId = trialId,
            Objective = objective,
            Value = stored,
            ValueType = type,
        });
    }

    /// <summary>Inserts or updates a tagged objective-value entity in the current context.</summary>
    /// <param name="ctx">The operation-scoped EF Core context; the caller owns its lifetime and transaction.</param>
    /// <param name="trialId">The storage-wide trial identifier, distinct from its study-local number.</param>
    /// <param name="objective">The zero-based objective index.</param>
    /// <param name="value">The value to serialize, persist, or convert to the requested representation.</param>
    private static void SetTrialValueUpsert(OptunaRdbContext ctx, int trialId, int objective, double value)
    {
        var (stored, type) = RdbModels.ValueToStoredRepr(value);
        var row = ctx.TrialValues.FirstOrDefault(v => v.TrialId == trialId && v.Objective == objective);
        if (row is null)
            ctx.TrialValues.Add(new TrialValueEntity { TrialId = trialId, Objective = objective, Value = stored, ValueType = type });
        else
        {
            row.Value = stored;
            row.ValueType = type;
        }
    }

    /// <inheritdoc />
    public override void SetTrialIntermediateValue(int trialId, int step, double intermediateValue) => Mutate(ctx =>
    {
        using var tx = ctx.Database.BeginTransaction();
        var trial = FindTrialForUpdate(ctx, trialId);
        CheckTrialIsUpdatableOrRaise(trial);

        var (stored, type) = RdbModels.IntermediateValueToStoredRepr(intermediateValue);
        var row = ctx.TrialIntermediateValues.FirstOrDefault(v => v.TrialId == trialId && v.Step == step);
        if (row is null)
            ctx.TrialIntermediateValues.Add(new TrialIntermediateValueEntity
            {
                TrialId = trialId, Step = step, IntermediateValue = stored, IntermediateValueType = type,
            });
        else
        {
            row.IntermediateValue = stored;
            row.IntermediateValueType = type;
        }
        ctx.SaveChanges();
        tx.Commit();
    });

    /// <summary>Adds a tagged intermediate-report entity to the current context without saving it.</summary>
    /// <param name="ctx">The operation-scoped EF Core context; the caller owns its lifetime and transaction.</param>
    /// <param name="trialId">The storage-wide trial identifier, distinct from its study-local number.</param>
    /// <param name="step">The nonnegative intermediate-report step.</param>
    /// <param name="intermediateValue">The reported value, including nonfinite values represented by separate database tags.</param>
    private static void AddTrialIntermediateValue(OptunaRdbContext ctx, int trialId, int step, double intermediateValue)
    {
        var (stored, type) = RdbModels.IntermediateValueToStoredRepr(intermediateValue);
        ctx.TrialIntermediateValues.Add(new TrialIntermediateValueEntity
        {
            TrialId = trialId, Step = step, IntermediateValue = stored, IntermediateValueType = type,
        });
    }

    /// <inheritdoc />
    public override void SetTrialUserAttr(int trialId, string key, object? value)
        => SetTrialAttr(isUser: true, trialId, key, value);

    /// <inheritdoc />
    public override void SetTrialSystemAttr(int trialId, string key, object? value)
        => SetTrialAttr(isUser: false, trialId, key, value);

    /// <summary>Upserts a JSON-serializable attribute after verifying that its trial is still mutable.</summary>
    /// <param name="isUser">Whether to use user-attribute records instead of system-attribute records.</param>
    /// <param name="trialId">The storage-wide trial identifier, distinct from its study-local number.</param>
    /// <param name="key">The attribute name or exact storage key to access.</param>
    /// <param name="value">The value to serialize, persist, or convert to the requested representation.</param>
    private void SetTrialAttr(bool isUser, int trialId, string key, object? value) => Mutate(ctx =>
    {
        using var tx = ctx.Database.BeginTransaction();
        var trial = FindTrialForUpdate(ctx, trialId);
        CheckTrialIsUpdatableOrRaise(trial);
        string json = PyJson.Dumps(value);

        if (isUser)
        {
            var row = ctx.TrialUserAttributes.FirstOrDefault(a => a.TrialId == trialId && a.Key == key);
            if (row is null)
                ctx.TrialUserAttributes.Add(new TrialUserAttributeEntity { TrialId = trialId, Key = key, ValueJson = json });
            else
                row.ValueJson = json;
        }
        else
        {
            var row = ctx.TrialSystemAttributes.FirstOrDefault(a => a.TrialId == trialId && a.Key == key);
            if (row is null)
                ctx.TrialSystemAttributes.Add(new TrialSystemAttributeEntity { TrialId = trialId, Key = key, ValueJson = json });
            else
                row.ValueJson = json;
        }
        ctx.SaveChanges();
        tx.Commit();
    });

    // ------------------------------------------------------------------ trial access

    /// <inheritdoc />
    /// <exception cref="NumSharp.KeyError">The requested trial number does not exist in the study.</exception>
    public override int GetTrialIdFromStudyIdTrialNumber(int studyId, int trialNumber)
    {
        using var ctx = NewContext();
        var trialId = ctx.Trials.AsNoTracking()
            .Where(t => t.Number == trialNumber && t.StudyId == studyId)
            .Select(t => (int?)t.TrialId).FirstOrDefault();
        if (trialId is null)
            throw new KeyError(
                $"No trial with trial number {trialNumber} exists in study with study_id {studyId}.");
        return trialId.Value;
    }

    /// <inheritdoc />
    public override FrozenTrial GetTrial(int trialId)
    {
        using var ctx = NewContext();
        EnsureTrialExists(ctx, trialId);
        var trials = GetTrialsInternal(ctx, studyId: null, singleTrialId: trialId, states: null,
            includedTrialIds: null, trialIdGreaterThan: -1);
        return trials[0];
    }

    /// <inheritdoc />
    public override List<FrozenTrial> GetAllTrials(
        int studyId, bool deepcopy = true, IReadOnlyCollection<TrialState>? states = null)
    {
        using var ctx = NewContext();
        EnsureStudyExists(ctx, studyId);
        return GetTrialsInternal(ctx, studyId, singleTrialId: null, states, includedTrialIds: null, trialIdGreaterThan: -1);
    }

    /// <summary>Loads explicitly included trials and trials newer than an incremental trial-id cursor.</summary>
    /// <param name="studyId">The storage-wide study identifier, not its display name.</param>
    /// <param name="states">The allowed trial states, or null to include every state.</param>
    /// <param name="includedTrialIds">Trial identifiers to include even when they are at or below the incremental cursor.</param>
    /// <param name="trialIdGreaterThan">The exclusive lower trial-id bound for incremental reads.</param>
    /// <returns>Matching trial snapshots, ordered by trial number.</returns>
    public List<FrozenTrial> GetTrialsPartial(
        int studyId,
        IReadOnlyCollection<TrialState>? states,
        IReadOnlyCollection<int> includedTrialIds,
        int trialIdGreaterThan)
    {
        using var ctx = NewContext();
        EnsureStudyExists(ctx, studyId);
        return GetTrialsInternal(ctx, studyId, singleTrialId: null, states, includedTrialIds, trialIdGreaterThan);
    }

    /// <summary>Materializes matching trial records and their parameters, values, and attributes.</summary>
    /// <param name="ctx">The operation-scoped EF Core context; the caller owns its lifetime and transaction.</param>
    /// <param name="studyId">The storage-wide study identifier, not its display name.</param>
    /// <param name="singleTrialId">An optional identifier restricting the query to one trial.</param>
    /// <param name="states">The allowed trial states, or null to include every state.</param>
    /// <param name="includedTrialIds">Trial identifiers to include even when they are at or below the incremental cursor.</param>
    /// <param name="trialIdGreaterThan">The exclusive lower trial-id bound for incremental reads.</param>
    /// <returns>Matching trial snapshots with their child records materialized.</returns>
    private static List<FrozenTrial> GetTrialsInternal(
        OptunaRdbContext ctx,
        int? studyId,
        int? singleTrialId,
        IReadOnlyCollection<TrialState>? states,
        IReadOnlyCollection<int>? includedTrialIds,
        int trialIdGreaterThan)
    {
        // Python `_get_trials`: keep only tracked ids at or below the watermark; the rest are already
        // covered by the `trial_id > trial_id_greater_than` clause.
        var included = includedTrialIds is null
            ? new List<int>()
            : includedTrialIds.Where(id => id <= trialIdGreaterThan).ToList();

        IQueryable<TrialEntity> q = ctx.Trials.AsNoTracking();
        if (singleTrialId is not null)
        {
            q = q.Where(t => t.TrialId == singleTrialId.Value);
        }
        else
        {
            int sid = studyId!.Value;
            q = q.Where(t => t.StudyId == sid);

            if (states is not null)
            {
                if (states.Count == 0)
                    return new List<FrozenTrial>();
                var dbStates = states.Select(RdbModels.StateToDb).ToList();
                q = q.Where(t => dbStates.Contains(t.State));
            }

            if (included.Count > 0 && trialIdGreaterThan > -1)
                q = q.Where(t => t.TrialId > trialIdGreaterThan || included.Contains(t.TrialId));
            else if (trialIdGreaterThan > -1)
                q = q.Where(t => t.TrialId > trialIdGreaterThan);
        }

        var trialRows = q.OrderBy(t => t.TrialId)
            .Select(t => new TrialRow(t.TrialId, t.Number, t.State, t.DatetimeStart, t.DatetimeComplete))
            .ToList();

        if (trialRows.Count == 0)
            return new List<FrozenTrial>();

        var trialIds = trialRows.Select(t => t.TrialId).ToList();
        var idSet = trialIds.ToHashSet();

        var paramsByTrial = ctx.TrialParams.AsNoTracking()
            .Where(p => idSet.Contains(p.TrialId!.Value)).OrderBy(p => p.ParamId)
            .Select(p => new { p.TrialId, p.ParamName, p.ParamValue, p.DistributionJson })
            .AsEnumerable().GroupBy(p => p.TrialId!.Value)
            .ToDictionary(g => g.Key, g => g.ToList());

        var valuesByTrial = ctx.TrialValues.AsNoTracking()
            .Where(v => idSet.Contains(v.TrialId))
            .Select(v => new { v.TrialId, v.Objective, v.Value, v.ValueType })
            .AsEnumerable().GroupBy(v => v.TrialId).ToDictionary(g => g.Key, g => g.ToList());

        var intermediateByTrial = ctx.TrialIntermediateValues.AsNoTracking()
            .Where(v => idSet.Contains(v.TrialId))
            .Select(v => new { v.TrialId, v.Step, v.IntermediateValue, v.IntermediateValueType })
            .AsEnumerable().GroupBy(v => v.TrialId).ToDictionary(g => g.Key, g => g.ToList());

        var userByTrial = ctx.TrialUserAttributes.AsNoTracking()
            .Where(a => idSet.Contains(a.TrialId!.Value)).OrderBy(a => a.TrialUserAttributeId)
            .Select(a => new { a.TrialId, a.Key, a.ValueJson })
            .AsEnumerable().GroupBy(a => a.TrialId!.Value).ToDictionary(g => g.Key, g => g.ToList());

        var systemByTrial = ctx.TrialSystemAttributes.AsNoTracking()
            .Where(a => idSet.Contains(a.TrialId!.Value)).OrderBy(a => a.TrialSystemAttributeId)
            .Select(a => new { a.TrialId, a.Key, a.ValueJson })
            .AsEnumerable().GroupBy(a => a.TrialId!.Value).ToDictionary(g => g.Key, g => g.ToList());

        var result = new List<FrozenTrial>(trialRows.Count);
        foreach (var row in trialRows)
        {
            List<double>? values = null;
            if (valuesByTrial.TryGetValue(row.TrialId, out var valueRows) && valueRows.Count > 0)
            {
                values = new List<double>(new double[valueRows.Count]);
                foreach (var v in valueRows)
                    values[v.Objective] = RdbModels.StoredReprToValue(v.Value, v.ValueType);
            }

            var @params = new OrderedDict<string, object?>();
            var distributions = new OrderedDict<string, BaseDistribution>();
            if (paramsByTrial.TryGetValue(row.TrialId, out var paramRows))
            {
                foreach (var p in paramRows)
                {
                    var distribution = Distribution.FromJson(p.DistributionJson!);
                    @params[p.ParamName!] = distribution.ToExternalRepr(p.ParamValue!.Value);
                    distributions[p.ParamName!] = distribution;
                }
            }

            var userAttrs = new OrderedDict<string, object?>();
            if (userByTrial.TryGetValue(row.TrialId, out var userRows))
                foreach (var a in userRows)
                    userAttrs[a.Key!] = PyJson.Loads(a.ValueJson!);

            var systemAttrs = new OrderedDict<string, object?>();
            if (systemByTrial.TryGetValue(row.TrialId, out var systemRows))
                foreach (var a in systemRows)
                    systemAttrs[a.Key!] = PyJson.Loads(a.ValueJson!);

            var intermediateValues = new OrderedDict<int, double>();
            if (intermediateByTrial.TryGetValue(row.TrialId, out var intermediateRows))
                foreach (var v in intermediateRows)
                    intermediateValues[v.Step] = RdbModels.StoredReprToIntermediateValue(v.IntermediateValue, v.IntermediateValueType);

            result.Add(new FrozenTrial(
                number: row.Number ?? -1,
                state: RdbModels.StateFromDb(row.State),
                value: null,
                values: values,
                datetimeStart: RdbModels.FromDbUtc(row.DatetimeStart),
                datetimeComplete: RdbModels.FromDbUtc(row.DatetimeComplete),
                @params: @params,
                distributions: distributions,
                userAttrs: userAttrs,
                systemAttrs: systemAttrs,
                intermediateValues: intermediateValues,
                trialId: row.TrialId));
        }
        return result;
    }

    private readonly record struct TrialRow(
        int TrialId, int? Number, DbTrialState State, DateTime? DatetimeStart, DateTime? DatetimeComplete);

    // ------------------------------------------------------------------ best trial

    /// <inheritdoc />
    /// <exception cref="NumSharp.RuntimeError">The study has more than one objective.</exception>
    public override FrozenTrial GetBestTrial(int studyId)
    {
        var directions = GetStudyDirections(studyId);
        if (directions.Count > 1)
            throw new RuntimeError("Best trial can be obtained only for single-objective optimization.");
        int trialId = GetBestTrialId(studyId, directions[0]);
        return GetTrial(trialId);
    }

    /// <summary>Finds the completed trial with the best first-objective value for the requested direction.</summary>
    /// <param name="studyId">The storage-wide study identifier, not its display name.</param>
    /// <param name="direction">The objective direction used for ordering or representation conversion.</param>
    /// <returns>The storage-wide identifier of the best completed trial.</returns>
    /// <exception cref="NumSharp.ValueError">No completed trial has an objective value to compare.</exception>
    public int GetBestTrialId(int studyId, StudyDirection direction)
    {
        using var ctx = NewContext();
        var complete = RdbModels.StateToDb(TrialState.Complete);
        var candidates = (
            from t in ctx.Trials.AsNoTracking()
            join v in ctx.TrialValues.AsNoTracking() on t.TrialId equals v.TrialId
            where t.StudyId == studyId && t.State == complete && v.Objective == 0
            select new { t.TrialId, v.Value, v.ValueType })
            .AsEnumerable()
            .Select(r => new { r.TrialId, Eff = EffectiveValue(r.Value, r.ValueType) })
            .ToList();

        if (candidates.Count == 0)
            throw new ValueError(RdbModels.NotFoundMsg);

        // -inf < finite < +inf via the value_type tag folded into ±inf, then by the stored value
        // (Python find_max_value_trial_id / find_min_value_trial_id). LIMIT 1 keeps the first best.
        var best = candidates[0];
        foreach (var c in candidates.Skip(1))
        {
            bool better = direction == StudyDirection.Maximize ? c.Eff > best.Eff : c.Eff < best.Eff;
            if (better)
                best = c;
        }
        return best.TrialId;
    }

    /// <summary>Recovers the objective value represented by its database column and nonfinite-value tag.</summary>
    /// <param name="value">The value to serialize, persist, or convert to the requested representation.</param>
    /// <param name="type">The database tag distinguishing finite values, infinities, and, for intermediate reports, NaN.</param>
    /// <returns>The finite or infinite objective value used for comparison.</returns>
    private static double EffectiveValue(double? value, DbTrialValueType type) => type switch
    {
        DbTrialValueType.INF_POS => double.PositiveInfinity,
        DbTrialValueType.INF_NEG => double.NegativeInfinity,
        _ => value!.Value,
    };

    // ------------------------------------------------------------------ heartbeat

    /// <summary>Inserts or refreshes the trial heartbeat using the database clock or storage timestamp convention.</summary>
    /// <param name="trialId">The storage-wide trial identifier, distinct from its study-local number.</param>
    public void RecordHeartbeat(int trialId)
    {
        using var ctx = NewContext();
        string sql = _provider == RdbProvider.PostgreSql
            ? "INSERT INTO trial_heartbeats (trial_id, heartbeat) VALUES ({0}, CURRENT_TIMESTAMP) " +
              "ON CONFLICT (trial_id) DO UPDATE SET heartbeat = CURRENT_TIMESTAMP"
            : "INSERT INTO trial_heartbeats (trial_id, heartbeat) VALUES ({0}, CURRENT_TIMESTAMP) " +
              "ON DUPLICATE KEY UPDATE heartbeat = CURRENT_TIMESTAMP";
        ctx.Database.ExecuteSqlRaw(sql, trialId);
    }

    /// <summary>Finds running trials whose recorded heartbeat is older than the configured grace period.</summary>
    /// <param name="studyId">The storage-wide study identifier, not its display name.</param>
    /// <returns>Identifiers of running trials whose existing heartbeat has expired.</returns>
    public List<int> GetStaleTrialIds(int studyId)
    {
        int gracePeriod = _gracePeriod ?? 2 * _heartbeatInterval!.Value;
        using var ctx = NewContext();
        var running = RdbModels.StateToDb(TrialState.Running);

        // Everything on the DB clock (Optuna uses func.now()) — the elapsed-seconds expression differs
        // per dialect. Returns the trials whose most recent heartbeat is older than the grace period.
        string elapsed = _provider == RdbProvider.PostgreSql
            ? "EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - h.heartbeat))"
            : "TIMESTAMPDIFF(SECOND, h.heartbeat, CURRENT_TIMESTAMP)";
        string stateName = running.ToString();
        string sql =
            "SELECT t.trial_id FROM trials t JOIN trial_heartbeats h ON h.trial_id = t.trial_id " +
            $"WHERE t.study_id = {{0}} AND t.state = '{stateName}' AND {elapsed} > {{1}}";

        var ids = new List<int>();
        var conn = ctx.Database.GetDbConnection();
        bool opened = conn.State != System.Data.ConnectionState.Open;
        if (opened)
            conn.Open();
        try
        {
            using var cmd = conn.CreateCommand();
            cmd.CommandText = sql.Replace("{0}", "@p0").Replace("{1}", "@p1");
            AddParam(cmd, "@p0", studyId);
            AddParam(cmd, "@p1", gracePeriod);
            using var reader = cmd.ExecuteReader();
            while (reader.Read())
                ids.Add(Convert.ToInt32(reader.GetValue(0), RdbModels.Inv));
        }
        finally
        {
            if (opened)
                conn.Close();
        }
        return ids;
    }

    /// <summary>Adds a provider-created command parameter with its name and value.</summary>
    /// <param name="cmd">The command that receives the newly created parameter.</param>
    /// <param name="name">The identifier used by the provider or schema operation.</param>
    /// <param name="value">The value to serialize, persist, or convert to the requested representation.</param>
    private static void AddParam(DbCommand cmd, string name, object value)
    {
        var p = cmd.CreateParameter();
        p.ParameterName = name;
        p.Value = value;
        cmd.Parameters.Add(p);
    }

    /// <summary>Returns the configured heartbeat recording interval.</summary>
    /// <returns>The heartbeat interval in seconds, or null when heartbeat recording is disabled.</returns>
    public int? GetHeartbeatInterval() => _heartbeatInterval;

    /// <summary>Returns the callback invoked after a heartbeat-stale trial is failed.</summary>
    /// <returns>The configured stale-trial callback, or null if no callback was supplied.</returns>
    public Action<Study, FrozenTrial>? GetHeartbeatStaleTrialCallback() => _heartbeatStaleTrialCallback;

    // ------------------------------------------------------------------ helpers

    /// <summary>Verifies that the requested study record exists before accessing its children.</summary>
    /// <param name="ctx">The operation-scoped EF Core context; the caller owns its lifetime and transaction.</param>
    /// <param name="studyId">The storage-wide study identifier, not its display name.</param>
    /// <exception cref="NumSharp.KeyError">The study identifier does not exist.</exception>
    private static void EnsureStudyExists(OptunaRdbContext ctx, int studyId)
    {
        if (!ctx.Studies.AsNoTracking().Any(s => s.StudyId == studyId))
            throw new KeyError(RdbModels.NotFoundMsg);
    }

    /// <summary>Verifies that the requested trial record exists before accessing its children.</summary>
    /// <param name="ctx">The operation-scoped EF Core context; the caller owns its lifetime and transaction.</param>
    /// <param name="trialId">The storage-wide trial identifier, distinct from its study-local number.</param>
    /// <exception cref="NumSharp.KeyError">The trial identifier does not exist.</exception>
    private static void EnsureTrialExists(OptunaRdbContext ctx, int trialId)
    {
        if (!ctx.Trials.AsNoTracking().Any(t => t.TrialId == trialId))
            throw new KeyError(RdbModels.NotFoundMsg);
    }

    // SELECT * ... FOR UPDATE row lock (PostgreSQL/MySQL) — Optuna's TrialModel.find_or_raise_by_id(
    // for_update=True). AsEnumerable() keeps the SQL unwrapped so the FOR UPDATE actually locks.
    /// <summary>Loads the tracked trial record needed to validate and apply a mutation.</summary>
    /// <param name="ctx">The operation-scoped EF Core context; the caller owns its lifetime and transaction.</param>
    /// <param name="trialId">The storage-wide trial identifier, distinct from its study-local number.</param>
    /// <returns>The tracked entity used for update validation.</returns>
    /// <exception cref="NumSharp.KeyError">The trial identifier does not exist.</exception>
    private static TrialEntity FindTrialForUpdate(OptunaRdbContext ctx, int trialId)
    {
        var trial = ctx.Trials
            .FromSqlRaw("SELECT * FROM trials WHERE trial_id = {0} FOR UPDATE", trialId)
            .AsEnumerable().FirstOrDefault();
        if (trial is null)
            throw new KeyError(RdbModels.NotFoundMsg);
        return trial;
    }

    /// <summary>Rejects mutations of a trial that has reached a finished state.</summary>
    /// <param name="trial">The trial snapshot or database record being read or validated.</param>
    /// <exception cref="UpdateFinishedTrialError">The trial is already complete, pruned, or failed.</exception>
    private static void CheckTrialIsUpdatableOrRaise(TrialEntity trial)
    {
        if (RdbModels.StateFromDb(trial.State).IsFinished())
            throw new UpdateFinishedTrialError(
                $"Trial#{trial.Number} has already finished and can not be updated.");
    }

    /// <summary>Recognizes provider-specific uniqueness failures so duplicate studies can be translated consistently.</summary>
    /// <param name="e">The provider exception to classify or translate.</param>
    /// <returns>Whether the exception denotes a uniqueness-constraint violation.</returns>
    private static bool IsUniqueViolation(DbUpdateException e)
    {
        for (Exception? inner = e; inner is not null; inner = inner.InnerException)
        {
            if (inner is PostgresException pg && pg.SqlState == PostgresErrorCodes.UniqueViolation)
                return true;
            if (inner is MySqlException my && my.ErrorCode == MySqlErrorCode.DuplicateKeyEntry)
                return true;
        }
        return false;
    }

    /// <summary>
    ///     Disposes an injected PostgreSQL data source passed with <c>ownsDataSource: true</c> (and its
    ///     pooled connections). Everything else is left open: an injected data source with
    ///     <c>ownsDataSource: false</c> stays the caller's, and the URL / connection-string constructors use a
    ///     data source shared per connection string for the life of the process (see
    ///     <see cref="SharedPostgresDataSource" />), so disposing one storage never breaks another on the same
    ///     database. MySQL keeps no per-storage resource (MySqlConnector pools by connection string).
    /// </summary>
    public void Dispose()
    {
        if (_ownsDataSource)
            _dataSource?.Dispose();
    }
}

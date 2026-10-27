using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Text;
using Microsoft.Data.Sqlite;
using NumSharp;
using OptunaSharp.Collections;
using OptunaSharp.Distributions;
using OptunaSharp.Internal;
using OptunaSharp.Logging;

namespace OptunaSharp.Storages;

/// <summary>
///     Storage class for an RDB backend on SQLite. Port of
///     <c>optuna/storages/_rdb/storage.py: RDBStorage</c>, hand-written over
///     <c>Microsoft.Data.Sqlite</c> against the exact schema SQLAlchemy generates (see
///     <see cref="RdbModels" />), so Python Optuna and OptunaSharp can open the same
///     <c>.db</c> file.
///
///     <para>Like Optuna, this backend is meant to be wrapped in a <see cref="CachedStorage" />:
///     it implements <see cref="ICachedStorageBackend" /> and
///     <see cref="StorageFunctions.GetStorage(BaseStorage?)" /> wraps it automatically (Python's
///     <c>get_storage</c> returns <c>_CachedStorage(RDBStorage(...))</c>).</para>
///
///     <para><b>Concurrency.</b> A single SQLite connection is held for the storage's lifetime and
///     every access is serialized under one lock. This keeps <c>:memory:</c> databases alive across
///     operations and is the correct model for SQLite (Optuna itself warns against SQLite for
///     parallel optimization). The <c>FOR UPDATE</c> row locks Optuna uses on other RDBs — which
///     SQLite does not support — are subsumed by this lock.</para>
/// </summary>
/// <example>
///     <code>
///     using var storage = new RDBStorage("sqlite:///study.db");
///     var study = Optuna.CreateStudy(storage: storage);
///     study.Optimize(t => Math.Pow(t.SuggestFloat("x", -100, 100), 2), nTrials: 10);
///     </code>
/// </example>
public sealed partial class RDBStorage : BaseStorage, ICachedStorageBackend, IDisposable
{
    private static readonly Logger _logger = OptunaLogging.GetLogger("optuna.storages._rdb.storage");

    private readonly object _lock = new();

    private SqliteConnection _conn = null!;
    private bool _ownsConnection;
    private int? _heartbeatInterval;
    private int? _gracePeriod;
    private Action<Study, FrozenTrial>? _heartbeatStaleTrialCallback;

    /// <summary>The storage URL/connection string this storage was opened with.</summary>
    public string Url { get; private set; } = "";

    /// <summary>
    ///     The underlying connection — exposed to the test project (friend) so the effect of every
    ///     configured <c>PRAGMA</c>/connection-string option can be asserted on the exact connection
    ///     the storage uses. Not part of the public API.
    /// </summary>
    internal SqliteConnection Connection => _conn;

    /// <summary>
    ///     Create an RDB storage from a SQLAlchemy-style SQLite URL — the Optuna-faithful constructor
    ///     (<c>optuna.storages.RDBStorage(url, …)</c>).
    /// </summary>
    /// <param name="url">
    ///     A SQLite URL: <c>sqlite:///relative.db</c>, <c>sqlite:////absolute.db</c>,
    ///     <c>sqlite:///C:/path.db</c> or <c>sqlite:///:memory:</c>.
    /// </param>
    /// <param name="skipCompatibilityCheck">Skip the alembic schema-compatibility check.</param>
    /// <param name="skipTableCreation">Skip creating the tables (assume they already exist).</param>
    /// <param name="heartbeatInterval">Heartbeat recording interval in seconds (experimental).</param>
    /// <param name="gracePeriod">
    ///     Grace period before a running trial is failed from its last heartbeat. Defaults to
    ///     <c>2 * heartbeatInterval</c>.
    /// </param>
    /// <param name="heartbeatStaleTrialCallback">Callback invoked after each heartbeat-stale trial is failed.</param>
    /// <param name="failedTrialCallback">
    ///     Deprecated alias of <paramref name="heartbeatStaleTrialCallback" /> (Python
    ///     <c>failed_trial_callback</c>, deprecated in v4.9.0, removed in v6.0.0). Specify only one of
    ///     the two.
    /// </param>
    public RDBStorage(
        string url,
        bool skipCompatibilityCheck = false,
        bool skipTableCreation = false,
        int? heartbeatInterval = null,
        int? gracePeriod = null,
        Action<Study, FrozenTrial>? heartbeatStaleTrialCallback = null,
        Action<Study, FrozenTrial>? failedTrialCallback = null)
        : this(url, new SqliteRdbOptions
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

    /// <summary>
    ///     Create an RDB storage from a URL plus a native tweak of the
    ///     <see cref="SqliteConnectionStringBuilder" /> — the convenience form of
    ///     <see cref="SqliteRdbOptions.ConfigureConnectionString" />.
    /// </summary>
    /// <param name="url">The database connection URL to resolve.</param>
    /// <param name="configureConnectionString">The callback applied to the connection-string builder before opening the database.</param>
    public RDBStorage(string url, Action<SqliteConnectionStringBuilder> configureConnectionString)
        : this(url, new SqliteRdbOptions { ConfigureConnectionString = configureConnectionString })
    {
    }

    /// <summary>
    ///     Create an RDB storage from a URL with the full <see cref="SqliteRdbOptions" /> surface —
    ///     every connection-string option and load-bearing <c>PRAGMA</c>, beyond Optuna's own API.
    /// </summary>
    /// <param name="url">The database connection URL to resolve.</param>
    /// <param name="options">The storage and provider configuration applied before schema initialization.</param>
    public RDBStorage(string url, SqliteRdbOptions options)
    {
        ArgumentNullException.ThrowIfNull(url);
        ArgumentNullException.ThrowIfNull(options);
        string filled = FillStorageUrlTemplate(url);
        string connectionString = BuildConnectionStringFromUrl(filled, options);
        Initialize(new SqliteConnection(connectionString), ownsConnection: true, options, filled);
    }

    /// <summary>
    ///     Create an RDB storage from a fully-built <see cref="SqliteConnectionStringBuilder" /> (BYO
    ///     connection string). Any typed connection-string options on <paramref name="options" /> are
    ///     layered on top of the builder.
    /// </summary>
    /// <param name="connectionStringBuilder">The native provider connection-string builder, customized before the connection opens.</param>
    /// <param name="options">The storage and provider configuration applied before schema initialization.</param>
    public RDBStorage(SqliteConnectionStringBuilder connectionStringBuilder, SqliteRdbOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(connectionStringBuilder);
        options ??= new SqliteRdbOptions();
        var builder = new SqliteConnectionStringBuilder(connectionStringBuilder.ConnectionString);
        ApplyConnectionStringOptions(builder, options);
        Initialize(new SqliteConnection(builder.ConnectionString), ownsConnection: true, options, builder.ConnectionString);
    }

    /// <summary>
    ///     Create an RDB storage over an existing <see cref="SqliteConnection" /> (BYO connection —
    ///     the "feel like the client, not Optuna" form). The connection is opened if needed; its
    ///     connection-string options are used as-is (the typed connection-string options on
    ///     <paramref name="options" /> are ignored), while the <c>PRAGMA</c> options and
    ///     <see cref="SqliteRdbOptions.ConfigureConnection" /> still apply.
    /// </summary>
    /// <param name="connection">The connection to use.</param>
    /// <param name="options">Full configuration (PRAGMAs, heartbeat, schema knobs).</param>
    /// <param name="ownsConnection">
    ///     When <c>true</c>, <see cref="Dispose" /> disposes the connection; when <c>false</c>
    ///     (default), the caller retains ownership.
    /// </param>
    public RDBStorage(SqliteConnection connection, SqliteRdbOptions? options = null, bool ownsConnection = false)
    {
        ArgumentNullException.ThrowIfNull(connection);
        Initialize(connection, ownsConnection, options ?? new SqliteRdbOptions(), connection.ConnectionString);
    }

    // ------------------------------------------------------------------ connection bootstrap

    /// <summary>Opens and configures the storage resources, then initializes or verifies the Optuna schema.</summary>
    /// <param name="connection">The database or Redis connection used by the backend.</param>
    /// <param name="ownsConnection">Whether the storage should dispose the supplied connection when it is disposed.</param>
    /// <param name="options">The storage and provider configuration applied before schema initialization.</param>
    /// <param name="url">The database connection URL to resolve.</param>
    private void Initialize(SqliteConnection connection, bool ownsConnection, SqliteRdbOptions options, string url)
    {
        ValidateOptions(options);

        Url = url;
        _heartbeatInterval = options.HeartbeatInterval;
        _gracePeriod = options.GracePeriod;
        // Python: `heartbeat_stale_trial_callback or failed_trial_callback` — the deprecated alias is
        // used only when the primary is unset (ValidateOptions rejects specifying both).
        _heartbeatStaleTrialCallback = options.HeartbeatStaleTrialCallback ?? options.FailedTrialCallback;
        _conn = connection;
        _ownsConnection = ownsConnection;

        if (_conn.State != System.Data.ConnectionState.Open)
            _conn.Open();

        ApplyPragmas(options);

        if (!options.SkipTableCreation)
            CreateSchema();

        InitVersionInfo();
        InitAlembic();
        if (!options.SkipCompatibilityCheck)
            CheckTableSchemaCompatibility();
    }

    // Port of the heartbeat validation + warnings block of `RDBStorage.__init__`
    // (optuna/storages/_rdb/storage.py) — the exact order matters (the experimental warning fires
    // before the grace-period check; the deprecation warning fires after the "specify only one" check).
    /// <summary>Rejects invalid heartbeat settings and conflicting callback aliases before opening storage.</summary>
    /// <param name="options">The storage and provider configuration applied before schema initialization.</param>
    /// <exception cref="NumSharp.ValueError">A heartbeat interval or grace period is nonpositive, or both callback aliases are supplied.</exception>
    private static void ValidateOptions(SqliteRdbOptions options)
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
        => template.Replace("{SCHEMA_VERSION}", RdbModels.SchemaVersion.ToString(CultureInfo.InvariantCulture));

    /// <summary>Extracts the SQLite file or in-memory target from an Optuna-style URL.</summary>
    /// <param name="url">The database connection URL to resolve.</param>
    /// <returns>The SQLite data-source name used to open the connection.</returns>
    private static string ResolveDataSource(string url)
    {
        const string prefix = "sqlite:///";
        if (url.StartsWith(prefix, StringComparison.Ordinal))
            return url.Substring(prefix.Length);
        if (url.StartsWith("sqlite://", StringComparison.Ordinal))
            return url.Substring("sqlite://".Length);
        return url; // tolerate a bare file path
    }

    /// <summary>Builds the SQLite connection string from the resolved URL and typed options.</summary>
    /// <param name="filledUrl">The storage URL after schema-version template substitution.</param>
    /// <param name="options">The storage and provider configuration applied before schema initialization.</param>
    /// <returns>The configured native SQLite connection string.</returns>
    private static string BuildConnectionStringFromUrl(string filledUrl, SqliteRdbOptions options)
    {
        string dataSource = ResolveDataSource(filledUrl);
        var builder = new SqliteConnectionStringBuilder { DataSource = dataSource };
        // A :memory: data source needs Memory mode unless the caller overrode it explicitly.
        if (options.Mode is null && dataSource == ":memory:")
            builder.Mode = SqliteOpenMode.Memory;
        ApplyConnectionStringOptions(builder, options);
        return builder.ConnectionString;
    }

    /// <summary>Applies explicitly configured SQLite connection settings and the customization callback.</summary>
    /// <param name="builder">The native connection-string builder receiving the parsed options.</param>
    /// <param name="o">The configured storage options; null-valued settings leave provider defaults unchanged.</param>
    private static void ApplyConnectionStringOptions(SqliteConnectionStringBuilder builder, SqliteRdbOptions o)
    {
        if (o.Mode is not null)
            builder.Mode = o.Mode.Value;
        if (o.Cache is not null)
            builder.Cache = o.Cache.Value;
        if (o.Pooling is not null)
            builder.Pooling = o.Pooling.Value;
        if (o.Password is not null)
            builder.Password = o.Password;
        if (o.RecursiveTriggers is not null)
            builder.RecursiveTriggers = o.RecursiveTriggers.Value;
        if (o.DefaultTimeoutSeconds is not null)
            builder.DefaultTimeout = o.DefaultTimeoutSeconds.Value;
        o.ConfigureConnectionString?.Invoke(builder);
    }

    /// <summary>Creates the missing Optuna tables and indexes using the pinned SQLite schema.</summary>
    private void CreateSchema()
    {
        foreach (var stmt in RdbModels.SchemaStatements)
            Exec(stmt, null);
    }

    /// <summary>Applies opted-in SQLite runtime settings before schema initialization.</summary>
    /// <param name="o">The configured storage options; null-valued settings leave provider defaults unchanged.</param>
    private void ApplyPragmas(SqliteRdbOptions o)
    {
        // page_size / auto_vacuum only take effect on a fresh database, so apply them before any table.
        if (o.PageSize is not null)
            ExecPragma($"page_size={o.PageSize.Value}");
        if (o.AutoVacuum is not null)
            ExecPragma($"auto_vacuum={o.AutoVacuum.Value.ToString().ToUpperInvariant()}");
        if (o.JournalMode is not null)
            ExecPragma($"journal_mode={o.JournalMode.Value.ToString().ToUpperInvariant()}");
        if (o.Synchronous is not null)
            ExecPragma($"synchronous={o.Synchronous.Value.ToString().ToUpperInvariant()}");
        if (o.ForeignKeys is not null)
            ExecPragma($"foreign_keys={(o.ForeignKeys.Value ? "ON" : "OFF")}");
        if (o.BusyTimeoutMs is not null)
            ExecPragma($"busy_timeout={o.BusyTimeoutMs.Value}");
        if (o.CacheSize is not null)
            ExecPragma($"cache_size={o.CacheSize.Value}");
        if (o.MmapSize is not null)
            ExecPragma($"mmap_size={o.MmapSize.Value}");
        if (o.TempStore is not null)
            ExecPragma($"temp_store={o.TempStore.Value.ToString().ToUpperInvariant()}");
        if (o.SecureDelete is not null)
            ExecPragma($"secure_delete={(o.SecureDelete.Value ? "ON" : "OFF")}");
        if (o.AdditionalPragmas is not null)
            foreach (var pragma in o.AdditionalPragmas)
                ExecPragma(pragma);
        o.ConfigureConnection?.Invoke(_conn);
    }

    /// <summary>Executes one implementation-selected SQLite PRAGMA body.</summary>
    /// <param name="body">The trusted PRAGMA expression executed after the PRAGMA keyword.</param>
    private void ExecPragma(string body) => Exec("PRAGMA " + body, null);

    // Port of `_VersionManager._init_version_info_model`. The version tables are created here (not only
    // in CreateSchema) so the two version-tracking tables exist even under `skip_table_creation` — the
    // mode the CLI's `storage upgrade` opens with — mirroring Python, where the ORM insert / alembic
    // stamp materialize them regardless. Both are IF NOT EXISTS no-ops on the normal fresh-DB path.
    /// <summary>Creates the informational schema/library version record if it is absent.</summary>
    private void InitVersionInfo()
    {
        Exec(RdbModels.VersionInfoTableDdl, null);
        long count = ScalarLong("SELECT COUNT(*) FROM version_info", null);
        if (count > 0)
            return;
        Exec(
            "INSERT INTO version_info (version_info_id, schema_version, library_version) VALUES (1, $p0, $p1)",
            null,
            RdbModels.SchemaVersion,
            RdbModels.LibraryVersion);
    }

    // Port of `_VersionManager._init_alembic`. On an empty `alembic_version`, Python stamps the head
    // revision when alembic is "supported" (a freshly-created DB, or one already at the current
    // schema_version) and the base revision otherwise — the marker that says "this is a pre-alembic
    // database that must be migrated forward". The current OptunaSharp code always stamped head, which
    // silently mislabelled an old database as up-to-date; stamping the base is what lets the
    // compatibility check (and `storage upgrade`) see it needs migrating.
    /// <summary>Initializes the Alembic migration marker for a newly created or legacy database.</summary>
    private void InitAlembic()
    {
        Exec(RdbModels.AlembicVersionTableDdl, null);
        long count = ScalarLong("SELECT COUNT(*) FROM alembic_version", null);
        if (count > 0)
            return;
        string revision = IsAlembicSupported() ? RdbModels.HeadVersion : RdbMigrations.BaseVersion;
        Exec("INSERT INTO alembic_version (version_num) VALUES ($p0)", null, revision);
    }

    // Port of `_VersionManager._is_alembic_supported`: `version_info` is guaranteed to exist by the
    // preceding InitVersionInfo, so this is true when the stored schema_version equals the current one
    // (a new or already-current DB) and false for a pre-alembic database (an older schema_version).
    /// <summary>Checks whether the legacy schema version can be associated with the supported Alembic chain.</summary>
    /// <returns>Whether the database's legacy schema version is supported by the migration chain.</returns>
    private bool IsAlembicSupported()
        => ScalarNullableLong("SELECT schema_version FROM version_info WHERE version_info_id = 1", null)
            == RdbModels.SchemaVersion;

    /// <summary>Checks that the database migration revision is compatible with this storage implementation.</summary>
    /// <exception cref="NumSharp.RuntimeError">The Alembic marker is absent or the schema revision is incompatible.</exception>
    private void CheckTableSchemaCompatibility()
    {
        string? libraryVersion = ScalarString("SELECT library_version FROM version_info", null);
        string current = ScalarString("SELECT version_num FROM alembic_version", null)
            ?? throw new RuntimeError("The alembic_version table is empty.");
        if (current == RdbModels.HeadVersion)
            return;

        var message = new StringBuilder(
            $"The runtime optuna version {OptunaSharpVersion.OptunaCompatibilityTarget} is no longer " +
            "compatible with the table schema " +
            $"(set up by optuna {libraryVersion}). ");
        message.Append(RdbMigrations.RevisionsBaseToHead.Contains(current)
            ? "Please execute `$ optuna storage upgrade --storage $STORAGE_URL` for upgrading the storage."
            : "Please try updating optuna to the latest version by `$ pip install -U optuna`.");
        throw new RuntimeError(message.ToString());
    }

    // ------------------------------------------------------------------ study manipulation

    /// <inheritdoc />
    /// <exception cref="DuplicatedStudyError">Another study already uses the requested name.</exception>
    public override int CreateNewStudy(IReadOnlyList<StudyDirection> directions, string? studyName = null)
    {
        lock (_lock)
        {
            studyName ??= CreateUniqueStudyName();

            using var tx = _conn.BeginTransaction();
            long studyId;
            try
            {
                Exec("INSERT INTO studies (study_name) VALUES ($p0)", tx, studyName);
                studyId = ScalarLong("SELECT last_insert_rowid()", tx);
                for (int objective = 0; objective < directions.Count; objective++)
                {
                    Exec(
                        "INSERT INTO study_directions (direction, study_id, objective) VALUES ($p0, $p1, $p2)",
                        tx, RdbModels.DirectionToDb(directions[objective]), studyId, objective);
                }
                tx.Commit();
            }
            catch (SqliteException e) when (IsUniqueViolation(e))
            {
                tx.Rollback();
                throw new DuplicatedStudyError();
            }

            _logger.Info($"A new study created in RDB with name: {studyName}");
            return (int)studyId;
        }
    }

    /// <summary>Generates a study name and checks the database until the name is unused.</summary>
    /// <returns>A generated name that is not currently present in the studies table.</returns>
    private string CreateUniqueStudyName()
    {
        while (true)
        {
            string studyName = DefaultStudyNamePrefix + Guid.NewGuid();
            long count = ScalarLong("SELECT COUNT(*) FROM studies WHERE study_name = $p0", null, studyName);
            if (count == 0)
                return studyName;
        }
    }

    /// <inheritdoc />
    public override void DeleteStudy(int studyId)
    {
        lock (_lock)
        {
            using var tx = _conn.BeginTransaction();
            EnsureStudyExists(studyId, tx);

            // Foreign keys are not enforced (Optuna leaves PRAGMA foreign_keys off and cascades in the
            // ORM layer), so every child row is deleted explicitly, deepest first.
            const string trialScope = "trial_id IN (SELECT trial_id FROM trials WHERE study_id = $p0)";
            Exec($"DELETE FROM trial_params WHERE {trialScope}", tx, studyId);
            Exec($"DELETE FROM trial_values WHERE {trialScope}", tx, studyId);
            Exec($"DELETE FROM trial_intermediate_values WHERE {trialScope}", tx, studyId);
            Exec($"DELETE FROM trial_user_attributes WHERE {trialScope}", tx, studyId);
            Exec($"DELETE FROM trial_system_attributes WHERE {trialScope}", tx, studyId);
            Exec($"DELETE FROM trial_heartbeats WHERE {trialScope}", tx, studyId);
            Exec("DELETE FROM trials WHERE study_id = $p0", tx, studyId);
            Exec("DELETE FROM study_directions WHERE study_id = $p0", tx, studyId);
            Exec("DELETE FROM study_user_attributes WHERE study_id = $p0", tx, studyId);
            Exec("DELETE FROM study_system_attributes WHERE study_id = $p0", tx, studyId);
            Exec("DELETE FROM studies WHERE study_id = $p0", tx, studyId);
            tx.Commit();
        }
    }

    /// <inheritdoc />
    public override void SetStudyUserAttr(int studyId, string key, object? value)
        => SetStudyAttr("study_user_attributes", studyId, key, value);

    /// <inheritdoc />
    public override void SetStudySystemAttr(int studyId, string key, object? value)
        => SetStudyAttr("study_system_attributes", studyId, key, value);

    /// <summary>Upserts a JSON-serializable attribute after verifying that its study exists.</summary>
    /// <param name="table">The trusted schema table name selected by the storage implementation.</param>
    /// <param name="studyId">The storage-wide study identifier, not its display name.</param>
    /// <param name="key">The attribute name or exact storage key to access.</param>
    /// <param name="value">The value to serialize, persist, or convert to the requested representation.</param>
    private void SetStudyAttr(string table, int studyId, string key, object? value)
    {
        lock (_lock)
        {
            using var tx = _conn.BeginTransaction();
            EnsureStudyExists(studyId, tx);
            Exec(
                $"INSERT INTO {table} (study_id, \"key\", value_json) VALUES ($p0, $p1, $p2) " +
                "ON CONFLICT(study_id, \"key\") DO UPDATE SET value_json = excluded.value_json",
                tx, studyId, key, PyJson.Dumps(value));
            tx.Commit();
        }
    }

    /// <inheritdoc />
    /// <exception cref="NumSharp.KeyError">No study has the requested name.</exception>
    public override int GetStudyIdFromName(string studyName)
    {
        lock (_lock)
        {
            long? id = ScalarNullableLong("SELECT study_id FROM studies WHERE study_name = $p0", null, studyName);
            if (id is null)
                throw new KeyError(RdbModels.NotFoundMsg);
            return (int)id.Value;
        }
    }

    /// <inheritdoc />
    /// <exception cref="NumSharp.KeyError">The study identifier does not exist.</exception>
    public override string GetStudyNameFromId(int studyId)
    {
        lock (_lock)
        {
            string? name = ScalarString("SELECT study_name FROM studies WHERE study_id = $p0", null, studyId);
            if (name is null)
                throw new KeyError(RdbModels.NotFoundMsg);
            return name;
        }
    }

    /// <inheritdoc />
    public override List<StudyDirection> GetStudyDirections(int studyId)
    {
        lock (_lock)
        {
            EnsureStudyExists(studyId, null);
            var directions = new List<StudyDirection>();
            Query(
                "SELECT direction FROM study_directions WHERE study_id = $p0 ORDER BY objective",
                null,
                r => directions.Add(RdbModels.DirectionFromDb(r.GetString(0))),
                studyId);
            return directions;
        }
    }

    /// <inheritdoc />
    public override OrderedDict<string, object?> GetStudyUserAttrs(int studyId)
        => GetStudyAttrs("study_user_attributes", "study_user_attribute_id", studyId);

    /// <inheritdoc />
    public override OrderedDict<string, object?> GetStudySystemAttrs(int studyId)
        => GetStudyAttrs("study_system_attributes", "study_system_attribute_id", studyId);

    /// <summary>Loads and decodes all attributes of the selected kind for a study.</summary>
    /// <param name="table">The trusted schema table name selected by the storage implementation.</param>
    /// <param name="idColumn">The trusted identifier column used to group records.</param>
    /// <param name="studyId">The storage-wide study identifier, not its display name.</param>
    /// <returns>Decoded attributes in their stored traversal order.</returns>
    private OrderedDict<string, object?> GetStudyAttrs(string table, string idColumn, int studyId)
    {
        lock (_lock)
        {
            EnsureStudyExists(studyId, null);
            var attrs = new OrderedDict<string, object?>();
            Query(
                $"SELECT \"key\", value_json FROM {table} WHERE study_id = $p0 ORDER BY {idColumn}",
                null,
                r => attrs[r.GetString(0)] = PyJson.Loads(r.GetString(1)),
                studyId);
            return attrs;
        }
    }

    /// <inheritdoc />
    public override List<FrozenStudy> GetAllStudies()
    {
        lock (_lock)
        {
            var studyRows = new List<(int Id, string Name)>();
            Query(
                "SELECT study_id, study_name FROM studies ORDER BY study_id",
                null,
                r => studyRows.Add(((int)r.GetInt64(0), r.GetString(1))));

            var directions = new Dictionary<int, List<StudyDirection>>();
            Query(
                "SELECT study_id, direction FROM study_directions ORDER BY study_id, objective",
                null,
                r =>
                {
                    int sid = (int)r.GetInt64(0);
                    if (!directions.TryGetValue(sid, out var list))
                        directions[sid] = list = new List<StudyDirection>();
                    list.Add(RdbModels.DirectionFromDb(r.GetString(1)));
                });

            var userAttrs = LoadAllStudyAttrs("study_user_attributes", "study_user_attribute_id");
            var systemAttrs = LoadAllStudyAttrs("study_system_attributes", "study_system_attribute_id");

            var result = new List<FrozenStudy>(studyRows.Count);
            foreach (var (id, name) in studyRows)
            {
                result.Add(new FrozenStudy(
                    studyName: name,
                    direction: null,
                    directions: directions.TryGetValue(id, out var d) ? d : new List<StudyDirection>(),
                    userAttrs: userAttrs.TryGetValue(id, out var u) ? u : new OrderedDict<string, object?>(),
                    systemAttrs: systemAttrs.TryGetValue(id, out var s) ? s : new OrderedDict<string, object?>(),
                    studyId: id));
            }
            return result;
        }
    }

    /// <summary>Loads decoded study attributes in one pass, grouped by storage-wide study identifier.</summary>
    /// <param name="table">The trusted schema table name selected by the storage implementation.</param>
    /// <param name="idColumn">The trusted identifier column used to group records.</param>
    /// <returns>A dictionary mapping study identifiers to decoded attribute dictionaries.</returns>
    private Dictionary<int, OrderedDict<string, object?>> LoadAllStudyAttrs(string table, string idColumn)
    {
        var byStudy = new Dictionary<int, OrderedDict<string, object?>>();
        Query(
            $"SELECT study_id, \"key\", value_json FROM {table} ORDER BY {idColumn}",
            null,
            r =>
            {
                int sid = (int)r.GetInt64(0);
                if (!byStudy.TryGetValue(sid, out var dict))
                    byStudy[sid] = dict = new OrderedDict<string, object?>();
                dict[r.GetString(1)] = PyJson.Loads(r.GetString(2));
            });
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
    public FrozenTrial CreateNewTrialFull(int studyId, FrozenTrial? templateTrial = null)
    {
        lock (_lock)
        {
            using var tx = _conn.BeginTransaction();
            EnsureStudyExists(studyId, tx);

            long trialId;
            if (templateTrial is null)
            {
                Exec(
                    "INSERT INTO trials (number, study_id, state, datetime_start, datetime_complete) " +
                    "VALUES (NULL, $p0, $p1, $p2, NULL)",
                    tx, studyId, RdbModels.StateToDb(TrialState.Running), RdbModels.FormatDatetime(DateTime.Now));
                trialId = ScalarLong("SELECT last_insert_rowid()", tx);
            }
            else
            {
                // Only RUNNING trials are updatable, so the new trial is temporarily RUNNING while its
                // fields are written, then flipped to the template's state (Python's `_get_prepared_new_trial`).
                Exec(
                    "INSERT INTO trials (number, study_id, state, datetime_start, datetime_complete) " +
                    "VALUES (NULL, $p0, $p1, $p2, $p3)",
                    tx, studyId, RdbModels.StateToDb(TrialState.Running),
                    OrNull(templateTrial.DatetimeStart), OrNull(templateTrial.DatetimeComplete));
                trialId = ScalarLong("SELECT last_insert_rowid()", tx);

                var values = templateTrial.Values;
                if (values is { Count: > 1 })
                {
                    for (int objective = 0; objective < values.Count; objective++)
                        SetTrialValueNoCommit(tx, (int)trialId, objective, values[objective]);
                }
                else if (templateTrial.Value is not null)
                {
                    SetTrialValueNoCommit(tx, (int)trialId, 0, templateTrial.Value.Value);
                }

                foreach (var (paramName, paramValue) in templateTrial.Params)
                {
                    var distribution = templateTrial.Distributions[paramName];
                    double internalRepr = distribution.ToInternalRepr(paramValue);
                    SetTrialParamNoCommit(tx, (int)trialId, paramName, internalRepr, distribution, cachedDistribution: null);
                }

                foreach (var (key, value) in templateTrial.UserAttrs)
                    SetTrialAttrNoCommit(tx, "trial_user_attributes", (int)trialId, key, value);
                foreach (var (key, value) in templateTrial.SystemAttrs)
                    SetTrialAttrNoCommit(tx, "trial_system_attributes", (int)trialId, key, value);
                foreach (var (step, intermediateValue) in templateTrial.IntermediateValues)
                    SetTrialIntermediateValueNoCommit(tx, (int)trialId, step, intermediateValue);

                Exec("UPDATE trials SET state = $p0 WHERE trial_id = $p1",
                    tx, RdbModels.StateToDb(templateTrial.State), trialId);
            }

            // number = count of the study's trials created before this one (Python `count_past_trials`).
            long number = ScalarLong(
                "SELECT COUNT(trial_id) FROM trials WHERE study_id = $p0 AND trial_id < $p1", tx, studyId, trialId);
            Exec("UPDATE trials SET number = $p0 WHERE trial_id = $p1", tx, number, trialId);

            string? datetimeStartStr = ScalarString(
                "SELECT datetime_start FROM trials WHERE trial_id = $p0", tx, trialId);
            tx.Commit();

            DateTime? datetimeStart = RdbModels.ParseDatetime(datetimeStartStr);
            if (templateTrial is not null)
            {
                var frozen = templateTrial.DeepCopy();
                frozen.SetNumber((int)number);
                frozen.SetDatetimeStart(datetimeStart);
                frozen.TrialId = (int)trialId;
                return frozen;
            }

            return new FrozenTrial(
                number: (int)number,
                state: TrialState.Running,
                value: null,
                datetimeStart: datetimeStart,
                datetimeComplete: null,
                @params: new OrderedDict<string, object?>(),
                distributions: new OrderedDict<string, BaseDistribution>(),
                userAttrs: new OrderedDict<string, object?>(),
                systemAttrs: new OrderedDict<string, object?>(),
                intermediateValues: new OrderedDict<long, double>(),
                trialId: (int)trialId);
        }
    }

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
        BaseDistribution? cachedDistribution)
    {
        lock (_lock)
        {
            using var tx = _conn.BeginTransaction();
            SetTrialParamNoCommit(tx, trialId, paramName, paramValueInternal, distribution, cachedDistribution);
            tx.Commit();
        }
    }

    /// <summary>Upserts a compatible parameter inside the caller-owned transaction without committing it.</summary>
    /// <param name="tx">The current SQLite transaction, or null to execute outside an explicit transaction.</param>
    /// <param name="trialId">The storage-wide trial identifier, distinct from its study-local number.</param>
    /// <param name="paramName">The case-sensitive parameter name.</param>
    /// <param name="paramValueInternal">The numeric internal representation defined by the parameter distribution.</param>
    /// <param name="distribution">The distribution describing the parameter and its internal representation.</param>
    /// <param name="cachedDistribution">A previously validated distribution for this parameter, or null when storage must check compatibility.</param>
    private void SetTrialParamNoCommit(
        SqliteTransaction tx,
        int trialId,
        string paramName,
        double paramValueInternal,
        BaseDistribution distribution,
        BaseDistribution? cachedDistribution)
    {
        var trial = FindTrialOrRaise(trialId, tx);
        CheckTrialIsUpdatableOrRaise(trial);

        BaseDistribution? previousDistribution = cachedDistribution;
        if (previousDistribution is null)
        {
            string? previousJson = ScalarString(
                "SELECT tp.distribution_json FROM trial_params tp " +
                "JOIN trials t ON t.trial_id = tp.trial_id " +
                "WHERE t.study_id = $p0 AND tp.param_name = $p1 LIMIT 1",
                tx, trial.StudyId, paramName);
            if (previousJson is not null)
                previousDistribution = Distribution.FromJson(previousJson);
        }

        if (previousDistribution is not null)
            Distribution.CheckDistributionCompatibility(previousDistribution, distribution);

        // Optuna inserts (not upserts) under `ignore_integrity_error=True`: a duplicate
        // (trial_id, param_name) is silently ignored, so the first write for a parameter wins.
        try
        {
            Exec(
                "INSERT INTO trial_params (trial_id, param_name, param_value, distribution_json) " +
                "VALUES ($p0, $p1, $p2, $p3)",
                tx, trialId, paramName, paramValueInternal, Distribution.ToJson(distribution));
        }
        catch (SqliteException e) when (IsUniqueViolation(e))
        {
            // Timing race among threads/processes — another writer committed the same key first.
        }
    }

    /// <inheritdoc />
    /// <exception cref="NumSharp.KeyError">The trial or requested parameter does not exist.</exception>
    public override double GetTrialParam(int trialId, string paramName)
    {
        lock (_lock)
        {
            EnsureTrialExists(trialId, null);
            double? value = ScalarNullableDouble(
                "SELECT param_value FROM trial_params WHERE trial_id = $p0 AND param_name = $p1",
                null, trialId, paramName);
            if (value is null)
                throw new KeyError(RdbModels.NotFoundMsg);
            return value.Value;
        }
    }

    // ------------------------------------------------------------------ trial state / values

    /// <inheritdoc />
    public override bool SetTrialStateValues(int trialId, TrialState state, IReadOnlyList<double>? values = null)
    {
        lock (_lock)
        {
            using var tx = _conn.BeginTransaction();
            try
            {
                var trial = FindTrialOrRaise(trialId, tx);
                CheckTrialIsUpdatableOrRaise(trial);

                if (values is not null)
                {
                    for (int objective = 0; objective < values.Count; objective++)
                        SetTrialValueNoCommit(tx, trialId, objective, values[objective]);
                }

                if (state == TrialState.Running && trial.State != TrialState.Waiting)
                {
                    tx.Rollback();
                    return false;
                }

                Exec("UPDATE trials SET state = $p0 WHERE trial_id = $p1", tx, RdbModels.StateToDb(state), trialId);
                if (state == TrialState.Running)
                    Exec("UPDATE trials SET datetime_start = $p0 WHERE trial_id = $p1",
                        tx, RdbModels.FormatDatetime(DateTime.Now), trialId);
                if (state.IsFinished())
                    Exec("UPDATE trials SET datetime_complete = $p0 WHERE trial_id = $p1",
                        tx, RdbModels.FormatDatetime(DateTime.Now), trialId);

                tx.Commit();
            }
            catch (SqliteException e) when (IsUniqueViolation(e))
            {
                tx.Rollback();
                return false;
            }
            return true;
        }
    }

    /// <summary>Upserts a tagged objective value inside the caller-owned transaction.</summary>
    /// <param name="tx">The current SQLite transaction, or null to execute outside an explicit transaction.</param>
    /// <param name="trialId">The storage-wide trial identifier, distinct from its study-local number.</param>
    /// <param name="objective">The zero-based objective index.</param>
    /// <param name="value">The value to serialize, persist, or convert to the requested representation.</param>
    private void SetTrialValueNoCommit(SqliteTransaction tx, int trialId, int objective, double value)
    {
        var (stored, type) = RdbModels.ValueToStoredRepr(value);
        Exec(
            "INSERT INTO trial_values (trial_id, objective, value, value_type) VALUES ($p0, $p1, $p2, $p3) " +
            "ON CONFLICT(trial_id, objective) DO UPDATE SET value = excluded.value, value_type = excluded.value_type",
            tx, trialId, objective, (object?)stored, type);
    }

    /// <inheritdoc />
    public override void SetTrialIntermediateValue(int trialId, long step, double intermediateValue)
    {
        lock (_lock)
        {
            using var tx = _conn.BeginTransaction();
            SetTrialIntermediateValueNoCommit(tx, trialId, step, intermediateValue);
            tx.Commit();
        }
    }

    /// <summary>Upserts a tagged intermediate report inside the caller-owned transaction.</summary>
    /// <param name="tx">The current SQLite transaction, or null to execute outside an explicit transaction.</param>
    /// <param name="trialId">The storage-wide trial identifier, distinct from its study-local number.</param>
    /// <param name="step">The nonnegative intermediate-report step; SQLite's INTEGER column stores all 64 bits, as for Python.</param>
    /// <param name="intermediateValue">The reported value, including nonfinite values represented by separate database tags.</param>
    private void SetTrialIntermediateValueNoCommit(SqliteTransaction tx, int trialId, long step, double intermediateValue)
    {
        var trial = FindTrialOrRaise(trialId, tx);
        CheckTrialIsUpdatableOrRaise(trial);

        var (stored, type) = RdbModels.IntermediateValueToStoredRepr(intermediateValue);
        Exec(
            "INSERT INTO trial_intermediate_values (trial_id, step, intermediate_value, intermediate_value_type) " +
            "VALUES ($p0, $p1, $p2, $p3) " +
            "ON CONFLICT(trial_id, step) DO UPDATE SET " +
            "intermediate_value = excluded.intermediate_value, intermediate_value_type = excluded.intermediate_value_type",
            tx, trialId, step, (object?)stored, type);
    }

    /// <inheritdoc />
    public override void SetTrialUserAttr(int trialId, string key, object? value)
        => SetTrialAttr("trial_user_attributes", trialId, key, value);

    /// <inheritdoc />
    public override void SetTrialSystemAttr(int trialId, string key, object? value)
        => SetTrialAttr("trial_system_attributes", trialId, key, value);

    /// <summary>Upserts a JSON-serializable attribute after verifying that its trial is still mutable.</summary>
    /// <param name="table">The trusted schema table name selected by the storage implementation.</param>
    /// <param name="trialId">The storage-wide trial identifier, distinct from its study-local number.</param>
    /// <param name="key">The attribute name or exact storage key to access.</param>
    /// <param name="value">The value to serialize, persist, or convert to the requested representation.</param>
    private void SetTrialAttr(string table, int trialId, string key, object? value)
    {
        lock (_lock)
        {
            using var tx = _conn.BeginTransaction();
            SetTrialAttrNoCommit(tx, table, trialId, key, value);
            tx.Commit();
        }
    }

    /// <summary>Upserts a trial attribute inside the caller-owned transaction without committing it.</summary>
    /// <param name="tx">The current SQLite transaction, or null to execute outside an explicit transaction.</param>
    /// <param name="table">The trusted schema table name selected by the storage implementation.</param>
    /// <param name="trialId">The storage-wide trial identifier, distinct from its study-local number.</param>
    /// <param name="key">The attribute name or exact storage key to access.</param>
    /// <param name="value">The value to serialize, persist, or convert to the requested representation.</param>
    private void SetTrialAttrNoCommit(SqliteTransaction tx, string table, int trialId, string key, object? value)
    {
        var trial = FindTrialOrRaise(trialId, tx);
        CheckTrialIsUpdatableOrRaise(trial);
        Exec(
            $"INSERT INTO {table} (trial_id, \"key\", value_json) VALUES ($p0, $p1, $p2) " +
            "ON CONFLICT(trial_id, \"key\") DO UPDATE SET value_json = excluded.value_json",
            tx, trialId, key, PyJson.Dumps(value));
    }

    // ------------------------------------------------------------------ trial access

    /// <inheritdoc />
    /// <exception cref="NumSharp.KeyError">The requested trial number does not exist in the study.</exception>
    public override int GetTrialIdFromStudyIdTrialNumber(int studyId, int trialNumber)
    {
        lock (_lock)
        {
            long? trialId = ScalarNullableLong(
                "SELECT trial_id FROM trials WHERE number = $p0 AND study_id = $p1", null, trialNumber, studyId);
            if (trialId is null)
                throw new KeyError(
                    $"No trial with trial number {trialNumber} exists in study with study_id {studyId}.");
            return (int)trialId.Value;
        }
    }

    /// <inheritdoc />
    public override FrozenTrial GetTrial(int trialId)
    {
        lock (_lock)
        {
            EnsureTrialExists(trialId, null);
            var trials = GetTrialsInternal(
                studyId: null, singleTrialId: trialId, states: null, includedTrialIds: null, trialIdGreaterThan: -1);
            return trials[0];
        }
    }

    /// <inheritdoc />
    public override List<FrozenTrial> GetAllTrials(
        int studyId, bool deepcopy = true, IReadOnlyCollection<TrialState>? states = null)
    {
        lock (_lock)
        {
            EnsureStudyExists(studyId, null);
            // Every trial is reconstructed fresh from the database, so it is already an independent
            // copy — the `deepcopy` flag has no separate effect (matching Optuna's RDB read path).
            return GetTrialsInternal(studyId, singleTrialId: null, states, includedTrialIds: null, trialIdGreaterThan: -1);
        }
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
        lock (_lock)
        {
            EnsureStudyExists(studyId, null);
            return GetTrialsInternal(studyId, singleTrialId: null, states, includedTrialIds, trialIdGreaterThan);
        }
    }

    /// <summary>Materializes matching trial records and their parameters, values, and attributes.</summary>
    /// <param name="studyId">The storage-wide study identifier, not its display name.</param>
    /// <param name="singleTrialId">An optional identifier restricting the query to one trial.</param>
    /// <param name="states">The allowed trial states, or null to include every state.</param>
    /// <param name="includedTrialIds">Trial identifiers to include even when they are at or below the incremental cursor.</param>
    /// <param name="trialIdGreaterThan">The exclusive lower trial-id bound for incremental reads.</param>
    /// <returns>Matching trial snapshots with their child records materialized.</returns>
    private List<FrozenTrial> GetTrialsInternal(
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

        var sql = new StringBuilder(
            "SELECT trial_id, number, state, datetime_start, datetime_complete FROM trials WHERE ");
        var args = new List<object?>();

        if (singleTrialId is not null)
        {
            sql.Append("trial_id = $p0");
            args.Add(singleTrialId.Value);
        }
        else
        {
            sql.Append("study_id = $p0");
            args.Add(studyId!.Value);

            if (states is not null)
            {
                sql.Append(" AND state IN (");
                int i = 0;
                foreach (var st in states)
                {
                    if (i > 0)
                        sql.Append(',');
                    sql.Append("$p").Append(args.Count);
                    args.Add(RdbModels.StateToDb(st));
                    i++;
                }
                sql.Append(')');
                if (i == 0)
                    return new List<FrozenTrial>();
            }

            if (included.Count > 0 && trialIdGreaterThan > -1)
            {
                sql.Append(" AND (trial_id > $p").Append(args.Count);
                args.Add(trialIdGreaterThan);
                sql.Append(" OR trial_id IN (");
                for (int i = 0; i < included.Count; i++)
                {
                    if (i > 0)
                        sql.Append(',');
                    sql.Append("$p").Append(args.Count);
                    args.Add(included[i]);
                }
                sql.Append("))");
            }
            else if (trialIdGreaterThan > -1)
            {
                sql.Append(" AND trial_id > $p").Append(args.Count);
                args.Add(trialIdGreaterThan);
            }
        }

        sql.Append(" ORDER BY trial_id");

        var trialRows = new List<(int TrialId, int Number, TrialState State, string? Start, string? Complete)>();
        Query(sql.ToString(), null, r => trialRows.Add((
            (int)r.GetInt64(0),
            (int)r.GetInt64(1),
            RdbModels.StateFromDb(r.GetString(2)),
            r.IsDBNull(3) ? null : r.GetString(3),
            r.IsDBNull(4) ? null : r.GetString(4))), args.ToArray());

        if (trialRows.Count == 0)
            return new List<FrozenTrial>();

        var trialIds = trialRows.Select(t => t.TrialId).ToList();
        var paramsByTrial = LoadTrialParams(trialIds);
        var valuesByTrial = LoadTrialValues(trialIds);
        var intermediateByTrial = LoadTrialIntermediateValues(trialIds);
        var userByTrial = LoadTrialAttrs("trial_user_attributes", "trial_user_attribute_id", trialIds);
        var systemByTrial = LoadTrialAttrs("trial_system_attributes", "trial_system_attribute_id", trialIds);

        var result = new List<FrozenTrial>(trialRows.Count);
        foreach (var row in trialRows)
        {
            result.Add(BuildFrozenTrial(
                row.TrialId, row.Number, row.State, row.Start, row.Complete,
                paramsByTrial.GetValueOrDefault(row.TrialId),
                valuesByTrial.GetValueOrDefault(row.TrialId),
                intermediateByTrial.GetValueOrDefault(row.TrialId),
                userByTrial.GetValueOrDefault(row.TrialId),
                systemByTrial.GetValueOrDefault(row.TrialId)));
        }
        return result;
    }

    /// <summary>Combines a trial row and its child rows into the public trial snapshot.</summary>
    /// <param name="trialId">The storage-wide trial identifier, distinct from its study-local number.</param>
    /// <param name="number">The zero-based trial number within its study.</param>
    /// <param name="state">The trial lifecycle state to convert or persist.</param>
    /// <param name="datetimeStart">The stored trial-start timestamp, or null if the trial has not started.</param>
    /// <param name="datetimeComplete">The stored completion timestamp, or null for an unfinished trial.</param>
    /// <param name="paramRows">The parameter rows and serialized distributions belonging to this trial, or null when absent.</param>
    /// <param name="valueRows">Objective rows indexed by their objective number, or null when absent.</param>
    /// <param name="intermediateRows">Intermediate-report rows indexed by step, or null when absent.</param>
    /// <param name="userRows">Serialized user attributes for the trial, or null when absent.</param>
    /// <param name="systemRows">Serialized system attributes for the trial, or null when absent.</param>
    /// <returns>A detached snapshot carrying the supplied trial data.</returns>
    private FrozenTrial BuildFrozenTrial(
        int trialId,
        int number,
        TrialState state,
        string? datetimeStart,
        string? datetimeComplete,
        List<(long ParamId, string Name, double Value, string DistJson)>? paramRows,
        List<(int Objective, double? Value, string Type)>? valueRows,
        List<(long Step, double? Value, string Type)>? intermediateRows,
        List<(string Key, string Json)>? userRows,
        List<(string Key, string Json)>? systemRows)
    {
        List<double>? values = null;
        if (valueRows is { Count: > 0 })
        {
            values = new List<double>(new double[valueRows.Count]);
            foreach (var (objective, value, type) in valueRows)
                values[objective] = RdbModels.StoredReprToValue(value, type);
        }

        var @params = new OrderedDict<string, object?>();
        var distributions = new OrderedDict<string, BaseDistribution>();
        if (paramRows is not null)
        {
            foreach (var (_, name, value, distJson) in paramRows.OrderBy(p => p.ParamId))
            {
                var distribution = Distribution.FromJson(distJson);
                @params[name] = distribution.ToExternalRepr(value);
                distributions[name] = distribution;
            }
        }

        var userAttrs = new OrderedDict<string, object?>();
        if (userRows is not null)
            foreach (var (key, json) in userRows)
                userAttrs[key] = PyJson.Loads(json);

        var systemAttrs = new OrderedDict<string, object?>();
        if (systemRows is not null)
            foreach (var (key, json) in systemRows)
                systemAttrs[key] = PyJson.Loads(json);

        var intermediateValues = new OrderedDict<long, double>();
        if (intermediateRows is not null)
            foreach (var (step, value, type) in intermediateRows)
                intermediateValues[step] = RdbModels.StoredReprToIntermediateValue(value, type);

        return new FrozenTrial(
            number: number,
            state: state,
            value: null,
            values: values,
            datetimeStart: RdbModels.ParseDatetime(datetimeStart),
            datetimeComplete: RdbModels.ParseDatetime(datetimeComplete),
            @params: @params,
            distributions: distributions,
            userAttrs: userAttrs,
            systemAttrs: systemAttrs,
            intermediateValues: intermediateValues,
            trialId: trialId);
    }

    /// <summary>Loads serialized distributions and parameter values grouped by trial identifier.</summary>
    /// <param name="trialIds">The storage-wide identifiers whose child records are read.</param>
    /// <returns>Parameter rows grouped by storage-wide trial identifier.</returns>
    private Dictionary<int, List<(long ParamId, string Name, double Value, string DistJson)>> LoadTrialParams(
        IReadOnlyList<int> trialIds)
    {
        var byTrial = new Dictionary<int, List<(long, string, double, string)>>();
        ReadChildrenInChunks(
            "trial_id, param_id, param_name, param_value, distribution_json", "trial_params", trialIds,
            r =>
            {
                int tid = (int)r.GetInt64(0);
                if (!byTrial.TryGetValue(tid, out var list))
                    byTrial[tid] = list = new List<(long, string, double, string)>();
                list.Add((r.GetInt64(1), r.GetString(2), r.GetDouble(3), r.GetString(4)));
            });
        return byTrial;
    }

    /// <summary>Loads objective values and their nonfinite-value tags grouped by trial identifier.</summary>
    /// <param name="trialIds">The storage-wide identifiers whose child records are read.</param>
    /// <returns>Objective rows grouped by storage-wide trial identifier.</returns>
    private Dictionary<int, List<(int Objective, double? Value, string Type)>> LoadTrialValues(
        IReadOnlyList<int> trialIds)
    {
        var byTrial = new Dictionary<int, List<(int, double?, string)>>();
        ReadChildrenInChunks(
            "trial_id, objective, value, value_type", "trial_values", trialIds,
            r =>
            {
                int tid = (int)r.GetInt64(0);
                if (!byTrial.TryGetValue(tid, out var list))
                    byTrial[tid] = list = new List<(int, double?, string)>();
                list.Add(((int)r.GetInt64(1), r.IsDBNull(2) ? null : r.GetDouble(2), r.GetString(3)));
            });
        return byTrial;
    }

    /// <summary>Loads intermediate reports and their nonfinite-value tags grouped by trial identifier.</summary>
    /// <param name="trialIds">The storage-wide identifiers whose child records are read.</param>
    /// <returns>Intermediate-report rows grouped by storage-wide trial identifier.</returns>
    private Dictionary<int, List<(long Step, double? Value, string Type)>> LoadTrialIntermediateValues(
        IReadOnlyList<int> trialIds)
    {
        var byTrial = new Dictionary<int, List<(long, double?, string)>>();
        ReadChildrenInChunks(
            "trial_id, step, intermediate_value, intermediate_value_type", "trial_intermediate_values", trialIds,
            r =>
            {
                int tid = (int)r.GetInt64(0);
                if (!byTrial.TryGetValue(tid, out var list))
                    byTrial[tid] = list = new List<(long, double?, string)>();
                // The full 64-bit step: an (int) cast silently wrapped a Python-written step past Int32 (to
                // -1,294,967,296 for 3,000,000,000), and a C# process copying the trial persisted the corruption.
                list.Add((r.GetInt64(1), r.IsDBNull(2) ? null : r.GetDouble(2), r.GetString(3)));
            });
        return byTrial;
    }

    /// <summary>Loads serialized trial attributes grouped by trial identifier.</summary>
    /// <param name="table">The trusted schema table name selected by the storage implementation.</param>
    /// <param name="idColumn">The trusted identifier column used to group records.</param>
    /// <param name="trialIds">The storage-wide identifiers whose child records are read.</param>
    /// <returns>Attribute rows grouped by storage-wide trial identifier.</returns>
    private Dictionary<int, List<(string Key, string Json)>> LoadTrialAttrs(
        string table, string idColumn, IReadOnlyList<int> trialIds)
    {
        var byTrial = new Dictionary<int, List<(string, string)>>();
        ReadChildrenInChunks(
            $"trial_id, \"key\", value_json", table, trialIds,
            r =>
            {
                int tid = (int)r.GetInt64(0);
                if (!byTrial.TryGetValue(tid, out var list))
                    byTrial[tid] = list = new List<(string, string)>();
                list.Add((r.GetString(1), r.GetString(2)));
            },
            orderBy: idColumn);
        return byTrial;
    }

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
        lock (_lock)
        {
            // ORDER BY treats -inf < finite < +inf via the value_type tag, then by the stored value
            // (Python `find_max_value_trial_id` / `find_min_value_trial_id`).
            const string typeRank =
                "CASE tv.value_type WHEN 'INF_NEG' THEN -1 WHEN 'FINITE' THEN 0 WHEN 'INF_POS' THEN 1 END";
            string order = direction == StudyDirection.Maximize
                ? $"{typeRank} DESC, tv.value DESC"
                : $"{typeRank} ASC, tv.value ASC";
            long? trialId = ScalarNullableLong(
                "SELECT t.trial_id FROM trials t JOIN trial_values tv ON tv.trial_id = t.trial_id " +
                "WHERE t.study_id = $p0 AND t.state = $p1 AND tv.objective = 0 " +
                $"ORDER BY {order} LIMIT 1",
                null, studyId, RdbModels.StateToDb(TrialState.Complete));
            if (trialId is null)
                throw new ValueError(RdbModels.NotFoundMsg);
            return (int)trialId.Value;
        }
    }

    // ------------------------------------------------------------------ heartbeat

    /// <summary>Inserts or refreshes the trial heartbeat using the database clock or storage timestamp convention.</summary>
    /// <param name="trialId">The storage-wide trial identifier, distinct from its study-local number.</param>
    public void RecordHeartbeat(int trialId)
    {
        lock (_lock)
        {
            Exec(
                "INSERT INTO trial_heartbeats (trial_id, heartbeat) VALUES ($p0, CURRENT_TIMESTAMP) " +
                "ON CONFLICT(trial_id) DO UPDATE SET heartbeat = CURRENT_TIMESTAMP",
                null, trialId);
        }
    }

    /// <summary>Finds running trials whose recorded heartbeat is older than the configured grace period.</summary>
    /// <param name="studyId">The storage-wide study identifier, not its display name.</param>
    /// <returns>Identifiers of running trials whose existing heartbeat has expired.</returns>
    public List<int> GetStaleTrialIds(int studyId)
    {
        lock (_lock)
        {
            int gracePeriod = _gracePeriod ?? 2 * _heartbeatInterval!.Value;
            string? nowStr = ScalarString("SELECT CURRENT_TIMESTAMP", null);
            var now = DateTime.SpecifyKind(
                DateTime.ParseExact(nowStr!, "yyyy-MM-dd HH:mm:ss", CultureInfo.InvariantCulture, DateTimeStyles.None),
                DateTimeKind.Utc);

            var stale = new List<int>();
            Query(
                "SELECT t.trial_id, h.heartbeat FROM trials t JOIN trial_heartbeats h ON h.trial_id = t.trial_id " +
                "WHERE t.study_id = $p0 AND t.state = $p1",
                null,
                r =>
                {
                    var heartbeat = DateTime.SpecifyKind(
                        DateTime.ParseExact(r.GetString(1), "yyyy-MM-dd HH:mm:ss", CultureInfo.InvariantCulture, DateTimeStyles.None),
                        DateTimeKind.Utc);
                    if ((now - heartbeat).TotalSeconds > gracePeriod)
                        stale.Add((int)r.GetInt64(0));
                },
                studyId, RdbModels.StateToDb(TrialState.Running));
            return stale;
        }
    }

    /// <summary>Returns the configured heartbeat recording interval.</summary>
    /// <returns>The heartbeat interval in seconds, or null when heartbeat recording is disabled.</returns>
    public int? GetHeartbeatInterval() => _heartbeatInterval;

    /// <summary>Returns the callback invoked after a heartbeat-stale trial is failed.</summary>
    /// <returns>The configured stale-trial callback, or null if no callback was supplied.</returns>
    public Action<Study, FrozenTrial>? GetHeartbeatStaleTrialCallback() => _heartbeatStaleTrialCallback;

    // ------------------------------------------------------------------ helpers

    /// <summary>Verifies that the requested study record exists before accessing its children.</summary>
    /// <param name="studyId">The storage-wide study identifier, not its display name.</param>
    /// <param name="tx">The current SQLite transaction, or null to execute outside an explicit transaction.</param>
    /// <exception cref="NumSharp.KeyError">The study identifier does not exist.</exception>
    private void EnsureStudyExists(int studyId, SqliteTransaction? tx)
    {
        long? id = ScalarNullableLong("SELECT study_id FROM studies WHERE study_id = $p0", tx, studyId);
        if (id is null)
            throw new KeyError(RdbModels.NotFoundMsg);
    }

    /// <summary>Verifies that the requested trial record exists before accessing its children.</summary>
    /// <param name="trialId">The storage-wide trial identifier, distinct from its study-local number.</param>
    /// <param name="tx">The current SQLite transaction, or null to execute outside an explicit transaction.</param>
    /// <exception cref="NumSharp.KeyError">The trial identifier does not exist.</exception>
    private void EnsureTrialExists(int trialId, SqliteTransaction? tx)
    {
        long? id = ScalarNullableLong("SELECT trial_id FROM trials WHERE trial_id = $p0", tx, trialId);
        if (id is null)
            throw new KeyError(RdbModels.NotFoundMsg);
    }

    /// <summary>Reads the trial number, state, and owning study needed to validate an update.</summary>
    /// <param name="trialId">The storage-wide trial identifier, distinct from its study-local number.</param>
    /// <param name="tx">The current SQLite transaction, or null to execute outside an explicit transaction.</param>
    /// <returns>The trial's study-local number, state, and owning study identifier.</returns>
    /// <exception cref="NumSharp.KeyError">The trial identifier does not exist.</exception>
    private (int Number, TrialState State, int StudyId) FindTrialOrRaise(int trialId, SqliteTransaction? tx)
    {
        (int, TrialState, int)? found = null;
        Query(
            "SELECT number, state, study_id FROM trials WHERE trial_id = $p0", tx,
            r => found = (
                r.IsDBNull(0) ? -1 : (int)r.GetInt64(0),
                RdbModels.StateFromDb(r.GetString(1)),
                (int)r.GetInt64(2)),
            trialId);
        if (found is null)
            throw new KeyError(RdbModels.NotFoundMsg);
        return found.Value;
    }

    /// <summary>Rejects mutations of a trial that has reached a finished state.</summary>
    /// <param name="trial">The trial snapshot or database record being read or validated.</param>
    /// <exception cref="UpdateFinishedTrialError">The trial is already complete, pruned, or failed.</exception>
    private static void CheckTrialIsUpdatableOrRaise((int Number, TrialState State, int StudyId) trial)
    {
        if (trial.State.IsFinished())
            throw new UpdateFinishedTrialError(
                $"Trial#{trial.Number} has already finished and can not be updated.");
    }

    /// <summary>Converts an optional timestamp to its database representation without inventing a missing date.</summary>
    /// <param name="value">The value to serialize, persist, or convert to the requested representation.</param>
    /// <returns>The serialized timestamp or null when the timestamp is absent.</returns>
    private static object? OrNull(DateTime? value)
        => value is null ? null : RdbModels.FormatDatetime(value.Value);

    /// <summary>Recognizes provider-specific uniqueness failures so duplicate studies can be translated consistently.</summary>
    /// <param name="e">The provider exception to classify or translate.</param>
    /// <returns>Whether the exception denotes a uniqueness-constraint violation.</returns>
    private static bool IsUniqueViolation(SqliteException e)
        // SQLITE_CONSTRAINT (19) — includes the UNIQUE-constraint sub-code 2067.
        => e.SqliteErrorCode == 19;

    /// <summary>Reads trial child rows in bounded identifier batches to respect SQLite parameter limits.</summary>
    /// <param name="columns">The trusted projection columns selected from the schema.</param>
    /// <param name="table">The trusted schema table name selected by the storage implementation.</param>
    /// <param name="trialIds">The storage-wide identifiers whose child records are read.</param>
    /// <param name="onRow">The callback consuming each current reader row before the reader advances or is disposed.</param>
    /// <param name="orderBy">An optional trusted ordering clause applied to child-row queries.</param>
    private void ReadChildrenInChunks(
        string columns, string table, IReadOnlyList<int> trialIds, Action<SqliteDataReader> onRow, string? orderBy = null)
    {
        const int chunkSize = 500;
        for (int start = 0; start < trialIds.Count; start += chunkSize)
        {
            int n = Math.Min(chunkSize, trialIds.Count - start);
            var sql = new StringBuilder("SELECT ").Append(columns).Append(" FROM ").Append(table)
                .Append(" WHERE trial_id IN (");
            var args = new object?[n];
            for (int i = 0; i < n; i++)
            {
                if (i > 0)
                    sql.Append(',');
                sql.Append("$p").Append(i);
                args[i] = trialIds[start + i];
            }
            sql.Append(')');
            if (orderBy is not null)
                sql.Append(" ORDER BY ").Append(orderBy);
            Query(sql.ToString(), null, onRow, args);
        }
    }

    /// <summary>Creates a parameterized SQLite command associated with the selected transaction.</summary>
    /// <param name="sql">The implementation-owned SQL command text with positional parameter placeholders.</param>
    /// <param name="tx">The current SQLite transaction, or null to execute outside an explicit transaction.</param>
    /// <param name="args">Values bound to the SQL command parameters in placeholder order.</param>
    /// <returns>A command owned by the caller, which must dispose it after execution.</returns>
    private SqliteCommand NewCommand(string sql, SqliteTransaction? tx, params object?[] args)
    {
        var cmd = _conn.CreateCommand();
        cmd.CommandText = sql;
        if (tx is not null)
            cmd.Transaction = tx;
        for (int i = 0; i < args.Length; i++)
            cmd.Parameters.AddWithValue("$p" + i, args[i] ?? DBNull.Value);
        return cmd;
    }

    /// <summary>Executes a parameterized non-query and disposes its command.</summary>
    /// <param name="sql">The implementation-owned SQL command text with positional parameter placeholders.</param>
    /// <param name="tx">The current SQLite transaction, or null to execute outside an explicit transaction.</param>
    /// <param name="args">Values bound to the SQL command parameters in placeholder order.</param>
    private void Exec(string sql, SqliteTransaction? tx, params object?[] args)
    {
        using var cmd = NewCommand(sql, tx, args);
        cmd.ExecuteNonQuery();
    }

    /// <summary>Invokes a callback for each result row while the SQLite reader remains open.</summary>
    /// <param name="sql">The implementation-owned SQL command text with positional parameter placeholders.</param>
    /// <param name="tx">The current SQLite transaction, or null to execute outside an explicit transaction.</param>
    /// <param name="onRow">The callback consuming each current reader row before the reader advances or is disposed.</param>
    /// <param name="args">Values bound to the SQL command parameters in placeholder order.</param>
    private void Query(string sql, SqliteTransaction? tx, Action<SqliteDataReader> onRow, params object?[] args)
    {
        using var cmd = NewCommand(sql, tx, args);
        using var reader = cmd.ExecuteReader();
        while (reader.Read())
            onRow(reader);
    }

    /// <summary>Reads a required integer scalar from a parameterized SQLite query.</summary>
    /// <param name="sql">The implementation-owned SQL command text with positional parameter placeholders.</param>
    /// <param name="tx">The current SQLite transaction, or null to execute outside an explicit transaction.</param>
    /// <param name="args">Values bound to the SQL command parameters in placeholder order.</param>
    /// <returns>The first scalar converted to a 64-bit integer.</returns>
    private long ScalarLong(string sql, SqliteTransaction? tx, params object?[] args)
    {
        using var cmd = NewCommand(sql, tx, args);
        return Convert.ToInt64(cmd.ExecuteScalar(), CultureInfo.InvariantCulture);
    }

    /// <summary>Reads an optional integer scalar from a parameterized SQLite query.</summary>
    /// <param name="sql">The implementation-owned SQL command text with positional parameter placeholders.</param>
    /// <param name="tx">The current SQLite transaction, or null to execute outside an explicit transaction.</param>
    /// <param name="args">Values bound to the SQL command parameters in placeholder order.</param>
    /// <returns>The first scalar converted to a 64-bit integer, or null for SQL NULL or no result.</returns>
    private long? ScalarNullableLong(string sql, SqliteTransaction? tx, params object?[] args)
    {
        using var cmd = NewCommand(sql, tx, args);
        object? value = cmd.ExecuteScalar();
        return value is null or DBNull ? null : Convert.ToInt64(value, CultureInfo.InvariantCulture);
    }

    /// <summary>Reads an optional floating-point scalar from a parameterized SQLite query.</summary>
    /// <param name="sql">The implementation-owned SQL command text with positional parameter placeholders.</param>
    /// <param name="tx">The current SQLite transaction, or null to execute outside an explicit transaction.</param>
    /// <param name="args">Values bound to the SQL command parameters in placeholder order.</param>
    /// <returns>The first scalar converted to a double, or null for SQL NULL or no result.</returns>
    private double? ScalarNullableDouble(string sql, SqliteTransaction? tx, params object?[] args)
    {
        using var cmd = NewCommand(sql, tx, args);
        object? value = cmd.ExecuteScalar();
        return value is null or DBNull ? null : Convert.ToDouble(value, CultureInfo.InvariantCulture);
    }

    /// <summary>Reads an optional string scalar from a parameterized SQLite query.</summary>
    /// <param name="sql">The implementation-owned SQL command text with positional parameter placeholders.</param>
    /// <param name="tx">The current SQLite transaction, or null to execute outside an explicit transaction.</param>
    /// <param name="args">Values bound to the SQL command parameters in placeholder order.</param>
    /// <returns>The first scalar as a string, or null for SQL NULL or no result.</returns>
    private string? ScalarString(string sql, SqliteTransaction? tx, params object?[] args)
    {
        using var cmd = NewCommand(sql, tx, args);
        object? value = cmd.ExecuteScalar();
        return value is null or DBNull ? null : Convert.ToString(value, CultureInfo.InvariantCulture);
    }

    /// <summary>
    ///     Disposes the underlying <see cref="SqliteConnection" /> — unless it was injected with
    ///     <c>ownsConnection: false</c>, in which case the caller retains ownership and it is left open.
    /// </summary>
    public void Dispose()
    {
        if (_ownsConnection)
            _conn.Dispose();
    }
}

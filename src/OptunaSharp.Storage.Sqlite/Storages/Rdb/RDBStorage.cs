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
public sealed class RDBStorage : BaseStorage, ICachedStorageBackend, IDisposable
{
    private static readonly Logger _logger = OptunaLogging.GetLogger("optuna.storages._rdb.storage");

    private readonly SqliteConnection _conn;
    private readonly object _lock = new();

    private readonly int? _heartbeatInterval;
    private readonly int? _gracePeriod;
    private readonly Action<Study, FrozenTrial>? _heartbeatStaleTrialCallback;

    /// <summary>The storage URL, after <c>{SCHEMA_VERSION}</c> template substitution.</summary>
    public string Url { get; }

    /// <summary>Create an RDB storage from a SQLAlchemy-style SQLite URL.</summary>
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
    public RDBStorage(
        string url,
        bool skipCompatibilityCheck = false,
        bool skipTableCreation = false,
        int? heartbeatInterval = null,
        int? gracePeriod = null,
        Action<Study, FrozenTrial>? heartbeatStaleTrialCallback = null)
    {
        if (heartbeatInterval is not null and <= 0)
            throw new ValueError("The value of `heartbeat_interval` should be a positive integer.");
        if (gracePeriod is not null and <= 0)
            throw new ValueError("The value of `grace_period` should be a positive integer.");

        Url = FillStorageUrlTemplate(url);
        _heartbeatInterval = heartbeatInterval;
        _gracePeriod = gracePeriod;
        _heartbeatStaleTrialCallback = heartbeatStaleTrialCallback;

        _conn = new SqliteConnection(ResolveConnectionString(Url));
        _conn.Open();
        Exec("PRAGMA busy_timeout=30000", null);

        if (!skipTableCreation)
        {
            foreach (var stmt in RdbModels.SchemaStatements)
                Exec(stmt, null);
        }

        InitVersionInfo();
        InitAlembic();
        if (!skipCompatibilityCheck)
            CheckTableSchemaCompatibility();
    }

    // ------------------------------------------------------------------ URL + version bootstrap

    private static string FillStorageUrlTemplate(string template)
        => template.Replace("{SCHEMA_VERSION}", RdbModels.SchemaVersion.ToString(CultureInfo.InvariantCulture));

    private static string ResolveConnectionString(string url)
    {
        const string prefix = "sqlite:///";
        string dataSource;
        if (url.StartsWith(prefix, StringComparison.Ordinal))
            dataSource = url.Substring(prefix.Length);
        else if (url.StartsWith("sqlite://", StringComparison.Ordinal))
            dataSource = url.Substring("sqlite://".Length);
        else
            dataSource = url; // tolerate a bare file path

        var builder = new SqliteConnectionStringBuilder
        {
            DataSource = dataSource,
            Mode = dataSource == ":memory:" ? SqliteOpenMode.Memory : SqliteOpenMode.ReadWriteCreate,
        };
        return builder.ToString();
    }

    private void InitVersionInfo()
    {
        long count = ScalarLong("SELECT COUNT(*) FROM version_info", null);
        if (count > 0)
            return;
        Exec(
            "INSERT INTO version_info (version_info_id, schema_version, library_version) VALUES (1, $p0, $p1)",
            null,
            RdbModels.SchemaVersion,
            RdbModels.LibraryVersion);
    }

    private void InitAlembic()
    {
        long count = ScalarLong("SELECT COUNT(*) FROM alembic_version", null);
        if (count > 0)
            return;
        Exec("INSERT INTO alembic_version (version_num) VALUES ($p0)", null, RdbModels.HeadVersion);
    }

    private static readonly string[] KnownVersions =
    {
        "v0.9.0.a", "v1.2.0.a", "v1.3.0.a", "v2.4.0.a", "v2.6.0.a",
        "v3.0.0.a", "v3.0.0.b", "v3.0.0.c", "v3.0.0.d", "v3.2.0.a",
    };

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
        message.Append(KnownVersions.Contains(current)
            ? "Please execute `$ optuna storage upgrade --storage $STORAGE_URL` for upgrading the storage."
            : "Please try updating optuna to the latest version by `$ pip install -U optuna`.");
        throw new RuntimeError(message.ToString());
    }

    // ------------------------------------------------------------------ study manipulation

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

    public override void SetStudyUserAttr(int studyId, string key, object? value)
        => SetStudyAttr("study_user_attributes", studyId, key, value);

    public override void SetStudySystemAttr(int studyId, string key, object? value)
        => SetStudyAttr("study_system_attributes", studyId, key, value);

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

    public override OrderedDict<string, object?> GetStudyUserAttrs(int studyId)
        => GetStudyAttrs("study_user_attributes", "study_user_attribute_id", studyId);

    public override OrderedDict<string, object?> GetStudySystemAttrs(int studyId)
        => GetStudyAttrs("study_system_attributes", "study_system_attribute_id", studyId);

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

    public override int CreateNewTrial(int studyId, FrozenTrial? templateTrial = null)
        => CreateNewTrialFull(studyId, templateTrial).TrialId;

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
                intermediateValues: new OrderedDict<int, double>(),
                trialId: (int)trialId);
        }
    }

    // ------------------------------------------------------------------ trial params

    public override void SetTrialParam(
        int trialId, string paramName, double paramValueInternal, BaseDistribution distribution)
        => SetTrialParamWithCachedDistribution(trialId, paramName, paramValueInternal, distribution, null);

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

    private void SetTrialValueNoCommit(SqliteTransaction tx, int trialId, int objective, double value)
    {
        var (stored, type) = RdbModels.ValueToStoredRepr(value);
        Exec(
            "INSERT INTO trial_values (trial_id, objective, value, value_type) VALUES ($p0, $p1, $p2, $p3) " +
            "ON CONFLICT(trial_id, objective) DO UPDATE SET value = excluded.value, value_type = excluded.value_type",
            tx, trialId, objective, (object?)stored, type);
    }

    public override void SetTrialIntermediateValue(int trialId, int step, double intermediateValue)
    {
        lock (_lock)
        {
            using var tx = _conn.BeginTransaction();
            SetTrialIntermediateValueNoCommit(tx, trialId, step, intermediateValue);
            tx.Commit();
        }
    }

    private void SetTrialIntermediateValueNoCommit(SqliteTransaction tx, int trialId, int step, double intermediateValue)
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

    public override void SetTrialUserAttr(int trialId, string key, object? value)
        => SetTrialAttr("trial_user_attributes", trialId, key, value);

    public override void SetTrialSystemAttr(int trialId, string key, object? value)
        => SetTrialAttr("trial_system_attributes", trialId, key, value);

    private void SetTrialAttr(string table, int trialId, string key, object? value)
    {
        lock (_lock)
        {
            using var tx = _conn.BeginTransaction();
            SetTrialAttrNoCommit(tx, table, trialId, key, value);
            tx.Commit();
        }
    }

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

    private FrozenTrial BuildFrozenTrial(
        int trialId,
        int number,
        TrialState state,
        string? datetimeStart,
        string? datetimeComplete,
        List<(long ParamId, string Name, double Value, string DistJson)>? paramRows,
        List<(int Objective, double? Value, string Type)>? valueRows,
        List<(int Step, double? Value, string Type)>? intermediateRows,
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

        var intermediateValues = new OrderedDict<int, double>();
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

    private Dictionary<int, List<(int Step, double? Value, string Type)>> LoadTrialIntermediateValues(
        IReadOnlyList<int> trialIds)
    {
        var byTrial = new Dictionary<int, List<(int, double?, string)>>();
        ReadChildrenInChunks(
            "trial_id, step, intermediate_value, intermediate_value_type", "trial_intermediate_values", trialIds,
            r =>
            {
                int tid = (int)r.GetInt64(0);
                if (!byTrial.TryGetValue(tid, out var list))
                    byTrial[tid] = list = new List<(int, double?, string)>();
                list.Add(((int)r.GetInt64(1), r.IsDBNull(2) ? null : r.GetDouble(2), r.GetString(3)));
            });
        return byTrial;
    }

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

    public override FrozenTrial GetBestTrial(int studyId)
    {
        var directions = GetStudyDirections(studyId);
        if (directions.Count > 1)
            throw new RuntimeError("Best trial can be obtained only for single-objective optimization.");
        int trialId = GetBestTrialId(studyId, directions[0]);
        return GetTrial(trialId);
    }

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

    public int? GetHeartbeatInterval() => _heartbeatInterval;

    public Action<Study, FrozenTrial>? GetHeartbeatStaleTrialCallback() => _heartbeatStaleTrialCallback;

    // ------------------------------------------------------------------ helpers

    private void EnsureStudyExists(int studyId, SqliteTransaction? tx)
    {
        long? id = ScalarNullableLong("SELECT study_id FROM studies WHERE study_id = $p0", tx, studyId);
        if (id is null)
            throw new KeyError(RdbModels.NotFoundMsg);
    }

    private void EnsureTrialExists(int trialId, SqliteTransaction? tx)
    {
        long? id = ScalarNullableLong("SELECT trial_id FROM trials WHERE trial_id = $p0", tx, trialId);
        if (id is null)
            throw new KeyError(RdbModels.NotFoundMsg);
    }

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

    private static void CheckTrialIsUpdatableOrRaise((int Number, TrialState State, int StudyId) trial)
    {
        if (trial.State.IsFinished())
            throw new UpdateFinishedTrialError(
                $"Trial#{trial.Number} has already finished and can not be updated.");
    }

    private static object? OrNull(DateTime? value)
        => value is null ? null : RdbModels.FormatDatetime(value.Value);

    private static bool IsUniqueViolation(SqliteException e)
        // SQLITE_CONSTRAINT (19) — includes the UNIQUE-constraint sub-code 2067.
        => e.SqliteErrorCode == 19;

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

    private void Exec(string sql, SqliteTransaction? tx, params object?[] args)
    {
        using var cmd = NewCommand(sql, tx, args);
        cmd.ExecuteNonQuery();
    }

    private void Query(string sql, SqliteTransaction? tx, Action<SqliteDataReader> onRow, params object?[] args)
    {
        using var cmd = NewCommand(sql, tx, args);
        using var reader = cmd.ExecuteReader();
        while (reader.Read())
            onRow(reader);
    }

    private long ScalarLong(string sql, SqliteTransaction? tx, params object?[] args)
    {
        using var cmd = NewCommand(sql, tx, args);
        return Convert.ToInt64(cmd.ExecuteScalar(), CultureInfo.InvariantCulture);
    }

    private long? ScalarNullableLong(string sql, SqliteTransaction? tx, params object?[] args)
    {
        using var cmd = NewCommand(sql, tx, args);
        object? value = cmd.ExecuteScalar();
        return value is null or DBNull ? null : Convert.ToInt64(value, CultureInfo.InvariantCulture);
    }

    private double? ScalarNullableDouble(string sql, SqliteTransaction? tx, params object?[] args)
    {
        using var cmd = NewCommand(sql, tx, args);
        object? value = cmd.ExecuteScalar();
        return value is null or DBNull ? null : Convert.ToDouble(value, CultureInfo.InvariantCulture);
    }

    private string? ScalarString(string sql, SqliteTransaction? tx, params object?[] args)
    {
        using var cmd = NewCommand(sql, tx, args);
        object? value = cmd.ExecuteScalar();
        return value is null or DBNull ? null : Convert.ToString(value, CultureInfo.InvariantCulture);
    }

    public void Dispose() => _conn.Dispose();
}

using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Threading;
using NumSharp;
using OptunaSharp.Collections;
using OptunaSharp.Distributions;
using OptunaSharp.Internal;
using OptunaSharp.Logging;

namespace OptunaSharp.Storages.Journal;

/// <summary>Journal operation codes — Optuna's on-disk values.</summary>
public enum JournalOperation
{
    /// <summary>Creates study metadata and assigns its replay identifier.</summary>
    CreateStudy = 0,
    /// <summary>Removes a study from the replayed study registry.</summary>
    DeleteStudy = 1,
    /// <summary>Updates a study user attribute.</summary>
    SetStudyUserAttr = 2,
    /// <summary>Updates an internal study attribute.</summary>
    SetStudySystemAttr = 3,
    /// <summary>Creates a trial and assigns its replay identifier.</summary>
    CreateTrial = 4,
    /// <summary>Records a parameter and its distribution.</summary>
    SetTrialParam = 5,
    /// <summary>Changes trial state and objective values.</summary>
    SetTrialStateValues = 6,
    /// <summary>Records an objective observation at a step.</summary>
    SetTrialIntermediateValue = 7,
    /// <summary>Updates a trial user attribute.</summary>
    SetTrialUserAttr = 8,
    /// <summary>Updates an internal trial attribute.</summary>
    SetTrialSystemAttr = 9,
    /// <summary>Only supported (written) by Rustuna; skipped on replay.</summary>
    DiscardTrials = 10,
}

/// <summary>
///     Storage class for Journal storage backend.
///     Port of <c>optuna/storages/journal/_storage.py: JournalStorage</c>.
///
///     Journal storage writes a record of every operation to the database as it is executed and
///     at the same time, keeps a latest snapshot of the database in-memory. If the database
///     crashes for any reason, the storage can re-establish the contents in memory by replaying
///     the operations stored from the beginning.
/// </summary>
/// <example>
///     <code>
///     var storage = new JournalStorage(new JournalFileBackend("./optuna_journal_storage.log"));
///     var study = Optuna.CreateStudy(storage: storage);
///     study.Optimize(objective);
///     </code>
///     On Windows, prefer the open lock:
///     <code>
///     var filePath = "./optuna_journal_storage.log";
///     var lockObj = new JournalFileOpenLock(filePath);
///     var storage = new JournalStorage(new JournalFileBackend(filePath, lockObj: lockObj));
///     </code>
/// </example>
public class JournalStorage : BaseStorage
{
    /// <summary>Message used when journal replay cannot find a requested record.</summary>
    internal const string NotFoundMsg = "Record does not exist.";
    /// <summary>Format string used when a finished trial is modified.</summary>
    internal const string UnupdatableMsgFormat = "Trial#{0} has already finished and can not be updated.";
    // A heuristic interval number to dump snapshots.
    /// <summary>
    ///     Optuna's <c>SNAPSHOT_INTERVAL</c>, the number of study or trial identifiers between replay snapshots: what a
    ///     storage uses when neither its constructor nor its backend (<see cref="IJournalSnapshotInterval" />) chooses.
    /// </summary>
    internal const int OptunaSnapshotInterval = 100;

    private static readonly Logger _logger = OptunaLogging.GetLogger("optuna.storages.journal._storage");

    private readonly string _workerIdPrefix = Guid.NewGuid() + "-";
    private readonly BaseJournalBackend _backend;
    private readonly object _threadLock = new();
    private readonly int _snapshotInterval;
    private JournalStorageReplayResult _replayResult;

    /// <summary>Restores an optional snapshot and replays the journal to initialize a synchronized storage view.</summary>
    /// <param name="logStorage">Shared process-safe journal backend; its lifetime belongs to the caller.</param>
    /// <param name="snapshotInterval">
    ///     For a backend that stores snapshots (<see cref="IBaseJournalSnapshot" />: the Redis backends), how many study
    ///     or trial ids apart this storage saves one: the worker that creates a study or trial whose id is a multiple of
    ///     this value saves the whole replay state. <c>null</c> (the default) lets the backend choose through
    ///     <see cref="IJournalSnapshotInterval" /> (the Redis backends: Optuna's 100 in both read modes),
    ///     and falls back to Optuna's fixed <c>SNAPSHOT_INTERVAL</c> of 100. Optuna has no such parameter. See the
    ///     remarks for the trade-off; <see cref="SnapshotInterval" /> reports the value this storage uses.
    /// </param>
    /// <remarks>
    ///     <para>
    ///         A storage opened over a journal first restores the latest snapshot and then replays only the records
    ///         written after it, so its cold start reads at most about one interval of records. Each snapshot serializes
    ///         every trial of the journal under this storage's lock, so a snapshot costs more the longer the journal grows,
    ///         and a smaller interval saves one more often. Workers that open and close storages often (the
    ///         <c>optunasharp</c> CLI opens one per command) gain from a snapshot when replaying records is expensive;
    ///         long-lived workers sync incrementally and never read one.
    ///     </para>
    ///     <para>
    ///         Measured on 2026-11-07 at 20,000 two-parameter trials on a standalone Redis server (Release build, with the
    ///         binary snapshot of plan 10, item 2.7), where records are read in batches: a cold start took 109 ms, 170 ms
    ///         and 927 ms at an interval of 100, 1000 and no snapshot, and the writer 9.76 ms per trial at 100 against
    ///         10.74 with no snapshot. In cluster mode, where every record costs a round trip, the same cold start took
    ///         0.43 s, 3.3 s and 70.8 s. A snapshot of that study saves in about 20 ms and restores in about 40 ms.
    ///     </para>
    ///     <para>
    ///         OptunaSharp's snapshots are its own (a binary format, under their own Redis key): Python workers on the
    ///         same journal neither read nor write them, whatever interval either side uses.
    ///     </para>
    /// </remarks>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="snapshotInterval" /> is below 1.</exception>
    /// <exception cref="ArgumentException">
    ///     <paramref name="snapshotInterval" /> is <c>null</c> and the backend's
    ///     <see cref="IJournalSnapshotInterval.DefaultSnapshotInterval" /> is below 1.
    /// </exception>
    /// <exception cref="System.Exception">The backend cannot load a snapshot, read records, or replay their contents.</exception>
    public JournalStorage(BaseJournalBackend logStorage, int? snapshotInterval = null)
    {
        _snapshotInterval = ResolveSnapshotInterval(logStorage, snapshotInterval);
        _backend = logStorage;
        _replayResult = new JournalStorageReplayResult(_workerIdPrefix);

        lock (_threadLock)
        {
            if (_backend is IBaseJournalSnapshot snapshotBackend)
            {
                var snapshot = snapshotBackend.LoadSnapshot();
                if (snapshot is not null)
                    RestoreReplayResult(snapshot);
            }
            SyncWithBackend();
        }
    }

    /// <summary>
    ///     How many study or trial ids apart this storage saves a replay snapshot, resolved once by the constructor: its
    ///     <c>snapshotInterval</c> argument, else the backend's <see cref="IJournalSnapshotInterval.DefaultSnapshotInterval" />,
    ///     else Optuna's 100. Unused when the backend stores no snapshots (it does not implement
    ///     <see cref="IBaseJournalSnapshot" />).
    /// </summary>
    public int SnapshotInterval => _snapshotInterval;

    /// <summary>
    ///     Picks the snapshot interval in the constructor's order: the caller's value, the backend's, then Optuna's.
    /// </summary>
    /// <param name="logStorage">The backend, asked through <see cref="IJournalSnapshotInterval" /> when the caller gives no value.</param>
    /// <param name="snapshotInterval">The constructor's argument; <c>null</c> lets the backend choose.</param>
    /// <returns>The interval the storage uses, at least 1.</returns>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="snapshotInterval" /> is below 1.</exception>
    /// <exception cref="ArgumentException">
    ///     <paramref name="snapshotInterval" /> is <c>null</c> and the backend's interval is below 1.
    /// </exception>
    private static int ResolveSnapshotInterval(BaseJournalBackend logStorage, int? snapshotInterval)
    {
        if (snapshotInterval is { } chosen)
        {
            // An interval of 0 would divide by zero at the first save, trials later; a negative one would act as its
            // absolute value (C#'s remainder ignores the divisor's sign), which no caller means. Refuse both up front.
            if (chosen < 1)
                throw new ArgumentOutOfRangeException(nameof(snapshotInterval), chosen, "snapshotInterval must be at least 1.");
            return chosen;
        }

        if (logStorage is IJournalSnapshotInterval preferred)
        {
            // Read once: a backend whose answer could change (a derived class computing it) cannot move the interval
            // between the saves of one storage.
            int interval = preferred.DefaultSnapshotInterval;
            if (interval < 1)
                throw new ArgumentException(
                    $"The backend's DefaultSnapshotInterval must be at least 1, but it is {interval}.", nameof(logStorage));
            return interval;
        }

        return OptunaSnapshotInterval;
    }

    /// <summary>
    ///     Restore the in-memory replay state from a snapshot produced by
    ///     <see cref="JournalStorageReplayResult.Serialize" />.
    ///     [Misaligned] Python snapshots are pickle bytes; OptunaSharp uses its own binary format (v2 since plan 10,
    ///     item 2.7; the older v1 JSON still restores) under its own key (the Redis backends'
    ///     <c>{prefix}:snapshot:optunasharp</c>) — snapshots are a same-language acceleration, never a cross-language
    ///     artifact (the file backend has no snapshot support on either side).
    /// </summary>
    /// <param name="snapshot">OptunaSharp snapshot bytes, v2 or v1; anything else is ignored with a warning.</param>
    public void RestoreReplayResult(byte[] snapshot)
    {
        var r = JournalStorageReplayResult.TryDeserialize(snapshot);
        if (r is null)
        {
            _logger.Warning("Failed to restore `JournalStorageReplayResult`.");
            return;
        }
        r.ResetWorkerState(_workerIdPrefix);
        _replayResult = r;
    }

    /// <summary>Adds the current worker identity and appends one operation to the journal.</summary>
    /// <param name="opCode">Stable on-disk operation code.</param>
    /// <param name="extraFields">Operation payload fields merged after the operation code and worker identifier.</param>
    /// <exception cref="System.Exception">The configured backend cannot append the record.</exception>
    private void WriteLog(JournalOperation opCode, OrderedDict<string, object?> extraFields)
    {
        string workerId = _replayResult.WorkerId;
        var log = new OrderedDict<string, object?>
        {
            ["op_code"] = (int)opCode,
            ["worker_id"] = workerId,
        };
        foreach (var (key, value) in extraFields)
            log[key] = value;
        _backend.AppendLogs(new[] { log });
    }

    /// <summary>Replays records after the current cursor; callers serialize access with the storage lock.</summary>
    /// <exception cref="System.Exception">Reading or applying a journal record fails.</exception>
    private void SyncWithBackend()
    {
        var logs = _backend.ReadLogs(_replayResult.LogNumberRead);
        _replayResult.ApplyLogs(logs);
    }

    /// <param name="directions">Objective directions in the same order as the objective values.</param>
    /// <param name="studyName">Name identifying the study in the selected storage.</param>
    /// <inheritdoc />
    public override int CreateNewStudy(IReadOnlyList<StudyDirection> directions, string? studyName = null)
    {
        studyName ??= DefaultStudyNamePrefix + Guid.NewGuid();

        lock (_threadLock)
        {
            WriteLog(JournalOperation.CreateStudy, new OrderedDict<string, object?>
            {
                ["study_name"] = studyName,
                ["directions"] = directions.Select(d => (object?)(int)d).ToList(),
            });
            SyncWithBackend();

            foreach (var frozenStudy in _replayResult.GetAllStudies())
            {
                if (frozenStudy.StudyName != studyName)
                    continue;

                _logger.Info($"A new study created in Journal with name: {studyName}");
                int studyId = frozenStudy.StudyId;

                // Dump snapshot here.
                if (_backend is IBaseJournalSnapshot snapshotBackend
                    && studyId != 0
                    && studyId % _snapshotInterval == 0)
                {
                    snapshotBackend.SaveSnapshot(_replayResult.Serialize());
                }

                return studyId;
            }
            throw new InvalidOperationException("Should not reach.");
        }
    }

    /// <param name="studyId">Storage identifier of the study; distinct from its name.</param>
    /// <inheritdoc />
    public override void DeleteStudy(int studyId)
    {
        lock (_threadLock)
        {
            WriteLog(JournalOperation.DeleteStudy, new OrderedDict<string, object?> { ["study_id"] = studyId });
            SyncWithBackend();
        }
    }

    /// <param name="studyId">Storage identifier of the study; distinct from its name.</param>
    /// <param name="key">Attribute name; a later write replaces its stored value.</param>
    /// <param name="value">Attribute value; persistent backends require JSON-serializable data.</param>
    /// <inheritdoc />
    public override void SetStudyUserAttr(int studyId, string key, object? value)
    {
        var log = new OrderedDict<string, object?>
        {
            ["study_id"] = studyId,
            ["user_attr"] = new OrderedDict<string, object?> { [key] = value },
        };
        lock (_threadLock)
        {
            WriteLog(JournalOperation.SetStudyUserAttr, log);
            SyncWithBackend();
        }
    }

    /// <param name="studyId">Storage identifier of the study; distinct from its name.</param>
    /// <param name="key">Attribute name; a later write replaces its stored value.</param>
    /// <param name="value">Attribute value; persistent backends require JSON-serializable data.</param>
    /// <inheritdoc />
    public override void SetStudySystemAttr(int studyId, string key, object? value)
    {
        var log = new OrderedDict<string, object?>
        {
            ["study_id"] = studyId,
            ["system_attr"] = new OrderedDict<string, object?> { [key] = value },
        };
        lock (_threadLock)
        {
            WriteLog(JournalOperation.SetStudySystemAttr, log);
            SyncWithBackend();
        }
    }

    /// <param name="studyName">Name identifying the study in the selected storage.</param>
    /// <inheritdoc />
    public override int GetStudyIdFromName(string studyName)
    {
        lock (_threadLock)
        {
            SyncWithBackend();
            foreach (var study in _replayResult.GetAllStudies())
            {
                if (study.StudyName == studyName)
                    return study.StudyId;
            }
            throw new KeyError(NotFoundMsg);
        }
    }

    /// <param name="studyId">Storage identifier of the study; distinct from its name.</param>
    /// <inheritdoc />
    public override string GetStudyNameFromId(int studyId)
    {
        lock (_threadLock)
        {
            SyncWithBackend();
            return _replayResult.GetStudy(studyId).StudyName;
        }
    }

    /// <param name="studyId">Storage identifier of the study; distinct from its name.</param>
    /// <inheritdoc />
    public override List<StudyDirection> GetStudyDirections(int studyId)
    {
        lock (_threadLock)
        {
            SyncWithBackend();
            return _replayResult.GetStudy(studyId).Directions;
        }
    }

    /// <param name="studyId">Storage identifier of the study; distinct from its name.</param>
    /// <inheritdoc />
    public override OrderedDict<string, object?> GetStudyUserAttrs(int studyId)
    {
        lock (_threadLock)
        {
            SyncWithBackend();
            return _replayResult.GetStudy(studyId).UserAttrs;
        }
    }

    /// <param name="studyId">Storage identifier of the study; distinct from its name.</param>
    /// <inheritdoc />
    public override OrderedDict<string, object?> GetStudySystemAttrs(int studyId)
    {
        lock (_threadLock)
        {
            SyncWithBackend();
            return _replayResult.GetStudy(studyId).SystemAttrs;
        }
    }

    /// <inheritdoc />
    public override List<FrozenStudy> GetAllStudies()
    {
        lock (_threadLock)
        {
            SyncWithBackend();
            return _replayResult.GetAllStudies().Select(s => s.DeepCopy()).ToList();
        }
    }

    // ------------------------------------------------------------------ trial manipulation

    /// <summary>
    ///     Python's <c>to_internal_repr</c> returns an int for categoricals (the choice index)
    ///     and a float otherwise; the journal preserves that distinction byte-for-byte.
    /// </summary>
    /// <param name="distribution">Distribution defining the parameter representation and allowed values.</param>
    /// <param name="externalValue">Parameter value in its external representation.</param>
    /// <returns>The internal numeric representation used by journal records.</returns>
    /// <exception cref="NumSharp.ValueError">The external value cannot be represented by the distribution.</exception>
    private static object InternalReprForJournal(BaseDistribution distribution, object? externalValue)
    {
        double internalRepr = distribution.ToInternalRepr(externalValue);
        // The (object) casts stop the conditional from unifying long back to double.
        return distribution is CategoricalDistribution ? (object)(long)internalRepr : internalRepr;
    }

    /// <param name="studyId">Storage identifier of the study; distinct from its name.</param>
    /// <inheritdoc />
    /// <param name="templateTrial">Optional record to copy; null creates a new running trial with empty observations.</param>
    public override int CreateNewTrial(int studyId, FrozenTrial? templateTrial = null)
    {
        var datetimeStart = DateTime.UtcNow;
        var log = new OrderedDict<string, object?>
        {
            ["study_id"] = studyId,
            ["datetime_start"] = JournalDatetime.ToIsoUtc(datetimeStart),
        };

        if (templateTrial is not null)
        {
            log["state"] = (int)templateTrial.State;
            if (templateTrial.Values is not null && templateTrial.Values.Count > 1)
            {
                log["value"] = null;
                log["values"] = templateTrial.Values.Cast<object?>().ToList();
            }
            else
            {
                log["value"] = templateTrial.Values is null ? null : templateTrial.Value;
                log["values"] = null;
            }
            if (templateTrial.DatetimeStart is not null)
                log["datetime_start"] = JournalDatetime.ToIsoUtc(templateTrial.DatetimeStart.Value.ToUniversalTime());
            else
                log["datetime_start"] = null;
            if (templateTrial.DatetimeComplete is not null)
                log["datetime_complete"] = JournalDatetime.ToIsoUtc(templateTrial.DatetimeComplete.Value.ToUniversalTime());

            var distributions = new OrderedDict<string, object?>();
            foreach (var (k, dist) in templateTrial.DistributionsDict)
                distributions[k] = Distribution.ToJson(dist);
            log["distributions"] = distributions;

            var @params = new OrderedDict<string, object?>();
            foreach (var (k, param) in templateTrial.ParamsDict)
                @params[k] = InternalReprForJournal(templateTrial.DistributionsDict[k], param);
            log["params"] = @params;

            log["user_attrs"] = templateTrial.UserAttrsDict;
            log["system_attrs"] = templateTrial.SystemAttrsDict;
            log["intermediate_values"] = templateTrial.IntermediateValues;
        }

        lock (_threadLock)
        {
            WriteLog(JournalOperation.CreateTrial, log);
            SyncWithBackend();
            int trialId = _replayResult.LastCreatedTrialIdByThisProcess;

            // Dump snapshot here.
            if (_backend is IBaseJournalSnapshot snapshotBackend
                && trialId != 0
                && trialId % _snapshotInterval == 0)
            {
                snapshotBackend.SaveSnapshot(_replayResult.Serialize());
            }
            return trialId;
        }
    }

    /// <param name="trialId">Storage identifier of the trial; distinct from its per-study trial number.</param>
    /// <param name="paramName">Parameter name used as the storage and distribution lookup key.</param>
    /// <param name="distribution">Distribution defining the parameter representation and allowed values.</param>
    /// <inheritdoc />
    /// <param name="paramValueInternal">Parameter value in the distribution's internal numeric representation.</param>
    public override void SetTrialParam(
        int trialId, string paramName, double paramValueInternal, BaseDistribution distribution)
    {
        var log = new OrderedDict<string, object?>
        {
            ["trial_id"] = trialId,
            ["param_name"] = paramName,
            // Python's CategoricalDistribution.to_internal_repr returns a raw int index (despite
            // the float annotation), so its json.dumps emits `0`, not `0.0` — byte-compat follows.
            // The (object) cast stops the conditional from unifying long back to double.
            ["param_value_internal"] = distribution is CategoricalDistribution
                ? (object)(long)paramValueInternal
                : paramValueInternal,
            ["distribution"] = Distribution.ToJson(distribution),
        };

        lock (_threadLock)
        {
            WriteLog(JournalOperation.SetTrialParam, log);
            SyncWithBackend();
        }
    }

    /// <param name="studyId">Storage identifier of the study; distinct from its name.</param>
    /// <param name="trialNumber">Zero-based trial number within the study, not a storage identifier.</param>
    /// <inheritdoc />
    public override int GetTrialIdFromStudyIdTrialNumber(int studyId, int trialNumber)
    {
        lock (_threadLock)
        {
            SyncWithBackend();
            var trialIds = _replayResult.GetStudyTrialIds(studyId);
            if (trialIds.Count <= trialNumber)
                throw new KeyError(
                    $"No trial with trial_number={trialNumber} exists in study with study_id={studyId}.");
            return trialIds[trialNumber];
        }
    }

    /// <param name="trialId">Storage identifier of the trial; distinct from its per-study trial number.</param>
    /// <param name="state">Trial state to record or inspect.</param>
    /// <param name="values">Objective values in study direction order, or null when no objective values are recorded.</param>
    /// <inheritdoc />
    public override bool SetTrialStateValues(int trialId, TrialState state, IReadOnlyList<double>? values = null)
    {
        var log = new OrderedDict<string, object?>
        {
            ["trial_id"] = trialId,
            ["state"] = (int)state,
            ["values"] = values?.Cast<object?>().ToList(),
        };

        if (state == TrialState.Running)
            log["datetime_start"] = JournalDatetime.ToIsoUtc(DateTime.UtcNow);
        else if (state.IsFinished())
            log["datetime_complete"] = JournalDatetime.ToIsoUtc(DateTime.UtcNow);

        lock (_threadLock)
        {
            if (state == TrialState.Running)
            {
                // NOTE(nabenabe): This sync is kept to avoid confusion (see the Python source for
                // the full rationale). This section is only triggered by enqueue_trial pops.
                SyncWithBackend();
                var existingTrial = _replayResult.TryGetTrial(trialId);
                System.Diagnostics.Debug.Assert(
                    existingTrial is not null,
                    "Please report your bug on GitHub if this line fails your script.");
                if (existingTrial!.State.IsFinished())
                    throw new UpdateFinishedTrialError(
                        string.Format(CultureInfo.InvariantCulture, UnupdatableMsgFormat, existingTrial.Number));
                if (existingTrial.State != TrialState.Waiting)
                {
                    // This line is equivalent to `existingTrial.State == TrialState.Running`.
                    return false;
                }
            }
            WriteLog(JournalOperation.SetTrialStateValues, log);
            SyncWithBackend();
            return state != TrialState.Running || trialId == _replayResult.OwnedTrialId;
        }
    }

    /// <param name="trialId">Storage identifier of the trial; distinct from its per-study trial number.</param>
    /// <param name="step">Reporting step whose intermediate objective observation is stored; written as a JSON integer, 64-bit like Python's.</param>
    /// <inheritdoc />
    /// <param name="intermediateValue">Intermediate objective observation to store at the reporting step.</param>
    public override void SetTrialIntermediateValue(int trialId, long step, double intermediateValue)
    {
        var log = new OrderedDict<string, object?>
        {
            ["trial_id"] = trialId,
            ["step"] = step,
            ["intermediate_value"] = intermediateValue,
        };

        lock (_threadLock)
        {
            WriteLog(JournalOperation.SetTrialIntermediateValue, log);
            SyncWithBackend();
        }
    }

    /// <param name="trialId">Storage identifier of the trial; distinct from its per-study trial number.</param>
    /// <param name="key">Attribute name; a later write replaces its stored value.</param>
    /// <param name="value">Attribute value; persistent backends require JSON-serializable data.</param>
    /// <inheritdoc />
    public override void SetTrialUserAttr(int trialId, string key, object? value)
    {
        var log = new OrderedDict<string, object?>
        {
            ["trial_id"] = trialId,
            ["user_attr"] = new OrderedDict<string, object?> { [key] = value },
        };

        lock (_threadLock)
        {
            WriteLog(JournalOperation.SetTrialUserAttr, log);
            SyncWithBackend();
        }
    }

    /// <param name="trialId">Storage identifier of the trial; distinct from its per-study trial number.</param>
    /// <param name="key">Attribute name; a later write replaces its stored value.</param>
    /// <param name="value">Attribute value; persistent backends require JSON-serializable data.</param>
    /// <inheritdoc />
    public override void SetTrialSystemAttr(int trialId, string key, object? value)
    {
        var log = new OrderedDict<string, object?>
        {
            ["trial_id"] = trialId,
            ["system_attr"] = new OrderedDict<string, object?> { [key] = value },
        };

        lock (_threadLock)
        {
            WriteLog(JournalOperation.SetTrialSystemAttr, log);
            SyncWithBackend();
        }
    }

    /// <param name="studyId">Storage identifier of the study; distinct from its name.</param>
    /// <param name="attrs">The attributes, written in enumeration order.</param>
    /// <inheritdoc />
    /// <remarks>See <see cref="WriteAttributeBatch" />: one checked, appended batch of ordinary records.</remarks>
    public override void SetStudyUserAttrs(int studyId, IReadOnlyDictionary<string, object?> attrs)
        => WriteAttributeBatch(JournalOperation.SetStudyUserAttr, "study_id", studyId, "user_attr", attrs, isTrial: false);

    /// <param name="studyId">Storage identifier of the study; distinct from its name.</param>
    /// <param name="attrs">The attributes, written in enumeration order.</param>
    /// <inheritdoc />
    /// <remarks>See <see cref="WriteAttributeBatch" />: one checked, appended batch of ordinary records.</remarks>
    public override void SetStudySystemAttrs(int studyId, IReadOnlyDictionary<string, object?> attrs)
        => WriteAttributeBatch(JournalOperation.SetStudySystemAttr, "study_id", studyId, "system_attr", attrs, isTrial: false);

    /// <param name="trialId">Storage identifier of the trial; distinct from its per-study trial number.</param>
    /// <param name="attrs">The attributes, written in enumeration order.</param>
    /// <inheritdoc />
    /// <remarks>See <see cref="WriteAttributeBatch" />: one checked, appended batch of ordinary records.</remarks>
    public override void SetTrialUserAttrs(int trialId, IReadOnlyDictionary<string, object?> attrs)
        => WriteAttributeBatch(JournalOperation.SetTrialUserAttr, "trial_id", trialId, "user_attr", attrs, isTrial: true);

    /// <param name="trialId">Storage identifier of the trial; distinct from its per-study trial number.</param>
    /// <param name="attrs">The attributes, written in enumeration order.</param>
    /// <inheritdoc />
    /// <remarks>See <see cref="WriteAttributeBatch" />: one checked, appended batch of ordinary records.</remarks>
    public override void SetTrialSystemAttrs(int trialId, IReadOnlyDictionary<string, object?> attrs)
        => WriteAttributeBatch(JournalOperation.SetTrialSystemAttr, "trial_id", trialId, "system_attr", attrs, isTrial: true);

    /// <summary>
    ///     The bulk attribute writes (plan 13, item S8): one record per entry, exactly the record the single-attribute
    ///     write appends, all appended in one backend call.
    /// </summary>
    /// <param name="opCode">The single-attribute operation each record carries.</param>
    /// <param name="idField">The record's id field: <c>study_id</c> or <c>trial_id</c>.</param>
    /// <param name="id">The study or trial id.</param>
    /// <param name="attrField">The record's attribute field: <c>user_attr</c> or <c>system_attr</c>.</param>
    /// <param name="attrs">The attributes, one record each, in enumeration order.</param>
    /// <param name="isTrial">Whether <paramref name="id" /> is a trial (checked as updatable) or a study (checked as existing).</param>
    /// <exception cref="ArgumentNullException"><paramref name="attrs" /> is null.</exception>
    /// <exception cref="NumSharp.KeyError">The study or trial does not exist.</exception>
    /// <exception cref="OptunaSharp.UpdateFinishedTrialError">The trial has finished.</exception>
    /// <exception cref="Exception">A value cannot be serialized, or the backend cannot append; nothing is appended then.</exception>
    /// <remarks>
    ///     <para>
    ///         <b>One record per entry, on purpose.</b> Python's journal replay asserts that an attribute record carries
    ///         exactly one key (<c>assert len(log["user_attr"]) == 1</c>), so a journal Python workers share can only hold
    ///         single-key records. The batch is those records, appended together: the file backend writes them under one
    ///         lock with one flush, where single writes pay one each.
    ///     </para>
    ///     <para>
    ///         <b>Checked first.</b> The replay raises for this worker's own record on an unknown or finished trial, one
    ///         record per synchronization. Appending a whole batch the replay then rejects would leave records that make
    ///         later, unrelated calls of this storage raise, so the batch is checked against the replayed state first and
    ///         nothing is written when the check fails. If another process finishes the trial between the check and the
    ///         append, <see cref="SyncAfterBatch" /> consumes the batch's rejected records before the error surfaces.
    ///     </para>
    /// </remarks>
    private void WriteAttributeBatch(
        JournalOperation opCode, string idField, int id, string attrField,
        IReadOnlyDictionary<string, object?> attrs, bool isTrial)
    {
        ArgumentNullException.ThrowIfNull(attrs);
        if (attrs.Count == 0)
            return;

        var logs = new List<OrderedDict<string, object?>>(attrs.Count);
        foreach (var (key, value) in attrs)
        {
            logs.Add(new OrderedDict<string, object?>
            {
                ["op_code"] = (int)opCode,
                ["worker_id"] = null,  // filled under the lock: the worker id belongs to the replay state
                [idField] = id,
                [attrField] = new OrderedDict<string, object?> { [key] = value },
            });
        }

        lock (_threadLock)
        {
            SyncWithBackend();
            if (isTrial)
                _replayResult.ThrowIfTrialNotUpdatable(id);
            else
                _replayResult.ThrowIfStudyMissing(id);

            string workerId = _replayResult.WorkerId;
            foreach (var log in logs)
                log["worker_id"] = workerId;
            _backend.AppendLogs(logs);
            SyncAfterBatch(logs.Count);
        }
    }

    /// <summary>
    ///     Synchronizes after appending a batch of this worker's records, consuming every record of the batch the replay
    ///     rejects before raising the first rejection.
    /// </summary>
    /// <param name="batchSize">How many records the batch appended; each rejection consumes at least one of them.</param>
    /// <exception cref="Exception">The first exception the synchronization raised, rethrown with its original stack.</exception>
    /// <remarks>
    ///     A rejection here means another process finished or deleted the target between the check and the append. The
    ///     replay raises once per rejected record, and only for the worker that wrote it, so without draining, each of
    ///     the batch's remaining records would raise in a later call of this storage that has nothing to do with it.
    ///     Each failed synchronization advances past at least the record that failed, so at most one attempt per record
    ///     is needed.
    /// </remarks>
    private void SyncAfterBatch(int batchSize)
    {
        System.Runtime.ExceptionServices.ExceptionDispatchInfo? first = null;
        for (int attempt = 0; attempt < batchSize; attempt++)
        {
            try
            {
                SyncWithBackend();
                break;
            }
            catch (Exception exception)
            {
                first ??= System.Runtime.ExceptionServices.ExceptionDispatchInfo.Capture(exception);
            }
        }
        first?.Throw();
    }

    /// <summary>Test hook for <see cref="SyncAfterBatch" />: appends records as one batch, then drains as it does.</summary>
    /// <param name="logs">Complete records, worker id included.</param>
    /// <exception cref="Exception">The first rejection of the batch, after the batch is consumed.</exception>
    internal void AppendBatchAndSyncForTests(IReadOnlyList<OrderedDict<string, object?>> logs)
    {
        lock (_threadLock)
        {
            _backend.AppendLogs(logs);
            SyncAfterBatch(logs.Count);
        }
    }

    /// <summary>The worker id this storage writes into records, for tests that build records of their own.</summary>
    internal string WorkerIdForTests
    {
        get
        {
            lock (_threadLock)
                return _replayResult.WorkerId;
        }
    }

    /// <param name="trialId">Storage identifier of the trial; distinct from its per-study trial number.</param>
    /// <inheritdoc />
    public override FrozenTrial GetTrial(int trialId)
    {
        lock (_threadLock)
        {
            SyncWithBackend();
            return _replayResult.GetTrial(trialId);
        }
    }

    /// <param name="studyId">Storage identifier of the study; distinct from its name.</param>
    /// <param name="deepcopy">Whether to copy trial records; treat records returned with false as read-only.</param>
    /// <param name="states">States to include; null includes every state.</param>
    /// <inheritdoc />
    public override List<FrozenTrial> GetAllTrials(
        int studyId, bool deepcopy = true, IReadOnlyCollection<TrialState>? states = null)
    {
        lock (_threadLock)
        {
            SyncWithBackend();
            var frozenTrials = _replayResult.GetAllTrials(studyId, states);
            if (deepcopy)
                return frozenTrials.Select(t => t.DeepCopy()).ToList();
            return frozenTrials;
        }
    }
}

/// <summary>ISO-8601 datetime helpers matching Python's journal serialization exactly.</summary>
internal static class JournalDatetime
{
    /// <summary>Python <c>datetime.now(tz=timezone.utc).isoformat(timespec="microseconds")</c>.</summary>
    /// <param name="utc">UTC timestamp; the caller must convert to UTC before calling.</param>
    /// <returns>ISO-8601 timestamp with six fractional digits and an explicit +00:00 suffix.</returns>
    internal static string ToIsoUtc(DateTime utc)
        => utc.ToString("yyyy-MM-dd'T'HH:mm:ss.ffffff", CultureInfo.InvariantCulture) + "+00:00";

    /// <summary>Python <c>datetime.fromisoformat(s).astimezone().replace(tzinfo=None)</c>: to naive local time.</summary>
    /// <param name="iso">ISO-8601 timestamp, normally including an offset.</param>
    /// <returns>Local wall-clock timestamp with DateTimeKind.Unspecified.</returns>
    /// <exception cref="System.FormatException">The text is not a supported timestamp.</exception>
    internal static DateTime FromIsoToLocalNaive(string iso)
    {
        var dto = DateTimeOffset.Parse(iso, CultureInfo.InvariantCulture, System.Globalization.DateTimeStyles.AssumeLocal);
        return dto.ToLocalTime().DateTime;
    }
}

/// <summary>Port of <c>JournalStorageReplayResult</c>.</summary>
/// <remarks>
///     Partial: its snapshot codecs live in <c>JournalStorageReplayResult.Snapshot.cs</c> (the binary v2 format of plan
///     10, item 2.7, which it writes) and at the end of this file (the v1 JSON format, which it still reads).
/// </remarks>
internal sealed partial class JournalStorageReplayResult
{
    /// <summary>Number of log records already consumed by replay.</summary>
    internal int LogNumberRead;
    private string _workerIdPrefix;
    private readonly Dictionary<int, FrozenStudy> _studies = new();
    private readonly Dictionary<int, FrozenTrial> _trials = new();

    private readonly Dictionary<int, List<int>> _studyIdToTrialIds = new();
    private readonly Dictionary<int, int> _trialIdToStudyId = new();
    private int _nextStudyId;
    private Dictionary<string, int> _workerIdToOwnedTrialId = new();

    // One distribution object per distinct JSON text across records and restores (plan 10, item 2.7; the owner's
    // "Share one object"): parsing every record's text again was most of the replay's distribution cost.
    private readonly JournalDistributionCache _distributionCache = new();

    /// <summary>Last trial created by this process family, or -1 before a local creation.</summary>
    internal int LastCreatedTrialIdByThisProcess = -1;

    /// <summary>Initializes empty replay state for a process identity.</summary>
    /// <param name="workerIdPrefix">Unique process prefix used to distinguish local and remote journal operations.</param>
    internal JournalStorageReplayResult(string workerIdPrefix)
    {
        _workerIdPrefix = workerIdPrefix;
    }

    /// <summary>Clears process-local trial ownership after loading a reusable snapshot.</summary>
    /// <param name="workerIdPrefix">New process identity prefix; replayed study/trial records remain available.</param>
    internal void ResetWorkerState(string workerIdPrefix)
    {
        _workerIdPrefix = workerIdPrefix;
        _workerIdToOwnedTrialId = new Dictionary<string, int>();
        LastCreatedTrialIdByThisProcess = -1;
    }

    /// <summary>Applies ordered journal operations and advances the consumed-record cursor.</summary>
    /// <param name="logs">Decoded records beginning at the current replay cursor.</param>
    /// <exception cref="System.InvalidOperationException">A record uses an unknown operation code.</exception>
    /// <exception cref="System.Exception">A record is malformed or a local operation violates study/trial state constraints.</exception>
    internal void ApplyLogs(IEnumerable<OrderedDict<string, object?>> logs)
    {
        foreach (var log in logs)
        {
            // Consume rejected operations too: a local validation failure must not be rethrown
            // forever on every subsequent synchronization of this append-only journal.
            LogNumberRead += 1;
            var op = (JournalOperation)Convert.ToInt32(log["op_code"], CultureInfo.InvariantCulture);
            switch (op)
            {
                case JournalOperation.CreateStudy:
                    ApplyCreateStudy(log);
                    break;
                case JournalOperation.DeleteStudy:
                    ApplyDeleteStudy(log);
                    break;
                case JournalOperation.SetStudyUserAttr:
                    ApplySetStudyUserAttr(log);
                    break;
                case JournalOperation.SetStudySystemAttr:
                    ApplySetStudySystemAttr(log);
                    break;
                case JournalOperation.CreateTrial:
                    ApplyCreateTrial(log);
                    break;
                case JournalOperation.SetTrialParam:
                    ApplySetTrialParam(log);
                    break;
                case JournalOperation.SetTrialStateValues:
                    ApplySetTrialStateValues(log);
                    break;
                case JournalOperation.SetTrialIntermediateValue:
                    ApplySetTrialIntermediateValue(log);
                    break;
                case JournalOperation.SetTrialUserAttr:
                    ApplySetTrialUserAttr(log);
                    break;
                case JournalOperation.SetTrialSystemAttr:
                    ApplySetTrialSystemAttr(log);
                    break;
                case JournalOperation.DiscardTrials:
                    // This operation is only supported by Rustuna.
                    continue;
                default:
                    throw new InvalidOperationException("Should not reach.");
            }
        }
    }

    /// <param name="studyId">Storage identifier of the study; distinct from its name.</param>
    /// <summary>Reads records from the current replay state.</summary>
    /// <returns>The stored study record.</returns>
    /// <exception cref="NumSharp.KeyError">The requested study or trial does not exist.</exception>
    internal FrozenStudy GetStudy(int studyId)
    {
        if (!_studies.TryGetValue(studyId, out var study))
            throw new KeyError(JournalStorage.NotFoundMsg);
        return study;
    }

    /// <summary>Reads records from the current replay state.</summary>
    /// <returns>A new list containing the current shared study records.</returns>
    internal List<FrozenStudy> GetAllStudies() => _studies.Values.ToList();

    /// <param name="trialId">Storage identifier of the trial; distinct from its per-study trial number.</param>
    /// <summary>Reads records from the current replay state.</summary>
    /// <returns>The shared trial record.</returns>
    /// <exception cref="NumSharp.KeyError">The requested study or trial does not exist.</exception>
    internal FrozenTrial GetTrial(int trialId)
    {
        if (!_trials.TryGetValue(trialId, out var trial))
            throw new KeyError(JournalStorage.NotFoundMsg);
        return trial;
    }

    /// <param name="trialId">Storage identifier of the trial; distinct from its per-study trial number.</param>
    /// <summary>Reads records from the current replay state.</summary>
    /// <returns>The trial record, or null when absent.</returns>
    internal FrozenTrial? TryGetTrial(int trialId) => _trials.GetValueOrDefault(trialId);

    /// <param name="studyId">Storage identifier of the study; distinct from its name.</param>
    /// <summary>Reads records from the current replay state.</summary>
    /// <returns>Shared list of trial identifiers belonging to the study.</returns>
    /// <exception cref="NumSharp.KeyError">The requested study or trial does not exist.</exception>
    internal List<int> GetStudyTrialIds(int studyId)
    {
        if (!_studyIdToTrialIds.TryGetValue(studyId, out var trialIds))
            throw new KeyError(studyId.ToString(CultureInfo.InvariantCulture));
        return trialIds;
    }

    /// <param name="studyId">Storage identifier of the study; distinct from its name.</param>
    /// <param name="states">States to include; null includes every state.</param>
    /// <summary>Reads records from the current replay state.</summary>
    /// <returns>Shared trial records matching the state filter, in journal creation order.</returns>
    /// <exception cref="NumSharp.KeyError">The requested study or trial does not exist.</exception>
    internal List<FrozenTrial> GetAllTrials(int studyId, IReadOnlyCollection<TrialState>? states)
    {
        if (!_studies.ContainsKey(studyId))
            throw new KeyError(JournalStorage.NotFoundMsg);

        var frozenTrials = new List<FrozenTrial>();
        foreach (int trialId in _studyIdToTrialIds[studyId])
        {
            var trial = _trials[trialId];
            if (states is null || states.Contains(trial.State))
                frozenTrials.Add(trial);
        }
        return frozenTrials;
    }

    /// <summary>Current process prefix combined with the managed thread identifier.</summary>
    internal string WorkerId => _workerIdPrefix + Environment.CurrentManagedThreadId.ToString(CultureInfo.InvariantCulture);

    /// <summary>Trial claimed by this worker, or null before a successful claim.</summary>
    internal int? OwnedTrialId
        => _workerIdToOwnedTrialId.TryGetValue(WorkerId, out int trialId) ? trialId : null;

    /// <param name="log">Decoded journal operation, including its payload and originating worker identifier.</param>
    /// <summary>Checks whether this thread originated a journal record.</summary>
    /// <returns>True when the record worker identifier equals this process/thread identity.</returns>
    private bool IsIssuedByThisWorker(OrderedDict<string, object?> log)
        => (string?)log["worker_id"] == WorkerId;

    /// <param name="studyId">Storage identifier of the study; distinct from its name.</param>
    /// <param name="log">Decoded journal operation, including its payload and originating worker identifier.</param>
    /// <summary>Checks study existence while suppressing rejected operations from other workers.</summary>
    /// <returns>True when the study exists; false for a missing study referenced by another worker.</returns>
    /// <exception cref="NumSharp.KeyError">This worker issued an operation for a missing study.</exception>
    private bool StudyExists(int studyId, OrderedDict<string, object?> log)
    {
        if (_studies.ContainsKey(studyId))
            return true;
        // Only the originating worker receives its operation failure. Other workers must keep
        // replaying past rejected entries to converge on the same accepted storage state.
        if (IsIssuedByThisWorker(log))
            throw new KeyError(JournalStorage.NotFoundMsg);
        return false;
    }

    /// <param name="log">Decoded journal operation, including its payload and originating worker identifier.</param>
    /// <summary>Creates study metadata and assigns its replay identifier. Applies the entry in journal order.</summary>
    /// <exception cref="NumSharp.KeyError">A referenced record is absent and this worker issued the operation.</exception>
    /// <exception cref="System.InvalidCastException">A journal field has an incompatible decoded type.</exception>
    /// <exception cref="System.Collections.Generic.KeyNotFoundException">A required journal field is absent.</exception>
    /// <exception cref="OptunaSharp.DuplicatedStudyError">This worker attempts to create a study name that already exists.</exception>
    private void ApplyCreateStudy(OrderedDict<string, object?> log)
    {
        string studyName = (string)log["study_name"]!;
        var directions = ((List<object?>)log["directions"]!)
            .Select(d => (StudyDirection)Convert.ToInt32(d, CultureInfo.InvariantCulture))
            .ToList();

        if (_studies.Values.Any(s => s.StudyName == studyName))
        {
            if (IsIssuedByThisWorker(log))
                throw new DuplicatedStudyError();
            return;
        }

        int studyId = _nextStudyId;
        _nextStudyId += 1;

        _studies[studyId] = new FrozenStudy(
            studyName: studyName,
            direction: null,
            userAttrs: new OrderedDict<string, object?>(),
            systemAttrs: new OrderedDict<string, object?>(),
            studyId: studyId,
            directions: directions);
        _studyIdToTrialIds[studyId] = new List<int>();
    }

    /// <param name="log">Decoded journal operation, including its payload and originating worker identifier.</param>
    /// <summary>Removes a study from the replayed study registry. Applies the entry in journal order.</summary>
    /// <exception cref="NumSharp.KeyError">A referenced record is absent and this worker issued the operation.</exception>
    /// <exception cref="System.InvalidCastException">A journal field has an incompatible decoded type.</exception>
    /// <exception cref="System.Collections.Generic.KeyNotFoundException">A required journal field is absent.</exception>
    private void ApplyDeleteStudy(OrderedDict<string, object?> log)
    {
        int studyId = ToInt(log["study_id"]);

        if (StudyExists(studyId, log))
        {
            var fs = _studies[studyId];
            _studies.Remove(studyId);
            System.Diagnostics.Debug.Assert(fs.StudyId == studyId);
        }
    }

    /// <param name="log">Decoded journal operation, including its payload and originating worker identifier.</param>
    /// <summary>Updates a study user attribute. Applies the entry in journal order.</summary>
    /// <exception cref="NumSharp.KeyError">A referenced record is absent and this worker issued the operation.</exception>
    /// <exception cref="System.InvalidCastException">A journal field has an incompatible decoded type.</exception>
    /// <exception cref="System.Collections.Generic.KeyNotFoundException">A required journal field is absent.</exception>
    private void ApplySetStudyUserAttr(OrderedDict<string, object?> log)
    {
        int studyId = ToInt(log["study_id"]);

        if (StudyExists(studyId, log))
        {
            var attr = (OrderedDict<string, object?>)log["user_attr"]!;
            System.Diagnostics.Debug.Assert(attr.Count == 1);
            var userAttrs = _studies[studyId].UserAttrs.ShallowClone();
            foreach (var (key, value) in attr)
                userAttrs[key] = value;
            _studies[studyId].UserAttrs = userAttrs;
        }
    }

    /// <param name="log">Decoded journal operation, including its payload and originating worker identifier.</param>
    /// <summary>Updates an internal study attribute. Applies the entry in journal order.</summary>
    /// <exception cref="NumSharp.KeyError">A referenced record is absent and this worker issued the operation.</exception>
    /// <exception cref="System.InvalidCastException">A journal field has an incompatible decoded type.</exception>
    /// <exception cref="System.Collections.Generic.KeyNotFoundException">A required journal field is absent.</exception>
    private void ApplySetStudySystemAttr(OrderedDict<string, object?> log)
    {
        int studyId = ToInt(log["study_id"]);

        if (StudyExists(studyId, log))
        {
            var attr = (OrderedDict<string, object?>)log["system_attr"]!;
            System.Diagnostics.Debug.Assert(attr.Count == 1);
            var systemAttrs = _studies[studyId].SystemAttrs.ShallowClone();
            foreach (var (key, value) in attr)
                systemAttrs[key] = value;
            _studies[studyId].SystemAttrs = systemAttrs;
        }
    }

    /// <param name="log">Decoded journal operation, including its payload and originating worker identifier.</param>
    /// <summary>Creates a trial and assigns its replay identifier. Applies the entry in journal order.</summary>
    /// <exception cref="NumSharp.KeyError">A referenced record is absent and this worker issued the operation.</exception>
    /// <exception cref="System.InvalidCastException">A journal field has an incompatible decoded type.</exception>
    /// <exception cref="System.Collections.Generic.KeyNotFoundException">A required journal field is absent.</exception>
    private void ApplyCreateTrial(OrderedDict<string, object?> log)
    {
        int studyId = ToInt(log["study_id"]);

        if (!StudyExists(studyId, log))
            return;

        int trialId = _trials.Count;
        var distributions = new OrderedDict<string, BaseDistribution>();
        if (log.ContainsKey("distributions"))
        {
            foreach (var (k, v) in (OrderedDict<string, object?>)log["distributions"]!)
                distributions[k] = _distributionCache.FromJson((string)v!);
        }
        var @params = new OrderedDict<string, object?>();
        if (log.ContainsKey("params"))
        {
            foreach (var (k, p) in (OrderedDict<string, object?>)log["params"]!)
                @params[k] = distributions[k].ToExternalRepr(ToDouble(p));
        }
        DateTime? datetimeStart = log["datetime_start"] is string startIso
            ? JournalDatetime.FromIsoToLocalNaive(startIso)
            : null;
        DateTime? datetimeComplete = log.ContainsKey("datetime_complete") && log["datetime_complete"] is string completeIso
            ? JournalDatetime.FromIsoToLocalNaive(completeIso)
            : null;

        // JSON object keys are strings; Python writes the steps with str(int), which can exceed Int32.
        var intermediateValues = new OrderedDict<long, double>();
        if (log.GetValueOrDefault("intermediate_values") is OrderedDict<string, object?> intermediates)
        {
            foreach (var (k, v) in intermediates)
                intermediateValues[long.Parse(k, CultureInfo.InvariantCulture)] = ToDouble(v);
        }

        var userAttrs = log.GetValueOrDefault("user_attrs") as OrderedDict<string, object?>
            ?? new OrderedDict<string, object?>();
        var systemAttrs = log.GetValueOrDefault("system_attrs") as OrderedDict<string, object?>
            ?? new OrderedDict<string, object?>();

        double? value = log.GetValueOrDefault("value") is { } rawValue ? ToDouble(rawValue) : null;
        List<double>? values = log.GetValueOrDefault("values") is List<object?> rawValues
            ? rawValues.Select(v => ToDouble(v)).ToList()
            : null;

        _trials[trialId] = new FrozenTrial(
            trialId: trialId,
            number: _studyIdToTrialIds[studyId].Count,
            state: (TrialState)(log.GetValueOrDefault("state") is { } rawState
                ? Convert.ToInt32(rawState, CultureInfo.InvariantCulture)
                : (int)TrialState.Running),
            @params: @params,
            distributions: distributions,
            userAttrs: userAttrs,
            systemAttrs: systemAttrs,
            value: value,
            intermediateValues: intermediateValues,
            datetimeStart: datetimeStart,
            datetimeComplete: datetimeComplete,
            values: values);

        _studyIdToTrialIds[studyId].Add(trialId);
        _trialIdToStudyId[trialId] = studyId;

        if (IsIssuedByThisWorker(log))
        {
            LastCreatedTrialIdByThisProcess = trialId;
            if (_trials[trialId].State == TrialState.Running)
                _workerIdToOwnedTrialId[WorkerId] = trialId;
        }
    }

    /// <param name="log">Decoded journal operation, including its payload and originating worker identifier.</param>
    /// <summary>Records a parameter and its distribution. Applies the entry in journal order.</summary>
    /// <exception cref="NumSharp.KeyError">A referenced record is absent and this worker issued the operation.</exception>
    /// <exception cref="System.InvalidCastException">A journal field has an incompatible decoded type.</exception>
    /// <exception cref="System.Collections.Generic.KeyNotFoundException">A required journal field is absent.</exception>
    /// <exception cref="NumSharp.ValueError">The recorded parameter distribution conflicts with its previous distribution.</exception>
    private void ApplySetTrialParam(OrderedDict<string, object?> log)
    {
        int trialId = ToInt(log["trial_id"]);

        if (!TrialExistsAndUpdatable(trialId, log))
            return;

        string paramName = (string)log["param_name"]!;
        double paramValueInternal = ToDouble(log["param_value_internal"]);
        var distribution = _distributionCache.FromJson((string)log["distribution"]!);

        int studyId = _trialIdToStudyId[trialId];

        foreach (int prevTrialId in _studyIdToTrialIds[studyId])
        {
            var prevTrial = _trials[prevTrialId];
            if (prevTrial.ParamsDict.ContainsKey(paramName))
            {
                try
                {
                    Distribution.CheckDistributionCompatibility(
                        prevTrial.DistributionsDict[paramName], distribution);
                }
                catch (Exception)
                {
                    if (IsIssuedByThisWorker(log))
                        throw;
                    return;
                }
                break;
            }
        }

        var trial = _trials[trialId].ShallowCopy();
        var @params = trial.ParamsDict.ShallowClone();
        @params[paramName] = distribution.ToExternalRepr(paramValueInternal);
        trial.ParamsDict = @params;
        var distributions = trial.DistributionsDict.ShallowClone();
        distributions[paramName] = distribution;
        trial.DistributionsDict = distributions;
        _trials[trialId] = trial;
    }

    /// <param name="log">Decoded journal operation, including its payload and originating worker identifier.</param>
    /// <summary>Changes trial state and objective values. Applies the entry in journal order.</summary>
    /// <exception cref="NumSharp.KeyError">A referenced record is absent and this worker issued the operation.</exception>
    /// <exception cref="System.InvalidCastException">A journal field has an incompatible decoded type.</exception>
    /// <exception cref="System.Collections.Generic.KeyNotFoundException">A required journal field is absent.</exception>
    private void ApplySetTrialStateValues(OrderedDict<string, object?> log)
    {
        int trialId = ToInt(log["trial_id"]);

        if (!TrialExistsAndUpdatable(trialId, log))
            return;

        var state = (TrialState)Convert.ToInt32(log["state"], CultureInfo.InvariantCulture);
        if (state == _trials[trialId].State && state == TrialState.Running)
        {
            // Reject the operation as the popped trial is already run by another process.
            return;
        }

        var trial = _trials[trialId].ShallowCopy();
        if (state == TrialState.Running)
        {
            trial.SetDatetimeStart(JournalDatetime.FromIsoToLocalNaive((string)log["datetime_start"]!));
            if (IsIssuedByThisWorker(log))
                _workerIdToOwnedTrialId[WorkerId] = trialId;
        }
        if (state.IsFinished())
            trial.DatetimeComplete = JournalDatetime.FromIsoToLocalNaive((string)log["datetime_complete"]!);
        trial.State = state;
        if (log["values"] is List<object?> values)
            trial.SetValues(values.Select(v => ToDouble(v)));

        _trials[trialId] = trial;
    }

    /// <param name="log">Decoded journal operation, including its payload and originating worker identifier.</param>
    /// <summary>Records an objective observation at a step. Applies the entry in journal order.</summary>
    /// <exception cref="NumSharp.KeyError">A referenced record is absent and this worker issued the operation.</exception>
    /// <exception cref="System.InvalidCastException">A journal field has an incompatible decoded type.</exception>
    /// <exception cref="System.Collections.Generic.KeyNotFoundException">A required journal field is absent.</exception>
    /// <exception cref="System.OverflowException">The trial id exceeds Int32 or the step exceeds Int64.</exception>
    /// <exception cref="System.FormatException">A numeric field holds non-numeric text.</exception>
    private void ApplySetTrialIntermediateValue(OrderedDict<string, object?> log)
    {
        int trialId = ToInt(log["trial_id"]);

        if (TrialExistsAndUpdatable(trialId, log))
        {
            var trial = _trials[trialId].ShallowCopy();
            trial.IntermediateValues = trial.IntermediateValues.ShallowClone();
            // 64-bit: Python accepts any int step, and one past Int32 used to make every C# load of the study throw.
            trial.IntermediateValues[ToLong(log["step"])] = ToDouble(log["intermediate_value"]);
            _trials[trialId] = trial;
        }
    }

    /// <param name="log">Decoded journal operation, including its payload and originating worker identifier.</param>
    /// <summary>Updates a trial user attribute. Applies the entry in journal order.</summary>
    /// <exception cref="NumSharp.KeyError">A referenced record is absent and this worker issued the operation.</exception>
    /// <exception cref="System.InvalidCastException">A journal field has an incompatible decoded type.</exception>
    /// <exception cref="System.Collections.Generic.KeyNotFoundException">A required journal field is absent.</exception>
    private void ApplySetTrialUserAttr(OrderedDict<string, object?> log)
    {
        int trialId = ToInt(log["trial_id"]);

        if (TrialExistsAndUpdatable(trialId, log))
        {
            var attr = (OrderedDict<string, object?>)log["user_attr"]!;
            System.Diagnostics.Debug.Assert(attr.Count == 1);
            var trial = _trials[trialId].ShallowCopy();
            trial.UserAttrsDict = trial.UserAttrsDict.ShallowClone();
            foreach (var (key, value) in attr)
                trial.UserAttrsDict[key] = value;
            _trials[trialId] = trial;
        }
    }

    /// <param name="log">Decoded journal operation, including its payload and originating worker identifier.</param>
    /// <summary>Updates an internal trial attribute. Applies the entry in journal order.</summary>
    /// <exception cref="NumSharp.KeyError">A referenced record is absent and this worker issued the operation.</exception>
    /// <exception cref="System.InvalidCastException">A journal field has an incompatible decoded type.</exception>
    /// <exception cref="System.Collections.Generic.KeyNotFoundException">A required journal field is absent.</exception>
    private void ApplySetTrialSystemAttr(OrderedDict<string, object?> log)
    {
        int trialId = ToInt(log["trial_id"]);

        if (TrialExistsAndUpdatable(trialId, log))
        {
            var attr = (OrderedDict<string, object?>)log["system_attr"]!;
            System.Diagnostics.Debug.Assert(attr.Count == 1);
            var trial = _trials[trialId].ShallowCopy();
            trial.SystemAttrsDict = trial.SystemAttrsDict.ShallowClone();
            foreach (var (key, value) in attr)
                trial.SystemAttrsDict[key] = value;
            _trials[trialId] = trial;
        }
    }

    /// <summary>
    ///     Raises what replaying this worker's own attribute record for <paramref name="trialId" /> would raise, before
    ///     any record is written: the bulk attribute writes (plan 13, item S8) check once, then append their batch.
    /// </summary>
    /// <param name="trialId">Storage identifier of the trial about to be written.</param>
    /// <exception cref="NumSharp.KeyError">The trial does not exist (<see cref="JournalStorage.NotFoundMsg" />).</exception>
    /// <exception cref="OptunaSharp.UpdateFinishedTrialError">The trial has finished; the message is the replay's.</exception>
    internal void ThrowIfTrialNotUpdatable(int trialId)
    {
        if (!_trials.TryGetValue(trialId, out FrozenTrial? trial))
            throw new KeyError(JournalStorage.NotFoundMsg);
        if (trial.State.IsFinished())
            throw new UpdateFinishedTrialError(
                string.Format(CultureInfo.InvariantCulture, JournalStorage.UnupdatableMsgFormat, trial.Number));
    }

    /// <summary>
    ///     Raises what replaying this worker's own study-attribute record would raise for a missing study, before any
    ///     record is written (plan 13, item S8).
    /// </summary>
    /// <param name="studyId">Storage identifier of the study about to be written.</param>
    /// <exception cref="NumSharp.KeyError">The study does not exist (<see cref="JournalStorage.NotFoundMsg" />).</exception>
    internal void ThrowIfStudyMissing(int studyId)
    {
        if (!_studies.ContainsKey(studyId))
            throw new KeyError(JournalStorage.NotFoundMsg);
    }

    /// <param name="trialId">Storage identifier of the trial; distinct from its per-study trial number.</param>
    /// <param name="log">Decoded journal operation, including its payload and originating worker identifier.</param>
    /// <summary>Checks trial existence and completion state before applying a replay mutation.</summary>
    /// <returns>True when the trial exists and is unfinished; false for rejected remote operations.</returns>
    /// <exception cref="NumSharp.KeyError">This worker referenced a missing trial.</exception>
    /// <exception cref="OptunaSharp.UpdateFinishedTrialError">This worker attempted to change a finished trial.</exception>
    private bool TrialExistsAndUpdatable(int trialId, OrderedDict<string, object?> log)
    {
        if (!_trials.ContainsKey(trialId))
        {
            if (IsIssuedByThisWorker(log))
                throw new KeyError(JournalStorage.NotFoundMsg);
            return false;
        }
        if (_trials[trialId].State.IsFinished())
        {
            if (IsIssuedByThisWorker(log))
                throw new UpdateFinishedTrialError(
                    string.Format(CultureInfo.InvariantCulture, JournalStorage.UnupdatableMsgFormat, _trials[trialId].Number));
            return false;
        }
        return true;
    }

    /// <param name="value">Decoded numeric field from a journal record.</param>
    /// <summary>Converts a decoded journal scalar using invariant numeric conventions.</summary>
    /// <returns>The converted scalar.</returns>
    /// <exception cref="System.FormatException">The scalar text is not numeric.</exception>
    /// <exception cref="System.OverflowException">The scalar exceeds the conversion range.</exception>
    /// <exception cref="System.InvalidCastException">The scalar does not support numeric conversion.</exception>
    private static int ToInt(object? value) => Convert.ToInt32(value, CultureInfo.InvariantCulture);

    /// <param name="value">Decoded integer field from a journal record, such as a reporting step.</param>
    /// <summary>Converts a decoded journal integer to 64 bits using invariant numeric conventions.</summary>
    /// <returns>The converted integer.</returns>
    /// <exception cref="System.FormatException">The scalar text is not numeric.</exception>
    /// <exception cref="System.OverflowException">The scalar exceeds Int64 (Python ints are unbounded).</exception>
    /// <exception cref="System.InvalidCastException">The scalar does not support numeric conversion.</exception>
    private static long ToLong(object? value) => Convert.ToInt64(value, CultureInfo.InvariantCulture);

    /// <param name="value">Decoded numeric field from a journal record.</param>
    /// <summary>Converts a decoded journal scalar using invariant numeric conventions.</summary>
    /// <returns>The converted scalar.</returns>
    /// <exception cref="System.FormatException">The scalar text is not numeric.</exception>
    /// <exception cref="System.OverflowException">The scalar exceeds the conversion range.</exception>
    /// <exception cref="System.InvalidCastException">The scalar does not support numeric conversion.</exception>
    private static double ToDouble(object? value) => value switch
    {
        double d => d,
        long l => l,
        bool b => b ? 1.0 : 0.0,
        _ => Convert.ToDouble(value, CultureInfo.InvariantCulture),
    };

    // ------------------------------------------------------------------ snapshots

    /// <summary>
    ///     Serializes the replay state in the v1 JSON snapshot format, OptunaSharp's format until plan 10, item 2.7.
    ///     [Misaligned] Not pickle bytes (see <see cref="JournalStorage.RestoreReplayResult" />);
    ///     the format is an OptunaSharp-internal JSON dump.
    /// </summary>
    /// <returns>UTF-8 bytes containing the OptunaSharp versioned replay snapshot.</returns>
    /// <exception cref="NumSharp.TypeError">An attribute value has no JSON representation.</exception>
    /// <exception cref="NumSharp.RuntimeError">An attribute value nests deeper than <see cref="PyJson.MaxDepth" /> within the snapshot.</exception>
    /// <remarks>
    ///     <see cref="Serialize" /> writes the binary v2 format and falls back to this one only when its writer fails,
    ///     so the outcome stays this method's (its bytes or its exception). Every OptunaSharp version reads v1, which
    ///     also makes it the parity tests' reference for what a restore must give.
    /// </remarks>
    internal byte[] SerializeV1()
    {
        var studies = new OrderedDict<string, object?>();
        foreach (var (studyId, study) in _studies)
        {
            studies[studyId.ToString(CultureInfo.InvariantCulture)] = new OrderedDict<string, object?>
            {
                ["study_name"] = study.StudyName,
                ["directions"] = study.Directions.Select(d => (object?)(int)d).ToList(),
                ["user_attrs"] = study.UserAttrs,
                ["system_attrs"] = study.SystemAttrs,
                ["trial_ids"] = _studyIdToTrialIds[studyId].Select(t => (object?)t).ToList(),
            };
        }

        var trials = new OrderedDict<string, object?>();
        foreach (var (trialId, trial) in _trials)
        {
            var distributions = new OrderedDict<string, object?>();
            foreach (var (k, dist) in trial.DistributionsDict)
                distributions[k] = Distribution.ToJson(dist);
            var @params = new OrderedDict<string, object?>();
            foreach (var (k, param) in trial.ParamsDict)
                @params[k] = trial.DistributionsDict[k].ToInternalRepr(param);

            trials[trialId.ToString(CultureInfo.InvariantCulture)] = new OrderedDict<string, object?>
            {
                ["number"] = trial.Number,
                ["state"] = (int)trial.State,
                ["value"] = trial.Values is { Count: 1 } ? trial.Values[0] : null,
                ["values"] = trial.Values is { Count: > 1 } ? trial.Values.Cast<object?>().ToList() : null,
                ["datetime_start"] = trial.DatetimeStart is null
                    ? null
                    : JournalDatetime.ToIsoUtc(trial.DatetimeStart.Value.ToUniversalTime()),
                ["datetime_complete"] = trial.DatetimeComplete is null
                    ? null
                    : JournalDatetime.ToIsoUtc(trial.DatetimeComplete.Value.ToUniversalTime()),
                ["distributions"] = distributions,
                ["params"] = @params,
                ["user_attrs"] = trial.UserAttrsDict,
                ["system_attrs"] = trial.SystemAttrsDict,
                ["intermediate_values"] = trial.IntermediateValues,
                ["study_id"] = _trialIdToStudyId[trialId],
            };
        }

        var payload = new OrderedDict<string, object?>
        {
            ["format"] = "optunasharp-journal-snapshot-v1",
            ["log_number_read"] = LogNumberRead,
            ["next_study_id"] = _nextStudyId,
            ["studies"] = studies,
            ["trials"] = trials,
        };
        return System.Text.Encoding.UTF8.GetBytes(PyJson.DumpsCompact(payload));
    }

    /// <summary>Deserializes a v1 JSON snapshot (<see cref="SerializeV1" />); <c>null</c> on failure.</summary>
    /// <param name="snapshot">UTF-8 bytes produced by the v1 snapshot serializer.</param>
    /// <returns>Restored replay state, or null for any format or content failure.</returns>
    /// <remarks>
    ///     Older OptunaSharp versions write v1, so mixed fleets and snapshots saved before plan 10, item 2.7 still
    ///     restore. The text is read by <see cref="PyJsonUtf8" />, which gives <see cref="PyJson.Loads" />'s values from
    ///     the bytes, and distributions come from the result's <see cref="JournalDistributionCache" />.
    /// </remarks>
    internal static JournalStorageReplayResult? TryDeserializeV1(byte[] snapshot)
    {
        try
        {
            if (PyJsonUtf8.Loads(snapshot) is not OrderedDict<string, object?> payload
                || payload.GetValueOrDefault("format") is not "optunasharp-journal-snapshot-v1")
            {
                return null;
            }

            var result = new JournalStorageReplayResult(string.Empty)
            {
                LogNumberRead = Convert.ToInt32(payload["log_number_read"], CultureInfo.InvariantCulture),
                _nextStudyId = Convert.ToInt32(payload["next_study_id"], CultureInfo.InvariantCulture),
            };

            foreach (var (studyIdStr, rawStudy) in (OrderedDict<string, object?>)payload["studies"]!)
            {
                int studyId = int.Parse(studyIdStr, CultureInfo.InvariantCulture);
                var studyDict = (OrderedDict<string, object?>)rawStudy!;
                result._studies[studyId] = new FrozenStudy(
                    studyName: (string)studyDict["study_name"]!,
                    direction: null,
                    userAttrs: (OrderedDict<string, object?>)studyDict["user_attrs"]!,
                    systemAttrs: (OrderedDict<string, object?>)studyDict["system_attrs"]!,
                    studyId: studyId,
                    directions: ((List<object?>)studyDict["directions"]!)
                        .Select(d => (StudyDirection)Convert.ToInt32(d, CultureInfo.InvariantCulture)));
                result._studyIdToTrialIds[studyId] = ((List<object?>)studyDict["trial_ids"]!)
                    .Select(t => Convert.ToInt32(t, CultureInfo.InvariantCulture)).ToList();
            }

            foreach (var (trialIdStr, rawTrial) in (OrderedDict<string, object?>)payload["trials"]!)
            {
                int trialId = int.Parse(trialIdStr, CultureInfo.InvariantCulture);
                var trialDict = (OrderedDict<string, object?>)rawTrial!;

                var distributions = new OrderedDict<string, BaseDistribution>();
                foreach (var (k, v) in (OrderedDict<string, object?>)trialDict["distributions"]!)
                    distributions[k] = result._distributionCache.FromJson((string)v!);
                var @params = new OrderedDict<string, object?>();
                foreach (var (k, p) in (OrderedDict<string, object?>)trialDict["params"]!)
                    @params[k] = distributions[k].ToExternalRepr(ToDouble(p));
                var intermediateValues = new OrderedDict<long, double>();
                foreach (var (k, v) in (OrderedDict<string, object?>)trialDict["intermediate_values"]!)
                    intermediateValues[long.Parse(k, CultureInfo.InvariantCulture)] = ToDouble(v);

                result._trials[trialId] = new FrozenTrial(
                    trialId: trialId,
                    number: Convert.ToInt32(trialDict["number"], CultureInfo.InvariantCulture),
                    state: (TrialState)Convert.ToInt32(trialDict["state"], CultureInfo.InvariantCulture),
                    @params: @params,
                    distributions: distributions,
                    userAttrs: (OrderedDict<string, object?>)trialDict["user_attrs"]!,
                    systemAttrs: (OrderedDict<string, object?>)trialDict["system_attrs"]!,
                    value: trialDict["value"] is { } v2 ? ToDouble(v2) : null,
                    values: trialDict["values"] is List<object?> vs ? vs.Select(v => ToDouble(v)) : null,
                    intermediateValues: intermediateValues,
                    datetimeStart: trialDict["datetime_start"] is string ds ? JournalDatetime.FromIsoToLocalNaive(ds) : null,
                    datetimeComplete: trialDict["datetime_complete"] is string dc ? JournalDatetime.FromIsoToLocalNaive(dc) : null);
                result._trialIdToStudyId[trialId] = Convert.ToInt32(trialDict["study_id"], CultureInfo.InvariantCulture);
            }

            return result;
        }
        catch (Exception)
        {
            return null;
        }
    }
}

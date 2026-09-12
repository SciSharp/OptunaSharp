using System;
using System.Collections.Generic;
using System.Linq;
using NumSharp;
using OptunaSharp.Collections;
using OptunaSharp.Distributions;

namespace OptunaSharp.Storages;

/// <summary>
///     The backend contract <see cref="CachedStorage" /> wraps — the internal RDBStorage surface
///     it drives in Python (<c>_create_new_trial</c>, <c>_set_trial_param</c>,
///     <c>_get_best_trial_id</c>, <c>_get_trials</c>). The Tier X
///     <c>OptunaSharp.Storage.Sqlite</c>/<c>EFCore</c> RDB storages implement this
///     (PORTING_PLAN 4.1/4.2); the wrapper itself is Tier N because the read-through cache logic
///     is reused there unchanged.
/// </summary>
public interface ICachedStorageBackend : IBaseHeartbeat
{
    /// <summary>Python <c>RDBStorage._create_new_trial</c>: create and return the full frozen trial.</summary>
    /// <param name="studyId">Storage identifier of the study; distinct from its name.</param>
    /// <param name="templateTrial">Optional record to copy; null creates a new running trial with empty observations.</param>
    /// <returns>The created trial including assigned storage identifier and per-study number.</returns>
    /// <exception cref="NumSharp.KeyError">The study does not exist.</exception>
    FrozenTrial CreateNewTrialFull(int studyId, FrozenTrial? templateTrial = null);

    /// <summary>Python <c>RDBStorage._set_trial_param</c> with the cached distribution shortcut.</summary>
    /// <param name="trialId">Storage identifier of the trial; distinct from its per-study trial number.</param>
    /// <param name="paramName">Parameter name used as the storage and distribution lookup key.</param>
    /// <param name="distribution">Distribution defining the parameter representation and allowed values.</param>
    /// <param name="paramValueInternal">Parameter value in the distribution's internal numeric representation.</param>
    /// <param name="cachedDistribution">Previously cached distribution, or null when the backend must load it.</param>
    /// <exception cref="NumSharp.KeyError">The trial does not exist.</exception>
    /// <exception cref="OptunaSharp.UpdateFinishedTrialError">The trial has finished.</exception>
    /// <exception cref="NumSharp.ValueError">The proposed and stored distributions are incompatible.</exception>
    void SetTrialParamWithCachedDistribution(
        int trialId,
        string paramName,
        double paramValueInternal,
        BaseDistribution distribution,
        BaseDistribution? cachedDistribution);

    /// <summary>Python <c>RDBStorage._get_best_trial_id</c>.</summary>
    /// <param name="studyId">Storage identifier of the study; distinct from its name.</param>
    /// <param name="direction">Direction used to order completed objective values.</param>
    /// <returns>Identifier of the best completed trial according to the supplied direction.</returns>
    /// <exception cref="NumSharp.ValueError">The study has no completed trials.</exception>
    int GetBestTrialId(int studyId, StudyDirection direction);

    /// <summary>
    ///     Python <c>RDBStorage._get_trials(study_id, states, included_trial_ids, trial_id_greater_than)</c>:
    ///     trials whose ID is in <paramref name="includedTrialIds" /> or greater than
    ///     <paramref name="trialIdGreaterThan" />.
    /// </summary>
    /// <param name="studyId">Storage identifier of the study; distinct from its name.</param>
    /// <param name="states">States to include; null includes every state.</param>
    /// <param name="includedTrialIds">Identifiers to fetch even when at or below the high-water mark.</param>
    /// <param name="trialIdGreaterThan">Exclusive identifier threshold for fetching trials created since the previous refresh.</param>
    /// <returns>Trials satisfying the state filter and either identifier selection condition.</returns>
    List<FrozenTrial> GetTrialsPartial(
        int studyId,
        IReadOnlyCollection<TrialState>? states,
        IReadOnlyCollection<int> includedTrialIds,
        int trialIdGreaterThan);
}

/// <summary>
///     A wrapper class of storage backends. Port of
///     <c>optuna/storages/_cached_storage.py: _CachedStorage</c> (public in C# so the satellite
///     RDB packages can compose with it).
///
///     <para><b>Data persistence</b> — this class does not guarantee that write operations are
///     logged into a persistent storage, even when write methods succeed. Thus, when process
///     failure occurs, some writes might be lost. As exceptions, when a persistent storage is
///     available, any writes on any attributes of Study and writes on state of Trial are
///     guaranteed to be persistent. Additionally, any preceding writes on any attributes of
///     Trial are guaranteed to be written into a persistent storage before writes on state of
///     Trial succeed. The same applies for param, user_attrs, system_attrs and
///     intermediate_values attributes.</para>
/// </summary>
public class CachedStorage : BaseStorage, IBaseHeartbeat
{
    private sealed class StudyInfo
    {
        // Trial number to corresponding FrozenTrial.
        /// <summary>Cached trials keyed by their per-study number.</summary>
        internal readonly Dictionary<int, FrozenTrial> Trials = new();
        // Trials which require storage access to read latest attributes.
        /// <summary>Trial identifiers that must be refreshed because their state can still change.</summary>
        internal readonly HashSet<int> UnfinishedTrialIds = new();
        /// <summary>Highest finished identifier covered by a complete backend refresh.</summary>
        internal int LastFinishedTrialId = -1;
        // Cache distributions to avoid storage access on distribution consistency check.
        /// <summary>Previously recorded distributions used to validate parameters across trials.</summary>
        internal readonly Dictionary<string, BaseDistribution> ParamDistribution = new();
        /// <summary>Study directions in objective order.</summary>
        internal List<StudyDirection>? Directions;
        /// <summary>Registered study name.</summary>
        internal string? Name;
    }

    private readonly BaseStorage _backendStorage;
    private readonly ICachedStorageBackend _backend;
    private readonly Dictionary<int, StudyInfo> _studies = new();
    private readonly Dictionary<int, (int StudyId, int Number)> _trialIdToStudyIdAndNumber = new();
    private readonly Dictionary<(int StudyId, int Number), int> _studyIdAndNumberToTrialId = new();
    private readonly object _lock = new();

    /// <summary>Wraps an RDB-compatible backend with a synchronized cache.</summary>
    /// <param name="backend">Shared backend implementing ICachedStorageBackend; the wrapper does not own its lifetime.</param>
    /// <exception cref="System.ArgumentException">The backend does not implement ICachedStorageBackend.</exception>
    public CachedStorage(BaseStorage backend)
    {
        if (backend is not ICachedStorageBackend cachedBackend)
            throw new ArgumentException(
                "CachedStorage wraps RDB-style backends implementing ICachedStorageBackend.",
                nameof(backend));
        _backendStorage = backend;
        _backend = cachedBackend;
    }

    /// <summary>The wrapped backend.</summary>
    public BaseStorage Backend => _backendStorage;

    /// <param name="directions">Objective directions in the same order as the objective values.</param>
    /// <param name="studyName">Name identifying the study in the selected storage.</param>
    /// <inheritdoc />
    public override int CreateNewStudy(IReadOnlyList<StudyDirection> directions, string? studyName = null)
    {
        int studyId = _backendStorage.CreateNewStudy(directions, studyName);
        lock (_lock)
        {
            var study = new StudyInfo
            {
                Name = studyName,
                Directions = directions.ToList(),
            };
            _studies[studyId] = study;
        }

        return studyId;
    }

    /// <param name="studyId">Storage identifier of the study; distinct from its name.</param>
    /// <inheritdoc />
    public override void DeleteStudy(int studyId)
    {
        lock (_lock)
        {
            if (_studies.TryGetValue(studyId, out var study))
            {
                foreach (int trialNumber in study.Trials.Keys)
                {
                    if (_studyIdAndNumberToTrialId.TryGetValue((studyId, trialNumber), out int trialId))
                        _trialIdToStudyIdAndNumber.Remove(trialId);
                    _studyIdAndNumberToTrialId.Remove((studyId, trialNumber));
                }
                _studies.Remove(studyId);
            }
        }

        _backendStorage.DeleteStudy(studyId);
    }

    /// <param name="studyId">Storage identifier of the study; distinct from its name.</param>
    /// <param name="key">Attribute name; a later write replaces its stored value.</param>
    /// <param name="value">Attribute value; persistent backends require JSON-serializable data.</param>
    /// <inheritdoc />
    public override void SetStudyUserAttr(int studyId, string key, object? value)
        => _backendStorage.SetStudyUserAttr(studyId, key, value);

    /// <param name="studyId">Storage identifier of the study; distinct from its name.</param>
    /// <param name="key">Attribute name; a later write replaces its stored value.</param>
    /// <param name="value">Attribute value; persistent backends require JSON-serializable data.</param>
    /// <inheritdoc />
    public override void SetStudySystemAttr(int studyId, string key, object? value)
        => _backendStorage.SetStudySystemAttr(studyId, key, value);

    /// <param name="studyName">Name identifying the study in the selected storage.</param>
    /// <inheritdoc />
    public override int GetStudyIdFromName(string studyName)
        => _backendStorage.GetStudyIdFromName(studyName);

    /// <param name="studyId">Storage identifier of the study; distinct from its name.</param>
    /// <inheritdoc />
    public override string GetStudyNameFromId(int studyId)
    {
        lock (_lock)
        {
            if (_studies.TryGetValue(studyId, out var study) && study.Name is not null)
                return study.Name;
        }

        string name = _backendStorage.GetStudyNameFromId(studyId);
        lock (_lock)
        {
            if (!_studies.ContainsKey(studyId))
                _studies[studyId] = new StudyInfo();
            _studies[studyId].Name = name;
        }
        return name;
    }

    /// <param name="studyId">Storage identifier of the study; distinct from its name.</param>
    /// <inheritdoc />
    public override List<StudyDirection> GetStudyDirections(int studyId)
    {
        lock (_lock)
        {
            if (_studies.TryGetValue(studyId, out var study) && study.Directions is not null)
                return study.Directions;
        }

        var directions = _backendStorage.GetStudyDirections(studyId);
        lock (_lock)
        {
            if (!_studies.ContainsKey(studyId))
                _studies[studyId] = new StudyInfo();
            _studies[studyId].Directions = directions;
        }
        return directions;
    }

    /// <param name="studyId">Storage identifier of the study; distinct from its name.</param>
    /// <inheritdoc />
    public override OrderedDict<string, object?> GetStudyUserAttrs(int studyId)
        => _backendStorage.GetStudyUserAttrs(studyId);

    /// <param name="studyId">Storage identifier of the study; distinct from its name.</param>
    /// <inheritdoc />
    public override OrderedDict<string, object?> GetStudySystemAttrs(int studyId)
        => _backendStorage.GetStudySystemAttrs(studyId);

    /// <inheritdoc />
    public override List<FrozenStudy> GetAllStudies() => _backendStorage.GetAllStudies();

    /// <param name="studyId">Storage identifier of the study; distinct from its name.</param>
    /// <inheritdoc />
    /// <param name="templateTrial">Optional record to copy; null creates a new running trial with empty observations.</param>
    public override int CreateNewTrial(int studyId, FrozenTrial? templateTrial = null)
    {
        var frozenTrial = _backend.CreateNewTrialFull(studyId, templateTrial);
        int trialId = frozenTrial.TrialId;
        lock (_lock)
        {
            if (!_studies.ContainsKey(studyId))
                _studies[studyId] = new StudyInfo();
            AddTrialsToCache(studyId, new List<FrozenTrial> { frozenTrial });
        }
        return trialId;
    }

    /// <param name="trialId">Storage identifier of the trial; distinct from its per-study trial number.</param>
    /// <param name="paramName">Parameter name used as the storage and distribution lookup key.</param>
    /// <param name="distribution">Distribution defining the parameter representation and allowed values.</param>
    /// <inheritdoc />
    /// <param name="paramValueInternal">Parameter value in the distribution's internal numeric representation.</param>
    public override void SetTrialParam(
        int trialId, string paramName, double paramValueInternal, BaseDistribution distribution)
    {
        lock (_lock)
        {
            var (studyId, _) = _trialIdToStudyIdAndNumber[trialId];
            var cachedDist = _studies[studyId].ParamDistribution.GetValueOrDefault(paramName);
            _backend.SetTrialParamWithCachedDistribution(
                trialId, paramName, paramValueInternal, distribution, cachedDist);
            if (cachedDist is null)
                _studies[studyId].ParamDistribution[paramName] = distribution;
        }
    }

    /// <param name="studyId">Storage identifier of the study; distinct from its name.</param>
    /// <param name="trialNumber">Zero-based trial number within the study, not a storage identifier.</param>
    /// <inheritdoc />
    public override int GetTrialIdFromStudyIdTrialNumber(int studyId, int trialNumber)
    {
        var key = (studyId, trialNumber);
        lock (_lock)
        {
            if (_studyIdAndNumberToTrialId.TryGetValue(key, out int trialId))
                return trialId;
        }

        return _backendStorage.GetTrialIdFromStudyIdTrialNumber(studyId, trialNumber);
    }

    /// <param name="studyId">Storage identifier of the study; distinct from its name.</param>
    /// <inheritdoc />
    public override FrozenTrial GetBestTrial(int studyId)
    {
        var directions = GetStudyDirections(studyId);
        if (directions.Count > 1)
            throw new RuntimeError("Best trial can be obtained only for single-objective optimization.");
        var direction = directions[0];
        int trialId = _backend.GetBestTrialId(studyId, direction);
        return GetTrial(trialId);
    }

    /// <param name="trialId">Storage identifier of the trial; distinct from its per-study trial number.</param>
    /// <param name="state">Trial state to record or inspect.</param>
    /// <param name="values">Objective values in study direction order, or null when no objective values are recorded.</param>
    /// <inheritdoc />
    public override bool SetTrialStateValues(int trialId, TrialState state, IReadOnlyList<double>? values = null)
        => _backendStorage.SetTrialStateValues(trialId, state: state, values: values);

    /// <param name="trialId">Storage identifier of the trial; distinct from its per-study trial number.</param>
    /// <param name="step">Reporting step whose intermediate objective observation is stored.</param>
    /// <inheritdoc />
    /// <param name="intermediateValue">Intermediate objective observation to store at the reporting step.</param>
    public override void SetTrialIntermediateValue(int trialId, int step, double intermediateValue)
        => _backendStorage.SetTrialIntermediateValue(trialId, step, intermediateValue);

    /// <param name="trialId">Storage identifier of the trial; distinct from its per-study trial number.</param>
    /// <param name="key">Attribute name; a later write replaces its stored value.</param>
    /// <param name="value">Attribute value; persistent backends require JSON-serializable data.</param>
    /// <inheritdoc />
    public override void SetTrialUserAttr(int trialId, string key, object? value)
        => _backendStorage.SetTrialUserAttr(trialId, key: key, value: value);

    /// <param name="trialId">Storage identifier of the trial; distinct from its per-study trial number.</param>
    /// <param name="key">Attribute name; a later write replaces its stored value.</param>
    /// <param name="value">Attribute value; persistent backends require JSON-serializable data.</param>
    /// <inheritdoc />
    public override void SetTrialSystemAttr(int trialId, string key, object? value)
        => _backendStorage.SetTrialSystemAttr(trialId, key: key, value: value);

    /// <param name="trialId">Storage identifier of the trial; distinct from its per-study trial number.</param>
    /// <summary>Looks up a finished trial without consulting the backend.</summary>
    /// <returns>A cached finished trial, or null if the record is missing or unfinished.</returns>
    private FrozenTrial? GetCachedTrial(int trialId)
    {
        if (!_trialIdToStudyIdAndNumber.TryGetValue(trialId, out var location))
            return null;
        var (studyId, number) = location;
        var study = _studies[studyId];
        var trial = study.Trials[number];
        // Finished records are stable. Unfinished records must come from the backend because a
        // different worker can still report values, change attributes, or complete the trial.
        if (!trial.State.IsFinished())
            return null;
        return trial;
    }

    /// <param name="trialId">Storage identifier of the trial; distinct from its per-study trial number.</param>
    /// <inheritdoc />
    public override FrozenTrial GetTrial(int trialId)
    {
        lock (_lock)
        {
            var trial = GetCachedTrial(trialId);
            if (trial is not null)
                return trial;
        }

        return _backendStorage.GetTrial(trialId);
    }

    /// <param name="studyId">Storage identifier of the study; distinct from its name.</param>
    /// <param name="deepcopy">Whether to copy trial records; treat records returned with false as read-only.</param>
    /// <param name="states">States to include; null includes every state.</param>
    /// <inheritdoc />
    public override List<FrozenTrial> GetAllTrials(
        int studyId, bool deepcopy = true, IReadOnlyCollection<TrialState>? states = null)
    {
        ReadTrialsFromRemoteStorage(studyId);

        lock (_lock)
        {
            var study = _studies[studyId];
            // We need to sort trials by their number because some samplers assume this behavior.

            IEnumerable<FrozenTrial> trials = states is not null
                ? study.Trials.Values.Where(t => states.Contains(t.State))
                : study.Trials.Values;
            var sorted = trials.OrderBy(t => t.Number).ToList();
            return deepcopy ? sorted.Select(t => t.DeepCopy()).ToList() : sorted;
        }
    }

    /// <param name="studyId">Storage identifier of the study; distinct from its name.</param>
    /// <summary>Refreshes unfinished and newly created trials before advancing the completed-trial high-water mark.</summary>
    private void ReadTrialsFromRemoteStorage(int studyId)
    {
        lock (_lock)
        {
            if (!_studies.ContainsKey(studyId))
                _studies[studyId] = new StudyInfo();
            var study = _studies[studyId];
            var trials = _backend.GetTrialsPartial(
                studyId,
                states: null,
                includedTrialIds: study.UnfinishedTrialIds,
                trialIdGreaterThan: study.LastFinishedTrialId);
            if (trials.Count == 0)
                return;

            AddTrialsToCache(studyId, trials);
            foreach (var trial in trials)
            {
                if (!trial.State.IsFinished())
                {
                    study.UnfinishedTrialIds.Add(trial.TrialId);
                    continue;
                }

                // Updates to LastFinishedTrialId should only be performed here because they must
                // be executed only when all trials have been considered.
                study.LastFinishedTrialId = Math.Max(study.LastFinishedTrialId, trial.TrialId);
                study.UnfinishedTrialIds.Remove(trial.TrialId);
            }
        }
    }

    /// <param name="studyId">Storage identifier of the study; distinct from its name.</param>
    /// <summary>Updates both identifier indexes and the trial-number cache from backend records.</summary>
    /// <param name="trials">Trial records to process in enumeration order.</param>
    private void AddTrialsToCache(int studyId, List<FrozenTrial> trials)
    {
        var study = _studies[studyId];
        foreach (var trial in trials)
        {
            _trialIdToStudyIdAndNumber[trial.TrialId] = (studyId, trial.Number);
            _studyIdAndNumberToTrialId[(studyId, trial.Number)] = trial.TrialId;
            study.Trials[trial.Number] = trial;
        }
    }

    // ------------------------------------------------------------------ heartbeat passthrough

    /// <param name="trialId">Storage identifier of the trial; distinct from its per-study trial number.</param>
    /// <summary>Persists the latest liveness timestamp for a running trial.</summary>
    /// <exception cref="System.Exception">The heartbeat storage provider fails.</exception>
    public void RecordHeartbeat(int trialId) => _backend.RecordHeartbeat(trialId);

    /// <param name="studyId">Storage identifier of the study; distinct from its name.</param>
    /// <summary>Finds running trials whose heartbeat exceeds the backend grace period.</summary>
    /// <returns>Identifiers of running trials considered stale by the backend.</returns>
    public List<int> GetStaleTrialIds(int studyId) => _backend.GetStaleTrialIds(studyId);

    /// <summary>Reads the configured heartbeat period.</summary>
    /// <returns>Heartbeat interval in seconds, or null when disabled.</returns>
    public int? GetHeartbeatInterval() => _backend.GetHeartbeatInterval();

    /// <summary>Reads the callback used after a stale trial is failed.</summary>
    /// <returns>Configured callback, or null if no stale-trial callback is registered.</returns>
    public Action<Study, FrozenTrial>? GetHeartbeatStaleTrialCallback()
        => _backend.GetHeartbeatStaleTrialCallback();
}

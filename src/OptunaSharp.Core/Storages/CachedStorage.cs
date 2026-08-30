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
    FrozenTrial CreateNewTrialFull(int studyId, FrozenTrial? templateTrial = null);

    /// <summary>Python <c>RDBStorage._set_trial_param</c> with the cached distribution shortcut.</summary>
    void SetTrialParamWithCachedDistribution(
        int trialId,
        string paramName,
        double paramValueInternal,
        BaseDistribution distribution,
        BaseDistribution? cachedDistribution);

    /// <summary>Python <c>RDBStorage._get_best_trial_id</c>.</summary>
    int GetBestTrialId(int studyId, StudyDirection direction);

    /// <summary>
    ///     Python <c>RDBStorage._get_trials(study_id, states, included_trial_ids, trial_id_greater_than)</c>:
    ///     trials whose ID is in <paramref name="includedTrialIds" /> or greater than
    ///     <paramref name="trialIdGreaterThan" />.
    /// </summary>
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
        internal readonly Dictionary<int, FrozenTrial> Trials = new();
        // Trials which require storage access to read latest attributes.
        internal readonly HashSet<int> UnfinishedTrialIds = new();
        internal int LastFinishedTrialId = -1;
        // Cache distributions to avoid storage access on distribution consistency check.
        internal readonly Dictionary<string, BaseDistribution> ParamDistribution = new();
        internal List<StudyDirection>? Directions;
        internal string? Name;
    }

    private readonly BaseStorage _backendStorage;
    private readonly ICachedStorageBackend _backend;
    private readonly Dictionary<int, StudyInfo> _studies = new();
    private readonly Dictionary<int, (int StudyId, int Number)> _trialIdToStudyIdAndNumber = new();
    private readonly Dictionary<(int StudyId, int Number), int> _studyIdAndNumberToTrialId = new();
    private readonly object _lock = new();

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

    public override void SetStudyUserAttr(int studyId, string key, object? value)
        => _backendStorage.SetStudyUserAttr(studyId, key, value);

    public override void SetStudySystemAttr(int studyId, string key, object? value)
        => _backendStorage.SetStudySystemAttr(studyId, key, value);

    public override int GetStudyIdFromName(string studyName)
        => _backendStorage.GetStudyIdFromName(studyName);

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

    public override OrderedDict<string, object?> GetStudyUserAttrs(int studyId)
        => _backendStorage.GetStudyUserAttrs(studyId);

    public override OrderedDict<string, object?> GetStudySystemAttrs(int studyId)
        => _backendStorage.GetStudySystemAttrs(studyId);

    public override List<FrozenStudy> GetAllStudies() => _backendStorage.GetAllStudies();

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

    public override FrozenTrial GetBestTrial(int studyId)
    {
        var directions = GetStudyDirections(studyId);
        if (directions.Count > 1)
            throw new RuntimeError("Best trial can be obtained only for single-objective optimization.");
        var direction = directions[0];
        int trialId = _backend.GetBestTrialId(studyId, direction);
        return GetTrial(trialId);
    }

    public override bool SetTrialStateValues(int trialId, TrialState state, IReadOnlyList<double>? values = null)
        => _backendStorage.SetTrialStateValues(trialId, state: state, values: values);

    public override void SetTrialIntermediateValue(int trialId, int step, double intermediateValue)
        => _backendStorage.SetTrialIntermediateValue(trialId, step, intermediateValue);

    public override void SetTrialUserAttr(int trialId, string key, object? value)
        => _backendStorage.SetTrialUserAttr(trialId, key: key, value: value);

    public override void SetTrialSystemAttr(int trialId, string key, object? value)
        => _backendStorage.SetTrialSystemAttr(trialId, key: key, value: value);

    private FrozenTrial? GetCachedTrial(int trialId)
    {
        if (!_trialIdToStudyIdAndNumber.TryGetValue(trialId, out var location))
            return null;
        var (studyId, number) = location;
        var study = _studies[studyId];
        var trial = study.Trials[number];
        if (!trial.State.IsFinished())
            return null;
        return trial;
    }

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

    public void RecordHeartbeat(int trialId) => _backend.RecordHeartbeat(trialId);

    public List<int> GetStaleTrialIds(int studyId) => _backend.GetStaleTrialIds(studyId);

    public int? GetHeartbeatInterval() => _backend.GetHeartbeatInterval();

    public Action<Study, FrozenTrial>? GetHeartbeatStaleTrialCallback()
        => _backend.GetHeartbeatStaleTrialCallback();
}

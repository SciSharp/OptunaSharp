using System;
using System.Collections.Generic;
using System.Linq;
using NumSharp;
using OptunaSharp.Collections;
using OptunaSharp.Distributions;
using OptunaSharp.Internal;
using OptunaSharp.Logging;

namespace OptunaSharp.Storages;

/// <summary>
///     Storage class that stores data in memory of the current process.
///     Port of <c>optuna/storages/_in_memory.py: InMemoryStorage</c>.
/// </summary>
/// <example>
///     <code>
///     var storage = new InMemoryStorage();
///     var study = Optuna.CreateStudy(storage: storage);
///     study.Optimize(trial => Math.Pow(trial.SuggestFloat("x", -100, 100), 2), nTrials: 10);
///     </code>
/// </example>
public class InMemoryStorage : BaseStorage
{
    private static readonly Logger _logger = OptunaLogging.GetLogger("optuna.storages._in_memory");

    private readonly Dictionary<int, (int StudyId, int Number)> _trialIdToStudyIdAndNumber = new();
    private readonly Dictionary<string, int> _studyNameToId = new();
    private readonly Dictionary<int, StudyInfo> _studies = new();

    private int _maxStudyId = -1;
    private int _maxTrialId = -1;

    private readonly object _lock = new();
    private readonly Dictionary<int, int> _prevWaitingTrialNumber = new();

    public override int CreateNewStudy(IReadOnlyList<StudyDirection> directions, string? studyName = null)
    {
        lock (_lock)
        {
            int studyId = _maxStudyId + 1;
            _maxStudyId += 1;

            if (studyName is not null)
            {
                if (_studyNameToId.ContainsKey(studyName))
                    throw new DuplicatedStudyError();
            }
            else
            {
                string studyUuid = Guid.NewGuid().ToString();
                studyName = DefaultStudyNamePrefix + studyUuid;
            }

            _studies[studyId] = new StudyInfo(studyName, directions.ToList());
            _studyNameToId[studyName] = studyId;
            _prevWaitingTrialNumber[studyId] = 0;

            _logger.Info($"A new study created in memory with name: {studyName}");

            return studyId;
        }
    }

    public override void DeleteStudy(int studyId)
    {
        lock (_lock)
        {
            CheckStudyId(studyId);

            foreach (var trial in _studies[studyId].Trials)
                _trialIdToStudyIdAndNumber.Remove(trial.TrialId);
            string studyName = _studies[studyId].Name;
            _studyNameToId.Remove(studyName);
            _studies.Remove(studyId);
            _prevWaitingTrialNumber.Remove(studyId);
        }
    }

    public override void SetStudyUserAttr(int studyId, string key, object? value)
    {
        lock (_lock)
        {
            CheckStudyId(studyId);

            var userAttrs = _studies[studyId].UserAttrs.ShallowClone();
            userAttrs[key] = value;
            _studies[studyId].UserAttrs = userAttrs;
        }
    }

    public override void SetStudySystemAttr(int studyId, string key, object? value)
    {
        lock (_lock)
        {
            CheckStudyId(studyId);

            var systemAttrs = _studies[studyId].SystemAttrs.ShallowClone();
            systemAttrs[key] = value;
            _studies[studyId].SystemAttrs = systemAttrs;
        }
    }

    public override int GetStudyIdFromName(string studyName)
    {
        lock (_lock)
        {
            if (!_studyNameToId.TryGetValue(studyName, out int studyId))
                throw new KeyError($"No such study {studyName}.");

            return studyId;
        }
    }

    public override string GetStudyNameFromId(int studyId)
    {
        lock (_lock)
        {
            CheckStudyId(studyId);
            return _studies[studyId].Name;
        }
    }

    public override List<StudyDirection> GetStudyDirections(int studyId)
    {
        lock (_lock)
        {
            CheckStudyId(studyId);
            return _studies[studyId].Directions;
        }
    }

    public override OrderedDict<string, object?> GetStudyUserAttrs(int studyId)
    {
        lock (_lock)
        {
            CheckStudyId(studyId);
            return _studies[studyId].UserAttrs;
        }
    }

    public override OrderedDict<string, object?> GetStudySystemAttrs(int studyId)
    {
        lock (_lock)
        {
            CheckStudyId(studyId);
            return _studies[studyId].SystemAttrs;
        }
    }

    public override List<FrozenStudy> GetAllStudies()
    {
        lock (_lock)
        {
            return _studies.Keys.Select(BuildFrozenStudy).ToList();
        }
    }

    private FrozenStudy BuildFrozenStudy(int studyId)
    {
        var study = _studies[studyId];
        return new FrozenStudy(
            studyName: study.Name,
            direction: null,
            directions: study.Directions,
            userAttrs: new OrderedDict<string, object?>(
                study.UserAttrs.Select(kv => new KeyValuePair<string, object?>(kv.Key, PyJson.DeepCopy(kv.Value)))),
            systemAttrs: new OrderedDict<string, object?>(
                study.SystemAttrs.Select(kv => new KeyValuePair<string, object?>(kv.Key, PyJson.DeepCopy(kv.Value)))),
            studyId: studyId);
    }

    public override int CreateNewTrial(int studyId, FrozenTrial? templateTrial = null)
    {
        lock (_lock)
        {
            CheckStudyId(studyId);

            FrozenTrial trial = templateTrial is null ? CreateRunningTrial() : templateTrial.DeepCopy();

            int trialId = _maxTrialId + 1;
            _maxTrialId += 1;
            trial.SetNumber(_studies[studyId].Trials.Count);
            trial.TrialId = trialId;
            _trialIdToStudyIdAndNumber[trialId] = (studyId, trial.Number);
            _studies[studyId].Trials.Add(trial);
            UpdateCache(trialId, studyId);
            return trialId;
        }
    }

    private static FrozenTrial CreateRunningTrial() => new(
        trialId: -1,  // dummy value.
        number: -1,  // dummy value.
        state: TrialState.Running,
        @params: new OrderedDict<string, object?>(),
        distributions: new OrderedDict<string, BaseDistribution>(),
        userAttrs: new OrderedDict<string, object?>(),
        systemAttrs: new OrderedDict<string, object?>(),
        value: null,
        intermediateValues: new OrderedDict<int, double>(),
        datetimeStart: DateTime.Now,
        datetimeComplete: null);

    public override void SetTrialParam(
        int trialId, string paramName, double paramValueInternal, BaseDistribution distribution)
    {
        lock (_lock)
        {
            var trial = GetTrialUnlocked(trialId);

            CheckTrialIsUpdatable(trialId, trial.State);

            int studyId = _trialIdToStudyIdAndNumber[trialId].StudyId;
            // Check param distribution compatibility with previous trial(s).
            if (_studies[studyId].ParamDistribution.TryGetValue(paramName, out var existing))
                Distribution.CheckDistributionCompatibility(existing, distribution);

            // Set param distribution.
            _studies[studyId].ParamDistribution[paramName] = distribution;

            // Set param.
            trial = trial.ShallowCopy();
            trial.ParamsDict = trial.ParamsDict.ShallowClone();
            trial.ParamsDict[paramName] = distribution.ToExternalRepr(paramValueInternal);
            trial.DistributionsDict = trial.DistributionsDict.ShallowClone();
            trial.DistributionsDict[paramName] = distribution;
            SetTrial(trialId, trial);
        }
    }

    public override int GetTrialIdFromStudyIdTrialNumber(int studyId, int trialNumber)
    {
        lock (_lock)
        {
            if (!_studies.TryGetValue(studyId, out var study))
                throw new KeyError($"No study with study_id {studyId} exists.");

            var trials = study.Trials;
            if (trials.Count <= trialNumber)
                throw new KeyError(
                    $"No trial with trial number {trialNumber} exists " +
                    $"in study with study_id {studyId}.");

            var trial = trials[trialNumber];
            System.Diagnostics.Debug.Assert(trial.Number == trialNumber);

            return trial.TrialId;
        }
    }

    public override int GetTrialNumberFromId(int trialId)
    {
        lock (_lock)
        {
            CheckTrialId(trialId);

            return _trialIdToStudyIdAndNumber[trialId].Number;
        }
    }

    public override FrozenTrial GetBestTrial(int studyId)
    {
        lock (_lock)
        {
            CheckStudyId(studyId);

            int? bestTrialId = _studies[studyId].BestTrialId;

            if (bestTrialId is null)
                throw new ValueError("No trials are completed yet.");
            if (_studies[studyId].Directions.Count > 1)
                throw new RuntimeError("Best trial can be obtained only for single-objective optimization.");
            return GetTrialUnlocked(bestTrialId.Value);
        }
    }

    public override double GetTrialParam(int trialId, string paramName)
    {
        lock (_lock)
        {
            var trial = GetTrialUnlocked(trialId);

            var distribution = trial.Distributions[paramName];
            return distribution.ToInternalRepr(trial.Params[paramName]);
        }
    }

    public override bool SetTrialStateValues(int trialId, TrialState state, IReadOnlyList<double>? values = null)
    {
        lock (_lock)
        {
            var trial = GetTrialUnlocked(trialId).ShallowCopy();
            CheckTrialIsUpdatable(trialId, trial.State);

            if (state == TrialState.Running && trial.State != TrialState.Waiting)
                return false;

            trial.State = state;
            if (values is not null)
                trial.SetValues(values);

            if (state == TrialState.Running)
                trial.SetDatetimeStart(DateTime.Now);

            if (state.IsFinished())
            {
                trial.DatetimeComplete = DateTime.Now;
                SetTrial(trialId, trial);
                int studyId = _trialIdToStudyIdAndNumber[trialId].StudyId;
                UpdateCache(trialId, studyId);
            }
            else
            {
                SetTrial(trialId, trial);
            }

            return true;
        }
    }

    private void UpdateCache(int trialId, int studyId)
    {
        var trial = GetTrialUnlocked(trialId);

        if (trial.State != TrialState.Complete)
            return;

        int? bestTrialId = _studies[studyId].BestTrialId;
        if (bestTrialId is null)
        {
            _studies[studyId].BestTrialId = trialId;
            return;
        }

        var directions = _studies[studyId].Directions;
        if (directions.Count > 1)
            return;
        var direction = directions[0];

        var bestTrial = GetTrialUnlocked(bestTrialId.Value);
        if (bestTrial.Value is null)
        {
            _studies[studyId].BestTrialId = trialId;
            return;
        }
        // Complete trials do not have `null` values.
        System.Diagnostics.Debug.Assert(trial.Value is not null);
        double bestValue = bestTrial.Value.Value;
        double newValue = trial.Value!.Value;

        if (direction == StudyDirection.Maximize)
        {
            if (bestValue < newValue)
                _studies[studyId].BestTrialId = trialId;
        }
        else
        {
            if (bestValue > newValue)
                _studies[studyId].BestTrialId = trialId;
        }
    }

    public override void SetTrialIntermediateValue(int trialId, int step, double intermediateValue)
    {
        lock (_lock)
        {
            var trial = GetTrialUnlocked(trialId);
            CheckTrialIsUpdatable(trialId, trial.State);

            trial = trial.ShallowCopy();
            trial.IntermediateValues = trial.IntermediateValues.ShallowClone();
            trial.IntermediateValues[step] = intermediateValue;
            SetTrial(trialId, trial);
        }
    }

    public override void SetTrialUserAttr(int trialId, string key, object? value)
    {
        lock (_lock)
        {
            CheckTrialId(trialId);
            var trial = GetTrialUnlocked(trialId);
            CheckTrialIsUpdatable(trialId, trial.State);

            trial = trial.ShallowCopy();
            trial.UserAttrsDict = trial.UserAttrsDict.ShallowClone();
            trial.UserAttrsDict[key] = value;
            SetTrial(trialId, trial);
        }
    }

    public override void SetTrialSystemAttr(int trialId, string key, object? value)
    {
        lock (_lock)
        {
            var trial = GetTrialUnlocked(trialId);
            CheckTrialIsUpdatable(trialId, trial.State);

            trial = trial.ShallowCopy();
            trial.SystemAttrsDict = trial.SystemAttrsDict.ShallowClone();
            trial.SystemAttrsDict[key] = value;
            SetTrial(trialId, trial);
        }
    }

    public override FrozenTrial GetTrial(int trialId)
    {
        lock (_lock)
        {
            return GetTrialUnlocked(trialId);
        }
    }

    private FrozenTrial GetTrialUnlocked(int trialId)
    {
        CheckTrialId(trialId);
        var (studyId, trialNumber) = _trialIdToStudyIdAndNumber[trialId];
        return _studies[studyId].Trials[trialNumber];
    }

    private void SetTrial(int trialId, FrozenTrial trial)
    {
        var (studyId, trialNumber) = _trialIdToStudyIdAndNumber[trialId];
        _studies[studyId].Trials[trialNumber] = trial;
    }

    public override List<FrozenTrial> GetAllTrials(
        int studyId, bool deepcopy = true, IReadOnlyCollection<TrialState>? states = null)
    {
        lock (_lock)
        {
            CheckStudyId(studyId);

            List<FrozenTrial> trials;

            // Optimized retrieval of trials in the WAITING state to improve performance
            // for the call, `GetAllTrials(states: new[] { TrialState.Waiting })`.
            if (states is { Count: 1 } && states.Contains(TrialState.Waiting))
            {
                trials = new List<FrozenTrial>();
                var studyTrials = _studies[studyId].Trials;
                for (int i = _prevWaitingTrialNumber[studyId]; i < studyTrials.Count; i++)
                {
                    var trial = studyTrials[i];
                    if (trial.State == TrialState.Waiting)
                    {
                        if (trials.Count == 0)
                            _prevWaitingTrialNumber[studyId] = trial.Number;
                        trials.Add(trial);
                    }
                }
                if (trials.Count == 0)
                    _prevWaitingTrialNumber[studyId] = _studies[studyId].Trials.Count;
            }
            else
            {
                trials = states is null
                    ? new List<FrozenTrial>(_studies[studyId].Trials)
                    : _studies[studyId].Trials.Where(t => states.Contains(t.State)).ToList();
            }

            if (deepcopy)
                trials = trials.Select(t => t.DeepCopy()).ToList();

            return trials;
        }
    }

    private void CheckStudyId(int studyId)
    {
        if (!_studies.ContainsKey(studyId))
            throw new KeyError($"No study with study_id {studyId} exists.");
    }

    private void CheckTrialId(int trialId)
    {
        if (!_trialIdToStudyIdAndNumber.ContainsKey(trialId))
            throw new KeyError($"No trial with trial_id {trialId} exists.");
    }

    private sealed class StudyInfo
    {
        internal StudyInfo(string name, List<StudyDirection> directions)
        {
            Name = name;
            Directions = directions;
        }

        internal List<FrozenTrial> Trials { get; } = new();
        internal Dictionary<string, BaseDistribution> ParamDistribution { get; } = new();
        internal OrderedDict<string, object?> UserAttrs { get; set; } = new();
        internal OrderedDict<string, object?> SystemAttrs { get; set; } = new();
        internal string Name { get; }
        internal List<StudyDirection> Directions { get; }
        internal int? BestTrialId { get; set; }
    }
}

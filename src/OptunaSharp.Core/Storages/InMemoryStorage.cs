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

    /// <param name="directions">Objective directions in the same order as the objective values.</param>
    /// <param name="studyName">Name identifying the study in the selected storage.</param>
    /// <inheritdoc />
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

    /// <param name="studyId">Storage identifier of the study; distinct from its name.</param>
    /// <inheritdoc />
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

    /// <param name="studyId">Storage identifier of the study; distinct from its name.</param>
    /// <param name="key">Attribute name; a later write replaces its stored value.</param>
    /// <param name="value">Attribute value; persistent backends require JSON-serializable data.</param>
    /// <inheritdoc />
    public override void SetStudyUserAttr(int studyId, string key, object? value)
    {
        lock (_lock)
        {
            CheckStudyId(studyId);

            // Replace the dictionary instead of mutating it so readers holding an older snapshot
            // do not observe a write performed after their read.
            var userAttrs = _studies[studyId].UserAttrs.ShallowClone();
            userAttrs[key] = value;
            _studies[studyId].UserAttrs = userAttrs;
        }
    }

    /// <param name="studyId">Storage identifier of the study; distinct from its name.</param>
    /// <param name="key">Attribute name; a later write replaces its stored value.</param>
    /// <param name="value">Attribute value; persistent backends require JSON-serializable data.</param>
    /// <inheritdoc />
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

    /// <param name="studyName">Name identifying the study in the selected storage.</param>
    /// <inheritdoc />
    public override int GetStudyIdFromName(string studyName)
    {
        lock (_lock)
        {
            if (!_studyNameToId.TryGetValue(studyName, out int studyId))
                throw new KeyError($"No such study {studyName}.");

            return studyId;
        }
    }

    /// <param name="studyId">Storage identifier of the study; distinct from its name.</param>
    /// <inheritdoc />
    public override string GetStudyNameFromId(int studyId)
    {
        lock (_lock)
        {
            CheckStudyId(studyId);
            return _studies[studyId].Name;
        }
    }

    /// <param name="studyId">Storage identifier of the study; distinct from its name.</param>
    /// <inheritdoc />
    public override List<StudyDirection> GetStudyDirections(int studyId)
    {
        lock (_lock)
        {
            CheckStudyId(studyId);
            return _studies[studyId].Directions;
        }
    }

    /// <param name="studyId">Storage identifier of the study; distinct from its name.</param>
    /// <inheritdoc />
    public override OrderedDict<string, object?> GetStudyUserAttrs(int studyId)
    {
        lock (_lock)
        {
            CheckStudyId(studyId);
            return _studies[studyId].UserAttrs;
        }
    }

    /// <param name="studyId">Storage identifier of the study; distinct from its name.</param>
    /// <inheritdoc />
    public override OrderedDict<string, object?> GetStudySystemAttrs(int studyId)
    {
        lock (_lock)
        {
            CheckStudyId(studyId);
            return _studies[studyId].SystemAttrs;
        }
    }

    /// <inheritdoc />
    public override List<FrozenStudy> GetAllStudies()
    {
        lock (_lock)
        {
            return _studies.Keys.Select(BuildFrozenStudy).ToList();
        }
    }

    /// <param name="studyId">Storage identifier of the study; distinct from its name.</param>
    /// <summary>Copies study metadata while the storage lock is held.</summary>
    /// <returns>A study record with copied attribute containers.</returns>
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

    /// <param name="studyId">Storage identifier of the study; distinct from its name.</param>
    /// <inheritdoc />
    /// <param name="templateTrial">Optional record to copy; null creates a new running trial with empty observations.</param>
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

    /// <summary>Creates an empty running record before storage assigns its identifiers.</summary>
    /// <returns>A running trial with current start time and placeholder identifiers.</returns>
    private static FrozenTrial CreateRunningTrial() => new(
        trialId: -1,  // dummy value.
        number: -1,  // dummy value.
        state: TrialState.Running,
        @params: new OrderedDict<string, object?>(),
        distributions: new OrderedDict<string, BaseDistribution>(),
        userAttrs: new OrderedDict<string, object?>(),
        systemAttrs: new OrderedDict<string, object?>(),
        value: null,
        intermediateValues: new OrderedDict<long, double>(),
        datetimeStart: DateTime.Now,
        datetimeComplete: null);

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

    /// <param name="studyId">Storage identifier of the study; distinct from its name.</param>
    /// <param name="trialNumber">Zero-based trial number within the study, not a storage identifier.</param>
    /// <inheritdoc />
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

    /// <param name="trialId">Storage identifier of the trial; distinct from its per-study trial number.</param>
    /// <inheritdoc />
    public override int GetTrialNumberFromId(int trialId)
    {
        lock (_lock)
        {
            CheckTrialId(trialId);

            return _trialIdToStudyIdAndNumber[trialId].Number;
        }
    }

    /// <param name="studyId">Storage identifier of the study; distinct from its name.</param>
    /// <inheritdoc />
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

    /// <param name="trialId">Storage identifier of the trial; distinct from its per-study trial number.</param>
    /// <param name="paramName">Parameter name used as the storage and distribution lookup key.</param>
    /// <inheritdoc />
    public override double GetTrialParam(int trialId, string paramName)
    {
        lock (_lock)
        {
            var trial = GetTrialUnlocked(trialId);

            var distribution = trial.Distributions[paramName];
            return distribution.ToInternalRepr(trial.Params[paramName]);
        }
    }

    /// <param name="trialId">Storage identifier of the trial; distinct from its per-study trial number.</param>
    /// <param name="state">Trial state to record or inspect.</param>
    /// <param name="values">Objective values in study direction order, or null when no objective values are recorded.</param>
    /// <inheritdoc />
    public override bool SetTrialStateValues(int trialId, TrialState state, IReadOnlyList<double>? values = null)
    {
        lock (_lock)
        {
            var trial = GetTrialUnlocked(trialId).ShallowCopy();
            CheckTrialIsUpdatable(trialId, trial.State);

            // The lock makes claiming WAITING work atomic: only one worker can transition a queued
            // trial to RUNNING, and losers can continue searching without evaluating it twice.
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

    /// <param name="trialId">Storage identifier of the trial; distinct from its per-study trial number.</param>
    /// <param name="studyId">Storage identifier of the study; distinct from its name.</param>
    /// <summary>Updates the best-trial identifier after inserting or completing a trial; requires the storage lock.</summary>
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

    /// <param name="trialId">Storage identifier of the trial; distinct from its per-study trial number.</param>
    /// <param name="step">Reporting step whose intermediate objective observation is stored.</param>
    /// <inheritdoc />
    /// <param name="intermediateValue">Intermediate objective observation to store at the reporting step.</param>
    public override void SetTrialIntermediateValue(int trialId, long step, double intermediateValue)
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

    /// <param name="trialId">Storage identifier of the trial; distinct from its per-study trial number.</param>
    /// <param name="key">Attribute name; a later write replaces its stored value.</param>
    /// <param name="value">Attribute value; persistent backends require JSON-serializable data.</param>
    /// <inheritdoc />
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

    /// <param name="trialId">Storage identifier of the trial; distinct from its per-study trial number.</param>
    /// <param name="key">Attribute name; a later write replaces its stored value.</param>
    /// <param name="value">Attribute value; persistent backends require JSON-serializable data.</param>
    /// <inheritdoc />
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

    /// <param name="trialId">Storage identifier of the trial; distinct from its per-study trial number.</param>
    /// <inheritdoc />
    public override FrozenTrial GetTrial(int trialId)
    {
        lock (_lock)
        {
            return GetTrialUnlocked(trialId);
        }
    }

    /// <param name="trialId">Storage identifier of the trial; distinct from its per-study trial number.</param>
    /// <summary>Loads the shared trial record while the caller holds the storage lock.</summary>
    /// <returns>The stored trial instance.</returns>
    /// <exception cref="NumSharp.KeyError">The identifier does not exist.</exception>
    private FrozenTrial GetTrialUnlocked(int trialId)
    {
        CheckTrialId(trialId);
        var (studyId, trialNumber) = _trialIdToStudyIdAndNumber[trialId];
        return _studies[studyId].Trials[trialNumber];
    }

    /// <param name="trialId">Storage identifier of the trial; distinct from its per-study trial number.</param>
    /// <param name="trial">Trial record or active trial on which the operation acts.</param>
    /// <summary>Replaces a stored trial snapshot while the caller holds the storage lock.</summary>
    private void SetTrial(int trialId, FrozenTrial trial)
    {
        var (studyId, trialNumber) = _trialIdToStudyIdAndNumber[trialId];
        _studies[studyId].Trials[trialNumber] = trial;
    }

    /// <param name="studyId">Storage identifier of the study; distinct from its name.</param>
    /// <inheritdoc />
    /// <remarks>
    ///     O(1): wraps the live per-study list in a <see cref="TrialsPrefixView" /> (membership frozen at the
    ///     current count, reads see current copy-on-write instances) instead of copying it — the difference
    ///     between a linear and a quadratic optimization loop for GA samplers that read history every ask.
    /// </remarks>
    public override IReadOnlyList<FrozenTrial> GetTrialsPrefixView(int studyId)
    {
        lock (_lock)
        {
            CheckStudyId(studyId);
            List<FrozenTrial> trials = _studies[studyId].Trials;
            return new TrialsPrefixView(trials, trials.Count);
        }
    }

    /// <inheritdoc />
    /// <remarks>
    ///     <see langword="true" />: <see cref="GetTrialsPrefixView" /> above wraps the live list without copying it, so
    ///     samplers that read the history on every ask can afford to read it through the view.
    /// </remarks>
    internal override bool HasZeroCopyTrialsView => true;

    /// <param name="studyId">Storage identifier of the study; distinct from its name.</param>
    /// <param name="deepcopy">Whether to copy trial records; treat records returned with false as read-only.</param>
    /// <param name="states">States to include; null includes every state.</param>
    /// <inheritdoc />
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

    /// <param name="studyId">Storage identifier of the study; distinct from its name.</param>
    /// <summary>Rejects identifiers that are absent from this storage instance.</summary>
    /// <exception cref="NumSharp.KeyError">The identifier does not exist.</exception>
    private void CheckStudyId(int studyId)
    {
        if (!_studies.ContainsKey(studyId))
            throw new KeyError($"No study with study_id {studyId} exists.");
    }

    /// <param name="trialId">Storage identifier of the trial; distinct from its per-study trial number.</param>
    /// <summary>Rejects identifiers that are absent from this storage instance.</summary>
    /// <exception cref="NumSharp.KeyError">The identifier does not exist.</exception>
    private void CheckTrialId(int trialId)
    {
        if (!_trialIdToStudyIdAndNumber.ContainsKey(trialId))
            throw new KeyError($"No trial with trial_id {trialId} exists.");
    }

    private sealed class StudyInfo
    {
        /// <param name="name">Study name registered in this storage instance.</param>
        /// <param name="directions">Objective directions in the same order as the objective values.</param>
        /// <summary>Initializes the in-memory metadata and empty trial history for a study.</summary>
        internal StudyInfo(string name, List<StudyDirection> directions)
        {
            Name = name;
            Directions = directions;
        }

        /// <summary>Cached trials keyed by their per-study number.</summary>
        internal List<FrozenTrial> Trials { get; } = new();
        /// <summary>Previously recorded distributions used to validate parameters across trials.</summary>
        internal Dictionary<string, BaseDistribution> ParamDistribution { get; } = new();
        /// <summary>Copy-on-write user attributes for the study.</summary>
        internal OrderedDict<string, object?> UserAttrs { get; set; } = new();
        /// <summary>Copy-on-write internal attributes for the study.</summary>
        internal OrderedDict<string, object?> SystemAttrs { get; set; } = new();
        /// <summary>Registered study name.</summary>
        internal string Name { get; }
        /// <summary>Study directions in objective order.</summary>
        internal List<StudyDirection> Directions { get; }
        /// <summary>Identifier of the best completed trial cached for single-objective lookup.</summary>
        internal int? BestTrialId { get; set; }
    }
}

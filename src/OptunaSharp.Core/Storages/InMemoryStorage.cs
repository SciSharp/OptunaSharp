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
    private readonly bool _compactFinishedTrials;

    /// <summary>Creates an empty storage that keeps every trial's dictionaries as Optuna's in-memory storage does.</summary>
    public InMemoryStorage()
        : this(compactFinishedTrials: false)
    {
    }

    /// <summary>
    ///     Creates an empty storage, optionally keeping finished trials in a compact form (Rustuna plan 13, item S12).
    /// </summary>
    /// <param name="compactFinishedTrials">
    ///     When true, each trial is compacted as it finishes: parameters become one internal value per name against a
    ///     parameter schema the study's trials share, distributions come from that schema, attributes become arrays
    ///     with long ASCII strings stored as bytes, and empty dictionaries take no memory. False (the default) keeps
    ///     Optuna's representation exactly.
    /// </param>
    /// <remarks>
    ///     <para>
    ///         What stays the same: every value any API returns, and so every sampler's results; a seeded study gives
    ///         the same trials either way. A finished 40-parameter random-search trial took 6.7 KB in memory and takes
    ///         0.7 KB compact; a CMA-ES trial 13.5 KB and 2.7 KB, its optimizer states stored as bytes (100,000 trials:
    ///         1,287 MB and 257 MB). Run times stay the same within noise.
    ///     </para>
    ///     <para>
    ///         What changes, which is why it is opt-in: a finished trial's <see cref="FrozenTrial.Params" />,
    ///         <see cref="FrozenTrial.Distributions" />, <see cref="FrozenTrial.UserAttrs" /> and
    ///         <see cref="FrozenTrial.SystemAttrs" /> become read-only views rather than <see cref="OrderedDict{TKey,TValue}" />
    ///         objects (a cast to <c>OrderedDict</c> fails, a write through <c>IDictionary</c> throws), and numeric values
    ///         and long strings are rebuilt on each read, so two reads are equal but not the same object. The mutable
    ///         accessors (<see cref="FrozenTrial.ParamsDict" /> and the other <c>…Dict</c> properties, and
    ///         <see cref="FrozenTrial.IntermediateValues" />) still return real dictionaries: the first such read gives
    ///         the trial back a dictionary of its own, which it keeps, so changes made through them stay, at the cost of
    ///         that trial's saving. A finished trial no longer shares its dictionaries with the running record it was
    ///         copied from. A read of a numeric parameter through the views allocates a box; TPE, which reads every
    ///         trial on every ask, copies the stored internal values instead.
    ///     </para>
    /// </remarks>
    public InMemoryStorage(bool compactFinishedTrials) => _compactFinishedTrials = compactFinishedTrials;

    /// <summary>Whether this storage keeps finished trials in the compact form (see the constructor).</summary>
    public bool CompactFinishedTrials => _compactFinishedTrials;

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
                // Python's bare `raise DuplicatedStudyError`: str(e) is empty, where .NET's parameterless constructor
                // supplies "Exception of type ... was thrown." (QA-66). create_study replaces it with its own message.
                if (_studyNameToId.ContainsKey(studyName))
                    throw new DuplicatedStudyError(string.Empty);
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

    /// <param name="studyId">Storage identifier of the study; distinct from its name.</param>
    /// <param name="attrs">The attributes, written in enumeration order.</param>
    /// <inheritdoc />
    /// <remarks>
    ///     One lock and one new dictionary for the whole batch (plan 13, item S8), published only after every entry is
    ///     in it, so a reader sees all of them or none, and a reader holding the previous dictionary sees no change.
    /// </remarks>
    public override void SetStudyUserAttrs(int studyId, IReadOnlyDictionary<string, object?> attrs)
    {
        ArgumentNullException.ThrowIfNull(attrs);
        if (attrs.Count == 0)
            return;
        lock (_lock)
        {
            CheckStudyId(studyId);
            var userAttrs = _studies[studyId].UserAttrs.ShallowClone();
            foreach (var (key, value) in attrs)
                userAttrs[key] = value;
            _studies[studyId].UserAttrs = userAttrs;
        }
    }

    /// <param name="studyId">Storage identifier of the study; distinct from its name.</param>
    /// <param name="attrs">The attributes, written in enumeration order.</param>
    /// <inheritdoc />
    /// <remarks>One lock and one new dictionary for the whole batch, as <see cref="SetStudyUserAttrs" />.</remarks>
    public override void SetStudySystemAttrs(int studyId, IReadOnlyDictionary<string, object?> attrs)
    {
        ArgumentNullException.ThrowIfNull(attrs);
        if (attrs.Count == 0)
            return;
        lock (_lock)
        {
            CheckStudyId(studyId);
            var systemAttrs = _studies[studyId].SystemAttrs.ShallowClone();
            foreach (var (key, value) in attrs)
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
            // Both branches built a record with dictionaries nobody else holds, so its writes may go in place until a
            // reader receives it (see FrozenTrial.StorageExclusive).
            trial.StorageExclusive = true;
            // A trial added already finished (Study.AddTrial) is compacted as one finishing here would be; it is never
            // written again, so the exclusive flag above stays unused for it.
            if (_compactFinishedTrials && trial.State.IsFinished())
                trial.CompactFinished(SchemasOf(studyId));
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

            var study = _studies[_trialIdToStudyIdAndNumber[trialId].StudyId];
            // Check param distribution compatibility with previous trial(s).
            if (study.ParamDistribution.TryGetValue(paramName, out var existing))
            {
                Distribution.CheckDistributionCompatibility(existing, distribution);
                // Share one instance per parameter across the study's trials when the new one is the same distribution
                // to the bit (plan 13, item S7): memory only, the trial still holds a distribution identical in every
                // field. Bitwise, so a -0.0 bound or a categorical with equal-but-different choices (1 vs 1.0, which
                // Python calls equal) keeps the caller's own instance and prints as Optuna would.
                if (Distribution.IsBitwiseIdentical(existing, distribution))
                    distribution = existing;
            }

            // Set param distribution.
            study.ParamDistribution[paramName] = distribution;
            // One string instance per parameter name across the study (plan 13, S7): names built per call, such as
            // $"x{i}", would otherwise keep a copy alive in every trial.
            if (!study.ParamNames.TryGetValue(paramName, out string? name))
                study.ParamNames[paramName] = name = paramName;

            // Set param. In place while the storage alone holds the record; otherwise an own copy first, as Optuna's
            // copy-on-write does (WritableTrialUnlocked).
            trial = WritableTrialUnlocked(trialId);
            trial.ParamsDict[name] = distribution.ToExternalRepr(paramValueInternal);
            trial.DistributionsDict[name] = distribution;
        }
    }

    /// <summary>
    ///     The record of <paramref name="trialId" /> in a form this write may change: the stored object itself while the
    ///     storage alone holds it, otherwise a copy that owns all of its dictionaries, stored in its place (Rustuna plan
    ///     13, item S4).
    /// </summary>
    /// <param name="trialId">The trial being written; the caller holds the lock and has checked it is updatable.</param>
    /// <returns>A record the caller may change in place; it stays the stored record.</returns>
    /// <remarks>
    ///     <para>
    ///         Optuna copies the record and the dictionary it writes on every write, so a reader holding the old record
    ///         never sees a change. The copy here happens only when a reader may hold the record (any read clears
    ///         <see cref="FrozenTrial.StorageExclusive" />), and it clones all five dictionaries, because a later write
    ///         may touch any of them in place. A trial's suggestions, which nobody reads between, then cost one copy
    ///         instead of one per suggestion.
    ///     </para>
    ///     <para>
    ///         State changes (<see cref="SetTrialStateValues" />) keep copying: a finished record is read without the
    ///         lock through <see cref="GetTrialsPrefixView" />, so it must be published whole, never changed in place.
    ///     </para>
    /// </remarks>
    private FrozenTrial WritableTrialUnlocked(int trialId)
    {
        var trial = GetTrialUnlocked(trialId);
        if (trial.StorageExclusive)
            return trial;

        var own = trial.ShallowCopy();
        own.ParamsDict = trial.ParamsDict.ShallowClone();
        own.DistributionsDict = trial.DistributionsDict.ShallowClone();
        own.UserAttrsDict = trial.UserAttrsDict.ShallowClone();
        own.SystemAttrsDict = trial.SystemAttrsDict.ShallowClone();
        own.IntermediateValues = trial.IntermediateValues.ShallowClone();
        own.StorageExclusive = true;
        SetTrial(trialId, own);
        WriteCopyCount++;
        return own;
    }

    /// <summary>The study's schema cache for compact finished trials, created on first use (plan 13, item S12).</summary>
    /// <param name="studyId">A study of this storage; the caller holds the lock.</param>
    /// <returns>The cache the study's compact trials share schemas through.</returns>
    private ParamSchemaCache SchemasOf(int studyId)
    {
        StudyInfo study = _studies[studyId];
        return study.Schemas ??= new ParamSchemaCache();
    }

    /// <summary>
    ///     How many writes had to copy a trial's record first because a reader may hold it (Rustuna plan 13, item S4):
    ///     a diagnostic the tests read to prove that suggestions nobody reads between are written in place.
    /// </summary>
    /// <remarks>Counted under the storage's lock; read it while no write is running, or the value is approximate.</remarks>
    internal int WriteCopyCount { get; private set; }

    /// <summary>
    ///     Marks a stored record as held by a reader, so the next write copies it first: every path that returns a
    ///     stored record (rather than a copy) goes through here.
    /// </summary>
    /// <param name="trial">A record of this storage, read under the lock.</param>
    /// <returns><paramref name="trial" />, for chaining at the return statement.</returns>
    private static FrozenTrial HandOut(FrozenTrial trial)
    {
        trial.StorageExclusive = false;
        return trial;
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
            // A completed record is never written again, but handing it out keeps the rule uniform.
            return HandOut(GetTrialUnlocked(bestTrialId.Value));
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

            // A compact finished trial (plan 13, item S12) stores exactly the internal value the conversion below would
            // rebuild through its own distribution, so it is returned as stored. A missing name falls through to the
            // dictionary path's KeyError.
            if (trial.Params is CompactParams compact && compact.TryGetInternal(paramName, out double stored))
                return stored;

            // Python's `trial.distributions[param_name]`: KeyError(param_name), not KeyNotFoundException (QA-66).
            if (!trial.Distributions.TryGetValue(paramName, out var distribution))
                throw new KeyError(paramName);
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
                int studyId = _trialIdToStudyIdAndNumber[trialId].StudyId;
                // Compact the new record before it is published: no reader can see this object yet, and the
                // dictionaries it shares with the record it replaces are only read (FrozenTrial.CompactFinished).
                if (_compactFinishedTrials)
                    trial.CompactFinished(SchemasOf(studyId));
                SetTrial(trialId, trial);
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

            // In place unless a reader may hold the record (WritableTrialUnlocked).
            WritableTrialUnlocked(trialId).IntermediateValues[step] = intermediateValue;
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

            // In place unless a reader may hold the record (WritableTrialUnlocked).
            WritableTrialUnlocked(trialId).UserAttrsDict[key] = value;
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

            // In place unless a reader may hold the record (WritableTrialUnlocked).
            WritableTrialUnlocked(trialId).SystemAttrsDict[key] = value;
        }
    }

    /// <param name="trialId">Storage identifier of the trial; distinct from its per-study trial number.</param>
    /// <param name="attrs">The attributes, written in enumeration order.</param>
    /// <inheritdoc />
    /// <remarks>
    ///     One lock, one updatable check and at most one copy of the record for the whole batch (plan 13, item S8): an
    ///     unknown or finished trial gets nothing written.
    /// </remarks>
    public override void SetTrialUserAttrs(int trialId, IReadOnlyDictionary<string, object?> attrs)
    {
        ArgumentNullException.ThrowIfNull(attrs);
        if (attrs.Count == 0)
            return;
        lock (_lock)
        {
            var trial = GetTrialUnlocked(trialId);
            CheckTrialIsUpdatable(trialId, trial.State);

            // One writable record for every entry: in place unless a reader may hold it (WritableTrialUnlocked).
            var userAttrs = WritableTrialUnlocked(trialId).UserAttrsDict;
            foreach (var (key, value) in attrs)
                userAttrs[key] = value;
        }
    }

    /// <param name="trialId">Storage identifier of the trial; distinct from its per-study trial number.</param>
    /// <param name="attrs">The attributes, written in enumeration order.</param>
    /// <inheritdoc />
    /// <remarks>One lock, one check and at most one copy, as <see cref="SetTrialUserAttrs" />.</remarks>
    public override void SetTrialSystemAttrs(int trialId, IReadOnlyDictionary<string, object?> attrs)
    {
        ArgumentNullException.ThrowIfNull(attrs);
        if (attrs.Count == 0)
            return;
        lock (_lock)
        {
            var trial = GetTrialUnlocked(trialId);
            CheckTrialIsUpdatable(trialId, trial.State);

            var systemAttrs = WritableTrialUnlocked(trialId).SystemAttrsDict;
            foreach (var (key, value) in attrs)
                systemAttrs[key] = value;
        }
    }

    /// <param name="trialId">Storage identifier of the trial; distinct from its per-study trial number.</param>
    /// <inheritdoc />
    public override FrozenTrial GetTrial(int trialId)
    {
        lock (_lock)
        {
            // The caller receives the stored record itself, so later writes must copy it first.
            return HandOut(GetTrialUnlocked(trialId));
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
            // An unfinished record may be written in place while the storage alone holds it (WritableTrialUnlocked),
            // so the view reads one under the lock and hands it out; finished records stay lock-free.
            return new TrialsPrefixView(trials, trials.Count, index =>
            {
                lock (_lock)
                    return HandOut(trials[index]);
            });
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
            {
                trials = trials.Select(t => t.DeepCopy()).ToList();
            }
            else
            {
                // The caller receives the stored records themselves. Finished ones are never written again; an
                // unfinished one must be copied before its next write (WritableTrialUnlocked). Only those are marked,
                // which keeps this pass to the few running and waiting trials' fields.
                foreach (var trial in trials)
                    if (!trial.State.IsFinished())
                        HandOut(trial);
            }

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
        /// <summary>
        ///     The study's one string instance per parameter name, used as the key in every trial's dictionaries so
        ///     names built per suggestion (<c>$"x{i}"</c>) do not keep a copy alive in each trial (plan 13, S7).
        /// </summary>
        internal Dictionary<string, string> ParamNames { get; } = new();
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

        /// <summary>
        ///     The study's parameter schemas for compact finished trials (plan 13, item S12); created by the first trial
        ///     that finishes in a storage built with <c>compactFinishedTrials: true</c>, so other storages never hold one.
        /// </summary>
        internal ParamSchemaCache? Schemas { get; set; }
    }
}

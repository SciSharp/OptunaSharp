using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using NumSharp;
using OptunaSharp.Collections;
using OptunaSharp.Distributions;
using OptunaSharp.Internal;
using OptunaSharp.Logging;
using OptunaSharp.Pruners;
using OptunaSharp.Samplers;
using OptunaSharp.Storages;

namespace OptunaSharp;

/// <summary>
///     A study corresponds to an optimization task, i.e., a set of trials.
///     Port of <c>optuna/study/study.py: Study</c>.
///
///     This object provides interfaces to run a new <see cref="Trial" />, access trials'
///     history, set/get user-defined attributes of the study itself.
///
///     Note that the direct use of this constructor is not recommended. To create and load a
///     study, please refer to <see cref="Optuna.CreateStudy" /> and <see cref="Optuna.LoadStudy" />
///     respectively.
/// </summary>
public class Study
{
    internal const string SystemAttrMetricNames = "study:metric_names";

    private static readonly Logger _logger = OptunaLogging.GetLogger("optuna.study.study");

    internal sealed class ThreadLocalStudyAttribute
    {
        internal bool InOptimizeLoop;
        internal List<FrozenTrial>? CachedAllTrials;
    }

    private readonly ThreadLocal<ThreadLocalStudyAttribute> _threadLocal =
        new(() => new ThreadLocalStudyAttribute());

    internal ThreadLocalStudyAttribute ThreadLocal => _threadLocal.Value!;

    internal volatile bool StopFlag;

    internal readonly int StudyId;
    internal readonly BaseStorage Storage;
    private readonly List<StudyDirection> _directions;

    /// <summary>The study name.</summary>
    public string StudyName { get; }

    /// <summary>The sampler in use.</summary>
    public BaseSampler Sampler { get; set; }

    /// <summary>The pruner in use.</summary>
    public BasePruner Pruner { get; set; }

    public Study(
        string studyName,
        BaseStorage storage,
        BaseSampler? sampler = null,
        BasePruner? pruner = null)
    {
        StudyName = studyName;
        int studyId = storage.GetStudyIdFromName(studyName);
        StudyId = studyId;
        Storage = storage;
        _directions = storage.GetStudyDirections(studyId);

        Sampler = sampler ?? new TPESampler();
        Pruner = pruner ?? new MedianPruner();

        StopFlag = false;
    }

    /// <summary>
    ///     Return parameters of the best trial in the study.
    ///     This feature can only be used for single-objective optimization.
    /// </summary>
    public IReadOnlyDictionary<string, object?> BestParams => BestTrial.Params;

    /// <summary>
    ///     Return the best objective value in the study.
    ///     This feature can only be used for single-objective optimization.
    /// </summary>
    public double BestValue
    {
        get
        {
            double? bestValue = BestTrial.Value;
            System.Diagnostics.Debug.Assert(bestValue is not null);
            return bestValue!.Value;
        }
    }

    /// <summary>
    ///     Return the best trial in the study.
    ///
    ///     This feature can only be used for single-objective optimization. If your study is
    ///     multi-objective, use <see cref="BestTrials" /> instead.
    ///
    ///     In constrained optimization, the best trial is selected from trials that satisfy all
    ///     constraints. A trial is considered feasible when all of its constraint values are less
    ///     than or equal to 0.0.
    /// </summary>
    public FrozenTrial BestTrial => GetBestTrialInternal(deepcopy: true);

    /// <summary>
    ///     Return trials located at the Pareto front in the study.
    ///
    ///     A trial is located at the Pareto front if there are no trials that dominate the trial.
    ///     It's called that a trial <c>t0</c> dominates another trial <c>t1</c> if
    ///     <c>all(v0 &lt;= v1)</c> and <c>any(v0 &lt; v1)</c> over their values.
    /// </summary>
    public List<FrozenTrial> BestTrials
    {
        get
        {
            // Check whether the study is constrained optimization.
            var trials = GetTrials(deepcopy: false);
            bool isConstrained = ConstrainedOptimization.IsConstrainedOptimization(trials);

            return MultiObjective.GetParetoFrontTrials(this, considerConstraint: isConstrained);
        }
    }

    /// <summary>
    ///     Return the direction of the study. This feature can only be used for single-objective
    ///     optimization. If your study is multi-objective, use <see cref="Directions" /> instead.
    /// </summary>
    public StudyDirection Direction
    {
        get
        {
            if (IsMultiObjective())
                throw new RuntimeError(
                    "A single direction cannot be retrieved from a multi-objective study. Consider " +
                    "using Study.directions to retrieve a list containing all directions.");

            return Directions[0];
        }
    }

    /// <summary>Return the directions of the study.</summary>
    public IReadOnlyList<StudyDirection> Directions => _directions;

    /// <summary>
    ///     Return all trials in the study, ordered by trial number.
    ///     This is a short form of <c>GetTrials(deepcopy: true, states: null)</c>.
    /// </summary>
    public List<FrozenTrial> Trials => GetTrials(deepcopy: true, states: null);

    /// <summary>
    ///     Return all trials in the study, ordered by trial number.
    /// </summary>
    /// <param name="deepcopy">
    ///     Flag to control whether to apply deep copy to the trials. Note that if you set the
    ///     flag to <c>false</c>, you shouldn't mutate any fields of the returned trials.
    /// </param>
    /// <param name="states">Trial states to filter on. If <c>null</c>, include all states.</param>
    public virtual List<FrozenTrial> GetTrials(bool deepcopy = true, IReadOnlyCollection<TrialState>? states = null)
        => GetTrialsInternal(deepcopy, states, useCache: false);

    /// <summary>Python <c>Study._get_trials</c>.</summary>
    internal List<FrozenTrial> GetTrialsInternal(
        bool deepcopy = true, IReadOnlyCollection<TrialState>? states = null, bool useCache = false)
    {
        if (useCache)
        {
            ThreadLocal.CachedAllTrials ??= Storage.GetAllTrials(StudyId, deepcopy: false);
            var trials = ThreadLocal.CachedAllTrials;
            var filteredTrials = states is not null
                ? trials.Where(t => states.Contains(t.State)).ToList()
                : trials;
            return deepcopy ? filteredTrials.Select(t => t.DeepCopy()).ToList() : filteredTrials;
        }

        return Storage.GetAllTrials(StudyId, deepcopy: deepcopy, states: states);
    }

    /// <summary>Python <c>Study._get_best_trial(deepcopy)</c>.</summary>
    internal FrozenTrial GetBestTrialInternal(bool deepcopy)
    {
        if (IsMultiObjective())
            throw new RuntimeError(
                "A single best trial cannot be retrieved from a multi-objective study. Consider " +
                "using Study.best_trials to retrieve a list containing the best trials.");

        var bestTrial = Storage.GetBestTrial(StudyId);

        // If the trial with the best value is infeasible, select the best trial from all feasible
        // trials.
        if (bestTrial.Constraints.Values.Any(x => x > 0.0))
        {
            var completeTrials = GetTrials(deepcopy: false, states: new[] { TrialState.Complete });
            var feasibleTrials = ConstrainedOptimization.GetFeasibleTrials(completeTrials);
            if (feasibleTrials.Count == 0)
                throw new ValueError("No feasible trials are completed yet.");
            bestTrial = Direction == StudyDirection.Maximize
                ? feasibleTrials.Aggregate((best, t) => t.Value! > best.Value! ? t : best)
                : feasibleTrials.Aggregate((best, t) => t.Value! < best.Value! ? t : best);
        }

        return deepcopy ? bestTrial.DeepCopy() : bestTrial;
    }

    /// <summary>Return user attributes of the study.</summary>
    public OrderedDict<string, object?> UserAttrs
    {
        get
        {
            var attrs = Storage.GetStudyUserAttrs(StudyId);
            var copy = new OrderedDict<string, object?>(attrs.Count);
            foreach (var (key, value) in attrs)
                copy[key] = PyJson.DeepCopy(value);
            return copy;
        }
    }

    /// <summary>Return system attributes of the study.</summary>
    [Obsolete("Deprecated in v3.1.0. This feature will be removed in v5.0.0.")]
    public OrderedDict<string, object?> SystemAttrs
    {
        get
        {
            var attrs = Storage.GetStudySystemAttrs(StudyId);
            var copy = new OrderedDict<string, object?>(attrs.Count);
            foreach (var (key, value) in attrs)
                copy[key] = PyJson.DeepCopy(value);
            return copy;
        }
    }

    /// <summary>
    ///     Return metric names. Use <see cref="SetMetricNames" /> to set the metric names first.
    /// </summary>
    public List<string>? MetricNames
        => Storage.GetStudySystemAttrs(StudyId).GetValueOrDefault(SystemAttrMetricNames) is List<object?> names
            ? names.Select(n => (string)n!).ToList()
            : null;

    // ------------------------------------------------------------------ optimize / ask / tell

    /// <summary>
    ///     Optimize an objective function.
    ///
    ///     Optimization is done by choosing a suitable set of hyperparameter values from a given
    ///     range. Uses a sampler which implements the task of value suggestion based on a
    ///     specified distribution. The sampler is specified in <see cref="Optuna.CreateStudy" />
    ///     and the default choice for the sampler is TPE.
    /// </summary>
    /// <param name="func">A callable that implements objective function.</param>
    /// <param name="nTrials">
    ///     The number of trials for each process. <c>null</c> represents no limit in terms of
    ///     the number of trials.
    /// </param>
    /// <param name="timeout">
    ///     Stop study after the given number of second(s). <c>null</c> represents no limit.
    /// </param>
    /// <param name="nJobs">The number of parallel jobs. If this argument is set to -1, the number is set to CPU count.</param>
    /// <param name="catch">
    ///     A study continues to run even when a trial raises one of the exceptions specified in
    ///     this argument. Default is an empty array, i.e. the study will stop for any exception
    ///     except for <see cref="TrialPruned" />.
    /// </param>
    /// <param name="callbacks">
    ///     List of callback functions that are invoked at the end of each trial, receiving the
    ///     <see cref="Study" /> and the resulting <see cref="FrozenTrial" />.
    /// </param>
    /// <param name="gcAfterTrial">
    ///     Flag to determine whether to automatically run garbage collection after each trial.
    /// </param>
    /// <param name="showProgressBar">Flag to show progress bars or not.</param>
    public void Optimize(
        Func<Trial, double> func,
        int? nTrials = null,
        double? timeout = null,
        int nJobs = 1,
        Type[]? @catch = null,
        IEnumerable<Action<Study, FrozenTrial>>? callbacks = null,
        bool gcAfterTrial = false,
        bool showProgressBar = false)
        => OptimizeLoop.Optimize(
            study: this,
            func: trial => new[] { func(trial) },
            nTrials: nTrials,
            timeout: timeout,
            nJobs: nJobs,
            catchExceptions: @catch,
            callbacks: callbacks?.ToList(),
            gcAfterTrial: gcAfterTrial,
            showProgressBar: showProgressBar);

    /// <summary>
    ///     Optimize a multi-objective function. See
    ///     <see cref="Optimize(Func{Trial, double}, int?, double?, int, Type[], IEnumerable{Action{Study, FrozenTrial}}, bool, bool)" />.
    /// </summary>
    public void Optimize(
        Func<Trial, IReadOnlyList<double>> func,
        int? nTrials = null,
        double? timeout = null,
        int nJobs = 1,
        Type[]? @catch = null,
        IEnumerable<Action<Study, FrozenTrial>>? callbacks = null,
        bool gcAfterTrial = false,
        bool showProgressBar = false)
        => OptimizeLoop.Optimize(
            study: this,
            func: func,
            nTrials: nTrials,
            timeout: timeout,
            nJobs: nJobs,
            catchExceptions: @catch,
            callbacks: callbacks?.ToList(),
            gcAfterTrial: gcAfterTrial,
            showProgressBar: showProgressBar);

    /// <summary>
    ///     Create a new trial from which hyperparameters can be suggested.
    ///
    ///     This method is part of an alternative to
    ///     <see cref="Optimize(Func{Trial, double}, int?, double?, int, Type[], IEnumerable{Action{Study, FrozenTrial}}, bool, bool)" />
    ///     that allows controlling the lifetime of a trial outside the scope of <c>func</c>. Each
    ///     call to this method should be followed by a call to <see cref="Tell(Trial, double, bool)" />
    ///     to finish the created trial.
    /// </summary>
    /// <param name="fixedDistributions">
    ///     A dictionary containing the parameter names and parameter's distributions. Each
    ///     parameter in this dictionary is automatically suggested for the returned trial, even
    ///     when the suggest method is not explicitly invoked by the user.
    /// </param>
    public Trial Ask(OrderedDict<string, BaseDistribution>? fixedDistributions = null)
    {
        if (!ThreadLocal.InOptimizeLoop && HeartbeatHelpers.IsHeartbeatEnabled(Storage))
            OptunaWarnings.Warn("Heartbeat of storage is supposed to be used with Study.optimize.");

        var converted = new OrderedDict<string, BaseDistribution>();
        if (fixedDistributions is not null)
        {
            foreach (var (key, dist) in fixedDistributions)
                converted[key] = Distribution.ConvertOldDistributionToNewDistribution(dist);
        }

        // Sync storage once every trial.
        ThreadLocal.CachedAllTrials = null;

        int? trialId = PopWaitingTrialId();
        trialId ??= Storage.CreateNewTrial(StudyId);
        var trial = new Trial(this, trialId.Value);

        foreach (var (name, param) in converted)
            trial.Suggest(name, param);

        return trial;
    }

    /// <summary>
    ///     Finish a trial created with <see cref="Ask" /> reporting a single objective value.
    /// </summary>
    /// <returns>A <see cref="FrozenTrial" /> representing the resulting trial (deep-copied).</returns>
    public FrozenTrial Tell(Trial trial, double value, bool skipIfFinished = false)
        => TellInternal(trial, new[] { value }, null, skipIfFinished);

    /// <summary>
    ///     Finish a trial created with <see cref="Ask" />.
    /// </summary>
    /// <param name="trial">A <see cref="Trial" /> object.</param>
    /// <param name="values">
    ///     Optional objective value or a sequence of such values in case the study is used for
    ///     multi-objective optimization. Argument must be provided if <paramref name="state" />
    ///     is COMPLETE and should be <c>null</c> if it is FAIL or PRUNED.
    /// </param>
    /// <param name="state">
    ///     State to be reported. Must be <c>null</c>, COMPLETE, FAIL or PRUNED. If <c>null</c>,
    ///     it will be updated to COMPLETE or FAIL depending on whether validation for
    ///     <paramref name="values" /> reported succeed or not.
    /// </param>
    /// <param name="skipIfFinished">
    ///     Flag to control whether exception should be raised when values for already finished
    ///     trial are told. If <c>true</c>, tell is skipped without any error when the trial is
    ///     already finished.
    /// </param>
    /// <returns>A <see cref="FrozenTrial" /> representing the resulting trial (deep-copied).</returns>
    public FrozenTrial Tell(
        Trial trial, IReadOnlyList<double>? values = null, TrialState? state = null, bool skipIfFinished = false)
        => TellInternal(trial, values, state, skipIfFinished);

    /// <summary>Finish a trial identified by its number, reporting a single objective value.</summary>
    public FrozenTrial Tell(int trialNumber, double value, bool skipIfFinished = false)
        => TellInternal(trialNumber, new[] { value }, null, skipIfFinished);

    /// <summary>Finish a trial identified by its number.</summary>
    public FrozenTrial Tell(
        int trialNumber, IReadOnlyList<double>? values = null, TrialState? state = null, bool skipIfFinished = false)
        => TellInternal(trialNumber, values, state, skipIfFinished);

    private FrozenTrial TellInternal(Trial trial, IReadOnlyList<double>? values, TrialState? state, bool skipIfFinished)
    {
        TellHelpers.TellWithWarning(this, TellHelpers.GetFrozenTrial(this, trial), values, state, skipIfFinished);
        return TellHelpers.GetFrozenTrial(this, trial).DeepCopy();
    }

    private FrozenTrial TellInternal(int trialNumber, IReadOnlyList<double>? values, TrialState? state, bool skipIfFinished)
    {
        TellHelpers.TellWithWarning(this, TellHelpers.GetFrozenTrial(this, trialNumber), values, state, skipIfFinished);
        return TellHelpers.GetFrozenTrial(this, trialNumber).DeepCopy();
    }

    // ------------------------------------------------------------------ attributes

    /// <summary>
    ///     Export trials as a <see cref="TrialsTable" /> (the Core-tier analog of
    ///     <c>Study.trials_dataframe</c>; the pandas-equivalent adapter is a Tier X package).
    ///     If there are no trials, an empty table is returned.
    /// </summary>
    /// <param name="attrs">Field names of <see cref="FrozenTrial" /> to include as columns.</param>
    /// <param name="multiIndex">
    ///     Whether the returned table keeps two-level column labels; when <c>false</c>,
    ///     hierarchical columns such as <c>(params, x)</c> are flattened to <c>params_x</c>.
    /// </param>
    public TrialsTable TrialsDataframe(string[]? attrs = null, bool multiIndex = false)
        => OptunaSharp.TrialsDataframe.Build(this, attrs ?? OptunaSharp.TrialsDataframe.DefaultAttrs, multiIndex);

    /// <summary>Set a user attribute to the study.</summary>
    /// <param name="key">A key string of the attribute.</param>
    /// <param name="value">A value of the attribute. The value should be JSON serializable.</param>
    public void SetUserAttr(string key, object? value) => Storage.SetStudyUserAttr(StudyId, key, value);

    /// <summary>
    ///     Set a system attribute to the study. Note that Optuna internally uses this method to
    ///     save system messages.
    /// </summary>
    [Obsolete("Deprecated in v3.1.0. This feature will be removed in v5.0.0.")]
    public void SetSystemAttr(string key, object? value) => Storage.SetStudySystemAttr(StudyId, key, value);

    /// <summary>
    ///     Exit from the current optimization loop after the running trials finish.
    ///
    ///     This method lets the running
    ///     <see cref="Optimize(Func{Trial, double}, int?, double?, int, Type[], IEnumerable{Action{Study, FrozenTrial}}, bool, bool)" />
    ///     method return immediately after all trials which the method spawned finishes. This
    ///     method only works when it is called inside an objective function or callback.
    /// </summary>
    public virtual void Stop()
    {
        if (!ThreadLocal.InOptimizeLoop)
            throw new RuntimeError(
                "`Study.stop` is supposed to be invoked inside an objective function or a callback.");

        StopFlag = true;
    }

    /// <summary>
    ///     Enqueue a trial with given parameter values. You can fix the next sampling parameters
    ///     which will be evaluated in your objective function.
    /// </summary>
    /// <param name="params">Parameter values to pass your objective function.</param>
    /// <param name="userAttrs">A dictionary of user-specific attributes other than <c>params</c>.</param>
    /// <param name="skipIfExists">When <c>true</c>, prevents duplicate trials from being enqueued again.</param>
    public void EnqueueTrial(
        OrderedDict<string, object?> @params,
        OrderedDict<string, object?>? userAttrs = null,
        bool skipIfExists = false)
    {
        if (skipIfExists && ShouldSkipEnqueue(@params))
        {
            _logger.Info($"Trial with params {PythonRepr.DictRepr(@params)} already exists. Skipping enqueue.");
            return;
        }

        AddTrial(FrozenTrial.CreateTrial(
            state: TrialState.Waiting,
            systemAttrs: new OrderedDict<string, object?> { ["fixed_params"] = @params },
            userAttrs: userAttrs));
    }

    /// <summary>
    ///     Add trial to study. The trial is validated before being added.
    ///
    ///     This method should in general be used to add already evaluated trials
    ///     (<c>trial.State.IsFinished() == true</c>). To queue trials for evaluation, please
    ///     refer to <see cref="EnqueueTrial" />.
    /// </summary>
    public void AddTrial(FrozenTrial trial)
    {
        trial.Validate();

        if (trial.Values is not null && Directions.Count != trial.Values.Count)
            throw new ValueError(
                $"The added trial has {trial.Values.Count} values, which is different from the " +
                $"number of objectives {Directions.Count} in the study (determined by " +
                "Study.directions).");

        Storage.CreateNewTrial(StudyId, templateTrial: trial);
    }

    /// <summary>Add trials to study. The trials are validated before being added.</summary>
    public void AddTrials(IEnumerable<FrozenTrial> trials)
    {
        foreach (var trial in trials)
            AddTrial(trial);
    }

    /// <summary>
    ///     Set metric names. This method names each dimension of the returned values of the
    ///     objective function. It is particularly useful in multi-objective optimization.
    ///     (Experimental since Optuna 3.2.0.)
    /// </summary>
    public void SetMetricNames(IReadOnlyList<string> metricNames)
    {
        if (Directions.Count != metricNames.Count)
            throw new ValueError("The number of objectives must match the length of the metric names.");

        Storage.SetStudySystemAttr(
            StudyId, SystemAttrMetricNames, metricNames.Cast<object?>().ToList());
    }

    /// <summary>Return <c>true</c> if the study has multiple objectives.</summary>
    internal bool IsMultiObjective() => Directions.Count > 1;

    /// <summary>Python <c>Study._pop_waiting_trial_id</c>.</summary>
    internal int? PopWaitingTrialId()
    {
        foreach (var trial in Storage.GetAllTrials(
            StudyId, deepcopy: false, states: new[] { TrialState.Waiting }))
        {
            // Attempt to set the state to RUNNING.
            // - If another process or thread has already changed the state to RUNNING,
            //   SetTrialStateValues returns false.
            // - If another process or thread has already finished the trial,
            //   an UpdateFinishedTrialError is raised.
            try
            {
                if (!Storage.SetTrialStateValues(trial.TrialId, state: TrialState.Running))
                    continue;
            }
            catch (UpdateFinishedTrialError)
            {
                continue;
            }

            _logger.Debug($"Trial {trial.Number} popped from the trial queue.");
            return trial.TrialId;
        }

        return null;
    }

    /// <summary>Python <c>Study._should_skip_enqueue</c>.</summary>
    internal bool ShouldSkipEnqueue(OrderedDict<string, object?> @params)
    {
        foreach (var trial in GetTrials(deepcopy: false))
        {
            IReadOnlyDictionary<string, object?> trialParams =
                trial.SystemAttrsDict.GetValueOrDefault("fixed_params") as OrderedDict<string, object?>
                ?? (IReadOnlyDictionary<string, object?>)trial.ParamsDict;

            if (!new HashSet<string>(trialParams.Keys).SetEquals(@params.Keys))
            {
                // Can't have repeated trials if different params are suggested.
                continue;
            }

            var repeatedParams = new List<bool>();
            foreach (var (paramName, paramValue) in @params)
            {
                var existingParam = trialParams[paramName];
                if (paramValue is null != existingParam is null
                    || (paramValue is not null && existingParam is not null
                        && !AreSameParamKind(paramValue, existingParam)))
                {
                    // Enqueued param has distribution that does not match existing param
                    // (e.g. trying to enqueue categorical to float param).
                    repeatedParams.Add(false);
                    continue;
                }

                bool isRepeated;
                if (PyFloat.TryConvert(paramValue, out double newNumeric)
                    && paramValue is not string && paramValue is not null)
                {
                    PyFloat.TryConvert(existingParam, out double existingNumeric);
                    // np.isclose(..., atol=0.0, equal_nan=True).
                    isRepeated = double.IsNaN(newNumeric) && double.IsNaN(existingNumeric)
                        || Math.Abs(newNumeric - existingNumeric) <= 1e-08 * Math.Abs(existingNumeric);
                }
                else
                {
                    isRepeated = PyJson.ValueEquals(paramValue, existingParam);
                }
                repeatedParams.Add(isRepeated);
            }

            if (repeatedParams.All(x => x))
                return true;
        }

        return false;
    }

    private static bool AreSameParamKind(object a, object b)
    {
        // Python: isinstance(param_value, type(existing_param)) — bool/int/float/str kinds.
        static int Kind(object v) => v switch
        {
            bool => 0,
            sbyte or byte or short or ushort or int or uint or long or ulong => 1,
            float or double or decimal => 2,
            string => 3,
            _ => 4,
        };
        int kindA = Kind(a);
        int kindB = Kind(b);
        // Python's bool is an int subclass: isinstance(True, int) is True, so an int enqueued
        // over a bool matches only one way. Match on numeric compatibility like isinstance does.
        if (kindA == kindB)
            return true;
        return kindA == 0 && kindB == 1;  // isinstance(bool_value, int_type).
    }

    /// <summary>Python <c>Study._log_completed_trial</c>.</summary>
    internal void LogCompletedTrial(List<double> values, int number, OrderedDict<string, object?> @params)
    {
        if (!_logger.IsEnabledFor(LogLevel.Info))
            return;

        var metricNames = MetricNames;

        if (values.Count > 1)
        {
            string trialValues;
            if (metricNames is null)
            {
                trialValues = PythonRepr.ListRepr(values);
            }
            else
            {
                var named = new OrderedDict<string, object?>();
                for (int i = 0; i < values.Count && i < metricNames.Count; i++)
                    named[metricNames[i]] = values[i];
                trialValues = PythonRepr.DictRepr(named);
            }
            _logger.Info(
                $"Trial {number} finished with values: {trialValues} and parameters: {PythonRepr.DictRepr(@params)}.");
        }
        else if (values.Count == 1)
        {
            string trialValue;
            if (metricNames is null)
                trialValue = PythonRepr.FloatRepr(values[0]);
            else
                trialValue = PythonRepr.DictRepr(new OrderedDict<string, object?> { [metricNames[0]] = values[0] });

            string message =
                $"Trial {number} finished with value: {trialValue} and parameters: {PythonRepr.DictRepr(@params)}.";
            try
            {
                var bestTrial = GetBestTrialInternal(deepcopy: false);
                message += $" Best is trial {bestTrial.Number} with value: {PythonRepr.FloatRepr(bestTrial.Value ?? double.NaN)}.";
            }
            catch (ValueError)
            {
                // If no feasible trials are completed yet, study.best_trial raises ValueError.
            }
            _logger.Info(message);
        }
    }
}

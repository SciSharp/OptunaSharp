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
/// <seealso href="https://optuna.readthedocs.io/en/stable/reference/generated/optuna.study.Study.html">Official Optuna study contract.</seealso>
public class Study
{
    /// <summary>Reserved system-attribute key holding objective display names.</summary>
    internal const string SystemAttrMetricNames = "study:metric_names";

    private static readonly Logger _logger = OptunaLogging.GetLogger("optuna.study.study");

    /// <summary>Per-thread optimization-loop state and cached trial history.</summary>
    internal sealed class ThreadLocalStudyAttribute
    {
        /// <summary>Whether this thread is currently executing an objective or its callbacks.</summary>
        internal bool InOptimizeLoop;

        private List<FrozenTrial>? _cachedAllTrials;

        /// <summary>Trial-history snapshot reused within one trial; cleared when the next trial is asked.</summary>
        /// <remarks>
        ///     Setting it, to a new snapshot or to null, also drops <see cref="CachedFiltered" />: that list was filtered
        ///     from the snapshot being replaced, and a reset site (a tell, the end of an optimize worker) must not leave a
        ///     reused thread holding a finished study's trials through it either.
        /// </remarks>
        internal List<FrozenTrial>? CachedAllTrials
        {
            get => _cachedAllTrials;
            set
            {
                _cachedAllTrials = value;
                CachedFiltered = null;
                CachedFilteredSource = null;
            }
        }

        /// <summary>
        ///     The last list <see cref="GetTrialsCachedReadOnly" /> filtered, shared by every read-only read of the same
        ///     states until the snapshot changes (plan 15).
        /// </summary>
        internal List<FrozenTrial>? CachedFiltered;

        /// <summary>The snapshot <see cref="CachedFiltered" /> was filtered from: the memo answers only while it is still <see cref="CachedAllTrials" />.</summary>
        internal List<FrozenTrial>? CachedFilteredSource;

        /// <summary>The state mask <see cref="CachedFiltered" /> was filtered with: bit <c>s</c> set for each kept state <c>s</c>.</summary>
        internal int CachedFilteredMask;

        /// <summary>
        ///     The worker's own sampler instance while this thread runs a trial of an <c>Optimize(nJobs)</c> call with
        ///     <see cref="SamplerConcurrency.PerWorker" />; null otherwise, so <see cref="Sampler" /> serves. Set and
        ///     restored by the worker around each trial (THREADING.md §4), since dedicated and pool threads are reused.
        /// </summary>
        internal Samplers.BaseSampler? SamplerOverride;
    }

    private readonly ThreadLocal<ThreadLocalStudyAttribute> _threadLocal =
        new(() => new ThreadLocalStudyAttribute());

    /// <summary>Gets this thread's optimization state and trial cache.</summary>
    internal ThreadLocalStudyAttribute ThreadLocal => _threadLocal.Value!;

    /// <summary>Shared request to stop scheduling trials after active trials finish.</summary>
    internal volatile bool StopFlag;

    /// <summary>Storage identifier of the study.</summary>
    internal readonly int StudyId;
    /// <summary>Backend containing this study's persistent or in-memory records.</summary>
    internal readonly BaseStorage Storage;
    private readonly List<StudyDirection> _directions;

    /// <summary>The study name.</summary>
    public string StudyName { get; }

    /// <summary>The sampler in use.</summary>
    public BaseSampler Sampler { get; set; }

    /// <summary>
    ///     The sampler that serves a trial started on the calling thread: the worker's own instance during a trial of an
    ///     <c>Optimize(nJobs)</c> call with <see cref="SamplerConcurrency.PerWorker" />, otherwise <see cref="Sampler" />.
    ///     Read once per sampler call, like <see cref="Sampler" />, so the monitor taken belongs to the instance that runs.
    /// </summary>
    internal BaseSampler SamplerForCurrentThread => ThreadLocal.SamplerOverride ?? Sampler;

    /// <summary>The pruner in use.</summary>
    public BasePruner Pruner { get; set; }

    private TrialExecutorOptions? _executorOptions;

    /// <summary>
    ///     The executor options this study's <c>Optimize(nJobs: …)</c> calls use when a call passes none: which
    ///     threads run the trials. <see langword="null" /> (the default) defers to
    ///     <see cref="TrialExecutorOptions.Default" />, read at the start of each call. Optuna has no counterpart.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///         The second of the executor's levels: a call's <c>executorOptions</c> argument overrides it, and it
    ///         overrides <see cref="TrialExecutorOptions.Default" />. The DI study factory sets it from
    ///         <see cref="Extensions.OptunaStudyOptions.DefaultExecutorOptions" /> when it builds the study, so a
    ///         value set here afterwards replaces that DI default.
    ///     </para>
    ///     <para>
    ///         It belongs to this <see cref="Study" /> object and is not stored: a study opened again with
    ///         <see cref="Optuna.LoadStudy" /> starts with <see langword="null" />. Two threads running <c>Optimize</c>
    ///         on one study object share it; each call takes its own snapshot when it starts.
    ///     </para>
    /// </remarks>
    public TrialExecutorOptions? ExecutorOptions
    {
        get => Volatile.Read(ref _executorOptions);
        set => Volatile.Write(ref _executorOptions, value);
    }

    /// <summary>Opens an existing named study using a shared storage backend.</summary>
    /// <param name="studyName">Name identifying the study in the selected storage.</param>
    /// <param name="storage">
    ///     Storage backend holding study and trial records. An RDB backend (any
    ///     <see cref="ICachedStorageBackend" />, such as the SQLite or EF Core <c>RDBStorage</c>) is wrapped in
    ///     a new <see cref="CachedStorage" />; every other storage, including one that is already a
    ///     <see cref="CachedStorage" />, is used as given.
    /// </param>
    /// <param name="sampler">Sampler for parameter suggestions; null selects TPE.</param>
    /// <param name="pruner">Pruning strategy; null selects the median pruner.</param>
    /// <exception cref="NumSharp.KeyError">The named study does not exist in storage.</exception>
    /// <remarks>
    ///     Python's <c>Study.__init__</c> passes its storage through <c>storages.get_storage</c>, and so does
    ///     this constructor. Without the wrap, a study opened by <see cref="Optuna.LoadStudy" />, by the CLI
    ///     or by calling this constructor directly would read the whole study from the database each time a
    ///     sampler asks for the trial history. <see cref="Optuna.CreateStudy" /> has already wrapped its
    ///     storage, which then passes through unchanged. Each constructed study wraps separately, as each
    ///     Python <c>load_study</c> call does: two studies opened over one <c>RDBStorage</c> have two caches.
    /// </remarks>
    public Study(
        string studyName,
        BaseStorage storage,
        BaseSampler? sampler = null,
        BasePruner? pruner = null)
    {
        StudyName = studyName;
        // Python's Study.__init__ runs get_storage here, so a raw RDB backend gets the trial cache no matter
        // which entry point built the study (see remarks).
        storage = StorageFunctions.GetStorage(storage);
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
    /// <exception cref="NumSharp.RuntimeError">The study has multiple objectives.</exception>
    /// <exception cref="NumSharp.ValueError">No completed trial, or no feasible completed trial, is available.</exception>
    /// <exception cref="NumSharp.KeyError">The study has been removed from storage.</exception>
    public IReadOnlyDictionary<string, object?> BestParams => BestTrial.Params;

    /// <summary>
    ///     Return the best objective value in the study.
    ///     This feature can only be used for single-objective optimization.
    /// </summary>
    /// <exception cref="NumSharp.RuntimeError">The study has multiple objectives.</exception>
    /// <exception cref="NumSharp.ValueError">No completed trial, or no feasible completed trial, is available.</exception>
    /// <exception cref="NumSharp.KeyError">The study has been removed from storage.</exception>
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
    /// <exception cref="NumSharp.RuntimeError">The study has multiple objectives.</exception>
    /// <exception cref="NumSharp.ValueError">No completed trial, or no feasible completed trial, is available.</exception>
    /// <exception cref="NumSharp.KeyError">The study has been removed from storage.</exception>
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
    /// <exception cref="NumSharp.RuntimeError">The study has multiple objectives.</exception>
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
    /// <returns>Trials in ascending trial-number order, optionally filtered and copied.</returns>
    /// <exception cref="NumSharp.KeyError">The study no longer exists in storage.</exception>
    public virtual List<FrozenTrial> GetTrials(bool deepcopy = true, IReadOnlyCollection<TrialState>? states = null)
        => GetTrialsInternal(deepcopy, states, useCache: false);

    /// <summary>Python <c>Study._get_trials</c>.</summary>
    /// <param name="deepcopy">Whether to copy trial records; treat records returned with false as read-only.</param>
    /// <param name="states">States to include; null includes every state.</param>
    /// <param name="useCache">Whether to reuse the current thread's trial-history snapshot until the next Ask call.</param>
    /// <returns>Trials in ascending trial-number order, optionally filtered and copied.</returns>
    /// <exception cref="NumSharp.KeyError">The study no longer exists in storage.</exception>
    internal List<FrozenTrial> GetTrialsInternal(
        bool deepcopy = true, IReadOnlyCollection<TrialState>? states = null, bool useCache = false)
    {
        if (useCache)
        {
            // A sampler may request history repeatedly while suggesting one trial. Reuse a coherent
            // per-thread snapshot until Ask invalidates it for the next trial.
            ThreadLocal.CachedAllTrials ??= Storage.GetAllTrials(StudyId, deepcopy: false);
            var trials = ThreadLocal.CachedAllTrials;
            var filteredTrials = states is not null
                ? FilterByState(trials, states)
                : trials;
            return deepcopy ? filteredTrials.Select(t => t.DeepCopy()).ToList() : filteredTrials;
        }

        return Storage.GetAllTrials(StudyId, deepcopy: deepcopy, states: states);
    }

    /// <summary>
    ///     <c>GetTrialsInternal(deepcopy: false, states, useCache: true)</c> for a caller that only reads the result: the
    ///     same records in the same order, but one list shared by every such read of the same states until the trial
    ///     snapshot changes, instead of a new filtered copy per read.
    /// </summary>
    /// <param name="states">The states to keep; the list is shared only between reads of the same set of states.</param>
    /// <returns>
    ///     Trials in ascending trial-number order. <b>Never change it</b>: a later read of the same states in the same trial
    ///     returns this very list, so a change would leak into every one of them (the type says read-only for that reason;
    ///     a caller that needs to change the history copies it, or calls <see cref="GetTrialsInternal" />).
    /// </returns>
    /// <exception cref="ArgumentNullException"><paramref name="states" /> is null.</exception>
    /// <exception cref="NumSharp.KeyError">The study no longer exists in storage (on the read that takes the snapshot).</exception>
    /// <remarks>
    ///     <para>
    ///         <b>Why.</b> TPE reads the filtered history twice for each sampled trial in multivariate mode, and twice for
    ///         <i>each parameter</i> in independent mode (MOTPE's default): each read copied the whole history, growing the
    ///         list by doubling. Measured before plan 15 those copies were 12 % of a 1,000-trial TPE study's allocations.
    ///     </para>
    ///     <para>
    ///         <b>Why it is the same history.</b> The memo is keyed by the snapshot list's identity and the state set, and
    ///         the snapshot is the one <see cref="GetTrialsInternal" /> filters too, so both filter the same records with
    ///         the same test. A record's state cannot change under the memo: the storages hand out records they never
    ///         change afterwards (the in-memory and journal storages change a shallow copy and store that, plan 13 item S4),
    ///         and the snapshot itself is replaced, which drops the memo, by the tell that ends the trial.
    ///     </para>
    /// </remarks>
    internal IReadOnlyList<FrozenTrial> GetTrialsCachedReadOnly(TrialState[] states)
    {
        ArgumentNullException.ThrowIfNull(states);
        ThreadLocalStudyAttribute local = ThreadLocal;
        local.CachedAllTrials ??= Storage.GetAllTrials(StudyId, deepcopy: false);
        List<FrozenTrial> all = local.CachedAllTrials;

        int mask = StateMask(states);
        if (mask >= 0 && local.CachedFiltered is { } memo && ReferenceEquals(local.CachedFilteredSource, all)
            && local.CachedFilteredMask == mask)
            return memo;

        List<FrozenTrial> filtered = FilterByState(all, states);
        // A set no mask can hold (a value outside the enum's small range) is filtered every time and never memoized, so
        // the memo's key always describes its list exactly.
        if (mask >= 0)
        {
            local.CachedFiltered = filtered;
            local.CachedFilteredSource = all;
            local.CachedFilteredMask = mask;
        }
        return filtered;
    }

    /// <summary>
    ///     The trials of <paramref name="trials" /> whose state is in <paramref name="states" />, in order: what
    ///     <c>trials.Where(t =&gt; states.Contains(t.State)).ToList()</c> returns, in a list allocated once at its final size.
    /// </summary>
    /// <param name="trials">The trials to filter; not changed.</param>
    /// <param name="states">The states to keep; duplicates are harmless and an empty set keeps nothing.</param>
    /// <returns>A new list the caller may change, holding the kept trials as the same references, in their order.</returns>
    /// <remarks>
    ///     <para>
    ///         The membership test is a bit mask (<see cref="StateMask" />): enum equality is equality of the underlying
    ///         values, so for states in 0–30 testing bit <c>state</c> answers exactly what <c>Contains</c> answers. A set
    ///         holding any other value takes the LINQ expression itself. A trial whose own state lies outside 0–30 is never
    ///         in a mask, as it is never in such a set.
    ///     </para>
    ///     <para>
    ///         Two passes, count then fill, instead of a list grown by doubling: the doubling left about twice the final
    ///         array in garbage per call, while a single list sized for every trial would waste the whole history on a
    ///         narrow filter (QMC's read of the running trials).
    ///     </para>
    /// </remarks>
    internal static List<FrozenTrial> FilterByState(List<FrozenTrial> trials, IReadOnlyCollection<TrialState> states)
    {
        int mask = StateMask(states);
        if (mask < 0)
            return trials.Where(t => states.Contains(t.State)).ToList();

        int count = 0;
        foreach (FrozenTrial t in trials)
        {
            if (InStateMask(t.State, mask))
                count++;
        }

        var result = new List<FrozenTrial>(count);
        foreach (FrozenTrial t in trials)
        {
            if (InStateMask(t.State, mask))
                result.Add(t);
        }
        return result;
    }

    /// <summary>One bit per state of <paramref name="states" /> (bit <c>s</c> for state <c>s</c>), or -1 when a state lies outside 0–30.</summary>
    /// <param name="states">The states to encode; enumerated once.</param>
    /// <returns>The mask (never negative, and 0 for an empty set), or -1 when the set holds a state no mask can encode.</returns>
    /// <remarks>Bit 31 would make the mask negative and collide with the -1 refusal, so state 31 is refused too: the enum's real values are 0–4.</remarks>
    private static int StateMask(IReadOnlyCollection<TrialState> states)
    {
        int mask = 0;
        if (states is TrialState[] array)
        {
            // An array is the common case (every sampler passes one); indexing it avoids boxing an enumerator per read.
            for (int i = 0; i < array.Length; i++)
            {
                if ((uint)array[i] >= 31)
                    return -1;
                mask |= 1 << (int)array[i];
            }
            return mask;
        }
        foreach (TrialState state in states)
        {
            if ((uint)state >= 31)
                return -1;
            mask |= 1 << (int)state;
        }
        return mask;
    }

    /// <summary>Whether <paramref name="state" /> has its bit set in <paramref name="mask" />.</summary>
    /// <param name="state">The trial's state; a value outside 0–30 is in no mask.</param>
    /// <param name="mask">A mask from <see cref="StateMask" />.</param>
    /// <returns>True when the state is one of the mask's states.</returns>
    private static bool InStateMask(TrialState state, int mask)
        => (uint)state < 31 && ((mask >> (int)state) & 1) != 0;

    /// <summary>
    ///     This study's trials as a zero-copy, fixed-membership window (see
    ///     <see cref="BaseStorage.GetTrialsPrefixView" />) — the O(1) read GA samplers use on every ask where a
    ///     copied list would make the loop quadratic in trial count. Treat the result as read-only; reads see
    ///     each trial's current snapshot (fresher than a copied list under copy-on-write storages).
    /// </summary>
    /// <returns>Trials in ascending trial-number order, membership frozen at the call.</returns>
    /// <exception cref="NumSharp.KeyError">The study no longer exists in storage.</exception>
    internal IReadOnlyList<FrozenTrial> GetTrialsView() => Storage.GetTrialsPrefixView(StudyId);

    /// <summary>
    ///     Every trial of this study, for a sampler that reads the history on every ask: the zero-copy view where the
    ///     storage has one, otherwise exactly the read Optuna's samplers make
    ///     (<c>study._get_trials(deepcopy=False, use_cache=...)</c>).
    /// </summary>
    /// <param name="useCache">
    ///     For storages without a zero-copy view: whether to share the per-thread snapshot that lives until the next
    ///     ask (Optuna's <c>use_cache</c>). Ignored where the view is used, since the view costs nothing to take again.
    /// </param>
    /// <returns>Trials in ascending trial-number order; read-only.</returns>
    /// <exception cref="NumSharp.KeyError">The study no longer exists in storage.</exception>
    /// <remarks>
    ///     <para>
    ///         Same trials, same order: the two reads differ only in how fresh each snapshot is. The view returns each
    ///         trial's current snapshot, a copied list the one current when it was taken. In a run with one trial at
    ///         a time nothing changes in between, so seeded results are identical (Rustuna plan 13, phase 1).
    ///     </para>
    ///     <para>
    ///         The fallback keeps database-backed storages on their existing read, so switching a sampler to this
    ///         method never adds a database round trip (<see cref="BaseStorage.HasZeroCopyTrialsView" />).
    ///     </para>
    /// </remarks>
    internal IReadOnlyList<FrozenTrial> GetTrialsForSampler(bool useCache)
        => Storage.HasZeroCopyTrialsView
            ? Storage.GetTrialsPrefixView(StudyId)
            : GetTrialsInternal(deepcopy: false, states: null, useCache: useCache);

    /// <summary>Python <c>Study._get_best_trial(deepcopy)</c>.</summary>
    /// <param name="deepcopy">Whether to copy trial records; treat records returned with false as read-only.</param>
    /// <returns>The best feasible completed single-objective trial, copied when requested.</returns>
    /// <exception cref="NumSharp.RuntimeError">The study has multiple objectives.</exception>
    /// <exception cref="NumSharp.ValueError">No completed trial, or no feasible completed trial, is available.</exception>
    /// <exception cref="NumSharp.KeyError">The study has been removed from storage.</exception>
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

    // Study.system_attrs (deprecated 3.1.0) was removed in Optuna 5.0.0 (#6834). Study-level system
    // attributes are optuna-internal; read them through Storage.GetStudySystemAttrs(StudyId).

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
    /// <param name="executorOptions">
    ///     Which threads run the trials when <paramref name="nJobs" /> is not 1. <see langword="null" /> (the default)
    ///     uses <see cref="ExecutorOptions" />, then <see cref="TrialExecutorOptions.Default" />, which start as
    ///     OptunaSharp's reused dedicated threads. Read once, when the call starts. Not part of Optuna's API.
    /// </param>
    /// <param name="ct">
    ///     Cancels the run (OptunaSharp only; Python stops on <c>KeyboardInterrupt</c>). Once it is cancelled no trial
    ///     starts, the trials in flight finish and are stored, and this method throws
    ///     <see cref="OperationCanceledException" /> carrying it. A running objective may stop early through
    ///     <see cref="Trial.CancellationToken" />: its trial is then stored FAIL, whatever <paramref name="catch" />
    ///     holds. An already cancelled token runs no trial. <see cref="StopSignals" /> turns Ctrl+C and SIGTERM into
    ///     such a token; a .NET Generic Host passes <c>IHostApplicationLifetime.ApplicationStopping</c>.
    /// </param>
    /// <exception cref="NumSharp.RuntimeError">Optimization is invoked recursively on the same thread.</exception>
    /// <exception cref="NumSharp.ValueError"><paramref name="nJobs" /> is 0 or below -1, as Python's <c>ThreadPoolExecutor</c> refuses.</exception>
    /// <exception cref="ArgumentOutOfRangeException">The executor options in effect return an undefined executor kind or an invalid idle timeout; only a derived <see cref="TrialExecutorOptions" /> class can, because its setters validate.</exception>
    /// <exception cref="OperationCanceledException">
    ///     <paramref name="ct" /> was cancelled before the scheduling ended, or an objective threw this exception while it
    ///     was cancelled. Thrown after every trial in flight is stored; <see cref="OperationCanceledException.CancellationToken" />
    ///     is <paramref name="ct" />.
    /// </exception>
    /// <exception cref="System.Exception">An objective raises an exception outside catch, or a callback, sampler, pruner, or storage operation fails; the original failure propagates.</exception>
    /// <remarks>Running objectives are allowed to finish when the count, timeout, Stop request, or cancellation ends scheduling. Callback failures are not covered by the objective catch list.</remarks>
    public void Optimize(
        Func<Trial, double> func,
        int? nTrials = null,
        double? timeout = null,
        int nJobs = 1,
        Type[]? @catch = null,
        IEnumerable<Action<Study, FrozenTrial>>? callbacks = null,
        bool gcAfterTrial = false,
        bool showProgressBar = false,
        TrialExecutorOptions? executorOptions = null,
        CancellationToken ct = default)
        => OptimizeLoop.Optimize(
            study: this,
            func: trial => new[] { func(trial) },
            nTrials: nTrials,
            timeout: timeout,
            nJobs: nJobs,
            catchExceptions: @catch,
            callbacks: callbacks?.ToList(),
            gcAfterTrial: gcAfterTrial,
            showProgressBar: showProgressBar,
            executorOptions: executorOptions,
            ct: ct);

    /// <summary>
    ///     Optimize a multi-objective function. See
    ///     <see cref="Optimize(Func{Trial, double}, int?, double?, int, Type[], IEnumerable{Action{Study, FrozenTrial}}, bool, bool, TrialExecutorOptions, CancellationToken)" />.
    /// </summary>
    /// <param name="nTrials">Maximum trials started by this invocation; null leaves the count unlimited.</param>
    /// <param name="func">Objective delegate returning a scalar or values in study direction order.</param>
    /// <param name="timeout">Maximum scheduling duration in seconds; null leaves elapsed time unlimited.</param>
    /// <param name="nJobs">Concurrent trial workers; -1 uses the processor count.</param>
    /// <param name="catch">Objective exception types to mark failed and continue; other objective exceptions stop optimization.</param>
    /// <param name="callbacks">Callbacks invoked after a trial finishes, receiving the study and finished trial.</param>
    /// <param name="gcAfterTrial">Whether to request full garbage collection after each trial.</param>
    /// <param name="showProgressBar">Whether to show supported trial-count or timeout progress.</param>
    /// <param name="executorOptions">
    ///     Which threads run the trials when <paramref name="nJobs" /> is not 1; <see langword="null" /> uses
    ///     <see cref="ExecutorOptions" />, then <see cref="TrialExecutorOptions.Default" />. Read once, when the call starts.
    /// </param>
    /// <param name="ct">
    ///     Cancels the run: no trial starts once it is cancelled, the trials in flight finish, then
    ///     <see cref="OperationCanceledException" /> is thrown; see the single-objective overload.
    /// </param>
    /// <exception cref="NumSharp.RuntimeError">Optimization is invoked recursively on the same thread.</exception>
    /// <exception cref="NumSharp.ValueError"><paramref name="nJobs" /> is 0 or below -1, as Python's <c>ThreadPoolExecutor</c> refuses.</exception>
    /// <exception cref="ArgumentOutOfRangeException">The executor options in effect return an undefined executor kind or an invalid idle timeout; only a derived <see cref="TrialExecutorOptions" /> class can.</exception>
    /// <exception cref="OperationCanceledException">
    ///     <paramref name="ct" /> was cancelled before the scheduling ended, or an objective threw this exception while it
    ///     was cancelled. Thrown after every trial in flight is stored; <see cref="OperationCanceledException.CancellationToken" />
    ///     is <paramref name="ct" />.
    /// </exception>
    /// <exception cref="System.Exception">An objective raises an exception outside catch, or a callback, sampler, pruner, or storage operation fails; the original failure propagates.</exception>
    /// <remarks>Running objectives are allowed to finish when the count, timeout, Stop request, or cancellation ends scheduling. Callback failures are not covered by the objective catch list.</remarks>
    public void Optimize(
        Func<Trial, IReadOnlyList<double>> func,
        int? nTrials = null,
        double? timeout = null,
        int nJobs = 1,
        Type[]? @catch = null,
        IEnumerable<Action<Study, FrozenTrial>>? callbacks = null,
        bool gcAfterTrial = false,
        bool showProgressBar = false,
        TrialExecutorOptions? executorOptions = null,
        CancellationToken ct = default)
        => OptimizeLoop.Optimize(
            study: this,
            func: func,
            nTrials: nTrials,
            timeout: timeout,
            nJobs: nJobs,
            catchExceptions: @catch,
            callbacks: callbacks?.ToList(),
            gcAfterTrial: gcAfterTrial,
            showProgressBar: showProgressBar,
            executorOptions: executorOptions,
            ct: ct);

    /// <summary>
    ///     Create a new trial from which hyperparameters can be suggested.
    ///
    ///     This method is part of an alternative to
    ///     <see cref="Optimize(Func{Trial, double}, int?, double?, int, Type[], IEnumerable{Action{Study, FrozenTrial}}, bool, bool, TrialExecutorOptions, CancellationToken)" />
    ///     that allows controlling the lifetime of a trial outside the scope of <c>func</c>. Each
    ///     call to this method should be followed by a call to <see cref="Tell(Trial, double, bool)" />
    ///     to finish the created trial.
    /// </summary>
    /// <param name="fixedDistributions">
    ///     A dictionary containing the parameter names and parameter's distributions. Each
    ///     parameter in this dictionary is automatically suggested for the returned trial, even
    ///     when the suggest method is not explicitly invoked by the user.
    /// </param>
    /// <returns>An active trial claimed from the waiting queue or newly created; finish it with Tell.</returns>
    /// <exception cref="NumSharp.ValueError">A supplied distribution or pre-suggested parameter is invalid.</exception>
    /// <exception cref="System.Exception">Storage or a sampler hook fails while creating or initializing the trial.</exception>
    /// <remarks>A failure while pre-suggesting fixed distributions can leave an already-created trial running. This manual interface does not start a heartbeat worker.</remarks>
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
    /// <param name="trial">Trial record or active trial on which the operation acts.</param>
    /// <param name="value">Single objective value used to infer the completed state.</param>
    /// <param name="skipIfFinished">Whether to return an already-finished record without changing its state or values.</param>
    /// <exception cref="NumSharp.ValueError">The trial number is invalid, the trial is not running, its requested state and values disagree, or explicit completion has invalid objective values.</exception>
    /// <exception cref="NumSharp.KeyError">The requested trial cannot be found.</exception>
    /// <exception cref="System.Exception">The sampler AfterTrial hook or the storage backend fails.</exception>
    /// <exception cref="UpdateFinishedTrialError">Another storage instance or process finished the trial while this tell was writing it; the storage admits one of them.</exception>
    /// <remarks>
    ///     Thread-safe for one trial: tells of it through this study's storage instance, from any thread and from
    ///     the optimize loop, act in turn. Exactly one stores its result and reaches the sampler's <c>AfterTrial</c>;
    ///     every later one is refused with <c>Cannot tell a COMPLETE trial.</c> (or the trial's state), or skipped
    ///     when <c>skipIfFinished</c> is set. Optuna behaves the same in practice, because its GIL makes that order
    ///     near certain; without the ordering, two tells could both pass the check and both reach <c>AfterTrial</c>.
    /// </remarks>
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
    /// <exception cref="NumSharp.ValueError">The trial number is invalid, the trial is not running, its requested state and values disagree, or explicit completion has invalid objective values.</exception>
    /// <exception cref="NumSharp.KeyError">The requested trial cannot be found.</exception>
    /// <exception cref="System.Exception">The sampler AfterTrial hook or the storage backend fails.</exception>
    /// <exception cref="UpdateFinishedTrialError">Another storage instance or process finished the trial while this tell was writing it; the storage admits one of them.</exception>
    /// <remarks>
    ///     Thread-safe for one trial: tells of it through this study's storage instance, from any thread and from
    ///     the optimize loop, act in turn. Exactly one stores its result and reaches the sampler's <c>AfterTrial</c>;
    ///     every later one is refused with <c>Cannot tell a COMPLETE trial.</c> (or the trial's state), or skipped
    ///     when <c>skipIfFinished</c> is set. Optuna behaves the same in practice, because its GIL makes that order
    ///     near certain; without the ordering, two tells could both pass the check and both reach <c>AfterTrial</c>.
    /// </remarks>
    public FrozenTrial Tell(
        Trial trial, IReadOnlyList<double>? values = null, TrialState? state = null, bool skipIfFinished = false)
        => TellInternal(trial, values, state, skipIfFinished);

    /// <summary>Finish a trial identified by its number, reporting a single objective value.</summary>
    /// <param name="trialNumber">Zero-based trial number within the study, not a storage identifier.</param>
    /// <param name="value">Single objective value used to infer the completed state.</param>
    /// <param name="skipIfFinished">Whether to return an already-finished record without changing its state or values.</param>
    /// <returns>A detached copy of the resulting stored trial.</returns>
    /// <exception cref="NumSharp.ValueError">The trial number is invalid, the trial is not running, its requested state and values disagree, or explicit completion has invalid objective values.</exception>
    /// <exception cref="NumSharp.KeyError">The requested trial cannot be found.</exception>
    /// <exception cref="System.Exception">The sampler AfterTrial hook or the storage backend fails.</exception>
    /// <exception cref="UpdateFinishedTrialError">Another storage instance or process finished the trial while this tell was writing it; the storage admits one of them.</exception>
    /// <remarks>
    ///     Thread-safe for one trial: tells of it through this study's storage instance, from any thread and from
    ///     the optimize loop, act in turn. Exactly one stores its result and reaches the sampler's <c>AfterTrial</c>;
    ///     every later one is refused with <c>Cannot tell a COMPLETE trial.</c> (or the trial's state), or skipped
    ///     when <c>skipIfFinished</c> is set. Optuna behaves the same in practice, because its GIL makes that order
    ///     near certain; without the ordering, two tells could both pass the check and both reach <c>AfterTrial</c>.
    /// </remarks>
    public FrozenTrial Tell(int trialNumber, double value, bool skipIfFinished = false)
        => TellInternal(trialNumber, new[] { value }, null, skipIfFinished);

    /// <summary>Finish a trial identified by its number.</summary>
    /// <param name="trialNumber">Zero-based trial number within the study, not a storage identifier.</param>
    /// <param name="values">Objective values in study direction order, or null when no objective values are recorded.</param>
    /// <param name="state">Complete, Pruned, Fail, or null to infer Complete or Fail from objective-value validation.</param>
    /// <param name="skipIfFinished">Whether to return an already-finished record without changing its state or values.</param>
    /// <returns>A detached copy of the resulting stored trial.</returns>
    /// <exception cref="NumSharp.ValueError">The trial number is invalid, the trial is not running, its requested state and values disagree, or explicit completion has invalid objective values.</exception>
    /// <exception cref="NumSharp.KeyError">The requested trial cannot be found.</exception>
    /// <exception cref="System.Exception">The sampler AfterTrial hook or the storage backend fails.</exception>
    /// <exception cref="UpdateFinishedTrialError">Another storage instance or process finished the trial while this tell was writing it; the storage admits one of them.</exception>
    /// <remarks>
    ///     Thread-safe for one trial: tells of it through this study's storage instance, from any thread and from
    ///     the optimize loop, act in turn. Exactly one stores its result and reaches the sampler's <c>AfterTrial</c>;
    ///     every later one is refused with <c>Cannot tell a COMPLETE trial.</c> (or the trial's state), or skipped
    ///     when <c>skipIfFinished</c> is set. Optuna behaves the same in practice, because its GIL makes that order
    ///     near certain; without the ordering, two tells could both pass the check and both reach <c>AfterTrial</c>.
    /// </remarks>
    public FrozenTrial Tell(
        int trialNumber, IReadOnlyList<double>? values = null, TrialState? state = null, bool skipIfFinished = false)
        => TellInternal(trialNumber, values, state, skipIfFinished);

    /// <param name="trial">Trial record or active trial on which the operation acts.</param>
    /// <param name="values">Objective values in study direction order, or null when no objective values are recorded.</param>
    /// <param name="state">Complete, Pruned, Fail, or null to infer Complete or Fail from objective-value validation.</param>
    /// <summary>Finalizes a trial through validation, sampler postprocessing, and storage state updates.</summary>
    /// <param name="skipIfFinished">Whether to return an already-finished record without changing its state or values.</param>
    /// <returns>A detached copy of the resulting stored trial.</returns>
    /// <exception cref="NumSharp.ValueError">The trial number is invalid, the trial is not running, its requested state and values disagree, or explicit completion has invalid objective values.</exception>
    /// <exception cref="NumSharp.KeyError">The requested trial cannot be found.</exception>
    /// <exception cref="System.Exception">The sampler AfterTrial hook or the storage backend fails.</exception>
    private FrozenTrial TellInternal(Trial trial, IReadOnlyList<double>? values, TrialState? state, bool skipIfFinished)
    {
        // The snapshot is read, checked and written under the trial's tell monitor: one read before it could predate
        // another thread's tell of this trial, and both tells would pass the "still running" check
        // (see BaseStorage.TellLockFor).
        Storage.TellLocked(trial.TrialId,
            () => TellHelpers.TellWithWarning(this, TellHelpers.GetFrozenTrial(this, trial), values, state, skipIfFinished));
        return TellHelpers.GetFrozenTrial(this, trial).DeepCopy();
    }

    /// <param name="trialNumber">Zero-based trial number within the study, not a storage identifier.</param>
    /// <param name="values">Objective values in study direction order, or null when no objective values are recorded.</param>
    /// <param name="state">Complete, Pruned, Fail, or null to infer Complete or Fail from objective-value validation.</param>
    /// <summary>Finalizes a trial through validation, sampler postprocessing, and storage state updates.</summary>
    /// <param name="skipIfFinished">Whether to return an already-finished record without changing its state or values.</param>
    /// <returns>A detached copy of the resulting stored trial.</returns>
    /// <exception cref="NumSharp.ValueError">The trial number is invalid, the trial is not running, its requested state and values disagree, or explicit completion has invalid objective values.</exception>
    /// <exception cref="NumSharp.KeyError">The requested trial cannot be found.</exception>
    /// <exception cref="System.Exception">The sampler AfterTrial hook or the storage backend fails.</exception>
    private FrozenTrial TellInternal(int trialNumber, IReadOnlyList<double>? values, TrialState? state, bool skipIfFinished)
    {
        // The monitor is keyed by storage identifier, so the number is resolved first: a tell by number and a tell of
        // the same trial's Trial object then exclude each other (see BaseStorage.TellLockFor).
        int trialId = TellHelpers.GetTrialId(this, trialNumber);
        Storage.TellLocked(trialId,
            () => TellHelpers.TellWithWarning(this, Storage.GetTrial(trialId), values, state, skipIfFinished));
        return Storage.GetTrial(trialId).DeepCopy();
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
    /// <returns>An ordered TrialsTable of selected trial fields; this port does not require a pandas runtime.</returns>
    public TrialsTable TrialsDataframe(string[]? attrs = null, bool multiIndex = false)
        => OptunaSharp.TrialsDataframe.Build(this, attrs ?? OptunaSharp.TrialsDataframe.DefaultAttrs, multiIndex);

    /// <summary>Set a user attribute to the study.</summary>
    /// <param name="key">A key string of the attribute.</param>
    /// <param name="value">A value of the attribute. The value should be JSON serializable.</param>
    public void SetUserAttr(string key, object? value) => Storage.SetStudyUserAttr(StudyId, key, value);

    /// <summary>
    ///     Set several user attributes to the study at once (Rustuna's <c>set_user_attrs</c>; plan 13, item S8), with one
    ///     storage call: the stored result is what <see cref="SetUserAttr(string, object?)" /> once per entry leaves.
    /// </summary>
    /// <param name="attrs">The attributes, set in enumeration order. The values should be JSON serializable.</param>
    /// <exception cref="ArgumentNullException"><paramref name="attrs" /> is null.</exception>
    /// <exception cref="NumSharp.KeyError">The study no longer exists in storage; nothing is set.</exception>
    /// <exception cref="Exception">A configured storage fails; its exception propagates.</exception>
    public void SetUserAttrs(IReadOnlyDictionary<string, object?> attrs) => Storage.SetStudyUserAttrs(StudyId, attrs);

    /// <summary>Optuna's <c>set_user_attr</c> name for <see cref="SetUserAttrs" />: several user attributes in one call.</summary>
    /// <param name="attrs">The attributes, set in enumeration order. The values should be JSON serializable.</param>
    /// <exception cref="ArgumentNullException"><paramref name="attrs" /> is null.</exception>
    /// <exception cref="NumSharp.KeyError">The study no longer exists in storage; nothing is set.</exception>
    /// <exception cref="Exception">A configured storage fails; its exception propagates.</exception>
    public void SetUserAttr(IReadOnlyDictionary<string, object?> attrs) => SetUserAttrs(attrs);

    // Study.set_system_attr (deprecated 3.1.0) was removed in Optuna 5.0.0 (#6834). Optuna-internal
    // writers go through Storage.SetStudySystemAttr(StudyId, key, value); users keep SetUserAttr.

    /// <summary>
    ///     Exit from the current optimization loop after the running trials finish.
    ///
    ///     This method lets the running
    ///     <see cref="Optimize(Func{Trial, double}, int?, double?, int, Type[], IEnumerable{Action{Study, FrozenTrial}}, bool, bool, TrialExecutorOptions, CancellationToken)" />
    ///     method return immediately after all trials which the method spawned finishes. This
    ///     method only works when it is called inside an objective function or callback.
    /// </summary>
    /// <exception cref="NumSharp.RuntimeError">The call is outside this thread's active optimization loop.</exception>
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
    /// <remarks>Duplicate checking and insertion are separate storage operations; concurrent callers can enqueue the same parameters.</remarks>
    public void EnqueueTrial(
        OrderedDict<string, object?> @params,
        OrderedDict<string, object?>? userAttrs = null,
        bool skipIfExists = false)
    {
        // Write-boundary normalize: turn any boxed enum value into its member-NAME string BEFORE both
        // the duplicate check and the stored `fixed_params` payload, so (1) the value JSON-serializes as
        // the same string a SuggestCategorical<TEnum> emits, and (2) the skip comparison matches an
        // already-enqueued string. Returns the same dictionary untouched when it holds no enum values.
        @params = EnumCategorical.NormalizeParams(@params);

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
    /// <param name="trial">Trial record or active trial on which the operation acts.</param>
    /// <exception cref="NumSharp.ValueError">A trial has invalid state, timestamps, values, parameters, or distributions, or the objective count differs from this study.</exception>
    /// <exception cref="NumSharp.KeyError">The study has been removed from storage.</exception>
    /// <remarks>Records are inserted one at a time; a later failure does not roll back records already added.</remarks>
    public void AddTrial(FrozenTrial trial)
    {
        // Reject inconsistent state before any backend insertion so callers do not accumulate
        // unusable records when importing completed trials.
        trial.Validate();

        if (trial.Values is not null && Directions.Count != trial.Values.Count)
            throw new ValueError(
                $"The added trial has {trial.Values.Count} values, which is different from the " +
                $"number of objectives {Directions.Count} in the study (determined by " +
                "Study.directions).");

        Storage.CreateNewTrial(StudyId, templateTrial: trial);
    }

    /// <summary>Add trials to study. The trials are validated before being added.</summary>
    /// <param name="trials">Trial records to process in enumeration order.</param>
    /// <exception cref="NumSharp.ValueError">A trial has invalid state, timestamps, values, parameters, or distributions, or the objective count differs from this study.</exception>
    /// <exception cref="NumSharp.KeyError">The study has been removed from storage.</exception>
    /// <remarks>Records are inserted one at a time; a later failure does not roll back records already added.</remarks>
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
    /// <param name="metricNames">One display name per objective, in objective order.</param>
    /// <exception cref="NumSharp.ValueError">The number of names differs from the number of objectives.</exception>
    public void SetMetricNames(IReadOnlyList<string> metricNames)
    {
        if (Directions.Count != metricNames.Count)
            throw new ValueError("The number of objectives must match the length of the metric names.");

        Storage.SetStudySystemAttr(
            StudyId, SystemAttrMetricNames, metricNames.Cast<object?>().ToList());
    }

    /// <summary>Return <c>true</c> if the study has multiple objectives.</summary>
    /// <returns>True when the study has more than one direction.</returns>
    internal bool IsMultiObjective() => Directions.Count > 1;

    /// <summary>Python <c>Study._pop_waiting_trial_id</c>.</summary>
    /// <returns>Identifier of the first waiting trial successfully claimed, or null when none can be claimed.</returns>
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
    /// <param name="params">Parameter names mapped to their external values.</param>
    /// <returns>True when queued or recorded parameters match under the enqueue comparison rules.</returns>
    internal bool ShouldSkipEnqueue(OrderedDict<string, object?> @params)
    {
        foreach (var trial in GetTrials(deepcopy: false))
        {
            // The read-only accessors: this scans every trial, and the mutable ones would give each compact finished
            // record (plan 13, item S12) its dictionaries back.
            IReadOnlyDictionary<string, object?> trialParams =
                trial.SystemAttrs.GetValueOrDefault("fixed_params") as OrderedDict<string, object?>
                ?? trial.Params;

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
                    // np.isclose(a, b, atol=0.0, equal_nan=True): |a - b| <= atol + rtol*|b|.
                    // rtol is NOT passed, so it takes np.isclose's DEFAULT of 1e-5 (NOT 1e-8 — that
                    // is _truncnorm's explicit rtol; this call uses the numpy default).
                    isRepeated = double.IsNaN(newNumeric) && double.IsNaN(existingNumeric)
                        || Math.Abs(newNumeric - existingNumeric) <= 1e-05 * Math.Abs(existingNumeric);
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

    /// <summary>Checks Python-compatible scalar kind before comparing queued parameter values.</summary>
    /// <param name="a">Previously queued or recorded scalar.</param>
    /// <param name="b">Candidate scalar to enqueue.</param>
    /// <returns>True when the values have the same supported scalar kind.</returns>
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
    /// <param name="values">Completed objective values in study direction order.</param>
    /// <param name="params">Suggested parameter values included in the completion log.</param>
    /// <param name="number">Per-study trial number used in logs or replay.</param>
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

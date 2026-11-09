using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using OptunaSharp.Internal;

namespace OptunaSharp.Samplers;

/// <summary>
///     Base class for Genetic Algorithm (GA) samplers. Port of
///     <c>optuna/samplers/_ga/_base.py: BaseGASampler</c>.
///
///     GA samplers generate new trials by mimicking natural selection, using generations and
///     populations to iteratively improve solutions. This base class defines the interface and
///     provides utility methods for managing generations and populations. Parent selection is
///     handled by <see cref="SelectParent" />, which subclasses implement.
/// </summary>
/// <remarks>
///     Generation bookkeeping is <b>incremental</b>: each completed trial is inspected exactly once (via
///     <see cref="Study.GetTrialsView" />'s zero-copy window) and folded into a per-study tracker, instead of
///     re-scanning every completed trial on every ask as the reference implementation does. The numbers this
///     produces — the running generation, each generation's population, the cached parents — are identical to
///     the full scan's (same attrs read, same formula, same ascending-number order), so optimization
///     trajectories are bit-for-bit unchanged; only the cost drops from O(N) per ask (O(N²) per study) to
///     O(new trials). The generation attr each trial receives and the per-generation parent-id study attrs are
///     still written, so journals and resumed studies read exactly as before.
///     <para><b>One sampler, several studies.</b> The bookkeeping is kept per storage object and study id, not per
///     study id alone (every fresh in-memory storage starts at id 0), and it is dropped when the study stored under
///     that id is no longer the one it was built from, so a sampler instance reused for a second study starts it at
///     generation 0 with that study's own populations, exactly as Optuna's stateless scan does.</para>
/// </remarks>
public abstract class BaseGASampler : BaseSampler
{
    private int? _populationSize;

    /// <summary>
    ///     Per-study incremental generation bookkeeping — the cache that replaces the reference port's
    ///     every-ask full scans. All access goes through <see cref="Tracker" /> under <see cref="_trackerLock" />.
    /// </summary>
    private sealed class GenerationTracker
    {
        /// <summary>
        ///     Trial number 0 of the study when the tracker last caught up: if the study now stored under this id
        ///     has a different trial 0, it is a recreated study and everything below is stale. <c>null</c> until
        ///     the tracker has seen a trial.
        /// </summary>
        public TrialAnchor? Anchor;

        /// <summary>
        ///     A subclass's own per-study state (<see cref="GetStudyState{TState}" />), created on first use and
        ///     dropped together with the rest of this tracker.
        /// </summary>
        public object? Extension;

        /// <summary>First trial number not yet inspected; everything below it is folded or parked in <see cref="PendingNumbers" />.</summary>
        public int Cursor;

        /// <summary>Numbers below <see cref="Cursor" /> seen non-terminal (waiting/running) — rechecked each catch-up until they finish.</summary>
        public readonly List<int> PendingNumbers = new();

        /// <summary>Completed trials folded per generation — the incremental replacement of the reverse max-scan.</summary>
        public readonly Dictionary<int, int> CountByGeneration = new();

        /// <summary>Trial numbers per generation, in fold order (sorted ascending on read to match the scan's order).</summary>
        public readonly Dictionary<int, List<int>> NumbersByGeneration = new();

        /// <summary>Trial number by storage identifier for every folded trial — how cached parent ids resolve without a full scan.</summary>
        public readonly Dictionary<int, int> NumberById = new();

        /// <summary>Highest generation seen on a completed trial; 0 before any (the scan's default).</summary>
        public int MaxGeneration;

        /// <summary>Parents already selected (or resolved from the study attr) per generation — the scan-free hot path.</summary>
        public readonly Dictionary<int, List<FrozenTrial>> ParentCache = new();
    }

    private readonly object _trackerLock = new();
    // Keyed by storage object and study id, not the id alone: every fresh in-memory storage starts at study id 0
    // (pre-release review item H3).
    private readonly PerStudyState<GenerationTracker> _trackers = new();

    /// <summary>
    ///     The children this sampler handed to trials that are still running, so a child equal to one a running trial
    ///     evaluates is generated again instead of evaluated twice at once (<see cref="HandedOutParams" />). A subclass
    ///     claims each child through it in <c>SampleRelative</c> and forgets the trial in <c>AfterTrial</c>; a
    ///     sequential run never redraws, so its children, and Optuna parity, are unchanged.
    /// </summary>
    /// <remarks>Guarded by the sampler's monitor, under which every sampler hook runs; not by <c>_trackerLock</c>.</remarks>
    private protected HandedOutParams HandedOutChildren { get; } = new();

    /// <summary>Initializes generation bookkeeping with the requested number of completed individuals per generation.</summary>
    /// <param name="populationSize">Number of individuals retained or evaluated per generation.</param>
    protected BaseGASampler(int? populationSize)
    {
        _populationSize = populationSize;
    }

    // Python __init_subclass__ sets these per subclass to f"{cls.__name__}:...". GetType().Name at
    // runtime is the concrete subclass name, so "NSGAIISampler:generation" etc. — byte-identical.
    /// <summary>Trial system-attribute key namespaced by the concrete sampler type.</summary>
    internal string GenerationKey => $"{GetType().Name}:generation";
    /// <summary>Study system-attribute prefix used to cache parents by generation.</summary>
    internal string ParentCacheKeyPrefix => $"{GetType().Name}:parent:";

    /// <summary>Number of individuals (trials) in a generation.</summary>
    public int? PopulationSize
    {
        get => _populationSize;
        set => _populationSize = value;
    }

    /// <summary>
    ///     Select parent trials from the population for the given generation. Called once per
    ///     generation; the result is cached in the study system attributes.
    /// </summary>
    /// <param name="study">Study supplying trial history, objective directions, and storage.</param>
    /// <param name="generation">Zero-based generation whose individuals or optimizer state are requested.</param>
    /// <returns>The selected parent trials for the requested generation.</returns>
    public abstract List<FrozenTrial> SelectParent(Study study, int generation);

    /// <summary>
    ///     The tracker of <paramref name="study" />, created on first use and replaced by an empty one when it was
    ///     built for an earlier study stored under the same id. Callers hold <see cref="_trackerLock" />.
    /// </summary>
    /// <remarks>
    ///     A tracker is stale when the study's trial list is shorter than what the tracker already folded, or when
    ///     its trial 0 is not the trial the tracker last saw. Both happen only when the study was deleted and
    ///     another created under the same id on the same storage object (SQLite reuses ids). A live study only
    ///     grows, so neither check can drop a valid tracker: the returned tracker either continues incrementally or
    ///     starts from an empty fold, which is what a fresh process would compute from the same storage.
    /// </remarks>
    /// <param name="study">Study whose bookkeeping is requested; keyed by its storage object and study id.</param>
    /// <param name="view">The study's current trial list (<see cref="Study.GetTrialsView" />), used for the staleness check.</param>
    /// <returns>The study's tracker, valid for <paramref name="view" />.</returns>
    private GenerationTracker Tracker(Study study, IReadOnlyList<FrozenTrial> view)
        => _trackers.Resolve(
            study,
            static () => new GenerationTracker(),
            tracker => tracker.Cursor <= view.Count && (tracker.Anchor is not { } anchor || anchor.IsIn(view)));

    /// <summary>
    ///     Folds every not-yet-inspected trial into the tracker: completed trials are recorded once (their
    ///     generation attr read exactly as the reference scan reads it — absent counts as −1 and joins no
    ///     population); waiting/running trials are parked and rechecked until they finish, so out-of-order
    ///     completion (an enqueued trial finishing late) is handled without ever rescanning the folded prefix.
    /// </summary>
    /// <param name="tracker">The study's tracker, as returned by <see cref="Tracker" /> for <paramref name="view" />; caller holds <see cref="_trackerLock" />.</param>
    /// <param name="view">The study's current trial list, the one the tracker was validated against.</param>
    /// <exception cref="System.FormatException">A stored generation string cannot be parsed as an integer.</exception>
    /// <exception cref="System.InvalidCastException">A stored generation value does not support integer conversion.</exception>
    /// <exception cref="System.OverflowException">A stored generation value exceeds Int32.</exception>
    private void CatchUp(GenerationTracker tracker, IReadOnlyList<FrozenTrial> view)
    {
        // Parked numbers first: a trial that completed since the last catch-up must fold before anything
        // that might read its generation. Fold order does not affect results (max/count are order-free and
        // population lists are sorted on read), only completeness does.
        if (tracker.PendingNumbers.Count > 0)
        {
            for (int i = tracker.PendingNumbers.Count - 1; i >= 0; i--)
            {
                FrozenTrial parked = view[tracker.PendingNumbers[i]];
                if (!parked.State.IsFinished())
                    continue;
                Fold(tracker, parked);
                tracker.PendingNumbers.RemoveAt(i);
            }
        }
        for (int number = tracker.Cursor; number < view.Count; number++)
        {
            FrozenTrial trial = view[number];
            if (trial.State.IsFinished())
                Fold(tracker, trial);
            else
                tracker.PendingNumbers.Add(number);
        }
        tracker.Cursor = view.Count;
        // Re-anchored on every catch-up (the view was just validated), so a trial 0 that was still waiting, with no
        // start time, is anchored with its start time once it runs.
        if (view.Count > 0)
            tracker.Anchor = TrialAnchor.Of(view[0]);
    }

    /// <summary>Records one finished trial: only completed ones carry population weight, exactly like the reference scan's Complete filter.</summary>
    /// <param name="tracker">The study's tracker; caller holds <see cref="_trackerLock" />.</param>
    /// <param name="trial">A finished trial inspected for the first time.</param>
    /// <exception cref="System.FormatException">A stored generation string cannot be parsed as an integer.</exception>
    /// <exception cref="System.InvalidCastException">A stored generation value does not support integer conversion.</exception>
    /// <exception cref="System.OverflowException">A stored generation value exceeds Int32.</exception>
    private void Fold(GenerationTracker tracker, FrozenTrial trial)
    {
        if (trial.State != TrialState.Complete)
            return;   // failed/pruned trials never belonged to a population in the reference scan either
        tracker.NumberById[trial.TrialId] = trial.Number;
        object? raw = trial.SystemAttrs.GetValueOrDefault(GenerationKey);
        int generation = raw is null ? -1 : Convert.ToInt32(raw, CultureInfo.InvariantCulture);
        if (generation < 0)
            return;   // the scan's `raw is null ? -1` case: contributes to no maximum and no population
        tracker.CountByGeneration[generation] = tracker.CountByGeneration.GetValueOrDefault(generation) + 1;
        if (!tracker.NumbersByGeneration.TryGetValue(generation, out List<int>? numbers))
        {
            numbers = new List<int>();
            tracker.NumbersByGeneration[generation] = numbers;
        }
        numbers.Add(trial.Number);
        if (generation > tracker.MaxGeneration)
            tracker.MaxGeneration = generation;
    }

    /// <summary>
    ///     Get the generation number of the given trial. If not already set in the trial's system
    ///     attributes, it is calculated (from the max generation of all completed trials) and stored.
    /// </summary>
    /// <param name="study">Study supplying trial history, objective directions, and storage.</param>
    /// <param name="trial">Trial snapshot being examined; copy its contents before modifying them.</param>
    /// <returns>The stored or newly assigned zero-based generation.</returns>
    /// <exception cref="System.FormatException">A stored generation string cannot be parsed as an integer.</exception>
    /// <exception cref="System.InvalidCastException">A stored generation value does not support integer conversion.</exception>
    /// <exception cref="System.OverflowException">A stored generation value exceeds Int32.</exception>
    /// <exception cref="System.Exception">The configured storage provider fails; its exception propagates.</exception>
    public int GetTrialGeneration(Study study, FrozenTrial trial)
    {
        var stored = trial.SystemAttrs.GetValueOrDefault(GenerationKey);
        if (stored is not null)
            return Convert.ToInt32(stored, CultureInfo.InvariantCulture);

        lock (_trackerLock)
        {
            IReadOnlyList<FrozenTrial> view = study.GetTrialsView();
            GenerationTracker tracker = Tracker(study, view);
            CatchUp(tracker, view);

            // The reference reverse-scan's (maxGeneration, count-at-max) over completed trials, from the
            // incremental fold: identical numbers, O(new) instead of O(all).
            int maxGeneration = tracker.MaxGeneration;
            int maxGenerationCount = tracker.CountByGeneration.GetValueOrDefault(maxGeneration);

            System.Diagnostics.Debug.Assert(_populationSize is not null, "Population size must be set.");
            int result = maxGenerationCount < _populationSize ? maxGeneration : maxGeneration + 1;
            study.Storage.SetTrialSystemAttr(trial.TrialId, GenerationKey, result);
            return result;
        }
    }

    /// <summary>Get the population (completed trials) of the given generation.</summary>
    /// <param name="study">Study supplying trial history, objective directions, and storage.</param>
    /// <param name="generation">Zero-based generation whose individuals or optimizer state are requested.</param>
    /// <returns>Completed trial snapshots belonging to the requested generation, in ascending trial-number order.</returns>
    /// <exception cref="System.FormatException">A stored generation string cannot be parsed as an integer.</exception>
    /// <exception cref="System.InvalidCastException">A stored generation value does not support integer conversion.</exception>
    /// <exception cref="System.OverflowException">A stored generation value exceeds Int32.</exception>
    public List<FrozenTrial> GetPopulation(Study study, int generation)
    {
        lock (_trackerLock)
        {
            IReadOnlyList<FrozenTrial> view = study.GetTrialsView();
            GenerationTracker tracker = Tracker(study, view);
            CatchUp(tracker, view);
            var result = new List<FrozenTrial>();
            if (!tracker.NumbersByGeneration.TryGetValue(generation, out List<int>? numbers))
                return result;
            // Ascending trial-number order = the reference scan's order (it walked the trial list front to
            // back). Out-of-order completion can append out of order above, so order is restored here. Every
            // folded number is below view.Count, so the view the tracker caught up with resolves them all.
            foreach (int number in numbers.OrderBy(n => n))
                result.Add(view[number]);   // trial numbers equal list positions by construction
            return result;
        }
    }

    /// <summary>
    ///     Get the parent population of the given generation (cached in the study system attributes).
    ///     Returns an empty list when <paramref name="generation" /> is 0.
    /// </summary>
    /// <param name="study">Study supplying trial history, objective directions, and storage.</param>
    /// <param name="generation">Zero-based generation whose individuals or optimizer state are requested.</param>
    /// <returns>Selected parents, or an empty list for generation zero.</returns>
    /// <exception cref="System.FormatException">A cached parent identifier string cannot be parsed as an integer.</exception>
    /// <exception cref="System.InvalidCastException">Cached parent metadata has an unexpected collection or identifier representation.</exception>
    /// <exception cref="System.OverflowException">A cached parent identifier exceeds Int32.</exception>
    /// <exception cref="System.Exception">The configured parent-selection strategy or storage provider fails; its exception propagates.</exception>
    public List<FrozenTrial> GetParentPopulation(Study study, int generation)
    {
        if (generation == 0)
            return new List<FrozenTrial>();

        lock (_trackerLock)
        {
            // Validated before the parent cache is read: a tracker left from an earlier study under this id would
            // otherwise hand that study's parents to this one.
            IReadOnlyList<FrozenTrial> view = study.GetTrialsView();
            GenerationTracker tracker = Tracker(study, view);
            if (tracker.ParentCache.TryGetValue(generation, out List<FrozenTrial>? held))
                return new List<FrozenTrial>(held);

            // Not cached in this process yet: honor the study-attr cache first (a resumed study's parents
            // were selected by an earlier process) — same protocol, resolved against the zero-copy view.
            var studySystemAttrs = study.Storage.GetStudySystemAttrs(study.StudyId);
            var cached = studySystemAttrs.GetValueOrDefault(
                ParentCacheKeyPrefix + generation.ToString(CultureInfo.InvariantCulture));

            if (cached is not null)
            {
                var ids = new HashSet<int>(
                    ((IEnumerable<object?>)cached).Select(x => Convert.ToInt32(x, CultureInfo.InvariantCulture)));
                List<FrozenTrial> resolved = view.Where(t => ids.Contains(t.TrialId)).ToList();
                tracker.ParentCache[generation] = resolved;
                return new List<FrozenTrial>(resolved);
            }

            List<FrozenTrial> parentPopulation = SelectParent(study, generation);
            study.Storage.SetStudySystemAttr(
                study.StudyId,
                ParentCacheKeyPrefix + generation.ToString(CultureInfo.InvariantCulture),
                parentPopulation.Select(t => (object?)t.TrialId).ToList());

            // A fresh view: SelectParent may have read the study long after the one above was taken. Re-resolving
            // the tracker under the same lock returns the same instance (a live study only grows).
            view = study.GetTrialsView();
            tracker = Tracker(study, view);

            // Parity subtlety, load-bearing: the reference returns the elite strategy's OWN order on the ask
            // that ran the selection, but every LATER ask re-resolved the cached ids against the trial list —
            // i.e. ascending trial-number order. Child generation consumes parents by RNG index, so the two
            // orders are observable. The cache must therefore hold the number-ordered resolution (what every
            // later ask must see), while THIS ask returns the strategy's order — bit-for-bit the reference.
            CatchUp(tracker, view);   // every parent is a completed, folded trial; make NumberById complete
            List<FrozenTrial> numberOrdered = parentPopulation
                .Select(t => tracker.NumberById.TryGetValue(t.TrialId, out int number) ? view[number] : t)
                .OrderBy(t => t.Number)
                .ToList();
            tracker.ParentCache[generation] = numberOrdered;
            return parentPopulation;
        }
    }

    /// <summary>
    ///     A subclass's own state for <paramref name="study" /> (for example RVEA's adaptive reference vectors and
    ///     running ideal point), created on first use and kept with this class's generation bookkeeping: it is
    ///     per study, so a reused sampler gives every study a fresh one, and it is dropped whenever that
    ///     bookkeeping is (a recreated study under a reused id).
    /// </summary>
    /// <remarks>
    ///     Footguns: one sampler instance must always request the same <typeparamref name="TState" />, and the
    ///     returned object is only guarded by this class's lock while the call runs. Mutate it from
    ///     <see cref="SelectParent" /> (which this class only calls under that lock) or under a lock of your own.
    ///     Each call reads the study's trial list once, so call it per generation rather than per ask where the
    ///     storage is not in memory.
    /// </remarks>
    /// <typeparam name="TState">The subclass's per-study state type.</typeparam>
    /// <param name="study">The study whose state is requested.</param>
    /// <param name="create">Builds the state the first time this study needs it.</param>
    /// <returns>The study's state.</returns>
    /// <exception cref="System.InvalidCastException">This sampler instance stored a different state type for the study earlier.</exception>
    /// <exception cref="System.Exception"><paramref name="create" /> or the storage fails; its exception propagates.</exception>
    private protected TState GetStudyState<TState>(Study study, Func<TState> create) where TState : class
    {
        lock (_trackerLock)
        {
            IReadOnlyList<FrozenTrial> view = study.GetTrialsView();
            GenerationTracker tracker = Tracker(study, view);
            tracker.Extension ??= create();
            return (TState)tracker.Extension;
        }
    }
}

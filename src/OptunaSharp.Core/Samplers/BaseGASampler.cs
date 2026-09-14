using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;

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
    private readonly Dictionary<int, GenerationTracker> _trackers = new();

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

    /// <summary>The tracker of <paramref name="study" />, created on first use. Callers hold <see cref="_trackerLock" />.</summary>
    /// <param name="study">Study whose bookkeeping is requested; keyed by its storage identifier.</param>
    /// <returns>The study's tracker.</returns>
    private GenerationTracker Tracker(Study study)
    {
        if (!_trackers.TryGetValue(study.StudyId, out GenerationTracker? tracker))
        {
            tracker = new GenerationTracker();
            _trackers[study.StudyId] = tracker;
        }
        return tracker;
    }

    /// <summary>
    ///     Folds every not-yet-inspected trial into the tracker: completed trials are recorded once (their
    ///     generation attr read exactly as the reference scan reads it — absent counts as −1 and joins no
    ///     population); waiting/running trials are parked and rechecked until they finish, so out-of-order
    ///     completion (an enqueued trial finishing late) is handled without ever rescanning the folded prefix.
    /// </summary>
    /// <param name="study">Study supplying trial history, objective directions, and storage.</param>
    /// <param name="tracker">The study's tracker; caller holds <see cref="_trackerLock" />.</param>
    private void CatchUp(Study study, GenerationTracker tracker)
    {
        IReadOnlyList<FrozenTrial> view = study.GetTrialsView();
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
        object? raw = trial.SystemAttrsDict.GetValueOrDefault(GenerationKey);
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
        var stored = trial.SystemAttrsDict.GetValueOrDefault(GenerationKey);
        if (stored is not null)
            return Convert.ToInt32(stored, CultureInfo.InvariantCulture);

        lock (_trackerLock)
        {
            GenerationTracker tracker = Tracker(study);
            CatchUp(study, tracker);

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
            GenerationTracker tracker = Tracker(study);
            CatchUp(study, tracker);
            var result = new List<FrozenTrial>();
            if (!tracker.NumbersByGeneration.TryGetValue(generation, out List<int>? numbers))
                return result;
            // Ascending trial-number order = the reference scan's order (it walked the trial list front to
            // back). Out-of-order completion can append out of order above, so order is restored here.
            IReadOnlyList<FrozenTrial> view = study.GetTrialsView();
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
            GenerationTracker tracker = Tracker(study);
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
                List<FrozenTrial> resolved = study.GetTrialsView().Where(t => ids.Contains(t.TrialId)).ToList();
                tracker.ParentCache[generation] = resolved;
                return new List<FrozenTrial>(resolved);
            }

            List<FrozenTrial> parentPopulation = SelectParent(study, generation);
            study.Storage.SetStudySystemAttr(
                study.StudyId,
                ParentCacheKeyPrefix + generation.ToString(CultureInfo.InvariantCulture),
                parentPopulation.Select(t => (object?)t.TrialId).ToList());

            // Parity subtlety, load-bearing: the reference returns the elite strategy's OWN order on the ask
            // that ran the selection, but every LATER ask re-resolved the cached ids against the trial list —
            // i.e. ascending trial-number order. Child generation consumes parents by RNG index, so the two
            // orders are observable. The cache must therefore hold the number-ordered resolution (what every
            // later ask must see), while THIS ask returns the strategy's order — bit-for-bit the reference.
            CatchUp(study, tracker);   // every parent is a completed, folded trial; make NumberById complete
            IReadOnlyList<FrozenTrial> view = study.GetTrialsView();
            List<FrozenTrial> numberOrdered = parentPopulation
                .Select(t => tracker.NumberById.TryGetValue(t.TrialId, out int number) ? view[number] : t)
                .OrderBy(t => t.Number)
                .ToList();
            tracker.ParentCache[generation] = numberOrdered;
            return parentPopulation;
        }
    }
}

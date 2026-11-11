using System;
using System.Collections.Generic;
using System.Linq;

namespace OptunaSharp.Internal;

/// <summary>
///     Port of the trial-splitting helpers in <c>optuna/samplers/_tpe/sampler.py</c>, shared with
///     <c>PedAnovaImportanceEvaluator</c> and the multi-objective TPE split. The HSSP tie-break in
///     <see cref="SplitCompleteTrialsMultiObjective" /> routes through <see cref="Hypervolume" />
///     (Phase 2.10).
/// </summary>
internal static class TpeTrialSplit
{
    // -------------------------------------------------------------------------------------------
    // Module-level defaults from sampler.py (gamma / weights).
    // -------------------------------------------------------------------------------------------

    /// <summary>Python <c>default_gamma(x) = min(ceil(0.1 * x), 25)</c>.</summary>
    /// <param name="x">The number of observed trials for which to compute the TPE split or observation weights.</param>
    /// <returns>The preferred-group size, capped at 25 observations.</returns>
    internal static int DefaultGamma(int x) => Math.Min((int)Math.Ceiling(0.1 * x), 25);

    /// <summary>Python <c>default_gamma_multiobjective(x) = ceil(0.1 * x)</c>.</summary>
    /// <param name="x">The number of observed trials for which to compute the TPE split or observation weights.</param>
    /// <returns>The preferred-group size for multi-objective TPE.</returns>
    internal static int DefaultGammaMultiObjective(int x) => (int)Math.Ceiling(0.1 * x);

    /// <summary>Python <c>hyperopt_default_gamma(x) = min(ceil(0.25 * x**0.5), 25)</c>.</summary>
    /// <param name="x">The number of observed trials for which to compute the TPE split or observation weights.</param>
    /// <returns>The Hyperopt-compatible preferred-group size, capped at 25 observations.</returns>
    internal static int HyperoptDefaultGamma(int x) => Math.Min((int)Math.Ceiling(0.25 * Math.Pow(x, 0.5)), 25);

    /// <summary>Python <c>default_weights(x)</c> — uniform below 25, ramp+flat at and above 25.</summary>
    /// <param name="x">The number of observed trials for which to compute the TPE split or observation weights.</param>
    /// <returns>Observation weights in historical order, with the most recent 25 given unit weight.</returns>
    internal static double[] DefaultWeights(int x)
    {
        if (x == 0)
            return Array.Empty<double>();
        if (x < 25)
        {
            var ones = new double[x];
            for (int i = 0; i < x; i++)
                ones[i] = 1.0;
            return ones;
        }

        var ramp = Linspace(1.0 / x, 1.0, x - 25);
        var result = new double[ramp.Length + 25];
        Array.Copy(ramp, result, ramp.Length);
        for (int i = ramp.Length; i < result.Length; i++)
            result[i] = 1.0;  // flat = np.ones(25).
        return result;
    }

    /// <summary>NumPy <c>np.linspace(start, stop, num)</c> (endpoint=True) — bit-exact.</summary>
    /// <param name="start">The inclusive start of the interval or slice.</param>
    /// <param name="stop">The final value of the generated numerical interval.</param>
    /// <param name="num">The number of values in the output grid.</param>
    /// <returns>An evenly spaced grid including both endpoints when at least two values are requested.</returns>
    private static double[] Linspace(double start, double stop, int num)
    {
        if (num <= 0)
            return Array.Empty<double>();
        if (num == 1)
            return new[] { start };

        int div = num - 1;
        double delta = stop - start;
        double step = delta / div;
        var y = new double[num];
        for (int i = 0; i < num; i++)
            y[i] = i * step + start;  // arange(num) * step + start.
        y[^1] = stop;  // endpoint override.
        return y;
    }

    // -------------------------------------------------------------------------------------------
    // Trial splitting (below / above).
    // -------------------------------------------------------------------------------------------

    /// <summary>Python <c>_split_trials(study, trials, n_below)</c>.</summary>
    /// <param name="study">The study supplying trial history, objective directions, and storage.</param>
    /// <param name="trials">Trial snapshots to inspect; their order is preserved unless the operation explicitly ranks them.</param>
    /// <param name="nBelow">The number of trials to assign to the preferred group or to cover by Pareto ranks.</param>
    /// <returns>The preferred and remaining trials, each sorted by trial number.</returns>
    internal static (List<FrozenTrial> Below, List<FrozenTrial> Above) SplitTrials(
        Study study, IReadOnlyList<FrozenTrial> trials, int nBelow)
    {
        // Sized for the common case, where nearly every trial is complete, so the list never regrows (plan 15).
        var complete = new List<FrozenTrial>(trials.Count);
        var pruned = new List<FrozenTrial>();
        var running = new List<FrozenTrial>();
        var infeasible = new List<FrozenTrial>();

        foreach (var trial in trials)
        {
            if (trial.State == TrialState.Running)
                // Check RUNNING before feasibility: constraints are not yet set.
                running.Add(trial);
            else if (GetInfeasibleTrialScore(trial) > 0)
                infeasible.Add(trial);
            else if (trial.State == TrialState.Complete)
                complete.Add(trial);
            else if (trial.State == TrialState.Pruned)
                pruned.Add(trial);
            else
                System.Diagnostics.Debug.Assert(false, "Unexpected trial state in _split_trials.");
        }

        var (belowComplete, aboveComplete) = SplitCompleteTrials(complete, study, nBelow);
        nBelow = Math.Max(0, nBelow - belowComplete.Count);
        var (belowPruned, abovePruned) = SplitPrunedTrials(pruned, study, nBelow);
        nBelow = Math.Max(0, nBelow - belowPruned.Count);
        var (belowInfeasible, aboveInfeasible) = SplitInfeasibleTrials(infeasible, nBelow);

        var below = new List<FrozenTrial>(belowComplete.Count + belowPruned.Count + belowInfeasible.Count);
        below.AddRange(belowComplete);
        below.AddRange(belowPruned);
        below.AddRange(belowInfeasible);
        var above = new List<FrozenTrial>(aboveComplete.Count + abovePruned.Count + aboveInfeasible.Count + running.Count);
        above.AddRange(aboveComplete);
        above.AddRange(abovePruned);
        above.AddRange(aboveInfeasible);
        above.AddRange(running);

        SortByNumber(below);
        SortByNumber(above);
        return (below, above);
    }

    /// <summary>Python <c>_split_complete_trials(trials, study, n_below)</c>.</summary>
    /// <param name="trials">Trial snapshots to inspect; their order is preserved unless the operation explicitly ranks them.</param>
    /// <param name="study">The study supplying trial history, objective directions, and storage.</param>
    /// <param name="nBelow">The number of trials to assign to the preferred group or to cover by Pareto ranks.</param>
    /// <returns>The preferred and remaining completed trials according to the study objectives.</returns>
    private static (List<FrozenTrial> Below, List<FrozenTrial> Above) SplitCompleteTrials(
        IReadOnlyList<FrozenTrial> trials, Study study, int nBelow)
    {
        nBelow = Math.Min(nBelow, trials.Count);
        return study.Directions.Count <= 1
            ? SplitCompleteTrialsSingleObjective(trials, study, nBelow)
            : SplitCompleteTrialsMultiObjective(trials, study, nBelow);
    }

    /// <summary>Python <c>_split_complete_trials_single_objective(trials, study, n_below)</c>.</summary>
    /// <param name="trials">Trial snapshots to inspect; their order is preserved unless the operation explicitly ranks them.</param>
    /// <param name="study">The study supplying trial history, objective directions, and storage.</param>
    /// <param name="nBelow">The number of trials to assign to the preferred group or to cover by Pareto ranks.</param>
    /// <returns>The best requested completed trials and the remaining trials in objective order.</returns>
    private static (List<FrozenTrial> Below, List<FrozenTrial> Above) SplitCompleteTrialsSingleObjective(
        IReadOnlyList<FrozenTrial> trials, Study study, int nBelow)
    {
        // Python's `sorted` is stable: ties keep input order. The trials are sorted by (value, position), descending
        // values for a maximized objective, which is the order LINQ's stable OrderBy / OrderByDescending produced, with
        // the same double.CompareTo semantics (NaN first, -0.0 equal to 0.0). Plan 15: (value, position) structs that
        // compare themselves, so the generic sort calls CompareTo directly instead of a comparer through an interface,
        // and the two sides built at their final sizes instead of three intermediate lists per ask.
        int n = trials.Count;
        bool descending = study.Directions[0] != StudyDirection.Minimize;
        var keyed = new ValueThenPosition[n];
        for (int i = 0; i < n; i++)
            keyed[i] = new ValueThenPosition(trials[i].Value!.Value, i, descending);
        Array.Sort(keyed);

        var below = new List<FrozenTrial>(nBelow);
        var above = new List<FrozenTrial>(n - nBelow);
        for (int r = 0; r < n; r++)
            (r < nBelow ? below : above).Add(trials[keyed[r].Position]);
        return (below, above);
    }

    /// <summary>
    ///     A trial's objective value and position, ordered by value (ascending, or descending for a maximized
    ///     objective), ties by position: the order of a stable sort by value.
    /// </summary>
    /// <remarks>
    ///     Values compare with <see cref="double.CompareTo(double)" />, the comparison LINQ's default comparer made, so NaN
    ///     sorts first and -0.0 ties with 0.0; reversing its sign for the descending order is what
    ///     <c>OrderByDescending</c> did. Positions are distinct, so the order is total and any correct sort yields it.
    ///     Every element of one sort carries the same direction.
    /// </remarks>
    private readonly struct ValueThenPosition : IComparable<ValueThenPosition>
    {
        /// <summary>The trial's objective value.</summary>
        internal readonly double Value;

        /// <summary>The trial's position in the input list.</summary>
        internal readonly int Position;

        private readonly bool _descending;

        /// <summary>Pairs a value with its position.</summary>
        /// <param name="value">The trial's objective value.</param>
        /// <param name="position">The trial's position in the input list.</param>
        /// <param name="descending">Whether larger values come first; the same for every element of a sort.</param>
        internal ValueThenPosition(double value, int position, bool descending)
        {
            Value = value;
            Position = position;
            _descending = descending;
        }

        /// <summary>Compares by value, then by position.</summary>
        /// <param name="other">The element to compare with.</param>
        /// <returns>Negative when this sorts first, positive when <paramref name="other" /> does, zero only for the same position.</returns>
        public int CompareTo(ValueThenPosition other)
        {
            int byValue = Value.CompareTo(other.Value);
            if (_descending)
                byValue = -byValue;
            return byValue != 0 ? byValue : Position.CompareTo(other.Position);
        }
    }

    /// <summary>Python <c>_get_pruned_trial_score(trial, study)</c>.</summary>
    /// <param name="trial">The trial whose recorded state and values are used.</param>
    /// <param name="study">The study supplying trial history, objective directions, and storage.</param>
    /// <returns>A lexicographic score prioritizing later pruning steps and then better intermediate values.</returns>
    private static (double, double) GetPrunedTrialScore(FrozenTrial trial, Study study)
    {
        if (trial.IntermediateValuesForRead.Count > 0)
        {
            long step = trial.IntermediateValuesForRead.Keys.Max();  // max(dict.items()) compares by key (step).
            double intermediateValue = trial.IntermediateValuesForRead[step];
            if (double.IsNaN(intermediateValue))
                return (-step, double.PositiveInfinity);
            if (study.Directions[0] == StudyDirection.Minimize)
                return (-step, intermediateValue);
            return (-step, -intermediateValue);
        }
        return (1, 0.0);
    }

    /// <summary>Python <c>_split_pruned_trials(trials, study, n_below)</c>.</summary>
    /// <param name="trials">Trial snapshots to inspect; their order is preserved unless the operation explicitly ranks them.</param>
    /// <param name="study">The study supplying trial history, objective directions, and storage.</param>
    /// <param name="nBelow">The number of trials to assign to the preferred group or to cover by Pareto ranks.</param>
    /// <returns>The preferred and remaining pruned trials ranked by pruning step and intermediate value.</returns>
    private static (List<FrozenTrial> Below, List<FrozenTrial> Above) SplitPrunedTrials(
        IReadOnlyList<FrozenTrial> trials, Study study, int nBelow)
    {
        nBelow = Math.Min(nBelow, trials.Count);
        var sorted = trials
            .Select(t => (Trial: t, Score: GetPrunedTrialScore(t, study)))
            .OrderBy(x => x.Score.Item1)
            .ThenBy(x => x.Score.Item2)
            .Select(x => x.Trial)
            .ToList();
        return (sorted.Take(nBelow).ToList(), sorted.Skip(nBelow).ToList());
    }

    /// <summary>
    ///     Sorts trials by number in place, as <c>trials.sort(key=lambda trial: trial.number)</c> does in
    ///     <c>_split_trials</c>.
    /// </summary>
    /// <param name="trials">The trials to sort; their numbers are distinct.</param>
    /// <remarks>
    ///     Trial numbers are unique within a study, so every correct sort produces the same order, and sorting the
    ///     numbers as primitive keys with the trials alongside gives exactly what the comparison-delegate sort gave
    ///     (plan 15: the delegate was called about n log n times per ask on the above list).
    /// </remarks>
    private static void SortByNumber(List<FrozenTrial> trials)
    {
        int n = trials.Count;
        if (n < 2)
            return;
        var numbers = new int[n];
        var items = new FrozenTrial[n];
        for (int i = 0; i < n; i++)
        {
            items[i] = trials[i];
            numbers[i] = items[i].Number;
        }
        Array.Sort(numbers, items);
        for (int i = 0; i < n; i++)
            trials[i] = items[i];
    }

    /// <summary>Python <c>_get_infeasible_trial_score(trial) = sum(v for v in constraints.values() if v > 0)</c>.</summary>
    /// <param name="trial">The trial whose recorded state and values are used.</param>
    /// <returns>The sum of positive constraint violations, or positive infinity when feasibility information is unavailable.</returns>
    private static double GetInfeasibleTrialScore(FrozenTrial trial)
    {
        double sum = 0.0;
        // Most trials record no constraint, and then the loop below adds nothing: return its 0.0 without building the
        // empty constraints dictionary (plan 15; the split scores every trial on every ask, so that was one dictionary
        // per trial per ask, two thirds of what a 1,000-trial split allocated). A trial with any constraint, in the
        // old list format or the per-key one, takes the original path.
        if (!ConstrainedOptimization.HasConstraints(trial.SystemAttrs))
            return sum;
        foreach (var v in trial.Constraints.Values)
        {
            if (v > 0)
                sum += v;
        }
        return sum;
    }

    /// <summary>Python <c>_split_infeasible_trials(trials, n_below)</c>.</summary>
    /// <param name="trials">Trial snapshots to inspect; their order is preserved unless the operation explicitly ranks them.</param>
    /// <param name="nBelow">The number of trials to assign to the preferred group or to cover by Pareto ranks.</param>
    /// <returns>The trials with smallest constraint violation and the remaining infeasible trials.</returns>
    private static (List<FrozenTrial> Below, List<FrozenTrial> Above) SplitInfeasibleTrials(
        IReadOnlyList<FrozenTrial> trials, int nBelow)
    {
        nBelow = Math.Min(nBelow, trials.Count);
        var sorted = trials
            .Select(t => (Trial: t, Score: GetInfeasibleTrialScore(t)))
            .OrderBy(x => x.Score)
            .Select(x => x.Trial)
            .ToList();
        return (sorted.Take(nBelow).ToList(), sorted.Skip(nBelow).ToList());
    }

    /// <summary>Python <c>_split_complete_trials_multi_objective(trials, study, n_below)</c>.</summary>
    /// <param name="trials">Trial snapshots to inspect; their order is preserved unless the operation explicitly ranks them.</param>
    /// <param name="study">The study supplying trial history, objective directions, and storage.</param>
    /// <param name="nBelow">The number of trials to assign to the preferred group or to cover by Pareto ranks.</param>
    /// <returns>The preferred Pareto-ranked trials and the remainder, using hypervolume to split a partial rank.</returns>
    internal static (List<FrozenTrial> Below, List<FrozenTrial> Above) SplitCompleteTrialsMultiObjective(
        IReadOnlyList<FrozenTrial> trials, Study study, int nBelow)
    {
        if (nBelow == 0)
            return (new List<FrozenTrial>(), trials.ToList());
        if (nBelow == trials.Count)
            return (trials.ToList(), new List<FrozenTrial>());

        System.Diagnostics.Debug.Assert(0 < nBelow && nBelow < trials.Count);

        var directions = study.Directions;
        // lvals = np.array([trial.values]) * [(-1 if MAXIMIZE else 1) per objective].
        var lvals = trials
            .Select(trial => trial.Values!
                .Select((v, i) => (directions[i] == StudyDirection.Maximize ? -1.0 : 1.0) * v)
                .ToArray())
            .ToArray();

        var nondominationRanks = MultiObjective.FastNonDominationRank(lvals, nBelow: nBelow);

        // ranks, rank_counts = np.unique(nondomination_ranks, return_counts=True).
        var counts = new SortedDictionary<int, int>();
        foreach (int r in nondominationRanks)
            counts[r] = counts.TryGetValue(r, out int c) ? c + 1 : 1;
        var ranks = counts.Keys.ToArray();
        var rankCounts = counts.Values.ToArray();

        // last_rank_before_tiebreak = max(ranks[cumsum(rank_counts) <= n_below], initial=-1).
        int lastRankBeforeTiebreak = -1;
        int cumsum = 0;
        for (int i = 0; i < ranks.Length; i++)
        {
            cumsum += rankCounts[i];
            if (cumsum <= nBelow)
                lastRankBeforeTiebreak = Math.Max(lastRankBeforeTiebreak, ranks[i]);
        }

        var belowIndices = new List<int>();
        for (int i = 0; i < nondominationRanks.Length; i++)
        {
            if (nondominationRanks[i] <= lastRankBeforeTiebreak)
                belowIndices.Add(i);
        }

        if (belowIndices.Count < nBelow)
        {
            // Tie-break with the Hypervolume Subset Selection Problem (HSSP).
            System.Diagnostics.Debug.Assert(
                lastRankBeforeTiebreak + 1 < ranks.Length
                && ranks[lastRankBeforeTiebreak + 1] == lastRankBeforeTiebreak + 1);
            var needTiebreak = new List<int>();
            for (int i = 0; i < nondominationRanks.Length; i++)
                if (nondominationRanks[i] == lastRankBeforeTiebreak + 1)
                    needTiebreak.Add(i);

            var rankILvals = needTiebreak.Select(i => lvals[i]).ToArray();
            int subsetSize = nBelow - belowIndices.Count;
            var referencePoint = GetReferencePoint(rankILvals);
            var selected = Hypervolume.SolveHssp(rankILvals, needTiebreak.ToArray(), subsetSize, referencePoint);
            belowIndices.AddRange(selected);
        }

        var belowSet = new HashSet<int>(belowIndices);
        var below = new List<FrozenTrial>();
        var above = new List<FrozenTrial>();
        for (int i = 0; i < trials.Count; i++)
        {
            if (belowSet.Contains(i))
                below.Add(trials[i]);
            else
                above.Add(trials[i]);
        }
        return (below, above);
    }

    /// <summary>Python <c>_get_reference_point(loss_vals)</c> (sampler.py). <c>EPS = 1e-12</c>.</summary>
    /// <param name="lossVals">Objective rows converted to minimization, one solution per row.</param>
    /// <returns>A reference vector worse than the observed objective extremes in the operation's coordinate convention.</returns>
    private static double[] GetReferencePoint(double[][] lossVals)
    {
        int m = lossVals[0].Length;
        var referencePoint = new double[m];
        for (int k = 0; k < m; k++)
        {
            double worst = double.NegativeInfinity;  // np.max(loss_vals, axis=0)[k].
            foreach (var row in lossVals)
                worst = Math.Max(worst, row[k]);
            double r = Math.Max(1.1 * worst, 0.9 * worst);
            referencePoint[k] = r == 0.0 ? 1e-12 : r;
        }
        return referencePoint;
    }
}

/// <summary>
///     The last below/above split a TPE sampler computed, reused while the same trial records are split again (Rustuna
///     plan 13, item S5). Independent TPE (the default for several objectives) splits the history once per parameter, and
///     every parameter of a trial sees the same records, so a 40-parameter trial computed the same split 40 times: about
///     a quarter of a multi-objective run, in the non-domination ranks and the hypervolume tie-break.
/// </summary>
/// <remarks>
///     <para>
///         <b>Why the records themselves are the key.</b> <see cref="TpeTrialSplit.SplitTrials" /> is a function of the
///         records it is given (their state, values, intermediate values and constraints), the study's directions and
///         <c>n_below</c>. A finished record never changes, and a record changes reference whenever its trial is written
///         after a reader got it: storages publish a new record on every state change, and <c>InMemoryStorage</c>
///         copies a handed-out record before writing it. So the same references in the same order, of the same study,
///         with the same <c>n_below</c>, split the same way, and a cached split is returned only then. Anything else
///         (a new trial, another study, a running trial written by another worker under constant liar, a storage that
///         builds new records per read) misses and splits afresh, which is what the sampler did every time before.
///     </para>
///     <para>
///         Not thread-safe: the sampler calls it under its own monitor (THREADING.md §6.3), and a
///         <see cref="SamplerConcurrency.PerWorker" /> worker instance gets a cache of its own.
///     </para>
/// </remarks>
internal sealed class TpeSplitCache
{
    private OptunaSharp.Storages.BaseStorage? _storage;
    private int _studyId;
    private int _nBelow;
    private FrozenTrial[] _trials = Array.Empty<FrozenTrial>();
    private List<FrozenTrial> _below = new();
    private List<FrozenTrial> _above = new();

    /// <summary>How many splits were computed rather than reused: a diagnostic the tests read.</summary>
    internal int Computed { get; private set; }

    /// <summary>
    ///     <see cref="TpeTrialSplit.SplitTrials" /> for these records, reused when the previous call split the very same
    ///     records of the same study with the same <paramref name="nBelow" />.
    /// </summary>
    /// <param name="study">The study the records belong to; its storage and id are part of the key.</param>
    /// <param name="trials">The records to split, in the order the sampler read them.</param>
    /// <param name="nBelow">The number of trials the preferred group should hold.</param>
    /// <returns>New lists the caller may change, holding the preferred and the remaining records by trial number.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="study" /> or <paramref name="trials" /> is null.</exception>
    /// <exception cref="Exception">Whatever <see cref="TpeTrialSplit.SplitTrials" /> throws on a miss; nothing is cached then.</exception>
    internal (List<FrozenTrial> Below, List<FrozenTrial> Above) Split(Study study, IReadOnlyList<FrozenTrial> trials, int nBelow)
    {
        ArgumentNullException.ThrowIfNull(study);
        ArgumentNullException.ThrowIfNull(trials);

        if (!Matches(study, trials, nBelow))
        {
            var (below, above) = TpeTrialSplit.SplitTrials(study, trials, nBelow);
            // Stored only after the split succeeded, so a throwing split leaves the previous entry consistent.
            var key = new FrozenTrial[trials.Count];
            for (int i = 0; i < key.Length; i++)
                key[i] = trials[i];
            _storage = study.Storage;
            _studyId = study.StudyId;
            _nBelow = nBelow;
            _trials = key;
            _below = below;
            _above = above;
            Computed++;
        }

        // Copies: the cached lists must survive a caller that changes what it was given.
        return (new List<FrozenTrial>(_below), new List<FrozenTrial>(_above));
    }

    /// <summary>Whether the cached split was computed from exactly these records of this study with this size.</summary>
    /// <param name="study">The study being sampled.</param>
    /// <param name="trials">The records about to be split.</param>
    /// <param name="nBelow">The requested size of the preferred group.</param>
    /// <returns>True when every key part matches, the records compared by reference and in order.</returns>
    private bool Matches(Study study, IReadOnlyList<FrozenTrial> trials, int nBelow)
    {
        if (_storage is null || !ReferenceEquals(_storage, study.Storage) || _studyId != study.StudyId
            || _nBelow != nBelow || _trials.Length != trials.Count)
            return false;
        for (int i = 0; i < _trials.Length; i++)
        {
            if (!ReferenceEquals(_trials[i], trials[i]))
                return false;
        }
        return true;
    }
}

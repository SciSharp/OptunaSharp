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
    internal static int DefaultGamma(int x) => Math.Min((int)Math.Ceiling(0.1 * x), 25);

    /// <summary>Python <c>default_gamma_multiobjective(x) = ceil(0.1 * x)</c>.</summary>
    internal static int DefaultGammaMultiObjective(int x) => (int)Math.Ceiling(0.1 * x);

    /// <summary>Python <c>hyperopt_default_gamma(x) = min(ceil(0.25 * x**0.5), 25)</c>.</summary>
    internal static int HyperoptDefaultGamma(int x) => Math.Min((int)Math.Ceiling(0.25 * Math.Pow(x, 0.5)), 25);

    /// <summary>Python <c>default_weights(x)</c> — uniform below 25, ramp+flat at and above 25.</summary>
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
    internal static (List<FrozenTrial> Below, List<FrozenTrial> Above) SplitTrials(
        Study study, IReadOnlyList<FrozenTrial> trials, int nBelow)
    {
        var complete = new List<FrozenTrial>();
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

        var below = new List<FrozenTrial>(belowComplete);
        below.AddRange(belowPruned);
        below.AddRange(belowInfeasible);
        var above = new List<FrozenTrial>(aboveComplete);
        above.AddRange(abovePruned);
        above.AddRange(aboveInfeasible);
        above.AddRange(running);

        below.Sort((a, b) => a.Number.CompareTo(b.Number));
        above.Sort((a, b) => a.Number.CompareTo(b.Number));
        return (below, above);
    }

    /// <summary>Python <c>_split_complete_trials(trials, study, n_below)</c>.</summary>
    private static (List<FrozenTrial> Below, List<FrozenTrial> Above) SplitCompleteTrials(
        IReadOnlyList<FrozenTrial> trials, Study study, int nBelow)
    {
        nBelow = Math.Min(nBelow, trials.Count);
        return study.Directions.Count <= 1
            ? SplitCompleteTrialsSingleObjective(trials, study, nBelow)
            : SplitCompleteTrialsMultiObjective(trials, study, nBelow);
    }

    /// <summary>Python <c>_split_complete_trials_single_objective(trials, study, n_below)</c>.</summary>
    private static (List<FrozenTrial> Below, List<FrozenTrial> Above) SplitCompleteTrialsSingleObjective(
        IReadOnlyList<FrozenTrial> trials, Study study, int nBelow)
    {
        // Python's `sorted` is stable; OrderBy/OrderByDescending are too (ties keep input order).
        var sorted = study.Directions[0] == StudyDirection.Minimize
            ? trials.OrderBy(t => t.Value!.Value).ToList()
            : trials.OrderByDescending(t => t.Value!.Value).ToList();
        return (sorted.Take(nBelow).ToList(), sorted.Skip(nBelow).ToList());
    }

    /// <summary>Python <c>_get_pruned_trial_score(trial, study)</c>.</summary>
    private static (double, double) GetPrunedTrialScore(FrozenTrial trial, Study study)
    {
        if (trial.IntermediateValues.Count > 0)
        {
            int step = trial.IntermediateValues.Keys.Max();  // max(dict.items()) compares by key (step).
            double intermediateValue = trial.IntermediateValues[step];
            if (double.IsNaN(intermediateValue))
                return (-step, double.PositiveInfinity);
            if (study.Directions[0] == StudyDirection.Minimize)
                return (-step, intermediateValue);
            return (-step, -intermediateValue);
        }
        return (1, 0.0);
    }

    /// <summary>Python <c>_split_pruned_trials(trials, study, n_below)</c>.</summary>
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

    /// <summary>Python <c>_get_infeasible_trial_score(trial) = sum(v for v in constraints.values() if v > 0)</c>.</summary>
    private static double GetInfeasibleTrialScore(FrozenTrial trial)
    {
        double sum = 0.0;
        foreach (var v in trial.Constraints.Values)
        {
            if (v > 0)
                sum += v;
        }
        return sum;
    }

    /// <summary>Python <c>_split_infeasible_trials(trials, n_below)</c>.</summary>
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

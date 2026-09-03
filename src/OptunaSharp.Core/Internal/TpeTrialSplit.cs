using System;
using System.Collections.Generic;
using System.Linq;

namespace OptunaSharp.Internal;

/// <summary>
///     Port of the trial-splitting helpers in <c>optuna/samplers/_tpe/sampler.py</c> that are
///     shared with <c>PedAnovaImportanceEvaluator</c>. The TPE sampler itself lands in Phase 2.8;
///     this hosts only the pieces already needed. The HSSP tie-break (and its reference-point
///     helper) rides on <c>_hypervolume/*</c>, which arrives in Phase 2.10 — until then the
///     tie-break path throws.
/// </summary>
internal static class TpeTrialSplit
{
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
            // Tie-break with the Hypervolume Subset Selection Problem (HSSP) — Phase 2.10.
            throw new NotImplementedException(
                "Multi-objective trial splitting with an HSSP tie-break is not ported yet " +
                "(PORTING_PLAN Phase 2.10, _hypervolume/*). This path is reached by " +
                "PedAnovaImportanceEvaluator on a multi-objective study when target is null and " +
                "the non-domination ranks do not partition exactly at the quantile boundary. " +
                "Pass an explicit target (e.g. target: t => t.Values[0]) in the meantime.");

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
}

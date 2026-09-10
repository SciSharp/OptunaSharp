using System.Collections.Generic;

namespace OptunaSharp.Samplers.AgeMoea;

/// <summary>
///     Fast non-dominated sorting — port of pymoo's <c>fast_non_dominated_sort</c>
///     (<c>pymoo.util.nds.non_dominated_sorting</c> / <c>fast_non_dominated_sort.py</c>) wrapped by
///     <c>NonDominatedSorting.do</c>. pymoo loads the Cython build at runtime, but its output is a
///     deterministic front <em>partition</em>: within-front order is never used downstream on the
///     AGE path (ranks are assigned per individual and <c>survival_score</c> reindexes by original
///     index), so the standard Deb algorithm below yields identical rank assignments.
///
///     <see cref="Do" /> mirrors the <c>n_stop_if_ranked</c> early stop: fronts are produced only
///     until at least that many individuals are ranked (the front that crosses the threshold is the
///     one the environmental selection cuts).
/// </summary>
internal static class NonDominatedSorting
{
    /// <summary>
    ///     Sort <paramref name="f" /> into Pareto fronts (lists of row indices, best front first),
    ///     stopping once <paramref name="nStopIfRanked" /> individuals have been ranked.
    /// </summary>
    public static List<List<int>> Do(double[][] f, double[]? cv = null, int nStopIfRanked = int.MaxValue)
    {
        int n = f.Length;
        var fronts = new List<List<int>>();
        if (n == 0) return fronts;

        var isDominating = new List<int>[n];
        var nDominated = new int[n];
        for (int i = 0; i < n; i++) isDominating[i] = new List<int>();

        var currentFront = new List<int>();
        for (int i = 0; i < n; i++)
        {
            for (int j = i + 1; j < n; j++)
            {
                int rel = Dominator.GetRelation(
                    f[i], f[j],
                    cv is null ? null : cv[i],
                    cv is null ? null : cv[j]);
                if (rel == 1)
                {
                    isDominating[i].Add(j);
                    nDominated[j]++;
                }
                else if (rel == -1)
                {
                    isDominating[j].Add(i);
                    nDominated[i]++;
                }
            }

            if (nDominated[i] == 0)
                currentFront.Add(i);
        }

        int nRanked = currentFront.Count;
        fronts.Add(currentFront);

        while (nRanked < n && nRanked < nStopIfRanked)
        {
            var nextFront = new List<int>();
            foreach (int i in currentFront)
                foreach (int j in isDominating[i])
                {
                    nDominated[j]--;
                    if (nDominated[j] == 0)
                    {
                        nextFront.Add(j);
                        nRanked++;
                    }
                }

            if (nextFront.Count == 0) break;
            fronts.Add(nextFront);
            currentFront = nextFront;
        }

        return fronts;
    }
}

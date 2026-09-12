// SMS-EMOA environmental survival — "remove the individual with the least hypervolume contribution
// within its worst nondominated-sort front".
//
// Reference: Beume, Naujoks & Emmerich, "SMS-EMOA: Multiobjective selection based on dominated
// hypervolume", European Journal of Operational Research 181(3), 2007; survival details follow
// pymoo's `LeastHypervolumeContributionSurvival` (pymoo/algorithms/moo/sms.py, Apache-2.0). SMS-EMOA
// is NOT part of Optuna → this is a native port validated by convergence/HV, not bit-matched to an
// upstream (the `LmCmaSampler` / `JanusSampler` model).

using System;
using System.Collections.Generic;
using System.Linq;
using OptunaSharp.Internal;

namespace OptunaSharp.Samplers.SmsEmoa;

/// <summary>
///     The least-hypervolume-contribution survival operator shared by <see cref="SmsEmoaSampler" /> and
///     its unit tests. Given a merged population in the <b>minimization</b> objective convention, it
///     nondominated-sorts, keeps whole fronts while they fit, and trims the single overflowing front by
///     repeatedly dropping the individual whose <i>exclusive</i> hypervolume contribution is least
///     (<c>HV(front) − HV(front∖{i})</c>) — pymoo's <c>hv.hvc.argmin()</c>. Objectives are normalized to
///     the supplied <c>ideal</c>/<c>nadir</c> box with the reference point shifted to <c>1 + eps</c>
///     (pymoo default <c>eps = 10</c>), so the survival ranking is on the population's own scale.
/// </summary>
internal static class SmsEmoaSurvival
{
    /// <summary>pymoo's default reference-point offset (<c>ref = 1 + eps</c> in normalized space).</summary>
    internal const double DefaultEps = 10.0;

    /// <summary>
    ///     Select <paramref name="nSurvive" /> survivor indices out of <paramref name="mergedLoss" />
    ///     (minimization convention: lower is better on every objective). <paramref name="ideal" /> and
    ///     <paramref name="nadir" /> are the per-objective normalization bounds (in SMS-EMOA, estimated
    ///     from the pre-merge parent population, as pymoo's <c>_advance</c> does). Returns the surviving
    ///     row indices, best fronts first, in ascending-index order within a front.
    /// </summary>
    /// <param name="mergedLoss">Minimization-oriented objective rows of the merged parent and offspring population.</param>
    /// <param name="nSurvive">Maximum number of individuals to retain.</param>
    /// <param name="ideal">Ideal objective vector used as the normalization origin.</param>
    /// <param name="nadir">Upper objective normalization reference.</param>
    /// <param name="eps">Positive padding used to place the hypervolume reference beyond the observed front.</param>
    /// <returns>Indices of the surviving rows in the merged objective matrix.</returns>
    internal static int[] Select(
        double[][] mergedLoss, int nSurvive, double[] ideal, double[] nadir, double eps = DefaultEps)
    {
        int n = mergedLoss.Length;
        if (nSurvive <= 0) return Array.Empty<int>();
        if (nSurvive >= n) return Enumerable.Range(0, n).ToArray();

        int m = mergedLoss[0].Length;
        int[] ranks = MultiObjective.FastNonDominationRank(mergedLoss);

        // Group row indices by nondomination rank (0 = best front), fronts visited best-first.
        var byRank = new SortedDictionary<int, List<int>>();
        for (int i = 0; i < n; i++)
        {
            if (!byRank.TryGetValue(ranks[i], out var list))
            {
                list = new List<int>();
                byRank[ranks[i]] = list;
            }
            list.Add(i);
        }

        var survivors = new List<int>();
        foreach (var front in byRank.Values)
        {
            if (survivors.Count + front.Count <= nSurvive)
            {
                survivors.AddRange(front);
                if (survivors.Count == nSurvive) break;
                continue;
            }
            // The one overflowing front: keep only the top-HV-contribution members.
            int keep = nSurvive - survivors.Count;
            survivors.AddRange(TrimByHypervolume(front, mergedLoss, ideal, nadir, m, eps, keep));
            break;
        }
        return survivors.ToArray();
    }

    /// <summary>
    ///     Keep <paramref name="keep" /> of <paramref name="front" /> by repeatedly removing the
    ///     least-exclusive-HV-contribution individual (pymoo: <c>while ...: k = hv.hvc.argmin();
    ///     hv.delete(k)</c>). One objective has no hypervolume, so there the worst-value points are
    ///     dropped directly.
    /// </summary>
    /// <param name="front">Objective vectors of one nondominated front.</param>
    /// <param name="loss">Minimization-oriented objective coordinate values.</param>
    /// <param name="ideal">Ideal objective vector used as the normalization origin.</param>
    /// <param name="nadir">Upper objective normalization reference.</param>
    /// <param name="m">Number of objective or residual components.</param>
    /// <param name="eps">Positive padding used to place the hypervolume reference beyond the observed front.</param>
    /// <param name="keep">Number of split-front individuals to retain.</param>
    /// <returns>Indices retained after repeatedly removing the least hypervolume contributor.</returns>
    private static List<int> TrimByHypervolume(
        List<int> front, double[][] loss, double[] ideal, double[] nadir, int m, double eps, int keep)
    {
        var current = new List<int>(front);
        if (m == 1)
        {
            // No hypervolume in 1-D: the least "contribution" is the largest (worst) objective value.
            current.Sort((a, b) => loss[a][0].CompareTo(loss[b][0]));   // ascending (best first)
            return current.Take(keep).ToList();
        }

        var refPoint = new double[m];
        for (int j = 0; j < m; j++) refPoint[j] = 1.0 + eps;

        while (current.Count > keep)
        {
            var norm = new double[current.Count][];
            for (int i = 0; i < current.Count; i++)
                norm[i] = Normalize(loss[current[i]], ideal, nadir, refPoint);

            double hvAll = Hypervolume.ComputeHypervolume(norm, refPoint);
            int worst = 0;
            double minContribution = double.PositiveInfinity;
            for (int i = 0; i < current.Count; i++)
            {
                double hvWithout = Hypervolume.ComputeHypervolume(Without(norm, i), refPoint);
                double contribution = hvAll - hvWithout;
                if (contribution < minContribution)
                {
                    minContribution = contribution;
                    worst = i;
                }
            }
            current.RemoveAt(worst);
        }
        return current;
    }

    /// <summary>
    ///     pymoo <c>normalize(F, ideal, nadir)</c> = <c>(F − ideal) / (nadir − ideal)</c>, then clamped
    ///     to the reference point so a point outside the reference region satisfies the hypervolume
    ///     module's "all points dominate the reference" contract and simply contributes nothing there.
    /// </summary>
    /// <param name="loss">Minimization-oriented objective coordinate values.</param>
    /// <param name="ideal">Ideal objective vector used as the normalization origin.</param>
    /// <param name="nadir">Upper objective normalization reference.</param>
    /// <param name="refPoint">Hypervolume reference coordinates used for normalization.</param>
    /// <returns>The objective loss normalized between the ideal and reference points.</returns>
    private static double[] Normalize(double[] loss, double[] ideal, double[] nadir, double[] refPoint)
    {
        var r = new double[loss.Length];
        for (int j = 0; j < loss.Length; j++)
        {
            double denom = nadir[j] - ideal[j];
            double v = denom > 0 ? (loss[j] - ideal[j]) / denom : 0.0;
            r[j] = Math.Min(v, refPoint[j]);
        }
        return r;
    }

    /// <summary>Copies all row references except the row being excluded from a hypervolume calculation.</summary>
    /// <param name="rows">Observation rows to copy into the result.</param>
    /// <param name="skip">Zero-based row omitted from the result.</param>
    /// <returns>The retained row references in their original order.</returns>
    private static double[][] Without(double[][] rows, int skip)
    {
        var result = new double[rows.Length - 1][];
        int w = 0;
        for (int i = 0; i < rows.Length; i++)
            if (i != skip) result[w++] = rows[i];
        return result;
    }
}

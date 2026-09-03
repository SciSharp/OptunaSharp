using System;
using System.Collections.Generic;
using System.Linq;
using NumSharp;
using OptunaSharp.Internal;
using OptunaSharp.Logging;

namespace OptunaSharp.Visualization;

/// <summary>
///     The data backing a hypervolume-history plot: the running hypervolume after each completed
///     trial. Port of <c>_HypervolumeHistoryInfo</c> in
///     <c>optuna/visualization/_hypervolume_history.py</c>.
/// </summary>
public sealed record HypervolumeHistoryInfo(
    IReadOnlyList<int> TrialNumbers,
    IReadOnlyList<double> Values);

/// <summary>
///     Data layer for <c>optuna.visualization.plot_hypervolume_history</c> — the pure
///     <c>_get_hypervolume_history_info</c> half of <c>_hypervolume_history.py</c> (with the two
///     argument validations that live in <c>plot_hypervolume_history</c> folded into
///     <see cref="GetInfo" /> so the data-layer entry is self-checking). The experimental-function
///     warning is a Tier X rendering concern and is not emitted here.
/// </summary>
public static class HypervolumeHistory
{
    private static readonly Logger Logger =
        OptunaLogging.GetLogger("optuna.visualization._hypervolume_history");

    /// <summary>
    ///     Python <c>_get_hypervolume_history_info(study, reference_point)</c>, guarded by the
    ///     validations from <c>plot_hypervolume_history</c>.
    /// </summary>
    public static HypervolumeHistoryInfo GetInfo(Study study, IReadOnlyList<double> referencePoint)
    {
        if (!study.IsMultiObjective())
            throw new ValueError(
                "Study must be multi-objective. For single-objective optimization, " +
                "please use plot_optimization_history instead.");

        if (referencePoint.Count != study.Directions.Count)
            throw new ValueError(
                "The dimension of the reference point must be the same as the number of objectives.");

        var completedTrials = study.GetTrials(deepcopy: false, states: new[] { TrialState.Complete });

        if (completedTrials.Count == 0)
            Logger.Warning("Your study does not have any completed trials.");

        // Our hypervolume computation module assumes that all objectives are minimized.
        // Here we transform the objective values and the reference point.
        var directions = study.Directions;
        int m = directions.Count;
        var signs = new double[m];
        for (int j = 0; j < m; j++)
            signs[j] = directions[j] == StudyDirection.Minimize ? 1.0 : -1.0;

        var minimizationReferencePoint = new double[m];
        for (int j = 0; j < m; j++)
            minimizationReferencePoint[j] = signs[j] * referencePoint[j];

        // Only feasible trials are considered in hypervolume computation.
        var trialNumbers = new List<int>();
        var hypervolumeValues = new List<double>();
        var bestTrialsValuesNormalized = new List<double[]>();  // The current Pareto set (empty == None).
        bool bestInitialized = false;
        double hypervolume = 0.0;

        foreach (var trial in completedTrials)
        {
            trialNumbers.Add(trial.Number);

            if (trial.Constraints.Values.Any(x => x > 0.0))
            {
                // The trial is infeasible.
                hypervolumeValues.Add(hypervolume);
                continue;
            }

            var valuesNormalized = new double[m];
            for (int j = 0; j < m; j++)
                valuesNormalized[j] = signs[j] * trial.Values![j];

            if (bestInitialized)
            {
                // (best <= values_normalized).all(axis=1).any(axis=0): the trial is (weakly) dominated.
                bool dominated = bestTrialsValuesNormalized.Any(
                    p => AllLessOrEqual(p, valuesNormalized));
                if (dominated)
                {
                    hypervolumeValues.Add(hypervolume);
                    continue;
                }
            }

            // (values_normalized > minimization_reference_point).any(): out of the reference region.
            bool outOfRegion = false;
            for (int j = 0; j < m; j++)
                if (valuesNormalized[j] > minimizationReferencePoint[j])
                {
                    outOfRegion = true;
                    break;
                }
            if (outOfRegion)
            {
                hypervolumeValues.Add(hypervolume);
                continue;
            }

            hypervolume += Prod(minimizationReferencePoint, valuesNormalized);
            if (!bestInitialized)
            {
                bestTrialsValuesNormalized.Add(valuesNormalized);
                bestInitialized = true;
            }
            else
            {
                // limited_sols = np.maximum(best, values_normalized)
                var limitedSols = bestTrialsValuesNormalized
                    .Select(p => Maximum(p, valuesNormalized))
                    .ToArray();
                hypervolume -= Hypervolume.ComputeHypervolume(limitedSols, minimizationReferencePoint);

                // is_kept = (best < values_normalized).any(axis=1); keep those, then append the new point.
                var kept = bestTrialsValuesNormalized
                    .Where(p => AnyLess(p, valuesNormalized))
                    .ToList();
                kept.Add(valuesNormalized);
                bestTrialsValuesNormalized = kept;
            }
            hypervolumeValues.Add(hypervolume);
        }

        if (!bestInitialized)
            Logger.Warning("Your study does not have any feasible trials.");

        return new HypervolumeHistoryInfo(trialNumbers, hypervolumeValues);
    }

    private static bool AllLessOrEqual(double[] a, double[] b)
    {
        for (int j = 0; j < a.Length; j++)
            if (!(a[j] <= b[j]))
                return false;
        return true;
    }

    private static bool AnyLess(double[] a, double[] b)
    {
        for (int j = 0; j < a.Length; j++)
            if (a[j] < b[j])
                return true;
        return false;
    }

    /// <summary>NumPy <c>np.prod(ref - values)</c> — a sequential product (M ≤ 8, so no pairwise).</summary>
    private static double Prod(double[] reference, double[] values)
    {
        double p = 1.0;  // 1.0 * x == x exactly, so this matches multiply.reduce starting from ref[0]-values[0].
        for (int j = 0; j < reference.Length; j++)
            p *= reference[j] - values[j];
        return p;
    }

    private static double[] Maximum(double[] a, double[] b)
    {
        var r = new double[a.Length];
        for (int j = 0; j < a.Length; j++)
            r[j] = Math.Max(a[j], b[j]);
        return r;
    }
}

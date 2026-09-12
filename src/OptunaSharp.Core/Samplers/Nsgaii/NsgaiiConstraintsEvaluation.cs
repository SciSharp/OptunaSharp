using System.Collections.Generic;
using System.Linq;
using NumSharp;
using OptunaSharp.Internal;

namespace OptunaSharp.Samplers.Nsgaii;

/// <summary>
///     Port of <c>optuna/samplers/nsgaii/_constraints_evaluation.py</c>. Deterministic
///     constraint-domination logic (no RNG).
/// </summary>
internal static class NsgaiiConstraintsEvaluation
{
    /// <summary>
    ///     Python <c>_constrained_dominates</c>. A trial x constrained-dominates y iff:
    ///     (1) x is feasible and y is not; (2) both infeasible but x has smaller overall violation;
    ///     or (3) both feasible and x dominates y.
    /// </summary>
    /// <param name="trial0">First trial in the dominance comparison.</param>
    /// <param name="trial1">Second trial in the dominance comparison.</param>
    /// <param name="directions">One optimization direction per objective.</param>
    /// <returns>True when the first trial dominates under feasibility-first comparison.</returns>
    internal static bool ConstrainedDominates(
        FrozenTrial trial0, FrozenTrial trial1, IReadOnlyList<StudyDirection> directions)
    {
        var constraints0 = trial0.Constraints;
        var constraints1 = trial1.Constraints;

        if (trial0.State != TrialState.Complete)
            return false;

        if (trial1.State != TrialState.Complete)
            return true;

        bool satisfyConstraints0 = constraints0.Values.All(v => v <= 0);
        bool satisfyConstraints1 = constraints1.Values.All(v => v <= 0);

        if (satisfyConstraints0 && satisfyConstraints1)
            // Both trials satisfy the constraints.
            return MultiObjective.Dominates(trial0, trial1, directions);

        if (satisfyConstraints0)
            // trial0 satisfies the constraints, but trial1 violates them.
            return true;

        if (satisfyConstraints1)
            // trial1 satisfies the constraints, but trial0 violates them.
            return false;

        // Both trials violate the constraints.
        double violation0 = constraints0.Values.Where(v => v > 0).Sum();
        double violation1 = constraints1.Values.Where(v => v > 0).Sum();
        return violation0 < violation1;
    }

    /// <summary>Python <c>_evaluate_penalty</c>: per-trial sum of positive constraint values.</summary>
    /// <param name="population">Individuals available for survival or reproduction.</param>
    /// <returns>Summed positive constraint violations for each trial.</returns>
    /// <exception cref="NumSharp.ValueError">A trial constraint is NaN.</exception>
    internal static double[] EvaluatePenalty(IReadOnlyList<FrozenTrial> population)
    {
        var penalty = new double[population.Count];
        for (int i = 0; i < population.Count; i++)
            penalty[i] = population[i].Constraints.Values.Where(v => v > 0).Sum();
        return penalty;
    }

    /// <summary>Python <c>_validate_constraints</c>.</summary>
    /// <param name="population">Individuals available for survival or reproduction.</param>
    /// <param name="isConstrained">Whether constraint feasibility participates in ranking.</param>
    internal static void ValidateConstraints(IReadOnlyList<FrozenTrial> population, bool isConstrained)
    {
        if (!isConstrained)
            return;

        foreach (var trial in population)
        {
            if (trial.Constraints.Values.Any(double.IsNaN))
                throw new ValueError("NaN is not acceptable as constraint value.");
        }
    }
}

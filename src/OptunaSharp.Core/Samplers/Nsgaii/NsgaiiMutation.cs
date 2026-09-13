using NumSharp;
using OptunaSharp.Collections;
using OptunaSharp.Distributions;
using OptunaSharp.Internal;

namespace OptunaSharp.Samplers.Nsgaii;

/// <summary>
///     Port of <c>optuna/samplers/nsgaii/_mutation.py: perform_mutation</c>. Applies a mutation to a
///     single numerical parameter in the transformed space and clips it back into range; returns
///     <c>null</c> for categorical parameters (which are then resampled independently).
/// </summary>
internal static class NsgaiiMutation
{
    /// <summary>Transforms one parameter, applies numerical mutation, and restores its external representation.</summary>
    /// <param name="mutation">Numerical mutation operator used when an offspring coordinate is selected for mutation.</param>
    /// <param name="rng">Random stream whose draw order is shared with the reference algorithm.</param>
    /// <param name="study">Study supplying trial history, objective directions, and storage.</param>
    /// <param name="distribution">Distribution defining the coordinate being transformed or mutated.</param>
    /// <param name="value">External parameter value before mutation.</param>
    /// <returns>The mutated external parameter value.</returns>
    internal static object? PerformMutation(
        BaseMutation mutation, NumPyRandom rng, Study study, BaseDistribution distribution, object? value)
    {
        if (distribution is not (FloatDistribution or IntDistribution))
            return null;

        var transform = new SearchSpaceTransform(new OrderedDict<string, BaseDistribution> { [""] = distribution });
        double transValue = transform.Transform(new OrderedDict<string, object?> { [""] = value }).ToArray<double>()[0];

        var bounds = transform.RawBounds;
        double transMutatedValue = mutation.Mutation(
            transValue, rng, study, new[] { bounds[0, 0], bounds[0, 1] });
        transMutatedValue = MathUtil.Clip(transMutatedValue, bounds[0, 0], bounds[0, 1]);

        // View the single mutated value zero-copy; Untransform copies it out, so dispose here.
        using NDArray nd = np.array(new[] { transMutatedValue }, copy: false);
        return transform.Untransform(nd)[""];
    }
}

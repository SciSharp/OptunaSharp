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

        return transform.Untransform(np.array(new[] { transMutatedValue }))[""];
    }
}

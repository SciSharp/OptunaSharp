using NumSharp;

namespace OptunaSharp.Samplers.Nsgaii;

/// <summary>
///     Base class for mutations. Port of
///     <c>optuna/samplers/nsgaii/_mutations/_base.py: BaseMutation</c> (experimental v5.0.0).
///
///     A mutation operation is used by <see cref="NSGAIISampler" /> to mutate a numerical parameter
///     when creating a new individual.
/// </summary>
public abstract class BaseMutation
{
    /// <inheritdoc />
    public override string ToString() => GetType().Name;

    /// <summary>Mutate the given parameter.</summary>
    /// <param name="param">A parameter value in the transformed numerical search space.</param>
    /// <param name="rng">An instance of <c>numpy.random.RandomState</c>.</param>
    /// <param name="study">Target study object.</param>
    /// <param name="searchSpaceBounds">
    ///     A length-2 array <c>[lower, upper]</c> of numerical distribution bounds from the
    ///     transformed search space.
    /// </param>
    /// <returns>A mutated parameter value in the transformed numerical search space.</returns>
    public abstract double Mutation(double param, NumPyRandom rng, Study study, double[] searchSpaceBounds);
}

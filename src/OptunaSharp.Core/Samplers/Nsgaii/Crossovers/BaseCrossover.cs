using NumSharp;

namespace OptunaSharp.Samplers.Nsgaii;

/// <summary>
///     Base class for crossovers. Port of
///     <c>optuna/samplers/nsgaii/_crossovers/_base.py: BaseCrossover</c>.
///
///     A crossover operation is used by <see cref="NSGAIISampler" /> to create new parameter
///     combinations from parameters of <c>n</c> parent individuals.
///
///     Concrete implementations of this class are expected to only accept parameters from
///     numerical distributions. Only crossover for categorical parameters (uniform crossover) is
///     built into <see cref="NSGAIISampler" />.
/// </summary>
public abstract class BaseCrossover
{
    /// <inheritdoc />
    public override string ToString() => GetType().Name;

    /// <summary>Number of parent individuals required to perform crossover.</summary>
    public abstract int NParents { get; }

    /// <summary>
    ///     Perform crossover of selected parent individuals.
    /// </summary>
    /// <param name="parentsParams">
    ///     A <c>num_parents x num_parameters</c> array. Represents a parameter space for each
    ///     parent individual (continuous for numerical parameters). Row <c>i</c> is parent <c>i</c>.
    /// </param>
    /// <param name="rng">An instance of <c>numpy.random.RandomState</c>.</param>
    /// <param name="study">Target study object.</param>
    /// <param name="searchSpaceBounds">
    ///     A <c>len_search_space x 2</c> array of numerical distribution bounds constructed from the
    ///     transformed search space. Column 0 is lower, column 1 is upper.
    /// </param>
    /// <returns>A 1-dimensional array containing the new parameter combination.</returns>
    public abstract double[] Crossover(
        double[][] parentsParams, NumPyRandom rng, Study study, double[,] searchSpaceBounds);
}

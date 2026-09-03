using NumSharp;
using OptunaSharp.Internal;

namespace OptunaSharp.Samplers.Nsgaii;

/// <summary>
///     Uniform Crossover operation used by <see cref="NSGAIISampler" />. Port of
///     <c>optuna/samplers/nsgaii/_crossovers/_uniform.py: UniformCrossover</c>.
///
///     Select each parameter with equal probability from the two parent individuals.
/// </summary>
public class UniformCrossover : BaseCrossover
{
    private readonly double _swappingProb;

    /// <param name="swappingProb">Probability of swapping each parameter of the parents during crossover.</param>
    public UniformCrossover(double swappingProb = 0.5)
    {
        if (!(0.0 <= swappingProb && swappingProb <= 1.0))
            throw new ValueError("`swapping_prob` must be a float value within the range [0.0, 1.0].");
        _swappingProb = swappingProb;
    }

    public override int NParents => 2;

    public override double[] Crossover(
        double[][] parentsParams, NumPyRandom rng, Study study, double[,] searchSpaceBounds)
    {
        int nParams = searchSpaceBounds.GetLength(0);
        // masks = (rng.rand(n_params) >= swapping_prob).astype(int)
        // child_params = parents_params[masks, range(n_params)]
        var r = rng.rand(nParams).ToArray<double>();
        var child = new double[nParams];
        for (int j = 0; j < nParams; j++)
        {
            int mask = r[j] >= _swappingProb ? 1 : 0;
            child[j] = parentsParams[mask][j];
        }
        return child;
    }
}

using System;
using NumSharp;

namespace OptunaSharp.Samplers.Nsgaii;

/// <summary>
///     Blend Crossover operation used by <see cref="NSGAIISampler" />. Port of
///     <c>optuna/samplers/nsgaii/_crossovers/_blxalpha.py: BLXAlphaCrossover</c> (experimental v3.0.0).
///
///     Uniformly samples child individuals from the hyper-rectangles created by the two parent
///     individuals.
/// </summary>
public class BLXAlphaCrossover : BaseCrossover
{
    private readonly double _alpha;

    /// <param name="alpha">Parametrizes blend operation.</param>
    /// <summary>Configures how far uniform offspring intervals extend beyond the two parent coordinates.</summary>
    public BLXAlphaCrossover(double alpha = 0.5)
    {
        _alpha = alpha;
    }

    /// <inheritdoc />
    public override int NParents => 2;

    /// <inheritdoc />
    public override double[] Crossover(
        double[][] parentsParams, NumPyRandom rng, Study study, double[,] searchSpaceBounds)
    {
        int n = searchSpaceBounds.GetLength(0);
        var r = rng.rand(n).ToArray<double>();
        var child = new double[n];
        for (int j = 0; j < n; j++)
        {
            double parentsMin = Math.Min(parentsParams[0][j], parentsParams[1][j]);
            double parentsMax = Math.Max(parentsParams[0][j], parentsParams[1][j]);
            double diff = _alpha * (parentsMax - parentsMin);  // Equation (1).
            double low = parentsMin - diff;                    // Equation (1).
            double high = parentsMax + diff;                   // Equation (1).
            child[j] = (high - low) * r[j] + low;
        }
        return child;
    }
}

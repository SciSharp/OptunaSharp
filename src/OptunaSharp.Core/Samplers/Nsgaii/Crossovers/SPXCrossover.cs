using System;
using NumSharp;
using OptunaSharp.Internal;

namespace OptunaSharp.Samplers.Nsgaii;

/// <summary>
///     Simplex Crossover operation used by <see cref="NSGAIISampler" />. Port of
///     <c>optuna/samplers/nsgaii/_crossovers/_spx.py: SPXCrossover</c> (experimental v3.0.0).
///
///     Uniformly samples child individuals from within a single simplex similar to the simplex
///     produced by the (three) parent individuals.
/// </summary>
public class SPXCrossover : BaseCrossover
{
    private readonly double? _epsilon;

    /// <param name="epsilon">Expansion rate. If unset, defaults to <c>sqrt(len(search_space) + 2)</c>.</param>
    /// <summary>Configures expansion of the simplex formed by three numerical parent vectors.</summary>
    public SPXCrossover(double? epsilon = null)
    {
        _epsilon = epsilon;
    }

    /// <inheritdoc />
    public override int NParents => 3;

    /// <inheritdoc />
    public override double[] Crossover(
        double[][] parentsParams, NumPyRandom rng, Study study, double[,] searchSpaceBounds)
    {
        int nParents = NParents;
        int nParams = searchSpaceBounds.GetLength(0);
        int n = nParents - 1;

        // G = np.mean(parents_params, axis=0)  (Equation (1)).
        var g = new double[nParams];
        for (int c = 0; c < nParams; c++)
        {
            double sum = parentsParams[0][c];
            for (int k = 1; k < nParents; k++)
                sum += parentsParams[k][c];
            g[c] = sum / nParents;
        }

        // rs = np.power(rng.rand(n), 1 / (np.arange(n) + 1))  (Equation (2)).
        var rand = NumPyDraws.Rand(rng, n);
        var rs = new double[n];
        for (int i = 0; i < n; i++)
            rs[i] = Math.Pow(rand[i], 1.0 / (i + 1));

        double epsilon = _epsilon ?? Math.Sqrt(nParams + 2);

        // xks = [G + epsilon * (pk - G) for pk in parents_params]  (Equation (3)).
        var xks = new double[nParents][];
        for (int k = 0; k < nParents; k++)
        {
            xks[k] = new double[nParams];
            for (int c = 0; c < nParams; c++)
                xks[k][c] = g[c] + epsilon * (parentsParams[k][c] - g[c]);
        }

        // ck = 0; for k in 1..n_parents: ck = rs[k-1] * (xks[k-1] - xks[k] + ck)  (Equation (4)).
        var ck = new double[nParams];
        for (int k = 1; k < nParents; k++)
        {
            for (int c = 0; c < nParams; c++)
                ck[c] = rs[k - 1] * (xks[k - 1][c] - xks[k][c] + ck[c]);
        }

        // child_params = xks[-1] + ck  (Equation (5)).
        var child = new double[nParams];
        for (int c = 0; c < nParams; c++)
            child[c] = xks[nParents - 1][c] + ck[c];
        return child;
    }
}

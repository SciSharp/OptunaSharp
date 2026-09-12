using System;
using NumSharp;

namespace OptunaSharp.Samplers.Nsgaii;

/// <summary>
///     Modified Simulated Binary Crossover (vSBX) used by <see cref="NSGAIISampler" />. Port of
///     <c>optuna/samplers/nsgaii/_crossovers/_vsbx.py: VSBXCrossover</c> (experimental v3.0.0).
///
///     vSBX generates child individuals without excluding any region of the parameter space while
///     maintaining the properties of SBX.
/// </summary>
public class VSBXCrossover : BaseCrossover
{
    private readonly double? _eta;
    private readonly double _uniformCrossoverProb;
    private readonly double _useChildGeneProb;

    /// <param name="eta">Distribution index. Defaults to 2 for single-objective, 20 for multi-objective.</param>
    /// <param name="uniformCrossoverProb">Probability of uniform crossover between the two candidate children.</param>
    /// <param name="useChildGeneProb">Per-variable probability of using the generated child value over the parent value.</param>
    /// <summary>Configures variable-wise simulated-binary crossover spread and child-coordinate selection.</summary>
    /// <exception cref="NumSharp.ValueError">The distribution index is negative, uniform crossover probability is outside [0, 1], or child-gene probability is outside (0, 1].</exception>
    public VSBXCrossover(double? eta = null, double uniformCrossoverProb = 0.5, double useChildGeneProb = 0.5)
    {
        if (eta is not null && eta < 0.0)
            throw new ValueError("The value of `eta` must be greater than or equal to 0.0.");
        _eta = eta;

        if (uniformCrossoverProb < 0.0 || uniformCrossoverProb > 1.0)
            throw new ValueError("The value of `uniform_crossover_prob` must be in the range [0.0, 1.0].");
        if (useChildGeneProb <= 0.0 || useChildGeneProb > 1.0)
            throw new ValueError("The value of `use_child_gene_prob` must be in the range (0.0, 1.0].");
        _uniformCrossoverProb = uniformCrossoverProb;
        _useChildGeneProb = useChildGeneProb;
    }

    /// <inheritdoc />
    public override int NParents => 2;

    /// <inheritdoc />
    public override double[] Crossover(
        double[][] parentsParams, NumPyRandom rng, Study study, double[,] searchSpaceBounds)
    {
        int n = searchSpaceBounds.GetLength(0);
        double eta = _eta ?? (study.IsMultiObjective() ? 20.0 : 2.0);

        const double eps = 1e-10;
        var us = rng.rand(n).ToArray<double>();
        var beta1 = new double[n];
        var beta2 = new double[n];
        for (int j = 0; j < n; j++)
        {
            beta1[j] = Math.Pow(1 / Math.Max(2 * us[j], eps), 1 / (eta + 1));
            beta2[j] = Math.Pow(1 / Math.Max(2 * (1 - us[j]), eps), 1 / (eta + 1));
        }

        var p0 = parentsParams[0];
        var p1 = parentsParams[1];

        var c1 = new double[n];
        double u1 = (double)rng.rand();
        for (int j = 0; j < n; j++)
        {
            c1[j] = u1 <= 0.5
                ? 0.5 * ((1 + beta1[j]) * p0[j] + (1 - beta2[j]) * p1[j])
                : 0.5 * ((1 - beta1[j]) * p0[j] + (1 + beta2[j]) * p1[j]);
        }

        var c2 = new double[n];
        double u2 = (double)rng.rand();
        for (int j = 0; j < n; j++)
        {
            c2[j] = u2 <= 0.5
                ? 0.5 * ((3 - beta1[j]) * p0[j] - (1 - beta2[j]) * p1[j])
                : 0.5 * (-(1 - beta1[j]) * p0[j] + (3 - beta2[j]) * p1[j]);
        }

        return SbxLike.CollapseTwoChildren(c1, c2, p0, p1, rng, _useChildGeneProb, _uniformCrossoverProb);
    }
}

using System;
using NumSharp;

namespace OptunaSharp.Samplers.Nsgaii;

/// <summary>
///     Simulated Binary Crossover operation used by <see cref="NSGAIISampler" />. Port of
///     <c>optuna/samplers/nsgaii/_crossovers/_sbx.py: SBXCrossover</c> (experimental v3.0.0).
///
///     Generates a child from two parent individuals according to the polynomial probability
///     distribution. Optuna returns only one child, so it uses <c>uniform_crossover_prob</c> and
///     <c>use_child_gene_prob</c> to collapse the two SBX children into one.
/// </summary>
public class SBXCrossover : BaseCrossover
{
    private readonly double? _eta;
    private readonly double _uniformCrossoverProb;
    private readonly double _useChildGeneProb;

    /// <param name="eta">Distribution index. Defaults to 2 for single-objective, 20 for multi-objective.</param>
    /// <param name="uniformCrossoverProb">Probability of uniform crossover between the two candidate children.</param>
    /// <param name="useChildGeneProb">Per-variable probability of using the generated child value over the parent value.</param>
    public SBXCrossover(double? eta = null, double uniformCrossoverProb = 0.5, double useChildGeneProb = 0.5)
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

    public override int NParents => 2;

    public override double[] Crossover(
        double[][] parentsParams, NumPyRandom rng, Study study, double[,] searchSpaceBounds)
    {
        int n = searchSpaceBounds.GetLength(0);
        double eta = _eta ?? (study.IsMultiObjective() ? 20.0 : 2.0);

        var us = rng.rand(n).ToArray<double>();

        var c1 = new double[n];
        var c2 = new double[n];
        for (int j = 0; j < n; j++)
        {
            double xls = searchSpaceBounds[j, 0];
            double xus = searchSpaceBounds[j, 1];
            double xsMin = Math.Min(parentsParams[0][j], parentsParams[1][j]);
            double xsMax = Math.Max(parentsParams[0][j], parentsParams[1][j]);

            double xsDiff = Math.Max(xsMax - xsMin, 1e-10);  // np.clip(..., 1e-10, None).
            double beta1 = 1 + 2 * (xsMin - xls) / xsDiff;
            double beta2 = 1 + 2 * (xus - xsMax) / xsDiff;
            double alpha1 = 2 - Math.Pow(beta1, -(eta + 1));
            double alpha2 = 2 - Math.Pow(beta2, -(eta + 1));

            double u = us[j];
            double betaq1 = u > 1 / alpha1  // Equation (3).
                ? Math.Pow(1 / (2 - u * alpha1), 1 / (eta + 1))
                : Math.Pow(u * alpha1, 1 / (eta + 1));
            double betaq2 = u > 1 / alpha2  // Equation (3).
                ? Math.Pow(1 / (2 - u * alpha2), 1 / (eta + 1))
                : Math.Pow(u * alpha2, 1 / (eta + 1));

            c1[j] = 0.5 * ((xsMin + xsMax) - betaq1 * xsDiff);  // Equation (4).
            c2[j] = 0.5 * ((xsMin + xsMax) + betaq2 * xsDiff);  // Equation (5).
        }

        var child = SbxLike.CollapseTwoChildren(
            c1, c2, parentsParams[0], parentsParams[1], rng, _useChildGeneProb, _uniformCrossoverProb);
        return child;
    }
}

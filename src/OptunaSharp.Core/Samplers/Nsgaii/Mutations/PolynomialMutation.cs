using System;
using NumSharp;

namespace OptunaSharp.Samplers.Nsgaii;

/// <summary>
///     Polynomial mutation operation used by <see cref="NSGAIISampler" />. Port of
///     <c>optuna/samplers/nsgaii/_mutations/_polynomial.py: PolynomialMutation</c> (experimental v5.0.0).
///
///     Mutates a real-valued parameter according to the polynomial probability distribution,
///     following revision 1.1.6 of the original NSGA-II C implementation.
/// </summary>
public class PolynomialMutation : BaseMutation
{
    private readonly double _eta;

    /// <param name="eta">Distribution index. Larger values keep mutated values closer to the original.</param>
    /// <summary>Configures the polynomial perturbation distribution applied to bounded coordinates.</summary>
    /// <exception cref="NumSharp.ValueError">The polynomial distribution index is negative.</exception>
    public PolynomialMutation(double eta = 20.0)
    {
        if (eta < 0)
            throw new ValueError("`eta` must be a non-negative float value.");
        _eta = eta;
    }

    /// <inheritdoc />
    public override double Mutation(double param, NumPyRandom rng, Study study, double[] searchSpaceBounds)
    {
        double u = (double)rng.rand();
        double lb = searchSpaceBounds[0];
        double ub = searchSpaceBounds[1];
        double width = ub - lb;

        if (width <= 0.0)
            return param;

        double delta1 = (param - lb) / width;
        double delta2 = (ub - param) / width;
        double mutationPower = 1.0 / (_eta + 1.0);

        double deltaQ;
        if (u <= 0.5)
        {
            double xy = 1.0 - delta1;
            double value = 2.0 * u + (1.0 - 2.0 * u) * Math.Pow(xy, _eta + 1.0);
            deltaQ = Math.Pow(value, mutationPower) - 1.0;
        }
        else
        {
            double xy = 1.0 - delta2;
            double value = 2.0 * (1.0 - u) + 2.0 * (u - 0.5) * Math.Pow(xy, _eta + 1.0);
            deltaQ = 1.0 - Math.Pow(value, mutationPower);
        }

        return param + deltaQ * width;
    }
}

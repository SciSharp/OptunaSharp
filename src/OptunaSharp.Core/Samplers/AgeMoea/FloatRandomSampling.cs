using NumSharp;

namespace OptunaSharp.Samplers.AgeMoea;

/// <summary>
///     Uniform random sampling of real decision vectors within the box bounds. Port of
///     <c>pymoo.operators.sampling.rnd.FloatRandomSampling</c> (<c>@0.6.1.3</c>):
///     <code>X = np.random.random((n_samples, n_var)); X = xl + (xu - xl) * X</code>
///     The single <c>random((n_samples, n_var))</c> draw (C-order) is the first consumption of the
///     MT19937 stream in a run, so its shape/order is load-bearing for byte-exactness.
/// </summary>
internal sealed class FloatRandomSampling
{
    /// <summary>Draws independent uniform decision vectors inside the problem box.</summary>
    /// <param name="problem">Decision bounds and objective/constraint evaluation surface.</param>
    /// <param name="nSamples">Number of new decision vectors to draw.</param>
    /// <param name="rng">Random stream whose draw order is shared with the reference algorithm.</param>
    /// <returns>Uniformly sampled vectors with one row per individual.</returns>
    public double[][] Do(AgeMoeaProblem problem, int nSamples, NumPyRandom rng)
    {
        double[][] x = NpRandom.Random(rng, nSamples, problem.NVar);
        for (int i = 0; i < nSamples; i++)
            for (int j = 0; j < problem.NVar; j++)
                x[i][j] = problem.Xl[j] + (problem.Xu[j] - problem.Xl[j]) * x[i][j];
        return x;
    }
}

using NumSharp;
using OptunaSharp.Samplers;

namespace OptunaSharp.Gp;

/// <summary>
///     Random-sampling acquisition-function optimizer — port of <c>optuna/_gp/optim_sample.py</c>.
///     Draws <c>n_samples</c> normalized points from the search space (scrambled Sobol', seeded from
///     <c>rng</c>) and returns the argmax by the acquisition value. Used by
///     <see cref="OptunaSharp.Terminator.RegretBoundEvaluator" /> (Phase 3.3) to maximize the UCB over
///     the whole search space (Optuna uses this simple sampler, not the mixed local optimizer, for the
///     regret-bound estimate).
/// </summary>
internal static class OptimSample
{
    /// <summary>
    ///     <c>optimize_acqf_sample(acqf, n_samples, rng)</c> — sample <paramref name="nSamples" /> points,
    ///     evaluate the acqf, and return the best point and its value.
    /// </summary>
    /// <param name="acqf">The acquisition function to maximize over its normalized search space.</param>
    /// <param name="nSamples">The number of observations or generated sample rows.</param>
    /// <param name="rng">The random generator whose state is advanced when drawing samples.</param>
    /// <returns>
    ///     The best sampled normalized point and its acquisition value; when an acquisition value is NaN, the first NaN
    ///     point and NaN, as NumPy's <c>np.argmax</c> picks it.
    /// </returns>
    /// <exception cref="NumSharp.ValueError"><paramref name="nSamples" /> is 0 (NumPy's <c>attempt to get argmax of an empty sequence</c>).</exception>
    internal static (double[] X, double Value) OptimizeAcqfSample(
        IAcquisitionFunc acqf, int nSamples, NumPyRandom rng)
    {
        double[,] xs = acqf.SearchSpace.SampleNormalizedParams(nSamples, rng);
        double[] res = acqf.EvalAcqfNoGrad(xs);

        // best_i = np.argmax(res), through NumSharp: the first maximum, or the first NaN when one is NaN (NumPy treats a
        // NaN as maximal; a strict > loop would skip it). One call per optimization, after nSamples acqf evaluations.
        int bestI;
        using (NDArray values = np.array(res, copy: false))
        using (NDArray index = np.argmax(values))
            bestI = (int)(long)index;

        int dim = xs.GetLength(1);
        var x = new double[dim];
        for (int d = 0; d < dim; d++)
            x[d] = xs[bestI, d];
        return (x, res[bestI]);
    }
}

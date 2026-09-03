using System;
using System.Collections.Generic;
using System.Linq;
using NumSharp;
using OptunaSharp.Internal;

namespace OptunaSharp.Samplers.CmaEs;

/// <summary>
///     Warm Starting CMA-ES. Port of the external <c>cmaes</c> package's
///     <c>get_warm_start_mgd</c> (<c>cmaes/_warm_start.py</c>, v0.13.0): estimates a promising
///     multivariate-gaussian distribution (mean, sigma, covariance) from the top-<c>γ</c> solutions of
///     a source task, to initialise CMA-ES for a similar task (AAAI 2021, arXiv:2012.06932).
///
///     <para><b>Bit-exact vs the Python package.</b> The rank-one products (<c>np.dot(v.reshape(d,1),
///     v.reshape(1,d))</c> = outer products) carry no summation, the accumulations are plain
///     sequential <c>+=</c>, the determinant goes through <see cref="LinAlg.Det" /> (LAPACK, byte-exact
///     vs NumPy), and the scalar powers map to <see cref="Math.Pow" /> — so a shared source corpus
///     yields byte-identical (mean, sigma, cov).</para>
/// </summary>
internal static class WarmStart
{
    /// <summary>
    ///     Estimate (mean, sigma, covariance) from <paramref name="sourceSolutions" /> — a list of
    ///     (parameter vector, value) pairs on a source task. <paramref name="gamma" /> selects the
    ///     top-(γ·100)% by value; <paramref name="alpha" /> is the prior for the initial covariance.
    /// </summary>
    internal static (double[] Mean, double Sigma, double[,] Cov) GetWarmStartMgd(
        List<(double[] Params, double Value)> sourceSolutions, double gamma = 0.1, double alpha = 0.1)
    {
        if (!(gamma > 0 && gamma <= 1)) throw new ArgumentException("gamma should be in (0, 1]");
        if (sourceSolutions.Count == 0)
            throw new ValueError("solutions should contain one or more items.");

        // Select top-(gamma x 100)% solutions (stable sort by value, ascending).
        var sorted = sourceSolutions.OrderBy(s => s.Value).ToList();
        int gammaN = (int)Math.Floor(sorted.Count * gamma);
        if (gammaN < 1)
            throw new ArgumentException("One or more solutions must be selected from a source task");

        int dim = sorted[0].Params.Length;

        // first_term = alpha**2 * I ; cov_term = Σ outer(x_i, x_i) ; mean_term = Σ x_i.
        double alphaSq = Math.Pow(alpha, 2);
        var covTerm = new double[dim, dim];
        var meanTerm = new double[dim];
        for (int i = 0; i < gammaN; i++)
        {
            double[] x = sorted[i].Params;
            for (int r = 0; r < dim; r++)
            {
                meanTerm[r] += x[r];
                for (int c = 0; c < dim; c++)
                    covTerm[r, c] += x[r] * x[c];  // np.dot(x.reshape(d,1), x.reshape(1,d)) = outer(x,x)
            }
        }

        // second_term = cov_term / gamma_n ; mean_term /= gamma_n ; third_term = outer(mean, mean).
        for (int r = 0; r < dim; r++) meanTerm[r] /= gammaN;

        var sigma = new double[dim, dim];
        for (int r = 0; r < dim; r++)
            for (int c = 0; c < dim; c++)
            {
                double first = r == c ? alphaSq : 0.0;
                double second = covTerm[r, c] / gammaN;
                double third = meanTerm[r] * meanTerm[c];
                sigma[r, c] = first + second - third;  // Sigma = first + second - third
            }

        double detSigma = LinAlg.Det(sigma);
        double sigma0 = Math.Pow(detSigma, 1.0 / 2.0 / dim);
        double covScale = Math.Pow(detSigma, 1.0 / dim);
        var cov = new double[dim, dim];
        for (int r = 0; r < dim; r++)
            for (int c = 0; c < dim; c++)
                cov[r, c] = sigma[r, c] / covScale;

        return ((double[])meanTerm.Clone(), sigma0, cov);
    }
}

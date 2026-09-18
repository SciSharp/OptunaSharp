// SAASBO's acquisition — expected improvement averaged over the bag of hyperparameter samples (Eriksson &
// Jankowiak 2021, Eqn (10)) and its maximization (Eqn (11) / §B): screen a scrambled Sobol sequence of
// length Q, take the top K starts, and polish each with L-BFGS-B (bound-constrained to the unit box, ≤100
// function evaluations). Averaging EI across the L samples — "potentially from distinct modes of the
// posterior" — is precisely what the paper credits for NUTS's robustness over MAP (§5.2): it mitigates
// EI's tendency to seek the domain boundary.

using System;
using System.Collections.Generic;
using System.Linq;
using OptunaSharp.Internal;

namespace OptunaSharp.Samplers.ModelBased.Saas;

/// <summary>
///     The sample-averaged expected-improvement acquisition of Eriksson &amp; Jankowiak 2021 (Eqn (10)) and
///     its optimizer (Eqn (11)). Constructed from the training GP, a bag of hyperparameter samples
///     <c>{ψ_ℓ}</c> (NUTS draws or the single MAP estimate), and the incumbent, it conditions one
///     <see cref="SaasPosterior" /> per sample up front so <see cref="Value" /> / <see cref="ValueAndGrad" />
///     average EI cheaply across the samples. <see cref="Maximize" /> then runs the paper's Sobol-screen +
///     L-BFGS-B strategy over the normalized unit box.
///
///     <para>The convention is <b>maximization</b>: the GP models a maximization-oriented, standardized
///     objective, the incumbent <c>f0</c> is the best such value, and improvement is <c>f(x) − f0 &gt; 0</c>
///     — the caller sign-flips a minimizing study before fitting.</para>
/// </summary>
internal sealed class SaasAcquisition
{
    private const double VarianceFloor = 1e-12;   // keeps σ &gt; 0 at near-noise-free training points
    private static readonly double InvSqrt2Pi = 1.0 / Math.Sqrt(2.0 * Math.PI);

    private readonly SaasPosterior[] _posteriors;
    private readonly double _incumbent;
    private readonly int _dim;

    /// <summary>The input dimension over which the acquisition is optimized.</summary>
    internal int Dim => _dim;

    /// <summary>
    ///     Conditions the GP on each hyperparameter sample so EI can be averaged over the bag. A sample whose
    ///     covariance will not factor is dropped (its posterior is unusable); if every sample drops the
    ///     acquisition is empty and <see cref="Value" /> returns zero everywhere.
    /// </summary>
    /// <param name="gp">The training GP (standardized, maximization-oriented targets).</param>
    /// <param name="samples">The bag of hyperparameter samples to average EI over (NUTS draws or the MAP point).</param>
    /// <param name="incumbent">The incumbent <c>f0</c> — the best standardized maximization-oriented training value.</param>
    /// <exception cref="System.ArgumentException"><paramref name="samples" /> is empty.</exception>
    internal SaasAcquisition(SaasGaussianProcess gp, IReadOnlyList<SaasHyperparameters> samples, double incumbent)
    {
        if (samples.Count == 0)
            throw new ArgumentException("At least one hyperparameter sample is required.", nameof(samples));
        _dim = gp.Dim;
        _incumbent = incumbent;

        var conditioned = new List<SaasPosterior>(samples.Count);
        foreach (SaasHyperparameters hp in samples)
        {
            try
            {
                conditioned.Add(gp.Condition(hp));
            }
            catch (NumSharp.LinAlgError)
            {
                // A pathological sample whose covariance will not factor is skipped rather than aborting the
                // whole acquisition — the remaining samples still define a usable averaged EI.
            }
        }
        _posteriors = conditioned.ToArray();
    }

    /// <summary>The number of usable posterior samples EI is averaged over (samples with a factorable covariance).</summary>
    internal int SampleCount => _posteriors.Length;

    /// <summary>
    ///     The sample-averaged expected improvement at <paramref name="x" /> (Eqn (10)):
    ///     <c>(1/L) Σ_ℓ EI(x | ψ_ℓ)</c>, with each term the closed-form Gaussian EI
    ///     <c>(μ − f0)Φ(Z) + σφ(Z)</c>, <c>Z = (μ − f0)/σ</c>. Returns zero when no posterior sample is usable.
    /// </summary>
    /// <param name="x">One normalized query point in the unit box (length <see cref="Dim" />).</param>
    /// <returns>The averaged expected improvement (nonnegative).</returns>
    internal double Value(double[] x)
    {
        if (_posteriors.Length == 0)
            return 0.0;
        double sum = 0.0;
        foreach (SaasPosterior post in _posteriors)
        {
            (double mean, double var) = post.Predict(x);
            sum += ExpectedImprovement(mean, var, _incumbent).Ei;
        }
        return sum / _posteriors.Length;
    }

    /// <summary>
    ///     The sample-averaged expected improvement and its gradient w.r.t. <paramref name="x" /> — the
    ///     quantities the L-BFGS-B polish consumes. Averages the per-sample EI and, using
    ///     <c>∂EI/∂μ = Φ(Z)</c> and <c>∂EI/∂σ² = φ(Z)/(2σ)</c>, the chained input gradient
    ///     <c>Φ(Z)·∂μ/∂x + [φ(Z)/(2σ)]·∂σ²/∂x</c> over the bag.
    /// </summary>
    /// <param name="x">One normalized query point in the unit box (length <see cref="Dim" />).</param>
    /// <returns>The averaged expected improvement and its per-coordinate gradient.</returns>
    internal (double Value, double[] Grad) ValueAndGrad(double[] x)
    {
        var grad = new double[_dim];
        if (_posteriors.Length == 0)
            return (0.0, grad);

        double sum = 0.0;
        foreach (SaasPosterior post in _posteriors)
        {
            (double mean, double var, double[] dMean, double[] dVar, bool _) = post.PredictWithGrad(x);
            (double ei, double dMeanCoef, double dVarCoef) = ExpectedImprovement(mean, var, _incumbent);
            sum += ei;
            for (int d = 0; d < _dim; d++)
                grad[d] += dMeanCoef * dMean[d] + dVarCoef * dVar[d];
        }

        double inv = 1.0 / _posteriors.Length;
        for (int d = 0; d < _dim; d++)
            grad[d] *= inv;
        return (sum * inv, grad);
    }

    /// <summary>
    ///     The closed-form Gaussian expected improvement and the two sensitivity coefficients used by the
    ///     input gradient: with <c>σ = √(max(var, floor))</c>, <c>Z = (μ − f0)/σ</c>,
    ///     <c>EI = max(0, (μ − f0)Φ(Z) + σφ(Z))</c>, <c>∂EI/∂μ = Φ(Z)</c>, and
    ///     <c>∂EI/∂var = φ(Z)/(2σ)</c>.
    /// </summary>
    /// <param name="mean">The posterior mean at the query point (maximization-oriented, standardized).</param>
    /// <param name="var">The posterior variance at the query point.</param>
    /// <param name="incumbent">The incumbent <c>f0</c> to improve upon.</param>
    /// <returns>The expected improvement and the <c>∂EI/∂μ</c> and <c>∂EI/∂var</c> coefficients.</returns>
    private static (double Ei, double DMeanCoef, double DVarCoef) ExpectedImprovement(
        double mean, double var, double incumbent)
    {
        double sigma = Math.Sqrt(Math.Max(var, VarianceFloor));
        double improvement = mean - incumbent;
        double z = improvement / sigma;
        double cdf = Erf.NormCdf(z);
        double pdf = InvSqrt2Pi * Math.Exp(-0.5 * z * z);
        double ei = improvement * cdf + sigma * pdf;
        if (ei < 0.0) ei = 0.0;                 // EI is nonnegative by construction; guard tiny negatives from roundoff.
        return (ei, cdf, pdf / (2.0 * sigma));
    }

    /// <summary>
    ///     Maximizes the averaged EI over the normalized unit box <c>[0, 1]^Dim</c> with the paper's strategy
    ///     (§B): draw a scrambled Sobol sequence of length <paramref name="numCandidates" />, take the
    ///     <paramref name="numRestarts" /> highest-EI starts, and polish each with L-BFGS-B (bound-constrained,
    ///     ≤ <paramref name="maxLbfgsEvals" /> function evaluations). Returns the best point found (clamped to
    ///     the box). If L-BFGS-B never improves on the Sobol screen (e.g. a flat EI surface), the best Sobol
    ///     candidate is returned.
    /// </summary>
    /// <param name="numCandidates">The Sobol screen size <c>Q</c> (paper: 5000).</param>
    /// <param name="numRestarts">The number of L-BFGS-B restarts <c>K</c> (paper: 3).</param>
    /// <param name="seed">The scrambled-Sobol seed (a pure function of the sampler's per-trial RNG).</param>
    /// <param name="maxLbfgsEvals">The per-restart L-BFGS-B function-evaluation cap (paper: 100).</param>
    /// <returns>The normalized maximizer of the averaged EI in the unit box.</returns>
    /// <exception cref="System.ArgumentException">A budget is non-positive.</exception>
    internal double[] Maximize(int numCandidates, int numRestarts, int seed, int maxLbfgsEvals = 100)
    {
        if (numCandidates < 1)
            throw new ArgumentException("numCandidates must be at least 1.", nameof(numCandidates));
        if (numRestarts < 1)
            throw new ArgumentException("numRestarts must be at least 1.", nameof(numRestarts));
        if (maxLbfgsEvals < 1)
            throw new ArgumentException("maxLbfgsEvals must be at least 1.", nameof(maxLbfgsEvals));

        // ---- Sobol screen: evaluate averaged EI on a scrambled Sobol fill of the unit box ----
        double[,] candidates = ScrambledSobol.Random(_dim, numCandidates, seed);
        var eis = new double[numCandidates];
        for (int i = 0; i < numCandidates; i++)
        {
            var x = new double[_dim];
            for (int d = 0; d < _dim; d++) x[d] = candidates[i, d];
            eis[i] = Value(x);
        }

        // Top-K starts by EI (descending). Ties keep Sobol order — deterministic under a fixed seed.
        int[] order = Enumerable.Range(0, numCandidates).OrderByDescending(i => eis[i]).ToArray();
        int k = Math.Min(numRestarts, numCandidates);

        double[] bestX = new double[_dim];
        for (int d = 0; d < _dim; d++) bestX[d] = candidates[order[0], d];
        double bestEi = eis[order[0]];

        var bounds = new (double Lower, double Upper)[_dim];
        for (int d = 0; d < _dim; d++) bounds[d] = (0.0, 1.0);

        // ---- L-BFGS-B polish from each of the top-K starts (minimize −EI) ----
        for (int r = 0; r < k; r++)
        {
            var start = new double[_dim];
            for (int d = 0; d < _dim; d++) start[d] = candidates[order[r], d];

            try
            {
                LBfgsB.Result res = LBfgsB.Minimize(
                    x =>
                    {
                        (double ei, double[] g) = ValueAndGrad(x);
                        var negGrad = new double[_dim];
                        for (int d = 0; d < _dim; d++) negGrad[d] = -g[d];   // minimize −EI ⇒ negate the gradient.
                        return (-ei, negGrad);
                    },
                    start, bounds, m: 10, factr: 1e7, pgtol: 1e-5,
                    maxfun: maxLbfgsEvals, maxiter: maxLbfgsEvals, maxls: 20);

                double[] xOpt = res.X;
                double eiOpt = Value(xOpt);   // recompute at the returned point (L-BFGS-B leaves at its last probe).
                if (eiOpt > bestEi)
                {
                    bestEi = eiOpt;
                    for (int d = 0; d < _dim; d++)
                        bestX[d] = Math.Min(1.0, Math.Max(0.0, xOpt[d]));   // clamp any tiny out-of-box drift.
                }
            }
            catch (Exception e) when (e is InvalidOperationException or NumSharp.LinAlgError)
            {
                // A failed polish keeps the Sobol start's contribution; the screen already provided a valid
                // fallback (bestX), so this restart is simply skipped.
            }
        }

        return bestX;
    }
}

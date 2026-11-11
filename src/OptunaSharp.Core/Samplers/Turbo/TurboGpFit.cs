// The TuRBO local-GP fit — the paper's Appendix C bound-constrained marginal-likelihood optimization,
// driven over the SHARED GPRegressor without modifying it.
//
// Reference:
//   David Eriksson, Michael Pearce, Jacob Gardner, Ryan D. Turner, Matthias Poloczek.
//     "Scalable Global Optimization via Local Bayesian Optimization." NeurIPS 2019 (arXiv:1910.01739), §C:
//     "We use a Matérn-5/2 kernel with ARD for TuRBO and use the following bounds for the hyperparameters:
//      (lengthscale) λ_i ∈ [0.005, 2.0], (signal variance) s² ∈ [0.05, 20.0], (noise variance)
//      σ² ∈ [0.0005, 0.1]."
//
// Why this exists (and why it is NOT a change to GPRegressor): GPSampler fits the same Matérn-5/2 ARD GP
// but with Optuna's UNCONSTRAINED, soft-prior-regularized MLL (its L-BFGS-B call passes `bounds: null`).
// TuRBO §C instead pins HARD box bounds on the hyperparameters, and those bounds feed directly into the
// trust-region geometry (the TR side lengths are rescaled by the fitted length scales, TurboOptimizer.
// TrustRegionBounds). To honor §C without perturbing GPSampler, this helper reuses GPRegressor's exact
// forward + hand-derived gradient (LossAndGrad) and its posterior caching (CacheMatrix), but drives the
// managed L-BFGS-B with the §C box in the same log-transformed coordinates the fit optimizes in. It is a
// faithful bound-CONSTRAINED optimum (L-BFGS-B respects the box), not a post-hoc clamp of an unconstrained
// optimum — so the fitted parameters are guaranteed to satisfy §C, which the parity tests assert directly.

using System;
using OptunaSharp.Internal;
using OptunaSharp.Samplers.ModelBased.Gp;

namespace OptunaSharp.Samplers.Turbo;

/// <summary>
///     Fits TuRBO's local Matérn-5/2 ARD Gaussian process with the paper's §C hard hyperparameter bounds,
///     reusing the shared <see cref="GPRegressor" /> (kernel forward, analytic marginal-likelihood gradient,
///     OpenBLAS Cholesky, posterior cache) unchanged. The one difference from
///     <see cref="GPRegressor.FitKernelParams" /> — which GPSampler uses — is that the marginal-likelihood
///     optimization here is <b>bound-constrained</b> to the §C box rather than unconstrained, because the
///     fitted length scales rescale the trust-region box and TuRBO relies on §C to keep them in range.
///
///     <para><b>Footgun.</b> The §C noise-variance lower bound and the GP fit's <c>minimumNoise</c> floor are
///     the same quantity: <c>ν = exp(raw) + minimumNoise ≥ minimumNoise</c>, so the caller must pass
///     <c>minimumNoise = </c><see cref="NoiseVarMin" /> to reproduce the paper's <c>σ² ∈ [0.0005, 0.1]</c>.
///     A caller-supplied floor above <see cref="NoiseVarMax" /> would invert the noise box; the upper bound
///     is dropped (left unbounded) in that degenerate case rather than throwing.</para>
/// </summary>
internal static class TurboGpFit
{
    /// <summary>§C lower bound on each ARD length scale <c>λ_i</c> (input space is the unit box).</summary>
    internal const double LengthScaleMin = 0.005;

    /// <summary>§C upper bound on each ARD length scale <c>λ_i</c>; caps how wide a flat direction can stretch the TR box.</summary>
    internal const double LengthScaleMax = 2.0;

    /// <summary>§C lower bound on the signal variance <c>s²</c> (the kernel scale / amplitude).</summary>
    internal const double KernelScaleMin = 0.05;

    /// <summary>§C upper bound on the signal variance <c>s²</c>.</summary>
    internal const double KernelScaleMax = 20.0;

    /// <summary>§C lower bound on the observation-noise variance <c>σ²</c> — also the GP fit's noise floor
    ///     (the two are the same quantity in Optuna's <c>ν = exp(raw) + minimumNoise</c> parameterization).</summary>
    internal const double NoiseVarMin = 5e-4;   // 0.0005

    /// <summary>§C upper bound on the observation-noise variance <c>σ²</c>.</summary>
    internal const double NoiseVarMax = 0.1;

    /// <summary>
    ///     Fits the local GP by maximizing the marginal log likelihood over the log-transformed kernel
    ///     hyperparameters subject to the §C box, then caches the covariance factor. On optimizer failure
    ///     (non-converged L-BFGS-B or a non-positive-definite covariance) it falls back to the within-box
    ///     default parameters and still caches — mirroring <see cref="GPRegressor.FitKernelParams" />'s
    ///     robustness so a hard problem degrades to a usable prior rather than throwing mid-study.
    /// </summary>
    /// <param name="x">Normalized training inputs in <c>[0, 1]^d</c>, one point per row.</param>
    /// <param name="y">Standardized training targets (maximization-oriented), aligned with <paramref name="x" />'s rows.</param>
    /// <param name="isCategorical">Per-coordinate flag selecting Hamming (categorical) vs. numerical kernel distance.</param>
    /// <param name="minimumNoise">The observation-noise floor; pass <see cref="NoiseVarMin" /> for §C parity. It is the
    ///     effective lower bound of the noise box, so it must be below <see cref="NoiseVarMax" /> for a well-posed box.</param>
    /// <param name="deterministicObjective">Treat the objective as noise-free — fix <c>ν = minimumNoise</c> and drop the noise
    ///     dimension from the optimization (its gradient is exactly zero), matching <see cref="GPRegressor.LossAndGrad" />.</param>
    /// <param name="gtol">The projected-gradient stopping tolerance forwarded to L-BFGS-B (scipy's <c>pgtol</c>).</param>
    /// <returns>A fitted GP whose hyperparameters satisfy the §C box and whose covariance is cached for posterior queries.</returns>
    /// <exception cref="System.ArgumentException">The training shapes are inconsistent (surfaced by the <see cref="GPRegressor" /> constructor).</exception>
    internal static GPRegressor Fit(
        double[,] x, double[] y, bool[] isCategorical,
        double minimumNoise, bool deterministicObjective, double gtol = 1e-2)
    {
        int dim = x.GetLength(1);

        // ---- §C box, mapped into the fit's log coordinates -------------------------------------------
        // The fit optimizes raw = [log w_0 … log w_{d-1}, log s, log(ν − minimumNoise)] where the inverse
        // squared length scale w = 1/λ². Larger λ ⇒ smaller w, so the λ box flips orientation for w.
        double wLo = Math.Log(1.0 / (LengthScaleMax * LengthScaleMax));   // λ ≤ 2   ⇒ w ≥ 0.25
        double wHi = Math.Log(1.0 / (LengthScaleMin * LengthScaleMin));   // λ ≥ 0.005 ⇒ w ≤ 40000
        double sLo = Math.Log(KernelScaleMin);
        double sHi = Math.Log(KernelScaleMax);
        // ν = exp(raw) + minimumNoise, so raw's lower is −∞ (ν → minimumNoise) and its upper encodes ν ≤
        // NoiseVarMax. A floor at/above NoiseVarMax degenerates the box → leave the noise raw unbounded.
        double noiseHi = minimumNoise < NoiseVarMax
            ? Math.Log(NoiseVarMax - minimumNoise)
            : double.PositiveInfinity;

        var bounds = new (double Lower, double Upper)[dim + 2];
        for (int d = 0; d < dim; d++) bounds[d] = (wLo, wHi);
        bounds[dim] = (sLo, sHi);
        bounds[dim + 1] = deterministicObjective
            ? (double.NegativeInfinity, double.PositiveInfinity)   // ν fixed at the floor; the bound is inert
            : (double.NegativeInfinity, noiseHi);

        // ---- feasible within-box initial point -------------------------------------------------------
        // λ = 1 (w = 1) and s² = 1 sit inside the box; the noise starts one log-unit below its upper bound
        // (strictly interior) so L-BFGS-B does not begin pinned to a face. (L-BFGS-B projects x0 onto the
        // box regardless, but a feasible interior start avoids a wasted first projection.)
        double rawNoise0 = double.IsPositiveInfinity(noiseHi)
            ? Math.Log(Math.Max(minimumNoise, 1e-6))
            : noiseHi - 1.0;
        double nu0 = deterministicObjective ? minimumNoise : Math.Exp(rawNoise0) + minimumNoise;

        var w0 = new double[dim];
        Array.Fill(w0, 1.0);
        GPRegressor gpr = new(isCategorical, x, y, w0, kernelScale: 1.0, noiseVar: nu0);

        var initial = new double[dim + 2];
        // initial[0..dim-1] = log(1) = 0 (λ = 1); initial[dim] = log(1) = 0 (s² = 1).
        initial[dim + 1] = deterministicObjective ? 0.0 : rawNoise0;

        try
        {
            LBfgsB.Result res;
            try
            {
                res = LBfgsB.Minimize(
                    raw => gpr.LossAndGrad(raw, minimumNoise, deterministicObjective),
                    initial, bounds, m: 10, factr: 1e7, pgtol: gtol, maxfun: 15000, maxiter: 15000, maxls: 20);
            }
            finally
            {
                // The loss is not evaluated again; the regressor is kept for its posterior, without the fit's scratch.
                gpr.ReleaseFitScratch();
            }
            // A non-zero warn flag is scipy/Optuna's "Optimization failed" — treat as a fit failure and
            // fall back, rather than trusting an unconverged point that may violate the box.
            if (res.WarnFlag != 0)
                throw new InvalidOperationException("Optimization failed");

            // L-BFGS-B leaves the model at its LAST probe, not the optimum, so write the returned optimum
            // back explicitly (as GPRegressor.FitInner does) before caching the covariance.
            double[] xOpt = res.X;
            var wFit = new double[dim];
            for (int d = 0; d < dim; d++) wFit[d] = Math.Exp(xOpt[d]);
            gpr.InverseSquaredLengthscales = wFit;
            gpr.KernelScale = Math.Exp(xOpt[dim]);
            gpr.NoiseVar = deterministicObjective ? minimumNoise : minimumNoise + Math.Exp(xOpt[dim + 1]);
            gpr.CacheMatrix();
            return gpr;
        }
        catch (Exception e) when (e is InvalidOperationException or NumSharp.LinAlgError)
        {
            // Robustness fallback (mirrors GPRegressor.FitKernelParams): the within-box default prior
            // (λ = 1, s² = 1, ν = the feasible start) always yields a positive-definite covariance.
            var fallback = new GPRegressor(
                isCategorical, x, y, w0, kernelScale: 1.0,
                noiseVar: deterministicObjective ? minimumNoise : nu0);
            fallback.CacheMatrix();
            return fallback;
        }
    }
}

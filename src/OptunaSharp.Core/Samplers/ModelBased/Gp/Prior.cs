// Beyond-core copy of the GP kernel-hyperparameter prior (OptunaSharp.Gp.Prior), used by the copied
// Matérn GPRegressor in this toolkit namespace so the model-based samplers do not depend on the
// Optuna-parity GP stack (the project rule: copy rather than modify a shared preexisting component).
// Verbatim duplicate of the parity Prior.cs — keep diff-identical; changes meant for GPSampler's fit
// belong in OptunaSharp.Gp.

using System;
using OptunaSharp.Internal;

namespace OptunaSharp.Samplers.ModelBased.Gp;

/// <summary>
///     Prior distribution over the GP kernel hyperparameters — port of <c>optuna/_gp/prior.py</c>.
///     The prior regularizes the marginal-likelihood fit (<see cref="GPRegressor.LossAndGrad" />):
///     a hand-crafted term on the inverse squared lengthscales plus <c>Gamma</c> priors on the
///     kernel scale and the noise variance. Optuna omits every constant factor (they do not move the
///     optimum), so this reproduces the exact objective the L-BFGS-B fit sees.
///
///     <para>Optuna's <c>default_log_prior</c> is a pluggable <c>Callable[[GPRegressor], Tensor]</c>
///     whose gradient is taken by autograd; <see cref="OptunaSharp.Samplers.GPSampler" /> only ever installs this default
///     one. Because OptunaSharp hand-derives the fit gradient, the analytic derivative of this exact
///     closed form is folded into <see cref="GPRegressor.LossAndGrad" /> (documented there); a custom
///     prior would need its own derivative — the one boundary vs Optuna's callable seam.</para>
/// </summary>
internal static class Prior
{
    /// <summary><c>DEFAULT_MINIMUM_NOISE_VAR</c> — the floor added to the fitted noise variance.</summary>
    internal const double DefaultMinimumNoiseVar = 1e-6;

    /// <summary>
    ///     <c>default_log_prior(gpr)</c> — the (constant-omitted) log prior density of the kernel
    ///     hyperparameters. <c>gamma_log_prior(x, c, rate) = (c - 1)·log(x) - rate·x</c>.
    /// </summary>
    /// <param name="gpr">The Gaussian-process model supplying posterior predictions and kernel length scales.</param>
    /// <returns>The log prior density up to constants independent of the kernel parameters.</returns>
    internal static double DefaultLogPrior(GPRegressor gpr)
    {
        double[] w = gpr.InverseSquaredLengthscales;
        var lengthscaleTerms = new double[w.Length];
        for (int d = 0; d < w.Length; d++)
            lengthscaleTerms[d] = 0.1 / w[d] + 0.1 * w[d];

        double lengthscalePrior = -MathUtil.PairwiseSum(lengthscaleTerms);
        double kernelScalePrior = (2.0 - 1.0) * Math.Log(gpr.KernelScale) - 1.0 * gpr.KernelScale;
        double noiseVarPrior = (1.1 - 1.0) * Math.Log(gpr.NoiseVar) - 30.0 * gpr.NoiseVar;
        return lengthscalePrior + kernelScalePrior + noiseVarPrior;
    }
}

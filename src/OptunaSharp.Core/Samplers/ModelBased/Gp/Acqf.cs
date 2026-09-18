// Beyond-core copy of the scalar expected-improvement helpers (a trimmed subset of OptunaSharp.Gp.Acqf),
// duplicated into the model-based toolkit namespace so REMBO's paper-faithful acquisition does not depend
// on the Optuna-parity GP stack (the project rule: copy rather than modify a shared preexisting component).
//
// Scope: ONLY the numerically stable scalar log-expected-improvement is copied — REMBO maximizes EI
// (Wang et al. 2016 report the expected-improvement acquisition of Močkus 1982 / Bull 2011) with the
// derivative-free DIRECT + CMA-ES optimizers, so it needs the acquisition VALUE only, never the analytic
// gradient or the q-batch / multi-objective / constrained acquisitions the parity Acqf also carries.
// log-EI is monotone in EI, so an argmax over it is the argmax over EI, with far better numerical range.

using System;
using OptunaSharp.Internal;

namespace OptunaSharp.Samplers.ModelBased.Gp;

/// <summary>
///     The scalar log-expected-improvement acquisition REMBO maximizes over its low-dimensional box —
///     the stable log forms from "Unexpected Improvements to Expected Improvement" (arXiv:2310.20708),
///     identical to <c>OptunaSharp.Gp.Acqf</c>'s scalar helpers. It rides <see cref="Erf" /> (erfc/erfcx),
///     so it inherits the erf/ucrt tolerance class (not bit-exact against a foreign libm), which is
///     immaterial here because REMBO's acquisition is optimized derivative-free and only its argmax matters.
/// </summary>
internal static class Acqf
{
    private static readonly double SqrtHalf = Math.Sqrt(0.5);
    private static readonly double InvSqrt2Pi = 1.0 / Math.Sqrt(2.0 * Math.PI);
    private static readonly double SqrtHalfPi = Math.Sqrt(0.5 * Math.PI);
    private static readonly double LogSqrt2Pi = Math.Log(Math.Sqrt(2.0 * Math.PI));

    /// <summary>
    ///     <c>standard_logei(z) = log E_{x~N(0,1)}[max(0, x + z)]</c> — the stable log expected positive
    ///     improvement of a standard normal shifted by <paramref name="z" />. The primary branch is
    ///     <c>log(z·Φ(z) + φ(z))</c>; for <c>z &lt; -25</c> the <c>erfcx</c> form avoids the <c>0/0</c> as
    ///     the improvement vanishes (Eq. 9 of arXiv:2310.20708), so the value stays finite arbitrarily far
    ///     below the incumbent — the property that lets DIRECT/CMA-ES score even hopeless candidates without
    ///     producing <c>-inf</c> plateaus that would flatten the acquisition landscape.
    /// </summary>
    /// <param name="z">The standardized normal argument <c>(mean − incumbent)/σ</c>.</param>
    /// <returns>The log expected positive improvement for the standardized normal argument.</returns>
    internal static double StandardLogEi(double z)
    {
        if (z < -25.0)
        {
            double w = -SqrtHalf * z;
            return -0.5 * z * z - LogSqrt2Pi + Math.Log(1.0 + SqrtHalfPi * z * Erf.Erfcx(w));
        }
        double zHalf = 0.5 * z;
        return Math.Log(zHalf * Erf.Erfc(-SqrtHalf * z) + Math.Exp(-zHalf * z) * InvSqrt2Pi);
    }

    /// <summary>
    ///     <c>logei(mean, var, f0) = standard_logei((mean − f0)/σ) + log σ</c>, with <c>σ = √var</c> — the
    ///     log expected improvement of a Gaussian posterior <c>N(mean, var)</c> over the incumbent
    ///     <paramref name="f0" /> in the maximization convention REMBO's GP is standardized to.
    /// </summary>
    /// <param name="mean">The GP posterior mean at the candidate, in the standardized maximization coordinate.</param>
    /// <param name="var">The nonnegative GP posterior variance at the candidate (a small floor should be added by the caller to avoid a zero σ).</param>
    /// <param name="f0">The incumbent (best standardized value so far) that a candidate must improve upon.</param>
    /// <returns>The logarithm of expected improvement above the incumbent threshold.</returns>
    internal static double LogEi(double mean, double var, double f0)
    {
        double sigma = Math.Sqrt(var);
        return StandardLogEi((mean - f0) / sigma) + Math.Log(sigma);
    }
}

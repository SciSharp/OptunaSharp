// The CMA-ES population host for JANUS — OptunaSharp's own `Cma` behind `IPopulationHost`.
//
// Plan 03 extraction: this is exactly the CMA-ES–specific driving that Phase 1's `JanusOptimizer`
// entangled with the JANUS math — the host construction (initial mean/step in the transformed box),
// the native ask/tell, and the proposal statistics (mean, σ, trace(C), Mahalanobis whitening) CMIX
// and the bounded injection consume. Nothing here changes behavior; the Phase-1 JANUS tests are the
// contract, and the byte-for-byte candidate/elite stream is preserved (the RNG discipline — a single
// MT19937 `Cma` host stream — is unchanged; the CMIX noise stream stays in the driver).

using System;
using System.Collections.Generic;
using NumSharp;
using OptunaSharp.Internal;
using OptunaSharp.Samplers.CmaEs;

namespace OptunaSharp.Samplers.Janus;

/// <summary>
///     The single-objective CMA-ES <see cref="IPopulationHost" /> for <see cref="JanusOptimizer" />: a
///     <see cref="Cma" /> host that supplies native offspring (resample-then-repair ask), coherent
///     generation tell, and the full proposal statistics JANUS reads — mean, step-size σ, the
///     covariance trace, and the square-root-invariant Mahalanobis whitening ‖z‖ = √(dᵀC⁻¹d)/σ. With
///     the JANUS infill contributing nothing, a run over this host's <see cref="Ask" /> reduces to
///     plain CMA-ES (the same <see cref="Cma" /> <see cref="CmaEsSampler" /> drives).
/// </summary>
internal sealed class CmaPopulationHost : IPopulationHost
{
    private const double Eps = 1e-10;

    private readonly Cma _cma;

    /// <param name="trans">The transformed box the host proposes in (<c>transform01</c>).</param>
    /// <param name="sigma0">Accepted for signature parity with the reference; as there, the host's
    ///     initial step is <c>min_range/6</c> in the transformed box (the reference likewise ignores
    ///     its <c>sigma0=0.2</c>).</param>
    /// <param name="popsize">Population size λ (default <c>4 + ⌊3 ln d⌋</c>, the CMA-ES default).</param>
    /// <param name="seed">Seed for the host's MT19937 stream.</param>
    /// <summary>Creates the CMA-ES host and transformed box used by JANUS injection.</summary>
    internal CmaPopulationHost(SearchSpaceTransform trans, double? sigma0, int? popsize, int seed)
    {
        double[] boundsFlat = trans.Bounds.ToArray<double>();
        int d = boundsFlat.Length / 2;
        var boundsArr = new double[d, 2];
        var mean = new double[d];
        double sig = double.PositiveInfinity;
        for (int i = 0; i < d; i++)
        {
            double lo = boundsFlat[i * 2];
            double hi = boundsFlat[i * 2 + 1];
            boundsArr[i, 0] = lo;
            boundsArr[i, 1] = hi;
            mean[i] = lo + (hi - lo) / 2.0;
            sig = Math.Min(sig, (hi - lo) / 6.0);
        }
        // sigma0 is accepted for signature parity with the reference but, as there, the host's initial
        // step is min_range/6 in the transformed box (the reference likewise ignores its sigma0=0.2).
        _ = sigma0;
        sig = Math.Max(sig, Eps);
        int lambda = popsize ?? 4 + (int)(3 * Math.Log(d));

        _cma = new Cma(mean, sig, boundsArr, nMaxResampling: 10 * d, seed: seed,
            populationSize: lambda, cov: null, lrAdapt: false);
    }

    /// <summary>Number of evaluated individuals required for one generation.</summary>
    public int PopulationSize => _cma.PopulationSize;
    /// <summary>Current Gaussian proposal center in transformed coordinates.</summary>
    public double[] Mean => _cma.Mean;
    /// <summary>Global Gaussian proposal step size.</summary>
    public double Sigma => _cma.Sigma;

    /// <summary>Draws one candidate from the current proposal without advancing the generation.</summary>
    /// <returns>One candidate vector in transformed coordinates.</returns>
    public double[] Ask() => _cma.Ask();

    /// <summary>Updates the optimizer using evaluated candidates from one coherent generation.</summary>
    /// <param name="solutions">Evaluated proposal vectors and minimization-oriented values for one coherent generation.</param>
    public void Tell(IReadOnlyList<(double[] X, double Value)> solutions)
        // Cma.Tell wants a concrete List; the driver always passes one, so this cast avoids a copy.
        => _cma.Tell(solutions as List<(double[] X, double Value)> ?? new List<(double[] X, double Value)>(solutions));

    /// <summary>Whether this population host can expose and replace a proposal covariance.</summary>
    public bool HasProposalCovariance => true;

    /// <summary>Sum of proposal covariance diagonal entries used to match exploration scale.</summary>
    public double ProposalCovarianceTrace
    {
        get
        {
            double[,] c = _cma.Cov;
            int n = c.GetLength(0);
            double s = 0;
            for (int i = 0; i < n; i++) s += c[i, i];
            return s;
        }
    }

    /// <summary>Measures a displacement in the host proposal scale for bounded JANUS injection.</summary>
    /// <param name="diff">Displacement from the proposal center in transformed coordinates.</param>
    /// <returns>The host-scaled step norm; a nonpositive result signals unavailable whitening.</returns>
    public double WhitenedStepNorm(double[] diff)
    {
        try
        {
            double[] cinvDiff = LinAlg.Solve(_cma.Cov, diff);   // C⁻¹ diff
            double mahal2 = 0;
            for (int i = 0; i < diff.Length; i++) mahal2 += diff[i] * cinvDiff[i];
            return Math.Sqrt(Math.Max(mahal2, 0.0)) / Math.Max(_cma.Sigma, Eps);
        }
        catch (Exception)
        {
            return 0.0;   // no whitening — the driver injects unclipped (then box-clamped)
        }
    }

    /// <summary>Obtains a detached proposal covariance when the host supplies one.</summary>
    /// <returns>A detached covariance matrix, or null for a covariance-free host.</returns>
    public double[,]? GetProposalCovariance() => (double[,])_cma.Cov.Clone();

    /// <summary>Updates a covariance-capable host and invalidates factors derived from its previous covariance.</summary>
    /// <param name="c">Replacement proposal covariance in transformed coordinates.</param>
    /// <returns>True when the host accepted a proposal covariance; false when unsupported.</returns>
    public bool TrySetProposalCovariance(double[,] c)
    {
        _cma.SetCovariance(c);   // resets the cached eigendecomposition; next Ask re-derives it
        return true;
    }
}

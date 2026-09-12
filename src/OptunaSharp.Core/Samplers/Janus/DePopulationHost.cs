// The Differential Evolution population host for JANUS — OptunaSharp's own `De` behind `IPopulationHost`
// (Plan 05 Phase 2b: JANUS-DE).
//
// DE exposes no Gaussian proposal covariance, so this host reports `HasProposalCovariance = false` and
// the JANUS driver runs the paper's conservative GN-only configuration: the Gauss–Newton exploitation
// elite is bounded-injected into slot 0 of each DE generation, but no CMIX exploration candidates are
// offered (there is no covariance to build the inverse-curvature metric from). The bounded injection's
// whitening degrades from CMA's Mahalanobis norm to DE's covariance-free analogue — the per-dimension
// population-spread ("box and step-length") norm the paper describes for a covariance-free host.
//
// The DE population is cold-started uniformly in the transformed box; JANUS's window (harvested from ALL
// completed trials by `JanusOptimizer`, independent of this population) supplies the Jacobian for the GN
// elite, which enters the population through DE's ordinary greedy selection when it beats its target.

using System;
using System.Collections.Generic;
using NumSharp;
using OptunaSharp.Internal;
using OptunaSharp.Samplers.Differential;

namespace OptunaSharp.Samplers.Janus;

/// <summary>
///     The Differential Evolution <see cref="IPopulationHost" /> for <see cref="JanusOptimizer" />: a
///     <see cref="De" /> optimizer that supplies native offspring (one per target, cycled across the
///     generation), consumes a coherent generation via greedy selection, and reports the covariance-free
///     proposal statistics JANUS's GN-only infill reads — the population centroid (mean), a per-dimension
///     spread (σ), and the box/step-length whitening. It declares no proposal covariance, so JANUS offers
///     no CMIX candidates (the paper's <c>JANUS-GN</c> configuration for a covariance-free host).
/// </summary>
internal sealed class DePopulationHost : IPopulationHost
{
    private const double Eps = 1e-12;

    private readonly De _de;
    private readonly double[] _lo;
    private readonly double[] _hi;
    private int _askPtr;   // cycles 0..NP-1 across one generation of Ask() calls

    /// <param name="trans">The transformed box the host proposes in (<c>transform01</c>).</param>
    /// <param name="popsize">Population size NP (default <c>max(8, 4 + ⌊3 ln d⌋)</c>).</param>
    /// <param name="seed">Seed for the host's MT19937 stream and its cold-start population.</param>
    /// <param name="f">Differential weight F.</param>
    /// <param name="cr">Crossover rate CR.</param>
    /// <param name="strategy">DE mutation strategy.</param>
    /// <summary>Creates a differential-evolution host for JANUS using the transformed search-space box.</summary>
    internal DePopulationHost(SearchSpaceTransform trans, int? popsize, int seed, double f, double cr, DeStrategy strategy)
    {
        double[] boundsFlat = trans.Bounds.ToArray<double>();
        int d = boundsFlat.Length / 2;
        _lo = new double[d];
        _hi = new double[d];
        for (int i = 0; i < d; i++) { _lo[i] = boundsFlat[i * 2]; _hi[i] = boundsFlat[i * 2 + 1]; }

        int nPop = popsize ?? De.DefaultPopulationSize(d);
        if (nPop < 4) nPop = 4;

        // Cold-start the population uniformly in the box; values are "unknown" (+inf) until the first tell,
        // so DE's first greedy selection accepts any finite offspring (the generation fills in real values).
        var initRng = np.random.RandomState();
        initRng.seed((uint)seed);
        var pop = new double[nPop][];
        for (int i = 0; i < nPop; i++)
        {
            var row = new double[d];
            double[] u = initRng.uniform(0.0, 1.0, new Shape(d)).ToArray<double>();
            for (int j = 0; j < d; j++) row[j] = _lo[j] + u[j] * (_hi[j] - _lo[j]);
            pop[i] = row;
        }
        var vals = new double[nPop];
        for (int i = 0; i < nPop; i++) vals[i] = double.PositiveInfinity;

        // Seed the optimizer's own ask stream distinctly from the cold-start draw.
        _de = new De(pop, vals, f, cr, strategy, seed: seed ^ 0x5DE);
        _askPtr = 0;
    }

    /// <summary>Number of evaluated individuals required for one generation.</summary>
    public int PopulationSize => _de.PopulationSize;
    /// <summary>Population centroid used as the JANUS injection center in transformed coordinates.</summary>
    public double[] Mean => _de.Mean();

    /// <summary>Mean population standard deviation across coordinates, floored away from zero for step clipping.</summary>
    public double Sigma
    {
        get
        {
            double[] sd = _de.PerDimStd();
            double s = 0;
            foreach (double v in sd) s += v;
            return Math.Max(s / sd.Length, Eps);
        }
    }

    /// <summary>Draws one candidate from the current proposal without advancing the generation.</summary>
    /// <returns>One candidate vector in transformed coordinates.</returns>
    public double[] Ask()
    {
        int target = _askPtr % _de.PopulationSize;
        _askPtr++;
        double[] u = _de.AskCandidate(target);
        // The host owns feasibility: clamp the offspring into the box (DE is unconstrained).
        var x = new double[u.Length];
        for (int i = 0; i < u.Length; i++) x[i] = Math.Min(Math.Max(u[i], _lo[i]), _hi[i]);
        return x;
    }

    /// <summary>Updates the optimizer using evaluated candidates from one coherent generation.</summary>
    /// <param name="solutions">Evaluated proposal vectors and minimization-oriented values for one coherent generation.</param>
    public void Tell(IReadOnlyList<(double[] X, double Value)> solutions)
    {
        // The driver delivers a coherent generation in trial/ask order, whose per-slot target order is a
        // cyclic permutation (each solution's slot is trialNumber % popsize), so DE's one-to-one greedy
        // pairing cannot be reconstructed here. Use the order-independent (μ+λ) elitist selection instead —
        // it pools parents with the evaluated offspring (including the injected GN elite) and keeps the
        // best NP, exactly the behavior JANUS wants from an infill host.
        _de.TellElitist(solutions);
        _askPtr = 0;
    }

    /// <summary>Whether this population host can expose and replace a proposal covariance.</summary>
    public bool HasProposalCovariance => false;

    /// <summary>Sum of proposal covariance diagonal entries used to match exploration scale.</summary>
    public double ProposalCovarianceTrace
    {
        // Not read while HasProposalCovariance is false (the driver guards on it); kept meaningful anyway.
        get
        {
            double[] sd = _de.PerDimStd();
            double s = 0;
            foreach (double v in sd) s += v * v;
            return s;
        }
    }

    /// <summary>Measures a displacement in the host proposal scale for bounded JANUS injection.</summary>
    /// <param name="diff">Displacement from the proposal center in transformed coordinates.</param>
    /// <returns>The host-scaled step norm; a nonpositive result signals unavailable whitening.</returns>
    public double WhitenedStepNorm(double[] diff)
    {
        // DE's covariance-free analogue of the Mahalanobis norm: the step measured in per-dimension
        // population-spread units, √(Σ (diff_j / sd_j)²). The driver clips this to κ√d (the paper's box /
        // step-length constraint for a covariance-free host).
        double[] sd = _de.PerDimStd();
        double s = 0;
        for (int i = 0; i < diff.Length; i++)
        {
            double z = diff[i] / sd[i];
            s += z * z;
        }
        return Math.Sqrt(Math.Max(s, 0.0));
    }

    // DE has no Gaussian proposal covariance, so the reshape mode is a no-op here.
    /// <summary>Obtains a detached proposal covariance when the host supplies one.</summary>
    /// <returns>A detached covariance matrix, or null for a covariance-free host.</returns>
    public double[,]? GetProposalCovariance() => null;

    /// <summary>Updates a covariance-capable host and invalidates factors derived from its previous covariance.</summary>
    /// <param name="c">Replacement proposal covariance in transformed coordinates.</param>
    /// <returns>True when the host accepted a proposal covariance; false when unsupported.</returns>
    public bool TrySetProposalCovariance(double[,] c) => false;
}

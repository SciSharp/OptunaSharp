// The per-trial SAASBO step (Eriksson & Jankowiak 2021, Algorithm 1, lines 3-5): given the completed
// observations, fit the SAAS GP (NUTS or MAP), then maximize the sample-averaged expected improvement to
// obtain the next normalized query point. This is the glue between the model/inference/acquisition pieces
// of the toolkit and the public SaasboSampler, kept separate so it can be unit-tested without the study
// protocol.

using System;
using System.Collections.Generic;
using NumSharp;

namespace OptunaSharp.Samplers.ModelBased.Saas;

/// <summary>
///     The immutable configuration of one SAASBO run — the inference method plus every knob of the model,
///     inference, and acquisition. The <see cref="SaasboSampler" /> builds one of these from its
///     constructor arguments; the defaults follow the paper (α = 0.1, RBF, noise 1e-6, NUTS 512/256/16 with
///     max tree depth 6, Sobol Q = 5000 / K = 3 / ≤100 L-BFGS-B evals).
/// </summary>
/// <param name="Method">Whether to fit with NUTS (paper default) or MAP.</param>
/// <param name="Alpha">The SAAS global-shrinkage hyperparameter <c>α</c> (NUTS only; MAP ignores it).</param>
/// <param name="NoiseVariance">The fixed observation-noise variance (paper: <c>1e-6</c>, noise-free).</param>
/// <param name="KernelVarianceLogStd">The std of the log-Normal prior on <c>σ²_k</c> (paper: <c>10</c>).</param>
/// <param name="NumWarmup">NUTS warmup iterations (paper: 512).</param>
/// <param name="NumSamples">NUTS post-warmup iterations before thinning (paper: 256).</param>
/// <param name="Thinning">NUTS thinning factor; every <c>Thinning</c>-th post-warmup draw is retained (paper: 16 ⇒ L = 16).</param>
/// <param name="MaxTreeDepth">NUTS maximum tree depth (paper: 6).</param>
/// <param name="TargetAccept">NUTS dual-averaging target acceptance rate (0.8).</param>
/// <param name="MapTaus">MAP pre-selected shrinkage grid (paper: {1, 1e-1, 1e-2, 1e-3}); null uses the default.</param>
/// <param name="MapSteps">MAP Adam iterations per shrinkage value (paper: 1500).</param>
/// <param name="NumCandidates">EI-maximization Sobol screen size <c>Q</c> (paper: 5000).</param>
/// <param name="NumRestarts">EI-maximization L-BFGS-B restarts <c>K</c> (paper: 3).</param>
/// <param name="MaxLbfgsEvals">EI-maximization per-restart L-BFGS-B evaluation cap (paper: 100).</param>
internal readonly record struct SaasboConfig(
    SaasboInferenceMethod Method,
    double Alpha,
    double NoiseVariance,
    double KernelVarianceLogStd,
    int NumWarmup,
    int NumSamples,
    int Thinning,
    int MaxTreeDepth,
    double TargetAccept,
    IReadOnlyList<double>? MapTaus,
    int MapSteps,
    int NumCandidates,
    int NumRestarts,
    int MaxLbfgsEvals);

/// <summary>
///     Executes one SAASBO iteration: fit the SAAS GP to the standardized, maximization-oriented
///     observations and return the normalized point maximizing the sample-averaged EI. Stateless apart from
///     the supplied configuration — the sampler reconstructs the observation set each trial and calls
///     <see cref="Propose" /> once.
/// </summary>
internal static class SaasboOptimizer
{
    /// <summary>
    ///     Runs inference then acquisition and returns the next normalized query point in <c>[0, 1]^dim</c>.
    ///     The targets must be standardized and maximization-oriented (higher is better); the incumbent
    ///     <c>f0</c> is taken as their maximum.
    /// </summary>
    /// <param name="x">Normalized training inputs (<c>n × dim</c>, one completed trial per row).</param>
    /// <param name="y">Standardized, maximization-oriented training targets (length <c>n</c>).</param>
    /// <param name="config">The SAASBO configuration.</param>
    /// <param name="rng">The per-trial MT19937 stream driving NUTS and the Sobol screen.</param>
    /// <returns>The next normalized query point.</returns>
    /// <exception cref="System.ArgumentException">The training shapes are inconsistent.</exception>
    internal static double[] Propose(double[,] x, double[] y, SaasboConfig config, NumPyRandom rng)
    {
        var gp = new SaasGaussianProcess(x, y, config.NoiseVariance);
        int dim = gp.Dim;

        // Incumbent: the best standardized (maximization-oriented) observation.
        double incumbent = double.NegativeInfinity;
        foreach (double v in y)
            if (v > incumbent) incumbent = v;

        IReadOnlyList<SaasHyperparameters> samples =
            config.Method == SaasboInferenceMethod.Nuts
                ? RunNuts(gp, config, rng)
                : new[] { RunMap(gp, config) };

        var acq = new SaasAcquisition(gp, samples, incumbent);

        // The EI-maximization Sobol seed is a fresh draw from the per-trial stream, so it is reproducible yet
        // decorrelated from the NUTS momentum draws that preceded it.
        int seed = (int)rng.randint(0, int.MaxValue);
        return acq.Maximize(config.NumCandidates, config.NumRestarts, seed, config.MaxLbfgsEvals);
    }

    /// <summary>
    ///     Fits the SAAS GP with NUTS and returns the thinned bag of posterior hyperparameter samples. NUTS
    ///     runs in the reparameterized unconstrained coordinates (<see cref="SaasModel" />); the chain starts
    ///     at <c>τ = α</c>, <c>ρ̃ = 1</c>, <c>σ²_k = 1</c> (a prior-centered feasible point), and every
    ///     <see cref="SaasboConfig.Thinning" />-th post-warmup draw is kept.
    /// </summary>
    /// <param name="gp">The training GP.</param>
    /// <param name="config">The SAASBO configuration.</param>
    /// <param name="rng">The per-trial MT19937 stream.</param>
    /// <returns>The retained hyperparameter samples (at least one).</returns>
    private static IReadOnlyList<SaasHyperparameters> RunNuts(
        SaasGaussianProcess gp, SaasboConfig config, NumPyRandom rng)
    {
        var model = new SaasModel(gp, config.Alpha, config.KernelVarianceLogStd);

        // Prior-centered start: log τ = log α, log ρ̃_i = 0, log σ²_k = 0.
        var init = new double[model.LatentDim];
        init[0] = Math.Log(config.Alpha);
        // init[1..dim] and init[dim+1] stay 0 (ρ̃ = 1, σ²_k = 1).

        var nuts = new NoUTurnSampler(config.NumWarmup, config.NumSamples, config.MaxTreeDepth, config.TargetAccept);
        List<double[]> draws = nuts.Sample(model.LogJointAndGrad, init, rng);

        // Thin: keep every `Thinning`-th draw (the last of each block); always keep at least the final draw.
        var kept = new List<SaasHyperparameters>();
        for (int i = 0; i < draws.Count; i++)
            if ((i + 1) % config.Thinning == 0)
                kept.Add(model.HyperparametersFromTheta(draws[i]));
        if (kept.Count == 0 && draws.Count > 0)
            kept.Add(model.HyperparametersFromTheta(draws[^1]));
        if (kept.Count == 0)
            // Degenerate budget (no post-warmup samples): fall back to the prior-centered point so the
            // acquisition still has one usable posterior rather than throwing.
            kept.Add(model.HyperparametersFromTheta(init));
        return kept;
    }

    /// <summary>Fits the SAAS GP with MAP and returns the single leave-one-out-selected hyperparameter estimate.</summary>
    /// <param name="gp">The training GP.</param>
    /// <param name="config">The SAASBO configuration.</param>
    /// <returns>The MAP hyperparameter point estimate.</returns>
    private static SaasHyperparameters RunMap(SaasGaussianProcess gp, SaasboConfig config)
    {
        // MAP removes the τ prior, so α is unused; the log-Normal σ²_k prior std still applies.
        var model = new SaasModel(gp, alpha: 0.1, kernelVarLogStd: config.KernelVarianceLogStd);
        return SaasMapOptimizer.Optimize(model, config.MapTaus, config.MapSteps);
    }
}

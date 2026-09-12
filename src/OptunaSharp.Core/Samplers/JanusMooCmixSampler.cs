// JanusMooCmixSampler — multi-objective JANUS (Plan 05 Phase 3), a faithful port of the authors'
// reference `janus_optuna/janus_moo_cmix.py`.
//
// Reference paper:
//   Hongyuan Yu, Pufan Xu, Jiaojiao Yi, Yiding Tian, Mingrui Sun, Jiayuan Lu, Changyuan Wen.
//   "JANUS: Online Jacobian-Aligned Infill for Black-Box Optimization." arXiv:2608.22862 (2026).
// Reference code: https://github.com/hongyuanyu/JANUS  (Apache-2.0), attributed here.
//
// The reference subclasses Optuna's NSGA-II and, past a warm-up, replaces a child with probability
// `cmix_prob` by an inverse-curvature Gaussian sample estimated from the recent window. NSGA-II still
// owns Pareto selection; CMIX is a small, conservative candidate-generation nudge. Because OptunaSharp's
// `NSGAIISampler` is a bit-exact port and this subclass ALWAYS calls `base.SampleRelative` (advancing
// NSGA-II's MT19937 stream identically) before consulting a SEPARATE CMIX RNG, the NSGA-II trajectory is
// untouched — CMIX only substitutes the returned params on the fraction of trials where it fires. The
// CMIX stream is the reference's exact PCG64 (`np.random.default_rng`) and its Gaussian draw is numpy's
// SVD `multivariate_normal` via OpenBLAS gesdd, so the injection is byte-identical to the reference (not
// merely behavioral) — see JanusMooCmix.cs / JanusMooMetric.cs for the metric geometry and the RNG parity.

using System;
using System.Collections.Generic;
using OptunaSharp.Collections;
using OptunaSharp.Distributions;
using OptunaSharp.Internal;
using OptunaSharp.Samplers.Nsgaii;

namespace OptunaSharp.Samplers;

/// <summary>
///     Multi-objective JANUS: NSGA-II selection with JANUS-style CMIX candidate generation. Past a
///     warm-up, a small fraction (<paramref name="cmixProb" />) of children are replaced by an
///     inverse-curvature Gaussian sample drawn from a metric estimated over the recent window — either
///     from <c>component_losses</c> residuals (the recommended mode) or a diagonal-quadratic scalar
///     fallback — centered on a crowding-ranked point. NSGA-II keeps full control of Pareto selection, so
///     this <b>matches or beats</b> the bare NSGA-II host without a significant regression (the paper's
///     headline MOO property). Port of the authors' <c>janus_optuna.JanusMooCmixSampler</c>
///     (arXiv:2608.22862).
///
///     <para><b>Residual coordinates (optional but recommended).</b> If each objective decomposes into
///     components, expose the residual vector per trial via
///     <c>trial.SetComponentLosses(new double[]{ … })</c>; JANUS then estimates the local metric directly
///     from those residuals. Without them it uses a diagonal-quadratic fit on the window's scalarized
///     values.</para>
///
///     <para><b>Hosts.</b> This is the NSGA-II-hosted sampler the reference ships. The same CMIX engine is
///     host-agnostic; SMS-EMOA- and AGE-MOEA-2-hosted variants (for low- and many-objective suites) reuse
///     it — see <see cref="JanusSmsEmoaCmixSampler" /> and <see cref="JanusAgeMoea2CmixSampler" />.</para>
/// </summary>
/// <example>
///     <code>
///     var study = Optuna.CreateStudy(
///         directions: new[] { StudyDirection.Minimize, StudyDirection.Minimize },
///         sampler: new JanusMooCmixSampler(populationSize: 50, seed: 0));
///     study.Optimize(t =>
///     {
///         double x = t.SuggestFloat("x", 0, 1), y = t.SuggestFloat("y", 0, 1);
///         return new[] { x, (1 - x) + y };
///     }, nTrials: 400);
///     </code>
/// </example>
public class JanusMooCmixSampler : NSGAIISampler
{
    private readonly Janus.JanusMooCmix _cmix;

    /// <param name="populationSize">NSGA-II population size (default 100, the reference value).</param>
    /// <param name="seed">Random seed (seeds both NSGA-II and, separately, the CMIX stream).</param>
    /// <param name="cmixProb">Probability a child is replaced by a CMIX sample past the warm-up (default 0.02).</param>
    /// <param name="cmixWindow">Recent-window size for the metric fit: <c>"auto"</c> (= 3·populationSize) or an integer string.</param>
    /// <param name="cmixRidge">Ridge regularization for the metric regression (default 1.0).</param>
    /// <param name="cmixLambda">Levenberg–Marquardt damping added before inverting the metric (default 1.0).</param>
    /// <param name="cmixSigma">Normalized step size scaling the inverse-curvature covariance (default 0.002).</param>
    /// <param name="cmixStartGeneration">Warm-up in generations, used only when <paramref name="cmixStartTrial" /> is <c>null</c> (default 3).</param>
    /// <param name="cmixStartTrial">Warm-up in trials before CMIX may fire; <c>null</c> derives it from <paramref name="cmixStartGeneration" /> (default 100).</param>
    /// <param name="componentAttr">User-attribute key carrying the residual vector (default <c>component_losses</c>).</param>
    /// <param name="scalarFallback">Scalar metric when no components are present: <c>diag_quadratic</c> (default) or <c>none</c>.</param>
    /// <param name="mutation">NSGA-II mutation (forwarded).</param>
    /// <param name="mutationProb">NSGA-II mutation probability (forwarded).</param>
    /// <param name="crossover">NSGA-II crossover (forwarded).</param>
    /// <param name="crossoverProb">NSGA-II crossover probability (forwarded, default 0.9).</param>
    /// <param name="swappingProb">NSGA-II swapping probability (forwarded, default 0.5).</param>
    public JanusMooCmixSampler(
        int populationSize = 100,
        int? seed = null,
        double cmixProb = 0.02,
        string cmixWindow = "auto",
        double cmixRidge = 1.0,
        double cmixLambda = 1.0,
        double cmixSigma = 0.002,
        int cmixStartGeneration = 3,
        int? cmixStartTrial = 100,
        string componentAttr = "component_losses",
        string scalarFallback = "diag_quadratic",
        BaseMutation? mutation = null,
        double? mutationProb = null,
        BaseCrossover? crossover = null,
        double crossoverProb = 0.9,
        double swappingProb = 0.5)
        : base(populationSize: populationSize, mutation: mutation, mutationProb: mutationProb,
            crossover: crossover, crossoverProb: crossoverProb, swappingProb: swappingProb, seed: seed)
    {
        _cmix = new Janus.JanusMooCmix(
            populationSize, seed, cmixProb, cmixWindow, cmixRidge, cmixLambda, cmixSigma,
            cmixStartGeneration, cmixStartTrial, componentAttr, scalarFallback);
    }

    public override OrderedDict<string, object?> SampleRelative(
        Study study, FrozenTrial trial, OrderedDict<string, BaseDistribution> searchSpace)
    {
        // Always advance the NSGA-II host first (keeps its bit-exact stream identical to bare NSGA-II),
        // then let the shared CMIX engine decide whether to substitute the child.
        var nsgaParams = base.SampleRelative(study, trial, searchSpace);
        return _cmix.Apply(study, trial, searchSpace, nsgaParams);
    }
}

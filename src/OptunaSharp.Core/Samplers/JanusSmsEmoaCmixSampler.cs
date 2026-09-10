// JanusSmsEmoaCmixSampler — JANUS CMIX injection on an SMS-EMOA host (Plan 05 Phase 3).
//
// The multi-objective JANUS CMIX idea (Yu et al., arXiv:2608.22862; https://github.com/hongyuanyu/JANUS,
// Apache-2.0) is host-agnostic: NSGA-II owns Pareto selection and JANUS occasionally substitutes a child
// with an inverse-curvature Gaussian sample from the recent window. The reference ships the NSGA-II host
// (see JanusMooCmixSampler); this variant drives the SAME `JanusMooCmix` engine over the hypervolume-based
// SMS-EMOA host (Plan 04) — the plan's low-objective (2–3) MOO host. As with the NSGA-II sampler, the
// host's `base.SampleRelative` runs first (its stream untouched), then the shared CMIX gate decides.

using System.Collections.Generic;
using OptunaSharp.Collections;
using OptunaSharp.Distributions;
using OptunaSharp.Internal;
using OptunaSharp.Samplers.Nsgaii;
using OptunaSharp.Samplers.SmsEmoa;

namespace OptunaSharp.Samplers;

/// <summary>
///     SMS-EMOA with JANUS-style CMIX candidate generation — the hypervolume-hosted sibling of
///     <see cref="JanusMooCmixSampler" />, for low-objective (2–3) suites. SMS-EMOA keeps full control of
///     its steady-state hypervolume selection; past a warm-up a small fraction of children are replaced by
///     an inverse-curvature Gaussian sample (from <c>component_losses</c> or a diagonal-quadratic scalar
///     fallback). See <see cref="JanusMooCmixSampler" /> for the CMIX parameters and the RNG divergence.
/// </summary>
public sealed class JanusSmsEmoaCmixSampler : SmsEmoaSampler
{
    private readonly Janus.JanusMooCmix _cmix;

    /// <param name="populationSize">SMS-EMOA population size (default 50).</param>
    /// <param name="seed">Random seed (seeds both SMS-EMOA and, separately, the CMIX stream).</param>
    /// <param name="cmixProb">Probability a child is replaced by a CMIX sample past the warm-up (default 0.02).</param>
    /// <param name="cmixWindow">Recent-window size for the metric fit: <c>"auto"</c> (= 3·populationSize) or an integer string.</param>
    /// <param name="cmixRidge">Ridge regularization for the metric regression (default 1.0).</param>
    /// <param name="cmixLambda">Levenberg–Marquardt damping added before inverting the metric (default 1.0).</param>
    /// <param name="cmixSigma">Normalized step size scaling the inverse-curvature covariance (default 0.002).</param>
    /// <param name="cmixStartGeneration">Warm-up in generations, used only when <paramref name="cmixStartTrial" /> is <c>null</c> (default 3).</param>
    /// <param name="cmixStartTrial">Warm-up in trials before CMIX may fire; <c>null</c> derives it from <paramref name="cmixStartGeneration" /> (default 100).</param>
    /// <param name="componentAttr">User-attribute key carrying the residual vector (default <c>component_losses</c>).</param>
    /// <param name="scalarFallback">Scalar metric when no components are present: <c>diag_quadratic</c> (default) or <c>none</c>.</param>
    /// <param name="mutation">SMS-EMOA mutation (forwarded).</param>
    /// <param name="mutationProb">SMS-EMOA mutation probability (forwarded).</param>
    /// <param name="crossover">SMS-EMOA crossover (forwarded).</param>
    /// <param name="crossoverProb">SMS-EMOA crossover probability (forwarded, default 0.9).</param>
    /// <param name="swappingProb">SMS-EMOA swapping probability (forwarded, default 0.5).</param>
    /// <param name="hvReferenceEps">Hypervolume reference-point epsilon (forwarded).</param>
    public JanusSmsEmoaCmixSampler(
        int populationSize = 50,
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
        double swappingProb = 0.5,
        double hvReferenceEps = SmsEmoaSurvival.DefaultEps)
        : base(populationSize: populationSize, mutation: mutation, mutationProb: mutationProb,
            crossover: crossover, crossoverProb: crossoverProb, swappingProb: swappingProb, seed: seed,
            hvReferenceEps: hvReferenceEps)
    {
        _cmix = new Janus.JanusMooCmix(
            populationSize, seed, cmixProb, cmixWindow, cmixRidge, cmixLambda, cmixSigma,
            cmixStartGeneration, cmixStartTrial, componentAttr, scalarFallback);

        OptunaWarnings.Warn(
            "Argument ``JanusSmsEmoaCmixSampler`` is an experimental feature. The interface can change in the future.",
            WarningCategory.ExperimentalWarning);
    }

    public override OrderedDict<string, object?> SampleRelative(
        Study study, FrozenTrial trial, OrderedDict<string, BaseDistribution> searchSpace)
    {
        var hostParams = base.SampleRelative(study, trial, searchSpace);
        return _cmix.Apply(study, trial, searchSpace, hostParams);
    }
}

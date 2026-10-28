// JanusAgeMoea2CmixSampler — JANUS CMIX injection on an AGE-MOEA-2 host (Plan 05 Phase 3).
//
// The multi-objective JANUS CMIX idea (Yu et al., arXiv:2608.22862; https://github.com/hongyuanyu/JANUS,
// Apache-2.0) is host-agnostic. The reference ships the NSGA-II host (see JanusMooCmixSampler); this
// variant drives the SAME `JanusMooCmix` engine over the AGE-MOEA-2 host — the plan's many-objective
// (5/8/10) MOO host, whose survival keeps a well-spread front in high objective dimension. As with the
// NSGA-II sampler, the host's `base.SampleRelative` runs first (its stream untouched), then the shared
// CMIX gate decides whether to substitute the child.

using System.Collections.Generic;
using OptunaSharp.Collections;
using OptunaSharp.Distributions;
using OptunaSharp.Internal;
using OptunaSharp.Samplers.Nsgaii;

namespace OptunaSharp.Samplers;

/// <summary>
///     AGE-MOEA-2 with JANUS-style CMIX candidate generation — the many-objective (5/8/10) sibling of
///     <see cref="JanusMooCmixSampler" />. AGE-MOEA-2 keeps full control of its adaptive-geometry survival;
///     past a warm-up a small fraction of children are replaced by an inverse-curvature Gaussian sample
///     (from <c>component_losses</c> or a diagonal-quadratic scalar fallback). See
///     <see cref="JanusMooCmixSampler" /> for the CMIX parameters and the RNG divergence.
/// </summary>
public sealed class JanusAgeMoea2CmixSampler : AgeMoea2Sampler
{
    private readonly Janus.JanusMooCmix _cmix;

    /// <param name="populationSize">AGE-MOEA-2 population size (default 50).</param>
    /// <param name="seed">Random seed (seeds both AGE-MOEA-2 and, separately, the CMIX stream).</param>
    /// <param name="cmixProb">Probability a child is replaced by a CMIX sample past the warm-up (default 0.02).</param>
    /// <param name="cmixWindow">Recent-window size for the metric fit: <c>"auto"</c> (= 3·populationSize) or an integer string.</param>
    /// <param name="cmixRidge">Ridge regularization for the metric regression (default 1.0).</param>
    /// <param name="cmixLambda">Levenberg–Marquardt damping added before inverting the metric (default 1.0).</param>
    /// <param name="cmixSigma">Normalized step size scaling the inverse-curvature covariance (default 0.002).</param>
    /// <param name="cmixStartGeneration">Warm-up in generations, used only when <paramref name="cmixStartTrial" /> is <c>null</c> (default 3).</param>
    /// <param name="cmixStartTrial">Warm-up in trials before CMIX may fire; <c>null</c> derives it from <paramref name="cmixStartGeneration" /> (default 100).</param>
    /// <param name="componentAttr">
    ///     User-attribute key the CMIX metric reads each trial's residual vector from. The default is
    ///     <c>component_losses</c>, the key <c>trial.SetComponentLosses</c> writes.
    ///     <para>A custom key is read exactly as given, so store the vector with
    ///     <c>trial.SetUserAttr(componentAttr, …)</c>. Only the default key also accepts the legacy <c>rae</c> alias.</para>
    ///     <para>CMIX fits the component metric only when every trial in its window carries a numeric vector under
    ///     this key; otherwise it uses <paramref name="scalarFallback" />. <c>null</c> turns the component channel off.</para>
    /// </param>
    /// <param name="scalarFallback">Scalar metric when no components are present: <c>diag_quadratic</c> (default) or <c>none</c>.</param>
    /// <param name="mutation">AGE-MOEA-2 mutation (forwarded).</param>
    /// <param name="mutationProb">AGE-MOEA-2 mutation probability (forwarded).</param>
    /// <param name="crossover">AGE-MOEA-2 crossover (forwarded).</param>
    /// <param name="crossoverProb">AGE-MOEA-2 crossover probability (forwarded, default 0.9).</param>
    /// <param name="swappingProb">AGE-MOEA-2 swapping probability (forwarded, default 0.5).</param>
    /// <summary>Configures curvature-informed candidate injection into an AGE-MOEA-2 population.</summary>
    public JanusAgeMoea2CmixSampler(
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
        double swappingProb = 0.5)
        : base(populationSize: populationSize, mutation: mutation, mutationProb: mutationProb,
            crossover: crossover, crossoverProb: crossoverProb, swappingProb: swappingProb, seed: seed)
    {
        _cmix = new Janus.JanusMooCmix(
            populationSize, seed, cmixProb, cmixWindow, cmixRidge, cmixLambda, cmixSigma,
            cmixStartGeneration, cmixStartTrial, componentAttr, scalarFallback);
    }

    /// <inheritdoc />
    public override OrderedDict<string, object?> SampleRelative(
        Study study, FrozenTrial trial, OrderedDict<string, BaseDistribution> searchSpace)
    {
        var hostParams = base.SampleRelative(study, trial, searchSpace);
        return _cmix.Apply(study, trial, searchSpace, hostParams);
    }
}

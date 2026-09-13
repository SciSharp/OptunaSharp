// JanusAgeMoea2CmixCptSampler — JANUS CMIX injection on an AGE-MOEA-2-CPT host: the conditional-parameter
// ("Categorical Parameter Trees") variant of JanusAgeMoea2CmixSampler.
//
// The multi-objective JANUS CMIX idea (Yu et al., arXiv:2608.22862; https://github.com/hongyuanyu/JANUS,
// Apache-2.0) is host-agnostic. `JanusAgeMoea2CmixSampler` drives the shared `JanusMooCmix` engine over a
// bit-exact AGE-MOEA-2 host; this variant drives the SAME engine over `AgeMoea2CptSampler`, so it keeps
// AGE-MOEA-2's geometry survival AND evolves conditional (branch-defined) parameters. As with the other
// Janus MOO samplers, the host's `base.SampleRelative` runs first (its stream untouched), then the shared
// CMIX gate decides whether to substitute the child.
//
// Practical note for a categorical / conditional search space (e.g. Rock's gene trees): `JanusMooCmix`
// requires an all-Float/Int relative space to build its transforms (it bails to null on any categorical),
// so on such spaces CMIX NEVER fires and this sampler reduces exactly to `AgeMoea2CptSampler`. The CMIX
// layer earns its keep only on continuous (all-numeric) multi-objective spaces; it is carried here as the
// named JANUS variant so a single class covers both. When CMIX does fire (a numeric conditional space),
// the substituted child is pruned against the conditional rules so it cannot carry inactive-branch params.

using System.Collections.Generic;
using OptunaSharp.Collections;
using OptunaSharp.Distributions;
using OptunaSharp.Internal;
using OptunaSharp.Samplers.Nsgaii;
using OptunaSharp.Samplers.NsgaiiiCpt;

namespace OptunaSharp.Samplers;

/// <summary>
///     AGE-MOEA-2-CPT with JANUS-style CMIX candidate generation — the conditional-parameter sibling of
///     <see cref="JanusAgeMoea2CmixSampler" /> and the JANUS-augmented sibling of <see cref="AgeMoea2CptSampler" />.
///     AGE-MOEA-2 keeps full control of its adaptive-geometry survival, the <see cref="ConditionalParams" />
///     rules keep branch-defined parameters in the evolved space, and — past a warm-up — a small fraction of
///     children are replaced by an inverse-curvature Gaussian sample drawn from the recent window
///     (<c>component_losses</c> residuals or a diagonal-quadratic scalar fallback).
///
///     <para><b>When CMIX contributes nothing.</b> The CMIX engine needs an all-numeric relative search space
///     (it cannot transform categoricals). On a categorical-dominant or conditional-categorical space the gate
///     never fires and this sampler is exactly <see cref="AgeMoea2CptSampler" /> — which is the intended, useful
///     behavior for that setting (the value is AGE-MOEA-2's survival + CPT, not the infill). Reserve the CMIX
///     layer for continuous multi-objective problems; see <see cref="JanusMooCmixSampler" /> for the CMIX
///     parameters and the PCG64 RNG divergence.</para>
/// </summary>
/// <seealso cref="AgeMoea2CptSampler" />
/// <seealso cref="JanusAgeMoea2CmixSampler" />
public sealed class JanusAgeMoea2CmixCptSampler : AgeMoea2CptSampler
{
    private readonly Janus.JanusMooCmix _cmix;
    private readonly ConditionalParams? _conditionalParams;

    /// <param name="populationSize">AGE-MOEA-2 population size (default 50).</param>
    /// <param name="seed">Random seed (seeds AGE-MOEA-2, its independent sampler, and — separately — the CMIX stream).</param>
    /// <param name="cmixProb">Probability a child is replaced by a CMIX sample past the warm-up (default 0.02). Ineffective on a non-numeric space (the gate cannot fire there).</param>
    /// <param name="cmixWindow">Recent-window size for the metric fit: <c>"auto"</c> (= 3·populationSize) or an integer string.</param>
    /// <param name="cmixRidge">Ridge regularization for the metric regression (default 1.0).</param>
    /// <param name="cmixLambda">Levenberg–Marquardt damping added before inverting the metric (default 1.0).</param>
    /// <param name="cmixSigma">Normalized step size scaling the inverse-curvature covariance (default 0.002).</param>
    /// <param name="cmixStartGeneration">Warm-up in generations, used only when <paramref name="cmixStartTrial" /> is <c>null</c> (default 3).</param>
    /// <param name="cmixStartTrial">Warm-up in trials before CMIX may fire; <c>null</c> derives it from <paramref name="cmixStartGeneration" /> (default 100).</param>
    /// <param name="componentAttr">User-attribute key carrying the residual vector (default <c>component_losses</c>).</param>
    /// <param name="scalarFallback">Scalar metric when no components are present: <c>diag_quadratic</c> (default) or <c>none</c>.</param>
    /// <param name="mutation">AGE-MOEA-2 mutation (forwarded; <c>null</c> = the polynomial default, eta 20).</param>
    /// <param name="mutationProb">AGE-MOEA-2 mutation probability (forwarded; <c>null</c> = <c>1 / n_active_params</c>).</param>
    /// <param name="crossover">AGE-MOEA-2 crossover (forwarded; <c>null</c> = SBX, eta 15).</param>
    /// <param name="crossoverProb">AGE-MOEA-2 crossover probability (forwarded, default 0.9).</param>
    /// <param name="swappingProb">AGE-MOEA-2 swapping probability (forwarded, default 0.5).</param>
    /// <param name="conditionalParams">
    ///     Activation rules for branch-defined parameters, forwarded to <see cref="AgeMoea2CptSampler" />. <c>null</c>
    ///     makes the host a plain <see cref="AgeMoea2Sampler" /> under the CMIX layer.
    /// </param>
    /// <summary>Configures curvature-informed candidate injection into an AGE-MOEA-2-CPT population.</summary>
    /// <exception cref="NumSharp.ValueError">Population size is below two or the crossover parent count, or a configured reproduction probability is invalid.</exception>
    public JanusAgeMoea2CmixCptSampler(
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
        ConditionalParams? conditionalParams = null)
        : base(populationSize: populationSize, mutation: mutation, mutationProb: mutationProb,
            crossover: crossover, crossoverProb: crossoverProb, swappingProb: swappingProb, seed: seed,
            conditionalParams: conditionalParams)
    {
        _conditionalParams = conditionalParams;
        _cmix = new Janus.JanusMooCmix(
            populationSize, seed, cmixProb, cmixWindow, cmixRidge, cmixLambda, cmixSigma,
            cmixStartGeneration, cmixStartTrial, componentAttr, scalarFallback);
    }

    /// <inheritdoc />
    public override OrderedDict<string, object?> SampleRelative(
        Study study, FrozenTrial trial, OrderedDict<string, BaseDistribution> searchSpace)
    {
        // Advance the AGE-MOEA-2-CPT host first (its conditional child + dormant carry), keeping its stream
        // identical to the bare host, then let the shared CMIX engine decide whether to substitute the child.
        var hostParams = base.SampleRelative(study, trial, searchSpace);
        var result = _cmix.Apply(study, trial, searchSpace, hostParams);
        // If CMIX substituted (a numeric space where the gate fired), the injected sample spans the relative
        // space without regard to conditional activation; prune inactive-branch parameters so the genotype is
        // consistent with the rules. `ReferenceEquals` distinguishes a substitution from the no-fire pass-through
        // (Apply returns `hostParams` itself when it does not fire), so the host child is never re-pruned.
        if (_conditionalParams is not null && !ReferenceEquals(result, hostParams))
            _conditionalParams.Prune(result, searchSpace);
        return result;
    }
}

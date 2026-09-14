// The common ask/tell surface of the Natural-Evolution-Strategy optimizers (OpenAI-ES / SNES / xNES),
// so one shared `EvolutionStrategySampler` base can drive all three through the Optuna trial protocol.

using System.Collections.Generic;

namespace OptunaSharp.Samplers.EvolutionStrategy;

/// <summary>
///     The minimal ask/tell contract the shared <see cref="EvolutionStrategySampler" /> needs from an
///     Evolution-Strategy optimizer. Implementations work entirely in the transformed <c>[0, 1]^d</c>
///     search box (as produced by <c>SearchSpaceTransform</c> with <c>transform01: true</c>); the sampler
///     clamps each asked candidate back into the box and maps it to external parameter values, and — at
///     tell time — re-transforms the stored trial parameters to reconstruct the population. Every
///     implementation is therefore free of any dependence on persisted ε: state is <c>(mean, scale,
///     generation, seed)</c> plus algorithm-specific adaptation, and the population is recovered from the
///     actually-evaluated (possibly clamped/step-snapped) candidates.
/// </summary>
internal interface IEvolutionStrategyOptimizer
{
    /// <summary>Number of numerical decision coordinates the optimizer searches over.</summary>
    int Dim { get; }

    /// <summary>Number of evaluated candidates that constitute one generation (one adaptation step).</summary>
    int PopulationSize { get; }

    /// <summary>Zero-based count of completed adaptation steps; also selects the current generation's noise.</summary>
    int Generation { get; }

    /// <summary>
    ///     Proposes the <paramref name="index" />-th candidate of the current generation in the transformed
    ///     <c>[0, 1]^d</c> box (pre-clamp). Deterministic in <c>(seed, generation, index)</c>: the whole
    ///     generation's noise is re-derived from the persisted seed, so any trial can produce any index
    ///     reproducibly and the caller may clamp/step-snap the result without corrupting later
    ///     reconstruction.
    /// </summary>
    /// <param name="index">Zero-based position of the candidate within the current generation, in <c>[0, PopulationSize)</c>.</param>
    /// <returns>The proposed decision vector (length <see cref="Dim" />) before box clamping.</returns>
    double[] Ask(int index);

    /// <summary>
    ///     Consumes one full generation of evaluated candidates and performs the optimizer's adaptation
    ///     step, advancing <see cref="Generation" />. Candidates are the <b>actually-evaluated</b> decision
    ///     vectors (transformed back into <c>[0, 1]^d</c>), each paired with its <b>minimization-oriented</b>
    ///     objective value (the caller negates for a maximizing study). Order need not follow the ask index —
    ///     each optimizer recovers the per-candidate noise from the vector itself.
    /// </summary>
    /// <param name="solutions">Exactly <see cref="PopulationSize" /> (decision vector, value) pairs for one generation.</param>
    void Tell(IReadOnlyList<(double[] X, double Value)> solutions);
}

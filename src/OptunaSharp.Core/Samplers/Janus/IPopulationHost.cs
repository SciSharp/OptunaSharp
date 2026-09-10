// The population-host seam JANUS's infill driver plugs into (Plan 03 — host-agnostic infill infra).
//
// JANUS is, in the paper's words, *host-agnostic infill*: it estimates a local Jacobian from the
// recent evaluation trace and contributes a few Jacobian-aligned candidates into an evolutionary
// population, while the population host keeps full control of selection and its own proposal
// adaptation. This interface captures the minimum a host must expose so <see cref="JanusInfill" />
// can drive it without knowing whether the host is CMA-ES, DE, or a future MOO optimizer:
//   • sample a native offspring (Ask) and be told a coherent generation (Tell), and
//   • expose the proposal statistics the JANUS infill needs — the injection center (Mean), the step
//     length (Sigma), and — *optionally* — a proposal covariance used by the CMIX exploration metric
//     and by the bounded-injection whitening.
//
// The covariance is optional by design (paper §"analogous box and step-length constraints"): a host
// that cannot supply one (e.g. DE, which has no Gaussian covariance) reports
// <see cref="HasProposalCovariance" /> = false, and the driver then runs the conservative JANUS-GN
// configuration — the Gauss–Newton injection only, no CMIX candidates.

using System.Collections.Generic;

namespace OptunaSharp.Samplers.Janus;

/// <summary>
///     The population host <see cref="JanusInfill" /> drives: an evolutionary optimizer that proposes
///     a native generation and consumes a coherent one, plus the proposal statistics JANUS's infill
///     reads. <see cref="CmaPopulationHost" /> is the single-objective CMA-ES implementation; DE and
///     MOO hosts (Plan 05 Phases 2/3) plug into the same seam.
/// </summary>
internal interface IPopulationHost
{
    /// <summary>Population size λ the host proposes per generation. JANUS reserves a fixed set of its
    ///     slots (slot 0 for the Gauss–Newton elite, tail slots for CMIX candidates) and fills the
    ///     rest with native offspring, then tells the host generations of exactly this size.</summary>
    int PopulationSize { get; }

    /// <summary>The proposal center (in the transformed box) JANUS injects its Gauss–Newton elite and
    ///     mean-centered CMIX samples around — for CMA-ES the distribution mean <c>m</c>.</summary>
    double[] Mean { get; }

    /// <summary>The proposal step length σ scaling the injected candidates — for CMA-ES the CMA
    ///     step-size. A covariance-free host reports its analogous step length here.</summary>
    double Sigma { get; }

    /// <summary>Draw one native offspring in the transformed box, using the host's own feasibility
    ///     handling (CMA-ES resample-then-repair). With JANUS infill contributing nothing, a run over
    ///     these draws reduces to the bare host.</summary>
    double[] Ask();

    /// <summary>Tell the host exactly one coherent generation — <see cref="PopulationSize" /> solutions
    ///     drawn from a single proposal distribution — so the host's adaptation never mixes samples
    ///     from different distributions. JANUS batches its asked-and-completed trials into these
    ///     generations before calling this.</summary>
    void Tell(IReadOnlyList<(double[] X, double Value)> solutions);

    /// <summary>Whether the host exposes a proposal covariance. When <c>false</c>, JANUS offers no CMIX
    ///     exploration candidates (the paper's GN-only configuration) and the bounded injection
    ///     degrades to the host's box/step constraint via <see cref="WhitenedStepNorm" />.</summary>
    bool HasProposalCovariance { get; }

    /// <summary>trace(C) of the proposal covariance — the CMIX metric rescales its inverse-curvature
    ///     eigenvalues so their sum matches this, keeping the exploration samples on the host's scale.
    ///     Only read when <see cref="HasProposalCovariance" /> is <c>true</c>.</summary>
    double ProposalCovarianceTrace { get; }

    /// <summary>
    ///     The whitened length ‖z‖ of a step <paramref name="diff" /> = x − <see cref="Mean" />, scaled
    ///     by the step length, in the host's proposal metric — the quantity the bounded injection clips
    ///     to κ√d. For CMA-ES this is the square-root-invariant Mahalanobis norm √(diffᵀ C⁻¹ diff)/σ; a
    ///     covariance-free host (DE) returns its analogous box/step-length norm instead. Return a value
    ///     ≤ 0 to signal "no whitening available", in which case the driver injects the candidate
    ///     unclipped (still box-clamped).
    /// </summary>
    double WhitenedStepNorm(double[] diff);

    /// <summary>
    ///     The host's current proposal covariance <c>C</c> (a fresh copy), or <c>null</c> if it exposes
    ///     none. Only meaningful when <see cref="HasProposalCovariance" /> is <c>true</c>; used by JANUS's
    ///     <c>cmix_mode="reshape"</c> to mix an inverse-curvature term into it.
    /// </summary>
    double[,]? GetProposalCovariance();

    /// <summary>
    ///     Replace the host's proposal covariance with <paramref name="c" /> (the host re-derives any
    ///     cached decomposition). Returns <c>false</c> for a covariance-free host (DE), which has none to
    ///     set — the reshape mode is then a no-op there. Only called when <see cref="HasProposalCovariance" />
    ///     is <c>true</c>.
    /// </summary>
    bool TrySetProposalCovariance(double[,] c);
}

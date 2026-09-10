using System;
using OptunaSharp.Samplers;

namespace OptunaSharp.Benchmark.VolleyEval;

/// <summary>
///     A competitor in the profile matrix: a named sampler plus the knobs the <see cref="Evaluator" />
///     needs to run it fairly. A contender is a sampler <b>factory</b> (not an instance) because every
///     (challenge, seed) round must construct a fresh, seeded sampler — reusing one instance would leak
///     state across rounds and break reproducibility.
/// </summary>
public sealed class Contender
{
    /// <param name="name">Row header in the profile matrix (e.g. <c>"cmaes"</c>).</param>
    /// <param name="buildSampler">Constructs a fresh sampler seeded for this round (paper defaults).</param>
    /// <param name="trialCap">
    ///     A per-contender ceiling on trials. <c>null</c> = run the challenge's full budget. Set for
    ///     samplers whose per-trial cost is superlinear (GP's O(n³) refit; MO-TPE's growing density
    ///     estimate) so a large-budget challenge stays tractable — the concrete form of the plan's
    ///     Runner "throttle the expensive contenders". The cap binds only when the challenge budget
    ///     exceeds it, so cheap challenges are unaffected. This realizes the per-wall-clock (practical)
    ///     view: a sample-efficient but expensive sampler is limited to few evaluations, while a cheap
    ///     one runs to convergence.
    /// </param>
    /// <param name="tunedBuild">
    ///     The contender's hyperparameter space, as a single function that <b>suggests</b> its knobs from
    ///     a trial and <b>builds</b> a seeded sampler with them (the plan's <c>HyperparamSpace(Trial)</c> +
    ///     <c>BuildSampler(hp, seed)</c> fused). It runs against a real <see cref="Trial" /> during the
    ///     outer tuning study and against a <see cref="FixedTrial" /> to replay the winning recipe for
    ///     held-out validation — the same function, so the tuned and validated samplers are provably the
    ///     same construction. <c>null</c> for a sampler with no tunable knobs (e.g. random).
    /// </param>
    public Contender(
        string name,
        Func<int, BaseSampler> buildSampler,
        int? trialCap = null,
        Func<BaseTrial, int, BaseSampler>? tunedBuild = null)
    {
        Name = name;
        BuildSampler = buildSampler;
        TrialCap = trialCap;
        TunedBuild = tunedBuild;
    }

    /// <summary>Row header / identifier.</summary>
    public string Name { get; }

    /// <summary>Seed → fresh sampler (paper defaults).</summary>
    public Func<int, BaseSampler> BuildSampler { get; }

    /// <summary>Optional trial ceiling (see constructor); <c>null</c> = unlimited.</summary>
    public int? TrialCap { get; }

    /// <summary>Suggest-and-build over this contender's knob space; <c>null</c> when it has none.</summary>
    public Func<BaseTrial, int, BaseSampler>? TunedBuild { get; }

    /// <summary>True when the contender declares a tunable hyperparameter space.</summary>
    public bool Tunable => TunedBuild is not null;

    /// <summary>The trials this contender actually runs on a challenge of the given budget.</summary>
    public int EffectiveBudget(int challengeBudget)
        => TrialCap is int cap ? Math.Min(challengeBudget, cap) : challengeBudget;
}

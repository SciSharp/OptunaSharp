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
    /// <param name="buildSampler">Constructs a fresh sampler seeded for this round.</param>
    /// <param name="trialCap">
    ///     A per-contender ceiling on trials. <c>null</c> = run the challenge's full budget. Set for
    ///     samplers whose per-trial cost is superlinear (GP's O(n³) refit; MO-TPE's growing density
    ///     estimate) so a large-budget challenge stays tractable — the concrete form of the plan's
    ///     Runner "throttle the expensive contenders". The cap binds only when the challenge budget
    ///     exceeds it, so cheap challenges are unaffected. This realizes the per-wall-clock (practical)
    ///     view: a sample-efficient but expensive sampler is limited to few evaluations, while a cheap
    ///     one runs to convergence.
    /// </param>
    public Contender(string name, Func<int, BaseSampler> buildSampler, int? trialCap = null)
    {
        Name = name;
        BuildSampler = buildSampler;
        TrialCap = trialCap;
    }

    /// <summary>Row header / identifier.</summary>
    public string Name { get; }

    /// <summary>Seed → fresh sampler.</summary>
    public Func<int, BaseSampler> BuildSampler { get; }

    /// <summary>Optional trial ceiling (see constructor); <c>null</c> = unlimited.</summary>
    public int? TrialCap { get; }

    /// <summary>The trials this contender actually runs on a challenge of the given budget.</summary>
    public int EffectiveBudget(int challengeBudget)
        => TrialCap is int cap ? Math.Min(challengeBudget, cap) : challengeBudget;
}

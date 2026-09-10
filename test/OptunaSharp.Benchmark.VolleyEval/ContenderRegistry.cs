using System;
using System.Collections.Generic;
using System.Linq;
using OptunaSharp.Samplers;

namespace OptunaSharp.Benchmark.VolleyEval;

/// <summary>
///     The v0 contender set: the eight competitive native samplers, each with paper-default
///     hyperparameters (v0 does <b>not</b> tune — that is the nested-hyperopt protocol of the full
///     plan). Every sampler is seeded through OptunaSharp's <see cref="LazyRandomState" /> so a round is
///     bit-reproducible. BoTorch / pyCMA interop are intentionally out of scope for v0 (Tier-0, no
///     Python).
///
///     <para>
///         <b>Trial caps.</b> <c>gp</c> and <c>tpe</c> carry caps because their per-trial cost is
///         superlinear: GP refits an O(n³) posterior every trial, and multi-objective TPE's density
///         estimate grows with the trial count. The caps bind only on the large-budget challenge (ZDT1,
///         budget 8000) — GP is held to 80 trials (also keeping it low-D/low-budget, where the plan says
///         it is viable) and TPE to 300; on the small challenges (Rosenbrock 300, NK 200) neither cap
///         binds. This is deliberate and reported: at a large budget the cheap evolutionary champion
///         converges and wins while the sample-efficient-but-expensive samplers are throttled — the
///         plan's two-leaderboards (per-trial vs per-wall-clock) tension made concrete.
///     </para>
/// </summary>
public static class ContenderRegistry
{
    /// <summary>GP's trial ceiling — its cubic refit makes a full 8000-trial run take hours.</summary>
    public const int GpTrialCap = 80;

    /// <summary>Multi-objective TPE's trial ceiling on large-budget challenges.</summary>
    public const int TpeTrialCap = 300;

    /// <summary>The eight native contenders in profile-matrix order.</summary>
    public static IReadOnlyList<Contender> All { get; } = new[]
    {
        new Contender("random", seed => new RandomSampler(seed)),
        new Contender("qmc",    seed => new QMCSampler(seed: seed)),
        new Contender("tpe",    seed => new TPESampler(seed: seed), trialCap: TpeTrialCap),
        new Contender("gp",     seed => new GPSampler(seed: seed),  trialCap: GpTrialCap),
        new Contender("cmaes",  seed => new CmaEsSampler(seed: seed)),
        new Contender("lmcma",  seed => new LmCmaSampler(seed: seed)),
        new Contender("nsga2",  seed => new NSGAIISampler(seed: seed)),
        new Contender("nsga3",  seed => new NSGAIIISampler(seed: seed)),
    };

    /// <summary>
    ///     The cheap subset (everything except GP) — used by the smoke tests, which need the three-radical
    ///     invariants (all satisfied without GP) to run in a few seconds rather than minutes.
    /// </summary>
    public static IReadOnlyList<Contender> Cheap { get; } =
        All.Where(c => c.Name != "gp").ToArray();

    /// <summary>Look up contenders by name (case-insensitive); throws on an unknown name.</summary>
    public static IReadOnlyList<Contender> Select(IEnumerable<string> names)
    {
        var byName = All.ToDictionary(c => c.Name, StringComparer.OrdinalIgnoreCase);
        var result = new List<Contender>();
        foreach (var n in names)
        {
            if (!byName.TryGetValue(n, out var c))
                throw new ArgumentException(
                    $"Unknown contender '{n}'. Known: {string.Join(", ", byName.Keys)}.");
            result.Add(c);
        }
        return result;
    }
}

using System;
using System.Collections.Generic;
using System.Linq;
using OptunaSharp.Benchmark.VolleyEval.Challenges;

namespace OptunaSharp.Benchmark.VolleyEval;

/// <summary>
///     The VolleyEval challenge catalog. The first three are the <b>maximally-orthogonal radicals</b> —
///     continuous-smooth (Rosenbrock) vs discrete-rugged (NK) vs multi-objective (ZDT1), no two sharing a
///     champion. Two further radicals break the assumptions those three all share: <b>determinism</b>
///     (NoisySphere) and being <b>unconstrained</b> (ConstrainedQuadratic). Across the five, every sampler
///     family owns a home: CMA (smooth & noisy), TPE/QMC (rugged categorical), NSGA (Pareto), GP
///     (constrained). Each factory carries the v0-validated defaults.
/// </summary>
public static class ChallengeRegistry
{
    /// <summary>Build the full five-challenge catalog at instance <paramref name="instanceId" />.</summary>
    public static IReadOnlyList<IChallenge> Default(int instanceId = 0) => new IChallenge[]
    {
        new Rosenbrock(dimension: 40, instanceId: instanceId, budget: 300),
        new NkLandscape(n: 24, k: 2, instanceId: instanceId, budget: 200),
        new Zdt1(dimension: 30, instanceId: instanceId, budget: 8000),
        new NoisySphere(dimension: 10, sigma: 8.0, instanceId: instanceId, budget: 300),
        new ConstrainedQuadratic(dimension: 4, radius: 2.5, target: 3.5, instanceId: instanceId, budget: 300),
    };

    /// <summary>The original three maximally-orthogonal radicals (the subset whose winners must all differ).</summary>
    public static IReadOnlyList<IChallenge> ThreeRadicals(int instanceId = 0) => new IChallenge[]
    {
        new Rosenbrock(dimension: 40, instanceId: instanceId, budget: 300),
        new NkLandscape(n: 24, k: 2, instanceId: instanceId, budget: 200),
        new Zdt1(dimension: 30, instanceId: instanceId, budget: 8000),
    };

    /// <summary>Look up challenges by <see cref="IChallenge.Name" /> (case-insensitive); throws on an unknown name.</summary>
    public static IReadOnlyList<IChallenge> Select(IEnumerable<string> names, int instanceId = 0)
    {
        var byName = Default(instanceId).ToDictionary(c => c.Name, StringComparer.OrdinalIgnoreCase);
        var result = new List<IChallenge>();
        foreach (var n in names)
        {
            if (!byName.TryGetValue(n, out var c))
                throw new ArgumentException(
                    $"Unknown challenge '{n}'. Known: {string.Join(", ", byName.Keys)}.");
            result.Add(c);
        }
        return result;
    }
}

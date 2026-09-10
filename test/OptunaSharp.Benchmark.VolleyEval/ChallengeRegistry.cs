using System;
using System.Collections.Generic;
using System.Linq;
using OptunaSharp.Benchmark.VolleyEval.Challenges;

namespace OptunaSharp.Benchmark.VolleyEval;

/// <summary>
///     The v0 three-radical catalog. The challenges are chosen to be <b>maximally orthogonal</b> —
///     continuous-smooth (Rosenbrock) vs discrete-rugged (NK) vs multi-objective (ZDT1) — so no two
///     share a champion and every sampler family has a home on at least one. Each factory carries the
///     v0-validated defaults.
/// </summary>
public static class ChallengeRegistry
{
    /// <summary>Build the three v0 challenges at instance <paramref name="instanceId" /> with their default budgets.</summary>
    public static IReadOnlyList<IChallenge> Default(int instanceId = 0) => new IChallenge[]
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

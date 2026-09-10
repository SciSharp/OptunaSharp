using System;
using System.Collections.Generic;
using System.Linq;

namespace OptunaSharp.Benchmark.VolleyEval;

/// <summary>
///     The Definition of Done encoded as machine-checkable gates over a <see cref="ProfileMatrix" />,
///     shared by the CLI (<c>volley run --assert</c>) and the smoke tests so both enforce the same claims:
///     <list type="number">
///         <item>every challenge's column winner is one of that challenge's <b>home</b> samplers
///               (the best sampler per regime is the regime's home);</item>
///         <item>each home beats <c>random</c>;</item>
///         <item>where a challenge has an <b>away</b> roster, the home beats it;</item>
///         <item>the three maximally-orthogonal radicals (Rosenbrock, NK, ZDT1), when all present, have
///               three <i>different</i> winners.</item>
///     </list>
///     The rosters are the plan's coverage matrix reconciled with what the samplers actually do in
///     OptunaSharp — including the corrections surfaced by live runs (GP is not inert on categoricals;
///     CMA-family is noise-robust; GP is the constrained champion even trial-capped).
/// </summary>
public static class DodGate
{
    private static readonly Dictionary<string, string[]> Homes = new()
    {
        ["rosenbrock"] = new[] { "cmaes", "lmcma" },
        ["nk_landscape"] = new[] { "tpe", "qmc" },
        ["zdt1"] = new[] { "nsga2", "nsga3" },
        ["noisy_sphere"] = new[] { "cmaes", "lmcma" },       // rank-based selection is noise-robust
        // Constraint-aware in OptunaSharp: GP + NSGA auto-detect; TPE's split ranks infeasible trials
        // last. GP is the champion (best feasible with the fewest evaluations); TPE wins when GP is absent.
        ["constrained_quadratic"] = new[] { "gp", "tpe", "nsga2", "nsga3" },
    };

    private static readonly Dictionary<string, string[]> Aways = new()
    {
        ["rosenbrock"] = new[] { "qmc", "nsga2", "nsga3" },
        ["zdt1"] = new[] { "gp", "tpe", "qmc", "random" },
        ["constrained_quadratic"] = new[] { "cmaes", "lmcma" }, // constraint-blind: drive into the infeasible basin
    };

    /// <summary>The three maximally-orthogonal radicals whose winners must all differ.</summary>
    private static readonly string[] ThreeRadicals = { "rosenbrock", "nk_landscape", "zdt1" };

    /// <summary>One DoD gate outcome.</summary>
    public readonly record struct GateResult(string Name, bool Passed, string Detail);

    /// <summary>Evaluate every applicable gate for the matrix (checks skip when their challenge/contenders are absent).</summary>
    public static IReadOnlyList<GateResult> Evaluate(ProfileMatrix matrix)
    {
        var results = new List<GateResult>();
        var present = matrix.Contenders.Select(c => c.Name).ToHashSet(StringComparer.Ordinal);
        var challengeNames = matrix.Challenges.Select(c => c.Name).ToHashSet(StringComparer.Ordinal);

        // Gate 1 — column winner is a home sampler (the per-regime-champion thesis).
        foreach (var ch in matrix.Challenges)
        {
            if (!Homes.TryGetValue(ch.Name, out var homes)) continue;
            var homesPresent = homes.Where(present.Contains).ToArray();
            if (homesPresent.Length == 0) continue;
            string winner = matrix.ColumnWinner(ch.Name);
            bool pass = homesPresent.Contains(winner, StringComparer.Ordinal);
            results.Add(new GateResult($"{ch.Name}:home-wins-column", pass,
                $"winner={winner}, home={{{string.Join(",", homesPresent)}}}"));
        }

        // Gate 2 — home beats random.
        foreach (var ch in matrix.Challenges)
        {
            if (!Homes.TryGetValue(ch.Name, out var homes)) continue;
            var homesPresent = homes.Where(present.Contains).ToArray();
            if (homesPresent.Length == 0 || !present.Contains("random")) continue;
            bool pass = matrix.HomeBeatsRandom(ch.Name, homesPresent);
            double random = matrix.Score(ch.Name, "random");
            results.Add(new GateResult($"{ch.Name}:home>random", pass,
                $"home {{{string.Join(",", homesPresent.Select(h => $"{h}={matrix.Score(ch.Name, h):F4}"))}}} vs random={random:F4}"));
        }

        // Gate 3 — home beats away.
        foreach (var ch in matrix.Challenges)
        {
            if (!Aways.TryGetValue(ch.Name, out var aways) || !Homes.TryGetValue(ch.Name, out var homes)) continue;
            var homesPresent = homes.Where(present.Contains).ToArray();
            var awaysPresent = aways.Where(present.Contains).ToArray();
            if (homesPresent.Length == 0 || awaysPresent.Length == 0) continue;
            bool pass = matrix.HomeBeatsAway(ch.Name, homesPresent, awaysPresent);
            double worstHome = homesPresent.Min(h => matrix.Score(ch.Name, h));
            double bestAway = awaysPresent.Select(a => matrix.Score(ch.Name, a)).Where(v => !double.IsNaN(v)).DefaultIfEmpty(double.NegativeInfinity).Max();
            results.Add(new GateResult($"{ch.Name}:home>away", pass, $"worstHome={worstHome:F4} vs bestAway={bestAway:F4}"));
        }

        // Gate 4 — the three radicals have distinct winners (only when all three are present).
        if (ThreeRadicals.All(challengeNames.Contains))
        {
            var winners = ThreeRadicals.Select(matrix.ColumnWinner).ToArray();
            bool distinct = winners.Distinct(StringComparer.Ordinal).Count() == winners.Length;
            results.Add(new GateResult("three-radicals:distinct-winners", distinct,
                string.Join(", ", ThreeRadicals.Zip(winners, (c, w) => $"{c}={w}"))));
        }

        return results;
    }

    /// <summary>True when every applicable gate passed.</summary>
    public static bool AllPassed(ProfileMatrix matrix) => Evaluate(matrix).All(g => g.Passed);

    /// <summary>Human-readable gate report.</summary>
    public static string Render(ProfileMatrix matrix)
    {
        var lines = Evaluate(matrix)
            .Select(g => $"  [{(g.Passed ? "PASS" : "FAIL")}] {g.Name,-34} {g.Detail}");
        return "DoD gates:" + Environment.NewLine + string.Join(Environment.NewLine, lines);
    }
}

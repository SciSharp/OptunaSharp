using System;
using System.Collections.Generic;
using System.Linq;

namespace OptunaSharp.Benchmark.VolleyEval;

/// <summary>
///     The v0 "Definition of Done" encoded as machine-checkable gates over a <see cref="ProfileMatrix" />,
///     shared by the CLI (<c>volley run --assert</c>) and the smoke tests so both enforce exactly the same
///     claims:
///     <list type="number">
///         <item>the column winner differs across all three challenges (the three-radical thesis);</item>
///         <item>each challenge's <b>home</b> sampler beats <c>random</c>;</item>
///         <item>for the coupled-continuous (Rosenbrock) and multi-objective (ZDT1) radicals, the home
///               also beats the <b>away</b> samplers.</item>
///     </list>
///     The home/away rosters below are the plan's coverage-matrix expectations reconciled with what the
///     samplers actually do in OptunaSharp: NK's dependable engaging samplers are TPE/QMC (NSGA is
///     sample-inefficient at a small categorical budget), and ZDT1's "away" set includes GP (a strong but
///     trial-capped Bayesian optimizer) which the converged NSGA must still out-score.
/// </summary>
public static class DodGate
{
    private static readonly Dictionary<string, string[]> Homes = new()
    {
        ["rosenbrock"] = new[] { "cmaes", "lmcma" },
        ["nk_landscape"] = new[] { "tpe", "qmc" },
        ["zdt1"] = new[] { "nsga2", "nsga3" },
    };

    private static readonly Dictionary<string, string[]> Aways = new()
    {
        ["rosenbrock"] = new[] { "qmc", "nsga2", "nsga3" },
        ["zdt1"] = new[] { "gp", "tpe", "qmc", "random" },
    };

    /// <summary>One DoD gate outcome.</summary>
    public readonly record struct GateResult(string Name, bool Passed, string Detail);

    /// <summary>Evaluate every applicable gate for the matrix (checks are skipped when their challenge/contenders are absent).</summary>
    public static IReadOnlyList<GateResult> Evaluate(ProfileMatrix matrix)
    {
        var results = new List<GateResult>();
        var present = matrix.Contenders.Select(c => c.Name).ToHashSet(StringComparer.Ordinal);
        var challengeNames = matrix.Challenges.Select(c => c.Name).ToHashSet(StringComparer.Ordinal);

        // Gate 1 — distinct column winners (only meaningful with ≥ 2 columns).
        if (matrix.Challenges.Count >= 2)
        {
            bool distinct = matrix.WinnersDistinct();
            results.Add(new GateResult(
                "column-winners-distinct", distinct,
                string.Join(", ", matrix.Challenges.Select(c => $"{c.Name}={matrix.ColumnWinner(c.Name)}"))));
        }

        // Gate 2 — home beats random (each present challenge with a present home).
        foreach (var ch in matrix.Challenges)
        {
            if (!Homes.TryGetValue(ch.Name, out var homes)) continue;
            var homesPresent = homes.Where(present.Contains).ToArray();
            if (homesPresent.Length == 0 || !present.Contains("random")) continue;
            bool pass = matrix.HomeBeatsRandom(ch.Name, homesPresent);
            double random = matrix.Score(ch.Name, "random");
            string detail = $"home {{{string.Join(",", homesPresent.Select(h => $"{h}={matrix.Score(ch.Name, h):F4}"))}}} vs random={random:F4}";
            results.Add(new GateResult($"{ch.Name}:home>random", pass, detail));
        }

        // Gate 3 — home beats away (Rosenbrock, ZDT1).
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

        return results;
    }

    /// <summary>True when every applicable gate passed.</summary>
    public static bool AllPassed(ProfileMatrix matrix) => Evaluate(matrix).All(g => g.Passed);

    /// <summary>Human-readable gate report.</summary>
    public static string Render(ProfileMatrix matrix)
    {
        var lines = Evaluate(matrix)
            .Select(g => $"  [{(g.Passed ? "PASS" : "FAIL")}] {g.Name,-28} {g.Detail}");
        return "DoD gates:" + Environment.NewLine + string.Join(Environment.NewLine, lines);
    }
}

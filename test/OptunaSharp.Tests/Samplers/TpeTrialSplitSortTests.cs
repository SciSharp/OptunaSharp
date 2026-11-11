using System;
using System.Collections.Generic;
using System.Linq;
using AwesomeAssertions;
using OptunaSharp.Collections;
using OptunaSharp.Distributions;
using OptunaSharp.Internal;

namespace OptunaSharp.Tests.Samplers;

/// <summary>
///     Plan 15 (docs/plans/15-performance-pass-3.md), step F: TPE's single-objective split sorts (value, position) structs
///     instead of LINQ's stable <c>OrderBy</c>, builds both sides at their final sizes, sorts them by number on primitive
///     keys, and scores a trial without constraints without building its constraints dictionary. The split must be the
///     same lists in the same order; each test compares it with the replaced code, written out here.
/// </summary>
[TestClass]
public class TpeTrialSplitSortTests
{
    /// <summary>
    ///     Shuffled histories whose values come from a small pool, so ties are everywhere, with both zeros, both
    ///     infinities and NaN in the pool: for both directions and every size of the preferred group, the split equals the
    ///     LINQ stable sort, cut in two and sorted by number, trial for trial.
    /// </summary>
    /// <param name="seed">Seeds the history.</param>
    [TestMethod]
    [DataRow(1)]
    [DataRow(2)]
    [DataRow(3)]
    public void SingleObjectiveSplit_MatchesTheLinqStableSort_TrialForTrial(int seed)
    {
        var rng = new Random(seed);
        double[] pool = { -1.0, -0.0, 0.0, 0.5, 2.0, double.PositiveInfinity, double.NegativeInfinity, double.NaN, 1e-300 };
        int n = 60;
        var numbers = Enumerable.Range(0, n).OrderBy(_ => rng.Next()).ToArray();
        var trials = new List<FrozenTrial>(n);
        foreach (int number in numbers)
            trials.Add(Trial(number, pool[rng.Next(pool.Length)]));

        foreach (StudyDirection direction in new[] { StudyDirection.Minimize, StudyDirection.Maximize })
        {
            Study study = Optuna.CreateStudy(direction: direction);
            for (int nBelow = 0; nBelow <= n; nBelow += 7)
            {
                var (below, above) = TpeTrialSplit.SplitTrials(study, trials, nBelow);
                var (expectedBelow, expectedAbove) = ReplacedSplit(trials, direction, nBelow);
                below.Select(t => t.Number).Should().Equal(expectedBelow.Select(t => t.Number), $"{direction}, n_below={nBelow}");
                above.Select(t => t.Number).Should().Equal(expectedAbove.Select(t => t.Number), $"{direction}, n_below={nBelow}");
            }
        }
    }

    /// <summary>
    ///     A trial with a constraint, in the per-key format or the old list format, still scores as infeasible and goes
    ///     above; the fast path only skips trials that record no constraint at all.
    /// </summary>
    [TestMethod]
    public void InfeasibleTrials_StillScoreThroughTheirConstraints()
    {
        Study study = Optuna.CreateStudy();
        var trials = new List<FrozenTrial>
        {
            Trial(0, 1.0),
            Trial(1, 0.5, new OrderedDict<string, object?> { ["constraints:c"] = 2.0 }),
            Trial(2, 0.25, new OrderedDict<string, object?> { ["constraints"] = new List<object?> { 3.0, -1.0 } }),
            Trial(3, 2.0, new OrderedDict<string, object?> { ["constraints:c"] = -1.0 }),
        };

        var (below, above) = TpeTrialSplit.SplitTrials(study, trials, 2);

        // The feasible trials 0 and 3 are the preferred two (values 1.0 and 2.0); 1 and 2 violate a constraint.
        below.Select(t => t.Number).Should().Equal(0, 3);
        above.Select(t => t.Number).Should().Equal(1, 2);
    }

    /// <summary>
    ///     The single-objective split before plan 15, for complete trials: LINQ's stable sort by value (descending for a
    ///     maximized objective), the first <paramref name="nBelow" /> below and the rest above, each sorted by number.
    /// </summary>
    /// <param name="trials">The complete trials.</param>
    /// <param name="direction">The objective's direction.</param>
    /// <param name="nBelow">The size of the preferred group.</param>
    /// <returns>The two sides.</returns>
    private static (List<FrozenTrial> Below, List<FrozenTrial> Above) ReplacedSplit(
        List<FrozenTrial> trials, StudyDirection direction, int nBelow)
    {
        nBelow = Math.Min(nBelow, trials.Count);
        var sorted = direction == StudyDirection.Minimize
            ? trials.OrderBy(t => t.Value!.Value).ToList()
            : trials.OrderByDescending(t => t.Value!.Value).ToList();
        var below = sorted.Take(nBelow).ToList();
        var above = sorted.Skip(nBelow).ToList();
        below.Sort((a, b) => a.Number.CompareTo(b.Number));
        above.Sort((a, b) => a.Number.CompareTo(b.Number));
        return (below, above);
    }

    /// <summary>A complete trial with the given number, value and system attributes.</summary>
    /// <param name="number">The trial number.</param>
    /// <param name="value">The objective value.</param>
    /// <param name="systemAttrs">The system attributes, where constraints are recorded; none when null.</param>
    /// <returns>The trial.</returns>
    private static FrozenTrial Trial(int number, double value, OrderedDict<string, object?>? systemAttrs = null)
    {
        var parameters = new OrderedDict<string, object?>();
        var distributions = new OrderedDict<string, BaseDistribution>();
        var userAttrs = new OrderedDict<string, object?>();
        var intermediateValues = new OrderedDict<long, double>();
        return new FrozenTrial(number, TrialState.Complete, value, DateTime.UnixEpoch, DateTime.UnixEpoch, parameters,
            distributions, userAttrs, systemAttrs ?? new OrderedDict<string, object?>(), intermediateValues, number);
    }
}

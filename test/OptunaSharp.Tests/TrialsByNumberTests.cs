using System;
using System.Collections.Generic;
using System.Linq;
using AwesomeAssertions;
using OptunaSharp.Collections;
using OptunaSharp.Distributions;
using OptunaSharp.Internal;

namespace OptunaSharp.Tests;

/// <summary>
///     Plan 15 (docs/plans/15-performance-pass-3.md), step L: SMS-EMOA and MOEA/D find their population members by trial
///     number through <see cref="TrialsByNumber" /> instead of dictionaries over the whole history rebuilt per ask. Every
///     lookup must answer as the dictionary it replaces (<c>byNumber[t.Number] = t</c> over the list, in list order) would.
/// </summary>
[TestClass]
public class TrialsByNumberTests
{
    /// <summary>
    ///     Ascending histories with gaps (as a filtered history has): every number from below the first to above the last
    ///     is answered as the dictionary answers it, hits by reference and misses as misses.
    /// </summary>
    /// <param name="seed">Seeds the gaps.</param>
    [TestMethod]
    [DataRow(1)]
    [DataRow(2)]
    [DataRow(3)]
    public void AscendingHistory_AnswersAsTheDictionary(int seed)
    {
        var rng = new Random(seed);
        foreach (int size in new[] { 0, 1, 2, 3, 7, 64, 257 })
        {
            var trials = new List<FrozenTrial>();
            int number = rng.Next(3);
            for (int i = 0; i < size; i++)
            {
                trials.Add(Trial(number));
                number += 1 + rng.Next(3);
            }

            var lookup = TrialsByNumber.Over(trials);
            lookup.IsAscending.Should().BeTrue();
            AssertSameAnswers(lookup, Dictionary(trials), -2, number + 2, $"size {size}");
        }
    }

    /// <summary>
    ///     Histories out of order, or repeating a number: the lookup reports them as not ascending and answers from the
    ///     dictionary itself, the later record winning a repeated number.
    /// </summary>
    [TestMethod]
    public void OutOfOrderOrRepeatedHistory_UsesTheDictionaryItself()
    {
        var shuffled = new List<FrozenTrial> { Trial(4), Trial(1), Trial(9), Trial(0), Trial(5) };
        var outOfOrder = TrialsByNumber.Over(shuffled);
        outOfOrder.IsAscending.Should().BeFalse();
        AssertSameAnswers(outOfOrder, Dictionary(shuffled), -1, 11, "shuffled");

        FrozenTrial early = Trial(3);
        FrozenTrial late = Trial(3);
        var repeated = new List<FrozenTrial> { Trial(1), early, late, Trial(6) };
        var withRepeat = TrialsByNumber.Over(repeated);
        withRepeat.IsAscending.Should().BeFalse();
        withRepeat.TryGetValue(3, out FrozenTrial found).Should().BeTrue();
        ReferenceEquals(found, late).Should().BeTrue("the dictionary keeps the later record");
        AssertSameAnswers(withRepeat, Dictionary(repeated), 0, 8, "repeated");
    }

    /// <summary>A null history is refused.</summary>
    /// <exception cref="ArgumentNullException">Expected: thrown by <see cref="TrialsByNumber.Over" /> and asserted.</exception>
    [TestMethod]
    public void Over_RefusesANullHistory()
    {
        Action act = () => TrialsByNumber.Over(null!);
        act.Should().Throw<ArgumentNullException>();
    }

    /// <summary>Asserts that the lookup and the dictionary agree on every number in a range.</summary>
    /// <param name="lookup">The lookup under test.</param>
    /// <param name="expected">The dictionary the replaced code built.</param>
    /// <param name="from">The first number to ask for.</param>
    /// <param name="to">The last number to ask for.</param>
    /// <param name="label">Names the case in a failure.</param>
    private static void AssertSameAnswers(TrialsByNumber lookup, Dictionary<int, FrozenTrial> expected, int from, int to, string label)
    {
        for (int n = from; n <= to; n++)
        {
            bool hit = lookup.TryGetValue(n, out FrozenTrial actual);
            hit.Should().Be(expected.TryGetValue(n, out FrozenTrial? want), $"{label}: number {n}");
            if (hit)
                ReferenceEquals(actual, want).Should().BeTrue($"{label}: number {n}");
            else
                actual.Should().BeNull($"{label}: number {n}");
        }
    }

    /// <summary>The dictionary the replaced code built: every record by number, a later one replacing an earlier one.</summary>
    /// <param name="trials">The history, in list order.</param>
    /// <returns>The dictionary.</returns>
    private static Dictionary<int, FrozenTrial> Dictionary(List<FrozenTrial> trials)
    {
        var byNumber = new Dictionary<int, FrozenTrial>();
        foreach (FrozenTrial t in trials)
            byNumber[t.Number] = t;
        return byNumber;
    }

    /// <summary>A completed trial record with this number and nothing else.</summary>
    /// <param name="number">The trial number, also used as its id.</param>
    /// <returns>The record.</returns>
    private static FrozenTrial Trial(int number)
    {
        return new FrozenTrial(number, TrialState.Complete, 1.0, DateTime.UnixEpoch, DateTime.UnixEpoch,
            new OrderedDict<string, object?>(), new OrderedDict<string, BaseDistribution>(),
            new OrderedDict<string, object?>(), new OrderedDict<string, object?>(), new OrderedDict<long, double>(), number);
    }
}

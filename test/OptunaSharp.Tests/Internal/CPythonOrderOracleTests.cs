using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using AwesomeAssertions;
using OptunaSharp.Internal;
using OptunaSharp.Samplers.Nsgaiii;

namespace OptunaSharp.Tests.Internal;

/// <summary>
///     Replays the committed corpus <c>oracle/cpython_order.json</c> (CPython 3.12, recorded by
///     <c>oracle/gen_cpython_order_oracle.py</c>) with no Python: two orders Optuna inherits from CPython itself.
///     <list type="bullet">
///         <item><description>
///             <see cref="PyIntSet" />: GridSampler's <c>list(set(range(n)) - set(visited) - set(running))</c> and
///             <c>list(set(range(n)) - set(visited))</c>, in slot order (QA-68), over 271 histories up to 70,000 grids.
///         </description></item>
///         <item><description>
///             <see cref="PyListSort" /> with <see cref="NSGAIIIElitePopulationSelectionStrategy.DistanceIndexLessThan" />:
///             NSGA-III's <c>list.sort(reverse=True)</c> on <c>(distance, index)</c> tuples where distances can be NaN
///             (QA-56), over 353 lists up to 1,000 elements, both directions.
///         </description></item>
///     </list>
///     Long results are compared by count, first and last 32 elements, and the SHA-256 of the decimal elements joined by
///     commas, as the generator wrote them.
/// </summary>
[TestClass]
public class CPythonOrderOracleTests
{
    /// <summary>The generator's alphabet: a sort case stores each distance as two of these, its pool index in base 64.</summary>
    private const string Alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

    /// <summary>Every set-difference case yields CPython's iteration order, for both differences GridSampler takes.</summary>
    [TestMethod]
    public void SetDifferences_IterateInCPythonsSlotOrder()
    {
        JsonElement corpus = LoadCorpus();
        int checkedCases = 0;
        foreach (JsonElement c in corpus.GetProperty("set_difference").EnumerateArray())
        {
            int n = c.GetProperty("n").GetInt32();
            HashSet<int> visited = FromBitmap(c.GetProperty("visited").GetString()!, n);
            HashSet<int> running = FromBitmap(c.GetProperty("running").GetString()!, n);

            List<int> unvisited = PyIntSet.Range(n).Difference(visited).Difference(running).ToList();
            List<int> unfinished = PyIntSet.Range(n).Difference(visited).ToList();

            ShouldMatch(unvisited, c.GetProperty("unvisited"), $"n={n}, |visited|={visited.Count}, |running|={running.Count}");
            ShouldMatch(unfinished, c.GetProperty("unfinished"), $"n={n}, |visited|={visited.Count} (unfinished)");
            checkedCases++;
        }
        checkedCases.Should().Be(271);
    }

    /// <summary>The journal's example, spelled out: the fresh table of the difference wraps 8 and 11 onto slots 0 and 3.</summary>
    [TestMethod]
    public void SetDifference_TheQa68Example_Is_8_11_4()
    {
        var visited = new HashSet<int> { 0, 1, 2, 3, 5, 6, 7, 9, 10, 12, 13, 14, 15 };
        PyIntSet.Range(16).Difference(visited).Difference(new HashSet<int>()).ToList().Should().Equal(8, 11, 4);
    }

    /// <summary>Every sort case leaves the tuples in CPython's order, NaN distances included.</summary>
    [TestMethod]
    public void TupleSorts_WithNaNDistances_LeaveCPythonsOrder()
    {
        JsonElement corpus = LoadCorpus();
        int checkedCases = 0;
        foreach (JsonElement c in corpus.GetProperty("list_sort").EnumerateArray())
        {
            double[] distances = DecodeDistances(c.GetProperty("distances").GetString()!);
            bool reverse = c.GetProperty("reverse").GetBoolean();
            var pairs = distances.Select((d, i) => (Distance: d, Index: i)).ToList();

            PyListSort.Sort(pairs, NSGAIIIElitePopulationSelectionStrategy.DistanceIndexLessThan, reverse);

            ShouldMatch(pairs.Select(p => p.Index).ToList(), c.GetProperty("order"),
                $"{distances.Length} distances ({distances.Count(double.IsNaN)} NaN), reverse={reverse}, case {checkedCases}");
            checkedCases++;
        }
        checkedCases.Should().Be(353);
    }

    /// <summary>
    ///     With a total order the port is an ordinary stable sort: random lists with ties (no NaN) agree with LINQ's stable
    ///     <c>OrderBy</c>, ascending and descending (CPython's reverse keeps equal elements in their original order).
    /// </summary>
    [TestMethod]
    public void WithATotalOrder_ItIsAStableSort()
    {
        var rng = new Random(7);
        for (int trial = 0; trial < 200; trial++)
        {
            int size = rng.Next(0, 600);
            var items = Enumerable.Range(0, size).Select(i => (Key: rng.Next(0, 12), Original: i)).ToList();
            var ascending = items.ToList();
            PyListSort.Sort(ascending, (x, y) => x.Key < y.Key);
            ascending.Should().Equal(items.OrderBy(t => t.Key).ToList());

            var descending = items.ToList();
            PyListSort.Sort(descending, (x, y) => x.Key < y.Key, reverse: true);
            descending.Should().Equal(items.OrderByDescending(t => t.Key).ToList());
        }
    }

    /// <summary>Null arguments are refused; a negative set size is refused.</summary>
    [TestMethod]
    public void InvalidArguments_AreRefused()
    {
        Action noList = () => PyListSort.Sort<int>(null!, (x, y) => x < y);
        Action noComparison = () => PyListSort.Sort(new List<int> { 2, 1 }, null!);
        Action negativeRange = () => PyIntSet.Range(-1);
        Action noOther = () => PyIntSet.Range(3).Difference(null!);
        noList.Should().Throw<ArgumentNullException>();
        noComparison.Should().Throw<ArgumentNullException>();
        negativeRange.Should().Throw<ArgumentOutOfRangeException>();
        noOther.Should().Throw<ArgumentNullException>();
    }

    /// <summary>Loads the committed corpus copied next to the test assembly.</summary>
    /// <returns>The corpus root: <c>set_difference</c> and <c>list_sort</c> case arrays.</returns>
    /// <exception cref="FileNotFoundException">The corpus was not copied to the output directory.</exception>
    private static JsonElement LoadCorpus()
    {
        string path = Path.Combine(AppContext.BaseDirectory, "oracle", "cpython_order.json");
        return JsonDocument.Parse(File.ReadAllText(path)).RootElement.Clone();
    }

    /// <summary>Decodes the generator's base64 little-endian bitmap of a set of ints in <c>range(n)</c>.</summary>
    /// <param name="base64">The bitmap text.</param>
    /// <param name="n">The universe size.</param>
    /// <returns>The set the bitmap encodes.</returns>
    private static HashSet<int> FromBitmap(string base64, int n)
    {
        byte[] data = Convert.FromBase64String(base64);
        var set = new HashSet<int>();
        for (int k = 0; k < n; k++)
        {
            if ((data[k >> 3] & (1 << (k & 7))) != 0)
                set.Add(k);
        }
        return set;
    }

    /// <summary>
    ///     Decodes a sort case's distances: two alphabet characters per distance, a pool index whose values are NaN,
    ///     +inf, -inf, 0.0, -0.0 and then k / 4090 (the same double as Python's, IEEE division being exact-rounded).
    /// </summary>
    /// <param name="text">The encoded distances.</param>
    /// <returns>The distances.</returns>
    private static double[] DecodeDistances(string text)
    {
        var values = new double[text.Length / 2];
        for (int i = 0; i < values.Length; i++)
        {
            int code = (Alphabet.IndexOf(text[2 * i]) << 6) | Alphabet.IndexOf(text[2 * i + 1]);
            values[i] = code switch
            {
                0 => double.NaN,
                1 => double.PositiveInfinity,
                2 => double.NegativeInfinity,
                3 => 0.0,
                4 => -0.0,
                _ => (code - 5) / 4090.0,
            };
        }
        return values;
    }

    /// <summary>Asserts an order equals a corpus description: the full list, or count, head, tail and SHA-256.</summary>
    /// <param name="actual">The order the port produced.</param>
    /// <param name="expected">The corpus description.</param>
    /// <param name="because">What the case is, for the failure message.</param>
    private static void ShouldMatch(List<int> actual, JsonElement expected, string because)
    {
        actual.Should().HaveCount(expected.GetProperty("count").GetInt32(), because);
        if (expected.TryGetProperty("items", out JsonElement items))
        {
            actual.Should().Equal(items.EnumerateArray().Select(e => e.GetInt32()).ToList(), because);
            return;
        }
        actual.Take(32).Should().Equal(expected.GetProperty("head").EnumerateArray().Select(e => e.GetInt32()), because);
        actual.Skip(actual.Count - 32).Should()
            .Equal(expected.GetProperty("tail").EnumerateArray().Select(e => e.GetInt32()), because);
        string digest = Convert.ToHexString(SHA256.HashData(Encoding.UTF8.GetBytes(string.Join(",", actual))))
            .ToLowerInvariant();
        digest.Should().Be(expected.GetProperty("sha256").GetString(), because);
    }
}

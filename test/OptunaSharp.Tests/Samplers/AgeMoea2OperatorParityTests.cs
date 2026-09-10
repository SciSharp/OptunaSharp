using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using AwesomeAssertions;
using OptunaSharp.Samplers;
using OptunaSharp.Samplers.AgeMoea;

namespace OptunaSharp.Tests.Samplers;

/// <summary>
///     Phase 1 gate for the AGE-MOEA-2 port: the pymoo-faithful operators
///     (<c>OptunaSharp.Samplers.AgeMoea</c>) must reproduce pymoo 0.6.1.3's SBX / PM /
///     TournamentSelection / Mating / FloatRandomSampling <b>bit-for-bit</b>, replaying a committed
///     oracle (oracle/gen_agemoea2_oracle.py) with NO Python at test time — the repo's differential
///     corpus model. Each case seeds a <see cref="LazyRandomState" /> with the oracle's seed; the
///     MT19937 stream is 0-ULP identical to numpy's global (verified separately), so byte-exactness
///     hinges on drawing the same shapes in the same order as pymoo.
/// </summary>
[TestClass]
public class AgeMoea2OperatorParityTests
{
    private static JsonElement _oracle;

    [ClassInitialize]
    public static void Load(TestContext _)
    {
        string path = Path.Combine(AppContext.BaseDirectory, "oracle", "agemoea2_operators.json");
        _oracle = JsonDocument.Parse(File.ReadAllText(path)).RootElement.Clone();
    }

    private static AgeMoeaProblem Box(int nVar)
    {
        var xl = new double[nVar];
        var xu = Enumerable.Repeat(1.0, nVar).ToArray();
        return new FunctionProblem(nVar, 2, xl, xu, x => new[] { 0.0, 0.0 });
    }

    /// <summary>Read a double, decoding the non-finite sentinel strings the oracle writes.</summary>
    private static double Num(JsonElement v) => v.ValueKind == JsonValueKind.String
        ? v.GetString() switch
        {
            "Infinity" => double.PositiveInfinity,
            "-Infinity" => double.NegativeInfinity,
            "NaN" => double.NaN,
            var s => throw new FormatException($"unexpected numeric sentinel '{s}'")
        }
        : v.GetDouble();

    private static double[][] Rows(JsonElement e) =>
        e.EnumerateArray().Select(r => r.EnumerateArray().Select(Num).ToArray()).ToArray();

    private static int[][] IntRows(JsonElement e) =>
        e.EnumerateArray().Select(r => r.EnumerateArray().Select(v => v.GetInt32()).ToArray()).ToArray();

    private static void BitEqual(double[][] got, double[][] exp, string what)
    {
        got.Length.Should().Be(exp.Length, $"{what}: row count");
        for (int i = 0; i < got.Length; i++)
        {
            got[i].Length.Should().Be(exp[i].Length, $"{what}: row {i} width");
            for (int j = 0; j < got[i].Length; j++)
                BitConverter.DoubleToInt64Bits(got[i][j])
                    .Should().Be(BitConverter.DoubleToInt64Bits(exp[i][j]),
                        $"{what}: [{i}][{j}] must be bit-identical to pymoo (got {got[i][j]:R}, exp {exp[i][j]:R})");
        }
    }

    [TestMethod]
    public void FloatRandomSampling_IsBitExact()
    {
        foreach (var c in _oracle.GetProperty("sampling").EnumerateArray())
        {
            int seed = c.GetProperty("seed").GetInt32();
            int nVar = c.GetProperty("n_var").GetInt32();
            int n = c.GetProperty("n").GetInt32();
            var rng = new LazyRandomState(seed).Rng;
            double[][] got = new FloatRandomSampling().Do(Box(nVar), n, rng);
            BitEqual(got, Rows(c.GetProperty("X")), $"sampling seed={seed}");
        }
    }

    [TestMethod]
    public void Sbx_IsBitExact()
    {
        foreach (var c in _oracle.GetProperty("crossover").EnumerateArray())
        {
            int seed = c.GetProperty("seed").GetInt32();
            double[][] popX = Rows(c.GetProperty("pop_X"));
            int[][] parents = IntRows(c.GetProperty("parents"));
            var rng = new LazyRandomState(seed).Rng;
            double[][] got = new SimulatedBinaryCrossover(prob: 0.9, eta: 15)
                .Do(popX, parents, Box(popX[0].Length), rng);
            BitEqual(got, Rows(c.GetProperty("off_X")), $"sbx seed={seed}");
        }
    }

    [TestMethod]
    public void Pm_IsBitExact()
    {
        foreach (var c in _oracle.GetProperty("mutation").EnumerateArray())
        {
            int seed = c.GetProperty("seed").GetInt32();
            double[][] inX = Rows(c.GetProperty("in_X"));
            var rng = new LazyRandomState(seed).Rng;
            double[][] got = new PolynomialMutation(eta: 20).Do(inX, Box(inX[0].Length), rng);
            BitEqual(got, Rows(c.GetProperty("out_X")), $"pm seed={seed}");
        }
    }

    [TestMethod]
    public void TournamentSelection_MatchesParentIndices()
    {
        foreach (var c in _oracle.GetProperty("tournament").EnumerateArray())
        {
            int seed = c.GetProperty("seed").GetInt32();
            double[][] popX = Rows(c.GetProperty("pop_X"));
            int[] ranks = c.GetProperty("ranks").EnumerateArray().Select(v => v.GetInt32()).ToArray();
            double[] crowd = c.GetProperty("crowding").EnumerateArray().Select(Num).ToArray();
            int nSelect = c.GetProperty("n_select").GetInt32();

            var pop = Population.New(popX);
            pop.Rank = ranks;
            pop.Crowding = crowd;
            pop.CV = new double[popX.Length];

            var rng = new LazyRandomState(seed).Rng;
            int[][] got = new TournamentSelection().Do(pop, nSelect, 2, rng);
            int[][] exp = IntRows(c.GetProperty("parents"));

            got.Length.Should().Be(exp.Length, $"tournament seed={seed}: n_select");
            for (int i = 0; i < got.Length; i++)
                got[i].Should().Equal(exp[i], $"tournament seed={seed} mating {i}");
        }
    }

    [TestMethod]
    public void Mating_IsBitExact()
    {
        foreach (var c in _oracle.GetProperty("mating").EnumerateArray())
        {
            int seed = c.GetProperty("seed").GetInt32();
            double[][] popX = Rows(c.GetProperty("pop_X"));
            int[] ranks = c.GetProperty("ranks").EnumerateArray().Select(v => v.GetInt32()).ToArray();
            double[] crowd = c.GetProperty("crowding").EnumerateArray().Select(Num).ToArray();
            int nOff = c.GetProperty("n_off").GetInt32();

            var pop = Population.New(popX);
            pop.Rank = ranks;
            pop.Crowding = crowd;
            pop.CV = new double[popX.Length];

            var rng = new LazyRandomState(seed).Rng;
            var mating = new Mating(
                new TournamentSelection(),
                new SimulatedBinaryCrossover(prob: 0.9, eta: 15),
                new PolynomialMutation(eta: 20),
                new DuplicateElimination());
            double[][] got = mating.Do(Box(popX[0].Length), pop, nOff, rng).X;
            BitEqual(got, Rows(c.GetProperty("off_X")), $"mating seed={seed}");
        }
    }
}

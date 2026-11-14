using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
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
///
///     <para>The cases that pass through <c>np.linalg.solve</c> are host-pinned to x64 and report
///     Inconclusive elsewhere: normalization, survival scores, <c>_do</c> and the full driver. That solve is
///     the extreme-point hyperplane in <see cref="AgeMoeaSurvival" />. The committed oracle was generated
///     on OpenBLAS's x64 Haswell kernels, which CI selects with <c>OPENBLAS_CORETYPE=Haswell</c> (except on AMD
///     CPUs with AVX-512, where those cases report Inconclusive: see <c>RequireBitExactLinalgHost</c>), and arm64
///     OpenBLAS kernels land 1–2 ULP away; the macOS arm64 runner showed it on Normalize and Do. This is
///     the same host-pin model as the sklearn, SHAP and interop parity suites. The pure-RNG operators
///     (sampling, SBX, PM, tournament, mating) and the manifold geometry still run, and must hold
///     bit-for-bit, on every architecture.</para>
/// </summary>
[TestClass]
public class AgeMoea2OperatorParityTests
{
    private static JsonElement _oracle;

    /// <summary>
    ///     Skips (Inconclusive) a case whose oracle depends on the LAPACK solve in
    ///     <see cref="AgeMoeaSurvival" />'s normalization, unless the process runs on x64 AND OpenBLAS runs its
    ///     Haswell kernels, the ones that produced the oracle.
    /// </summary>
    /// <remarks>
    ///     The kernel check matters on x64 too: CI pins <c>OPENBLAS_CORETYPE=Haswell</c> except on AMD CPUs with
    ///     AVX-512, where OpenBLAS 0.3.34's forced Haswell kernels fault (<c>.github/scripts/openblas_coretype.sh</c>),
    ///     and OpenBLAS's own choice there (Zen 4/5, 2026-11-14) put Normalize and Do a last bit away. The same holds for
    ///     a local run on any CPU whose default kernels are not Haswell.
    /// </remarks>
    /// <exception cref="AssertInconclusiveException">The process is not x64, or OpenBLAS is not running its Haswell kernels.</exception>
    private static void RequireBitExactLinalgHost()
    {
        if (RuntimeInformation.ProcessArchitecture != Architecture.X64)
            Assert.Inconclusive(
                "AGE-MOEA-2 normalization solves the extreme-point hyperplane with np.linalg.solve (OpenBLAS); " +
                "the committed oracle was captured on x64 Haswell kernels, which " +
                $"ProcessArchitecture={RuntimeInformation.ProcessArchitecture} does not reproduce bit-for-bit.");
        // OpenBlasEngine.CoreName is OpenBLAS's openblas_get_corename(); null only when no OpenBLAS is loaded, which
        // OptunaSharp's module initializer rules out, and then the case cannot be bit-exact either.
        if (NumSharp.Interop.OpenBLAS.OpenBlasEngine.CoreName is not "Haswell")
            Assert.Inconclusive(
                "AGE-MOEA-2 normalization's oracle was captured on OpenBLAS's Haswell kernels; this process runs " +
                $"'{NumSharp.Interop.OpenBLAS.OpenBlasEngine.CoreName}' (OPENBLAS_CORETYPE=" +
                $"{Environment.GetEnvironmentVariable("OPENBLAS_CORETYPE") ?? "unset"}), whose last bits differ.");
    }

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

    // ---- Phase 2: non-dominated sort ranks, corner solutions, normalization ----

    [TestMethod]
    public void NonDominatedSorting_RanksMatch()
    {
        int cs = 0;
        foreach (var c in _oracle.GetProperty("nds").EnumerateArray())
        {
            double[][] f = Rows(c.GetProperty("F"));
            var fronts = NonDominatedSorting.Do(f);

            var rank = Enumerable.Repeat(-1, f.Length).ToArray();
            for (int i = 0; i < fronts.Count; i++)
                foreach (int idx in fronts[i])
                    rank[idx] = i;

            int[] exp = c.GetProperty("rank").EnumerateArray().Select(v => v.GetInt32()).ToArray();
            rank.Should().Equal(exp, $"NDS ranks must match pymoo (case {cs})");
            cs++;
        }
    }

    [TestMethod]
    public void FindCornerSolutions_MatchesExtremeIndices()
    {
        int cs = 0;
        foreach (var c in _oracle.GetProperty("corner").EnumerateArray())
        {
            double[][] front = Rows(c.GetProperty("front"));
            int[] got = AgeMoeaSurvival.FindCornerSolutions(front);
            int[] exp = c.GetProperty("extreme").EnumerateArray().Select(v => v.GetInt32()).ToArray();
            got.Should().Equal(exp, $"corner indices must match pymoo (case {cs})");
            cs++;
        }
    }

    [TestMethod]
    public void Normalize_IsBitExact()
    {
        RequireBitExactLinalgHost();
        int cs = 0;
        foreach (var c in _oracle.GetProperty("normalize").EnumerateArray())
        {
            double[][] front = Rows(c.GetProperty("front"));
            int[] extreme = c.GetProperty("extreme").EnumerateArray().Select(v => v.GetInt32()).ToArray();
            var (normFront, normalization) = AgeMoeaSurvival.Normalize(front, extreme);

            double[] expNorm = c.GetProperty("normalization").EnumerateArray().Select(Num).ToArray();
            for (int j = 0; j < normalization.Length; j++)
                BitConverter.DoubleToInt64Bits(normalization[j])
                    .Should().Be(BitConverter.DoubleToInt64Bits(expNorm[j]),
                        $"normalization[{j}] must be bit-identical to pymoo (case {cs})");

            BitEqual(normFront, Rows(c.GetProperty("norm_front")), $"normalize front (case {cs})");
            cs++;
        }
    }

    // ---- Phase 3: survival geometry (find_zero, project_on_manifold, survival_score, _do) ----

    private static void BitEqual1(double[] got, double[] exp, string what)
    {
        got.Length.Should().Be(exp.Length, $"{what}: length");
        for (int j = 0; j < got.Length; j++)
            BitConverter.DoubleToInt64Bits(got[j])
                .Should().Be(BitConverter.DoubleToInt64Bits(exp[j]),
                    $"{what}: [{j}] must be bit-identical (got {got[j]:R}, exp {exp[j]:R})");
    }

    [TestMethod]
    public void FindZero_IsBitExact()
    {
        int cs = 0;
        foreach (var c in _oracle.GetProperty("find_zero").EnumerateArray())
        {
            double[] point = c.GetProperty("point").EnumerateArray().Select(Num).ToArray();
            int n = c.GetProperty("n").GetInt32();
            double got = AgeMoea2Survival.FindZero(point, n, 0.001);
            double exp = Num(c.GetProperty("p"));
            BitConverter.DoubleToInt64Bits(got).Should().Be(BitConverter.DoubleToInt64Bits(exp),
                $"find_zero (case {cs}) must be bit-identical (got {got:R}, exp {exp:R})");
            cs++;
        }
    }

    [TestMethod]
    public void ProjectOnManifold_IsBitExact()
    {
        int cs = 0;
        foreach (var c in _oracle.GetProperty("project").EnumerateArray())
        {
            double[] point = c.GetProperty("point").EnumerateArray().Select(Num).ToArray();
            double p = Num(c.GetProperty("p"));
            double[] got = AgeMoea2Survival.ProjectOnManifold(point, p);
            BitEqual1(got, c.GetProperty("projected").EnumerateArray().Select(Num).ToArray(), $"project (case {cs})");
            cs++;
        }
    }

    [TestMethod]
    public void SurvivalScore_Age2_IsBitExact() => SurvivalScoreCase("survival_score_age2", new AgeMoea2Survival());

    [TestMethod]
    public void SurvivalScore_Age1_IsBitExact() => SurvivalScoreCase("survival_score_age1", new AgeMoeaSurvival());

    private static void SurvivalScoreCase(string key, AgeMoeaSurvival survival)
    {
        // survival_score normalizes the front first, so it inherits the LAPACK host pin.
        RequireBitExactLinalgHost();
        int cs = 0;
        foreach (var c in _oracle.GetProperty(key).EnumerateArray())
        {
            double[][] front1 = Rows(c.GetProperty("front1"));
            double[] ideal = c.GetProperty("ideal").EnumerateArray().Select(Num).ToArray();
            var (crowd, p, norm) = survival.SurvivalScore(front1, ideal);

            BitConverter.DoubleToInt64Bits(p).Should().Be(BitConverter.DoubleToInt64Bits(Num(c.GetProperty("p"))),
                $"{key} p (case {cs})");
            BitEqual1(crowd, c.GetProperty("crowd_dist").EnumerateArray().Select(Num).ToArray(), $"{key} crowd_dist (case {cs})");
            BitEqual1(norm, c.GetProperty("normalization").EnumerateArray().Select(Num).ToArray(), $"{key} normalization (case {cs})");
            cs++;
        }
    }

    [TestMethod]
    public void Do_Age2_IsBitExact()
    {
        RequireBitExactLinalgHost();
        int cs = 0;
        foreach (var c in _oracle.GetProperty("do_age2").EnumerateArray())
        {
            double[][] f = Rows(c.GetProperty("F"));
            int nSurvive = c.GetProperty("n_survive").GetInt32();
            int nVar = 4;

            // unique X per individual (matching the oracle) so survivors are identifiable
            var x = new double[f.Length][];
            for (int i = 0; i < f.Length; i++)
            {
                x[i] = new double[nVar];
                for (int j = 0; j < nVar; j++) x[i][j] = i * nVar + j;
            }
            var pop = Population.New(x);
            pop.F = f;
            pop.CV = new double[f.Length];

            Population survivors = new AgeMoea2Survival().Do(pop, nSurvive);

            // ranks: compare ranked-ness and the front index for ranked individuals
            // (pymoo's unranked sentinel is np.iinfo(int).max = int64 max; ours is int.MaxValue).
            long[] expRank = c.GetProperty("rank").EnumerateArray().Select(v => v.GetInt64()).ToArray();
            for (int i = 0; i < f.Length; i++)
            {
                bool expRanked = expRank[i] < 1000;
                bool gotRanked = pop.Rank![i] < 1000;
                gotRanked.Should().Be(expRanked, $"do (case {cs}) individual {i} ranked-ness");
                if (expRanked)
                    ((long)pop.Rank![i]).Should().Be(expRank[i], $"do (case {cs}) rank[{i}]");
            }

            BitEqual1(pop.Crowding!, c.GetProperty("crowding").EnumerateArray().Select(Num).ToArray(), $"do (case {cs}) crowding");
            BitEqual(survivors.F!, Rows(c.GetProperty("survivor_F")), $"do (case {cs}) survivor F");
            survivors.Count.Should().Be(nSurvive, $"do (case {cs}) survivor count");
            cs++;
        }
    }

    // ---- Phase 4: full minimize()-shaped driver, generation-by-generation ----

    /// <summary>The oracle's Box objective: distinct polynomial objectives, bit-exact across numpy/C#.</summary>
    private static AgeMoeaProblem BoxDriver(int nVar, int nObj)
    {
        var xl = new double[nVar];
        var xu = Enumerable.Repeat(1.0, nVar).ToArray();
        return new FunctionProblem(nVar, nObj, xl, xu, x =>
        {
            // np.sum(X**2, axis=1) etc. — element squares then numpy's PAIRWISE reduction
            // (8-way unrolled for n>=8), so C# matches numpy bit-for-bit at any n_var.
            var sq0 = new double[x.Length];
            var sq1 = new double[x.Length];
            var sq2 = new double[x.Length];
            for (int k = 0; k < x.Length; k++)
            {
                sq0[k] = x[k] * x[k];
                double d1 = x[k] - 1.0; sq1[k] = d1 * d1;
                double d2 = x[k] - 0.5; sq2[k] = d2 * d2;
            }
            double s0 = OptunaSharp.Internal.MathUtil.PairwiseSum(sq0);
            double s1 = OptunaSharp.Internal.MathUtil.PairwiseSum(sq1);
            double s2 = OptunaSharp.Internal.MathUtil.PairwiseSum(sq2);
            return nObj == 2 ? new[] { s0, s1 } : new[] { s0, s1, s2 };
        });
    }

    [TestMethod]
    public void Driver_IsGenerationByGenerationBitExact()
    {
        // Every generation's survival normalizes through the LAPACK solve.
        RequireBitExactLinalgHost();
        foreach (var c in _oracle.GetProperty("driver").EnumerateArray())
        {
            int seed = c.GetProperty("seed").GetInt32();
            int nVar = c.GetProperty("n_var").GetInt32();
            int nObj = c.GetProperty("n_obj").GetInt32();
            int popSize = c.GetProperty("pop_size").GetInt32();
            int nGen = c.GetProperty("n_gen").GetInt32();

            var optimizer = new AgeMoea2Optimizer(BoxDriver(nVar, nObj), popSize, nGen, seed);
            List<Population> history = optimizer.Run();

            var gens = c.GetProperty("gens").EnumerateArray().ToList();
            history.Count.Should().Be(gens.Count, $"driver seed={seed}: generation count");

            for (int g = 0; g < gens.Count; g++)
            {
                BitEqual(history[g].X, Rows(gens[g].GetProperty("X")), $"driver seed={seed} gen {g + 1} X");
                BitEqual(history[g].F!, Rows(gens[g].GetProperty("F")), $"driver seed={seed} gen {g + 1} F");
            }
        }
    }
}

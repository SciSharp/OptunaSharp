using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.Json;
using AwesomeAssertions;
using OptunaSharp.Samplers;          // LazyRandomState
using OptunaSharp.Samplers.AgeMoea;  // AgeMoea2Optimizer, Population, FunctionProblem + operators (internal, IVT friend)
using Python.Runtime;

namespace OptunaSharp.Tests.Interop;

/// <summary>
///     The <b>live</b> counterpart of the committed AGE-MOEA-2 differential corpus
///     (<c>test/OptunaSharp.Tests/oracle/agemoea2_operators.json</c>, replayed with no Python by
///     <c>AgeMoea2OperatorParityTests</c>): drive the <b>real</b> pymoo 0.6.1.3 AGE-MOEA-2 in-process via
///     pythonnet and assert OptunaSharp's native <see cref="AgeMoea2Optimizer" /> reproduces it
///     <b>generation-by-generation bit-exact</b> — the same equality the corpus checks offline, but
///     against whatever pymoo / numpy / numba the CI box actually has, every run. This closes the
///     AGE-MOEA-2 Phase-5 gate named in <c>docs/AGE_MOEA_2_PORTING.md</c> §7 and turns the fastmath
///     decision-equivalence prediction (§1.1) into a standing check.
///
///     <para><b>Why bit-exactness is reachable live.</b> pymoo 0.6.1.3 seeds numpy's <b>legacy</b> global
///     stream (MT19937 == <see cref="LazyRandomState" />), which is version-stable; libm is the machine's
///     (matched on x64); the intercept solve goes through OpenBLAS (byte-identical to numpy); and the
///     <c>@jit(fastmath=True)</c> survival kernels reduce over only 2–3 objectives, so there is no
///     reassociation freedom — the same reasons the corpus proved 0-ULP across numpy 2.0.2 → 2.4.2. We run
///     <b>stock (JIT-on) pymoo</b>, exactly what Mimicry runs and what generated the corpus; the plan's
///     JIT-off "bit-exact target" is unavailable (pymoo's <c>find_corner_solutions</c> uses
///     <c>numba.intp</c> as a numpy dtype, which breaks un-jitted) and, as measured, identical anyway.</para>
///
///     <para><b>Decision-equivalence ledger.</b> Because the whole run is asserted bit-identical, the
///     max ULP gap vs live pymoo is <b>0</b> — which trivially preserves every survivor / argmin / argmax /
///     tournament decision (§1.1). The driver test accumulates that gap and surfaces it: green means 0 ULP;
///     a future pymoo/numpy/libm drift makes the assertion bite and reports the worst offender.</para>
///
///     <para><b>Crossing.</b> pymoo's per-generation <c>X</c>/<c>F</c> cross as JSON built from Python's
///     round-trippable <c>repr</c> floats — bit-lossless for finite doubles (the box objective is bounded,
///     so no non-finite values appear; <c>allow_nan=False</c> would raise loudly if that ever changed) —
///     the same lossless channel the committed corpus uses. Native generations are computed first in pure
///     C# (no Python); each <see cref="OptunaInteropTestBase" /> helper opens/closes its own GIL scope.</para>
///
///     <para>Self-skips <see cref="Assert.Inconclusive(string)" /> without a usable Python+pymoo (or off the
///     x64 bit-exact host); hard-fails under <c>OPTUNASHARP_PYTHONNET_REQUIRE_ENGINE=1</c> — the interop
///     suite's contract, which CI sets after installing Python + pymoo.</para>
/// </summary>
[TestClass]
public class AgeMoea2ParityTests : OptunaInteropTestBase
{
    /// <summary>Bit-exact parity is host-pinned to x64 (.NET's arm64 libm differs from numpy's by ~1 ULP,
    /// matching the repo's CMA-ES / linalg parity tiers) AND requires pymoo 0.6.1.3 in the interop
    /// interpreter; otherwise Inconclusive.</summary>
    private void RequirePymooOrInconclusive()
    {
        PythonSession.EnsureBitExactHostOrInconclusive();
        try
        {
            // age2 pulls numba (its @jit geometry kernels) — probe the exact class we drive.
            PyExec("from pymoo.algorithms.moo.age2 import AGEMOEA2\n");
        }
        catch (PythonException)
        {
            Assert.Inconclusive(
                "pymoo 0.6.1.3 is not installed in the interop interpreter " +
                "(`pip install pymoo==0.6.1.3 --no-deps` + `pip install autograd alive_progress numba`).");
        }
    }

    /// <summary>
    ///     The oracle's box objective, reproduced bit-for-bit in C#: distinct polynomial objectives
    ///     (<c>Σxᵢ²</c>, <c>Σ(xᵢ−1)²</c>, <c>Σ(xᵢ−½)²</c>) whose element squares match numpy's
    ///     (<c>X**2 == x*x</c>) and whose row reduction uses numpy's PAIRWISE summation
    ///     (<see cref="OptunaSharp.Internal.MathUtil.PairwiseSum" />, 8-way unrolled for <c>n≥8</c>) — so C#
    ///     equals <c>np.sum(X**2, axis=1)</c> at every <c>n_var</c>. Identical to
    ///     <c>AgeMoea2OperatorParityTests.BoxDriver</c> and to the oracle's <c>Box._evaluate</c>.
    /// </summary>
    private static AgeMoeaProblem Box(int nVar, int nObj)
    {
        var xl = new double[nVar];
        var xu = Enumerable.Repeat(1.0, nVar).ToArray();
        return new FunctionProblem(nVar, nObj, xl, xu, x =>
        {
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

    // The live pymoo entry points — the same calls gen_agemoea2_oracle.py makes, one per operator plus the
    // full AGEMOEA2 driver. Each returns round-trippable JSON so the C# side can bit-compare. Defined once
    // per test (the interop scope is per-test); operator inputs come from an independent RandomState(12345)
    // (so they never perturb the seeded global stream the operator draws from) and are returned alongside
    // the output so the native side runs on identical inputs.
    private const string PymooSource =
"""
import numpy as np, json, types
from pymoo.core.problem import Problem
from pymoo.core.population import Population
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.selection.tournament import TournamentSelection
from pymoo.algorithms.moo.nsga2 import binary_tournament
from pymoo.algorithms.moo.age2 import AGEMOEA2


class _OssBox(Problem):
    def __init__(self, n_var, n_obj):
        super().__init__(n_var=n_var, n_obj=n_obj, xl=np.zeros(n_var), xu=np.ones(n_var))

    def _evaluate(self, X, out, *args, **kwargs):
        cols = [np.sum(X ** 2, axis=1), np.sum((X - 1.0) ** 2, axis=1), np.sum((X - 0.5) ** 2, axis=1)]
        out["F"] = np.column_stack(cols[:self.n_obj])


def _oss_agemoea2_driver(seed, n_var, n_obj, pop_size, n_gen):
    problem = _OssBox(n_var, n_obj)
    algo = AGEMOEA2(pop_size=pop_size)
    algo.setup(problem, termination=("n_gen", n_gen), seed=seed, verbose=False)
    gens = []
    while algo.has_next():
        algo.next()
        gens.append(dict(X=algo.pop.get("X").tolist(), F=algo.pop.get("F").tolist()))
    return json.dumps(dict(gens=gens), allow_nan=False)


def _oss_op_sampling(seed, n_var, n):
    problem = _OssBox(n_var, 2)
    np.random.seed(seed)
    X = FloatRandomSampling()(problem, n).get("X")
    return json.dumps(dict(X=X.tolist()), allow_nan=False)


def _oss_op_sbx(seed):
    rng = np.random.RandomState(12345)
    pop_X = rng.random((8, 6))
    parents = rng.randint(0, 8, size=(4, 2))
    problem = _OssBox(6, 2)
    pop = Population.new(X=pop_X)
    np.random.seed(seed)
    off = SBX(prob=0.9, eta=15)(problem, pop, parents=parents).get("X")
    return json.dumps(dict(pop_X=pop_X.tolist(), parents=parents.tolist(), off_X=off.tolist()), allow_nan=False)


def _oss_op_pm(seed):
    rng = np.random.RandomState(12345)
    in_X = rng.random((8, 6))
    problem = _OssBox(6, 2)
    pop = Population.new(X=in_X)
    np.random.seed(seed)
    out = PM(eta=20)(problem, pop).get("X")
    return json.dumps(dict(in_X=in_X.tolist(), out_X=out.tolist()), allow_nan=False)


def _oss_op_tournament(seed):
    rng = np.random.RandomState(12345)
    pop_X = rng.random((10, 5))
    ranks = rng.randint(0, 3, size=10)
    crowd = rng.random(10)
    problem = _OssBox(5, 2)
    pop = Population.new(X=pop_X)
    pop.set("F", np.zeros((10, 2)))
    pop.set("rank", ranks)
    pop.set("crowding", crowd)
    pop.set("CV", np.zeros((10, 1)))
    algo = types.SimpleNamespace(tournament_type="comp_by_rank_and_crowding")
    sel = TournamentSelection(func_comp=binary_tournament)
    np.random.seed(seed)
    parents = sel(problem, pop, 5, n_parents=2, algorithm=algo, to_pop=False)
    return json.dumps(dict(pop_X=pop_X.tolist(), ranks=ranks.tolist(), crowding=crowd.tolist(),
                           parents=np.array(parents).tolist()), allow_nan=False)
""";

    // ---- headline: full driver, generation-by-generation bit-exact vs live pymoo -------------------

    /// <summary>
    ///     For each config on the corpus grid (plus a larger front as the fastmath stress case), run live
    ///     <c>AGEMOEA2</c> and the native <see cref="AgeMoea2Optimizer" /> and assert every generation's
    ///     <c>X</c> and <c>F</c> are bit-identical. The final assertion is the decision-equivalence ledger:
    ///     the max ULP gap over the whole run must be 0.
    /// </summary>
    [TestMethod]
    [TestCategory("RequiresPython")]
    [DataRow(42, 6, 2, 8, 5)]
    [DataRow(7, 5, 2, 10, 6)]
    [DataRow(0, 4, 3, 12, 5)]
    [DataRow(2024, 8, 2, 16, 4)]
    [DataRow(2024, 10, 3, 40, 5)]   // larger 3-objective front — the fastmath decision-equivalence stress case
    public void Driver_IsGenerationByGenerationBitExact_VsLivePymoo(
        int seed, int nVar, int nObj, int popSize, int nGen)
    {
        RequirePymooOrInconclusive();

        // Native side first (pure C#, no Python held across the GIL).
        List<Population> history = new AgeMoea2Optimizer(Box(nVar, nObj), popSize, nGen, seed).Run();

        // Live pymoo side.
        PyExec(PymooSource);
        string json = PyStr($"_oss_agemoea2_driver({seed}, {nVar}, {nObj}, {popSize}, {nGen})");
        using JsonDocument doc = JsonDocument.Parse(json);
        List<JsonElement> gens = doc.RootElement.GetProperty("gens").EnumerateArray().ToList();

        history.Count.Should().Be(gens.Count, $"driver seed={seed}: generation count vs live pymoo");

        int mismatches = 0;
        ulong maxUlp = 0;
        string worst = "(bit-identical)";
        for (int g = 0; g < gens.Count; g++)
        {
            Accumulate(history[g].X, Rows(gens[g].GetProperty("X")),
                $"seed={seed} gen {g + 1} X", ref mismatches, ref maxUlp, ref worst);
            Accumulate(history[g].F!, Rows(gens[g].GetProperty("F")),
                $"seed={seed} gen {g + 1} F", ref mismatches, ref maxUlp, ref worst);
        }

        mismatches.Should().Be(0,
            "AGE-MOEA-2 must be generation-by-generation BIT-EXACT vs live pymoo across the whole run " +
            $"(fastmath decision-equivalence ledger: max ULP gap must be 0; observed {maxUlp} ULP, worst = {worst})");
    }

    // ---- operator spot-checks: the corpus sections, live (localizes any RNG-stream drift) -----------

    [TestMethod]
    [TestCategory("RequiresPython")]
    [DataRow(42, 6, 8)]
    [DataRow(7, 4, 10)]
    [DataRow(0, 8, 20)]
    [DataRow(2024, 3, 5)]
    public void FloatRandomSampling_IsBitExact_VsLivePymoo(int seed, int nVar, int n)
    {
        RequirePymooOrInconclusive();
        PyExec(PymooSource);
        using JsonDocument doc = JsonDocument.Parse(PyStr($"_oss_op_sampling({seed}, {nVar}, {n})"));

        double[][] got = new FloatRandomSampling().Do(Box(nVar, 2), n, new LazyRandomState(seed).Rng);
        BitEqual(got, Rows(doc.RootElement.GetProperty("X")), $"sampling seed={seed}");
    }

    [TestMethod]
    [TestCategory("RequiresPython")]
    [DataRow(42)]
    [DataRow(7)]
    [DataRow(0)]
    [DataRow(2024)]
    public void Sbx_IsBitExact_VsLivePymoo(int seed)
    {
        RequirePymooOrInconclusive();
        PyExec(PymooSource);
        using JsonDocument doc = JsonDocument.Parse(PyStr($"_oss_op_sbx({seed})"));
        JsonElement root = doc.RootElement;

        double[][] popX = Rows(root.GetProperty("pop_X"));
        int[][] parents = IntRows(root.GetProperty("parents"));
        double[][] got = new SimulatedBinaryCrossover(prob: 0.9, eta: 15)
            .Do(popX, parents, Box(popX[0].Length, 2), new LazyRandomState(seed).Rng);
        BitEqual(got, Rows(root.GetProperty("off_X")), $"sbx seed={seed}");
    }

    [TestMethod]
    [TestCategory("RequiresPython")]
    [DataRow(42)]
    [DataRow(7)]
    [DataRow(0)]
    [DataRow(2024)]
    public void Pm_IsBitExact_VsLivePymoo(int seed)
    {
        RequirePymooOrInconclusive();
        PyExec(PymooSource);
        using JsonDocument doc = JsonDocument.Parse(PyStr($"_oss_op_pm({seed})"));
        JsonElement root = doc.RootElement;

        double[][] inX = Rows(root.GetProperty("in_X"));
        double[][] got = new PolynomialMutation(eta: 20).Do(inX, Box(inX[0].Length, 2), new LazyRandomState(seed).Rng);
        BitEqual(got, Rows(root.GetProperty("out_X")), $"pm seed={seed}");
    }

    [TestMethod]
    [TestCategory("RequiresPython")]
    [DataRow(42)]
    [DataRow(7)]
    [DataRow(0)]
    [DataRow(2024)]
    public void TournamentSelection_MatchesLivePymoo(int seed)
    {
        RequirePymooOrInconclusive();
        PyExec(PymooSource);
        using JsonDocument doc = JsonDocument.Parse(PyStr($"_oss_op_tournament({seed})"));
        JsonElement root = doc.RootElement;

        double[][] popX = Rows(root.GetProperty("pop_X"));
        int[] ranks = root.GetProperty("ranks").EnumerateArray().Select(v => v.GetInt32()).ToArray();
        double[] crowd = root.GetProperty("crowding").EnumerateArray().Select(v => v.GetDouble()).ToArray();

        var pop = Population.New(popX);
        pop.Rank = ranks;
        pop.Crowding = crowd;
        pop.CV = new double[popX.Length];

        int[][] got = new TournamentSelection().Do(pop, 5, 2, new LazyRandomState(seed).Rng);
        int[][] exp = IntRows(root.GetProperty("parents"));

        got.Length.Should().Be(exp.Length, $"tournament seed={seed}: n_select");
        for (int i = 0; i < got.Length; i++)
            got[i].Should().Equal(exp[i], $"tournament seed={seed} mating {i}");
    }

    // ---- helpers -----------------------------------------------------------------------------------

    private static double[][] Rows(JsonElement e) =>
        e.EnumerateArray().Select(r => r.EnumerateArray().Select(v => v.GetDouble()).ToArray()).ToArray();

    private static int[][] IntRows(JsonElement e) =>
        e.EnumerateArray().Select(r => r.EnumerateArray().Select(v => v.GetInt32()).ToArray()).ToArray();

    /// <summary>Strict bit-equality (used by the operator spot-checks — no ledger; an operator either
    /// reproduces pymoo's stream exactly or it does not).</summary>
    private static void BitEqual(double[][] got, double[][] exp, string what)
    {
        got.Length.Should().Be(exp.Length, $"{what}: row count");
        for (int i = 0; i < got.Length; i++)
        {
            got[i].Length.Should().Be(exp[i].Length, $"{what}: row {i} width");
            for (int j = 0; j < got[i].Length; j++)
                BitConverter.DoubleToInt64Bits(got[i][j])
                    .Should().Be(BitConverter.DoubleToInt64Bits(exp[i][j]),
                        $"{what}: [{i}][{j}] must be bit-identical to live pymoo (got {got[i][j]:R}, exp {exp[i][j]:R})");
        }
    }

    /// <summary>Compare a generation's matrix bit-for-bit, accumulating the mismatch count and the worst
    /// ULP gap for the driver's decision-equivalence ledger (structural mismatches fail fast).</summary>
    private static void Accumulate(
        double[][] got, double[][] exp, string what, ref int mismatches, ref ulong maxUlp, ref string worst)
    {
        got.Length.Should().Be(exp.Length, $"{what}: row count");
        for (int i = 0; i < got.Length; i++)
        {
            got[i].Length.Should().Be(exp[i].Length, $"{what}: row {i} width");
            for (int j = 0; j < got[i].Length; j++)
            {
                if (BitConverter.DoubleToInt64Bits(got[i][j]) == BitConverter.DoubleToInt64Bits(exp[i][j]))
                    continue;
                mismatches++;
                ulong gap = UlpGap(got[i][j], exp[i][j]);
                if (gap >= maxUlp)
                {
                    maxUlp = gap;
                    worst = $"{what}[{i}][{j}] got {got[i][j]:R} exp {exp[i][j]:R} ({gap} ULP)";
                }
            }
        }
    }

    /// <summary>Unsigned ULP distance between two finite doubles (total-order transform across the sign
    /// boundary). Only reached when bits already differ, purely to size the ledger entry.</summary>
    private static ulong UlpGap(double a, double b)
    {
        long ba = BitConverter.DoubleToInt64Bits(a);
        long bb = BitConverter.DoubleToInt64Bits(b);
        if (ba < 0) ba = long.MinValue - ba;
        if (bb < 0) bb = long.MinValue - bb;
        long d = ba - bb;
        return d < 0 ? (ulong)-d : (ulong)d;
    }
}

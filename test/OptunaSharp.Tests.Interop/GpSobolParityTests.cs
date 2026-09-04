using System;
using System.Globalization;
using System.Text;
using AwesomeAssertions;
using OptunaSharp.Distributions;
using OptunaSharp.Gp;
using OptunaSharp.Internal;
using OptunaSharp.Samplers;

namespace OptunaSharp.Tests.Interop;

/// <summary>
///     Byte-exact parity of the scrambled Sobol' engine (PORTING_PLAN Phase 3.2 part 3c) against
///     <c>scipy.stats.qmc.Sobol(scramble=True)</c>, and of <see cref="GpSearchSpace.SampleNormalizedParams" />
///     against Optuna's <c>_sample_normalized_params</c>. The scramble is bit-exact because NumSharp's
///     <c>np.random.default_rng</c> reproduces NumPy's PCG64 stream and the LMS + Gray-code generation
///     are pure integer bit manipulation — closing the 2.12 scrambling gap.
/// </summary>
[TestClass]
public class GpSobolParityTests : OptunaInteropTestBase
{
    private const string SobolHelper =
        "import warnings\n" +
        "from scipy.stats.qmc import Sobol\n" +
        "def _sobol(d, n, seed):\n" +
        "    with warnings.catch_warnings():\n" +
        "        warnings.simplefilter('ignore')\n" +
        "        return Sobol(d=d, scramble=True, seed=seed).random(n).tolist()\n";

    [TestMethod]
    [TestCategory("RequiresPython")]
    public void ScrambledSobol_ByteExact_Sweep()
    {
        PythonSession.EnsureOrInconclusive();
        PyExec("import numpy as np\n" + SobolHelper);
        var rng = new Random(2718);
        for (int trial = 0; trial < 25; trial++)
        {
            int d = 1 + rng.Next(6);       // 1..6
            int n = 1 << (1 + rng.Next(5)); // 2,4,...,32 (power of 2)
            int seed = rng.Next(1, 2000000000);
            double[,] cs = ScrambledSobol.Random(d, n, seed);
            PyExec(
                $"cs = np.array({Lit(cs)})\n" +
                $"py = np.array(_sobol({d}, {n}, {seed}))\n" +
                "diff = float(np.max(np.abs(py - cs)))\n");
            PyFloat("diff").Should().Be(0.0, $"scrambled Sobol' d={d} n={n} seed={seed}");
        }
    }

    [TestMethod]
    [TestCategory("RequiresPython")]
    public void SampleNormalizedParams_ByteExact()
    {
        PythonSession.EnsureOrInconclusive();
        PyExec(
            "import numpy as np\n" +
            "import optuna._gp.search_space as _ss\n" +
            "from optuna.distributions import FloatDistribution, IntDistribution, CategoricalDistribution\n" +
            "def _sample(seed, n):\n" +
            "    space = {\n" +
            "        'f': FloatDistribution(-3.0, 5.0),\n" +
            "        'lf': FloatDistribution(1e-3, 10.0, log=True),\n" +
            "        'i': IntDistribution(0, 9),\n" +
            "        'df': FloatDistribution(0.0, 1.0, step=0.25),\n" +
            "        'c': CategoricalDistribution(['a','b','c']),\n" +
            "    }\n" +
            "    return _ss.SearchSpace(space).sample_normalized_params(n, np.random.RandomState(seed)).tolist()\n");

        var space = new System.Collections.Generic.Dictionary<string, BaseDistribution>
        {
            ["f"] = new FloatDistribution(-3.0, 5.0),
            ["lf"] = new FloatDistribution(1e-3, 10.0, log: true),
            ["i"] = new IntDistribution(0, 9),
            ["df"] = new FloatDistribution(0.0, 1.0, step: 0.25),
            ["c"] = new CategoricalDistribution((CategoricalChoice)"a", (CategoricalChoice)"b", (CategoricalChoice)"c"),
        };
        var gpSpace = new GpSearchSpace(space);
        foreach (int seed in new[] { 0, 7, 42, 123, 999 })
        {
            double[,] cs = gpSpace.SampleNormalizedParams(8, new LazyRandomState(seed).Rng);
            PyExec($"cs = np.array({Lit(cs)})\npy = np.array(_sample({seed}, 8))\ndiff = float(np.max(np.abs(py - cs)))\n");
            PyFloat("diff").Should().Be(0.0, $"sample_normalized_params seed={seed}");
        }
    }

    private static string Lit(double[,] a)
    {
        int rows = a.GetLength(0), cols = a.GetLength(1);
        var sb = new StringBuilder("[");
        for (int i = 0; i < rows; i++)
        {
            if (i > 0) sb.Append(", ");
            sb.Append('[');
            for (int j = 0; j < cols; j++)
            {
                if (j > 0) sb.Append(", ");
                sb.Append(a[i, j].ToString("R", CultureInfo.InvariantCulture));
            }
            sb.Append(']');
        }
        return sb.Append(']').ToString();
    }
}

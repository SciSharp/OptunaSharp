using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Text;
using AwesomeAssertions;
using OptunaSharp.Distributions;
using OptunaSharp.Gp;
using OptunaSharp.Samplers;

namespace OptunaSharp.Tests.Interop;

/// <summary>
///     Live parity of the mixed continuous/discrete acqf optimizer (PORTING_PLAN Phase 3.2 part 3d)
///     against <c>optuna/_gp/optim_mixed.py: optimize_acqf_mixed</c>. Given an identical fitted GPR,
///     search space, best-warmstart point, and RNG seed, OptunaSharp's <see cref="OptimMixed" /> must
///     reproduce Optuna's chosen <c>(normalized_params, acqf_value)</c>. This deterministically
///     exercises the whole new algorithm — scrambled-Sobol' preliminary sampling, the roulette start
///     selection (<c>rng.choice(replace=False, p=)</c>), the L-BFGS-B gradient ascent with lengthscale
///     preconditioning, and the exhaustive / Brent line searches over the discrete dims.
///
///     <para><b>Tolerance, not bit-for-bit</b> — the GP path rides the managed L-BFGS-B + triangular
///     solve (ULP drift vs scipy's OpenBLAS) and LogEI's <c>erfc</c>/<c>erfcx</c> (the erf/ucrt class),
///     the documented GP boundary. The preliminary samples are bit-exact (scrambled Sobol'), so the
///     start points line up; the local optima then agree to a small tolerance.</para>
/// </summary>
[TestClass]
public class GpOptimMixedParityTests : OptunaInteropTestBase
{
    private const string Helper =
        "import numpy as np, torch\n" +
        "import optuna._gp.gp as _gp\n" +
        "import optuna._gp.acqf as _acqf\n" +
        "import optuna._gp.search_space as _ss\n" +
        "import optuna._gp.optim_mixed as _om\n" +
        "from optuna.distributions import FloatDistribution, IntDistribution, CategoricalDistribution\n" +
        "def _optim_mixed(X, Y, isc, isl, ks, nv, threshold, best, space, seed, cs_param):\n" +
        "    gpr = _gp.GPRegressor(is_categorical=torch.from_numpy(isc), X_train=torch.from_numpy(X),\n" +
        "        y_train=torch.from_numpy(Y), inverse_squared_lengthscales=torch.from_numpy(isl),\n" +
        "        kernel_scale=torch.tensor(ks, dtype=torch.float64), noise_var=torch.tensor(nv, dtype=torch.float64))\n" +
        "    gpr._cache_matrix()\n" +
        "    S = _ss.SearchSpace(space)\n" +
        "    a = _acqf.LogEI(gpr=gpr, search_space=S, threshold=threshold)\n" +
        "    rng = np.random.RandomState(seed)\n" +
        "    param, val = _om.optimize_acqf_mixed(a, warmstart_normalized_params_array=best,\n" +
        "        n_preliminary_samples=2048, n_local_search=10, tol=1e-4, rng=rng)\n" +
        "    acqf_at_cs = float(a.eval_acqf_no_grad(cs_param))\n" +  // Optuna's acqf at OptunaSharp's optimum
        "    return [float(v) for v in param], float(val), acqf_at_cs\n";

    [TestMethod]
    [TestCategory("RequiresPython")]
    public void OptimizeAcqfMixed_MatchesOptuna_MixedSpaces()
    {
        PythonSession.EnsureOrInconclusive();
        PyExec(Helper);

        // (name, distributions, inverse-squared-lengthscales, kernel_scale, noise_var, gen-seed, rng-seed)
        RunScenario("float2_int_small",
            new BaseDistribution[] { new FloatDistribution(0, 1), new FloatDistribution(0, 1), new IntDistribution(0, 10) },
            new[] { 2.0, 3.0, 1.5 }, 1.0, 0.01, 7, 42);
        RunScenario("float_int_big",  // 101 int choices → the Brent discrete line search
            new BaseDistribution[] { new FloatDistribution(0, 1), new IntDistribution(0, 100) },
            new[] { 4.0, 2.5 }, 1.2, 0.02, 3, 123);
        RunScenario("float_cat",
            new BaseDistribution[] { new FloatDistribution(0, 1), new CategoricalDistribution("a", "b", "c", "d") },
            new[] { 3.0, 5.0 }, 0.8, 0.05, 11, 55);
        RunScenario("all_float3",
            new BaseDistribution[] { new FloatDistribution(0, 1), new FloatDistribution(0, 1), new FloatDistribution(0, 1) },
            new[] { 2.0, 4.0, 6.0 }, 1.0, 0.03, 21, 999);
        RunScenario("logfloat_int",
            new BaseDistribution[] { new FloatDistribution(1e-3, 1.0, log: true), new IntDistribution(1, 20) },
            new[] { 3.0, 2.0 }, 1.1, 0.02, 31, 77);
    }

    private void RunScenario(
        string name, BaseDistribution[] dists, double[] isl, double ks, double nv, int genSeed, int rngSeed)
    {
        int dim = dists.Length;
        var space = new Dictionary<string, BaseDistribution>();
        for (int i = 0; i < dim; i++)
            space[$"p{i}"] = dists[i];
        var gpSpace = new GpSearchSpace(space);

        // Realistic normalized training data — discrete dims land on-grid, categoricals are codes.
        int nPoints = 10 + dim;
        var gen = new LazyRandomState(genSeed).Rng;
        double[,] x = gpSpace.SampleNormalizedParams(nPoints, gen);
        var y = new double[nPoints];
        var yRaw = gen.random_sample((long)nPoints).ToArray<double>();
        for (int i = 0; i < nPoints; i++) y[i] = yRaw[i] * 2 - 1;

        var isc = new bool[dim];
        for (int i = 0; i < dim; i++) isc[i] = dists[i] is CategoricalDistribution;

        var gpr = new GPRegressor(isc, x, y, isl, ks, nv);
        gpr.CacheMatrix();
        double threshold = y.Max();
        var acqf = new LogEI(gpr, gpSpace, threshold);

        int argmax = 0;
        for (int i = 1; i < nPoints; i++) if (y[i] > y[argmax]) argmax = i;
        var best = new double[1, dim];
        for (int d = 0; d < dim; d++) best[0, d] = x[argmax, d];

        var rng = new LazyRandomState(rngSeed).Rng;
        (double[] csParam, double csVal) = OptimMixed.OptimizeAcqfMixed(acqf, best, 2048, 10, 1e-4, rng);

        PyExec(
            $"X = np.array({Lit(x)}, dtype=float)\nY = np.array({Lit(y)}, dtype=float)\n" +
            $"isc = np.array({Lit(isc)})\nisl = np.array({Lit(isl)}, dtype=float)\n" +
            $"ks = {Fmt(ks)}\nnv = {Fmt(nv)}\nthr = {Fmt(threshold)}\n" +
            $"best = np.array({Lit(best)}, dtype=float)\nspace = {PySpace(dists)}\n" +
            $"cs_param = np.array({Lit(csParam)})\ncs_val = {Fmt(csVal)}\n" +
            $"py_param, py_val, acqf_at_cs = _optim_mixed(X, Y, isc, isl, ks, nv, thr, best, space, {rngSeed}, cs_param)\n" +
            "py_param = np.array(py_param)\n" +
            "d_param = float(np.max(np.abs(py_param - cs_param)))\n" +
            "d_val = abs(py_val - cs_val) / (1.0 + abs(py_val))\n" +
            // Is OptunaSharp's optimum as good as Optuna's, scored by Optuna's own acqf?
            "d_opt = abs(py_val - acqf_at_cs) / (1.0 + abs(py_val))\n");

        // The achieved acqf value is the meaningful quantity: L-BFGS-B stops at pgtol=sqrt(1e-4)=1e-2,
        // so the argmin can drift ~1e-4 in a flat acqf region (managed vs scipy BLAS), while the value
        // agrees tightly — the documented GP boundary ("validate by achieved value, not argmin").
        PyFloat("d_val").Should().BeLessThan(1e-6, $"{name}: achieved acqf value");
        PyFloat("d_opt").Should().BeLessThan(1e-5, $"{name}: OptunaSharp's optimum under Optuna's acqf");
        PyFloat("d_param").Should().BeLessThan(5e-3, $"{name}: normalized params (loose — flat-region argmin drift)");
    }

    // ---- python literal emitters ----------------------------------------------------------------

    private static string PySpace(BaseDistribution[] dists)
    {
        var sb = new StringBuilder("{");
        for (int i = 0; i < dists.Length; i++)
        {
            if (i > 0) sb.Append(", ");
            sb.Append($"'p{i}': ");
            switch (dists[i])
            {
                case FloatDistribution f:
                    sb.Append($"FloatDistribution({Fmt(f.Low)}, {Fmt(f.High)}, log={(f.Log ? "True" : "False")}");
                    sb.Append(f.Step.HasValue ? $", step={Fmt(f.Step.Value)})" : ")");
                    break;
                case IntDistribution n:
                    sb.Append($"IntDistribution({n.Low}, {n.High}, log={(n.Log ? "True" : "False")}, step={n.Step})");
                    break;
                case CategoricalDistribution c:
                    sb.Append("CategoricalDistribution([");
                    sb.Append(string.Join(", ", c.Choices.Select(ch => $"'{ch.Value}'")));
                    sb.Append("])");
                    break;
            }
        }
        return sb.Append('}').ToString();
    }

    private static string Fmt(double d) => d.ToString("R", CultureInfo.InvariantCulture);

    private static string Lit(double[] a)
    {
        var sb = new StringBuilder("[");
        for (int i = 0; i < a.Length; i++) { if (i > 0) sb.Append(", "); sb.Append(Fmt(a[i])); }
        return sb.Append(']').ToString();
    }

    private static string Lit(bool[] a)
    {
        var sb = new StringBuilder("[");
        for (int i = 0; i < a.Length; i++) { if (i > 0) sb.Append(", "); sb.Append(a[i] ? "True" : "False"); }
        return sb.Append(']').ToString();
    }

    private static string Lit(double[,] a)
    {
        int rows = a.GetLength(0), cols = a.GetLength(1);
        var sb = new StringBuilder("[");
        for (int i = 0; i < rows; i++)
        {
            if (i > 0) sb.Append(", ");
            sb.Append('[');
            for (int j = 0; j < cols; j++) { if (j > 0) sb.Append(", "); sb.Append(Fmt(a[i, j])); }
            sb.Append(']');
        }
        return sb.Append(']').ToString();
    }
}

using System;
using System.Globalization;
using System.Text;
using AwesomeAssertions;
using OptunaSharp.Distributions;
using OptunaSharp.Gp;

namespace OptunaSharp.Tests.Interop;

/// <summary>
///     Live parity of the multi-objective GP acquisition function <see cref="LogEHVI" /> (PORTING_PLAN
///     Phase 3.2 part 3e-4) against <c>optuna/_gp/acqf.py: LogEHVI</c>. Each objective's GP is built from
///     <b>fixed</b> kernel hyperparameters (no fit) and both sides draw the QMC fixed samples from the
///     same <c>qmc_seed</c> (<c>sample_from_normal_sobol</c>, matched to ~1e-15 in part 3e-3), so the
///     comparison isolates the box-decomposition + log-sum-exp acqf math and its hand-derived gradient
///     from the fit's ULP drift. LogEHVI rides the GP posterior and a <c>log</c>/<c>logsumexp</c> — the
///     erf/ucrt-class GP boundary — so these are magnitude-relative tolerance compares, not bit-for-bit.
///
///     <para>Optuna's autograd can emit <c>nan</c> gradients at box-clamp kinks; those are excluded
///     (finite-only), matching the single-objective acqf parity policy.</para>
/// </summary>
[TestClass]
public class GpMultiObjectiveAcqfParityTests : OptunaInteropTestBase
{
    private const string Helper =
        "import numpy as np, torch\n" +
        "import optuna._gp.gp as _gp\n" +
        "import optuna._gp.acqf as _acqf\n" +
        "import optuna._gp.search_space as _ss\n" +
        "from optuna.distributions import FloatDistribution\n" +
        "def _build(X, Y, isc, isl, ks, nv):\n" +
        "    gpr = _gp.GPRegressor(is_categorical=torch.from_numpy(isc), X_train=torch.from_numpy(X),\n" +
        "        y_train=torch.from_numpy(Y), inverse_squared_lengthscales=torch.from_numpy(isl),\n" +
        "        kernel_scale=torch.tensor(ks, dtype=torch.float64), noise_var=torch.tensor(nv, dtype=torch.float64))\n" +
        "    gpr._cache_matrix()\n" +
        "    return gpr\n" +
        "def _logehvi(X, isc, Ytrain, Ys, isls, kss, nvs, n_qmc, qmc_seed, xs):\n" +
        "    gprs = [_build(X, np.array(Ys[i]), isc, np.array(isls[i]), kss[i], nvs[i]) for i in range(len(Ys))]\n" +
        "    S = _ss.SearchSpace({f'p{i}': FloatDistribution(0.0, 1.0) for i in range(X.shape[1])})\n" +
        "    a = _acqf.LogEHVI(gpr_list=gprs, search_space=S, Y_train=torch.from_numpy(Ytrain),\n" +
        "        n_qmc_samples=n_qmc, qmc_seed=qmc_seed)\n" +
        "    ng = a.eval_acqf_no_grad(xs).tolist()\n" +
        "    grads = []\n" +
        "    for x in xs:\n" +
        "        _, g = a.eval_acqf_with_grad(x)\n" +
        "        grads.append([float(v) for v in g])\n" +
        "    return ng, grads\n";

    [TestMethod]
    [TestCategory("RequiresPython")]
    public void LogEHVI_EvalAndGrad_MatchesOptuna_Sweep()
    {
        PythonSession.EnsureOrInconclusive();
        PyExec(Helper);
        var rng = new Random(24680);
        for (int trial = 0; trial < 12; trial++)
        {
            int nParams = 1 + rng.Next(2);       // 1..2 continuous dims
            int nObj = 2 + rng.Next(2);          // 2..3 objectives
            int nPoints = nParams + 5 + rng.Next(4);
            int nQuery = 3 + rng.Next(3);
            int nQmc = 64;
            int qmcSeed = rng.Next(1, 1_000_000);
            var isCat = new bool[nParams];

            var x = new double[nPoints, nParams];
            for (int i = 0; i < nPoints; i++)
                for (int d = 0; d < nParams; d++)
                    x[i, d] = rng.NextDouble();

            var ys = new double[nObj][];
            var isls = new double[nObj][];
            var kss = new double[nObj];
            var nvs = new double[nObj];
            var yTrain = new double[nPoints, nObj];
            var gprs = new GPRegressor[nObj];
            for (int o = 0; o < nObj; o++)
            {
                var y = new double[nPoints];
                for (int i = 0; i < nPoints; i++) { y[i] = rng.NextDouble() * 2 - 1; yTrain[i, o] = y[i]; }
                var isl = new double[nParams];
                for (int d = 0; d < nParams; d++) isl[d] = 0.5 + rng.NextDouble() * 8;
                double ks = 0.3 + rng.NextDouble() * 2;
                double nv = 1e-4 + rng.NextDouble() * 0.2;
                ys[o] = y; isls[o] = isl; kss[o] = ks; nvs[o] = nv;
                var gpr = new GPRegressor(isCat, x, y, isl, ks, nv);
                gpr.CacheMatrix();
                gprs[o] = gpr;
            }

            var acqf = new LogEHVI(gprs, DummySpace(nParams), yTrain, nQmc, qmcSeed);
            var xs = new double[nQuery, nParams];
            for (int i = 0; i < nQuery; i++)
                for (int d = 0; d < nParams; d++)
                    xs[i, d] = rng.NextDouble();

            double[] ng = acqf.EvalAcqfNoGrad(xs);
            var grads = new double[nQuery][];
            for (int i = 0; i < nQuery; i++)
            {
                var row = new double[nParams];
                for (int d = 0; d < nParams; d++) row[d] = xs[i, d];
                grads[i] = acqf.EvalAcqfWithGrad(row).Grad;
            }

            PyExec(
                $"X = np.array({Lit(x)}, dtype=float)\nisc = np.array({Lit(isCat)})\n" +
                $"Ytrain = np.array({Lit(yTrain)}, dtype=float)\n" +
                $"Ys = {Lit2(ys)}\nisls = {Lit2(isls)}\nkss = {Lit(kss)}\nnvs = {Lit(nvs)}\n" +
                $"xs = np.array({Lit(xs)}, dtype=float)\n" +
                $"ng_cs = np.array({Lit(ng)})\ngr_cs = {Lit2(grads)}\n" +
                $"ng_py, gr_py = _logehvi(X, isc, Ytrain, Ys, isls, kss, nvs, {nQmc}, {qmcSeed}, xs)\n" +
                "ng_py = np.array(ng_py)\n" +
                "d_eval = float(np.max(np.abs(ng_py - ng_cs) / (1.0 + np.abs(ng_py))))\n" +
                "d_grad = 0.0\n" +
                "for i in range(len(gr_py)):\n" +
                "    for j in range(len(gr_py[i])):\n" +
                "        gp_ = gr_py[i][j]\n" +
                "        if np.isfinite(gp_):\n" +
                "            d_grad = max(d_grad, abs(gp_ - gr_cs[i][j]) / (1.0 + abs(gp_)))\n");
            PyFloat("d_eval").Should().BeLessThan(1e-7, $"LogEHVI eval trial={trial} nObj={nObj}");
            PyFloat("d_grad").Should().BeLessThan(1e-5, $"LogEHVI grad trial={trial} nObj={nObj}");
        }
    }

    private static GpSearchSpace DummySpace(int dim)
    {
        var d = new System.Collections.Generic.Dictionary<string, BaseDistribution>();
        for (int i = 0; i < dim; i++)
            d[$"p{i}"] = new FloatDistribution(0.0, 1.0);
        return new GpSearchSpace(d);
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

    private static string Lit2(double[][] a)
    {
        var sb = new StringBuilder("[");
        for (int i = 0; i < a.Length; i++) { if (i > 0) sb.Append(", "); sb.Append(Lit(a[i])); }
        return sb.Append(']').ToString();
    }
}

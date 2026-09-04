using System;
using System.Globalization;
using System.Text;
using AwesomeAssertions;
using OptunaSharp.Distributions;
using OptunaSharp.Gp;

namespace OptunaSharp.Tests.Interop;

/// <summary>
///     Live parity of the GP LogEI acquisition function (PORTING_PLAN Phase 3.2 part 3b) against
///     <c>optuna/_gp/acqf.py</c>: the <c>standard_logei</c>/<c>logei</c> scalars, LogEI's batched
///     <c>eval_acqf_no_grad</c>, and the hand-derived gradient vs Optuna's autograd
///     <c>eval_acqf_with_grad</c>. LogEI rides <c>erfc</c>/<c>erfcx</c> → the erf/ucrt tolerance class,
///     so these are magnitude-relative tolerance compares, not bit-for-bit.
///
///     <para>Optuna's autograd produces <c>nan</c> gradients at pathological points where the
///     posterior collapses (<c>z ≲ −35</c>, acqf ≈ −10³) — points the sampler's roulette never
///     selects; OptunaSharp's analytic gradient stays finite there, so the gradient parity is asserted
///     only where Optuna is finite.</para>
/// </summary>
[TestClass]
public class GpAcqfParityTests : OptunaInteropTestBase
{
    private const string ScalarHelper =
        "import torch\n" +
        "import optuna._gp.acqf as _acqf\n" +
        "def _std_logei(z):\n" +
        "    return float(_acqf.standard_logei(torch.tensor([z], dtype=torch.float64))[0])\n" +
        "def _logei(mean, var, f0):\n" +
        "    return float(_acqf.logei(torch.tensor([mean], dtype=torch.float64), torch.tensor([var], dtype=torch.float64), f0)[0])\n";

    private const string AcqfHelper =
        "import numpy as np, torch\n" +
        "import optuna._gp.gp as _gp\n" +
        "import optuna._gp.acqf as _acqf\n" +
        "import optuna._gp.search_space as _ss\n" +
        "from optuna.distributions import FloatDistribution\n" +
        "def _gp_logei(X, Y, isc, isl, ks, nv, threshold, xs):\n" +
        "    gpr = _gp.GPRegressor(is_categorical=torch.from_numpy(isc), X_train=torch.from_numpy(X),\n" +
        "        y_train=torch.from_numpy(Y), inverse_squared_lengthscales=torch.from_numpy(isl),\n" +
        "        kernel_scale=torch.tensor(ks, dtype=torch.float64), noise_var=torch.tensor(nv, dtype=torch.float64))\n" +
        "    gpr._cache_matrix()\n" +
        "    dim = X.shape[1]\n" +
        "    S = _ss.SearchSpace({f'p{i}': FloatDistribution(0.0, 1.0) for i in range(dim)})\n" +
        "    a = _acqf.LogEI(gpr=gpr, search_space=S, threshold=threshold)\n" +
        "    ng = a.eval_acqf_no_grad(xs).tolist()\n" +
        "    grads = []\n" +
        "    for x in xs:\n" +
        "        _, g = a.eval_acqf_with_grad(x)\n" +
        "        grads.append([float(v) for v in g])\n" +
        "    return ng, grads\n";

    [TestMethod]
    [TestCategory("RequiresPython")]
    public void StandardLogEiAndLogEi_MatchesOptuna_Sweep()
    {
        PythonSession.EnsureOrInconclusive();
        PyExec(ScalarHelper);
        var rng = new Random(808);
        for (int trial = 0; trial < 40; trial++)
        {
            double z = rng.NextDouble() * 70 - 60;  // in [-60, 10], spanning both branches
            double cs = Acqf.StandardLogEi(z);
            PyExec($"d = abs(_std_logei({Fmt(z)}) - {Fmt(cs)}) / (1.0 + abs({Fmt(cs)}))\n");
            PyFloat("d").Should().BeLessThan(1e-9, $"standard_logei z={z}");

            double mean = rng.NextDouble() * 4 - 2;
            double var = 0.01 + rng.NextDouble() * 2;
            double f0 = rng.NextDouble() * 4 - 2;
            double lc = Acqf.LogEi(mean, var, f0);
            PyExec($"dl = abs(_logei({Fmt(mean)}, {Fmt(var)}, {Fmt(f0)}) - {Fmt(lc)}) / (1.0 + abs({Fmt(lc)}))\n");
            PyFloat("dl").Should().BeLessThan(1e-9, $"logei mean={mean} var={var} f0={f0}");
        }
    }

    [TestMethod]
    [TestCategory("RequiresPython")]
    public void LogEI_EvalAndGrad_MatchesOptuna_Sweep()
    {
        PythonSession.EnsureOrInconclusive();
        PyExec(AcqfHelper);
        var rng = new Random(12321);
        for (int trial = 0; trial < 15; trial++)
        {
            int nParams = 1 + rng.Next(2);   // 1..2
            int nPoints = nParams + 4 + rng.Next(5);
            int nQuery = 3 + rng.Next(4);
            var isCat = new bool[nParams];   // continuous (gradients are only over continuous dims)

            var x = new double[nPoints, nParams];
            for (int i = 0; i < nPoints; i++)
                for (int dd = 0; dd < nParams; dd++)
                    x[i, dd] = rng.NextDouble();
            var y = new double[nPoints];
            for (int i = 0; i < nPoints; i++) y[i] = rng.NextDouble() * 2 - 1;
            var isl = new double[nParams];
            for (int dd = 0; dd < nParams; dd++) isl[dd] = 0.5 + rng.NextDouble() * 8;
            double ks = 0.3 + rng.NextDouble() * 2;
            double nv = 1e-4 + rng.NextDouble() * 0.2;
            double threshold = 0.0;
            for (int i = 0; i < nPoints; i++) threshold = Math.Max(threshold, y[i]);
            var xs = new double[nQuery, nParams];
            for (int i = 0; i < nQuery; i++)
                for (int dd = 0; dd < nParams; dd++)
                    xs[i, dd] = rng.NextDouble();

            var gpr = new GPRegressor(isCat, x, y, isl, ks, nv);
            gpr.CacheMatrix();
            var acqf = new LogEI(gpr, DummySpace(nParams), threshold);
            double[] ng = acqf.EvalAcqfNoGrad(xs);
            var grads = new double[nQuery][];
            for (int i = 0; i < nQuery; i++)
            {
                var row = new double[nParams];
                for (int dd = 0; dd < nParams; dd++) row[dd] = xs[i, dd];
                grads[i] = acqf.EvalAcqfWithGrad(row).Grad;
            }

            PyExec(
                $"X = np.array({Lit(x)}, dtype=float)\nY = np.array({Lit(y)}, dtype=float)\n" +
                $"isc = np.array({Lit(isCat)})\nisl = np.array({Lit(isl)}, dtype=float)\n" +
                $"ks = {Fmt(ks)}\nnv = {Fmt(nv)}\nthr = {Fmt(threshold)}\n" +
                $"xs = np.array({Lit(xs)}, dtype=float)\n" +
                $"ng_cs = np.array({Lit(ng)})\ngr_cs = {Lit2(grads)}\n" +
                "ng_py, gr_py = _gp_logei(X, Y, isc, isl, ks, nv, thr, xs)\n" +
                "ng_py = np.array(ng_py)\n" +
                "d_eval = float(np.max(np.abs(ng_py - ng_cs) / (1.0 + np.abs(ng_py))))\n" +
                "d_grad = 0.0\n" +
                "for i in range(len(gr_py)):\n" +
                "    for j in range(len(gr_py[i])):\n" +
                "        gp_ = gr_py[i][j]\n" +
                "        if np.isfinite(gp_):\n" +
                "            d_grad = max(d_grad, abs(gp_ - gr_cs[i][j]) / (1.0 + abs(gp_)))\n");
            PyFloat("d_eval").Should().BeLessThan(1e-8, $"eval trial={trial}");
            PyFloat("d_grad").Should().BeLessThan(1e-6, $"grad trial={trial}");
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

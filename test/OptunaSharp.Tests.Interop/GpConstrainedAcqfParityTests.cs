using System;
using System.Globalization;
using System.Text;
using AwesomeAssertions;
using OptunaSharp.Distributions;
using OptunaSharp.Gp;

namespace OptunaSharp.Tests.Interop;

/// <summary>
///     Live parity of the constrained single-objective GP acquisition functions (PORTING_PLAN Phase 3.2
///     part 3e-2) against <c>optuna/_gp/acqf.py</c>: <see cref="LogPI" /> (log feasible probability) and
///     <see cref="LogCEI" /> (= LogEI + Σ LogPI). Both GPs are constructed from <b>fixed</b> kernel
///     hyperparameters (no fit) so the comparison isolates the acqf math from the fit's ULP drift. LogPI
///     rides <c>log_ndtr</c> and LogEI rides <c>erfc</c>/<c>erfcx</c> → the erf/ucrt tolerance class, so
///     these are magnitude-relative tolerance compares, not bit-for-bit.
///
///     <para>Optuna's autograd can produce <c>nan</c> gradients at pathological points where the
///     posterior collapses; those are excluded (finite-only), matching the LogEI parity policy.</para>
/// </summary>
[TestClass]
public class GpConstrainedAcqfParityTests : OptunaInteropTestBase
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
        "def _space(dim):\n" +
        "    return _ss.SearchSpace({f'p{i}': FloatDistribution(0.0, 1.0) for i in range(dim)})\n" +
        "def _eval_grad(a, xs):\n" +
        "    ng = a.eval_acqf_no_grad(xs).tolist()\n" +
        "    grads = []\n" +
        "    for x in xs:\n" +
        "        _, g = a.eval_acqf_with_grad(x)\n" +
        "        grads.append([float(v) for v in g])\n" +
        "    return ng, grads\n" +
        "def _logpi(X, Y, isc, isl, ks, nv, threshold, xs):\n" +
        "    a = _acqf.LogPI(gpr=_build(X, Y, isc, isl, ks, nv), search_space=_space(X.shape[1]), threshold=threshold)\n" +
        "    return _eval_grad(a, xs)\n" +
        "def _logcei(X, Yo, isc, islo, kso, nvo, thr, Yc, islc, ksc, nvc, cthr, xs):\n" +
        "    g = _build(X, Yo, isc, islo, kso, nvo)\n" +
        "    cg = _build(X, Yc, isc, islc, ksc, nvc)\n" +
        "    a = _acqf.LogCEI(gpr=g, search_space=_space(X.shape[1]), threshold=thr,\n" +
        "        constraints_gpr_list=[cg], constraints_threshold_list=[cthr])\n" +
        "    return _eval_grad(a, xs)\n";

    [TestMethod]
    [TestCategory("RequiresPython")]
    public void LogPI_EvalAndGrad_MatchesOptuna_Sweep()
    {
        PythonSession.EnsureOrInconclusive();
        PyExec(Helper);
        var rng = new Random(51512);
        for (int trial = 0; trial < 15; trial++)
        {
            var s = RandomScenario(rng);
            var gpr = new GPRegressor(s.IsCat, s.X, s.Y, s.Isl, s.Ks, s.Nv);
            gpr.CacheMatrix();
            var acqf = new LogPI(gpr, DummySpace(s.NParams), s.Threshold);
            (double[] ng, double[][] grads) = EvalGrad(acqf, s.Xs, s.NParams);

            PyExec(
                ScenarioLits(s) +
                $"ng_cs = np.array({Lit(ng)})\ngr_cs = {Lit2(grads)}\n" +
                "ng_py, gr_py = _logpi(X, Y, isc, isl, ks, nv, thr, xs)\n" +
                MaxDiffSnippet);
            PyFloat("d_eval").Should().BeLessThan(1e-8, $"LogPI eval trial={trial}");
            PyFloat("d_grad").Should().BeLessThan(1e-6, $"LogPI grad trial={trial}");
        }
    }

    [TestMethod]
    [TestCategory("RequiresPython")]
    public void LogCEI_EvalAndGrad_MatchesOptuna_Sweep()
    {
        PythonSession.EnsureOrInconclusive();
        PyExec(Helper);
        var rng = new Random(97531);
        for (int trial = 0; trial < 15; trial++)
        {
            var s = RandomScenario(rng);
            // A second (constraint) GP over the same inputs, with its own Y / hyperparameters.
            var yc = new double[s.NPoints];
            for (int i = 0; i < s.NPoints; i++) yc[i] = rng.NextDouble() * 2 - 1;
            var islc = new double[s.NParams];
            for (int d = 0; d < s.NParams; d++) islc[d] = 0.5 + rng.NextDouble() * 8;
            double ksc = 0.3 + rng.NextDouble() * 2;
            double nvc = 1e-4 + rng.NextDouble() * 0.2;
            double cthr = rng.NextDouble() * 2 - 1;

            var gpr = new GPRegressor(s.IsCat, s.X, s.Y, s.Isl, s.Ks, s.Nv);
            gpr.CacheMatrix();
            var cgpr = new GPRegressor(s.IsCat, s.X, yc, islc, ksc, nvc);
            cgpr.CacheMatrix();
            var acqf = new LogCEI(gpr, DummySpace(s.NParams), s.Threshold, new[] { cgpr }, new[] { cthr });
            (double[] ng, double[][] grads) = EvalGrad(acqf, s.Xs, s.NParams);

            PyExec(
                ScenarioLits(s) +
                $"Yc = np.array({Lit(yc)}, dtype=float)\nislc = np.array({Lit(islc)}, dtype=float)\n" +
                $"ksc = {Fmt(ksc)}\nnvc = {Fmt(nvc)}\ncthr = {Fmt(cthr)}\n" +
                $"ng_cs = np.array({Lit(ng)})\ngr_cs = {Lit2(grads)}\n" +
                "ng_py, gr_py = _logcei(X, Y, isc, isl, ks, nv, thr, Yc, islc, ksc, nvc, cthr, xs)\n" +
                MaxDiffSnippet);
            PyFloat("d_eval").Should().BeLessThan(1e-8, $"LogCEI eval trial={trial}");
            PyFloat("d_grad").Should().BeLessThan(1e-6, $"LogCEI grad trial={trial}");
        }
    }

    private const string MaxDiffSnippet =
        "ng_py = np.array(ng_py)\n" +
        "d_eval = float(np.max(np.abs(ng_py - ng_cs) / (1.0 + np.abs(ng_py))))\n" +
        "d_grad = 0.0\n" +
        "for i in range(len(gr_py)):\n" +
        "    for j in range(len(gr_py[i])):\n" +
        "        gp_ = gr_py[i][j]\n" +
        "        if np.isfinite(gp_):\n" +
        "            d_grad = max(d_grad, abs(gp_ - gr_cs[i][j]) / (1.0 + abs(gp_)))\n";

    private readonly struct Scenario
    {
        public int NParams { get; init; }
        public int NPoints { get; init; }
        public bool[] IsCat { get; init; }
        public double[,] X { get; init; }
        public double[] Y { get; init; }
        public double[] Isl { get; init; }
        public double Ks { get; init; }
        public double Nv { get; init; }
        public double Threshold { get; init; }
        public double[,] Xs { get; init; }
    }

    private static Scenario RandomScenario(Random rng)
    {
        int nParams = 1 + rng.Next(2);            // 1..2 (gradients are over continuous dims)
        int nPoints = nParams + 4 + rng.Next(5);
        int nQuery = 3 + rng.Next(4);
        var x = new double[nPoints, nParams];
        for (int i = 0; i < nPoints; i++)
            for (int d = 0; d < nParams; d++)
                x[i, d] = rng.NextDouble();
        var y = new double[nPoints];
        for (int i = 0; i < nPoints; i++) y[i] = rng.NextDouble() * 2 - 1;
        var isl = new double[nParams];
        for (int d = 0; d < nParams; d++) isl[d] = 0.5 + rng.NextDouble() * 8;
        var xs = new double[nQuery, nParams];
        for (int i = 0; i < nQuery; i++)
            for (int d = 0; d < nParams; d++)
                xs[i, d] = rng.NextDouble();
        return new Scenario
        {
            NParams = nParams, NPoints = nPoints, IsCat = new bool[nParams], X = x, Y = y, Isl = isl,
            Ks = 0.3 + rng.NextDouble() * 2, Nv = 1e-4 + rng.NextDouble() * 0.2,
            Threshold = rng.NextDouble() * 2 - 1, Xs = xs,
        };
    }

    private static (double[] Ng, double[][] Grads) EvalGrad(IAcquisitionFunc acqf, double[,] xs, int nParams)
    {
        double[] ng = acqf.EvalAcqfNoGrad(xs);
        var grads = new double[xs.GetLength(0)][];
        for (int i = 0; i < xs.GetLength(0); i++)
        {
            var row = new double[nParams];
            for (int d = 0; d < nParams; d++) row[d] = xs[i, d];
            grads[i] = acqf.EvalAcqfWithGrad(row).Grad;
        }
        return (ng, grads);
    }

    private static string ScenarioLits(Scenario s) =>
        $"X = np.array({Lit(s.X)}, dtype=float)\nY = np.array({Lit(s.Y)}, dtype=float)\n" +
        $"isc = np.array({Lit(s.IsCat)})\nisl = np.array({Lit(s.Isl)}, dtype=float)\n" +
        $"ks = {Fmt(s.Ks)}\nnv = {Fmt(s.Nv)}\nthr = {Fmt(s.Threshold)}\n" +
        $"xs = np.array({Lit(s.Xs)}, dtype=float)\n";

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

using System;
using System.Globalization;
using System.Text;
using AwesomeAssertions;
using OptunaSharp.Distributions;
using OptunaSharp.Gp;

namespace OptunaSharp.Tests.Interop;

/// <summary>
///     Live parity of the <b>q-batch</b> GP acquisition functions (PORTING_PLAN Phase 3.2 part 3e-6)
///     against Optuna — the running-trial surface built on <see cref="ConditionalGPRegressor" />
///     (<c>gp.py: ConditionalGPRegressor.sample_joint_posterior</c>): <c>qLogEI</c> (this file grows to
///     cover <c>qLogPI</c>/<c>qLogCEI</c>/<c>qLogEHVI</c> as they land). Both sides fit <b>fixed</b>
///     kernel hyperparameters (no fit) and share the QMC seed, so the comparison isolates the
///     conditional-posterior conditioning + the <c>amax</c>/<c>logsumexp</c> reduction (and its
///     hand-derived gradient — the hardest gradient in the GP stack) from the fit's ULP drift.
///
///     <para>Tolerance (not bit-for-bit): the conditional posterior rides the managed triangular solve
///     and the QMC samples match torch ~1e-15 — the documented GP boundary. Optuna's autograd can emit
///     <c>nan</c>/non-finite gradients at the <c>amax</c> kinks (where the per-sample max flips between
///     the query and a fantasy column); those are excluded (finite-only).</para>
/// </summary>
[TestClass]
public class GpQBatchAcqfParityTests : OptunaInteropTestBase
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
        "def _grads(a, xs):\n" +
        "    out = []\n" +
        "    for x in xs:\n" +
        "        _, g = a.eval_acqf_with_grad(x)\n" +
        "        out.append([float(v) for v in g])\n" +
        "    return out\n" +
        "def _qlogei(X, isc, Y, isl, ks, nv, thr, n_qmc, seed, Xr, xs):\n" +
        "    a = _acqf.qLogEI(_build(X, Y, isc, isl, ks, nv), _space(X.shape[1]), thr, n_qmc, seed, Xr)\n" +
        "    return a.eval_acqf_no_grad(xs).tolist(), _grads(a, xs)\n" +
        "def _qlogpi(X, isc, Y, isl, ks, nv, thr, n_qmc, seed, Xr, xs):\n" +
        "    a = _acqf.qLogPI(_build(X, Y, isc, isl, ks, nv), _space(X.shape[1]), thr, n_qmc, seed, Xr)\n" +
        "    return a.eval_acqf_no_grad(xs).tolist(), _grads(a, xs)\n" +
        "def _qlogcei(X, isc, Yo, islo, kso, nvo, thr, Yc, islc, ksc, nvc, cthrs, n_qmc, seed, Xr, xs):\n" +
        "    go = _build(X, Yo, isc, islo, kso, nvo)\n" +
        "    gc = [_build(X, np.array(Yc[i]), isc, np.array(islc[i]), ksc[i], nvc[i]) for i in range(len(Yc))]\n" +
        "    a = _acqf.qLogCEI(go, _space(X.shape[1]), thr, n_qmc, seed, gc, list(cthrs), Xr)\n" +
        "    return a.eval_acqf_no_grad(xs).tolist(), _grads(a, xs)\n" +
        "def _qlogehvi(X, isc, Ytrain, Ys, isls, kss, nvs, n_qmc, seed, Xr, xs):\n" +
        "    gprs = [_build(X, np.array(Ys[i]), isc, np.array(isls[i]), kss[i], nvs[i]) for i in range(len(Ys))]\n" +
        "    a = _acqf.qLogEHVI(gprs, _space(X.shape[1]), torch.from_numpy(Ytrain), n_qmc, seed, Xr)\n" +
        "    return a.eval_acqf_no_grad(xs).tolist(), _grads(a, xs)\n";

    // Compare grads element-wise, excluding entries where Optuna's autograd is non-finite.
    private const string CompareGrad =
        "d_grad = 0.0\n" +
        "for i in range(len(gr_py)):\n" +
        "    for j in range(len(gr_py[i])):\n" +
        "        gp_ = gr_py[i][j]\n" +
        "        if np.isfinite(gp_):\n" +
        "            d_grad = max(d_grad, abs(gp_ - gr_cs[i][j]) / (1.0 + abs(gp_)))\n";

    [TestMethod]
    [TestCategory("RequiresPython")]
    public void QLogEI_EvalAndGrad_MatchesOptuna_Sweep()
    {
        PythonSession.EnsureOrInconclusive();
        PyExec(Helper);
        var rng = new Random(556677);
        for (int trial = 0; trial < 12; trial++)
        {
            int dim = 1 + rng.Next(2);
            int nPoints = dim + 6 + rng.Next(3);
            int nRunning = 1 + rng.Next(3);
            int nQuery = 5;
            int nQmc = 32;
            int qmcSeed = rng.Next(1, 1_000_000);
            var isCat = new bool[dim];

            double[,] x = RandMatrix(rng, nPoints, dim);
            var y = new double[nPoints];
            double ymin = double.MaxValue, ymax = double.MinValue;
            for (int i = 0; i < nPoints; i++) { y[i] = rng.NextDouble() * 2 - 1; ymin = Math.Min(ymin, y[i]); ymax = Math.Max(ymax, y[i]); }
            var isl = new double[dim];
            for (int d = 0; d < dim; d++) isl[d] = 0.5 + rng.NextDouble() * 8;
            double ks = 0.3 + rng.NextDouble() * 2;
            double nv = 1e-4 + rng.NextDouble() * 0.2;
            // Sweep the threshold across (and a bit past) the data range so gradients are non-trivial.
            double thr = ymin - 0.3 + rng.NextDouble() * (ymax - ymin + 0.6);
            double[,] xr = RandMatrix(rng, nRunning, dim);
            double[,] xs = RandMatrix(rng, nQuery, dim);

            var gpr = new GPRegressor(isCat, x, y, isl, ks, nv);
            gpr.CacheMatrix();
            var acqf = new QLogEI(gpr, DummySpace(dim), thr, nQmc, qmcSeed, xr);
            double[] ng = acqf.EvalAcqfNoGrad(xs);
            double[][] grads = RowGrads(acqf, xs);

            PyExec(
                $"X = np.array({Lit(x)}, dtype=float)\nisc = np.array({Lit(isCat)})\n" +
                $"Y = np.array({Lit(y)}, dtype=float)\nisl = np.array({Lit(isl)}, dtype=float)\n" +
                $"Xr = np.array({Lit(xr)}, dtype=float)\nxs = np.array({Lit(xs)}, dtype=float)\n" +
                $"ng_cs = np.array({Lit(ng)})\ngr_cs = {Lit2(grads)}\n" +
                $"ng_py, gr_py = _qlogei(X, isc, Y, isl, {Fmt(ks)}, {Fmt(nv)}, {Fmt(thr)}, {nQmc}, {qmcSeed}, Xr, xs)\n" +
                "ng_py = np.array(ng_py)\n" +
                "d_eval = float(np.max(np.abs(ng_py - ng_cs) / (1.0 + np.abs(ng_py))))\n" +
                CompareGrad);
            PyFloat("d_eval").Should().BeLessThan(1e-7, $"qLogEI eval trial={trial} dim={dim} nRun={nRunning}");
            PyFloat("d_grad").Should().BeLessThan(1e-5, $"qLogEI grad trial={trial} dim={dim} nRun={nRunning}");
        }
    }

    [TestMethod]
    [TestCategory("RequiresPython")]
    public void QLogPI_EvalAndGrad_MatchesOptuna_Sweep()
    {
        PythonSession.EnsureOrInconclusive();
        PyExec(Helper);
        var rng = new Random(778899);
        for (int trial = 0; trial < 12; trial++)
        {
            int dim = 1 + rng.Next(2);
            int nPoints = dim + 6 + rng.Next(3);
            int nRunning = 1 + rng.Next(3);
            int nQuery = 5;
            int nQmc = 32;
            int qmcSeed = rng.Next(1, 1_000_000);
            var isCat = new bool[dim];

            double[,] x = RandMatrix(rng, nPoints, dim);
            var y = new double[nPoints];
            double ymin = double.MaxValue, ymax = double.MinValue;
            for (int i = 0; i < nPoints; i++) { y[i] = rng.NextDouble() * 2 - 1; ymin = Math.Min(ymin, y[i]); ymax = Math.Max(ymax, y[i]); }
            var isl = new double[dim];
            for (int d = 0; d < dim; d++) isl[d] = 0.5 + rng.NextDouble() * 8;
            double ks = 0.3 + rng.NextDouble() * 2;
            double nv = 1e-4 + rng.NextDouble() * 0.2;
            double thr = ymin - 0.3 + rng.NextDouble() * (ymax - ymin + 0.6);
            double[,] xr = RandMatrix(rng, nRunning, dim);
            double[,] xs = RandMatrix(rng, nQuery, dim);

            var gpr = new GPRegressor(isCat, x, y, isl, ks, nv);
            gpr.CacheMatrix();
            var acqf = new QLogPI(gpr, DummySpace(dim), thr, nQmc, qmcSeed, xr);
            double[] ng = acqf.EvalAcqfNoGrad(xs);
            double[][] grads = RowGrads(acqf, xs);

            PyExec(
                $"X = np.array({Lit(x)}, dtype=float)\nisc = np.array({Lit(isCat)})\n" +
                $"Y = np.array({Lit(y)}, dtype=float)\nisl = np.array({Lit(isl)}, dtype=float)\n" +
                $"Xr = np.array({Lit(xr)}, dtype=float)\nxs = np.array({Lit(xs)}, dtype=float)\n" +
                $"ng_cs = np.array({Lit(ng)})\ngr_cs = {Lit2(grads)}\n" +
                $"ng_py, gr_py = _qlogpi(X, isc, Y, isl, {Fmt(ks)}, {Fmt(nv)}, {Fmt(thr)}, {nQmc}, {qmcSeed}, Xr, xs)\n" +
                "ng_py = np.array(ng_py)\n" +
                "d_eval = float(np.max(np.abs(ng_py - ng_cs) / (1.0 + np.abs(ng_py))))\n" +
                CompareGrad);
            PyFloat("d_eval").Should().BeLessThan(1e-7, $"qLogPI eval trial={trial} dim={dim} nRun={nRunning}");
            PyFloat("d_grad").Should().BeLessThan(1e-5, $"qLogPI grad trial={trial} dim={dim} nRun={nRunning}");
        }
    }

    [TestMethod]
    [TestCategory("RequiresPython")]
    public void QLogCEI_EvalAndGrad_MatchesOptuna_Sweep()
    {
        PythonSession.EnsureOrInconclusive();
        PyExec(Helper);
        var rng = new Random(224466);
        for (int trial = 0; trial < 12; trial++)
        {
            int dim = 1 + rng.Next(2);
            int nCons = 1 + rng.Next(2);
            int nPoints = dim + 6 + rng.Next(3);
            int nRunning = 1 + rng.Next(3);
            int nQuery = 5;
            int nQmc = 32;
            int qmcSeed = rng.Next(1, 1_000_000);
            var isCat = new bool[dim];

            double[,] x = RandMatrix(rng, nPoints, dim);
            (double[] yo, double[] islo, double kso, double nvo, double ymin, double ymax) = RandGp(rng, nPoints, dim);
            double thr = ymin - 0.3 + rng.NextDouble() * (ymax - ymin + 0.6);

            var yc = new double[nCons][];
            var islc = new double[nCons][];
            var ksc = new double[nCons];
            var nvc = new double[nCons];
            var cthrs = new double[nCons];
            var consGprs = new GPRegressor[nCons];
            for (int c = 0; c < nCons; c++)
            {
                (double[] y, double[] isl, double ks, double nv, double cmin, double cmax) = RandGp(rng, nPoints, dim);
                yc[c] = y; islc[c] = isl; ksc[c] = ks; nvc[c] = nv;
                cthrs[c] = cmin - 0.3 + rng.NextDouble() * (cmax - cmin + 0.6);
                var g = new GPRegressor(isCat, x, y, isl, ks, nv);
                g.CacheMatrix();
                consGprs[c] = g;
            }
            double[,] xr = RandMatrix(rng, nRunning, dim);
            double[,] xs = RandMatrix(rng, nQuery, dim);

            var gObj = new GPRegressor(isCat, x, yo, islo, kso, nvo);
            gObj.CacheMatrix();
            var acqf = new QLogCEI(gObj, DummySpace(dim), thr, nQmc, qmcSeed, consGprs, cthrs, xr);
            double[] ng = acqf.EvalAcqfNoGrad(xs);
            double[][] grads = RowGrads(acqf, xs);

            PyExec(
                $"X = np.array({Lit(x)}, dtype=float)\nisc = np.array({Lit(isCat)})\n" +
                $"Yo = np.array({Lit(yo)}, dtype=float)\nislo = np.array({Lit(islo)}, dtype=float)\n" +
                $"Yc = {Lit2(yc)}\nislc = {Lit2(islc)}\nksc = {Lit(ksc)}\nnvc = {Lit(nvc)}\ncthrs = {Lit(cthrs)}\n" +
                $"Xr = np.array({Lit(xr)}, dtype=float)\nxs = np.array({Lit(xs)}, dtype=float)\n" +
                $"ng_cs = np.array({Lit(ng)})\ngr_cs = {Lit2(grads)}\n" +
                $"ng_py, gr_py = _qlogcei(X, isc, Yo, islo, {Fmt(kso)}, {Fmt(nvo)}, {Fmt(thr)}, Yc, islc, ksc, nvc, cthrs, {nQmc}, {qmcSeed}, Xr, xs)\n" +
                "ng_py = np.array(ng_py)\n" +
                "d_eval = float(np.max(np.abs(ng_py - ng_cs) / (1.0 + np.abs(ng_py))))\n" +
                CompareGrad);
            PyFloat("d_eval").Should().BeLessThan(1e-7, $"qLogCEI eval trial={trial} dim={dim} nCons={nCons} nRun={nRunning}");
            PyFloat("d_grad").Should().BeLessThan(1e-5, $"qLogCEI grad trial={trial} dim={dim} nCons={nCons} nRun={nRunning}");
        }
    }

    [TestMethod]
    [TestCategory("RequiresPython")]
    public void QLogEHVI_EvalAndGrad_MatchesOptuna_Sweep()
    {
        PythonSession.EnsureOrInconclusive();
        PyExec(Helper);
        var rng = new Random(135790);
        for (int trial = 0; trial < 12; trial++)
        {
            int dim = 1 + rng.Next(2);
            int nObj = 2 + rng.Next(2);
            int nPoints = dim + 6 + rng.Next(3);
            int nRunning = 1 + rng.Next(3);
            int nQuery = 5;
            int nQmc = 32;
            int qmcSeed = rng.Next(1, 1_000_000);
            var isCat = new bool[dim];

            double[,] x = RandMatrix(rng, nPoints, dim);
            var ys = new double[nObj][];
            var isls = new double[nObj][];
            var kss = new double[nObj];
            var nvs = new double[nObj];
            var yTrain = new double[nPoints, nObj];
            var gprs = new GPRegressor[nObj];
            for (int o = 0; o < nObj; o++)
            {
                (double[] y, double[] isl, double ks, double nv, _, _) = RandGp(rng, nPoints, dim);
                for (int i = 0; i < nPoints; i++) yTrain[i, o] = y[i];
                ys[o] = y; isls[o] = isl; kss[o] = ks; nvs[o] = nv;
                var gpr = new GPRegressor(isCat, x, y, isl, ks, nv);
                gpr.CacheMatrix();
                gprs[o] = gpr;
            }
            double[,] xr = RandMatrix(rng, nRunning, dim);
            double[,] xs = RandMatrix(rng, nQuery, dim);

            var acqf = new QLogEHVI(gprs, DummySpace(dim), yTrain, nQmc, qmcSeed, xr);
            double[] ng = acqf.EvalAcqfNoGrad(xs);
            double[][] grads = RowGrads(acqf, xs);

            PyExec(
                $"X = np.array({Lit(x)}, dtype=float)\nisc = np.array({Lit(isCat)})\n" +
                $"Ytrain = np.array({Lit(yTrain)}, dtype=float)\n" +
                $"Ys = {Lit2(ys)}\nisls = {Lit2(isls)}\nkss = {Lit(kss)}\nnvs = {Lit(nvs)}\n" +
                $"Xr = np.array({Lit(xr)}, dtype=float)\nxs = np.array({Lit(xs)}, dtype=float)\n" +
                $"ng_cs = np.array({Lit(ng)})\ngr_cs = {Lit2(grads)}\n" +
                $"ng_py, gr_py = _qlogehvi(X, isc, Ytrain, Ys, isls, kss, nvs, {nQmc}, {qmcSeed}, Xr, xs)\n" +
                "ng_py = np.array(ng_py)\n" +
                "d_eval = float(np.max(np.abs(ng_py - ng_cs) / (1.0 + np.abs(ng_py))))\n" +
                CompareGrad);
            PyFloat("d_eval").Should().BeLessThan(1e-7, $"qLogEHVI eval trial={trial} nObj={nObj} nRun={nRunning}");
            PyFloat("d_grad").Should().BeLessThan(1e-5, $"qLogEHVI grad trial={trial} nObj={nObj} nRun={nRunning}");
        }
    }

    private static (double[] Y, double[] Isl, double Ks, double Nv, double Ymin, double Ymax) RandGp(Random rng, int nPoints, int dim)
    {
        var y = new double[nPoints];
        double ymin = double.MaxValue, ymax = double.MinValue;
        for (int i = 0; i < nPoints; i++) { y[i] = rng.NextDouble() * 2 - 1; ymin = Math.Min(ymin, y[i]); ymax = Math.Max(ymax, y[i]); }
        var isl = new double[dim];
        for (int d = 0; d < dim; d++) isl[d] = 0.5 + rng.NextDouble() * 8;
        return (y, isl, 0.3 + rng.NextDouble() * 2, 1e-4 + rng.NextDouble() * 0.2, ymin, ymax);
    }

    private static double[][] RowGrads(IAcquisitionFunc acqf, double[,] xs)
    {
        int n = xs.GetLength(0), dim = xs.GetLength(1);
        var grads = new double[n][];
        for (int i = 0; i < n; i++)
        {
            var row = new double[dim];
            for (int d = 0; d < dim; d++) row[d] = xs[i, d];
            grads[i] = acqf.EvalAcqfWithGrad(row).Grad;
        }
        return grads;
    }

    private static double[,] RandMatrix(Random rng, int rows, int cols)
    {
        var m = new double[rows, cols];
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                m[i, j] = rng.NextDouble();
        return m;
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

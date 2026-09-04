using System;
using System.Globalization;
using System.Text;
using AwesomeAssertions;
using OptunaSharp.Gp;

namespace OptunaSharp.Tests.Interop;

/// <summary>
///     Live parity of the GP fit objective (PORTING_PLAN Phase 3.2 — the §7 autodiff decision gate)
///     against Optuna's own <c>loss_func</c> from <c>_fit_kernel_params</c>, i.e. PyTorch
///     <c>autograd</c>. For arbitrary raw (log-transformed) kernel parameters, OptunaSharp's
///     hand-derived analytic <see cref="GPRegressor.LossAndGrad" /> reproduces the loss AND the full
///     gradient that torch computes by reverse-mode differentiation.
///
///     <para>This is the evidence that keeps the GP stack in Tier N (BCL + NumSharp only, no
///     TorchSharp): the closed-form Matérn-5/2 marginal-likelihood gradient, in pure managed code,
///     matches autograd to a machine-precision tolerance (~1e-11 over these sweeps). It is a
///     <b>tolerance</b> compare, not bit-for-bit — the analytic gradient is a different float-op
///     sequence than autograd's reverse pass, and the forward rides the managed
///     <c>SolveTriangular</c> — the documented GP boundary.</para>
/// </summary>
[TestClass]
public class GpFitParityTests : OptunaInteropTestBase
{
    private const double Tol = 1e-9;
    private const double MinNoise = 1e-6;

    private const string Helper =
        "import numpy as np, torch\n" +
        "import optuna._gp.gp as _gp\n" +
        "import optuna._gp.prior as _gp_prior\n" +
        "def _gp_loss_grad(raw, X, Y, isc, min_noise, deterministic):\n" +
        "    n = X.shape[1]\n" +
        "    gpr = _gp.GPRegressor(\n" +
        "        is_categorical=torch.from_numpy(isc), X_train=torch.from_numpy(X),\n" +
        "        y_train=torch.from_numpy(Y),\n" +
        "        inverse_squared_lengthscales=torch.ones(n, dtype=torch.float64),\n" +
        "        kernel_scale=torch.ones((), dtype=torch.float64),\n" +
        "        noise_var=torch.ones((), dtype=torch.float64))\n" +
        "    rpt = torch.from_numpy(raw).clone().requires_grad_(True)\n" +
        "    with torch.enable_grad():\n" +
        "        gpr.inverse_squared_lengthscales = torch.exp(rpt[:n])\n" +
        "        gpr.kernel_scale = torch.exp(rpt[n])\n" +
        "        gpr.noise_var = (torch.tensor(min_noise, dtype=torch.float64) if deterministic\n" +
        "                         else torch.exp(rpt[n+1]) + min_noise)\n" +
        "        loss = -gpr.marginal_log_likelihood() - _gp_prior.default_log_prior(gpr)\n" +
        "        loss.backward()\n" +
        "    return float(loss), rpt.grad.detach().numpy()\n";

    [TestMethod]
    [TestCategory("RequiresPython")]
    public void LossAndGrad_Numerical_MatchesAutograd_Sweep()
    {
        PythonSession.EnsureOrInconclusive();
        PyExec(Helper);
        var rng = new Random(2024);
        for (int trial = 0; trial < 25; trial++)
        {
            int nParams = 1 + rng.Next(3);       // 1..3
            int nPoints = nParams + 3 + rng.Next(6);
            var isCat = new bool[nParams];        // all numerical here
            AssertLossGradParity(rng, nPoints, nParams, isCat, deterministic: false, trial);
        }
    }

    [TestMethod]
    [TestCategory("RequiresPython")]
    public void LossAndGrad_Categorical_MatchesAutograd_Sweep()
    {
        PythonSession.EnsureOrInconclusive();
        PyExec(Helper);
        var rng = new Random(99);
        for (int trial = 0; trial < 20; trial++)
        {
            int nParams = 2 + rng.Next(2);        // 2..3
            int nPoints = nParams + 4 + rng.Next(6);
            var isCat = new bool[nParams];
            isCat[rng.Next(nParams)] = true;      // at least one categorical dim (Hamming)
            AssertLossGradParity(rng, nPoints, nParams, isCat, deterministic: false, trial);
        }
    }

    private const string FitHelper =
        "import numpy as np, torch\n" +
        "import optuna._gp.gp as _gp\n" +
        "import optuna._gp.prior as _gp_prior\n" +
        "def _gp_fit(X, Y, isc):\n" +
        "    gpr = _gp.fit_kernel_params(X=X, Y=Y, is_categorical=isc,\n" +
        "        log_prior=_gp_prior.default_log_prior, minimum_noise=1e-6,\n" +
        "        deterministic_objective=False, gpr_cache=None, gtol=1e-2)\n" +
        "    loss = float(-gpr.marginal_log_likelihood() - _gp_prior.default_log_prior(gpr))\n" +
        "    return (gpr.inverse_squared_lengthscales.detach().numpy().tolist(),\n" +
        "            float(gpr.kernel_scale), float(gpr.noise_var), loss)\n";

    [TestMethod]
    [TestCategory("RequiresPython")]
    public void FitKernelParams_MatchesOptuna_Sweep()
    {
        PythonSession.EnsureOrInconclusive();
        PyExec(FitHelper);
        var rng = new Random(31337);
        for (int trial = 0; trial < 20; trial++)
        {
            int nParams = 1 + rng.Next(3);
            int nPoints = nParams + 5 + rng.Next(6);
            var isCat = new bool[nParams];
            if (rng.NextDouble() < 0.4)
                isCat[rng.Next(nParams)] = true;

            var x = new double[nPoints, nParams];
            for (int i = 0; i < nPoints; i++)
                for (int dd = 0; dd < nParams; dd++)
                    x[i, dd] = isCat[dd] ? rng.Next(3) : rng.NextDouble();
            var y = new double[nPoints];
            for (int i = 0; i < nPoints; i++)
                y[i] = rng.NextDouble() * 3 - 1.5;

            GPRegressor gpr = GPRegressor.FitKernelParams(x, y, isCat, 1e-6, false, null, 1e-2);
            var fitted = new double[nParams + 2];
            for (int dd = 0; dd < nParams; dd++) fitted[dd] = gpr.InverseSquaredLengthscales[dd];
            fitted[nParams] = gpr.KernelScale;
            fitted[nParams + 1] = gpr.NoiseVar;
            double lossCs = -gpr.MarginalLogLikelihood() - Prior.DefaultLogPrior(gpr);

            PyExec(
                $"X = np.array({Lit(x)})\n" +
                $"Y = np.array({Lit(y)})\n" +
                $"isc = np.array({Lit(isCat)})\n" +
                $"fcs = np.array({Lit(fitted)})\n" +
                $"loss_cs = {lossCs.ToString("R", CultureInfo.InvariantCulture)}\n" +
                "isl, ks, nv, loss_py = _gp_fit(X, Y, isc)\n" +
                "fpy = np.array(list(isl) + [ks, nv])\n" +
                "d_rel = float(np.max(np.abs(fpy - fcs) / (1.0 + np.abs(fpy))))\n" +
                "d_loss = float(abs(loss_cs - loss_py))\n");

            // "Same optimum" is judged by the achieved objective, which is well-defined even where
            // the argmin sits in a flat region: with Optuna's loose gtol=1e-2 the projected gradient
            // is tiny at the stop, so ULP-level path differences let the two fits settle ~1e-3 apart
            // in the hyperparameters while the minimized loss agrees far more tightly.
            PyFloat("d_loss").Should().BeLessThan(1e-4, $"fit loss trial={trial} nP={nPoints} nD={nParams}");
            PyFloat("d_rel").Should().BeLessThan(2e-2, $"fit params trial={trial} nP={nPoints} nD={nParams}");
        }
    }

    [TestMethod]
    [TestCategory("RequiresPython")]
    public void LossAndGrad_Deterministic_MatchesAutograd_Sweep()
    {
        PythonSession.EnsureOrInconclusive();
        PyExec(Helper);
        var rng = new Random(7);
        for (int trial = 0; trial < 15; trial++)
        {
            int nParams = 1 + rng.Next(3);
            int nPoints = nParams + 3 + rng.Next(5);
            var isCat = new bool[nParams];
            AssertLossGradParity(rng, nPoints, nParams, isCat, deterministic: true, trial);
        }
    }

    private void AssertLossGradParity(Random rng, int nPoints, int nParams, bool[] isCat, bool deterministic, int trial)
    {
        var x = new double[nPoints, nParams];
        for (int i = 0; i < nPoints; i++)
            for (int d = 0; d < nParams; d++)
                x[i, d] = isCat[d] ? rng.Next(3) : rng.NextDouble();  // categories in {0,1,2}
        var y = new double[nPoints];
        for (int i = 0; i < nPoints; i++)
            y[i] = rng.NextDouble() * 2 - 1;

        // Raw params in the fit's operating regime (lengthscale/scale ~O(1), noise not too tiny
        // to keep the covariance well conditioned).
        var raw = new double[nParams + 2];
        for (int d = 0; d < nParams; d++)
            raw[d] = rng.NextDouble() * 3 - 0.5;
        raw[nParams] = rng.NextDouble() * 1.5 - 0.5;
        raw[nParams + 1] = rng.NextDouble() * -6 - 1;  // exp(raw)+min_noise ∈ ~[1e-3, 0.37]

        var gpr = new GPRegressor(isCat, x, y, Ones(nParams), 1.0, 1.0);
        (double lossCs, double[] gradCs) = gpr.LossAndGrad(raw, MinNoise, deterministic);

        PyExec(
            $"raw = np.array({Lit(raw)})\n" +
            $"X = np.array({Lit(x)})\n" +
            $"Y = np.array({Lit(y)})\n" +
            $"isc = np.array({Lit(isCat)})\n" +
            $"lcs = {lossCs.ToString("R", CultureInfo.InvariantCulture)}\n" +
            $"gcs = np.array({Lit(gradCs)})\n" +
            $"lpy, gpy = _gp_loss_grad(raw, X, Y, isc, {MinNoise.ToString("R", CultureInfo.InvariantCulture)}, {(deterministic ? "True" : "False")})\n" +
            // Magnitude-relative tolerance: the loss is a log-determinant whose absolute size grows
            // with the covariance conditioning (deterministic mode pins noise to 1e-6, cond ~1e4+),
            // so the managed SolveTriangular drift scales with |loss|. For the well-conditioned O(1)
            // cases this is ~absolute; it never exceeds ~1e-10 relative anywhere.
            "d_loss = abs(lpy - lcs) / (1.0 + abs(lpy))\n" +
            "d_grad = float(np.max(np.abs(gpy - gcs)) / (1.0 + np.max(np.abs(gpy))))\n");

        PyFloat("d_loss").Should().BeLessThan(Tol, $"loss trial={trial} nP={nPoints} nD={nParams}");
        PyFloat("d_grad").Should().BeLessThan(Tol, $"grad trial={trial} nP={nPoints} nD={nParams}");
    }

    private static double[] Ones(int n)
    {
        var a = new double[n];
        Array.Fill(a, 1.0);
        return a;
    }

    // ---- literals -----------------------------------------------------------------------------

    private static string Lit(double[] a)
    {
        var sb = new StringBuilder("[");
        for (int i = 0; i < a.Length; i++)
        {
            if (i > 0) sb.Append(", ");
            sb.Append(a[i].ToString("R", CultureInfo.InvariantCulture));
        }
        return sb.Append(']').ToString();
    }

    private static string Lit(bool[] a)
    {
        var sb = new StringBuilder("[");
        for (int i = 0; i < a.Length; i++)
        {
            if (i > 0) sb.Append(", ");
            sb.Append(a[i] ? "True" : "False");
        }
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

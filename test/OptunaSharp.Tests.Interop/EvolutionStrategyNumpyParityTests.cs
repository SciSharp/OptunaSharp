using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using AwesomeAssertions;
using OptunaSharp.Samplers.EvolutionStrategy;

namespace OptunaSharp.Tests.Interop;

/// <summary>
///     Live-numpy parity for the Natural-Evolution-Strategy math. OpenAI-ES / SNES / xNES have no Optuna
///     counterpart to differentially test against, so this gate instead pins OptunaSharp's shared
///     rank-based shaping and the SNES natural-gradient step against an <b>independent</b> reference written
///     directly in numpy (the plan's "optional bit-parity vs a pinned numpy reference"). The MT19937 draw
///     is byte-identical to numpy, and the arithmetic is plain float64, so the reference and the C#
///     implementation agree to a tight tolerance — the residual is only summation-order rounding, not an
///     algorithmic difference. Self-skips (Inconclusive) when no Python/numpy engine is discoverable.
/// </summary>
[TestClass]
public class EvolutionStrategyNumpyParityTests : OptunaInteropTestBase
{
    private const double Tol = 1e-9;

    /// <summary>Parses a Python <c>','.join(repr(float(v)) …)</c> payload back into full-precision doubles.</summary>
    private static double[] ParseCsv(string csv)
        => csv.Length == 0
            ? Array.Empty<double>()
            : csv.Split(',').Select(s => double.Parse(s, CultureInfo.InvariantCulture)).ToArray();

    [TestMethod]
    [TestCategory("RequiresPython")]
    public void CenteredRanks_MatchNumpyReference()
    {
        double[] scores = { 3.0, 1.0, 2.0, 0.0, 5.5, -1.0, 2.0, 9.0 };
        PySet("scores_json", "[" + string.Join(",", scores.Select(v => v.ToString("R", CultureInfo.InvariantCulture))) + "]");
        PyExec(
            "import numpy as np, json\n" +
            "x = np.array(json.loads(scores_json), dtype=np.float64)\n" +
            // Ascending competition ranks (stable), then rank/(n-1) − 0.5 in float64 (OptunaSharp's formula).
            "order = np.argsort(x, kind='stable')\n" +
            "ranks = np.empty(len(x), dtype=np.float64)\n" +
            "ranks[order] = np.arange(len(x), dtype=np.float64)\n" +
            "y = ranks / (len(x) - 1) - 0.5\n" +
            "cr = ','.join(repr(float(v)) for v in y)\n");

        double[] py = ParseCsv(PyStr("cr"));
        double[] cs = EsFitnessShaping.CenteredRanks(scores);

        cs.Length.Should().Be(py.Length);
        for (int i = 0; i < cs.Length; i++)
            cs[i].Should().BeApproximately(py[i], Tol, $"centered rank[{i}] matches numpy");
    }

    [TestMethod]
    [TestCategory("RequiresPython")]
    public void NesUtilities_MatchNumpyReference()
    {
        foreach (int lambda in new[] { 4, 7, 10, 16 })
        {
            PyExec(
                "import numpy as np\n" +
                $"lam = {lambda}\n" +
                // u_k = max(0, ln(λ/2+1) − ln k)/Σ − 1/λ for k = 1..λ (OptunaSharp's utility function).
                "k = np.arange(1, lam + 1, dtype=np.float64)\n" +
                "raw = np.maximum(0.0, np.log(lam / 2.0 + 1.0) - np.log(k))\n" +
                "u = raw / raw.sum() - 1.0 / lam\n" +
                "us = ','.join(repr(float(v)) for v in u)\n");

            double[] py = ParseCsv(PyStr("us"));
            double[] cs = EsFitnessShaping.NesUtilities(lambda);

            cs.Length.Should().Be(lambda);
            for (int i = 0; i < lambda; i++)
                cs[i].Should().BeApproximately(py[i], Tol, $"λ={lambda} utility[{i}] matches numpy");
        }
    }

    [TestMethod]
    [TestCategory("RequiresPython")]
    public void SnesUpdate_MatchesNumpyReference()
    {
        // A single Separable-NES generation on a sphere, computed independently in numpy, must reproduce the
        // C# optimizer's (mean, σ) update. numpy draws the perturbations (byte-identical to NumSharp), and
        // both sides run the same natural-gradient step from them.
        const int popsize = 8;
        const int d = 4;
        double[] mu0 = { 0.5, 0.4, 0.6, 0.5 };
        double[] sigma0 = { 0.3, 0.25, 0.35, 0.3 };
        double etaMu = 1.0;
        double etaSigma = SnesOptimizer.DefaultEtaSigma(d);

        PySet("mu0_json", "[" + string.Join(",", mu0.Select(v => v.ToString("R", CultureInfo.InvariantCulture))) + "]");
        PySet("sigma0_json", "[" + string.Join(",", sigma0.Select(v => v.ToString("R", CultureInfo.InvariantCulture))) + "]");
        PyExec(
            "import numpy as np, json\n" +
            $"pop, d = {popsize}, {d}\n" +
            "mu = np.array(json.loads(mu0_json), dtype=np.float64)\n" +
            "sigma = np.array(json.loads(sigma0_json), dtype=np.float64)\n" +
            $"eta_mu, eta_sigma = {etaMu.ToString("R", CultureInfo.InvariantCulture)}, {etaSigma.ToString("R", CultureInfo.InvariantCulture)}\n" +
            "rng = np.random.RandomState(20260915)\n" +
            "s = rng.randn(pop, d)\n" +                              // perturbations (shared with C#)
            "z = mu[None, :] + sigma[None, :] * s\n" +
            "vals = (z * z).sum(axis=1)\n" +                         // sphere objective (minimize)
            // Ascending by value (best first); utilities aligned by rank.
            "order = np.argsort(vals, kind='stable')\n" +
            "lam = pop\n" +
            "kk = np.arange(1, lam + 1, dtype=np.float64)\n" +
            "raw = np.maximum(0.0, np.log(lam / 2.0 + 1.0) - np.log(kk))\n" +
            "u = raw / raw.sum() - 1.0 / lam\n" +
            "grad_mu = np.zeros(d); grad_sigma = np.zeros(d)\n" +
            "for r in range(lam):\n" +
            "    sr = s[order[r]]\n" +
            "    grad_mu += u[r] * sr\n" +
            "    grad_sigma += u[r] * (sr * sr - 1.0)\n" +
            "mu_new = mu + eta_mu * sigma * grad_mu\n" +
            "sigma_new = sigma * np.exp(0.5 * eta_sigma * grad_sigma)\n" +
            "s_json = json.dumps([[float(v) for v in row] for row in s])\n" +
            "mu_str = ','.join(repr(float(v)) for v in mu_new)\n" +
            "sigma_str = ','.join(repr(float(v)) for v in sigma_new)\n");

        // Recover the exact perturbations numpy drew, build the evaluated candidates, and drive the C#
        // optimizer one generation. Tell reconstructs s = (x − μ)/σ from the candidates (exact up to the
        // reconstruction rounding the tolerance absorbs).
        string sJson = PyStr("s_json");
        double[][] s = ParseMatrix(sJson, popsize, d);

        var opt = new SnesOptimizer(mu0, sigma0, etaMu, etaSigma, popsize, seed: 1);
        var solutions = new List<(double[] X, double Value)>(popsize);
        for (int i = 0; i < popsize; i++)
        {
            var x = new double[d];
            double val = 0;
            for (int j = 0; j < d; j++) { x[j] = mu0[j] + sigma0[j] * s[i][j]; val += x[j] * x[j]; }
            solutions.Add((x, val));
        }
        opt.Tell(solutions);

        var state = opt.ToState();
        double[] muPy = ParseCsv(PyStr("mu_str"));
        double[] sigmaPy = ParseCsv(PyStr("sigma_str"));

        for (int j = 0; j < d; j++)
        {
            state.Mean[j].Should().BeApproximately(muPy[j], Tol, $"SNES μ[{j}] matches numpy");
            state.Sigma[j].Should().BeApproximately(sigmaPy[j], Tol, $"SNES σ[{j}] matches numpy");
        }
    }

    /// <summary>Parses a Python <c>json.dumps</c> of a rows×cols float matrix into a jagged double array.</summary>
    private static double[][] ParseMatrix(string json, int rows, int cols)
    {
        // The payload is a JSON array of arrays of full-precision floats; System.Text.Json round-trips them.
        using var doc = System.Text.Json.JsonDocument.Parse(json);
        var root = doc.RootElement;
        var m = new double[rows][];
        for (int r = 0; r < rows; r++)
        {
            var row = new double[cols];
            var pyRow = root[r];
            for (int c = 0; c < cols; c++) row[c] = pyRow[c].GetDouble();
            m[r] = row;
        }
        return m;
    }
}

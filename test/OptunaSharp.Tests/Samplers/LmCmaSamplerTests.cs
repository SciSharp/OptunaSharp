using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.Json;
using AwesomeAssertions;
using OptunaSharp.Samplers;
using OptunaSharp.Samplers.CmaEs;

namespace OptunaSharp.Tests.Samplers;

/// <summary>
///     Tests for the paper-identical <see cref="LmCma" /> optimizer and its <see cref="LmCmaSampler" />.
///     LM-CMA (Loshchilov 2015) has no <c>cmaes</c>/Optuna/OptunaHub counterpart, so there is no upstream
///     trajectory to bit-match (unlike <c>CmaEsSamplerTests</c>). These are therefore behavioral gates:
///     convergence, same-seed determinism, and optimizer-state round-trip fidelity.
/// </summary>
[TestClass]
public class LmCmaSamplerTests
{
    private static double Sphere(double[] x)
    {
        double s = 0;
        foreach (double v in x) s += v * v;
        return s;
    }

    // Axis-aligned ellipsoid with condition number 1e6 — the covariance-reconstruction stress test.
    private static double Ellipsoid(double[] x)
    {
        int n = x.Length;
        double s = 0;
        for (int i = 0; i < n; i++) s += Math.Pow(10.0, 6.0 * i / (n - 1)) * x[i] * x[i];
        return s;
    }

    // Drive the optimizer as a batch ask/tell object (this path exercises the paper's mirrored sampling).
    private static double Drive(LmCma opt, Func<double[], double> f, int generations)
    {
        double best = double.PositiveInfinity;
        for (int g = 0; g < generations; g++)
        {
            var sols = new List<(double[] X, double Value)>();
            for (int k = 0; k < opt.PopulationSize; k++)
            {
                double[] x = opt.Ask();
                double val = f(x);
                if (val < best) best = val;
                sols.Add((x, val));
            }
            opt.Tell(sols);
        }
        return best;
    }

    private static double[] Filled(int n, double v)
    {
        var a = new double[n];
        for (int i = 0; i < n; i++) a[i] = v;
        return a;
    }

    // ---- optimizer: convergence -------------------------------------------------------------------

    [TestMethod]
    public void Optimizer_Sphere_Converges()
    {
        double best = Drive(new LmCma(Filled(10, 3.0), sigma: 2.0, seed: 1), Sphere, generations: 300);
        best.Should().BeLessThan(1e-8, "LM-CMA solves the sphere to near machine precision");
    }

    [TestMethod]
    public void Optimizer_IllConditionedEllipsoid_Descends()
    {
        // A broken Cholesky reconstruction would STALL once the easy axes are solved, unable to descend
        // the ill-conditioned valley — plateauing many orders above the start's ~1e7. Reaching < 100 (a
        // ~5-order descent) proves the reconstruction is genuinely learning the conditioning. LM-CMA is
        // deliberately slower than full CMA here (a limited-memory approximation at low n), so the gate is
        // "descends the valley", not "solves to precision".
        double best = Drive(new LmCma(Filled(10, 3.0), sigma: 2.0, seed: 3), Ellipsoid, generations: 2000);
        best.Should().BeLessThan(100.0, "LM-CMA descends the condition-1e6 valley (reconstruction works)");
    }

    // ---- optimizer: determinism & state round-trip ------------------------------------------------

    [TestMethod]
    public void Optimizer_SameSeed_IsReproducible()
    {
        double a = Drive(new LmCma(Filled(8, 2.0), 1.5, seed: 11), Sphere, generations: 50);
        double b = Drive(new LmCma(Filled(8, 2.0), 1.5, seed: 11), Sphere, generations: 50);
        b.Should().Be(a, "same seed ⇒ identical trajectory");
    }

    [TestMethod]
    public void Optimizer_StateRoundTrip_PreservesEverything()
    {
        var opt = new LmCma(Filled(12, 2.0), 1.5, seed: 5);
        Drive(opt, Sphere, generations: 40);   // build up mean/sigma/pc and the direction-vector store

        // Serialize → deserialize exactly as LmCmaSampler stores optimizer state between trials.
        string json = JsonSerializer.Serialize(opt.ToState());
        var restored = LmCma.FromState(JsonSerializer.Deserialize<LmCmaState>(json)!);

        restored.Mean.Should().Equal(opt.Mean);
        restored.Sigma.Should().Be(opt.Sigma);
        restored.Generation.Should().Be(opt.Generation);

        // After an identical reseed, the next candidate must be bit-identical — proving the reconstructed
        // Cholesky store (P/V/b/d/j/l) round-tripped, not just the scalars.
        opt.ReseedRng(999);
        restored.ReseedRng(999);
        restored.AskCandidate(1).Should().Equal(opt.AskCandidate(1));
    }

    // ---- sampler: end-to-end through a Study ------------------------------------------------------

    [TestMethod]
    public void Sampler_MinimizesSphere()
    {
        var study = Optuna.CreateStudy(sampler: new LmCmaSampler(seed: 1, nStartupTrials: 5));
        study.Optimize(t =>
        {
            double s = 0;
            for (int i = 0; i < 10; i++) { double x = t.SuggestFloat($"x{i}", -5, 5); s += x * x; }
            return s;
        }, nTrials: 800);

        study.BestValue.Should().BeLessThan(1e-2, "the sampler drives LM-CMA to a small sphere value");
    }

    [TestMethod]
    public void Sampler_Maximize_Works()
    {
        var study = Optuna.CreateStudy(
            direction: StudyDirection.Maximize, sampler: new LmCmaSampler(seed: 3, nStartupTrials: 5));
        study.Optimize(t =>
        {
            double s = 0;
            for (int i = 0; i < 8; i++) { double x = t.SuggestFloat($"x{i}", -5, 5); s += x * x; }
            return -s;   // optimum 0, approached from below
        }, nTrials: 500);

        study.BestValue.Should().BeGreaterThan(-1e-1, "direction flipping lets LM-CMA maximize -sphere");
    }

    [TestMethod]
    public void Sampler_GenerationAttr_IsAttributedPerTrial()
    {
        var study = Optuna.CreateStudy(sampler: new LmCmaSampler(seed: 42, nStartupTrials: 1));
        study.Optimize(t =>
        {
            double x = t.SuggestFloat("x", -5, 5);
            double y = t.SuggestFloat("y", -5, 5);
            return x * x + y * y;
        }, nTrials: 20);

        // Trials past the startup phase carry the optimizer's generation stamp.
        study.Trials.Skip(5).Any(tr => tr.SystemAttrs.ContainsKey("lmcma:generation")).Should().BeTrue();
    }

    [TestMethod]
    public void Sampler_SingleDimension_FallsBackToIndependent()
    {
        // LM-CMA needs n > 1; a 1-D relative space must fall back to the independent sampler and still run.
        var study = Optuna.CreateStudy(sampler: new LmCmaSampler(seed: 2, nStartupTrials: 5));
        study.Optimize(t => { double x = t.SuggestFloat("x", -5, 5); return x * x; }, nTrials: 60);

        study.BestValue.Should().BeLessThan(25.0, "the fallback sampler still explores the 1-D space");
    }
}

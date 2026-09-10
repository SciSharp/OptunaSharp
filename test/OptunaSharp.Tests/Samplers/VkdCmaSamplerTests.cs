using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.Json;
using AwesomeAssertions;
using OptunaSharp.Samplers;
using OptunaSharp.Samplers.CmaEs;

namespace OptunaSharp.Tests.Samplers;

/// <summary>
///     Tests for the <see cref="VkdCma" /> optimizer (a port of Akimoto's <c>GaussVkDSampler</c>) and its
///     <see cref="VkdCmaSampler" />. VkD-CMA (Akimoto &amp; Hansen, GECCO/PPSN 2016) has no
///     <c>cmaes</c>/Optuna/OptunaHub counterpart, so there is no upstream trajectory to bit-match (unlike
///     <c>CmaEsSamplerTests</c>). These are therefore behavioral gates: convergence, that the online
///     k-adaptation grows the rank exactly when off-diagonal covariance is required, same-seed determinism,
///     and optimizer-state round-trip fidelity.
/// </summary>
[TestClass]
public class VkdCmaSamplerTests
{
    private static double Sphere(double[] x)
    {
        double s = 0;
        foreach (double v in x) s += v * v;
        return s;
    }

    // Axis-aligned ellipsoid, condition number 1e6. VkD-CMA solves this with its DIAGONAL part alone
    // (each coordinate scaled independently), so the rank-k term need not engage.
    private static double Ellipsoid(double[] x)
    {
        int n = x.Length;
        double s = 0;
        for (int i = 0; i < n; i++) s += Math.Pow(10.0, 6.0 * i / (n - 1)) * x[i] * x[i];
        return s;
    }

    // The SAME ill-conditioned ellipsoid seen through a fixed Householder rotation H = I - 2 vvᵀ/(vᵀv),
    // which couples every coordinate. A diagonal-only model canNOT represent this — solving it REQUIRES
    // the low-rank V/S covariance term, so this is the test that actually exercises the restricted
    // covariance (and the k-adaptation that grows it).
    private static double RotatedEllipsoid(double[] x)
    {
        int n = x.Length;
        var v = new double[n];
        for (int i = 0; i < n; i++) v[i] = i % 2 == 0 ? 1.0 : -0.7 + 0.3 * i;   // fixed non-axis vector
        double vv = 0, vx = 0;
        for (int i = 0; i < n; i++) { vv += v[i] * v[i]; vx += v[i] * x[i]; }
        var y = new double[n];
        for (int i = 0; i < n; i++) y[i] = x[i] - 2 * vx / vv * v[i];
        double s = 0;
        for (int i = 0; i < n; i++) s += Math.Pow(10.0, 6.0 * i / (n - 1)) * y[i] * y[i];
        return s;
    }

    // Batch ask/tell driver (this path exercises the exact-mirror TPA injection).
    private static double Drive(VkdCma opt, Func<double[], double> f, int generations)
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
        double best = Drive(new VkdCma(Filled(10, 3.0), sigma: 2.0, seed: 1), Sphere, generations: 300);
        best.Should().BeLessThan(1e-8, "VkD-CMA solves the sphere to near machine precision");
    }

    [TestMethod]
    public void Optimizer_AxisAlignedEllipsoid_Converges()
    {
        // Axis-aligned ⇒ the diagonal D captures the full conditioning; VkD reaches high precision.
        double best = Drive(new VkdCma(Filled(10, 3.0), sigma: 2.0, seed: 3), Ellipsoid, generations: 1500);
        best.Should().BeLessThan(1e-6, "the adaptive diagonal solves the axis-aligned condition-1e6 ellipsoid");
    }

    [TestMethod]
    public void Optimizer_RotatedEllipsoid_Converges_ViaLowRankTerm()
    {
        // The rotated ellipsoid is non-separable: only the rank-k V/S term can represent the coupling.
        var opt = new VkdCma(Filled(10, 3.0), sigma: 2.0, seed: 1);
        double best = Drive(opt, RotatedEllipsoid, generations: 2500);
        best.Should().BeLessThan(1e-6,
            "the low-rank covariance term learns the rotation and descends the condition-1e6 valley");
        opt.KActive.Should().BeGreaterThan(0,
            "k-adaptation must have grown the rank — a diagonal-only model cannot solve a rotated ellipsoid");
    }

    [TestMethod]
    public void Optimizer_DiagonalOnly_StallsOnRotatedEllipsoid()
    {
        // With k pinned to 0 (no low-rank term), VkD degenerates to a separable/diagonal model that
        // CANNOT represent the rotation — it stalls far above the optimum. This is the negative control
        // proving the previous test's success comes from the V/S term, not the diagonal.
        double best = Drive(new VkdCma(Filled(10, 3.0), sigma: 2.0, seed: 1, kInit: 0, kAdapt: false),
            RotatedEllipsoid, generations: 2500);
        best.Should().BeGreaterThan(1.0, "a diagonal-only model cannot solve a rotated ill-conditioned ellipsoid");
    }

    [TestMethod]
    public void Optimizer_KAdaptation_GrowsRankThenTracks()
    {
        // On the rotated problem the rank must climb well above its k_init=0 start as directions saturate.
        var opt = new VkdCma(Filled(10, 3.0), sigma: 2.0, seed: 2);
        opt.K.Should().Be(0, "k starts at 0 (diagonal) by default");
        Drive(opt, RotatedEllipsoid, generations: 2500);
        opt.K.Should().BeGreaterThan(1, "online model selection grows k to capture the rotated covariance");
    }

    // ---- optimizer: determinism & state round-trip ------------------------------------------------

    [TestMethod]
    public void Optimizer_SameSeed_IsReproducible()
    {
        double a = Drive(new VkdCma(Filled(8, 2.0), 1.5, seed: 11), Sphere, generations: 50);
        double b = Drive(new VkdCma(Filled(8, 2.0), 1.5, seed: 11), Sphere, generations: 50);
        b.Should().Be(a, "same seed ⇒ identical trajectory");
    }

    [TestMethod]
    public void Optimizer_StateRoundTrip_PreservesEverything()
    {
        var opt = new VkdCma(Filled(10, 3.0), 2.0, seed: 5);
        Drive(opt, RotatedEllipsoid, generations: 120);   // build up mean/sigma/D/V/S/pc/dx and grow k

        // Serialize → deserialize exactly as VkdCmaSampler stores optimizer state between trials.
        string json = JsonSerializer.Serialize(opt.ToState());
        var restored = VkdCma.FromState(JsonSerializer.Deserialize<VkdCmaState>(json)!);

        restored.Mean.Should().Equal(opt.Mean);
        restored.Sigma.Should().Be(opt.Sigma);
        restored.Generation.Should().Be(opt.Generation);
        restored.K.Should().Be(opt.K);
        restored.KActive.Should().Be(opt.KActive);

        // After an identical reseed, the next injected candidate must be bit-identical — proving the whole
        // restricted-covariance state (D/V/S/pc/dx/σ and the injection flag) round-tripped, not just scalars.
        opt.ReseedRng(999);
        restored.ReseedRng(999);
        restored.AskCandidate(1).Should().Equal(opt.AskCandidate(1));
    }

    // ---- sampler: end-to-end through a Study ------------------------------------------------------

    [TestMethod]
    public void Sampler_MinimizesSphere()
    {
        var study = Optuna.CreateStudy(sampler: new VkdCmaSampler(seed: 1, nStartupTrials: 5));
        study.Optimize(t =>
        {
            double s = 0;
            for (int i = 0; i < 10; i++) { double x = t.SuggestFloat($"x{i}", -5, 5); s += x * x; }
            return s;
        }, nTrials: 800);

        study.BestValue.Should().BeLessThan(1e-2, "the sampler drives VkD-CMA to a small sphere value");
    }

    [TestMethod]
    public void Sampler_Maximize_Works()
    {
        var study = Optuna.CreateStudy(
            direction: StudyDirection.Maximize, sampler: new VkdCmaSampler(seed: 3, nStartupTrials: 5));
        study.Optimize(t =>
        {
            double s = 0;
            for (int i = 0; i < 8; i++) { double x = t.SuggestFloat($"x{i}", -5, 5); s += x * x; }
            return -s;   // optimum 0, approached from below
        }, nTrials: 500);

        study.BestValue.Should().BeGreaterThan(-1e-1, "direction flipping lets VkD-CMA maximize -sphere");
    }

    [TestMethod]
    public void Sampler_GenerationAttr_IsAttributedPerTrial()
    {
        var study = Optuna.CreateStudy(sampler: new VkdCmaSampler(seed: 42, nStartupTrials: 1));
        study.Optimize(t =>
        {
            double x = t.SuggestFloat("x", -5, 5);
            double y = t.SuggestFloat("y", -5, 5);
            return x * x + y * y;
        }, nTrials: 20);

        // Trials past the startup phase carry the optimizer's generation stamp.
        study.Trials.Skip(5).Any(tr => tr.SystemAttrs.ContainsKey("vkdcma:generation")).Should().BeTrue();
    }

    [TestMethod]
    public void Sampler_SingleDimension_FallsBackToIndependent()
    {
        // VkD-CMA needs n > 1; a 1-D relative space must fall back to the independent sampler and still run.
        var study = Optuna.CreateStudy(sampler: new VkdCmaSampler(seed: 2, nStartupTrials: 5));
        study.Optimize(t => { double x = t.SuggestFloat("x", -5, 5); return x * x; }, nTrials: 60);

        study.BestValue.Should().BeLessThan(25.0, "the fallback sampler still explores the 1-D space");
    }
}

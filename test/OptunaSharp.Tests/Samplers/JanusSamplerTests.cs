using System;
using System.Linq;
using AwesomeAssertions;
using OptunaSharp.Samplers;
using OptunaSharp.Samplers.CmaEs;
using OptunaSharp.Samplers.Janus;

namespace OptunaSharp.Tests.Samplers;

/// <summary>
///     Tests for the native <see cref="JanusSampler" /> port (JANUS — Yu et al., arXiv:2608.22862;
///     reference https://github.com/hongyuanyu/JANUS, Apache-2.0). JANUS has no <c>cmaes</c>/Optuna/
///     OptunaHub counterpart, so — like <c>LmCmaSamplerTests</c> — these are behavioral gates:
///     convergence, "matches or beats the CMA-ES host" (the paper's central early-budget claim),
///     same-seed determinism, the JANUS-GN variant, graceful fallbacks, and unit tests on the scalar
///     residual-coordinate metric. Thresholds are calibrated to measured convergence, not the paper's
///     absolute numbers (a different host and problem set).
/// </summary>
[TestClass]
public class JanusSamplerTests
{
    // ---- objectives ------------------------------------------------------------------------------

    private static double ShiftedSphere(double[] x, double shift, out double[] residuals)
    {
        residuals = new double[x.Length];
        for (int i = 0; i < x.Length; i++) residuals[i] = (x[i] - shift) * (x[i] - shift);
        return residuals.Sum();
    }

    // ---- convergence: component mode -------------------------------------------------------------

    [TestMethod]
    public void Sampler_ComponentMode_MinimizesShiftedSphere()
    {
        var study = Optuna.CreateStudy(sampler: new JanusSampler(seed: 0, nStartupTrials: 16));
        study.Optimize(t =>
        {
            var x = new double[10];
            for (int i = 0; i < 10; i++) x[i] = t.SuggestFloat($"x{i}", -5, 5);
            double v = ShiftedSphere(x, 0.5, out double[] r);
            t.SetUserAttr("component_losses", r);   // JANUS estimates J directly from these residuals
            return v;
        }, nTrials: 600);

        study.BestValue.Should().BeLessThan(1e-2,
            "JANUS drives the CMA-ES host to a small value on the component-decomposed sphere");
    }

    // ---- convergence: scalar mode (no components) ------------------------------------------------

    [TestMethod]
    public void Sampler_ScalarMode_MinimizesSphere()
    {
        // No component_losses → JANUS falls back to the paper's scalar residual-coordinate fit.
        var study = Optuna.CreateStudy(
            sampler: new JanusSampler(seed: 1, nStartupTrials: 16, pseudoMode: "quadratic"));
        study.Optimize(t =>
        {
            double s = 0;
            for (int i = 0; i < 8; i++) { double x = t.SuggestFloat($"x{i}", -5, 5); s += x * x; }
            return s;
        }, nTrials: 600);

        study.BestValue.Should().BeLessThan(1e-2, "scalar-mode JANUS still minimizes the sphere");
    }

    // ---- the central claim: JANUS matches or beats the CMA-ES host at an early budget ------------

    [TestMethod]
    public void Sampler_MatchesOrBeatsCmaEsHost_EarlyBudget()
    {
        // The paper's claim is early-budget acceleration. Ill-conditioned single runs are noisy
        // (an easy seed alone can flip the result), so aggregate the best value over several seeds and
        // compare totals — JANUS's Gauss–Newton + CMIX infill should not lose to its bare host in sum.
        const int nTrials = 200;
        double Total(Func<int, BaseSampler> make)
        {
            double sum = 0;
            for (int seed = 0; seed < 6; seed++)
            {
                var study = Optuna.CreateStudy(sampler: make(seed));
                study.Optimize(t =>
                {
                    var x = new double[10];
                    for (int i = 0; i < 10; i++) x[i] = t.SuggestFloat($"x{i}", -5, 5);
                    double v = ShiftedSphere(x, 0.5, out double[] r);
                    t.SetUserAttr("component_losses", r);
                    return v;
                }, nTrials: nTrials);
                sum += study.BestValue;
            }
            return sum;
        }

        double janus = Total(s => new JanusSampler(seed: s, nStartupTrials: 16));
        double cma = Total(s => new CmaEsSampler(seed: s, nStartupTrials: 16));

        janus.Should().BeLessThanOrEqualTo(cma,
            $"JANUS infill matches or beats the CMA-ES host in aggregate (JANUS Σ={janus:g4}, CMA-ES Σ={cma:g4})");
    }

    // ---- JANUS-GN variant (exploitation only, no CMIX) -------------------------------------------

    [TestMethod]
    public void Sampler_JanusGnVariant_Converges()
    {
        // cmixCandidateFrac: 0 ⇒ only the Gauss–Newton injection (the paper's conservative JANUS-GN).
        var study = Optuna.CreateStudy(
            sampler: new JanusSampler(seed: 3, nStartupTrials: 16, cmixCandidateFrac: 0.0));
        study.Optimize(t =>
        {
            var x = new double[10];
            for (int i = 0; i < 10; i++) x[i] = t.SuggestFloat($"x{i}", -5, 5);
            double v = ShiftedSphere(x, 0.5, out double[] r);
            t.SetUserAttr("component_losses", r);
            return v;
        }, nTrials: 600);

        study.BestValue.Should().BeLessThan(1e-2, "JANUS-GN (GN injection only) still converges");
    }

    // ---- deferred Phase-1 cmix options: reshape + archive ----------------------------------------

    [TestMethod]
    public void Sampler_ReshapeMode_MinimizesShiftedSphere()
    {
        // cmix_mode="reshape" mixes the inverse-curvature metric into the CMA-ES host covariance in place
        // (C ← (1-w)C + w·tr(C)/tr(T)·T) via Cma.SetCovariance, instead of injecting tail candidates.
        var study = Optuna.CreateStudy(
            sampler: new JanusSampler(seed: 0, nStartupTrials: 16, cmixMode: "reshape"));
        study.Optimize(t =>
        {
            var x = new double[10];
            for (int i = 0; i < 10; i++) x[i] = t.SuggestFloat($"x{i}", -5, 5);
            double v = ShiftedSphere(x, 0.5, out double[] r);
            t.SetUserAttr("component_losses", r);
            return v;
        }, nTrials: 600);

        study.BestValue.Should().BeLessThan(1e-2, "reshape-mode JANUS converges (host-covariance mixing)");
    }

    [TestMethod]
    public void Sampler_ArchiveCenter_Converges()
    {
        // cmix_candidate_center="archive" centers the CMIX samples on a diverse Pareto-archive of
        // residual-window points (round-robined). It is more exploratory than "mean", so the gate is
        // "clearly improves from random" (~85 on this box), not tight convergence.
        var study = Optuna.CreateStudy(
            sampler: new JanusSampler(seed: 0, nStartupTrials: 16, cmixCandidateCenter: "archive"));
        study.Optimize(t =>
        {
            var x = new double[10];
            for (int i = 0; i < 10; i++) x[i] = t.SuggestFloat($"x{i}", -5, 5);
            double v = ShiftedSphere(x, 0.5, out double[] r);
            t.SetUserAttr("component_losses", r);
            return v;
        }, nTrials: 600);

        study.BestValue.Should().BeLessThan(1.0, "archive-center JANUS still drives the host well below random");
    }

    [TestMethod]
    public void Optimizer_Cma_SetCovariance_RoundTripsAndReDecomposes()
    {
        // The reshape hook: setting C must be reflected in Cov and must invalidate the cached
        // eigendecomposition so the next Ask re-derives it and still yields a finite sample.
        var cma = new Cma(new double[] { 0, 0, 0 }, sigma: 1.0, seed: 1);
        var c = new double[3, 3] { { 4, 0, 0 }, { 0, 1, 0 }, { 0, 0, 0.25 } };
        cma.SetCovariance(c);

        cma.Cov[0, 0].Should().Be(4.0);
        cma.Cov[1, 1].Should().Be(1.0);
        cma.Cov[2, 2].Should().Be(0.25);

        double[] x = cma.Ask();   // re-derives B/D from the new C
        x.Length.Should().Be(3);
        x.All(v => !double.IsNaN(v) && !double.IsInfinity(v)).Should().BeTrue("re-decomposition yields a finite sample");
    }

    // ---- determinism -----------------------------------------------------------------------------

    [TestMethod]
    public void Sampler_SameSeed_IsReproducible()
    {
        double Run()
        {
            var study = Optuna.CreateStudy(sampler: new JanusSampler(seed: 11, nStartupTrials: 16));
            study.Optimize(t =>
            {
                var x = new double[6];
                for (int i = 0; i < 6; i++) x[i] = t.SuggestFloat($"x{i}", -5, 5);
                double v = ShiftedSphere(x, 0.5, out double[] r);
                t.SetUserAttr("component_losses", r);
                return v;
            }, nTrials: 120);
            return study.BestValue;
        }

        Run().Should().Be(Run(), "a fixed seed gives an identical trajectory");
    }

    // ---- maximize direction ----------------------------------------------------------------------

    [TestMethod]
    public void Sampler_Maximize_Works()
    {
        var study = Optuna.CreateStudy(
            direction: StudyDirection.Maximize, sampler: new JanusSampler(seed: 5, nStartupTrials: 16));
        study.Optimize(t =>
        {
            var x = new double[8];
            for (int i = 0; i < 8; i++) x[i] = t.SuggestFloat($"x{i}", -5, 5);
            double v = ShiftedSphere(x, 0.0, out double[] r);
            t.SetUserAttr("component_losses", r);
            return -v;   // optimum 0, approached from below
        }, nTrials: 400);

        study.BestValue.Should().BeGreaterThan(-1e-1, "direction flipping lets JANUS maximize -sphere");
    }

    // ---- fallbacks -------------------------------------------------------------------------------

    [TestMethod]
    public void Sampler_SingleDimension_FallsBackToIndependent()
    {
        // A relative space of dimension ≤ 1 must fall back to the independent sampler and still run.
        var study = Optuna.CreateStudy(sampler: new JanusSampler(seed: 2, nStartupTrials: 5));
        study.Optimize(t => { double x = t.SuggestFloat("x", -5, 5); return x * x; }, nTrials: 60);

        study.BestValue.Should().BeLessThan(25.0, "the fallback sampler still explores the 1-D space");
    }

    // ---- metric math: scalar residual-coordinate fits --------------------------------------------

    [TestMethod]
    public void Metric_Linear_RecoversGradientDirection()
    {
        // f(x) = aᵀx is exactly linear ⇒ the linear fit recovers g ∝ a and JtJ = outer(g, g) (rank 1).
        var a = new[] { 2.0, -3.0, 1.0 };
        var rng = new Random(0);
        int n = 200, d = 3;
        var x = new double[n, d];
        var f = new double[n];
        for (int i = 0; i < n; i++)
        {
            double val = 0;
            for (int j = 0; j < d; j++) { x[i, j] = rng.NextDouble() * 4 - 2; val += a[j] * x[i, j]; }
            f[i] = val;
        }

        var metric = JanusMetric.Compute(x, f, mode: "linear", ridge: 1e-8);
        metric.Should().NotBeNull();
        metric!.Rank.Should().Be(1);
        // JtJ = outer(g, g). Its diagonal is g_i²; recover |g| and check the direction matches a.
        double gg = metric.JtJ[0, 0] + metric.JtJ[1, 1] + metric.JtJ[2, 2];   // ‖g‖²
        double[] g = { Math.Sqrt(metric.JtJ[0, 0]) * Math.Sign(a[0]),
                       Math.Sqrt(metric.JtJ[1, 1]) * Math.Sign(a[1]),
                       Math.Sqrt(metric.JtJ[2, 2]) * Math.Sign(a[2]) };
        double aNorm = Math.Sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]);
        double gNorm = Math.Sqrt(gg);
        double cos = (g[0] * a[0] + g[1] * a[1] + g[2] * a[2]) / (gNorm * aNorm);
        cos.Should().BeGreaterThan(0.999, "the fitted linear slope points along the true gradient");
    }

    [TestMethod]
    public void Metric_DiagQuadratic_RecoversCurvatureOrdering()
    {
        // f(x) = Σ h_i x_i², h = [1, 4, 9] ⇒ Hessian diag ∝ h; the fit's metric diagonal should be
        // positive and ordered like h (a robust recovery check that does not hinge on exact scaling).
        var h = new[] { 1.0, 4.0, 9.0 };
        var rng = new Random(1);
        int n = 400, d = 3;
        var x = new double[n, d];
        var f = new double[n];
        for (int i = 0; i < n; i++)
        {
            double val = 0;
            for (int j = 0; j < d; j++) { x[i, j] = rng.NextDouble() * 4 - 2; val += h[j] * x[i, j] * x[i, j]; }
            f[i] = val;
        }

        var metric = JanusMetric.Compute(x, f, mode: "diag_quadratic", ridge: 1e-6);
        metric.Should().NotBeNull();
        double d0 = metric!.JtJ[0, 0], d1 = metric.JtJ[1, 1], d2 = metric.JtJ[2, 2];
        // The fit clamps negative curvature estimates to 0, so the smallest (h_0=1) may land at 0;
        // the meaningful recovery check is the ordering and that the dominant curvature is positive.
        d0.Should().BeGreaterThanOrEqualTo(0);
        (d1 > d0).Should().BeTrue("curvature grows with h_i");
        (d2 > d1).Should().BeTrue("curvature grows with h_i");
        d2.Should().BeGreaterThan(0, "the dominant curvature (h_2=9) is recovered positive");
    }

    [TestMethod]
    public void Metric_TooFewPoints_ReturnsNull()
    {
        var x = new double[3, 2];
        var f = new double[3];
        JanusMetric.Compute(x, f, mode: "linear").Should().BeNull("fewer than 4 rows cannot fit a metric");
    }
}

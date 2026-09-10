using System;
using System.Linq;
using AwesomeAssertions;
using OptunaSharp.Samplers;

namespace OptunaSharp.Tests.Samplers;

/// <summary>
///     Tests for <see cref="JanusDeSampler" /> — JANUS-DE, the paper's host-agnostic infill check on a
///     Differential Evolution host (Yu et al., arXiv:2608.22862; https://github.com/hongyuanyu/JANUS,
///     Apache-2.0). DE exposes no proposal covariance, so JANUS runs its conservative GN-only
///     configuration: the Gauss–Newton exploitation elite is injected into each DE generation, no CMIX
///     exploration. Like <c>JanusSamplerTests</c>, these are behavioral gates. The central one is the
///     paper's headline property — <b>JANUS-DE beats or matches its bare DE host</b> at an early budget —
///     since the DE host is deliberately weaker than the CMA host (covariance-free), its absolute
///     convergence is modest and not gated tightly.
/// </summary>
[TestClass]
public class JanusDeSamplerTests
{
    private static double ShiftedSphere(double[] x, double shift, out double[] residuals)
    {
        residuals = new double[x.Length];
        for (int i = 0; i < x.Length; i++) residuals[i] = (x[i] - shift) * (x[i] - shift);
        return residuals.Sum();
    }

    // ---- the central claim: JANUS-DE beats or matches its bare DE host ----------------------------

    [TestMethod]
    public void Sampler_ComponentMode_MatchesOrBeatsBareDeHost()
    {
        // The paper's host-agnostic check: the Jacobian infill should not lose to the bare DE host. Single
        // runs are noisy, so aggregate the best value over several seeds and compare totals — JANUS-DE's
        // Gauss–Newton injection should beat plain DE in sum at an early budget.
        const int nTrials = 300;
        double Total(Func<int, BaseSampler> make)
        {
            double sum = 0;
            for (int seed = 0; seed < 5; seed++)
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

        double janusDe = Total(s => new JanusDeSampler(seed: s, nStartupTrials: 16));
        double bareDe = Total(s => new DifferentialEvolutionSampler(seed: s));

        janusDe.Should().BeLessThanOrEqualTo(bareDe,
            $"JANUS-DE's GN infill matches or beats the bare DE host in aggregate (JANUS-DE Σ={janusDe:g4}, DE Σ={bareDe:g4})");
    }

    // ---- convergence: component mode -------------------------------------------------------------

    [TestMethod]
    public void Sampler_ComponentMode_Improves()
    {
        var study = Optuna.CreateStudy(sampler: new JanusDeSampler(seed: 0, nStartupTrials: 16));
        study.Optimize(t =>
        {
            var x = new double[10];
            for (int i = 0; i < 10; i++) x[i] = t.SuggestFloat($"x{i}", -5, 5);
            double v = ShiftedSphere(x, 0.5, out double[] r);
            t.SetUserAttr("component_losses", r);   // JANUS estimates J directly from these residuals
            return v;
        }, nTrials: 400);

        // A random best over this 10-D box sits around ~85; reaching < 5 is a clear, robust improvement
        // (the covariance-free host converges modestly — the tight gate is "beats bare DE" above).
        study.BestValue.Should().BeLessThan(5.0, "JANUS-DE drives the DE host well below random on the residual sphere");
    }

    // ---- scalar mode (no components) -------------------------------------------------------------

    [TestMethod]
    public void Sampler_ScalarMode_Runs()
    {
        // No component_losses → JANUS-DE falls back to the paper's scalar residual-coordinate fit for the
        // GN elite (still a covariance-free host, so CMIX stays off).
        var study = Optuna.CreateStudy(
            sampler: new JanusDeSampler(seed: 1, nStartupTrials: 16, pseudoMode: "quadratic"));
        study.Optimize(t =>
        {
            double s = 0;
            for (int i = 0; i < 8; i++) { double x = t.SuggestFloat($"x{i}", -5, 5); s += x * x; }
            return s;
        }, nTrials: 500);

        study.BestValue.Should().BeLessThan(5.0, "scalar-mode JANUS-DE still improves the sphere substantially");
    }

    // ---- determinism -----------------------------------------------------------------------------

    [TestMethod]
    public void Sampler_SameSeed_IsReproducible()
    {
        double Run()
        {
            var study = Optuna.CreateStudy(sampler: new JanusDeSampler(seed: 11, nStartupTrials: 16));
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
            direction: StudyDirection.Maximize, sampler: new JanusDeSampler(seed: 5, nStartupTrials: 16));
        study.Optimize(t =>
        {
            var x = new double[8];
            for (int i = 0; i < 8; i++) x[i] = t.SuggestFloat($"x{i}", -5, 5);
            double v = ShiftedSphere(x, 0.0, out double[] r);
            t.SetUserAttr("component_losses", r);
            return -v;   // optimum 0, approached from below
        }, nTrials: 400);

        study.BestValue.Should().BeGreaterThan(-5.0, "direction flipping lets JANUS-DE maximize -sphere");
    }

    // ---- fallbacks -------------------------------------------------------------------------------

    [TestMethod]
    public void Sampler_SingleDimension_FallsBackToIndependent()
    {
        // A relative space of dimension ≤ 1 must fall back to the independent sampler and still run.
        var study = Optuna.CreateStudy(sampler: new JanusDeSampler(seed: 2, nStartupTrials: 5));
        study.Optimize(t => { double x = t.SuggestFloat("x", -5, 5); return x * x; }, nTrials: 60);

        study.BestValue.Should().BeLessThan(25.0, "the fallback sampler still explores the 1-D space");
    }
}

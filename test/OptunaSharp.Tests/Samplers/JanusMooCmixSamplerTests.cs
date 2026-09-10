using System;
using System.Collections.Generic;
using System.Linq;
using AwesomeAssertions;
using OptunaSharp.Samplers;
using OptunaSharp.Samplers.Janus;

namespace OptunaSharp.Tests.Samplers;

/// <summary>
///     Tests for multi-objective JANUS — <see cref="JanusMooCmixSampler" /> and its SMS-EMOA / AGE-MOEA-2
///     host variants (Yu et al., arXiv:2608.22862; https://github.com/hongyuanyu/JANUS, Apache-2.0). The
///     reference subclasses NSGA-II and occasionally substitutes a child with an inverse-curvature
///     Gaussian sample. These are behavioral gates: the load-bearing one is that, with CMIX disabled, the
///     sampler is <b>bit-identical</b> to its bare NSGA-II host (the subclass never perturbs the host's
///     stream); plus the paper's zero-significant-regression property, the host variants running, and unit
///     tests on the MOO metric geometry. The pure-numpy metric fits get separate byte-parity coverage in
///     <c>JanusMooMetricParityTests</c> (interop).
/// </summary>
[TestClass]
public class JanusMooCmixSamplerTests
{
    // ZDT1 (2-objective) — a shared decision space where NSGA-II builds a clean convex front.
    private static double[] Zdt1(Trial t, int d, bool components)
    {
        var x = new double[d];
        for (int i = 0; i < d; i++) x[i] = t.SuggestFloat($"x{i}", 0, 1);
        double f1 = x[0];
        double gsum = 0; for (int i = 1; i < d; i++) gsum += x[i];
        double g = 1.0 + 9.0 * gsum / (d - 1);
        double f2 = g * (1.0 - Math.Sqrt(f1 / g));
        if (components) t.SetComponentLosses(new[] { f1, f2 });
        return new[] { f1, f2 };
    }

    private static double Hv2D(List<double[]> pts, double r1, double r2)
    {
        var feasible = pts.Where(p => p[0] < r1 && p[1] < r2).ToList();
        var front = feasible
            .Where(p => !feasible.Any(q => q[0] <= p[0] && q[1] <= p[1] && (q[0] < p[0] || q[1] < p[1])))
            .OrderBy(p => p[0]).ToList();
        double hv = 0, prevF1 = r1;
        for (int i = front.Count - 1; i >= 0; i--) { hv += (prevF1 - front[i][0]) * (r2 - front[i][1]); prevF1 = front[i][0]; }
        return hv;
    }

    private static double Hypervolume(Study study) =>
        Hv2D(study.Trials.Where(tr => tr.Values is not null).Select(tr => tr.Values!.ToArray()).ToList(), 1.1, 7.0);

    // ---- the load-bearing gate: CMIX off ⇒ bit-identical to the NSGA-II host ----------------------

    [TestMethod]
    public void Sampler_CmixDisabled_IsBitIdenticalToNsgaHost()
    {
        // cmix_prob = 0 (and start_trial = 0 so the gate is fully exercised) ⇒ every trial keeps the
        // NSGA-II child. Because the subclass always calls base.SampleRelative first and only consults a
        // SEPARATE CMIX stream, the whole study must match bare NSGA-II param-for-param, value-for-value.
        Study Run(BaseSampler s)
        {
            var study = Optuna.CreateStudy(
                directions: new[] { StudyDirection.Minimize, StudyDirection.Minimize }, sampler: s);
            study.Optimize(t => Zdt1(t, 6, components: false), nTrials: 90);
            return study;
        }

        var nsga = Run(new NSGAIISampler(populationSize: 20, seed: 7));
        var janus = Run(new JanusMooCmixSampler(populationSize: 20, seed: 7, cmixProb: 0.0, cmixStartTrial: 0));

        janus.Trials.Count.Should().Be(nsga.Trials.Count);
        for (int i = 0; i < nsga.Trials.Count; i++)
        {
            janus.Trials[i].Values.Should().Equal(nsga.Trials[i].Values, $"trial {i} objective values match NSGA-II");
            foreach (var (k, v) in nsga.Trials[i].Params)
                janus.Trials[i].Params[k].Should().Be(v, $"trial {i} param {k} matches NSGA-II");
        }
    }

    // ---- zero-significant-regression vs the bare host (the paper's MOO property) -------------------

    [TestMethod]
    public void Sampler_DefaultConfig_MatchesNsgaHypervolume()
    {
        // At the conservative default firing rate the inverse-curvature nudge is tiny and NSGA-II filters
        // it, so the achieved hypervolume must not significantly regress from the bare host.
        double nsga = 0, janus = 0;
        for (int seed = 0; seed < 3; seed++)
        {
            var a = Optuna.CreateStudy(directions: new[] { StudyDirection.Minimize, StudyDirection.Minimize },
                sampler: new NSGAIISampler(populationSize: 50, seed: seed));
            a.Optimize(t => Zdt1(t, 8, components: false), nTrials: 250);
            nsga += Hypervolume(a);

            var b = Optuna.CreateStudy(directions: new[] { StudyDirection.Minimize, StudyDirection.Minimize },
                sampler: new JanusMooCmixSampler(populationSize: 50, seed: seed, cmixStartTrial: 60));
            b.Optimize(t => Zdt1(t, 8, components: false), nTrials: 250);
            janus += Hypervolume(b);
        }

        janus.Should().BeGreaterThan(nsga * 0.95,
            $"JANUS-MOO does not significantly regress the host hypervolume (JANUS Σ={janus:g4}, NSGA-II Σ={nsga:g4})");
    }

    // ---- component / scalar modes run and the variants run ----------------------------------------

    [TestMethod]
    public void Sampler_ComponentMode_ProducesAFront()
    {
        var study = Optuna.CreateStudy(
            directions: new[] { StudyDirection.Minimize, StudyDirection.Minimize },
            sampler: new JanusMooCmixSampler(populationSize: 30, seed: 1, cmixProb: 0.2, cmixStartTrial: 20));
        study.Optimize(t => Zdt1(t, 6, components: true), nTrials: 200);
        Hypervolume(study).Should().BeGreaterThan(0.0, "the component-mode run builds a non-trivial Pareto front");
    }

    [TestMethod]
    public void Sampler_ScalarMode_ProducesAFront()
    {
        var study = Optuna.CreateStudy(
            directions: new[] { StudyDirection.Minimize, StudyDirection.Minimize },
            sampler: new JanusMooCmixSampler(populationSize: 30, seed: 2, cmixProb: 0.2, cmixStartTrial: 20));
        study.Optimize(t => Zdt1(t, 6, components: false), nTrials: 200);
        Hypervolume(study).Should().BeGreaterThan(0.0, "scalar-fallback mode builds a non-trivial Pareto front");
    }

    [TestMethod]
    public void Sampler_SmsEmoaVariant_Runs()
    {
        var study = Optuna.CreateStudy(
            directions: new[] { StudyDirection.Minimize, StudyDirection.Minimize },
            sampler: new JanusSmsEmoaCmixSampler(populationSize: 30, seed: 1, cmixProb: 0.2, cmixStartTrial: 20));
        study.Optimize(t => Zdt1(t, 6, components: true), nTrials: 200);
        Hypervolume(study).Should().BeGreaterThan(0.0, "the SMS-EMOA-hosted variant builds a Pareto front");
    }

    [TestMethod]
    public void Sampler_AgeMoea2Variant_Runs()
    {
        var study = Optuna.CreateStudy(
            directions: new[] { StudyDirection.Minimize, StudyDirection.Minimize },
            sampler: new JanusAgeMoea2CmixSampler(populationSize: 30, seed: 1, cmixProb: 0.2, cmixStartTrial: 20));
        study.Optimize(t => Zdt1(t, 6, components: true), nTrials: 200);
        Hypervolume(study).Should().BeGreaterThan(0.0, "the AGE-MOEA-2-hosted variant builds a Pareto front");
    }

    [TestMethod]
    public void Sampler_SameSeed_IsReproducible()
    {
        double Run()
        {
            var study = Optuna.CreateStudy(
                directions: new[] { StudyDirection.Minimize, StudyDirection.Minimize },
                sampler: new JanusMooCmixSampler(populationSize: 30, seed: 11, cmixProb: 0.3, cmixStartTrial: 20));
            study.Optimize(t => Zdt1(t, 6, components: true), nTrials: 150);
            return Hypervolume(study);
        }
        Run().Should().Be(Run(), "a fixed seed gives an identical trajectory (including CMIX draws)");
    }

    // ---- MOO metric unit tests --------------------------------------------------------------------

    [TestMethod]
    public void Metric_DiagQuadratic_RecoversCurvatureOrdering()
    {
        // f(x) = Σ h_i x_i², h = [1, 4, 9] ⇒ the diagonal metric should grow with h_i.
        var h = new[] { 1.0, 4.0, 9.0 };
        var rng = new Random(1);
        int n = 400, d = 3;
        var x = new double[n, d];
        var y = new double[n];
        for (int i = 0; i < n; i++)
        {
            double val = 0;
            for (int j = 0; j < d; j++) { x[i, j] = rng.NextDouble() * 4 - 2; val += h[j] * x[i, j] * x[i, j]; }
            y[i] = val;
        }

        var metric = JanusMooMetric.FitDiagQuadraticMetric(x, y, ridge: 1e-6);
        metric.Should().NotBeNull();
        double d0 = metric![0, 0], d1 = metric[1, 1], d2 = metric[2, 2];
        d0.Should().BeGreaterThan(0);
        (d1 > d0).Should().BeTrue("curvature grows with h_i");
        (d2 > d1).Should().BeTrue("curvature grows with h_i");
    }

    [TestMethod]
    public void Metric_Component_ReturnsSymmetricMetric()
    {
        // f = Σ r_i with r_i = (x_i - c_i)²; the residual Jacobian metric should be a symmetric d×d PSD.
        var c = new[] { 0.2, -0.4, 0.7 };
        var rng = new Random(3);
        int n = 300, d = 3;
        var x = new double[n, d];
        var comp = new double[n, d];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < d; j++) { x[i, j] = rng.NextDouble() * 4 - 2; double r = x[i, j] - c[j]; comp[i, j] = r * r; }

        var metric = JanusMooMetric.FitComponentMetric(x, comp, ridge: 1.0);
        metric.Should().NotBeNull();
        for (int i = 0; i < d; i++)
            for (int j = 0; j < d; j++)
                metric![i, j].Should().BeApproximately(metric[j, i], 1e-9, "the metric is symmetric");
    }

    [TestMethod]
    public void Metric_NondominatedIndices_MatchesReferenceQuirk()
    {
        // Pinned against the LIVE reference: _nondominated_indices returns the DOMINATED points (a faithful
        // reference quirk), not the Pareto front. If this ever "fixes" to the true front it must be a
        // deliberate, documented divergence — not an accident.
        JanusMooMetric.NondominatedIndices(new[]
        {
            new[] { 0.0, 0.0 }, new[] { 1.0, 1.0 }, new[] { 0.0, 1.0 },
        }).Should().Equal(1);
        JanusMooMetric.NondominatedIndices(new[]
        {
            new[] { 3.0 }, new[] { 1.0 }, new[] { 2.0 },
        }).Should().Equal(0);
    }

    [TestMethod]
    public void Metric_CrowdingLike_BoundariesAreInfinite()
    {
        double[] cd = JanusMooMetric.CrowdingLike(new[]
        {
            new[] { 0.0, 3.0 }, new[] { 1.0, 2.0 }, new[] { 2.0, 1.0 }, new[] { 3.0, 0.0 },
        });
        // The two extremes on each objective get +∞; interior points are finite.
        cd.Count(double.IsPositiveInfinity).Should().BeGreaterThanOrEqualTo(2);
        double.IsInfinity(cd[1]).Should().BeFalse("an interior point has finite crowding");
    }
}

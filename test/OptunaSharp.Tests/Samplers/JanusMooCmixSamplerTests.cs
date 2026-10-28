using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using AwesomeAssertions;
using OptunaSharp.Collections;
using OptunaSharp.Distributions;
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

    // ---- componentAttr: the configured key is the component channel -------------------------------
    //
    // Regression gates for review item P2-6 (2026-10-27): every MOO sampler stored `componentAttr` but the
    // CMIX engine always read the constant `component_losses`, so residuals written under a custom key were
    // ignored (scalar fallback) and residuals under the default key were used even when another key was
    // configured. The reference reads `user_attrs.get(self._component_attr)`.

    /// <summary>The custom attribute key these tests store residual vectors under.</summary>
    private const string CustomKey = "residuals";

    /// <summary>
    ///     ZDT1 over four parameters that stores its residual vector <c>[f1, f2]</c> under
    ///     <paramref name="residualKey" />. The canonical key goes through the typed writer users are told to
    ///     call, any other key through <see cref="BaseTrial.SetUserAttr" />, and <c>null</c> stores nothing.
    /// </summary>
    /// <param name="t">The trial being evaluated.</param>
    /// <param name="residualKey">User-attribute key for the residual vector, or null to store none.</param>
    /// <returns>The two ZDT1 objective values.</returns>
    private static IReadOnlyList<double> Zdt1StoringResidualsUnder(Trial t, string? residualKey)
    {
        var x = new double[4];
        for (int i = 0; i < x.Length; i++) x[i] = t.SuggestFloat($"x{i}", 0, 1);
        double f1 = x[0];
        double g = 1.0 + 9.0 * (x[1] + x[2] + x[3]) / 3.0;
        double f2 = g * (1.0 - Math.Sqrt(f1 / g));
        if (residualKey == ComponentLosses.Key)
            t.SetComponentLosses(new[] { f1, f2 });
        else if (residualKey is not null)
            t.SetUserAttr(residualKey, new[] { f1, f2 });
        return new[] { f1, f2 };
    }

    /// <summary>
    ///     Builds one of the four JANUS MOO samplers with CMIX firing on every trial from trial 16 on, so the
    ///     residual source decides the metric of most children in a short study.
    /// </summary>
    /// <param name="host">The host label: <c>NSGA-II</c>, <c>SMS-EMOA</c>, <c>AGE-MOEA-2</c> or <c>AGE-MOEA-2-CPT</c>.</param>
    /// <param name="componentAttr">The <c>componentAttr</c> constructor argument under test.</param>
    /// <returns>A seeded sampler; equal arguments give equal trajectories.</returns>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="host" /> names no JANUS MOO sampler.</exception>
    private static BaseSampler JanusMooSampler(string host, string componentAttr) => host switch
    {
        "NSGA-II" => new JanusMooCmixSampler(
            populationSize: 8, seed: 5, cmixProb: 1.0, cmixStartTrial: 16, componentAttr: componentAttr),
        "SMS-EMOA" => new JanusSmsEmoaCmixSampler(
            populationSize: 8, seed: 5, cmixProb: 1.0, cmixStartTrial: 16, componentAttr: componentAttr),
        "AGE-MOEA-2" => new JanusAgeMoea2CmixSampler(
            populationSize: 8, seed: 5, cmixProb: 1.0, cmixStartTrial: 16, componentAttr: componentAttr),
        "AGE-MOEA-2-CPT" => new JanusAgeMoea2CmixCptSampler(
            populationSize: 8, seed: 5, cmixProb: 1.0, cmixStartTrial: 16, componentAttr: componentAttr),
        _ => throw new ArgumentOutOfRangeException(nameof(host), host, "Not a JANUS MOO host."),
    };

    /// <summary>
    ///     Runs 48 trials and returns each trial's parameters in round-trip form, so two runs compare
    ///     bit-for-bit.
    /// </summary>
    /// <param name="host">The host label passed to <see cref="JanusMooSampler" />.</param>
    /// <param name="componentAttr">The sampler's configured residual key.</param>
    /// <param name="writtenUnder">The key the objective actually stores its residuals under.</param>
    /// <returns>One <c>name=value</c> line per trial, in trial order.</returns>
    private static List<string> ParamTrajectory(string host, string componentAttr, string writtenUnder)
    {
        var study = Optuna.CreateStudy(
            directions: new[] { StudyDirection.Minimize, StudyDirection.Minimize },
            sampler: JanusMooSampler(host, componentAttr));
        study.Optimize(t => Zdt1StoringResidualsUnder(t, writtenUnder), nTrials: 48);
        return study.Trials
            .Select(tr => string.Join(";", tr.Params.Select(p =>
                $"{p.Key}={Convert.ToDouble(p.Value, CultureInfo.InvariantCulture).ToString("R", CultureInfo.InvariantCulture)}")))
            .ToList();
    }

    /// <summary>
    ///     End-to-end through each public JANUS MOO sampler. Residuals under a custom <c>componentAttr</c> must
    ///     reproduce the default-key run bit-for-bit, and residuals under a key other than the configured one must
    ///     not. Before the fix the first comparison failed: the custom key was ignored and CMIX fell back to the
    ///     scalar metric.
    /// </summary>
    /// <param name="host">The host label of the sampler under test.</param>
    [TestMethod]
    [DataRow("NSGA-II")]
    [DataRow("SMS-EMOA")]
    [DataRow("AGE-MOEA-2")]
    [DataRow("AGE-MOEA-2-CPT")]
    public void Sampler_CustomComponentAttr_IsTheResidualSource(string host)
    {
        // The hosts never read user attributes, so the only thing that can separate these runs is which residual
        // vector the CMIX metric is fitted on.
        var custom = ParamTrajectory(host, componentAttr: CustomKey, writtenUnder: CustomKey);
        var canonical = ParamTrajectory(host, componentAttr: ComponentLosses.Key, writtenUnder: ComponentLosses.Key);
        var otherKey = ParamTrajectory(host, componentAttr: CustomKey, writtenUnder: ComponentLosses.Key);

        custom.Should().Equal(canonical,
            $"{host}: residuals under a custom componentAttr drive CMIX exactly like the same residuals under the default key");
        custom.Should().NotEqual(otherKey,
            $"{host}: with componentAttr set, residuals stored under another key must not feed the CMIX metric");
    }

    /// <summary>
    ///     Twelve completed two-objective trials over <c>x0, x1 ∈ [0, 1]</c>, each storing the residual vector
    ///     <c>[(x0 − 0.3)², (x1 − 0.7)²]</c> under every key in <paramref name="residualKeys" />.
    /// </summary>
    /// <param name="jsonListForm">
    ///     Store each vector as the <c>List&lt;object?&gt;</c> of mixed integers and doubles that a persistent storage
    ///     deserializes to, instead of the in-process <c>double[]</c>.
    /// </param>
    /// <param name="residualKeys">The user-attribute keys to store the vector under.</param>
    /// <returns>An in-memory study whose window fills one <c>populationSize: 4</c> CMIX engine.</returns>
    private static Study ResidualHistory(bool jsonListForm, params string[] residualKeys)
    {
        var study = Optuna.CreateStudy(
            directions: new[] { StudyDirection.Minimize, StudyDirection.Minimize },
            sampler: new RandomSampler(seed: 3));
        for (int i = 0; i < 12; i++)
        {
            var trial = study.Ask();
            double x0 = trial.SuggestFloat("x0", 0, 1), x1 = trial.SuggestFloat("x1", 0, 1);
            double r0 = (x0 - 0.3) * (x0 - 0.3), r1 = (x1 - 0.7) * (x1 - 0.7);
            foreach (string key in residualKeys)
            {
                // The JSON form mixes a boxed long with a double, as journal/RDB reads of "[0, 0.25]" do.
                object vector = jsonListForm ? new List<object?> { (long)Math.Round(r0), r1 } : new[] { r0, r1 };
                trial.SetUserAttr(key, vector);
            }
            study.Tell(trial, new[] { x0, 1 - x0 + x1 * x1 });
        }
        return study;
    }

    /// <summary>
    ///     Runs one CMIX draw over <see cref="ResidualHistory" /> and reports the metric family it used:
    ///     <c>component_losses</c> when every windowed trial yielded a residual vector, <c>diag_quadratic</c>
    ///     when it fell back to the scalar fit.
    /// </summary>
    /// <param name="componentAttr">The engine's configured residual key (null turns the channel off).</param>
    /// <param name="jsonListForm">Whether the history stores the persistent-storage list form.</param>
    /// <param name="residualKeys">The keys the history stores its residual vectors under.</param>
    /// <returns>The engine's <see cref="JanusMooCmix.LastMode" /> after the draw.</returns>
    private static string CmixModeFor(string? componentAttr, bool jsonListForm, params string[] residualKeys)
    {
        var cmix = new JanusMooCmix(
            populationSize: 4, seed: 0, cmixProb: 1.0, cmixWindow: "auto", cmixRidge: 1.0, cmixLambda: 1.0,
            cmixSigma: 0.002, cmixStartGeneration: 0, cmixStartTrial: 0, componentAttr: componentAttr,
            scalarFallback: "diag_quadratic");
        var space = new OrderedDict<string, BaseDistribution>
        {
            ["x0"] = new FloatDistribution(0, 1),
            ["x1"] = new FloatDistribution(0, 1),
        };
        cmix.TrySample(ResidualHistory(jsonListForm, residualKeys), space)
            .Should().NotBeNull("a twelve-trial window over a continuous space always yields a CMIX candidate");
        return cmix.LastMode;
    }

    /// <summary>
    ///     The engine fits its metric on residual vectors stored under a custom key, in both the in-process
    ///     <c>double[]</c> form and the list form persistent storages return. Before the fix it read only
    ///     <c>component_losses</c> and took the scalar fallback here.
    /// </summary>
    [TestMethod]
    public void Cmix_CustomComponentAttr_FitsTheMetricOnThatKey()
    {
        CmixModeFor(CustomKey, jsonListForm: false, CustomKey)
            .Should().Be("component_losses", "the residual vectors live under the configured key");
        CmixModeFor(CustomKey, jsonListForm: true, CustomKey)
            .Should().Be("component_losses", "the list form persistent storages return is read under a custom key too");
    }

    /// <summary>
    ///     With a custom key configured, vectors under <c>component_losses</c> or its <c>rae</c> alias are not
    ///     component data, so the engine takes the scalar fallback. Before the fix it used them.
    /// </summary>
    [TestMethod]
    public void Cmix_CustomComponentAttr_NeverBorrowsTheDefaultKeyOrItsAlias()
    {
        CmixModeFor(CustomKey, jsonListForm: false, ComponentLosses.Key, ComponentLosses.LegacyKey)
            .Should().Be("diag_quadratic",
                "a custom key is read exactly as configured, so component_losses and rae are not its residuals");
    }

    /// <summary>
    ///     Guard for the unchanged default: <c>component_losses</c> still reads its own vector and the
    ///     <c>rae</c> read alias, and nothing else.
    /// </summary>
    [TestMethod]
    public void Cmix_DefaultComponentAttr_KeepsItsLegacyAliasAndNothingElse()
    {
        CmixModeFor(ComponentLosses.Key, jsonListForm: false, ComponentLosses.Key)
            .Should().Be("component_losses", "the default key reads the vector SetComponentLosses writes");
        CmixModeFor(ComponentLosses.Key, jsonListForm: false, ComponentLosses.LegacyKey)
            .Should().Be("component_losses", "rae stays a read alias of the default key");
        CmixModeFor(ComponentLosses.Key, jsonListForm: false, CustomKey)
            .Should().Be("diag_quadratic", "the default key does not pick up arbitrary attributes");
    }

    /// <summary>
    ///     A <c>null</c> <c>componentAttr</c> disables the component channel even when residuals exist under
    ///     every key. Before the fix it still read <c>component_losses</c>.
    /// </summary>
    [TestMethod]
    public void Cmix_NullComponentAttr_TurnsTheComponentChannelOff()
    {
        // Mirrors the reference, whose user_attrs.get(None) is always None: no residual vector is ever read.
        CmixModeFor(null, jsonListForm: false, ComponentLosses.Key, CustomKey)
            .Should().Be("diag_quadratic", "a null componentAttr leaves only the scalar fallback");
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

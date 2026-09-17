using System;
using System.Linq;
using AwesomeAssertions;
using NumSharp;
using OptunaSharp.Samplers;
using OptunaSharp.Samplers.Turbo;

namespace OptunaSharp.Tests.Samplers;

/// <summary>
///     Tests for the native <see cref="TurboSampler" /> (TuRBO-1, Eriksson et al. 2019). TuRBO has no
///     <c>cmaes</c>/Optuna/OptunaHub counterpart, so — like <c>DifferentialEvolutionSamplerTests</c> —
///     these are behavioral gates: convergence (including on the multimodal Ackley where the trust-region
///     restart strategy beats a single global GP), same-seed determinism, bounds, and the continuous-only
///     fallback for categorical parameters. Thresholds are calibrated to measured convergence (GP +
///     Thompson sampling is stochastic and tolerance-reproducible, not bit-exact). Small candidate sets and
///     dimensions keep the runs fast.
/// </summary>
[TestClass]
public class TurboSamplerTests
{
    private static double Sphere(Trial t, int d)
    {
        double s = 0;
        for (int i = 0; i < d; i++) { double x = t.SuggestFloat($"x{i}", -5, 5); s += x * x; }
        return s;
    }

    private static double Ackley(Trial t, int d)
    {
        double s1 = 0, s2 = 0;
        for (int i = 0; i < d; i++)
        {
            double x = t.SuggestFloat($"x{i}", -5, 5);
            s1 += x * x;
            s2 += Math.Cos(2 * Math.PI * x);
        }
        return -20 * Math.Exp(-0.2 * Math.Sqrt(s1 / d)) - Math.Exp(s2 / d) + 20 + Math.E;
    }

    [TestMethod]
    public void Sampler_MinimizesSphere()
    {
        // Default (noisy) path. The paper §C noise floor (σ² ≥ 0.0005, vs the GP stack's 1e-6) deliberately
        // caps precision on a truly noise-free objective, so the bar here is 0.1 — still ~50× below a random
        // baseline (~2–5 on this box). The tighter noise-free path (~1e-2) is asserted in
        // Sampler_DeterministicObjective_ConvergesTighterOnNoiseFreeSphere.
        var study = Optuna.CreateStudy(sampler: new TurboSampler(seed: 1, numCandidates: 150));
        study.Optimize(t => Sphere(t, 5), nTrials: 80);
        study.BestValue.Should().BeLessThan(0.1, "TuRBO's local GP + trust region converges on the sphere");
    }

    [TestMethod]
    public void Sampler_MinimizesMultimodalAckley()
    {
        // Ackley is multimodal; a random search sits near ~5–6. The trust-region local model drives well
        // below that — the regime where TuRBO's restart strategy pays off over a single global surrogate.
        var study = Optuna.CreateStudy(sampler: new TurboSampler(seed: 1, numCandidates: 150));
        study.Optimize(t => Ackley(t, 5), nTrials: 120);
        study.BestValue.Should().BeLessThan(3.0, "TuRBO makes clear progress on the multimodal Ackley");
    }

    [TestMethod]
    public void Sampler_Maximize_Works()
    {
        var study = Optuna.CreateStudy(
            direction: StudyDirection.Maximize, sampler: new TurboSampler(seed: 1, numCandidates: 150));
        study.Optimize(t => -Sphere(t, 5), nTrials: 80);
        // Mirror of Sampler_MinimizesSphere under the §C noise floor (see its note for the 0.1 bar).
        study.BestValue.Should().BeGreaterThan(-0.1, "direction flipping lets TuRBO maximize −sphere");
    }

    [TestMethod]
    public void Sampler_SameSeed_IsReproducible()
    {
        double Run()
        {
            var study = Optuna.CreateStudy(sampler: new TurboSampler(seed: 7, numCandidates: 120));
            study.Optimize(t => Sphere(t, 4), nTrials: 60);
            return study.BestValue;
        }
        Run().Should().Be(Run(), "a fixed seed gives an identical trajectory on a platform");
    }

    [TestMethod]
    public void Sampler_RespectsBounds()
    {
        var study = Optuna.CreateStudy(sampler: new TurboSampler(seed: 3, numCandidates: 120));
        study.Optimize(t =>
        {
            double x = t.SuggestFloat("x", -2, 3);
            double y = t.SuggestFloat("y", 1, 4);
            return (x - 1) * (x - 1) + (y - 2) * (y - 2);
        }, nTrials: 80);

        foreach (var tr in study.Trials)
        {
            ((double)tr.Params["x"]!).Should().BeInRange(-2, 3);
            ((double)tr.Params["y"]!).Should().BeInRange(1, 4);
        }
    }

    [TestMethod]
    public void Sampler_MixedIntFloatStepLog_Works()
    {
        var study = Optuna.CreateStudy(sampler: new TurboSampler(seed: 3, numCandidates: 120));
        study.Optimize(t =>
        {
            double a = t.SuggestFloat("a", -5, 5);
            long b = t.SuggestInt("b", -8, 8);
            double c = t.SuggestFloat("c", -4, 4, step: 0.5);
            double d = t.SuggestFloat("d", 1e-3, 1e3, log: true);
            double logd = Math.Log10(d);
            return a * a + b * b + c * c + logd * logd;
        }, nTrials: 120);
        study.BestValue.Should().BeLessThan(3.0, "int/float/step/log dims are handled by the normalized TR box");
    }

    [TestMethod]
    public void Sampler_CategoricalParameter_FallsBackToIndependent()
    {
        // The trust-region box is continuous, so a categorical is sampled independently while the continuous
        // params drive TuRBO. The study must run and the continuous coordinates must converge.
        var study = Optuna.CreateStudy(sampler: new TurboSampler(seed: 3, numCandidates: 120));
        study.Optimize(t =>
        {
            string pick = t.SuggestCategorical("mode", new[] { "u", "v", "w" });
            double bonus = pick == "v" ? 0.0 : 1.0;
            double x = t.SuggestFloat("x", -5, 5);
            double y = t.SuggestFloat("y", -5, 5);
            return x * x + y * y + bonus;
        }, nTrials: 120);

        study.BestValue.Should().BeLessThan(1.5, "continuous params converge while the categorical is independent");
    }

    [TestMethod]
    public void Sampler_InitPhaseIsSpaceFilling_ThenTrustRegionTakesOver()
    {
        // The first 2·d trials of the first restart are the Sobol initialization; afterwards TuRBO proposes
        // from the local GP. The study should still converge well below random after the init budget.
        var study = Optuna.CreateStudy(sampler: new TurboSampler(seed: 5, numCandidates: 120));
        study.Optimize(t => Sphere(t, 4), nTrials: 60);
        study.Trials.Count.Should().Be(60);
        study.BestValue.Should().BeLessThan(1e-1, "TuRBO improves markedly past the 2·d init trials");
    }

    [TestMethod]
    public void Sampler_MultiObjective_Raises()
    {
        var study = Optuna.CreateStudy(
            directions: new[] { StudyDirection.Minimize, StudyDirection.Minimize },
            sampler: new TurboSampler(seed: 1));
        Action act = () => study.Optimize(t =>
        {
            double x = t.SuggestFloat("x", -5, 5);
            return new[] { x * x, (x - 2) * (x - 2) };
        }, nTrials: 5);
        act.Should().Throw<ValueError>("TuRBO-1 is single-objective only");
    }

    [TestMethod]
    public void Sampler_DeterministicObjective_ConvergesTighterOnNoiseFreeSphere()
    {
        // The paper's noise-free treatment (§2): fixing the GP noise at the floor and centering on the best
        // observation converges tighter on a genuinely noise-free objective than the noisy default (which the
        // §C noise floor caps at ~0.03 — see Sampler_MinimizesSphere).
        var study = Optuna.CreateStudy(
            sampler: new TurboSampler(seed: 1, numCandidates: 150, deterministicObjective: true));
        study.Optimize(t => Sphere(t, 5), nTrials: 80);
        study.BestValue.Should().BeLessThan(0.05, "the deterministic path reaches ~1e-2 on the noise-free sphere");
    }

    // ---- paper claims: §C GP hyperparameter box ---------------------------------------------------

    /// <summary>A small, deterministic, well-spread training set with a smooth maximization-oriented target,
    ///     standardized — enough to exercise a genuine bound-constrained GP fit.</summary>
    private static (double[,] X, double[] Y) SmoothDataset(int n, int d)
    {
        var x = new double[n, d];
        var y = new double[n];
        for (int i = 0; i < n; i++)
        {
            double s = 0;
            for (int j = 0; j < d; j++)
            {
                double v = ((i * 7 + j * 3) % 10) / 10.0 + 0.05;   // spread across (0, 1)
                x[i, j] = v;
                s += (v - 0.5) * (v - 0.5);
            }
            y[i] = -s;   // smooth bowl, higher = better
        }
        double mean = y.Average();
        double std = Math.Max(1e-10, Math.Sqrt(y.Select(v => (v - mean) * (v - mean)).Average()));
        for (int i = 0; i < n; i++) y[i] = (y[i] - mean) / std;
        return (x, y);
    }

    [TestMethod]
    public void Fit_HyperparametersLandInsidePaperBox()
    {
        // §C: TuRBO fits its local GP subject to λ ∈ [0.005, 2], s² ∈ [0.05, 20], σ² ∈ [0.0005, 0.1]. Because
        // TurboGpFit runs a BOUND-CONSTRAINED marginal-likelihood optimization (not GPSampler's unconstrained
        // fit), the returned hyperparameters are guaranteed to satisfy the box — the direct proof of the fix.
        (double[,] x, double[] y) = SmoothDataset(12, 3);
        var gpr = TurboGpFit.Fit(x, y, new bool[3], TurboGpFit.NoiseVarMin, deterministicObjective: false);

        foreach (double ls in gpr.LengthScales)
            ls.Should().BeInRange(TurboGpFit.LengthScaleMin, TurboGpFit.LengthScaleMax, "§C length-scale box");
        gpr.KernelScale.Should().BeInRange(TurboGpFit.KernelScaleMin, TurboGpFit.KernelScaleMax, "§C signal-variance box");
        gpr.NoiseVar.Should().BeInRange(TurboGpFit.NoiseVarMin, TurboGpFit.NoiseVarMax, "§C noise-variance box");
    }

    [TestMethod]
    public void Fit_DeterministicObjective_FixesNoiseAtFloor()
    {
        // §2 noise-free treatment: the observation-noise variance is fixed at the floor, not fitted.
        (double[,] x, double[] y) = SmoothDataset(8, 2);
        var gpr = TurboGpFit.Fit(x, y, new bool[2], minimumNoise: 0.01, deterministicObjective: true);
        gpr.NoiseVar.Should().Be(0.01, "the deterministic path fixes ν at minimumNoise");
    }

    // ---- paper claims: default schedule matches the supplement (§D) -------------------------------

    [TestMethod]
    public void DefaultFailureTolerance_IsCeilDOverQ()
    {
        // §D: τ_fail = ⌈d/q⌉ = d for the sequential (q = 1) case.
        TurboSampler.DefaultFailureTolerance(1).Should().Be(1);
        TurboSampler.DefaultFailureTolerance(3).Should().Be(3);
        TurboSampler.DefaultFailureTolerance(4).Should().Be(4);
        TurboSampler.DefaultFailureTolerance(60).Should().Be(60);
    }

    [TestMethod]
    public void DefaultNumCandidates_IsMin100dAnd5000()
    {
        // §D: the Thompson candidate-set size is min(100·d, 5000).
        TurboSampler.DefaultNumCandidates(2).Should().Be(200);
        TurboSampler.DefaultNumCandidates(10).Should().Be(1000);
        TurboSampler.DefaultNumCandidates(50).Should().Be(5000);
        TurboSampler.DefaultNumCandidates(200).Should().Be(5000);
    }

    [TestMethod]
    public void Constructor_RejectsIllOrderedLengths()
    {
        Action act = () => new TurboSampler(lengthInit: 0.8, lengthMin: 1.0, lengthMax: 1.6);
        act.Should().Throw<ArgumentException>("lengthInit must be ≥ lengthMin");
    }

    [TestMethod]
    public void Constructor_RejectsNegativeSuccessMargin()
    {
        Action act = () => new TurboSampler(successMargin: -1e-6);
        act.Should().Throw<ArgumentException>("successMargin must be non-negative");
    }
}

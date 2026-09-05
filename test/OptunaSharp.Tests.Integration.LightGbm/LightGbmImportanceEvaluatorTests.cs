using System;
using System.Collections.Generic;
using System.Linq;
using AwesomeAssertions;
using Microsoft.ML.Trainers.LightGbm;
using NumSharp;
using OptunaSharp.Importance;
using OptunaSharp.Integration.LightGbm;
using OptunaSharp.Samplers;

namespace OptunaSharp.Tests.Integration.LightGbm;

/// <summary>
///     Behavioural tests for the <b>experimental LightGBM</b> importance backends (fANOVA + MDI), run
///     across both via the same <c>Backends[]</c> pattern as the forests / booster suites. These assert
///     only the algorithm's <b>invariants</b> (reproducibility, hyperparameter effects, normalization,
///     ordering, single-value / degenerate handling, the multi-objective guard). There is deliberately
///     <b>no value-parity assertion</b>: LightGBM is a gradient booster, off-label for fANOVA/MDI (its
///     trees fit residuals, not the objective), so the importances are a non-Optuna quantity.
///
///     <para>As with the boosters there is no "different seed differs" test: with full features and no
///     bagging LightGBM is deterministic and seed-independent, so a different seed legitimately produces
///     the identical fit. Determinism (same inputs → identical result) is still asserted.</para>
/// </summary>
[TestClass]
public class LightGbmImportanceEvaluatorTests
{
    private static double Objective(Trial t)
    {
        double x = t.SuggestFloat("x", -10, 10);
        double y = t.SuggestFloat("y", -10, 10);
        long z = t.SuggestInt("z", 0, 10);
        return 10.0 * x * x + 0.001 * y + 0.0001 * z;
    }

    private static Study MakeStudy(int seed = 0, int nTrials = 60)
    {
        var study = Optuna.CreateStudy(sampler: new RandomSampler(seed));
        study.Optimize(Objective, nTrials: nTrials);
        return study;
    }

    private sealed record Backend(string Name, bool IsFanova, Func<int, int, int?, BaseImportanceEvaluator> Make);

    private static readonly Backend[] Backends =
    {
        new("lightgbm-fanova", true, (nt, md, s) => new LightGbmFanovaImportanceEvaluator(nt, md, s)),
        new("lightgbm-mdi", false, (nt, md, s) => new LightGbmMeanDecreaseImpurityImportanceEvaluator(nt, md, s)),
    };

    private static bool SameImportances(IReadOnlyDictionary<string, double> a, IReadOnlyDictionary<string, double> b)
    {
        if (a.Count != b.Count) return false;
        foreach (var (k, v) in a)
            if (!b.TryGetValue(k, out double w) || Math.Abs(v - w) > 1e-12) return false;
        return true;
    }

    [TestMethod]
    public void SameSeed_IsReproducible()
    {
        var study = MakeStudy(seed: 0);
        foreach (var b in Backends)
        {
            var a = b.Make(64, 64, 0).Evaluate(study);
            var c = b.Make(64, 64, 0).Evaluate(study);
            SameImportances(a, c).Should().BeTrue($"{b.Name} with the same inputs must be reproducible");
        }
    }

    [TestMethod]
    public void NTrees_AffectsResult()
    {
        var study = MakeStudy(seed: 0);
        foreach (var b in Backends)
            SameImportances(b.Make(10, 64, 0).Evaluate(study), b.Make(40, 64, 0).Evaluate(study))
                .Should().BeFalse($"{b.Name}: n_trees (boosting rounds) should affect the result");
    }

    [TestMethod]
    public void MaxDepth_AffectsResult()
    {
        var study = MakeStudy(seed: 0);
        foreach (var b in Backends)
            SameImportances(b.Make(64, 1, 0).Evaluate(study), b.Make(64, 8, 0).Evaluate(study))
                .Should().BeFalse($"{b.Name}: max_depth should affect the result");
    }

    [TestMethod]
    public void Normalized_SumsToOne_AndDescending()
    {
        var study = MakeStudy(seed: 0);
        foreach (var b in Backends)
        {
            var imp = ImportanceFunctions.GetParamImportances(study, b.Make(64, 64, 0));
            imp.Values.Sum().Should().BeApproximately(1.0, 1e-9, b.Name);
            var values = imp.Values.ToList();
            for (int i = 1; i < values.Count; i++)
                values[i].Should().BeLessThanOrEqualTo(values[i - 1] + 1e-12, $"{b.Name} importances are descending");
        }
    }

    [TestMethod]
    public void DominantParam_RankedFirst()
    {
        var study = MakeStudy(seed: 0, nTrials: 80);
        foreach (var b in Backends)
            ImportanceFunctions.GetParamImportances(study, b.Make(64, 64, 0)).Keys.First()
                .Should().Be("x", $"{b.Name}: x dominates the objective (10*x^2)");
    }

    [TestMethod]
    public void MultiObjective_WithoutTarget_Throws()
    {
        var study = MakeMultiObjectiveStudy();
        foreach (var b in Backends)
        {
            var act = () => b.Make(64, 64, 0).Evaluate(study);
            act.Should().Throw<ValueError>().WithMessage("*multi-objective*", $"{b.Name} MO guard");
        }
    }

    [TestMethod]
    public void MultiObjective_WithTarget_Works()
    {
        var study = MakeMultiObjectiveStudy();
        foreach (var b in Backends)
        {
            var imp = ImportanceFunctions.GetParamImportances(study, b.Make(64, 64, 0), target: t => t.Values![0]);
            imp.Values.Sum().Should().BeApproximately(1.0, 1e-9, b.Name);
            imp.Keys.Should().Contain(new[] { "a", "b" });
        }
    }

    [TestMethod]
    public void SingleTrial_ReturnsZeros()
    {
        var study = MakeStudy(seed: 0, nTrials: 1);
        foreach (var b in Backends)
            b.Make(64, 64, 0).Evaluate(study).Values.Should().OnlyContain(v => v == 0.0, b.Name);
    }

    [TestMethod]
    public void EmptyParams_ReturnsEmpty()
    {
        var study = MakeStudy(seed: 0);
        foreach (var b in Backends)
            b.Make(64, 64, 0).Evaluate(study, @params: Array.Empty<string>()).Should().BeEmpty(b.Name);
    }

    [TestMethod]
    public void Fanova_SingleValueDistribution_IsZero()
    {
        var study = Optuna.CreateStudy(sampler: new RandomSampler(0));
        study.Optimize((Func<Trial, double>)(t =>
        {
            double x = t.SuggestFloat("x", -10, 10);
            long k = t.SuggestInt("k", 5, 5);  // single-valued
            return 10.0 * x * x + 0.0 * k;
        }), nTrials: 60);

        foreach (var b in Backends.Where(b => b.IsFanova))
        {
            var imp = b.Make(64, 64, 0).Evaluate(study);
            imp["k"].Should().Be(0.0, $"{b.Name}: a single-valued distribution has importance exactly 0");
            imp["x"].Should().BeGreaterThan(0.0, b.Name);
        }
    }

    // ---- Full-API options overload: the caller supplies the real LightGbmRegressionTrainer.Options,
    //      exposing the entire LightGBM surface (LearningRate, NumberOfLeaves, MaximumBinCountPerFeature,
    //      the nested Booster, …) — not just (n_trees, max_depth, seed). ----

    private static LightGbmRegressionTrainer.Options LgbmOptions(double learningRate = 0.2, int seed = 0)
        => new()
        {
            NumberOfIterations = 64, NumberOfLeaves = 32, MinimumExampleCountPerLeaf = 1,
            NumberOfThreads = 1, Seed = seed, LearningRate = learningRate,
        };

    [TestMethod]
    public void Options_Overload_RanksNormalizesAndReproducible()
    {
        var study = MakeStudy(seed: 0, nTrials: 80);
        var evaluators = new (string Name, Func<BaseImportanceEvaluator> Make)[]
        {
            ("lightgbm-fanova", () => new LightGbmFanovaImportanceEvaluator(LgbmOptions())),
            ("lightgbm-mdi", () => new LightGbmMeanDecreaseImpurityImportanceEvaluator(LgbmOptions())),
        };
        foreach (var (name, make) in evaluators)
        {
            var imp = ImportanceFunctions.GetParamImportances(study, make());
            imp.Keys.First().Should().Be("x", $"options-{name}: x dominates the objective (10*x^2)");
            imp.Values.Sum().Should().BeApproximately(1.0, 1e-9, name);
            SameImportances(make().Evaluate(study), make().Evaluate(study))
                .Should().BeTrue($"options-{name}: a fixed options config (Seed + 1 thread) is reproducible");
        }
    }

    [TestMethod]
    public void Options_LearningRate_ExposesControlBeyondSimpleCtor()
    {
        var study = MakeStudy(seed: 0, nTrials: 80);
        // LearningRate is not reachable through the simple (n_trees, max_depth, seed) ctor; the options
        // overload exposes it, and it changes the boosted fit -> different importances.
        SameImportances(
                new LightGbmFanovaImportanceEvaluator(LgbmOptions(learningRate: 0.05)).Evaluate(study),
                new LightGbmFanovaImportanceEvaluator(LgbmOptions(learningRate: 0.40)).Evaluate(study))
            .Should().BeFalse("LightGBM LearningRate (a knob absent from the simple ctor) must reach the fit");
    }

    [TestMethod]
    public void Configure_Overload_TweaksDefaultsRanksAndReproducible()
    {
        var study = MakeStudy(seed: 0, nTrials: 80);
        // Start from the reproducible defaults and tweak only the learning rate (absent from the simple ctor).
        BaseImportanceEvaluator Make() => new LightGbmFanovaImportanceEvaluator(
            o => { o.LearningRate = 0.05; o.Seed = 0; o.NumberOfThreads = 1; });

        var imp = ImportanceFunctions.GetParamImportances(study, Make());
        imp.Keys.First().Should().Be("x", "configure: x dominates the objective (10*x^2)");
        imp.Values.Sum().Should().BeApproximately(1.0, 1e-9);
        SameImportances(Make().Evaluate(study), Make().Evaluate(study)).Should().BeTrue("configure is reproducible");
        SameImportances(Make().Evaluate(study), new LightGbmFanovaImportanceEvaluator(seed: 0).Evaluate(study))
            .Should().BeFalse("the configure tweak (LearningRate=0.05) must reach the fit");
    }

    private static Study MakeMultiObjectiveStudy()
    {
        var study = Optuna.CreateStudy(
            directions: new[] { StudyDirection.Minimize, StudyDirection.Minimize },
            sampler: new RandomSampler(0));
        study.Optimize((Func<Trial, IReadOnlyList<double>>)(t =>
        {
            double a = t.SuggestFloat("a", -3, 3);
            long b = t.SuggestInt("b", 0, 10);
            return new[] { 10.0 * a * a + 0.001 * b, (a - 1) * (a - 1) - b };
        }), nTrials: 60);
        return study;
    }
}

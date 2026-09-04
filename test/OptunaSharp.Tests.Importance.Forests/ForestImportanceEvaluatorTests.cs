using System;
using System.Collections.Generic;
using System.Linq;
using AwesomeAssertions;
using NumSharp;
using OptunaSharp.Importance;
using OptunaSharp.Samplers;

namespace OptunaSharp.Tests.Importance.Forests;

/// <summary>
///     Behavioural tests for the forest-based importance evaluators (fANOVA + MDI). These exercise
///     the ML.NET-backed public API — they assert the algorithm's invariants (reproducibility,
///     hyperparameter effects, normalization, ordering, single-value / degenerate handling, the
///     multi-objective guard), NOT numerical parity with scikit-learn (the forests diverge; see the
///     package README and the live ranking-parity in the interop suite). Adapted from Optuna's
///     <c>tests/importance_tests/test_evaluator.py</c>.
/// </summary>
[TestClass]
public class ForestImportanceEvaluatorTests
{
    // Objective with a clearly dominant parameter (x), a near-irrelevant one (y), and an int (z).
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

    private static IEnumerable<BaseImportanceEvaluator> Evaluators(int? seed = 0, int nTrees = 64, int maxDepth = 64)
    {
        yield return new FanovaImportanceEvaluator(nTrees: nTrees, maxDepth: maxDepth, seed: seed);
        yield return new MeanDecreaseImpurityImportanceEvaluator(nTrees: nTrees, maxDepth: maxDepth, seed: seed);
    }

    private static bool SameImportances(IReadOnlyDictionary<string, double> a, IReadOnlyDictionary<string, double> b)
    {
        if (a.Count != b.Count)
            return false;
        foreach (var (k, v) in a)
        {
            if (!b.TryGetValue(k, out double w) || Math.Abs(v - w) > 1e-12)
                return false;
        }
        return true;
    }

    [TestMethod]
    public void SameSeed_IsReproducible()
    {
        var study = MakeStudy(seed: 0);
        foreach (var _ in new[] { 0 })
        {
            var a = new FanovaImportanceEvaluator(seed: 2).Evaluate(study);
            var b = new FanovaImportanceEvaluator(seed: 2).Evaluate(study);
            SameImportances(a, b).Should().BeTrue("fANOVA with the same seed must be reproducible");

            var c = new MeanDecreaseImpurityImportanceEvaluator(seed: 2).Evaluate(study);
            var d = new MeanDecreaseImpurityImportanceEvaluator(seed: 2).Evaluate(study);
            SameImportances(c, d).Should().BeTrue("MDI with the same seed must be reproducible");
        }
    }

    [TestMethod]
    public void DifferentSeed_Differs()
    {
        var study = MakeStudy(seed: 0);

        var f2 = new FanovaImportanceEvaluator(seed: 2).Evaluate(study);
        var f3 = new FanovaImportanceEvaluator(seed: 3).Evaluate(study);
        SameImportances(f2, f3).Should().BeFalse("fANOVA with different seeds should differ");

        var m2 = new MeanDecreaseImpurityImportanceEvaluator(seed: 2).Evaluate(study);
        var m3 = new MeanDecreaseImpurityImportanceEvaluator(seed: 3).Evaluate(study);
        SameImportances(m2, m3).Should().BeFalse("MDI with different seeds should differ");
    }

    [TestMethod]
    public void NTrees_AffectsResult()
    {
        var study = MakeStudy(seed: 0);
        foreach (var kind in new[] { "fanova", "mdi" })
        {
            BaseImportanceEvaluator Make(int nTrees) => kind == "fanova"
                ? new FanovaImportanceEvaluator(nTrees: nTrees, seed: 0)
                : new MeanDecreaseImpurityImportanceEvaluator(nTrees: nTrees, seed: 0);
            SameImportances(Make(10).Evaluate(study), Make(30).Evaluate(study))
                .Should().BeFalse($"{kind}: n_trees should affect the result");
        }
    }

    [TestMethod]
    public void MaxDepth_AffectsResult()
    {
        var study = MakeStudy(seed: 0);
        foreach (var kind in new[] { "fanova", "mdi" })
        {
            BaseImportanceEvaluator Make(int maxDepth) => kind == "fanova"
                ? new FanovaImportanceEvaluator(maxDepth: maxDepth, seed: 0)
                : new MeanDecreaseImpurityImportanceEvaluator(maxDepth: maxDepth, seed: 0);
            SameImportances(Make(1).Evaluate(study), Make(8).Evaluate(study))
                .Should().BeFalse($"{kind}: max_depth should affect the result");
        }
    }

    [TestMethod]
    public void Normalized_SumsToOne_AndDescending()
    {
        var study = MakeStudy(seed: 0);
        foreach (var evaluator in Evaluators())
        {
            var imp = ImportanceFunctions.GetParamImportances(study, evaluator);
            imp.Values.Sum().Should().BeApproximately(1.0, 1e-9);

            var values = imp.Values.ToList();
            for (int i = 1; i < values.Count; i++)
                values[i].Should().BeLessThanOrEqualTo(values[i - 1] + 1e-12, "importances are descending");
        }
    }

    [TestMethod]
    public void DominantParam_RankedFirst()
    {
        var study = MakeStudy(seed: 0, nTrials: 80);
        foreach (var evaluator in Evaluators())
        {
            var imp = ImportanceFunctions.GetParamImportances(study, evaluator);
            imp.Keys.First().Should().Be("x", "x dominates the objective (10*x^2)");
        }
    }

    [TestMethod]
    public void MultiObjective_WithoutTarget_Throws()
    {
        var study = Optuna.CreateStudy(
            directions: new[] { StudyDirection.Minimize, StudyDirection.Minimize },
            sampler: new RandomSampler(0));
        study.Optimize((Func<Trial, IReadOnlyList<double>>)(t =>
        {
            double a = t.SuggestFloat("a", -3, 3);
            long b = t.SuggestInt("b", 0, 10);
            return new[] { a * a + b, (a - 1) * (a - 1) - b };
        }), nTrials: 30);

        foreach (var evaluator in Evaluators())
        {
            var act = () => evaluator.Evaluate(study);
            act.Should().Throw<ValueError>()
                .WithMessage("*multi-objective*", "the MO-without-target guard must fire");
        }
    }

    [TestMethod]
    public void MultiObjective_WithTarget_Works()
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

        foreach (var evaluator in Evaluators())
        {
            var imp = ImportanceFunctions.GetParamImportances(study, evaluator, target: t => t.Values![0]);
            imp.Values.Sum().Should().BeApproximately(1.0, 1e-9);
            imp.Keys.Should().Contain(new[] { "a", "b" });
        }
    }

    [TestMethod]
    public void SingleTrial_ReturnsZeros()
    {
        var study = MakeStudy(seed: 0, nTrials: 1);
        foreach (var evaluator in Evaluators())
        {
            var imp = evaluator.Evaluate(study);
            imp.Values.Should().OnlyContain(v => v == 0.0);
        }
    }

    [TestMethod]
    public void EmptyParams_ReturnsEmpty()
    {
        var study = MakeStudy(seed: 0);
        foreach (var evaluator in Evaluators())
            evaluator.Evaluate(study, @params: Array.Empty<string>()).Should().BeEmpty();
    }

    [TestMethod]
    public void Fanova_SingleValueDistribution_IsZero()
    {
        // A parameter fixed to a single value is a single() distribution; fANOVA short-circuits it
        // to 0 importance (it never reaches the forest).
        var study = Optuna.CreateStudy(sampler: new RandomSampler(0));
        study.Optimize((Func<Trial, double>)(t =>
        {
            double x = t.SuggestFloat("x", -10, 10);
            long k = t.SuggestInt("k", 5, 5);  // single-valued
            return 10.0 * x * x + 0.0 * k;
        }), nTrials: 60);

        var imp = new FanovaImportanceEvaluator(seed: 0).Evaluate(study);
        imp["k"].Should().Be(0.0, "a single-valued distribution has importance exactly 0");
        imp["x"].Should().BeGreaterThan(0.0);
    }
}

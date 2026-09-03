using System;
using System.Collections.Generic;
using System.Linq;
using AwesomeAssertions;
using OptunaSharp.Collections;
using OptunaSharp.Importance;
using OptunaSharp.Importance.PedAnova;
using OptunaSharp.Samplers;
using OptunaSharp.Storages;

namespace OptunaSharp.Tests.Importance;

/// <summary>
///     Port of Optuna's <c>tests/importance_tests/pedanova_tests/test_evaluator.py</c>, plus
///     end-to-end importance anchors captured from live Optuna 5.0.0.dev. The single-objective
///     end-to-end cases build the identical study in C# (bit-exact MT19937 <see cref="RandomSampler" />)
///     and assert the importances Python produced, bit-for-bit.
/// </summary>
[TestClass]
public class PedAnovaEvaluatorTests
{
    // ------------------------------------------------------------------ _QuantileFilter

    private static FrozenTrial ValueTrial(int index, params double[] values)
        => FrozenTrial.CreateTrial(
            values: values,
            userAttrs: new OrderedDict<string, object?> { ["index"] = index });

    // _VALUES = [[9], [8], ..., [0]]; _MULTI_VALUES = [[0, 9], [1, 8], ..., [9, 0]].
    private static List<FrozenTrial> SingleValueTrials()
        => Enumerable.Range(0, 10).Select(i => ValueTrial(i, 9 - i)).ToList();

    private static List<FrozenTrial> MultiValueTrials()
        => Enumerable.Range(0, 10).Select(i => ValueTrial(i, i, 9 - i)).ToList();

    private static void AssertFilter(
        double quantile, bool isLowerBetter, List<FrozenTrial> trials,
        Func<FrozenTrial, double>? target, int[] expected)
    {
        var filtered = new QuantileFilter(quantile, isLowerBetter, target).Filter(trials);
        filtered.Select(t => (int)t.UserAttrs["index"]!).Should().Equal(expected);
    }

    [TestMethod]
    public void QuantileFilter_SingleObjective()
    {
        AssertFilter(0.1, true, new List<FrozenTrial> { ValueTrial(0, 1), ValueTrial(1, 2) }, null, new[] { 0 });
        AssertFilter(0.49, true, SingleValueTrials(), null, new[] { 5, 6, 7, 8, 9 });
        AssertFilter(0.5, true, SingleValueTrials(), null, new[] { 5, 6, 7, 8, 9 });
        AssertFilter(0.51, true, SingleValueTrials(), null, new[] { 4, 5, 6, 7, 8, 9 });
        AssertFilter(1.0, true, new List<FrozenTrial> { ValueTrial(0, 1), ValueTrial(1, 2) }, null, new[] { 0, 1 });
        AssertFilter(0.49, false, SingleValueTrials(), null, new[] { 0, 1, 2, 3, 4 });
        AssertFilter(0.5, false, SingleValueTrials(), null, new[] { 0, 1, 2, 3, 4 });
        AssertFilter(0.51, false, SingleValueTrials(), null, new[] { 0, 1, 2, 3, 4, 5 });
    }

    [TestMethod]
    public void QuantileFilter_MultiObjectiveTarget()
    {
        AssertFilter(0.49, true, MultiValueTrials(), t => t.Values![0], new[] { 0, 1, 2, 3, 4 });
        AssertFilter(0.5, true, MultiValueTrials(), t => t.Values![0], new[] { 0, 1, 2, 3, 4 });
        AssertFilter(0.51, true, MultiValueTrials(), t => t.Values![0], new[] { 0, 1, 2, 3, 4, 5 });
        AssertFilter(0.49, true, MultiValueTrials(), t => t.Values![1], new[] { 5, 6, 7, 8, 9 });
        AssertFilter(0.5, true, MultiValueTrials(), t => t.Values![1], new[] { 5, 6, 7, 8, 9 });
        AssertFilter(0.51, true, MultiValueTrials(), t => t.Values![1], new[] { 4, 5, 6, 7, 8, 9 });
    }

    // ------------------------------------------------------------------ end-to-end importances

    // optuna.testing.pytest_importance._objective.
    private static double GetStudyObjective(Trial t)
    {
        double x1 = t.SuggestFloat("x1", 0.1, 3);
        double x2 = t.SuggestFloat("x2", 0.1, 3, log: true);
        double x3 = t.SuggestFloat("x3", 2, 4, log: true);
        return x1 + x2 * x3;
    }

    private static Study GetStudy(int seed, int nTrials)
    {
        var study = Optuna.CreateStudy(storage: new InMemoryStorage(), sampler: new RandomSampler(seed));
        study.Optimize(GetStudyObjective, nTrials: nTrials);
        return study;
    }

    // The importance values match Python to a few ULP (the erf/ucrt divergence documented on
    // TruncNorm); the trial selection and dict ordering are exact.
    private const double ImportanceTolerance = 1e-12;

    [TestMethod]
    public void GetParamImportances_SingleObjective_MatchesOptuna()
    {
        var importances = ImportanceFunctions.GetParamImportances(GetStudy(seed: 0, nTrials: 20));
        importances["x1"].Should().BeApproximately(0.46822886157675137, ImportanceTolerance);
        importances["x2"].Should().BeApproximately(0.4384947335292527, ImportanceTolerance);
        importances["x3"].Should().BeApproximately(0.09327640489399588, ImportanceTolerance);
        // Descending order (dict order matches Optuna's sort-by-importance).
        importances.Keys.Should().Equal("x1", "x2", "x3");
    }

    [TestMethod]
    public void GetParamImportances_IntFloatMix_NormalizedAndRaw()
    {
        var study = Optuna.CreateStudy(storage: new InMemoryStorage(), sampler: new RandomSampler(42));
        study.Optimize(t =>
        {
            long x = t.SuggestInt("x", 0, 2);
            double y = t.SuggestFloat("y", -1.0, 1.0);
            double z = t.SuggestFloat("z", 0.0, 1.5);
            return x * x + Math.Pow(y, 3) - Math.Pow(z, 4);
        }, nTrials: 100);

        var norm = ImportanceFunctions.GetParamImportances(study);
        norm["z"].Should().BeApproximately(0.9265534130647286, ImportanceTolerance);
        norm["x"].Should().BeApproximately(0.04114064352930909, ImportanceTolerance);
        norm["y"].Should().BeApproximately(0.03230594340596233, ImportanceTolerance);
        norm.Keys.Should().Equal("z", "x", "y");

        var raw = ImportanceFunctions.GetParamImportances(study, normalize: false);
        raw["z"].Should().BeApproximately(0.055565226964442854, ImportanceTolerance);
        raw["x"].Should().BeApproximately(0.002467196346088683, ImportanceTolerance);
        raw["y"].Should().BeApproximately(0.0019373811075987467, ImportanceTolerance);
    }

    // ------------------------------------------------------------------ behaviour

    [DataTestMethod]
    [DataRow(0)]
    [DataRow(1)]
    public void NTrialsLessThanTwo_ImportancesAreZero(int nTrials)
    {
        var importances = new PedAnovaImportanceEvaluator().Evaluate(GetStudy(seed: 0, nTrials: nTrials));
        // With 0 completed trials there are no parameters at all (an empty dict), matching Python's
        // np.allclose([], zeros(0)); with 1 trial every parameter's importance is exactly 0.
        if (nTrials == 0)
            importances.Should().BeEmpty();
        else
            importances.Values.Should().OnlyContain(v => v == 0.0);
    }

    [TestMethod]
    public void TwoTrials_ImportancesNotAllZero()
    {
        var importances = new PedAnovaImportanceEvaluator().Evaluate(GetStudy(seed: 0, nTrials: 2));
        importances.Values.Should().Contain(v => v != 0.0);
    }

    [TestMethod]
    public void TargetQuantile_ChangesResult()
    {
        var study = GetStudy(seed: 0, nTrials: 20);
        var byDefault = new PedAnovaImportanceEvaluator(targetQuantile: 0.1).Evaluate(study);
        var alt = new PedAnovaImportanceEvaluator(targetQuantile: 0.3).Evaluate(study);
        ImportancesEqual(byDefault, alt).Should().BeFalse();
    }

    [TestMethod]
    public void RegionQuantile_ChangesResult()
    {
        var study = GetStudy(seed: 0, nTrials: 20);
        var byDefault = new PedAnovaImportanceEvaluator(regionQuantile: 1.0).Evaluate(study);
        var alt = new PedAnovaImportanceEvaluator(regionQuantile: 0.5).Evaluate(study);
        ImportancesEqual(byDefault, alt).Should().BeFalse();
    }

    [TestMethod]
    public void EvaluateOnLocal_ChangesResult()
    {
        var study = GetStudy(seed: 0, nTrials: 20);
        var local = new PedAnovaImportanceEvaluator(evaluateOnLocal: true).Evaluate(study);
        var global = new PedAnovaImportanceEvaluator(evaluateOnLocal: false).Evaluate(study);
        ImportancesEqual(local, global).Should().BeFalse();
    }

    [TestMethod]
    public void Direction_ChangesResult()
    {
        var minStudy = GetStudy(seed: 0, nTrials: 20);
        var maxStudy = Optuna.CreateStudy(
            storage: new InMemoryStorage(), direction: StudyDirection.Maximize);
        maxStudy.AddTrials(minStudy.Trials);

        var evaluator = new PedAnovaImportanceEvaluator();
        ImportancesEqual(evaluator.Evaluate(minStudy), evaluator.Evaluate(maxStudy)).Should().BeFalse();
    }

    // ------------------------------------------------------------------ multi-objective / HSSP seam

    [TestMethod]
    public void MultiObjectiveTargetNone_ExactRankPartition_DoesNotThrow()
    {
        // values = [[i, i] for i in range(6)]: each trial is its own non-domination rank, so the
        // target (n_below = 3) and region (n_below = 5) select whole ranks — no HSSP tie-break.
        var study = Optuna.CreateStudy(
            storage: new InMemoryStorage(),
            directions: new[] { StudyDirection.Minimize, StudyDirection.Minimize });
        study.AddTrials(Enumerable.Range(0, 6).Select(i => FrozenTrial.CreateTrial(values: new double[] { i, i })));

        var evaluator = new PedAnovaImportanceEvaluator(targetQuantile: 0.5, regionQuantile: 0.8);
        var importances = evaluator.Evaluate(study);
        importances.Should().BeEmpty();  // No parameters were suggested.
    }

    [TestMethod]
    public void MultiObjectiveTargetNone_HsspTieBreak_Completes()
    {
        // values with a shared non-domination rank at the quantile boundary force an HSSP tie-break
        // in _split_complete_trials_multi_objective (Phase 2.10 — this path used to throw).
        var values = new[]
        {
            new double[] { 0, 0 }, new double[] { 1, 5 }, new double[] { 2, 4 }, new double[] { 3, 3 },
            new double[] { 4, 2 }, new double[] { 5, 1 }, new double[] { 6, 6 }, new double[] { 7, 7 },
        };
        var study = Optuna.CreateStudy(
            storage: new InMemoryStorage(),
            directions: new[] { StudyDirection.Minimize, StudyDirection.Minimize });
        study.AddTrials(values.Select(v => FrozenTrial.CreateTrial(values: v)));

        var evaluator = new PedAnovaImportanceEvaluator(targetQuantile: 0.5, regionQuantile: 0.75);
        var act = () => evaluator.Evaluate(study);
        act.Should().NotThrow();
        act().Should().BeEmpty();  // The trials carry no parameters.
    }

    private static bool ImportancesEqual(OrderedDict<string, double> a, OrderedDict<string, double> b)
    {
        if (a.Count != b.Count)
            return false;
        foreach (var (k, v) in a)
        {
            if (!b.TryGetValue(k, out double other) || other != v)
                return false;
        }
        return true;
    }
}

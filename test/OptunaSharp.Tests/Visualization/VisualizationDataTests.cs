using System;
using System.Collections.Generic;
using System.Linq;
using AwesomeAssertions;
using NumSharp;
using OptunaSharp.Collections;
using OptunaSharp.Distributions;
using OptunaSharp.Importance;
using OptunaSharp.Internal;
using OptunaSharp.Samplers;
using OptunaSharp.Visualization;

namespace OptunaSharp.Tests.Visualization;

/// <summary>
///     Unit tests for the Phase 3.1 part-1 visualization data layer (<see cref="IntermediateValues" />,
///     <see cref="OptimizationHistory" />, <see cref="Edf" />) and the numpy helpers they rely on.
///     Studies are built from hand-constructed <see cref="FrozenTrial" />s so the expected info is
///     exact; live-Optuna bit-parity lives in the interop suite.
/// </summary>
[TestClass]
public class VisualizationDataTests
{
    private static FrozenTrial Complete(double value, OrderedDict<string, double>? constraints = null,
        OrderedDict<int, double>? intermediateValues = null)
        => FrozenTrial.CreateTrial(
            state: TrialState.Complete, value: value, constraints: constraints,
            intermediateValues: intermediateValues);

    // ------------------------------------------------------------------ intermediate values

    [TestMethod]
    public void IntermediateValues_SortsItems_AndFlagsFeasibility()
    {
        var study = Optuna.CreateStudy();
        study.AddTrials(new[]
        {
            // Out-of-order steps: must be sorted ascending by step.
            Complete(1.0, intermediateValues: new OrderedDict<int, double> { { 2, 30.0 }, { 0, 10.0 }, { 1, 20.0 } }),
            // Infeasible (constraint > 0), with intermediate values.
            Complete(2.0, constraints: new OrderedDict<string, double> { { "c", 5.0 } },
                intermediateValues: new OrderedDict<int, double> { { 0, 1.0 } }),
            // No intermediate values → excluded from trial_infos entirely.
            Complete(3.0),
        });

        var info = IntermediateValues.GetPlotInfo(study);

        info.TrialInfos.Should().HaveCount(2);

        var t0 = info.TrialInfos[0];
        t0.TrialNumber.Should().Be(0);
        t0.Feasible.Should().BeTrue();
        t0.SortedIntermediateValues.Should().Equal((0, 10.0), (1, 20.0), (2, 30.0));

        var t1 = info.TrialInfos[1];
        t1.TrialNumber.Should().Be(1);
        t1.Feasible.Should().BeFalse();
        t1.SortedIntermediateValues.Should().Equal((0, 1.0));
    }

    [TestMethod]
    public void IntermediateValues_EmptyStudy_ReturnsNoTrialInfos()
    {
        var study = Optuna.CreateStudy();
        IntermediateValues.GetPlotInfo(study).TrialInfos.Should().BeEmpty();
    }

    // ------------------------------------------------------------------ optimization history

    [TestMethod]
    public void OptimizationHistory_Minimize_AccumulatesRunningBest()
    {
        var study = Optuna.CreateStudy(direction: StudyDirection.Minimize);
        study.AddTrials(new[] { Complete(3.0), Complete(1.0), Complete(2.0) });

        var info = OptimizationHistory.GetInfoList(study).Single();

        info.TrialNumbers.Should().Equal(0, 1, 2);
        info.ValuesInfo.Values.Should().Equal(3.0, 1.0, 2.0);
        info.ValuesInfo.LabelName.Should().Be("Objective Value");
        info.ValuesInfo.States.Should().Equal(ValueState.Feasible, ValueState.Feasible, ValueState.Feasible);
        info.ValuesInfo.Stds.Should().BeNull();

        info.BestValuesInfo!.Values.Should().Equal(3.0, 1.0, 1.0);  // running min.
        info.BestValuesInfo.LabelName.Should().Be("Best Value");
    }

    [TestMethod]
    public void OptimizationHistory_Maximize_AccumulatesRunningBest()
    {
        var study = Optuna.CreateStudy(direction: StudyDirection.Maximize);
        study.AddTrials(new[] { Complete(1.0), Complete(3.0), Complete(2.0) });

        var info = OptimizationHistory.GetInfoList(study).Single();

        info.BestValuesInfo!.Values.Should().Equal(1.0, 3.0, 3.0);  // running max.
    }

    [TestMethod]
    public void OptimizationHistory_InfeasibleAndIncomplete_UseSentinelsInBest()
    {
        var study = Optuna.CreateStudy(direction: StudyDirection.Minimize);
        study.AddTrials(new[]
        {
            Complete(5.0),                                                                 // feasible
            Complete(1.0, constraints: new OrderedDict<string, double> { { "c", 2.0 } }),  // infeasible
            FrozenTrial.CreateTrial(state: TrialState.Running),                            // incomplete
            Complete(3.0),                                                                 // feasible
        });

        var info = OptimizationHistory.GetInfoList(study).Single();

        info.ValuesInfo.States.Should().Equal(
            ValueState.Feasible, ValueState.Infeasible, ValueState.Incomplete, ValueState.Feasible);
        // Incomplete → NaN in the raw values.
        double.IsNaN(info.ValuesInfo.Values[2]).Should().BeTrue();
        // Best-value accumulate skips the infeasible (value 1.0) and incomplete: 5,5,5,3.
        info.BestValuesInfo!.Values.Should().Equal(5.0, 5.0, 5.0, 3.0);
    }

    [TestMethod]
    public void OptimizationHistory_TargetFunction_HasNoBestValues()
    {
        var study = Optuna.CreateStudy(direction: StudyDirection.Minimize);
        study.AddTrials(new[] { Complete(3.0), Complete(1.0) });

        // A custom target (target_name left at default → a UserWarning is emitted, but no error).
        var info = OptimizationHistory.GetInfoList(study, target: t => -(t.Value ?? 0.0)).Single();

        info.ValuesInfo.Values.Should().Equal(-3.0, -1.0);
        info.BestValuesInfo.Should().BeNull();
    }

    [TestMethod]
    public void OptimizationHistory_MultiObjective_NullTarget_Throws()
    {
        var study = Optuna.CreateStudy(directions: new[] { StudyDirection.Minimize, StudyDirection.Minimize });
        study.AddTrials(new[] { FrozenTrial.CreateTrial(values: new[] { 1.0, 2.0 }) });

        var act = () => OptimizationHistory.GetInfoList(study);
        act.Should().Throw<ValueError>()
            .WithMessage("If the `study` is being used for multi-objective optimization, *");
    }

    [TestMethod]
    public void OptimizationHistory_NoCompleteTrials_ReturnsEmpty()
    {
        var study = Optuna.CreateStudy();  // no trials at all.
        OptimizationHistory.GetInfoList(study).Should().BeEmpty();
    }

    [TestMethod]
    public void OptimizationHistory_ErrorBar_AggregatesMeanAndStd()
    {
        var studyA = Optuna.CreateStudy(studyName: "a", direction: StudyDirection.Minimize);
        studyA.AddTrials(new[] { Complete(2.0), Complete(4.0) });
        var studyB = Optuna.CreateStudy(studyName: "b", direction: StudyDirection.Minimize);
        studyB.AddTrials(new[] { Complete(4.0), Complete(6.0) });

        var info = OptimizationHistory.GetInfoList(new[] { studyA, studyB }, errorBar: true).Single();

        info.TrialNumbers.Should().Equal(0, 1);
        // n=0: {2,4} → mean 3, std 1;  n=1: {4,6} → mean 5, std 1.
        info.ValuesInfo.Values.Should().Equal(3.0, 5.0);
        info.ValuesInfo.Stds.Should().Equal(1.0, 1.0);
        info.ValuesInfo.LabelName.Should().Be("Objective Value");
        // Best values also aggregate (running min per study is {2,4}/{2,2} then {4,4}... averaged).
        info.BestValuesInfo!.LabelName.Should().Be("Best Value");
    }

    // ------------------------------------------------------------------ EDF

    [TestMethod]
    public void Edf_LinspaceGrid_AndCumulativeCounts()
    {
        var study = Optuna.CreateStudy(direction: StudyDirection.Minimize);
        study.AddTrials(new[] { Complete(0.0), Complete(1.0), Complete(2.0), Complete(3.0) });

        var info = Edf.GetInfo(study);

        info.XValues.Should().HaveCount(100);
        info.XValues[0].Should().Be(0.0);
        info.XValues[99].Should().Be(3.0);

        info.Lines.Should().HaveCount(1);
        var y = info.Lines[0].YValues;
        y.Should().HaveCount(100);
        y[0].Should().Be(0.25);   // exactly one of {0,1,2,3} is ≤ 0.0 → 1/4.
        y[99].Should().Be(1.0);   // all four are ≤ 3.0.
        // Monotonically non-decreasing (an EDF).
        for (int i = 1; i < y.Length; i++)
            y[i].Should().BeGreaterThanOrEqualTo(y[i - 1]);
    }

    [TestMethod]
    public void Edf_NoCompleteTrials_ReturnsEmpty()
    {
        var study = Optuna.CreateStudy();
        var info = Edf.GetInfo(study);
        info.Lines.Should().BeEmpty();
        info.XValues.Should().BeEmpty();
    }

    [TestMethod]
    public void Edf_NonFiniteValues_AreFilteredOut()
    {
        var study = Optuna.CreateStudy(direction: StudyDirection.Minimize);
        study.AddTrials(new[] { Complete(1.0), Complete(double.PositiveInfinity), Complete(2.0) });

        var info = Edf.GetInfo(study);

        // Only the two finite values contribute → grid spans [1, 2], y[99] == 1.0 over size 2.
        info.XValues[0].Should().Be(1.0);
        info.XValues[99].Should().Be(2.0);
        info.Lines[0].YValues[99].Should().Be(1.0);
    }

    // ------------------------------------------------------------------ numpy helpers

    [TestMethod]
    public void MathUtil_Linspace_MatchesNumpy()
    {
        MathUtil.Linspace(0.0, 1.0, 5).Should().Equal(0.0, 0.25, 0.5, 0.75, 1.0);
        MathUtil.Linspace(2.0, 2.0, 3).Should().Equal(2.0, 2.0, 2.0);
        MathUtil.Linspace(0.0, 1.0, 1).Should().Equal(0.0);
        MathUtil.Linspace(0.0, 1.0, 0).Should().BeEmpty();
    }

    [TestMethod]
    public void MathUtil_MeanAndStd_MatchNumpy()
    {
        MathUtil.Mean(new[] { 2.0, 4.0 }).Should().Be(3.0);
        MathUtil.Std(new[] { 2.0, 4.0 }).Should().Be(1.0);            // population std.
        MathUtil.Std(new[] { 7.0 }).Should().Be(0.0);
        double.IsNaN(MathUtil.Mean(Array.Empty<double>())).Should().BeTrue();
    }

    [TestMethod]
    public void Builders_AreDeterministic()
    {
        var study = Optuna.CreateStudy(direction: StudyDirection.Minimize);
        study.AddTrials(new[] { Complete(3.0), Complete(1.0), Complete(2.0) });

        var a = OptimizationHistory.GetInfoList(study).Single();
        var b = OptimizationHistory.GetInfoList(study).Single();
        a.BestValuesInfo!.Values.Should().Equal(b.BestValuesInfo!.Values);

        Edf.GetInfo(study).XValues.Should().Equal(Edf.GetInfo(study).XValues);
    }

    // ------------------------------------------------------------------ hypervolume history

    [TestMethod]
    public void HypervolumeHistory_IncrementalHypervolume_MatchesHandComputed()
    {
        var study = Optuna.CreateStudy(directions: new[] { StudyDirection.Minimize, StudyDirection.Minimize });
        study.AddTrials(new[]
        {
            FrozenTrial.CreateTrial(values: new[] { 1.0, 3.0 }),
            FrozenTrial.CreateTrial(values: new[] { 3.0, 1.0 }),
            FrozenTrial.CreateTrial(values: new[] { 2.0, 2.0 }),
        });

        var info = HypervolumeHistory.GetInfo(study, new[] { 4.0, 4.0 });

        info.TrialNumbers.Should().Equal(0, 1, 2);
        // Dominated-hypervolume of the growing Pareto front wrt (4,4): 3 → 5 → 6 (hand-computed).
        info.Values.Should().Equal(3.0, 5.0, 6.0);
    }

    [TestMethod]
    public void HypervolumeHistory_InfeasibleTrial_CarriesPreviousHypervolume()
    {
        var study = Optuna.CreateStudy(directions: new[] { StudyDirection.Minimize, StudyDirection.Minimize });
        study.AddTrials(new[]
        {
            FrozenTrial.CreateTrial(values: new[] { 1.0, 3.0 }),
            // Infeasible (constraint > 0) → its hypervolume value repeats the previous one.
            FrozenTrial.CreateTrial(values: new[] { 0.5, 0.5 },
                constraints: new OrderedDict<string, double> { { "c", 1.0 } }),
            FrozenTrial.CreateTrial(values: new[] { 3.0, 1.0 }),
        });

        var info = HypervolumeHistory.GetInfo(study, new[] { 4.0, 4.0 });

        info.TrialNumbers.Should().Equal(0, 1, 2);
        info.Values.Should().Equal(3.0, 3.0, 5.0);  // trial 1 infeasible → repeats 3.
    }

    [TestMethod]
    public void HypervolumeHistory_SingleObjective_Throws()
    {
        var study = Optuna.CreateStudy(direction: StudyDirection.Minimize);
        study.AddTrials(new[] { Complete(1.0) });

        var act = () => HypervolumeHistory.GetInfo(study, new[] { 2.0 });
        act.Should().Throw<ValueError>().WithMessage("Study must be multi-objective.*");
    }

    [TestMethod]
    public void HypervolumeHistory_WrongReferenceDimension_Throws()
    {
        var study = Optuna.CreateStudy(directions: new[] { StudyDirection.Minimize, StudyDirection.Minimize });
        study.AddTrials(new[] { FrozenTrial.CreateTrial(values: new[] { 1.0, 2.0 }) });

        var act = () => HypervolumeHistory.GetInfo(study, new[] { 4.0, 4.0, 4.0 });
        act.Should().Throw<ValueError>()
            .WithMessage("The dimension of the reference point must be the same*");
    }

    // ------------------------------------------------------------------ param importances

    [TestMethod]
    public void ParamImportances_ReversesAndLabels_AgainstImportanceFunctions()
    {
        var study = Optuna.CreateStudy(sampler: new RandomSampler(0));
        study.Optimize(t =>
        {
            double x1 = t.SuggestFloat("x1", -5, 5);
            long x2 = t.SuggestInt("x2", 0, 20);
            return x1 * x1 + x2;
        }, nTrials: 60);

        var raw = ImportanceFunctions.GetParamImportances(study);          // descending
        var info = ParamImportances.GetInfo(study);

        // The info is the reverse of get_param_importances (ascending for a bottom-to-top bar chart).
        info.ParamNames.Should().Equal(raw.Keys.Reverse());
        info.ImportanceValues.Should().Equal(raw.Values.Reverse());

        // Labels: "F2" when >= 0.01, else "<0.01".
        for (int i = 0; i < info.ImportanceValues.Count; i++)
        {
            double v = info.ImportanceValues[i];
            string expected = v >= 0.01 ? v.ToString("F2", System.Globalization.CultureInfo.InvariantCulture) : "<0.01";
            info.ImportanceLabels[i].Should().Be(expected);
        }

        info.TargetName.Should().Be("Objective Value");
    }

    [TestMethod]
    public void ParamImportances_MultiObjective_OneInfoPerObjective()
    {
        var study = Optuna.CreateStudy(
            directions: new[] { StudyDirection.Minimize, StudyDirection.Minimize },
            sampler: new RandomSampler(0));
        study.Optimize((System.Func<Trial, System.Collections.Generic.IReadOnlyList<double>>)(t =>
        {
            double x = t.SuggestFloat("x", -3, 3);
            long y = t.SuggestInt("y", 0, 10);
            return new[] { x * x + y, (x - 1.0) * (x - 1.0) - y };
        }), nTrials: 50);

        var infos = ParamImportances.GetInfos(study);

        infos.Should().HaveCount(2);
        infos[0].TargetName.Should().Be("Objective Value 0");
        infos[1].TargetName.Should().Be("Objective Value 1");
    }

    [TestMethod]
    public void ParamImportances_NoCompleteTrials_ReturnsEmptyInfo()
    {
        var study = Optuna.CreateStudy();
        var info = ParamImportances.GetInfo(study);
        info.ImportanceValues.Should().BeEmpty();
        info.ParamNames.Should().BeEmpty();
        info.TargetName.Should().Be("Objective Value");
    }

    // ------------------------------------------------------------------ slice

    private static Study BuildSliceStudy()
    {
        var study = Optuna.CreateStudy(sampler: new RandomSampler(0));
        study.Optimize(t =>
        {
            double a = t.SuggestFloat("a", -5, 5);
            double b = t.SuggestFloat("b", 1e-3, 1e3, log: true);
            var d = t.SuggestCategorical("d", new CategoricalChoice[] { "x", "y", "z" });
            return a * a + b + (d.Value as string == "x" ? 1.0 : 0.0);
        }, nTrials: 12);
        return study;
    }

    [TestMethod]
    public void Slice_SortsSubplots_AndClassifiesParams()
    {
        var study = BuildSliceStudy();
        var info = SlicePlot.GetPlotInfo(study);

        info.TargetName.Should().Be("Objective Value");
        info.Subplots.Select(s => s.ParamName).Should().Equal("a", "b", "d");

        // 'a': numerical, not log. 'b': numerical, log. 'd': categorical (x_labels present).
        info.Subplots[0].IsNumerical.Should().BeTrue();
        info.Subplots[0].IsLog.Should().BeFalse();
        info.Subplots[0].XLabels.Should().BeNull();
        info.Subplots[1].IsNumerical.Should().BeTrue();
        info.Subplots[1].IsLog.Should().BeTrue();
        info.Subplots[2].IsNumerical.Should().BeFalse();
        info.Subplots[2].XLabels!.Select(c => c.Value).Should().Equal("x", "y", "z");

        // Non-conditional params → every trial contributes; y == the objective value.
        foreach (var sub in info.Subplots)
        {
            sub.X.Should().HaveCount(12);
            sub.Y.Should().HaveCount(12);
            sub.TrialNumbers.Should().Equal(Enumerable.Range(0, 12));
            sub.Constraints.Should().OnlyContain(c => c);  // no constraints → all feasible.
        }
        // The 'a' subplot's x values are exactly the trials' 'a' params.
        var trials = study.GetTrials();
        info.Subplots[0].X.Should().Equal(trials.Select(t => t.Params["a"]));
        info.Subplots[0].Y.Should().Equal(trials.Select(t => t.Value!.Value));
    }

    [TestMethod]
    public void Slice_ParamsSubset_SortsAndValidates()
    {
        var study = BuildSliceStudy();

        SlicePlot.GetPlotInfo(study, @params: new[] { "d", "a" })
            .Subplots.Select(s => s.ParamName).Should().Equal("a", "d");

        var act = () => SlicePlot.GetPlotInfo(study, @params: new[] { "zzz" });
        act.Should().Throw<ValueError>().WithMessage("Parameter zzz does not exist in your study.");
    }

    [TestMethod]
    public void Slice_NoCompleteTrials_ReturnsEmpty()
    {
        var study = Optuna.CreateStudy();
        var info = SlicePlot.GetPlotInfo(study);
        info.Subplots.Should().BeEmpty();
        info.TargetName.Should().Be("Objective Value");
    }

    // ------------------------------------------------------------------ pareto front

    [TestMethod]
    public void ParetoFront_PartitionsBestNonBestInfeasible()
    {
        var study = Optuna.CreateStudy(directions: new[] { StudyDirection.Minimize, StudyDirection.Minimize });
        study.AddTrials(new[]
        {
            FrozenTrial.CreateTrial(values: new[] { 1.0, 3.0 }),  // 0: on front
            FrozenTrial.CreateTrial(values: new[] { 3.0, 1.0 }),  // 1: on front
            FrozenTrial.CreateTrial(values: new[] { 2.0, 2.0 }),  // 2: on front
            FrozenTrial.CreateTrial(values: new[] { 3.0, 3.0 }),  // 3: dominated → non-best
            FrozenTrial.CreateTrial(values: new[] { 0.0, 0.0 },   // 4: infeasible
                constraints: new OrderedDict<string, double> { { "c", 1.0 } }),
        });

        var info = ParetoFront.GetInfo(study);

        info.NTargets.Should().Be(2);
        info.TargetNames.Should().Equal("Objective 0", "Objective 1");
        info.AxisOrder.Should().Equal(0, 1);
        info.IncludeDominatedTrials.Should().BeTrue();
        info.HasConstraints.Should().BeTrue();

        info.BestTrialsWithValues.Select(x => x.Trial.Number).Should().Equal(0, 1, 2);
        info.NonBestTrialsWithValues.Select(x => x.Trial.Number).Should().Equal(3);
        info.InfeasibleTrialsWithValues.Select(x => x.Trial.Number).Should().Equal(4);
        info.BestTrialsWithValues[0].Values.Should().Equal(1.0, 3.0);
    }

    [TestMethod]
    public void ParetoFront_IncludeDominatedFalse_EmptyNonBest()
    {
        var study = Optuna.CreateStudy(directions: new[] { StudyDirection.Minimize, StudyDirection.Minimize });
        study.AddTrials(new[]
        {
            FrozenTrial.CreateTrial(values: new[] { 1.0, 3.0 }),
            FrozenTrial.CreateTrial(values: new[] { 3.0, 3.0 }),  // dominated
        });

        var info = ParetoFront.GetInfo(study, includeDominatedTrials: false);
        info.NonBestTrialsWithValues.Should().BeEmpty();
        info.HasConstraints.Should().BeFalse();
    }

    [TestMethod]
    public void ParetoFront_UnsupportedObjectiveCount_Throws()
    {
        var study = Optuna.CreateStudy(directions: new[]
        {
            StudyDirection.Minimize, StudyDirection.Minimize,
            StudyDirection.Minimize, StudyDirection.Minimize,
        });
        study.AddTrials(new[] { FrozenTrial.CreateTrial(values: new[] { 1.0, 2.0, 3.0, 4.0 }) });

        var act = () => ParetoFront.GetInfo(study);  // targets=null, 4 objectives → ValueError.
        act.Should().Throw<ValueError>().WithMessage("*only supports 2 or 3 objective*");
    }

    // ------------------------------------------------------------------ contour

    private static FrozenTrial ContourTrial(double value, double x, double y)
        => FrozenTrial.CreateTrial(
            value: value,
            @params: new OrderedDict<string, object?> { { "x", x }, { "y", y } },
            distributions: new OrderedDict<string, BaseDistribution>
            {
                { "x", new FloatDistribution(-5, 5) },
                { "y", new FloatDistribution(-5, 5) },
            });

    [TestMethod]
    public void Contour_TwoNumericalParams_BuildsPaddedGrid()
    {
        var study = Optuna.CreateStudy(direction: StudyDirection.Minimize);
        study.AddTrials(new[]
        {
            ContourTrial(10.0, 0.0, 0.0),
            ContourTrial(20.0, 1.0, 1.0),
            ContourTrial(5.0, 0.0, 1.0),
            ContourTrial(8.0, 0.0, 0.0),  // duplicate (x=0,y=0) → minimize keeps min(10, 8) = 8.
        });

        var info = Contour.GetInfo(study);

        info.SortedParams.Should().Equal("x", "y");
        info.ReverseScale.Should().BeTrue();  // minimize → reversed scale.
        info.TargetName.Should().Be("Objective Value");

        var sub = info.SubPlotInfos[0][0];
        sub.XAxis.IsCat.Should().BeFalse();
        sub.XAxis.IsLog.Should().BeFalse();

        // Indices = padded endpoints + the sorted unique values {0, 1}.
        sub.XAxis.Indices.Should().HaveCount(4);
        Convert.ToDouble(sub.XAxis.Indices[0]).Should().BeApproximately(-0.05, 1e-12);
        Convert.ToDouble(sub.XAxis.Indices[1]).Should().Be(0.0);
        Convert.ToDouble(sub.XAxis.Indices[2]).Should().Be(1.0);
        Convert.ToDouble(sub.XAxis.Indices[3]).Should().BeApproximately(1.05, 1e-12);

        // z grid keyed by (x_index, y_index) into the padded indices; dedup keeps the minimum.
        sub.ZValues.Should().HaveCount(3);
        sub.ZValues[(1, 1)].Should().Be(8.0);
        sub.ZValues[(2, 2)].Should().Be(20.0);
        sub.ZValues[(1, 2)].Should().Be(5.0);
        sub.Constraints.Should().OnlyContain(c => c);
    }

    [TestMethod]
    public void Contour_Maximize_NotReverseScale()
    {
        var study = Optuna.CreateStudy(direction: StudyDirection.Maximize);
        study.AddTrials(new[] { ContourTrial(1.0, 0.0, 0.0), ContourTrial(2.0, 1.0, 1.0) });
        Contour.GetInfo(study).ReverseScale.Should().BeFalse();
    }

    [TestMethod]
    public void Contour_UnknownParam_Throws()
    {
        var study = Optuna.CreateStudy(direction: StudyDirection.Minimize);
        study.AddTrials(new[] { ContourTrial(1.0, 0.0, 0.0), ContourTrial(2.0, 1.0, 1.0) });

        var act = () => Contour.GetInfo(study, @params: new[] { "x", "nope" });
        act.Should().Throw<ValueError>().WithMessage("Parameter nope does not exist in your study.");
    }

    [TestMethod]
    public void Contour_ThreeParams_BuildsNbyNGrid()
    {
        var study = Optuna.CreateStudy(direction: StudyDirection.Minimize);
        // Three params via Optimize (deterministic); grid is 3×3 with same-param diagonals empty.
        study.Optimize(t =>
        {
            double a = t.SuggestFloat("a", -5, 5);
            double b = t.SuggestFloat("b", -5, 5);
            double c = t.SuggestFloat("c", -5, 5);
            return a + b + c;
        }, nTrials: 10);

        var info = Contour.GetInfo(study);
        info.SortedParams.Should().Equal("a", "b", "c");
        info.SubPlotInfos.Should().HaveCount(3);
        info.SubPlotInfos.Should().OnlyContain(row => row.Count == 3);
        // Diagonal (same param) subplots carry no z grid.
        info.SubPlotInfos[0][0].ZValues.Should().BeEmpty();
        info.SubPlotInfos[1][0].ZValues.Should().NotBeEmpty();
    }

    // ------------------------------------------------------------------ rank

    private static FrozenTrial RankTrial(double value, double x, double y,
        OrderedDict<string, double>? constraints = null)
        => FrozenTrial.CreateTrial(
            value: value, constraints: constraints,
            @params: new OrderedDict<string, object?> { { "x", x }, { "y", y } },
            distributions: new OrderedDict<string, BaseDistribution>
            {
                { "x", new FloatDistribution(-5, 5) },
                { "y", new FloatDistribution(-5, 5) },
            });

    [TestMethod]
    public void Rank_ColorIdxs_AreScaledAverageRanks()
    {
        var study = Optuna.CreateStudy(direction: StudyDirection.Minimize);
        study.AddTrials(new[] { RankTrial(3.0, 0, 0), RankTrial(1.0, 1, 1), RankTrial(2.0, 0, 1) });

        var info = Rank.GetInfo(study);

        info.Zs.Should().Equal(3.0, 1.0, 2.0);
        // order([3,1,2]) = [2,0,1]; /(3-1) = [1.0, 0.0, 0.5].
        info.ColorIdxs.Should().Equal(1.0, 0.0, 0.5);
        info.HasCustomTarget.Should().BeFalse();
        info.Params.Should().Equal("x", "y");

        // Two params → a single off-diagonal subplot.
        info.SubPlotInfos.Should().HaveCount(1);
        info.SubPlotInfos[0].Should().HaveCount(1);
        var sub = info.SubPlotInfos[0][0];
        sub.XAxis.Name.Should().Be("x");
        sub.YAxis.Name.Should().Be("y");
        sub.Zs.Should().Equal(3.0, 1.0, 2.0);
        sub.ColorIdxs.Should().Equal(1.0, 0.0, 0.5);
        sub.Infeasible.Should().OnlyContain(f => !f);
    }

    [TestMethod]
    public void Rank_TiedValues_ShareAveragedRank()
    {
        var study = Optuna.CreateStudy(direction: StudyDirection.Minimize);
        study.AddTrials(new[] { RankTrial(1.0, 0, 0), RankTrial(1.0, 1, 1), RankTrial(2.0, 0, 1) });

        // sorted [1,1,2]: the two 1's share positions {0,1} → 0.5; the 2 → 2. /(3-1) = [0.25,0.25,1.0].
        Rank.GetInfo(study).ColorIdxs.Should().Equal(0.25, 0.25, 1.0);
    }

    [TestMethod]
    public void Rank_PreservesGivenParamOrder()
    {
        var study = Optuna.CreateStudy(direction: StudyDirection.Minimize);
        study.AddTrials(new[] { RankTrial(1.0, 0, 0), RankTrial(2.0, 1, 1) });

        // Unlike slice/contour, rank keeps the caller's param order (no sort).
        var info = Rank.GetInfo(study, @params: new[] { "y", "x" });
        info.Params.Should().Equal("y", "x");
        info.SubPlotInfos[0][0].XAxis.Name.Should().Be("y");
        info.SubPlotInfos[0][0].YAxis.Name.Should().Be("x");
    }

    [TestMethod]
    public void Rank_InfeasibleTrials_AreFlagged()
    {
        var study = Optuna.CreateStudy(direction: StudyDirection.Minimize);
        study.AddTrials(new[]
        {
            RankTrial(1.0, 0, 0),
            RankTrial(2.0, 1, 1, constraints: new OrderedDict<string, double> { { "c", 1.0 } }),
        });

        Rank.GetInfo(study).SubPlotInfos[0][0].Infeasible.Should().Equal(false, true);
    }

    [TestMethod]
    public void Rank_UnknownParam_Throws()
    {
        var study = Optuna.CreateStudy(direction: StudyDirection.Minimize);
        study.AddTrials(new[] { RankTrial(1.0, 0, 0), RankTrial(2.0, 1, 1) });

        var act = () => Rank.GetInfo(study, @params: new[] { "nope", "x" });
        act.Should().Throw<ValueError>().WithMessage("Parameter nope does not exist in your study.");
    }

    // ------------------------------------------------------------------ parallel coordinate

    [TestMethod]
    public void MathUtil_FormatG3_MatchesPython()
    {
        // C's %g / Python's f"{x:.3g}" — see MathUtil.FormatG3.
        MathUtil.FormatG3(0.0).Should().Be("0");
        MathUtil.FormatG3(1.5).Should().Be("1.5");
        MathUtil.FormatG3(1.23456).Should().Be("1.23");
        MathUtil.FormatG3(100.0).Should().Be("100");
        MathUtil.FormatG3(1234.5).Should().Be("1.23e+03");
        MathUtil.FormatG3(1000.0).Should().Be("1e+03");
        MathUtil.FormatG3(999.9).Should().Be("1e+03");     // rounds up, bumping the exponent.
        MathUtil.FormatG3(0.000123).Should().Be("0.000123");
        MathUtil.FormatG3(9.999e-5).Should().Be("0.0001");  // rounds up to 1e-4 → fixed.
        MathUtil.FormatG3(1e-5).Should().Be("1e-05");
        MathUtil.FormatG3(-3.14159).Should().Be("-3.14");
    }

    [TestMethod]
    public void ParallelCoordinate_SingleObjective_BuildsObjectivePlusParamDims()
    {
        var study = Optuna.CreateStudy(direction: StudyDirection.Minimize, sampler: new RandomSampler(0));
        study.Optimize(t =>
        {
            double a = t.SuggestFloat("a", -5, 5);
            double b = t.SuggestFloat("b", -5, 5);
            return a * a + b * b;
        }, nTrials: 10);

        var info = ParallelCoordinate.GetInfo(study);

        info.ColorbarTitle.Should().Be("Objective Value");
        info.IsRankColor.Should().BeFalse();
        info.ReverseScale.Should().BeTrue();  // minimize.
        info.DimsObjectives.Should().HaveCount(1);
        info.DimsObjectives[0].Label.Should().Be("Objective Value");
        info.DimsParams.Select(d => d.Label).Should().Equal("a", "b");
        info.DrawOrder.Should().Equal(Enumerable.Range(0, 10));  // trivial for single-objective.
        info.Feasibility.Should().BeNull();  // no constraints.
        info.ColorValues.Should().HaveCount(10);
    }

    [TestMethod]
    public void ParallelCoordinate_Constraints_ProduceFeasibility()
    {
        var study = Optuna.CreateStudy(direction: StudyDirection.Minimize, sampler: new RandomSampler(0));
        study.Optimize(t =>
        {
            double a = t.SuggestFloat("a", -5, 5);
            t.SetConstraint("c", a);  // feasible when a <= 0.
            return a * a;
        }, nTrials: 8);

        var info = ParallelCoordinate.GetInfo(study);
        info.Feasibility.Should().NotBeNull();
        info.Feasibility!.Should().HaveCount(8);
    }

    // ------------------------------------------------------------------ timeline

    private static FrozenTrial TimelineComplete(double value, DateTime start, DateTime complete,
        OrderedDict<string, object?>? @params = null, OrderedDict<string, BaseDistribution>? dists = null,
        OrderedDict<string, double>? constraints = null)
    {
        var t = FrozenTrial.CreateTrial(
            state: TrialState.Complete, value: value,
            @params: @params, distributions: dists, constraints: constraints);
        t.SetDatetimeStart(start);
        t.DatetimeComplete = complete;
        return t;
    }

    [TestMethod]
    public void Timeline_CompleteTrials_ProduceBars()
    {
        var s0 = new DateTime(2024, 1, 1, 0, 0, 0);
        var c0 = new DateTime(2024, 1, 1, 0, 0, 1);
        var s1 = new DateTime(2024, 1, 1, 0, 0, 2);
        var c1 = new DateTime(2024, 1, 1, 0, 0, 5);

        var study = Optuna.CreateStudy();
        study.AddTrials(new[]
        {
            TimelineComplete(1.0, s0, c0),
            TimelineComplete(2.0, s1, c1, constraints: new OrderedDict<string, double> { { "c", 1.0 } }),
        });

        var info = Timeline.GetInfo(study);

        info.Bars.Should().HaveCount(2);
        info.Bars[0].Number.Should().Be(0);
        info.Bars[0].State.Should().Be(TrialState.Complete);
        info.Bars[0].Start.Should().Be(s0);
        info.Bars[0].Complete.Should().Be(c0);
        info.Bars[0].Infeasible.Should().BeFalse();
        info.Bars[1].Infeasible.Should().BeTrue();  // constraint 1.0 > 0.
    }

    [TestMethod]
    public void Timeline_Hovertext_IsIndentedJsonWithBrNewlines()
    {
        var study = Optuna.CreateStudy();
        study.AddTrials(new[]
        {
            TimelineComplete(1.5, new DateTime(2024, 1, 1), new DateTime(2024, 1, 1, 0, 0, 1),
                @params: new OrderedDict<string, object?> { { "x", 0.5 } },
                dists: new OrderedDict<string, BaseDistribution> { { "x", new FloatDistribution(0, 1) } }),
        });

        string hovertext = Timeline.GetInfo(study).Bars[0].Hovertext;

        hovertext.Should().Be(
            "{<br>  \"number\": 0,<br>  \"values\": [<br>    1.5<br>  ],<br>" +
            "  \"params\": {<br>    \"x\": 0.5<br>  }<br>}");
        hovertext.Should().NotContain("\n");
    }

    [TestMethod]
    public void Timeline_NRecentTrials_SlicesToLast()
    {
        var study = Optuna.CreateStudy();
        var baseTime = new DateTime(2024, 1, 1);
        study.AddTrials(Enumerable.Range(0, 4)
            .Select(i => TimelineComplete(i, baseTime.AddSeconds(i), baseTime.AddSeconds(i + 1)))
            .ToList());

        var info = Timeline.GetInfo(study, nRecentTrials: 2);
        info.Bars.Select(b => b.Number).Should().Equal(2, 3);
    }

    [TestMethod]
    public void Timeline_NRecentTrials_NonPositive_Throws()
    {
        var study = Optuna.CreateStudy();
        study.AddTrials(new[] { TimelineComplete(1.0, new DateTime(2024, 1, 1), new DateTime(2024, 1, 1, 0, 0, 1)) });

        ((Func<TimelineInfo>)(() => Timeline.GetInfo(study, nRecentTrials: 0)))
            .Should().Throw<ValueError>().WithMessage("n_recent_trials must be a positive integer or None.");
    }

    [TestMethod]
    public void Timeline_RunningTrial_CompleteAfterStart()
    {
        var study = Optuna.CreateStudy();
        var running = FrozenTrial.CreateTrial(state: TrialState.Running);
        running.SetDatetimeStart(new DateTime(2024, 1, 1));
        study.AddTrials(new[] { running });

        var bar = Timeline.GetInfo(study).Bars[0];
        bar.State.Should().Be(TrialState.Running);
        // A running trial's bar ends at max_datetime + 1s (max_datetime is "now" when a trial is running).
        bar.Complete.Should().BeAfter(bar.Start);
    }

    [TestMethod]
    public void Timeline_EmptyStudy_NoBars()
    {
        Timeline.GetInfo(Optuna.CreateStudy()).Bars.Should().BeEmpty();
    }

    [TestMethod]
    public void ParallelCoordinate_MultiObjective_UsesParetoRankColor()
    {
        var study = Optuna.CreateStudy(
            directions: new[] { StudyDirection.Minimize, StudyDirection.Minimize },
            sampler: new RandomSampler(0));
        study.Optimize((System.Func<Trial, System.Collections.Generic.IReadOnlyList<double>>)(t =>
        {
            double x = t.SuggestFloat("x", -3, 3);
            double y = t.SuggestFloat("y", -3, 3);
            return new[] { x * x + y * y, (x - 1) * (x - 1) + (y - 1) * (y - 1) };
        }), nTrials: 20);

        var info = ParallelCoordinate.GetInfo(study);

        info.IsRankColor.Should().BeTrue();
        info.ColorbarTitle.Should().Be("Pareto Rank");
        info.ReverseScale.Should().BeTrue();
        info.DimsObjectives.Select(d => d.Label).Should().Equal("Objective 0", "Objective 1");
        // Pareto rank 0 exists (the front) and color_values are whole numbers.
        info.ColorValues.Should().Contain(0.0);
        info.ColorValues.Should().OnlyContain(v => v == Math.Floor(v));
        // draw_order sorts ranks non-increasingly.
        for (int i = 1; i < info.DrawOrder.Count; i++)
            info.ColorValues[info.DrawOrder[i]].Should().BeLessThanOrEqualTo(info.ColorValues[info.DrawOrder[i - 1]]);
    }
}

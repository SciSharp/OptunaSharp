using System;
using System.Collections.Generic;
using System.Linq;
using AwesomeAssertions;
using NumSharp;
using OptunaSharp.Collections;
using OptunaSharp.Internal;
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
}

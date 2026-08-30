using System;
using System.Collections.Generic;
using System.Linq;
using AwesomeAssertions;
using NumSharp;
using OptunaSharp.Collections;
using OptunaSharp.Distributions;
using OptunaSharp.Internal;
using OptunaSharp.Pruners;
using OptunaSharp.Samplers;
using OptunaSharp.Storages;

namespace OptunaSharp.Tests;

/// <summary>
///     The Trial suggest/report/attr surface (plan 0.5): suggestion caching and compatibility,
///     fixed params (enqueue), single-distribution shortcut, report validation, should_prune via
///     MedianPruner, attrs and constraints.
/// </summary>
[TestClass]
public class TrialTests
{
    private static Study NewStudy(BasePruner? pruner = null, int seed = 0)
        => Optuna.CreateStudy(sampler: new RandomSampler(seed), pruner: pruner);

    [TestMethod]
    public void Suggest_Same_Name_Returns_Cached_Value()
    {
        var study = NewStudy();
        var trial = study.Ask();
        double first = trial.SuggestFloat("x", 0, 1);
        double second = trial.SuggestFloat("x", 0, 1);
        second.Should().Be(first, "already-suggested params are not re-sampled");
        study.Tell(trial, first);
    }

    [TestMethod]
    public void Suggest_Same_Name_Incompatible_Kind_Raises()
    {
        var study = NewStudy();
        var trial = study.Ask();
        trial.SuggestFloat("x", 0, 1);
        var act = () => trial.SuggestInt("x", 0, 5);
        act.Should().Throw<ValueError>()
            .WithMessage("Cannot set different distribution kind to the same parameter name.");
    }

    [TestMethod]
    public void Suggest_Single_Distribution_ShortCircuits_Sampler()
    {
        var study = NewStudy();
        var trial = study.Ask();
        trial.SuggestFloat("s", 3.5, 3.5).Should().Be(3.5);
        trial.SuggestInt("n", 7, 7).Should().Be(7L);
        trial.SuggestCategorical("c", new[] { "only" }).Should().Be("only");
        study.Tell(trial, 0.0);
    }

    [TestMethod]
    public void Suggest_All_Overloads_Return_Typed_Values()
    {
        var study = NewStudy();
        var trial = study.Ask();
        trial.SuggestFloat("f", -5, 5).Should().BeInRange(-5, 5);
        trial.SuggestFloat("fl", 1e-5, 1e-1, log: true).Should().BeInRange(1e-5, 1e-1);
        double fs = trial.SuggestFloat("fs", 0, 1, step: 0.25);
        (fs / 0.25).Should().BeApproximately(Math.Round(fs / 0.25), 1e-9);
        trial.SuggestInt("i", 1, 10).Should().BeInRange(1, 10);
        trial.SuggestInt("il", 1, 64, log: true).Should().BeInRange(1, 64);
        long stepInt = trial.SuggestInt("istep", 0, 20, step: 5);
        (stepInt % 5).Should().Be(0);
        string c = trial.SuggestCategorical("c", new[] { "a", "b" });
        c.Should().BeOneOf("a", "b");
        long ci = trial.SuggestCategorical("ci", new[] { -1, 0, 1 });
        ci.Should().BeInRange(-1, 1);
        bool cb = trial.SuggestCategorical("cb", new[] { true, false });
        var choice = trial.SuggestCategorical("mixed",
            new CategoricalChoice[] { CategoricalChoice.None, 1L, "s" });
        study.Tell(trial, 0.0);

        var stored = study.Trials.Single();
        stored.Params.Keys.Should().Contain(new[] { "f", "fl", "fs", "i", "il", "istep", "c", "ci", "cb", "mixed" });
        stored.Distributions["fl"].Should().Be(new FloatDistribution(1e-5, 1e-1, log: true));
        _ = cb;
        _ = choice;
    }

    [TestMethod]
    public void Fixed_Params_Take_Priority_And_OutOfRange_Warns_But_Applies()
    {
        var study = NewStudy();
        study.EnqueueTrial(new OrderedDict<string, object?> { ["x"] = 25.0 });

        string? warning = null;
        void Capture(string message, WarningCategory category) => warning ??= message;
        OptunaWarnings.WarningRaised += Capture;
        try
        {
            study.Optimize(t => t.SuggestFloat("x", 0, 10), nTrials: 1);
        }
        finally
        {
            OptunaWarnings.WarningRaised -= Capture;
        }

        study.Trials.Single().Params["x"].Should().Be(25.0, "fixed params apply even out of range");
        warning.Should().Be("Fixed parameter x with value 25.0 is out of range for distribution " +
            "FloatDistribution(high=10.0, log=False, low=0.0, step=None).");
    }

    [TestMethod]
    public void Report_Validates_Step_And_Ignores_Duplicates()
    {
        var study = NewStudy();
        var trial = study.Ask();

        var negative = () => trial.Report(1.0, -1);
        negative.Should().Throw<ValueError>().WithMessage("`step=-1` must be non-negative.");

        trial.Report(1.0, 0);
        trial.Report(999.0, 0);  // Ignored with a warning.
        study.Tell(trial, values: null, state: TrialState.Pruned);
        study.Trials.Single().IntermediateValues[0].Should().Be(1.0);
    }

    [TestMethod]
    public void Report_And_ShouldPrune_Reject_MultiObjective()
    {
        var study = Optuna.CreateStudy(
            sampler: new RandomSampler(0),
            directions: new[] { StudyDirection.Minimize, StudyDirection.Minimize });
        var trial = study.Ask();
        var report = () => trial.Report(1.0, 0);
        report.Should().Throw<NotImplementedException>()
            .WithMessage("Trial.report is not supported for multi-objective optimization.");
        var prune = () => trial.ShouldPrune();
        prune.Should().Throw<NotImplementedException>()
            .WithMessage("Trial.should_prune is not supported for multi-objective optimization.");
    }

    [TestMethod]
    public void ShouldPrune_Uses_The_Study_Pruner()
    {
        // MedianPruner scenario pinned against live optuna (n_startup=2, warmup=1):
        // four completed histories, then a fifth trial reporting 6.0/4.0/3.5/2.9 -> all False.
        var study = Optuna.CreateStudy(
            sampler: new RandomSampler(5),
            pruner: new MedianPruner(nStartupTrials: 2, nWarmupSteps: 1, intervalSteps: 1));
        double[][] hist =
        {
            new[] { 10.0, 9, 8, 7 },
            new[] { 5.0, 6, 7, 8 },
            new[] { 3.0, 3, 3, 3 },
            new[] { 8.0, 2, 1, 0 },
        };
        foreach (var values in hist)
        {
            var t = study.Ask();
            t.SuggestFloat("x", 0, 1);
            for (int i = 0; i < values.Length; i++)
                t.Report(values[i], i);
            study.Tell(t, values[^1]);
        }

        var probe = study.Ask();
        probe.SuggestFloat("x", 0, 1);
        var decisions = new List<bool>();
        double[] reports = { 6.0, 4.0, 3.5, 2.9 };
        for (int i = 0; i < reports.Length; i++)
        {
            probe.Report(reports[i], i);
            decisions.Add(probe.ShouldPrune());
        }
        decisions.Should().Equal(false, false, false, false);
        study.Tell(probe, 1.0);
    }

    [TestMethod]
    public void Attrs_And_Constraints_RoundTrip()
    {
        var study = NewStudy();
        var trial = study.Ask();

        trial.SetUserAttr("BATCHSIZE", 128L);
        trial.UserAttrs["BATCHSIZE"].Should().Be(128L);

        trial.SetConstraint("c0", -1.0);
        trial.SetConstraint("c0", 5.0);  // Ignored: already set.
        trial.Constraints["c0"].Should().Be(-1.0);

        var nan = () => trial.SetConstraint("bad", double.NaN);
        nan.Should().Throw<ValueError>()
            .WithMessage("Attempted to set a constraint for 'bad', but NaN is not allowed.");

        study.Tell(trial, 0.0);
        study.Trials.Single().Constraints["c0"].Should().Be(-1.0);
    }

    [TestMethod]
    public void FixedTrial_Suggests_Given_Values_Only()
    {
        var fixedTrial = new FixedTrial(new OrderedDict<string, object?> { ["x"] = 1.0, ["y"] = 0L });
        double Objective(BaseTrial t)
        {
            double x = t.SuggestFloat("x", -100, 100);
            long y = t.SuggestCategorical("y", new[] { -1, 0, 1 });
            return x * x + y;
        }

        Objective(fixedTrial).Should().Be(1.0);
        fixedTrial.Params.Keys.Should().Equal("x", "y");

        var missing = () => fixedTrial.SuggestFloat("z", 0, 1);
        missing.Should().Throw<ValueError>()
            .WithMessage("The value of the parameter 'z' is not found. Please set it at the construction of the FixedTrial object.");
        fixedTrial.ShouldPrune().Should().BeFalse();
    }
}

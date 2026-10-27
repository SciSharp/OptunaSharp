using System;
using System.Collections.Generic;
using AwesomeAssertions;
using NumSharp;
using OptunaSharp.Collections;
using OptunaSharp.Distributions;

namespace OptunaSharp.Tests;

/// <summary>
///     <see cref="FrozenTrial" /> semantics: <c>create_trial</c> validation with verbatim
///     messages, value/values, suggest-from-params, duration/last_step, equality and copying.
/// </summary>
[TestClass]
public class FrozenTrialTests
{
    [TestMethod]
    public void CreateTrial_Complete_Requires_Values()
    {
        var act = () => FrozenTrial.CreateTrial(state: TrialState.Complete);
        act.Should().Throw<ValueError>().WithMessage("values should be set for a complete trial.");

        var nan = () => FrozenTrial.CreateTrial(value: double.NaN);
        nan.Should().Throw<ValueError>().WithMessage("values should not contain NaN.");
    }

    [TestMethod]
    public void CreateTrial_Fail_Rejects_Values()
    {
        var act = () => FrozenTrial.CreateTrial(state: TrialState.Fail, value: 1.0);
        act.Should().Throw<ValueError>()
            .WithMessage("values should be None for a failed trial, but got [1.0].");
    }

    [TestMethod]
    public void CreateTrial_Rejects_Value_And_Values_Together()
    {
        var act = () => FrozenTrial.CreateTrial(value: 1.0, values: new[] { 1.0 });
        act.Should().Throw<ValueError>().WithMessage("Specify only one of `value` and `values`.");
    }

    [TestMethod]
    public void CreateTrial_Validates_Params_Distributions_Consistency()
    {
        var act = () => FrozenTrial.CreateTrial(
            value: 1.0,
            @params: new OrderedDict<string, object?> { ["x"] = 1.0 });
        act.Should().Throw<ValueError>().WithMessage("Inconsistent parameters {'x'} and distributions set().");

        var outOfRange = () => FrozenTrial.CreateTrial(
            value: 1.0,
            @params: new OrderedDict<string, object?> { ["x"] = 99.0 },
            distributions: new OrderedDict<string, BaseDistribution> { ["x"] = new FloatDistribution(0, 10) });
        outOfRange.Should().Throw<ValueError>()
            .WithMessage("The value 99.0 of parameter 'x' isn't contained in the distribution FloatDistribution(high=10.0, log=False, low=0.0, step=None).");
    }

    [TestMethod]
    public void CreateTrial_Waiting_Has_No_Datetimes()
    {
        var waiting = FrozenTrial.CreateTrial(state: TrialState.Waiting);
        waiting.DatetimeStart.Should().BeNull();
        waiting.DatetimeComplete.Should().BeNull();

        var complete = FrozenTrial.CreateTrial(value: 1.0);
        complete.DatetimeStart.Should().NotBeNull();
        complete.DatetimeComplete.Should().Be(complete.DatetimeStart);
    }

    [TestMethod]
    public void CreateTrial_Converts_Deprecated_Distributions()
    {
#pragma warning disable CS0618
        var trial = FrozenTrial.CreateTrial(
            value: 1.0,
            @params: new OrderedDict<string, object?> { ["x"] = 2.0 },
            distributions: new OrderedDict<string, BaseDistribution> { ["x"] = new UniformDistribution(0, 10) });
#pragma warning restore CS0618
        trial.Distributions["x"].Should().BeOfType<FloatDistribution>();
        trial.Distributions["x"].Should().Be(new FloatDistribution(0, 10));
    }

    [TestMethod]
    public void Value_Property_Guards_MultiObjective()
    {
        var trial = FrozenTrial.CreateTrial(values: new[] { 1.0, 2.0 });
        var act = () => trial.Value;
        act.Should().Throw<RuntimeError>()
            .WithMessage("This attribute is not available during multi-objective optimization.");
        trial.Values.Should().Equal(1.0, 2.0);
    }

    [TestMethod]
    public void Suggest_Replays_Stored_Params()
    {
        var trial = FrozenTrial.CreateTrial(
            value: 5.0,
            @params: new OrderedDict<string, object?> { ["x"] = 1.0, ["y"] = 0L },
            distributions: new OrderedDict<string, BaseDistribution>
            {
                ["x"] = new FloatDistribution(0, 10),
                ["y"] = new CategoricalDistribution(new CategoricalChoice[] { -1L, 0L, 1L }),
            });

        // The docs example: objective(study.best_trial) == study.best_value.
        trial.SuggestFloat("x", 0, 10).Should().Be(1.0);
        trial.SuggestCategorical("y", new[] { -1, 0, 1 }).Should().Be(0L);

        var missing = () => trial.SuggestFloat("z", 0, 1);
        missing.Should().Throw<ValueError>()
            .WithMessage("The value of the parameter 'z' is not found. Please set it at the construction of the FrozenTrial object.");
    }

    [TestMethod]
    public void LastStep_And_Duration()
    {
        var trial = FrozenTrial.CreateTrial(
            value: 1.0,
            intermediateValues: new OrderedDict<long, double> { [3] = 1.0, [1] = 2.0 });
        trial.LastStep.Should().Be(3);
        trial.Duration.Should().Be(TimeSpan.Zero);

        FrozenTrial.CreateTrial(value: 1.0).LastStep.Should().BeNull();
    }

    [TestMethod]
    public void Equality_Is_Full_State_Comparison()
    {
        FrozenTrial Make() => new(
            number: 1,
            state: TrialState.Complete,
            value: 2.0,
            datetimeStart: new DateTime(2026, 1, 1),
            datetimeComplete: new DateTime(2026, 1, 2),
            @params: new OrderedDict<string, object?> { ["x"] = 1.0 },
            distributions: new OrderedDict<string, BaseDistribution> { ["x"] = new FloatDistribution(0, 10) },
            userAttrs: new OrderedDict<string, object?> { ["u"] = new List<object?> { 1L } },
            systemAttrs: new OrderedDict<string, object?>(),
            intermediateValues: new OrderedDict<long, double> { [0] = 1.0 },
            trialId: 7);

        Make().Should().Be(Make());
        var different = Make();
        different.Value = 3.0;
        different.Should().NotBe(Make());
    }

    [TestMethod]
    public void DeepCopy_Isolates_Mutable_State()
    {
        var original = FrozenTrial.CreateTrial(
            value: 1.0,
            userAttrs: new OrderedDict<string, object?> { ["list"] = new List<object?> { 1L } });
        var copy = original.DeepCopy();
        ((List<object?>)copy.UserAttrsDict["list"]!).Add(2L);
        copy.IntermediateValues[9] = 1.0;

        ((List<object?>)original.UserAttrsDict["list"]!).Should().HaveCount(1);
        original.IntermediateValues.Should().BeEmpty();
        copy.Should().NotBe(original);
    }
}

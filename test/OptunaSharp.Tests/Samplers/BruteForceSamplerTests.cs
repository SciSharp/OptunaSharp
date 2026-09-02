using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using AwesomeAssertions;
using OptunaSharp.Collections;
using OptunaSharp.Distributions;
using OptunaSharp.Samplers;
using OptunaSharp.Storages;

namespace OptunaSharp.Tests.Samplers;

/// <summary>
///     BruteForceSampler parity against the vendored Optuna reference (v5.0.0.dev, <c>f6898da</c>).
///     These mirror Optuna's own <c>tests/samplers_tests/test_brute_force.py</c>: coverage tests
///     assert the exhaustive-search set (the sampler self-stops at exhaustion), and the two
///     <c>seed=42</c> ask/tell cases pin the exact RNG-driven suggestions hardcoded in that test —
///     the bit-exact proof of the <c>rng.choice(p=weights)</c> port (NumPy pairwise-sum weight
///     normalization + <c>searchsorted(side='right')</c> over the <see cref="MT19937" /> stream).
///
///     <para>The install-time pip Optuna (4.6.0) predates this rewrite (older non-blended
///     <c>sample_child</c>, no NaN-subset fix), so these are validated here as unit tests against the
///     pinned reference rather than through the live-interop suite.</para>
/// </summary>
[TestClass]
public class BruteForceSamplerTests
{
    /// <summary>Port of the test module's <c>conditional_objective</c>.</summary>
    private static double ConditionalObjective(Trial trial, bool prune = false)
    {
        long a = trial.SuggestInt("a", 0, 2);
        if (a == 0)
        {
            double b = trial.SuggestFloat("b", -1.0, 1.0, step: 0.5);
            if (prune)
                throw new TrialPruned();
            return a + b;
        }

        if (a == 1)
        {
            var c = trial.SuggestCategorical("c", new CategoricalChoice[] { "x", "y", CategoricalChoice.None });
            return (c.Value as string) == "x" ? a + 1 : a - 1;
        }

        return a * 2;
    }

    // The nine complete `params` of the conditional search space (Optuna's expected_suggested_values).
    private static readonly HashSet<string> ExpectedConditional = new()
    {
        "a=0,b=-1.0", "a=0,b=-0.5", "a=0,b=0.0", "a=0,b=0.5", "a=0,b=1.0",
        "a=1,c=x", "a=1,c=y", "a=1,c=None", "a=2",
    };

    private static string ParamsKey(FrozenTrial trial)
        => string.Join(",", trial.Params.Select(kv => $"{kv.Key}={FormatValue(kv.Value)}"));

    private static string FormatValue(object? value) => value switch
    {
        null => "None",
        double d => d.ToString("0.0###############", CultureInfo.InvariantCulture),
        _ => Convert.ToString(value, CultureInfo.InvariantCulture)!,
    };

    private static void AssertCoversConditionalSpace(Study study)
    {
        var suggested = study.Trials.Select(ParamsKey).ToList();
        suggested.Should().HaveCount(ExpectedConditional.Count);
        suggested.Should().BeEquivalentTo(ExpectedConditional);
    }

    private static Study NewStudy(BruteForceSampler sampler, string name)
        => Optuna.CreateStudy(storage: new InMemoryStorage(), sampler: sampler, studyName: name);

    [TestMethod]
    public void StudyOptimize_WithSingleSearchSpace_CoversEntireSpace()
    {
        var study = NewStudy(new BruteForceSampler(), "single");
        study.Optimize(t => ConditionalObjective(t));
        AssertCoversConditionalSpace(study);
    }

    [TestMethod]
    public void StudyOptimize_WithPrunedTrials_CoversEntireSpace()
    {
        var study = NewStudy(new BruteForceSampler(), "pruned");
        study.Optimize(t => ConditionalObjective(t, prune: true));
        AssertCoversConditionalSpace(study);
    }

    [TestMethod]
    public void StudyOptimize_WithInfiniteSearchSpace_Throws()
    {
        var study = NewStudy(new BruteForceSampler(), "inf");
        var act = () => study.Optimize(t => t.SuggestFloat("a", 0, 2), nTrials: 1);
        act.Should().Throw<NumSharp.ValueError>()
            .WithMessage("FloatDistribution.step must be given for BruteForceSampler*");
    }

    [TestMethod]
    public void StudyOptimize_WithNan_TriesBothChoices()
    {
        var study = NewStudy(new BruteForceSampler(), "nan");
        study.Optimize(t =>
        {
            t.SuggestCategorical("a", new CategoricalChoice[] { 0.0, double.NaN });
            return 1.0;
        });

        var values = study.Trials.Select(t => (double)t.Params["a"]!).ToList();
        values.Should().HaveCount(2);
        values.Should().Contain(v => v == 0.0);
        values.Should().Contain(v => double.IsNaN(v));
    }

    [TestMethod]
    public void StudyOptimize_WithUserAddedTrial_DoesNotRetryIt()
    {
        var study = NewStudy(new BruteForceSampler(), "added");
        // Manually add a trial that is one of the nine grid points. It must not be tried again.
        study.AddTrial(FrozenTrial.CreateTrial(
            value: 0.0,
            @params: new OrderedDict<string, object?> { ["a"] = 0L, ["b"] = -1.0 },
            distributions: new OrderedDict<string, BaseDistribution>
            {
                ["a"] = new IntDistribution(0, 2),
                ["b"] = new FloatDistribution(-1.0, 1.0, step: 0.5),
            }));

        study.Optimize(t => ConditionalObjective(t));
        AssertCoversConditionalSpace(study);
    }

    [TestMethod]
    public void StudyOptimize_WithDynamicRangeSearchSpace_Throws()
    {
        var study = NewStudy(new BruteForceSampler(), "dyn");
        var act = () => study.Optimize(t => t.SuggestInt("x", -1, t.Number), nTrials: 10);
        act.Should().Throw<NumSharp.ValueError>();
    }

    [TestMethod]
    public void StudyOptimize_WithIncreasingSearchSpace_Throws()
    {
        var study = NewStudy(new BruteForceSampler(), "grow");
        study.AddTrial(FrozenTrial.CreateTrial(
            value: 0.0,
            @params: new OrderedDict<string, object?> { ["x0"] = 0L },
            distributions: new OrderedDict<string, BaseDistribution> { ["x0"] = new IntDistribution(0, 0) }));

        var act = () => study.Optimize(t => (double)(t.SuggestInt("x0", 0, 0) + t.SuggestInt("x1", 0, 0)), nTrials: 10);
        act.Should().Throw<NumSharp.ValueError>();
    }

    [TestMethod]
    public void StudyOptimize_WithDecreasingSearchSpace_Throws()
    {
        var study = NewStudy(new BruteForceSampler(), "shrink");
        study.AddTrial(FrozenTrial.CreateTrial(
            value: 0.0,
            @params: new OrderedDict<string, object?> { ["x0"] = 0L, ["x1"] = 0L },
            distributions: new OrderedDict<string, BaseDistribution>
            {
                ["x0"] = new IntDistribution(0, 0),
                ["x1"] = new IntDistribution(0, 0),
            }));

        var act = () => study.Optimize(t => (double)t.SuggestInt("x0", 0, 0), nTrials: 10);
        act.Should().Throw<NumSharp.ValueError>();
    }

    [TestMethod]
    public void StudyOptimize_WithFailedTrials_CoversEntireSpace()
    {
        var study = NewStudy(new BruteForceSampler(), "fail");
        study.Optimize(t =>
        {
            t.SuggestInt("x", 0, 99);
            return double.NaN;
        }, nTrials: 100);

        study.Trials.Should().HaveCount(100);
        study.Trials.Select(t => (long)t.Params["x"]!).OrderBy(v => v)
            .Should().Equal(Enumerable.Range(0, 100).Select(i => (long)i));
    }

    [TestMethod]
    public void ObjectiveWithWeirdChoices_TriesTheWholeProduct()
    {
        // [inf, -inf, nan, None] over three params — 4**3 = 64 unique combinations.
        var weird = new CategoricalChoice[]
            { double.PositiveInfinity, double.NegativeInfinity, double.NaN, CategoricalChoice.None };
        var study = NewStudy(new BruteForceSampler(seed: 0), "weird");
        study.Optimize(t =>
        {
            for (int i = 0; i < 3; i++)
                t.SuggestCategorical($"c{i}", weird);
            return 0.0;
        });

        study.Trials.Should().HaveCount(weird.Length * weird.Length * weird.Length);
    }

    [TestMethod]
    public void ParallelOptimize_AsksDistinctChoices()
    {
        var study = NewStudy(new BruteForceSampler(), "par");
        var trial1 = study.Ask();
        var trial2 = study.Ask();
        var x1 = trial1.SuggestCategorical("x", new CategoricalChoice[] { "a", "b" });
        var x2 = trial2.SuggestCategorical("x", new CategoricalChoice[] { "a", "b" });

        new HashSet<string?> { x1.Value as string, x2.Value as string }
            .Should().BeEquivalentTo(new HashSet<string?> { "a", "b" });
    }

    [TestMethod]
    public void NotAvoidPrematureStop_Seed42_IsBitExact()
    {
        // Hardcoded from Optuna's test_not_avoid_premature_stop (the reference's own assertions).
        var study = NewStudy(new BruteForceSampler(seed: 42), "nps");
        var trials = new[] { study.Ask(), study.Ask(), study.Ask() };

        trials[0].SuggestInt("x", 0, 1).Should().Be(0);
        trials[0].SuggestInt("y", 0, 1).Should().Be(1);
        study.Tell(trials[0], 0.0);
        trials[1].SuggestInt("x", 0, 1).Should().Be(1);
        trials[2].SuggestInt("x", 0, 1).Should().Be(0);
        trials[2].SuggestInt("y", 0, 1).Should().Be(0);

        // trials[0]: x=0,y=1 (complete); trials[1]: x=1; trials[2]: x=0,y=0 (both running). The
        // sampler assumes running trials already suggest all parameters, so it considers all
        // combinations of x and y exhausted and stops for both remaining tells.
        int stopCalls = 0;
        var counting = new StopCountingStudy(study, () => stopCalls++);
        counting.Tell(trials[1], 0.0);
        counting.Tell(trials[2], 0.0);
        stopCalls.Should().Be(2);
    }

    [TestMethod]
    public void AvoidPrematureStop_Seed42_IsBitExact()
    {
        // Hardcoded from Optuna's test_avoid_premature_stop.
        var study = NewStudy(new BruteForceSampler(seed: 42, avoidPrematureStop: true), "aps");
        var trials = Enumerable.Range(0, 5).Select(_ => study.Ask()).ToArray();

        trials[0].SuggestInt("x", 0, 1).Should().Be(0);
        trials[0].SuggestInt("y", 0, 1).Should().Be(1);
        study.Tell(trials[0], 0.0);
        trials[1].SuggestInt("x", 0, 1).Should().Be(1);
        trials[2].SuggestInt("x", 0, 1).Should().Be(0);
        trials[2].SuggestInt("y", 0, 1).Should().Be(0);

        // With avoid_premature_stop, running trials may still suggest new params, so telling
        // trials[2] does not stop (trials[1] could still suggest y).
        int stopAfterT2 = 0;
        new StopCountingStudy(study, () => stopAfterT2++).Tell(trials[2], 0.0);
        stopAfterT2.Should().Be(0);

        trials[1].SuggestInt("y", 0, 1).Should().Be(0);
        trials[3].SuggestInt("x", 0, 1).Should().Be(1);
        trials[3].SuggestInt("y", 0, 1).Should().Be(1);

        // trials[3] (x=1,y=1) is still running and might have another parameter, so telling
        // trials[1] does not stop yet either.
        int stopAfterT1 = 0;
        new StopCountingStudy(study, () => stopAfterT1++).Tell(trials[1], 0.0);
        stopAfterT1.Should().Be(0);

        // Now all of {x,y} in {0,1}^2 are complete except trials[3], whose tell exhausts the space.
        int stopAfterT3 = 0;
        new StopCountingStudy(study, () => stopAfterT3++).Tell(trials[3], 0.0);
        stopAfterT3.Should().Be(1);
    }

    [TestMethod]
    public void SampleIndependent_DoesNotCrash_OnExhaustedSubspace()
    {
        var study = NewStudy(new BruteForceSampler(seed: 0), "exhausted");
        study.EnqueueTrial(new OrderedDict<string, object?> { ["x"] = 0L, ["y"] = 0L });
        study.EnqueueTrial(new OrderedDict<string, object?> { ["x"] = 0L, ["y"] = 1L });
        // The subtree of x=0 is exhausted, but no error should happen even if x=0 shows up again.
        study.EnqueueTrial(new OrderedDict<string, object?> { ["x"] = 0L });
        var act = () => study.Optimize(t => (double)(t.SuggestInt("x", 0, 1) + t.SuggestInt("y", 0, 1)));
        act.Should().NotThrow();
    }

    [DataTestMethod]
    [DataRow(1.0, 3.0, 0.5, false)]
    [DataRow(0.0, 10.0, 1.0, true)]
    [DataRow(0.0, 10.0, 3.0, true)]
    public void EnumerateCandidates_AreSortedAndUniform(double low, double high, double step, bool isInt)
    {
        var candidates = BruteForceSampler.EnumerateCandidates(
            new BruteForceSampler.ChoicesArgs(low, high, step, isInt));

        candidates.Should().BeInAscendingOrder();
        if (candidates.Count >= 2)
        {
            var diffs = Enumerable.Range(0, candidates.Count - 1).Select(i => candidates[i + 1] - candidates[i]).ToList();
            diffs.Should().OnlyContain(d => Math.Abs(d - diffs[0]) < 1e-12);
        }
    }

    [TestMethod]
    public void EnumerateCandidates_WithNullStep_Throws()
    {
        var act = () => BruteForceSampler.EnumerateCandidates(new BruteForceSampler.ChoicesArgs(1.0, 3.0, null, false));
        act.Should().Throw<NumSharp.ValueError>()
            .WithMessage("FloatDistribution.step must be given for BruteForceSampler*");
    }

    [TestMethod]
    public void NonDivisibleStep_ThatFailsToFallbackToDivisibleRange_Throws()
    {
        var study = NewStudy(new BruteForceSampler(), "nd-fail");
        study.Ask(new OrderedDict<string, BaseDistribution> { ["x"] = new FloatDistribution(0.0, 0.3, step: 0.1) });
        var act = () => study.Ask(new OrderedDict<string, BaseDistribution>
            { ["x"] = new FloatDistribution(0.0, 0.3, step: 0.1 - 1e-17) });
        act.Should().Throw<NumSharp.ValueError>();
    }

    [TestMethod]
    public void NonDivisibleStep_WithSuccessfulFallback_DoesNotThrow()
    {
        var study = NewStudy(new BruteForceSampler(), "nd-ok");
        var act = () => study.Ask(new OrderedDict<string, BaseDistribution>
            { ["x"] = new FloatDistribution(0.0, 0.5, step: 0.2) });
        act.Should().NotThrow();
    }

    /// <summary>
    ///     A <see cref="Study" /> proxy that counts <see cref="Study.Stop" /> calls (Optuna's tests
    ///     use <c>patch.object(study, "stop")</c>). It shares the underlying storage/sampler so the
    ///     sampler observes the same trials, letting us assert the exact stop decisions.
    /// </summary>
    private sealed class StopCountingStudy : Study
    {
        private readonly Action _onStop;

        internal StopCountingStudy(Study inner, Action onStop)
            : base(inner.StudyName, inner.Storage, inner.Sampler, inner.Pruner)
        {
            _onStop = onStop;
        }

        public override void Stop() => _onStop();
    }
}

using System;
using System.Collections.Generic;
using System.Linq;
using AwesomeAssertions;
using NumSharp;
using OptunaSharp.Collections;
using OptunaSharp.Distributions;
using OptunaSharp.Internal;
using OptunaSharp.Samplers;
using OptunaSharp.Samplers.Nsgaii;
using OptunaSharp.Samplers.NsgaiiiCpt;
using OptunaSharp.SearchSpace;

namespace OptunaSharp.Tests;

/// <summary>
///     Coverage for <see cref="ConditionalParams" /> — the OptunaSharp-only activation rules that let the
///     GA samplers evolve define-by-run conditional parameters instead of dropping them from the
///     intersection search space. There is no Optuna analog, hence no parity oracle: the tests pin the
///     rule algebra, the widened search space, the child-generation behavior (inheritance vs. random
///     resampling) and — most important — that with rules that never fire the sampler still reproduces
///     Optuna's sequence bit-for-bit.
/// </summary>
[TestClass]
public class ConditionalParamsTests
{
    private static readonly StudyDirection[] TwoObjectives = { StudyDirection.Minimize, StudyDirection.Minimize };

    private static ConditionalParams OptimizerRules() => new ConditionalParams()
        .When("optimizer", "adam", "adamw").Activates("beta1", "beta2")
        .When("optimizer", "sgd").Activates("momentum");

    /// <summary>
    ///     A two-objective surrogate shaped like an optimizer search: `beta1`/`beta2` exist under adam and
    ///     adamw, `momentum` under sgd. Plain (non-log, step-less) ranges so uniform crossover round-trips
    ///     every value through the transform bit-for-bit.
    /// </summary>
    private static IReadOnlyList<double> OptimizerObjective(Trial t)
    {
        string optimizer = t.SuggestCategorical("optimizer", new[] { "adam", "adamw", "sgd" });
        double lr = t.SuggestFloat("lr", 0.0, 1.0);
        double penalty;
        if (optimizer is "adam" or "adamw")
        {
            double beta1 = t.SuggestFloat("beta1", 0.5, 0.999);
            double beta2 = t.SuggestFloat("beta2", 0.9, 0.9999);
            penalty = Math.Pow(beta1 - 0.9, 2) + Math.Pow(beta2 - 0.99, 2);
        }
        else
        {
            double momentum = t.SuggestFloat("momentum", 0.0, 0.99);
            penalty = Math.Pow(momentum - 0.9, 2);
        }
        return new[] { Math.Pow(lr - 0.3, 2) + penalty, optimizer == "sgd" ? lr : 1.0 - lr };
    }

    // ---- rule algebra --------------------------------------------------------------------------

    [TestMethod]
    public void When_Activates_DeclaresConditionalParameters()
    {
        var rules = OptimizerRules();
        rules.Parameters.Should().Equal("beta1", "beta2", "momentum");
        rules.IsConditional("beta1").Should().BeTrue();
        rules.IsConditional("optimizer").Should().BeFalse();
        rules.ActivatorsOf("momentum").Should().Equal("optimizer");
        rules.ActivatorsOf("optimizer").Should().BeEmpty();
        rules.ToString().Should().Be(
            "ConditionalParams(`beta1` when `optimizer` in {'adam', 'adamw'}; " +
            "`beta2` when `optimizer` in {'adam', 'adamw'}; `momentum` when `optimizer` in {'sgd'})");
    }

    [TestMethod]
    public void IsActive_SameActivatorOrs_DifferentActivatorsAnd()
    {
        var rules = new ConditionalParams()
            .When("a", 1).Activates("x")
            .When("a", 2).Activates("x")        // OR with the rule above (same activator)
            .When("b", true).Activates("x");    // AND with `a` (different activator)

        rules.IsActive("x", P(("a", 1L), ("b", true))).Should().BeTrue();
        rules.IsActive("x", P(("a", 2L), ("b", true))).Should().BeTrue();
        rules.IsActive("x", P(("a", 3L), ("b", true))).Should().BeFalse();
        rules.IsActive("x", P(("a", 1L), ("b", false))).Should().BeFalse();
        rules.IsActive("x", P(("a", 1L))).Should().BeFalse("a missing activator means inactive");
        rules.IsActive("a", P()).Should().BeTrue("unconditional parameters are always active");
    }

    [TestMethod]
    public void IsActive_Chain_RequiresEveryLink()
    {
        var rules = new ConditionalParams()
            .When("a", "on").Activates("b")
            .When("b", "deep").Activates("c");

        rules.IsActive("c", P(("a", "on"), ("b", "deep"))).Should().BeTrue();
        rules.IsActive("c", P(("a", "off"), ("b", "deep"))).Should().BeFalse("b is present but itself inactive");
        rules.IsActive("c", P(("a", "on"), ("b", "shallow"))).Should().BeFalse();
    }

    [TestMethod]
    public void When_Values_UsePythonEquality()
    {
        // Int parameters arrive boxed as long, categoricals as their raw choice value. Python `==`:
        // 2 == 2L == 2.0, True == 1, strings ordinal, None == null.
        new ConditionalParams().When("n", 2).Activates("x").IsActive("x", P(("n", 2L))).Should().BeTrue();
        new ConditionalParams().When("n", 2.0).Activates("x").IsActive("x", P(("n", 2L))).Should().BeTrue();
        new ConditionalParams().When("f", true).Activates("x").IsActive("x", P(("f", 1L))).Should().BeTrue();
        new ConditionalParams().When("s", "Adam").Activates("x").IsActive("x", P(("s", "adam"))).Should().BeFalse();
        // A bare `null` literal is ambiguous with the predicate overload, so None is spelled `(object?)null`.
        new ConditionalParams().When("s", (object?)null).Activates("x").IsActive("x", P(("s", null))).Should().BeTrue();
        new ConditionalParams().When("s", (object?)null).Activates("x").IsActive("x", P(("s", "adam"))).Should().BeFalse();

        var byPredicate = new ConditionalParams().When("n", v => (long)v! >= 2).Activates("x");
        byPredicate.IsActive("x", P(("n", 3L))).Should().BeTrue();
        byPredicate.IsActive("x", P(("n", 1L))).Should().BeFalse();
    }

    [TestMethod]
    public void Declarations_RejectSelfActivation_Cycles_AndEmptyRules()
    {
        Action self = () => new ConditionalParams().When("a", 1).Activates("a");
        self.Should().Throw<ValueError>().WithMessage("*cannot activate itself*");

        var chain = new ConditionalParams().When("a", 1).Activates("b").When("b", 1).Activates("c");
        Action cycle = () => chain.When("c", 1).Activates("a");
        cycle.Should().Throw<ValueError>().WithMessage("*cycle*");

        Action noValues = () => new ConditionalParams().When("a");
        noValues.Should().Throw<ValueError>();
        Action noParameters = () => new ConditionalParams().When("a", 1).Activates();
        noParameters.Should().Throw<ValueError>();
    }

    // ---- pruning of a child individual ---------------------------------------------------------

    [TestMethod]
    public void Prune_RemovesInactiveDependents_InDependencyOrder()
    {
        var rules = new ConditionalParams()
            .When("a", "on").Activates("b")
            .When("b", "deep").Activates("c");
        var space = new OrderedDict<string, BaseDistribution>
        {
            ["a"] = new CategoricalDistribution("on", "off"),
            ["b"] = new CategoricalDistribution("deep", "shallow"),
            ["c"] = new FloatDistribution(0, 1),
        };

        // a=off: b goes (inactive), and c goes with it although b's own value would have admitted it.
        var child = Child(("a", "off"), ("b", "deep"), ("c", 0.5));
        rules.Prune(child, space);
        child.Keys.Should().Equal("a");

        child = Child(("a", "on"), ("b", "deep"), ("c", 0.5));
        rules.Prune(child, space);
        child.Keys.Should().Equal("a", "b", "c");
    }

    [TestMethod]
    public void Prune_MissingActivator_InactiveWhenManaged_UnknownWhenNot()
    {
        var rules = new ConditionalParams().When("a", "on").Activates("b");

        var spaceWithA = new OrderedDict<string, BaseDistribution>
        {
            ["a"] = new CategoricalDistribution("on", "off"),
            ["b"] = new FloatDistribution(0, 1),
        };
        var child = Child(("b", 0.5));
        rules.Prune(child, spaceWithA);
        child.Should().BeEmpty("`a` is relatively managed but absent, so `b` cannot be justified");

        var spaceWithoutA = new OrderedDict<string, BaseDistribution> { ["b"] = new FloatDistribution(0, 1) };
        child = Child(("b", 0.5));
        rules.Prune(child, spaceWithoutA);
        child.Keys.Should().Equal(new[] { "b" }, "`a` is not relatively managed, so the objective's suggest calls decide");
    }

    // ---- search space --------------------------------------------------------------------------

    [TestMethod]
    public void SearchSpace_ConditionalParameters_SurviveAbsence()
    {
        var study = Optuna.CreateStudy(sampler: new RandomSampler(seed: 0), directions: TwoObjectives);
        study.Optimize(OptimizerObjective, nTrials: 30);
        var trials = study.Trials;
        trials.Should().Contain(t => t.Params.ContainsKey("beta1"));
        trials.Should().Contain(t => t.Params.ContainsKey("momentum"));

        // Optuna's intersection: only what every trial has.
        SearchSpaceFunctions.IntersectionSearchSpaceOf(trials).Keys.Should().Equal("lr", "optimizer");
        // With rules: the conditional parameters stay, sorted by name like the intersection.
        NsgaiiiCptSearchSpace.IntersectionSearchSpaceOf(trials, OptimizerRules()).Keys
            .Should().Equal("beta1", "beta2", "lr", "momentum", "optimizer");

        // The incremental (cached) study-level fold agrees and raises no contradiction warning, since the
        // objective's branches match the rules exactly.
        var space = new NsgaiiiCptSearchSpace(OptimizerRules());
        var warnings = CaptureWarnings(() =>
        {
            space.Calculate(study).Keys.Should().Equal("beta1", "beta2", "lr", "momentum", "optimizer");
            study.Optimize(OptimizerObjective, nTrials: 5);
            space.Calculate(study).Keys.Should().Equal("beta1", "beta2", "lr", "momentum", "optimizer");
        });
        warnings.Should().BeEmpty();
    }

    [TestMethod]
    public void SearchSpace_ConditionalParameterWithChangingDistribution_IsExcluded()
    {
        var study = Optuna.CreateStudy(sampler: new RandomSampler(seed: 0), direction: StudyDirection.Minimize);
        study.Optimize(t =>
        {
            t.SuggestCategorical("optimizer", new[] { "adam" });
            // Dynamic range: the distribution differs between trials, so the parameter is not evolvable.
            return t.Number % 2 == 0 ? t.SuggestFloat("beta1", 0.5, 0.999) : t.SuggestFloat("beta1", 0.0, 0.999);
        }, nTrials: 6);

        var rules = new ConditionalParams().When("optimizer", "adam").Activates("beta1");
        NsgaiiiCptSearchSpace.IntersectionSearchSpaceOf(study.Trials, rules).Keys.Should().Equal("optimizer");
    }

    [TestMethod]
    public void SearchSpace_ContradictingTrials_WarnOncePerParameter()
    {
        var rules = new ConditionalParams().When("optimizer", "adam").Activates("beta1");
        var space = new NsgaiiiCptSearchSpace(rules);
        var study = Optuna.CreateStudy(sampler: new RandomSampler(seed: 0), direction: StudyDirection.Minimize);
        study.Optimize(t =>
        {
            t.SuggestCategorical("optimizer", new[] { "adam", "sgd" });
            return t.SuggestFloat("beta1", 0, 1);  // suggested regardless of `optimizer`: contradicts the rule under sgd
        }, nTrials: 12);
        study.Trials.Should().Contain(t => (string)t.Params["optimizer"]! == "sgd");

        var warnings = CaptureWarnings(() =>
        {
            space.Calculate(study);
            space.Calculate(study);  // cached: nothing is re-judged
        });
        warnings.Should().ContainSingle().Which.Should().Contain("`beta1`").And.Contain("present");
    }

    // ---- the samplers --------------------------------------------------------------------------

    [TestMethod]
    public void Nsgaiii_ConditionalParams_InheritsConditionalValues_FromParents()
    {
        var rules = OptimizerRules();
        var study = Optuna.CreateStudy(
            sampler: new NSGAIIICptSampler(populationSize: 8, seed: 1, mutationProb: 0.0, conditionalParams: rules),
            directions: TwoObjectives);
        study.Optimize(OptimizerObjective, nTrials: 56);
        var trials = study.Trials;
        AssertRuleConsistent(trials, rules);

        // With mutation off, uniform crossover copies parent values bit-for-bit and the no-crossover branch
        // copies a whole parent, so every conditional value after generation 0 must already exist in an
        // earlier trial. Under the plain intersection (control test below) they are random draws instead.
        int inherited = 0;
        foreach (var name in rules.Parameters)
        {
            foreach (var t in trials.Where(t => Generation(t, "NSGAIIICptSampler") >= 1 && t.Params.ContainsKey(name)))
            {
                long bits = BitConverter.DoubleToInt64Bits((double)t.Params[name]!);
                trials.Where(e => e.Number < t.Number && e.Params.ContainsKey(name))
                    .Select(e => BitConverter.DoubleToInt64Bits((double)e.Params[name]!))
                    .Should().Contain(bits, $"`{name}` of trial {t.Number} should come from a parent");
                inherited++;
            }
        }
        inherited.Should().BeGreaterThan(20);
    }

    [TestMethod]
    public void Nsgaiii_WithoutConditionalParams_ResamplesConditionalValues()
    {
        var study = Optuna.CreateStudy(
            sampler: new NSGAIIISampler(populationSize: 8, seed: 1, mutationProb: 0.0),
            directions: TwoObjectives);
        study.Optimize(OptimizerObjective, nTrials: 56);
        var trials = study.Trials;

        int fresh = trials
            .Where(t => Generation(t, "NSGAIIISampler") >= 1 && t.Params.ContainsKey("beta1"))
            .Count(t => !trials.Any(e => e.Number < t.Number && e.Params.ContainsKey("beta1")
                                         && BitConverter.DoubleToInt64Bits((double)e.Params["beta1"]!)
                                            == BitConverter.DoubleToInt64Bits((double)t.Params["beta1"]!)));
        fresh.Should().BeGreaterThan(0, "the intersection space drops `beta1`, so it is drawn at random every trial");
    }

    [TestMethod]
    public void NsgaiiiCpt_ConditionalParams_ArithmeticCrossover_NumericalActivator_Runs()
    {
        var rules = new ConditionalParams().When("n_layers", 3).Activates("width_3");
        var study = Optuna.CreateStudy(
            sampler: new NSGAIIICptSampler(
                populationSize: 6, seed: 3, crossover: new SBXCrossover(), mutation: new PolynomialMutation(),
                conditionalParams: rules),
            directions: TwoObjectives);
        study.Optimize(t =>
        {
            long nLayers = t.SuggestInt("n_layers", 1, 3);
            double width = t.SuggestFloat("width", 8, 64);
            double width3 = nLayers == 3 ? t.SuggestFloat("width_3", 8, 64) : 0.0;
            return (IReadOnlyList<double>)new[] { Math.Pow(width - 32, 2) + Math.Pow(width3 - 16, 2), (double)nLayers };
        }, nTrials: 36);

        AssertRuleConsistent(study.Trials, rules);
        study.Trials.Should().Contain(t => Generation(t, "NSGAIIICptSampler") >= 1 && t.Params.ContainsKey("width_3"));
    }

    [TestMethod]
    public void Nsgaiii_ConditionalParams_ThreeParentCrossover_Runs()
    {
        var rules = OptimizerRules();
        var study = Optuna.CreateStudy(
            sampler: new NSGAIIICptSampler(populationSize: 6, seed: 5, crossover: new SPXCrossover(), conditionalParams: rules),
            directions: TwoObjectives);
        study.Optimize(OptimizerObjective, nTrials: 36);

        AssertRuleConsistent(study.Trials, rules);
        study.Trials.Should().Contain(t => Generation(t, "NSGAIIICptSampler") >= 1 && t.Params.ContainsKey("beta1"));
        study.Trials.Should().Contain(t => Generation(t, "NSGAIIICptSampler") >= 1 && t.Params.ContainsKey("momentum"));
    }

    [TestMethod]
    public void RulesThatNeverFire_KeepTheOptunaSequence_BitExact()
    {
        // Same objective, seed and expected sequence as NsgaiiiSamplerTests.SingleObjective_MatchesOptunaSequence
        // (real Optuna 5.0.0.dev output). `y` is declared conditional on every choice of `z`, so it is never
        // absent and never pruned: the conditional path must reduce to the parity path draw for draw.
        var rules = new ConditionalParams().When("z", "a", "b", "c").Activates("y");
        var study = Optuna.CreateStudy(
            sampler: new NSGAIIICptSampler(populationSize: 4, seed: 42, conditionalParams: rules),
            direction: StudyDirection.Minimize);
        study.Optimize(t =>
        {
            double x = t.SuggestFloat("x", -5, 5);
            long y = t.SuggestInt("y", 0, 10);
            t.SuggestCategorical("z", new CategoricalChoice[] { "a", "b", "c" });
            return Math.Pow(x - 2, 2) + y;
        }, nTrials: 12);

        var trials = study.Trials;
        double[] expectedX =
        {
            -1.254598811526375, -3.4400547966379733, -4.7941550570419755, -3.165954901465662,
            -1.254598811526375, -3.6050613934795814, -3.165954901465662, -3.4400547966379733,
            -1.254598811526375, -3.4400547966379733, -1.254598811526375, -3.165954901465662,
        };
        long[] expectedY = { 10, 0, 10, 3, 6, 3, 0, 10, 0, 0, 0, 10 };
        string[] expectedZ = { "a", "a", "a", "a", "a", "a", "a", "c", "c", "b", "a", "a" };

        for (int i = 0; i < 12; i++)
        {
            BitConverter.DoubleToInt64Bits((double)trials[i].Params["x"]!)
                .Should().Be(BitConverter.DoubleToInt64Bits(expectedX[i]), $"x at trial {i}");
            Convert.ToInt64(trials[i].Params["y"]).Should().Be(expectedY[i], $"y at trial {i}");
            CategoricalChoice.From(trials[i].Params["z"]).Value.Should().Be(expectedZ[i], $"z at trial {i}");
        }
    }

    // ---- helpers -------------------------------------------------------------------------------

    private static OrderedDict<string, object?> P(params (string Name, object? Value)[] entries)
    {
        var dict = new OrderedDict<string, object?>();
        foreach (var (name, value) in entries)
            dict[name] = value;
        return dict;
    }

    private static OrderedDict<string, object?> Child(params (string Name, object? Value)[] entries) => P(entries);

    private static int Generation(FrozenTrial trial, string sampler)
        => Convert.ToInt32(trial.SystemAttrs[$"{sampler}:generation"]);

    /// <summary>A conditional parameter must be recorded exactly when the rules say it is active.</summary>
    private static void AssertRuleConsistent(IEnumerable<FrozenTrial> trials, ConditionalParams rules)
    {
        foreach (var trial in trials)
        {
            foreach (var name in rules.Parameters)
            {
                trial.Params.ContainsKey(name).Should().Be(
                    rules.IsActive(name, trial.Params), $"`{name}` in trial {trial.Number}");
            }
        }
    }

    private static List<string> CaptureWarnings(Action action)
    {
        var captured = new List<string>();
        void Handler(string message, WarningCategory category) => captured.Add(message);

        OptunaWarnings.WarningRaised += Handler;
        try
        {
            action();
        }
        finally
        {
            OptunaWarnings.WarningRaised -= Handler;
        }
        return captured;
    }
}

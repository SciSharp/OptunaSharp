using System;
using System.Collections.Generic;
using System.Linq;
using AwesomeAssertions;
using NumSharp;
using OptunaSharp.Collections;
using OptunaSharp.Distributions;
using OptunaSharp.Samplers;
using OptunaSharp.Samplers.Nsgaii;
using OptunaSharp.SearchSpace;

namespace OptunaSharp.Tests;

/// <summary>
///     Does deactivating a parameter affect the <b>math</b> of NSGA-II/III? These tests answer it the only
///     way that is falsifiable: for a child on one branch (so the other branch's parameter is deactivated),
///     compare the generated child <i>and the RNG state after it</i> against a counterfactual world in which
///     that parameter <b>never existed at all</b> (removed from the parents, the search space and the rules).
///     Bit-identity means the deactivated parameter touched nothing.
///
///     <para>Selection, non-domination ranking, crowding distance / reference-point niching and the parent
///     tournament read only objective values and population indices, so they are parameter-independent by
///     construction (nothing to test). The mutation rate uses the active-parameter count and the mutation
///     loop iterates the active parameters, so a deactivated parameter is simply not there. The remaining
///     question — the one these tests pin — is the <b>crossover</b>: a definitely-inactive parameter must not
///     enter the operator (dimensionality / internal draws) nor trigger the partial-carrier draw before it is
///     pruned. That holds for copy-type, arithmetic and — the case the prune-aware fix added — multi-parent
///     crossovers.</para>
///
///     <para>The one residual is documented in <see cref="BoxRejectingCrossover_CanCoupleViaActiveRejectedAttempts" />:
///     the conditional crossover rejects out-of-box children and redraws, so an operator that overshoots the
///     box (vSBX) can let a parameter that is <i>active on a rejected attempt</i> change the redraw count — an
///     active-parameter effect on a discarded attempt, not a deactivated-parameter effect.</para>
/// </summary>
[TestClass]
public class ConditionalParamsMathNeutralityTests
{
    private const int Seeds = 400;

    // Five of the six crossovers: a deactivated parameter changes nothing, bit-for-bit, over 400 seeds.
    // The multi-parent pair (spx, undx) is the point of the prune-aware crossover fix — before it they
    // diverged on ~93 / ~99 of ~275 deactivating children (a to-be-pruned param drew a partial-carrier
    // rng.randint); after it, zero.
    [DataTestMethod]
    [DataRow("uniform")]
    [DataRow("blx")]
    [DataRow("sbx")]
    [DataRow("spx")]
    [DataRow("undx")]
    public void DeactivatedParameter_DoesNotAffectCrossover_NorTheRngStream(string crossoverKind)
    {
        var (decided, diverged) = CountDivergences(crossoverKind);
        decided.Should().BeGreaterThan(100, "the probe must actually produce deactivating children");
        diverged.Should().Be(0,
            $"a parameter inactive for the child must not touch the {crossoverKind} crossover or the RNG stream");
    }

    [TestMethod]
    public void BoxRejectingCrossover_CanCoupleViaActiveRejectedAttempts()
    {
        // vSBX overshoots the search box, so the conditional crossover's reject-and-redraw loop fires. On a
        // rejected attempt whose (intermediate) child was the OTHER branch, the conditional parameter is
        // genuinely active and its crossing changes how often that attempt is rejected — shifting the RNG for
        // the child that is eventually produced. This is a parameter affecting the math WHILE ACTIVE on a
        // discarded attempt, not a deactivated parameter affecting it; it is rare and confined to
        // overshoot-prone operators (the default UniformCrossover never overshoots, so it is exactly neutral).
        var (decided, diverged) = CountDivergences("vsbx");
        decided.Should().BeGreaterThan(100);
        diverged.Should().BeLessThanOrEqualTo(12, "the coupling is a rare box-rejection artifact (measured 5/264)");
    }

    // ---- the counterfactual probe ---------------------------------------------------------------

    private static (int Decided, int Diverged) CountDivergences(string crossoverKind)
    {
        int decided = 0, diverged = 0;
        for (int seed = 0; seed < Seeds; seed++)
        {
            var (child, next) = Generate(seed, crossoverKind, FullWorld(), FullSpace(), Rules());
            if (!child.TryGetValue("optimizer", out var optimizer))
                continue;   // the activator itself was not produced: not a deactivation case.
            decided++;

            // The counterfactual: the same run in a world where the parameter deactivated here never existed.
            OrderedDict<string, object?> counterfactual;
            double counterfactualNext;
            if ((string)optimizer! == "sgd")
                (counterfactual, counterfactualNext) = Generate(seed, crossoverKind, WithoutBeta1(), SpaceWithout("beta1"), RulesWithout("beta1"));
            else
                (counterfactual, counterfactualNext) = Generate(seed, crossoverKind, WithoutMomentum(), SpaceWithout("momentum"), RulesWithout("momentum"));

            bool identical = child.Keys.SequenceEqual(counterfactual.Keys)
                && child.All(kv => SameValue(kv.Value, counterfactual[kv.Key]))
                && BitConverter.DoubleToInt64Bits(next) == BitConverter.DoubleToInt64Bits(counterfactualNext);
            if (!identical)
                diverged++;
        }
        return (decided, diverged);
    }

    private static (OrderedDict<string, object?> Child, double NextDraw) Generate(
        int seed, string crossoverKind, List<FrozenTrial> population,
        OrderedDict<string, BaseDistribution> searchSpace, ConditionalParams rules)
    {
        var rng = new LazyRandomState(seed);
        var study = Optuna.CreateStudy(direction: StudyDirection.Minimize);
        var strategy = new NSGAIIChildGenerationStrategy(
            mutation: null, mutationProb: null, Crossover(crossoverKind), crossoverProb: 0.9, swappingProb: 0.5,
            constraintsFunc: null, rng, rules);
        var child = strategy.Generate(study, searchSpace, population);
        return (child, (double)rng.Rng.rand());
    }

    private static BaseCrossover Crossover(string kind) => kind switch
    {
        "blx" => new BLXAlphaCrossover(),
        "sbx" => new SBXCrossover(),
        "vsbx" => new VSBXCrossover(),
        "spx" => new SPXCrossover(),
        "undx" => new UNDXCrossover(),
        _ => new UniformCrossover(0.5),
    };

    private static ConditionalParams Rules() => new ConditionalParams()
        .When("optimizer", "adam").Activates("beta1").When("optimizer", "sgd").Activates("momentum");

    private static ConditionalParams RulesWithout(string name) => name == "beta1"
        ? new ConditionalParams().When("optimizer", "sgd").Activates("momentum")
        : new ConditionalParams().When("optimizer", "adam").Activates("beta1");

    private static bool SameValue(object? a, object? b)
        => a is double da && b is double db ? BitConverter.DoubleToInt64Bits(da) == BitConverter.DoubleToInt64Bits(db) : Equals(a, b);

    private static FloatDistribution Lr() => new(0.0, 1.0);
    private static FloatDistribution Beta1() => new(0.5, 0.999);
    private static FloatDistribution Mom() => new(0.0, 0.99);
    private static CategoricalDistribution Cat() => new("adam", "sgd");

    private static FrozenTrial T(int number, params (string Name, object? Value, BaseDistribution Distribution)[] entries)
    {
        var @params = new OrderedDict<string, object?>();
        var distributions = new OrderedDict<string, BaseDistribution>();
        foreach (var (name, value, distribution) in entries)
        {
            @params[name] = value;
            distributions[name] = distribution;
        }
        return new FrozenTrial(
            number, TrialState.Complete, value: number, DateTime.UtcNow, DateTime.UtcNow, @params, distributions,
            new OrderedDict<string, object?>(), new OrderedDict<string, object?>(), new OrderedDict<int, double>(), trialId: number);
    }

    private static List<FrozenTrial> FullWorld() => new()
    {
        T(0, ("optimizer", "adam", Cat()), ("lr", 0.10, Lr()), ("beta1", 0.80, Beta1())),
        T(1, ("optimizer", "adam", Cat()), ("lr", 0.20, Lr()), ("beta1", 0.85, Beta1())),
        T(2, ("optimizer", "sgd", Cat()), ("lr", 0.30, Lr()), ("momentum", 0.50, Mom())),
        T(3, ("optimizer", "sgd", Cat()), ("lr", 0.40, Lr()), ("momentum", 0.60, Mom())),
    };

    private static List<FrozenTrial> WithoutBeta1() => new()
    {
        T(0, ("optimizer", "adam", Cat()), ("lr", 0.10, Lr())),
        T(1, ("optimizer", "adam", Cat()), ("lr", 0.20, Lr())),
        T(2, ("optimizer", "sgd", Cat()), ("lr", 0.30, Lr()), ("momentum", 0.50, Mom())),
        T(3, ("optimizer", "sgd", Cat()), ("lr", 0.40, Lr()), ("momentum", 0.60, Mom())),
    };

    private static List<FrozenTrial> WithoutMomentum() => new()
    {
        T(0, ("optimizer", "adam", Cat()), ("lr", 0.10, Lr()), ("beta1", 0.80, Beta1())),
        T(1, ("optimizer", "adam", Cat()), ("lr", 0.20, Lr()), ("beta1", 0.85, Beta1())),
        T(2, ("optimizer", "sgd", Cat()), ("lr", 0.30, Lr())),
        T(3, ("optimizer", "sgd", Cat()), ("lr", 0.40, Lr())),
    };

    private static OrderedDict<string, BaseDistribution> FullSpace() => new()
    {
        ["beta1"] = Beta1(), ["lr"] = Lr(), ["momentum"] = Mom(), ["optimizer"] = Cat(),
    };

    private static OrderedDict<string, BaseDistribution> SpaceWithout(string name)
    {
        var space = FullSpace();
        space.Remove(name);
        return space;
    }
}

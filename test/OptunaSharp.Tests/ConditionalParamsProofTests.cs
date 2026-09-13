using System;
using System.Collections.Generic;
using System.Linq;
using AwesomeAssertions;
using NumSharp;
using OptunaSharp.Collections;
using OptunaSharp.Distributions;
using OptunaSharp.Samplers;
using OptunaSharp.Samplers.Nsgaii;
using OptunaSharp.Samplers.NsgaiiiCpt;
using OptunaSharp.SearchSpace;

namespace OptunaSharp.Tests;

/// <summary>
///     The three behavioral claims behind <see cref="ConditionalParams" />, each pinned as a test:
///     <list type="number">
///         <item><b>Controllable</b> — every rule kind (a categorical value set, a single value, a numeric
///         predicate, AND across activators, a chain) drives the sampler's <i>own proposals</i>, not merely
///         the objective's branches: a parameter is never proposed next to an activator value that
///         deactivates it, and a proposal the objective asks for is exactly what the trial gets.</item>
///         <item><b>Learns when active</b> — under the rules a branch-only parameter converges on its
///         optimum, while the same seeded studies under Optuna's plain intersection leave it at the
///         random-draw level; the unconditional parameter converges either way.</item>
///         <item><b>Dormant when inactive</b> — for every seed in which a parameter is deactivated, the
///         child individual and the sampler's RNG state are bit-identical to a world in which that
///         parameter never existed: nothing about it is blended, counted or drawn.</item>
///     </list>
///     The runnable, printed version of claim 2 is <c>examples/ConditionalParameters.cs</c>.
/// </summary>
[TestClass]
public class ConditionalParamsProofTests
{
    private const double LrOpt = 0.3, Beta1Opt = 0.9, MomentumOpt = 0.7;

    // ---- claim 1: controllable ------------------------------------------------------------------

    [TestMethod]
    public void Controllable_EveryRuleKind_DrivesTheSamplersOwnProposals()
    {
        var rules = new ConditionalParams()
            .When("optimizer", "adam", "adamw").Activates("beta1", "warmup")   // a categorical value set
            .When("optimizer", "sgd").Activates("momentum")                    // a single categorical value
            .When("n_layers", v => (long)v! >= 3).Activates("width_3")          // a numeric predicate on an int
            .When("use_sched", true).Activates("sched", "warmup")               // a bool; `warmup` needs this AND adam-family
            .When("sched", "step").Activates("step_size");                      // a chain: use_sched -> sched -> step_size

        // The objective is deliberately indifferent to every activator (only `lr` matters), so no branch
        // is selected against and each rule keeps carriers in the population for the whole run.
        var proposals = new Dictionary<int, OrderedDict<string, object?>>();
        var study = Optuna.CreateStudy(
            sampler: new NSGAIIICptSampler(populationSize: 16, seed: 7, conditionalParams: rules),
            direction: StudyDirection.Minimize);
        study.Optimize(t =>
        {
            // The sampler's proposal for this trial, captured before the objective suggests anything.
            proposals[t.Number] = new OrderedDict<string, object?>(t.RelativeParams);
            return MixedRulesObjective(t);
        }, nTrials: 160);

        var proposed = rules.Parameters.ToDictionary(n => n, _ => 0, StringComparer.Ordinal);
        var withheld = rules.Parameters.ToDictionary(n => n, _ => 0, StringComparer.Ordinal);
        foreach (var trial in study.Trials)
        {
            var record = trial.Params;
            var proposal = proposals[trial.Number];
            foreach (var name in rules.Parameters)
            {
                // (a) the trial record holds the parameter exactly when the rules say it is active
                record.ContainsKey(name).Should().Be(rules.IsActive(name, record), $"`{name}` in trial {trial.Number}");

                if (Generation(trial, "NSGAIIICptSampler") == 0)
                    continue;   // generation 0 is pure random sampling: there is no proposal to judge

                bool inProposal = proposal.ContainsKey(name);
                foreach (var activator in rules.ActivatorsOf(name))
                {
                    if (!proposal.TryGetValue(activator, out var value))
                        continue;   // the activator was dropped for mutation and is drawn at suggest time
                    bool admits = rules.ActivatorAdmits(name, activator, value);
                    // (b) never proposed next to an activator value that deactivates it
                    if (inProposal)
                        admits.Should().BeTrue($"`{name}` was proposed in trial {trial.Number} although `{activator}`={value} deactivates it");
                    if (!admits)
                        withheld[name]++;
                }

                if (!inProposal)
                    continue;
                proposed[name]++;
                if (record.ContainsKey(name))
                {
                    // (c) a proposal the objective asked for is exactly what the trial got: the evolved value feeds the trial
                    SameValue(record[name], proposal[name]).Should().BeTrue($"`{name}` in trial {trial.Number} should be the proposed value");
                }
                else
                {
                    // and one it did not ask for was deactivated by a freshly drawn activator — never a stale leak
                    rules.IsActive(name, record).Should().BeFalse($"`{name}` was proposed but not recorded in trial {trial.Number}");
                }
            }
        }

        // Non-vacuous: every rule kind both proposed its parameter and withheld it, many times.
        foreach (var name in rules.Parameters)
        {
            proposed[name].Should().BeGreaterThan(5, $"`{name}` should be proposed while active");
            withheld[name].Should().BeGreaterThan(5, $"`{name}` should be withheld when its activator deactivates it");
        }
    }

    // ---- claim 2: learns when active, random without the rules ----------------------------------

    [TestMethod]
    public void Learns_WhenActive_StaysRandom_UnderThePlainIntersection()
    {
        // The conditional feature lives only in NSGA-III-CPT now, so both arms use it: the rules arm carries
        // the activation rules; the "plain intersection" arm passes null rules — which is bit-identical to
        // NSGAIIISampler parity (see ConditionalParamsGateTests). The claim: the rules make the branch-only
        // parameters learnable where the plain intersection leaves them at the random-draw level.
        const string samplerName = "NSGAIIICptSampler";
        var rules = new ConditionalParams()
            .When("optimizer", "adam").Activates("beta1")
            .When("optimizer", "sgd").Activates("momentum");

        // Three seeds, pooled: the late (last four of sixteen) generations of the same seeded study with
        // the rules and under Optuna's plain intersection. Mutation resampling is kept at 10% so the
        // inheritance signal is not drowned in uniform re-draws — the plain runs use the same setting.
        var lateRules = new List<FrozenTrial>();
        var latePlain = new List<FrozenTrial>();
        for (int seed = 0; seed < 3; seed++)
        {
            var withRules = RunTwoBranch(seed, rules);
            var plain = RunTwoBranch(seed, rules: null);

            // Generation 0 is identical (same seed, pure random sampling): only the rules differ afterwards.
            withRules.Trials.Where(t => Generation(t, samplerName) == 0).Select(t => t.Params["lr"])
                .Should().Equal(plain.Trials.Where(t => Generation(t, samplerName) == 0).Select(t => t.Params["lr"]));

            lateRules.AddRange(withRules.Trials.Where(t => Generation(t, samplerName) >= 12));
            latePlain.AddRange(plain.Trials.Where(t => Generation(t, samplerName) >= 12));
        }
        foreach (var late in new[] { lateRules, latePlain })
        {
            late.Count(t => t.Params.ContainsKey("beta1")).Should().BeGreaterThan(40, "both branches stay on the front");
            late.Count(t => t.Params.ContainsKey("momentum")).Should().BeGreaterThan(40, "both branches stay on the front");
        }

        // Random-draw levels for reference: median |U(0.5,0.999)-0.9| = 0.150, median |U(0,0.99)-0.7| = 0.245;
        // P(within 0.05) = 0.20 and 0.10.
        Median(lateRules, "beta1", Beta1Opt).Should().BeLessThan(0.05, "`beta1` is evolved whenever adam is active");
        Median(lateRules, "momentum", MomentumOpt).Should().BeLessThan(0.06, "`momentum` is evolved whenever sgd is active");
        HitRate(lateRules, "beta1", Beta1Opt).Should().BeGreaterThan(0.5);
        HitRate(lateRules, "momentum", MomentumOpt).Should().BeGreaterThan(0.5);

        Median(latePlain, "beta1", Beta1Opt).Should().BeGreaterThan(0.08, "the plain intersection re-draws `beta1` uniformly every trial");
        Median(latePlain, "momentum", MomentumOpt).Should().BeGreaterThan(0.15, "the plain intersection re-draws `momentum` uniformly every trial");
        HitRate(latePlain, "beta1", Beta1Opt).Should().BeLessThan(0.35);
        HitRate(latePlain, "momentum", MomentumOpt).Should().BeLessThan(0.25);

        Median(lateRules, "beta1", Beta1Opt).Should().BeLessThan(Median(latePlain, "beta1", Beta1Opt) / 2.5);
        Median(lateRules, "momentum", MomentumOpt).Should().BeLessThan(Median(latePlain, "momentum", MomentumOpt) / 2.5);

        // The unconditional parameter converges either way: the rules change nothing for it.
        Median(lateRules, "lr", LrOpt).Should().BeLessThan(0.05);
        Median(latePlain, "lr", LrOpt).Should().BeLessThan(0.05);
    }

    // ---- claim 3: dormant when inactive ---------------------------------------------------------

    [TestMethod]
    [DataRow("uniform")]
    [DataRow("sbx")]
    public void Dormant_WhenDeactivated_ChildAndRngState_MatchAWorldWithoutTheParameter(string crossoverKind)
    {
        var study = Optuna.CreateStudy(direction: StudyDirection.Minimize);   // supplies the tournament's directions
        var optimizer = new CategoricalDistribution("adam", "sgd");
        var lr = new FloatDistribution(0.0, 1.0);
        var beta1 = new FloatDistribution(0.5, 0.999);
        var momentum = new FloatDistribution(0.0, 0.99);

        FrozenTrial Adam(int number, double lrValue, double beta1Value, double value) => Frozen(
            number, value, ("optimizer", "adam", optimizer), ("lr", lrValue, lr), ("beta1", beta1Value, beta1));
        FrozenTrial Sgd(int number, double lrValue, double momentumValue, double value) => Frozen(
            number, value, ("optimizer", "sgd", optimizer), ("lr", lrValue, lr), ("momentum", momentumValue, momentum));

        // The full world: both branch-only parameters exist, each in the parents of its branch.
        var population = new List<FrozenTrial>
        {
            Adam(0, 0.10, 0.90, 1.0), Sgd(1, 0.20, 0.50, 2.0), Adam(2, 0.30, 0.80, 3.0), Sgd(3, 0.40, 0.60, 4.0),
        };
        var space = Space(("beta1", beta1), ("lr", lr), ("momentum", momentum), ("optimizer", optimizer));   // sorted, like InferRelativeSearchSpace
        var rules = new ConditionalParams()
            .When("optimizer", "adam").Activates("beta1")
            .When("optimizer", "sgd").Activates("momentum");

        // Two counterfactual worlds in which one of the parameters never existed at all.
        var noBeta1 = population.Select(t => Without(t, "beta1")).ToList();
        var noBeta1Space = Without(space, "beta1");
        var noBeta1Rules = new ConditionalParams().When("optimizer", "sgd").Activates("momentum");
        var noMomentum = population.Select(t => Without(t, "momentum")).ToList();
        var noMomentumSpace = Without(space, "momentum");
        var noMomentumRules = new ConditionalParams().When("optimizer", "adam").Activates("beta1");

        int sgdChildren = 0, adamChildren = 0;
        for (int seed = 0; seed < 300; seed++)
        {
            var (child, next) = Generate(seed, crossoverKind, study, space, population, rules);
            if (!child.TryGetValue("optimizer", out var chosen))
                continue;   // the activator itself was dropped for mutation: undecidable until suggest time, by design

            OrderedDict<string, object?> counterfactual;
            double counterfactualNext;
            if ((string)chosen! == "sgd")
            {
                child.Should().NotContainKey("beta1", $"seed {seed}: `beta1` is deactivated for an sgd child");
                (counterfactual, counterfactualNext) = Generate(seed, crossoverKind, study, noBeta1Space, noBeta1, noBeta1Rules);
                sgdChildren++;
            }
            else
            {
                child.Should().NotContainKey("momentum", $"seed {seed}: `momentum` is deactivated for an adam child");
                (counterfactual, counterfactualNext) = Generate(seed, crossoverKind, study, noMomentumSpace, noMomentum, noMomentumRules);
                adamChildren++;
            }

            // Bit-identical child (same keys, same order, same bits) ...
            child.Keys.Should().Equal(counterfactual.Keys, $"seed {seed}");
            foreach (var (name, value) in child)
                SameValue(value, counterfactual[name]).Should().BeTrue($"seed {seed}: `{name}`");
            // ... and a bit-identical RNG state afterwards: the dormant parameter consumed no draw.
            BitConverter.DoubleToInt64Bits(next).Should().Be(BitConverter.DoubleToInt64Bits(counterfactualNext), $"seed {seed}: next draw");
        }

        sgdChildren.Should().BeGreaterThan(50);
        adamChildren.Should().BeGreaterThan(50);
    }

    // ---- helpers -------------------------------------------------------------------------------

    /// <summary>One child from a fresh strategy seeded with <paramref name="seed" />, plus the draw that follows it.</summary>
    private static (OrderedDict<string, object?> Child, double NextDraw) Generate(
        int seed, string crossoverKind, Study study, OrderedDict<string, BaseDistribution> space,
        List<FrozenTrial> population, ConditionalParams rules)
    {
        var rng = new LazyRandomState(seed);
        BaseCrossover crossover = crossoverKind == "sbx" ? new SBXCrossover() : new UniformCrossover(0.5);
        BaseMutation? mutation = crossoverKind == "sbx" ? new PolynomialMutation() : null;
        var strategy = new NsgaiiiCptChildGenerationStrategy(
            mutation, mutationProb: null, crossover, crossoverProb: 0.9, swappingProb: 0.5,
            constraintsFunc: null, rng, rules);
        var child = strategy.Generate(study, space, population);
        return (child, (double)rng.Rng.rand());
    }

    /// <summary>Every rule kind at once; only `lr` matters, so every branch keeps carriers in the population.</summary>
    private static double MixedRulesObjective(Trial t)
    {
        string optimizer = t.SuggestCategorical("optimizer", new[] { "adam", "adamw", "sgd" });
        long nLayers = t.SuggestInt("n_layers", 1, 4);
        bool useSched = t.SuggestCategorical("use_sched", new[] { true, false });
        double lr = t.SuggestFloat("lr", 0.0, 1.0);
        if (optimizer is "adam" or "adamw")
            t.SuggestFloat("beta1", 0.5, 0.999);
        else
            t.SuggestFloat("momentum", 0.0, 0.99);
        if (nLayers >= 3)
            t.SuggestFloat("width_3", 8, 64);
        if (useSched)
        {
            string sched = t.SuggestCategorical("sched", new[] { "cosine", "step" });
            if (sched == "step")
                t.SuggestInt("step_size", 1, 10);
            if (optimizer is "adam" or "adamw")
                t.SuggestInt("warmup", 0, 100);
        }
        return Math.Pow(lr - LrOpt, 2.0);
    }

    /// <summary>
    ///     Two tasks sharing `lr`: adam trials can only lower f1, sgd trials only f2, so no adam trial ever
    ///     dominates an sgd trial or vice versa — both branches survive elite selection and keep evolving.
    /// </summary>
    private static IReadOnlyList<double> TwoBranchObjective(Trial t)
    {
        string optimizer = t.SuggestCategorical("optimizer", new[] { "adam", "sgd" });
        double lr = t.SuggestFloat("lr", 0.0, 1.0);
        double shared = Math.Pow(lr - LrOpt, 2.0);
        if (optimizer == "adam")
        {
            double beta1 = t.SuggestFloat("beta1", 0.5, 0.999);
            return new[] { shared + Math.Pow(beta1 - Beta1Opt, 2.0), 1.0 + shared };
        }
        double momentum = t.SuggestFloat("momentum", 0.0, 0.99);
        return new[] { 1.0 + shared, shared + Math.Pow(momentum - MomentumOpt, 2.0) };
    }

    private static Study RunTwoBranch(int seed, ConditionalParams? rules)
    {
        // NSGA-III-CPT is the only sampler carrying the conditional feature; null rules give the plain
        // NSGA-III intersection behavior (bit-identical to NSGAIIISampler — ConditionalParamsGateTests).
        var sampler = new NSGAIIICptSampler(populationSize: 24, seed: seed, mutationProb: 0.1, conditionalParams: rules);
        var study = Optuna.CreateStudy(
            sampler: sampler, directions: new[] { StudyDirection.Minimize, StudyDirection.Minimize });
        study.Optimize(TwoBranchObjective, nTrials: 24 * 16);
        return study;
    }

    private static List<double> Distances(IEnumerable<FrozenTrial> trials, string name, double optimum)
        => trials.Where(t => t.Params.ContainsKey(name)).Select(t => Math.Abs((double)t.Params[name]! - optimum)).ToList();

    private static double Median(IEnumerable<FrozenTrial> trials, string name, double optimum)
    {
        var sorted = Distances(trials, name, optimum).OrderBy(d => d).ToList();
        return sorted[sorted.Count / 2];
    }

    /// <summary>The share of trials carrying the parameter within 0.05 of its optimum.</summary>
    private static double HitRate(IEnumerable<FrozenTrial> trials, string name, double optimum)
    {
        var distances = Distances(trials, name, optimum);
        return distances.Count(d => d <= 0.05) / (double)distances.Count;
    }

    private static int Generation(FrozenTrial trial, string sampler)
        => Convert.ToInt32(trial.SystemAttrs[$"{sampler}:generation"]);

    private static bool SameValue(object? a, object? b)
        => a is double da && b is double db ? BitConverter.DoubleToInt64Bits(da) == BitConverter.DoubleToInt64Bits(db) : Equals(a, b);

    private static FrozenTrial Frozen(int number, double value, params (string Name, object? Value, BaseDistribution Distribution)[] entries)
    {
        var @params = new OrderedDict<string, object?>();
        var distributions = new OrderedDict<string, BaseDistribution>();
        foreach (var (name, v, distribution) in entries)
        {
            @params[name] = v;
            distributions[name] = distribution;
        }
        return new FrozenTrial(
            number, TrialState.Complete, value, DateTime.UtcNow, DateTime.UtcNow, @params, distributions,
            new OrderedDict<string, object?>(), new OrderedDict<string, object?>(), new OrderedDict<int, double>(),
            trialId: number);
    }

    private static FrozenTrial Without(FrozenTrial trial, string name)
    {
        var @params = new OrderedDict<string, object?>(trial.ParamsDict.Where(kv => kv.Key != name));
        var distributions = new OrderedDict<string, BaseDistribution>(trial.DistributionsDict.Where(kv => kv.Key != name));
        return new FrozenTrial(
            trial.Number, trial.State, trial.Value, trial.DatetimeStart, trial.DatetimeComplete, @params, distributions,
            new OrderedDict<string, object?>(), new OrderedDict<string, object?>(), new OrderedDict<int, double>(),
            trialId: trial.TrialId);
    }

    private static OrderedDict<string, BaseDistribution> Space(params (string Name, BaseDistribution Distribution)[] entries)
    {
        var space = new OrderedDict<string, BaseDistribution>();
        foreach (var (name, distribution) in entries)
            space[name] = distribution;
        return space;
    }

    private static OrderedDict<string, BaseDistribution> Without(OrderedDict<string, BaseDistribution> space, string name)
        => new(space.Where(kv => kv.Key != name));
}

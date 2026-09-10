using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using AwesomeAssertions;
using NumSharp;
using OptunaSharp.Distributions;
using OptunaSharp.Samplers;
using OptunaSharp.Samplers.Nsgaii;
using OptunaSharp.SearchSpace;

namespace OptunaSharp.Tests;

/// <summary>
///     The <b>null control</b> for <see cref="ConditionalParams" />: the conditional-parameter effect
///     occurs <i>only</i> when explicitly activated, and then only for the parameters actually declared.
///     Where <see cref="ConditionalParamsProofTests" /> proves the feature does what it claims when turned
///     on, these prove it changes nothing when it is off — the measurements behind the claim are in
///     <c>examples/ConditionalParameters.cs</c> and were taken with the two-branch study below.
///     <list type="number">
///         <item><b>No activation ⇒ the Optuna baseline, bit-for-bit.</b> Passing no rules, an empty
///         <see cref="ConditionalParams" />, or rules that always admit all reproduce the historical
///         real-Optuna sequence to the bit on a study whose parameters are always present.</item>
///         <item><b>An empty <see cref="ConditionalParams" /> is a bit-exact no-op</b> on a branchy study
///         too — across both samplers and a copy-type and an arithmetic crossover — so the effect is gated
///         behind <i>declaring a rule</i>, not behind passing the object.</item>
///         <item><b>Activation is per-parameter.</b> Declaring one branch-only parameter leaves an
///         undeclared sibling exactly at the random-draw level it sits at with no rules at all, while the
///         declared one is learned.</item>
///     </list>
/// </summary>
[TestClass]
public class ConditionalParamsGateTests
{
    private const double LrOpt = 0.3, Beta1Opt = 0.9, MomentumOpt = 0.7;

    // Real Optuna 5.0.0.dev, NSGA-II population_size=4 seed=42 (the sequence pinned by
    // NsgaiiSamplerTests.DefaultSingleObjective_MatchesOptunaSequence). The objective has no branch, so
    // every trial suggests x, y and z — nothing for a conditional rule to prune.
    private static readonly double[] GoldenX =
    {
        -1.254598811526375, -3.4400547966379733, -4.7941550570419755, -3.165954901465662,
        -1.254598811526375, -3.6050613934795814, -3.165954901465662, -3.4400547966379733,
        -1.254598811526375, -3.4400547966379733, -1.254598811526375, -3.165954901465662,
    };
    private static readonly long[] GoldenY = { 10, 0, 10, 3, 6, 3, 0, 10, 0, 0, 0, 10 };
    private static readonly string[] GoldenZ = { "a", "a", "a", "a", "a", "a", "a", "c", "c", "b", "a", "a" };

    // ---- 1. no activation reproduces the Optuna baseline, bit-for-bit ---------------------------

    [TestMethod]
    public void WithoutActivation_ReproducesOptunaBaseline_BitExact()
    {
        // No rules: the plain Optuna-parity path.
        AssertMatchesGolden(RunGolden(conditionalParams: null));
        // An empty rule set: declared, but activates nothing.
        AssertMatchesGolden(RunGolden(new ConditionalParams()));
        // A rule that always admits (z is always one of a/b/c, so `y` is always active): present in every
        // trial anyway, so it is never pruned and the sequence is unchanged.
        AssertMatchesGolden(RunGolden(new ConditionalParams().When("z", "a", "b", "c").Activates("y")));
    }

    private static void AssertMatchesGolden(Study study)
    {
        var trials = study.Trials;
        trials.Should().HaveCount(12);
        for (int i = 0; i < 12; i++)
        {
            BitConverter.DoubleToInt64Bits((double)trials[i].Params["x"]!)
                .Should().Be(BitConverter.DoubleToInt64Bits(GoldenX[i]), $"x at trial {i}");
            Convert.ToInt64(trials[i].Params["y"]).Should().Be(GoldenY[i], $"y at trial {i}");
            CategoricalChoice.From(trials[i].Params["z"]).Value.Should().Be(GoldenZ[i], $"z at trial {i}");
        }
    }

    // ---- 2. an empty ConditionalParams is a bit-exact no-op on a branchy study ------------------

    [DataTestMethod]
    [DataRow("NSGAIISampler", "uniform")]
    [DataRow("NSGAIISampler", "sbx")]
    [DataRow("NSGAIIISampler", "uniform")]
    [DataRow("NSGAIIISampler", "sbx")]
    public void EmptyRules_AreABitExactNoOp_OnABranchyStudy(string samplerName, string crossoverKind)
    {
        // The two-branch study really exercises the conditional code paths (beta1/momentum are branch-only),
        // yet with nothing declared the whole trajectory — every trial's generation, params, and bits —
        // must equal the no-rules run exactly. Proven here for copy-type (uniform) and arithmetic (SBX)
        // crossover, on both samplers, across several seeds.
        for (int seed = 0; seed < 3; seed++)
        {
            string none = Canonical(RunTwoBranch(samplerName, seed, rules: null, crossoverKind), samplerName);
            string empty = Canonical(RunTwoBranch(samplerName, seed, new ConditionalParams(), crossoverKind), samplerName);
            empty.Should().Be(none, $"an empty ConditionalParams must not change anything (seed {seed})");
        }
    }

    // ---- 3. activation is per-parameter ---------------------------------------------------------

    [TestMethod]
    public void Activation_IsPerParameter_UndeclaredSiblingStaysAtBaseline()
    {
        // Pool the late generations over several seeds for each declaration, and compare how close each
        // branch-only parameter gets to its optimum. Random-draw levels: median |beta1-0.9| = 0.150,
        // median |momentum-0.7| = 0.245.
        var none = Pool("NSGAIISampler", rules: null);
        var beta1Only = Pool("NSGAIISampler", new ConditionalParams().When("optimizer", "adam").Activates("beta1"));
        var momentumOnly = Pool("NSGAIISampler", new ConditionalParams().When("optimizer", "sgd").Activates("momentum"));

        // With nothing declared, neither branch-only parameter is learned — both sit near the random level.
        Median(none, "beta1", Beta1Opt).Should().BeGreaterThan(0.10, "no rules ⇒ `beta1` is re-drawn at random");
        Median(none, "momentum", MomentumOpt).Should().BeGreaterThan(0.13, "no rules ⇒ `momentum` is re-drawn at random");

        // Declaring `beta1` alone learns `beta1` ...
        Median(beta1Only, "beta1", Beta1Opt).Should().BeLessThan(0.05, "`beta1` is declared, so it is evolved");
        HitRate(beta1Only, "beta1", Beta1Opt).Should().BeGreaterThan(0.45);
        // ... and leaves the undeclared `momentum` exactly where it was with no rules: still at random level.
        Median(beta1Only, "momentum", MomentumOpt).Should().BeGreaterThan(0.13, "`momentum` was not declared, so nothing changes for it");
        HitRate(beta1Only, "momentum", MomentumOpt).Should().BeLessThan(0.40);
        Median(beta1Only, "beta1", Beta1Opt).Should().BeLessThan(Median(beta1Only, "momentum", MomentumOpt) / 3.0,
            "within one study the declared parameter is learned and the undeclared one is not");

        // Symmetrically, declaring `momentum` alone learns `momentum` and leaves `beta1` at random level.
        Median(momentumOnly, "momentum", MomentumOpt).Should().BeLessThan(0.06, "`momentum` is declared, so it is evolved");
        HitRate(momentumOnly, "momentum", MomentumOpt).Should().BeGreaterThan(0.45);
        Median(momentumOnly, "beta1", Beta1Opt).Should().BeGreaterThan(0.10, "`beta1` was not declared, so nothing changes for it");
        HitRate(momentumOnly, "beta1", Beta1Opt).Should().BeLessThan(0.40);
    }

    // ---- helpers -------------------------------------------------------------------------------

    private static Study RunGolden(ConditionalParams? conditionalParams)
    {
        var study = Optuna.CreateStudy(
            sampler: new NSGAIISampler(populationSize: 4, seed: 42, conditionalParams: conditionalParams),
            direction: StudyDirection.Minimize);
        study.Optimize(t =>
        {
            double x = t.SuggestFloat("x", -5, 5);
            long y = t.SuggestInt("y", 0, 10);
            t.SuggestCategorical("z", new CategoricalChoice[] { "a", "b", "c" });
            return Math.Pow(x - 2, 2) + y;
        }, nTrials: 12);
        return study;
    }

    private static Study RunTwoBranch(string samplerName, int seed, ConditionalParams? rules, string crossoverKind)
    {
        BaseCrossover? crossover = crossoverKind == "sbx" ? new SBXCrossover() : null;
        BaseMutation? mutation = crossoverKind == "sbx" ? new PolynomialMutation() : null;
        BaseSampler sampler = samplerName == "NSGAIISampler"
            ? new NSGAIISampler(populationSize: 24, seed: seed, mutationProb: 0.1, crossover: crossover, mutation: mutation, conditionalParams: rules)
            : new NSGAIIISampler(populationSize: 24, seed: seed, mutationProb: 0.1, crossover: crossover, mutation: mutation, conditionalParams: rules);
        var study = Optuna.CreateStudy(
            sampler: sampler, directions: new[] { StudyDirection.Minimize, StudyDirection.Minimize });
        study.Optimize(TwoBranchObjective, nTrials: 24 * 16);
        return study;
    }

    /// <summary>adam lowers f1 only, sgd lowers f2 only, so both branches stay on the Pareto front.</summary>
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

    /// <summary>The last four generations of the two-branch study, pooled over three seeds.</summary>
    private static List<FrozenTrial> Pool(string samplerName, ConditionalParams? rules)
    {
        var pooled = new List<FrozenTrial>();
        for (int seed = 0; seed < 3; seed++)
        {
            var study = RunTwoBranch(samplerName, seed, rules, crossoverKind: "uniform");
            pooled.AddRange(study.Trials.Where(t =>
                Convert.ToInt32(t.SystemAttrs[$"{samplerName}:generation"]) >= 12));
        }
        return pooled;
    }

    /// <summary>A bit-exact, order-sensitive serialization of the whole study (generation + params).</summary>
    private static string Canonical(Study study, string samplerName)
    {
        var sb = new StringBuilder();
        foreach (var t in study.Trials)
        {
            sb.Append(t.Number).Append(':').Append(t.SystemAttrs[$"{samplerName}:generation"]).Append('|');
            foreach (var key in t.Params.Keys.OrderBy(k => k, StringComparer.Ordinal))
            {
                object? v = t.Params[key];
                string repr = v is double d ? BitConverter.DoubleToInt64Bits(d).ToString() : v?.ToString() ?? "null";
                sb.Append(key).Append('=').Append(repr).Append(',');
            }
            sb.Append(';');
        }
        return sb.ToString();
    }

    private static List<double> Distances(IEnumerable<FrozenTrial> trials, string name, double optimum)
        => trials.Where(t => t.Params.ContainsKey(name)).Select(t => Math.Abs((double)t.Params[name]! - optimum)).ToList();

    private static double Median(IEnumerable<FrozenTrial> trials, string name, double optimum)
    {
        var sorted = Distances(trials, name, optimum).OrderBy(d => d).ToList();
        sorted.Should().NotBeEmpty($"`{name}` should appear in its branch's trials");
        return sorted[sorted.Count / 2];
    }

    private static double HitRate(IEnumerable<FrozenTrial> trials, string name, double optimum)
    {
        var distances = Distances(trials, name, optimum);
        return distances.Count(d => d <= 0.05) / (double)distances.Count;
    }
}

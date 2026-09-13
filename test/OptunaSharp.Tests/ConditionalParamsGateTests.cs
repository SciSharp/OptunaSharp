using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using AwesomeAssertions;
using NumSharp;
using OptunaSharp.Distributions;
using OptunaSharp.Samplers;
using OptunaSharp.Samplers.Nsgaii;
using OptunaSharp.Samplers.NsgaiiiCpt;
using OptunaSharp.SearchSpace;

namespace OptunaSharp.Tests;

/// <summary>
///     The <b>null control</b> for <see cref="ConditionalParams" />: the conditional-parameter effect
///     occurs <i>only</i> when explicitly activated, and then only for the parameters actually declared.
///     Where <see cref="ConditionalParamsProofTests" /> proves the feature does what it claims when turned
///     on, these prove it changes nothing when it is off — the measurements behind the claim are in
///     <c>examples/ConditionalParameters.cs</c> and were taken with the two-branch study below.
///
///     <para>Since the extraction, the conditional feature lives only in <see cref="NSGAIIICptSampler" />
///     (NSGA-III-CPT), so these controls compare NSGA-III-CPT against its own parity baseline
///     <see cref="NSGAIIISampler" /> (constructed with no rules), and separately pin that the
///     Optuna-parity <see cref="NSGAIISampler" /> still reproduces real Optuna bit-for-bit.</para>
///     <list type="number">
///         <item><b>No activation ⇒ the parity baseline, bit-for-bit.</b> The parity
///         <see cref="NSGAIISampler" /> reproduces the historical real-Optuna sequence; and
///         <see cref="NSGAIIICptSampler" /> with an empty <see cref="ConditionalParams" /> or rules that
///         always admit reproduces <see cref="NSGAIIISampler" />'s trajectory to the bit on a study whose
///         parameters are always present.</item>
///         <item><b>An empty <see cref="ConditionalParams" /> is a bit-exact no-op</b> on a branchy study
///         too — for a copy-type and an arithmetic crossover — so the effect is gated behind
///         <i>declaring a rule</i>, not behind passing the object.</item>
///         <item><b>Activation is per-parameter.</b> Declaring one branch-only parameter leaves an
///         undeclared sibling exactly at the random-draw level it sits at with no rules at all, while the
///         declared one is learned.</item>
///     </list>
/// </summary>
[TestClass]
public class ConditionalParamsGateTests
{
    private const double LrOpt = 0.3, Beta1Opt = 0.9, MomentumOpt = 0.7;

    // The concrete NSGA-III-CPT sampler class name — the prefix BaseGASampler stamps on its generation
    // system attribute (`$"{GetType().Name}:generation"`), read below to slice the study by generation.
    private const string CptGenerationSampler = "NSGAIIICptSampler";

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

    // ---- 1. no activation reproduces the parity baseline, bit-for-bit ---------------------------

    [TestMethod]
    public void WithoutActivation_ReproducesOptunaBaseline_BitExact()
    {
        // The Optuna-parity NSGAIISampler (which no longer carries the conditional extension) still
        // reproduces the real-Optuna NSGA-II sequence to the bit.
        AssertMatchesGolden(RunGolden(new NSGAIISampler(populationSize: 4, seed: 42)));

        // NSGA-III-CPT with no active rule reproduces NSGA-III parity bit-for-bit: an empty rule set, and a
        // rule that always admits (so it never prunes), both leave NSGAIIISampler's trajectory unchanged.
        AssertCptMatchesNsgaiiiParity(new ConditionalParams());
        AssertCptMatchesNsgaiiiParity(new ConditionalParams().When("z", "a", "b", "c").Activates("y"));
    }

    /// <summary>Runs NSGA-III-CPT with the given rules and asserts it equals NSGA-III parity to the bit.</summary>
    private static void AssertCptMatchesNsgaiiiParity(ConditionalParams rules)
    {
        var cpt = RunGolden(new NSGAIIICptSampler(populationSize: 4, seed: 42, conditionalParams: rules));
        var parity = RunGolden(new NSGAIIISampler(populationSize: 4, seed: 42));
        AssertSameTrials(cpt, parity);
    }

    /// <summary>Asserts two studies produced bit-identical (x, y, z) over all 12 trials.</summary>
    private static void AssertSameTrials(Study actual, Study expected)
    {
        var a = actual.Trials;
        var e = expected.Trials;
        a.Should().HaveCount(e.Count);
        for (int i = 0; i < e.Count; i++)
        {
            BitConverter.DoubleToInt64Bits((double)a[i].Params["x"]!)
                .Should().Be(BitConverter.DoubleToInt64Bits((double)e[i].Params["x"]!), $"x at trial {i}");
            Convert.ToInt64(a[i].Params["y"]).Should().Be(Convert.ToInt64(e[i].Params["y"]), $"y at trial {i}");
            CategoricalChoice.From(a[i].Params["z"]).Value.Should().Be(CategoricalChoice.From(e[i].Params["z"]).Value, $"z at trial {i}");
        }
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
    [DataRow("uniform")]
    [DataRow("sbx")]
    public void EmptyRules_AreABitExactNoOp_OnABranchyStudy(string crossoverKind)
    {
        // The two-branch study really exercises the conditional code paths (beta1/momentum are branch-only),
        // yet with nothing declared the whole NSGA-III-CPT trajectory — every trial's generation, params, and
        // bits — must equal the no-rules run exactly. Proven here for copy-type (uniform) and arithmetic (SBX)
        // crossover, across several seeds.
        for (int seed = 0; seed < 3; seed++)
        {
            string none = Canonical(RunTwoBranch(seed, rules: null, crossoverKind));
            string empty = Canonical(RunTwoBranch(seed, new ConditionalParams(), crossoverKind));
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
        var none = Pool(rules: null);
        var beta1Only = Pool(new ConditionalParams().When("optimizer", "adam").Activates("beta1"));
        var momentumOnly = Pool(new ConditionalParams().When("optimizer", "sgd").Activates("momentum"));

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

    /// <summary>Runs the branch-free golden objective (x float, y int, z categorical) under the given sampler.</summary>
    private static Study RunGolden(BaseSampler sampler)
    {
        var study = Optuna.CreateStudy(sampler: sampler, direction: StudyDirection.Minimize);
        study.Optimize(t =>
        {
            double x = t.SuggestFloat("x", -5, 5);
            long y = t.SuggestInt("y", 0, 10);
            t.SuggestCategorical("z", new CategoricalChoice[] { "a", "b", "c" });
            return Math.Pow(x - 2, 2) + y;
        }, nTrials: 12);
        return study;
    }

    /// <summary>The two-branch study under NSGA-III-CPT (the only sampler carrying the conditional feature).</summary>
    private static Study RunTwoBranch(int seed, ConditionalParams? rules, string crossoverKind)
    {
        BaseCrossover? crossover = crossoverKind == "sbx" ? new SBXCrossover() : null;
        BaseMutation? mutation = crossoverKind == "sbx" ? new PolynomialMutation() : null;
        var sampler = new NSGAIIICptSampler(
            populationSize: 24, seed: seed, mutationProb: 0.1, crossover: crossover, mutation: mutation,
            conditionalParams: rules);
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

    /// <summary>The last four generations of the two-branch NSGA-III-CPT study, pooled over three seeds.</summary>
    private static List<FrozenTrial> Pool(ConditionalParams? rules)
    {
        var pooled = new List<FrozenTrial>();
        for (int seed = 0; seed < 3; seed++)
        {
            var study = RunTwoBranch(seed, rules, crossoverKind: "uniform");
            pooled.AddRange(study.Trials.Where(t =>
                Convert.ToInt32(t.SystemAttrs[$"{CptGenerationSampler}:generation"]) >= 12));
        }
        return pooled;
    }

    /// <summary>A bit-exact, order-sensitive serialization of the whole study (generation + params).</summary>
    private static string Canonical(Study study)
    {
        var sb = new StringBuilder();
        foreach (var t in study.Trials)
        {
            sb.Append(t.Number).Append(':').Append(t.SystemAttrs[$"{CptGenerationSampler}:generation"]).Append('|');
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

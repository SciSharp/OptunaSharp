using System;
using System.Linq;
using AwesomeAssertions;
using OptunaSharp.Samplers;
using OptunaSharp.Samplers.NsgaiiiCpt;

namespace OptunaSharp.Tests.Samplers;

/// <summary>
///     Tests for <see cref="AgeMoea2CptSampler" /> — AGE-MOEA-2 survival + the reused "Categorical Parameter
///     Trees" conditional-parameter seams. These are <b>wiring</b> gates, not a re-test of the conditional
///     machinery itself (that is covered exhaustively by the <c>ConditionalParams*Tests</c> suites, which
///     exercise the same shared <see cref="NsgaiiiCptSearchSpace" /> / <see cref="NsgaiiiCptChildGenerationStrategy" />
///     objects this sampler holds). The load-bearing gate is that with <c>conditionalParams: null</c> the
///     sampler is <b>bit-identical</b> to the bit-exact <see cref="AgeMoea2Sampler" /> — i.e. adding CPT does
///     not perturb the base algorithm — plus that a conditional multi-objective study runs end to end and keeps
///     each branch's parameters present exactly when active.
/// </summary>
[TestClass]
public class AgeMoea2CptSamplerTests
{
    // ZDT1 (2-objective) — a continuous, non-conditional space for the parity gate.
    private static double[] Zdt1(Trial t, int d)
    {
        var x = new double[d];
        for (int i = 0; i < d; i++) x[i] = t.SuggestFloat($"x{i}", 0, 1);
        double f1 = x[0];
        double gsum = 0; for (int i = 1; i < d; i++) gsum += x[i];
        double g = 1.0 + 9.0 * gsum / (d - 1);
        double f2 = g * (1.0 - Math.Sqrt(f1 / g));
        return new[] { f1, f2 };
    }

    // A conditional 2-objective objective: an "optimizer" categorical gates one branch-specific float, so the
    // set of suggested parameters changes per trial — the situation CPT exists to evolve rather than resample.
    private static double[] OptimizerBranch(Trial t)
    {
        string optimizer = t.SuggestCategorical("optimizer", new[] { "adam", "sgd" });
        double x = t.SuggestFloat("x", 0, 1);
        double extra = optimizer == "adam"
            ? t.SuggestFloat("beta1", 0.80, 0.999)
            : t.SuggestFloat("momentum", 0.0, 0.99);
        double f1 = x;
        double f2 = (1.0 - x) + 0.1 * extra;
        return new[] { f1, f2 };
    }

    /// <summary>The declarations used by the conditional tests, built fresh so no per-study cache is shared.</summary>
    private static ConditionalParams Rules() => new ConditionalParams()
        .When("optimizer", "adam").Activates("beta1")
        .When("optimizer", "sgd").Activates("momentum");

    // ---- the load-bearing gate: null rules ⇒ bit-identical to AgeMoea2Sampler --------------------

    /// <summary>
    ///     With no rules, AGE-MOEA-2-CPT must reproduce <see cref="AgeMoea2Sampler" /> trial-for-trial: same
    ///     elite strategy, same operator defaults, and — because the CPT search space and child generation both
    ///     fall through to the shared Optuna-parity fold under null rules — the same RNG draw order.
    /// </summary>
    [TestMethod]
    public void NullRules_MatchesAgeMoea2Sampler_BitForBit()
    {
        Study Run(BaseSampler s)
        {
            var study = Optuna.CreateStudy(
                directions: new[] { StudyDirection.Minimize, StudyDirection.Minimize }, sampler: s);
            study.Optimize(t => Zdt1(t, 6), nTrials: 90);
            return study;
        }

        var baseHost = Run(new AgeMoea2Sampler(populationSize: 20, seed: 7));
        var cpt = Run(new AgeMoea2CptSampler(populationSize: 20, seed: 7, conditionalParams: null));

        cpt.Trials.Count.Should().Be(baseHost.Trials.Count);
        for (int i = 0; i < baseHost.Trials.Count; i++)
        {
            cpt.Trials[i].Values.Should().Equal(baseHost.Trials[i].Values, $"trial {i} objective values match AGE-MOEA-2");
            foreach (var (k, v) in baseHost.Trials[i].Params)
                cpt.Trials[i].Params[k].Should().Be(v, $"trial {i} param {k} matches AGE-MOEA-2");
        }
    }

    // ---- conditional multi-objective study runs and stays rule-consistent ------------------------

    /// <summary>
    ///     A conditional 2-objective study runs to completion, builds a non-trivial Pareto front, visits both
    ///     branches, and every completed trial carries exactly its active branch's conditional parameter — the
    ///     end-to-end health check that the conditional path is exercised without error.
    /// </summary>
    [TestMethod]
    public void ConditionalParams_RunsAndStaysRuleConsistent()
    {
        var study = Optuna.CreateStudy(
            directions: new[] { StudyDirection.Minimize, StudyDirection.Minimize },
            sampler: new AgeMoea2CptSampler(populationSize: 20, seed: 3, conditionalParams: Rules()));
        study.Optimize(OptimizerBranch, nTrials: 160);

        study.BestTrials.Count.Should().BeGreaterThan(0, "the conditional run builds a Pareto front");

        var completed = study.Trials.Where(t => t.Values is not null).ToList();
        completed.Count.Should().Be(160);
        bool sawAdam = false, sawSgd = false;
        foreach (var trial in completed)
        {
            bool isAdam = trial.Params["optimizer"]?.ToString() == "adam";
            sawAdam |= isAdam;
            sawSgd |= !isAdam;
            trial.Params.ContainsKey("beta1").Should().Be(isAdam, "beta1 is present exactly for the adam branch");
            trial.Params.ContainsKey("momentum").Should().Be(!isAdam, "momentum is present exactly for the sgd branch");
        }
        sawAdam.Should().BeTrue("the search visits the adam branch");
        sawSgd.Should().BeTrue("the search visits the sgd branch");
    }

    /// <summary>A fixed seed replays the whole conditional trajectory (DR-9-style determinism).</summary>
    [TestMethod]
    public void SameSeed_IsReproducible()
    {
        double[][] Run()
        {
            var study = Optuna.CreateStudy(
                directions: new[] { StudyDirection.Minimize, StudyDirection.Minimize },
                sampler: new AgeMoea2CptSampler(populationSize: 20, seed: 11, conditionalParams: Rules()));
            study.Optimize(OptimizerBranch, nTrials: 90);
            return study.Trials.Where(t => t.Values is not null).Select(t => t.Values!.ToArray()).ToArray();
        }

        var a = Run();
        var b = Run();
        a.Length.Should().Be(b.Length);
        for (int i = 0; i < a.Length; i++) a[i].Should().Equal(b[i], $"trial {i} replays identically");
    }
}

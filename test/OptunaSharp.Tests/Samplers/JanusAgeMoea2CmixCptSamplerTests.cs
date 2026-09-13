using System;
using System.Linq;
using AwesomeAssertions;
using OptunaSharp.Samplers;
using OptunaSharp.Samplers.NsgaiiiCpt;

namespace OptunaSharp.Tests.Samplers;

/// <summary>
///     Tests for <see cref="JanusAgeMoea2CmixCptSampler" /> — JANUS CMIX over an AGE-MOEA-2-CPT host. Two
///     load-bearing gates: (1) with CMIX disabled the sampler is <b>bit-identical</b> to its
///     <see cref="AgeMoea2CptSampler" /> host (the subclass never perturbs the host stream — CMIX runs on a
///     separate RNG); (2) on a categorical / conditional search space CMIX is <b>inert even when set to fire
///     on every trial</b>, because its transforms cannot represent a categorical, so the sampler again reduces
///     to the bare AGE-MOEA-2-CPT host. Plus an end-to-end conditional run.
/// </summary>
[TestClass]
public class JanusAgeMoea2CmixCptSamplerTests
{
    // ZDT1 (2-objective) continuous space — for the CMIX-off parity gate.
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

    // A conditional 2-objective objective with an "optimizer" categorical gate (a non-numeric relative space).
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

    /// <summary>The conditional declarations, built fresh per sampler so no per-study cache is shared.</summary>
    private static ConditionalParams Rules() => new ConditionalParams()
        .When("optimizer", "adam").Activates("beta1")
        .When("optimizer", "sgd").Activates("momentum");

    // ---- gate 1: CMIX off ⇒ bit-identical to the AGE-MOEA-2-CPT host -----------------------------

    /// <summary>
    ///     <c>cmixProb = 0</c> (with <c>cmixStartTrial = 0</c> so the gate is fully reached) ⇒ every trial keeps
    ///     the host child. Because the CMIX stream is separate from the host's, the study must match the bare
    ///     <see cref="AgeMoea2CptSampler" /> param-for-param and value-for-value.
    /// </summary>
    [TestMethod]
    public void CmixDisabled_MatchesAgeMoea2CptHost_BitForBit()
    {
        Study Run(BaseSampler s)
        {
            var study = Optuna.CreateStudy(
                directions: new[] { StudyDirection.Minimize, StudyDirection.Minimize }, sampler: s);
            study.Optimize(t => Zdt1(t, 6), nTrials: 90);
            return study;
        }

        var host = Run(new AgeMoea2CptSampler(populationSize: 20, seed: 7, conditionalParams: null));
        var janus = Run(new JanusAgeMoea2CmixCptSampler(
            populationSize: 20, seed: 7, cmixProb: 0.0, cmixStartTrial: 0, conditionalParams: null));

        janus.Trials.Count.Should().Be(host.Trials.Count);
        for (int i = 0; i < host.Trials.Count; i++)
        {
            janus.Trials[i].Values.Should().Equal(host.Trials[i].Values, $"trial {i} objective values match the host");
            foreach (var (k, v) in host.Trials[i].Params)
                janus.Trials[i].Params[k].Should().Be(v, $"trial {i} param {k} matches the host");
        }
    }

    // ---- gate 2: CMIX inert on a categorical space, even firing every trial ----------------------

    /// <summary>
    ///     With <c>cmixProb = 1</c> and <c>cmixStartTrial = 0</c>, CMIX attempts to fire on every trial — yet on
    ///     a conditional-categorical space its transforms bail (a categorical cannot be normalized), so no
    ///     substitution ever occurs and the study is <b>identical</b> to the bare <see cref="AgeMoea2CptSampler" />
    ///     with the same rules. This pins the "CMIX is inert on a categorical genome" property the Rock decision
    ///     rests on.
    /// </summary>
    [TestMethod]
    public void CmixInertOnCategoricalSpace_MatchesHost_EvenWhenAlwaysFiring()
    {
        Study Run(BaseSampler s)
        {
            var study = Optuna.CreateStudy(
                directions: new[] { StudyDirection.Minimize, StudyDirection.Minimize }, sampler: s);
            study.Optimize(OptimizerBranch, nTrials: 120);
            return study;
        }

        var host = Run(new AgeMoea2CptSampler(populationSize: 20, seed: 5, conditionalParams: Rules()));
        var janus = Run(new JanusAgeMoea2CmixCptSampler(
            populationSize: 20, seed: 5, cmixProb: 1.0, cmixStartTrial: 0, conditionalParams: Rules()));

        janus.Trials.Count.Should().Be(host.Trials.Count);
        for (int i = 0; i < host.Trials.Count; i++)
        {
            janus.Trials[i].Values.Should().Equal(host.Trials[i].Values, $"trial {i} matches the host (CMIX never fires on categoricals)");
            foreach (var (k, v) in host.Trials[i].Params)
                janus.Trials[i].Params[k].Should().Be(v, $"trial {i} param {k} matches the host");
        }
    }

    // ---- end-to-end conditional run --------------------------------------------------------------

    /// <summary>The default configuration runs a conditional 2-objective study to a Pareto front, rule-consistent.</summary>
    [TestMethod]
    public void Default_RunsConditionalStudyAndBuildsFront()
    {
        var study = Optuna.CreateStudy(
            directions: new[] { StudyDirection.Minimize, StudyDirection.Minimize },
            sampler: new JanusAgeMoea2CmixCptSampler(populationSize: 20, seed: 1, cmixStartTrial: 20, conditionalParams: Rules()));
        study.Optimize(OptimizerBranch, nTrials: 160);

        study.BestTrials.Count.Should().BeGreaterThan(0, "the conditional run builds a Pareto front");
        foreach (var trial in study.Trials.Where(t => t.Values is not null))
        {
            bool isAdam = trial.Params["optimizer"]?.ToString() == "adam";
            trial.Params.ContainsKey("beta1").Should().Be(isAdam, "beta1 is present exactly for the adam branch");
            trial.Params.ContainsKey("momentum").Should().Be(!isAdam, "momentum is present exactly for the sgd branch");
        }
    }
}

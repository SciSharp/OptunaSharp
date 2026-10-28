using System;
using System.Collections.Generic;
using System.Linq;
using AwesomeAssertions;
using OptunaSharp.Distributions;
using OptunaSharp.Samplers;

namespace OptunaSharp.Tests.Samplers;

/// <summary>
///     Every Core sampler under <c>Optimize(nJobs: 8)</c>: each trial completes and the numbering is contiguous. All
///     workers share one sampler, so every sampler call runs under the sampler's monitor; before that, MOEA/D, JANUS and
///     JANUS-DE failed trials, REMBO threw before the first one, and Random and TPE repeated draws. This is the
///     regression form of the 2026-10-28 sweep, which ran all 32 samplers at <c>nJobs: 16</c> on a journal file.
/// </summary>
/// <remarks>
///     A sampler may still give one point to two concurrent trials when it derives the point from state that changes
///     only when trials finish (JANUS without its own lanes, REMBO's acquisition, the genetic samplers' unchanged
///     copies); that is documented, so only completion and numbering are asserted here. SAASBO, the one slow sampler, is
///     in the <c>Slow</c> category; every other case runs on each push.
/// </remarks>
[TestClass]
public class SamplerNJobsSweepTests
{
    private static readonly HashSet<string> MultiObjective = new()
    {
        "moead", "nsga2", "nsga3", "nsga3cpt", "smsemoa", "agemoea2", "agemoea2cpt", "rvea",
        "janusmoo", "janussms", "janusage2", "janusage2cpt",
    };

    /// <summary>Builds the named sampler with a fixed seed; the genetic ones use a small population.</summary>
    /// <param name="name">A key of the data rows below.</param>
    /// <returns>A new sampler instance.</returns>
    /// <exception cref="ArgumentException"><paramref name="name" /> names no sampler.</exception>
    private static BaseSampler Make(string name) => name switch
    {
        "random" => new RandomSampler(seed: 1),
        "tpe" => new TPESampler(seed: 1),
        "tpe-independent" => new TPESampler(seed: 1, multivariate: false),
        "cmaes" => new CmaEsSampler(seed: 1),
        "qmc" => new QMCSampler(seed: 1),
        "qmc-scrambled" => new QMCSampler(scramble: true, seed: 1),
        "partialfixed" => new PartialFixedSampler(new Dictionary<string, object?> { ["z"] = 0.5 }, new TPESampler(seed: 1)),
        "janus" => new JanusSampler(seed: 1),
        "janusde" => new JanusDeSampler(seed: 1),
        "de" => new DifferentialEvolutionSampler(seed: 1),
        "pso" => new PsoSampler(seed: 1),
        "lmcma" => new LmCmaSampler(seed: 1),
        "vkd" => new VkdCmaSampler(seed: 1),
        "openaies" => new OpenAiEsSampler(seed: 1),
        "snes" => new SnesSampler(seed: 1),
        "xnes" => new XnesSampler(seed: 1),
        "moead" => new MoeaDSampler(populationSize: 12, seed: 1),
        "nsga2" => new NSGAIISampler(populationSize: 12, seed: 1),
        "nsga3" => new NSGAIIISampler(populationSize: 12, seed: 1),
        "nsga3cpt" => new NSGAIIICptSampler(populationSize: 12, seed: 1),
        "smsemoa" => new SmsEmoaSampler(populationSize: 12, seed: 1),
        "agemoea2" => new AgeMoea2Sampler(populationSize: 12, seed: 1),
        "agemoea2cpt" => new AgeMoea2CptSampler(populationSize: 12, seed: 1),
        "rvea" => new RveaSampler(populationSize: 12, seed: 1),
        "janusmoo" => new JanusMooCmixSampler(populationSize: 12, seed: 1),
        "janussms" => new JanusSmsEmoaCmixSampler(populationSize: 12, seed: 1),
        "janusage2" => new JanusAgeMoea2CmixSampler(populationSize: 12, seed: 1),
        "janusage2cpt" => new JanusAgeMoea2CmixCptSampler(populationSize: 12, seed: 1),
        "gp" => new GPSampler(seed: 1),
        "turbo" => new TurboSampler(seed: 1),
        "rembo" => new RandomEmbeddingSampler(2, seed: 1),
        "saasbo" => new SaasboSampler(seed: 1, numSamples: 32),
        _ => throw new ArgumentException("unknown sampler " + name, nameof(name)),
    };

    /// <summary>The samplers whose trials are cheap enough for every push.</summary>
    /// <param name="name">The sampler under test (see <see cref="Make" />).</param>
    [TestMethod]
    [DataRow("random")]
    [DataRow("tpe")]
    [DataRow("tpe-independent")]
    [DataRow("cmaes")]
    [DataRow("qmc")]
    [DataRow("qmc-scrambled")]
    [DataRow("partialfixed")]
    [DataRow("janus")]
    [DataRow("janusde")]
    [DataRow("de")]
    [DataRow("pso")]
    [DataRow("lmcma")]
    [DataRow("vkd")]
    [DataRow("openaies")]
    [DataRow("snes")]
    [DataRow("xnes")]
    [DataRow("moead")]
    [DataRow("nsga2")]
    [DataRow("nsga3")]
    [DataRow("nsga3cpt")]
    [DataRow("smsemoa")]
    [DataRow("agemoea2")]
    [DataRow("agemoea2cpt")]
    [DataRow("rvea")]
    [DataRow("janusmoo")]
    [DataRow("janussms")]
    [DataRow("janusage2")]
    [DataRow("janusage2cpt")]
    public void NJobs_Sampler_CompletesEveryTrialWithContiguousNumbers(string name) => RunAndCheck(name, 60);

    /// <summary>
    ///     The Gaussian-process samplers, which fit a model on every trial after their startup: the same check with fewer
    ///     trials (about half a second each in memory).
    /// </summary>
    /// <param name="name">The sampler under test (see <see cref="Make" />).</param>
    [TestMethod]
    [DataRow("gp")]
    [DataRow("turbo")]
    [DataRow("rembo")]
    public void NJobs_ModelBasedSampler_CompletesEveryTrialWithContiguousNumbers(string name) => RunAndCheck(name, 24);

    /// <summary>SAASBO, whose NUTS posterior sampling takes about ten seconds for these trials: the same check, slow category.</summary>
    [TestMethod]
    [Slow]
    public void NJobs_Saasbo_CompletesEveryTrialWithContiguousNumbers() => RunAndCheck("saasbo", 24);

    /// <summary>
    ///     Grid and BruteForce on a 90-point discrete space: every trial completes, the numbering is contiguous, and
    ///     every point is covered before the sampler stops the study. Concurrent workers can take one point twice while
    ///     it is still running, as Optuna documents for both, so the trial count may exceed 90.
    /// </summary>
    /// <param name="name"><c>grid</c> or <c>bruteforce</c>.</param>
    [TestMethod]
    [DataRow("grid")]
    [DataRow("bruteforce")]
    public void NJobs_ExhaustiveSampler_CoversEveryPointAndCompletesEveryTrial(string name)
    {
        BaseSampler sampler = name == "grid"
            ? new GridSampler(new Dictionary<string, IReadOnlyList<CategoricalChoice>>
            {
                ["x"] = Enumerable.Range(0, 9).Select(i => CategoricalChoice.From(-10.0 + 2.5 * i)).ToList(),
                ["y"] = Enumerable.Range(0, 5).Select(i => CategoricalChoice.From((long)i)).ToList(),
                ["c"] = new[] { "a", "b" }.Select(s => CategoricalChoice.From(s)).ToList(),
            }, seed: 1)
            : new BruteForceSampler(seed: 1);
        var study = Optuna.CreateStudy(sampler: sampler);

        study.Optimize(t =>
        {
            double x = t.SuggestFloat("x", -10, 10, step: 2.5);
            long y = t.SuggestInt("y", 0, 4);
            string c = t.SuggestCategorical("c", new[] { "a", "b" });
            return x * x + y + (c == "a" ? 0 : 1);
        }, nTrials: 200, nJobs: 8);

        var trials = study.Trials;
        trials.Select(t => t.Number).Should().Equal(Enumerable.Range(0, trials.Count));
        trials.Should().OnlyContain(t => t.State == TrialState.Complete);
        trials.Select(t => $"{t.Params["x"]}|{t.Params["y"]}|{t.Params["c"]}").Distinct().Should().HaveCount(90,
            "the sampler stops only once every point has been taken");
    }

    /// <summary>
    ///     The second-pass edge: every cheap sampler at <c>nJobs: 8</c> on a conditional search space whose branch is a
    ///     categorical choice, with a fixed share of trials failing (a caught exception) and of trials pruned. Samplers
    ///     reading the history while it changes see trials of both branches and of every state, which is how
    ///     relative-space inference, the genetic samplers' populations and the evolution strategies' generations meet
    ///     FAIL and PRUNED trials mid-run. Every trial must end in the state its number dictates, with exactly the
    ///     parameters of its branch, and the numbering stays contiguous.
    /// </summary>
    /// <param name="name">The sampler under test (see <see cref="Make" />).</param>
    /// <exception cref="NumSharp.ValueError">Thrown by every seventh trial; caught, so the trial is recorded as FAIL.</exception>
    /// <exception cref="TrialPruned">Thrown by trials numbered 5 mod 11 that do not fail; recorded as PRUNED.</exception>
    [TestMethod]
    [DataRow("random")]
    [DataRow("tpe")]
    [DataRow("tpe-independent")]
    [DataRow("cmaes")]
    [DataRow("qmc")]
    [DataRow("qmc-scrambled")]
    [DataRow("partialfixed")]
    [DataRow("janus")]
    [DataRow("janusde")]
    [DataRow("de")]
    [DataRow("pso")]
    [DataRow("lmcma")]
    [DataRow("vkd")]
    [DataRow("openaies")]
    [DataRow("snes")]
    [DataRow("xnes")]
    [DataRow("moead")]
    [DataRow("nsga2")]
    [DataRow("nsga3")]
    [DataRow("nsga3cpt")]
    [DataRow("smsemoa")]
    [DataRow("agemoea2")]
    [DataRow("agemoea2cpt")]
    [DataRow("rvea")]
    [DataRow("janusmoo")]
    [DataRow("janussms")]
    [DataRow("janusage2")]
    [DataRow("janusage2cpt")]
    public void NJobs_Sampler_OnAConditionalSpaceWithFailedAndPrunedTrials_EndsEveryTrialAsDictated(string name)
    {
        const int nTrials = 60;
        var sampler = Make(name);
        bool mo = MultiObjective.Contains(name);
        var study = mo
            ? Optuna.CreateStudy(sampler: sampler, directions: new[] { StudyDirection.Minimize, StudyDirection.Minimize })
            : Optuna.CreateStudy(sampler: sampler);

        study.Optimize(t =>
        {
            // "z" is common to both branches (and the parameter the partial-fixed sampler pins); the branch decides the rest.
            double z = t.SuggestFloat("z", 0, 1);
            string kind = t.SuggestCategorical("kind", new[] { "a", "b" });
            double loss = kind == "a"
                ? Math.Pow(t.SuggestFloat("a_x", -5, 5) - 1, 2) + t.SuggestInt("a_n", 1, 8) * 0.1
                : Math.Log(t.SuggestFloat("b_y", 1e-3, 1e3, log: true)) + (t.SuggestCategorical("b_c", new[] { "p", "q", "r" }) == "q" ? 0 : 1);
            if (t.Number % 7 == 3)
                throw new NumSharp.ValueError("failing trial");
            if (t.Number % 11 == 5)
                throw new TrialPruned();
            return mo ? new[] { loss + z, (1 - z) * (1 - z) } : new[] { loss + z };
        }, nTrials: nTrials, nJobs: 8, @catch: new[] { typeof(NumSharp.ValueError) });

        var trials = study.Trials;
        trials.Select(t => t.Number).Should().Equal(Enumerable.Range(0, nTrials));
        foreach (var trial in trials)
        {
            var expected = trial.Number % 7 == 3 ? TrialState.Fail
                : trial.Number % 11 == 5 ? TrialState.Pruned
                : TrialState.Complete;
            trial.State.Should().Be(expected, $"trial {trial.Number} ends as its number dictates");
            string kind = (string)trial.Params["kind"]!;
            var branch = kind == "a" ? new[] { "a_n", "a_x" } : new[] { "b_c", "b_y" };
            trial.Params.Keys.Order().Should().Equal(branch.Append("kind").Append("z").Order(),
                $"trial {trial.Number} holds exactly the parameters of branch {kind}");
        }
    }

    /// <summary>
    ///     Runs the named sampler over a three-parameter space at <c>nJobs: 8</c> and checks the study: the trial count,
    ///     contiguous numbering, and every trial complete with all three parameters.
    /// </summary>
    /// <param name="name">The sampler under test (see <see cref="Make" />).</param>
    /// <param name="nTrials">The trial count.</param>
    private static void RunAndCheck(string name, int nTrials)
    {
        var sampler = Make(name);
        bool mo = MultiObjective.Contains(name);
        var study = mo
            ? Optuna.CreateStudy(sampler: sampler, directions: new[] { StudyDirection.Minimize, StudyDirection.Minimize })
            : Optuna.CreateStudy(sampler: sampler);

        if (mo)
        {
            study.Optimize(t =>
            {
                double x = t.SuggestFloat("x", -10, 10);
                double y = t.SuggestFloat("y", -10, 10);
                double z = t.SuggestFloat("z", 0, 1);
                return new[] { (x - 2) * (x - 2) + y * y + z, x * x + (y - 1) * (y - 1) };
            }, nTrials: nTrials, nJobs: 8);
        }
        else
        {
            study.Optimize(t =>
            {
                double x = t.SuggestFloat("x", -10, 10);
                double y = t.SuggestFloat("y", -10, 10);
                double z = t.SuggestFloat("z", 0, 1);
                return (x - 2) * (x - 2) + (y + 1) * (y + 1) + z;
            }, nTrials: nTrials, nJobs: 8);
        }

        var trials = study.Trials;
        trials.Select(t => t.Number).Should().Equal(Enumerable.Range(0, nTrials));
        trials.Should().OnlyContain(t => t.State == TrialState.Complete
            && t.Params.ContainsKey("x") && t.Params.ContainsKey("y") && t.Params.ContainsKey("z"));
    }
}

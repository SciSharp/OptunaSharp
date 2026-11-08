using System;
using System.Collections.Generic;
using System.Linq;
using AwesomeAssertions;
using OptunaSharp.Samplers;
using OptunaSharp.Tests.Samplers;

namespace OptunaSharp.Tests.Concurrency;

/// <summary>
///     One sampler instance serving a study per shooter, every shooter driving its own study at the same instant
///     (THREADING.md §1.5, §6.3). Each study lives in its own in-memory storage, so all of them have study id 0, which is
///     the case that per-study state must tell apart by storage object (<c>PerStudyState</c>). A genetic sampler gives
///     every study the generations a study run alone gets, and every Core sampler completes every trial of every study.
/// </summary>
/// <remarks>
///     A shared sampler draws from one generator for all studies, so the values differ from solo runs; what must not
///     differ is each study's structure. Samplers whose state is derived from each study's own history (TPE, CMA-ES, the
///     evolution strategies, QMC) share safely by construction; the genetic samplers, JANUS, JANUS-DE, MOEA/D, SMS-EMOA
///     and REMBO keep state per study.
/// </remarks>
[TestClass]
public class SamplerSharingConcurrencyCoverageTests
{
    private const int Threads = ConcurrencyGun.DefaultThreads;

    /// <summary>
    ///     Every shooter resolves the per-study state of its own study at once, as a sampler does under its own lock, every
    ///     round. Every study id is 0, so only the storage object tells the studies apart. Each study gets a state of its own,
    ///     and the same state again in every later round.
    /// </summary>
    /// <remarks>
    ///     This pins the keying itself. The generation test below cannot: a genetic sampler also anchors its state to a
    ///     trial of its study (<c>TrialAnchor</c>), and a state found under another study's key fails that check and is
    ///     rebuilt from the study's own history. A mutation check that keyed the state by study id alone therefore left
    ///     every generation right, at the cost of a rebuild each time the studies took turns.
    /// </remarks>
    [TestMethod]
    public void PerStudyState_ResolvedForAStudyPerShooterAtOnce_KeepsOneStatePerStudy()
    {
        const int rounds = 50;
        var perStudy = new OptunaSharp.Internal.PerStudyState<object>();
        // PerStudyState is not synchronized; its callers hold a lock of their own around Resolve, as samplers do.
        var callerLock = new object();
        Study[] studies = Enumerable.Range(0, Threads).Select(_ => Optuna.CreateStudy(sampler: new RandomSampler(5))).ToArray();
        var resolved = new object[Threads];
        object[]? firstRound = null;

        using var gun = new ConcurrencyGun(Threads);
        GunReport report = gun.Fire(rounds,
            shot =>
            {
                lock (callerLock)
                    resolved[shot.Thread] = perStudy.Resolve(studies[shot.Thread], static () => new object());
            },
            afterRound: r =>
            {
                resolved.Distinct(ReferenceEqualityComparer.Instance).Should()
                    .HaveCount(Threads, "round {0}: studies with equal ids in different storages have different states", r);
                firstRound ??= (object[])resolved.Clone();
                for (int t = 0; t < Threads; t++)
                    resolved[t].Should().BeSameAs(firstRound[t], "round {0}: study {1} keeps its state", r, t);
                return false;
            });

        report.AssertClean();
        studies.Should().OnlyContain(s => s.StudyId == 0, "the case under test: every study has the same id");
    }

    /// <summary>
    ///     One NSGA-II instance with a population of 4 serves a study per shooter at once, each study asking and telling
    ///     two trials a round for six rounds. Every study gets exactly the generation sequence of a study run alone (four
    ///     trials per generation), because the sampler keeps each study's generation bookkeeping apart, even though every
    ///     study id is 0.
    /// </summary>
    [TestMethod]
    public void OneGaSampler_ServingAStudyPerShooterAtOnce_GivesEveryStudyTheGenerationsOfASoloRun()
    {
        const int population = 4, rounds = 6, trialsPerShot = 2;
        const string generationKey = "NSGAIISampler:generation";

        var solo = Optuna.CreateStudy(sampler: new NSGAIISampler(populationSize: population, seed: 3), directions: TwoMinimized);
        for (int n = 0; n < rounds * trialsPerShot; n++)
            AskAndTell(solo, multiObjective: true);
        List<object?> soloGenerations = solo.Trials.Select(t => t.SystemAttrs.GetValueOrDefault(generationKey)).ToList();

        var shared = new NSGAIISampler(populationSize: population, seed: 3);
        Study[] studies = Enumerable.Range(0, Threads)
            .Select(_ => Optuna.CreateStudy(sampler: shared, directions: TwoMinimized))
            .ToArray();

        using var quiet = new QuietLogs();
        using var gun = new ConcurrencyGun(Threads);
        GunReport report = gun.Fire(rounds, shot =>
        {
            for (int k = 0; k < trialsPerShot; k++)
                AskAndTell(studies[shot.Thread], multiObjective: true);
        });

        report.AssertClean();
        soloGenerations.Should().NotContainNulls("the solo run records a generation for every trial");
        foreach (Study study in studies)
        {
            study.Trials.Select(t => t.SystemAttrs.GetValueOrDefault(generationKey)).Should()
                .Equal(soloGenerations, "each study keeps its own generations however its sampler is shared");
        }
    }

    /// <summary>
    ///     Every cheap Core sampler, one instance serving a study per shooter at once: in every study every trial
    ///     completes and the numbering is contiguous. The genetic samplers (population 12) reach their second generation,
    ///     so their crossover runs on each study's own parents.
    /// </summary>
    /// <param name="name">The sampler, as <see cref="SamplerNJobsSweepTests" /> builds it.</param>
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
    [DataRow("rembo")]
    public void OneSampler_ServingAStudyPerShooterAtOnce_CompletesEveryTrialOfEveryStudy(string name)
    {
        const int rounds = 8, trialsPerShot = 2;
        BaseSampler sampler = SamplerNJobsSweepTests.Make(name);
        bool multiObjective = SamplerNJobsSweepTests.MultiObjective.Contains(name);
        Study[] studies = Enumerable.Range(0, Threads)
            .Select(_ => multiObjective
                ? Optuna.CreateStudy(sampler: sampler, directions: TwoMinimized)
                : Optuna.CreateStudy(sampler: sampler))
            .ToArray();

        using var quiet = new QuietLogs();
        using var gun = new ConcurrencyGun(Threads);
        GunReport report = gun.Fire(rounds, shot =>
        {
            for (int k = 0; k < trialsPerShot; k++)
                AskAndTell(studies[shot.Thread], multiObjective);
        });

        report.AssertClean();
        foreach (Study study in studies)
        {
            var trials = study.Trials;
            trials.Select(t => t.Number).Should().Equal(Enumerable.Range(0, rounds * trialsPerShot));
            trials.Should().OnlyContain(t => t.State == TrialState.Complete, "a shared {0} completes every trial of every study", name);
        }
    }

    /// <summary>Two minimized objectives, the multi-objective studies of these tests.</summary>
    private static readonly StudyDirection[] TwoMinimized = { StudyDirection.Minimize, StudyDirection.Minimize };

    /// <summary>
    ///     Asks one trial, suggests the sweep's three parameters and tells the result, as the <c>nJobs</c> sweep's objective
    ///     does: one value, or two for a multi-objective study.
    /// </summary>
    /// <param name="study">The shooter's own study.</param>
    /// <param name="multiObjective">Whether the study minimizes two objectives.</param>
    private static void AskAndTell(Study study, bool multiObjective)
    {
        Trial trial = study.Ask();
        double x = trial.SuggestFloat("x", -10, 10);
        double y = trial.SuggestFloat("y", -10, 10);
        double z = trial.SuggestFloat("z", 0, 1);
        if (multiObjective)
            study.Tell(trial, new[] { (x - 2) * (x - 2) + y * y + z, x * x + (y - 1) * (y - 1) });
        else
            study.Tell(trial, (x - 2) * (x - 2) + (y + 1) * (y + 1) + z);
    }
}

using System;
using System.Collections.Generic;
using System.Linq;
using AwesomeAssertions;
using OptunaSharp.Distributions;
using OptunaSharp.Logging;
using OptunaSharp.Samplers;
using OptunaSharp.Tests.Samplers;

namespace OptunaSharp.Tests.Concurrency;

/// <summary>
///     Parallel reads never hand one point to two trials. For every Core sampler, rounds of eight trials ask and suggest
///     at the same instant on the same history (<see cref="ParallelReads" />), and no two trials of a round may receive
///     the same new point (a collision). A sampler that never re-proposes a finished point must not start doing so under
///     parallel reads either.
/// </summary>
/// <remarks>
///     <para>
///         Before the fixes of 2026-11-08 (<c>docs/THREADING.md</c> §6.3), measured over 8 rounds of 8 parallel reads:
///         LM-CMA handed one point to 8-10 trials, REMBO to up to 21, JANUS to 7, SAASBO to 7 (every startup trial of a
///         round took the same Sobol row, as TuRBO's did), PSO to 1-9 and DE to up to 2; NSGA-II and NSGA-III collided
///         in one run of 12 to 20, and BruteForce in Optuna's strict mode repeated 3-8 running points per 88 trials.
///         Each sampler now steers concurrent asks apart with the trials that are running: it takes an index (population
///         slot, particle, target, design row) no running trial holds, redraws a point or child a running trial holds,
///         or lies about the running trials' outcomes to its acquisition. JANUS hands each slot of a generation out once,
///         whether its trial still runs or has finished. A sequential run has no other running trial, so none of it
///         applies: seeded results, Optuna parity included, are unchanged.
///     </para>
///     <para>
///         The volleys run on eight threads and in lockstep (every read of a round before any write, each on a thread of
///         its own: the genetic samplers steer apart only children asked on different threads, QA-57). Rounds release a
///         generation together, so <see cref="ParallelReads_GenerationSpanningTwoRounds_HandsNoCandidateTwice" /> uses a
///         population larger than a round, and
///         <see cref="Optimize_SixteenWorkers_SamplerThatNeverRepeats_EvaluatesNoPointTwice" /> runs real
///         <c>Optimize(nJobs: 16)</c> calls, where trials finish at any time: a first JANUS fix passed every volley and
///         still repeated 41-65 points in 120 trials there.
///     </para>
///     <para>
///         <b>History repeats.</b> Genetic samplers, DE, JANUS-DE, PSO and LM-CMA re-propose finished points by design:
///         a child that copies a parent, a mutant clamped onto a corner, a particle at rest, a Rademacher point of an
///         earlier generation. They do so one trial at a time as well
///         (<see cref="ReadOneAtATime_SamplerThatRepeatsFinishedPoints_RepeatsThemSequentiallyToo" />), so only their
///         collisions are asserted. Every other sampler must not repeat a told trial either, except that CMA-ES and VkD-CMA
///         may repeat one point a volley through Optuna's per-trial seeding (<see cref="SeedCollisionOnce" />).
///     </para>
///     <para>
///         The record of handed-out points lives in the sampler instance, so these guarantees hold for the asks one
///         process makes through one sampler: two processes sharing a storage are steered apart only by the slot and
///         row stamps written to the storage.
///     </para>
/// </remarks>
[TestClass]
public class SamplerParallelReadTests
{
    /// <summary>Rounds per volley: 64 trials, past every sampler's startup and into its later generations.</summary>
    private const int Rounds = 8;

    /// <summary>Rounds on the 90-point discrete space: 88 trials, so the last rounds pick among the last free points.</summary>
    private const int DiscreteRounds = 11;

    /// <summary>The trials that ask at once in each round, as <see cref="ParallelReads.Fire" /> fires them by default.</summary>
    private const int Shooters = 8;

    /// <summary>
    ///     The samplers that re-propose finished points by design, one trial at a time as well (see the class remarks):
    ///     only their collisions are asserted. JANUS-DE is among them through its DE host, which clamps mutants onto the
    ///     box, so two of them can land on one corner.
    /// </summary>
    private static readonly HashSet<string> RepeatsFinishedPoints = new()
    {
        "de", "pso", "lmcma", "janusde", "moead", "nsga2", "nsga3", "nsga3cpt", "smsemoa", "agemoea2", "agemoea2cpt",
        "rvea", "janusmoo", "janussms", "janusage2", "janusage2cpt",
    };

    /// <summary>
    ///     The samplers allowed one repeated point per volley: each trial reseeds CMA-ES's sampling with
    ///     <c>randint(1, 65536) + trial.number</c>, as Optuna does, so two trials of one generation draw the same point
    ///     about once in 65,535 pairs, one trial at a time too. About 170 such pairs meet in a volley, so the chance is
    ///     about 1 in 400; the fixed seeding is kept for Optuna parity.
    /// </summary>
    private static readonly HashSet<string> SeedCollisionOnce = new() { "cmaes", "vkd" };

    /// <summary>
    ///     Every Core sampler, shared by eight trials asked at once, as <c>Optimize(nJobs: 8)</c> shares it by default:
    ///     no two trials of a round receive one new point, and a sampler that never re-proposes a finished point does not
    ///     receive one either.
    /// </summary>
    /// <param name="name">The sampler under test (see <see cref="SamplerNJobsSweepTests.Make" />).</param>
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
    [DataRow("gp")]
    [DataRow("turbo")]
    [DataRow("rembo")]
    public void ParallelReads_SharedSampler_HandsNoNewPointToTwoTrials(string name)
    {
        // Samplers warn about independent sampling while their startup fills; the warnings are not under test.
        using var quiet = new QuietLogs();
        bool multiObjective = SamplerNJobsSweepTests.MultiObjective.Contains(name);

        // On eight threads, whatever interleaving they take; then in lockstep, where every read of a round lands
        // before any write, the interleaving a sampler that waits for stored parameters handles worst. The lockstep
        // reads run on threads of their own, as concurrent workers' do: the genetic samplers steer apart only children
        // asked on different threads, and leave one caller's batch to Optuna's draws (QA-57, pinned against Optuna by
        // Run2SamplerParityBugTests.Nsga_BatchedAskTell_PartialChildIsNotRedrawn).
        ParallelReads.Outcome threaded = ParallelReads.Fire(
            SamplerNJobsSweepTests.Make(name), multiObjective, ParallelReads.Space.Continuous, Rounds);
        ParallelReads.Outcome lockstep = ParallelReads.ReadInLockstep(
            SamplerNJobsSweepTests.Make(name), multiObjective, ParallelReads.Space.Continuous, Rounds,
            readsOnOwnThreads: true);

        AssertNoDuplicates(name, threaded, Rounds);
        AssertNoDuplicates(name, lockstep, Rounds);
    }

    /// <summary>
    ///     A population larger than a round (12 against 8 trials), so a generation spans two rounds and the second round
    ///     asks while trials of the generation are finished but the generation is not yet complete. A first version of
    ///     the JANUS fix freed a slot once its trial finished and handed that evaluated candidate out again; the
    ///     population of 7 in the main test fits in one round, so it never saw this, while <c>Optimize(nJobs: 16)</c>
    ///     repeated 41-65 points in 120 trials.
    /// </summary>
    /// <param name="name">The population sampler under test, with a population of 12.</param>
    [TestMethod]
    [DataRow("janus")]
    [DataRow("janusde")]
    [DataRow("de")]
    [DataRow("pso")]
    [DataRow("lmcma")]
    public void ParallelReads_GenerationSpanningTwoRounds_HandsNoCandidateTwice(string name)
    {
        using var quiet = new QuietLogs();
        BaseSampler sampler = name switch
        {
            "janus" => new JanusSampler(seed: 1, popsize: 12),
            "janusde" => new JanusDeSampler(seed: 1, popsize: 12),
            "de" => new DifferentialEvolutionSampler(seed: 1, popsize: 12),
            "pso" => new PsoSampler(seed: 1, popsize: 12),
            _ => new LmCmaSampler(seed: 1, popsize: 12),
        };

        ParallelReads.Outcome threaded = ParallelReads.Fire(sampler, false, ParallelReads.Space.Continuous, Rounds);
        ParallelReads.Outcome lockstep = ParallelReads.ReadInLockstep(
            name switch
            {
                "janus" => new JanusSampler(seed: 1, popsize: 12),
                "janusde" => new JanusDeSampler(seed: 1, popsize: 12),
                "de" => new DifferentialEvolutionSampler(seed: 1, popsize: 12),
                "pso" => new PsoSampler(seed: 1, popsize: 12),
                _ => new LmCmaSampler(seed: 1, popsize: 12),
            }, false, ParallelReads.Space.Continuous, Rounds, readsOnOwnThreads: true);

        AssertNoDuplicates(name, threaded, Rounds);
        AssertNoDuplicates(name, lockstep, Rounds);
    }

    /// <summary>
    ///     A whole <c>Optimize(nJobs: 16)</c> run, where trials finish at any time and ask while others of their
    ///     generation are running or finished: a sampler that never re-proposes a finished point evaluates no point
    ///     twice in 120 trials (60 for the Gaussian-process samplers). JANUS and JANUS-DE repeated 41-65 points here
    ///     before each slot of a generation went to one trial only.
    /// </summary>
    /// <remarks>
    ///     xNES is left out: one trial at a time, over 100 seeds of this objective, its distribution degenerated in 6
    ///     within 120 trials. One then clamped candidates onto corners of the box and repeated them; five failed the
    ///     study with a NaN parameter or a singular matrix (found 2026-11-08, not concurrency). An interleaving acts as
    ///     another seed, so the run could repeat a corner or fail. The volleys above, 64 trials, stay clear of it.
    /// </remarks>
    /// <param name="name">The sampler under test (see <see cref="SamplerNJobsSweepTests.Make" />).</param>
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
    [DataRow("vkd")]
    [DataRow("openaies")]
    [DataRow("snes")]
    [DataRow("gp")]
    [DataRow("turbo")]
    [DataRow("rembo")]
    public void Optimize_SixteenWorkers_SamplerThatNeverRepeats_EvaluatesNoPointTwice(string name)
    {
        using var quiet = new QuietLogs();
        int trials = name is "gp" or "turbo" or "rembo" ? 60 : 120;
        var study = Optuna.CreateStudy(sampler: SamplerNJobsSweepTests.Make(name));

        study.Optimize(t =>
        {
            double x = t.SuggestFloat("x", -10, 10), y = t.SuggestFloat("y", -10, 10), z = t.SuggestFloat("z", 0, 1);
            return (x - 2) * (x - 2) + (y + 1) * (y + 1) + z;
        }, nTrials: trials, nJobs: 16);

        AssertNoPointEvaluatedTwice(name, study, trials);
    }

    /// <summary>
    ///     SAASBO's Sobol startup design: every trial of a round used to take the row the completed count names, so 7 of
    ///     8 parallel reads received one point. With 12 startup trials the second round needs rows 8 to 15, past the
    ///     design's end, and the third round is the Bayesian-optimization phase. A light NUTS budget keeps this off the
    ///     slow category; the startup rows do not depend on it.
    /// </summary>
    [TestMethod]
    public void ParallelReads_Saasbo_StartupRowsAndProposals_HandNoNewPointToTwoTrials()
    {
        using var quiet = new QuietLogs();
        var sampler = new SaasboSampler(
            seed: 1, nStartupTrials: 12, numWarmup: 16, numSamples: 16, thinning: 4, numCandidates: 64);

        ParallelReads.Outcome outcome = ParallelReads.Fire(sampler, false, ParallelReads.Space.Continuous, rounds: 3);

        AssertNoDuplicates("saasbo", outcome, rounds: 3);
    }

    /// <summary>
    ///     <c>SamplerConcurrency.PerWorker</c>: each of the eight trials samples with an instance of its own, reseeded
    ///     before the trial as <c>Optimize</c> reseeds a worker's instance, so the instances share no state and cannot
    ///     steer each other apart. Their reseeded streams must still never meet on one point.
    /// </summary>
    /// <param name="name">A sampler that makes worker instances (see <see cref="SamplerNJobsSweepTests.Make" />).</param>
    [TestMethod]
    [DataRow("random")]
    [DataRow("tpe")]
    [DataRow("tpe-independent")]
    [DataRow("cmaes")]
    [DataRow("vkd")]
    [DataRow("partialfixed")]
    public void ParallelReads_PerWorkerInstances_HandNoNewPointToTwoTrials(string name)
    {
        using var quiet = new QuietLogs();

        ParallelReads.Outcome outcome = ParallelReads.Fire(
            SamplerNJobsSweepTests.Make(name), false, ParallelReads.Space.Continuous, Rounds, perWorker: true);

        AssertNoDuplicates(name, outcome, Rounds);
    }

    /// <summary>
    ///     The grid sampler on a 90-point grid, 88 trials: every trial receives a grid point no other trial received,
    ///     as Optuna's grid sampler hands out the grid ids no finished or running trial holds.
    /// </summary>
    [TestMethod]
    public void ParallelReads_Grid_HandsEveryTrialADistinctGridPoint()
    {
        static GridSampler Grid() => new(new Dictionary<string, IReadOnlyList<CategoricalChoice>>
        {
            ["x"] = Enumerable.Range(0, 9).Select(i => CategoricalChoice.From(-10.0 + 2.5 * i)).ToList(),
            ["y"] = Enumerable.Range(0, 5).Select(i => CategoricalChoice.From((long)i)).ToList(),
            ["c"] = new[] { "a", "b" }.Select(s => CategoricalChoice.From(s)).ToList(),
        }, seed: 1);

        ParallelReads.Outcome threaded = ParallelReads.Fire(Grid(), false, ParallelReads.Space.Discrete, DiscreteRounds);
        ParallelReads.Outcome lockstep = ParallelReads.ReadInLockstep(
            Grid(), false, ParallelReads.Space.Discrete, DiscreteRounds, readsOnOwnThreads: true);

        AssertNoDuplicates("grid", threaded, DiscreteRounds);
        AssertNoDuplicates("grid", lockstep, DiscreteRounds);
    }

    /// <summary>
    ///     BruteForce on a 90-point space, 88 trials, in both modes: no trial receives a point another trial holds or
    ///     held, down to the last free points. It samples one parameter per call, so a trial still choosing counts as
    ///     taking one of the points below its path, and a branch whose points are all held is never entered.
    /// </summary>
    /// <remarks>
    ///     <b>Misaligned:</b> Optuna's strict mode (<c>avoid_premature_stop=True</c>) hands running points out again on
    ///     purpose, so a dead worker's point still gets evaluated, and both of its modes count a point as free while a
    ///     trial still choosing is about to take it. Owner's decision 2026-11-08: concurrent asks never share a point.
    ///     A sequential run is unchanged.
    /// </remarks>
    /// <param name="avoidPrematureStop">Whether the sampler runs in Optuna's strict exhaustive mode.</param>
    [TestMethod]
    [Misaligned]
    [DataRow(false)]
    [DataRow(true)]
    public void ParallelReads_BruteForce_NeverHandsOutAPointARunningTrialHolds(bool avoidPrematureStop)
    {
        ParallelReads.Outcome threaded = ParallelReads.Fire(
            new BruteForceSampler(seed: 1, avoidPrematureStop: avoidPrematureStop), false, ParallelReads.Space.Discrete,
            DiscreteRounds);
        ParallelReads.Outcome lockstep = ParallelReads.ReadInLockstep(
            new BruteForceSampler(seed: 1, avoidPrematureStop: avoidPrematureStop), false, ParallelReads.Space.Discrete,
            DiscreteRounds, readsOnOwnThreads: true);

        AssertNoDuplicates("bruteforce", threaded, DiscreteRounds);
        AssertNoDuplicates("bruteforce", lockstep, DiscreteRounds);
    }

    /// <summary>
    ///     More trials at once than the population has members: DE and PSO give each running trial its own target or
    ///     particle, and the trials left over are sampled independently on purpose, so no point is shared and no
    ///     "sampled independently" warning is logged for them (only for trials the sampler could not serve).
    /// </summary>
    /// <param name="name"><c>de</c> or <c>pso</c>, with a population of 4 against 8 trials per round.</param>
    [TestMethod]
    [DataRow("de")]
    [DataRow("pso")]
    public void ParallelReads_MoreTrialsThanPopulation_SampleTheRestIndependentlyWithoutWarning(string name)
    {
        BaseSampler sampler = name == "de"
            ? new DifferentialEvolutionSampler(seed: 1, popsize: 4)
            : new PsoSampler(seed: 1, popsize: 4);
        var messages = new List<string>();
        var handler = new CollectingHandler(messages);
        int verbosity = OptunaLogging.GetVerbosity();
        ParallelReads.Outcome outcome;
        try
        {
            OptunaLogging.SetVerbosity(LogLevel.Warning);
            OptunaLogging.AddHandler(handler);
            outcome = ParallelReads.Fire(sampler, false, ParallelReads.Space.Continuous, Rounds);
        }
        finally
        {
            OptunaLogging.RemoveHandler(handler);
            OptunaLogging.SetVerbosity(verbosity);
        }

        AssertNoDuplicates(name, outcome, Rounds);
        messages.Should().NotContain(m => m.Contains("sampled independently"),
            "the trials past the population are sampled independently on purpose");
    }

    /// <summary>
    ///     The repeats the parallel-read test allows these samplers are their own: read one trial at a time, with no
    ///     parallelism at all, the same sampler re-proposes finished points too. If one stops doing so, take it out of
    ///     <see cref="RepeatsFinishedPoints" /> so the parallel test asserts its history repeats as well.
    /// </summary>
    /// <param name="name">A sampler in <see cref="RepeatsFinishedPoints" />.</param>
    [TestMethod]
    [DataRow("de")]
    [DataRow("pso")]
    [DataRow("lmcma")]
    [DataRow("janusde")]
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
    public void ReadOneAtATime_SamplerThatRepeatsFinishedPoints_RepeatsThemSequentiallyToo(string name)
    {
        using var quiet = new QuietLogs();
        RepeatsFinishedPoints.Should().Contain(name);

        // JANUS-DE's first 64 trials at its default population of 7 happen to reach no corner twice; with 12 its DE
        // host clamps two mutants onto the corner (-10, -10, 1), at trial 36.
        BaseSampler sampler = name == "janusde"
            ? new JanusDeSampler(seed: 1, popsize: 12)
            : SamplerNJobsSweepTests.Make(name);
        ParallelReads.Outcome outcome = ParallelReads.ReadOneAtATime(
            sampler, SamplerNJobsSweepTests.MultiObjective.Contains(name), ParallelReads.Space.Continuous,
            Rounds * Shooters);

        outcome.HistoryRepeats.Should().BeGreaterThan(0,
            "the sampler re-proposes finished points by design, without any parallel read");
    }

    /// <summary>
    ///     Asserts that every trial of every round was told, that no two trials of a round received one new point, and,
    ///     unless the sampler re-proposes finished points by design, that no trial received a told trial's point. The
    ///     CMA samplers of <see cref="SeedCollisionOnce" /> may repeat one point, either way.
    /// </summary>
    /// <param name="name">The sampler's name, which selects whether history repeats are allowed.</param>
    /// <param name="outcome">What the volley produced.</param>
    /// <param name="rounds">How many rounds the volley fired.</param>
    /// <exception cref="AssertFailedException">A trial is missing, two trials of a round shared a point, or a sampler that should not repeat did.</exception>
    private static void AssertNoDuplicates(string name, ParallelReads.Outcome outcome, int rounds)
    {
        outcome.Trials.Should().Be(rounds * Shooters, "every trial of every round is asked and told");
        if (SeedCollisionOnce.Contains(name))
        {
            outcome.Duplicates.Should().BeLessThanOrEqualTo(1,
                "{0} repeats a point only when two trials' seeds collide, about once in 65,535 pairs: {1}", name, outcome);
            return;
        }

        outcome.Collisions.Should().Be(0, "no two trials read at once may receive one new point: {0}", outcome);
        if (!RepeatsFinishedPoints.Contains(name))
            outcome.HistoryRepeats.Should().Be(0, "{0} never re-proposes a told trial's point: {1}", name, outcome);
    }

    /// <summary>
    ///     Asserts that no point was evaluated twice in a whole <c>Optimize</c> run, or at most once for the CMA samplers
    ///     of <see cref="SeedCollisionOnce" />, and at most twice for JANUS-DE, whose DE host can clamp two mutants onto one
    ///     corner of the box (it does so one trial at a time too). The slot bug this pins repeated 47-65 JANUS-DE points.
    /// </summary>
    /// <param name="name">The sampler's name.</param>
    /// <param name="study">The study after the run.</param>
    /// <param name="trials">How many trials the run had.</param>
    /// <exception cref="AssertFailedException">A trial is missing or a point was evaluated more often than allowed.</exception>
    private static void AssertNoPointEvaluatedTwice(string name, Study study, int trials)
    {
        var vectors = study.Trials.Select(t => ParallelReads.VectorOf(t.Params)).ToList();
        vectors.Should().HaveCount(trials);
        string[] repeated = vectors.GroupBy(v => v, StringComparer.Ordinal).Where(g => g.Count() > 1)
            .Select(g => $"{g.Key} x{g.Count()}").ToArray();
        int duplicates = vectors.Count - vectors.Distinct(StringComparer.Ordinal).Count();
        int allowed = name == "janusde" ? 2 : SeedCollisionOnce.Contains(name) ? 1 : 0;
        duplicates.Should().BeLessThanOrEqualTo(allowed,
            "{0} evaluated these points more than once: {1}", name, string.Join("; ", repeated));
    }

    /// <summary>Collects every log message, under a lock, because the trials of a round log from their own threads.</summary>
    /// <param name="sink">The list the messages are appended to.</param>
    private sealed class CollectingHandler(List<string> sink) : ILogHandler
    {
        /// <summary>Appends the record's message.</summary>
        /// <param name="record">The record emitted.</param>
        public void Emit(in LogRecord record)
        {
            lock (sink)
                sink.Add(record.Message);
        }
    }
}

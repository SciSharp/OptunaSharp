using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Runtime.ExceptionServices;
using System.Threading;
using OptunaSharp.Samplers;

namespace OptunaSharp.Tests.Concurrency;

/// <summary>
///     Fires rounds of parallel reads at one study's sampler: in each round every shooter of a <see cref="ConcurrencyGun" />
///     asks a trial and suggests its parameters at the same instant, and only once every shooter has read does any of
///     them tell its trial. So every read of a round sees the same history (the trials of the earlier rounds), and a
///     point two shooters of one round received is a duplicate the parallel reads produced, not a copy of a finished
///     trial.
/// </summary>
/// <remarks>
///     <para>
///         Two kinds of repeated point are told apart. A <b>collision</b> is one new point handed to two or more
///         trials of the same round: the reads ran in parallel on the same history, so this is what parallel reading
///         can produce and sequential reading cannot. A <b>history repeat</b> is a point equal to a trial told in an
///         earlier round: the sampler re-proposed a finished point, which it can do sequentially as well (a genetic
///         sampler copying a parent unchanged).
///     </para>
///     <para>
///         With <c>perWorker</c> each shooter samples with an instance of its own, made by
///         <see cref="BaseSampler.CreateWorkerInstance" /> and reseeded before each of its trials, bound to the shooter's
///         thread exactly as <c>Optimize</c> binds a worker's instance (<see cref="Study.ThreadLocalStudyAttribute.SamplerOverride" />),
///         and each shooter tells its own trial on its own thread, so <c>AfterTrial</c> reaches that instance.
///     </para>
///     <para>
///         <see cref="Fire" /> runs whatever interleaving the threads take, so a sampler whose result depends on which
///         reads landed before which writes repeats a varying count. <see cref="ReadInLockstep" /> makes the one
///         interleaving where every read of a round lands before any write, in a fixed order (on the calling thread, or
///         each read on a thread of its own), so its counts repeat exactly and a test can pin them.
///     </para>
/// </remarks>
internal static class ParallelReads
{
    /// <summary>How long the shooters of a round wait for each other between reading and telling.</summary>
    private static readonly TimeSpan BarrierTimeout = TimeSpan.FromSeconds(60);

    /// <summary>The search space a volley samples.</summary>
    internal enum Space
    {
        /// <summary>Three floats: <c>x</c> and <c>y</c> in [-10, 10], <c>z</c> in [0, 1].</summary>
        Continuous,

        /// <summary>A 90-point grid: <c>x</c> in -10..10 by 2.5, <c>y</c> in 0..4, <c>c</c> in {a, b}.</summary>
        Discrete,
    }

    /// <summary>One repeated point: where it appeared and which trials received it.</summary>
    /// <param name="Round">The round in which the point was received again.</param>
    /// <param name="Vector">The parameter vector, keys in ordinal order.</param>
    /// <param name="Trials">The numbers of the trials of that round that received it.</param>
    /// <param name="InHistory">Whether a trial of an earlier round already had it (a history repeat), or not (a collision).</param>
    internal sealed record Repeat(int Round, string Vector, IReadOnlyList<int> Trials, bool InHistory);

    /// <summary>One trial's point, for a test that compares the points themselves (a parity check against Python).</summary>
    /// <param name="Number">The trial's number.</param>
    /// <param name="Params">The trial's parameters as it holds them after suggesting.</param>
    internal sealed record Point(int Number, IReadOnlyDictionary<string, object?> Params);

    /// <summary>What a volley produced.</summary>
    /// <param name="Trials">The trials asked and told.</param>
    /// <param name="Collisions">Trials beyond the first of a round that received the same new point.</param>
    /// <param name="HistoryRepeats">Trials that received a point an earlier round's trial already had.</param>
    /// <param name="Repeats">Every repeated point, for the failure message.</param>
    /// <param name="Points">Every trial's point, in round order and, within a round, in trial-number order.</param>
    internal sealed record Outcome(
        int Trials, int Collisions, int HistoryRepeats, IReadOnlyList<Repeat> Repeats, IReadOnlyList<Point> Points)
    {
        /// <summary>Every trial whose point another trial already had or received at the same time.</summary>
        internal int Duplicates => Collisions + HistoryRepeats;

        /// <summary>The repeats, one per line, for an assertion's reason.</summary>
        /// <returns>A readable list; empty when nothing repeated.</returns>
        public override string ToString()
            => string.Join("; ", Repeats.Select(r =>
                $"round {r.Round}, trials {string.Join(",", r.Trials)}: {r.Vector}{(r.InHistory ? " (already a told trial)" : " (new, received twice)")}"));
    }

    /// <summary>
    ///     Fires <paramref name="rounds" /> rounds of <paramref name="shooters" /> parallel reads at one study and
    ///     classifies every repeated point.
    /// </summary>
    /// <param name="sampler">The study's sampler; with <paramref name="perWorker" />, the prototype of the shooters' instances.</param>
    /// <param name="multiObjective">Whether the study minimizes two objectives instead of one.</param>
    /// <param name="space">The search space each trial suggests.</param>
    /// <param name="rounds">How many rounds to fire; each asks and tells <paramref name="shooters" /> trials.</param>
    /// <param name="shooters">How many trials read at once in each round.</param>
    /// <param name="perWorker">Whether each shooter samples with its own worker instance, as <c>SamplerConcurrency.PerWorker</c> does.</param>
    /// <param name="roundTimeout">
    ///     How long a round may take, and the shooters may wait for each other between reading and telling, before it
    ///     counts as hung; the gun's default when null. A sampler fitting a model per read (BoTorch) needs more.
    /// </param>
    /// <returns>The trial count and the repeated points, classified.</returns>
    /// <exception cref="InvalidOperationException">
    ///     <paramref name="perWorker" /> is set and the sampler makes no worker instance (it stays shared under
    ///     <c>PerWorker</c>, so the volley would not test what it says).
    /// </exception>
    /// <exception cref="AssertFailedException">A shot failed or a round hung (the gun's report).</exception>
    internal static Outcome Fire(BaseSampler sampler, bool multiObjective, Space space, int rounds, int shooters = 8,
        bool perWorker = false, TimeSpan? roundTimeout = null)
    {
        Study study = multiObjective
            ? Optuna.CreateStudy(sampler: sampler, directions: new[] { StudyDirection.Minimize, StudyDirection.Minimize })
            : Optuna.CreateStudy(sampler: sampler);

        BaseSampler[]? workers = null;
        if (perWorker)
        {
            workers = new BaseSampler[shooters];
            for (int i = 0; i < shooters; i++)
                workers[i] = sampler.CreateWorkerInstance()
                             ?? throw new InvalidOperationException(
                                 $"{sampler.GetType().Name} makes no worker instance; under PerWorker it stays shared.");
        }

        var tally = new Tally();
        var received = new (int Number, string Vector, IReadOnlyDictionary<string, object?> Params)[shooters];
        // Holds every shooter after its read until all have read, so no read of a round sees a trial of the same round
        // told; a broken or timed-out barrier fails the round instead of letting a late read see a told trial.
        using var readsDone = new Barrier(shooters);
        using var gun = new ConcurrencyGun(shooters, roundTimeout);
        TimeSpan barrierTimeout = roundTimeout ?? BarrierTimeout;

        GunReport report = gun.Fire(rounds, shot =>
        {
            BaseSampler? own = workers?[shot.Thread];
            if (own is not null)
            {
                // What Optimize does before a worker's trial: reseed the instance, then bind it to the thread.
                own.Locked(own.ReseedRng);
                study.ThreadLocal.SamplerOverride = own;
            }

            try
            {
                Trial trial = study.Ask();
                double[] values = Suggest(trial, space, multiObjective);
                received[shot.Thread] = (trial.Number, VectorOf(trial.Params), trial.Params);
                if (!readsDone.SignalAndWait(barrierTimeout))
                    throw new TimeoutException("The shooters of the round did not all read within the timeout.");
                study.Tell(trial, values);
            }
            finally
            {
                if (own is not null)
                    study.ThreadLocal.SamplerOverride = null;
            }
        }, afterRound: round =>
        {
            tally.Add(round, received);
            return false;
        });

        report.AssertClean();
        return tally.Result();
    }

    /// <summary>
    ///     The deterministic parallel read: in each round <paramref name="shooters" /> trials are asked, then each samples
    ///     its relative parameters in turn before any of them stores a parameter, then they suggest in turn and are told.
    ///     So every relative read of a round sees the round's other trials running with no parameters yet: the
    ///     interleaving of <see cref="Fire" /> in which every read lands before any write, made on one thread so that it
    ///     repeats exactly, for counts a test can pin.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///         Independent parameters (all of BruteForce's and Grid's, the startup draws of the others) are sampled in the
    ///         suggests, one trial after another, so each of those reads sees the earlier trials' parameters, as a sampler
    ///         monitor serializes them in <see cref="Fire" /> too. Each relative read starts from an empty history cache,
    ///         as a trial's first read on a thread of its own does.
    ///     </para>
    ///     <para>
    ///         <b>Which thread reads.</b> By default every relative read runs on the calling thread: a batch one caller asks
    ///         before telling any of it, as Optuna's batch-optimization tutorial does. The genetic samplers follow Optuna
    ///         there and do not steer the batch's children apart (QA-57). With <paramref name="readsOnOwnThreads" /> each
    ///         relative read runs on a thread of its own, started and joined in turn, so the order of the reads, and with
    ///         it the counts, stay exactly repeatable while every trial is asked for by a different thread, as concurrent
    ///         workers ask.
    ///     </para>
    /// </remarks>
    /// <param name="sampler">The study's sampler, shared by every trial.</param>
    /// <param name="multiObjective">Whether the study minimizes two objectives instead of one.</param>
    /// <param name="space">The search space each trial suggests.</param>
    /// <param name="rounds">How many rounds to read; each asks and tells <paramref name="shooters" /> trials.</param>
    /// <param name="shooters">How many trials read in each round before any of them writes.</param>
    /// <param name="readsOnOwnThreads">
    ///     Whether each trial's relative read runs on a dedicated thread (concurrent workers) rather than on the calling
    ///     thread (one caller's batch); see the remarks.
    /// </param>
    /// <returns>The trial count and the repeated points, classified as by <see cref="Fire" />.</returns>
    /// <exception cref="Exception">Whatever the sampler, a suggest or a tell throws, unchanged (a read's on its own thread too).</exception>
    internal static Outcome ReadInLockstep(
        BaseSampler sampler, bool multiObjective, Space space, int rounds, int shooters = 8,
        bool readsOnOwnThreads = false)
    {
        Study study = multiObjective
            ? Optuna.CreateStudy(sampler: sampler, directions: new[] { StudyDirection.Minimize, StudyDirection.Minimize })
            : Optuna.CreateStudy(sampler: sampler);
        var tally = new Tally();
        for (int round = 0; round < rounds; round++)
        {
            var trials = new Trial[shooters];
            for (int i = 0; i < shooters; i++)
                trials[i] = study.Ask();

            foreach (Trial trial in trials)
            {
                if (readsOnOwnThreads)
                {
                    // A fresh thread per read, joined before the next starts: the reads keep their order, and each
                    // trial's child is claimed on a thread of its own, as a concurrent worker's is. A new thread's history
                    // cache is empty, so no reset is needed. A failure is rethrown here with its original stack.
                    ExceptionDispatchInfo? failure = null;
                    var reader = new Thread(() =>
                    {
                        try
                        {
                            _ = trial.RelativeParams;
                        }
                        catch (Exception e)
                        {
                            failure = ExceptionDispatchInfo.Capture(e);
                        }
                    });
                    reader.Start();
                    reader.Join();
                    failure?.Throw();
                    continue;
                }

                // In Fire each trial reads on its own thread, whose history cache its first read fills; on this one
                // thread the previous trial's read filled it, so drop it and let this read take its own snapshot.
                study.ThreadLocal.CachedAllTrials = null;
                _ = trial.RelativeParams;
            }

            var received = new (int Number, string Vector, IReadOnlyDictionary<string, object?> Params)[shooters];
            var values = new double[shooters][];
            for (int i = 0; i < shooters; i++)
            {
                values[i] = Suggest(trials[i], space, multiObjective);
                received[i] = (trials[i].Number, VectorOf(trials[i].Params), trials[i].Params);
            }

            for (int i = 0; i < shooters; i++)
                study.Tell(trials[i], values[i]);
            tally.Add(round, received);
        }

        return tally.Result();
    }

    /// <summary>
    ///     The sequential baseline of <see cref="Fire" />: <paramref name="trials" /> reads one after another, each trial
    ///     told before the next is asked, on the calling thread. Every repeat is then a history repeat, the kind a sampler
    ///     produces without any parallelism.
    /// </summary>
    /// <param name="sampler">The study's sampler.</param>
    /// <param name="multiObjective">Whether the study minimizes two objectives instead of one.</param>
    /// <param name="space">The search space each trial suggests.</param>
    /// <param name="trials">How many trials to ask and tell.</param>
    /// <returns>The trial count and the repeated points (all history repeats).</returns>
    internal static Outcome ReadOneAtATime(BaseSampler sampler, bool multiObjective, Space space, int trials)
    {
        Study study = multiObjective
            ? Optuna.CreateStudy(sampler: sampler, directions: new[] { StudyDirection.Minimize, StudyDirection.Minimize })
            : Optuna.CreateStudy(sampler: sampler);
        var history = new HashSet<string>(StringComparer.Ordinal);
        var repeats = new List<Repeat>();
        var points = new List<Point>(trials);
        for (int i = 0; i < trials; i++)
        {
            Trial trial = study.Ask();
            double[] values = Suggest(trial, space, multiObjective);
            string vector = VectorOf(trial.Params);
            if (!history.Add(vector))
                repeats.Add(new Repeat(i, vector, new[] { trial.Number }, InHistory: true));
            points.Add(new Point(trial.Number, trial.Params));
            study.Tell(trial, values);
        }

        return new Outcome(trials, 0, repeats.Count, repeats, points);
    }

    /// <summary>Suggests the trial's parameters in <paramref name="space" /> and computes its objective values.</summary>
    /// <param name="trial">The asked trial.</param>
    /// <param name="space">The search space to suggest.</param>
    /// <param name="multiObjective">Whether to return two objective values.</param>
    /// <returns>The values to tell: one, or two for a multi-objective study.</returns>
    private static double[] Suggest(Trial trial, Space space, bool multiObjective)
    {
        if (space == Space.Discrete)
        {
            double x = trial.SuggestFloat("x", -10, 10, step: 2.5);
            long y = trial.SuggestInt("y", 0, 4);
            string c = trial.SuggestCategorical("c", new[] { "a", "b" });
            double loss = x * x + y + (c == "a" ? 0 : 1);
            return multiObjective ? new[] { loss, (x - 5) * (x - 5) + (4 - y) } : new[] { loss };
        }

        double px = trial.SuggestFloat("x", -10, 10), py = trial.SuggestFloat("y", -10, 10), pz = trial.SuggestFloat("z", 0, 1);
        return multiObjective
            ? new[] { (px - 2) * (px - 2) + py * py + pz, px * px + (py - 1) * (py - 1) }
            : new[] { (px - 2) * (px - 2) + (py + 1) * (py + 1) + pz };
    }

    /// <summary>
    ///     Classifies the points of round after round against everything received in the earlier rounds: a point received
    ///     again in a later round is a history repeat, a new point received by several trials of one round a collision.
    /// </summary>
    private sealed class Tally
    {
        private readonly HashSet<string> _history = new(StringComparer.Ordinal);
        private readonly List<Repeat> _repeats = new();
        private readonly List<Point> _points = new();
        private int _collisions, _historyRepeats, _told;

        /// <summary>Classifies one round's points, then adds them to the history the later rounds are checked against.</summary>
        /// <param name="round">The round's index.</param>
        /// <param name="received">The number, point and parameters of every trial of the round.</param>
        internal void Add(int round, (int Number, string Vector, IReadOnlyDictionary<string, object?> Params)[] received)
        {
            foreach (var group in received.GroupBy(r => r.Vector, StringComparer.Ordinal))
            {
                int[] numbers = group.Select(r => r.Number).Order().ToArray();
                bool inHistory = _history.Contains(group.Key);
                if (inHistory)
                    _historyRepeats += numbers.Length;
                else if (numbers.Length > 1)
                    _collisions += numbers.Length - 1;
                if (inHistory || numbers.Length > 1)
                    _repeats.Add(new Repeat(round, group.Key, numbers, inHistory));
            }

            foreach (var (_, vector, _) in received)
                _history.Add(vector);
            foreach (var (number, _, @params) in received.OrderBy(r => r.Number))
                _points.Add(new Point(number, @params));
            _told += received.Length;
        }

        /// <summary>What the rounds added so far produced.</summary>
        /// <returns>The trial count and the repeated points, classified.</returns>
        internal Outcome Result() => new(_told, _collisions, _historyRepeats, _repeats, _points);
    }

    /// <summary>A trial's parameters as one comparable string, keys in ordinal order and values culture-invariant.</summary>
    /// <param name="parameters">The trial's parameters.</param>
    /// <returns>The vector; equal strings mean the same point exactly.</returns>
    internal static string VectorOf(IReadOnlyDictionary<string, object?> parameters)
        => string.Join("|", parameters.OrderBy(p => p.Key, StringComparer.Ordinal)
            .Select(p => p.Key + "=" + Convert.ToString(p.Value, CultureInfo.InvariantCulture)));
}

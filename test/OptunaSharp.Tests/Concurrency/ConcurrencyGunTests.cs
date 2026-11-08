using System;
using System.Diagnostics;
using System.Linq;
using System.Threading;
using AwesomeAssertions;
using NumSharp;
using OptunaSharp.Samplers;

namespace OptunaSharp.Tests.Concurrency;

/// <summary>
///     The <see cref="ConcurrencyGun" />'s own contract, and the proof that it catches what it is built to catch. Its
///     shooters are dedicated threads of its own, halted on a <see cref="ManualResetEventSlim" /> gate and released into
///     the shot together; the callbacks run while every shooter is halted. Races in a shot are caught: a lost update in an
///     unguarded read-modify-write, and NumSharp's lock-free generator drawn without the sampler monitor, the race
///     THREADING.md §6.3 describes. A lock-order deadlock is reported with its threads, and the threads are freed. Failures
///     are reported by thread and round.
/// </summary>
/// <remarks>
///     The race-catching tests stop the volley at the first round that shows the race, so on a multi-core machine they
///     take one round. On a single logical processor they are inconclusive (<see cref="ConcurrencyGun.RequireParallelHardware" />).
///     Their controls, the same volleys with the race removed, must show nothing in every round: that is what makes a
///     caught race the shot's fault and not the gun's.
/// </remarks>
[TestClass]
public class ConcurrencyGunTests
{
    // ------------------------------------------------------------------ the gun's contract

    /// <summary>
    ///     The shooters are the gun's own threads: never pool threads, one dedicated thread per index for the gun's whole
    ///     life, named for the gun. Each round releases all of them into the shot together: every shooter is inside the
    ///     shot at once, which the shot checks with a countdown that only completes when all eight have arrived, and the
    ///     report's timestamps agree.
    /// </summary>
    [TestMethod]
    public void Shooters_AreTheGunsOwnThreads_ReleasedIntoTheShotTogether_EveryRound()
    {
        const int threads = 8, rounds = 5;
        var threadIds = new int[threads, rounds];
        var names = new string?[threads];
        var everyoneMet = new bool[threads, rounds];
        int onPool = 0, inside = 0, peakInside = 0;
        CountdownEvent everyoneInside = null!;

        using var gun = new ConcurrencyGun(threads);
        GunReport report = gun.Fire(rounds,
            shot =>
            {
                threadIds[shot.Thread, shot.Round] = Environment.CurrentManagedThreadId;
                names[shot.Thread] = Thread.CurrentThread.Name;
                if (Thread.CurrentThread.IsThreadPoolThread)
                    Interlocked.Increment(ref onPool);
                Watermark.Raise(ref peakInside, Interlocked.Increment(ref inside));
                // Every shooter waits here until all have arrived. A gun that ran the shots one after another would let
                // no shooter through, and the bounded wait would end false.
                everyoneInside.Signal();
                everyoneMet[shot.Thread, shot.Round] = everyoneInside.Wait(TimeSpan.FromSeconds(20));
                Interlocked.Decrement(ref inside);
            },
            beforeRound: _ => everyoneInside = new CountdownEvent(threads));

        Console.WriteLine(report);
        report.AssertClean();
        report.RoundsFired.Should().Be(rounds);
        onPool.Should().Be(0, "shooters are threads of the gun's own, never thread-pool threads");
        for (int t = 0; t < threads; t++)
        {
            Enumerable.Range(0, rounds).Select(r => threadIds[t, r]).Distinct().Should()
                .ContainSingle("shooter {0} is one dedicated thread for the gun's whole life", t);
        }
        Enumerable.Range(0, threads).Select(t => threadIds[t, 0]).Should().OnlyHaveUniqueItems("each index is its own thread");
        names.Should().OnlyContain(n => n != null && n.StartsWith("ConcurrencyGun shooter ", StringComparison.Ordinal));
        everyoneMet.Cast<bool>().Should().OnlyContain(met => met, "the gate releases every shooter into the shot together");
        peakInside.Should().Be(threads);
        report.PeakConcurrency.Should().Be(threads, "the shots' own timestamps show all of them running at once");
    }

    /// <summary>
    ///     <c>beforeRound</c> and <c>afterRound</c> run while every shooter is halted, which is what lets a test set up and
    ///     check shared state without locks: every shot of a round starts after <c>beforeRound</c> returned and ends before
    ///     <c>afterRound</c> starts, even when the callbacks are slow. <c>afterRound</c> returning true stops the volley.
    /// </summary>
    [TestMethod]
    public void BeforeAndAfterRound_RunWhileEveryShooterIsHalted_AndAfterRoundCanStopTheVolley()
    {
        const int threads = 4, rounds = 6, stopAfter = 3;
        var beforeEnded = new long[rounds];
        var afterStarted = new long[rounds];
        var shotStarted = new long[rounds, threads];
        var shotEnded = new long[rounds, threads];

        using var gun = new ConcurrencyGun(threads);
        GunReport report = gun.Fire(rounds,
            shot =>
            {
                shotStarted[shot.Round, shot.Thread] = Stopwatch.GetTimestamp();
                Thread.Sleep(2);
                shotEnded[shot.Round, shot.Thread] = Stopwatch.GetTimestamp();
            },
            beforeRound: r =>
            {
                // A slow setup: a shooter that was not halted would start meanwhile.
                Thread.Sleep(15);
                beforeEnded[r] = Stopwatch.GetTimestamp();
            },
            afterRound: r =>
            {
                afterStarted[r] = Stopwatch.GetTimestamp();
                Thread.Sleep(15);
                return r == stopAfter;
            });

        report.AssertClean();
        report.RoundsFired.Should().Be(stopAfter + 1, "afterRound returned true after round {0}", stopAfter);
        for (int r = 0; r <= stopAfter; r++)
        {
            for (int t = 0; t < threads; t++)
            {
                shotStarted[r, t].Should().BeGreaterThan(beforeEnded[r], "round {0}'s shots wait for beforeRound", r);
                shotEnded[r, t].Should().BeLessThanOrEqualTo(afterStarted[r], "afterRound waits for round {0}'s shots", r);
            }
        }
        Enumerable.Range(stopAfter + 1, rounds - stopAfter - 1).Should().OnlyContain(r => beforeEnded[r] == 0, "no round is fired after the stop");
    }

    /// <summary>
    ///     A shot that throws is reported with its round and thread, the other shots of its round still run, and the volley
    ///     goes on; <see cref="GunReport.AssertClean" /> then fails the test, naming the failures.
    /// </summary>
    [TestMethod]
    public void ShotsThatThrow_AreReportedByRoundAndThread_AndTheVolleyGoesOn()
    {
        const int threads = 4, rounds = 6;
        int shots = 0;

        using var gun = new ConcurrencyGun(threads);
        GunReport report = gun.Fire(rounds, shot =>
        {
            Interlocked.Increment(ref shots);
            FailOnOddRoundsOfShooter3(shot);
        });

        report.RoundsFired.Should().Be(rounds);
        shots.Should().Be(threads * rounds, "a failed shot stops neither its round nor the volley");
        report.Failures.Select(f => (f.Round, f.Thread)).Should().Equal((1, 3), (3, 3), (5, 3));
        report.Failures.Should().OnlyContain(f => f.Exception is InvalidOperationException);
        Action assertClean = report.AssertClean;
        assertClean.Should().Throw<AssertFailedException>().WithMessage("*round 1, thread 3*");
    }

    /// <summary>A gun needs two shooters to race anything, and a disposed gun fires no more; disposing twice is harmless.</summary>
    [TestMethod]
    public void Gun_RefusesFewerThanTwoShooters_AndAnyVolleyAfterDispose()
    {
        Action oneShooter = () => new ConcurrencyGun(threads: 1);
        oneShooter.Should().Throw<ArgumentOutOfRangeException>();
        Action noTimeout = () => new ConcurrencyGun(threads: 2, roundTimeout: TimeSpan.Zero);
        noTimeout.Should().Throw<ArgumentOutOfRangeException>();

        var gun = new ConcurrencyGun(threads: 2);
        gun.Dispose();
        Action fire = () => gun.Fire(1, _ => { });
        fire.Should().Throw<ObjectDisposedException>();
        gun.Dispose();
    }

    // ------------------------------------------------------------------ the gun catches races

    /// <summary>
    ///     The gun catches a classic race: eight shooters increment one counter by reading it and writing the read value
    ///     plus one. Two shooters that read the same value write the same sum, and an increment is lost. The volley stops
    ///     at the first round that lost one; on a multi-core machine that is the first round.
    /// </summary>
    /// <remarks>
    ///     This is the test that expects the gun to catch a concurrency problem. Its control,
    ///     <see cref="InterlockedIncrement_LosesNothing_InAnyRound" />, runs the same volley with an atomic increment and
    ///     must lose nothing: a lost update here is the shot's race, not a shot the gun ran twice or skipped.
    /// </remarks>
    [TestMethod]
    public void UnguardedReadModifyWrite_IsCaughtLosingUpdates()
    {
        ConcurrencyGun.RequireParallelHardware();
        const int threads = 8, increments = 20_000, maxRounds = 100;
        var cell = new Cell();
        int roundsWithLosses = 0;
        long lostUpdates = 0;

        using var gun = new ConcurrencyGun(threads);
        GunReport report = gun.Fire(maxRounds,
            _ =>
            {
                for (int k = 0; k < increments; k++)
                {
                    // Check, then act: the read and the write are separate steps another shooter can come between.
                    int seen = Volatile.Read(ref cell.Value);
                    Volatile.Write(ref cell.Value, seen + 1);
                }
            },
            afterRound: _ =>
            {
                int lost = threads * increments - cell.Value;
                if (lost > 0)
                {
                    roundsWithLosses++;
                    lostUpdates += lost;
                }
                cell.Value = 0;
                return roundsWithLosses > 0;
            });

        Console.WriteLine($"{report}; {lostUpdates} increments lost in {roundsWithLosses} of {report.RoundsFired} rounds");
        report.AssertClean();
        roundsWithLosses.Should().BeGreaterThan(0, "eight shooters released together race on an unguarded read-modify-write ({0})", report);
    }

    /// <summary>
    ///     The control of <see cref="UnguardedReadModifyWrite_IsCaughtLosingUpdates" />: the same volley with
    ///     <see cref="Interlocked.Increment(ref int)" /> loses nothing in any round, so every shot ran exactly once and the
    ///     gun's own bookkeeping is sound.
    /// </summary>
    [TestMethod]
    public void InterlockedIncrement_LosesNothing_InAnyRound()
    {
        const int threads = 8, increments = 20_000, rounds = 20;
        var cell = new Cell();
        var totals = new int[rounds];

        using var gun = new ConcurrencyGun(threads);
        GunReport report = gun.Fire(rounds,
            _ =>
            {
                for (int k = 0; k < increments; k++)
                    Interlocked.Increment(ref cell.Value);
            },
            afterRound: r =>
            {
                totals[r] = cell.Value;
                cell.Value = 0;
                return false;
            });

        report.AssertClean();
        totals.Should().OnlyContain(total => total == threads * increments);
    }

    /// <summary>
    ///     The gun catches the race the sampler monitor exists for (THREADING.md §6.3): NumSharp's MT19937 generator takes
    ///     no lock, where numpy's does. Eight shooters drawing from one sampler's generator without
    ///     <see cref="BaseSampler" />'s monitor tear its state: two shooters read the same key word and draw the same value,
    ///     or the position runs past the key and a draw throws. Both count as caught, and the volley stops at the first
    ///     round that shows either.
    /// </summary>
    /// <remarks>
    ///     Draws of a whole generator are 53-bit doubles, and 40,000 of them repeat a value with probability about 1e-7, so
    ///     a repeat is the tear, not chance. Its control is
    ///     <see cref="NumSharpGenerator_DrawnUnderTheSamplerMonitor_NeverRepeatsOrThrows" />.
    /// </remarks>
    [TestMethod]
    public void NumSharpGenerator_DrawnWithoutTheSamplerMonitor_IsCaughtTearing()
    {
        ConcurrencyGun.RequireParallelHardware();
        const int threads = 8, draws = 5_000, maxRounds = 50;
        var sampler = new RandomSampler(seed: 7);
        NumPyRandom rng = sampler.Rng.Rng;
        double[][] drawn = NewDrawBuffers(threads, draws);
        int failedShots = 0, repeatedDraws = 0;

        using var gun = new ConcurrencyGun(threads);
        GunReport report = gun.Fire(maxRounds,
            shot =>
            {
                double[] mine = drawn[shot.Thread];
                try
                {
                    // No monitor: what every sampler did before BaseSampler.SyncRoot existed.
                    for (int k = 0; k < draws; k++)
                        mine[k] = (double)rng.rand();
                }
                catch (Exception)
                {
                    // A torn position runs past the key; that is the race caught too.
                    Interlocked.Increment(ref failedShots);
                }
            },
            beforeRound: _ => ClearDrawBuffers(drawn),
            afterRound: _ =>
            {
                repeatedDraws += RepeatedDraws(drawn);
                return repeatedDraws > 0 || failedShots > 0;
            });

        Console.WriteLine($"{report}; {repeatedDraws} repeated draws, {failedShots} failed shots");
        (repeatedDraws + failedShots).Should().BeGreaterThan(0, "an unguarded MT19937 drawn by eight shooters at once tears ({0})", report);
    }

    /// <summary>
    ///     The control of <see cref="NumSharpGenerator_DrawnWithoutTheSamplerMonitor_IsCaughtTearing" />, and the
    ///     production path: the same volley with every draw taken under the sampler's monitor
    ///     (<c>BaseSampler.Locked</c>, as <c>Trial</c> and the <c>nJobs</c> workers call it) never repeats a value and never
    ///     throws, in any round.
    /// </summary>
    [TestMethod]
    public void NumSharpGenerator_DrawnUnderTheSamplerMonitor_NeverRepeatsOrThrows()
    {
        const int threads = 8, draws = 2_000, rounds = 10;
        var sampler = new RandomSampler(seed: 7);
        NumPyRandom rng = sampler.Rng.Rng;
        double[][] drawn = NewDrawBuffers(threads, draws);
        int repeatedDraws = 0;

        using var gun = new ConcurrencyGun(threads);
        GunReport report = gun.Fire(rounds,
            shot =>
            {
                double[] mine = drawn[shot.Thread];
                for (int k = 0; k < draws; k++)
                    mine[k] = sampler.Locked(() => (double)rng.rand());
            },
            beforeRound: _ => ClearDrawBuffers(drawn),
            afterRound: _ =>
            {
                repeatedDraws += RepeatedDraws(drawn);
                return false;
            });

        report.AssertClean();
        repeatedDraws.Should().Be(0, "the monitor admits one draw at a time");
    }

    // ------------------------------------------------------------------ the gun catches deadlocks

    /// <summary>
    ///     A lock-order inversion, the deadlock THREADING.md §5.2's lock order prevents, is reported instead of hanging the
    ///     run. Shooter 0 takes <c>a</c> then <c>b</c>, shooter 1 takes <c>b</c> then <c>a</c>, and neither takes its second
    ///     lock before both hold their first, so the round deadlocks. After the round timeout the report names the round
    ///     and both shooters, interrupting them frees both (each records <see cref="ThreadInterruptedException" />), and the
    ///     spent gun refuses another volley.
    /// </summary>
    [TestMethod]
    public void LockOrderInversion_IsReportedAsAHang_AndItsShootersAreFreed()
    {
        object a = new(), b = new();
        CountdownEvent bothHoldTheirFirstLock = null!;

        using var gun = new ConcurrencyGun(threads: 2, roundTimeout: TimeSpan.FromSeconds(2));
        GunReport report = gun.Fire(rounds: 3,
            shot =>
            {
                object first = shot.Thread == 0 ? a : b;
                object second = shot.Thread == 0 ? b : a;
                lock (first)
                {
                    // Makes the inversion certain rather than likely: each shooter holds its first lock before either
                    // reaches for its second.
                    bothHoldTheirFirstLock.Signal();
                    bothHoldTheirFirstLock.Wait(TimeSpan.FromSeconds(10));
                    lock (second)
                    {
                    }
                }
            },
            beforeRound: _ => bothHoldTheirFirstLock = new CountdownEvent(2));

        report.Hung.Should().BeTrue();
        report.HungRound.Should().Be(0);
        report.RoundsFired.Should().Be(1, "a hang ends the volley");
        report.HungThreads.Should().Equal(0, 1);
        report.RecoveredFromHang.Should().BeTrue("an interrupt ends a wait for a monitor");
        report.Failures.Should().HaveCount(2).And.OnlyContain(f => f.Exception is ThreadInterruptedException);
        Action assertClean = report.AssertClean;
        assertClean.Should().Throw<AssertFailedException>().WithMessage("*Round 0 hung with shooters 0, 1*");
        Action fireAgain = () => gun.Fire(1, _ => { });
        fireAgain.Should().Throw<InvalidOperationException>().WithMessage("*spent*");
    }

    // ------------------------------------------------------------------ helpers

    /// <summary>A shared integer the race tests increment; a field, so shooters can pass it by reference.</summary>
    private sealed class Cell
    {
        /// <summary>The counter.</summary>
        public int Value;
    }

    /// <summary>The failing shot of <see cref="ShotsThatThrow_AreReportedByRoundAndThread_AndTheVolleyGoesOn" />.</summary>
    /// <param name="shot">The shot's thread and round.</param>
    /// <exception cref="InvalidOperationException">On shooter 3 in every odd round, naming the thread and round.</exception>
    private static void FailOnOddRoundsOfShooter3(Shot shot)
    {
        if (shot.Thread == 3 && shot.Round % 2 == 1)
            throw new InvalidOperationException($"thread {shot.Thread}, round {shot.Round}");
    }

    /// <summary>One draw buffer per shooter, so shooters never write the same array.</summary>
    /// <param name="threads">The shooter count.</param>
    /// <param name="draws">The draws each shooter takes per round.</param>
    /// <returns>The buffers.</returns>
    private static double[][] NewDrawBuffers(int threads, int draws)
        => Enumerable.Range(0, threads).Select(_ => new double[draws]).ToArray();

    /// <summary>Fills every buffer with NaN, so a shot that threw part-way leaves no stale value of an earlier round.</summary>
    /// <param name="drawn">The shooters' buffers.</param>
    private static void ClearDrawBuffers(double[][] drawn)
    {
        foreach (double[] buffer in drawn)
            Array.Fill(buffer, double.NaN);
    }

    /// <summary>How many draws of one round repeat an earlier draw of that round, over every shooter's buffer.</summary>
    /// <param name="drawn">The shooters' buffers; NaN marks a draw that never happened.</param>
    /// <returns>The number of repeated values; 0 for a generator that stayed whole.</returns>
    private static int RepeatedDraws(double[][] drawn)
    {
        var values = drawn.SelectMany(buffer => buffer).Where(v => !double.IsNaN(v)).ToList();
        return values.Count - values.Distinct().Count();
    }
}

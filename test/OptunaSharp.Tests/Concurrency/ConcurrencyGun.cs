using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading;

namespace OptunaSharp.Tests.Concurrency;

/// <summary>
///     Fires a fixed set of dedicated threads into the same code at the same instant, round after round, so that the
///     interleavings a race needs happen in a test instead of in production. In every round each shooter thread halts on
///     the round's <see cref="ManualResetEventSlim" /> gate with a blocking <c>Wait</c> (never <c>await</c>); once all of
///     them are parked the gate opens, they line up, and all of them run the same shot together.
/// </summary>
/// <remarks>
///     <para>
///         <b>Why a gun, not <c>Parallel.For</c> or <c>Task.Run</c>.</b> Pool threads start whenever the pool gets to
///         them, often one after another, so two of them rarely meet inside a short critical section, and a test written
///         that way passes on code that races. Every shooter here is a thread of the gun's own, already blocked at the gate
///         when the round starts, so the shots begin within microseconds of each other (see
///         <see cref="GunReport.MedianReleaseSpread" />), and they do so every round: hundreds of rounds turn a race window
///         of nanoseconds into one a test hits.
///     </para>
///     <para>
///         <b>The lineup.</b> Opening the gate wakes the shooters one kernel wake-up after another, tens of microseconds
///         apart. Each woken shooter therefore spins until every shooter is awake, and only then runs its shot. The spin
///         uses <see cref="SpinWait" />, which yields and sleeps as it goes, so a machine with fewer cores than shooters
///         still makes progress; it only lines the shots up less tightly.
///     </para>
///     <para>
///         <b>Hangs.</b> A round that does not finish within the round timeout is reported, never waited on forever: the
///         shooters still inside the shot are interrupted (<see cref="Thread.Interrupt" />), which ends a wait for a lock, a
///         sleep or a join with <see cref="ThreadInterruptedException" />, and the gun is spent. A deadlocked test therefore
///         fails with the threads named instead of hanging the run. A shooter that spins forever or blocks in native code
///         cannot be interrupted: it stays stuck as a background thread, which never keeps the process alive.
///     </para>
///     <para>
///         <b>Use.</b> One gun per test, disposed when the test ends. <c>beforeRound</c> and <c>afterRound</c> run on the
///         thread that called <see cref="Fire" /> while every shooter is halted, so they may set up and check shared state
///         without locks: the gate and the round's countdowns order every write of a round before the next callback reads
///         it. A test that <em>expects</em> a race to show should first call <see cref="RequireParallelHardware" />: on one
///         core, two shooters never truly run at the same time.
///     </para>
/// </remarks>
internal sealed class ConcurrencyGun : IDisposable
{
    /// <summary>The shooter count a test gets when it does not choose one: enough to contend on any machine.</summary>
    internal const int DefaultThreads = 8;

    /// <summary>
    ///     How long one round may take before the gun calls it hung. Generous on purpose: a busy CI runner can stall a
    ///     healthy round for seconds, and a false hang is a flaky test.
    /// </summary>
    internal static readonly TimeSpan DefaultRoundTimeout = TimeSpan.FromSeconds(30);

    // How long interrupted shooters get to leave a hung shot before the report says they did not recover.
    private static readonly TimeSpan RecoveryTimeout = TimeSpan.FromSeconds(5);

    // How long Dispose waits for each shooter to end; a shooter still stuck in a hung shot is left behind.
    private static readonly TimeSpan JoinTimeout = TimeSpan.FromSeconds(5);

    private readonly Thread[] _shooters;
    private readonly TimeSpan _roundTimeout;

    // The rounds handed to the shooters: one copy per shooter per round. Never disposed: a shooter recovering from a hung
    // shot may still reach Take after Dispose, and an ObjectDisposedException on a background thread would end the test
    // host. It holds no kernel handle, so leaving it to the garbage collector costs nothing.
    private readonly BlockingCollection<Round> _magazine = new(new ConcurrentQueue<Round>());

    // Exceptions that ended a shooter outside its shot: harness faults, kept for the next Fire to report.
    private readonly ConcurrentQueue<Exception> _harnessFaults = new();

    // Set once a round hung or stalled, or a shooter died: its shooters may be stuck, so no further round is fired.
    private volatile bool _spent;

    private bool _disposed;

    /// <summary>
    ///     Allocates <paramref name="threads" /> dedicated shooter threads and starts them; each waits for a round, and
    ///     the same thread serves the same index in every round and every volley of this gun.
    /// </summary>
    /// <param name="threads">How many threads enter the shot together; at least 2.</param>
    /// <param name="roundTimeout">
    ///     How long one round may take before the gun reports it hung; null takes <see cref="DefaultRoundTimeout" />. A test
    ///     that expects a deadlock passes a short one, so it fails fast.
    /// </param>
    /// <exception cref="ArgumentOutOfRangeException">
    ///     <paramref name="threads" /> is below 2, or <paramref name="roundTimeout" /> is zero or negative.
    /// </exception>
    internal ConcurrencyGun(int threads = DefaultThreads, TimeSpan? roundTimeout = null)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(threads, 2);
        _roundTimeout = roundTimeout ?? DefaultRoundTimeout;
        ArgumentOutOfRangeException.ThrowIfLessThanOrEqual(_roundTimeout, TimeSpan.Zero, nameof(roundTimeout));

        _shooters = new Thread[threads];
        for (int index = 0; index < threads; index++)
        {
            _shooters[index] = new Thread(Shoot)
            {
                // A shooter stuck in a hung shot must never keep the test host alive.
                IsBackground = true,
                Name = $"ConcurrencyGun shooter {index}",
            };
            _shooters[index].Start(index);
        }
    }

    /// <summary>How many shooters run each shot together.</summary>
    internal int Threads => _shooters.Length;

    /// <summary>
    ///     Marks the calling test inconclusive on a machine with one logical processor. Call it first in a test that
    ///     <em>expects</em> the gun to catch a race: on one core two shooters never run at the same instant, so a race
    ///     shows only when the scheduler happens to switch threads inside the window, and its absence proves nothing.
    /// </summary>
    /// <exception cref="AssertInconclusiveException">The machine has a single logical processor.</exception>
    internal static void RequireParallelHardware()
    {
        if (Environment.ProcessorCount < 2)
            Assert.Inconclusive("Catching a race needs two threads running at the same instant: this machine has one logical processor.");
    }

    /// <summary>
    ///     Fires up to <paramref name="rounds" /> rounds. In each, every shooter runs <paramref name="shot" /> once, all of
    ///     them starting together. An exception a shot throws is recorded with its thread and round, and the volley goes on.
    /// </summary>
    /// <param name="rounds">The most rounds to fire; <paramref name="afterRound" /> can stop the volley earlier.</param>
    /// <param name="shot">The code every shooter runs each round, told its thread index and the round number.</param>
    /// <param name="beforeRound">
    ///     Runs on the calling thread before each round, given the round number, while every shooter is halted: set up the
    ///     round's shared state here (a fresh trial, a fresh countdown).
    /// </param>
    /// <param name="afterRound">
    ///     Runs on the calling thread after each round, given the round number, while every shooter is halted: check the
    ///     round's outcome here. Returning <c>true</c> stops the volley, for example once a race was caught.
    /// </param>
    /// <returns>What happened: failures by thread and round, a hang, and how closely the shots started together.</returns>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="rounds" /> is zero or negative.</exception>
    /// <exception cref="ArgumentNullException"><paramref name="shot" /> is null.</exception>
    /// <exception cref="ObjectDisposedException">The gun was disposed.</exception>
    /// <exception cref="InvalidOperationException">
    ///     An earlier round of this gun hung or stalled, or a shooter failed outside its shot: some shooters may still be
    ///     stuck, so the gun refuses to fire again. Use a new gun.
    /// </exception>
    /// <exception cref="Exception">
    ///     Whatever <paramref name="beforeRound" /> or <paramref name="afterRound" /> throws. No round is in flight at that
    ///     point, so the gun stays usable.
    /// </exception>
    /// <remarks>
    ///     Not thread-safe: one volley at a time, from one thread, which is how a test uses it. A hung round ends the
    ///     volley: the report then names the round and the threads still inside the shot, and says whether interrupting
    ///     them freed them.
    /// </remarks>
    internal GunReport Fire(int rounds, Action<Shot> shot, Action<int>? beforeRound = null, Func<int, bool>? afterRound = null)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(rounds);
        ArgumentNullException.ThrowIfNull(shot);
        ThrowIfSpent();

        var report = new GunReport(Threads);
        for (int number = 0; number < rounds; number++)
        {
            beforeRound?.Invoke(number);

            var round = new Round(number, Threads, shot);
            // One copy per shooter. Each shooter takes exactly one: a shooter that took a copy stays halted at the gate,
            // and the gate opens only once every copy was taken, so no shooter can come back for a second copy.
            for (int index = 0; index < Threads; index++)
                _magazine.Add(round);

            // Every shooter must be halted at the gate before it opens; a shooter that arrived late would start its shot
            // after the others, and the round would not be simultaneous.
            if (!round.Parked.Wait(_roundTimeout))
            {
                Stall(round);
                report.RecordStall(number);
                break;
            }

            // Fire: one Set releases every parked shooter.
            round.Gate.Set();

            if (!round.Finished.Wait(_roundTimeout))
            {
                _spent = true;
                int[] hung = round.UnfinishedThreads();
                // Ends a wait for a monitor, a sleep or a join inside the shot with ThreadInterruptedException, which the
                // round records as that shot's failure; the shooter then returns to the magazine.
                foreach (int index in hung)
                    _shooters[index].Interrupt();
                bool recovered = round.Finished.Wait(RecoveryTimeout);
                report.RecordRound(number, round.StartTicks, round.EndTicks, round.Errors);
                report.RecordHang(number, hung, recovered);
                break;
            }

            report.RecordRound(number, round.StartTicks, round.EndTicks, round.Errors);
            if (afterRound?.Invoke(number) == true)
                break;
        }

        return report;
    }

    /// <summary>
    ///     Stops the shooters and waits briefly for each to end. Shooters still stuck in a hung shot are left behind as
    ///     background threads. Calling it again does nothing.
    /// </summary>
    public void Dispose()
    {
        if (_disposed)
            return;
        _disposed = true;

        // Ends every shooter's wait for the next round once the magazine is empty.
        _magazine.CompleteAdding();
        foreach (Thread shooter in _shooters)
            shooter.Join(JoinTimeout);
    }

    /// <summary>
    ///     Gives up on a round in which some shooter never reached the gate: removes the copies nobody took, so no shooter
    ///     picks one up later, and releases the parked shooters without running the shot.
    /// </summary>
    /// <param name="round">The round that stalled before it was fired.</param>
    private void Stall(Round round)
    {
        _spent = true;
        while (_magazine.TryTake(out _))
        {
        }

        round.Cancelled = true;
        round.Gate.Set();
    }

    /// <summary>Refuses to fire a gun whose shooters may be stuck or gone.</summary>
    /// <exception cref="InvalidOperationException">An earlier round hung or stalled, or a shooter failed outside its shot.</exception>
    private void ThrowIfSpent()
    {
        if (!_spent)
            return;

        string faults = _harnessFaults.IsEmpty
            ? "an earlier round hung or stalled"
            : "a shooter failed outside its shot: " + string.Join("; ", _harnessFaults.Select(e => e.GetType().Name + ": " + e.Message));
        throw new InvalidOperationException($"This gun is spent ({faults}); its shooters may still be stuck. Use a new gun.");
    }

    /// <summary>
    ///     A shooter thread: runs every round handed to it, as the shooter of index <paramref name="state" />, until the
    ///     gun is disposed.
    /// </summary>
    /// <param name="state">The shooter's index, boxed.</param>
    private void Shoot(object? state)
    {
        int index = (int)state!;
        try
        {
            foreach (Round round in _magazine.GetConsumingEnumerable())
                round.Run(index);
        }
        catch (ThreadInterruptedException)
        {
            // An interrupt meant for a hung shot can arrive after that shot ended, at this shooter's next wait. The gun is
            // spent by then, so the shooter simply ends.
        }
        catch (Exception e)
        {
            // Never let a harness fault end the test host: keep it, and refuse further volleys, whose rounds would stall
            // without this shooter.
            _harnessFaults.Enqueue(e);
            _spent = true;
        }
    }

    /// <summary>
    ///     One round: the gate the shooters halt on, the countdowns the gun waits on, and what every shot of the round
    ///     recorded. A new one per round, so a straggler of an earlier round can never be released by a later gate.
    /// </summary>
    /// <remarks>
    ///     The events are never disposed: a shooter may still be inside <c>Signal</c> or <c>Set</c> when the gun's wait
    ///     returns, and disposing under it would throw on that shooter. They hold no kernel handle, since nothing reads
    ///     their wait handles, so the garbage collector reclaims them.
    /// </remarks>
    private sealed class Round
    {
        private readonly int _threads;
        private readonly Action<Shot> _shot;

        // How many shooters woke from the gate and reached the lineup.
        private int _awake;

        /// <summary>Prepares round <paramref name="number" /> for <paramref name="threads" /> shooters.</summary>
        /// <param name="number">The round number, from 0.</param>
        /// <param name="threads">How many shooters take part.</param>
        /// <param name="shot">The code every shooter runs.</param>
        internal Round(int number, int threads, Action<Shot> shot)
        {
            Number = number;
            _threads = threads;
            _shot = shot;
            Parked = new CountdownEvent(threads);
            Finished = new CountdownEvent(threads);
            StartTicks = new long[threads];
            EndTicks = new long[threads];
            Errors = new Exception?[threads];
        }

        /// <summary>The round number, from 0.</summary>
        internal int Number { get; }

        /// <summary>The gate every shooter halts on until the gun fires; one <c>Set</c> releases them all.</summary>
        internal ManualResetEventSlim Gate { get; } = new(false);

        /// <summary>Reaches zero once every shooter is halted at the gate.</summary>
        internal CountdownEvent Parked { get; }

        /// <summary>Reaches zero once every shooter has finished its shot, normally or not.</summary>
        internal CountdownEvent Finished { get; }

        /// <summary><see cref="Stopwatch" /> timestamp at which each shooter started its shot; 0 while it has not.</summary>
        internal long[] StartTicks { get; }

        /// <summary><see cref="Stopwatch" /> timestamp at which each shooter finished its shot; 0 while it has not.</summary>
        internal long[] EndTicks { get; }

        /// <summary>The exception each shooter's shot threw, or null.</summary>
        internal Exception?[] Errors { get; }

        /// <summary>Set when the round stalled before it was fired: released shooters skip the shot.</summary>
        internal volatile bool Cancelled;

        /// <summary>
        ///     Runs the round on shooter <paramref name="thread" />: parks it at the gate, then, once the gate opens and
        ///     every shooter is awake, runs the shot and records its outcome. The shot's exception is recorded, never thrown.
        /// </summary>
        /// <param name="thread">The shooter's index.</param>
        /// <exception cref="ThreadInterruptedException">
        ///     An interrupt meant for an earlier hung shot arrived while this shooter waited at the gate; the gun is spent
        ///     by then.
        /// </exception>
        internal void Run(int thread)
        {
            Parked.Signal();
            // The halt: a blocking wait, so the shooter is parked in the kernel until the gun fires.
            Gate.Wait();
            if (Cancelled)
                return;

            try
            {
                LineUp();
                Volatile.Write(ref StartTicks[thread], Stopwatch.GetTimestamp());
                _shot(new Shot(thread, Number));
            }
            catch (Exception e)
            {
                // The shot's own failure, or ThreadInterruptedException from the gun ending a hung shot; either way it
                // belongs to the report, and this shooter must still count as finished.
                Errors[thread] = e;
            }
            finally
            {
                Volatile.Write(ref EndTicks[thread], Stopwatch.GetTimestamp());
                Finished.Signal();
            }
        }

        /// <summary>The shooters that have not finished their shot: the hung ones, once the round timed out.</summary>
        /// <returns>Their indices, ascending.</returns>
        internal int[] UnfinishedThreads()
            => Enumerable.Range(0, _threads).Where(i => Volatile.Read(ref EndTicks[i]) == 0).ToArray();

        /// <summary>
        ///     Spins until every shooter of the round woke from the gate, so the shots start together instead of one kernel
        ///     wake-up apart.
        /// </summary>
        /// <exception cref="ThreadInterruptedException">The gun interrupted this shooter while <see cref="SpinWait" /> yielded.</exception>
        /// <remarks>
        ///     <c>SpinOnce(sleep1Threshold: -1)</c>: the default spin falls back to <c>Thread.Sleep(1)</c>, which lasts a
        ///     timer tick, up to 15.6 ms on Windows. Shooters that slept that long started their shots 11–15 ms apart, longer
        ///     than most shots last, so they barely overlapped. Without it the spinner busy-spins and then only yields
        ///     (<c>Thread.Sleep(0)</c>, <c>Thread.Yield</c>), which still hands a busy core to a shooter that has yet to wake.
        /// </remarks>
        private void LineUp()
        {
            Interlocked.Increment(ref _awake);
            var spinner = new SpinWait();
            while (Volatile.Read(ref _awake) < _threads)
                spinner.SpinOnce(sleep1Threshold: -1);
        }
    }
}

/// <summary>Where one shot runs: which shooter, and in which round.</summary>
/// <param name="Thread">
///     The shooter's index, from 0 to the gun's thread count minus 1. The same index is always the same dedicated thread,
///     so a test may keep per-index state in an array without locks.
/// </param>
/// <param name="Round">The round number, from 0.</param>
internal readonly record struct Shot(int Thread, int Round);

/// <summary>A shot that threw: in which round, on which shooter, and what.</summary>
/// <param name="Round">The round number, from 0.</param>
/// <param name="Thread">The shooter's index.</param>
/// <param name="Exception">What the shot threw; <see cref="ThreadInterruptedException" /> for a shot the gun ended because the round hung.</param>
internal sealed record ShotFailure(int Round, int Thread, Exception Exception);

/// <summary>
///     What a volley of a <see cref="ConcurrencyGun" /> did: the shots that failed, a hang, and how closely the shots of each
///     round started together. <see cref="AssertClean" /> fails a test on any failure or hang.
/// </summary>
internal sealed class GunReport
{
    private readonly List<ShotFailure> _failures = new();
    private readonly List<TimeSpan> _spreads = new();

    /// <summary>Starts an empty report for a gun of <paramref name="threads" /> shooters.</summary>
    /// <param name="threads">The gun's shooter count.</param>
    internal GunReport(int threads) => Threads = threads;

    /// <summary>The gun's shooter count.</summary>
    internal int Threads { get; }

    /// <summary>How many rounds were fired, a hung one included.</summary>
    internal int RoundsFired { get; private set; }

    /// <summary>Every shot that threw, in round order and then thread order.</summary>
    internal IReadOnlyList<ShotFailure> Failures => _failures;

    /// <summary>The round that did not finish within the round timeout, or null.</summary>
    internal int? HungRound { get; private set; }

    /// <summary>Whether a round hung, which ended the volley.</summary>
    internal bool Hung => HungRound is not null;

    /// <summary>The shooters still inside the shot when the round timed out.</summary>
    internal IReadOnlyList<int> HungThreads { get; private set; } = Array.Empty<int>();

    /// <summary>
    ///     Whether every hung shooter left the shot once interrupted. False when a shooter spins, or waits in native code,
    ///     where an interrupt cannot reach it.
    /// </summary>
    internal bool RecoveredFromHang { get; private set; }

    /// <summary>The round in which some shooter never reached the gate (a harness fault), or null.</summary>
    internal int? StalledRound { get; private set; }

    /// <summary>
    ///     The most shots that were running at one instant, over every round, from the shots' start and end timestamps. A
    ///     shot too short to overlap its peers (a few nanoseconds) can lower it; a working gun reaches the shooter count for
    ///     shots that last a few milliseconds.
    /// </summary>
    internal int PeakConcurrency { get; private set; }

    /// <summary>
    ///     The median, over the rounds, of the time between the first and the last shooter starting its shot. Diagnostic
    ///     only: it depends on the machine and its load, so tests print it rather than bound it.
    /// </summary>
    internal TimeSpan MedianReleaseSpread
        => _spreads.Count == 0 ? TimeSpan.Zero : _spreads.OrderBy(s => s).ElementAt(_spreads.Count / 2);

    /// <summary>The largest release spread of any round.</summary>
    internal TimeSpan MaxReleaseSpread => _spreads.Count == 0 ? TimeSpan.Zero : _spreads.Max();

    /// <summary>Adds one round's outcome: its failures, its release spread and its peak concurrency.</summary>
    /// <param name="number">The round number.</param>
    /// <param name="startTicks">Each shooter's shot start timestamp; 0 for a shot that never started.</param>
    /// <param name="endTicks">Each shooter's shot end timestamp; 0 for a shot that never ended.</param>
    /// <param name="errors">Each shooter's exception, or null.</param>
    internal void RecordRound(int number, long[] startTicks, long[] endTicks, Exception?[] errors)
    {
        RoundsFired = number + 1;
        for (int thread = 0; thread < errors.Length; thread++)
        {
            Exception? error = Volatile.Read(ref errors[thread]);
            if (error is not null)
                _failures.Add(new ShotFailure(number, thread, error));
        }

        // Element by element through Volatile.Read: after a hang a recovering shooter may still be writing its stamps, and a
        // volatile read of a long is atomic even in a 32-bit process.
        long[] starts = Enumerable.Range(0, startTicks.Length)
            .Select(i => Volatile.Read(ref startTicks[i]))
            .Where(t => t != 0)
            .ToArray();
        // A spread over some shooters only would understate it; a round where a shot never started has none.
        if (starts.Length == Threads)
            _spreads.Add(Stopwatch.GetElapsedTime(starts.Min(), starts.Max()));

        PeakConcurrency = Math.Max(PeakConcurrency, Peak(startTicks, endTicks));
    }

    /// <summary>Records the round that hung and what interrupting its shooters achieved.</summary>
    /// <param name="number">The hung round.</param>
    /// <param name="threads">The shooters still inside the shot at the timeout.</param>
    /// <param name="recovered">Whether all of them left the shot after the interrupt.</param>
    internal void RecordHang(int number, int[] threads, bool recovered)
    {
        HungRound = number;
        HungThreads = threads;
        RecoveredFromHang = recovered;
    }

    /// <summary>Records the round in which some shooter never reached the gate.</summary>
    /// <param name="number">The stalled round; it was never fired.</param>
    internal void RecordStall(int number) => StalledRound = number;

    /// <summary>
    ///     Fails the test unless every shot of every round returned normally: lists the failures by round and thread (the
    ///     first ten, with their exceptions), the hang and the stall.
    /// </summary>
    /// <exception cref="AssertFailedException">A shot threw, a round hung, or a round stalled.</exception>
    internal void AssertClean()
    {
        if (_failures.Count == 0 && !Hung && StalledRound is null)
            return;

        var text = new StringBuilder("The volley was not clean: ").Append(this).AppendLine();
        if (Hung)
        {
            text.Append($"Round {HungRound} hung with shooters {string.Join(", ", HungThreads)} still inside the shot; ")
                .AppendLine(RecoveredFromHang ? "interrupting them freed them." : "interrupting them did not free them.");
        }
        if (StalledRound is not null)
            text.AppendLine($"Round {StalledRound} stalled: a shooter never reached the gate.");
        foreach (ShotFailure failure in _failures.Take(10))
            text.AppendLine($"round {failure.Round}, thread {failure.Thread}: {failure.Exception}");
        if (_failures.Count > 10)
            text.AppendLine($"... and {_failures.Count - 10} more failures.");
        Assert.Fail(text.ToString());
    }

    /// <summary>A one-line summary for test output: shooters, rounds, failures, peak concurrency and release spreads.</summary>
    /// <returns>The summary.</returns>
    public override string ToString()
        => $"{Threads} shooters, {RoundsFired} rounds, {_failures.Count} failed shots, hung: {(Hung ? $"round {HungRound}" : "no")}, " +
           $"peak concurrency {PeakConcurrency}, release spread median {Micro(MedianReleaseSpread)} / max {Micro(MaxReleaseSpread)}";

    /// <summary>
    ///     The most intervals [start, end) that overlap at one instant: a sweep over the sorted endpoints, where an end
    ///     sorts before a start at the same timestamp, so touching shots do not count as overlapping.
    /// </summary>
    /// <param name="startTicks">The shots' start timestamps; 0 for a shot that never started.</param>
    /// <param name="endTicks">The shots' end timestamps; 0 for a shot that never ended.</param>
    /// <returns>The peak number of shots running at once; 0 when no shot ran.</returns>
    private static int Peak(long[] startTicks, long[] endTicks)
    {
        var edges = new List<(long Time, int Delta)>();
        for (int i = 0; i < startTicks.Length; i++)
        {
            long start = Volatile.Read(ref startTicks[i]), end = Volatile.Read(ref endTicks[i]);
            if (start == 0 || end == 0)
                continue;
            edges.Add((start, +1));
            edges.Add((end, -1));
        }

        int running = 0, peak = 0;
        foreach (var (_, delta) in edges.OrderBy(e => e.Time).ThenBy(e => e.Delta))
        {
            running += delta;
            peak = Math.Max(peak, running);
        }
        return peak;
    }

    /// <summary>Formats a duration in microseconds with one decimal.</summary>
    /// <param name="span">The duration.</param>
    /// <returns>For example <c>12.3 µs</c>.</returns>
    private static string Micro(TimeSpan span) => $"{span.TotalMilliseconds * 1000:0.0} µs";
}

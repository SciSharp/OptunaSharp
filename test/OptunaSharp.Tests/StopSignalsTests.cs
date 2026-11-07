using System;
using System.Linq;
using System.Runtime.InteropServices;
using System.Threading;
using System.Threading.Tasks;
using AwesomeAssertions;
using OptunaSharp.Samplers;

namespace OptunaSharp.Tests;

/// <summary>
///     <see cref="StopSignals" /> (plan 10, item 2.4, the owner's choice of 2026-11-07): the first SIGINT or SIGTERM is
///     cancelled and cancels the token, a later one is left to its default action (which ends the process), and
///     <see cref="StopSignals.Dispose" /> unhooks both handlers.
/// </summary>
/// <remarks>
///     Most cases drive the handler directly with a <see cref="PosixSignalContext" />: on Windows a real Ctrl+C reaches
///     every process on the console, the test runner included. The real-signal case sends SIGTERM to this process on
///     Linux and macOS only, where the runtime delivers it to the registered handler alone.
/// </remarks>
[TestClass]
[DoNotParallelize]
public class StopSignalsTests
{
    /// <summary>Listening registers both handlers on this platform without an error, and the token starts uncancelled.</summary>
    [TestMethod]
    public void Listen_Registers_AndTheTokenStartsUncancelled()
    {
        using var signals = StopSignals.Listen();

        signals.Token.CanBeCanceled.Should().BeTrue();
        signals.Token.IsCancellationRequested.Should().BeFalse();
    }

    /// <summary>
    ///     The first signal keeps the process alive and cancels the token; the second, of either kind, is left to its
    ///     default action, the way out when the trials in flight take too long.
    /// </summary>
    /// <param name="first">The first signal.</param>
    /// <param name="second">The second signal.</param>
    [TestMethod]
    [DataRow(PosixSignal.SIGINT, PosixSignal.SIGINT)]
    [DataRow(PosixSignal.SIGTERM, PosixSignal.SIGINT)]
    [DataRow(PosixSignal.SIGINT, PosixSignal.SIGTERM)]
    public void FirstSignal_IsHeldAndCancelsTheToken_SecondFallsThrough(PosixSignal first, PosixSignal second)
    {
        using var signals = StopSignals.Listen();
        var firstContext = new PosixSignalContext(first);
        var secondContext = new PosixSignalContext(second);

        signals.OnSignal(firstContext);
        firstContext.Cancel.Should().BeTrue("the first signal must not end the process");
        signals.Token.IsCancellationRequested.Should().BeTrue();

        signals.OnSignal(secondContext);
        secondContext.Cancel.Should().BeFalse("the second signal takes its default action");
    }

    /// <summary>After disposal a signal takes its default action again, the token stays readable, and Dispose is idempotent.</summary>
    [TestMethod]
    public void AfterDispose_SignalsFallThrough_AndTheTokenStaysReadable()
    {
        var signals = StopSignals.Listen();
        signals.Dispose();
        signals.Dispose();

        var context = new PosixSignalContext(PosixSignal.SIGINT);
        signals.OnSignal(context);

        context.Cancel.Should().BeFalse();
        signals.Token.IsCancellationRequested.Should().BeFalse("a signal after Dispose does not cancel");
    }

    /// <summary>
    ///     A callback registered on the token that throws does not escape the handler, which runs on the runtime's signal
    ///     thread and must not take the process down; the token is still cancelled.
    /// </summary>
    [TestMethod]
    public void ACallbackThatThrows_DoesNotEscapeTheHandler()
    {
        using var signals = StopSignals.Listen();
        using var registration = signals.Token.Register(FailingCallback);
        var context = new PosixSignalContext(PosixSignal.SIGTERM);

        var act = () => signals.OnSignal(context);

        act.Should().NotThrow();
        context.Cancel.Should().BeTrue();
        signals.Token.IsCancellationRequested.Should().BeTrue();
    }

    /// <summary>A token callback that fails, as a careless application callback might.</summary>
    /// <exception cref="InvalidOperationException">Always.</exception>
    private static void FailingCallback() => throw new InvalidOperationException("callback failed");

    /// <summary>
    ///     End to end through <c>Optimize</c>: a signal during a run with four workers stops the scheduling, the trials
    ///     in flight finish, and the call throws carrying the listener's token.
    /// </summary>
    [TestMethod]
    public void ASignalDuringOptimize_StopsTheRunGracefully()
    {
        using var signals = StopSignals.Listen();
        var study = Optuna.CreateStudy(sampler: new RandomSampler(seed: 0));
        int started = 0;

        var act = () => study.Optimize(t =>
        {
            if (Interlocked.Increment(ref started) == 8)
                Task.Run(() => signals.OnSignal(new PosixSignalContext(PosixSignal.SIGINT)));
            Thread.Sleep(20);
            return t.SuggestFloat("x", 0, 1);
        }, nTrials: 500, nJobs: 4, ct: signals.Token);

        act.Should().Throw<OperationCanceledException>().Which.CancellationToken.Should().Be(signals.Token);
        study.Trials.Should().NotContain(t => t.State == TrialState.Running);
        study.Trials.Count.Should().BeLessThan(500);
    }

    /// <summary>
    ///     A real SIGTERM sent to this process reaches the registered handler: the process lives on and the token is
    ///     cancelled. Linux and macOS only; Windows has no way to send a console event to one process alone.
    /// </summary>
    [TestMethod]
    public void ARealSigterm_CancelsTheToken_AndTheProcessLivesOn()
    {
        if (OperatingSystem.IsWindows())
            Assert.Inconclusive("Windows cannot send a console control event to this process alone; the handler cases cover it.");

        using var signals = StopSignals.Listen();
        int result = kill(Environment.ProcessId, 15);   // SIGTERM

        result.Should().Be(0, "kill(2) accepts the signal");
        signals.Token.WaitHandle.WaitOne(TimeSpan.FromSeconds(10)).Should().BeTrue("the runtime hands the signal to the handler");
    }

    /// <summary>libc's <c>kill(2)</c>, to send a real signal to this process.</summary>
    /// <param name="pid">The process to signal.</param>
    /// <param name="signal">The signal number (15 is SIGTERM on Linux and macOS).</param>
    /// <returns>0 on success, -1 on failure.</returns>
    [DllImport("libc", SetLastError = true)]
    private static extern int kill(int pid, int signal);
}

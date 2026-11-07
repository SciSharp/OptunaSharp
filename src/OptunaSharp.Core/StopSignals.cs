using System;
using System.Runtime.InteropServices;
using System.Threading;
using OptunaSharp.Logging;

namespace OptunaSharp;

/// <summary>
///     Turns the first Ctrl+C (SIGINT) or SIGTERM the process receives into a cancelled <see cref="Token" />, so that a
///     console program stops <c>Study.Optimize</c> gracefully: no trial starts after the signal, the trials in flight
///     finish and are stored, and <c>Optimize</c> throws <see cref="OperationCanceledException" />. A second signal ends
///     the process as it would without this class. OptunaSharp only, and opt-in: OptunaSharp never installs a handler by
///     itself, because a signal handler is process-wide.
/// </summary>
/// <example>
///     <code>
///     using var signals = StopSignals.Listen();
///     study.Optimize(Objective, nTrials: 500, nJobs: 4, ct: signals.Token);
///     </code>
/// </example>
/// <remarks>
///     <para>
///         <b>Why.</b> By default .NET ends the process at Ctrl+C or SIGTERM, and the trials in flight stay RUNNING in a
///         persistent storage: a journal never fails them, and an RDB storage with heartbeats fails them later as stale.
///         Python Optuna raises <c>KeyboardInterrupt</c> instead; at <c>n_jobs</c> above 1 its trials in flight finish
///         first, and at <c>n_jobs</c> 1 the trial the interrupt lands in is stored FAIL (THREADING.md §3.4). With this
///         class the trials in flight finish, and an objective that observes <see cref="Trial.CancellationToken" /> is
///         stored FAIL, as in Python.
///     </para>
///     <para>
///         <b>Cost.</b> None until a signal arrives: the handlers are operating-system callbacks registered through
///         <see cref="PosixSignalRegistration" />. Measured on 2026-11-07 over 10 idle seconds, a process used the same
///         CPU with and without them (Windows creates the handler thread only when the signal comes; Linux adds one
///         runtime thread asleep in a blocking read).
///     </para>
///     <para>
///         <b>Containers.</b> <c>docker stop</c> and Kubernetes send SIGTERM, then kill the process after a grace period
///         (10 s and 30 s by default): the trials in flight must finish within it. On Windows the runtime raises SIGTERM
///         for the console's shutdown events.
///     </para>
///     <para>
///         <b>Hosts.</b> A .NET Generic Host already handles both signals; pass its
///         <c>IHostApplicationLifetime.ApplicationStopping</c> token to <c>Optimize</c> instead of using this class.
///     </para>
///     <para>
///         <b>Lifetime.</b> <see cref="Dispose" /> unhooks the handlers, after which a signal takes its default action
///         again; the token stays valid and keeps its state. Several instances may listen at once: the first signal
///         cancels every token, and the second ends the process.
///     </para>
/// </remarks>
public sealed class StopSignals : IDisposable
{
    private static readonly Logger _logger = OptunaLogging.GetLogger("optuna.stop_signals");

    // Never disposed: a handler running on the runtime's signal thread may cancel it after Dispose has unhooked the
    // registrations, and a source without a timer or linked tokens holds nothing that needs releasing. Leaving it alive
    // also keeps Token readable after Dispose, which a disposed source would refuse with ObjectDisposedException.
    private readonly CancellationTokenSource _source = new();
    private readonly PosixSignalRegistration[] _registrations;
    private int _disposed;

    /// <summary>Registers the SIGINT and SIGTERM handlers.</summary>
    /// <exception cref="PlatformNotSupportedException">The platform has no signal handling (a browser runtime).</exception>
    /// <exception cref="System.IO.IOException">The operating system refuses the registration.</exception>
    private StopSignals()
    {
        var sigint = PosixSignalRegistration.Create(PosixSignal.SIGINT, OnSignal);
        try
        {
            _registrations = new[] { sigint, PosixSignalRegistration.Create(PosixSignal.SIGTERM, OnSignal) };
        }
        catch
        {
            // Never leave a half-installed listener behind: the caller gets the exception and no SIGINT handler that
            // nothing will ever dispose.
            sigint.Dispose();
            throw;
        }
    }

    /// <summary>
    ///     Starts listening for SIGINT (Ctrl+C) and SIGTERM. Dispose the result, usually with <c>using</c>, to stop.
    /// </summary>
    /// <returns>A listener whose <see cref="Token" /> is cancelled by the first signal.</returns>
    /// <exception cref="PlatformNotSupportedException">The platform has no signal handling (a browser runtime).</exception>
    /// <exception cref="System.IO.IOException">The operating system refuses the registration.</exception>
    public static StopSignals Listen() => new();

    /// <summary>
    ///     Cancelled by the first SIGINT or SIGTERM after <see cref="Listen" />; pass it as <c>Optimize</c>'s <c>ct</c>.
    /// </summary>
    /// <remarks>Readable after <see cref="Dispose" />, which leaves its state unchanged.</remarks>
    public CancellationToken Token => _source.Token;

    /// <summary>
    ///     The handler of both signals: the first one is cancelled (the process keeps running) and cancels
    ///     <see cref="Token" />; any later one is left to its default action, which ends the process.
    /// </summary>
    /// <param name="context">The runtime's context for this signal; setting its <c>Cancel</c> keeps the process alive.</param>
    /// <remarks>
    ///     Internal so that tests can drive it with a <see cref="PosixSignalContext" /> of their own: a real Ctrl+C
    ///     cannot be sent to a single process on Windows without reaching the whole console.
    /// </remarks>
    internal void OnSignal(PosixSignalContext context)
    {
        // A second signal falls through to the default action: the way out when the trials in flight take too long.
        if (Volatile.Read(ref _disposed) != 0 || _source.IsCancellationRequested)
            return;

        context.Cancel = true;
        _logger.Warning(
            $"Received {context.Signal}: no new trial starts, and the trials in flight finish before Optimize stops. " +
            "Send the signal again (Ctrl+C) to end the process now.");
        try
        {
            _source.Cancel();
        }
        catch (AggregateException e)
        {
            // Callbacks registered on the token run here, on the signal's thread; one that throws must not take the
            // process down from inside the handler that was meant to keep it alive.
            _logger.Warning("A callback registered on the stop token failed.", e);
        }
    }

    /// <summary>Unhooks both handlers, so that the next signal takes its default action. Calling it again does nothing.</summary>
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
            return;
        foreach (var registration in _registrations)
            registration.Dispose();
    }
}

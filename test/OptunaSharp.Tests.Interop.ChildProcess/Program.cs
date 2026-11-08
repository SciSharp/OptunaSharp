using System;
using System.Linq;
using System.Threading;
using OptunaSharp.Distributions;
using OptunaSharp.Interop.PythonNet;
using OptunaSharp.Logging;
using OptunaSharp.Samplers;
using Python.Runtime;

namespace OptunaSharp.Tests.Interop.ChildProcess;

/// <summary>
///     The contract between <c>GilAwareMonitorTests</c> and this child process: the modes it accepts, the line it prints once
///     a mode has finished, and its usage exit code. The parent compiles these constants in through its project reference,
///     so the two sides cannot drift apart.
/// </summary>
public static class ChildProcessProtocol
{
    /// <summary>
    ///     No sub-interpreter; runs all three scenarios. It passes on fixed and unfixed code alike, which shows that a failure
    ///     of another mode comes from the sub-interpreter, not from the scenario.
    /// </summary>
    public const string Control = "control";

    /// <summary>
    ///     A sub-interpreter, then <c>Optimize(nJobs: 2)</c>: the caller's submit loop waits for its workers through
    ///     <c>MonitorGate.Wait</c>, whose hook decides whether the waiting thread holds the GIL.
    /// </summary>
    public const string SubInterpreterOptimize = "subinterp-optimize";

    /// <summary>
    ///     A sub-interpreter, then a thread that waits for a sampler monitor another thread holds: the contended path of
    ///     <c>MonitorGate.Enter</c>'s hook, which decides whether the waiting thread holds the GIL.
    /// </summary>
    public const string SubInterpreterMonitor = "subinterp-monitor";

    /// <summary>
    ///     A sub-interpreter, then a <c>PythonRuntime.Run</c> call from the main thread: the hand-off to OptunaSharp's Python
    ///     thread, which decides whether the caller holds the GIL before it waits.
    /// </summary>
    public const string SubInterpreterBridge = "subinterp-bridge";

    /// <summary>
    ///     Printed on standard output once every scenario of the mode has finished. The parent requires it together with exit
    ///     code 0: a CPython fatal error, an unhandled exception and a usage error all end the process without it.
    /// </summary>
    public const string DoneMarker = "child-process: done";

    /// <summary>The exit code for a missing or unknown mode; it differs from a crash's codes, so a mistyped mode is plain.</summary>
    public const int UsageExitCode = 2;
}

/// <summary>
///     Runs one <see cref="ChildProcessProtocol" /> mode in a process of its own, for <c>GilAwareMonitorTests</c> (catalog
///     Open item 28).
/// </summary>
/// <remarks>
///     <para>
///         <b>What is pinned.</b> Once any CPython sub-interpreter has existed in a process, CPython's
///         <c>PyGILState_Check()</c> answers 1 on every thread, whether the thread holds the GIL or not (measured on 3.11.15,
///         3.12.12, 3.13.7, 3.14.2 and free-threaded 3.14.2; it stays so after the sub-interpreter is destroyed).
///         OptunaSharp's GIL-aware hooks used to trust that answer: a thread that held no GIL called
///         <c>PyEval_SaveThread</c> to release it, and the process ended, with "Fatal Python error: PyEval_SaveThread: … the
///         current Python thread state is NULL" on 3.12 and an access violation in <c>PyEval_SaveThread</c> on 3.14. The
///         hooks now read the thread's own CPython thread state instead.
///     </para>
///     <para>
///         <b>Why a process of its own.</b> The sub-interpreter changes CPython for the rest of the process, and the bug ends
///         the process. Inside the interop suite, either would take every later test with it.
///     </para>
/// </remarks>
internal static class Program
{
    /// <summary>
    ///     The longest any one wait inside a scenario may take before the scenario counts as hung. Fixed runs take well under
    ///     a second; the parent's own timeout is longer, so a hang is reported from here, with its cause.
    /// </summary>
    private static readonly TimeSpan Watchdog = TimeSpan.FromSeconds(60);

    /// <summary>Runs the mode named by the only argument, then exits.</summary>
    /// <param name="args">Exactly one argument: a mode of <see cref="ChildProcessProtocol" />.</param>
    /// <returns>
    ///     0 once the mode has finished, after printing <see cref="ChildProcessProtocol.DoneMarker" />; or
    ///     <see cref="ChildProcessProtocol.UsageExitCode" /> for a missing or unknown mode.
    /// </returns>
    /// <exception cref="PythonInteropException">No usable libpython could be started; the parent passes <c>PYTHONNET_PYDLL</c>.</exception>
    /// <exception cref="PythonException">The sub-interpreter could not be created, or the bridge call failed in Python.</exception>
    /// <exception cref="InvalidOperationException">A scenario ended with the wrong result, or hung past <see cref="Watchdog" />.</exception>
    /// <remarks>
    ///     Exceptions are left unhandled on purpose: the runtime prints them to standard error and exits non-zero, which the
    ///     parent quotes. The bug this process pins throws nothing at all: CPython prints its fatal error and aborts.
    /// </remarks>
    private static int Main(string[] args)
    {
        string? mode = args.Length == 1 ? args[0] : null;
        Action? scenario = mode switch
        {
            ChildProcessProtocol.Control => () =>
            {
                OptimizeWithTwoWorkers();
                ContendForASamplerMonitor();
                CallTheBridgeFromTheMainThread();
            },
            ChildProcessProtocol.SubInterpreterOptimize => OptimizeWithTwoWorkers,
            ChildProcessProtocol.SubInterpreterMonitor => ContendForASamplerMonitor,
            ChildProcessProtocol.SubInterpreterBridge => CallTheBridgeFromTheMainThread,
            _ => null,
        };
        if (scenario is null)
        {
            Console.Error.WriteLine(
                "usage: dotnet OptunaSharp.Tests.Interop.ChildProcess.dll " +
                $"<{ChildProcessProtocol.Control}|{ChildProcessProtocol.SubInterpreterOptimize}|" +
                $"{ChildProcessProtocol.SubInterpreterMonitor}|{ChildProcessProtocol.SubInterpreterBridge}>");
            return ChildProcessProtocol.UsageExitCode;
        }

        // Warnings only: the parent quotes this process's output when a mode fails, and per-trial INFO lines would bury the
        // cause.
        OptunaLogging.SetVerbosity(LogLevel.Warning);

        // Starts the engine on OptunaSharp's Python thread and installs the GIL-aware hooks into Core, as the first use of
        // any bridge does. This call runs before any sub-interpreter exists, so it is safe on fixed and unfixed code.
        string version = PythonRuntime.Run(() => PythonEngine.Version.Split(' ')[0]);
        Console.WriteLine($"child-process: CPython {version}, mode {mode}");

        if (mode != ChildProcessProtocol.Control)
            CreateAndDestroyASubInterpreter();

        scenario();
        Console.WriteLine(ChildProcessProtocol.DoneMarker);
        return 0;
    }

    /// <summary>
    ///     Creates a CPython sub-interpreter and destroys it again. From then on CPython's <c>PyGILState_Check()</c> answers 1
    ///     on every thread of this process, which is the condition of catalog Open item 28.
    /// </summary>
    /// <exception cref="PythonException">
    ///     Neither <c>_interpreters</c> (3.13 and later) nor <c>_xxsubinterpreters</c> (3.12 and earlier) can be imported, or
    ///     the sub-interpreter cannot be created or destroyed.
    /// </exception>
    /// <remarks>
    ///     The body runs on OptunaSharp's Python thread. The main thread decides whether it holds the GIL before the body
    ///     runs, while CPython's check still answers correctly, so this call itself is safe on fixed and unfixed code.
    /// </remarks>
    private static void CreateAndDestroyASubInterpreter()
    {
        PythonRuntime.Run(() => PythonEngine.Exec(
            "try:\n" +
            "    import _interpreters as _subinterpreters\n" +
            "except ImportError:\n" +
            "    import _xxsubinterpreters as _subinterpreters\n" +
            "_subinterpreters.destroy(_subinterpreters.create())\n"));
        Console.WriteLine("child-process: created and destroyed a sub-interpreter");
    }

    /// <summary>
    ///     Runs <c>Optimize(nTrials: 6, nJobs: 2)</c> over a managed sampler. The main thread holds no GIL. With two trials in
    ///     flight, its submit loop waits for one to finish through <c>MonitorGate.Wait</c>, whose hook first decides whether
    ///     this thread holds the GIL.
    /// </summary>
    /// <exception cref="InvalidOperationException">The study did not end with six COMPLETE trials.</exception>
    private static void OptimizeWithTwoWorkers()
    {
        var study = Optuna.CreateStudy(sampler: new RandomSampler(seed: 0));
        study.Optimize(trial => trial.SuggestFloat("x", 0, 1), nTrials: 6, nJobs: 2);

        int complete = study.Trials.Count(t => t.State == TrialState.Complete);
        if (study.Trials.Count != 6 || complete != 6)
            throw new InvalidOperationException(
                $"Optimize(nJobs: 2) ended with {study.Trials.Count} trials, {complete} of them COMPLETE; expected 6 and 6.");
        Console.WriteLine("child-process: Optimize(nJobs: 2) completed 6 trials");
    }

    /// <summary>
    ///     Makes a thread wait for a sampler monitor that another thread holds, so the wait takes the hook's contended path:
    ///     <c>Monitor.TryEnter</c> fails, and the hook then decides whether the waiting thread holds the GIL.
    /// </summary>
    /// <exception cref="InvalidOperationException">
    ///     A thread threw, the holder never parked, the waiter never blocked on the monitor, a thread hung past
    ///     <see cref="Watchdog" />, or the study did not end with two COMPLETE trials.
    /// </exception>
    /// <remarks>
    ///     The holder parks inside <c>SampleIndependent</c>, under the sampler monitor. The waiter then asks for a trial, whose
    ///     <c>BeforeTrial</c> needs that monitor; it is the first lock the waiter's <c>Ask</c> finds taken. The holder is
    ///     released only once the waiter has been blocked for a while, so the wait really is contended.
    /// </remarks>
    private static void ContendForASamplerMonitor()
    {
        var sampler = new ParkingSampler(seed: 1);
        var study = Optuna.CreateStudy(sampler: sampler);
        Exception? holderError = null, waiterError = null;

        Thread holder = StartThread("monitor holder", () => AskAndTell(study), e => holderError = e);
        if (!sampler.Parked.Wait(Watchdog))
            throw new InvalidOperationException("The holder never parked inside the sampler monitor.");

        Thread waiter = StartThread("monitor waiter", () => AskAndTell(study), e => waiterError = e);
        bool contended = WaitUntilBlocked(waiter);
        // Released whatever happened, so a failed check still ends both threads instead of leaving the holder parked.
        sampler.Release.Set();

        if (!holder.Join(Watchdog) || !waiter.Join(Watchdog))
            throw new InvalidOperationException("The monitor holder or the monitor waiter hung.");
        if (holderError is not null || waiterError is not null)
            throw new InvalidOperationException($"The holder threw: {holderError}{Environment.NewLine}The waiter threw: {waiterError}");
        if (!contended)
            throw new InvalidOperationException("The waiter never blocked on the sampler monitor, so the contended path did not run.");

        int complete = study.Trials.Count(t => t.State == TrialState.Complete);
        if (study.Trials.Count != 2 || complete != 2)
            throw new InvalidOperationException(
                $"The contended run ended with {study.Trials.Count} trials, {complete} of them COMPLETE; expected 2 and 2.");
        Console.WriteLine("child-process: a contended sampler monitor was waited for, and both trials completed");
    }

    /// <summary>
    ///     Makes one bridge call from the main thread, which holds no GIL: <c>PythonRuntime.Run</c> hands the body to
    ///     OptunaSharp's Python thread and first decides whether this thread holds the GIL, to release it while it waits.
    /// </summary>
    /// <exception cref="InvalidOperationException">The body returned the wrong value.</exception>
    /// <exception cref="PythonException">The body failed in Python.</exception>
    private static void CallTheBridgeFromTheMainThread()
    {
        long sum = PythonRuntime.Run(() =>
        {
            // Disposed inside the body: a PyObject may only be released under the GIL.
            using PyObject result = PythonEngine.Eval("sum(range(10))");
            return result.As<long>();
        });
        if (sum != 45)
            throw new InvalidOperationException($"The bridge call returned {sum}; expected 45.");
        Console.WriteLine("child-process: a bridge call from the main thread returned");
    }

    /// <summary>Asks for a trial, suggests one parameter and tells the trial its value.</summary>
    /// <param name="study">The study to ask and tell.</param>
    /// <exception cref="Exception">Whatever the study, its sampler or its storage throws.</exception>
    private static void AskAndTell(Study study)
    {
        Trial trial = study.Ask();
        double x = trial.SuggestFloat("x", 0, 1);
        study.Tell(trial, x);
    }

    /// <summary>Starts a background thread that runs <paramref name="body" /> and reports its exception instead of crashing.</summary>
    /// <param name="name">The thread's name, shown in a dump of a hung run.</param>
    /// <param name="body">The work to run.</param>
    /// <param name="onError">Receives the exception <paramref name="body" /> threw; read it only after joining the thread.</param>
    /// <returns>The started thread.</returns>
    /// <remarks>
    ///     A background thread never keeps this process alive, so a hung scenario ends with the process's exit instead of
    ///     blocking the parent until its own timeout.
    /// </remarks>
    private static Thread StartThread(string name, Action body, Action<Exception> onError)
    {
        var thread = new Thread(() =>
        {
            try
            {
                body();
            }
            catch (Exception e)
            {
                onError(e);
            }
        }) { IsBackground = true, Name = name };
        thread.Start();
        return thread;
    }

    /// <summary>
    ///     Waits until <paramref name="thread" /> has been blocked (<see cref="ThreadState.WaitSleepJoin" />) for 100 ms in a
    ///     row, which only a real wait on the monitor lasts.
    /// </summary>
    /// <param name="thread">The thread expected to block on the sampler monitor.</param>
    /// <returns>True once the thread has stayed blocked for 100 ms; false if it ended first or <see cref="Watchdog" /> passed.</returns>
    /// <remarks>
    ///     A thread blocked in <c>Monitor.Enter</c> reports <see cref="ThreadState.WaitSleepJoin" />. Requiring 100 ms keeps a
    ///     brief block elsewhere on the way to the monitor from releasing the holder too early, which would let the waiter
    ///     pass uncontended and the scenario pass without testing anything.
    /// </remarks>
    private static bool WaitUntilBlocked(Thread thread)
    {
        DateTime deadline = DateTime.UtcNow + Watchdog;
        DateTime? blockedSince = null;
        while (DateTime.UtcNow < deadline && thread.IsAlive)
        {
            if ((thread.ThreadState & ThreadState.WaitSleepJoin) != 0)
            {
                blockedSince ??= DateTime.UtcNow;
                if (DateTime.UtcNow - blockedSince.Value >= TimeSpan.FromMilliseconds(100))
                    return true;
            }
            else
            {
                blockedSince = null;
            }

            Thread.Sleep(5);
        }

        return false;
    }

    /// <summary>
    ///     A <see cref="RandomSampler" /> whose first <c>SampleIndependent</c> holds the sampler monitor until
    ///     <see cref="Release" /> is set; later calls sample at once.
    /// </summary>
    /// <param name="seed">The random sampler's seed.</param>
    private sealed class ParkingSampler(int seed) : RandomSampler(seed)
    {
        // 1 until the first call parks; Interlocked so exactly one call parks even if two threads arrive together.
        private int _parkOnce = 1;

        /// <summary>Set once the first call holds the sampler monitor.</summary>
        internal ManualResetEventSlim Parked { get; } = new();

        /// <summary>Set to let the parked call go on and release the monitor.</summary>
        internal ManualResetEventSlim Release { get; } = new();

        /// <summary>Parks the first call under the sampler monitor (bounded by <see cref="Watchdog" />), then samples.</summary>
        /// <param name="study">The study being optimized.</param>
        /// <param name="trial">The trial being sampled.</param>
        /// <param name="paramName">The parameter's name.</param>
        /// <param name="paramDistribution">The parameter's distribution.</param>
        /// <returns>The value <see cref="RandomSampler" /> samples.</returns>
        /// <exception cref="TimeoutException">The parked call was not released within <see cref="Watchdog" />.</exception>
        public override object? SampleIndependent(
            Study study, FrozenTrial trial, string paramName, BaseDistribution paramDistribution)
        {
            if (Interlocked.Exchange(ref _parkOnce, 0) == 1)
            {
                Parked.Set();
                if (!Release.Wait(Watchdog))
                    throw new TimeoutException("The parked SampleIndependent call was never released.");
            }

            return base.SampleIndependent(study, trial, paramName, paramDistribution);
        }
    }
}

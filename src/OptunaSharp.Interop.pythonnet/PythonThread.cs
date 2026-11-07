using System;
using System.Collections.Concurrent;
using System.Runtime.ExceptionServices;
using System.Threading;
using Python.Runtime;

namespace OptunaSharp.Interop.PythonNet;

/// <summary>
///     The one thread on which <see cref="PythonRuntime" /> runs every Python call of OptunaSharp's bridges (OptunaHub,
///     BoTorch, PyCma, Pymoo): a long-lived background thread that owns a single CPython thread state for the life of the
///     process and has a 16 MB stack. Callers on any other thread hand it their body and wait for it.
/// </summary>
/// <remarks>
///     <para>
///         <b>Why (plan 10, item 2.5; review item 19b).</b> On a thread CPython did not create, pythonnet's
///         <c>Py.GIL()</c> creates a CPython thread state when the outermost scope opens and frees it when that scope
///         ends. pybind11 extensions remember the thread state of the thread on which their process-wide internals load,
///         and use it again later, after it was freed. Measured on 2026-11-07: <c>PymooSampler</c> off the main thread
///         crashed with <c>0xC0000005</c> in scipy's pybind11 <c>cdist</c> (pymoo imports <c>scipy.spatial</c> lazily) in
///         7 of 8 runs, on a 16 MB thread too, and when built and driven on one thread; a bare probe failed 5 of 5 with
///         scipy (a crash) and 5 of 5 with torch (a hang). With one thread state kept for the thread's life, every run
///         passed. The main thread and Python's own threads keep theirs, which is why Python never sees this.
///     </para>
///     <para>
///         <b>Why one thread</b> (the owner's choice of 2026-11-07, over keeping a state per caller thread or an owner
///         thread per sampler): no thread state is ever freed, so no library can keep a dangling one; nothing leaks when
///         .NET threads end; thread-local Python state (torch's grad mode, CUDA's current device, numpy's error state)
///         is the same for every call; and CPython's 16 MB thread stack (its macOS <c>THREAD_STACK_SIZE</c>) lets numba
///         compile pymoo's AGE-MOEA kernels, which overflowed .NET's default stacks on macOS arm64. Python already runs one
///         thread at a time under the GIL, so calls lose no parallelism within a study, whose sampler calls are serial
///         anyway. The hop costs about 1 µs a call (measured 1.1–1.3 µs against 0.5–0.8 µs for the old per-call thread
///         state), where a bridge call takes milliseconds. What is lost: concurrent studies whose Python code releases the
///         GIL (torch kernels, large numpy operations) no longer overlap.
///     </para>
///     <para>
///         <b>Rules.</b>
///     </para>
///     <list type="bullet">
///         <item>A body that calls <see cref="PythonRuntime.Run(Action)" /> again, directly or from .NET code that Python
///         calls back, runs inline: it is already on the Python thread.</item>
///         <item>A caller that holds the GIL (user code inside <c>using (Py.GIL())</c>) releases it while it waits, as
///         CPython's own C extensions release it around blocking waits; otherwise the Python thread could never take it.</item>
///         <item>A body must never wait for another thread that may itself be waiting for the Python thread: the body
///         occupies the only thread that could run the other thread's call. In particular, do not call <c>Trial</c> or
///         <c>Study</c> methods inside <see cref="PythonRuntime.Run(Action)" /> while other threads optimize the same
///         study with a Python sampler; take the GIL with <c>Py.GIL()</c> for your own Python code instead. OptunaSharp's
///         bridges follow this rule: their bodies take only their own locks and the storages' locks.</item>
///         <item>The caller's <see cref="ExecutionContext" /> (culture, <c>AsyncLocal</c> values) flows into the body, and
///         the body's exception reaches the caller with its original stack trace.</item>
///     </list>
/// </remarks>
internal static class PythonThread
{
    /// <summary>The Python thread's stack: CPython's own thread stack size on macOS (<c>THREAD_STACK_SIZE</c>, 16 MB).</summary>
    internal const int StackSize = 16 * 1024 * 1024;

    // The calls waiting for the Python thread. It lives with the process, so a Python thread that failed to start and a
    // later one that succeeds share it; only a running Python thread takes from it.
    private static readonly BlockingCollection<WorkItem> _queue = new(new ConcurrentQueue<WorkItem>());

    // True on the Python thread only: a nested call there must run inline, since queuing it would wait for itself.
    [ThreadStatic] private static bool _onPythonThread;

    // Set once a Python thread is running calls; it then runs them for the life of the process.
    private static volatile bool _started;

    // Keeps the GIL state the Python thread took when a host had already started CPython on another thread. Never
    // released: dropping it would free the thread state, the very churn this thread exists to prevent.
    private static Py.GILState? _hostEngineAnchor;

    /// <summary>Whether the calling thread is the Python thread.</summary>
    internal static bool IsCurrent => _onPythonThread;

    /// <summary>Whether a Python thread is running calls (it then does so for the life of the process).</summary>
    internal static bool IsStarted => _started;

    /// <summary>
    ///     Starts the Python thread and waits until it can run calls: it starts the embedded engine itself (it then holds
    ///     CPython's main thread state for the life of the process), or, when a host already started the engine, takes a
    ///     thread state of its own and keeps it.
    /// </summary>
    /// <param name="startEngine">
    ///     Starts the engine on the calling thread (the Python thread); called only when no engine is running yet.
    /// </param>
    /// <exception cref="PythonInteropException">The engine could not be started; no Python thread is left running.</exception>
    /// <remarks>
    ///     <see cref="PythonRuntime.Initialize" /> calls this under its lock; once a start has succeeded, it is never
    ///     called again. A failed start ends its thread, so a later <see cref="PythonRuntime.Initialize" /> tries again with
    ///     a new one, as it did before the Python thread existed.
    /// </remarks>
    internal static void Start(Action startEngine)
    {
        var started = new StartSignal(startEngine);
        var thread = new Thread(Main, StackSize)
        {
            // Background: an idle Python thread must not keep the process alive, as no other bridge thread does.
            IsBackground = true,
            Name = "OptunaSharp Python",
        };
        thread.Start(started);
        // Not disposed: the Python thread may still be inside Set when the wait returns, and Dispose must not race it.
        // The event allocates a kernel handle only for a long wait, and the finalizer releases that one.
        started.Ready.Wait();
        started.Error?.Throw();
    }

    /// <summary>
    ///     Runs <paramref name="body" /> under the GIL on the Python thread and returns its result. Inline when the caller
    ///     already is the Python thread.
    /// </summary>
    /// <typeparam name="T">The body's result type; it must not need the GIL once the call returns.</typeparam>
    /// <param name="body">The work to run under the GIL; it must not wait for a thread that waits for the Python thread.</param>
    /// <returns>The body's result.</returns>
    /// <exception cref="Exception">Whatever <paramref name="body" /> throws, with its original stack trace.</exception>
    /// <remarks>
    ///     Requires a started Python thread (<see cref="Start" />, through <see cref="PythonRuntime.Initialize" />):
    ///     otherwise the call waits for one.
    /// </remarks>
    internal static T Invoke<T>(Func<T> body)
    {
        if (_onPythonThread)
        {
            // Nested: this thread's state persists, so the nested GIL scope only counts up and down.
            using (Py.GIL())
                return body();
        }

        var item = new WorkItem(() => body(), ExecutionContext.Capture());
        _queue.Add(item);
        Await(item);
        return (T)item.Result!;
    }

    /// <summary>
    ///     Waits for a queued call, releasing the GIL meanwhile when this thread holds it, as CPython's C extensions
    ///     release it around a blocking wait; then rethrows the body's exception, if any.
    /// </summary>
    /// <param name="item">The queued call.</param>
    /// <exception cref="Exception">Whatever the call's body threw, with its original stack trace.</exception>
    private static void Await(WorkItem item)
    {
        // Done is never disposed: the Python thread may still be inside Set when this wait returns, and Dispose must not
        // race it (an ObjectDisposedException there would end the Python thread). The event allocates a kernel handle
        // only for a long wait, and the finalizer releases that one.
        if (!GilAwareMonitor.CurrentThreadHoldsGil())
        {
            item.Done.Wait();
        }
        else
        {
            // The caller is inside its own Py.GIL() (an objective running Python, say). Holding the GIL through the wait
            // would keep the Python thread from ever taking it: a deadlock on the first call.
            IntPtr threadState = PythonEngine.BeginAllowThreads();
            try
            {
                item.Done.Wait();
            }
            finally
            {
                PythonEngine.EndAllowThreads(threadState);
            }
        }

        item.Error?.Throw();
    }

    /// <summary>
    ///     The Python thread: takes its thread state for good, reports that it is ready, then runs queued calls one at a
    ///     time, each under the GIL, for the life of the process.
    /// </summary>
    /// <param name="state">The <see cref="StartSignal" /> of <see cref="Start" />.</param>
    private static void Main(object? state)
    {
        var started = (StartSignal)state!;
        _onPythonThread = true;
        try
        {
            if (!PythonEngine.IsInitialized)
            {
                // Started here, the engine's main thread state belongs to this thread, and the engine start already
                // releases the GIL while keeping that state: nothing more is needed to keep it for good.
                started.StartEngine();
            }
            else
            {
                // A host started CPython on a thread of its own. Take a thread state here and release the GIL, not the
                // state: later scopes on this thread then nest on it instead of creating and freeing one each time.
                _hostEngineAnchor = Py.GIL();
                PythonEngine.BeginAllowThreads();
            }
        }
        catch (Exception e)
        {
            // The caller of Start rethrows this; the thread ends, and a later Initialize starts a new one.
            started.Error = ExceptionDispatchInfo.Capture(e);
            _onPythonThread = false;
            started.Ready.Set();
            return;
        }

        _started = true;
        started.Ready.Set();
        foreach (WorkItem item in _queue.GetConsumingEnumerable())
            item.Run();
    }

    /// <summary>The start handshake between <see cref="Start" /> and the Python thread.</summary>
    /// <param name="startEngine">Starts the embedded engine on the Python thread.</param>
    private sealed class StartSignal(Action startEngine)
    {
        /// <summary>Starts the embedded engine on the Python thread.</summary>
        internal Action StartEngine { get; } = startEngine;

        /// <summary>Set once the Python thread is ready, or has failed to start.</summary>
        internal ManualResetEventSlim Ready { get; } = new();

        /// <summary>Why the start failed, or <c>null</c>.</summary>
        internal ExceptionDispatchInfo? Error { get; set; }
    }

    /// <summary>One call waiting for the Python thread: its body, its caller's context, and its outcome.</summary>
    /// <param name="body">The body; its result is boxed for the caller to unbox.</param>
    /// <param name="context">The caller's execution context, or <c>null</c> when the caller suppressed its flow.</param>
    private sealed class WorkItem(Func<object?> body, ExecutionContext? context)
    {
        /// <summary>Set when the body has finished, normally or not.</summary>
        internal ManualResetEventSlim Done { get; } = new();

        /// <summary>The body's result, once <see cref="Done" /> is set.</summary>
        internal object? Result { get; private set; }

        /// <summary>The body's exception, or <c>null</c>.</summary>
        internal ExceptionDispatchInfo? Error { get; private set; }

        /// <summary>Runs the body on the Python thread, in the caller's context and under the GIL, and signals the caller.</summary>
        /// <remarks>Never throws: an exception belongs to the caller, and the Python thread must survive every body.</remarks>
        internal void Run()
        {
            try
            {
                if (context is null)
                    RunUnderGil();
                else
                    ExecutionContext.Run(context, static self => ((WorkItem)self!).RunUnderGil(), this);
            }
            catch (Exception e)
            {
                Error = ExceptionDispatchInfo.Capture(e);
            }
            finally
            {
                Done.Set();
            }
        }

        /// <summary>Takes the GIL on the Python thread's own state and runs the body.</summary>
        /// <exception cref="Exception">Whatever the body throws; <see cref="Run" /> hands it to the caller.</exception>
        private void RunUnderGil()
        {
            using (Py.GIL())
                Result = body();
        }
    }
}

using System;
using System.Runtime.InteropServices;
using System.Threading;
using OptunaSharp.Internal;
using Python.Runtime;

namespace OptunaSharp.Interop.PythonNet;

/// <summary>
///     The <c>MonitorGate</c> hooks this package installs into Core: a thread that holds the Python GIL releases it while
///     it waits for an OptunaSharp tell, trial, sampler or pruner monitor, then takes it back once the monitor is held,
///     and an <c>Optimize(nJobs &gt; 1)</c> caller releases it while it waits for its workers.
/// </summary>
/// <remarks>
///     <para>Why (review item P3-1): every Python bridge (OptunaHub sampler and pruner, BoTorch, PyCma, Pymoo) takes the
///     GIL inside calls that Core runs under those monitors. An objective that holds <c>Py.GIL()</c> while it calls a
///     <c>Trial</c> method blocked on the monitor with the GIL held. The monitor's holder needed the GIL, which CPython
///     even drops every switch interval, so at <c>nJobs &gt; 1</c> both threads waited forever. A hub sampler hung after 6
///     trials, the hub median pruner hung, BoTorch stalled.</para>
///     <para>The fix is the rule CPython's C extensions follow: never block on a lock while holding the GIL
///     (<c>Py_BEGIN_ALLOW_THREADS</c> around the wait). The monitor is tried first, so an uncontended call costs nothing
///     extra, and the GIL is only touched when the thread would otherwise block while holding it. The thread's Python
///     state is saved and restored by <see cref="PythonEngine.BeginAllowThreads" /> /
///     <see cref="PythonEngine.EndAllowThreads" />. That is legal at any GIL nesting depth, so user code inside
///     <c>using (Py.GIL())</c> is unaffected.</para>
///     <para>GIL ownership is read from the calling thread's own CPython thread state, through an export resolved from
///     the library pythonnet loaded (<c>Runtime.PythonDLL</c>); pythonnet exposes no public equivalent. A thread holds the
///     GIL exactly when it has a current thread state, which is also the precondition of <c>PyEval_SaveThread</c>, the call
///     that releases it. <c>PyGILState_Check()</c> is only the last resort: once any sub-interpreter has existed, CPython
///     turns that check off for the whole process and it answers 1 on every thread, so a thread without the GIL released
///     one and CPython ended the process (catalog Open item 28). If nothing can be resolved, the hook falls back to a plain
///     <c>Monitor.Enter</c>, the behavior before P3-1.</para>
/// </remarks>
internal static class GilAwareMonitor
{
    /// <summary>
    ///     CPython's <c>PyThreadState* PyThreadState_GetUnchecked(void)</c>, exported from 3.13 on, or the same function under
    ///     its earlier name <c>_PyThreadState_UncheckedGet</c> (3.11 and 3.12): the calling thread's current thread state.
    /// </summary>
    /// <returns>The current thread state while the calling thread holds the GIL; <see cref="IntPtr.Zero" /> otherwise.</returns>
    /// <remarks>
    ///     Unlike <c>PyThreadState_Get</c>, which ends the process with a fatal error when there is no current thread state,
    ///     it simply answers NULL, and unlike <c>PyGILState_Check</c> no sub-interpreter changes its answer.
    /// </remarks>
    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate IntPtr PyThreadStateGetUnchecked();

    /// <summary>
    ///     CPython's <c>int PyGILState_Check(void)</c>, the last resort for a build that exports neither thread-state function.
    /// </summary>
    /// <returns>1 if the calling thread holds the GIL, otherwise 0; but 1 on every thread once a sub-interpreter has existed.</returns>
    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate int PyGilStateCheck();

    /// <summary>
    ///     The thread-state exports, newest name first. Each CPython version exports exactly one of them: measured on Windows,
    ///     3.11.15 and 3.12.12 export the second; 3.13.7, 3.14.2 and free-threaded 3.14.2 the first.
    /// </summary>
    private static readonly string[] ThreadStateExports = new[] { "PyThreadState_GetUnchecked", "_PyThreadState_UncheckedGet" };

    /// <summary>
    ///     The resolved "does the calling thread hold the GIL?" check, or null when nothing could be resolved. Built once, on
    ///     the first question after the engine started, when <c>Runtime.PythonDLL</c> names the loaded library.
    /// </summary>
    private static readonly Lazy<Func<bool>?> HoldsGil = new(ResolveHoldsGil, LazyThreadSafetyMode.ExecutionAndPublication);

    /// <summary>
    ///     Installs both hooks into Core's <c>MonitorGate</c>: monitor entry and the <c>Optimize(nJobs)</c> wait for its
    ///     workers. It is idempotent and cheap, so it is called on every <see cref="PythonRuntime.Initialize" />, including
    ///     when the host started CPython itself.
    /// </summary>
    internal static void Install()
    {
        MonitorGate.Hook = Enter;
        MonitorGate.WaitHook = Wait;
    }

    /// <summary>
    ///     Runs a blocking wait that is not a monitor (an <c>nJobs</c> caller waiting for its workers), releasing the GIL
    ///     around it when this thread holds the GIL, as Python's <c>concurrent.futures.wait</c> does.
    /// </summary>
    /// <param name="wait">The wait; it touches no Python object, so it may run without the GIL.</param>
    /// <exception cref="Exception">Whatever <paramref name="wait" /> throws, rethrown once the GIL is back.</exception>
    private static void Wait(Action wait)
    {
        if (!CurrentThreadHoldsGil())
        {
            wait();
            return;
        }

        // A caller inside Py.GIL() would otherwise block with the GIL held while its workers need it (P3-1).
        IntPtr threadState = PythonEngine.BeginAllowThreads();
        try
        {
            wait();
        }
        finally
        {
            PythonEngine.EndAllowThreads(threadState);
        }
    }

    /// <summary>
    ///     Enters <paramref name="monitor" /> as <c>Monitor.Enter(monitor, ref lockTaken)</c> does, releasing the GIL
    ///     around the wait when the monitor is contended and this thread holds the GIL.
    /// </summary>
    /// <param name="monitor">The tell, trial, sampler or pruner monitor to enter.</param>
    /// <param name="lockTaken">Set to true once the monitor is held; must be false on entry.</param>
    private static void Enter(object monitor, ref bool lockTaken)
    {
        // Fast path: an uncontended monitor never needs the GIL released.
        Monitor.TryEnter(monitor, ref lockTaken);
        if (lockTaken)
            return;

        if (!CurrentThreadHoldsGil())
        {
            Monitor.Enter(monitor, ref lockTaken);
            return;
        }

        // Blocking with the GIL held is the P3-1 deadlock: the monitor's holder may need the GIL to finish. Release it
        // for the wait, as Py_BEGIN_ALLOW_THREADS does, and take it back before returning with the monitor held.
        IntPtr threadState = PythonEngine.BeginAllowThreads();
        try
        {
            Monitor.Enter(monitor, ref lockTaken);
        }
        finally
        {
            PythonEngine.EndAllowThreads(threadState);
        }
    }

    /// <summary>Whether the calling thread holds the GIL; false when the engine is down or no check could be resolved.</summary>
    /// <returns>
    ///     True only when the calling thread has a current CPython thread state; on a build without the thread-state exports,
    ///     when <c>PyGILState_Check</c> says so.
    /// </returns>
    /// <remarks>
    ///     <para>Also used by <see cref="PythonThread" />: a caller that holds the GIL releases it while it waits for the
    ///     Python thread.</para>
    ///     <para>Footgun this avoids (catalog Open item 28): every true answer leads to <c>PyEval_SaveThread</c>, so a true
    ///     answer for a thread without the GIL ends the process: "Fatal Python error: PyEval_SaveThread" on 3.12, an access
    ///     violation inside <c>PyEval_SaveThread</c> on 3.14. That is the answer <c>PyGILState_Check</c> gives on every
    ///     thread once a sub-interpreter has existed (measured on 3.11.15, 3.12.12, 3.13.7, 3.14.2 and free-threaded
    ///     3.14.2), which is why it is only the last resort.</para>
    /// </remarks>
    internal static bool CurrentThreadHoldsGil()
        => PythonEngine.IsInitialized && HoldsGil.Value is { } holdsGil && holdsGil();

    /// <summary>
    ///     Resolves the check from the Python library pythonnet loaded: the first thread-state export found, else
    ///     <c>PyGILState_Check</c>. The library is already in the process, so loading it by the same name returns that module.
    /// </summary>
    /// <returns>The check, or null when neither the library nor any of the exports can be found.</returns>
    private static Func<bool>? ResolveHoldsGil()
    {
        try
        {
            string? library = Runtime.PythonDLL;
            if (string.IsNullOrEmpty(library) || !NativeLibrary.TryLoad(library, out IntPtr handle))
                return null;

            foreach (string name in ThreadStateExports)
            {
                if (NativeLibrary.TryGetExport(handle, name, out IntPtr export))
                {
                    var currentThreadState = Marshal.GetDelegateForFunctionPointer<PyThreadStateGetUnchecked>(export);
                    return () => currentThreadState() != IntPtr.Zero;
                }
            }

            // Last resort, for a build that exports neither name: right until a sub-interpreter is created (Open item 28).
            if (NativeLibrary.TryGetExport(handle, "PyGILState_Check", out IntPtr check))
            {
                var gilStateCheck = Marshal.GetDelegateForFunctionPointer<PyGilStateCheck>(check);
                return () => gilStateCheck() == 1;
            }

            return null;
        }
        catch (Exception)
        {
            // Any failure to resolve an export only costs the GIL release, never correctness of the monitor itself.
            return null;
        }
    }
}

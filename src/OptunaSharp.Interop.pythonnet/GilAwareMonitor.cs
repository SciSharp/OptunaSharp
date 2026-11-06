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
///     <para>GIL ownership comes from CPython's own <c>PyGILState_Check()</c>, resolved from the library pythonnet
///     loaded (<c>Runtime.PythonDLL</c>); pythonnet exposes no public equivalent. If it cannot be resolved, the hook
///     falls back to a plain <c>Monitor.Enter</c>, the behavior before the fix.</para>
/// </remarks>
internal static class GilAwareMonitor
{
    /// <summary>CPython's <c>int PyGILState_Check(void)</c>: 1 when the calling thread holds the GIL.</summary>
    /// <returns>1 if the current thread holds the GIL, otherwise 0.</returns>
    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate int PyGilStateCheck();

    private static readonly Lazy<PyGilStateCheck?> GilStateCheck = new(ResolveGilStateCheck, LazyThreadSafetyMode.ExecutionAndPublication);

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

    /// <summary>Whether the calling thread holds the GIL; false when the engine is down or the check is unavailable.</summary>
    /// <returns>True only when CPython reports that this thread holds the GIL.</returns>
    private static bool CurrentThreadHoldsGil()
        => PythonEngine.IsInitialized && GilStateCheck.Value is { } check && check() == 1;

    /// <summary>
    ///     Resolves <c>PyGILState_Check</c> from the Python library pythonnet loaded. It is already in the process, so
    ///     loading it by the same name returns that module.
    /// </summary>
    /// <returns>A callable for <c>PyGILState_Check</c>, or null when the library or the export cannot be found.</returns>
    private static PyGilStateCheck? ResolveGilStateCheck()
    {
        try
        {
            string? library = Runtime.PythonDLL;
            if (string.IsNullOrEmpty(library) || !NativeLibrary.TryLoad(library, out IntPtr handle))
                return null;
            return NativeLibrary.TryGetExport(handle, "PyGILState_Check", out IntPtr function)
                ? Marshal.GetDelegateForFunctionPointer<PyGilStateCheck>(function)
                : null;
        }
        catch (Exception)
        {
            // Any failure to resolve the export only costs the GIL release, never correctness of the monitor itself.
            return null;
        }
    }
}

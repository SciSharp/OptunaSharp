using System;
using System.Threading;

namespace OptunaSharp.Internal;

/// <summary>
///     How OptunaSharp enters the monitors that come before the Python GIL in its lock order: the storage's tell monitor
///     (<c>BaseStorage.TellLockFor</c>), each <c>Trial</c>'s own monitor, and a sampler's or pruner's <c>SyncRoot</c>.
///     Without a hook it is <c>Monitor.Enter</c>. An embedding runtime with a global interpreter lock installs
///     <see cref="Hook" />, and the hook releases that lock while the thread waits for a monitor. Storage locks come
///     after the GIL and stay plain <c>lock</c>s: their holders never wait for it.
/// </summary>
/// <remarks>
///     <para>Why (review item P3-1, a regression of <c>3a75811</c>): Python-bridged samplers and pruners (OptunaHub,
///     BoTorch, PyCma, Pymoo) take the GIL inside their calls, which run under the monitor. An objective that holds
///     <c>Py.GIL()</c> while it calls a <c>Trial</c> method blocks on the monitor. Meanwhile the thread holding the
///     monitor needs the GIL, and CPython even drops the GIL from that thread every switch interval. At
///     <c>nJobs &gt; 1</c> both threads waited forever: a hub sampler hung after 6 trials, the hub median pruner (which
///     worked before <c>3a75811</c>) hung, BoTorch stalled.</para>
///     <para>The rule CPython's own C extensions follow fixes all of these: never block on a lock while holding the GIL.
///     <c>OptunaSharp.Interop.pythonnet</c> installs a hook that tries the monitor first. Only when the monitor is
///     contended and the current thread holds the GIL does it release the GIL around the wait (the equivalent of
///     <c>Py_BEGIN_ALLOW_THREADS</c>) and take it back once the monitor is held. One process-wide hook covers every
///     composition: a managed sampler wrapping a bridged one, a bridged independent sampler, a bridged pruner, and the
///     tell monitor above them all. Core itself stays free of any Python reference.</para>
///     <para>The same rule covers the one blocking wait that is not a monitor: <c>Optimize(nJobs &gt; 1)</c> waiting for
///     its workers. A caller inside <c>Py.GIL()</c> blocked there with the GIL held while every worker that needed it
///     waited, so <see cref="Wait" /> runs that wait through <see cref="WaitHook" />, which releases the GIL around it as
///     Python's <c>concurrent.futures.wait</c> does.</para>
/// </remarks>
internal static class MonitorGate
{
    /// <summary>Enters <paramref name="monitor" /> exactly as <c>Monitor.Enter(monitor, ref lockTaken)</c> does.</summary>
    /// <param name="monitor">The monitor to enter.</param>
    /// <param name="lockTaken">Set to true once the monitor is held; must be false on entry.</param>
    internal delegate void EnterHook(object monitor, ref bool lockTaken);

    private static EnterHook? _hook;

    /// <summary>
    ///     The process-wide hook, or null for plain <c>Monitor.Enter</c>. Installed once by
    ///     <c>OptunaSharp.Interop.pythonnet</c> when it starts or adopts CPython. A hook must leave
    ///     <c>lockTaken</c> as <c>Monitor.Enter(object, ref bool)</c> would, even when it throws.
    /// </summary>
    internal static EnterHook? Hook
    {
        get => Volatile.Read(ref _hook);
        set => Volatile.Write(ref _hook, value);
    }

    /// <summary>Runs <paramref name="wait" />, a blocking wait that is not a monitor, as the hook's runtime requires.</summary>
    /// <param name="wait">The blocking wait, for example on worker tasks.</param>
    internal delegate void BlockingWaitHook(Action wait);

    private static BlockingWaitHook? _waitHook;

    /// <summary>
    ///     The process-wide hook for blocking waits that are not monitors, or null to run them as they are. Installed with
    ///     <see cref="Hook" />; it releases the GIL around the wait when the waiting thread holds it, as Python's own
    ///     <c>concurrent.futures.wait</c> does. Footgun: the wait must not touch Python objects, it runs without the GIL.
    /// </summary>
    internal static BlockingWaitHook? WaitHook
    {
        get => Volatile.Read(ref _waitHook);
        set => Volatile.Write(ref _waitHook, value);
    }

    /// <summary>
    ///     Runs <paramref name="wait" />, through <see cref="WaitHook" /> when one is installed. <c>Optimize(nJobs &gt; 1)</c>
    ///     waits for its workers this way, so a caller holding the GIL does not starve workers that need it.
    /// </summary>
    /// <param name="wait">The blocking wait; it must not touch Python objects.</param>
    /// <exception cref="Exception">Whatever <paramref name="wait" /> throws, after the hook has restored the GIL.</exception>
    internal static void Wait(Action wait)
    {
        var hook = WaitHook;
        if (hook is null)
            wait();
        else
            hook(wait);
    }

    /// <summary>Enters <paramref name="monitor" />, through <see cref="Hook" /> when one is installed.</summary>
    /// <param name="monitor">The monitor to enter.</param>
    /// <param name="lockTaken">Set to true once the monitor is held; must be false on entry.</param>
    /// <exception cref="System.Threading.ThreadInterruptedException">The waiting thread was interrupted; the monitor is not held.</exception>
    internal static void Enter(object monitor, ref bool lockTaken)
    {
        var hook = Hook;
        if (hook is null)
            Monitor.Enter(monitor, ref lockTaken);
        else
            hook(monitor, ref lockTaken);
    }

    /// <summary>
    ///     Enters <paramref name="monitor" /> through <see cref="Enter" /> and returns the scope that exits it, so a
    ///     <c>lock (monitor) { … }</c> block becomes <c>using (MonitorGate.Scope(monitor)) { … }</c> with the same shape,
    ///     early returns included.
    /// </summary>
    /// <param name="monitor">The monitor to enter; the trial monitor is the one that needs this form.</param>
    /// <returns>A scope whose <see cref="MonitorScope.Dispose" /> exits <paramref name="monitor" />.</returns>
    /// <exception cref="System.Threading.ThreadInterruptedException">The waiting thread was interrupted; the monitor is not held.</exception>
    /// <remarks>
    ///     Footgun: the scope must be disposed exactly once, on the thread that entered, which a <c>using</c> statement
    ///     guarantees. A hook that throws after taking the monitor would otherwise leak it, so the monitor is exited here
    ///     before the exception leaves.
    /// </remarks>
    internal static MonitorScope Scope(object monitor)
    {
        bool taken = false;
        try
        {
            Enter(monitor, ref taken);
        }
        catch
        {
            // Monitor.Enter(object, ref bool) reports a taken monitor even when it throws; never leave it held.
            if (taken)
                Monitor.Exit(monitor);
            throw;
        }

        return new MonitorScope(monitor);
    }

    /// <summary>
    ///     A held monitor, released by <see cref="Dispose" />. A <c>ref struct</c>, so it cannot be boxed, captured or
    ///     carried across an <c>await</c>: the monitor stays bound to the thread that entered it, as with <c>lock</c>.
    /// </summary>
    internal readonly ref struct MonitorScope
    {
        private readonly object _monitor;

        /// <summary>Wraps a monitor the caller already holds.</summary>
        /// <param name="monitor">The held monitor.</param>
        internal MonitorScope(object monitor) => _monitor = monitor;

        /// <summary>Exits the monitor. Called once, by the <c>using</c> statement that owns the scope.</summary>
        /// <exception cref="SynchronizationLockException">The current thread does not hold the monitor (a misuse of the scope).</exception>
        public void Dispose() => Monitor.Exit(_monitor);
    }
}

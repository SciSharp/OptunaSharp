using System;
using System.Collections.Generic;
using System.Linq;
using NumSharp;

namespace OptunaSharp.Internal;

/// <summary>
///     Holds OpenBLAS's thread count for the duration of <c>Optimize</c> calls, as <see cref="TrialExecutorOptions.BlasThreads" />
///     asks: single-threaded while several workers run (the automatic policy), or an explicit count.
/// </summary>
/// <remarks>
///     <para>
///         <b>Why.</b> OpenBLAS's threads are one pool for the whole process, sized for one caller. Several
///         <c>nJobs</c> workers in linear algebra at once oversubscribe the cores: 2,000 trials of a 40-dimensional
///         CMA-ES at <c>nJobs: 8</c> took 5.22 s with 24 unpinned threads and 3.58 s pinned to one (32 cores,
///         2026-11-08). Optuna's GPSampler limits its own BLAS threads for the same reason
///         (<c>optuna/_gp/thread_limiting.py</c>).
///     </para>
///     <para>
///         <b>Process-wide, so shared.</b> The count is one native setting for every thread of the process. Each call
///         holds the count it asks for; the count in effect is the smallest of the counts held, so concurrent parallel
///         runs never oversubscribe; the count in effect before the first holder is restored when the last one releases.
///         The count is changed only on the first hold, on a change of the smallest count, and on the last release, so
///         calls that agree never touch the native library while another call's workers are inside it.
///     </para>
///     <para>
///         <b>A performance policy only.</b> No failure here may fail an optimization: without a loaded OpenBLAS, or if
///         NumSharp's threading knob refuses a change, the call runs with whatever count is in effect.
///     </para>
/// </remarks>
internal static class BlasThreadPolicy
{
    /// <summary>The environment variable NumSharp's OpenBLAS knob writes when it changes the count.</summary>
    private const string ThreadsVariable = "OPENBLAS_NUM_THREADS";

    private static readonly object _gate = new();

    // The counts held by the calls running now, one entry per call; guarded by _gate.
    private static readonly List<int> _held = new();

    // The live count before the first current holder, restored when the last one releases; guarded by _gate.
    private static int _original;

    // The count this policy last set (or found), so an unchanged smallest count costs no native call; guarded by _gate.
    private static int _applied;

    // Whether OPENBLAS_NUM_THREADS was in the process environment before the first current holder; NumSharp's knob
    // writes it on every change, so it is removed again on restore when it was not there; guarded by _gate.
    private static bool _variableWasSet;

    /// <summary>
    ///     Holds <paramref name="threads" /> OpenBLAS threads until the returned lease is disposed.
    /// </summary>
    /// <param name="threads">
    ///     The count to hold, or null to leave OpenBLAS alone (the lease then does nothing). Zero and negative counts hold
    ///     nothing either; <see cref="TrialExecutorOptions.ValidateBlasThreads" /> refuses them before a call starts.
    /// </param>
    /// <returns>A lease to dispose when the call ends, on the thread that took it; disposing it twice is a misuse.</returns>
    /// <remarks>Never throws: a missing OpenBLAS or a refused change leaves the count as it is.</remarks>
    internal static Lease Hold(int? threads)
    {
        // A non-positive count would be held but never released: the empty lease is 0, so its Dispose releases nothing.
        // Measured with the validation removed (a mutation run): a 0 left in the held list pinned every later call.
        if (threads is not int requested || requested <= 0)
            return default;

        lock (_gate)
        {
            if (_held.Count == 0)
            {
                // The first holder records what to restore. Nothing to hold when OpenBLAS is absent or reports no count.
                if (ReadThreads() is not int live)
                    return default;
                _original = live;
                _applied = live;
                _variableWasSet = Environment.GetEnvironmentVariable(ThreadsVariable) is not null;
            }

            _held.Add(requested);
            Apply(_held.Min());
        }

        return new Lease(requested);
    }

    /// <summary>Releases one hold of <paramref name="threads" />: applies the next smallest held count, or restores the original.</summary>
    /// <param name="threads">The count the released lease held.</param>
    private static void Release(int threads)
    {
        lock (_gate)
        {
            _held.Remove(threads);
            if (_held.Count > 0)
            {
                Apply(_held.Min());
                return;
            }

            Apply(_original);
            // The knob wrote the variable on every change; a process that did not set it gets its environment back, so
            // child processes it starts do not inherit a count the user never chose.
            if (!_variableWasSet)
                Environment.SetEnvironmentVariable(ThreadsVariable, null);
        }
    }

    /// <summary>Sets the native count to <paramref name="threads" /> unless it already is; the caller holds <see cref="_gate" />.</summary>
    /// <param name="threads">The count to set.</param>
    private static void Apply(int threads)
    {
        if (threads == _applied)
            return;
        try
        {
            TensorEngine.Threading.SetThreads(TensorEngine.Threading.OpenBlas, threads);
            _applied = threads;
        }
        catch (Exception)
        {
            // A performance policy must not fail the optimization (see remarks); the count stays as it was.
        }
    }

    /// <summary>The live OpenBLAS thread count, read through NumSharp's knob.</summary>
    /// <returns>The count, or null when OpenBLAS is not loaded, its knob is not registered, or the read fails.</returns>
    private static int? ReadThreads()
    {
        try
        {
            return TensorEngine.Threading.TryGet(TensorEngine.Threading.OpenBlas, out TensorEngine.Threading.Variable knob)
                ? knob.Threads
                : null;
        }
        catch (Exception)
        {
            return null;
        }
    }

    /// <summary>
    ///     One hold of a thread count, released by <see cref="Dispose" />. The default value holds nothing, so a call with
    ///     nothing to hold can dispose it all the same.
    /// </summary>
    internal readonly struct Lease : IDisposable
    {
        // The count this lease holds; 0 (the default) holds nothing, since every held count is at least 1.
        private readonly int _threads;

        /// <summary>Wraps a hold that <see cref="Hold" /> has registered.</summary>
        /// <param name="threads">The count held; at least 1.</param>
        internal Lease(int threads) => _threads = threads;

        /// <summary>Releases the hold; does nothing for the default lease.</summary>
        public void Dispose()
        {
            if (_threads != 0)
                Release(_threads);
        }
    }
}

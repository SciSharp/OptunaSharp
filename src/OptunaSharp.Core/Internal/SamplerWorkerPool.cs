using System.Collections.Generic;
using OptunaSharp.Samplers;

namespace OptunaSharp.Internal;

/// <summary>
///     The sampler instances of one <c>Optimize(nJobs)</c> call with <see cref="SamplerConcurrency.PerWorker" />: a
///     worker rents one for its trial and returns it afterwards, so a call holds at most <c>nJobs</c> instances, each
///     serving one trial at a time and reused by later trials of the call.
/// </summary>
/// <remarks>
///     <para>
///         Instances are keyed by the sampler they were made from (the study's <see cref="Study.Sampler" /> when the
///         worker started its trial), so a sampler swapped on the study mid-run gets instances of its own and the old
///         ones are never handed out again.
///     </para>
///     <para>
///         A sampler whose <see cref="BaseSampler.CreateWorkerInstance" /> returns null is remembered, so its
///         workers fall back to the shared instance, one call at a time, without asking again per trial. The first call
///         per sampler is made under the pool's lock, so workers that start together ask it once; later instances of a
///         sampler that makes them are built outside the lock.
///     </para>
///     <para>Thread-safe: every worker of the call rents and returns concurrently, under one short lock.</para>
/// </remarks>
internal sealed class SamplerWorkerPool
{
    private readonly object _gate = new();

    // Idle instances, each with the sampler it was made from; guarded by _gate.
    private readonly List<(BaseSampler Prototype, BaseSampler Instance)> _idle = new();

    // Samplers that have no worker instance (they stay shared); guarded by _gate.
    private readonly HashSet<BaseSampler> _shared = new(ReferenceEqualityComparer.Instance);

    // Samplers whose first CreateWorkerInstance call returned an instance, so later ones are made outside the lock;
    // guarded by _gate.
    private readonly HashSet<BaseSampler> _makesInstances = new(ReferenceEqualityComparer.Instance);

    /// <summary>
    ///     An instance made from <paramref name="prototype" /> for one trial: an idle one, or a new one; null when the
    ///     sampler must stay shared.
    /// </summary>
    /// <param name="prototype">The study's sampler, read once by the worker for this trial.</param>
    /// <returns>An instance owned by the caller until <see cref="Return" />, or null for shared access.</returns>
    /// <exception cref="System.Exception">Whatever the sampler's <see cref="BaseSampler.CreateWorkerInstance" /> throws.</exception>
    internal BaseSampler? Rent(BaseSampler prototype)
    {
        lock (_gate)
        {
            if (_shared.Contains(prototype))
                return null;
            for (int i = _idle.Count - 1; i >= 0; i--)
            {
                if (ReferenceEquals(_idle[i].Prototype, prototype))
                {
                    BaseSampler instance = _idle[i].Instance;
                    _idle.RemoveAt(i);
                    return instance;
                }
            }

            if (!_makesInstances.Contains(prototype))
            {
                // The first call for this sampler is made under the lock, so workers that start together ask it once:
                // two of them asking outside the lock both asked a sampler that stays shared (a flaky
                // PerWorker_ASamplerWithoutWorkerInstances_StaysSharedOneCallAtATime). It only copies configuration,
                // so the others wait microseconds, once per call.
                BaseSampler? first = prototype.CreateWorkerInstance();
                if (first is null)
                {
                    _shared.Add(prototype);
                    return null;
                }

                _makesInstances.Add(prototype);
                return first;
            }
        }

        // Built outside the lock: other workers keep renting meanwhile. It reads only the prototype's configuration,
        // which never changes after construction, so it needs no sampler monitor either.
        BaseSampler? created = prototype.CreateWorkerInstance();
        if (created is null)
        {
            lock (_gate)
                _shared.Add(prototype);
        }

        return created;
    }

    /// <summary>Gives an instance back after its trial, for a later trial of the call.</summary>
    /// <param name="prototype">The sampler the instance was rented for.</param>
    /// <param name="instance">The instance; the caller no longer uses it.</param>
    internal void Return(BaseSampler prototype, BaseSampler instance)
    {
        lock (_gate)
            _idle.Add((prototype, instance));
    }
}

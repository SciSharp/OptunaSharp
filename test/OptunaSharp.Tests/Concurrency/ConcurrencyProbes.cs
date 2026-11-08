using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using OptunaSharp.Collections;
using OptunaSharp.Distributions;
using OptunaSharp.Logging;
using OptunaSharp.Pruners;
using OptunaSharp.Samplers;

namespace OptunaSharp.Tests.Concurrency;

/// <summary>A running maximum that several threads raise at once without a lock.</summary>
internal static class Watermark
{
    /// <summary>Raises <paramref name="target" /> to <paramref name="value" /> unless it already is at least that, atomically.</summary>
    /// <param name="target">The running maximum, shared by the threads.</param>
    /// <param name="value">A new observation.</param>
    internal static void Raise(ref int target, int value)
    {
        int seen = Volatile.Read(ref target);
        // Retry while another thread raised it between the read and the exchange and the value is still higher.
        while (value > seen)
        {
            int previous = Interlocked.CompareExchange(ref target, value, seen);
            if (previous == seen)
                return;
            seen = previous;
        }
    }
}

/// <summary>
///     A sampler that watches how a study drives it: it counts each kind of call, per trial for <see cref="AfterTrial" />,
///     and records the most calls that were ever inside it at once. Every study-driven call into a sampler runs under the
///     sampler's monitor (THREADING.md §6.3), so <see cref="PeakOverlap" /> must stay 1 however many threads drive it.
/// </summary>
/// <remarks>
///     Sampling is delegated to an inner <see cref="RandomSampler" />, called directly as THREADING.md §5.3 asks of a
///     wrapping sampler, since the probe's own monitor already covers it. Each call keeps the probe busy a little longer
///     than it needs (a short spin), which widens the window in which a call that slipped past the monitor would be seen.
/// </remarks>
internal sealed class OverlapProbeSampler : BaseSampler
{
    private readonly RandomSampler _inner;
    private readonly ConcurrentDictionary<int, int> _afterTrialCalls = new();
    private int _inside;
    private int _peakOverlap;
    private int _sampleIndependentCalls;

    /// <summary>Creates a probe over a random sampler.</summary>
    /// <param name="seed">The inner random sampler's seed; null seeds it from OS entropy.</param>
    internal OverlapProbeSampler(int? seed = null) => _inner = new RandomSampler(seed);

    /// <summary>The most calls that were inside the probe at one instant; 1 when the monitor held, 0 before any call.</summary>
    internal int PeakOverlap => Volatile.Read(ref _peakOverlap);

    /// <summary>How many times the study asked for an independent sample.</summary>
    internal int SampleIndependentCalls => Volatile.Read(ref _sampleIndependentCalls);

    /// <summary>How many times the study ran <see cref="AfterTrial" /> for one trial; exactly one tell must.</summary>
    /// <param name="trialId">The trial's storage identifier.</param>
    /// <returns>The call count; 0 for a trial never told.</returns>
    internal int AfterTrialCallsOf(int trialId) => _afterTrialCalls.TryGetValue(trialId, out int calls) ? calls : 0;

    /// <inheritdoc />
    public override OrderedDict<string, BaseDistribution> InferRelativeSearchSpace(Study study, FrozenTrial trial)
        => Watch(() => _inner.InferRelativeSearchSpace(study, trial));

    /// <inheritdoc />
    public override OrderedDict<string, object?> SampleRelative(
        Study study, FrozenTrial trial, OrderedDict<string, BaseDistribution> searchSpace)
        => Watch(() => _inner.SampleRelative(study, trial, searchSpace));

    /// <inheritdoc />
    public override object? SampleIndependent(Study study, FrozenTrial trial, string paramName, BaseDistribution paramDistribution)
        => Watch(() =>
        {
            Interlocked.Increment(ref _sampleIndependentCalls);
            return _inner.SampleIndependent(study, trial, paramName, paramDistribution);
        });

    /// <inheritdoc />
    public override void BeforeTrial(Study study, FrozenTrial trial)
        => Watch(() =>
        {
            _inner.BeforeTrial(study, trial);
            return 0;
        });

    /// <inheritdoc />
    public override void AfterTrial(Study study, FrozenTrial trial, TrialState state, IReadOnlyList<double>? values)
        => Watch(() =>
        {
            _afterTrialCalls.AddOrUpdate(trial.TrialId, 1, (_, calls) => calls + 1);
            _inner.AfterTrial(study, trial, state, values);
            return 0;
        });

    /// <inheritdoc />
    public override void ReseedRng()
        => Watch(() =>
        {
            _inner.ReseedRng();
            return 0;
        });

    /// <summary>Runs one call of the study's, counting the calls inside the probe meanwhile.</summary>
    /// <typeparam name="T">The call's result type.</typeparam>
    /// <param name="call">The sampler call.</param>
    /// <returns>What the call returned.</returns>
    /// <exception cref="Exception">Whatever the inner sampler throws; the probe's count is restored first.</exception>
    private T Watch<T>(Func<T> call)
    {
        Watermark.Raise(ref _peakOverlap, Interlocked.Increment(ref _inside));
        try
        {
            // Hold the probe a little: a second call that slipped past the monitor would now be inside with this one.
            Thread.SpinWait(300);
            return call();
        }
        finally
        {
            Interlocked.Decrement(ref _inside);
        }
    }
}

/// <summary>
///     A pruner that watches how trials drive it: it counts its calls and records the most that were ever inside it at
///     once, which the pruner's monitor keeps at 1 (THREADING.md §6.4). It prunes a trial exactly when the trial's last
///     reported value is odd, so a test can tell from each decision that it was made on that trial's own values.
/// </summary>
internal sealed class OverlapProbePruner : BasePruner
{
    private int _inside;
    private int _peakOverlap;
    private int _calls;

    /// <summary>The most calls that were inside the pruner at one instant; 1 when the monitor held.</summary>
    internal int PeakOverlap => Volatile.Read(ref _peakOverlap);

    /// <summary>How many pruning decisions the pruner made.</summary>
    internal int Calls => Volatile.Read(ref _calls);

    /// <summary>Prunes the trial when its last reported value is an odd integer.</summary>
    /// <param name="study">The study the trial belongs to; unused.</param>
    /// <param name="trial">The trial to decide on; it must have reported at least once.</param>
    /// <returns>True when the trial's last intermediate value is odd.</returns>
    /// <exception cref="InvalidOperationException">The trial has reported nothing.</exception>
    public override bool Prune(Study study, FrozenTrial trial)
    {
        Watermark.Raise(ref _peakOverlap, Interlocked.Increment(ref _inside));
        try
        {
            Interlocked.Increment(ref _calls);
            // Hold the pruner a little, as the sampler probe does, to widen the window a second decision would be seen in.
            Thread.SpinWait(300);
            double last = trial.IntermediateValues.Values.Last();
            return (long)last % 2 != 0;
        }
        finally
        {
            Interlocked.Decrement(ref _inside);
        }
    }
}

/// <summary>
///     Lowers OptunaSharp's log verbosity for the life of the scope, then puts back the level it found. For volleys whose
///     shots log by design (a report of a step already reported warns once per losing shooter), which would otherwise
///     bury the test output; the tests run one at a time, so the process-wide level is theirs meanwhile.
/// </summary>
internal sealed class QuietLogs : IDisposable
{
    private readonly int _saved;

    /// <summary>Keeps only records at <paramref name="level" /> or above until disposed.</summary>
    /// <param name="level">The verbosity to apply; <see cref="LogLevel.Error" /> by default.</param>
    internal QuietLogs(int level = LogLevel.Error)
    {
        _saved = OptunaLogging.GetVerbosity();
        OptunaLogging.SetVerbosity(level);
    }

    /// <summary>Restores the verbosity in effect when the scope began.</summary>
    public void Dispose() => OptunaLogging.SetVerbosity(_saved);
}

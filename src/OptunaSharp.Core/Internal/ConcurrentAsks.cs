using System;
using System.Collections.Generic;
using OptunaSharp.Collections;

namespace OptunaSharp.Internal;

/// <summary>
///     What a sampler that hands out candidates from shared state (a generation's particles, a population's slots, a
///     fixed candidate set) needs so that two trials asked at the same time never receive the same candidate: the
///     trials that are running right now, read from the storage.
/// </summary>
/// <remarks>
///     <para>
///         <b>Why the storage and not the per-trial snapshot.</b> <c>Study.GetTrialsInternal(useCache: true)</c> returns
///         the snapshot this thread took the first time the trial read the history, possibly in an earlier sampler
///         call, before another worker stamped the candidate it took. Every stamp a sampler writes inside its
///         monitor is in the storage before the monitor is released, so a read made inside the next monitor-held call
///         sees all of them.
///     </para>
///     <para>
///         A sequential run has no running trial besides the one being sampled, so every sampler that consults this
///         behaves exactly as before; only concurrent asks (<c>Optimize(nJobs)</c>, several threads asking one study)
///         are steered apart.
///     </para>
/// </remarks>
internal static class ConcurrentAsks
{
    /// <summary>The study's running trials other than <paramref name="current" />, as stored now.</summary>
    /// <param name="study">The study whose trials are read.</param>
    /// <param name="current">The trial being sampled; excluded, since it holds nothing yet.</param>
    /// <returns>The other running trials in number order; read-only snapshots (no deep copy).</returns>
    /// <exception cref="NumSharp.KeyError">The study no longer exists in storage.</exception>
    internal static List<FrozenTrial> OtherRunningTrials(Study study, FrozenTrial current)
    {
        var running = study.GetTrialsInternal(deepcopy: false, states: new[] { TrialState.Running }, useCache: false);
        running.RemoveAll(t => t.TrialId == current.TrialId);
        return running;
    }

    /// <summary>
    ///     The study's complete and running trials other than <paramref name="current" />, read together from the
    ///     storage now: what an index a trial holds (a population slot, a design row) must be looked up in.
    /// </summary>
    /// <remarks>
    ///     One read, not the history snapshot plus a read of the running trials: a trial told by another worker
    ///     between the two (a tell does not take the sampler's monitor) would be in neither, and the index it holds
    ///     would look free. In one read every trial is in exactly one state. In a sequential run the read equals the
    ///     snapshot, so the indices found, and every seeded result, are unchanged.
    /// </remarks>
    /// <param name="study">The study whose trials are read.</param>
    /// <param name="current">The trial being sampled; excluded, since it holds nothing yet.</param>
    /// <returns>The other complete and running trials in number order; read-only snapshots (no deep copy).</returns>
    /// <exception cref="NumSharp.KeyError">The study no longer exists in storage.</exception>
    internal static List<FrozenTrial> OtherCompleteOrRunningTrials(Study study, FrozenTrial current)
    {
        var trials = study.GetTrialsInternal(
            deepcopy: false, states: new[] { TrialState.Complete, TrialState.Running }, useCache: false);
        trials.RemoveAll(t => t.TrialId == current.TrialId);
        return trials;
    }

    /// <summary>
    ///     The first index (a population slot, a row of a space-filling design) that no trial holds: the indices
    ///     stamped on finished and running trials, plus, for each finished trial with no stamp (one written before
    ///     stamps existed, or one the sampler did not propose), the lowest index still free, which is the one it took
    ///     when indices were handed out in order.
    /// </summary>
    /// <param name="held">The stamped indices of finished and running trials; duplicates and negatives are harmless.</param>
    /// <param name="unstampedFinished">How many finished trials hold an index without a stamp.</param>
    /// <returns>The free index. With only finished trials holding 0..n-1 (a sequential run), it is <c>n</c>.</returns>
    internal static int FirstFreeIndex(IEnumerable<long> held, int unstampedFinished)
    {
        var taken = new HashSet<long>(held);
        long next = 0;
        for (int i = 0; i < unstampedFinished; i++)
        {
            while (taken.Contains(next))
                next++;
            taken.Add(next);
        }

        long free = 0;
        while (taken.Contains(free))
            free++;
        return (int)free;
    }
}

/// <summary>
///     The candidate points a sampler handed to trials that are still running, per study, so a point one of them is
///     evaluating is drawn again instead of handed to a second trial. A trial's parameters reach the storage only after
///     its sampler call returns, so this record, written inside the call, is the only complete view.
/// </summary>
/// <remarks>
///     <para>
///         For samplers whose candidates can coincide by construction: few distinct points per generation (LM-CMA's
///         Rademacher draws), or the same donors drawn twice (DE), or two particles clamped to one corner of the box
///         (PSO). Points are compared exactly, in the optimizer's unit box.
///     </para>
///     <para>
///         Not synchronized: every member is called from a sampler hook, under the sampler's monitor. A sequential run
///         never has another running trial's point here (each trial's entry leaves with <c>AfterTrial</c>), so it
///         never redraws and its random stream is unchanged.
///     </para>
///     <para>
///         <b>One process only.</b> The record lives in this sampler instance, so two processes sharing a storage can
///         still hand one point to two trials; the slot and row stamps the samplers write to the storage are what
///         steers different processes apart.
///     </para>
/// </remarks>
internal sealed class HandedOutPoints
{
    /// <summary>
    ///     The most draws past the first that <see cref="Claim" /> makes before giving up on a point no other running
    ///     trial holds. Candidates that coincide by construction take few distinct values per generation (LM-CMA's
    ///     Rademacher draws take 2^d), so a few redraws find a free one whenever one exists; when none does (a PSO
    ///     particle at rest at the global best moves nowhere, whatever it draws), the cap ends the search and the
    ///     trial is sampled independently.
    /// </summary>
    internal const int MaxRedraws = 32;

    private readonly PerStudyState<Dictionary<int, (int Generation, double[] Point)>> _byStudy = new();

    /// <summary>
    ///     Hands the trial the first of <paramref name="candidate" /> and up to <see cref="MaxRedraws" /> redraws that no
    ///     other running trial of the generation holds, and records it as the trial's point for the asks made while it
    ///     runs.
    /// </summary>
    /// <param name="study">The trial's study, whose running trials are read from the storage.</param>
    /// <param name="trial">The trial being sampled.</param>
    /// <param name="generation">The generation the candidate belongs to.</param>
    /// <param name="candidate">The first draw, already clamped to the box. Recorded without a copy.</param>
    /// <param name="redraw">
    ///     Draws another candidate of the same kind, clamped the same way. Called only while the previous draw is held,
    ///     so a sequential run never calls it and its random stream, and with it every seeded result, is unchanged.
    /// </param>
    /// <returns>
    ///     The point handed to the trial, or <c>null</c> when every draw was held: the caller then samples the trial
    ///     independently (an overrun) rather than give two running trials one point. Nothing is recorded then.
    /// </returns>
    /// <exception cref="NumSharp.KeyError">The study no longer exists in storage.</exception>
    /// <exception cref="System.Exception">Whatever <paramref name="redraw" /> throws, unchanged.</exception>
    internal double[]? Claim(Study study, FrozenTrial trial, int generation, double[] candidate, Func<double[]> redraw)
    {
        List<double[]> held = HeldByOtherRunningTrials(study, trial, generation);
        for (int redraws = 0; IsHeld(held, candidate); redraws++)
        {
            if (redraws == MaxRedraws)
                return null;
            candidate = redraw();
        }

        Record(study, trial.Number, generation, candidate);
        return candidate;
    }

    /// <summary>
    ///     The points of <paramref name="generation" /> that other running trials are evaluating. Entries of trials no
    ///     longer running are dropped on the way.
    /// </summary>
    /// <param name="study">The study, whose running trials are read from the storage.</param>
    /// <param name="current">The trial being sampled; never its own holder.</param>
    /// <param name="generation">The current generation; points of earlier generations no longer matter.</param>
    /// <returns>The held points; empty without a storage read when nothing is recorded (a sequential run).</returns>
    /// <exception cref="NumSharp.KeyError">The study no longer exists in storage.</exception>
    internal List<double[]> HeldByOtherRunningTrials(Study study, FrozenTrial current, int generation)
    {
        var points = new List<double[]>();
        Dictionary<int, (int Generation, double[] Point)> handedOut = Of(study);
        if (handedOut.Count == 0)
            return points;

        var running = new HashSet<int>();
        foreach (FrozenTrial other in ConcurrentAsks.OtherRunningTrials(study, current))
            running.Add(other.Number);
        foreach (int number in new List<int>(handedOut.Keys))
        {
            if (!running.Contains(number))
            {
                // Finished (AfterTrial forgets it too) or told by another process: it holds nothing any more.
                handedOut.Remove(number);
                continue;
            }

            var (g, point) = handedOut[number];
            if (g == generation)
                points.Add(point);
        }

        return points;
    }

    /// <summary>Records the point handed to a trial, for the asks made while it runs.</summary>
    /// <param name="study">The trial's study.</param>
    /// <param name="trialNumber">The trial's number.</param>
    /// <param name="generation">The generation the point belongs to.</param>
    /// <param name="point">The point, in the optimizer's unit box; not copied, so the caller must not change it.</param>
    internal void Record(Study study, int trialNumber, int generation, double[] point)
        => Of(study)[trialNumber] = (generation, point);

    /// <summary>Forgets a finished trial's point.</summary>
    /// <param name="study">The trial's study.</param>
    /// <param name="trialNumber">The trial's number.</param>
    internal void Forget(Study study, int trialNumber) => Of(study).Remove(trialNumber);

    /// <summary>Whether <paramref name="point" /> is exactly one of <paramref name="held" />.</summary>
    /// <param name="held">The points other running trials hold.</param>
    /// <param name="point">The candidate.</param>
    /// <returns><c>true</c> when handing out the candidate would give two running trials one point.</returns>
    internal static bool IsHeld(List<double[]> held, double[] point)
    {
        foreach (double[] p in held)
            if (p.AsSpan().SequenceEqual(point))
                return true;
        return false;
    }

    /// <summary>This study's record, created on first use.</summary>
    /// <param name="study">The study.</param>
    /// <returns>The record, keyed by trial number.</returns>
    private Dictionary<int, (int Generation, double[] Point)> Of(Study study)
        => _byStudy.Resolve(study, static () => new Dictionary<int, (int, double[])>());
}

/// <summary>
///     The relative parameters a sampler handed to trials that are still running, per study, so a parameter set a
///     running trial evaluates is generated again instead of handed to a second trial: <see cref="HandedOutPoints" />
///     for samplers whose candidates are parameter sets in external representation, not unit-box points (the genetic
///     samplers' children).
/// </summary>
/// <remarks>
///     <para>
///         Genetic samplers repeat points by construction: uniform crossover can rebuild a parent, and two trials that
///         draw the same parents and crossover mask, with no mutation, build one child. One after another that repeats
///         a finished point, which Optuna does too and which is kept. Asked at the same time, the same child went to
///         two running trials (NSGA-II and NSGA-III: one run in 12 to 20 at 8 workers).
///     </para>
///     <para>
///         The rules of <see cref="HandedOutPoints" /> hold: every member is called under the sampler's monitor, a
///         sequential run never redraws (so its random stream and seeded results are unchanged), and the record steers
///         apart only the asks made through this sampler instance in this process. Parameter sets are compared with
///         Python's <c>==</c> (<see cref="PyJson.ValueEquals(object, object)" />), so a NaN never matches.
///     </para>
/// </remarks>
internal sealed class HandedOutParams
{
    private readonly PerStudyState<Dictionary<int, OrderedDict<string, object?>>> _byStudy = new();

    /// <summary>
    ///     Hands the trial the first of <paramref name="candidate" /> and up to <see cref="HandedOutPoints.MaxRedraws" />
    ///     redraws that no other running trial holds, and records it as the trial's parameters for the asks made while it
    ///     runs.
    /// </summary>
    /// <param name="study">The trial's study, whose running trials are read from the storage.</param>
    /// <param name="trial">The trial being sampled.</param>
    /// <param name="candidate">
    ///     The first draw. An empty one sends the trial to the independent sampler, which draws every parameter afresh,
    ///     so it is returned as it is and nothing is recorded. Recorded without a copy, so the caller must not change it
    ///     after handing it out.
    /// </param>
    /// <param name="redraw">
    ///     Draws another candidate the same way. Called only while the previous draw is held, so a sequential run never
    ///     calls it and its random stream, and with it every seeded result, is unchanged.
    /// </param>
    /// <returns>
    ///     The parameters handed to the trial, or <c>null</c> when every draw was held (a population collapsed onto one
    ///     point): the caller then samples the trial independently rather than give two running trials one point.
    /// </returns>
    /// <exception cref="NumSharp.KeyError">The study no longer exists in storage.</exception>
    /// <exception cref="System.Exception">Whatever <paramref name="redraw" /> throws, unchanged.</exception>
    internal OrderedDict<string, object?>? Claim(
        Study study, FrozenTrial trial, OrderedDict<string, object?> candidate,
        Func<OrderedDict<string, object?>> redraw)
    {
        if (candidate.Count == 0)
            return candidate;

        List<OrderedDict<string, object?>> held = HeldByOtherRunningTrials(study, trial);
        for (int redraws = 0; IsHeld(held, candidate); redraws++)
        {
            if (redraws == HandedOutPoints.MaxRedraws)
                return null;
            candidate = redraw();
        }

        // A redraw can come back empty (the strategy found no parents); then there is nothing to hold either.
        if (candidate.Count > 0)
            Of(study)[trial.Number] = candidate;
        return candidate;
    }

    /// <summary>Forgets a finished trial's parameters.</summary>
    /// <param name="study">The trial's study.</param>
    /// <param name="trialNumber">The trial's number.</param>
    internal void Forget(Study study, int trialNumber) => Of(study).Remove(trialNumber);

    /// <summary>
    ///     The parameter sets other running trials are evaluating. Entries of trials no longer running are dropped on the
    ///     way.
    /// </summary>
    /// <param name="study">The study, whose running trials are read from the storage.</param>
    /// <param name="current">The trial being sampled; never its own holder.</param>
    /// <returns>The held parameter sets; empty without a storage read when nothing is recorded (a sequential run).</returns>
    /// <exception cref="NumSharp.KeyError">The study no longer exists in storage.</exception>
    private List<OrderedDict<string, object?>> HeldByOtherRunningTrials(Study study, FrozenTrial current)
    {
        var sets = new List<OrderedDict<string, object?>>();
        Dictionary<int, OrderedDict<string, object?>> handedOut = Of(study);
        if (handedOut.Count == 0)
            return sets;

        var running = new HashSet<int>();
        foreach (FrozenTrial other in ConcurrentAsks.OtherRunningTrials(study, current))
            running.Add(other.Number);
        foreach (int number in new List<int>(handedOut.Keys))
        {
            if (running.Contains(number))
                sets.Add(handedOut[number]);
            else
                handedOut.Remove(number);   // finished, or told by another process: it holds nothing any more
        }

        return sets;
    }

    /// <summary>Whether <paramref name="candidate" /> equals one of <paramref name="held" /> under Python's <c>==</c>.</summary>
    /// <param name="held">The parameter sets other running trials hold.</param>
    /// <param name="candidate">The candidate.</param>
    /// <returns><c>true</c> when handing out the candidate would give two running trials one point.</returns>
    /// <exception cref="NumSharp.RuntimeError">A parameter value nests lists or dictionaries deeper than JSON allows.</exception>
    private static bool IsHeld(List<OrderedDict<string, object?>> held, OrderedDict<string, object?> candidate)
    {
        foreach (OrderedDict<string, object?> set in held)
            if (PyJson.ValueEquals(set, candidate))
                return true;
        return false;
    }

    /// <summary>This study's record, created on first use.</summary>
    /// <param name="study">The study.</param>
    /// <returns>The record, keyed by trial number.</returns>
    private Dictionary<int, OrderedDict<string, object?>> Of(Study study)
        => _byStudy.Resolve(study, static () => new Dictionary<int, OrderedDict<string, object?>>());
}

/// <summary>
///     The trials of each study that a population sampler sent to its independent sampler on purpose, because every
///     candidate of the current generation was held by a running trial (an overrun under <c>Optimize(nJobs)</c>):
///     their independent draws are intended, so the "independent sampling" warning is not logged for them.
/// </summary>
/// <remarks>
///     Not synchronized: every member is called from a sampler hook, which runs under the sampler's monitor. An entry
///     is removed by <see cref="Remove" /> from the sampler's <c>AfterTrial</c>; one left behind (a process that died
///     mid-trial) only silences a warning for a later trial of the same number.
/// </remarks>
internal sealed class OverrunTrials
{
    private readonly PerStudyState<HashSet<int>> _numbers = new();

    /// <summary>Records that the trial was sent to the independent sampler on purpose.</summary>
    /// <param name="study">The trial's study.</param>
    /// <param name="trialNumber">The trial's number.</param>
    internal void Add(Study study, int trialNumber)
        => _numbers.Resolve(study, static () => new HashSet<int>()).Add(trialNumber);

    /// <summary>Whether the trial was sent to the independent sampler on purpose.</summary>
    /// <param name="study">The trial's study.</param>
    /// <param name="trialNumber">The trial's number.</param>
    /// <returns><c>true</c> for an overrun trial, whose independent draws must not be warned about.</returns>
    internal bool Contains(Study study, int trialNumber)
        => _numbers.Resolve(study, static () => new HashSet<int>()).Contains(trialNumber);

    /// <summary>Forgets the trial once it has finished.</summary>
    /// <param name="study">The trial's study.</param>
    /// <param name="trialNumber">The trial's number.</param>
    internal void Remove(Study study, int trialNumber)
        => _numbers.Resolve(study, static () => new HashSet<int>()).Remove(trialNumber);
}

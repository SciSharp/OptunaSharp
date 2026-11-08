using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Reflection;
using System.Threading;
using AwesomeAssertions;
using OptunaSharp.Collections;
using OptunaSharp.Distributions;
using OptunaSharp.Internal;
using OptunaSharp.Samplers;

namespace OptunaSharp.Tests.Concurrency;

/// <summary>
///     <see cref="TrialExecutorOptions.SamplerConcurrency" /> (THREADING.md §6.3): with <see cref="SamplerConcurrency.Shared" />,
///     the default and Optuna's model, every <c>nJobs</c> worker samples through the study's one sampler, one call at a time.
///     With <see cref="SamplerConcurrency.PerWorker" />, each worker samples with an instance of its own made from the study's
///     sampler, so sampling runs in parallel, each trial is sampled and told by one instance, and the instances never reach
///     another call or another thread. Samplers that would hand workers duplicate samples make no instance and stay shared.
/// </summary>
/// <remarks>
///     The probe sampler here overrides the internal <see cref="BaseSampler.CreateWorkerInstance" /> (the test assembly is a
///     friend of Core), which no sampler outside OptunaSharp can do: an outside sampler always stays shared.
/// </remarks>
[TestClass]
public class SamplerConcurrencyModeTests
{
    /// <summary>Options asking for per-worker sampler instances on the dedicated threads.</summary>
    /// <returns>A fresh instance, so no test sees another's change.</returns>
    private static TrialExecutorOptions PerWorker() => new() { SamplerConcurrency = SamplerConcurrency.PerWorker };

    // ------------------------------------------------------------------ the default

    /// <summary>
    ///     With the default options every trial of an <c>nJobs</c> run is sampled, set up and told by the study's own
    ///     sampler, one call at a time, as in Optuna, and no worker instance is ever made.
    /// </summary>
    [TestMethod]
    public void Default_IsShared_EveryCallReachesTheStudysSamplerOneAtATime()
    {
        var ledger = new WorkerProbeLedger(makesWorkerInstances: true);
        var study = Optuna.CreateStudy(sampler: new WorkerProbeSampler(ledger));

        study.Optimize(t => t.SuggestFloat("x", 0, 1), nTrials: 24, nJobs: 4);

        ledger.InstancesCreated.Should().Be(0, "the default never asks the sampler for a worker instance");
        ledger.InstancesServing().Should().Equal(new[] { WorkerProbeLedger.PrototypeId }, "only the study's sampler served");
        ledger.PeakInside.Should().Be(1, "the shared sampler's monitor admits one call at a time");
    }

    // ------------------------------------------------------------------ per worker

    /// <summary>
    ///     With <see cref="SamplerConcurrency.PerWorker" /> the four workers sample at the same instant: the first four
    ///     sampling calls wait for each other inside the sampler, which only distinct instances (distinct monitors) allow.
    ///     The study's own sampler serves no trial, and the call makes exactly as many instances as it has workers.
    /// </summary>
    [TestMethod]
    public void PerWorker_SamplesOnEveryWorkerAtOnce_EachWithItsOwnInstance()
    {
        ConcurrencyGun.RequireParallelHardware();
        var ledger = new WorkerProbeLedger(makesWorkerInstances: true, rendezvous: 4);
        var study = Optuna.CreateStudy(sampler: new WorkerProbeSampler(ledger));

        study.Optimize(t => t.SuggestFloat("x", 0, 1), nTrials: 16, nJobs: 4, executorOptions: PerWorker());

        ledger.MissedRendezvous.Should().Be(0, "four sampling calls must be inside the sampler at once");
        ledger.PeakInside.Should().Be(4);
        ledger.InstancesCreated.Should().Be(4, "one instance per worker, reused by the call's later trials");
        ledger.InstancesServing().Should().NotContain(WorkerProbeLedger.PrototypeId, "the study's sampler served no trial");
        study.Trials.Should().HaveCount(16).And.OnlyContain(t => t.State == TrialState.Complete);
    }

    /// <summary>
    ///     Each trial is set up, sampled and told by one instance, the one its worker rented: <c>AfterTrial</c> reaches the
    ///     instance whose state sampled the trial, exactly once per trial, and each instance is reseeded before each of
    ///     its trials as Optuna reseeds a worker.
    /// </summary>
    [TestMethod]
    public void PerWorker_EachTrialIsSetUpSampledAndToldByOneInstance()
    {
        var ledger = new WorkerProbeLedger(makesWorkerInstances: true);
        var study = Optuna.CreateStudy(sampler: new WorkerProbeSampler(ledger));

        study.Optimize(t => t.SuggestFloat("x", 0, 1) + t.SuggestFloat("y", 0, 1), nTrials: 40, nJobs: 6,
            executorOptions: PerWorker());

        foreach (FrozenTrial trial in study.Trials)
        {
            IReadOnlyList<(string Call, int Instance)> calls = ledger.CallsOf(trial.Number);
            calls.Select(c => c.Instance).Distinct().Should().ContainSingle($"trial {trial.Number} belongs to one instance");
            calls.Count(c => c.Call == "before").Should().Be(1);
            calls.Count(c => c.Call == "sample").Should().Be(2);
            calls.Count(c => c.Call == "after").Should().Be(1, $"trial {trial.Number} is told once, to its own instance");
        }

        ledger.InstancesCreated.Should().BeInRange(1, 6, "never more instances than workers");
        ledger.Reseeds.Should().Be(40, "each worker reseeds its instance before its trial");
    }

    /// <summary>
    ///     A sampler that cannot make worker instances (every sampler written outside OptunaSharp, and the ones that would
    ///     hand workers duplicate samples) keeps the shared instance under <see cref="SamplerConcurrency.PerWorker" />, one
    ///     call at a time, and is asked only once whether it can.
    /// </summary>
    [TestMethod]
    public void PerWorker_ASamplerWithoutWorkerInstances_StaysSharedOneCallAtATime()
    {
        var ledger = new WorkerProbeLedger(makesWorkerInstances: false);
        var study = Optuna.CreateStudy(sampler: new WorkerProbeSampler(ledger));

        study.Optimize(t => t.SuggestFloat("x", 0, 1), nTrials: 24, nJobs: 4, executorOptions: PerWorker());

        ledger.InstancesServing().Should().Equal(new[] { WorkerProbeLedger.PrototypeId });
        ledger.PeakInside.Should().Be(1);
        ledger.CreationRequests.Should().Be(1, "the call remembers that the sampler stays shared");
    }

    /// <summary><c>nJobs: 1</c> ignores the setting: the calling thread samples with the study's sampler, as always.</summary>
    [TestMethod]
    public void PerWorker_WithOneJob_UsesTheStudysOwnSampler()
    {
        var ledger = new WorkerProbeLedger(makesWorkerInstances: true);
        var study = Optuna.CreateStudy(sampler: new WorkerProbeSampler(ledger));

        study.Optimize(t => t.SuggestFloat("x", 0, 1), nTrials: 8, nJobs: 1, executorOptions: PerWorker());

        ledger.CreationRequests.Should().Be(0);
        ledger.InstancesServing().Should().Equal(new[] { WorkerProbeLedger.PrototypeId });
    }

    /// <summary>
    ///     A worker's instance is bound to its trials only: once the run ends no thread keeps it, so a later call on the
    ///     same reused threads (dedicated or pool) samples with the study's sampler again, and so does the calling thread.
    /// </summary>
    /// <param name="kind">The executor whose reused threads are checked.</param>
    [TestMethod]
    [DataRow(TrialExecutorKind.DedicatedThreads)]
    [DataRow(TrialExecutorKind.ThreadPool)]
    public void PerWorker_LeavesNoWorkerInstanceOnTheReusedThreads(TrialExecutorKind kind)
    {
        var ledger = new WorkerProbeLedger(makesWorkerInstances: true);
        var study = Optuna.CreateStudy(sampler: new WorkerProbeSampler(ledger));
        var perWorker = new TrialExecutorOptions { Executor = kind, SamplerConcurrency = SamplerConcurrency.PerWorker };
        var shared = new TrialExecutorOptions { Executor = kind };

        study.Optimize(t => t.SuggestFloat("x", 0, 1), nTrials: 16, nJobs: 4, executorOptions: perWorker);
        var stale = new ConcurrentBag<int>();
        study.Optimize(t =>
        {
            // The same threads as the run before (the executor reuses them): none may still serve a worker's instance.
            if (!ReferenceEquals(study.SamplerForCurrentThread, study.Sampler))
                stale.Add(t.Number);
            return t.SuggestFloat("x", 0, 1);
        }, nTrials: 16, nJobs: 4, executorOptions: shared);

        stale.Should().BeEmpty();
        study.ThreadLocal.SamplerOverride.Should().BeNull("the calling thread never takes a worker's instance");
    }

    // ------------------------------------------------------------------ the pool

    /// <summary>
    ///     The pool hands an instance back to later trials of the same sampler, makes new instances for a sampler swapped
    ///     onto the study meanwhile (never handing out the old sampler's), and remembers a sampler that has none.
    /// </summary>
    [TestMethod]
    public void WorkerPool_ReusesInstancesPerSampler_AndRemembersASharedOne()
    {
        var pool = new SamplerWorkerPool();
        var first = new RandomSampler(1);
        var swapped = new RandomSampler(2);
        var shared = new WorkerProbeSampler(new WorkerProbeLedger(makesWorkerInstances: false));

        BaseSampler a = pool.Rent(first)!;
        a.Should().NotBeNull().And.NotBeSameAs(first);
        pool.Return(first, a);

        BaseSampler b = pool.Rent(swapped)!;
        b.Should().NotBeSameAs(a, "an instance of the old sampler is never handed out for the new one");
        pool.Return(swapped, b);

        pool.Rent(first).Should().BeSameAs(a, "an idle instance of the same sampler is reused");
        pool.Rent(shared).Should().BeNull();
        pool.Rent(shared).Should().BeNull();
        shared.Ledger.CreationRequests.Should().Be(1);
    }

    // ------------------------------------------------------------------ which samplers take part

    /// <summary>
    ///     Exactly the samplers THREADING.md §6.3 and <see cref="SamplerConcurrency.PerWorker" /> list make worker
    ///     instances. A sampler added to the list must be added to the documentation and to the duplicate sweep below.
    /// </summary>
    [TestMethod]
    public void OnlyTheListedSamplers_MakeWorkerInstances()
    {
        string[] overriding = typeof(BaseSampler).Assembly.GetTypes()
            .Where(t => t != typeof(BaseSampler) && typeof(BaseSampler).IsAssignableFrom(t))
            .Where(t => t.GetMethod(nameof(BaseSampler.CreateWorkerInstance),
                BindingFlags.Instance | BindingFlags.NonPublic, Type.EmptyTypes)?.DeclaringType == t)
            .Select(t => t.Name)
            .OrderBy(n => n, StringComparer.Ordinal)
            .ToArray();

        overriding.Should().Equal("CmaEsSampler", "GridSampler", "PartialFixedSampler", "RandomSampler", "TPESampler",
            "VkdCmaSampler");
    }

    /// <summary>
    ///     A taking-part sampler makes an instance of its own type with a generator of its own; a subclass (which may add
    ///     state the copy would not carry) and a wrapper over a non-taking-part sampler make none.
    /// </summary>
    [TestMethod]
    public void WorkerInstances_AreExactCopies_AndSubclassesAndSharedWrappedSamplersMakeNone()
    {
        var tpe = new TPESampler(seed: 3);
        BaseSampler? copy = tpe.CreateWorkerInstance();
        copy.Should().NotBeNull().And.BeOfType<TPESampler>().And.NotBeSameAs(tpe);

        new SubclassedRandomSampler().CreateWorkerInstance().Should().BeNull();
        new PartialFixedSampler(new Dictionary<string, object?> { ["x"] = 0.5 }, new QMCSampler(seed: 1))
            .CreateWorkerInstance().Should().BeNull("QMC hands every instance the same sequence");
        new CmaEsSampler(independentSampler: new QMCSampler(seed: 1)).CreateWorkerInstance()
            .Should().BeNull("its independent sampler cannot be copied");
        new GPSampler(seed: 1).CreateWorkerInstance()
            .Should().BeNull("instances that see the same history optimize the same acquisition function");
    }

    /// <summary>
    ///     Every taking-part sampler, driven by eight workers with instances of their own on one study, never proposes the
    ///     same point twice: the instances share only the storage, and each draws from fresh OS entropy.
    /// </summary>
    /// <param name="name">The sampler under test.</param>
    /// <remarks>
    ///     CMA-ES and VkD-CMA tolerate one repeat, which is Optuna's own and not the parallelism's: each trial seeds the
    ///     optimizer with a draw from 1 to 65535 plus its number, so two trials of one generation (same optimizer state)
    ///     repeat a point when those seeds collide, about once in 65,535 pairs, under <see cref="SamplerConcurrency.Shared" />
    ///     as well. Two such collisions in one run (about 2 in a million) would be a real failure.
    /// </remarks>
    [TestMethod]
    [DataRow("random")]
    [DataRow("tpe")]
    [DataRow("tpe-multivariate")]
    [DataRow("cmaes")]
    [DataRow("vkd")]
    [DataRow("grid")]
    [DataRow("partialfixed")]
    public void TakingPartSamplers_UnderPerWorker_ProposeNoPointTwice(string name)
    {
        using var quiet = new QuietLogs();
        BaseSampler sampler = MakeSampler(name);
        sampler.CreateWorkerInstance().Should().NotBeNull($"{name} takes part in SamplerConcurrency.PerWorker");
        var study = Optuna.CreateStudy(sampler: sampler);
        int trials = name == "grid" ? 36 : 48;

        study.Optimize(t =>
        {
            if (name == "grid")
                return t.SuggestFloat("x", -10, 10, step: 4) + t.SuggestInt("y", 0, 2)
                    + (t.SuggestCategorical("c", new[] { "a", "b" }) == "a" ? 0 : 1);
            double x = t.SuggestFloat("x", -10, 10), y = t.SuggestFloat("y", -10, 10), z = t.SuggestFloat("z", 0, 1);
            return (x - 2) * (x - 2) + (y + 1) * (y + 1) + z;
        }, nTrials: trials, nJobs: 8, executorOptions: PerWorker());

        study.Trials.Should().HaveCount(trials).And.OnlyContain(t => t.State == TrialState.Complete);
        int repeats = Repeats(study);
        if (name is "cmaes" or "vkd")
            repeats.Should().BeLessThanOrEqualTo(1, "only Optuna's own seed collision may repeat a CMA point");
        else
            repeats.Should().Be(0);
    }

    /// <summary>Builds the sampler a sweep case names, seeded so a failure names a reproducible configuration.</summary>
    /// <param name="name">One of the names of <see cref="TakingPartSamplers_UnderPerWorker_ProposeNoPointTwice" />.</param>
    /// <returns>A new sampler.</returns>
    /// <exception cref="ArgumentException"><paramref name="name" /> is not a sweep case.</exception>
    private static BaseSampler MakeSampler(string name) => name switch
    {
        "random" => new RandomSampler(seed: 1),
        "tpe" => new TPESampler(seed: 1),
        "tpe-multivariate" => new TPESampler(seed: 1, multivariate: true),
        "cmaes" => new CmaEsSampler(seed: 1),
        "vkd" => new VkdCmaSampler(seed: 1),
        // 6 × 3 × 2 = 36 points, one per trial: the grid's first trials take grid ids by trial number, so a repeat would
        // mean two workers read one point.
        "grid" => new GridSampler(new Dictionary<string, IReadOnlyList<CategoricalChoice>>
        {
            ["x"] = Enumerable.Range(0, 6).Select(i => CategoricalChoice.From(-10.0 + 4.0 * i)).ToList(),
            ["y"] = Enumerable.Range(0, 3).Select(i => CategoricalChoice.From((long)i)).ToList(),
            ["c"] = new[] { "a", "b" }.Select(s => CategoricalChoice.From(s)).ToList(),
        }, seed: 1),
        "partialfixed" => new PartialFixedSampler(new Dictionary<string, object?> { ["z"] = 0.5 }, new TPESampler(seed: 1)),
        _ => throw new ArgumentException("Unknown sweep case " + name, nameof(name)),
    };

    /// <summary>Counts the trials whose whole parameter vector an earlier trial already had (exact values).</summary>
    /// <param name="study">The study whose trials are compared.</param>
    /// <returns>The trial count minus the number of distinct vectors.</returns>
    private static int Repeats(Study study)
    {
        List<string> vectors = study.Trials
            .Select(t => string.Join("|", t.Params.OrderBy(p => p.Key, StringComparer.Ordinal)
                .Select(p => p.Key + "=" + Convert.ToString(p.Value, CultureInfo.InvariantCulture))))
            .ToList();
        return vectors.Count - vectors.Distinct(StringComparer.Ordinal).Count();
    }

    /// <summary>A sampler outside the list by being a subclass: its copy would lose whatever the subclass adds.</summary>
    private sealed class SubclassedRandomSampler : RandomSampler
    {
    }
}

/// <summary>
///     The record that every <see cref="WorkerProbeSampler" /> made from one prototype shares: which instance served each
///     call of each trial, how many sampling calls were inside the instances at once, how often a worker instance was
///     asked for and made, and whether the first sampling calls met inside the sampler.
/// </summary>
internal sealed class WorkerProbeLedger
{
    /// <summary>The identifier of the prototype, the instance the study was created with.</summary>
    internal const int PrototypeId = 0;

    private static readonly TimeSpan RendezvousTimeout = TimeSpan.FromSeconds(10);

    private readonly ConcurrentQueue<(int Trial, string Call, int Instance)> _calls = new();
    private readonly int _rendezvous;
    private readonly CountdownEvent? _arrivals;
    private int _arrived;
    private int _missedRendezvous;
    private int _inside;
    private int _peakInside;
    private int _creationRequests;
    private int _created;
    private int _reseeds;

    /// <summary>Creates a ledger for one prototype.</summary>
    /// <param name="makesWorkerInstances">Whether the probe answers <see cref="BaseSampler.CreateWorkerInstance" /> with an instance.</param>
    /// <param name="rendezvous">
    ///     How many of the first sampling calls must be inside the probe at once before any proceeds (0 for none). Calls
    ///     that never meet (a serialized sampler) wait <see cref="RendezvousTimeout" />, then proceed and count a miss.
    /// </param>
    internal WorkerProbeLedger(bool makesWorkerInstances, int rendezvous = 0)
    {
        MakesWorkerInstances = makesWorkerInstances;
        _rendezvous = rendezvous;
        _arrivals = rendezvous > 0 ? new CountdownEvent(rendezvous) : null;
    }

    /// <summary>Whether the probe makes worker instances.</summary>
    internal bool MakesWorkerInstances { get; }

    /// <summary>The most sampling calls inside the probe's instances at one instant.</summary>
    internal int PeakInside => Volatile.Read(ref _peakInside);

    /// <summary>How many first calls gave up waiting for the others (each waited the whole timeout).</summary>
    internal int MissedRendezvous => Volatile.Read(ref _missedRendezvous);

    /// <summary>How many times a worker instance was asked for.</summary>
    internal int CreationRequests => Volatile.Read(ref _creationRequests);

    /// <summary>How many worker instances were made.</summary>
    internal int InstancesCreated => Volatile.Read(ref _created);

    /// <summary>How many times any instance was reseeded.</summary>
    internal int Reseeds => Volatile.Read(ref _reseeds);

    /// <summary>The identifiers of the instances that served at least one trial call, ascending.</summary>
    /// <returns>The identifiers; <see cref="PrototypeId" /> is the study's own sampler.</returns>
    internal int[] InstancesServing() => _calls.Select(c => c.Instance).Distinct().OrderBy(i => i).ToArray();

    /// <summary>The calls one trial received, in the order they were recorded.</summary>
    /// <param name="trialNumber">The trial's number.</param>
    /// <returns>Each call's kind (<c>before</c>, <c>sample</c>, <c>after</c>) and the instance that served it.</returns>
    internal IReadOnlyList<(string Call, int Instance)> CallsOf(int trialNumber)
        => _calls.Where(c => c.Trial == trialNumber).Select(c => (c.Call, c.Instance)).ToList();

    /// <summary>Records a request for a worker instance and, when the probe makes them, the identifier of the new one.</summary>
    /// <returns>The new instance's identifier, or null when the probe makes none.</returns>
    internal int? RequestInstance()
    {
        Interlocked.Increment(ref _creationRequests);
        return MakesWorkerInstances ? Interlocked.Increment(ref _created) : null;
    }

    /// <summary>Records one call of a trial on an instance.</summary>
    /// <param name="trialNumber">The trial's number.</param>
    /// <param name="call">The call's kind.</param>
    /// <param name="instance">The serving instance's identifier.</param>
    internal void Record(int trialNumber, string call, int instance) => _calls.Enqueue((trialNumber, call, instance));

    /// <summary>Records a reseed of any instance.</summary>
    internal void RecordReseed() => Interlocked.Increment(ref _reseeds);

    /// <summary>
    ///     Runs one sampling call's stay inside the probe: counts it, makes the first <c>rendezvous</c> calls wait for each
    ///     other, and holds it briefly so a second call admitted at the same time would be seen.
    /// </summary>
    /// <param name="sample">The sampling itself.</param>
    /// <returns>What <paramref name="sample" /> returned.</returns>
    internal double Inside(Func<double> sample)
    {
        Watermark.Raise(ref _peakInside, Interlocked.Increment(ref _inside));
        try
        {
            // Only the first calls take part; the CountdownEvent would throw on a signal past zero.
            if (_arrivals is not null && Interlocked.Increment(ref _arrived) <= _rendezvous)
            {
                _arrivals.Signal();
                if (!_arrivals.Wait(RendezvousTimeout))
                    Interlocked.Increment(ref _missedRendezvous);
            }

            Thread.SpinWait(300);
            return sample();
        }
        finally
        {
            Interlocked.Decrement(ref _inside);
        }
    }
}

/// <summary>
///     A sampler that records which of its instances served each call (THREADING.md §6.3). It places each trial's floats
///     by the trial's number (a low-discrepancy sequence, no generator to share) and makes worker instances when its
///     ledger says so, through the internal <see cref="BaseSampler.CreateWorkerInstance" /> that only OptunaSharp's
///     samplers (and this friend assembly) can override.
/// </summary>
internal sealed class WorkerProbeSampler : BaseSampler
{
    // The golden-ratio fraction: trial n samples at frac(n · φ⁻¹) of the range, distinct for every trial number.
    private const double GoldenFraction = 0.6180339887498949;

    private readonly int _id;

    /// <summary>Creates the prototype of a ledger, the instance a study is created with.</summary>
    /// <param name="ledger">The record this instance and its worker instances share.</param>
    internal WorkerProbeSampler(WorkerProbeLedger ledger) : this(ledger, WorkerProbeLedger.PrototypeId)
    {
    }

    /// <summary>Creates an instance with a given identifier.</summary>
    /// <param name="ledger">The record this instance shares.</param>
    /// <param name="id">The identifier recorded with every call it serves.</param>
    private WorkerProbeSampler(WorkerProbeLedger ledger, int id)
    {
        Ledger = ledger;
        _id = id;
    }

    /// <summary>The record this instance shares with the others made from its prototype.</summary>
    internal WorkerProbeLedger Ledger { get; }

    /// <summary>A new instance sharing the ledger when the ledger allows one, else null (shared access).</summary>
    /// <returns>The worker instance, or null.</returns>
    internal override BaseSampler? CreateWorkerInstance()
        => Ledger.RequestInstance() is int id ? new WorkerProbeSampler(Ledger, id) : null;

    /// <inheritdoc />
    public override OrderedDict<string, BaseDistribution> InferRelativeSearchSpace(Study study, FrozenTrial trial)
        => new();

    /// <inheritdoc />
    public override OrderedDict<string, object?> SampleRelative(
        Study study, FrozenTrial trial, OrderedDict<string, BaseDistribution> searchSpace)
        => new();

    /// <inheritdoc />
    public override object? SampleIndependent(Study study, FrozenTrial trial, string paramName, BaseDistribution paramDistribution)
    {
        Ledger.Record(trial.Number, "sample", _id);
        var range = (FloatDistribution)paramDistribution;
        double fraction = trial.Number * GoldenFraction % 1.0;
        return Ledger.Inside(() => range.Low + (range.High - range.Low) * fraction);
    }

    /// <inheritdoc />
    public override void BeforeTrial(Study study, FrozenTrial trial) => Ledger.Record(trial.Number, "before", _id);

    /// <inheritdoc />
    public override void AfterTrial(Study study, FrozenTrial trial, TrialState state, IReadOnlyList<double>? values)
        => Ledger.Record(trial.Number, "after", _id);

    /// <inheritdoc />
    public override void ReseedRng() => Ledger.RecordReseed();
}

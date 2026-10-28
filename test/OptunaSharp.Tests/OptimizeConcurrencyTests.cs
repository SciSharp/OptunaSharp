using System;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Threading;
using System.Threading.Tasks;
using AwesomeAssertions;
using NumSharp;
using OptunaSharp.Pruners;
using OptunaSharp.Samplers;

namespace OptunaSharp.Tests;

/// <summary>
///     In-process concurrency of one study: <c>Optimize(nJobs)</c> and its threads sharing one sampler and pruner,
///     and the optimize loop's own semantics, which follow Python's <c>ThreadPoolExecutor</c> block. The expected
///     outcomes were observed on Optuna 5.0 (see each test).
/// </summary>
[TestClass]
public class OptimizeConcurrencyTests
{
    /// <summary>
    ///     Eight threads share one seeded sampler: no two trials may get the same draw. NumSharp's generator takes no
    ///     lock (numpy's <c>RandomState</c> does), so two threads drawing at once received bit-identical values.
    /// </summary>
    [TestMethod]
    public void NJobs_SharedRandomSampler_NeverRepeatsADraw()
    {
        var study = Optuna.CreateStudy(sampler: new RandomSampler(seed: 1));

        study.Optimize(t => t.SuggestFloat("x", -10, 10), nTrials: 400, nJobs: 8);

        var xs = study.Trials.Select(t => Convert.ToDouble(t.Params["x"])).ToList();
        xs.Should().HaveCount(400).And.OnlyHaveUniqueItems("concurrent trials must not consume the same generator output");
    }

    /// <summary>The same for TPE, whose startup trials and candidate draws share its generator.</summary>
    [TestMethod]
    public void NJobs_SharedTpeSampler_NeverRepeatsADraw()
    {
        var study = Optuna.CreateStudy(sampler: new TPESampler(seed: 2));

        study.Optimize(t => Math.Pow(t.SuggestFloat("x", -10, 10), 2), nTrials: 120, nJobs: 8);

        study.Trials.Select(t => Convert.ToDouble(t.Params["x"])).Should().HaveCount(120).And.OnlyHaveUniqueItems();
    }

    /// <summary>
    ///     Samplers with state of their own, shared by eight workers: every trial must complete. Before every sampler
    ///     call ran under the sampler's monitor, MOEA/D failed 5–34 of 400 trials (<c>KeyNotFoundException</c>), and
    ///     JANUS and JANUS-DE 21–52 ("Collection was modified"). REMBO's default configuration, which has no inner
    ///     sampler, threw <c>NullReferenceException</c> from <c>ReseedRng</c> in every worker, before the first trial.
    /// </summary>
    /// <param name="name">The sampler under test.</param>
    /// <param name="nTrials">The trial count; REMBO's Gaussian-process proposer is slow, and it failed before trial 0.</param>
    [TestMethod]
    [DataRow("moead", 120)]
    [DataRow("janus", 120)]
    [DataRow("janusde", 120)]
    [DataRow("rembo", 24)]
    public void NJobs_SamplersWithTheirOwnState_CompleteEveryTrial(string name, int nTrials)
    {
        BaseSampler sampler = name switch
        {
            "moead" => new MoeaDSampler(populationSize: 20, seed: 8),
            "janus" => new JanusSampler(seed: 8),
            "janusde" => new JanusDeSampler(seed: 8),
            _ => new RandomEmbeddingSampler(2, seed: 8),
        };

        if (name == "moead")
        {
            var mo = Optuna.CreateStudy(sampler: sampler, directions: new[] { StudyDirection.Minimize, StudyDirection.Minimize });
            mo.Optimize(t =>
            {
                double x = t.SuggestFloat("x", -10, 10);
                double y = t.SuggestFloat("y", -10, 10);
                return new[] { (x - 2) * (x - 2) + y * y, x * x + (y - 1) * (y - 1) };
            }, nTrials: nTrials, nJobs: 8);
            mo.Trials.Should().HaveCount(nTrials).And.OnlyContain(t => t.State == TrialState.Complete);
            return;
        }

        var study = Optuna.CreateStudy(sampler: sampler);
        study.Optimize(t =>
        {
            double x = t.SuggestFloat("x", -10, 10);
            double y = t.SuggestFloat("y", -10, 10);
            double z = t.SuggestFloat("z", -10, 10);
            return (x - 2) * (x - 2) + (y + 1) * (y + 1) + z * z;
        }, nTrials: nTrials, nJobs: 8);
        study.Trials.Should().HaveCount(nTrials).And.OnlyContain(t => t.State == TrialState.Complete);
    }

    /// <summary>
    ///     The natural-evolution-strategy samplers give every concurrent trial its own candidate. They took a
    ///     generation's member from the number of FINISHED members, so every worker that asked while members were still
    ///     running got the same one: at nJobs 16, 48 of 120 trials evaluated a point another trial already had. The
    ///     index now counts running members too, and an ask past the population draws an extra candidate from the
    ///     generation's noise stream instead of repeating the last member.
    /// </summary>
    /// <param name="name">The sampler under test.</param>
    [TestMethod]
    [DataRow("openaies")]
    [DataRow("snes")]
    [DataRow("xnes")]
    public void NJobs_EvolutionStrategySamplers_NeverRepeatACandidate(string name)
    {
        BaseSampler sampler = name switch
        {
            "openaies" => new OpenAiEsSampler(seed: 9),
            "snes" => new SnesSampler(seed: 9),
            _ => new XnesSampler(seed: 9),
        };
        var study = Optuna.CreateStudy(sampler: sampler);

        study.Optimize(t =>
        {
            double x = t.SuggestFloat("x", -10, 10);
            double y = t.SuggestFloat("y", -10, 10);
            double z = t.SuggestFloat("z", -10, 10);
            return (x - 2) * (x - 2) + (y + 1) * (y + 1) + z * z;
        }, nTrials: 120, nJobs: 8);

        var points = study.Trials
            .Select(t => (Convert.ToDouble(t.Params["x"]), Convert.ToDouble(t.Params["y"]), Convert.ToDouble(t.Params["z"])))
            .ToList();
        points.Should().HaveCount(120).And.OnlyHaveUniqueItems("concurrent trials must not evaluate one candidate twice");
    }

    /// <summary>
    ///     Hyperband with an automatic maximum resource builds its brackets lazily, on the first prune check after
    ///     a trial completes. Several threads used to build them at once into shared lists and crash Optimize.
    /// </summary>
    /// <exception cref="TrialPruned">Thrown only inside the objective, where Optimize records the trial as PRUNED.</exception>
    [TestMethod]
    public void NJobs_HyperbandPruner_BuildsItsBracketsOnceAndPrunes()
    {
        var study = Optuna.CreateStudy(sampler: new RandomSampler(3), pruner: new HyperbandPruner());

        study.Optimize(t =>
        {
            double x = t.SuggestFloat("x", 0, 1);
            for (int step = 0; step < 9; step++)
            {
                t.Report(x + step, step);
                if (t.ShouldPrune())
                    throw new TrialPruned();
            }
            return x;
        }, nTrials: 120, nJobs: 8);

        var trials = study.Trials;
        trials.Should().HaveCount(120).And.OnlyContain(t => t.State == TrialState.Complete || t.State == TrialState.Pruned);
        trials.Should().Contain(t => t.State == TrialState.Pruned, "the brackets were built and used");
    }

    /// <summary>
    ///     After an nJobs run, later single-job Optimize calls of the same study on thread-pool threads must work.
    ///     Python's executor threads die with their loop flag; .NET's pool threads are reused, and the stale flag
    ///     made 61 of 64 such calls fail with "Nested invocation of `Study.optimize` method isn't allowed."
    /// </summary>
    [TestMethod]
    public void Optimize_OnPoolThreadsAfterAnNJobsRun_IsNotANestedInvocation()
    {
        var study = Optuna.CreateStudy(sampler: new RandomSampler(4));
        study.Optimize(t => t.SuggestFloat("x", 0, 1), nTrials: 64, nJobs: 8);

        int nested = 0;
        Parallel.For(0, 64, _ =>
        {
            try
            {
                study.Optimize(t => t.SuggestFloat("x", 0, 1), nTrials: 1);
            }
            catch (RuntimeError)
            {
                Interlocked.Increment(ref nested);
            }
        });

        nested.Should().Be(0);
        study.Trials.Should().HaveCount(128);
    }

    /// <summary>
    ///     An uncaught objective error with nJobs: Python raises it as is, but only after every trial still running
    ///     has finished (observed on Optuna 5.0: <c>RuntimeError boom</c>, 0 trials left RUNNING). Optimize used to
    ///     return with three trials still running in the background.
    /// </summary>
    /// <exception cref="InvalidOperationException">Thrown by trial 3's objective; the assertion expects Optimize to rethrow it.</exception>
    [TestMethod]
    public void NJobs_ObjectiveError_WaitsForRunningTrials_ThenRaisesTheOriginalException()
    {
        var study = Optuna.CreateStudy(sampler: new RandomSampler(5));

        var act = () => study.Optimize(t =>
        {
            double x = t.SuggestFloat("x", 0, 1);
            if (t.Number == 3)
                throw new InvalidOperationException("boom");
            Thread.Sleep(300);
            return x;
        }, nTrials: 40, nJobs: 4);

        act.Should().Throw<InvalidOperationException>().WithMessage("boom");
        study.Trials.Should().NotContain(t => t.State == TrialState.Running, "Optimize returns only once its trials have finished");
    }

    /// <summary>
    ///     A failure among the last trials, after the final submission, is not raised: Python never calls
    ///     <c>result()</c> on them (observed on Optuna 5.0: returns normally, three FAIL trials). It used to surface
    ///     as an <see cref="AggregateException" />.
    /// </summary>
    /// <exception cref="InvalidOperationException">Thrown by every objective; Optimize records each trial as FAIL and does not rethrow.</exception>
    [TestMethod]
    public void NJobs_ErrorInTheFinalBatch_IsNotRaised_AsInPython()
    {
        var study = Optuna.CreateStudy(sampler: new RandomSampler(6));

        study.Optimize((Func<Trial, double>)(_ =>
        {
            Thread.Sleep(50);
            throw new InvalidOperationException("late");
        }), nTrials: 3, nJobs: 4);

        study.Trials.Should().HaveCount(3).And.OnlyContain(t => t.State == TrialState.Fail);
    }

    /// <summary>
    ///     n_jobs workers run on dedicated threads, as Python's <c>ThreadPoolExecutor</c> does, never on the .NET
    ///     thread pool. A worker blocks for most of a trial (storage lock, I/O, objective), and blocking more pool
    ///     threads than the pool's minimum starved StackExchange.Redis, which completes even synchronous calls on the
    ///     pool: <c>Optimize(nJobs: 96)</c> on a Redis journal timed out after 88 trials.
    /// </summary>
    [TestMethod]
    public void NJobs_WorkersRunOnDedicatedThreads_NotOnTheThreadPool()
    {
        var study = Optuna.CreateStudy(sampler: new RandomSampler(7));
        var onPool = new System.Collections.Concurrent.ConcurrentBag<bool>();

        study.Optimize(t =>
        {
            onPool.Add(Thread.CurrentThread.IsThreadPoolThread);
            return t.SuggestFloat("x", 0, 1);
        }, nTrials: 32, nJobs: 4);

        onPool.Should().HaveCount(32).And.OnlyContain(p => !p, "a worker must not occupy a thread-pool thread");
    }

    /// <summary>
    ///     When several workers fail in one completion batch, Optimize raises the first failure, as Python's
    ///     <c>for f in completed: f.result()</c> does, and observes the others, so none resurfaces later through
    ///     <see cref="TaskScheduler.UnobservedTaskException" /> when the garbage collector finalizes their tasks. Every
    ///     objective here fails at once, so the first batch holds several failed workers.
    /// </summary>
    [TestMethod]
    public void NJobs_SeveralFailuresInOneBatch_LeaveNoUnobservedTaskException()
    {
        int unobserved = 0;
        void Handler(object? sender, UnobservedTaskExceptionEventArgs e)
        {
            // Only this test's failures count: another test's leaked task may be finalized during the same collection.
            if (e.Exception.Flatten().InnerExceptions.Any(x => x.Message == "batch failure"))
                Interlocked.Increment(ref unobserved);
        }

        TaskScheduler.UnobservedTaskException += Handler;
        try
        {
            RunAnAllFailingNJobsStudy();
            GC.Collect();
            GC.WaitForPendingFinalizers();
            GC.Collect();
            GC.WaitForPendingFinalizers();
        }
        finally
        {
            TaskScheduler.UnobservedTaskException -= Handler;
        }

        unobserved.Should().Be(0, "every failed worker's exception is observed before Optimize returns");
    }

    /// <summary>
    ///     Runs a study whose every trial fails, at <c>nJobs: 4</c>, and checks that Optimize raised the failure. Kept in
    ///     its own non-inlined method so its tasks are unreachable by the time the caller collects garbage.
    /// </summary>
    /// <exception cref="InvalidOperationException">Thrown by every objective; Optimize rethrows the first.</exception>
    [MethodImpl(MethodImplOptions.NoInlining)]
    private static void RunAnAllFailingNJobsStudy()
    {
        var study = Optuna.CreateStudy(sampler: new RandomSampler(12));
        var act = () => study.Optimize(
            (Func<Trial, double>)(_ => throw new InvalidOperationException("batch failure")), nTrials: 40, nJobs: 4);
        act.Should().Throw<InvalidOperationException>().WithMessage("batch failure");
    }

    /// <summary>
    ///     A trial's own writes never show up in a snapshot someone else took before them. The trial used to cache the
    ///     storage's live trial object and mutate it after its first write, so a reader holding that object from
    ///     <c>GetAllTrials(deepcopy: false)</c>, as TPE's constant liar does, saw its dictionaries change underneath it:
    ///     a data race when the reader is another thread. The trial now caches a detached copy.
    /// </summary>
    [TestMethod]
    public void Trial_Writes_DoNotChangeASnapshotTakenBeforeThem()
    {
        var study = Optuna.CreateStudy(sampler: new RandomSampler(13));
        var trial = study.Ask();
        FrozenTrial snapshot = study.Storage.GetAllTrials(study.StudyId, deepcopy: false).Single();

        trial.SuggestFloat("x", 0, 1);
        trial.Report(0.5, 0);
        trial.SetUserAttr("a", 1);

        snapshot.Params.Should().BeEmpty("the snapshot was taken before the suggestion");
        snapshot.Distributions.Should().BeEmpty();
        snapshot.IntermediateValues.Should().BeEmpty();
        snapshot.UserAttrs.Should().BeEmpty();
        var stored = study.Storage.GetTrial(trial.TrialId);
        stored.Params.Should().ContainKey("x", "the storage itself holds every write");
        stored.UserAttrs.Should().ContainKey("a");
    }

    /// <summary>
    ///     Eight threads setting different user attributes of one trial over the fast in-memory storage: every attribute
    ///     is stored, in the storage and in the trial's own view. A guard: the trial's cache dictionary is not
    ///     thread-safe and is now written under the trial's monitor. Before that, the storage's own lock happened to
    ///     space the unsynchronized cache writes apart, so this case did not fail on the old code (unlike the constraint
    ///     and report races, whose check-then-write spans two storage calls).
    /// </summary>
    [TestMethod]
    public void OneTrial_ConcurrentUserAttrs_AreAllStored()
    {
        var study = Optuna.CreateStudy(sampler: new RandomSampler(14));
        var trial = study.Ask();
        var errors = new System.Collections.Concurrent.ConcurrentQueue<Exception>();

        Parallel.For(0, 8, new ParallelOptions { MaxDegreeOfParallelism = 8 }, w =>
        {
            try
            {
                for (int i = 0; i < 250; i++)
                    trial.SetUserAttr($"w{w}-{i}", i);
            }
            catch (Exception e)
            {
                errors.Enqueue(e);
            }
        });

        errors.Should().BeEmpty();
        trial.UserAttrs.Should().HaveCount(2000, "the trial's own view keeps every attribute");
        study.Storage.GetTrial(trial.TrialId).UserAttrs.Should().HaveCount(2000);
    }

    /// <summary>
    ///     Python passes <c>n_jobs</c> to <c>ThreadPoolExecutor(max_workers=...)</c>, so 0 and every value below -1
    ///     raise its <c>ValueError</c>, verbatim. They used to reach <c>Task.WhenAny</c> on an empty set.
    /// </summary>
    [TestMethod]
    public void NJobs_ZeroOrBelowMinusOne_RaisesPythonsExecutorError()
    {
        var study = Optuna.CreateStudy();
        foreach (int nJobs in new[] { 0, -2 })
        {
            var act = () => study.Optimize(_ => 0.0, nTrials: 1, nJobs: nJobs);
            act.Should().Throw<ValueError>().WithMessage("max_workers must be greater than 0");
        }
        study.Trials.Should().BeEmpty();
    }
}

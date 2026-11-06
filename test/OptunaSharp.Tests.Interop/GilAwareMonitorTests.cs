using System;
using System.Linq;
using System.Threading;
using OptunaSharp;
using OptunaSharp.Distributions;
using OptunaSharp.Internal;
using OptunaSharp.Interop.PythonNet;
using OptunaSharp.Pruners;
using OptunaSharp.Samplers;
using Python.Runtime;

namespace OptunaSharp.Tests.Interop;

/// <summary>
///     Pins review item P3-1: code that holds the Python GIL while it calls a <see cref="Trial" /> or <see cref="Study" />
///     method no longer deadlocks against a sampler or pruner that needs the GIL under its OptunaSharp monitor.
/// </summary>
/// <remarks>
///     <para>Before the fix, an objective inside <c>using (Py.GIL())</c> blocked on the sampler, pruner or tell monitor
///     with the GIL held, while the monitor's holder (a bridged sampler or pruner) waited for that GIL: both threads
///     waited forever. A scratch probe of exactly the <c>Optimize</c> shapes below hung after 4 of 60 trials without the
///     hook and finished all 60 in about 3 s with it. <c>OptunaSharp.Interop.pythonnet</c> now installs
///     <c>GilAwareMonitor</c> into Core's <see cref="MonitorGate" />, which releases the GIL while a thread waits.</para>
///     <para>The three "parked" tests build the deadlock deterministically: thread B holds the monitor and needs the
///     GIL, thread A holds the GIL and needs the monitor. With <c>MonitorGate.Hook</c> removed they deadlock by
///     construction (a scratch probe of each hung). A regression there is recovered (thread A is interrupted out of its managed
///     wait, which drops its GIL), so it fails this test instead of hanging the run. The <c>Optimize</c> and
///     ask-and-tell tests reproduce the real-world shapes at <c>nJobs = 4</c>. Where the objective holds the GIL a
///     regression cannot be recovered and fails through the watchdog; an <c>Optimize</c> called while holding the GIL is
///     recovered like the parked tests. All of them self-skip (Inconclusive) without a usable Python engine.</para>
/// </remarks>
[TestClass]
public class GilAwareMonitorTests
{
    /// <summary>How long a fixed run may take before it counts as deadlocked; fixed runs take a few seconds.</summary>
    private static readonly TimeSpan Watchdog = TimeSpan.FromSeconds(120);

    /// <summary>
    ///     <see cref="PythonRuntime.Initialize" /> installs the hook even when the host started CPython first, as this
    ///     suite's <c>PythonSession</c> does. Without the hook every other test here would deadlock instead of pass.
    /// </summary>
    [TestMethod]
    [TestCategory("RequiresPython")]
    public void Initialize_InstallsTheMonitorGateHook()
    {
        RequireEngineWithHook();

        Assert.IsNotNull(MonitorGate.Hook, "PythonRuntime.Initialize must install GilAwareMonitor into MonitorGate");
    }

    /// <summary>
    ///     Thread B holds the sampler monitor inside <c>SampleIndependent</c> and needs the GIL; thread A holds the GIL
    ///     and calls <see cref="Study.Ask" />, whose <c>BeforeTrial</c> needs the monitor. Thread A must release the GIL
    ///     while it waits, so both trials complete.
    /// </summary>
    [TestMethod]
    [TestCategory("RequiresPython")]
    public void Ask_WhileHoldingTheGil_SamplerMonitorHeldByAThreadThatNeedsTheGil_Completes()
    {
        RequireEngineWithHook();

        var sampler = new ParkingGilSampler(seed: 3);
        var study = Optuna.CreateStudy(sampler: sampler);

        RunParked(
            sampler.Parked,
            holderB: () =>
            {
                Trial trial = study.Ask();
                double x = trial.SuggestFloat("x", 0, 1);   // parks in SampleIndependent, under the sampler monitor
                study.Tell(trial, x);
            },
            gilHolderA: () =>
            {
                sampler.Release.Set();                       // B now wants the GIL this thread holds
                Trial trial = study.Ask();                   // BeforeTrial needs the monitor B holds
                double x = trial.SuggestFloat("x", 0, 1);
                study.Tell(trial, x);
            },
            what: "Ask while holding the GIL against a sampler parked in SampleIndependent");

        Assert.AreEqual(2, study.Trials.Count);
        Assert.IsTrue(study.Trials.All(t => t.State == TrialState.Complete), "both trials must complete");
    }

    /// <summary>
    ///     Two threads drive one <see cref="Trial" />. Thread B holds the trial's own monitor (and the sampler's) inside
    ///     <see cref="Trial.SuggestFloat" /> and needs the GIL; thread A holds the GIL and writes a user attribute to the
    ///     same trial, which needs the trial monitor. The trial monitor also comes before the GIL, so it is entered through
    ///     <c>MonitorGate</c> as well, and thread A must release the GIL while it waits.
    /// </summary>
    [TestMethod]
    [TestCategory("RequiresPython")]
    public void TrialWrite_WhileHoldingTheGil_TrialMonitorHeldByAThreadThatNeedsTheGil_Completes()
    {
        RequireEngineWithHook();

        var sampler = new ParkingGilSampler(seed: 3);
        var study = Optuna.CreateStudy(sampler: sampler);
        Trial shared = study.Ask();
        double suggested = double.NaN;

        RunParked(
            sampler.Parked,
            holderB: () => suggested = shared.SuggestFloat("x", 0, 1),   // parks under the trial and sampler monitors
            gilHolderA: () =>
            {
                sampler.Release.Set();                       // B now wants the GIL this thread holds
                shared.SetUserAttr("seen", true);            // needs the trial monitor B holds
            },
            what: "SetUserAttr while holding the GIL against a trial parked in SuggestFloat");

        FrozenTrial told = study.Tell(shared, suggested);
        Assert.AreEqual(TrialState.Complete, told.State);
        Assert.AreEqual(true, told.UserAttrs["seen"]);
        Assert.AreEqual(suggested, Convert.ToDouble(told.Params["x"]));
    }

    /// <summary>
    ///     Thread B holds the pruner monitor inside <c>Prune</c> and needs the GIL; thread A holds the GIL and calls
    ///     <see cref="Trial.ShouldPrune" />, which needs the same monitor. Thread A must release the GIL while it waits.
    /// </summary>
    /// <remarks>
    ///     Thread A's trial is asked and reported before B parks: <c>Ask</c> enters the pruner monitor as well (through
    ///     <c>FilterStudy</c>, Optuna's <c>_filter_study</c> in <c>Trial.__init__</c>), and a parked holder would make it
    ///     wait for a release that only comes after the ask. That way <c>ShouldPrune</c> is A's only contended call.
    /// </remarks>
    [TestMethod]
    [TestCategory("RequiresPython")]
    public void ShouldPrune_WhileHoldingTheGil_PrunerMonitorHeldByAThreadThatNeedsTheGil_Completes()
    {
        RequireEngineWithHook();

        var pruner = new ParkingGilPruner();
        var study = Optuna.CreateStudy(sampler: new RandomSampler(seed: 3), pruner: pruner);
        Trial trialA = study.Ask();
        trialA.Report(2.0, 0);

        RunParked(
            pruner.Parked,
            holderB: () =>
            {
                Trial trial = study.Ask();
                trial.Report(1.0, 0);
                trial.ShouldPrune();                         // parks in Prune, under the pruner monitor
                study.Tell(trial, 1.0);
            },
            gilHolderA: () =>
            {
                pruner.Release.Set();                        // B now wants the GIL this thread holds
                trialA.ShouldPrune();                        // needs the pruner monitor B holds
                study.Tell(trialA, 2.0);
            },
            what: "ShouldPrune while holding the GIL against a pruner parked in Prune");

        Assert.AreEqual(2, study.Trials.Count);
        Assert.IsTrue(study.Trials.All(t => t.State == TrialState.Complete), "both trials must complete");
    }

    /// <summary>
    ///     The probe's sampler shape: four <c>nJobs</c> workers run an objective that holds the GIL around Python work and
    ///     <see cref="Trial.SuggestFloat" />, over a sampler whose study-driven calls take the GIL, as every Python bridge
    ///     does. All 60 trials must complete.
    /// </summary>
    [TestMethod]
    [TestCategory("RequiresPython")]
    public void Optimize_ObjectiveHoldsTheGil_GilTakingSampler_NJobs4_Completes()
    {
        RequireEngineWithHook();

        var study = Optuna.CreateStudy(sampler: new GilTakingSampler(seed: 1));

        RunWithinWatchdog(() => study.Optimize(t =>
        {
            using (Py.GIL())
            {
                RunPython(100_000);
                return t.SuggestFloat("x", 0, 1);
            }
        }, nTrials: 60, nJobs: 4), "Optimize(nJobs: 4) with a GIL-holding objective over a GIL-taking sampler");

        Assert.AreEqual(60, study.Trials.Count);
        Assert.IsTrue(study.Trials.All(t => t.State == TrialState.Complete), "every trial must complete");
    }

    /// <summary>
    ///     The probe's pruner shape: four <c>nJobs</c> workers report and call <see cref="Trial.ShouldPrune" /> while
    ///     holding the GIL, over a pruner whose <c>Prune</c> takes the GIL, as the OptunaHub pruner bridge does.
    /// </summary>
    [TestMethod]
    [TestCategory("RequiresPython")]
    public void Optimize_ObjectiveHoldsTheGil_GilTakingPruner_NJobs4_Completes()
    {
        RequireEngineWithHook();

        var study = Optuna.CreateStudy(sampler: new RandomSampler(seed: 1), pruner: new GilTakingPruner());

        RunWithinWatchdog(() => study.Optimize(t =>
        {
            using (Py.GIL())
            {
                RunPython(100_000);
                double x = t.SuggestFloat("x", 0, 1);
                t.Report(x, 0);
                t.ShouldPrune();
                return x;
            }
        }, nTrials: 60, nJobs: 4), "Optimize(nJobs: 4) with a GIL-holding objective over a GIL-taking pruner");

        Assert.AreEqual(60, study.Trials.Count);
        Assert.IsTrue(study.Trials.All(t => t.State == TrialState.Complete), "every trial must complete");
    }

    /// <summary>
    ///     <c>Optimize(nJobs: 4)</c> called while the caller holds the GIL, over a sampler that takes the GIL. The caller
    ///     waits for its workers, and that wait is not a monitor: it must release the GIL, as Python's
    ///     <c>wait(FIRST_COMPLETED)</c> does, or no worker can finish. Without the hooks a scratch probe hung after 4 trials;
    ///     with them all 40 finished in under a second.
    /// </summary>
    [TestMethod]
    [TestCategory("RequiresPython")]
    public void Optimize_CalledWhileHoldingTheGil_NJobs4_GilTakingSampler_Completes()
    {
        RequireEngineWithHook();

        var study = Optuna.CreateStudy(sampler: new GilTakingSampler(seed: 4));

        RunRecoverably(() =>
        {
            using (Py.GIL())
                study.Optimize(t => t.SuggestFloat("x", 0, 1), nTrials: 40, nJobs: 4);
        }, "Optimize(nJobs: 4) called while holding the GIL over a GIL-taking sampler");

        Assert.AreEqual(40, study.Trials.Count);
        Assert.IsTrue(study.Trials.All(t => t.State == TrialState.Complete), "every trial must complete");
    }

    /// <summary>
    ///     Four threads run the ask-and-tell loop while holding the GIL, over a sampler whose <c>BeforeTrial</c>,
    ///     <c>SampleIndependent</c> and <c>AfterTrial</c> take the GIL. This covers <c>Ask</c>, the suggest call and
    ///     <c>Tell</c> (the tell monitor, then <c>AfterTrial</c> under the sampler monitor).
    /// </summary>
    [TestMethod]
    [TestCategory("RequiresPython")]
    public void AskTell_FourThreadsHoldingTheGil_GilTakingSampler_Completes()
    {
        RequireEngineWithHook();

        var study = Optuna.CreateStudy(sampler: new GilTakingSampler(seed: 2));

        RunWithinWatchdog(() =>
        {
            var threads = Enumerable.Range(0, 4).Select(_ => new Thread(() =>
            {
                for (int i = 0; i < 15; i++)
                {
                    using (Py.GIL())
                    {
                        Trial trial = study.Ask();
                        double x = trial.SuggestFloat("x", 0, 1);
                        RunPython(50_000);
                        study.Tell(trial, x);
                    }
                }
            }) { IsBackground = true }).ToList();

            foreach (Thread thread in threads)
                thread.Start();
            foreach (Thread thread in threads)
                thread.Join();
        }, "four ask-and-tell threads holding the GIL over a GIL-taking sampler");

        Assert.AreEqual(60, study.Trials.Count);
        Assert.IsTrue(study.Trials.All(t => t.State == TrialState.Complete), "every trial must complete");
    }

    // ---- helpers --------------------------------------------------------------------------------

    /// <summary>
    ///     Skips the test without a usable Python engine, then makes sure the bridge's runtime (and so the
    ///     <see cref="MonitorGate" /> hook) is initialized; this suite's engine is started by <c>PythonSession</c>.
    /// </summary>
    /// <exception cref="AssertInconclusiveException">No usable Python/Optuna engine on this machine.</exception>
    private static void RequireEngineWithHook()
    {
        PythonSession.EnsureOrInconclusive();
        PythonRuntime.Initialize();
    }

    /// <summary>Runs a CPython loop of <paramref name="iterations" /> steps, long enough to cross the switch interval.</summary>
    /// <param name="iterations">The loop length; 100 000 takes a few milliseconds.</param>
    /// <remarks>Takes the GIL itself (re-entrantly when the caller already holds it), as a bridge call does.</remarks>
    private static void RunPython(int iterations)
    {
        using (Py.GIL())
            PythonEngine.Exec($"s = 0\nfor i in range({iterations}): s += i");
    }

    /// <summary>
    ///     Runs the deterministic two-thread deadlock: <paramref name="holderB" /> on thread B until it parks inside a
    ///     monitor (<paramref name="parked" /> set), then <paramref name="gilHolderA" /> on thread A inside
    ///     <c>Py.GIL()</c>. Fails the test if either thread throws or the pair is still blocked after the watchdog.
    /// </summary>
    /// <param name="parked">Set by the parking sampler or pruner once thread B holds its monitor.</param>
    /// <param name="holderB">Thread B's work; it parks inside the monitor and then needs the GIL.</param>
    /// <param name="gilHolderA">Thread A's work, run while holding the GIL; it releases B and then needs the monitor.</param>
    /// <param name="what">The scenario, for the failure message.</param>
    /// <exception cref="AssertFailedException">A thread threw, or the threads deadlocked.</exception>
    /// <remarks>
    ///     On a deadlock thread A is blocked in a managed monitor wait with the GIL held. <see cref="Thread.Interrupt" />
    ///     throws it out of that wait, the <c>using (Py.GIL())</c> scope then drops the GIL, and thread B can finish, so a
    ///     regression fails this test without hanging the rest of the run.
    /// </remarks>
    private static void RunParked(ManualResetEventSlim parked, Action holderB, Action gilHolderA, string what)
    {
        Exception? failureA = null, failureB = null;

        var threadB = new Thread(() =>
        {
            try
            {
                holderB();
            }
            catch (Exception ex)
            {
                failureB = ex;
            }
        }) { IsBackground = true, Name = "P3-1 monitor holder" };

        var threadA = new Thread(() =>
        {
            try
            {
                if (!parked.Wait(Watchdog))
                    throw new TimeoutException("thread B never parked inside the monitor");
                using (Py.GIL())
                    gilHolderA();
            }
            catch (Exception ex)
            {
                failureA = ex;
            }
        }) { IsBackground = true, Name = "P3-1 GIL holder" };

        threadB.Start();
        threadA.Start();

        if (!threadA.Join(Watchdog))
        {
            // Recover from the deadlock so the rest of the run can still use Python, then fail.
            threadA.Interrupt();
            threadA.Join(TimeSpan.FromSeconds(30));
            threadB.Join(TimeSpan.FromSeconds(30));
            Assert.Fail($"{what}: deadlocked (the GIL holder blocked on the monitor with the GIL held)");
        }

        Assert.IsTrue(threadB.Join(Watchdog), $"{what}: the monitor holder never finished");
        Assert.IsNull(failureA, $"{what}: the GIL holder threw {failureA}");
        Assert.IsNull(failureB, $"{what}: the monitor holder threw {failureB}");
    }

    /// <summary>
    ///     Runs <paramref name="callerHoldingTheGil" /> on a dedicated thread and fails the test if it is not done within
    ///     <see cref="Watchdog" />, recovering the deadlock first so the rest of the run can still use Python.
    /// </summary>
    /// <param name="callerHoldingTheGil">The run; it takes the GIL itself and blocks in managed waits only.</param>
    /// <param name="what">The scenario, for the failure message.</param>
    /// <exception cref="AssertFailedException">The run threw, or deadlocked.</exception>
    /// <remarks>
    ///     A deadlocked <c>Optimize</c> caller sits in a managed wait with the GIL held: first the wait for a free worker,
    ///     then, once interrupted out of it, the final wait for every worker in its <c>finally</c>. Each
    ///     <see cref="Thread.Interrupt" /> breaks one of them, so the watchdog interrupts until the thread is out and its
    ///     <c>using (Py.GIL())</c> has dropped the GIL.
    /// </remarks>
    private static void RunRecoverably(Action callerHoldingTheGil, string what)
    {
        Exception? failure = null;
        var caller = new Thread(() =>
        {
            try
            {
                callerHoldingTheGil();
            }
            catch (Exception ex)
            {
                failure = ex;
            }
        }) { IsBackground = true, Name = "P3-1 GIL-holding caller" };

        caller.Start();
        if (!caller.Join(Watchdog))
        {
            for (int interrupts = 0; interrupts < 10 && !caller.Join(TimeSpan.FromSeconds(2)); interrupts++)
                caller.Interrupt();
            Assert.Fail($"{what}: deadlocked (the caller waited for its workers with the GIL held)");
        }

        Assert.IsNull(failure, $"{what}: threw {failure}");
    }

    /// <summary>
    ///     Runs <paramref name="action" /> on a background thread and fails the test if it is not done within
    ///     <see cref="Watchdog" />, the idiom of the hub bridge's own nJobs deadlock test.
    /// </summary>
    /// <param name="action">The run to watch.</param>
    /// <param name="what">The scenario, for the failure message.</param>
    /// <exception cref="AssertFailedException">The run threw, or did not finish in time.</exception>
    /// <remarks>
    ///     A deadlock here leaves a worker blocked with the GIL held, which no one can release from outside, so later
    ///     Python tests in the same process may hang too. The parked tests above catch the same regression recoverably.
    /// </remarks>
    private static void RunWithinWatchdog(Action action, string what)
    {
        Exception? failure = null;
        var worker = new Thread(() =>
        {
            try
            {
                action();
            }
            catch (Exception ex)
            {
                failure = ex;
            }
        }) { IsBackground = true, Name = "P3-1 watched run" };

        worker.Start();
        Assert.IsTrue(worker.Join(Watchdog), $"{what}: deadlocked");
        Assert.IsNull(failure, $"{what}: threw {failure}");
    }

    /// <summary>A <see cref="RandomSampler" /> whose study-driven calls take the GIL and run Python, as a bridge's do.</summary>
    /// <param name="seed">The random sampler's seed.</param>
    private sealed class GilTakingSampler(int seed) : RandomSampler(seed)
    {
        /// <summary>Takes the GIL and runs Python inside the sampler monitor, then defers to <see cref="RandomSampler" />.</summary>
        /// <param name="study">The study being optimized.</param>
        /// <param name="trial">The trial being sampled.</param>
        /// <param name="paramName">The parameter's name.</param>
        /// <param name="distribution">The parameter's distribution.</param>
        /// <returns>The random sampler's value.</returns>
        public override object? SampleIndependent(Study study, FrozenTrial trial, string paramName, BaseDistribution distribution)
        {
            RunPython(300_000);
            return base.SampleIndependent(study, trial, paramName, distribution);
        }

        /// <summary>Takes the GIL and runs Python inside the sampler monitor, as <c>Ask</c> drives it.</summary>
        /// <param name="study">The study being optimized.</param>
        /// <param name="trial">The trial that starts.</param>
        public override void BeforeTrial(Study study, FrozenTrial trial) => RunPython(50_000);

        /// <summary>Takes the GIL and runs Python inside the sampler monitor, as <c>Tell</c> drives it.</summary>
        /// <param name="study">The study being optimized.</param>
        /// <param name="trial">The trial that finished.</param>
        /// <param name="state">Its final state.</param>
        /// <param name="values">Its values, or null.</param>
        public override void AfterTrial(Study study, FrozenTrial trial, TrialState state, System.Collections.Generic.IReadOnlyList<double>? values)
            => RunPython(50_000);

        /// <summary>Takes the GIL and runs Python inside the sampler monitor, as each <c>nJobs</c> worker's start does.</summary>
        public override void ReseedRng()
        {
            RunPython(100_000);
            base.ReseedRng();
        }
    }

    /// <summary>A pruner whose <c>Prune</c> takes the GIL and runs Python, as the OptunaHub pruner bridge does; never prunes.</summary>
    private sealed class GilTakingPruner : BasePruner
    {
        /// <summary>Takes the GIL and runs Python inside the pruner monitor.</summary>
        /// <param name="study">The study being optimized.</param>
        /// <param name="trial">The trial being judged.</param>
        /// <returns>Always false.</returns>
        public override bool Prune(Study study, FrozenTrial trial)
        {
            RunPython(300_000);
            return false;
        }
    }

    /// <summary>
    ///     A <see cref="RandomSampler" /> whose first <c>SampleIndependent</c> parks inside the sampler monitor until
    ///     <see cref="Release" /> is set, then takes the GIL: thread B of the deterministic deadlock.
    /// </summary>
    /// <param name="seed">The random sampler's seed.</param>
    private sealed class ParkingGilSampler(int seed) : RandomSampler(seed)
    {
        private int _parkOnce = 1;

        /// <summary>Set once the first <c>SampleIndependent</c> holds the sampler monitor.</summary>
        public ManualResetEventSlim Parked { get; } = new();

        /// <summary>Set by thread A, while it holds the GIL, to let the parked call go on and ask for the GIL.</summary>
        public ManualResetEventSlim Release { get; } = new();

        /// <summary>Parks the first call (bounded by the watchdog), then takes the GIL and runs Python.</summary>
        /// <param name="study">The study being optimized.</param>
        /// <param name="trial">The trial being sampled.</param>
        /// <param name="paramName">The parameter's name.</param>
        /// <param name="distribution">The parameter's distribution.</param>
        /// <returns>The random sampler's value.</returns>
        public override object? SampleIndependent(Study study, FrozenTrial trial, string paramName, BaseDistribution distribution)
        {
            if (Interlocked.Exchange(ref _parkOnce, 0) == 1)
            {
                Parked.Set();
                Release.Wait(Watchdog);
            }

            RunPython(100_000);
            return base.SampleIndependent(study, trial, paramName, distribution);
        }
    }

    /// <summary>
    ///     A pruner whose first <c>Prune</c> parks inside the pruner monitor until <see cref="Release" /> is set, then
    ///     takes the GIL: thread B of the deterministic deadlock. It never prunes.
    /// </summary>
    private sealed class ParkingGilPruner : BasePruner
    {
        private int _parkOnce = 1;

        /// <summary>Set once the first <c>Prune</c> holds the pruner monitor.</summary>
        public ManualResetEventSlim Parked { get; } = new();

        /// <summary>Set by thread A, while it holds the GIL, to let the parked call go on and ask for the GIL.</summary>
        public ManualResetEventSlim Release { get; } = new();

        /// <summary>Parks the first call (bounded by the watchdog), then takes the GIL and runs Python.</summary>
        /// <param name="study">The study being optimized.</param>
        /// <param name="trial">The trial being judged.</param>
        /// <returns>Always false.</returns>
        public override bool Prune(Study study, FrozenTrial trial)
        {
            if (Interlocked.Exchange(ref _parkOnce, 0) == 1)
            {
                Parked.Set();
                Release.Wait(Watchdog);
            }

            RunPython(100_000);
            return false;
        }
    }
}

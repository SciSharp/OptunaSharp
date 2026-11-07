using System;
using System.Collections.Concurrent;
using System.Globalization;
using System.Linq;
using System.Runtime.InteropServices;
using System.Threading;
using AwesomeAssertions;
using OptunaSharp.Distributions;
using OptunaSharp.Interop.Pymoo;
using OptunaSharp.Interop.PythonNet;
using OptunaSharp.Samplers;
using Python.Runtime;

namespace OptunaSharp.Tests.Interop;

/// <summary>
///     The Python thread of <see cref="PythonRuntime" /> (plan 10, item 2.5; review item 19b): every
///     <see cref="PythonRuntime.Run(Action)" /> body runs on one long-lived thread that keeps one CPython thread state for
///     the life of the process.
/// </summary>
/// <remarks>
///     <para>
///         The bug it fixes: on a .NET thread, pythonnet's <c>Py.GIL()</c> created a CPython thread state when the
///         outermost scope opened and freed it at the end, and pybind11 libraries kept using the freed one. Measured on
///         2026-11-07: <c>PymooSampler</c> off the main thread crashed with <c>0xC0000005</c> in scipy's <c>cdist</c> in 7
///         of 8 runs, and torch hung. Several cases here fail deterministically on the old runtime: a Python
///         <c>threading.local</c> lost its value between calls, and calls from different .NET threads ran on different
///         Python threads.
///     </para>
///     <para>
///         The engine-level checks need only Python; the pymoo and torch cases self-skip (Inconclusive) when the package
///         is missing.
///     </para>
/// </remarks>
[TestClass]
public class PythonThreadTests
{
    /// <summary>How long a case may wait before it counts as a deadlock.</summary>
    private static readonly TimeSpan Watchdog = TimeSpan.FromSeconds(120);

    /// <summary>
    ///     Runs <paramref name="work" /> on a new thread and fails the test when it throws or has not finished within the
    ///     watchdog, which is how a deadlock or a hang shows here.
    /// </summary>
    /// <param name="work">The work, usually one or more <see cref="PythonRuntime.Run(Action)" /> calls.</param>
    /// <param name="what">The scenario, for the failure message.</param>
    /// <exception cref="AssertFailedException">The work threw, or did not finish within the watchdog.</exception>
    private static void OnNewThread(Action work, string what)
    {
        Exception? error = null;
        var thread = new Thread(() =>
        {
            try
            {
                work();
            }
            catch (Exception e)
            {
                error = e;
            }
        })
        {
            IsBackground = true,
        };
        thread.Start();
        if (!thread.Join(Watchdog))
            Assert.Fail($"{what}: still blocked after {Watchdog.TotalSeconds} s (deadlock or hang).");
        if (error is not null)
            Assert.Fail($"{what} threw: {error}");
    }

    /// <summary>The ident of the Python thread that runs this call (<c>threading.get_ident()</c>).</summary>
    /// <returns>The ident. Call under the GIL.</returns>
    private static long PythonThreadIdent() => PythonEngine.Eval("__import__('threading').get_ident()").As<long>();

    /// <summary>
    ///     Calls from many .NET threads all run on one Python thread, and a Python <c>threading.local</c> keeps its value
    ///     between them: one thread state that is never freed. On the old runtime each .NET thread got a Python thread of
    ///     its own, and every call a new thread state, so the local's value was gone at the next call.
    /// </summary>
    [TestMethod]
    [TestCategory("RequiresPython")]
    public void Run_FromManyDotNetThreads_RunsOnOnePythonThread_WhoseStateSurvives()
    {
        PythonSession.EnsureOrInconclusive();
        long first = PythonRuntime.Run(() =>
        {
            PythonEngine.Exec(
                "import builtins, threading\n" +
                "builtins._optunasharp_thread_local = threading.local()\n" +
                "builtins._optunasharp_thread_local.value = 42\n");
            return PythonThreadIdent();
        });
        try
        {
            var seen = new ConcurrentBag<(long Ident, int? Value)>();
            for (int i = 0; i < 6; i++)
            {
                OnNewThread(() => seen.Add(PythonRuntime.Run(() =>
                {
                    using PyObject value = PythonEngine.Eval(
                        "__import__('builtins')._optunasharp_thread_local.__dict__.get('value')");
                    return (PythonThreadIdent(), value.IsNone() ? (int?)null : value.As<int>());
                })), "a call from another .NET thread");
            }

            seen.Should().HaveCount(6).And.OnlyContain(s => s.Ident == first && s.Value == 42);
        }
        finally
        {
            PythonRuntime.Run(() => PythonEngine.Exec("import builtins\ndel builtins._optunasharp_thread_local\n"));
        }
    }

    /// <summary>A body that calls <see cref="PythonRuntime.Run(Action)" /> again runs inline, on the same Python thread.</summary>
    [TestMethod]
    [TestCategory("RequiresPython")]
    public void NestedRun_RunsInlineOnThePythonThread()
    {
        PythonSession.EnsureOrInconclusive();
        (long outer, long inner) = (0, 0);

        OnNewThread(() => (outer, inner) = PythonRuntime.Run(() => (PythonThreadIdent(), PythonRuntime.Run(PythonThreadIdent))),
            "a nested Run");

        inner.Should().Be(outer);
    }

    /// <summary>
    ///     A caller that holds the GIL (user code inside <c>Py.GIL()</c>) releases it while it waits for the Python thread;
    ///     otherwise the Python thread could never take the GIL and the first call would deadlock.
    /// </summary>
    [TestMethod]
    [TestCategory("RequiresPython")]
    public void Run_FromACallerHoldingTheGil_Completes()
    {
        PythonSession.EnsureOrInconclusive();
        int result = 0;

        OnNewThread(() =>
        {
            using (Py.GIL())
                result = PythonRuntime.Run(() => PythonEngine.Eval("2 + 2").As<int>());
        }, "Run inside the caller's own Py.GIL()");

        result.Should().Be(4);
    }

    /// <summary>The caller's culture and <c>AsyncLocal</c> values flow into the body, as they did when it ran on the caller.</summary>
    [TestMethod]
    [TestCategory("RequiresPython")]
    public void Run_FlowsTheCallersCultureAndAsyncLocals()
    {
        PythonSession.EnsureOrInconclusive();
        var local = new AsyncLocal<string>();
        (string culture, string? value) seen = (string.Empty, null);

        OnNewThread(() =>
        {
            CultureInfo.CurrentCulture = CultureInfo.GetCultureInfo("de-DE");
            local.Value = "from the caller";
            seen = PythonRuntime.Run(() => (CultureInfo.CurrentCulture.Name, local.Value));
        }, "Run with a culture and an AsyncLocal");

        seen.Should().Be(("de-DE", "from the caller"));
    }

    /// <summary>A body's exception reaches the caller as the very object thrown, and a Python error as a <see cref="PythonException" />.</summary>
    [TestMethod]
    [TestCategory("RequiresPython")]
    public void Run_HandsTheBodysExceptionToTheCaller()
    {
        PythonSession.EnsureOrInconclusive();
        var thrown = new InvalidOperationException("from the body");

        var managed = () => PythonRuntime.Run(() => Throw(thrown));
        managed.Should().Throw<InvalidOperationException>().Which.Should().BeSameAs(thrown);

        var python = () => PythonRuntime.Run(() => PythonEngine.Exec("raise ValueError('boom from python')"));
        python.Should().Throw<PythonException>().WithMessage("*boom from python*");
    }

    /// <summary>Throws <paramref name="error" />, for a body that fails.</summary>
    /// <param name="error">The exception to throw.</param>
    /// <exception cref="InvalidOperationException">Always: <paramref name="error" />.</exception>
    private static void Throw(InvalidOperationException error) => throw error;

    /// <summary>
    ///     The Python thread reserves 16 MB of stack, CPython's own thread size on macOS, where numba's JIT compiling
    ///     pymoo's AGE-MOEA kernels overflowed .NET's default stacks. Read from Windows' stack limits; Inconclusive
    ///     elsewhere.
    /// </summary>
    [TestMethod]
    [TestCategory("RequiresPython")]
    public void PythonThread_Reserves16MBOfStack()
    {
        PythonSession.EnsureOrInconclusive();
        if (!OperatingSystem.IsWindows())
            Assert.Inconclusive("The stack limits are read with GetCurrentThreadStackLimits, a Windows API.");

        long reserved = PythonRuntime.Run(() =>
        {
            GetCurrentThreadStackLimits(out nuint low, out nuint high);
            return (long)(high - low);
        });

        reserved.Should().BeGreaterThanOrEqualTo(16L * 1024 * 1024);
    }

    /// <summary>Windows' bounds of the calling thread's reserved stack.</summary>
    /// <param name="lowLimit">The lowest address of the stack.</param>
    /// <param name="highLimit">The address just above the stack.</param>
    [DllImport("kernel32.dll")]
    private static extern void GetCurrentThreadStackLimits(out nuint lowLimit, out nuint highLimit);

    /// <summary>
    ///     Review item 19b: <c>PymooSampler</c> driven off the thread that built it, and at <c>nJobs: 4</c>, completes every
    ///     trial. Before the Python thread it crashed the process with <c>0xC0000005</c> in scipy's <c>cdist</c> (7 of 8
    ///     runs off the main thread), and <c>nJobs</c> above 1 was not supported.
    /// </summary>
    [TestMethod]
    [TestCategory("RequiresPython")]
    public void PymooSampler_OffItsThreadAndAtNJobs4_CompletesEveryTrial()
    {
        PythonSession.EnsureOrInconclusive();
        RequireModuleOrInconclusive("pymoo");
        var directions = new[] { StudyDirection.Minimize, StudyDirection.Minimize };
        double[] Objective(Trial t)
        {
            double x = t.SuggestFloat("x", 0, 1);
            double y = t.SuggestFloat("y", 0, 1);
            return new[] { x, (1 - x) + y };
        }

        var offThread = Optuna.CreateStudy(directions: directions, sampler: new PymooSampler(PymooAlgorithm.Nsga2, populationSize: 10, seed: 1));
        OnNewThread(() => offThread.Optimize(Objective, nTrials: 40), "PymooSampler on another thread");
        offThread.Trials.Should().HaveCount(40).And.OnlyContain(t => t.State == TrialState.Complete);

        var parallel = Optuna.CreateStudy(directions: directions, sampler: new PymooSampler(PymooAlgorithm.Nsga2, populationSize: 10, seed: 2));
        OnNewThread(() => parallel.Optimize(Objective, nTrials: 40, nJobs: 4), "PymooSampler at nJobs: 4");
        parallel.Trials.Should().HaveCount(40).And.OnlyContain(t => t.State == TrialState.Complete);
    }

    /// <summary>
    ///     torch, imported by a call from one .NET thread and used by calls from others, does not hang. Before the Python
    ///     thread it hung in 5 of 5 runs once the importing call's thread state had been freed.
    /// </summary>
    [TestMethod]
    [TestCategory("RequiresPython")]
    public void Torch_ImportedFromOneThread_UsedFromOthers_DoesNotHang()
    {
        PythonSession.EnsureOrInconclusive();
        RequireModuleOrInconclusive("torch");
        PyModule? scope = null;
        OnNewThread(() => PythonRuntime.Run(() =>
        {
            scope = Py.CreateScope();
            scope.Exec("import torch\n");
        }), "importing torch");
        try
        {
            var sums = new ConcurrentBag<double>();
            for (int i = 0; i < 4; i++)
            {
                OnNewThread(() => sums.Add(PythonRuntime.Run(() =>
                {
                    scope!.Exec(
                        "a = torch.eye(32, dtype=torch.float64) * 2\n" +
                        "r = float(torch.linalg.cholesky(a @ a.T).sum())\n");
                    return scope.Get("r").As<double>();
                })), "a torch call from another .NET thread");
            }

            sums.Should().HaveCount(4).And.OnlyContain(s => Math.Abs(s - 64.0) < 1e-9);
        }
        finally
        {
            PythonRuntime.Run(() => scope?.Dispose());
        }
    }

    /// <summary>
    ///     The P3-1 deadlock shape through the Python thread: thread B holds the sampler monitor and calls
    ///     <see cref="PythonRuntime.Run(Action)" />; thread A holds the GIL and asks, which needs that monitor. A releases
    ///     the GIL while it waits for the monitor, so the Python thread can take it for B's call, and both finish.
    /// </summary>
    [TestMethod]
    [TestCategory("RequiresPython")]
    public void Ask_WhileHoldingTheGil_AgainstASamplerInsideRun_Completes()
    {
        PythonSession.EnsureOrInconclusive();
        var sampler = new ParkingRunSampler(seed: 3);
        var study = Optuna.CreateStudy(sampler: sampler);
        Exception? errorB = null;
        var threadB = new Thread(() =>
        {
            try
            {
                Trial trial = study.Ask();
                double x = trial.SuggestFloat("x", 0, 1);   // parks in SampleIndependent, under the sampler monitor
                study.Tell(trial, x);
            }
            catch (Exception e)
            {
                errorB = e;
            }
        })
        {
            IsBackground = true,
        };
        threadB.Start();
        sampler.Parked.Wait(Watchdog).Should().BeTrue("thread B reaches the sampler");

        OnNewThread(() =>
        {
            using (Py.GIL())
            {
                sampler.Release.Set();                       // B now calls Run, which needs the GIL this thread holds
                Trial trial = study.Ask();                   // BeforeTrial needs the monitor B holds
                double x = trial.SuggestFloat("x", 0, 1);
                study.Tell(trial, x);
            }
        }, "Ask while holding the GIL against a sampler inside Run");

        threadB.Join(Watchdog).Should().BeTrue("thread B finishes once the GIL is free");
        errorB.Should().BeNull();
        study.Trials.Should().HaveCount(2).And.OnlyContain(t => t.State == TrialState.Complete);
    }

    /// <summary>Self-skips (Inconclusive) when <paramref name="module" /> cannot be imported in the interop interpreter.</summary>
    /// <param name="module">The Python module the case needs.</param>
    /// <exception cref="AssertInconclusiveException">The module is not installed.</exception>
    private static void RequireModuleOrInconclusive(string module)
    {
        try
        {
            PythonRuntime.Run(() => PythonRuntime.Import(module));
        }
        catch (PythonInteropException)
        {
            Assert.Inconclusive($"{module} is not installed in the interop interpreter (`pip install {module}`).");
        }
    }

    /// <summary>
    ///     A sampler that parks its first <c>SampleIndependent</c> under the sampler monitor until released, then runs
    ///     Python through <see cref="PythonRuntime.Run(Action)" />, as every bridge does.
    /// </summary>
    /// <param name="seed">The random sampler's seed.</param>
    private sealed class ParkingRunSampler(int seed) : RandomSampler(seed)
    {
        private int _parkOnce = 1;

        /// <summary>Set once the first <c>SampleIndependent</c> holds the sampler monitor.</summary>
        public ManualResetEventSlim Parked { get; } = new();

        /// <summary>Set by the thread that holds the GIL, to let the parked call go on into Python.</summary>
        public ManualResetEventSlim Release { get; } = new();

        /// <summary>Parks the first call (bounded by the watchdog), then runs Python on the Python thread.</summary>
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

            PythonRuntime.Run(() => PythonEngine.Exec("s = 0\nfor i in range(100000): s += i"));
            return base.SampleIndependent(study, trial, paramName, distribution);
        }
    }
}

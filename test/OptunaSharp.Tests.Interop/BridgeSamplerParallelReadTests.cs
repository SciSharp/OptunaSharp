using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text.Json;
using AwesomeAssertions;
using OptunaSharp.Interop.BoTorch;
using OptunaSharp.Interop.PyCma;
using OptunaSharp.Interop.Pymoo;
using OptunaSharp.Interop.PythonNet;
using OptunaSharp.Samplers;
using OptunaSharp.Tests.Concurrency;
using Python.Runtime;

namespace OptunaSharp.Tests.Interop;

/// <summary>
///     The four Python bridges under parallel reads, measured and pinned: what each repeats when eight trials read at
///     once. Unlike the Core samplers (<c>SamplerParallelReadTests</c> in the unit suite), the bridges are not changed to
///     steer concurrent asks apart: each hands out what the Python code it drives hands out, so its repeats are that
///     code's, and these tests record them so a change to a bridge, or to the Python package under it, shows.
/// </summary>
/// <remarks>
///     <para>
///         Two volleys per bridge, both from <see cref="ParallelReads" />. <see cref="ParallelReads.ReadInLockstep" />
///         makes the interleaving where every read of a round lands before any write, on one thread, so its counts
///         repeat exactly and are pinned. <see cref="ParallelReads.Fire" /> runs eight real threads, so a bridge whose
///         result depends on which reads saw which writes repeats a varying count; only what holds in every
///         interleaving is asserted there, and the counts measured on 2026-11-08 are noted.
///     </para>
///     <para>
///         Measured (8 rounds of 8 reads; collisions/history repeats): PyCma lockstep 49/0 (every trial of a round after
///         the startup receives one point, as <c>optuna_integration.PyCmaSampler</c> does, which counts only the
///         running trials whose parameters are stored), threaded 0-1/0. pymoo NSGA-II and AGE-MOEA-2 lockstep 42/0
///         (one offspring per round), threaded 0-2 collisions and 7-8 history repeats: under real threads the bridge
///         re-hands offspring a told trial already evaluated, which one trial at a time never happens. OptunaHub
///         <c>random_search</c> and <c>pso</c> 0/0 in both, the hub's <c>differential_evolution</c> 49/0 in both (it
///         ignores running trials). BoTorch 0/0 in both, with and without <c>considerRunningTrials</c>.
///     </para>
///     <para>
///         Self-skips (Inconclusive) without the engine or the bridge's Python package, like the other RequiresPython
///         tests. pycma must be 4.x: the 3.2.2 that <c>pip install pymoo</c> pins fails under NumPy 2.
///     </para>
/// </remarks>
[TestClass]
public class BridgeSamplerParallelReadTests : OptunaInteropTestBase
{
    /// <summary>Rounds per volley: 64 trials, past every bridge's startup.</summary>
    private const int Rounds = 8;

    /// <summary>
    ///     The lockstep emulation of <see cref="ParallelReads.ReadInLockstep" /> in Python, over the continuous space of
    ///     <see cref="ParallelReads.Space.Continuous" />, with the same objective and the same operation order, so both
    ///     sides compute bit-identical values. Defines <c>_lockstep(sampler, rounds, shooters)</c>, which returns
    ///     <c>[number, x, y, z]</c> per trial in round order and, within a round, in trial-number order.
    /// </summary>
    private const string PythonLockstep =
        "import json, warnings\n" +
        "warnings.simplefilter('ignore')\n" +
        "def _lockstep(sampler, rounds, shooters):\n" +
        "    study = optuna.create_study(sampler=sampler)\n" +
        "    points = []\n" +
        "    for _ in range(rounds):\n" +
        "        trials = [study.ask() for _ in range(shooters)]\n" +
        "        for t in trials:\n" +
        "            study._thread_local.cached_all_trials = None\n" +
        "            t.relative_params\n" +
        "        values = []\n" +
        "        for t in trials:\n" +
        "            x = t.suggest_float('x', -10, 10)\n" +
        "            y = t.suggest_float('y', -10, 10)\n" +
        "            z = t.suggest_float('z', 0, 1)\n" +
        "            values.append((x - 2) * (x - 2) + (y + 1) * (y + 1) + z)\n" +
        "            points.append([t.number, x, y, z])\n" +
        "        for t, v in zip(trials, values):\n" +
        "            study.tell(t, v)\n" +
        "    return points\n";

    /// <summary>
    ///     In lockstep the PyCma bridge hands every trial of a round after the startup one point, 49 collisions in 64
    ///     trials, and these are exactly the points <c>optuna_integration.PyCmaSampler</c> hands out read the same way:
    ///     both count the trials after the last told generation that have stored parameters, which no running trial has
    ///     yet when every read lands first.
    /// </summary>
    [TestMethod]
    [TestCategory("RequiresPython")]
    public void PyCma_InLockstep_RepeatsExactlyWhatOptunasPyCmaSamplerRepeats()
    {
        RequirePyCmaOrInconclusive();

        ParallelReads.Outcome outcome = ParallelReads.ReadInLockstep(
            new PyCmaSampler(seed: 1), false, ParallelReads.Space.Continuous, Rounds);

        outcome.Collisions.Should().Be(7 * (Rounds - 1), "every trial of a round after the startup gets one point: {0}", outcome);
        outcome.HistoryRepeats.Should().Be(0, "{0}", outcome);

        PyExec(PythonLockstep +
               "from optuna.integration import PyCmaSampler\n" +
               "_points = json.dumps(_lockstep(PyCmaSampler(seed=1), 8, 8))\n");
        AssertSamePoints(outcome, PyStr("_points"));
    }

    /// <summary>
    ///     PyCma under eight real threads: a trial counts the round's other trials whose parameters it saw stored, so
    ///     reads that missed each other's writes share a point (0-1 per volley measured), never more than in lockstep,
    ///     and no trial receives a told trial's point.
    /// </summary>
    [TestMethod]
    [TestCategory("RequiresPython")]
    public void PyCma_ParallelReads_CollideAtMostAsInLockstep_AndNeverRepeatATrial()
    {
        RequirePyCmaOrInconclusive();

        ParallelReads.Outcome outcome = ParallelReads.Fire(
            new PyCmaSampler(seed: 1), false, ParallelReads.Space.Continuous, Rounds);

        outcome.Trials.Should().Be(Rounds * 8);
        outcome.Collisions.Should().BeLessThanOrEqualTo(7 * (Rounds - 1), "{0}", outcome);
        outcome.HistoryRepeats.Should().Be(0, "{0}", outcome);
    }

    /// <summary>
    ///     In lockstep the pymoo bridge hands every trial of a round one offspring once generation 0 is told (rounds 2
    ///     to 7: 42 collisions), for both algorithms: it gives a trial the offspring whose index is the count of trials
    ///     asked since the last told generation, and none of the round's other trials stored parameters yet.
    /// </summary>
    /// <param name="algorithm">The pymoo algorithm the bridge drives.</param>
    [TestMethod]
    [TestCategory("RequiresPython")]
    [DataRow(PymooAlgorithm.Nsga2)]
    [DataRow(PymooAlgorithm.AgeMoea2)]
    public void Pymoo_InLockstep_HandsEveryTrialOfARoundOneOffspring(PymooAlgorithm algorithm)
    {
        RequirePymooOrInconclusive();

        ParallelReads.Outcome outcome = ParallelReads.ReadInLockstep(
            new PymooSampler(algorithm, populationSize: 12, seed: 1), true, ParallelReads.Space.Continuous, Rounds);

        outcome.Collisions.Should().Be(42, "rounds 2 to 7 each hand one offspring to all 8 trials: {0}", outcome);
        outcome.HistoryRepeats.Should().Be(0, "{0}", outcome);
    }

    /// <summary>
    ///     The pymoo bridge under eight real threads completes every trial and collides at most as in lockstep. Not
    ///     asserted, because it varies with the interleaving: it also re-hands offspring that told trials already
    ///     evaluated (7-8 of 64 trials measured), which read one at a time it never does.
    /// </summary>
    /// <param name="algorithm">The pymoo algorithm the bridge drives.</param>
    [TestMethod]
    [TestCategory("RequiresPython")]
    [DataRow(PymooAlgorithm.Nsga2)]
    [DataRow(PymooAlgorithm.AgeMoea2)]
    public void Pymoo_ParallelReads_CompleteEveryTrial_AndCollideAtMostAsInLockstep(PymooAlgorithm algorithm)
    {
        RequirePymooOrInconclusive();

        ParallelReads.Outcome outcome = ParallelReads.Fire(
            new PymooSampler(algorithm, populationSize: 12, seed: 1), true, ParallelReads.Space.Continuous, Rounds);

        outcome.Trials.Should().Be(Rounds * 8);
        outcome.Collisions.Should().BeLessThanOrEqualTo(42, "{0}", outcome);
    }

    /// <summary>
    ///     The OptunaHub bridge adds no repeat of its own: a hub sampler that draws every point afresh
    ///     (<c>random_search</c>) or queues distinct candidates (<c>pso</c>) repeats nothing in either volley, and one
    ///     that ignores running trials (<c>differential_evolution</c>) hands every trial of a round one point in both,
    ///     49 collisions, because each read sees the same told trials.
    /// </summary>
    /// <param name="package">The registry package under <c>refs/optunahub-registry/package</c>.</param>
    /// <param name="className">The sampler class the package exports.</param>
    /// <param name="collisions">The collisions pinned for both volleys.</param>
    [TestMethod]
    [TestCategory("RequiresPython")]
    [DataRow("samplers/random_search", "RandomSampler", 0)]
    [DataRow("samplers/pso", "PSOSampler", 0)]
    [DataRow("samplers/differential_evolution", "DESampler", 49)]
    public void OptunaHub_BothVolleys_RepeatWhatTheHubSamplerRepeats(string package, string className, int collisions)
    {
        RequireOptunaHubOrInconclusive();
        string registry = Path.Combine(RepoRoot(), "refs", "optunahub-registry", "package");
        if (!Directory.Exists(Path.Combine(registry, package)))
            Assert.Inconclusive(
                "refs/optunahub-registry submodule not initialized " +
                "(`git submodule update --init refs/optunahub-registry`).");

        using var module = OptunaHub.LoadLocalModule(package, registry);
        using var lockstepSampler = module.CreateSampler(className, new() { ["seed"] = 1 });
        using var threadedSampler = module.CreateSampler(className, new() { ["seed"] = 1 });

        ParallelReads.Outcome lockstep = ParallelReads.ReadInLockstep(
            lockstepSampler, false, ParallelReads.Space.Continuous, Rounds);
        ParallelReads.Outcome threaded = ParallelReads.Fire(
            threadedSampler, false, ParallelReads.Space.Continuous, Rounds);

        lockstep.Collisions.Should().Be(collisions, "lockstep: {0}", lockstep);
        lockstep.HistoryRepeats.Should().Be(0, "lockstep: {0}", lockstep);
        threaded.Collisions.Should().Be(collisions, "threaded: {0}", threaded);
        threaded.HistoryRepeats.Should().Be(0, "threaded: {0}", threaded);
    }

    /// <summary>
    ///     The BoTorch bridge repeats nothing in either volley, whether or not it conditions on running trials: each read
    ///     optimizes the acquisition under its own seed, and the optimum it lands on differs bit for bit. Each read fits
    ///     a model, so a round of eight takes up to a minute; slow category.
    /// </summary>
    /// <param name="considerRunningTrials">Whether the bridge passes running trials to the acquisition as pending points.</param>
    [TestMethod]
    [TestCategory("RequiresPython")]
    [TestCategory("Slow")]
    [DataRow(false)]
    [DataRow(true)]
    public void BoTorch_BothVolleys_RepeatNothing(bool considerRunningTrials)
    {
        RequireBoTorchOrInconclusive();

        ParallelReads.Outcome lockstep = ParallelReads.ReadInLockstep(
            new BoTorchSampler(nStartupTrials: 8, considerRunningTrials: considerRunningTrials, seed: 1),
            false, ParallelReads.Space.Continuous, rounds: 3);
        ParallelReads.Outcome threaded = ParallelReads.Fire(
            new BoTorchSampler(nStartupTrials: 8, considerRunningTrials: considerRunningTrials, seed: 1),
            false, ParallelReads.Space.Continuous, rounds: 3, roundTimeout: TimeSpan.FromMinutes(5));

        lockstep.Duplicates.Should().Be(0, "lockstep: {0}", lockstep);
        threaded.Duplicates.Should().Be(0, "threaded: {0}", threaded);
    }

    /// <summary>
    ///     Asserts that the Python lockstep handed out exactly the C# lockstep's points: the same trial numbers, and every
    ///     coordinate bit for bit.
    /// </summary>
    /// <param name="outcome">The C# volley.</param>
    /// <param name="pythonPoints">The Python volley, as JSON <c>[[number, x, y, z], ...]</c>.</param>
    /// <exception cref="AssertFailedException">A trial's number or coordinate differs.</exception>
    /// <exception cref="JsonException">The Python output is not the expected JSON.</exception>
    private static void AssertSamePoints(ParallelReads.Outcome outcome, string pythonPoints)
    {
        double[][] python = JsonSerializer.Deserialize<double[][]>(pythonPoints)!;
        python.Should().HaveCount(outcome.Points.Count);
        string[] names = { "x", "y", "z" };
        for (int i = 0; i < python.Length; i++)
        {
            ParallelReads.Point point = outcome.Points[i];
            ((int)python[i][0]).Should().Be(point.Number);
            for (int j = 0; j < names.Length; j++)
            {
                double csharp = Convert.ToDouble(point.Params[names[j]], CultureInfo.InvariantCulture);
                BitConverter.DoubleToInt64Bits(csharp).Should().Be(BitConverter.DoubleToInt64Bits(python[i][j + 1]),
                    "trial {0} {1}: C# {2:R} vs Python {3:R}", point.Number, names[j], csharp, python[i][j + 1]);
            }
        }
    }

    /// <summary>
    ///     Skips (Inconclusive) unless pycma and <c>optuna_integration</c> import and pycma can run a bounded strategy
    ///     under the installed NumPy (pycma 3.2.2 cannot under NumPy 2).
    /// </summary>
    /// <exception cref="AssertInconclusiveException">The engine, pycma or <c>optuna_integration</c> is missing or broken.</exception>
    private void RequirePyCmaOrInconclusive()
    {
        PythonSession.EnsureOrInconclusive();
        try
        {
            PyExec("import cma\nfrom optuna.integration import PyCmaSampler\n" +
                   "_probe = cma.CMAEvolutionStrategy([0.0, 0.0], 0.5,\n" +
                   "    {'bounds': [[-1.0, -1.0], [1.0, 1.0]], 'seed': 1, 'verbose': -9})\n" +
                   "_xs = _probe.ask()\n" +
                   "_probe.tell(_xs, [1.0] * len(_xs))\n");
        }
        catch (PythonException ex)
        {
            Assert.Inconclusive("pycma / optuna_integration cannot run a bounded strategy here (pip install \"cma>=4\"): " +
                                ex.Message.Split('\n')[0]);
        }
    }

    /// <summary>Skips (Inconclusive) unless pymoo imports.</summary>
    /// <exception cref="AssertInconclusiveException">The engine or pymoo is missing.</exception>
    private void RequirePymooOrInconclusive()
    {
        PythonSession.EnsureOrInconclusive();
        try
        {
            PyExec("import pymoo\n");
        }
        catch (PythonException)
        {
            Assert.Inconclusive("pymoo is not installed in the interop interpreter (`pip install pymoo`).");
        }
    }

    /// <summary>Skips (Inconclusive) unless optunahub imports.</summary>
    /// <exception cref="AssertInconclusiveException">The engine or optunahub is missing.</exception>
    private static void RequireOptunaHubOrInconclusive()
    {
        PythonSession.EnsureOrInconclusive();
        try
        {
            PythonRuntime.Run(() => { using var _ = Py.Import("optunahub"); });
        }
        catch (Exception ex)
        {
            Assert.Inconclusive($"optunahub not importable (`pip install optunahub`): {ex.Message}");
        }
    }

    /// <summary>Skips (Inconclusive) unless botorch and <c>optuna_integration</c> import.</summary>
    /// <exception cref="AssertInconclusiveException">The engine, botorch or <c>optuna_integration</c> is missing.</exception>
    private void RequireBoTorchOrInconclusive()
    {
        PythonSession.EnsureOrInconclusive();
        try
        {
            // A direct import is robust (find_spec can misbehave once torch is loaded into the engine).
            PyExec("import botorch, optuna_integration\n");
        }
        catch (PythonException)
        {
            Assert.Inconclusive("botorch / optuna_integration are not installed in the interop interpreter.");
        }
    }

    /// <summary>The repository root, two folders above this source file.</summary>
    /// <param name="thisFile">This file's path, filled in by the compiler.</param>
    /// <returns>The absolute path of the repository root.</returns>
    private static string RepoRoot([CallerFilePath] string? thisFile = null)
        => Path.GetFullPath(Path.Combine(Path.GetDirectoryName(thisFile)!, "..", ".."));
}

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.CompilerServices;
using AwesomeAssertions;
using OptunaSharp;
using OptunaSharp.Interop.PythonNet;
using Python.Runtime;

namespace OptunaSharp.Tests.Interop;

/// <summary>
///     Open parity bugs in the OptunaHub sampler bridge (<see cref="OptunaHubSampler" />) found by the 2026-11-10 parity
///     QA run (<c>docs/qa/PARITY_QA_2026-11-10.md</c>, QA-72). Each test fails today and passes once fixed, at which
///     point its <c>OpenBugs</c> category (and the CI guard below) should be removed. QA-73 is the same defect in
///     <see cref="OptunaHubPruner" />.
/// </summary>
/// <remarks>
///     The interop CI job filters only <c>TestCategory!=Slow</c>, not <c>OpenBugs</c>, so these tests report
///     Inconclusive when the <c>CI</c> environment variable is <c>true</c> (GitHub Actions sets it) instead of turning
///     the job red; run them locally to reproduce. They need Python with Optuna and OptunaHub and the pinned
///     <c>refs/optunahub-registry</c> submodule, and self-skip without them like the other live tests.
/// </remarks>
[TestClass]
public class Run2OptunaHubBridgeParityBugTests
{
    /// <summary>
    ///     QA-72: OptunaHub's <c>PSOSampler</c> (<c>samplers/pso</c>) does all of its swarm work in <c>after_trial</c> —
    ///     it collects each generation's completed trials there, builds and steps the swarm, and queues the next
    ///     generation for <c>sample_relative</c>. <see cref="OptunaHubSampler" /> never forwards <c>after_trial</c> (nor
    ///     <c>before_trial</c>) to the Python sampler, so through the bridge the swarm is never built and every trial
    ///     stays in PSO's random phase.
    /// </summary>
    /// <remarks>
    ///     Measured with Optuna 5.0.0 and OptunaHub 0.4.1 on the pinned registry: run natively,
    ///     <c>PSOSampler(seed=0)</c> on this objective serves its first swarm candidate at trial 11
    ///     (<c>x = 0.5201450987386327, y = 1.0528363508210972, n = 3</c>), reproducibly; through the bridge trial 11 is
    ///     another random draw, and all trials from 11 on differ. The other half of the gap is QA-72's second test.
    /// </remarks>
    [TestMethod]
    [TestCategory("RequiresPython")]
    [TestCategory("OpenBugs")]
    public void PsoThroughBridge_BuildsItsSwarmInAfterTrialLikeNativePython()
    {
        using var module = LoadHubModuleOrInconclusive("samplers/pso");
        using var sampler = module.CreateSampler("PSOSampler", new() { ["seed"] = 0 });

        var study = Optuna.CreateStudy(sampler: sampler);
        study.Optimize(t => Objective(t)[0], nTrials: 12);

        var eleventh = study.Trials[11].Params;
        // Native Python: optunahub.load_local_module("samplers/pso", registry).PSOSampler(seed=0), the same objective.
        ((double)eleventh["x"]!).Should().Be(0.5201450987386327);
        ((double)eleventh["y"]!).Should().Be(1.0528363508210972);
        Convert.ToInt64(eleventh["n"]).Should().Be(3);
    }

    /// <summary>
    ///     QA-72: the bridge's shadow study mirrors each finished trial as <c>create_trial(state, values, params,
    ///     distributions)</c> only and is rebuilt for every trial, so the trial system attributes a hub sampler writes in
    ///     <c>sample_relative</c> are lost, and the C# trials' user / system attributes, intermediate values and
    ///     constraints never reach it. OptunaHub's <c>MoCmaSampler</c> (<c>samplers/mocma</c>) stores its generation in
    ///     <c>trial.system_attrs["mocma:g"]</c> and reads it back from the completed trials, so through the bridge it fails
    ///     with <c>KeyError: 'mocma:g'</c>, where the native run completes all 30 trials.
    /// </summary>
    /// <remarks>
    ///     The same loss changes the trajectories of the NSGA-II-based hub samplers (<c>nsgaii_with_initial_trials</c>,
    ///     <c>speaii</c>: from the first generation boundary, trial 8 with <c>population_size=8</c>) and of
    ///     <c>implicit_natural_gradient</c> (from trial 7), all reproducible natively. Unaffected in the same comparison:
    ///     <c>differential_evolution</c>, <c>restart_cmaes</c>, <c>levy_flight_sampler</c>, <c>nelder_mead</c>,
    ///     <c>tpe_union_multivariate</c>, <c>random_search</c>. In the registry 26 samplers implement <c>after_trial</c>
    ///     and 25 read or write system attributes.
    /// </remarks>
    [TestMethod]
    [TestCategory("RequiresPython")]
    [TestCategory("OpenBugs")]
    public void MoCmaThroughBridge_KeepsTheSystemAttributesItWrites()
    {
        using var module = LoadHubModuleOrInconclusive("samplers/mocma");
        using var sampler = module.CreateSampler("MoCmaSampler", new() { ["seed"] = 0 });

        var study = Optuna.CreateStudy(sampler: sampler,
            directions: new[] { StudyDirection.Minimize, StudyDirection.Minimize });
        Action optimize = () => study.Optimize(Objective, nTrials: 30);

        optimize.Should().NotThrow("the native MoCmaSampler(seed=0) completes the 30 trials");
        study.Trials.Count(t => t.State == TrialState.Complete).Should().Be(30);
    }

    /// <summary>
    ///     QA-73: OptunaHub's <c>pruners/successive_halving</c> re-exports Optuna's own <c>SuccessiveHalvingPruner</c>,
    ///     which records each rung a trial passes in that trial's system attributes (<c>completed_rung_N</c>) and compares
    ///     a trial with the other trials' values in the same rung. <see cref="OptunaHubPruner" /> rebuilds its shadow study
    ///     for every <c>prune</c> call from <c>create_trial(state, values, params, distributions, intermediate_values)</c>,
    ///     so the rung records of earlier calls and of other trials are gone; each trial then competes only with itself,
    ///     is always promotable, and nothing is ever pruned.
    /// </summary>
    /// <remarks>
    ///     Measured: natively (Optuna 5.0.0, <c>RandomSampler(seed=0)</c>, 40 trials of this ten-step objective) 36 trials
    ///     are pruned, the first ones (trials 3 and 4) after step 1; through the bridge, with the port's bit-exact
    ///     <c>RandomSampler(seed: 0)</c> drawing the same <c>x</c> values, none is. <c>pruners/hyperband</c> (also
    ///     Optuna's, built on successive halving) prunes 31 natively and none through the bridge; <c>median</c>,
    ///     <c>percentile</c>, <c>wilcoxon</c> and <c>threshold</c>, which keep no state, agree trial for trial.
    /// </remarks>
    [TestMethod]
    [TestCategory("RequiresPython")]
    [TestCategory("OpenBugs")]
    public void SuccessiveHalvingThroughBridge_PrunesLikeNativePython()
    {
        using var module = LoadHubModuleOrInconclusive("pruners/successive_halving");
        using var pruner = module.CreatePruner("SuccessiveHalvingPruner", new());

        var study = Optuna.CreateStudy(sampler: new OptunaSharp.Samplers.RandomSampler(seed: 0), pruner: pruner);
        study.Optimize(ReportingObjective, nTrials: 40);

        study.Trials[3].State.Should().Be(TrialState.Pruned, "native Optuna prunes trial 3 after step 1");
        study.Trials[3].IntermediateValues.Keys.Should().Equal(0L, 1L);
        study.Trials.Count(t => t.State == TrialState.Pruned).Should().Be(36);
    }

    /// <summary>
    ///     The pruning comparison's objective (<c>hub_prune.py</c> in the QA scratch): one float, ten reported steps
    ///     <c>(x - 0.3)² + 1 / (step + 1)</c>, asking the pruner after each.
    /// </summary>
    /// <param name="trial">The trial to suggest on and report to.</param>
    /// <returns>The last reported value, when the trial is never pruned.</returns>
    /// <exception cref="TrialPruned">The study's pruner asks to prune the trial; <c>Optimize</c> records it as pruned.</exception>
    private static double ReportingObjective(Trial trial)
    {
        double x = trial.SuggestFloat("x", -1.0, 1.0);
        double value = 0.0;
        for (int step = 0; step < 10; step++)
        {
            value = Math.Pow(x - 0.3, 2) + 1.0 / (step + 1);
            trial.Report(value, step);
            if (trial.ShouldPrune())
                throw new TrialPruned();
        }
        return value;
    }

    /// <summary>
    ///     The objective both sides of the comparison ran (<c>hub_run.py</c> in the QA scratch): two floats and an int;
    ///     the second value is used by the multi-objective test only.
    /// </summary>
    /// <param name="trial">The trial to suggest on.</param>
    /// <returns>The two objective values; single-objective callers take the first.</returns>
    private static IReadOnlyList<double> Objective(Trial trial)
    {
        double x = trial.SuggestFloat("x", -1.0, 1.0);
        double y = trial.SuggestFloat("y", 0.0, 2.0);
        long n = trial.SuggestInt("n", 0, 5);
        double first = Math.Pow(x - 0.3, 2) + Math.Pow(y - 0.4, 2) + 0.1 * n;
        double second = Math.Pow(x + 0.5, 2) + Math.Pow(y - 1.5, 2) + 0.05 * (5 - n);
        return new[] { first, second };
    }

    /// <summary>
    ///     Skips on CI (see the class remarks), then loads <paramref name="package" /> from the pinned registry, or reports
    ///     Inconclusive when Python, OptunaHub or the submodule is missing.
    /// </summary>
    /// <param name="package">The registry package, e.g. <c>samplers/pso</c>.</param>
    /// <returns>The loaded module; the caller disposes it.</returns>
    private static OptunaHubModule LoadHubModuleOrInconclusive(string package)
    {
        // The interop CI job does not filter OpenBugs; keep it green until the bug is fixed.
        if (string.Equals(Environment.GetEnvironmentVariable("CI"), "true", StringComparison.OrdinalIgnoreCase))
            Assert.Inconclusive("Open bug QA-72; run locally (the interop CI job does not filter OpenBugs).");
        PythonSession.EnsureOrInconclusive();
        try
        {
            PythonRuntime.Run(() => { using var _ = Py.Import("optunahub"); });
        }
        catch (Exception ex)
        {
            Assert.Inconclusive($"optunahub not importable (`pip install optunahub`): {ex.Message}");
        }
        string root = Path.Combine(RepoRoot(), "refs", "optunahub-registry", "package");
        if (!Directory.Exists(Path.Combine(root, package)))
            Assert.Inconclusive("refs/optunahub-registry submodule not initialized " +
                                "(`git submodule update --init refs/optunahub-registry`).");
        return OptunaHub.LoadLocalModule(package, root);
    }

    /// <summary>Returns the repository root, two levels above this source file.</summary>
    /// <param name="thisFile">Filled in by the compiler with this file's path.</param>
    /// <returns>The absolute repository root.</returns>
    private static string RepoRoot([CallerFilePath] string? thisFile = null)
        => Path.GetFullPath(Path.Combine(Path.GetDirectoryName(thisFile)!, "..", ".."));
}

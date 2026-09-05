using System;
using System.IO;
using System.Linq;
using System.Runtime.CompilerServices;
using OptunaSharp;
using OptunaSharp.Distributions;
using OptunaSharp.Interop.PythonNet;
using Python.Runtime;

namespace OptunaSharp.Tests.Interop;

/// <summary>
///     Live end-to-end coverage for the OptunaHub bridge (<c>OptunaSharp.Interop.pythonnet</c>): it
///     loads real OptunaHub packages through the same <c>optunahub.load_local_module</c> path Python
///     uses and drives an OptunaSharp <see cref="Study" /> with them — a hub <c>BaseProblem</c> as an
///     objective (Float/Int/Categorical marshalling) and a hub sampler over the shadow study.
///     <para>Self-skips (Inconclusive) when Python / Optuna / OptunaHub or the pinned
///     <c>refs/optunahub-registry</c> submodule is unavailable, like the other RequiresPython tests.</para>
/// </summary>
[TestClass]
public class OptunaHubBridgeParityTests
{
    [TestMethod]
    [TestCategory("RequiresPython")]
    public void BenchmarkBridge_ToyProblem_ExposesSpaceAndOptimizes()
    {
        RequireOptunaHubOrInconclusive();

        // A dependency-free BaseProblem written into a temp local registry, loaded via the real path.
        string root = WriteToyBenchmarkRegistry();
        try
        {
            using var module = OptunaHub.LoadLocalModule("benchmarks/toy", root);
            var problem = module.CreateProblem("Problem");

            // directions + search space marshal to OptunaSharp types, in declaration order.
            Assert.AreEqual(1, problem.Directions.Count);
            Assert.AreEqual(StudyDirection.Minimize, problem.Directions[0]);
            Assert.AreEqual(3, problem.SearchSpace.Count);
            Assert.IsInstanceOfType(problem.SearchSpace["x"], typeof(FloatDistribution));
            Assert.IsInstanceOfType(problem.SearchSpace["n"], typeof(IntDistribution));
            Assert.IsInstanceOfType(problem.SearchSpace["c"], typeof(CategoricalDistribution));

            var study = Optuna.CreateStudy(directions: problem.Directions);
            study.Optimize(problem.SingleObjective, nTrials: 40);

            Assert.AreEqual(40, study.Trials.Count);
            Assert.IsFalse(double.IsNaN(study.BestValue));
            // Minimum is 0 at (x=2, n=0, c="b"); 40 TPE trials comfortably beat the trivial bound.
            Assert.IsTrue(study.BestValue < 1.0, $"best={study.BestValue}");
            problem.Dispose();
        }
        finally
        {
            TryDelete(root);
        }
    }

    [TestMethod]
    [TestCategory("RequiresPython")]
    public void SamplerBridge_HubRandomSampler_DrivesStudy()
    {
        RequireOptunaHubOrInconclusive();
        string registry = SubmoduleRegistryOrInconclusive();

        // samplers/random_search re-exports optuna.samplers.RandomSampler — a dependency-free hub sampler.
        using var module = OptunaHub.LoadLocalModule("samplers/random_search", registry);
        using var sampler = module.CreateSampler("RandomSampler", new() { ["seed"] = 0 });

        var study = Optuna.CreateStudy(sampler: sampler);
        study.Optimize(t => { var x = t.SuggestFloat("x", -10, 10); return (x - 2) * (x - 2); }, nTrials: 30);

        Assert.AreEqual(30, study.Trials.Count);
        Assert.IsFalse(double.IsNaN(study.BestValue));
        Assert.IsTrue(study.BestValue >= 0.0, "objective is a square, cannot be negative");
        // The suggestions came from the real Python sampler and stayed in-range.
        double best = Convert.ToDouble(study.BestParams["x"]);
        Assert.IsTrue(best is >= -10 and <= 10, $"x out of range: {best}");
    }

    // ---- helpers --------------------------------------------------------------------------------

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

    private static string SubmoduleRegistryOrInconclusive()
    {
        string root = Path.Combine(RepoRoot(), "refs", "optunahub-registry", "package");
        if (!Directory.Exists(Path.Combine(root, "samplers", "random_search")))
            Assert.Inconclusive(
                "refs/optunahub-registry submodule not initialized " +
                "(`git submodule update --init refs/optunahub-registry`).");
        return root;
    }

    private static string WriteToyBenchmarkRegistry()
    {
        string root = Path.Combine(Path.GetTempPath(), "optunasharp_toy_registry_" + Guid.NewGuid().ToString("N"));
        string dir = Path.Combine(root, "benchmarks", "toy");
        Directory.CreateDirectory(dir);
        File.WriteAllText(Path.Combine(dir, "__init__.py"),
            "from __future__ import annotations\n" +
            "from typing import Any\n" +
            "import optuna\n" +
            "import optunahub\n" +
            "\n" +
            "class Problem(optunahub.benchmarks.BaseProblem):\n" +
            "    @property\n" +
            "    def search_space(self) -> dict[str, optuna.distributions.BaseDistribution]:\n" +
            "        return {\n" +
            "            'x': optuna.distributions.FloatDistribution(-5.0, 5.0),\n" +
            "            'n': optuna.distributions.IntDistribution(0, 10),\n" +
            "            'c': optuna.distributions.CategoricalDistribution(['a', 'b', 'c']),\n" +
            "        }\n" +
            "    @property\n" +
            "    def directions(self) -> list[optuna.study.StudyDirection]:\n" +
            "        return [optuna.study.StudyDirection.MINIMIZE]\n" +
            "    def evaluate(self, params: dict[str, Any]) -> float:\n" +
            "        return (params['x'] - 2.0) ** 2 + 0.1 * params['n'] + (0.0 if params['c'] == 'b' else 1.0)\n");
        return root;
    }

    private static void TryDelete(string dir)
    {
        try { Directory.Delete(dir, recursive: true); } catch { /* best effort */ }
    }

    private static string RepoRoot([CallerFilePath] string? thisFile = null)
    {
        // thisFile = <repo>/test/OptunaSharp.Tests.Interop/OptunaHubBridgeParityTests.cs
        string dir = Path.GetDirectoryName(thisFile)!;
        return Path.GetFullPath(Path.Combine(dir, "..", ".."));
    }
}

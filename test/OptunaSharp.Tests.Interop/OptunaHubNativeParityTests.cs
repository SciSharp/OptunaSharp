using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text.Json;
using AwesomeAssertions;
using OptunaSharp;
using OptunaSharp.Interop.PythonNet;
using OptunaSharp.Samplers;
using Python.Runtime;

namespace OptunaSharp.Tests.Interop;

/// <summary>
///     Registry samplers and pruners through the OptunaHub bridges against the same objects run natively, trial for
///     trial: the comparison that found QA-72 and QA-73 (<c>docs/qa/PARITY_QA_2026-11-10.md</c>), kept as a gate for the
///     persistent shadow study (<c>OptunaHubShadow</c>, 2026-11-12).
/// </summary>
/// <remarks>
///     <para>
///         Both sides run in the one embedded interpreter, against the pinned <c>refs/optunahub-registry</c>, with a
///         seeded sampler and the study name <c>hub-parity</c> on both (some hub objects derive state from the name:
///         <c>HyperbandPruner</c>'s brackets do). The native side is a plain Python study; the OptunaSharp side drives
///         the same class through <see cref="OptunaHubSampler" /> or <see cref="OptunaHubPruner" /> (with the port's
///         bit-exact <see cref="RandomSampler" /> under a pruner). Every trial's state, parameters (bit for bit) and
///         reported steps must agree.
///     </para>
///     <para>
///         The samplers cover what the QA run found broken (PSO builds its swarm in <c>after_trial</c>; MO-CMA, the
///         NSGA-II variants and the implicit natural gradient keep generations in system attributes) and what it found
///         unaffected; the pruners cover the stateful successive halving and Hyperband and the stateless rest. A package
///         whose imports fail (a missing optional dependency) self-skips, like the other RequiresPython tests.
///     </para>
/// </remarks>
[TestClass]
public class OptunaHubNativeParityTests
{
    /// <summary>The study name both sides use.</summary>
    private const string StudyName = "hub-parity";

    /// <summary>The objective both sides run: two floats and an int, one or two values.</summary>
    private const string PythonObjective = """
        def objective(trial):
            x = trial.suggest_float("x", -1.0, 1.0)
            y = trial.suggest_float("y", 0.0, 2.0)
            n = trial.suggest_int("n", 0, 5)
            first = (x - 0.3) ** 2 + (y - 0.4) ** 2 + 0.1 * n
            second = (x + 0.5) ** 2 + (y - 1.5) ** 2 + 0.05 * (5 - n)
            return first if objectives == 1 else (first, second)
        """;

    /// <summary>The pruning objective both sides run: one float, ten reported steps, the pruner asked after each.</summary>
    private const string PythonReportingObjective = """
        def objective(trial):
            x = trial.suggest_float("x", -1.0, 1.0)
            value = 0.0
            for step in range(10):
                value = (x - 0.3) ** 2 + 1.0 / (step + 1)
                trial.report(value, step)
                if trial.should_prune():
                    raise optuna.TrialPruned()
            return value
        """;

    /// <summary>A registry sampler, seeded, through the bridge and natively: the same trials.</summary>
    /// <param name="package">The registry package.</param>
    /// <param name="className">The sampler class it exports.</param>
    /// <param name="objectives">1 or 2.</param>
    /// <param name="nTrials">Trials per side.</param>
    /// <param name="populationSize">A <c>population_size</c> kwarg, or 0 for none.</param>
    [TestMethod]
    [TestCategory("RequiresPython")]
    [DataRow("samplers/pso", "PSOSampler", 1, 30, 0)]
    [DataRow("samplers/mocma", "MoCmaSampler", 2, 30, 0)]
    [DataRow("samplers/nsgaii_with_initial_trials", "NSGAIIwITSampler", 2, 30, 8)]
    [DataRow("samplers/speaii", "SPEAIISampler", 2, 30, 8)]
    [DataRow("samplers/implicit_natural_gradient", "ImplicitNaturalGradientSampler", 1, 30, 0)]
    [DataRow("samplers/differential_evolution", "DESampler", 1, 30, 0)]
    [DataRow("samplers/restart_cmaes", "RestartCmaEsSampler", 1, 30, 0)]
    [DataRow("samplers/levy_flight_sampler", "LevyFlightSampler", 1, 30, 0)]
    [DataRow("samplers/nelder_mead", "NelderMeadSampler", 1, 30, 0)]
    [DataRow("samplers/tpe_union_multivariate", "TPEUnionMultivariateSampler", 1, 30, 0)]
    [DataRow("samplers/random_search", "RandomSampler", 1, 30, 0)]
    public void Sampler_ThroughTheBridge_MatchesNativePython(
        string package, string className, int objectives, int nTrials, int populationSize)
        => CompareSampler(package, className, objectives, nTrials, populationSize);

    /// <summary>
    ///     The Optuna team's <c>AutoSampler</c> (<c>samplers/auto_sampler</c>), which differed from trial 1 through the
    ///     rebuilt shadow (QA-72): the same trials as natively. It switches to Optuna's GP sampler, so 30 trials take about
    ///     20 seconds a side; slow category.
    /// </summary>
    [TestMethod]
    [TestCategory("RequiresPython")]
    [TestCategory("Slow")]
    public void AutoSampler_ThroughTheBridge_MatchesNativePython()
        => CompareSampler("samplers/auto_sampler", "AutoSampler", objectives: 1, nTrials: 30, populationSize: 0);

    /// <summary>Runs a registry sampler natively and through the bridge, then compares every trial.</summary>
    /// <param name="package">The registry package.</param>
    /// <param name="className">The sampler class it exports.</param>
    /// <param name="objectives">1 or 2.</param>
    /// <param name="nTrials">Trials per side.</param>
    /// <param name="populationSize">A <c>population_size</c> kwarg, or 0 for none.</param>
    private static void CompareSampler(string package, string className, int objectives, int nTrials, int populationSize)
    {
        string registry = RegistryOrInconclusive(package);
        var kwargs = new Dictionary<string, object?> { ["seed"] = 0 };
        if (populationSize > 0)
            kwargs["population_size"] = populationSize;

        string native = RunNativeOrInconclusive(package, registry, className, kwargs, objectives, nTrials, sampler: true);

        using var module = OptunaHub.LoadLocalModule(package, registry);
        using var sampler = module.CreateSampler(className, kwargs);
        var directions = Enumerable.Repeat(StudyDirection.Minimize, objectives).ToArray();
        var study = Optuna.CreateStudy(studyName: StudyName, sampler: sampler, directions: directions);
        study.Optimize(t =>
        {
            double x = t.SuggestFloat("x", -1.0, 1.0);
            double y = t.SuggestFloat("y", 0.0, 2.0);
            long n = t.SuggestInt("n", 0, 5);
            double first = Math.Pow(x - 0.3, 2) + Math.Pow(y - 0.4, 2) + 0.1 * n;
            double second = Math.Pow(x + 0.5, 2) + Math.Pow(y - 1.5, 2) + 0.05 * (5 - n);
            return objectives == 1 ? new[] { first } : new[] { first, second };
        }, nTrials: nTrials);

        ShouldMatch(native, study.Trials, new[] { "x", "y", "n" });
    }

    /// <summary>A registry pruner through the bridge and natively, under the same seeded random sampler: the same trials.</summary>
    /// <param name="package">The registry package.</param>
    /// <param name="className">The pruner class it exports.</param>
    /// <param name="kwargName">One constructor kwarg's name, or empty for the defaults.</param>
    /// <param name="kwargValue">That kwarg's value.</param>
    /// <exception cref="TrialPruned">
    ///     Thrown inside the objective when the bridged pruner prunes a trial; <c>Optimize</c> records the trial as pruned,
    ///     so it never leaves the test.
    /// </exception>
    [TestMethod]
    [TestCategory("RequiresPython")]
    [DataRow("pruners/successive_halving", "SuccessiveHalvingPruner", "", 0.0)]
    [DataRow("pruners/hyperband", "HyperbandPruner", "", 0.0)]
    [DataRow("pruners/median", "MedianPruner", "", 0.0)]
    [DataRow("pruners/percentile", "PercentilePruner", "percentile", 25.0)]
    [DataRow("pruners/wilcoxon", "WilcoxonPruner", "", 0.0)]
    [DataRow("pruners/threshold", "ThresholdPruner", "upper", 1.2)]
    public void Pruner_ThroughTheBridge_MatchesNativePython(
        string package, string className, string kwargName, double kwargValue)
    {
        string registry = RegistryOrInconclusive(package);
        var kwargs = new Dictionary<string, object?>();
        if (kwargName.Length > 0)
            kwargs[kwargName] = kwargValue;

        string native = RunNativeOrInconclusive(package, registry, className, kwargs, objectives: 1, nTrials: 40,
            sampler: false);

        using var module = OptunaHub.LoadLocalModule(package, registry);
        using var pruner = module.CreatePruner(className, kwargs);
        var study = Optuna.CreateStudy(studyName: StudyName, sampler: new RandomSampler(seed: 0), pruner: pruner);
        study.Optimize(t =>
        {
            double x = t.SuggestFloat("x", -1.0, 1.0);
            double value = 0.0;
            for (int step = 0; step < 10; step++)
            {
                value = Math.Pow(x - 0.3, 2) + 1.0 / (step + 1);
                t.Report(value, step);
                if (t.ShouldPrune())
                    throw new TrialPruned();
            }
            return value;
        }, nTrials: 40);

        ShouldMatch(native, study.Trials, new[] { "x" });
    }

    /// <summary>
    ///     Runs the package natively and returns its trials as JSON: <c>[state, {param: value}, [reported steps]]</c> each,
    ///     floats as <c>float.hex</c> so the comparison is bit for bit.
    /// </summary>
    /// <param name="package">The registry package.</param>
    /// <param name="registry">The registry root.</param>
    /// <param name="className">The class to instantiate.</param>
    /// <param name="kwargs">Its constructor kwargs (scalars).</param>
    /// <param name="objectives">1 or 2.</param>
    /// <param name="nTrials">Trials to run.</param>
    /// <param name="sampler">Whether the class is a sampler (else a pruner under <c>RandomSampler(seed=0)</c>).</param>
    /// <returns>The JSON text.</returns>
    /// <exception cref="PythonException">The native run fails.</exception>
    private static string RunNative(string package, string registry, string className,
        IReadOnlyDictionary<string, object?> kwargs, int objectives, int nTrials, bool sampler)
    {
        string kwargText = string.Join(", ", kwargs.Select(kv => $"{kv.Key}={PythonLiteral(kv.Value)}"));
        string code = $$"""
            import json
            import optuna
            import optunahub
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            module = optunahub.load_local_module(package, registry_root=registry)
            obj = getattr(module, class_name)({{kwargText}})
            objectives = {{objectives}}
            {{(sampler ? PythonObjective : PythonReportingObjective)}}
            if sampler:
                study = optuna.create_study(study_name="{{StudyName}}", sampler=obj, directions=["minimize"] * objectives)
            else:
                study = optuna.create_study(study_name="{{StudyName}}", sampler=optuna.samplers.RandomSampler(seed=0),
                                            pruner=obj)
            study.optimize(objective, n_trials={{nTrials}})
            def enc(v):
                return float(v).hex() if isinstance(v, float) else v
            result = json.dumps([[t.state.name, {k: enc(v) for k, v in t.params.items()}, sorted(t.intermediate_values)]
                                 for t in study.trials])
            """;
        return PythonRuntime.Run(() =>
        {
            using PyModule scope = Py.CreateScope();
            using (var p = package.ToPython()) scope.Set("package", p);
            using (var r = registry.ToPython()) scope.Set("registry", r);
            using (var c = className.ToPython()) scope.Set("class_name", c);
            using (var s = sampler.ToPython()) scope.Set("sampler", s);
            scope.Exec(code);
            return scope.Get<string>("result");
        });
    }

    /// <summary>
    ///     <see cref="RunNative" />, reporting Inconclusive when the package needs an optional dependency this interpreter
    ///     lacks: several import theirs lazily (restart CMA-ES needs <c>cmaes</c>, the Wilcoxon pruner <c>scipy</c>,
    ///     <c>AutoSampler</c> <c>torch</c>), so the import check of <see cref="RegistryOrInconclusive" /> passes and the
    ///     run fails.
    /// </summary>
    /// <param name="package">The registry package.</param>
    /// <param name="registry">The registry root.</param>
    /// <param name="className">The class to instantiate.</param>
    /// <param name="kwargs">Its constructor kwargs (scalars).</param>
    /// <param name="objectives">1 or 2.</param>
    /// <param name="nTrials">Trials to run.</param>
    /// <param name="sampler">Whether the class is a sampler (else a pruner).</param>
    /// <returns>The native trials' JSON.</returns>
    /// <exception cref="PythonException">The native run fails for another reason than a missing module.</exception>
    private static string RunNativeOrInconclusive(string package, string registry, string className,
        IReadOnlyDictionary<string, object?> kwargs, int objectives, int nTrials, bool sampler)
    {
        try
        {
            return RunNative(package, registry, className, kwargs, objectives, nTrials, sampler);
        }
        catch (PythonException ex) when (ex.Type.Name is "ModuleNotFoundError" or "ImportError")
        {
            Assert.Inconclusive($"{package} needs a module this interpreter lacks: {ex.Message}");
            throw;  // unreachable: Inconclusive throws
        }
    }

    /// <summary>A scalar kwarg as Python source.</summary>
    /// <param name="value">An int, long or double.</param>
    /// <returns>Its Python literal.</returns>
    /// <exception cref="ArgumentException">Another type.</exception>
    private static string PythonLiteral(object? value) => value switch
    {
        int or long => Convert.ToString(value, CultureInfo.InvariantCulture)!,
        double d => d.ToString("R", CultureInfo.InvariantCulture),
        _ => throw new ArgumentException($"Unsupported kwarg value {value}.", nameof(value)),
    };

    /// <summary>
    ///     Compares the native trials with the bridge's: the first trial that differs fails the test, showing both.
    /// </summary>
    /// <param name="nativeJson">The native trials (see <see cref="RunNative" />).</param>
    /// <param name="trials">The OptunaSharp study's trials.</param>
    /// <param name="paramNames">The parameters to compare.</param>
    private static void ShouldMatch(string nativeJson, IReadOnlyList<FrozenTrial> trials, string[] paramNames)
    {
        using JsonDocument doc = JsonDocument.Parse(nativeJson);
        JsonElement[] native = doc.RootElement.EnumerateArray().ToArray();
        trials.Should().HaveCount(native.Length);
        for (int i = 0; i < native.Length; i++)
        {
            string expected = Describe(native[i], paramNames);
            string actual = Describe(trials[i], paramNames);
            actual.Should().Be(expected, $"trial {i} (native first, then the bridge's)");
        }
    }

    /// <summary>One native trial as text: state, parameters (floats as hex), reported steps.</summary>
    /// <param name="trial">The native trial's JSON.</param>
    /// <param name="paramNames">The parameters to show.</param>
    /// <returns>The description.</returns>
    private static string Describe(JsonElement trial, string[] paramNames)
    {
        string state = trial[0].GetString()!;
        JsonElement ps = trial[1];
        var parts = paramNames.Select(name => ps.TryGetProperty(name, out JsonElement v)
            ? $"{name}={(v.ValueKind == JsonValueKind.String ? v.GetString() : v.GetRawText())}"
            : $"{name}=-");
        string steps = string.Join(",", trial[2].EnumerateArray().Select(e => e.GetRawText()));
        return $"{state} {string.Join(" ", parts)} steps=[{steps}]";
    }

    /// <summary>One OptunaSharp trial in <see cref="Describe(JsonElement, string[])" />'s format.</summary>
    /// <param name="trial">The trial.</param>
    /// <param name="paramNames">The parameters to show.</param>
    /// <returns>The description.</returns>
    private static string Describe(FrozenTrial trial, string[] paramNames)
    {
        string state = trial.State.ToString().ToUpperInvariant();
        var parts = paramNames.Select(name => trial.Params.TryGetValue(name, out object? v)
            ? $"{name}={(v is double d ? Hex(d) : Convert.ToString(v, CultureInfo.InvariantCulture))}"
            : $"{name}=-");
        string steps = string.Join(",", trial.IntermediateValues.Keys.OrderBy(k => k));
        return $"{state} {string.Join(" ", parts)} steps=[{steps}]";
    }

    /// <summary>Python's <c>float.hex()</c> of a double, for a bit-exact text comparison.</summary>
    /// <param name="value">The double.</param>
    /// <returns>The hex text, e.g. <c>0x1.0000000000000p+0</c>.</returns>
    private static string Hex(double value)
        => PythonRuntime.Run(() =>
        {
            using PyObject f = value.ToPython();
            using PyObject hex = f.InvokeMethod("hex");
            return hex.As<string>();
        });

    /// <summary>
    ///     The pinned registry root, after checking Python, OptunaHub and the package's own imports, or Inconclusive.
    /// </summary>
    /// <param name="package">The registry package the test loads.</param>
    /// <returns>The registry root.</returns>
    private static string RegistryOrInconclusive(string package)
    {
        PythonSession.EnsureOrInconclusive();
        string registry = Path.Combine(RepoRoot(), "refs", "optunahub-registry", "package");
        if (!Directory.Exists(Path.Combine(registry, package)))
            Assert.Inconclusive("refs/optunahub-registry submodule not initialized " +
                                "(`git submodule update --init refs/optunahub-registry`).");
        try
        {
            PythonRuntime.Run(() =>
            {
                using PyObject optunahub = Py.Import("optunahub");
                using PyObject load = optunahub.GetAttr("load_local_module");
                using var kwargs = new PyDict();
                using (var r = registry.ToPython()) kwargs["registry_root"] = r;
                using var p = package.ToPython();
                using PyObject _ = load.Invoke(new[] { p }, kwargs);
            });
        }
        catch (Exception ex)
        {
            Assert.Inconclusive($"{package} does not import here (an optional dependency is missing?): {ex.Message}");
        }
        return registry;
    }

    /// <summary>Returns the repository root, two levels above this source file.</summary>
    /// <param name="thisFile">Filled in by the compiler with this file's path.</param>
    /// <returns>The absolute repository root.</returns>
    private static string RepoRoot([CallerFilePath] string? thisFile = null)
        => Path.GetFullPath(Path.Combine(Path.GetDirectoryName(thisFile)!, "..", ".."));
}

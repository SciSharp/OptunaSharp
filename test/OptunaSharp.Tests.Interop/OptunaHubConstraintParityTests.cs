using System;
using System.IO;
using System.Linq;
using AwesomeAssertions;
using OptunaSharp.Interop.PythonNet;
using OptunaSharp.Samplers;

namespace OptunaSharp.Tests.Interop;

/// <summary>
///     Live parity of the OptunaHub bridge's constraint step against OptunaHub's own
///     <c>BaseProblem.__call__</c> (<see cref="OptunaHubProblem" />). OptunaHub 0.5.0 made a problem's
///     <c>evaluate_constraints(params)</c> part of <c>__call__</c>: each value is written onto the trial with
///     <c>trial.set_constraint</c> before <c>evaluate</c> runs, and a legacy sequence return is named by index
///     with a <c>FutureWarning</c>. Each test optimizes the same toy hub package twice with
///     <c>RandomSampler(seed=3)</c> — once through the C# bridge, once through Python's
///     <c>study.optimize(problem)</c> — and asserts full study-state parity (params, values and every system
///     attribute, i.e. the <c>constraints:*</c> entries).
///     <para>The installed OptunaHub decides which behavior is correct: the ≥ 0.5.0 tests go Inconclusive on
///     an older OptunaHub and the pre-0.5.0 test goes Inconclusive on ≥ 0.5.0, so one environment never
///     passes a test vacuously.</para>
/// </summary>
[TestClass]
public class OptunaHubConstraintParityTests : OptunaInteropTestBase
{
    /// <summary>Trials per study: enough RandomSampler draws to cover both signs of every constraint.</summary>
    private const int NTrials = 12;

    /// <summary>The study name used on BOTH sides, so the snapshot's <c>study_name</c> comparison holds.</summary>
    private const string StudyName = "optunahub-constraint-parity";

    /// <summary>A toy problem whose <c>evaluate_constraints</c> returns a <c>dict</c> (the OptunaHub ≥ 0.5.0 contract), including an <c>int</c> value Optuna's <c>set_constraint</c> casts to <c>float</c>.</summary>
    private const string DictConstraintsProblem =
        "from __future__ import annotations\n" +
        "from typing import Any\n" +
        "import optuna\n" +
        "import optunahub\n" +
        "\n" +
        "class Problem(optunahub.benchmarks.BaseProblem):\n" +
        "    @property\n" +
        "    def search_space(self) -> dict[str, optuna.distributions.BaseDistribution]:\n" +
        "        return {\n" +
        "            'x': optuna.distributions.FloatDistribution(-3.0, 3.0),\n" +
        "            'y': optuna.distributions.FloatDistribution(-3.0, 3.0),\n" +
        "        }\n" +
        "    @property\n" +
        "    def directions(self) -> list[optuna.study.StudyDirection]:\n" +
        "        return [optuna.study.StudyDirection.MINIMIZE]\n" +
        "    def evaluate(self, params: dict[str, Any]) -> float:\n" +
        "        return params['x'] * params['x'] + params['y'] * params['y']\n" +
        "    def evaluate_constraints(self, params: dict[str, Any]) -> dict[str, float]:\n" +
        "        return {\n" +
        "            'sum': params['x'] + params['y'] - 1.0,\n" +
        "            'x_cap': params['x'] - 0.5,\n" +
        "            'flag': int(params['x'] > 0),\n" +
        "        }\n";

    /// <summary>The same problem written for an older OptunaHub: <c>evaluate_constraints</c> returns a list.</summary>
    private const string SequenceConstraintsProblem =
        "from __future__ import annotations\n" +
        "from typing import Any\n" +
        "import optuna\n" +
        "import optunahub\n" +
        "\n" +
        "class Problem(optunahub.benchmarks.BaseProblem):\n" +
        "    @property\n" +
        "    def search_space(self) -> dict[str, optuna.distributions.BaseDistribution]:\n" +
        "        return {\n" +
        "            'x': optuna.distributions.FloatDistribution(-3.0, 3.0),\n" +
        "            'y': optuna.distributions.FloatDistribution(-3.0, 3.0),\n" +
        "        }\n" +
        "    @property\n" +
        "    def directions(self) -> list[optuna.study.StudyDirection]:\n" +
        "        return [optuna.study.StudyDirection.MINIMIZE]\n" +
        "    def evaluate(self, params: dict[str, Any]) -> float:\n" +
        "        return params['x'] * params['x'] + params['y'] * params['y']\n" +
        "    def evaluate_constraints(self, params: dict[str, Any]) -> list[float]:\n" +
        "        return [params['x'] + params['y'] - 1.0, params['x'] - 0.5]\n";

    /// <summary>OptunaHub 0.5.0's verbatim warning for a sequence return (see <c>_as_constraints_dict</c>).</summary>
    private const string LegacySequenceWarning =
        "Returning a sequence from `evaluate_constraints` is deprecated in v0.5.0. " +
        "The benchmark package cached in your local directory is outdated. " +
        "Please update it by `optunahub.load_module(..., force_reload=True)`.";

    /// <summary>
    ///     OptunaHub ≥ 0.5.0, dict constraints: the bridge must record exactly the <c>constraints:*</c>
    ///     system attributes (names, float values, insertion order) that <c>BaseProblem.__call__</c> records,
    ///     alongside identical params and values.
    /// </summary>
    [TestMethod]
    [TestCategory("RequiresPython")]
    public void DictConstraints_MatchBaseProblemCall()
    {
        RequireOptunaHub(withEvaluateConstraints: true);
        string root = WriteRegistry("toy_dict_constraints", DictConstraintsProblem);
        try
        {
            Study csStudy = RunThroughBridge("benchmarks/toy_dict_constraints", root);
            RunThroughPython("benchmarks/toy_dict_constraints", root);

            StudySnapshot.AssertParity(PyStudyState("study"), StudySnapshot.Capture(csStudy), DatetimeComparison.NullnessOnly);
            AssertSystemAttrOrderMatches(csStudy);
            // Non-vacuous: every trial really carries all three constraints, the int one cast to float.
            csStudy.Trials.Should().OnlyContain(t =>
                t.SystemAttrs.ContainsKey("constraints:sum")
                && t.SystemAttrs.ContainsKey("constraints:x_cap")
                && t.SystemAttrs["constraints:flag"] is double);
        }
        finally
        {
            TryDelete(root);
        }
    }

    /// <summary>
    ///     OptunaHub ≥ 0.5.0, legacy list constraints: the bridge must name them <c>"0"</c>, <c>"1"</c> like
    ///     <c>_as_constraints_dict</c> and raise OptunaHub's own <c>FutureWarning</c> once per trial, exactly
    ///     as many times and with exactly the text Python raises.
    /// </summary>
    [TestMethod]
    [TestCategory("RequiresPython")]
    public void SequenceConstraints_IndexNamedWithFutureWarning_MatchBaseProblemCall()
    {
        RequireOptunaHub(withEvaluateConstraints: true);
        string root = WriteRegistry("toy_sequence_constraints", SequenceConstraintsProblem);
        try
        {
            PyExec(
                "import warnings\n" +
                "_cs_cm = warnings.catch_warnings(record=True)\n" +
                "_cs_w = _cs_cm.__enter__()\n" +
                "warnings.simplefilter('always')\n");
            Study csStudy;
            try
            {
                csStudy = RunThroughBridge("benchmarks/toy_sequence_constraints", root);
            }
            finally
            {
                PyExec("_cs_cm.__exit__(None, None, None)\n");
            }

            PyExec(
                "_py_cm = warnings.catch_warnings(record=True)\n" +
                "_py_w = _py_cm.__enter__()\n" +
                "warnings.simplefilter('always')\n");
            try
            {
                RunThroughPython("benchmarks/toy_sequence_constraints", root);
            }
            finally
            {
                PyExec("_py_cm.__exit__(None, None, None)\n");
            }

            StudySnapshot.AssertParity(PyStudyState("study"), StudySnapshot.Capture(csStudy), DatetimeComparison.NullnessOnly);
            AssertSystemAttrOrderMatches(csStudy);
            csStudy.Trials.Should().OnlyContain(t =>
                t.SystemAttrs.ContainsKey("constraints:0") && t.SystemAttrs.ContainsKey("constraints:1"));

            PySet("_expected_msg", LegacySequenceWarning);
            PyExec(
                "_cs_n = sum(1 for w in _cs_w if issubclass(w.category, FutureWarning) and str(w.message) == _expected_msg)\n" +
                "_py_n = sum(1 for w in _py_w if issubclass(w.category, FutureWarning) and str(w.message) == _expected_msg)\n");
            PyLong("_py_n").Should().Be(NTrials, "OptunaHub warns once per __call__");
            PyLong("_cs_n").Should().Be(PyLong("_py_n"), "the bridge must warn exactly as often as BaseProblem.__call__");
        }
        finally
        {
            TryDelete(root);
        }
    }

    /// <summary>
    ///     OptunaHub &lt; 0.5.0: <c>BaseProblem.__call__</c> never applied constraints, so the bridge must not
    ///     either — a problem that defines <c>evaluate_constraints</c> yields no <c>constraints:*</c> attributes
    ///     on either side.
    /// </summary>
    [TestMethod]
    [TestCategory("RequiresPython")]
    public void PreV050_AppliesNoConstraints_MatchBaseProblemCall()
    {
        RequireOptunaHub(withEvaluateConstraints: false);
        string root = WriteRegistry("toy_pre050_constraints", DictConstraintsProblem);
        try
        {
            Study csStudy = RunThroughBridge("benchmarks/toy_pre050_constraints", root);
            RunThroughPython("benchmarks/toy_pre050_constraints", root);

            StudySnapshot.AssertParity(PyStudyState("study"), StudySnapshot.Capture(csStudy), DatetimeComparison.NullnessOnly);
            csStudy.Trials.Should().OnlyContain(t => !t.SystemAttrs.Keys.Any(k => k.StartsWith("constraints:", StringComparison.Ordinal)));
        }
        finally
        {
            TryDelete(root);
        }
    }

    // ---- helpers --------------------------------------------------------------------------------

    /// <summary>
    ///     Goes Inconclusive unless OptunaHub is importable AND its <c>BaseProblem</c> has (or lacks)
    ///     <c>evaluate_constraints</c> — the public signal of OptunaHub ≥ 0.5.0.
    /// </summary>
    /// <param name="withEvaluateConstraints">Whether the test needs OptunaHub ≥ 0.5.0 (<c>true</c>) or &lt; 0.5.0 (<c>false</c>).</param>
    private void RequireOptunaHub(bool withEvaluateConstraints)
    {
        try
        {
            PyExec("import optunahub\n_has_ec = hasattr(optunahub.benchmarks.BaseProblem, 'evaluate_constraints')\n");
        }
        catch (Exception ex)
        {
            Assert.Inconclusive($"optunahub not importable (`pip install optunahub`): {ex.Message}");
        }

        bool hasEvaluateConstraints = PyBool("_has_ec");
        if (hasEvaluateConstraints != withEvaluateConstraints)
            Assert.Inconclusive(withEvaluateConstraints
                ? $"needs optunahub >= 0.5.0 (BaseProblem.evaluate_constraints); installed {PyStr("optunahub.__version__")}"
                : $"needs optunahub < 0.5.0; installed {PyStr("optunahub.__version__")}");
    }

    /// <summary>Optimizes the hub problem through the C# bridge with <c>RandomSampler(seed: 3)</c>.</summary>
    /// <param name="package">The registry package path, e.g. <c>benchmarks/toy_dict_constraints</c>.</param>
    /// <param name="root">The temporary local registry root.</param>
    /// <returns>The finished C# study.</returns>
    private static Study RunThroughBridge(string package, string root)
    {
        using var module = OptunaHub.LoadLocalModule(package, root);
        using OptunaHubProblem problem = module.CreateProblem("Problem");
        var study = Optuna.CreateStudy(
            sampler: new RandomSampler(seed: 3), studyName: StudyName, directions: problem.Directions);
        study.Optimize(problem.SingleObjective, nTrials: NTrials);
        return study;
    }

    /// <summary>Optimizes the same package through Python's own <c>BaseProblem.__call__</c>, leaving the study in the scope as <c>study</c>.</summary>
    /// <param name="package">The registry package path.</param>
    /// <param name="root">The temporary local registry root.</param>
    private void RunThroughPython(string package, string root)
    {
        PySet("_root", root);
        PySet("_package", package);
        PySet("_study_name", StudyName);
        PyExec(
            "import optunahub\n" +
            "_mod = optunahub.load_local_module(_package, registry_root=_root)\n" +
            "study = optuna.create_study(study_name=_study_name, sampler=optuna.samplers.RandomSampler(seed=3))\n" +
            $"study.optimize(_mod.Problem(), n_trials={NTrials})\n");
    }

    /// <summary>
    ///     <see cref="StudySnapshot.AssertParity" /> compares attribute dicts as sets of keys, so this pins the
    ///     one thing it cannot see: each trial's <c>system_attrs</c> keys appear in the same ORDER on both
    ///     sides — i.e. the bridge applies the constraints in the problem's dict/sequence order, as
    ///     <c>__call__</c> does. Run after <see cref="RunThroughPython" /> (reads the scope's <c>study</c>).
    /// </summary>
    /// <param name="csStudy">The C# study produced by <see cref="RunThroughBridge" />.</param>
    private void AssertSystemAttrOrderMatches(Study csStudy)
    {
        string csOrder = string.Join("|", csStudy.Trials.Select(t => string.Join(",", t.SystemAttrs.Keys)));
        string pyOrder = PyStr("'|'.join(','.join(t.system_attrs) for t in study.trials)");
        csOrder.Should().Be(pyOrder, "constraints must be recorded in the order BaseProblem.__call__ records them");
    }

    /// <summary>Writes one toy benchmark package into a fresh temporary local registry.</summary>
    /// <param name="name">The package directory name under <c>benchmarks/</c>.</param>
    /// <param name="source">The package's <c>__init__.py</c> source.</param>
    /// <returns>The registry root to pass to <c>load_local_module</c>.</returns>
    private static string WriteRegistry(string name, string source)
    {
        string root = Path.Combine(Path.GetTempPath(), "optunasharp_hub_constraints_" + Guid.NewGuid().ToString("N"));
        string dir = Path.Combine(root, "benchmarks", name);
        Directory.CreateDirectory(dir);
        File.WriteAllText(Path.Combine(dir, "__init__.py"), source);
        return root;
    }

    /// <summary>Best-effort removal of a temporary registry (a locked file must not fail the test).</summary>
    /// <param name="dir">The directory to delete recursively.</param>
    private static void TryDelete(string dir)
    {
        try { Directory.Delete(dir, recursive: true); } catch { /* best effort */ }
    }
}

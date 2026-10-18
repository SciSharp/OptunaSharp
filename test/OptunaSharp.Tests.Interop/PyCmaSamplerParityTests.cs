using System;
using System.Collections.Generic;
using AwesomeAssertions;
using OptunaSharp.Distributions;
using OptunaSharp.Interop.PyCma;
using OptunaSharp.Storages;
using Python.Runtime;

namespace OptunaSharp.Tests.Interop;

/// <summary>
///     Full-state byte-parity of the native <see cref="PyCmaSampler" /> against the real
///     <c>optuna.integration.PyCmaSampler</c> (which wraps pycma — Nikolaus Hansen's <c>cma</c> package).
///     OptunaSharp runs the study (its <see cref="PyCmaSampler" /> drives the genuine pycma via
///     pythonnet); Python then runs the identical study with <c>optuna.integration.PyCmaSampler</c> in the
///     same embedded interpreter, and every trial's params (external + internal IEEE bits), value, and
///     distributions must agree byte-for-byte.
///
///     <para><b>Why this is bit-exact (not tolerance):</b> both sides call the <em>same</em> pycma with
///     the <em>same</em> seed, and the only C# computation on the boundary is the parameter ⇄ cma-space
///     transform — <c>log</c>/<c>exp</c>/<c>round</c>/<c>mean</c>, each bit-exact vs numpy/CPython — so the
///     <c>x0</c>/<c>sigma0</c>/<c>bounds</c> fed in and the params read back match to 0 ULP. The startup
///     trials come from a bit-exact <see cref="OptunaSharp.Samplers.RandomSampler" />, and PyCmaSampler
///     rebuilds+reseeds the optimizer on every ask (it stores no state), so the two studies never
///     interleave and reproduce each other trial-for-trial. This is the whole point of the bridge: it is
///     pycma, so it agrees with pycma exactly.</para>
///
///     <para>Self-skips (Inconclusive) when <c>cma</c> / <c>optuna_integration</c> are not installed in the
///     interop interpreter.</para>
/// </summary>
[TestClass]
public class PyCmaSamplerParityTests : OptunaInteropTestBase
{
    /// <summary>
    ///     PyCmaSampler's ctor imports pycma (and the Python side needs optuna_integration); probe both up front
    ///     and self-skip if either is missing — or if the installed pycma cannot run at all under the installed
    ///     NumPy.
    /// </summary>
    /// <remarks>
    ///     The second probe exists because <c>pip install pymoo</c> pins <c>cma==3.2.2</c>, which predates NumPy 2:
    ///     its box-constraint handler calls <c>np.array(..., copy=False)</c>, which NumPy 2 rejects, so every
    ///     bounded strategy dies in pycma's own constructor. That is a broken interpreter, not a bridge defect —
    ///     the parity tests then failed 13-of-15 on every run. The probe builds one bounded strategy in pure
    ///     Python (the exact path the bridge drives) and turns that environment into an explained skip.
    /// </remarks>
    private void RequirePyCmaOrInconclusive()
    {
        PythonSession.EnsureOrInconclusive();
        try
        {
            PyExec("import cma\nfrom optuna.integration import PyCmaSampler\n");
        }
        catch (PythonException)
        {
            Assert.Inconclusive("pycma (`cma`) / optuna_integration are not installed in the interop interpreter.");
        }

        try
        {
            PyExec(
                "_probe = cma.CMAEvolutionStrategy([0.0, 0.0], 0.5,\n" +
                "    {'bounds': [[-1.0, -1.0], [1.0, 1.0]], 'seed': 1, 'verbose': -9})\n" +
                "_xs = _probe.ask()\n" +
                "_probe.tell(_xs, [1.0] * len(_xs))\n");
        }
        catch (PythonException ex)
        {
            Assert.Inconclusive(
                $"the installed pycma {PyStr("cma.__version__")} cannot run a bounded strategy under numpy " +
                $"{PyStr("__import__('numpy').__version__")} ({ex.Message.Split('\n')[0]}). " +
                "Install a NumPy-2-compatible pycma: pip install \"cma>=4\".");
        }
    }

    /// <summary>Run the C# study, run the equivalent Python study, and diff the complete state bit-for-bit
    /// (datetimes only compared for null-ness, as elsewhere in the interop suite).</summary>
    private void AssertFullStateParity(Study csStudy, string pyCreateAndRun)
    {
        var csState = StudySnapshot.Capture(csStudy);
        PyExec(pyCreateAndRun);
        var pyState = PyStudyState("study");
        StudySnapshot.AssertParity(pyState, csState, DatetimeComparison.NullnessOnly);
    }

    // ---- float sphere (relative CMA over continuous params) -------------------------------------

    [TestMethod]
    [TestCategory("RequiresPython")]
    // NOTE: seeds must be non-zero — pycma treats a falsy seed (0) as "pick a random seed", which is
    // non-reproducible (see cma.evolution_strategy: `if not opts['seed']: np.random.seed()`).
    [DataRow(17, 30)]
    [DataRow(42, 30)]
    public void FloatSphere2D_FullState_IsBitIdentical(int seed, int nTrials)
    {
        RequirePyCmaOrInconclusive();

        var csStudy = Optuna.CreateStudy(
            storage: new InMemoryStorage(), studyName: "pycma",
            sampler: new PyCmaSampler(seed: seed));
        csStudy.Optimize(t =>
        {
            double x = t.SuggestFloat("x", -5, 5);
            double y = t.SuggestFloat("y", -5, 5);
            return Math.Pow(x - 1.5, 2) + Math.Pow(y + 2.0, 2);
        }, nTrials: nTrials);

        AssertFullStateParity(csStudy,
            "import warnings; warnings.simplefilter('ignore')\n" +
            $"study = optuna.create_study(study_name='pycma', sampler=optuna.integration.PyCmaSampler(seed={seed}))\n" +
            "def _obj(t):\n" +
            "    x = t.suggest_float('x', -5, 5)\n" +
            "    y = t.suggest_float('y', -5, 5)\n" +
            "    return (x - 1.5) ** 2 + (y + 2.0) ** 2\n" +
            $"study.optimize(_obj, n_trials={nTrials})\n");
    }

    [TestMethod]
    [TestCategory("RequiresPython")]
    [DataRow(7, 28)]
    [DataRow(13, 26)]
    public void FloatSphere3D_FullState_IsBitIdentical(int seed, int nTrials)
    {
        RequirePyCmaOrInconclusive();

        var csStudy = Optuna.CreateStudy(
            storage: new InMemoryStorage(), studyName: "pycma",
            sampler: new PyCmaSampler(seed: seed));
        csStudy.Optimize(t =>
        {
            double a = t.SuggestFloat("a", -5, 5);
            double b = t.SuggestFloat("b", -5, 5);
            double c = t.SuggestFloat("c", -5, 5);
            return Math.Pow(a - 0.7, 2) + Math.Pow(b + 1.3, 2) + Math.Pow(c - 2.1, 2);
        }, nTrials: nTrials);

        AssertFullStateParity(csStudy,
            "import warnings; warnings.simplefilter('ignore')\n" +
            $"study = optuna.create_study(study_name='pycma', sampler=optuna.integration.PyCmaSampler(seed={seed}))\n" +
            "def _obj(t):\n" +
            "    a = t.suggest_float('a', -5, 5)\n" +
            "    b = t.suggest_float('b', -5, 5)\n" +
            "    c = t.suggest_float('c', -5, 5)\n" +
            "    return (a - 0.7) ** 2 + (b + 1.3) ** 2 + (c - 2.1) ** 2\n" +
            $"study.optimize(_obj, n_trials={nTrials})\n");
    }

    // ---- mixed float + int (both optimized by pycma) --------------------------------------------

    [TestMethod]
    [TestCategory("RequiresPython")]
    public void MixedFloatInt_FullState_IsBitIdentical()
    {
        RequirePyCmaOrInconclusive();

        var csStudy = Optuna.CreateStudy(
            storage: new InMemoryStorage(), studyName: "pycma",
            sampler: new PyCmaSampler(seed: 11));
        csStudy.Optimize(t =>
        {
            double a = t.SuggestFloat("a", -4, 4);
            long b = t.SuggestInt("bb", -5, 5);
            double c = t.SuggestFloat("c", 0.01, 100, log: true);
            return Math.Pow(a - 1.5, 2) + Math.Pow(b + 2, 2) + Math.Pow(Math.Log(c) - 1.0, 2);
        }, nTrials: 26);

        AssertFullStateParity(csStudy,
            "import warnings; warnings.simplefilter('ignore')\n" +
            "import numpy as np\n" +
            "study = optuna.create_study(study_name='pycma', sampler=optuna.integration.PyCmaSampler(seed=11))\n" +
            "def _obj(t):\n" +
            "    a = t.suggest_float('a', -4, 4)\n" +
            "    b = t.suggest_int('bb', -5, 5)\n" +
            "    c = t.suggest_float('c', 0.01, 100, log=True)\n" +
            "    return (a - 1.5) ** 2 + (b + 2) ** 2 + (np.log(c) - 1.0) ** 2\n" +
            "study.optimize(_obj, n_trials=26)\n");
    }

    // ---- float + categorical: PyCma optimizes categoricals as ordinals (unlike CmaEsSampler) -----

    [TestMethod]
    [TestCategory("RequiresPython")]
    public void FloatCategorical_FullState_IsBitIdentical()
    {
        RequirePyCmaOrInconclusive();

        var csStudy = Optuna.CreateStudy(
            storage: new InMemoryStorage(), studyName: "pycma",
            sampler: new PyCmaSampler(seed: 3));
        csStudy.Optimize(t =>
        {
            double x = t.SuggestFloat("x", -5, 5);
            double y = t.SuggestFloat("y", -5, 5);
            // The categorical is sampled by pycma as an ordinal (same choice on both sides for the same
            // seed); it need not influence the objective for the parity to exercise the categorical path.
            t.SuggestCategorical("k", new CategoricalChoice[] { "a", "b", "c", "d" });
            return Math.Pow(x - 1.0, 2) + Math.Pow(y + 0.5, 2);
        }, nTrials: 24);

        AssertFullStateParity(csStudy,
            "import warnings; warnings.simplefilter('ignore')\n" +
            "study = optuna.create_study(study_name='pycma', sampler=optuna.integration.PyCmaSampler(seed=3))\n" +
            "def _obj(t):\n" +
            "    x = t.suggest_float('x', -5, 5)\n" +
            "    y = t.suggest_float('y', -5, 5)\n" +
            "    t.suggest_categorical('k', ['a', 'b', 'c', 'd'])\n" +
            "    return (x - 1.0) ** 2 + (y + 0.5) ** 2\n" +
            "study.optimize(_obj, n_trials=24)\n");
    }

    // ---- explicit n_startup_trials (independent RandomSampler prefix, then pycma) -----------------

    [TestMethod]
    [TestCategory("RequiresPython")]
    public void NStartupTrials_FullState_IsBitIdentical()
    {
        RequirePyCmaOrInconclusive();

        var csStudy = Optuna.CreateStudy(
            storage: new InMemoryStorage(), studyName: "pycma",
            sampler: new PyCmaSampler(seed: 9, nStartupTrials: 5));
        csStudy.Optimize(t =>
        {
            double x = t.SuggestFloat("x", -5, 5);
            double y = t.SuggestFloat("y", -5, 5);
            return Math.Pow(x - 2.0, 2) + Math.Pow(y - 2.0, 2);
        }, nTrials: 28);

        AssertFullStateParity(csStudy,
            "import warnings; warnings.simplefilter('ignore')\n" +
            "study = optuna.create_study(study_name='pycma', sampler=optuna.integration.PyCmaSampler(seed=9, n_startup_trials=5))\n" +
            "def _obj(t):\n" +
            "    x = t.suggest_float('x', -5, 5)\n" +
            "    y = t.suggest_float('y', -5, 5)\n" +
            "    return (x - 2.0) ** 2 + (y - 2.0) ** 2\n" +
            "study.optimize(_obj, n_trials=28)\n");
    }

    // ---- direction = maximize (the tell sign-flip path) -----------------------------------------

    [TestMethod]
    [TestCategory("RequiresPython")]
    public void Maximize_FullState_IsBitIdentical()
    {
        RequirePyCmaOrInconclusive();

        // MAXIMIZE flips the sign of each told value (`ys = [-y ...]`); the negated sphere peaks at 0.
        var csStudy = Optuna.CreateStudy(
            storage: new InMemoryStorage(), studyName: "pycma", direction: StudyDirection.Maximize,
            sampler: new PyCmaSampler(seed: 21));
        csStudy.Optimize(t =>
        {
            double x = t.SuggestFloat("x", -5, 5);
            double y = t.SuggestFloat("y", -5, 5);
            return -(Math.Pow(x - 1.5, 2) + Math.Pow(y + 2.0, 2));
        }, nTrials: 26);

        AssertFullStateParity(csStudy,
            "import warnings; warnings.simplefilter('ignore')\n" +
            "study = optuna.create_study(study_name='pycma', direction='maximize', sampler=optuna.integration.PyCmaSampler(seed=21))\n" +
            "def _obj(t):\n" +
            "    x = t.suggest_float('x', -5, 5)\n" +
            "    y = t.suggest_float('y', -5, 5)\n" +
            "    return -((x - 1.5) ** 2 + (y + 2.0) ** 2)\n" +
            "study.optimize(_obj, n_trials=26)\n");
    }

    // ---- cma_stds (per-parameter sigma multipliers → the CMA_stds option) -----------------------

    [TestMethod]
    [TestCategory("RequiresPython")]
    public void CmaStds_FullState_IsBitIdentical()
    {
        RequirePyCmaOrInconclusive();

        var csStudy = Optuna.CreateStudy(
            storage: new InMemoryStorage(), studyName: "pycma",
            sampler: new PyCmaSampler(seed: 8, cmaStds: new Dictionary<string, double> { ["x"] = 2.0, ["y"] = 0.5 }));
        csStudy.Optimize(t =>
        {
            double x = t.SuggestFloat("x", -5, 5);
            double y = t.SuggestFloat("y", -5, 5);
            return Math.Pow(x - 1.5, 2) + Math.Pow(y + 2.0, 2);
        }, nTrials: 26);

        AssertFullStateParity(csStudy,
            "import warnings; warnings.simplefilter('ignore')\n" +
            "study = optuna.create_study(study_name='pycma', sampler=optuna.integration.PyCmaSampler(seed=8, cma_stds={'x': 2.0, 'y': 0.5}))\n" +
            "def _obj(t):\n" +
            "    x = t.suggest_float('x', -5, 5)\n" +
            "    y = t.suggest_float('y', -5, 5)\n" +
            "    return (x - 1.5) ** 2 + (y + 2.0) ** 2\n" +
            "study.optimize(_obj, n_trials=26)\n");
    }

    // ---- explicit popsize via cma_opts (a different generation boundary — the numpy-vs-list zone) --

    [TestMethod]
    [TestCategory("RequiresPython")]
    public void ExplicitPopsizeViaCmaOpts_FullState_IsBitIdentical()
    {
        RequirePyCmaOrInconclusive();

        // popsize=10 puts generation boundaries at 10/20/30 — well past the several-generation point
        // where the tell-argument-type (numpy vs list) divergence first surfaces.
        var csStudy = Optuna.CreateStudy(
            storage: new InMemoryStorage(), studyName: "pycma",
            sampler: new PyCmaSampler(seed: 4, cmaOpts: new Dictionary<string, object?> { ["popsize"] = 10 }));
        csStudy.Optimize(t =>
        {
            double x = t.SuggestFloat("x", -5, 5);
            double y = t.SuggestFloat("y", -5, 5);
            return Math.Pow(x - 1.5, 2) + Math.Pow(y + 2.0, 2);
        }, nTrials: 34);

        AssertFullStateParity(csStudy,
            "import warnings; warnings.simplefilter('ignore')\n" +
            "study = optuna.create_study(study_name='pycma', sampler=optuna.integration.PyCmaSampler(seed=4, cma_opts={'popsize': 10}))\n" +
            "def _obj(t):\n" +
            "    x = t.suggest_float('x', -5, 5)\n" +
            "    y = t.suggest_float('y', -5, 5)\n" +
            "    return (x - 1.5) ** 2 + (y + 2.0) ** 2\n" +
            "study.optimize(_obj, n_trials=34)\n");
    }

    // ---- explicit x0 + sigma0 (skips _initialize_x0 / _initialize_sigma0) -----------------------

    [TestMethod]
    [TestCategory("RequiresPython")]
    public void ExplicitX0AndSigma0_FullState_IsBitIdentical()
    {
        RequirePyCmaOrInconclusive();

        var csStudy = Optuna.CreateStudy(
            storage: new InMemoryStorage(), studyName: "pycma",
            sampler: new PyCmaSampler(
                seed: 6,
                x0: new Dictionary<string, object?> { ["x"] = 1.0, ["y"] = -1.0 },
                sigma0: 2.5));
        csStudy.Optimize(t =>
        {
            double x = t.SuggestFloat("x", -5, 5);
            double y = t.SuggestFloat("y", -5, 5);
            return Math.Pow(x - 1.5, 2) + Math.Pow(y + 2.0, 2);
        }, nTrials: 26);

        AssertFullStateParity(csStudy,
            "import warnings; warnings.simplefilter('ignore')\n" +
            "study = optuna.create_study(study_name='pycma', sampler=optuna.integration.PyCmaSampler(seed=6, x0={'x': 1.0, 'y': -1.0}, sigma0=2.5))\n" +
            "def _obj(t):\n" +
            "    x = t.suggest_float('x', -5, 5)\n" +
            "    y = t.suggest_float('y', -5, 5)\n" +
            "    return (x - 1.5) ** 2 + (y + 2.0) ** 2\n" +
            "study.optimize(_obj, n_trials=26)\n");
    }

    // ---- int(log) + float(step) + int(step): the remaining _to_cma / _to_optuna transform branches -

    [TestMethod]
    [TestCategory("RequiresPython")]
    public void IntLogAndSteppedParams_FullState_IsBitIdentical()
    {
        RequirePyCmaOrInconclusive();

        var csStudy = Optuna.CreateStudy(
            storage: new InMemoryStorage(), studyName: "pycma",
            sampler: new PyCmaSampler(seed: 19));
        csStudy.Optimize(t =>
        {
            long p = t.SuggestInt("p", 1, 100, log: true);
            double q = t.SuggestFloat("q", -10, 10, step: 0.5);
            long r = t.SuggestInt("r", -20, 20, step: 5);
            return Math.Pow(Math.Log(p) - 3.0, 2) + Math.Pow(q - 2.5, 2) + Math.Pow(r + 10, 2);
        }, nTrials: 26);

        AssertFullStateParity(csStudy,
            "import warnings; warnings.simplefilter('ignore')\n" +
            "import numpy as np\n" +
            "study = optuna.create_study(study_name='pycma', sampler=optuna.integration.PyCmaSampler(seed=19))\n" +
            "def _obj(t):\n" +
            "    p = t.suggest_int('p', 1, 100, log=True)\n" +
            "    q = t.suggest_float('q', -10, 10, step=0.5)\n" +
            "    r = t.suggest_int('r', -20, 20, step=5)\n" +
            "    return (np.log(p) - 3.0) ** 2 + (q - 2.5) ** 2 + (r + 10) ** 2\n" +
            "study.optimize(_obj, n_trials=26)\n");
    }

    // ---- 1-D search space: PyCmaSampler defers to the independent sampler (verbatim behavior) -----

    [TestMethod]
    [TestCategory("RequiresPython")]
    public void OneDimensional_FallsBackToIndependentSampler()
    {
        RequirePyCmaOrInconclusive();

        // A 1-D relative space triggers the "`PyCmaSampler` does not support optimization of 1-D search
        // space" path: sample_relative returns {} and every trial is drawn by the independent
        // RandomSampler — so the whole study is bit-exact via RandomSampler.
        var csStudy = Optuna.CreateStudy(
            storage: new InMemoryStorage(), studyName: "pycma",
            sampler: new PyCmaSampler(seed: 2));
        csStudy.Optimize(t => Math.Pow(t.SuggestFloat("x", -5, 5) - 1.5, 2), nTrials: 15);

        AssertFullStateParity(csStudy,
            "import warnings; warnings.simplefilter('ignore')\n" +
            "study = optuna.create_study(study_name='pycma', sampler=optuna.integration.PyCmaSampler(seed=2))\n" +
            "def _obj(t):\n" +
            "    return (t.suggest_float('x', -5, 5) - 1.5) ** 2\n" +
            "study.optimize(_obj, n_trials=15)\n");
    }

    // ---- multi-objective is unsupported — both sides raise the identical ValueError --------------

    [TestMethod]
    [TestCategory("RequiresPython")]
    public void MultiObjective_RaisesLikeOptuna()
    {
        RequirePyCmaOrInconclusive();

        const string expected =
            "If the study is being used for multi-objective optimization, PyCmaSampler cannot be used.";

        var csStudy = Optuna.CreateStudy(
            directions: new[] { StudyDirection.Minimize, StudyDirection.Minimize },
            sampler: new PyCmaSampler(seed: 1));
        Action act = () => csStudy.Optimize(t =>
        {
            double x = t.SuggestFloat("x", -5, 5);
            double y = t.SuggestFloat("y", -5, 5);
            return (IReadOnlyList<double>)new[] { x * x + y * y, (x - 2) * (x - 2) + y * y };
        }, nTrials: 3);
        act.Should().Throw<NumSharp.ValueError>().Which.Message.Should().Be(expected);

        // Cross-check: live optuna_integration.PyCmaSampler raises ValueError with the same message.
        PyExec(
            "import warnings; warnings.simplefilter('ignore')\n" +
            "_study = optuna.create_study(directions=['minimize', 'minimize'], sampler=optuna.integration.PyCmaSampler(seed=1))\n" +
            "def _mo(t):\n" +
            "    x = t.suggest_float('x', -5, 5)\n" +
            "    y = t.suggest_float('y', -5, 5)\n" +
            "    return [x * x + y * y, (x - 2) ** 2 + y * y]\n" +
            "_err = None\n" +
            "try:\n" +
            "    _study.optimize(_mo, n_trials=3)\n" +
            "except ValueError as _e:\n" +
            "    _err = str(_e)\n");
        PyStr("_err").Should().Be(expected);
    }

    // ---- C#-side determinism (same seed → identical trajectory) ---------------------------------

    [TestMethod]
    [TestCategory("RequiresPython")]
    public void SameSeed_IsReproducible()
    {
        RequirePyCmaOrInconclusive();

        List<double> Run()
        {
            var study = Optuna.CreateStudy(sampler: new PyCmaSampler(seed: 5));
            study.Optimize(t =>
            {
                double x = t.SuggestFloat("x", -5, 5);
                double y = t.SuggestFloat("y", -5, 5);
                return Math.Pow(x - 1.5, 2) + Math.Pow(y + 2.0, 2);
            }, nTrials: 20);
            var values = new List<double>();
            foreach (var t in study.Trials)
                values.Add(t.Value!.Value);
            return values;
        }

        Run().Should().Equal(Run());
    }
}

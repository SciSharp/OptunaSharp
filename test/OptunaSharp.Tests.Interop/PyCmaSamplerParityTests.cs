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
    /// <summary>PyCmaSampler's ctor imports pycma (and the Python side needs optuna_integration); probe
    /// both up front and self-skip if either is missing.</summary>
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

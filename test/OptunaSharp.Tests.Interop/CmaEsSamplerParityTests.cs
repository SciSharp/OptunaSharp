using System;
using System.Collections.Generic;
using System.Linq;
using AwesomeAssertions;
using OptunaSharp.Collections;
using OptunaSharp.Distributions;
using OptunaSharp.Internal;
using OptunaSharp.Samplers;
using OptunaSharp.Samplers.CmaEs;
using OptunaSharp.Storages;

namespace OptunaSharp.Tests.Interop;

/// <summary>
///     Full-state bit-parity of <see cref="CmaEsSampler" /> against live Optuna (5.0.0.dev + cmaes
///     0.13.1). Every trial's params (external + internal IEEE bits), value, distributions, and the
///     <c>cma:generation</c> attribution must agree byte-for-byte: the CMA optimizer is bit-exact
///     (MT19937 draws, <c>eigh</c> via OpenBLAS, numpy pairwise/sequential reductions, and the
///     upper-triangle covariance symmetry preserved across the per-trial optimizer round-trip), the
///     per-trial reseed <c>randint(1,2**16)+trial.number</c> matches, and the first trial is the
///     independent <see cref="RandomSampler" />.
///
///     <para>The stored optimizer states (<c>cma:optimizer:*</c>) are Optuna's pickle on both sides
///     (review item P3-2) and are compared decoded: byte-identical once the <c>_funhist_values</c>
///     entries cmaes leaves uninitialized (<c>np.empty</c>, so Python's own bytes there are leftover
///     memory) are taken from Python's state. The rest of the snapshot is then compared bit-for-bit,
///     <c>cma:generation</c> included. The tolerance-bounded with-margin test compares trajectories only.</para>
/// </summary>
[TestClass]
public class CmaEsSamplerParityTests : OptunaInteropTestBase
{
    /// <summary>Removes the stored optimizer states, which <see cref="AssertOptimizerStatesMatch" /> compares decoded.</summary>
    /// <param name="s">The snapshot to edit in place.</param>
    private static void StripOptimizerAttrs(StudySnapshot s)
    {
        foreach (var t in s.Trials)
            foreach (var key in t.SystemAttrs.Keys.Where(k => k.Contains(":optimizer:")).ToList())
                t.SystemAttrs.Remove(key);
    }

    /// <summary>
    ///     Runs the Python study and asserts full parity with the C# one: the stored optimizer states decoded and
    ///     byte-compared (<see cref="AssertOptimizerStatesMatch" />), then every other field bit for bit.
    /// </summary>
    /// <param name="csStudy">The finished OptunaSharp study.</param>
    /// <param name="pyCreateAndRun">Python that creates and runs the same study as <c>study</c>.</param>
    private void AssertParityExceptOptimizerBlob(Study csStudy, string pyCreateAndRun)
    {
        var csState = StudySnapshot.Capture(csStudy);
        PyExec(pyCreateAndRun);
        var pyState = PyStudyState("study");
        AssertOptimizerStatesMatch(pyState, csState);
        StripOptimizerAttrs(csState);
        StripOptimizerAttrs(pyState);
        StudySnapshot.AssertParity(pyState, csState, DatetimeComparison.NullnessOnly);
    }

    /// <summary>
    ///     Every trial stores an optimizer state in both languages or in neither, and each pair is the same pickle,
    ///     byte for byte, once the funhist entries cmaes has not written yet are taken from Python's state.
    /// </summary>
    /// <param name="py">Python's snapshot.</param>
    /// <param name="cs">OptunaSharp's snapshot.</param>
    private static void AssertOptimizerStatesMatch(StudySnapshot py, StudySnapshot cs)
    {
        int compared = 0;
        for (int i = 0; i < Math.Min(py.Trials.Count, cs.Trials.Count); i++)
        {
            string? pyHex = OptimizerState(py.Trials[i].SystemAttrs);
            string? csHex = OptimizerState(cs.Trials[i].SystemAttrs);
            (csHex is null).Should().Be(pyHex is null, $"trial {i} must store an optimizer state in both languages or neither");
            if (pyHex is null || csHex is null)
                continue;
            byte[] pyBytes = CmaPickle.FromHexString(pyHex);
            var pyOptimizer = (PyInstance)PythonPickle.Loads(pyBytes)!;
            var csOptimizer = (PyInstance)PythonPickle.Loads(CmaPickle.FromHexString(csHex))!;
            CopyUninitializedFunhist(csOptimizer, pyOptimizer);
            PythonPickle.Dumps(csOptimizer).Should().Equal(pyBytes, $"trial {i}'s stored optimizer state");
            compared++;
        }
        compared.Should().BePositive("the study must have stored optimizer states to compare");
    }

    /// <summary>A trial's stored optimizer state, its chunks concatenated, or null when it stored none.</summary>
    /// <param name="attrs">The trial's system attrs.</param>
    /// <returns>The hex text.</returns>
    private static string? OptimizerState(OrderedDict<string, object?> attrs)
    {
        var keys = attrs.Keys.Where(k => k.Contains(":optimizer:")).ToList();
        if (keys.Count == 0)
            return null;
        string stem = keys[0][..(keys[0].IndexOf(":optimizer:", StringComparison.Ordinal) + ":optimizer:".Length)];
        return string.Concat(Enumerable.Range(0, keys.Count).Select(j => (string)attrs[stem + j]!));
    }

    /// <summary>
    ///     Copies Python's values into the <c>_funhist_values</c> entries cmaes has not written yet: generation
    ///     <c>g</c> writes entries <c>2(g mod term)</c> and the next one, and the others still hold what
    ///     <c>np.empty</c> returned, which no other process can reproduce (OptunaSharp starts them at zero).
    /// </summary>
    /// <param name="cs">OptunaSharp's state, edited in place.</param>
    /// <param name="py">Python's state for the same trial.</param>
    private static void CopyUninitializedFunhist(PyInstance cs, PyInstance py)
    {
        if (cs.TryGet("_cma", out var inner) && inner is PyInstance csInner)
        {
            CopyUninitializedFunhist(csInner, (PyInstance)py.Get("_cma")!);
            return;
        }
        double[] csValues = ((NpArray)cs.Get("_funhist_values")!).Doubles!;
        double[] pyValues = ((NpArray)py.Get("_funhist_values")!).Doubles!;
        long generation = (long)cs.Get("_g")!, term = (long)cs.Get("_funhist_term")!;
        var written = new HashSet<long>();
        for (long g = 1; g <= generation; g++)
        {
            written.Add(2 * (g % term));
            written.Add(2 * (g % term) + 1);
        }
        for (int i = 0; i < csValues.Length && i < pyValues.Length; i++)
            if (!written.Contains(i))
                csValues[i] = pyValues[i];
    }

    [TestMethod]
    [TestCategory("RequiresPython")]
    [DataRow(2, 0, 30)]
    [DataRow(2, 42, 30)]
    [DataRow(3, 7, 28)]
    [DataRow(5, 13, 26)]
    public void FloatSphere_FullState_IsBitIdentical(int dim, int seed, int nTrials)
    {
        PythonSession.EnsureOrInconclusive();

        var csStudy = Optuna.CreateStudy(
            storage: new InMemoryStorage(), studyName: "cma",
            sampler: new CmaEsSampler(seed: seed));
        csStudy.Optimize(SphereCs(dim), nTrials: nTrials);

        AssertParityExceptOptimizerBlob(csStudy,
            $"study = optuna.create_study(study_name='cma', sampler=optuna.samplers.CmaEsSampler(seed={seed}))\n" +
            SpherePy(dim) +
            $"study.optimize(_obj, n_trials={nTrials})\n");
    }

    [TestMethod]
    [TestCategory("RequiresPython")]
    [DataRow(3, 2)]
    [DataRow(4, 9)]
    public void LrAdapt_FullState_IsBitIdentical(int dim, int seed)
    {
        PythonSession.EnsureOrInconclusive();

        var csStudy = Optuna.CreateStudy(
            storage: new InMemoryStorage(), studyName: "cma",
            sampler: new CmaEsSampler(seed: seed, lrAdapt: true));
        csStudy.Optimize(SphereCs(dim), nTrials: 32);

        AssertParityExceptOptimizerBlob(csStudy,
            $"study = optuna.create_study(study_name='cma', sampler=optuna.samplers.CmaEsSampler(seed={seed}, lr_adapt=True))\n" +
            SpherePy(dim) +
            "study.optimize(_obj, n_trials=32)\n");
    }

    [TestMethod]
    [TestCategory("RequiresPython")]
    [DataRow(10)]
    [DataRow(20)]
    public void ExplicitPopsize_FullState_IsBitIdentical(int popsize)
    {
        PythonSession.EnsureOrInconclusive();

        var csStudy = Optuna.CreateStudy(
            storage: new InMemoryStorage(), studyName: "cma",
            sampler: new CmaEsSampler(seed: popsize, popsize: popsize));
        csStudy.Optimize(SphereCs(3), nTrials: popsize * 2 + 6);

        AssertParityExceptOptimizerBlob(csStudy,
            $"study = optuna.create_study(study_name='cma', sampler=optuna.samplers.CmaEsSampler(seed={popsize}, popsize={popsize}))\n" +
            SpherePy(3) +
            $"study.optimize(_obj, n_trials={popsize * 2 + 6})\n");
    }

    [TestMethod]
    [TestCategory("RequiresPython")]
    [DataRow(2, 0)]
    [DataRow(3, 7)]
    [DataRow(5, 13)]
    public void SeparableCma_FullState_IsBitIdentical(int dim, int seed)
    {
        PythonSession.EnsureOrInconclusive();

        // Separable (diagonal-covariance) CMA-ES — no eigendecomposition; bit-exact end-to-end.
        var csStudy = Optuna.CreateStudy(
            storage: new InMemoryStorage(), studyName: "cma",
            sampler: new CmaEsSampler(seed: seed, useSeparableCma: true));
        csStudy.Optimize(SphereCs(dim), nTrials: 34);

        AssertParityExceptOptimizerBlob(csStudy,
            "import warnings; warnings.simplefilter('ignore')\n" +
            $"study = optuna.create_study(study_name='cma', sampler=optuna.samplers.CmaEsSampler(seed={seed}, use_separable_cma=True))\n" +
            SpherePy(dim) +
            "study.optimize(_obj, n_trials=34)\n");
    }

    [TestMethod]
    [TestCategory("RequiresPython")]
    [DataRow(3, 1, 10)]
    [DataRow(2, 7, 3)]
    [DataRow(5, 2, 8)]
    public void WarmStart_FullState_IsBitIdentical(int dim, int srcSeed, int tgtSeed)
    {
        PythonSession.EnsureOrInconclusive();

        // Warm-Starting: a source study (independent RandomSampler → bit-exact) primes the target
        // CMA-ES via get_warm_start_mgd. Both source and target trajectories are byte-identical.
        var src = Optuna.CreateStudy(sampler: new RandomSampler(srcSeed));
        src.Optimize(SphereCs(dim), nTrials: 30);
        var csStudy = Optuna.CreateStudy(
            storage: new InMemoryStorage(), studyName: "cma",
            sampler: new CmaEsSampler(seed: tgtSeed, sourceTrials: src.Trials.ToList()));
        csStudy.Optimize(SphereCs(dim), nTrials: 32);

        AssertParityExceptOptimizerBlob(csStudy,
            "import warnings; warnings.simplefilter('ignore')\n" +
            SpherePy(dim) +
            $"_src = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed={srcSeed}))\n" +
            "_src.optimize(_obj, n_trials=30)\n" +
            $"study = optuna.create_study(study_name='cma', sampler=optuna.samplers.CmaEsSampler(seed={tgtSeed}, source_trials=_src.trials))\n" +
            "study.optimize(_obj, n_trials=32)\n");
    }

    /// <summary>
    ///     Review item H1 (2026-10-26): infinite objective values. Optuna pickles the optimizer, so the ±inf that
    ///     cmaes records in its <c>funhist</c> is harmless there; OptunaSharp's JSON state used to throw on it and
    ///     fail every later ask. Trial 4 reports +inf and trial 9 −inf, inside the first two generations, and the
    ///     whole trajectory through and after them must stay bit-identical to live Optuna, in both directions
    ///     (maximize negates the values, so the infinities swap sign inside the optimizer).
    /// </summary>
    /// <param name="separable">Whether the separable (diagonal) CMA-ES variant is used.</param>
    /// <param name="maximize">Whether the study maximizes instead of minimizing.</param>
    [TestMethod]
    [TestCategory("RequiresPython")]
    [DataRow(false, false)]
    [DataRow(false, true)]
    [DataRow(true, false)]
    public void InfiniteObjectiveValues_FullState_IsBitIdentical(bool separable, bool maximize)
    {
        PythonSession.EnsureOrInconclusive();

        var sphere = SphereCs(3);
        var csStudy = Optuna.CreateStudy(
            storage: new InMemoryStorage(), studyName: "cma",
            sampler: new CmaEsSampler(seed: 3, useSeparableCma: separable),
            direction: maximize ? StudyDirection.Maximize : StudyDirection.Minimize);
        csStudy.Optimize(t =>
        {
            // Suggest first, so the infinite trials carry the same parameters as Python's.
            double v = sphere(t);
            return t.Number switch
            {
                4 => double.PositiveInfinity,
                9 => double.NegativeInfinity,
                _ => v,
            };
        }, nTrials: 36);

        AssertParityExceptOptimizerBlob(csStudy,
            "import warnings; warnings.simplefilter('ignore')\n" +
            $"study = optuna.create_study(study_name='cma', direction='{(maximize ? "maximize" : "minimize")}', " +
            $"sampler=optuna.samplers.CmaEsSampler(seed=3, use_separable_cma={(separable ? "True" : "False")}))\n" +
            SpherePy(3) +
            "def _obj_inf(t):\n" +
            "    v = _obj(t)\n" +
            "    if t.number == 4: return float('inf')\n" +
            "    if t.number == 9: return float('-inf')\n" +
            "    return v\n" +
            "study.optimize(_obj_inf, n_trials=36)\n");
    }

    [TestMethod]
    [TestCategory("RequiresPython")]
    public void MixedSpace_FullState_IsBitIdentical()
    {
        PythonSession.EnsureOrInconclusive();

        // float (relative CMA) + int (relative CMA) + categorical (independent RandomSampler).
        var csStudy = Optuna.CreateStudy(
            storage: new InMemoryStorage(), studyName: "cma",
            sampler: new CmaEsSampler(seed: 5));
        csStudy.Optimize(t =>
        {
            double a = t.SuggestFloat("a", -4, 4);
            long b = t.SuggestInt("bb", -5, 5);
            double c = t.SuggestFloat("c", 0.01, 100, log: true);
            t.SuggestCategorical("dd", new CategoricalChoice[] { "p", "q", "r" });
            return Math.Pow(a - 1.5, 2) + Math.Pow(b + 2, 2) + Math.Pow(Math.Log(c) - 1.0, 2);
        }, nTrials: 24);

        AssertParityExceptOptimizerBlob(csStudy,
            "study = optuna.create_study(study_name='cma', sampler=optuna.samplers.CmaEsSampler(seed=5))\n" +
            "import numpy as np\n" +
            "def _obj(t):\n" +
            "    a = t.suggest_float('a', -4, 4)\n" +
            "    b = t.suggest_int('bb', -5, 5)\n" +
            "    c = t.suggest_float('c', 0.01, 100, log=True)\n" +
            "    t.suggest_categorical('dd', ['p', 'q', 'r'])\n" +
            "    return (a - 1.5) ** 2 + (b + 2) ** 2 + (np.log(c) - 1.0) ** 2\n" +
            "study.optimize(_obj, n_trials=24)\n");
    }

    // ---- CMA-ES with Margin (with_margin → cmaes.CMAwM) ---------------------------------------

    private const double MarginTol = 1e-9;

    /// <summary>
    ///     Trajectory parity for <c>with_margin=True</c>. Unlike the bit-exact CMA/SepCMA/warm-start
    ///     paths, CMAwM's margin correction rides SciPy's <c>norm.cdf</c>/<c>chi2.ppf</c> (reproduced
    ///     managed via <c>Erf</c>/<c>TruncNorm</c> — the erf class), so parity is asserted to a small
    ///     tolerance. The whole study still lines up trial-for-trial (number, state, param set, and the
    ///     <c>cmawm:generation</c> attribution) because the raw draws come from the bit-exact CMA and
    ///     the discrete dims snap to an exact grid; in practice the observable params/values agree
    ///     bit-for-bit (the erf drift stays inside the internal mean/scaling), but the tolerance keeps
    ///     the gate honest about the documented divergence and robust to an encoding-boundary tie.
    /// </summary>
    private void AssertMarginTrajectoryParity(Study csStudy, string pyCreateAndRun)
    {
        var csState = StudySnapshot.Capture(csStudy);
        PyExec(pyCreateAndRun);
        var pyState = PyStudyState("study");

        csState.Trials.Count.Should().Be(pyState.Trials.Count, "trajectory length must match");
        for (int i = 0; i < pyState.Trials.Count; i++)
        {
            var e = pyState.Trials[i];
            var a = csState.Trials[i];
            a.Number.Should().Be(e.Number);
            a.State.Should().Be(e.State, $"trial {e.Number} state");
            a.Params.Keys.Should().BeEquivalentTo(e.Params.Keys, $"trial {e.Number} param set");

            foreach (var key in e.Params.Keys)
            {
                object? ev = e.Params[key];
                object? av = a.Params[key];
                if (ev is double ed && av is double ad)
                    ad.Should().BeApproximately(ed, Math.Max(MarginTol, MarginTol * Math.Abs(ed)),
                        $"trial {e.Number} param '{key}'");
                else
                    av.Should().Be(ev, $"trial {e.Number} param '{key}'");   // int y snaps to the exact grid
            }

            (a.Values is null).Should().Be(e.Values is null, $"trial {e.Number} values null-ness");
            if (e.Values is not null)
                for (int k = 0; k < e.Values.Count; k++)
                    a.Values![k].Should().BeApproximately(e.Values[k],
                        Math.Max(MarginTol, MarginTol * Math.Abs(e.Values[k])), $"trial {e.Number} values[{k}]");

            // Generation attribution is exact (integer), under the `cmawm:` prefix.
            e.SystemAttrs.TryGetValue("cmawm:generation", out var eg);
            a.SystemAttrs.TryGetValue("cmawm:generation", out var ag);
            (ag is null).Should().Be(eg is null, $"trial {e.Number} cmawm:generation presence");
            if (eg is not null)
                Convert.ToInt64(ag).Should().Be(Convert.ToInt64(eg), $"trial {e.Number} cmawm:generation");
        }
    }

    [TestMethod]
    [TestCategory("RequiresPython")]
    [DataRow(42, 16)]
    [DataRow(7, 20)]
    [DataRow(0, 24)]
    public void WithMargin_Trajectory_MatchesLiveOptuna(int seed, int nTrials)
    {
        PythonSession.EnsureOrInconclusive();

        // Mixed float (continuous CMA) + int (margin-corrected discrete) space — the setting
        // with_margin exists for: it keeps the discrete `y` from collapsing to one point.
        var csStudy = Optuna.CreateStudy(
            storage: new InMemoryStorage(), studyName: "cmawm",
            sampler: new CmaEsSampler(seed: seed, withMargin: true));
        csStudy.Optimize(t =>
        {
            double x = t.SuggestFloat("x", -5, 5);
            long y = t.SuggestInt("y", -5, 5);
            double z = t.SuggestFloat("z", -5, 5);
            return Math.Pow(x - 0.7, 2) + Math.Pow(y + 2, 2) + Math.Pow(z - 1.3, 2);
        }, nTrials: nTrials);

        AssertMarginTrajectoryParity(csStudy,
            "import warnings; warnings.simplefilter('ignore')\n" +
            $"study = optuna.create_study(study_name='cmawm', sampler=optuna.samplers.CmaEsSampler(seed={seed}, with_margin=True))\n" +
            "def _obj(t):\n" +
            "    x = t.suggest_float('x', -5, 5)\n" +
            "    y = t.suggest_int('y', -5, 5)\n" +
            "    z = t.suggest_float('z', -5, 5)\n" +
            "    return (x - 0.7) ** 2 + (y + 2) ** 2 + (z - 1.3) ** 2\n" +
            $"study.optimize(_obj, n_trials={nTrials})\n");
    }

    // Deterministic shifted sphere with an explicit left-to-right sum (NOT Python's builtin sum(),
    // which since 3.12 uses Neumaier compensated summation and would drift ~1 ULP from the C# sum).
    private static Func<Trial, double> SphereCs(int dim)
    {
        var sh = Enumerable.Range(0, dim).Select(i => (i - (dim - 1) * 0.5) * 0.4).ToArray();
        return t =>
        {
            double v = 0;
            for (int i = 0; i < dim; i++) v += Math.Pow(t.SuggestFloat($"x{i}", -5, 5) - sh[i], 2);
            return v;
        };
    }

    private static string SpherePy(int dim)
    {
        var sh = string.Join(", ", Enumerable.Range(0, dim).Select(i => ((i - (dim - 1) * 0.5) * 0.4).ToString("R")));
        return
            $"_sh = [{sh}]\n" +
            "def _obj(t):\n" +
            $"    _x = [t.suggest_float(f'x{{i}}', -5, 5) for i in range({dim})]\n" +
            "    _v = 0.0\n" +
            $"    for i in range({dim}): _v = _v + (_x[i] - _sh[i]) ** 2\n" +
            "    return _v\n";
    }
}

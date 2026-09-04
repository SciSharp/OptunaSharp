using System;
using OptunaSharp.Pruners;
using OptunaSharp.Samplers;
using OptunaSharp.Storages;

namespace OptunaSharp.Tests.Interop;

/// <summary>
///     Byte-for-byte parity for the <b>search spaces of the most-starred public Optuna gists</b>
///     (ranked by <c>Gist.stargazerCount</c>), ported verbatim under <c>examples/</c>. A gist's real
///     objective is a heavy external process (ffmpeg encode, XGBoost train), but a sampler's
///     trajectory is fixed by <c>(seed, search space, value sequence)</c> alone — so each real search
///     space is paired with a <b>deterministic surrogate</b> and a fixed seed, and OptunaSharp must
///     reproduce live Optuna's run exactly. This pins that OptunaSharp handles the exact
///     distributions, dynamic/conditional branches and dynamic parameter names these battle-tested
///     scripts use.
///
///     <para>Parity is asserted on the RNG-exact <see cref="RandomSampler" /> path (full-state,
///     bit-identical). TPE parity over these particular spaces is intentionally NOT asserted: both
///     carry an integer parameter with few distinct values, whose repeated observations hit the
///     documented NumPy x86-simd-sort tie-order divergence inside TPE's <c>compute_sigmas</c> (see the
///     remark on <c>ParzenEstimator</c> / <see cref="TpeSamplerParityTests" />).</para>
///
///     <para>The C# workloads here are the same ones the runnable <c>examples/*.cs</c> illustrate;
///     this class is the authoritative parity gate for them.</para>
/// </summary>
[TestClass]
public class GistExampleParityTests : OptunaInteropTestBase
{
    // ---------------------------------------------------------------- ffmpeg x264 (sile, 25*)

    // Search space of sile/8aa1ff7808dd55298f51dd70c8b83092 ("Optunaを使ってFFmpegのエンコードパラメータを
    // 最適化してみる", maintainer sile) — a conditional x264 space with two branches (psy-rd under
    // trellis>0; deblock alpha/beta under not-no-deblock) and dynamic categorical names
    // (partitions-*, feature flags). suggest_uniform (pre-3.0) is modernized to suggest_float — same
    // uniform-float semantics. The real SSIM objective + SuccessiveHalvingPruner are kept structurally
    // (report at the pruner's 100/300/900 rungs) with a deterministic surrogate SSIM.
    private static double FfmpegX264(Trial t)
    {
        t.SuggestInt("refs", 1, 16);
        double qcomp = t.SuggestFloat("qcomp", 0.0, 1.0);
        t.SuggestInt("b-adapt", 0, 2);
        t.SuggestInt("subme", 1, 9);
        long trellis = t.SuggestInt("trellis", 0, 2);
        if (trellis > 0)
        {
            t.SuggestFloat("psy-rd.0", 0.0, 1.0);
            t.SuggestFloat("psy-rd.1", 0.0, 1.0);
        }

        foreach (var p in new[] { "p8x8", "p4x4", "b8x8", "i8x8", "i4x4" })
            t.SuggestCategorical($"partitions-{p}", new[] { 0, 1 });

        t.SuggestCategorical("me", new[] { "dia", "hex", "umh", "esa" });
        t.SuggestInt("me_range", 4, 16);
        t.SuggestCategorical("direct", new[] { "none", "spatial", "temporal", "auto" });
        t.SuggestInt("rc-lookahead", 10, 100);
        t.SuggestInt("min-keyint", 1, 12);
        t.SuggestInt("scenecut", 10, 100);
        t.SuggestInt("qdiff", 1, 51);

        long noDeblock = t.SuggestCategorical("no-deblock", new[] { 0, 1 });
        if (noDeblock == 0)  // Python: `if not no_deblock:` -> the deblock alpha/beta branch
        {
            t.SuggestInt("deblockalpha", -6, 6);
            t.SuggestInt("deblockbeta", -6, 6);
        }

        foreach (var f in new[]
                 {
                     "b-pyramid", "no-cabac", "mixed-refs", "no-chroma-me", "8x8dct",
                     "no-fast-pskip", "no-dct-decimate", "deadzone-inter", "deadzone-intra", "nr",
                 })
            t.SuggestCategorical(f, new[] { 0, 1 });

        // Deterministic stand-in for the ffmpeg SSIM measurement, reported at the SHA pruner's rungs.
        foreach (int d in new[] { 100, 300, 900 })
        {
            t.Report(qcomp + 0.001 * d, d);
            if (t.ShouldPrune())
                throw new TrialPruned();
        }
        return qcomp;
    }

    private const string PyFfmpegX264 =
        "def _obj(t):\n" +
        "    t.suggest_int('refs', 1, 16)\n" +
        "    qcomp = t.suggest_float('qcomp', 0.0, 1.0)\n" +
        "    t.suggest_int('b-adapt', 0, 2)\n" +
        "    t.suggest_int('subme', 1, 9)\n" +
        "    trellis = t.suggest_int('trellis', 0, 2)\n" +
        "    if trellis > 0:\n" +
        "        t.suggest_float('psy-rd.0', 0.0, 1.0)\n" +
        "        t.suggest_float('psy-rd.1', 0.0, 1.0)\n" +
        "    for p in ['p8x8', 'p4x4', 'b8x8', 'i8x8', 'i4x4']:\n" +
        "        t.suggest_categorical('partitions-%s' % p, [0, 1])\n" +
        "    t.suggest_categorical('me', ['dia', 'hex', 'umh', 'esa'])\n" +
        "    t.suggest_int('me_range', 4, 16)\n" +
        "    t.suggest_categorical('direct', ['none', 'spatial', 'temporal', 'auto'])\n" +
        "    t.suggest_int('rc-lookahead', 10, 100)\n" +
        "    t.suggest_int('min-keyint', 1, 12)\n" +
        "    t.suggest_int('scenecut', 10, 100)\n" +
        "    t.suggest_int('qdiff', 1, 51)\n" +
        "    no_deblock = t.suggest_categorical('no-deblock', [0, 1])\n" +
        "    if not no_deblock:\n" +
        "        t.suggest_int('deblockalpha', -6, 6)\n" +
        "        t.suggest_int('deblockbeta', -6, 6)\n" +
        "    for f in ['b-pyramid', 'no-cabac', 'mixed-refs', 'no-chroma-me', '8x8dct', " +
        "'no-fast-pskip', 'no-dct-decimate', 'deadzone-inter', 'deadzone-intra', 'nr']:\n" +
        "        t.suggest_categorical(f, [0, 1])\n" +
        "    for d in [100, 300, 900]:\n" +
        "        t.report(qcomp + 0.001 * d, d)\n" +
        "        if t.should_prune():\n" +
        "            raise optuna.TrialPruned()\n" +
        "    return qcomp\n";

    [TestMethod]
    [TestCategory("RequiresPython")]
    [DataRow(0)]
    [DataRow(42)]
    public void FfmpegX264SearchSpace_RandomSampler_BitIdentical(int seed)
    {
        PythonSession.EnsureOrInconclusive();

        var cs = Optuna.CreateStudy(
            storage: new InMemoryStorage(), sampler: new RandomSampler(seed),
            pruner: new SuccessiveHalvingPruner(minResource: 100, reductionFactor: 3),
            direction: StudyDirection.Maximize, studyName: "ffmpeg");
        cs.Optimize(FfmpegX264, nTrials: 40);
        var csState = StudySnapshot.Capture(cs);

        PyExec(
            $"study = optuna.create_study(study_name='ffmpeg', direction='maximize', " +
            $"sampler=optuna.samplers.RandomSampler(seed={seed}), " +
            "pruner=optuna.pruners.SuccessiveHalvingPruner(min_resource=100, reduction_factor=3))\n" +
            PyFfmpegX264 +
            "study.optimize(_obj, n_trials=40)\n");

        StudySnapshot.AssertParity(PyStudyState("study"), csState, DatetimeComparison.NullnessOnly);
    }

    // ---------------------------------------------------------------- xgboost hpo (bexgboost, 16*)

    // Search space of bexgboost/823df08d2e3760538e9b931d38439a68 (the classic XGBoost HPO grid).
    // The XGBoostPruningCallback + xgb.train objective is replaced by a deterministic surrogate; only
    // the flat 8-parameter search space and direction='minimize' are the portable, battle-tested part.
    private static double XgboostHpo(Trial t)
    {
        double eta = t.SuggestFloat("eta", 0.01, 0.3);
        t.SuggestInt("max_depth", 3, 10);
        double subsample = t.SuggestFloat("subsample", 0.5, 1.0);
        t.SuggestFloat("colsample_bytree", 0.5, 1.0);
        t.SuggestFloat("gamma", 0.0, 10.0);
        t.SuggestFloat("min_child_weight", 0.1, 10.0);
        t.SuggestFloat("lambda", 0.1, 10.0);
        t.SuggestFloat("alpha", 0.0, 10.0);
        return Math.Pow(eta - 0.1, 2.0) + Math.Pow(subsample - 0.8, 2.0);  // surrogate RMSE
    }

    private const string PyXgboostHpo =
        "def _obj(t):\n" +
        "    eta = t.suggest_float('eta', 0.01, 0.3)\n" +
        "    t.suggest_int('max_depth', 3, 10)\n" +
        "    subsample = t.suggest_float('subsample', 0.5, 1.0)\n" +
        "    t.suggest_float('colsample_bytree', 0.5, 1.0)\n" +
        "    t.suggest_float('gamma', 0.0, 10.0)\n" +
        "    t.suggest_float('min_child_weight', 0.1, 10.0)\n" +
        "    t.suggest_float('lambda', 0.1, 10.0)\n" +
        "    t.suggest_float('alpha', 0.0, 10.0)\n" +
        "    return (eta - 0.1) ** 2 + (subsample - 0.8) ** 2\n";

    [TestMethod]
    [TestCategory("RequiresPython")]
    [DataRow(0)]
    [DataRow(42)]
    public void XgboostSearchSpace_RandomSampler_BitIdentical(int seed)
    {
        PythonSession.EnsureOrInconclusive();

        var cs = Optuna.CreateStudy(
            storage: new InMemoryStorage(), sampler: new RandomSampler(seed),
            direction: StudyDirection.Minimize, studyName: "xgb");
        cs.Optimize(XgboostHpo, nTrials: 25);
        var csState = StudySnapshot.Capture(cs);

        PyExec(
            $"study = optuna.create_study(study_name='xgb', direction='minimize', " +
            $"sampler=optuna.samplers.RandomSampler(seed={seed}))\n" +
            PyXgboostHpo +
            "study.optimize(_obj, n_trials=25)\n");

        StudySnapshot.AssertParity(PyStudyState("study"), csState, DatetimeComparison.NullnessOnly);
    }
}

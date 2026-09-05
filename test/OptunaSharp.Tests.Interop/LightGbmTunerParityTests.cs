using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using AwesomeAssertions;
using OptunaSharp.Integration.LightGbm;
using Python.Runtime;

namespace OptunaSharp.Tests.Interop;

/// <summary>
///     LIVE parity of the LightGBM stepwise tuner (<see cref="LightGBMTuner" /> /
///     <see cref="LightGBMTunerCV" />) against real Optuna's <c>optuna_integration.lightgbm</c>, run
///     in-process through pythonnet. This is the differential gate the porting plan describes (§4): the
///     committed-oracle test (<c>OptunaSharp.Tests.Integration.LightGbm</c>) replays a captured JSON, this
///     one re-derives the sequence from live Optuna every run.
///
///     <para>The training call is replaced — in BOTH languages — by the same deterministic synthetic score
///     of the accumulated LightGBM parameters, using only <c>+,-,*</c> so <c>synth()</c> itself is
///     bit-identical; the trial sequence then depends only on Optuna's samplers, which OptunaSharp ports.
///     Parity is asserted at the project's usual two grades:</para>
///     <list type="bullet">
///         <item><description><b>Byte-perfect (0 ULP)</b>: trial count, direction, every trial number /
///         state / step name, and the grid + integer parameters (<c>feature_fraction</c>,
///         <c>num_leaves</c>, <c>bagging_freq</c>, <c>min_child_samples</c>) — the tuner's whole decision
///         surface.</description></item>
///         <item><description><b>Tolerance</b>: the TPE-sampled continuous parameters
///         (<c>bagging_fraction</c> ~1 ULP; <c>lambda_l1</c>/<c>lambda_l2</c> up to ~64 ULP, amplified from
///         the erf boundary by their <c>log=True</c> transform) and the objective values (~6 ULP) — the
///         documented Core-wide erf/ucrt divergence (<c>ParzenEstimator</c> / truncnorm). Absolute
///         deviation is ~1e-19; it never flips a decision, so the best trial and best params still
///         track.</description></item>
///     </list>
///
///     <para>Self-skips (Inconclusive) when the Python engine, <c>lightgbm</c>, or
///     <c>optuna_integration.lightgbm</c> is not importable.</para>
/// </summary>
[TestClass]
public class LightGbmTunerParityTests : OptunaInteropTestBase
{
    // Grid- and integer-valued params: asserted 0-ULP bit-exact. Everything else the tuner suggests
    // (bagging_fraction, lambda_l1, lambda_l2) rides TPE's truncated-normal sampling through the managed
    // erf and is asserted to tolerance.
    private static readonly HashSet<string> BitExactParams =
        new() { "feature_fraction", "num_leaves", "bagging_freq", "min_child_samples" };

    private const string StepNameKey = "lightgbm_tuner:step_name";

    // ---- the deterministic synthetic trainer (C# side), bit-identical to the Python fake_train/fake_cv --

    private sealed class SynthTrainer : ILightGbmTrainer
    {
        public Booster Train(TrainRequest request)
        {
            double score = Synth(request.Params);
            string metric = ResolvedMetric(request.Params);
            var byName = new Dictionary<string, IReadOnlyDictionary<string, double>>();
            foreach (var name in request.ValidNames)
                byName[name] = new Dictionary<string, double> { [metric] = score };
            if (byName.Count == 0)
                byName["valid_0"] = new Dictionary<string, double> { [metric] = score };
            return new Booster(byName, request.NumBoostRound);
        }

        public CvResult Cv(CvRequest request)
        {
            double score = Synth(request.Params);
            string metric = ResolvedMetric(request.Params);
            return new CvResult(new Dictionary<string, IReadOnlyList<double>>
            {
                [$"{metric}-mean"] = new[] { score },
                [$"{metric}-stdv"] = new[] { 0.0 },
            });
        }
    }

    private static double Synth(IReadOnlyDictionary<string, object?> p)
    {
        double ff = D(p, "feature_fraction", 1.0);
        double nl = D(p, "num_leaves", 31);
        double bf = D(p, "bagging_fraction", 1.0);
        double bfreq = D(p, "bagging_freq", 0);
        double l1 = D(p, "lambda_l1", 0.0);
        double l2 = D(p, "lambda_l2", 0.0);
        double mcs = D(p, "min_child_samples", 20);
        return 1.0 * (ff - 0.75) * (ff - 0.75)
               + 0.0001 * (nl - 100.0) * (nl - 100.0)
               + 1.0 * (bf - 0.8) * (bf - 0.8)
               + 0.01 * (bfreq - 3.0) * (bfreq - 3.0)
               + 0.1 * l1 + 0.1 * l2
               + 0.0001 * (mcs - 15.0) * (mcs - 15.0);

        static double D(IReadOnlyDictionary<string, object?> d, string k, double fallback)
            => d.TryGetValue(k, out var v) && v is not null
                ? Convert.ToDouble(v, CultureInfo.InvariantCulture)
                : fallback;
    }

    private static string ResolvedMetric(IReadOnlyDictionary<string, object?> p)
        => p.TryGetValue("metric", out var m) && m is string s ? s : "binary_logloss";

    private static Dataset Dummy(int rows) => new(new double[rows, 3], new double[rows]);

    private static Dictionary<string, object?> RegressionParams() => new()
    {
        ["objective"] = "regression",
        ["metric"] = "l2",
        ["verbosity"] = -1L,
    };

    // ---- the Python side: monkeypatch lightgbm.train/cv inside the tuner, run it, expose `tuner` ---------

    // Bit-identical synth() + fake train/cv, then the tuner is bound as `tuner` in the scope.
    private const string PythonTunerSetup = """
        import types, numpy as np
        import optuna_integration.lightgbm as lgb
        from optuna_integration.lightgbm._lightgbm_tuner import optimize as _opt

        def _synth(params):
            ff = params.get("feature_fraction", 1.0)
            nl = float(params.get("num_leaves", 31))
            bf = params.get("bagging_fraction", 1.0)
            bfreq = float(params.get("bagging_freq", 0))
            l1 = params.get("lambda_l1", 0.0)
            l2 = params.get("lambda_l2", 0.0)
            mcs = float(params.get("min_child_samples", 20))
            return (1.0 * (ff - 0.75) * (ff - 0.75)
                    + 0.0001 * (nl - 100.0) * (nl - 100.0)
                    + 1.0 * (bf - 0.8) * (bf - 0.8)
                    + 0.01 * (bfreq - 3.0) * (bfreq - 3.0)
                    + 0.1 * l1 + 0.1 * l2
                    + 0.0001 * (mcs - 15.0) * (mcs - 15.0))

        def _metric(params):
            metric = params.get("metric", "binary_logloss")
            return metric if isinstance(metric, str) else list(metric)[-1]

        def _fake_train(params, train_set, **kwargs):
            b = types.SimpleNamespace()
            b.best_score = {"valid_0": {_metric(params): _synth(params)}}
            b.current_iteration = lambda: kwargs.get("num_boost_round", 20)
            return b

        def _fake_cv(params, train_set, **kwargs):
            m = _metric(params)
            return {m + "-mean": [_synth(params)], m + "-stdv": [0.0]}

        _opt.lgb.train = _fake_train
        _opt.lgb.cv = _fake_cv

        def _run_tuner(cv):
            dtrain = lgb.Dataset(np.zeros((10, 3)), label=np.zeros(10))
            params = {"objective": "regression", "metric": "l2", "verbosity": -1}
            if cv:
                t = lgb.LightGBMTunerCV(params, dtrain, num_boost_round=20, optuna_seed=42, show_progress_bar=False)
            else:
                dval = lgb.Dataset(np.zeros((4, 3)), label=np.zeros(4))
                t = lgb.LightGBMTuner(params, dtrain, valid_sets=[dval], num_boost_round=20, optuna_seed=42, show_progress_bar=False)
            t.run()
            return t
        """;

    private void RequireTunerOrInconclusive()
    {
        using (Gil())
        {
            try
            {
                Scope.Exec("import lightgbm\nimport optuna_integration.lightgbm\n");
            }
            catch (PythonException e)
            {
                Assert.Inconclusive(
                    $"optuna_integration.lightgbm / lightgbm not importable in the interop Python: {e.Message}");
            }
        }
    }

    [TestMethod]
    [TestCategory("RequiresPython")]
    public void LightGBMTuner_MatchesLiveOptuna()
    {
        RequireTunerOrInconclusive();

        PyExec(PythonTunerSetup);
        PyExec("tuner = _run_tuner(cv=False)");
        var py = PyStudyState("tuner.study");
        double pyBestScore = PyFloat("tuner.best_score");
        long pyBestNumber = PyLong("tuner.study.best_trial.number");

        var tuner = new LightGBMTuner(
            RegressionParams(), Dummy(10), validSets: new[] { Dummy(4) },
            numBoostRound: 20, optunaSeed: 42, trainer: new SynthTrainer());
        tuner.Run();

        AssertTunerParity(py, tuner, pyBestScore, pyBestNumber, "train");
    }

    [TestMethod]
    [TestCategory("RequiresPython")]
    public void LightGBMTunerCV_MatchesLiveOptuna()
    {
        RequireTunerOrInconclusive();

        PyExec(PythonTunerSetup);
        PyExec("tuner = _run_tuner(cv=True)");
        var py = PyStudyState("tuner.study");
        double pyBestScore = PyFloat("tuner.best_score");
        long pyBestNumber = PyLong("tuner.study.best_trial.number");

        var tuner = new LightGBMTunerCV(
            RegressionParams(), Dummy(10),
            numBoostRound: 20, optunaSeed: 42, trainer: new SynthTrainer());
        tuner.Run();

        AssertTunerParity(py, tuner, pyBestScore, pyBestNumber, "cv");
    }

    private static void AssertTunerParity(
        StudySnapshot py, LightGbmBaseTuner tuner, double pyBestScore, long pyBestNumber, string ctx)
    {
        var cs = StudySnapshot.Capture(tuner.Study);

        cs.Directions.Should().Equal(py.Directions, $"{ctx}: study direction is inferred from the metric");
        cs.Trials.Count.Should().Be(py.Trials.Count, $"{ctx}: the tuner runs 7+20+10+6+20+5 = 68 trials");

        var pyTrials = py.Trials.OrderBy(t => t.Number).ToList();
        var csTrials = cs.Trials.OrderBy(t => t.Number).ToList();

        for (int i = 0; i < pyTrials.Count; i++)
        {
            var e = pyTrials[i];
            var a = csTrials[i];
            string tctx = $"{ctx} trial #{e.Number}";

            a.Number.Should().Be(e.Number, $"{tctx} number");
            a.State.Should().Be(e.State, $"{tctx} state");
            StepName(a).Should().Be(StepName(e), $"{tctx} step name");

            a.Params.Count.Should().Be(e.Params.Count, $"{tctx} param count");
            foreach (var (name, expected) in e.Params)
            {
                a.Params.Should().ContainKey(name, $"{tctx} has param {name}");
                AssertParam(name, a.Params[name], expected, $"{tctx} param {name}");
            }

            // Single-objective; the value = synth(params), inheriting the erf tolerance of the TPE floats.
            e.Values.Should().NotBeNull($"{tctx} python value present");
            a.Values.Should().NotBeNull($"{tctx} C# value present");
            AssertClose(a.Values![0], e.Values![0], $"{tctx} objective value");
        }

        AssertClose(tuner.BestScore, pyBestScore, $"{ctx} best_score");
        tuner.Study.BestTrial.Number.Should().Be((int)pyBestNumber, $"{ctx} best trial number");
    }

    private static string? StepName(TrialSnapshot t)
        => t.SystemAttrs.TryGetValue(StepNameKey, out var v) ? v as string : null;

    private static void AssertParam(string name, object? actual, object? expected, string ctx)
    {
        double a = Convert.ToDouble(actual, CultureInfo.InvariantCulture);
        double e = Convert.ToDouble(expected, CultureInfo.InvariantCulture);
        if (BitExactParams.Contains(name))
            a.Should().Be(e, $"{ctx} (bit-exact grid/int)");
        else
            AssertClose(a, e, ctx);
    }

    // numpy-allclose style: |a-b| <= atol + rtol*|b|, atol=rtol=1e-9 (observed absolute deviation ~1e-19).
    private static void AssertClose(double actual, double expected, string ctx)
        => Math.Abs(actual - expected).Should()
            .BeLessThanOrEqualTo(1e-9 + 1e-9 * Math.Abs(expected), $"{ctx} (erf tolerance)");
}

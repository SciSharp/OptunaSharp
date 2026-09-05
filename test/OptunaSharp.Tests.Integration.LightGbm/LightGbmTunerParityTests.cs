using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using AwesomeAssertions;
using OptunaSharp.Integration.LightGbm;

namespace OptunaSharp.Tests.Integration.LightGbm;

/// <summary>
///     Parity of the LightGBM tuner's <b>stepwise search</b> against live Optuna's
///     <c>optuna_integration.lightgbm</c>. The training call is replaced (in BOTH languages) by a
///     deterministic synthetic score of the accumulated LightGBM parameters that uses only <c>+,-,*</c>
///     (so it is bit-identical across Python and C#). Because the whole search rides OptunaSharp's
///     bit-exact <c>GridSampler</c> / <c>TPESampler</c>, the C# tuner reproduces Optuna's exact trial
///     sequence — for both <see cref="LightGBMTuner" /> and <see cref="LightGBMTunerCV" />.
///
///     <para>Parity is asserted at two grades, the project's usual "bit-exact vs erf-tolerance" split.
///     <b>Bit-exact</b>: the trial count (7+20+10+6+20+5), every step name, every parameter key set, and
///     the grid- and integer-valued parameters (<c>feature_fraction</c>, <c>num_leaves</c>,
///     <c>bagging_freq</c>, <c>min_child_samples</c>). <b>Tolerance</b> (~1e-9): the TPE-sampled continuous
///     parameters (<c>bagging_fraction</c>, <c>lambda_l1</c>, <c>lambda_l2</c>) and the objective values,
///     which ride TPE's truncated-normal sampling through the managed erf/ndtri (the documented erf/ucrt
///     boundary). Measured deviation vs the oracle: ~1 ULP for <c>bagging_fraction</c> (linear scale) but
///     up to ~64 ULP for <c>lambda_l1</c>/<c>lambda_l2</c> (the erf error amplified by their
///     <c>log=True</c> transform through <c>exp</c>) and ~6 ULP for the objective values — absolute
///     deviation ~1e-19, never flipping a decision, so the whole sequence, the best trial, and the best
///     params all track.</para>
///
///     <para>The oracle (<c>oracle/lightgbm_tuner_synth.json</c>) is captured from live Optuna 5.x +
///     optuna-integration 5.x by <c>oracle/gen_lightgbm_tuner_oracle.py</c> — the committed-oracle model
///     the porting plan describes (§4). This is the differential gate for the Phase 6 tuner's search
///     subsystem; the ML.NET training adapter (which has no oracle) is covered separately. A LIVE
///     counterpart that re-derives the sequence from real Optuna in-process every run (pythonnet) is
///     <c>OptunaSharp.Tests.Interop.LightGbmTunerParityTests</c>.</para>
/// </summary>
[TestClass]
public class LightGbmTunerParityTests
{
    /// <summary>
    ///     The deterministic synthetic trainer — a stand-in for <c>lightgbm.train</c>/<c>lightgbm.cv</c>.
    ///     Its <c>best_score</c> is <see cref="Synth" /> of the accumulated params, matching the Python
    ///     oracle's <c>fake_train</c>/<c>fake_cv</c> exactly.
    /// </summary>
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

    // Bit-identical to synth() in gen_lightgbm_tuner_oracle.py: only +,-,* (no pow/log), so the double
    // result matches Python bit-for-bit.
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
                ? Convert.ToDouble(v, System.Globalization.CultureInfo.InvariantCulture)
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

    [TestMethod]
    public void LightGBMTuner_ReproducesOptunaSequence()
    {
        var tuner = new LightGBMTuner(
            RegressionParams(), Dummy(10), validSets: new[] { Dummy(4) },
            numBoostRound: 20, optunaSeed: 42, trainer: new SynthTrainer());
        tuner.Run();

        AssertMatchesOracle(tuner, "train");
    }

    [TestMethod]
    public void LightGBMTunerCV_ReproducesOptunaSequence()
    {
        var tuner = new LightGBMTunerCV(
            RegressionParams(), Dummy(10),
            numBoostRound: 20, optunaSeed: 42, trainer: new SynthTrainer());
        tuner.Run();

        AssertMatchesOracle(tuner, "cv");
    }

    // Grid- and integer-valued parameters — deterministic, asserted bit-exact. The remaining tuned
    // parameters (bagging_fraction, lambda_l1, lambda_l2) ride TPE's truncated-normal sampling through
    // the managed erf and are asserted to the erf tolerance.
    private static readonly HashSet<string> BitExactParams =
        new() { "feature_fraction", "num_leaves", "bagging_freq", "min_child_samples", "verbosity" };

    private static void AssertMatchesOracle(LightGbmBaseTuner tuner, string key)
    {
        var oracle = LoadOracle().GetProperty(key);

        ((int)tuner.Study.Direction).Should().Be(oracle.GetProperty("direction").GetInt32(),
            "the study direction is inferred from the metric");

        var trials = tuner.Study.Trials.OrderBy(t => t.Number).ToList();
        var oracleTrials = oracle.GetProperty("trials");
        trials.Count.Should().Be(oracleTrials.GetArrayLength(), "the tuner runs 7+20+10+6+20+5 = 68 trials");

        for (int i = 0; i < trials.Count; i++)
        {
            var t = trials[i];
            var ot = oracleTrials[i];
            string ctx = $"{key} trial #{ot.GetProperty("number").GetInt32()}";

            t.Number.Should().Be(ot.GetProperty("number").GetInt32(), ctx);

            string oracleStep = ot.GetProperty("step").GetString()!;
            (t.SystemAttrs.TryGetValue("lightgbm_tuner:step_name", out var step) ? step as string : null)
                .Should().Be(oracleStep, $"{ctx} step name");

            var oracleParams = ot.GetProperty("params");
            t.Params.Count.Should().Be(oracleParams.EnumerateObject().Count(), $"{ctx} param count");
            foreach (var op in oracleParams.EnumerateObject())
            {
                t.Params.Should().ContainKey(op.Name, $"{ctx} has param {op.Name}");
                AssertNumeric(t.Params[op.Name], op.Value.GetDouble(), op.Name, $"{ctx} param {op.Name}");
            }

            // The objective value = synth(params); it inherits the erf tolerance of the TPE floats.
            AssertClose(t.Value!.Value, ot.GetProperty("value").GetDouble(), $"{ctx} objective value");
        }

        AssertClose(tuner.BestScore, oracle.GetProperty("best_score").GetDouble(), $"{key} best_score");

        var best = tuner.BestParams;
        var oracleBest = oracle.GetProperty("best_params");
        best.Count.Should().Be(oracleBest.EnumerateObject().Count(), $"{key} best_params key count");
        foreach (var ob in oracleBest.EnumerateObject())
        {
            best.Should().ContainKey(ob.Name, $"{key} best_params has {ob.Name}");
            object? cs = best[ob.Name];
            switch (ob.Value.ValueKind)
            {
                case JsonValueKind.String:
                    (cs as string).Should().Be(ob.Value.GetString(), $"{key} best_params[{ob.Name}]");
                    break;
                case JsonValueKind.True or JsonValueKind.False:
                    cs.Should().Be(ob.Value.GetBoolean(), $"{key} best_params[{ob.Name}]");
                    break;
                default:
                    AssertNumeric(cs, ob.Value.GetDouble(), ob.Name, $"{key} best_params[{ob.Name}]");
                    break;
            }
        }
    }

    // Bit-exact for grid/integer params; erf-tolerance for the TPE-sampled continuous params.
    private static void AssertNumeric(object? actual, double expected, string paramName, string ctx)
    {
        double a = Convert.ToDouble(actual, System.Globalization.CultureInfo.InvariantCulture);
        if (BitExactParams.Contains(paramName))
            a.Should().Be(expected, $"{ctx} (bit-exact grid/int)");
        else
            AssertClose(a, expected, ctx);
    }

    // numpy-allclose style: |a-b| <= atol + rtol*|b|, atol=rtol=1e-9 (observed absolute deviation ~1e-19;
    // up to ~64 ULP on the log-scale lambda params, ~6 ULP on the objective values).
    private static void AssertClose(double actual, double expected, string ctx)
        => Math.Abs(actual - expected).Should()
            .BeLessThanOrEqualTo(1e-9 + 1e-9 * Math.Abs(expected), $"{ctx} (erf tolerance)");

    private static JsonElement LoadOracle()
    {
        string path = Path.Combine(AppContext.BaseDirectory, "oracle", "lightgbm_tuner_synth.json");
        File.Exists(path).Should().BeTrue($"the oracle must be copied to the output at {path}");
        using var doc = JsonDocument.Parse(File.ReadAllText(path));
        return doc.RootElement.Clone();
    }
}

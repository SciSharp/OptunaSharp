using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.Json;
using AwesomeAssertions;
using OptunaSharp.Internal;
using OptunaSharp.Samplers;
using OptunaSharp.Samplers.CmaEs;

namespace OptunaSharp.Tests.Samplers;

/// <summary>
///     <see cref="CmaEsSampler" /> tests. The hardcoded trajectory below is the literal output of
///     <c>optuna.samplers.CmaEsSampler(seed=42)</c> (Optuna 5.0.0.dev + cmaes 0.13.0) — OptunaSharp
///     reproduces it bit-for-bit because the whole pipeline (MT19937 draws, the CMA <c>eigh</c> via
///     OpenBLAS, the pairwise/sequential numpy reductions, and the covariance symmetry preserved
///     across the optimizer-state round-trip) is bit-exact. Live full-state parity lives in the
///     interop suite; these run without Python.
/// </summary>
[TestClass]
public class CmaEsSamplerTests
{
    private static double FromHex(string h) => BitConverter.ToDouble(Convert.FromHexString(h));

    // seed=42, dim2 sphere shift (0.7,-1.3), 10 trials, from live Optuna 5.0.0.dev + cmaes 0.13.0.
    private static readonly (string X0, string X1, string Value)[] Seed42 =
    {
        ("32b31134d612f4bf", "e8cbea8250071240", "8a2b710d8dc54240"),  // t0 gen=None (RandomSampler startup)
        ("90ff5f606706d4bf", "6a50cb40e761f5bf", "00a17fd2b56ff03f"),  // t1 gen=0
        ("00903529e5d2babf", "7caad78f542ff9bf", "38397135ff20e73f"),  // t2 gen=0
        ("fa634b79aef00040", "e0dd551cf269e13f", "f4421f1939a41540"),  // t3 gen=0
        ("125e4522046913c0", "e0066fd8426fca3f", "35b7872adc8c4040"),  // t4 gen=0
        ("c431eb19d27ff73f", "ccd16507a7f400c0", "b43d8719e232f43f"),  // t5 gen=0
        ("24841974498bfd3f", "984ec14c285efbbf", "9ee3603344baf73f"),  // t6 gen=0
        ("6c159d5f9cbbf3bf", "581d3723dd070dc0", "f98cfb2283522240"),  // t7 gen=1
        ("805a5307d6f9c0bf", "3453d63fd386f1bf", "22c334e61586e73f"),  // t8 gen=1
        ("50586a8118a4e33f", "20f43f0f320ef8bf", "20b156228100a93f"),  // t9 gen=1
    };

    private static double Sphere(Trial t)
    {
        double[] sh = { 0.7, -1.3 };
        double x0 = t.SuggestFloat("x0", -5, 5);
        double x1 = t.SuggestFloat("x1", -5, 5);
        return Math.Pow(x0 - sh[0], 2) + Math.Pow(x1 - sh[1], 2);
    }

    [TestMethod]
    public void Seed42_Trajectory_IsBitIdenticalToLiveOptuna()
    {
        var study = Optuna.CreateStudy(sampler: new CmaEsSampler(seed: 42));
        study.Optimize(Sphere, nTrials: 10);

        var trials = study.Trials.OrderBy(t => t.Number).ToList();
        trials.Should().HaveCount(10);
        for (int i = 0; i < Seed42.Length; i++)
        {
            ((double)trials[i].Params["x0"]!).Should().Be(FromHex(Seed42[i].X0), $"x0 of trial {i}");
            ((double)trials[i].Params["x1"]!).Should().Be(FromHex(Seed42[i].X1), $"x1 of trial {i}");
            trials[i].Value!.Value.Should().Be(FromHex(Seed42[i].Value), $"value of trial {i}");
        }
    }

    [TestMethod]
    public void SameSeed_IsReproducible()
    {
        List<(object?, object?)> Run()
        {
            var study = Optuna.CreateStudy(sampler: new CmaEsSampler(seed: 7));
            study.Optimize(Sphere, nTrials: 25);
            return study.Trials.OrderBy(t => t.Number)
                .Select(t => (t.Params["x0"], t.Params["x1"])).ToList();
        }

        Run().Should().Equal(Run());
    }

    [TestMethod]
    public void Optimize_MinimizesTheQuadratic()
    {
        var study = Optuna.CreateStudy(sampler: new CmaEsSampler(seed: 1));
        study.Optimize(Sphere, nTrials: 80);
        // The global minimum of the shifted sphere is 0; CMA-ES drives well below 1e-3.
        study.BestValue.Should().BeLessThan(1e-3);
    }

    [TestMethod]
    public void Generation_SystemAttr_IsAttributedPerTrial()
    {
        var study = Optuna.CreateStudy(sampler: new CmaEsSampler(seed: 42));
        study.Optimize(Sphere, nTrials: 10);
        var trials = study.Trials.OrderBy(t => t.Number).ToList();
        // Trial 0 is the independent-sampler startup (no generation attr); popsize is 6 for dim 2,
        // so trials 1-6 are generation 0 and 7-9 are generation 1.
        trials[0].SystemAttrs.ContainsKey("cma:generation").Should().BeFalse();
        Convert.ToInt64(trials[1].SystemAttrs["cma:generation"]).Should().Be(0);
        Convert.ToInt64(trials[7].SystemAttrs["cma:generation"]).Should().Be(1);
    }

    [TestMethod]
    public void MultiObjectiveStudy_Throws()
    {
        var study = Optuna.CreateStudy(
            sampler: new CmaEsSampler(seed: 0),
            directions: new[] { StudyDirection.Minimize, StudyDirection.Minimize });
        Action act = () => study.Optimize(
            t => new[] { t.SuggestFloat("x", -5, 5), t.SuggestFloat("y", -5, 5) }, nTrials: 5);
        act.Should().Throw<NumSharp.ValueError>();
    }

    // ---- construction validation (verbatim Optuna ValueErrors) --------------------------------

    [TestMethod]
    public void SourceTrialsWithX0_ThrowsValueError()
    {
        Action act = () => new CmaEsSampler(
            x0: new Dictionary<string, object?> { ["x"] = 0.0 }, sourceTrials: new List<FrozenTrial>());
        act.Should().Throw<NumSharp.ValueError>()
            .WithMessage("*prohibited to pass `source_trials` argument when x0 or sigma0*");
    }

    [TestMethod]
    public void SourceTrialsWithSeparable_ThrowsValueError()
    {
        Action act = () => new CmaEsSampler(useSeparableCma: true, sourceTrials: new List<FrozenTrial>());
        act.Should().Throw<NumSharp.ValueError>()
            .WithMessage("*prohibited to pass `source_trials` argument when using separable CMA-ES*");
    }

    [TestMethod]
    public void LrAdaptWithSeparable_ThrowsValueError()
    {
        Action act = () => new CmaEsSampler(useSeparableCma: true, lrAdapt: true);
        act.Should().Throw<NumSharp.ValueError>()
            .WithMessage("*prohibited to pass `use_separable_cma` or `with_margin`*");
    }

    [TestMethod]
    public void SeparableAndMargin_ThrowsValueError()
    {
        Action act = () => new CmaEsSampler(useSeparableCma: true, withMargin: true);
        act.Should().Throw<NumSharp.ValueError>()
            .WithMessage("*do not support `use_separable_cma=True` and `with_margin=True`*");
    }

    // ---- experimental backends not yet ported (deferred; guard throws NotSupported) -----------

    [TestMethod]
    public void UseSeparableCma_AtSamplingTime_ThrowsNotSupported()
    {
        var study = Optuna.CreateStudy(sampler: new CmaEsSampler(seed: 0, useSeparableCma: true));
        Action act = () => study.Optimize(t => Sphere(t), nTrials: 5);
        act.Should().Throw<NotSupportedException>().WithMessage("*Separable CMA-ES*");
    }

    [TestMethod]
    public void WithMargin_AtSamplingTime_ThrowsNotSupported()
    {
        var study = Optuna.CreateStudy(sampler: new CmaEsSampler(seed: 0, withMargin: true));
        Action act = () => study.Optimize(
            t => Math.Pow(t.SuggestFloat("x", -5, 5), 2) + t.SuggestInt("y", -5, 5), nTrials: 5);
        act.Should().Throw<NotSupportedException>().WithMessage("*margin*");
    }

    // ---- warnings -----------------------------------------------------------------------------

    [TestMethod]
    [DataRow("lr_adapt")]
    [DataRow("use_separable_cma")]
    [DataRow("with_margin")]
    [DataRow("consider_pruned_trials")]
    public void ExperimentalArgument_EmitsExperimentalWarning(string which)
    {
        var messages = new List<(string, WarningCategory)>();
        void Handler(string m, WarningCategory c) => messages.Add((m, c));
        OptunaWarnings.WarningRaised += Handler;
        try
        {
            _ = which switch
            {
                "lr_adapt" => new CmaEsSampler(lrAdapt: true),
                "use_separable_cma" => new CmaEsSampler(useSeparableCma: true),
                "with_margin" => new CmaEsSampler(withMargin: true),
                _ => new CmaEsSampler(considerPrunedTrials: true),
            };
        }
        finally { OptunaWarnings.WarningRaised -= Handler; }

        messages.Should().Contain(m =>
            m.Item2 == WarningCategory.ExperimentalWarning && m.Item1.Contains("experimental feature"));
    }

    [TestMethod]
    [DataRow("x0")]
    [DataRow("sigma0")]
    public void DeprecatedArgument_EmitsFutureWarning(string which)
    {
        var messages = new List<(string, WarningCategory)>();
        void Handler(string m, WarningCategory c) => messages.Add((m, c));
        OptunaWarnings.WarningRaised += Handler;
        try
        {
            _ = which == "x0"
                ? new CmaEsSampler(x0: new Dictionary<string, object?> { ["x"] = 0.0 })
                : new CmaEsSampler(sigma0: 0.5);
        }
        finally { OptunaWarnings.WarningRaised -= Handler; }

        messages.Should().Contain(m =>
            m.Item2 == WarningCategory.FutureWarning && m.Item1.Contains("deprecated"));
    }

    // ---- optimizer-state serialization (the pickle analogue; [Misaligned] own JSON) -----------

    [TestMethod]
    public void CmaState_JsonRoundTrip_IsIdempotentAndDeterministic()
    {
        static double Obj(double[] x) { double s = 0; foreach (var xi in x) s += Math.Pow(xi - 0.3, 2); return s; }
        static void Advance(Cma c, int seed)
        {
            c.ReseedRng(seed);
            var sols = new List<(double[], double)>();
            for (int k = 0; k < c.PopulationSize; k++) { var x = c.Ask(); sols.Add((x, Obj(x))); }
            c.Tell(sols);
        }

        var bounds = new double[4, 2];
        for (int i = 0; i < 4; i++) { bounds[i, 0] = 0; bounds[i, 1] = 1; }
        var mean = new[] { 0.5, 0.5, 0.5, 0.5 };

        var opt = new Cma(mean, 1.0 / 6, bounds: bounds, nMaxResampling: 40, seed: 9, populationSize: null, lrAdapt: false);
        for (int g = 0; g < 3; g++) Advance(opt, 100 + g);
        string json0 = JsonSerializer.Serialize(opt.ToState());

        // Idempotent: restoring then re-serializing (no evolution) reproduces the stored form —
        // the covariance is already stored symmetric-from-upper, mirroring Optuna's pickle so a
        // study can restore/re-store across trials without drift.
        string jsonIdem = JsonSerializer.Serialize(Cma.FromState(JsonSerializer.Deserialize<CmaState>(json0)!).ToState());
        jsonIdem.Should().Be(json0);

        // Deterministic: two optimizers restored from the same state evolve bit-identically.
        var r1 = Cma.FromState(JsonSerializer.Deserialize<CmaState>(json0)!);
        var r2 = Cma.FromState(JsonSerializer.Deserialize<CmaState>(json0)!);
        for (int g = 0; g < 5; g++)
        {
            Advance(r1, 500 + g);
            Advance(r2, 500 + g);
        }
        r1.Mean.Should().Equal(r2.Mean);
        r1.Sigma.Should().Be(r2.Sigma);
        for (int i = 0; i < r1.Dim; i++)
            for (int j = 0; j < r1.Dim; j++)
                r1.Cov[i, j].Should().Be(r2.Cov[i, j]);
    }
}

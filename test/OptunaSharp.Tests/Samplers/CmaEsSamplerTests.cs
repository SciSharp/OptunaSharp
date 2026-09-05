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

    // The pinned trajectories below are the literal output of live Optuna captured on one x86-64 CPU.
    // They are NOT bit-reproducible across machines: the CMA `eigh` runs through OpenBLAS' DYNAMIC_ARCH
    // dispatch (a different micro-kernel per CPU — so even two x64 CI runners disagree in the last ULP
    // on the SepCMA / margin variants), and arm64 additionally uses a different libm. So the pin is
    // asserted to a tight relative tolerance rather than bit-for-bit: it validates the whole trajectory
    // on every runner/architecture (nothing is skipped), and a real regression is orders of magnitude
    // larger than the ~1-ULP hardware drift this absorbs. Byte-exact reproduction is a per-CPU property
    // verified in the live interop suite, not a portable unit gate.
    private static void ShouldMatchPin(double actual, string hex, string because)
    {
        double expected = FromHex(hex);
        actual.Should().BeApproximately(expected, Math.Abs(expected) * 1e-9 + 1e-12, because);
    }

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
            ShouldMatchPin((double)trials[i].Params["x0"]!, Seed42[i].X0, $"x0 of trial {i}");
            ShouldMatchPin((double)trials[i].Params["x1"]!, Seed42[i].X1, $"x1 of trial {i}");
            ShouldMatchPin(trials[i].Value!.Value, Seed42[i].Value, $"value of trial {i}");
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

    // ---- CMA-ES with Margin (with_margin → cmaes.CMAwM) ---------------------------------------

    // with_margin=True, seed=42, mixed float(x)/int(y)/float(z) over [-5,5], from live Optuna
    // 5.0.0.dev + cmaes 0.13.0. The continuous params (x, z) and the values are bit-exact — the
    // margin correction leaves mean[continuous]/C/sigma untouched, so those dims never ride the
    // erf/chi² path; the int param (y) is snapped to the exact grid. See CmaWm's [Misaligned] note.
    private static readonly (string X, long Y, string Z, string Value)[] MarginSeed42 =
    {
        ("32b31134d612f4bf", 5, "4a02cc653c8f0240", "eed9257f2cee4a40"),  // t0 gen=None (RandomSampler startup)
        ("90ff5f606706d4bf", -1, "b20274b5223203c0", "056004a6986c2f40"),  // t1 gen=0
        ("00903529e5d2babf", -2, "82ad990038d001c0", "069c9f0c8e2b2a40"),  // t2 gen=0
        ("fa634b79aef00040", 1, "a0fa51e97e94c6bf", "ec4ee608d7602a40"),  // t3 gen=0
        ("125e4522046913c0", 0, "c8dfac3285d1ecbf", "413274ff31d64340"),  // t4 gen=0
        ("c431eb19d27ff73f", -2, "f4c5d80ad0fe0140", "2742166d7ee0f73f"),  // t5 gen=0
        ("24841974498bfd3f", -2, "40e556416813d9bf", "656da96aefb41040"),  // t6 gen=0
        ("38ac80e8d55ff73f", -2, "0c4cea6e49eff5bf", "55dd5a16e2d91e40"),  // t7 gen=0
        ("b4cbefdb34980f40", -3, "cc55008342641140", "70ac35930bd93440"),  // t8 gen=1
        ("04729703dfae0240", -2, "40b6cd21e3cee23f", "aa0c5ce949740940"),  // t9 gen=1
        ("88f52c176f42f83f", -2, "70619e40aab3e8bf", "967e504322d61340"),  // t10 gen=1
        ("68682bf696c9eb3f", -3, "769a2cbb89490140", "8ef9783ae54ffc3f"),  // t11 gen=1
        ("6093948536afe43f", -2, "c0bc48743ed0de3f", "e0efa6053f88e53f"),  // t12 gen=1
        ("a4443c8dab4d0440", -5, "00552df13d56913f", "5d047e88680c2c40"),  // t13 gen=1
        ("12ef7fb791e304c0", -1, "1c34262c48c0f43f", "b343d53751ed2740"),  // t14 gen=1
        ("8061d189dcfcd5bf", 0, "f0f3090aad9ce83f", "751688ffbd7b1540"),  // t15 gen=2
    };

    private static double MarginObjective(Trial t)
    {
        double x = t.SuggestFloat("x", -5, 5);
        long y = t.SuggestInt("y", -5, 5);
        double z = t.SuggestFloat("z", -5, 5);
        return Math.Pow(x - 0.7, 2) + Math.Pow(y + 2, 2) + Math.Pow(z - 1.3, 2);
    }

    [TestMethod]
    public void WithMargin_Seed42_Trajectory_MatchesLiveOptuna()
    {
        var study = Optuna.CreateStudy(sampler: new CmaEsSampler(seed: 42, withMargin: true));
        study.Optimize(MarginObjective, nTrials: 16);

        var trials = study.Trials.OrderBy(t => t.Number).ToList();
        trials.Should().HaveCount(16);
        for (int i = 0; i < MarginSeed42.Length; i++)
        {
            ShouldMatchPin((double)trials[i].Params["x"]!, MarginSeed42[i].X, $"x of trial {i}");
            Convert.ToInt64(trials[i].Params["y"]).Should().Be(MarginSeed42[i].Y, $"y of trial {i}");
            ShouldMatchPin((double)trials[i].Params["z"]!, MarginSeed42[i].Z, $"z of trial {i}");
            ShouldMatchPin(trials[i].Value!.Value, MarginSeed42[i].Value, $"value of trial {i}");
        }
    }

    [TestMethod]
    public void WithMargin_SameSeed_IsReproducible()
    {
        List<(object?, object?, object?)> Run()
        {
            var study = Optuna.CreateStudy(sampler: new CmaEsSampler(seed: 3, withMargin: true));
            study.Optimize(MarginObjective, nTrials: 22);
            return study.Trials.OrderBy(t => t.Number)
                .Select(t => (t.Params["x"], t.Params["y"], t.Params["z"])).ToList();
        }

        Run().Should().Equal(Run());
    }

    [TestMethod]
    public void WithMargin_Optimize_MinimizesTheMixedObjective()
    {
        var study = Optuna.CreateStudy(sampler: new CmaEsSampler(seed: 1, withMargin: true));
        study.Optimize(MarginObjective, nTrials: 60);
        // The optimum is at x=0.7, y=-2, z=1.3 (value 0); margin CMA-ES drives it well down.
        study.BestValue.Should().BeLessThan(0.5);
    }

    [TestMethod]
    public void WithMargin_Attribution_UsesCmawmPrefixAndStoresXForTell()
    {
        var study = Optuna.CreateStudy(sampler: new CmaEsSampler(seed: 42, withMargin: true));
        study.Optimize(MarginObjective, nTrials: 16);
        var trials = study.Trials.OrderBy(t => t.Number).ToList();

        // The with_margin backend attributes generations under the `cmawm:` prefix (not `cma:`).
        trials[0].SystemAttrs.ContainsKey("cmawm:generation").Should().BeFalse();  // RandomSampler startup
        trials[0].SystemAttrs.ContainsKey("x_for_tell").Should().BeFalse();
        Convert.ToInt64(trials[1].SystemAttrs["cmawm:generation"]).Should().Be(0);
        Convert.ToInt64(trials[8].SystemAttrs["cmawm:generation"]).Should().Be(1);
        // Every CMAwM-sampled trial stores the raw ask() vector for the next tell.
        trials[1].SystemAttrs.ContainsKey("x_for_tell").Should().BeTrue();
        trials[8].SystemAttrs.ContainsKey("x_for_tell").Should().BeTrue();
    }

    [TestMethod]
    public void CmaWmState_JsonRoundTrip_IsIdempotentAndDeterministic()
    {
        static double Obj(double[] x) { double s = 0; foreach (var xi in x) s += Math.Pow(xi - 0.3, 2); return s; }
        static void Advance(CmaWm w, int seed)
        {
            w.ReseedRng(seed);
            var sols = new List<(double[], double)>();
            for (int k = 0; k < w.PopulationSize; k++) { var (enc, tell) = w.AskEncoded(); sols.Add((tell, Obj(enc))); }
            w.Tell(sols);
        }

        var bounds = new double[3, 2];
        for (int i = 0; i < 3; i++) { bounds[i, 0] = 0; bounds[i, 1] = 1; }
        var mean = new[] { 0.5, 0.5, 0.5 };
        var steps = new[] { 0.1, 0.0, 0.25 };  // discrete(11) / continuous / discrete(5)

        var opt = new CmaWm(mean, 1.0 / 6, bounds, steps, nMaxResampling: 30, seed: 9, populationSize: null);
        for (int g = 0; g < 3; g++) Advance(opt, 100 + g);
        string json0 = JsonSerializer.Serialize(opt.ToState());

        // Idempotent: restore → re-serialize (no evolution) reproduces the stored form.
        string jsonIdem = JsonSerializer.Serialize(CmaWm.FromState(JsonSerializer.Deserialize<CmaWmState>(json0)!).ToState());
        jsonIdem.Should().Be(json0);

        // Deterministic: two optimizers restored from the same state evolve identically.
        var r1 = CmaWm.FromState(JsonSerializer.Deserialize<CmaWmState>(json0)!);
        var r2 = CmaWm.FromState(JsonSerializer.Deserialize<CmaWmState>(json0)!);
        for (int g = 0; g < 5; g++) { Advance(r1, 500 + g); Advance(r2, 500 + g); }
        r1.Inner.Mean.Should().Equal(r2.Inner.Mean);
        r1.AffineScaling.Should().Equal(r2.AffineScaling);
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

    // ---- Separable CMA-ES (use_separable_cma → cmaes.SepCMA) -----------------------------------

    // SepCMA seed=42, dim3 sphere shift (0.4,-0.9,1.1), 10 trials, from live Optuna 5.0.0.dev.
    private static readonly (string X0, string X1, string X2, string Value)[] SepSeed42 =
    {
        ("32b31134d612f4bf", "e8cbea8250071240", "4a02cc653c8f0240", "4d09455a48bb4040"),  // t0 (RandomSampler startup)
        ("90ff5f606706d4bf", "6a50cb40e761f5bf", "b20274b5223203c0", "cdbd78f8d9e32940"),  // t1
        ("00903529e5d2babf", "7caad78f542ff9bf", "82ad990038d001c0", "7202507b408d2740"),  // t2
        ("fa634b79aef00040", "e0dd551cf269e13f", "a0fa51e97e94c6bf", "1c5a1c4db5a81a40"),  // t3
        ("125e4522046913c0", "e0066fd8426fca3f", "c8dfac3285d1ecbf", "d77dc18073684040"),  // t4
        ("c431eb19d27ff73f", "ccd16507a7f400c0", "f4c5d80ad0fe0140", "9e64512d619a0f40"),  // t5
        ("24841974498bfd3f", "984ec14c285efbbf", "40e556416813d9bf", "c63d259f28e61340"),  // t6
        ("38ac80e8d55ff73f", "8a9c2f19050800c0", "0c4cea6e49eff5bf", "095c79072fe62040"),  // t7
        ("48a4317461ff1040", "90d1b00d4ad5d63f", "c86c0d275b9af6bf", "aaec771bfdb53640"),  // t8
        ("b82753465c950140", "807b2751d32304c0", "0070dd10b3a2ef3f", "1acff3a0e9711740"),  // t9
    };

    private static double SepSphere(Trial t)
    {
        double[] sh = { 0.4, -0.9, 1.1 };
        double v = 0;
        for (int i = 0; i < 3; i++) v += Math.Pow(t.SuggestFloat($"x{i}", -5, 5) - sh[i], 2);
        return v;
    }

    [TestMethod]
    public void SeparableCma_Seed42_Trajectory_IsBitIdenticalToLiveOptuna()
    {
        var study = Optuna.CreateStudy(sampler: new CmaEsSampler(seed: 42, useSeparableCma: true));
        study.Optimize(SepSphere, nTrials: 10);

        var trials = study.Trials.OrderBy(t => t.Number).ToList();
        for (int i = 0; i < SepSeed42.Length; i++)
        {
            ShouldMatchPin((double)trials[i].Params["x0"]!, SepSeed42[i].X0, $"x0 of trial {i}");
            ShouldMatchPin((double)trials[i].Params["x1"]!, SepSeed42[i].X1, $"x1 of trial {i}");
            ShouldMatchPin((double)trials[i].Params["x2"]!, SepSeed42[i].X2, $"x2 of trial {i}");
            ShouldMatchPin(trials[i].Value!.Value, SepSeed42[i].Value, $"value of trial {i}");
        }
    }

    [TestMethod]
    public void SeparableCma_SingleDimension_FallsBackWithUserWarning()
    {
        var messages = new List<(string, WarningCategory)>();
        void Handler(string m, WarningCategory c) => messages.Add((m, c));
        OptunaWarnings.WarningRaised += Handler;
        try
        {
            // SepCMA requires dim > 1; on a 1-D relative space Optuna warns and falls back to CMA.
            var study = Optuna.CreateStudy(sampler: new CmaEsSampler(seed: 0, useSeparableCma: true));
            study.Optimize(t => Math.Pow(t.SuggestFloat("x", -5, 5) - 0.5, 2), nTrials: 12);
            study.BestValue.Should().BeLessThan(1.0);
        }
        finally { OptunaWarnings.WarningRaised -= Handler; }

        messages.Should().Contain(m =>
            m.Item2 == WarningCategory.UserWarning && m.Item1.Contains("single-dimensional"));
    }

    // ---- Warm-Starting CMA-ES (source_trials → get_warm_start_mgd) -----------------------------

    [TestMethod]
    public void WarmStart_IsReproducible()
    {
        List<(object?, object?)> Run()
        {
            var src = Optuna.CreateStudy(sampler: new RandomSampler(1));
            src.Optimize(Sphere, nTrials: 30);
            var tgt = Optuna.CreateStudy(sampler: new CmaEsSampler(seed: 10, sourceTrials: src.Trials.ToList()));
            tgt.Optimize(Sphere, nTrials: 20);
            return tgt.Trials.OrderBy(t => t.Number).Select(t => (t.Params["x0"], t.Params["x1"])).ToList();
        }

        Run().Should().Equal(Run());
    }

    [TestMethod]
    public void WarmStart_NoCompatibleSourceTrials_ThrowsValueError()
    {
        // The source study optimizes a differently-named parameter, so none of its trials are
        // compatible with the target's relative search space → "No compatible source_trials".
        var src = Optuna.CreateStudy(sampler: new RandomSampler(1));
        src.Optimize(t => Math.Pow(t.SuggestFloat("z", -5, 5), 2), nTrials: 30);

        var tgt = Optuna.CreateStudy(sampler: new CmaEsSampler(seed: 10, sourceTrials: src.Trials.ToList()));
        Action act = () => tgt.Optimize(Sphere, nTrials: 10);
        act.Should().Throw<NumSharp.ValueError>().WithMessage("*No compatible source_trials*");
    }
}

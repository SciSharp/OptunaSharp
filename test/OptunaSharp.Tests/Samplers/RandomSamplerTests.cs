using System;
using System.Linq;
using AwesomeAssertions;
using OptunaSharp.Collections;
using OptunaSharp.Distributions;
using OptunaSharp.Samplers;
using OptunaSharp.SearchSpace;

namespace OptunaSharp.Tests.Samplers;

/// <summary>
///     RandomSampler bit-parity against live Optuna (the plan §2 RNG-discipline payoff): every
///     value below is the literal output of <c>optuna.samplers.RandomSampler(seed=123)</c> over
///     a mixed search space — float, log-float, stepped float, int, log-int, stepped int and
///     categorical — captured from a real Python run. If any of these drift, the MT19937 draw
///     order or a transform edge-case broke.
/// </summary>
[TestClass]
public class RandomSamplerTests
{
    private static Study RunMixedStudy(int nTrials)
    {
        var study = Optuna.CreateStudy(sampler: new RandomSampler(seed: 123));
        study.Optimize(t =>
        {
            double x = t.SuggestFloat("x", -10, 10);
            double y = t.SuggestFloat("y", 1e-5, 1e-1, log: true);
            double z = t.SuggestFloat("z", 0.1, 0.9, step: 0.2);
            long n = t.SuggestInt("n", 1, 100);
            long m = t.SuggestInt("m", 1, 128, log: true);
            long k = t.SuggestInt("k", 0, 20, step: 5);
            string c = t.SuggestCategorical("c", new[] { "adam", "sgd", "rmsprop" });
            return x * x + y + z + n + m + k + (c == "adam" ? 1 : 0);
        }, nTrials: nTrials);
        return study;
    }

    // Literal live-optuna output for seed=123 (repr for floats, so bit-exact).
    private static readonly (string X, string Y, string Z, long N, long M, long K, string C)[] Expected =
    {
        ("3.9293837119572324", "0.00013949458198611188", "0.30000000000000004", 56, 27, 10, "adam"),
        ("-2.1576496361169895", "0.00023589137648150635", "0.7000000000000001", 44, 1, 5, "adam"),
        ("0.631027476836767", "0.0013406343673102114", "0.7000000000000001", 85, 28, 15, "adam"),
        ("-5.434735382420888", "0.0001495740267170668", "0.7000000000000001", 10, 6, 10, "adam"),
        ("-1.472973860743835", "0.0374590408092628", "0.9", 51, 16, 0, "rmsprop"),
        ("-4.990892692069866", "0.0008553366021440326", "0.9", 52, 15, 0, "adam"),
        ("-3.1447233245138317", "0.00016462021179757346", "0.5", 69, 64, 10, "adam"),
        ("3.493781019756497", "0.02340829330353508", "0.1", 77, 2, 0, "rmsprop"),
    };

    [TestMethod]
    public void Seeded_MixedSpace_Is_BitIdentical_To_Python()
    {
        var study = RunMixedStudy(Expected.Length);
        for (int i = 0; i < Expected.Length; i++)
        {
            var p = study.Trials[i].Params;
            OptunaSharp.Internal.PythonRepr.FloatRepr((double)p["x"]!).Should().Be(Expected[i].X, $"trial {i} x");
            OptunaSharp.Internal.PythonRepr.FloatRepr((double)p["y"]!).Should().Be(Expected[i].Y, $"trial {i} y");
            OptunaSharp.Internal.PythonRepr.FloatRepr((double)p["z"]!).Should().Be(Expected[i].Z, $"trial {i} z");
            p["n"].Should().Be(Expected[i].N, $"trial {i} n");
            p["m"].Should().Be(Expected[i].M, $"trial {i} m");
            p["k"].Should().Be(Expected[i].K, $"trial {i} k");
            p["c"].Should().Be(Expected[i].C, $"trial {i} c");
        }
    }

    [TestMethod]
    public void Same_Seed_Reproduces_Same_Sequence()
    {
        var a = RunMixedStudy(3);
        var b = RunMixedStudy(3);
        for (int i = 0; i < 3; i++)
        {
            a.Trials[i].Params["x"].Should().Be(b.Trials[i].Params["x"]);
            a.Trials[i].Params["c"].Should().Be(b.Trials[i].Params["c"]);
        }
    }

    [TestMethod]
    public void ReseedRng_Diverges_The_Sequence()
    {
        var sampler = new RandomSampler(seed: 1);
        var study1 = Optuna.CreateStudy(sampler: sampler);
        study1.Optimize(t => t.SuggestFloat("x", 0, 1), nTrials: 1);
        double first = (double)study1.Trials[0].Params["x"]!;

        sampler.ReseedRng();
        var study2 = Optuna.CreateStudy(sampler: sampler);
        study2.Optimize(t => t.SuggestFloat("x", 0, 1), nTrials: 1);
        ((double)study2.Trials[0].Params["x"]!).Should().NotBe(first);
    }

    [TestMethod]
    public void IntersectionSearchSpace_Excludes_Dynamic_Ranges()
    {
        var study = Optuna.CreateStudy(sampler: new RandomSampler(3));
        study.Optimize(t =>
        {
            double x = t.SuggestFloat("x", 0, 1);
            double y = t.Number % 2 == 0 ? t.SuggestFloat("y", 0, 1) : t.SuggestFloat("y", 0, 2);
            if (t.Number < 2)
                t.SuggestInt("only_early", 0, 5);
            return x + y;
        }, nTrials: 4);

        var space = new IntersectionSearchSpace().Calculate(study);
        space.Keys.Should().Equal("x");  // y has a dynamic range; only_early missing later.

        var byTrials = SearchSpaceFunctions.IntersectionSearchSpaceOf(study.GetTrials(deepcopy: false));
        byTrials.Keys.Should().Equal("x");
        byTrials["x"].Should().Be(new FloatDistribution(0, 1));
    }

    [TestMethod]
    public void IntersectionSearchSpace_Rejects_Second_Study()
    {
        var space = new IntersectionSearchSpace();
        var study1 = Optuna.CreateStudy(sampler: new RandomSampler(1), storage: new OptunaSharp.Storages.InMemoryStorage());
        space.Calculate(study1);

        // A second, different storage/study id (InMemory ids collide by design; use trick:
        // the check is only meaningful with distinct study ids, so create two studies in ONE storage).
        var storage = new OptunaSharp.Storages.InMemoryStorage();
        var a = Optuna.CreateStudy(sampler: new RandomSampler(1), storage: storage, studyName: "a");
        var b = Optuna.CreateStudy(sampler: new RandomSampler(1), storage: storage, studyName: "b");
        var shared = new IntersectionSearchSpace();
        shared.Calculate(a);
        var act = () => shared.Calculate(b);
        act.Should().Throw<NumSharp.ValueError>()
            .WithMessage("`IntersectionSearchSpace` cannot handle multiple studies.");
    }
}

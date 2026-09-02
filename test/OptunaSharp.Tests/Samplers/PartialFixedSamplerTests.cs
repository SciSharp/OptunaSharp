using System.Collections.Generic;
using System.Linq;
using AwesomeAssertions;
using OptunaSharp.Internal;
using OptunaSharp.Samplers;

namespace OptunaSharp.Tests.Samplers;

/// <summary>
///     PartialFixedSampler bit-parity against live Optuna. The sampler wraps a base sampler,
///     pinning some params and delegating the rest — so once <c>y</c> is fixed, the unfixed
///     <c>x</c> is drawn by the same base RandomSampler whose RNG simply continues from where
///     phase 1 left off. Every literal below is the output of a real Python run.
/// </summary>
[TestClass]
public class PartialFixedSamplerTests
{
    private static double Objective(Trial t)
    {
        double x = t.SuggestFloat("x", -1, 1);
        long y = t.SuggestInt("y", -1, 1);
        return x * x + y;
    }

    // Live-optuna phase-2 x values (RandomSampler(seed=5) continued after 6 phase-1 trials,
    // y fixed to the phase-1 best of -1). repr strings so the compare is bit-exact.
    private static readonly string[] ExpectedPhase2X =
    {
        "-0.11738155420809382",
        "-0.6833802645746976",
        "0.7598740624025577",
        "-0.4518270760155507",
        "-0.17152996183789737",
        "-0.40784013453270407",
    };

    [TestMethod]
    public void Fixes_Param_And_Phase2_Is_BitIdentical_To_Python()
    {
        var study = Optuna.CreateStudy(sampler: new RandomSampler(seed: 5));
        study.Optimize(Objective, nTrials: 6);

        study.BestParams["y"].Should().Be(-1L);
        PythonRepr.FloatRepr((double)study.BestParams["x"]!).Should().Be("-0.4063989968475561");

        var fixedParams = new Dictionary<string, object?> { ["y"] = study.BestParams["y"] };
        study.Sampler = new PartialFixedSampler(fixedParams, study.Sampler);
        study.Optimize(Objective, nTrials: 6);

        var phase2 = study.Trials.Skip(6).ToList();
        phase2.Should().HaveCount(6);
        for (int i = 0; i < phase2.Count; i++)
        {
            phase2[i].Params["y"].Should().Be(-1L, $"phase-2 trial {i} y must stay fixed");
            PythonRepr.FloatRepr((double)phase2[i].Params["x"]!).Should().Be(ExpectedPhase2X[i], $"phase-2 trial {i} x");
        }
    }

    [TestMethod]
    public void Out_Of_Range_Fixed_Param_Warns_But_Is_Used()
    {
        string? warned = null;
        void Handler(string message, WarningCategory category) => warned ??= message;
        OptunaWarnings.WarningRaised += Handler;
        try
        {
            var study = Optuna.CreateStudy(sampler: new PartialFixedSampler(
                new Dictionary<string, object?> { ["y"] = 99L },
                new RandomSampler(seed: 1)));
            study.Optimize(t => t.SuggestFloat("x", -1, 1) + t.SuggestInt("y", -1, 1), nTrials: 1);

            study.Trials[0].Params["y"].Should().Be(99L);
        }
        finally
        {
            OptunaWarnings.WarningRaised -= Handler;
        }

        warned.Should().Be(
            "Fixed parameter 'y' with value 99 is out of range for distribution " +
            "IntDistribution(high=1, log=False, low=-1, step=1).");
    }

    [TestMethod]
    public void Unfixed_Params_Delegate_To_Base_Sampler()
    {
        // With an empty fixed-params dict the sampler is a transparent pass-through of the base.
        var wrapped = Optuna.CreateStudy(sampler: new PartialFixedSampler(
            new Dictionary<string, object?>(), new RandomSampler(seed: 7)));
        wrapped.Optimize(Objective, nTrials: 4);

        var direct = Optuna.CreateStudy(sampler: new RandomSampler(seed: 7));
        direct.Optimize(Objective, nTrials: 4);

        for (int i = 0; i < 4; i++)
        {
            wrapped.Trials[i].Params["x"].Should().Be(direct.Trials[i].Params["x"], $"trial {i} x");
            wrapped.Trials[i].Params["y"].Should().Be(direct.Trials[i].Params["y"], $"trial {i} y");
        }
    }
}

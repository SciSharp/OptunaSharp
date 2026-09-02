using System;
using System.Collections.Generic;
using System.Linq;
using AwesomeAssertions;
using OptunaSharp.Distributions;
using OptunaSharp.Internal;
using OptunaSharp.Samplers;

namespace OptunaSharp.Tests.Samplers;

/// <summary>
///     GridSampler bit-parity against live Optuna. The grid order is fixed by the seeded shuffle
///     of the itertools-product enumeration, so every trial's params, its <c>grid_id</c> system
///     attr, and the exhaustion-triggered stop are deterministic — all literals below are the
///     output of <c>optuna.samplers.GridSampler(search_space, seed=…)</c> from a real Python run.
/// </summary>
[TestClass]
public class GridSamplerTests
{
    private static Dictionary<string, IReadOnlyList<CategoricalChoice>> XySearchSpace() => new()
    {
        ["x"] = new CategoricalChoice[] { -50, 0, 50 },
        ["y"] = new CategoricalChoice[] { -99, 0, 99 },
    };

    // Literal live-optuna output for GridSampler(seed=0) over {x:[-50,0,50], y:[-99,0,99]}.
    // params come back through storage: x (float dist) is a double, y (int dist) is a long.
    private static readonly (double X, long Y, int GridId)[] Expected =
    {
        (50.0, 0, 0),
        (-50.0, 99, 1),
        (-50.0, 0, 2),
        (0.0, 0, 3),
        (50.0, 99, 4),
        (50.0, -99, 5),
        (0.0, -99, 6),
        (-50.0, -99, 7),
        (0.0, 99, 8),
    };

    [TestMethod]
    public void ExhaustiveGrid_Order_And_GridIds_Are_BitIdentical_To_Python()
    {
        var study = Optuna.CreateStudy(sampler: new GridSampler(XySearchSpace(), seed: 0));
        // n_trials generous; GridSampler stops itself once the 9-point grid is exhausted.
        study.Optimize(t =>
        {
            double x = t.SuggestFloat("x", -100, 100);
            long y = t.SuggestInt("y", -100, 100);
            return x * x + y * y;
        }, nTrials: 20);

        study.Trials.Should().HaveCount(Expected.Length, "GridSampler stops after evaluating the whole grid");
        for (int i = 0; i < Expected.Length; i++)
        {
            var t = study.Trials[i];
            ((double)t.Params["x"]!).Should().Be(Expected[i].X, $"trial {i} x");
            t.Params["y"].Should().Be(Expected[i].Y, $"trial {i} y");
            Convert.ToInt32(t.SystemAttrs["grid_id"]).Should().Be(Expected[i].GridId, $"trial {i} grid_id");
        }
    }

    [TestMethod]
    public void SearchSpace_SystemAttr_Preserves_IntTypes()
    {
        var study = Optuna.CreateStudy(sampler: new GridSampler(XySearchSpace(), seed: 0));
        study.Optimize(t => t.SuggestFloat("x", -100, 100) + t.SuggestInt("y", -100, 100), nTrials: 20);

        var searchSpace = (OptunaSharp.Collections.OrderedDict<string, object?>)study.Trials[0].SystemAttrs["search_space"]!;
        // Grid values were ints, so they must serialize as ints (longs), never floats — the journal
        // format depends on this.
        ((List<object?>)searchSpace["x"]!).Should().Equal(-50L, 0L, 50L);
        ((List<object?>)searchSpace["y"]!).Should().Equal(-99L, 0L, 99L);
    }

    [TestMethod]
    public void MoreTrialsThanGrid_StopsEarly_And_Reevaluates_Like_Python()
    {
        // Grid of size 2, seed=3: fast path assigns grid_id 0,1 (a=2,1); after the 2nd trial the
        // grid is exhausted, so the study stops even though n_trials=5.
        var study = Optuna.CreateStudy(sampler: new GridSampler(
            new Dictionary<string, IReadOnlyList<CategoricalChoice>> { ["a"] = new CategoricalChoice[] { 1, 2 } },
            seed: 3));
        study.Optimize(t => (double)t.SuggestInt("a", 1, 5), nTrials: 5);

        study.Trials.Should().HaveCount(2);
        study.Trials.Select(t => t.Params["a"]).Should().Equal(2L, 1L);
        study.Trials.Select(t => Convert.ToInt32(t.SystemAttrs["grid_id"])).Should().Equal(0, 1);
    }

    [TestMethod]
    public void IsExhausted_Reports_Grid_Completion()
    {
        var sampler = new GridSampler(
            new Dictionary<string, IReadOnlyList<CategoricalChoice>> { ["a"] = new CategoricalChoice[] { 1, 2, 3 } },
            seed: 0);
        var study = Optuna.CreateStudy(sampler: sampler);

        sampler.IsExhausted(study).Should().BeFalse();
        study.Optimize(t => (double)t.SuggestInt("a", 1, 3), nTrials: 10);
        sampler.IsExhausted(study).Should().BeTrue();
    }

    [TestMethod]
    public void Categorical_Grid_Values_Round_Trip()
    {
        var study = Optuna.CreateStudy(sampler: new GridSampler(
            new Dictionary<string, IReadOnlyList<CategoricalChoice>>
            {
                ["opt"] = new CategoricalChoice[] { "adam", "sgd" },
            },
            seed: 1));
        study.Optimize(t => t.SuggestCategorical("opt", new CategoricalChoice[] { "adam", "sgd" }) == "adam" ? 1.0 : 0.0,
            nTrials: 10);

        study.Trials.Should().HaveCount(2);
        study.Trials.Select(t => (string)t.Params["opt"]!).Should().BeEquivalentTo(new[] { "adam", "sgd" });
    }

    [TestMethod]
    public void Value_Out_Of_Grid_Range_Warns_But_Is_Used()
    {
        // Grid specifies a non-integer point for a stepped-integer suggestion — Optuna warns and
        // uses the value anyway (GridSampler ignores quantization).
        string? warned = null;
        void Handler(string message, WarningCategory category) => warned ??= message;
        OptunaWarnings.WarningRaised += Handler;
        try
        {
            var study = Optuna.CreateStudy(sampler: new GridSampler(
                new Dictionary<string, IReadOnlyList<CategoricalChoice>> { ["x"] = new CategoricalChoice[] { -0.5, 0.5 } },
                seed: 0));
            study.Optimize(t => t.SuggestFloat("x", -5, 5, step: 1), nTrials: 5);

            study.Trials.Select(t => (double)t.Params["x"]!).Should().BeEquivalentTo(new[] { -0.5, 0.5 });
        }
        finally
        {
            OptunaWarnings.WarningRaised -= Handler;
        }

        warned.Should().Contain("is out of range of the parameter");
    }

    [TestMethod]
    public void Unknown_Param_Throws()
    {
        var study = Optuna.CreateStudy(sampler: new GridSampler(XySearchSpace(), seed: 0));
        var act = () => study.Optimize(t =>
        {
            t.SuggestFloat("x", -100, 100);
            t.SuggestInt("y", -100, 100);
            return t.SuggestFloat("z", 0, 1);  // z is not in the grid.
        }, nTrials: 1);

        act.Should().Throw<NumSharp.ValueError>()
            .WithMessage("The parameter name, z, is not found in the given grid.");
    }
}

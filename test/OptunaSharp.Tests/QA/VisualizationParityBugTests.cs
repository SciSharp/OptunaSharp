using System.Linq;
using AwesomeAssertions;
using OptunaSharp.Collections;
using OptunaSharp.Distributions;
using OptunaSharp.Visualization;

namespace OptunaSharp.Tests.QA;

/// <summary>
///     Open parity bugs in the visualization info layer, found by the 2026-11-02 parity QA run
///     (<c>docs/qa/PARITY_QA_2026-11-02.md</c>). A dump of every info function over seven studies matched
///     Optuna 5.0.0 apart from the case below; each test fails today and passes once fixed, at which point its
///     <c>[OpenBugs]</c> attribute should be removed.
/// </summary>
[TestClass]
public class VisualizationParityBugTests
{
    /// <summary>
    ///     QA-26: the parallel-coordinate data must handle a categorical parameter whose choices include
    ///     <c>None</c>, as Optuna's <c>_get_parallel_coordinate_info</c> does; the port throws
    ///     <see cref="System.ArgumentNullException" /> ("Value cannot be null. (Parameter 'key')").
    /// </summary>
    /// <remarks>
    ///     The port collects the categorical vocabulary in a dictionary keyed by the raw choice value, and a .NET
    ///     dictionary rejects a null key. So <c>plot_parallel_coordinate</c> fails for any study with a
    ///     <c>suggest_categorical(..., [None, ...])</c> parameter, a common way to make an option optional.
    ///     Optuna 5.0.0 for these four trials: the <c>c</c> dimension has values <c>(0, 1, 2, 0)</c>, tickvals
    ///     <c>[0, 1, 2]</c> and ticktext <c>['None', 'a', 'b']</c>.
    /// </remarks>
    [TestMethod]
    [OpenBugs]
    public void ParallelCoordinate_NoneCategoricalChoice_HandledLikePython()
    {
        var study = Optuna.CreateStudy();
        var cd = new CategoricalDistribution("a", CategoricalChoice.None, "b");
        var fd = new FloatDistribution(0, 1);
        foreach (var (c, x, v) in new (string? C, double X, double V)[] { (null, 0.1, 1.0), ("a", 0.5, 2.0), ("b", 0.9, 3.0), (null, 0.3, 0.5) })
        {
            study.AddTrial(FrozenTrial.CreateTrial(
                value: v,
                @params: new OrderedDict<string, object?> { ["c"] = c, ["x"] = x },
                distributions: new OrderedDict<string, BaseDistribution> { ["c"] = cd, ["x"] = fd }));
        }

        var info = ParallelCoordinate.GetInfo(study);

        var dim = info.DimsParams.Single(d => d.Label == "c");
        dim.Values.Should().Equal(0.0, 1.0, 2.0, 0.0);
        dim.Tickvals.Should().Equal(0.0, 1.0, 2.0);
        dim.Ticktext.Should().Equal("None", "a", "b");
    }
}

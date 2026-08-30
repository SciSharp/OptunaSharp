using AwesomeAssertions;
using OptunaSharp.Distributions;

namespace OptunaSharp.Tests.Distributions;

/// <summary>
///     Pins <c>distribution_to_json</c> / <c>json_to_distribution</c> to Python's exact bytes —
///     every expected string below is the literal output of optuna's <c>distribution_to_json</c>
///     (storage interop depends on this; plan 0.2).
/// </summary>
[TestClass]
public class DistributionJsonTests
{
    [TestMethod]
    public void ToJson_Is_ByteIdentical_To_Python()
    {
        Distribution.ToJson(new FloatDistribution(1, 10)).Should().Be(
            "{\"name\": \"FloatDistribution\", \"attributes\": {\"step\": null, \"low\": 1.0, \"high\": 10.0, \"log\": false}}");
        Distribution.ToJson(new FloatDistribution(1e-5, 1e-1, log: true)).Should().Be(
            "{\"name\": \"FloatDistribution\", \"attributes\": {\"step\": null, \"low\": 1e-05, \"high\": 0.1, \"log\": true}}");
        Distribution.ToJson(new FloatDistribution(0.1, 1.0, step: 0.1)).Should().Be(
            "{\"name\": \"FloatDistribution\", \"attributes\": {\"step\": 0.1, \"low\": 0.1, \"high\": 1.0, \"log\": false}}");
        Distribution.ToJson(new IntDistribution(1, 10)).Should().Be(
            "{\"name\": \"IntDistribution\", \"attributes\": {\"log\": false, \"step\": 1, \"low\": 1, \"high\": 10}}");
        Distribution.ToJson(new IntDistribution(1, 100, log: true)).Should().Be(
            "{\"name\": \"IntDistribution\", \"attributes\": {\"log\": true, \"step\": 1, \"low\": 1, \"high\": 100}}");
        Distribution.ToJson(new IntDistribution(0, 10, step: 2)).Should().Be(
            "{\"name\": \"IntDistribution\", \"attributes\": {\"log\": false, \"step\": 2, \"low\": 0, \"high\": 10}}");
        Distribution.ToJson(new CategoricalDistribution(new CategoricalChoice[] { "a", "b", "c" })).Should().Be(
            "{\"name\": \"CategoricalDistribution\", \"attributes\": {\"choices\": [\"a\", \"b\", \"c\"]}}");
        Distribution.ToJson(new CategoricalDistribution(
            new CategoricalChoice[] { CategoricalChoice.None, true, 1L, 2.5, "x" })).Should().Be(
            "{\"name\": \"CategoricalDistribution\", \"attributes\": {\"choices\": [null, true, 1, 2.5, \"x\"]}}");
    }

    [TestMethod]
    public void ToJson_Deprecated_Aliases_Match_Python()
    {
#pragma warning disable CS0618
        Distribution.ToJson(new UniformDistribution(1, 9)).Should().Be(
            "{\"name\": \"UniformDistribution\", \"attributes\": {\"low\": 1.0, \"high\": 9.0}}");
        Distribution.ToJson(new LogUniformDistribution(0.1, 10)).Should().Be(
            "{\"name\": \"LogUniformDistribution\", \"attributes\": {\"low\": 0.1, \"high\": 10.0}}");
        Distribution.ToJson(new DiscreteUniformDistribution(0, 1, 0.1)).Should().Be(
            "{\"name\": \"DiscreteUniformDistribution\", \"attributes\": {\"low\": 0.0, \"high\": 1.0, \"q\": 0.1}}");
        Distribution.ToJson(new IntUniformDistribution(0, 10, 2)).Should().Be(
            "{\"name\": \"IntUniformDistribution\", \"attributes\": {\"step\": 2, \"low\": 0, \"high\": 10}}");
        Distribution.ToJson(new IntLogUniformDistribution(1, 100)).Should().Be(
            "{\"name\": \"IntLogUniformDistribution\", \"attributes\": {\"step\": 1, \"low\": 1, \"high\": 100}}");
#pragma warning restore CS0618
    }

    [TestMethod]
    public void FromJson_RoundTrips_All_Classes()
    {
        BaseDistribution[] dists =
        {
            new FloatDistribution(1, 10),
            new FloatDistribution(1e-5, 1e-1, log: true),
            new FloatDistribution(0.1, 1.0, step: 0.1),
            new IntDistribution(1, 10),
            new IntDistribution(1, 100, log: true),
            new IntDistribution(0, 10, step: 2),
            new CategoricalDistribution(new CategoricalChoice[] { CategoricalChoice.None, true, 1L, 2.5, "x" }),
        };
        foreach (var d in dists)
        {
            var back = Distribution.FromJson(Distribution.ToJson(d));
            back.Should().Be(d, $"round trip must preserve {d}");
            back.GetType().Should().Be(d.GetType());
        }
    }

    [TestMethod]
    public void FromJson_Parses_Python_Output_From_Live_Optuna()
    {
        // Literal strings produced by optuna 4.6 (matching the pinned clone's format).
        var f = (FloatDistribution)Distribution.FromJson(
            "{\"name\": \"FloatDistribution\", \"attributes\": {\"step\": null, \"low\": 1e-05, \"high\": 0.1, \"log\": true}}");
        f.Low.Should().Be(1e-05);
        f.High.Should().Be(0.1);
        f.Log.Should().BeTrue();
        f.Step.Should().BeNull();

        var c = (CategoricalDistribution)Distribution.FromJson(
            "{\"name\": \"CategoricalDistribution\", \"attributes\": {\"choices\": [null, true, 1, 2.5, \"x\"]}}");
        c.Choices.Should().HaveCount(5);
        c.Choices[0].IsNone.Should().BeTrue();
        c.Choices[1].Value.Should().Be(true);
        c.Choices[2].Value.Should().Be(1L);
        c.Choices[3].Value.Should().Be(2.5);
        c.Choices[4].Value.Should().Be("x");
    }

    [TestMethod]
    public void FromJson_Supports_Abbreviated_Format()
    {
        var f = (FloatDistribution)Distribution.FromJson("{\"type\": \"float\", \"low\": 0.0, \"high\": 1.0}");
        f.Low.Should().Be(0.0);
        f.Step.Should().BeNull();

        var i = (IntDistribution)Distribution.FromJson("{\"type\": \"int\", \"low\": 1, \"high\": 9, \"log\": true}");
        i.Log.Should().BeTrue();
        i.Step.Should().Be(1);

        var c = (CategoricalDistribution)Distribution.FromJson("{\"type\": \"categorical\", \"choices\": [1, \"a\"]}");
        c.Choices.Should().HaveCount(2);
    }

    [TestMethod]
    public void FromJson_Unknown_Class_Raises_Verbatim()
    {
        var act = () => Distribution.FromJson("{\"name\": \"WeirdDistribution\", \"attributes\": {}}");
        act.Should().Throw<NumSharp.ValueError>()
            .WithMessage("Unknown distribution class: WeirdDistribution");

        var act2 = () => Distribution.FromJson("{\"type\": \"weird\"}");
        act2.Should().Throw<NumSharp.ValueError>()
            .WithMessage("Unknown distribution type: weird");
    }
}

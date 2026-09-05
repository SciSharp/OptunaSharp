using System;
using AwesomeAssertions;
using OptunaSharp.Cli;
using OptunaSharp.Samplers;

namespace OptunaSharp.Tests.Cli;

/// <summary>
///     Unit tests for the CLI's reflection sampler construction (<see cref="SamplerFactory" />, the port
///     of the <c>ask</c> command's <c>getattr(optuna.samplers, name)(**json.loads(kwargs))</c>).
/// </summary>
[TestClass]
public class CliSamplerFactoryTests
{
    [TestMethod]
    public void RandomSampler_NoKwargs_Constructs()
        => SamplerFactory.Create("RandomSampler", kwargsJson: null).Should().BeOfType<RandomSampler>();

    [TestMethod]
    public void RandomSampler_SeedKwarg_Constructs()
        => SamplerFactory.Create("RandomSampler", "{\"seed\": 0}").Should().BeOfType<RandomSampler>();

    [TestMethod]
    public void TPESampler_NoKwargs_Constructs()
        => SamplerFactory.Create("TPESampler", kwargsJson: null).Should().BeOfType<TPESampler>();

    [TestMethod]
    public void TPESampler_SeedKwarg_Constructs()
        => SamplerFactory.Create("TPESampler", "{\"seed\": 42}").Should().BeOfType<TPESampler>();

    [TestMethod]
    public void GridSampler_WithoutRequiredSearchSpace_Throws()
    {
        // GridSampler's constructor requires search_space (no default), which no kwarg supplies.
        Action act = () => SamplerFactory.Create("GridSampler", kwargsJson: null);
        act.Should().Throw<NumSharp.ValueError>();
    }

    [TestMethod]
    public void UnknownSampler_ThrowsAttributeError()
    {
        Action act = () => SamplerFactory.Create("NoSuchSampler", kwargsJson: null);
        act.Should().Throw<NumSharp.AttributeError>().WithMessage("*optuna.samplers*NoSuchSampler*");
    }

    [TestMethod]
    public void NonObjectKwargsJson_ThrowsValueError()
    {
        Action act = () => SamplerFactory.Create("RandomSampler", "[1, 2, 3]");
        act.Should().Throw<NumSharp.ValueError>();
    }

    [TestMethod]
    public void UnmatchedKwarg_ThrowsValueError()
    {
        // A kwarg that binds to no constructor parameter is an error, like Python's unexpected keyword.
        Action act = () => SamplerFactory.Create("RandomSampler", "{\"no_such_parameter\": 1}");
        act.Should().Throw<NumSharp.ValueError>();
    }
}

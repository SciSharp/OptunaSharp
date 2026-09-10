using System;
using System.Linq;
using AwesomeAssertions;
using OptunaSharp.Interop.Pymoo;

namespace OptunaSharp.Tests.Interop;

/// <summary>
///     Pure-logic checks for the pymoo bridge's static surface that need <b>no</b> embedded interpreter:
///     the <see cref="PymooAlgorithm" /> → (module, class) map, the construction-family partition, and
///     the population-size guard that runs before pymoo is ever imported. (The behavioral / value-parity
///     tests that actually drive pymoo live in <see cref="PymooSamplerParityTests" />.)
/// </summary>
[TestClass]
public class PymooApiUnitTests
{
    [TestMethod]
    public void EveryAlgorithm_ResolvesToAPymooMooModuleAndClass()
    {
        foreach (PymooAlgorithm algorithm in Enum.GetValues<PymooAlgorithm>())
        {
            (string module, string className) = PymooAlgorithmMap.Resolve(algorithm);
            module.Should().StartWith("pymoo.algorithms.moo.", $"{algorithm} maps into pymoo's MOO package");
            className.Should().NotBeNullOrWhiteSpace();
        }
    }

    [TestMethod]
    public void ConstructionFamilies_AreDispatchableAndCoverEveryAlgorithm()
    {
        foreach (PymooAlgorithm algorithm in Enum.GetValues<PymooAlgorithm>())
        {
            bool refDirs = PymooAlgorithmMap.UsesReferenceDirections(algorithm);
            bool refPoints = PymooAlgorithmMap.UsesReferencePoints(algorithm);

            // BuildAlgorithmKwargs dispatches: ref-directions, else ref-points, else pop_size. The two
            // explicit branches must not overlap. (R-NSGA-II additionally takes a pop_size — so it is in
            // UsesPopulationSize too — but it is dispatched as a reference-point algorithm.)
            (refDirs && refPoints).Should().BeFalse($"{algorithm} is not both ref-directions and ref-points");
            (refDirs || refPoints || PymooAlgorithmMap.UsesPopulationSize(algorithm))
                .Should().BeTrue($"{algorithm} must be covered by a construction path");
        }
    }

    [TestMethod]
    public void TheEnum_ExposesTheElevenSupportedAlgorithms_AndOmitsMoeadAndDnsga2()
    {
        var names = Enum.GetNames<PymooAlgorithm>().ToList();
        names.Should().HaveCount(11);
        names.Should().NotContain("Moead");
        names.Should().NotContain("Dnsga2");
    }

    [TestMethod]
    public void PopulationSizeBelowTwo_ThrowsBeforeTouchingPython()
    {
        // The guard runs before the `import pymoo` in the ctor, so this needs no engine.
        Action act = () => new PymooSampler(PymooAlgorithm.Nsga2, populationSize: 1);
        act.Should().Throw<NumSharp.ValueError>().WithMessage("*population_size*");
    }
}

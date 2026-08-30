using System;
using AwesomeAssertions;
using NumSharp;
using OptunaSharp.Distributions;

namespace OptunaSharp.Tests.Distributions;

/// <summary>
///     Behavior of the distribution classes — validation (with Optuna's verbatim error
///     messages), <c>single()</c>, <c>_contains</c>, internal/external conversion, and the
///     Decimal-exact step adjustment.
/// </summary>
[TestClass]
public class DistributionTests
{
    // ------------------------------------------------------------------ FloatDistribution

    [TestMethod]
    public void Float_Rejects_Log_With_Step()
    {
        var act = () => new FloatDistribution(1, 10, log: true, step: 0.5);
        act.Should().Throw<ValueError>()
            .WithMessage("The parameter `step` is not supported when `log` is true.");
    }

    [TestMethod]
    public void Float_Rejects_Low_Above_High_Verbatim()
    {
        var act = () => new FloatDistribution(2.0, 1.0);
        act.Should().Throw<ValueError>()
            .WithMessage("`low <= high` must hold, but got (low=2.0, high=1.0).");
    }

    [TestMethod]
    public void Float_Log_Requires_Positive_Low()
    {
        var act = () => new FloatDistribution(0.0, 1.0, log: true);
        act.Should().Throw<ValueError>()
            .WithMessage("`low > 0` must hold for `log=True`, but got (low=0.0, high=1.0).");
    }

    [TestMethod]
    public void Float_Step_Adjusts_High_Like_Python_Decimal()
    {
        // Python: FloatDistribution(0.0, 1.0, step=0.3) -> high replaced by 0.9 (Decimal math).
        var d = new FloatDistribution(0.0, 1.0, step: 0.3);
        d.High.Should().Be(0.9);
        // And a divisible range is untouched despite binary-float noise:
        new FloatDistribution(0.1, 1.0, step: 0.1).High.Should().Be(1.0);
    }

    [TestMethod]
    public void Float_Single_Uses_Decimal_Step_Comparison()
    {
        new FloatDistribution(1.0, 1.0).Single().Should().BeTrue();
        new FloatDistribution(1.0, 2.0).Single().Should().BeFalse();
        new FloatDistribution(0.0, 0.05, step: 0.1).Single().Should().BeTrue("high-low < step");
        new FloatDistribution(0.0, 0.1, step: 0.1).Single().Should().BeFalse();
    }

    [TestMethod]
    public void Float_Contains_Respects_Step_Grid()
    {
        var d = new FloatDistribution(0.0, 1.0, step: 0.1);
        d.Contains(0.30000000000000004).Should().BeTrue("k rounds within 1e-8");
        d.Contains(0.35).Should().BeFalse();
        d.Contains(-0.1).Should().BeFalse();
    }

    [TestMethod]
    public void Float_ToInternalRepr_Validates()
    {
        var d = new FloatDistribution(1, 10);
        d.ToInternalRepr(2.5).Should().Be(2.5);
        d.ToInternalRepr(3L).Should().Be(3.0);
        d.ToInternalRepr("4.5").Should().Be(4.5, "Python float() accepts numeric strings");
        d.ToInternalRepr(true).Should().Be(1.0, "Python bool is float-castable");

        var nan = () => d.ToInternalRepr(double.NaN);
        nan.Should().Throw<ValueError>().WithMessage("`nan` is invalid value.");

        var invalid = () => d.ToInternalRepr(null);
        invalid.Should().Throw<ValueError>()
            .WithMessage("'None' is not a valid type. float-castable value is expected.");

        var logDist = new FloatDistribution(0.1, 10, log: true);
        var nonPositive = () => logDist.ToInternalRepr(-1.0);
        nonPositive.Should().Throw<ValueError>()
            .WithMessage("`-1.0` is invalid value for the case log=True.");
    }

    // ------------------------------------------------------------------ IntDistribution

    [TestMethod]
    public void Int_Rejects_Log_With_Step()
    {
        var act = () => new IntDistribution(1, 10, log: true, step: 2);
        act.Should().Throw<ValueError>()
            .WithMessage("Samplers and other components in Optuna only accept step is 1 when `log` argument is True.");
    }

    [TestMethod]
    public void Int_Log_Requires_Low_AtLeast_One()
    {
        var act = () => new IntDistribution(0, 10, log: true);
        act.Should().Throw<ValueError>()
            .WithMessage("`low >= 1` must hold for `log=True`, but got (low=0, high=10).");
    }

    [TestMethod]
    public void Int_Step_Adjusts_High()
    {
        // Python: IntDistribution(0, 10, step=3) -> high becomes 9.
        var d = new IntDistribution(0, 10, step: 3);
        d.High.Should().Be(9);
    }

    [TestMethod]
    public void Int_External_Repr_Is_Long()
    {
        var d = new IntDistribution(1, 10);
        d.ToExternalRepr(7.0).Should().BeOfType<long>().And.Be(7L);
    }

    [TestMethod]
    public void Int_Single_And_Contains()
    {
        new IntDistribution(3, 3).Single().Should().BeTrue();
        new IntDistribution(1, 10).Single().Should().BeFalse();
        new IntDistribution(0, 4, step: 5).Single().Should().BeTrue("high adjusted to 0");

        var d = new IntDistribution(0, 10, step: 2);
        d.Contains(4.0).Should().BeTrue();
        d.Contains(5.0).Should().BeFalse();
        d.Contains(12.0).Should().BeFalse();
    }

    // ------------------------------------------------------------------ CategoricalDistribution

    [TestMethod]
    public void Categorical_Requires_Choices()
    {
        var act = () => new CategoricalDistribution(System.Array.Empty<CategoricalChoice>());
        act.Should().Throw<ValueError>()
            .WithMessage("The `choices` must contain one or more elements.");
    }

    [TestMethod]
    public void Categorical_InternalRepr_Is_Index_With_Python_Equality()
    {
        var d = new CategoricalDistribution(new CategoricalChoice[] { true, 2L, "x", double.NaN });
        d.ToInternalRepr(true).Should().Be(0);
        d.ToInternalRepr(1L).Should().Be(0, "Python: True == 1 — the documented index-collision quirk");
        d.ToInternalRepr(2.0).Should().Be(1, "1.0-vs-1 style numeric equality");
        d.ToInternalRepr("x").Should().Be(2);
        d.ToInternalRepr(double.NaN).Should().Be(3, "the nan-equal fallback loop");

        var missing = () => d.ToInternalRepr("missing");
        missing.Should().Throw<ValueError>()
            .WithMessage("'missing' not in (True, 2, 'x', nan).");
    }

    [TestMethod]
    public void Categorical_ExternalRepr_Returns_Choice_Value()
    {
        var d = new CategoricalDistribution(new CategoricalChoice[] { "a", "b" });
        d.ToExternalRepr(1.0).Should().Be("b");
        d.ChoiceAt(0).Should().Be((CategoricalChoice)"a");
    }

    // ------------------------------------------------------------------ equality / repr

    [TestMethod]
    public void Equality_Requires_Exact_Type_And_Attributes()
    {
        new FloatDistribution(1, 2).Should().Be(new FloatDistribution(1, 2));
        new FloatDistribution(1, 2).Should().NotBe(new FloatDistribution(1, 2, log: true));
        new FloatDistribution(1, 2).Should().NotBe(new IntDistribution(1, 2));
#pragma warning disable CS0618
        // Python: type(self) is not type(other) -> deprecated subclasses never equal the base.
        new FloatDistribution(1, 2).Should().NotBe(new UniformDistribution(1, 2));
#pragma warning restore CS0618

        var a = new CategoricalDistribution(new CategoricalChoice[] { double.NaN, 1L });
        var b = new CategoricalDistribution(new CategoricalChoice[] { double.NaN, 1L });
        a.Should().Be(b, "categorical equality treats NaN choices as equal");
    }

    [TestMethod]
    public void Repr_Matches_Python_Format()
    {
        // Python: repr sorts _asdict() keys.
        new FloatDistribution(1, 10).ToString()
            .Should().Be("FloatDistribution(high=10.0, log=False, low=1.0, step=None)");
        new IntDistribution(1, 10, log: true).ToString()
            .Should().Be("IntDistribution(high=10, log=True, low=1, step=1)");
        new CategoricalDistribution(new CategoricalChoice[] { "a", 1L, CategoricalChoice.None }).ToString()
            .Should().Be("CategoricalDistribution(choices=('a', 1, None))");
    }

    // ------------------------------------------------------------------ compatibility

    [TestMethod]
    public void CheckCompatibility_Enforces_Kind_Log_And_Choices()
    {
        var kind = () => Distribution.CheckDistributionCompatibility(
            new FloatDistribution(1, 2), new IntDistribution(1, 2));
        kind.Should().Throw<ValueError>()
            .WithMessage("Cannot set different distribution kind to the same parameter name.");

        var log = () => Distribution.CheckDistributionCompatibility(
            new FloatDistribution(1, 2), new FloatDistribution(1, 2, log: true));
        log.Should().Throw<ValueError>()
            .WithMessage("Cannot set different log configuration to the same parameter name.");

        var choices = () => Distribution.CheckDistributionCompatibility(
            new CategoricalDistribution(new CategoricalChoice[] { "a" }),
            new CategoricalDistribution(new CategoricalChoice[] { "b" }));
        choices.Should().Throw<ValueError>()
            .WithMessage("CategoricalDistribution does not support dynamic value space.");

        // Compatible pairs pass silently (dynamic numeric ranges allowed).
        Distribution.CheckDistributionCompatibility(
            new FloatDistribution(0, 1), new FloatDistribution(-5, 5));
    }
}

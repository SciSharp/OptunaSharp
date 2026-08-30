using AwesomeAssertions;
using OptunaSharp.Distributions;

namespace OptunaSharp.Tests.Distributions;

/// <summary>
///     Pins <see cref="CategoricalChoice" />'s Python <c>==</c> semantics — the plan §2.3 ground
///     rule and "the one place C# object equality silently diverges" (risk register). Each case
///     mirrors a CPython evaluation.
/// </summary>
[TestClass]
public class CategoricalChoiceTests
{
    [TestMethod]
    public void Bool_Equals_Int_And_Float_Like_Python()
    {
        // Python: True == 1 == 1.0; False == 0 == 0.0.
        ((CategoricalChoice)true).Should().Be((CategoricalChoice)1L);
        ((CategoricalChoice)true).Should().Be((CategoricalChoice)1.0);
        ((CategoricalChoice)false).Should().Be((CategoricalChoice)0L);
        ((CategoricalChoice)false).Should().Be((CategoricalChoice)0.0);
        ((CategoricalChoice)true).Should().NotBe((CategoricalChoice)2L);
    }

    [TestMethod]
    public void Int_Equals_Float_When_Numerically_Equal()
    {
        ((CategoricalChoice)1L == (CategoricalChoice)1.0).Should().BeTrue();
        ((CategoricalChoice)1L == (CategoricalChoice)1.5).Should().BeFalse();
        ((CategoricalChoice)(-3L) == (CategoricalChoice)(-3.0)).Should().BeTrue();
    }

    [TestMethod]
    public void NaN_NotEqual_To_NaN_Under_DoubleEquals()
    {
        // Python: float("nan") == float("nan") -> False.
        ((CategoricalChoice)double.NaN == (CategoricalChoice)double.NaN).Should().BeFalse();
    }

    [TestMethod]
    public void ChoicesEqual_Treats_NaN_As_Equal()
    {
        // Python _categorical_choice_equal: nan == nan for storage lookups.
        CategoricalChoice.ChoicesEqual(double.NaN, double.NaN).Should().BeTrue();
        CategoricalChoice.ChoicesEqual(double.NaN, 1.0).Should().BeFalse();
        CategoricalChoice.ChoicesEqual("a", "a").Should().BeTrue();
    }

    [TestMethod]
    public void CrossType_Number_Vs_String_Is_False()
    {
        ((CategoricalChoice)1L == (CategoricalChoice)"1").Should().BeFalse();
        ((CategoricalChoice)"True" == (CategoricalChoice)true).Should().BeFalse();
    }

    [TestMethod]
    public void None_Equals_Only_None()
    {
        (CategoricalChoice.None == CategoricalChoice.None).Should().BeTrue();
        (CategoricalChoice.None == (CategoricalChoice)0L).Should().BeFalse();
        (CategoricalChoice.None == (CategoricalChoice)false).Should().BeFalse();
        (CategoricalChoice.None == (CategoricalChoice)"").Should().BeFalse();
    }

    [TestMethod]
    public void Strings_Compare_Ordinal()
    {
        ((CategoricalChoice)"abc" == (CategoricalChoice)"abc").Should().BeTrue();
        ((CategoricalChoice)"ABC" == (CategoricalChoice)"abc").Should().BeFalse();
    }

    [TestMethod]
    public void Equal_Values_Hash_Alike()
    {
        // Python: hash(True) == hash(1) == hash(1.0).
        ((CategoricalChoice)true).GetHashCode().Should().Be(((CategoricalChoice)1L).GetHashCode());
        ((CategoricalChoice)1L).GetHashCode().Should().Be(((CategoricalChoice)1.0).GetHashCode());
    }

    [TestMethod]
    public void From_Normalizes_Numerics()
    {
        CategoricalChoice.From(3).Kind.Should().Be(CategoricalChoiceKind.Int);
        CategoricalChoice.From(3.0f).Kind.Should().Be(CategoricalChoiceKind.Float);
        CategoricalChoice.From(null).IsNone.Should().BeTrue();
        CategoricalChoice.From((byte)7).Value.Should().Be(7L);
    }

    [TestMethod]
    public void ToString_Renders_Python_Str()
    {
        CategoricalChoice.None.ToString().Should().Be("None");
        ((CategoricalChoice)true).ToString().Should().Be("True");
        ((CategoricalChoice)2.5).ToString().Should().Be("2.5");
        ((CategoricalChoice)"x").ToString().Should().Be("x");
        ((CategoricalChoice)"x").Repr().Should().Be("'x'");
    }
}

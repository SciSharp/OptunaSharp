using AwesomeAssertions;
using OptunaSharp.Collections;
using OptunaSharp.Internal;

namespace OptunaSharp.Tests.Internal;

/// <summary>
///     Pins <see cref="PythonRepr" /> to CPython's exact <c>repr()</c> output. Every expectation
///     below is copy-pasted from a live CPython 3.12 session — this formatting backs verbatim
///     error messages and the byte-compatible JSON float encoding, so it must not drift.
/// </summary>
[TestClass]
public class PythonReprTests
{
    [TestMethod]
    [DataRow(1.0, "1.0")]
    [DataRow(-1.0, "-1.0")]
    [DataRow(0.1, "0.1")]
    [DataRow(1e16, "1e+16")]
    [DataRow(1e15, "1000000000000000.0")]
    [DataRow(9999999999999998.0, "9999999999999998.0")]
    [DataRow(1e-4, "0.0001")]
    [DataRow(1e-5, "1e-05")]
    [DataRow(123456789012345.6, "123456789012345.6")]
    [DataRow(1.5e16, "1.5e+16")]
    [DataRow(-0.0, "-0.0")]
    [DataRow(5e-324, "5e-324")]
    [DataRow(1.7976931348623157e308, "1.7976931348623157e+308")]
    [DataRow(2.5, "2.5")]
    [DataRow(0.3333333333333333, "0.3333333333333333")]
    [DataRow(3.141592653589793, "3.141592653589793")]
    [DataRow(100.0, "100.0")]
    [DataRow(1e100, "1e+100")]
    [DataRow(-2.5e-10, "-2.5e-10")]
    [DataRow(double.NaN, "nan")]
    [DataRow(double.PositiveInfinity, "inf")]
    [DataRow(double.NegativeInfinity, "-inf")]
    [DataRow(0.0, "0.0")]
    public void FloatRepr_Matches_CPython(double value, string expected)
    {
        PythonRepr.FloatRepr(value).Should().Be(expected);
    }

    [TestMethod]
    public void FloatRepr_RoundTrips_RandomDoubles()
    {
        // The repr contract: shortest string that parses back to the identical double.
        var rng = new System.Random(1234);
        for (int i = 0; i < 5000; i++)
        {
            long bits = ((long)rng.Next() << 32) | (uint)rng.Next();
            double value = System.BitConverter.Int64BitsToDouble(bits);
            if (double.IsNaN(value) || double.IsInfinity(value))
                continue;
            string repr = PythonRepr.FloatRepr(value);
            double.Parse(repr, System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture)
                .Should().Be(value, $"repr {repr} must round-trip");
        }
    }

    [TestMethod]
    public void Repr_Renders_PythonLiterals()
    {
        PythonRepr.Repr(null).Should().Be("None");
        PythonRepr.Repr(true).Should().Be("True");
        PythonRepr.Repr(false).Should().Be("False");
        PythonRepr.Repr(42L).Should().Be("42");
        PythonRepr.Repr(1.0).Should().Be("1.0");
        PythonRepr.Repr("abc").Should().Be("'abc'");
        PythonRepr.Repr("it's").Should().Be("\"it's\"");
        PythonRepr.Repr("both'\"").Should().Be("'both\\'\"'");
    }

    [TestMethod]
    public void Str_IsRawForStrings_ReprOtherwise()
    {
        PythonRepr.Str("abc").Should().Be("abc");
        PythonRepr.Str(1.5).Should().Be("1.5");
        PythonRepr.Str(null).Should().Be("None");
        PythonRepr.Str(true).Should().Be("True");
    }

    [TestMethod]
    public void Collection_Reprs_Match_Python()
    {
        PythonRepr.ListRepr(new object?[] { 1L, 2.5, "x" }).Should().Be("[1, 2.5, 'x']");
        PythonRepr.TupleRepr(new object?[] { 1L, 2.5, "x" }).Should().Be("(1, 2.5, 'x')");
        PythonRepr.TupleRepr(new object?[] { "only" }).Should().Be("('only',)");
        PythonRepr.TupleRepr(new object?[] { }).Should().Be("()");

        var dict = new OrderedDict<string, object?> { ["b"] = 1L, ["a"] = null };
        PythonRepr.DictRepr(dict).Should().Be("{'b': 1, 'a': None}");
        PythonRepr.SetRepr(new object?[] { }).Should().Be("set()");
        PythonRepr.SetRepr(new object?[] { "x" }).Should().Be("{'x'}");
    }
}

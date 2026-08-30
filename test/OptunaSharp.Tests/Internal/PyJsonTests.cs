using System.Collections.Generic;
using AwesomeAssertions;
using OptunaSharp.Collections;
using OptunaSharp.Internal;

namespace OptunaSharp.Tests.Internal;

/// <summary>
///     Pins <see cref="PyJson" /> to CPython <c>json</c> module semantics — the byte-compat
///     basis for the journal-file format and <c>distribution_to_json</c>.
/// </summary>
[TestClass]
public class PyJsonTests
{
    [TestMethod]
    public void Dumps_Uses_Python_Default_Separators_And_FloatRepr()
    {
        var payload = new OrderedDict<string, object?>
        {
            ["a"] = 1.0,
            ["b"] = 1L,
            ["c"] = true,
            ["d"] = null,
            ["e"] = "x",
        };
        PyJson.Dumps(payload).Should().Be("{\"a\": 1.0, \"b\": 1, \"c\": true, \"d\": null, \"e\": \"x\"}");
    }

    [TestMethod]
    public void DumpsCompact_Uses_Journal_Separators()
    {
        var payload = new OrderedDict<string, object?>
        {
            ["op_code"] = 0L,
            ["directions"] = new List<object?> { 1L, 2L },
        };
        PyJson.DumpsCompact(payload).Should().Be("{\"op_code\":0,\"directions\":[1,2]}");
    }

    [TestMethod]
    public void Dumps_Writes_NonStandard_Nan_Infinity_Literals()
    {
        // Python: json.dumps(float("nan")) -> "NaN" (allow_nan=True default).
        PyJson.Dumps(double.NaN).Should().Be("NaN");
        PyJson.Dumps(double.PositiveInfinity).Should().Be("Infinity");
        PyJson.Dumps(double.NegativeInfinity).Should().Be("-Infinity");
    }

    [TestMethod]
    public void Loads_Parses_Nan_Infinity_Literals()
    {
        PyJson.Loads("NaN").Should().Be(double.NaN);
        PyJson.Loads("Infinity").Should().Be(double.PositiveInfinity);
        PyJson.Loads("-Infinity").Should().Be(double.NegativeInfinity);
        PyJson.Loads("[NaN, -Infinity]").Should().BeOfType<List<object?>>();
    }

    [TestMethod]
    public void Dumps_EnsureAscii_Escapes_NonAscii()
    {
        // Python default ensure_ascii=True: json.dumps("héllo") -> "\"h\\u00e9llo\"".
        PyJson.Dumps("héllo").Should().Be("\"h\\u00e9llo\"");
        PyJson.Dumps("日本").Should().Be("\"\\u65e5\\u672c\"");
    }

    [TestMethod]
    public void Loads_Preserves_Object_Key_Order()
    {
        var parsed = (OrderedDict<string, object?>)PyJson.Loads("{\"z\": 1, \"a\": 2, \"m\": 3}")!;
        parsed.Keys.Should().Equal("z", "a", "m");
    }

    [TestMethod]
    public void Loads_Integers_As_Long_And_Reals_As_Double()
    {
        PyJson.Loads("5").Should().BeOfType<long>().And.Be(5L);
        PyJson.Loads("5.0").Should().BeOfType<double>().And.Be(5.0);
        PyJson.Loads("1e3").Should().BeOfType<double>().And.Be(1000.0);
        PyJson.Loads("-7").Should().Be(-7L);
    }

    [TestMethod]
    public void RoundTrip_Preserves_Structure_And_Float_Bits()
    {
        var payload = new OrderedDict<string, object?>
        {
            ["values"] = new List<object?> { 0.1, 1e-05, 9999999999999998.0, -0.0 },
            ["nested"] = new OrderedDict<string, object?> { ["k"] = "v\n\"q\"" },
        };
        string json = PyJson.Dumps(payload);
        var back = (OrderedDict<string, object?>)PyJson.Loads(json)!;
        PyJson.ValueEquals(payload, back).Should().BeTrue();
        var values = (List<object?>)back["values"]!;
        System.BitConverter.DoubleToInt64Bits((double)values[3]!)
            .Should().Be(System.BitConverter.DoubleToInt64Bits(-0.0), "-0.0 must survive the round trip");
    }

    [TestMethod]
    public void Loads_Handles_Escapes_And_Surrogate_Pairs()
    {
        PyJson.Loads("\"a\\n\\t\\\"b\\\\\"").Should().Be("a\n\t\"b\\");
        // 😀 U+1F600 as surrogate pair.
        PyJson.Loads("\"\\ud83d\\ude00\"").Should().Be("😀");
        PyJson.Dumps("😀").Should().Be("\"\\ud83d\\ude00\"");
    }

    [TestMethod]
    public void IntKeys_Stringify_Like_Python()
    {
        var intermediate = new OrderedDict<int, double> { [0] = 1.5, [3] = double.NaN };
        PyJson.Dumps(intermediate).Should().Be("{\"0\": 1.5, \"3\": NaN}");
    }

    [TestMethod]
    public void Enums_Serialize_As_Ints()
    {
        PyJson.Dumps(new List<object?> { StudyDirection.Minimize, TrialState.Complete })
            .Should().Be("[1, 1]");
    }

    [TestMethod]
    public void DeepCopy_Isolates_Nested_Containers()
    {
        var original = new OrderedDict<string, object?>
        {
            ["list"] = new List<object?> { 1L },
        };
        var copy = (OrderedDict<string, object?>)PyJson.DeepCopy(original)!;
        ((List<object?>)copy["list"]!).Add(2L);
        ((List<object?>)original["list"]!).Should().HaveCount(1);
    }
}

using System;
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
        var intermediate = new OrderedDict<long, double> { [0] = 1.5, [3] = double.NaN };
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

    /// <summary>
    ///     Review item P2-2: <see cref="PyJson.Loads" /> accepts nesting up to <see cref="PyJson.MaxDepth" /> and
    ///     refuses one level more with the <see cref="NumSharp.RuntimeError" /> text CPython's <c>RecursionError</c>
    ///     carries for the same kind of container. Before the fix there was no limit, and a few thousand levels
    ///     overflowed the stack, which ends a .NET process instead of throwing.
    /// </summary>
    /// <param name="open">The opening of one nesting level.</param>
    /// <param name="innermost">The innermost container, written once.</param>
    /// <param name="close">The closing of one nesting level.</param>
    /// <param name="message">CPython's <c>RecursionError</c> message for this container kind.</param>
    [TestMethod]
    [DataRow("[", "[]", "]", "maximum recursion depth exceeded while decoding a JSON array from a unicode string")]
    [DataRow("{\"k\":", "{}", "}", "maximum recursion depth exceeded while decoding a JSON object from a unicode string")]
    public void Loads_NestingBeyondMaxDepth_RaisesPythonsRecursionError(
        string open, string innermost, string close, string message)
    {
        string Nested(int depth) => string.Concat(System.Linq.Enumerable.Repeat(open, depth - 1)) + innermost +
                                    string.Concat(System.Linq.Enumerable.Repeat(close, depth - 1));

        PyJson.Loads(Nested(PyJson.MaxDepth)).Should().NotBeNull("the limit itself is accepted");
        var tooDeep = () => PyJson.Loads(Nested(PyJson.MaxDepth + 1));

        tooDeep.Should().Throw<NumSharp.RuntimeError>().Which.Message.Should().Be(message);
    }

    /// <summary>
    ///     The pass-2B repro sizes: a 100 KB document of 50,000 nested arrays, and a 5,000-deep object, fail with a
    ///     catchable error instead of a stack overflow (which used to end the test host itself).
    /// </summary>
    [TestMethod]
    public void Loads_HostileDepth_FailsCatchably()
    {
        var arrays = () => PyJson.Loads(new string('[', 50_000) + new string(']', 50_000));
        var objects = () => PyJson.Loads(
            string.Concat(System.Linq.Enumerable.Repeat("{\"k\":", 5_000)) + "1" + new string('}', 5_000));

        arrays.Should().Throw<NumSharp.RuntimeError>();
        objects.Should().Throw<NumSharp.RuntimeError>();
    }

    /// <summary>
    ///     The stack guard behind the depth limit: on a thread with a 256 KB stack, a document at the depth limit needs
    ///     more stack than the thread has, and the parse must end in a catchable <see cref="NumSharp.RuntimeError" />
    ///     (or succeed, on a platform whose frames are small enough), never in a stack overflow that ends the process.
    /// </summary>
    [TestMethod]
    public void Loads_OnASmallStack_FailsCatchablyInsteadOfOverflowing()
    {
        string document = new string('[', PyJson.MaxDepth) + new string(']', PyJson.MaxDepth);
        Exception? failure = null;
        bool parsed = false;
        var thread = new System.Threading.Thread(() =>
        {
            try
            {
                parsed = PyJson.Loads(document) is not null;
            }
            catch (Exception e)
            {
                failure = e;
            }
        }, maxStackSize: 256 * 1024);

        thread.Start();
        thread.Join();

        if (!parsed)
            failure.Should().BeOfType<NumSharp.RuntimeError>()
                .Which.Message.Should().StartWith("maximum recursion depth exceeded");
    }

    /// <summary>
    ///     The writers, the deep copy and the comparison walk the same nesting and stop at the same depth, each with
    ///     the wording CPython uses for the operation: its C encoder says "while encoding a JSON object" for lists
    ///     too, its pure-Python indenting encoder and <c>copy.deepcopy</c> add no suffix, and <c>==</c> says
    ///     "in comparison". A value built in memory can be nested arbitrarily deep, so each walk needs its own guard.
    /// </summary>
    [TestMethod]
    public void Writers_DeepCopy_And_ValueEquals_StopAtMaxDepth()
    {
        object NestedList(int depth)
        {
            var value = new List<object?>();
            for (int i = 1; i < depth; i++)
                value = new List<object?> { value };
            return value;
        }

        object atLimit = NestedList(PyJson.MaxDepth);
        object tooDeep = NestedList(PyJson.MaxDepth + 1);

        PyJson.Dumps(atLimit).Should().HaveLength(2 * PyJson.MaxDepth);
        PyJson.DumpsCompact(atLimit).Should().HaveLength(2 * PyJson.MaxDepth);
        PyJson.DumpsIndented(atLimit, 0).Should().StartWith("[\n[\n");
        PyJson.ValueEquals(PyJson.DeepCopy(atLimit), atLimit).Should().BeTrue();

        ((System.Action)(() => PyJson.Dumps(tooDeep))).Should().Throw<NumSharp.RuntimeError>()
            .WithMessage("maximum recursion depth exceeded while encoding a JSON object");
        ((System.Action)(() => PyJson.DumpsCompact(tooDeep))).Should().Throw<NumSharp.RuntimeError>()
            .WithMessage("maximum recursion depth exceeded while encoding a JSON object");
        ((System.Action)(() => PyJson.DumpsIndented(tooDeep, 2))).Should().Throw<NumSharp.RuntimeError>()
            .WithMessage("maximum recursion depth exceeded");
        ((System.Action)(() => PyJson.DeepCopy(tooDeep))).Should().Throw<NumSharp.RuntimeError>()
            .WithMessage("maximum recursion depth exceeded");
        ((System.Action)(() => PyJson.ValueEquals(tooDeep, NestedList(PyJson.MaxDepth + 1)))).Should()
            .Throw<NumSharp.RuntimeError>().WithMessage("maximum recursion depth exceeded in comparison");
    }
}

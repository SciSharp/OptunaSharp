using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using AwesomeAssertions;
using OptunaSharp.Collections;
using OptunaSharp.Internal;

namespace OptunaSharp.Tests.Internal;

/// <summary>
///     Holds <see cref="PyJsonUtf8" />, the journal's UTF-8 reader (plan 10, item 2.7), to its contract: for any bytes it
///     returns exactly what <c>PyJson.Loads(Encoding.UTF8.GetString(bytes))</c> returns, with the same runtime types,
///     key order and value bits, or throws the same exception type with the same message.
/// </summary>
/// <remarks>
///     <para>
///         The owner's condition for this item was "very high parity test coverage", so every case is differential:
///         <see cref="PyJson" /> is the oracle, and equality is strict (<see cref="Same" />): a <see cref="long" /> must
///         stay a <see cref="long" />, a <see cref="double" /> must match bit for bit (NaN payloads and -0.0 included),
///         and dictionaries must list their keys in the same order.
///     </para>
///     <para>
///         The reader may hand a document to PyJson instead of finishing it, so a passing comparison alone could hide a
///         fast path that never runs. Valid documents therefore also assert that the fast reader itself accepted them
///         (<see cref="PyJsonUtf8Reader.TryParse" />), and documents PyJson refuses assert that the fast reader refused
///         them too: it may decline what PyJson accepts, but never accept what PyJson refuses.
///     </para>
/// </remarks>
[TestClass]
public class PyJsonUtf8Tests
{
    /// <summary>The outcome of one parse: a value, or an exception's type and message.</summary>
    /// <param name="Value">The parsed value when <paramref name="Error" /> is null.</param>
    /// <param name="Error">The exception's type and message, or null.</param>
    private sealed record Outcome(object? Value, string? Error);

    /// <summary>Runs a parse and captures its value or its exception.</summary>
    /// <param name="parse">The parse.</param>
    /// <returns>The outcome.</returns>
    private static Outcome Capture(Func<object?> parse)
    {
        try
        {
            return new Outcome(parse(), null);
        }
        catch (Exception e)
        {
            return new Outcome(null, e.GetType().FullName + ": " + e.Message);
        }
    }

    /// <summary>
    ///     Strict equality over PyJson's object model: same runtime types, same key order, doubles compared by their
    ///     bits.
    /// </summary>
    /// <param name="a">The first value.</param>
    /// <param name="b">The second value.</param>
    /// <returns>Whether the two are the same value in every observable respect but object identity.</returns>
    /// <exception cref="InvalidOperationException">A value outside PyJson's object model.</exception>
    internal static bool Same(object? a, object? b)
    {
        switch (a)
        {
            case null:
                return b is null;
            case bool ab:
                return b is bool bb && ab == bb;
            case long al:
                return b is long bl && al == bl;
            case double ad:
                return b is double bd && BitConverter.DoubleToInt64Bits(ad) == BitConverter.DoubleToInt64Bits(bd);
            case string astr:
                return b is string bstr && string.Equals(astr, bstr, StringComparison.Ordinal);
            case List<object?> alist:
                if (b is not List<object?> blist || alist.Count != blist.Count)
                    return false;
                for (int i = 0; i < alist.Count; i++)
                {
                    if (!Same(alist[i], blist[i]))
                        return false;
                }
                return true;
            case OrderedDict<string, object?> adict:
                if (b is not OrderedDict<string, object?> bdict || adict.Count != bdict.Count)
                    return false;
                using (var ea = adict.GetEnumerator())
                using (var eb = bdict.GetEnumerator())
                {
                    while (ea.MoveNext() && eb.MoveNext())
                    {
                        if (!string.Equals(ea.Current.Key, eb.Current.Key, StringComparison.Ordinal)
                            || !Same(ea.Current.Value, eb.Current.Value))
                        {
                            return false;
                        }
                    }
                }
                return true;
            default:
                throw new InvalidOperationException($"Not PyJson's object model: {a.GetType()}");
        }
    }

    /// <summary>
    ///     The differential check behind every case: the reader's outcome equals PyJson's, and the fast reader accepted
    ///     exactly when it had to.
    /// </summary>
    /// <param name="utf8">The document's bytes.</param>
    /// <param name="fastMustAccept">
    ///     Whether the fast reader itself must parse the document (the journal's and the snapshot's documents); when
    ///     false it may decline a document PyJson accepts.
    /// </param>
    /// <returns>PyJson's outcome, for further checks.</returns>
    private static Outcome AssertSameAsPyJson(byte[] utf8, bool fastMustAccept)
    {
        Outcome expected = Capture(() => PyJson.Loads(Encoding.UTF8.GetString(utf8)));
        Outcome actual = Capture(() => PyJsonUtf8.Loads(utf8));
        bool fastAccepted = PyJsonUtf8Reader.TryParse(utf8, out object? fastValue);

        string shown = Describe(utf8);
        actual.Error.Should().Be(expected.Error, "the reader must fail exactly as PyJson does on {0}", shown);
        if (expected.Error is null)
        {
            Same(expected.Value, actual.Value).Should().BeTrue("the reader must return PyJson's value for {0}", shown);
            if (fastAccepted)
                Same(expected.Value, fastValue).Should().BeTrue("the fast reader's own value must be PyJson's for {0}", shown);
            if (fastMustAccept)
                fastAccepted.Should().BeTrue("the fast reader must parse {0} itself", shown);
        }
        else
        {
            fastAccepted.Should().BeFalse("the fast reader must not accept what PyJson refuses: {0}", shown);
        }
        return expected;
    }

    /// <summary>A short, printable description of a document for assertion messages.</summary>
    /// <param name="utf8">The document's bytes.</param>
    /// <returns>Its hexadecimal bytes when short, else its length and first bytes.</returns>
    private static string Describe(byte[] utf8)
        => utf8.Length <= 48
            ? "[" + Convert.ToHexString(utf8) + "]"
            : $"[{utf8.Length} bytes: {Convert.ToHexString(utf8, 0, 48)}...]";

    /// <summary>Every scalar literal and number form PyJson accepts parses the same, as the fast reader's own result.</summary>
    /// <param name="text">The document.</param>
    [TestMethod]
    [DataRow("null")]
    [DataRow("true")]
    [DataRow("false")]
    [DataRow("NaN")]
    [DataRow("Infinity")]
    [DataRow("-Infinity")]
    [DataRow("0")]
    [DataRow("-0")]
    [DataRow("-0.0")]
    [DataRow("0.0")]
    [DataRow("1.0")]
    [DataRow("1e5")]
    [DataRow("1E5")]
    [DataRow("1e+5")]
    [DataRow("1.5e-3")]
    [DataRow("1e400")]
    [DataRow("-1e400")]
    [DataRow("1e-400")]
    [DataRow("5e-324")]
    [DataRow("2.2250738585072014e-308")]
    [DataRow("2.225073858507201e-308")]
    [DataRow("1.7976931348623157e+308")]
    [DataRow("0.1")]
    [DataRow("0.30000000000000004")]
    [DataRow("123456789012345680000.0")]
    [DataRow("9223372036854775807")]
    [DataRow("-9223372036854775808")]
    [DataRow("9223372036854775808")]
    [DataRow("-9223372036854775809")]
    [DataRow("123456789012345678901234567890")]
    [DataRow("-128")]
    [DataRow("-129")]
    [DataRow("1023")]
    [DataRow("1024")]
    [DataRow("\"\"")]
    [DataRow("\"text\"")]
    [DataRow("  \t\r\n  42  \n")]
    [DataRow("[]")]
    [DataRow("{}")]
    [DataRow("[ ]")]
    [DataRow("{ }")]
    public void Scalars_AndNumberForms_ParseAsPyJson(string text)
        => AssertSameAsPyJson(Encoding.UTF8.GetBytes(text), fastMustAccept: true);

    /// <summary>
    ///     Forms Python's own parser refuses but PyJson's lenient number scan accepts (a plus sign, leading zeros), and the
    ///     ones .NET's parser refuses inside PyJson, give PyJson's outcome either way.
    /// </summary>
    /// <param name="text">The document.</param>
    [TestMethod]
    [DataRow("+5")]
    [DataRow("01")]
    [DataRow("-01")]
    [DataRow("00.5")]
    [DataRow(".5")]
    [DataRow("5.")]
    [DataRow("1.2.3")]
    [DataRow("1-2")]
    [DataRow("--5")]
    [DataRow("1e")]
    [DataRow("1e+")]
    [DataRow("e5")]
    [DataRow("-")]
    [DataRow("+")]
    [DataRow(".")]
    [DataRow("1ee5")]
    [DataRow("1e5.5")]
    [DataRow("-Infinityx")]
    [DataRow("-I")]
    [DataRow("-In")]
    [DataRow("Inf")]
    [DataRow("nan")]
    [DataRow("NaNa")]
    [DataRow("tru")]
    [DataRow("nul")]
    [DataRow("True")]
    public void LenientAndRefusedNumberForms_GivePyJsonsOutcome(string text)
        => AssertSameAsPyJson(Encoding.UTF8.GetBytes(text), fastMustAccept: false);

    /// <summary>Every escape PyJson knows, and PyJson's lenient <c>\u</c> parse, decode the same.</summary>
    /// <param name="text">The document.</param>
    [TestMethod]
    [DataRow("\"\\\"\\\\\\/\\b\\f\\n\\r\\t\"")]
    [DataRow("\"\\u00e9\\u0041\\u00ff\"")]
    [DataRow("\"\\uD83D\\uDE00\"")]
    [DataRow("\"\\ud83d\\ude00\"")]
    [DataRow("\"\\ud800\"")]
    [DataRow("\"\\udc00 lone low\"")]
    [DataRow("\"\\ude00\\ud83d reversed\"")]
    [DataRow("\"\\u0000\\u001f\"")]
    [DataRow("\"tail\\u0041\"")]
    [DataRow("\"\\\"start\"")]
    [DataRow("\"a\\nb\\nc\\\\d\"")]
    public void Escapes_DecodeAsPyJson(string text)
        => AssertSameAsPyJson(Encoding.UTF8.GetBytes(text), fastMustAccept: true);

    /// <summary>
    ///     Escape forms PyJson reads leniently (whitespace inside <c>\u</c>, which .NET's hexadecimal parse allows) or
    ///     refuses (short, non-hexadecimal, unknown escapes, unterminated text) give PyJson's outcome.
    /// </summary>
    /// <param name="text">The document.</param>
    [TestMethod]
    [DataRow("\"\\u 12a\"")]
    [DataRow("\"\\u12a \"")]
    [DataRow("\"\\u1 2a\"")]
    [DataRow("\"\\u12\"")]
    [DataRow("\"\\u12")]
    [DataRow("\"\\uXYZW\"")]
    [DataRow("\"\\x41\"")]
    [DataRow("\"\\a\"")]
    [DataRow("\"\\")]
    [DataRow("\"abc")]
    [DataRow("\"")]
    [DataRow("\"\\u00e9")]
    public void LenientAndRefusedEscapes_GivePyJsonsOutcome(string text)
        => AssertSameAsPyJson(Encoding.UTF8.GetBytes(text), fastMustAccept: false);

    /// <summary>
    ///     Raw UTF-8 text in strings, valid or not, decodes as the whole document's decode does: multi-byte characters
    ///     whole, invalid sequences as U+FFFD, also next to escapes and quotes.
    /// </summary>
    /// <param name="hex">The document's bytes in hexadecimal.</param>
    /// <remarks>
    ///     The rows, in order: "é", "😀", "€"; an invalid continuation byte; a three-byte and a four-byte sequence cut
    ///     short by the closing quote; a byte UTF-8 never uses (FF); an overlong NUL; an encoded surrogate; a cut
    ///     sequence before an escape; an escape before a multi-byte character; é, an escaped A, é; the key "é"; a key
    ///     with a cut sequence; raw control characters, which PyJson accepts.
    /// </remarks>
    [TestMethod]
    [DataRow("22C3A922")]
    [DataRow("22F09F988022")]
    [DataRow("22E282AC22")]
    [DataRow("22C32822")]
    [DataRow("22E28222")]
    [DataRow("22F09F9822")]
    [DataRow("22FF22")]
    [DataRow("22C08022")]
    [DataRow("22EDA08022")]
    [DataRow("22E2825C6E22")]
    [DataRow("225C6EC3A922")]
    [DataRow("22C3A95C7530303431C3A922")]
    [DataRow("7B22C3A9223A317D")]
    [DataRow("7B22E282223A317D")]
    [DataRow("22010203090A0D22")]
    public void RawUtf8InStrings_DecodesAsTheWholeDocument(string hex)
        => AssertSameAsPyJson(Convert.FromHexString(hex), fastMustAccept: true);

    /// <summary>
    ///     Non-ASCII bytes outside strings, a byte order mark (which PyJson does not skip), and invalid bytes there give
    ///     PyJson's errors, whose positions count characters, not bytes.
    /// </summary>
    /// <param name="hex">The document's bytes in hexadecimal.</param>
    /// <remarks>
    ///     The rows, in order: a byte order mark before <c>{}</c>; "é" outside a string; <c>["é", é]</c>;
    ///     <c>["éé",]</c>, whose error column counts characters after multi-byte text; <c>{"é":</c> followed by FF;
    ///     a lone FF; an empty document.
    /// </remarks>
    [TestMethod]
    [DataRow("EFBBBF7B7D")]
    [DataRow("C3A9")]
    [DataRow("5B22C3A9222C20C3A95D")]
    [DataRow("5B22C3A9C3A9222C5D")]
    [DataRow("7B22C3A9223AFF7D")]
    [DataRow("FF")]
    [DataRow("")]
    public void NonAsciiOutsideStrings_GivePyJsonsError(string hex)
        => AssertSameAsPyJson(Convert.FromHexString(hex), fastMustAccept: false);

    /// <summary>Malformed structure of every kind gives PyJson's exact error.</summary>
    /// <param name="text">The document.</param>
    [TestMethod]
    [DataRow("[1,]")]
    [DataRow("[1 2]")]
    [DataRow("[1,")]
    [DataRow("[")]
    [DataRow("]")]
    [DataRow("{\"a\":1,}")]
    [DataRow("{\"a\" 1}")]
    [DataRow("{\"a\":}")]
    [DataRow("{a:1}")]
    [DataRow("{\"a\":1")]
    [DataRow("{\"a\":1 \"b\":2}")]
    [DataRow("{")]
    [DataRow("}")]
    [DataRow("1 2")]
    [DataRow("[] []")]
    [DataRow("{} x")]
    [DataRow("   ")]
    [DataRow(",")]
    public void MalformedStructure_GivesPyJsonsError(string text)
        => AssertSameAsPyJson(Encoding.UTF8.GetBytes(text), fastMustAccept: false);

    /// <summary>
    ///     A repeated key keeps the place of its first occurrence and the value of its last, at every level, as Python's
    ///     dict and PyJson do.
    /// </summary>
    [TestMethod]
    public void RepeatedKeys_KeepFirstPlace_AndLastValue()
    {
        byte[] doc = Encoding.UTF8.GetBytes("{\"a\":1,\"b\":2,\"a\":3,\"c\":{\"x\":1,\"x\":[1,{\"y\":1,\"y\":2}]},\"b\":4}");
        Outcome expected = AssertSameAsPyJson(doc, fastMustAccept: true);
        var top = (OrderedDict<string, object?>)expected.Value!;
        top.Keys.Should().Equal("a", "b", "c");
        top["a"].Should().Be(3L);
        top["b"].Should().Be(4L);
    }

    /// <summary>
    ///     Nesting up to <see cref="PyJson.MaxDepth" /> parses on the fast path; one level deeper fails with PyJson's
    ///     <see cref="NumSharp.RuntimeError" /> and message, for arrays and objects alike.
    /// </summary>
    /// <param name="open">The opening of one level.</param>
    /// <param name="close">The closing of one level.</param>
    [TestMethod]
    [DataRow("[", "]")]
    [DataRow("{\"k\":", "}")]
    public void Depth_UpToTheLimitParses_OneMoreFailsAsPyJson(string open, string close)
    {
        string AtDepth(int depth)
            => string.Concat(Enumerable.Repeat(open, depth)) + "1" + string.Concat(Enumerable.Repeat(close, depth));

        AssertSameAsPyJson(Encoding.UTF8.GetBytes(AtDepth(PyJson.MaxDepth)), fastMustAccept: true);
        Outcome deeper = AssertSameAsPyJson(Encoding.UTF8.GetBytes(AtDepth(PyJson.MaxDepth + 1)), fastMustAccept: false);
        deeper.Error.Should().StartWith(typeof(NumSharp.RuntimeError).FullName + ": maximum recursion depth exceeded");
    }

    /// <summary>
    ///     Journal records of every operation, written the two ways the backends write them (the file journal's compact
    ///     separators, the Redis journal's default ones), parse on the fast path to PyJson's values.
    /// </summary>
    [TestMethod]
    public void JournalRecordsOfEveryKind_ParseOnTheFastPath()
    {
        foreach (OrderedDict<string, object?> record in JournalRecordsOfEveryKind())
        {
            AssertSameAsPyJson(Encoding.UTF8.GetBytes(PyJson.DumpsCompact(record)), fastMustAccept: true);
            AssertSameAsPyJson(Encoding.UTF8.GetBytes(PyJson.Dumps(record)), fastMustAccept: true);
            AssertSameAsPyJson(Encoding.UTF8.GetBytes(PyJson.DumpsIndented(record, 2)), fastMustAccept: true);
        }
    }

    /// <summary>
    ///     Records shaped as Optuna's <c>JournalStorage</c> writes them, one per operation code, with attribute values of
    ///     every kind JSON can carry.
    /// </summary>
    /// <returns>The records.</returns>
    internal static IEnumerable<OrderedDict<string, object?>> JournalRecordsOfEveryKind()
    {
        const string worker = "1c3e4a2b-9f0d-4e3a-8c1b-2d3e4f5a6b7c-140245";
        const string floatDist = "{\"name\": \"FloatDistribution\", \"attributes\": {\"step\": null, \"low\": 0.0, \"high\": 1.0, \"log\": false}}";
        const string intDist = "{\"name\": \"IntDistribution\", \"attributes\": {\"log\": true, \"step\": 1, \"low\": 1, \"high\": 1024}}";
        const string catDist = "{\"name\": \"CategoricalDistribution\", \"attributes\": {\"choices\": [null, true, 1, 2.5, \"a\\u00e9\"]}}";
        var attrValue = new List<object?>
        {
            null, true, false, 0L, -1L, long.MaxValue, long.MinValue, 0.0, -0.0, 1.5, double.NaN,
            double.PositiveInfinity, double.NegativeInfinity, double.Epsilon, 1e300, "text", "\u00e9\u20ac\U0001F600",
            "\ud800", "quote \" backslash \\ newline \n", new List<object?>(),
            new OrderedDict<string, object?> { ["nested"] = new OrderedDict<string, object?> { ["deeper"] = new List<object?> { 1L, "two" } } },
        };

        yield return new() { ["op_code"] = 0L, ["worker_id"] = worker, ["study_name"] = "study \u00e9", ["directions"] = new List<object?> { 1L, 2L } };
        yield return new() { ["op_code"] = 1L, ["worker_id"] = worker, ["study_id"] = 0L };
        yield return new() { ["op_code"] = 2L, ["worker_id"] = worker, ["study_id"] = 0L, ["user_attr"] = new OrderedDict<string, object?> { ["values"] = attrValue } };
        yield return new() { ["op_code"] = 3L, ["worker_id"] = worker, ["study_id"] = 0L, ["system_attr"] = new OrderedDict<string, object?> { ["k"] = "v" } };
        yield return new()
        {
            ["op_code"] = 4L, ["worker_id"] = worker, ["study_id"] = 0L,
            ["datetime_start"] = "2026-11-07T12:34:56.123456+00:00",
            ["datetime_complete"] = null,
            ["distributions"] = new OrderedDict<string, object?> { ["x"] = floatDist, ["k"] = intDist, ["c"] = catDist },
            ["params"] = new OrderedDict<string, object?> { ["x"] = 0.123456789, ["k"] = 16.0, ["c"] = 4.0 },
            ["user_attrs"] = new OrderedDict<string, object?> { ["a"] = attrValue },
            ["system_attrs"] = new OrderedDict<string, object?>(),
            ["intermediate_values"] = new OrderedDict<string, object?> { ["0"] = 1.0, ["9223372036854775807"] = double.NaN },
            ["state"] = 1L, ["value"] = 0.5, ["values"] = null,
        };
        yield return new() { ["op_code"] = 5L, ["worker_id"] = worker, ["trial_id"] = 7L, ["param_name"] = "x", ["param_value_internal"] = 0.25, ["distribution"] = floatDist };
        yield return new() { ["op_code"] = 6L, ["worker_id"] = worker, ["trial_id"] = 7L, ["state"] = 1L, ["values"] = new List<object?> { 0.5, -1e-310 }, ["datetime_complete"] = "2026-11-07T12:34:57.000001+00:00" };
        yield return new() { ["op_code"] = 6L, ["worker_id"] = worker, ["trial_id"] = 8L, ["state"] = 0L, ["values"] = null, ["datetime_start"] = "2026-11-07T12:34:58.999999+00:00" };
        yield return new() { ["op_code"] = 7L, ["worker_id"] = worker, ["trial_id"] = 7L, ["step"] = 12345678901234L, ["intermediate_value"] = double.NegativeInfinity };
        yield return new() { ["op_code"] = 8L, ["worker_id"] = worker, ["trial_id"] = 7L, ["user_attr"] = new OrderedDict<string, object?> { ["k"] = attrValue } };
        yield return new() { ["op_code"] = 9L, ["worker_id"] = worker, ["trial_id"] = 7L, ["system_attr"] = new OrderedDict<string, object?> { ["constraints"] = new List<object?> { -1.0, 0.0, 2.5 } } };
    }

    /// <summary>
    ///     Random documents of every value kind, written compact, with Python's default separators, and with random
    ///     whitespace, parse on the fast path to PyJson's values. Seeded, so a failure reproduces.
    /// </summary>
    /// <param name="seed">The generator's seed.</param>
    [TestMethod]
    [DataRow(1)]
    [DataRow(2)]
    [DataRow(3)]
    public void RandomDocuments_ParseOnTheFastPath(int seed)
    {
        var rng = new Random(seed);
        for (int i = 0; i < 1500; i++)
        {
            object? value = RandomValue(rng, depth: 0);
            AssertSameAsPyJson(Encoding.UTF8.GetBytes(PyJson.DumpsCompact(value)), fastMustAccept: true);
            AssertSameAsPyJson(Encoding.UTF8.GetBytes(PyJson.Dumps(value)), fastMustAccept: true);
            AssertSameAsPyJson(Encoding.UTF8.GetBytes(WithRandomWhitespace(rng, PyJson.Dumps(value))), fastMustAccept: true);
            // Raw UTF-8 instead of escapes: the same values, written as a non-ASCII writer would.
            AssertSameAsPyJson(Encoding.UTF8.GetBytes(Unescaped(PyJson.DumpsCompact(value))), fastMustAccept: true);
        }
    }

    /// <summary>
    ///     Corrupted documents (bytes flipped, deleted, inserted, truncated) give PyJson's outcome: the same value when
    ///     PyJson still accepts the bytes, else the same exception and message.
    /// </summary>
    /// <param name="seed">The generator's seed.</param>
    [TestMethod]
    [DataRow(11)]
    [DataRow(12)]
    [DataRow(13)]
    public void CorruptedDocuments_GivePyJsonsOutcome(int seed)
    {
        var rng = new Random(seed);
        byte[] interesting = Encoding.UTF8.GetBytes("{}[]\":,\\.-+eE0123456789NaInfityrulsU \t\r\n");
        for (int i = 0; i < 1500; i++)
        {
            byte[] doc = Encoding.UTF8.GetBytes(rng.Next(2) == 0
                ? PyJson.DumpsCompact(RandomValue(rng, depth: 0))
                : Unescaped(PyJson.Dumps(RandomValue(rng, depth: 0))));
            var bytes = new List<byte>(doc);
            int edits = 1 + rng.Next(3);
            for (int e = 0; e < edits && bytes.Count > 0; e++)
            {
                int at = rng.Next(bytes.Count);
                switch (rng.Next(5))
                {
                    case 0: bytes[at] = interesting[rng.Next(interesting.Length)]; break;
                    case 1: bytes[at] = (byte)rng.Next(256); break;
                    case 2: bytes.RemoveAt(at); break;
                    case 3: bytes.Insert(at, rng.Next(2) == 0 ? interesting[rng.Next(interesting.Length)] : (byte)rng.Next(256)); break;
                    default: bytes.RemoveRange(at, bytes.Count - at); break;
                }
            }
            AssertSameAsPyJson(bytes.ToArray(), fastMustAccept: false);
        }
    }

    /// <summary>Text the ServiceStack client already decoded parses as PyJson parses it, lone surrogates included.</summary>
    /// <param name="text">The document.</param>
    [TestMethod]
    [DataRow("{\"op_code\": 4, \"worker_id\": \"w\"}")]
    [DataRow("[\"\u00e9\u20ac\U0001F600\"]")]
    [DataRow("[\"\ud800\"]")]
    [DataRow("[\"\udc00x\"]")]
    [DataRow("\"\ud83d\"")]
    [DataRow("[1,]")]
    [DataRow("")]
    public void DecodedText_ParsesAsPyJson(string text)
    {
        Outcome expected = Capture(() => PyJson.Loads(text));
        Outcome actual = Capture(() => PyJsonUtf8.Loads(text));
        actual.Error.Should().Be(expected.Error);
        if (expected.Error is null)
            Same(expected.Value, actual.Value).Should().BeTrue();
    }

    /// <summary>
    ///     Threads parse at the same time without sharing state: each thread has its own reader and buffers, and every
    ///     result is still PyJson's.
    /// </summary>
    /// <exception cref="InvalidOperationException">A thread parsed a document to a value other than PyJson's.</exception>
    [TestMethod]
    public void ConcurrentThreads_EachGetPyJsonsValues()
    {
        var docs = Enumerable.Range(0, 64)
            .Select(i => Encoding.UTF8.GetBytes(PyJson.Dumps(RandomValue(new Random(100 + i), depth: 0))))
            .ToArray();
        var expected = docs.Select(d => PyJson.Loads(Encoding.UTF8.GetString(d))).ToArray();

        Parallel.For(0, 8, new ParallelOptions { MaxDegreeOfParallelism = 8 }, _ =>
        {
            for (int round = 0; round < 50; round++)
            {
                for (int i = 0; i < docs.Length; i++)
                {
                    if (!Same(expected[i], PyJsonUtf8.Loads(docs[i])))
                        throw new InvalidOperationException($"document {i} differed under concurrency");
                }
            }
        });
    }

    /// <summary>
    ///     A malformed document leaves the thread's reader clean: the documents after it parse on the fast path to
    ///     PyJson's values, including one that needs the value stack and unescaping buffer a huge document grew.
    /// </summary>
    [TestMethod]
    public void AfterAFailedOrHugeParse_TheNextParseIsUnaffected()
    {
        var huge = new List<object?>();
        for (int i = 0; i < 100_000; i++)
            huge.Add(new List<object?> { (long)i, "s\\\"" + i });
        AssertSameAsPyJson(Encoding.UTF8.GetBytes(PyJson.DumpsCompact(huge)), fastMustAccept: true);
        AssertSameAsPyJson(Encoding.UTF8.GetBytes("[1,{\"a\":[2,3"), fastMustAccept: false);
        AssertSameAsPyJson(Encoding.UTF8.GetBytes("{\"a\":[1,2,{\"b\":\"c\\n\"}],\"d\":null}"), fastMustAccept: true);
    }

    /// <summary>A random value of PyJson's object model, with the edge cases each kind has.</summary>
    /// <param name="rng">The generator.</param>
    /// <param name="depth">The number of containers enclosing the value.</param>
    /// <returns>The value.</returns>
    internal static object? RandomValue(Random rng, int depth)
    {
        int kind = rng.Next(depth >= 4 ? 6 : 8);
        switch (kind)
        {
            case 0: return null;
            case 1: return rng.Next(2) == 0;
            case 2: return RandomLong(rng);
            case 3: return RandomDouble(rng);
            case 4:
            case 5: return RandomString(rng);
            case 6:
            {
                var list = new List<object?>();
                int n = rng.Next(5);
                for (int i = 0; i < n; i++)
                    list.Add(RandomValue(rng, depth + 1));
                return list;
            }
            default:
            {
                var dict = new OrderedDict<string, object?>();
                int n = rng.Next(5);
                for (int i = 0; i < n; i++)
                    dict[RandomString(rng)] = RandomValue(rng, depth + 1);
                return dict;
            }
        }
    }

    /// <summary>A random 64-bit integer, often at a boundary PyJson's number paths treat differently.</summary>
    /// <param name="rng">The generator.</param>
    /// <returns>The integer.</returns>
    private static long RandomLong(Random rng) => rng.Next(8) switch
    {
        0 => long.MaxValue,
        1 => long.MinValue,
        2 => rng.Next(-130, 1030),
        3 => -rng.NextInt64(),
        _ => rng.NextInt64(),
    };

    /// <summary>A random double, from random bit patterns (NaN payloads, subnormals, -0.0) and ordinary values alike.</summary>
    /// <param name="rng">The generator.</param>
    /// <returns>The double.</returns>
    private static double RandomDouble(Random rng) => rng.Next(10) switch
    {
        0 => double.NaN,
        1 => double.PositiveInfinity,
        2 => double.NegativeInfinity,
        3 => -0.0,
        4 => double.Epsilon * rng.Next(1, 1000),
        5 => BitConverter.Int64BitsToDouble(rng.NextInt64()),
        6 => Math.Round(rng.NextDouble() * 1000, rng.Next(0, 6)),
        7 => rng.NextDouble() * Math.Pow(10, rng.Next(-320, 309)),
        _ => rng.NextDouble(),
    };

    /// <summary>A random string mixing ASCII, escapes, non-ASCII, non-BMP characters and lone surrogates.</summary>
    /// <param name="rng">The generator.</param>
    /// <returns>The string.</returns>
    private static string RandomString(Random rng)
    {
        int n = rng.Next(12);
        var sb = new StringBuilder(n);
        for (int i = 0; i < n; i++)
        {
            switch (rng.Next(9))
            {
                case 0: sb.Append((char)rng.Next(0, 0x20)); break;
                case 1: sb.Append("\"\\/"[rng.Next(3)]); break;
                case 2: sb.Append((char)rng.Next(0x80, 0x800)); break;
                case 3: sb.Append((char)rng.Next(0x800, 0xD800)); break;
                case 4: sb.Append(char.ConvertFromUtf32(rng.Next(0x10000, 0x110000))); break;
                case 5: sb.Append((char)rng.Next(0xD800, 0xE000)); break;
                default: sb.Append((char)rng.Next(0x20, 0x7F)); break;
            }
        }
        return sb.ToString();
    }

    /// <summary>Adds random JSON whitespace around a document's structural characters.</summary>
    /// <param name="rng">The generator.</param>
    /// <param name="json">A document written by PyJson.</param>
    /// <returns>The same document with whitespace inserted outside strings.</returns>
    private static string WithRandomWhitespace(Random rng, string json)
    {
        var sb = new StringBuilder(json.Length * 2);
        bool inString = false;
        for (int i = 0; i < json.Length; i++)
        {
            char c = json[i];
            if (inString)
            {
                sb.Append(c);
                if (c == '\\')
                    sb.Append(json[++i]);
                else if (c == '"')
                    inString = false;
                continue;
            }
            if ("{}[],:".IndexOf(c) >= 0 && rng.Next(2) == 0)
                sb.Append(" \t\r\n"[rng.Next(4)]);
            sb.Append(c);
            if (c == '"')
                inString = true;
        }
        return sb.ToString();
    }

    /// <summary>
    ///     Replaces a document's <c>\uXXXX</c> escapes of non-surrogate, non-control characters with the characters
    ///     themselves, as a writer without <c>ensure_ascii</c> would write them.
    /// </summary>
    /// <param name="json">A document written by PyJson, which escapes all non-ASCII text.</param>
    /// <returns>The document with raw non-ASCII text where that keeps the same value.</returns>
    private static string Unescaped(string json)
    {
        var sb = new StringBuilder(json.Length);
        for (int i = 0; i < json.Length; i++)
        {
            if (json[i] == '\\' && i + 5 < json.Length && json[i + 1] == 'u'
                && int.TryParse(json.AsSpan(i + 2, 4), NumberStyles.HexNumber, CultureInfo.InvariantCulture, out int code)
                && code >= 0x80 && (code < 0xD800 || code > 0xDFFF))
            {
                sb.Append((char)code);
                i += 5;
                continue;
            }
            sb.Append(json[i]);
            if (json[i] == '\\' && i + 1 < json.Length)
                sb.Append(json[++i]);
        }
        return sb.ToString();
    }
}

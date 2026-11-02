using System;
using System.Collections.Generic;
using System.Numerics;
using AwesomeAssertions;
using OptunaSharp.Collections;
using OptunaSharp.Internal;

namespace OptunaSharp.Tests.QA;

/// <summary>
///     Open parity bugs in the Python-compatible JSON layer (<see cref="PyJson" />), found by the 2026-11-02
///     parity QA run (<c>docs/qa/PARITY_QA_2026-11-02.md</c>). A fuzz of 3,346 nested values (every Unicode
///     class, control characters, lone surrogates, special floats) found <c>Dumps</c> and
///     <see cref="PythonRepr.Repr" /> byte-identical to CPython's <c>json.dumps</c> and <c>repr</c>; the tests
///     below cover what remained. Each fails today and passes once fixed, at which point its <c>[OpenBugs]</c>
///     attribute should be removed.
/// </summary>
[TestClass]
public class JsonParityBugTests
{
    /// <summary>
    ///     QA-20: serializing an integer beyond 64 bits must write the exact digits, as Python's
    ///     <c>json.dumps</c> does for its unbounded <c>int</c>; the port throws <c>TypeError</c>.
    /// </summary>
    /// <remarks>
    ///     <c>json.dumps({"n": 10**25})</c> is <c>{"n": 10000000000000000000000000}</c>. A
    ///     <see cref="BigInteger" /> user attribute therefore cannot be stored by the port at all. Related to the
    ///     known read side (catalog item 21c, "big ints parse as double"): together they mean a Python-written
    ///     big int never survives a C# read and re-write.
    /// </remarks>
    [TestMethod]
    [OpenBugs]
    public void Dumps_BigInteger_WritesExactDigitsLikePython()
    {
        var value = new OrderedDict<string, object?>
        {
            ["n"] = BigInteger.Pow(10, 25),
            ["m"] = new List<object?> { -BigInteger.Pow(10, 30), BigInteger.Pow(2, 64) },
        };

        PyJson.Dumps(value).Should().Be(
            "{\"n\": 10000000000000000000000000, \"m\": [-1000000000000000000000000000000, 18446744073709551616]}");
    }

    /// <summary>
    ///     QA-21: <see cref="PyJson.Loads" /> must reject the malformed JSON that Python's strict
    ///     <c>json.loads</c> rejects; the port accepts it.
    /// </summary>
    /// <remarks>
    ///     Python raises <c>json.JSONDecodeError</c> for a raw control character inside a string
    ///     (<c>"\t"</c> typed literally, <c>"\x01"</c>), a leading zero (<c>01</c>), a bare trailing or leading
    ///     dot (<c>1.</c>, <c>.5</c>) and a leading plus (<c>+1</c>). The port parses them as <c>"\t"</c>,
    ///     <c>"\u0001"</c>, <c>1</c>, <c>1.0</c>, <c>0.5</c> and <c>1</c>. User-reachable through the CLI's
    ///     <c>--search-space</c> and <c>--sampler-kwargs</c> arguments, which Python's <c>optuna</c> CLI refuses.
    /// </remarks>
    [TestMethod]
    [OpenBugs]
    public void Loads_MalformedJson_RejectedLikePythonStrictDecoder()
    {
        var malformed = new[] { "\"\t\"", "\"\u0001\"", "01", "1.", ".5", "+1" };
        foreach (var text in malformed)
            ((Action)(() => PyJson.Loads(text))).Should().Throw<Exception>($"json.loads({PythonRepr.StringRepr(text)}) raises JSONDecodeError");
    }
}

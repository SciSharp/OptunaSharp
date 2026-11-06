using System.Collections.Generic;
using System.Linq;
using System.Text.Json;
using AwesomeAssertions;
using OptunaSharp.Artifacts;
using Python.Runtime;

namespace OptunaSharp.Tests.Interop;

/// <summary>
///     Live parity of the port's MIME guessing (<c>MimeTypes.GuessType</c>, used by <c>upload_artifact</c>) with
///     CPython's own <c>mimetypes</c> built-in tables (QA-25 of the 2026-11-02 parity QA run; the owner chose the
///     built-in table over the per-machine lookup, question Q2).
/// </summary>
/// <remarks>
///     The unit test <c>ArtifactParityBugTests.GuessType_MatchesCPython312BuiltInTable</c> pins 42 names generated
///     from CPython; this test asks the running interpreter about every suffix in its tables, so a mistyped table entry
///     cannot hide. It self-skips (Inconclusive) without a Python engine, and on a CPython other than 3.12.10–3.12.x,
///     whose built-in table the port reproduces (3.12.0 lacks <c>.md</c>, <c>.markdown</c>, <c>.rst</c>; 3.13 adds
///     types).
/// </remarks>
[TestClass]
public class MimeTypesParityTests
{
    /// <summary>
    ///     Every key of <c>_types_map_default</c>, <c>_suffix_map_default</c> and <c>_encodings_map_default</c>, in seven
    ///     spellings (as is, uppercased, with a case-sensitive encoding suffix, with an uppercase one, after <c>.tar</c>,
    ///     behind an <c>http:</c> scheme, and with a fragment that only a scheme would strip), guesses exactly what
    ///     <c>mimetypes.MimeTypes().guess_type</c> answers.
    /// </summary>
    [TestMethod]
    [TestCategory("RequiresPython")]
    public void GuessType_EveryBuiltInSuffix_MatchesLiveCPython()
    {
        PythonSession.EnsureOrInconclusive();

        string payload;
        using (Py.GIL())
        {
            using PyModule scope = Py.CreateScope();
            scope.Exec(
                "import sys, json, mimetypes\n" +
                "version = '%d.%d.%d' % sys.version_info[:3]\n" +
                "supported = (3, 12, 10) <= sys.version_info[:3] < (3, 13, 0)\n" +
                "keys = (list(mimetypes._types_map_default) + list(mimetypes._suffix_map_default)\n" +
                "        + list(mimetypes._encodings_map_default))\n" +
                "names = []\n" +
                "for k in keys:\n" +
                "    names += ['f' + k, 'F' + k.upper(), 'f' + k + '.gz', 'f' + k + '.GZ', 'f.tar' + k,\n" +
                "              'http:f' + k, 'f' + k + '#x']\n" +
                "db = mimetypes.MimeTypes()\n" +
                "payload = json.dumps([[n, *db.guess_type(n)] for n in names])\n");
            if (!scope.Get("supported").As<bool>())
                Assert.Inconclusive(
                    "MimeTypes reproduces CPython 3.12.10-3.12.x's built-in table; this interpreter is " +
                    scope.Get("version").As<string>() + ".");
            payload = scope.Get("payload").As<string>();
        }

        // Compare outside the GIL: the payload is plain JSON, no Python object outlives the block above.
        List<string> mismatches = new();
        int rows = 0;
        foreach (JsonElement row in JsonDocument.Parse(payload).RootElement.EnumerateArray())
        {
            rows++;
            string name = row[0].GetString()!;
            (string? Mimetype, string? Encoding) expected = (row[1].GetString(), row[2].GetString());
            var actual = MimeTypes.GuessType(name);
            if (actual != expected)
                mismatches.Add($"{name}: port {actual}, CPython {expected}");
        }

        rows.Should().BeGreaterThan(1000, "163 table keys in seven spellings each");
        mismatches.Should().BeEmpty(string.Join("; ", mismatches.Take(10)));
    }
}

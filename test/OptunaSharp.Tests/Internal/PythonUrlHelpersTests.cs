using System;
using System.Linq;
using System.Numerics;
using AwesomeAssertions;
using NumSharp;
using OptunaSharp.Internal;
using OptunaSharp.Storages.SqlAlchemy;

namespace OptunaSharp.Tests.Internal;

/// <summary>
///     The Python building blocks behind storage-URL handling, each pinned to what CPython 3.12 and SQLAlchemy 2.0.45
///     answer for the same input (every expectation was printed by Python, not derived): <c>int()</c> and
///     <c>float()</c> of a string, <c>urllib.parse</c>'s <c>urlsplit</c>/<c>parse_qs</c>/<c>unquote</c>,
///     <c>str.format</c> with Optuna's <c>SCHEMA_VERSION</c>, <c>ntpath</c>/<c>posixpath.normpath</c>, <c>repr()</c> of a
///     string, and SQLAlchemy's <c>make_url</c> and dialect lookup. The interop suite's
///     <c>StorageUrlStandardParityTests</c> compares whole URLs live; these keep the pieces honest without Python.
/// </summary>
[TestClass]
public class PythonUrlHelpersTests
{
    /// <summary><c>int(text)</c>: whitespace, underscores between digits and Unicode decimal digits are Python's; the result.</summary>
    /// <param name="text">The input.</param>
    /// <param name="expected">Python's answer.</param>
    [TestMethod]
    [DataRow(" 1_000 ", "1000")]
    [DataRow("\u0663", "3")]
    [DataRow("+7", "7")]
    [DataRow("-0", "0")]
    [DataRow("12\n", "12")]
    [DataRow("\uff11\uff12", "12")]
    public void ParseInt_AcceptsWhatPythonIntAccepts(string text, string expected)
        => PyNumbers.ParseInt(text).Should().Be(BigInteger.Parse(expected));

    /// <summary><c>int(text)</c> refusals carry Python's message, the input's <c>repr()</c> included.</summary>
    /// <param name="text">The input.</param>
    /// <param name="message">Python's message.</param>
    [TestMethod]
    [DataRow("0x10", "invalid literal for int() with base 10: '0x10'")]
    [DataRow("1__0", "invalid literal for int() with base 10: '1__0'")]
    [DataRow("", "invalid literal for int() with base 10: ''")]
    [DataRow(" ", "invalid literal for int() with base 10: ' '")]
    public void ParseInt_RefusesWithPythonsMessage(string text, string message)
    {
        Action act = () => PyNumbers.ParseInt(text);
        act.Should().Throw<ValueError>().WithMessage(message);
    }

    /// <summary><c>float(text)</c>: exponents, infinities, NaN, underscores and Unicode digits as Python reads them.</summary>
    /// <param name="text">The input.</param>
    /// <param name="expected">Python's answer.</param>
    [TestMethod]
    [DataRow("1e3", 1000.0)]
    [DataRow(" 2.5 ", 2.5)]
    [DataRow("inf", double.PositiveInfinity)]
    [DataRow("-Infinity", double.NegativeInfinity)]
    [DataRow("1_0.5", 10.5)]
    [DataRow("\u0663.5", 3.5)]
    [DataRow("1e400", double.PositiveInfinity)]
    public void ParseFloat_AcceptsWhatPythonFloatAccepts(string text, double expected)
        => PyNumbers.ParseFloat(text).Should().Be(expected);

    /// <summary><c>float("nan")</c> is NaN, and the refusals carry Python's message.</summary>
    [TestMethod]
    public void ParseFloat_NanAndRefusals()
    {
        double.IsNaN(PyNumbers.ParseFloat("nan")).Should().BeTrue();
        foreach (string text in new[] { "1,5", "", "0x10" })
        {
            Action act = () => PyNumbers.ParseFloat(text);
            act.Should().Throw<ValueError>().WithMessage($"could not convert string to float: {PythonRepr.StringRepr(text)}");
        }
    }

    /// <summary>
    ///     <c>urlsplit</c>: the five components, user name and password split at the last <c>@</c>, the host
    ///     lower-cased (except an IPv6 zone), brackets dropped, newlines removed, and the port as an int.
    /// </summary>
    /// <param name="url">The URL.</param>
    /// <param name="expected">Python's <c>(scheme, netloc, path, query, fragment, username, password, hostname, port)</c>, <c>|</c>-joined, <c>None</c> as empty.</param>
    [TestMethod]
    [DataRow("redis://H.Ex.COM:6380/0?x=1#f", "redis|H.Ex.COM:6380|/0|x=1|f|||h.ex.com|6380")]
    [DataRow("redis://[::1]:7000", "redis|[::1]:7000||||||::1|7000")]
    [DataRow("redis://[fe80::1%25ETH0]", "redis|[fe80::1%25ETH0]||||||fe80::1%25ETH0|")]
    [DataRow("redis://[v1.x]", "redis|[v1.x]||||||v1.x|")]
    [DataRow("redis://u:p@h", "redis|u:p@h||||u|p|h|")]
    [DataRow("redis://u@v@h", "redis|u@v@h||||u@v||h|")]
    [DataRow("redis://h\n:1", "redis|h:1||||||h|1")]
    [DataRow("x://h:007", "x|h:007||||||h|7")]
    public void UrlSplit_MatchesUrllib(string url, string expected)
    {
        PySplitResult r = PyUrllib.UrlSplit(url);
        string.Join("|", r.Scheme, r.Netloc, r.Path, r.Query, r.Fragment, r.Username, r.Password, r.Hostname, r.Port)
            .Should().Be(expected);
    }

    /// <summary><c>urlsplit</c>'s refusals, eager (brackets) or on reading the port, with <c>urllib</c>'s messages.</summary>
    /// <param name="url">The URL.</param>
    /// <param name="message">Python's message.</param>
    [TestMethod]
    [DataRow("redis://h:abc", "Port could not be cast to integer value as 'abc'")]
    [DataRow("redis://h:99999", "Port out of range 0-65535")]
    [DataRow("redis://[::1", "Invalid IPv6 URL")]
    [DataRow("redis://[127.0.0.1]", "An IPv4 address cannot be in brackets")]
    public void UrlSplit_RefusesWithUrllibsMessages(string url, string message)
    {
        Action act = () => _ = PyUrllib.UrlSplit(url).Port;
        act.Should().Throw<ValueError>().WithMessage(message);
    }

    /// <summary>
    ///     <c>parse_qs</c>: blank values dropped, repeated keys collected in order, <c>+</c> as a space, invalid UTF-8
    ///     replaced, and only <c>&amp;</c> as the separator.
    /// </summary>
    /// <param name="query">The query string.</param>
    /// <param name="expected">Python's dict, as <c>key=v1,v2</c> entries joined with <c>;</c>.</param>
    [TestMethod]
    [DataRow("a=1&a=2&b=", "a=1,2")]
    [DataRow("a+b=c+d&e=%2B", "a b=c d;e=+")]
    [DataRow("x&y=1", "y=1")]
    [DataRow("a=%E2%82%AC&b=%E2", "a=\u20ac;b=\ufffd")]
    [DataRow(";a=1", ";a=1")]
    public void ParseQs_MatchesUrllib(string query, string expected)
        => string.Join(";", PyUrllib.ParseQs(query).Select(p => $"{p.Key}={string.Join(",", p.Value)}")).Should().Be(expected);

    /// <summary><c>unquote</c> leaves a bad escape as it is and replaces invalid UTF-8; <c>unquote_plus</c> also turns <c>+</c> into a space.</summary>
    [TestMethod]
    public void Unquote_MatchesUrllib()
    {
        PyUrllib.Unquote("%41%zz").Should().Be("A%zz");
        PyUrllib.Unquote("a+b").Should().Be("a+b");
        PyUrllib.UnquotePlus("a+b").Should().Be("a b");
        PyUrllib.Unquote("%E2%82").Should().Be("\ufffd");
        PyUrllib.Unquote("%C3%A9").Should().Be("\u00e9");
    }

    /// <summary>
    ///     Optuna's <c>url.format(SCHEMA_VERSION=12)</c>: the field with format specs, conversions and attributes, and
    ///     doubled braces, as <c>str.format</c> renders them.
    /// </summary>
    /// <param name="template">The storage URL template.</param>
    /// <param name="expected">Python's result.</param>
    [TestMethod]
    [DataRow("{SCHEMA_VERSION}", "12")]
    [DataRow("{SCHEMA_VERSION:03d}", "012")]
    [DataRow("{{x}}", "{x}")]
    [DataRow("{SCHEMA_VERSION.real}", "12")]
    [DataRow("{SCHEMA_VERSION:,}", "12")]
    [DataRow("{SCHEMA_VERSION!r:>5}", "   12")]
    [DataRow("{SCHEMA_VERSION:c}", "\x0c")]
    [DataRow("{SCHEMA_VERSION:x<4}", "12xx")]
    [DataRow("{SCHEMA_VERSION:#b}", "0b1100")]
    [DataRow("{SCHEMA_VERSION:\U0001F600^5}", "\U0001F60012\U0001F600\U0001F600")]
    [DataRow("{SCHEMA_VERSION!r:\U0001F600>5}", "\U0001F600\U0001F600\U0001F60012")]
    [DataRow("{SCHEMA_VERSION:\U0001F600=+5}", "+\U0001F600\U0001F60012")]
    [DataRow("{SCHEMA_VERSION:\U0001F600<4c}", "\x0c\U0001F600\U0001F600\U0001F600")]
    [DataRow("{SCHEMA_VERSION!s:\U0001F600^6.1}", "\U0001F600\U0001F6001\U0001F600\U0001F600\U0001F600")]
    [DataRow("{SCHEMA_VERSION!\x00:>4}", "  12")]
    public void TemplateFill_MatchesStrFormat(string template, string expected)
        => StorageUrlTemplate.Fill(template).Should().Be(expected);

    /// <summary>
    ///     A zero-padded, grouped width of three million characters renders as Python renders it (3,000,001 characters,
    ///     <c>0,000,00…,000,012</c>) and in linear time: the grouping used to insert each group at the front, which made
    ///     this width take minutes.
    /// </summary>
    [TestMethod]
    public void TemplateFill_HugeGroupedWidthIsLinear()
    {
        var watch = System.Diagnostics.Stopwatch.StartNew();
        string filled = StorageUrlTemplate.Fill("{SCHEMA_VERSION:03000000,}");
        watch.Stop();
        filled.Length.Should().Be(3000001);
        filled.Should().StartWith("0,000,00").And.EndWith(",000,012");
        watch.Elapsed.Should().BeLessThan(TimeSpan.FromSeconds(10));
    }

    /// <summary>
    ///     A width past the longest .NET string (Python builds a 3 GB URL for <c>{SCHEMA_VERSION:3000000000}</c> when memory
    ///     allows) fails with <see cref="OutOfMemoryException" /> before any allocation, not with an argument error from a
    ///     negative count.
    /// </summary>
    [TestMethod]
    public void TemplateFill_WidthBeyondDotNetStringsIsOutOfMemory()
    {
        Action padded = () => StorageUrlTemplate.Fill("{SCHEMA_VERSION:3000000000}");
        padded.Should().Throw<OutOfMemoryException>();
        Action grouped = () => StorageUrlTemplate.Fill("{SCHEMA_VERSION:03000000000,}");
        grouped.Should().Throw<OutOfMemoryException>();
    }

    /// <summary><c>str.format</c>'s refusals keep Python's exception kinds and messages.</summary>
    /// <param name="template">The storage URL template.</param>
    /// <param name="kind">Python's exception type, as the port's.</param>
    /// <param name="message">Python's message.</param>
    [TestMethod]
    [DataRow("{a}", typeof(KeyError), "'a'")]
    [DataRow("{0}", typeof(IndexError), "Replacement index 0 out of range for positional args tuple")]
    [DataRow("}", typeof(ValueError), "Single '}' encountered in format string")]
    [DataRow("{", typeof(ValueError), "Single '{' encountered in format string")]
    [DataRow("{SCHEMA_VERSION!x}", typeof(ValueError), "Unknown conversion specifier x")]
    [DataRow("{SCHEMA_VERSION.foo}", typeof(AttributeError), "'int' object has no attribute 'foo'")]
    [DataRow("{SCHEMA_VERSION:s}", typeof(ValueError), "Unknown format code 's' for object of type 'int'")]
    [DataRow("{SCHEMA_VERSION[0]}", typeof(TypeError), "'int' object is not subscriptable")]
    [DataRow("{SCHEMA_VERSION:\U0001F600}", typeof(ValueError), "Unknown format code '\\x1f600' for object of type 'int'")]
    [DataRow("{SCHEMA_VERSION!\U0001F600}", typeof(ValueError), "Unknown conversion specifier \\x1f600")]
    [DataRow("{SCHEMA_VERSION!\U0001F600:>4}", typeof(ValueError), "Unknown conversion specifier \\x1f600")]
    [DataRow("{SCHEMA_VERSION:,\U0001F600}", typeof(ValueError), "Cannot specify ',' with '\\x1f600'.")]
    [DataRow("{SCHEMA_VERSION:\U0001F600\U0001F600}", typeof(ValueError), "Invalid format specifier '\U0001F600\U0001F600' for object of type 'int'")]
    [DataRow("{\U0001F600}", typeof(KeyError), "'\U0001F600'")]
    [DataRow("{\U000E0001}", typeof(KeyError), "'\\U000e0001'")]
    public void TemplateFill_RefusesLikeStrFormat(string template, Type kind, string message)
    {
        Action act = () => StorageUrlTemplate.Fill(template);
        Exception e = act.Should().Throw<Exception>().Which;
        e.GetType().Should().Be(kind);
        e.Message.Should().Be(message);
    }

    /// <summary><c>ntpath.normpath</c>: drives, UNC shares, <c>\\?\</c> paths and <c>..</c> at the root, as on Windows Python.</summary>
    /// <param name="path">The path.</param>
    /// <param name="expected">Python's answer.</param>
    [TestMethod]
    [DataRow("C:/a/../b", "C:\\b")]
    [DataRow("C:a/./b", "C:a\\b")]
    [DataRow("//server/share/x/..", "\\\\server\\share\\")]
    [DataRow("\\\\?\\C:\\x\\..\\y", "\\\\?\\C:\\y")]
    [DataRow("a/b/../../..", "..")]
    [DataRow("C:\\..", "C:\\")]
    [DataRow("/a//b/./c/..", "\\a\\b")]
    public void NtNormPath_MatchesNtpath(string path, string expected) => PyOsPath.NtNormPath(path).Should().Be(expected);

    /// <summary><c>posixpath.normpath</c>: two leading slashes kept, three collapsed, the empty path as <c>.</c>.</summary>
    /// <param name="path">The path.</param>
    /// <param name="expected">Python's answer.</param>
    [TestMethod]
    [DataRow("/a//b/./c/..", "/a/b")]
    [DataRow("//a/b", "//a/b")]
    [DataRow("///a", "/a")]
    [DataRow("a/../..", "..")]
    [DataRow("", ".")]
    [DataRow("./", ".")]
    public void PosixNormPath_MatchesPosixpath(string path, string expected) => PyOsPath.PosixNormPath(path).Should().Be(expected);

    /// <summary>
    ///     <c>repr()</c> of a string escapes what <c>str.isprintable()</c> rejects (no-break space, format characters,
    ///     separators, lone surrogates, tags) and keeps printable non-ASCII such as an emoji.
    /// </summary>
    /// <param name="text">The string.</param>
    /// <param name="expected">Python's <c>repr()</c>.</param>
    [TestMethod]
    [DataRow("\u00a0", "'\\xa0'")]
    [DataRow("\u200b", "'\\u200b'")]
    [DataRow("\U0001F600", "'\U0001F600'")]
    [DataRow("\U000e0001", "'\\U000e0001'")]
    [DataRow("a\tb", "'a\\tb'")]
    [DataRow("\x7f", "'\\x7f'")]
    [DataRow("\u2028", "'\\u2028'")]
    public void StringRepr_EscapesWhatPythonCannotPrint(string text, string expected) => PythonRepr.StringRepr(text).Should().Be(expected);

    /// <summary>
    ///     A lone surrogate is escaped as <c>\ud800</c>, as Python escapes one. (Built in code: attribute arguments are
    ///     stored as UTF-8, which cannot carry a lone surrogate.)
    /// </summary>
    [TestMethod]
    public void StringRepr_EscapesALoneSurrogate() => PythonRepr.StringRepr(new string('\ud800', 1)).Should().Be("'\\ud800'");

    /// <summary>
    ///     SQLAlchemy's <c>make_url</c>: components (the password percent-decoded), a repeated query key as a tuple, the
    ///     IPv6 host without brackets, and <c>str()</c> with the password hidden.
    /// </summary>
    [TestMethod]
    public void SqlAlchemyUrl_MatchesMakeUrl()
    {
        SqlAlchemyUrl pg = SqlAlchemyUrl.Parse("postgresql://u:p@h:5433/db?a=1&a=2&b=x");
        (pg.Drivername, pg.Username, pg.Password, pg.Host, pg.Port, pg.Database).Should().Be(("postgresql", "u", "p", "h", (BigInteger?)5433, "db"));
        ((string[])pg.Query["a"]).Should().Equal("1", "2");
        pg.Query["b"].Should().Be("x");
        pg.ToString().Should().Be("postgresql://u:***@h:5433/db?a=1&a=2&b=x");

        SqlAlchemyUrl.Parse("mysql://u:p%40s@h/db").Password.Should().Be("p@s");
        SqlAlchemyUrl.Parse("sqlite:///rel.db").Database.Should().Be("rel.db");
        SqlAlchemyUrl.Parse("sqlite:////abs.db").Database.Should().Be("/abs.db");
        SqlAlchemyUrl.Parse("postgresql://u:p@[::1]:5432/db").ToString().Should().Be("postgresql://u:***@[::1]:5432/db");
        SqlAlchemyUrl multi = SqlAlchemyUrl.Parse("postgresql://u@h,h2:5/db");
        (multi.Host, multi.Port).Should().Be(("h,h2", (BigInteger?)5));
        SqlAlchemyUrl bare = SqlAlchemyUrl.Parse("dialect+driver://");
        (bare.Drivername, bare.Host, bare.Database).Should().Be(("dialect+driver", (string?)null, (string?)null));
    }

    /// <summary><c>make_url</c>'s refusals: a string that is not a URL (<c>ArgumentError</c>) and a port that is not an int.</summary>
    [TestMethod]
    public void SqlAlchemyUrl_RefusesLikeMakeUrl()
    {
        Action notUrl = () => SqlAlchemyUrl.Parse("just-a-path.db");
        notUrl.Should().Throw<ArgumentException>().Which.GetType().Should().Be(typeof(ArgumentException));
        notUrl.Should().Throw<ArgumentException>().WithMessage("Could not parse SQLAlchemy URL from given URL string");
        Action port = () => SqlAlchemyUrl.Parse("postgresql://u:p@h:abc/db");
        port.Should().Throw<ValueError>().WithMessage("invalid literal for int() with base 10: 'abc'");
    }

    /// <summary>
    ///     SQLAlchemy's dialect and plugin lookup: unknown dialects, drivers and plugins raise <c>NoSuchModuleError</c>
    ///     (an <c>ArgumentError</c>) with its text; a known dialect resolves to its default driver.
    /// </summary>
    [TestMethod]
    public void SqlAlchemyRegistry_MatchesTheEntryPointLookup()
    {
        foreach (var (url, message) in new[]
                 {
                     ("postgres://h/db", "Can't load plugin: sqlalchemy.dialects:postgres"),
                     ("sqlite+nosuch:///x", "Can't load plugin: sqlalchemy.dialects:sqlite.nosuch"),
                     ("sqlite:///x?plugin=foo", "Can't load plugin: sqlalchemy.plugins:foo"),
                     ("redis://h", "Can't load plugin: sqlalchemy.dialects:redis"),
                 })
        {
            Action act = () => SqlAlchemyRegistry.Resolve(SqlAlchemyUrl.Parse(url), out _);
            act.Should().Throw<ArgumentException>().WithMessage(message);
        }
        SqlAlchemyDriver driver = SqlAlchemyRegistry.Resolve(SqlAlchemyUrl.Parse("postgresql://h/db"), out _);
        (driver.Dialect, driver.Driver, driver.IsAsync).Should().Be(("postgresql", "psycopg2", false));
        SqlAlchemyRegistry.Resolve(SqlAlchemyUrl.Parse("mysql+asyncmy://h/db"), out _).IsAsync.Should().BeTrue();
    }

    /// <summary>
    ///     Python's <c>str.lower()</c>: U+0130 becomes two code points, a word-final sigma becomes <c>ς</c> (case-ignorable
    ///     characters such as <c>.</c>, <c>'</c> and combining marks are skipped on both sides), the rest maps as .NET's
    ///     invariant lower case does. Each expectation is CPython 3.12's output.
    /// </summary>
    /// <param name="text">The input.</param>
    /// <param name="expected">Python's <c>text.lower()</c>.</param>
    [TestMethod]
    [DataRow("\U00000130", "i\U00000307")]
    [DataRow("\U00000391\U000003A3", "\U000003B1\U000003C2")]
    [DataRow("\U00000391\U000003A3.", "\U000003B1\U000003C2.")]
    [DataRow("\U00000391\U000003A3\U00000391", "\U000003B1\U000003C3\U000003B1")]
    [DataRow("\U000003A3", "\U000003C3")]
    [DataRow("A\U000003A3'B", "a\U000003C3'b")]
    [DataRow("A\U000003A3\U00000345", "a\U000003C2\U00000345")]
    [DataRow("\U000003A3\U000003A3", "\U000003C3\U000003C2")]
    [DataRow("a\U00000130b", "ai\U00000307b")]
    [DataRow("AB\U000003A3 C", "ab\U000003C2 c")]
    [DataRow("\U0000212A", "k")]
    [DataRow("\U00000391\U000003A3\U0001F600", "\U000003B1\U000003C2\U0001F600")]
    [DataRow("MixedASCII", "mixedascii")]
    public void Lower_MatchesPythonStrLower(string text, string expected) => PyStrings.Lower(text).Should().Be(expected);

    /// <summary>A lone surrogate is a character of its own in Python: <c>lower()</c> keeps it and a sigma before it is not final.</summary>
    [TestMethod]
    public void Lower_KeepsALoneSurrogate()
    {
        string lone = new('\ud800', 1);
        PyStrings.Lower(lone).Should().Be(lone);
        PyStrings.Lower("A\U000003A3" + lone).Should().Be("a\U000003C2" + lone);
    }

    /// <summary>
    ///     SQLAlchemy's <c>asbool</c> quotes the stripped, lower-cased text in its error, so the message carries Python's
    ///     <c>lower()</c> (CPython: <c>String is not true/false: 'i̇'</c> for <c>İ</c>).
    /// </summary>
    [TestMethod]
    public void AsBool_QuotesPythonsLowerCasedText()
    {
        foreach ((string text, string message) in new[]
                 {
                     ("\U00000130", "String is not true/false: 'i\U00000307'"),
                     (" T\U00000130 ", "String is not true/false: 'ti\U00000307'"),
                     ("\U00000391\U000003A3", "String is not true/false: '\U000003B1\U000003C2'"),
                 })
        {
            Action act = () => PyValues.AsBool(text);
            act.Should().Throw<ValueError>().Which.Message.Should().Be(message);
        }
        PyValues.AsBool(" Yes ").Should().BeTrue();
    }

    /// <summary>urllib's <c>.hostname</c> lower-cases with <c>str.lower()</c> before a <c>%</c> zone and leaves the zone alone.</summary>
    [TestMethod]
    public void Hostname_LowerCasesLikeUrllib()
    {
        PyUrllib.UrlSplit("redis://\U00000391\U000003A3:1/0").Hostname.Should().Be("\U000003B1\U000003C2");
        PyUrllib.UrlSplit("redis://[FE80::1%25\U00000391\U000003A3]:1").Hostname.Should().Be("fe80::1%25\U00000391\U000003A3");
    }

    /// <summary>Code-point order keeps a lone surrogate's own value (U+D800 sorts before U+E000), not U+FFFD's.</summary>
    [TestMethod]
    public void CompareCodePoints_KeepsALoneSurrogatesValue()
    {
        PyStrings.CompareCodePoints(new string('\ud800', 1), "\U0000E000").Should().BeNegative();
        PyStrings.CompareCodePoints("\U0001F600", "\U0000FFFF").Should().BePositive();
    }

    /// <summary>CPython's strict UTF-8 decoding: the same refusals with the same reason, byte and position.</summary>
    /// <param name="bytes">The input.</param>
    /// <param name="message">CPython's <c>UnicodeDecodeError</c> message.</param>
    [TestMethod]
    [DataRow(new byte[] { 0xff }, "'utf-8' codec can't decode byte 0xff in position 0: invalid start byte")]
    [DataRow(new byte[] { 0xe2, 0x82 }, "'utf-8' codec can't decode bytes in position 0-1: unexpected end of data")]
    [DataRow(new byte[] { 0xe2, 0x41 }, "'utf-8' codec can't decode byte 0xe2 in position 0: invalid continuation byte")]
    [DataRow(new byte[] { 0xed, 0xa0, 0x80 }, "'utf-8' codec can't decode byte 0xed in position 0: invalid continuation byte")]
    [DataRow(new byte[] { 0xf0, 0x9f, 0x98 }, "'utf-8' codec can't decode bytes in position 0-2: unexpected end of data")]
    [DataRow(new byte[] { 0xf0, 0x9f, 0x41 }, "'utf-8' codec can't decode bytes in position 0-1: invalid continuation byte")]
    [DataRow(new byte[] { 0x61, 0xc0, 0x80 }, "'utf-8' codec can't decode byte 0xc0 in position 1: invalid start byte")]
    [DataRow(new byte[] { 0xf4, 0x90, 0x80, 0x80 }, "'utf-8' codec can't decode byte 0xf4 in position 0: invalid continuation byte")]
    [DataRow(new byte[] { 0xe0, 0x80, 0x80 }, "'utf-8' codec can't decode byte 0xe0 in position 0: invalid continuation byte")]
    [DataRow(new byte[] { 0xf5 }, "'utf-8' codec can't decode byte 0xf5 in position 0: invalid start byte")]
    [DataRow(new byte[] { 0xc2 }, "'utf-8' codec can't decode byte 0xc2 in position 0: unexpected end of data")]
    public void Utf8Decode_RefusesWithCPythonsMessage(byte[] bytes, string message)
    {
        Action act = () => PyUtf8.DecodeStrict(bytes);
        act.Should().Throw<UnicodeDecodeError>().Which.Message.Should().Be(message);
    }

    /// <summary>Well-formed UTF-8 decodes, and the error is a <see cref="ValueError" /> as Python's is.</summary>
    [TestMethod]
    public void Utf8Decode_AcceptsWellFormedText()
    {
        PyUtf8.DecodeStrict(new byte[] { 0xe2, 0x82, 0xac, 0xf0, 0x9f, 0x98, 0x80 }).Should().Be("\U000020AC\U0001F600");
        Action act = () => PyUtf8.DecodeStrict(new byte[] { 0xff });
        act.Should().Throw<ValueError>();
    }

    /// <summary><c>text.encode("latin1")</c>: CPython's message for the first run of characters above U+00FF.</summary>
    /// <param name="text">The input.</param>
    /// <param name="message">CPython's <c>UnicodeEncodeError</c> message.</param>
    [TestMethod]
    [DataRow("\U000020AC", "'latin-1' codec can't encode character '\\u20ac' in position 0: ordinal not in range(256)")]
    [DataRow("a\U000020AC\U000020ACb", "'latin-1' codec can't encode characters in position 1-2: ordinal not in range(256)")]
    [DataRow("\U0001F600", "'latin-1' codec can't encode character '\\U0001f600' in position 0: ordinal not in range(256)")]
    [DataRow("ab\U00000100", "'latin-1' codec can't encode character '\\u0100' in position 2: ordinal not in range(256)")]
    [DataRow("\U000000E9\U000020AC", "'latin-1' codec can't encode character '\\u20ac' in position 1: ordinal not in range(256)")]
    public void Latin1Encode_RefusesLikeStrEncode(string text, string message)
    {
        Action act = () => PyLatin1.CheckEncodable(text);
        act.Should().Throw<UnicodeEncodeError>().Which.Message.Should().Be(message);
    }

    /// <summary>Latin-1 text (ASCII, <c>é</c>, <c>ÿ</c>) encodes without error.</summary>
    [TestMethod]
    public void Latin1Encode_AcceptsLatin1Text()
    {
        Action act = () => PyLatin1.CheckEncodable("p\U000000E9\U000000FF");
        act.Should().NotThrow();
    }

    /// <summary>
    ///     <c>os.path.expanduser</c> as CPython 3.12 runs it on this platform: a path without a leading <c>~</c> is left
    ///     alone; on Windows <c>~</c> is <c>USERPROFILE</c> and <c>~other</c> a sibling profile when the current
    ///     profile's last component is <c>USERNAME</c> (Python here gave <c>C:\Users\bob\x</c> for <c>~bob\x</c>);
    ///     elsewhere <c>~</c> is <c>HOME</c>.
    /// </summary>
    [TestMethod]
    public void ExpandUser_MatchesOsPath()
    {
        PyOsPath.ExpandUser("x~").Should().Be("x~");
        if (OperatingSystem.IsWindows())
        {
            string? profile = Environment.GetEnvironmentVariable("USERPROFILE");
            if (profile is null)
                return;
            PyOsPath.ExpandUser("~").Should().Be(profile);
            PyOsPath.ExpandUser("~\\my.cnf").Should().Be(profile + "\\my.cnf");
            PyOsPath.ExpandUser("~/my.cnf").Should().Be(profile + "/my.cnf");
            string? user = Environment.GetEnvironmentVariable("USERNAME");
            string parent = System.IO.Path.GetDirectoryName(profile.TrimEnd('\\'))!;
            if (user is not null && System.IO.Path.GetFileName(profile.TrimEnd('\\')) == user)
                PyOsPath.ExpandUser("~bob\\x").Should().Be(parent + "\\bob\\x");
            else
                PyOsPath.ExpandUser("~bob\\x").Should().Be("~bob\\x");
        }
        else
        {
            string? home = Environment.GetEnvironmentVariable("HOME");
            if (home is not null)
                PyOsPath.ExpandUser("~/my.cnf").Should().Be(home.TrimEnd('/') + "/my.cnf");
            PyOsPath.ExpandUser("~no-such-user-for-optunasharp/x").Should().Be("~no-such-user-for-optunasharp/x");
        }
    }
}

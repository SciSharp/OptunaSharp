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
    public void TemplateFill_MatchesStrFormat(string template, string expected)
        => StorageUrlTemplate.Fill(template).Should().Be(expected);

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
}

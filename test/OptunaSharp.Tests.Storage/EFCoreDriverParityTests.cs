using System;
using System.IO;
using System.Linq;
using AwesomeAssertions;
using MySqlConnector;
using Npgsql;
using NumSharp;
using OptunaSharp.Internal;
using OptunaSharp.Storages;
using OptunaSharp.Storages.EFCore;
// NumSharp 0.80.0 ships its own UnicodeDecodeError / UnicodeEncodeError; these tests mean OptunaSharp's.
using UnicodeDecodeError = OptunaSharp.Internal.UnicodeDecodeError;
using UnicodeEncodeError = OptunaSharp.Internal.UnicodeEncodeError;

namespace OptunaSharp.Tests.Storage;

/// <summary>
///     The Python drivers' own handling of a storage URL's arguments, past SQLAlchemy: psycopg2 and libpq 18 for
///     PostgreSQL, PyMySQL 2.2 and mysqlclient 2.2 for MySQL. Every expectation was printed by those drivers for the
///     same URL (psycopg2 2.9.12 over libpq 18.3, PyMySQL 2.2.8, mysqlclient 2.2.7 on CPython 3.12); the scratch
///     differential that found these cases ran the drivers live, these pin the results without Python or a server.
/// </summary>
[TestClass]
public class EFCoreDriverParityTests
{
    /// <summary>The libpq parameters psycopg2 would hand libpq for a URL, as <c>parse_dsn</c> lists them.</summary>
    /// <param name="url">The storage URL.</param>
    /// <returns><c>key=value</c> pairs sorted by key, joined with <c>;</c>.</returns>
    private static string Libpq(string url)
        => string.Join(";", PostgresUrl.LibpqParameters(StorageUrlResolver.Prepare(url)).OrderBy(p => p.Key, StringComparer.Ordinal)
            .Select(p => p.Key + "=" + p.Value));

    /// <summary>The Npgsql builder for a PostgreSQL URL.</summary>
    /// <param name="url">The storage URL.</param>
    /// <returns>The builder.</returns>
    private static NpgsqlConnectionStringBuilder Pg(string url) => (NpgsqlConnectionStringBuilder)RdbUrl.Parse(url).Builder;

    /// <summary>The MySqlConnector builder for a MySQL URL.</summary>
    /// <param name="url">The storage URL.</param>
    /// <returns>The builder.</returns>
    private static MySqlConnectionStringBuilder My(string url) => (MySqlConnectionStringBuilder)RdbUrl.Parse(url).Builder;

    /// <summary>Asserts that parsing a URL fails with exactly this exception type and message.</summary>
    /// <param name="url">The storage URL.</param>
    /// <param name="kind">The exception type.</param>
    /// <param name="message">The message.</param>
    private static void Fails(string url, Type kind, string message)
    {
        Action act = () => RdbUrl.Parse(url);
        Exception e = act.Should().Throw<Exception>().Which;
        e.GetType().Should().Be(kind, url);
        e.Message.Should().Be(message, url);
    }

    // ------------------------------------------------------------------------------------------------ PostgreSQL

    /// <summary>
    ///     A <c>?dsn=</c> in libpq's URI form is parsed as libpq parses it, then the URL's own components override it
    ///     the way <c>make_dsn</c>'s dict update does: hosts and ports lists, IPv6 brackets, percent-decoding and JDBC's
    ///     <c>ssl=true</c>.
    /// </summary>
    /// <param name="url">The storage URL.</param>
    /// <param name="expected">psycopg2's <c>parse_dsn(make_dsn(...))</c>, sorted.</param>
    [TestMethod]
    [DataRow("postgresql://u@h/db?dsn=postgresql%3A%2F%2Fx%3A5%2Fy%3Fsslmode%3Drequire", "dbname=db;host=h;port=5;sslmode=require;user=u")]
    [DataRow("postgresql:///?dsn=postgresql%3A%2F%2Fa%40x%3A5%2Cz%3A6%2Fy%3Fapplication_name%3Dq%2520r", "application_name=q r;dbname=y;host=x,z;port=5,6;user=a")]
    [DataRow("postgresql:///?dsn=postgresql%3A%2F%2F%5B%3A%3A1%5D%3A7%2Fd", "dbname=d;host=::1;port=7")]
    [DataRow("postgresql:///?dsn=postgresql%3A%2F%2Fh%2Fd%3Fssl%3Dtrue", "dbname=d;host=h;sslmode=require")]
    public void Postgres_UriDsn_MergesLikeMakeDsn(string url, string expected) => Libpq(url).Should().Be(expected);

    /// <summary>
    ///     A URI dsn's errors are libpq's (with psycopg2's <c>invalid dsn: </c> prefix); a repeated <c>?dsn=</c> is a
    ///     tuple psycopg2 refuses; a percent-escape that is not UTF-8 fails as psycopg2 decodes the result.
    /// </summary>
    [TestMethod]
    public void Postgres_UriDsn_FailsLikePsycopg2()
    {
        Action bogus = () => Libpq("postgresql:///?dsn=postgresql%3A%2F%2Fh%2Fd%3Fbogus%3D1");
        bogus.Should().Throw<FormatException>().Which.Message.Should().Be("invalid dsn: invalid URI query parameter: \"bogus\"");
        Action tuple = () => Libpq("postgresql://u@h/db?dsn=a&dsn=b");
        tuple.Should().Throw<TypeError>().Which.Message.Should().Be("Expected bytes or unicode string, got tuple instead");
        Action utf8 = () => Libpq("postgresql:///?dsn=postgresql%3A%2F%2Fh%2F%25ff");
        utf8.Should().Throw<UnicodeDecodeError>().Which.Message.Should().Be("'utf-8' codec can't decode byte 0xff in position 0: invalid start byte");
        Action spaces = () => Libpq("postgresql:///?dsn=postgresql%3A%2F%2Fh%2Fd%3Fapplication_name%3Da%20b");
        spaces.Should().Throw<FormatException>().Which.Message.Should()
            .Be("invalid dsn: unexpected spaces found in \"a b\", use percent-encoded spaces (%20) instead");
    }

    /// <summary>
    ///     A NUL in a value: psycopg2's <c>make_dsn</c> validates the string up to the NUL, then <c>connect()</c>
    ///     refuses the NUL (<c>embedded null character</c>), so an error before it still wins; psycopg 3 hands libpq the
    ///     C string, which simply ends there.
    /// </summary>
    [TestMethod]
    public void Postgres_Nul_IsPsycopg2sValueErrorAndPsycopgsTruncation()
    {
        Action psycopg2 = () => Libpq("postgresql://u@h/db?options=%00x&sslmode=require");
        psycopg2.Should().Throw<ValueError>().Which.Message.Should().Be("embedded null character");
        Action earlier = () => Libpq("postgresql://u@h/db?dsn=bogus%00");
        earlier.Should().Throw<FormatException>().Which.Message.Should().Be("invalid dsn: missing \"=\" after \"bogus\" in connection info string");
        Libpq("postgresql+psycopg://u@h/db?options=%00x&sslmode=require").Should().Be("dbname=db;host=h;options=;user=u");
    }

    /// <summary>
    ///     libpq's defaults: <c>sslrootcert=system</c> raises the default <c>sslmode</c> to <c>verify-full</c> (so the URL
    ///     connects), while an explicit weaker mode is refused; an explicit empty enumerated value is invalid, as libpq
    ///     validates it rather than defaulting it.
    /// </summary>
    [TestMethod]
    public void Postgres_LibpqDefaults_SslRootCertSystemAndEmptyValues()
    {
        Pg("postgresql://u@h/db?sslrootcert=system").SslMode.Should().Be(SslMode.VerifyFull);
        Fails("postgresql://u@h/db?sslrootcert=system&sslmode=require", typeof(FormatException),
            "weak sslmode \"require\" may not be used with sslrootcert=system (use \"verify-full\")");
        Fails("postgresql://u@h/db?dsn=sslmode%3D%27%27", typeof(FormatException), "invalid sslmode value: \"\"");
        Fails("postgresql://u@h/db?dsn=channel_binding%3D%27%27", typeof(FormatException), "invalid channel_binding value: \"\"");
    }

    /// <summary>
    ///     <c>require_auth</c> is validated with libpq's rules (all or none negated, each method once, names exact),
    ///     then refused: Npgsql cannot restrict the authentication requests it answers.
    /// </summary>
    /// <param name="value">The URL-encoded value.</param>
    /// <param name="message">libpq's message, or <see langword="null" /> for a valid list.</param>
    [TestMethod]
    [DataRow("bogus", "invalid require_auth value: \"bogus\"")]
    [DataRow("PASSWORD", "invalid require_auth value: \"PASSWORD\"")]
    [DataRow("password,", "invalid require_auth value: \"\"")]
    [DataRow("%21password,md5", "require_auth method \"md5\" cannot be mixed with negative methods")]
    [DataRow("password,%21md5", "negative require_auth method \"!md5\" cannot be mixed with non-negative methods")]
    [DataRow("password,password", "require_auth method \"password\" is specified more than once")]
    [DataRow("%21none,%21none", "require_auth method \"!none\" is specified more than once")]
    [DataRow("scram-sha-256,scram-sha-256", "require_auth method \"scram-sha-256\" is specified more than once")]
    [DataRow("gss,sspi", null)]
    [DataRow("%21gss,%21sspi", null)]
    [DataRow("oauth", null)]
    public void Postgres_RequireAuth_ValidatesLikeLibpqThenRefuses(string value, string? message)
    {
        Action act = () => RdbUrl.Parse($"postgresql://u@h/db?require_auth={value}");
        if (message is null)
            act.Should().Throw<NotSupportedException>().Which.Message.Should().Contain("require_auth");
        else
            act.Should().Throw<FormatException>().Which.Message.Should().Be(message);
    }

    /// <summary>The SCRAM keys are libpq 18's strict base64 of exactly 32 bytes; a valid one is then refused (Npgsql takes no precomputed key).</summary>
    [TestMethod]
    public void Postgres_ScramKeys_ValidateLikeLibpqThenRefuse()
    {
        Fails("postgresql://u@h/db?scram_client_key=x", typeof(FormatException), "invalid SCRAM client key");
        Fails("postgresql://u@h/db?scram_client_key=AAAAAAAAAAAAAAAAAAAAAA%3D%3D", typeof(FormatException), "invalid SCRAM client key length: 16");
        Fails("postgresql://u@h/db?scram_server_key=AAAA%20AAAA", typeof(FormatException), "invalid SCRAM server key");
        Fails("postgresql://u@h/db?dsn=scram_client_key%3D%27%27", typeof(FormatException), "invalid SCRAM client key length: 0");
        Action valid = () => RdbUrl.Parse("postgresql://u@h/db?scram_client_key=AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA%3D");
        valid.Should().Throw<NotSupportedException>();
    }

    /// <summary>
    ///     TLS and protocol versions: names compare in any ASCII case, the compiled-in minimum <c>TLSv1.2</c> makes a
    ///     lone lower maximum an invalid range, and a protocol minimum may not pass the maximum (an unset maximum
    ///     follows a minimum above 3.0).
    /// </summary>
    [TestMethod]
    public void Postgres_VersionRanges_FollowLibpq()
    {
        Pg("postgresql://u@h/db?ssl_min_protocol_version=tlsv1.2").Host.Should().Be("h");
        Fails("postgresql://u@h/db?ssl_max_protocol_version=TLSv1.1", typeof(FormatException), "invalid SSL protocol version range");
        Fails("postgresql://u@h/db?ssl_min_protocol_version=TLSv1.4", typeof(FormatException),
            "invalid \"ssl_min_protocol_version\" value: \"TLSv1.4\"");
        Fails("postgresql://u@h/db?min_protocol_version=latest&max_protocol_version=3.0", typeof(FormatException),
            "\"min_protocol_version\" is greater than \"max_protocol_version\"");
        Fails("postgresql://u@h/db?min_protocol_version=3.1", typeof(FormatException), "invalid min_protocol_version value: \"3.1\"");
        Pg("postgresql://u@h/db?max_protocol_version=latest").Host.Should().Be("h");
        Action latest = () => RdbUrl.Parse("postgresql://u@h/db?min_protocol_version=3.2");
        latest.Should().Throw<NotSupportedException>().Which.Message.Should().Contain("min_protocol_version");
    }

    /// <summary>
    ///     libpq's errors win over OptunaSharp's refusals: a URL with a parameter Npgsql cannot express and an invalid
    ///     value fails with libpq's message, as it does in Python.
    /// </summary>
    [TestMethod]
    public void Postgres_LibpqErrors_WinOverRefusals()
    {
        Fails("postgresql://u@h/db?require_auth=password&sslmode=bogus", typeof(FormatException), "invalid sslmode value: \"bogus\"");
        Fails("postgresql://u@h/db?hostaddr=127.0.0.1&sslmode=bogus", typeof(FormatException), "invalid sslmode value: \"bogus\"");
        Fails("postgresql://u@h/db?keepalives_count=3&gssencmode=bogus", typeof(FormatException), "invalid gssencmode value: \"bogus\"");
        Fails("postgresql://u@h/db?require_auth=x&channel_binding=x", typeof(FormatException), "invalid require_auth value: \"x\"");
    }

    /// <summary>
    ///     <c>client_encoding</c> goes through PostgreSQL's <c>clean_encoding_name</c>: every UTF-8 spelling is accepted
    ///     (Npgsql speaks UTF-8); another encoding is refused.
    /// </summary>
    [TestMethod]
    public void Postgres_ClientEncoding_AcceptsEveryUtf8Spelling()
    {
        foreach (string spelling in new[] { "UTF-8", "utf_8", "Unicode", "u.t.f.8" })
            Pg($"postgresql://u@h/db?client_encoding={spelling}").Host.Should().Be("h", spelling);
        Action latin1 = () => RdbUrl.Parse("postgresql://u@h/db?client_encoding=latin1");
        latin1.Should().Throw<NotSupportedException>().Which.Message.Should().Contain("client_encoding");
    }

    /// <summary>
    ///     A connection service file is read as libpq reads it: the group fills in what the URL leaves out (explicit
    ///     values win), and a missing group, a nested service, a bad line or a missing <c>PGSERVICEFILE</c> fail with
    ///     libpq's messages.
    /// </summary>
    [TestMethod]
    [DoNotParallelize]
    public void Postgres_ServiceFile_FollowsLibpq()
    {
        string dir = Path.Combine(Path.GetTempPath(), "optunasharp-pgservice-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        string file = Path.Combine(dir, "pg_service.conf").Replace('\\', '/');
        string? previous = Environment.GetEnvironmentVariable("PGSERVICEFILE");
        try
        {
            File.WriteAllText(file,
                "# test service file\n[good]\nhost=svc-host\nport=6543\nsslmode=require\n\n[nested]\nservice=x\n\n[badkey]\nbogus=1\n\n" +
                "[noeq]\nhost\n[spaced]\nport = 1\n");
            Environment.SetEnvironmentVariable("PGSERVICEFILE", file);

            NpgsqlConnectionStringBuilder good = Pg("postgresql://u@/db?service=good");
            (good.Host, good.Port, good.SslMode).Should().Be(("svc-host", 6543, SslMode.Require));
            NpgsqlConnectionStringBuilder explicitWins = Pg("postgresql://u@other:1/db?service=good");
            (explicitWins.Host, explicitWins.Port).Should().Be(("other", 1));
            Fails("postgresql://u@h/db?service=missing", typeof(FormatException), "definition of service \"missing\" not found");
            Fails("postgresql://u@h/db?service=nested", typeof(FormatException),
                $"nested service specifications not supported in service file \"{file}\", line 8");
            Fails("postgresql://u@h/db?service=badkey", typeof(FormatException), $"syntax error in service file \"{file}\", line 11");
            Fails("postgresql://u@h/db?service=noeq", typeof(FormatException), $"syntax error in service file \"{file}\", line 14");
            Fails("postgresql://u@h/db?service=spaced", typeof(FormatException), $"syntax error in service file \"{file}\", line 16");

            // libpq checks every line's length as it reads, so a lookup that scans past a long line fails on it.
            File.WriteAllText(file, "[good]\nport=1\n[long]\nhost=" + new string('h', 1100) + "\n");
            Pg("postgresql://u@h/db?service=good").Port.Should().Be(1);
            Fails("postgresql://u@h/db?service=long", typeof(FormatException), $"line 4 too long in service file \"{file}\"");
            Fails("postgresql://u@h/db?service=missing", typeof(FormatException), $"line 4 too long in service file \"{file}\"");

            string missing = dir.Replace('\\', '/') + "/nonexistent.conf";
            Environment.SetEnvironmentVariable("PGSERVICEFILE", missing);
            Fails("postgresql://u@h/db?service=good", typeof(FormatException), $"service file \"{missing}\" not found");
        }
        finally
        {
            Environment.SetEnvironmentVariable("PGSERVICEFILE", previous);
            Directory.Delete(dir, recursive: true);
        }
    }

    /// <summary>The deprecated <c>PGREQUIRESSL=1</c> means <c>sslmode=require</c> when neither the URL nor <c>PGSSLMODE</c> sets it.</summary>
    [TestMethod]
    [DoNotParallelize]
    public void Postgres_PgRequireSsl_MeansRequire()
    {
        string? previous = Environment.GetEnvironmentVariable("PGREQUIRESSL");
        string? sslmode = Environment.GetEnvironmentVariable("PGSSLMODE");
        try
        {
            Environment.SetEnvironmentVariable("PGSSLMODE", null);
            Environment.SetEnvironmentVariable("PGREQUIRESSL", "1");
            Pg("postgresql://u@h/db").SslMode.Should().Be(SslMode.Require);
            Pg("postgresql://u@h/db?sslmode=disable").SslMode.Should().Be(SslMode.Disable);
            Environment.SetEnvironmentVariable("PGREQUIRESSL", "0");
            Pg("postgresql://u@h/db").SslMode.Should().Be(SslMode.Prefer);
        }
        finally
        {
            Environment.SetEnvironmentVariable("PGREQUIRESSL", previous);
            Environment.SetEnvironmentVariable("PGSSLMODE", sslmode);
        }
    }

    // ------------------------------------------------------------------------------------------------ MySQL: MySQLdb

    /// <summary>
    ///     mysqlclient compares <c>ssl_mode</c> with <c>strcmp</c>, so only the five upper-case names work, and anything
    ///     else is its <c>NotSupportedError("Unknown ssl_mode specification")</c>.
    /// </summary>
    [TestMethod]
    public void MySqlDb_SslMode_IsCaseSensitive()
    {
        My("mysql://u:p@h/db?ssl_mode=REQUIRED").SslMode.Should().Be(MySqlSslMode.Required);
        My("mysql://u:p@h/db?ssl_mode=VERIFY_IDENTITY").SslMode.Should().Be(MySqlSslMode.VerifyFull);
        foreach (string mode in new[] { "required", "Required", "bogus", "%20REQUIRED" })
            Fails($"mysql://u:p@h/db?ssl_mode={mode}", typeof(FormatException), "Unknown ssl_mode specification");
    }

    /// <summary>
    ///     CPython's argument parsing for <c>_mysql.connect</c>: every given argument is converted in keyword-list order
    ///     before unknown keywords are reported, a tuple where text belongs names its position, and a NUL is refused.
    /// </summary>
    [TestMethod]
    public void MySqlDb_ArgumentParsing_FollowsCPython()
    {
        Fails("mysql://u:p@h/db?ssl_mode=a&ssl_mode=b", typeof(TypeError), "connect() argument 16 must be str, not tuple");
        Fails("mysql://u:p@h/db?ssl_mode=a&ssl_mode=b&foo=1", typeof(TypeError), "connect() argument 16 must be str, not tuple");
        Fails("mysql://u:p@h/db?host=a&host=b", typeof(TypeError), "connect() argument 1 must be str, not tuple");
        Fails("mysql://u:p@h/db?passwd=a&passwd=b", typeof(TypeError), "connect() argument 3 must be str, not tuple");
        Fails("mysql://u:p@h/db?unix_socket=%00", typeof(ValueError), "embedded null character");
        Fails("mysql://u:p@h/db?ssl_mode=REQUIRED&foo=1", typeof(TypeError), "'foo' is an invalid keyword argument for connect()");
        Fails("mysql://u:p@h/db?named_pipe=1", typeof(TypeError), "'str' object cannot be interpreted as an integer");
    }

    /// <summary>
    ///     An integer argument outside the C <c>int</c> range raises CPython's <c>OverflowError</c>, whose message depends
    ///     on the width of C <c>long</c>: 32-bit on Windows, 64-bit elsewhere.
    /// </summary>
    [TestMethod]
    public void MySqlDb_IntegerOverflow_UsesThePlatformsMessage()
    {
        string large = OperatingSystem.IsWindows() ? "Python int too large to convert to C long" : "signed integer is greater than maximum";
        Fails("mysql://u:p@h/db?connect_timeout=99999999999", typeof(OverflowError), large);
        Fails("mysql://u:p@h/db?client_flag=99999999999", typeof(OverflowError), large);
        Fails("mysql://u:p@h:99999999999/db", typeof(OverflowError), large);
        string small = OperatingSystem.IsWindows() ? "Python int too large to convert to C long" : "signed integer is less than minimum";
        Fails("mysql://u:p@h/db?client_flag=-2147483649", typeof(OverflowError), small);
    }

    /// <summary>
    ///     <c>Connection.__init__</c> calls <c>conv.items()</c> first, so a URL's text <c>conv</c> fails there (both
    ///     drivers), before any refusal of OptunaSharp's.
    /// </summary>
    [TestMethod]
    public void MySql_Conv_FailsOnItems()
    {
        Fails("mysql://u:p@h/db?conv=x&foo=1", typeof(AttributeError), "'str' object has no attribute 'items'");
        Fails("mysql+pymysql://u:p@h/db?conv=x", typeof(AttributeError), "'str' object has no attribute 'items'");
        Fails("mysql+pymysql://u:p@h/db?conv=a&conv=b", typeof(AttributeError), "'tuple' object has no attribute 'items'");
        Fails("mysql+pymysql://u:p@h/db?init_command=x&conv=y", typeof(AttributeError), "'str' object has no attribute 'items'");
    }

    /// <summary>
    ///     <c>client_flag</c>: the capability bits the drivers set on their own are accepted, <c>INTERACTIVE</c> maps to
    ///     <c>InteractiveSession</c>, MySQLdb's <c>COMPRESS</c> to compression; bits that change how SQL parses are refused.
    /// </summary>
    [TestMethod]
    public void MySql_ClientFlag_AcceptsWhatMySqlConnectorHonors()
    {
        My("mysql+pymysql://u:p@h/db?client_flag=1").Server.Should().Be("h");
        My("mysql+pymysql://u:p@h/db?client_flag=196608").Server.Should().Be("h");
        My("mysql+pymysql://u:p@h/db?client_flag=1024").InteractiveSession.Should().BeTrue();
        My("mysql://u:p@h/db?client_flag=32").UseCompression.Should().BeTrue();
        foreach (string url in new[] { "mysql+pymysql://u:p@h/db?client_flag=16", "mysql+pymysql://u:p@h/db?client_flag=32", "mysql://u:p@h/db?client_flag=256" })
        {
            Action act = () => RdbUrl.Parse(url);
            act.Should().Throw<NotSupportedException>(url).Which.Message.Should().Contain("client_flag");
        }
    }

    // ------------------------------------------------------------------------------------------------ MySQL: PyMySQL

    /// <summary>
    ///     PyMySQL's own failures on values a URL can only give as text: <c>?ssl=</c> has no <c>get</c>, an unknown
    ///     <c>charset</c> looks up <c>None</c>; known names match case-insensitively (<c>name.lower()</c>).
    /// </summary>
    [TestMethod]
    public void PyMySql_TextValues_FailLikePyMySql()
    {
        Fails("mysql+pymysql://u:p@h/db?ssl=x", typeof(AttributeError), "'str' object has no attribute 'get'");
        Fails("mysql+pymysql://u:p@h/db?ssl=a&ssl=b", typeof(AttributeError), "'tuple' object has no attribute 'get'");
        Fails("mysql+pymysql://u:p@h/db?charset=bogus", typeof(AttributeError), "'NoneType' object has no attribute 'encoding'");
        Fails("mysql+pymysql://u:p@h/db?init_command=x&charset=bogus", typeof(AttributeError), "'NoneType' object has no attribute 'encoding'");
        Fails("mysql+pymysql://u:p@h/db?charset=bogus&connect_timeout=0", typeof(ValueError), "connect_timeout should be >0 and <=31536000");
        My("mysql+pymysql://u:p@h/db?charset=UTF8MB4").CharacterSet.Should().Be("utf8mb4");
        My("mysql+pymysql://u:p@h/db?charset=Utf8").CharacterSet.Should().Be("utf8mb4");
    }

    /// <summary>PyMySQL encodes a text password as Latin-1, so a character outside it fails before connecting, with CPython's message.</summary>
    [TestMethod]
    public void PyMySql_Password_MustBeLatin1()
    {
        Fails("mysql+pymysql://u:%E2%82%AC@h/db", typeof(UnicodeEncodeError),
            "'latin-1' codec can't encode character '\\u20ac' in position 0: ordinal not in range(256)");
        Fails("mysql+pymysql://u:a%E2%82%AC%E2%82%ACb@h/db", typeof(UnicodeEncodeError),
            "'latin-1' codec can't encode characters in position 1-2: ordinal not in range(256)");
        My("mysql+pymysql://u:%C3%A9@h/db").Password.Should().Be("\U000000E9");
        // MySQLdb sends the password as UTF-8, so the same password is fine there.
        My("mysql://u:%E2%82%AC@h/db").Password.Should().Be("\U000020AC");
    }

    /// <summary>
    ///     The SSL files load before connecting in Python: a missing CA or certificate is
    ///     <c>FileNotFoundError: [Errno 2] No such file or directory</c>; <c>ssl_key_password</c> from a URL never reaches
    ///     PyMySQL's <c>load_cert_chain</c> (SQLAlchemy moved the certificate into the dictionary PyMySQL replaces), so
    ///     it is accepted and ignored.
    /// </summary>
    [TestMethod]
    public void PyMySql_SslFiles_LoadLikePython()
    {
        Fails("mysql+pymysql://u:p@h/db?ssl_ca=nonexistent", typeof(FileNotFoundError), "[Errno 2] No such file or directory");
        Fails("mysql+pymysql://u:p@h/db?ssl_cert=nonexistent", typeof(FileNotFoundError), "[Errno 2] No such file or directory");
        Fails("mysql+pymysql://u:p@h/db?ssl_capath=x&ssl_ca=nonexistent", typeof(FileNotFoundError), "[Errno 2] No such file or directory");
        My("mysql+pymysql://u:p@h/db?ssl_key_password=x").Server.Should().Be("h");
        My("mysql+pymysql://u:p@h/db?ssl_key_password=x&ssl_verify_cert=1").SslMode.Should().Be(MySqlSslMode.VerifyCA);
    }

    /// <summary>
    ///     Option files: one <c>configparser</c> cannot open is skipped, leaving the arguments as they were except that
    ///     the port goes through <c>int()</c> (so a query <c>?port=5</c> works there, and <c>?port=x</c> fails); one it
    ///     would read is refused, since OptunaSharp does not apply option files.
    /// </summary>
    [TestMethod]
    public void PyMySql_OptionFiles_FollowConfigParser()
    {
        My("mysql+pymysql://u:p@h/db?read_default_file=nonexistent&port=5").Port.Should().Be(5u);
        Fails("mysql+pymysql://u:p@h/db?read_default_file=nonexistent&port=x", typeof(ValueError), "invalid literal for int() with base 10: 'x'");
        Fails("mysql+pymysql://u:p@h/db?port=5", typeof(ValueError), "port should be of type int");
        string file = Path.GetTempFileName();
        try
        {
            File.WriteAllText(file, "[client]\nport=3307\n");
            Action act = () => RdbUrl.Parse("mysql+pymysql://u:p@h/db?read_default_file=" + Uri.EscapeDataString(file));
            act.Should().Throw<NotSupportedException>().Which.Message.Should().Contain("read_default_file");
        }
        finally
        {
            File.Delete(file);
        }
    }
}

using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Numerics;
using AwesomeAssertions;
using Microsoft.Data.Sqlite;
using NumSharp;
using OptunaSharp.Internal;
using OptunaSharp.Storages;
using OptunaSharp.Storages.EFCore;
using OptunaSharp.Storages.Journal;
using SqliteRdbStorage = OptunaSharp.Storages.RDBStorage;

namespace OptunaSharp.Tests.Interop;

/// <summary>
///     Review item P3-3 and the connection-string audit, live: every storage URL OptunaSharp accepts is read the way
///     the Python stack Optuna runs on reads it — SQLAlchemy plus pysqlite for SQLite, SQLAlchemy plus psycopg2's
///     <c>make_dsn</c> and libpq's <c>parse_dsn</c> for PostgreSQL, SQLAlchemy's <c>create_connect_args</c> for MySQL,
///     and redis-py's <c>ConnectionPool.from_url</c> (plus what its first connect checks) for Redis. Each URL goes
///     through both sides in this process, and the canonical outcomes must be equal: what was opened or parsed, or
///     which error was raised with which message.
/// </summary>
/// <remarks>
///     <para>Python error kinds map onto the port's as follows: <c>ArgumentError</c> and its <c>NoSuchModuleError</c>
///     subclass are a plain <see cref="ArgumentException" />; psycopg2's <c>ProgrammingError</c> for a malformed DSN is
///     a <see cref="FormatException" />; redis-py's <c>ConnectionError</c>, <c>RedisError</c> and <c>DataError</c> are a
///     <see cref="ValueError" /> with the same text; <c>ValueError</c>, <c>TypeError</c> and <c>OverflowError</c> keep
///     their names.</para>
///     <para>Documented divergences are tested separately, not excused: the URLs a .NET client cannot honor and the
///     values Python only refuses at the server must be accepted by Python and refused by the port up front. SQLite's
///     default busy timeout (30 s in the port, Python's 5 s) is compared only for URLs that set <c>?timeout=</c>.
///     PostgreSQL, MySQL and Redis cases need their Python driver (psycopg2, redis-py); without it they report
///     Inconclusive.</para>
/// </remarks>
[TestClass]
public class StorageUrlStandardParityTests : OptunaInteropTestBase
{
    /// <summary>The separator between URLs and outcomes crossing the language boundary; URLs may hold newlines.</summary>
    private const char Separator = '\x1e';

    /// <summary>The scratch folder for the SQLite cases' databases.</summary>
    private string _dir = null!;

    /// <summary>
    ///     Creates the scratch folder and keeps its real path (every symlinked component resolved), so the paths both
    ///     sides hand to SQLite are the ones SQLite reports back through <c>pragma_database_list</c>.
    /// </summary>
    /// <remarks>
    ///     On macOS the temp folder lives under <c>/var</c>, a symlink to <c>/private/var</c>; SQLite resolves it, so an
    ///     unresolved folder never matched the reported <c>/private/var/…</c> file (the macOS leg failed here at 49f53dc and
    ///     7d1db43). Windows and Linux temp folders have no symlinked component, and their paths are unchanged.
    /// </remarks>
    /// <exception cref="IOException">The folder cannot be created or a link in its path cannot be read.</exception>
    [TestInitialize]
    public void CreateFolder()
    {
        string created = Path.Combine(Path.GetTempPath(), $"optunasharp-urlstd-{Guid.NewGuid():N}");
        Directory.CreateDirectory(created);
        _dir = RealPath(created);
    }

    /// <summary>
    ///     <paramref name="path" /> with every symlinked component replaced by its final target, the form SQLite reports for a
    ///     database file (like POSIX <c>realpath</c>, which .NET does not expose).
    /// </summary>
    /// <param name="path">An existing folder.</param>
    /// <returns>The absolute path with no symlinked component.</returns>
    /// <exception cref="IOException">A link in the path cannot be read.</exception>
    private static string RealPath(string path)
    {
        string full = Path.GetFullPath(path);
        string root = Path.GetPathRoot(full)!;
        string current = root;
        foreach (string part in full[root.Length..].Split(
                     new[] { Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar }, StringSplitOptions.RemoveEmptyEntries))
        {
            current = Path.Combine(current, part);
            // ResolveLinkTarget returns null for a plain folder; a link (macOS /var -> private/var) becomes its final
            // target, resolved against the link's own folder when the target is relative.
            if (new DirectoryInfo(current).ResolveLinkTarget(returnFinalTarget: true) is { } target)
                current = target.FullName;
        }

        return current;
    }

    /// <summary>Deletes the scratch folder once both sides have released their files.</summary>
    [TestCleanup]
    public void DeleteFolder()
    {
        SqliteConnection.ClearAllPools();
        try
        {
            Directory.Delete(_dir, recursive: true);
        }
        catch (IOException)
        {
            // A Python handle can still be closing; the temp folder is cleaned by the OS eventually.
        }
    }

    /// <summary>Reports Inconclusive when <paramref name="module" /> cannot be imported by the embedded Python.</summary>
    /// <param name="module">The Python module the case needs.</param>
    private void RequireModule(string module)
    {
        try
        {
            PyExec($"import {module}");
        }
        catch (Exception e)
        {
            Assert.Inconclusive($"Python module '{module}' is not installed ({e.Message.Split('\n')[0]}).");
        }
    }

    /// <summary>Runs <paramref name="function" /> (a Python function of one URL) over the URLs and returns its outcomes.</summary>
    /// <param name="urls">The URLs.</param>
    /// <param name="function">The name of a function the scope already defines.</param>
    /// <returns>One outcome per URL.</returns>
    private string[] PythonOutcomes(IReadOnlyList<string> urls, string function)
    {
        PySet("_sus_urls", string.Join(Separator, urls));
        PyExec($"_sus_result = '\\x1e'.join({function}(u) for u in _sus_urls.split('\\x1e'))");
        return PyStr("_sus_result").Split(Separator);
    }

    /// <summary>The port's outcome for an exception, in the canonical <c>kind|message</c> form.</summary>
    /// <param name="e">The exception.</param>
    /// <returns>The canonical outcome.</returns>
    private static string Failure(Exception e) => e switch
    {
        // Python's UnicodeDecodeError and UnicodeEncodeError are ValueErrors; they report their own names.
        UnicodeDecodeError => $"UnicodeDecodeError|{e.Message}",
        UnicodeEncodeError => $"UnicodeEncodeError|{e.Message}",
        ValueError => $"ValueError|{e.Message}",
        TypeError => $"TypeError|{e.Message}",
        OverflowException => $"OverflowError|{e.Message}",
        FormatException => $"ProgrammingError|{e.Message}",
        NotSupportedException when e.Message.StartsWith(StorageUrlResolver.ImportErrorMessage, StringComparison.Ordinal)
            => $"ImportError|{StorageUrlResolver.ImportErrorMessage}",
        NotSupportedException => $"NotSupported|{e.Message}",
        SqliteException => "OperationalError|",
        ArgumentException when e.GetType() == typeof(ArgumentException) => $"ArgumentError|{e.Message}",
        _ => $"{e.GetType().Name}|{e.Message}",
    };

    // ------------------------------------------------------------------------------------------------ SQLite

    /// <summary>
    ///     SQLite URLs; <c>{dir}</c> is the scratch folder. Every query argument pysqlite reads or ignores, the template,
    ///     URI mode, the in-memory forms, and the refusals.
    /// </summary>
    private static readonly string[] SqliteCases =
    {
        "sqlite:///{dir}/a.db",
        "sqlite:///{dir}/b.db?timeout=10",
        "sqlite:///{dir}/c.db?timeout=0.5",
        "sqlite:///{dir}/d.db?timeout=0",
        "sqlite:///{dir}/e.db?timeout=-1",
        "sqlite:///{dir}/f.db?foo=bar&timeout=3",
        "sqlite:///{dir}/g.db?timeout=abc",
        "sqlite:///{dir}/h.db?check_same_thread=maybe",
        "sqlite:///{dir}/i.db?check_same_thread=false&cached_statements=10&detect_types=0",
        "sqlite:///{dir}/j{SCHEMA_VERSION}.db",
        "sqlite:///{dir}/k.db?isolation_level=IMMEDIATE",
        "sqlite:///file:{dir}/l.db?uri=true",
        "sqlite:///{dir}/m.db?timeout=1&timeout=2",
        "sqlite+pysqlite:///{dir}/n.db?timeout=4",
        "sqlite://",
        "sqlite:///:memory:",
        "sqlite://host/x.db",
        "study.db",
        "sqlite+pysqlcipher:///{dir}/o.db",
        "sqlite:///{dir}/p.db?timeout=1e9",
    };

    /// <summary>
    ///     Every SQLite URL opens the same file with the same busy timeout through <c>optuna.storages.RDBStorage</c> and
    ///     the SQLite package's <see cref="SqliteRdbStorage" />, or fails with the same error. The query string is never
    ///     part of the file name (P3-3).
    /// </summary>
    [TestMethod]
    [TestCategory("RequiresPython")]
    public void Sqlite_UrlsOpenWhatPythonOpens()
    {
        string dir = _dir.Replace('\\', '/');
        // Each side gets its own folder, so the two never share (or pre-create) a file.
        string[] pyUrls = SqliteCases.Select(c => c.Replace("{dir}", dir + "/py", StringComparison.Ordinal)).ToArray();
        string[] csUrls = SqliteCases.Select(c => c.Replace("{dir}", dir + "/cs", StringComparison.Ordinal)).ToArray();
        Directory.CreateDirectory(Path.Combine(_dir, "py"));
        Directory.CreateDirectory(Path.Combine(_dir, "cs"));

        string[] cs = csUrls.Select(url =>
        {
            try
            {
                using var storage = new SqliteRdbStorage(url);
                using SqliteCommand command = storage.Connection.CreateCommand();
                command.CommandText = "SELECT file FROM pragma_database_list WHERE name = 'main'";
                string file = (command.ExecuteScalar() as string ?? "").Replace('\\', '/').Replace(dir + "/cs", "{dir}", StringComparison.Ordinal);
                command.CommandText = "PRAGMA busy_timeout";
                return $"OK|{file}|{(long)command.ExecuteScalar()!}";
            }
            catch (Exception e)
            {
                return Failure(e);
            }
        }).ToArray();
        PySet("_sus_dir", dir + "/py");
        PyExec(
            "def _sus_sqlite(url):\n" +
            "    try:\n" +
            "        st = optuna.storages.RDBStorage(url)\n" +
            "    except Exception as e:\n" +
            "        name = type(e).__name__\n" +
            "        if name == 'NoSuchModuleError': name = 'ArgumentError'\n" +
            "        if name == 'OperationalError': return 'OperationalError|'\n" +
            "        if name == 'ImportError': return 'ImportError|Failed to import DB access module for the specified storage URL. Please install appropriate one.'\n" +
            "        return f'{name}|{e}'\n" +
            "    try:\n" +
            "        with st.engine.connect() as c:\n" +
            "            rows = c.exec_driver_sql('PRAGMA database_list').fetchall()\n" +
            "            busy = c.exec_driver_sql('PRAGMA busy_timeout').scalar()\n" +
            "    finally:\n" +
            "        st.engine.dispose()\n" +
            "    f = ([r[2] for r in rows if r[1] == 'main'][0] or '').replace('\\\\', '/').replace(_sus_dir, '{dir}')\n" +
            "    return f'OK|{f}|{busy}'\n");
        string[] py = PythonOutcomes(pyUrls, "_sus_sqlite");

        for (int i = 0; i < SqliteCases.Length; i++)
        {
            // The port's default busy timeout is 30 s where Python's is 5 s; compare it only when the URL sets one.
            // CPython computes `(int)(timeout * 1000)` in C, which is undefined beyond the 32-bit range: x86-64
            // yields INT_MIN (busy handler off, 0) and arm64 saturates to INT_MAX. The port gives x86-64's answer on
            // every host by design (PyNumbers.CTruncateToInt32), so `timeout=1e9` is compared on x64 hosts only.
            bool cUndefinedOnThisHost = !PythonSession.BitExactHost && SqliteCases[i].Contains("timeout=1e9", StringComparison.Ordinal);
            bool timed = SqliteCases[i].Contains("timeout=", StringComparison.Ordinal) && !cUndefinedOnThisHost;
            string Shape(string outcome) => !timed && outcome.StartsWith("OK|", StringComparison.Ordinal)
                ? outcome[..outcome.LastIndexOf('|')]
                : outcome;
            Shape(cs[i]).Should().Be(Shape(py[i]), $"'{SqliteCases[i]}' must open what Python opens");
        }
        cs[0].Should().StartWith("OK|{dir}/a.db", "a plain file URL must open on both sides");
        cs[1].Should().Be("OK|{dir}/b.db|10000", "the query string must not become part of the file name (P3-3)");
    }

    // ------------------------------------------------------------------------------------------------ PostgreSQL

    /// <summary>PostgreSQL URLs over psycopg2 (the default driver): components, libpq parameters, multi-host forms and refusals.</summary>
    private static readonly string[] PostgresCases =
    {
        "postgresql://u:p@localhost/db",
        "postgresql://u:p@localhost:5433/db?sslmode=require",
        "postgresql+psycopg2://u:p@h/db?connect_timeout=7&application_name=optuna",
        "postgresql://u@/db?host=/var/run/postgresql",
        "postgresql://u:p@/db?host=h1:5433&host=h2",
        "postgresql://a%40b:p%3Aw@h/db",
        "postgresql://u:p@h/db?SSL%20Mode=Require",
        "postgresql://u:p@h/db?Pooling=false",
        "postgresql://u:p@h/db?options=-c%20search_path%3Doptuna",
        "postgresql://u:p@h/db?keepalives=0&keepalives_idle=5",
        "postgresql://u:p@h/db?sslmode=verify-full&sslrootcert=system",
        "postgresql://u:p@[::1]:5432/db",
        "postgresql://u:p@h:abc/db",
        "postgresql://u:p@h/db?target_session_attrs=read-write&load_balance_hosts=random",
        "postgresql://u:p@h/db?client_encoding=utf8",
        "postgresql://u:p@h/db?dsn=host%3Dother%20port%3D6000",
        "postgresql://u@h/db?dsn=postgresql%3A%2F%2Fx%3A5%2Fy%3Fsslmode%3Drequire",
        "postgresql:///?dsn=postgresql%3A%2F%2Fa%40x%3A5%2Cz%3A6%2Fy%3Fapplication_name%3Dq%2520r",
        "postgresql:///?dsn=postgresql%3A%2F%2F%5B%3A%3A1%5D%3A7%2Fd%3Fssl%3Dtrue",
        "postgresql:///?dsn=postgresql%3A%2F%2Fh%2Fd%3Fbogus%3D1",
        "postgresql:///?dsn=postgresql%3A%2F%2Fh%2F%25ff",
        "postgresql://u@h/db?dsn=a&dsn=b",
        "postgresql://u:p@h/db?requiressl=1",
        "postgresql://u:p@h/{SCHEMA_VERSION}",
        "postgres://u:p@h/db",
        "postgresql+nosuchdriver://u:p@h/db",
    };

    /// <summary>
    ///     Every PostgreSQL URL yields the libpq parameters psycopg2 builds for it (<c>parse_dsn(make_dsn(...))</c> over
    ///     SQLAlchemy's <c>create_connect_args</c>), or the same error.
    /// </summary>
    [TestMethod]
    [TestCategory("RequiresPython")]
    public void Postgres_UrlsYieldPsycopg2sLibpqParameters()
    {
        RequireModule("psycopg2");
        string[] cs = PostgresCases.Select(url =>
        {
            try
            {
                OrderedDictLike parameters = new(PostgresUrl.LibpqParameters(StorageUrlResolver.Prepare(url)));
                return "OK|" + parameters.Canonical();
            }
            catch (Exception e)
            {
                return Failure(e);
            }
        }).ToArray();
        PyExec(
            "import psycopg2.extensions as _sus_ext\n" +
            "import sqlalchemy as _sus_sa\n" +
            "def _sus_pg(url):\n" +
            "    try:\n" +
            "        filled = url.format(SCHEMA_VERSION=12)\n" +
            "        engine = _sus_sa.create_engine(filled)\n" +
            "        cargs, cparams = engine.dialect.create_connect_args(engine.url)\n" +
            "        cparams = {k: v for k, v in cparams.items() if k not in ('async', 'async_')}\n" +
            "        d = _sus_ext.parse_dsn(_sus_ext.make_dsn(*cargs, **cparams))\n" +
            "    except Exception as e:\n" +
            "        name = type(e).__name__\n" +
            "        if name == 'NoSuchModuleError': name = 'ArgumentError'\n" +
            "        return f'{name}|{str(e).strip()}'\n" +
            "    return 'OK|' + ';'.join(f'{k}={d[k]}' for k in sorted(d))\n");
        string[] py = PythonOutcomes(PostgresCases, "_sus_pg");
        cs.Should().Equal(py, "every PostgreSQL URL must yield psycopg2's libpq parameters");
    }

    // ------------------------------------------------------------------------------------------------ MySQL

    /// <summary>MySQL and MariaDB URLs over MySQLdb, PyMySQL and MySQL Connector: components, typed arguments, SSL and refusals.</summary>
    private static readonly string[] MySqlCases =
    {
        "mysql://u:p@h/db",
        "mysql://u:p@h:3307/db?charset=utf8mb4",
        "mysql+pymysql://u:p@h/db?ssl_ca=/x/ca.pem&ssl_check_hostname=false",
        "mysql+pymysql://u:p@h/db?connect_timeout=5&read_timeout=10",
        "mysql://u@h/db?unix_socket=/tmp/m.sock",
        "mariadb://u:p@h/db",
        "mysql+mysqldb://u:p@h/db?ssl_mode=REQUIRED",
        "mysql+pymysql://u:p@h/db?local_infile=1",
        "mysql://u:p@h/db?compress=true",
        "mysql+pymysql://u:p@h/db?foo=1",
        "mysql://u:p@h:abc/db",
        "mysql+mysqlconnector://u:p@h/db?charset=utf8mb4",
        "mysql://u:p@h/db?ssl_key=/k&ssl_cert=/c",
        "mysql+pymysql://u:%40p@h/db",
        "mysql://u:p@h/db?connect_timeout=x",
        "mysql://u:p@h/{SCHEMA_VERSION}",
    };

    /// <summary>
    ///     Every MySQL URL yields the keyword arguments SQLAlchemy's <c>create_connect_args</c> hands the driver (without
    ///     the <c>FOUND_ROWS</c> client flag SQLAlchemy adds once the driver module is imported), or the same error.
    /// </summary>
    [TestMethod]
    [TestCategory("RequiresPython")]
    public void MySql_UrlsYieldSqlAlchemysConnectArgs()
    {
        string[] cs = MySqlCases.Select(url =>
        {
            try
            {
                var args = new OrderedDictLike(MySqlUrl.ConnectArgs(StorageUrlResolver.Prepare(url)));
                return "OK|" + args.Json();
            }
            catch (Exception e)
            {
                return Failure(e);
            }
        }).ToArray();
        PyExec(
            "import sqlalchemy as _sus_sa\n" +
            "from sqlalchemy.engine import make_url as _sus_make_url\n" +
            "def _sus_mysql(url):\n" +
            "    try:\n" +
            "        u = _sus_make_url(url.format(SCHEMA_VERSION=12))\n" +
            "        u2, _p, _k = u._instantiate_plugins({})\n" +
            "        dialect_cls = u2._get_entrypoint().get_dialect_cls(u2)\n" +
            "        try:\n" +
            "            d = _sus_sa.create_engine(u).dialect\n" +
            "        except ImportError:\n" +
            "            d = dialect_cls()\n" +
            "        cargs, cparams = d.create_connect_args(u2)\n" +
            "    except Exception as e:\n" +
            "        name = type(e).__name__\n" +
            "        if name == 'NoSuchModuleError': name = 'ArgumentError'\n" +
            "        return f'{name}|{str(e).strip()}'\n" +
            "    cparams = {k: (list(v) if isinstance(v, tuple) else v) for k, v in cparams.items() if k != 'client_flag'}\n" +
            "    return 'OK|' + json.dumps(cparams, sort_keys=True)\n");
        string[] py = PythonOutcomes(MySqlCases, "_sus_mysql");
        cs.Should().Equal(py, "every MySQL URL must yield SQLAlchemy's connect arguments");
    }

    // ------------------------------------------------------------------------------------------------ Redis

    /// <summary>Redis URLs on which the port and redis-py must agree exactly, errors included.</summary>
    private static readonly string[] RedisCases =
    {
        "redis://localhost:6379", "redis://", "redis://:pw@h:6380/3", "redis://u:p%40s@h", "redis://u:p@s@h",
        "rediss://h:6380", "unix:///tmp/redis.sock?db=1", "unix://:pw@/tmp/redis.sock", "unix://tmp/redis.sock",
        "redis://h/2?db=5", "redis://h/abc", "redis://h/1/2", "redis://h/%31", "redis://h/ 1", "redis://h?db=abc",
        "redis://h?db=1&db=2", "redis://h?db=", "redis://h?password=a+b", "redis://u:p@h?username=x&password=y",
        "redis://u:@h", "redis://h?socket_timeout=x", "redis://h?socket_connect_timeout=2.5", "rediss://h?ssl_cert_reqs=none",
        "rediss://h?ssl_ca_certs=/x/ca.pem&ssl_certfile=c&ssl_keyfile=k", "rediss://h?ssl_check_hostname=false",
        "redis://h?foo=bar", "redis://H.Example.COM:6379", "redis://[::1]:6379", "redis://h:abc", "redis://h:99999",
        "redis://h:0", "REDIS://h", "redis:/h", "http://h", "redis://h#frag", "redis://h?protocol=3", "redis://h?protocol=x",
        "redis://h?client_name=x", "redis://h?ssl_min_version=3", "redis://h%20x", "redis://h?timeout=3", "redis://h/+1",
        "redis:///2", "unix://", "redis://h ", "redis://h\n", "redis://h?socket_timeout=-1", "redis://h?socket_timeout=nan",
        "redis://h?socket_timeout=inf", "redis://h?socket_timeout=1e10", "redis://?host=qh&port=6390", "redis://h:6391?port=6390",
        "redis://?port=abc", "unix:///s?host=x", "unix://?path=/qp", "redis://h?path=/x", "redis://h?protocol=4",
        "redis://h?driver_info=x", "redis://h?max_connections=-1", "redis://h?connection_class=x", "rediss://h?connection_class=x",
        "redis://h?self=1", "redis://u:p@h?credential_provider=x", "redis://h?parser_class=x", "redis://h?=1",
        "rediss://h?ssl_cert_reqs=NONE", "rediss://h?ssl_cert_reqs=none&ssl_check_hostname=true", "rediss://h?ssl_min_version=771",
        "rediss://h?ssl_min_version=999", "rediss://h?ssl_min_version=-3", "rediss://h?ssl_include_verify_flags=BOGUS",
        "rediss://h?ssl_validate_ocsp=1&ssl_validate_ocsp_stapled=1", "redis://[::1", "redis://h:6379:1", "redis://H%4A",
        "rediss://u:p@H:6380/2?ssl_cert_reqs=required&ssl_check_hostname=false&socket_timeout=3&client_name=c&protocol=3",
        "unix:///var/run/redis.sock?password=p&db=4&socket_timeout=2", "redis://h?a=1;b=2", "redis://h?client_name=ok&lib_name=lib",
        "redis://h?max_connections=0&retry_on_timeout=yes&health_check_interval=30&decode_responses=true&encoding=UTF8",
    };

    /// <summary>
    ///     URLs redis-py accepts (it fails on them only at the server or the socket, or not at all) that the port refuses
    ///     up front, each with the exception type it must raise.
    /// </summary>
    private static readonly (string Url, Type Kind)[] RedisRefusedUpFront =
    {
        ("redis://h?client_name=a%20b", typeof(ValueError)),
        ("redis://h?db=-1", typeof(ValueError)),
        ("redis://?port=99999", typeof(ValueError)),
        ("redis://?port=0", typeof(ValueError)),
        ("redis://h?retry=x", typeof(NotSupportedException)),
        ("redis://h?event_dispatcher=x", typeof(NotSupportedException)),
        ("redis://h?encoding=utf-16", typeof(NotSupportedException)),
        ("rediss://h?ssl_ciphers=HIGH", typeof(NotSupportedException)),
        ("rediss://h?ssl_include_verify_flags=VERIFY_X509_STRICT", typeof(NotSupportedException)),
        ("rediss://h?ssl_validate_ocsp_stapled=true", typeof(NotSupportedException)),
        ("redis://h?socket_timeout=0", typeof(NotSupportedException)),
    };

    /// <summary>
    ///     The redis-py twin: <c>ConnectionPool.from_url</c>, the client, the connection object, then the checks the
    ///     first connect makes before any network traffic (socket timeouts, the TLS version, the OCSP modes), as one
    ///     canonical line.
    /// </summary>
    private const string RedisTwin =
        "import socket as _sus_socket, ssl as _sus_ssl, redis as _sus_redis\n" +
        "from redis.connection import ConnectionPool as _SusPool, SSLConnection as _SusSsl, UnixDomainSocketConnection as _SusUnix\n" +
        "_SUS_KIND = {'ConnectionError': 'ValueError', 'RedisError': 'ValueError', 'DataError': 'ValueError'}\n" +
        "def _sus_redis_err(e):\n" +
        "    return f'{_SUS_KIND.get(type(e).__name__, type(e).__name__)}|{e}'\n" +
        "def _sus_redis_url(url):\n" +
        "    try:\n" +
        "        pool = _SusPool.from_url(url)\n" +
        "        _sus_redis.Redis(connection_pool=pool)\n" +
        "        c = pool.make_connection()\n" +
        "    except Exception as e:\n" +
        "        return _sus_redis_err(e)\n" +
        "    try:\n" +
        "        s = _sus_socket.socket()\n" +
        "        try:\n" +
        "            s.settimeout(c.socket_connect_timeout)\n" +
        "            s.settimeout(c.socket_timeout)\n" +
        "        finally:\n" +
        "            s.close()\n" +
        "        if isinstance(c, _SusSsl):\n" +
        "            ctx = _sus_ssl.create_default_context()\n" +
        "            if c.ssl_min_version is not None:\n" +
        "                ctx.minimum_version = c.ssl_min_version\n" +
        "            if c.ssl_validate_ocsp_stapled and c.ssl_validate_ocsp:\n" +
        "                raise _sus_redis.exceptions.RedisError('Either an OCSP staple or pure OCSP connection must be validated - not both.')\n" +
        "    except Exception as e:\n" +
        "        return _sus_redis_err(e)\n" +
        "    unix = isinstance(c, _SusUnix)\n" +
        "    p = ['unix' if unix else 'ssl' if isinstance(c, _SusSsl) else 'tcp']\n" +
        "    p += [f'path={c.path}'] if unix else [f'host={c.host}', f'port={c.port}']\n" +
        "    p += [f'db={c.db}', f'username={c.username}', f'password={c.password}',\n" +
        "          f'socket_timeout={float(c.socket_timeout)!r}', f'socket_connect_timeout={float(c.socket_connect_timeout)!r}',\n" +
        "          f'client_name={c.client_name}', f'lib_name={c.driver_info.formatted_name if c.driver_info else None}', f'protocol={c.protocol}']\n" +
        "    if isinstance(c, _SusSsl):\n" +
        "        reqs = {_sus_ssl.CERT_NONE: 'none', _sus_ssl.CERT_OPTIONAL: 'optional', _sus_ssl.CERT_REQUIRED: 'required'}[c.cert_reqs]\n" +
        "        p += [f'cert_reqs={reqs}', f'check_hostname={c.check_hostname}', f'ca_certs={c.ca_certs}', f'ca_data={c.ca_data}',\n" +
        "              f'ca_path={c.ca_path}', f'certfile={c.certfile}', f'keyfile={c.keyfile}', f'ssl_password={c.certificate_password}',\n" +
        "              f'min_version={None if c.ssl_min_version is None else int(c.ssl_min_version)}']\n" +
        "    return 'OK|' + ';'.join(p)\n";

    /// <summary>The port's canonical line for a Redis URL, in the twin's format.</summary>
    /// <param name="url">The URL.</param>
    /// <returns>The outcome.</returns>
    private static string CSharpRedis(string url)
    {
        RedisEndpoint e;
        try
        {
            e = RedisUrl.Parse(url);
        }
        catch (Exception ex)
        {
            return Failure(ex);
        }
        static string S(object? v) => v switch
        {
            null => "None",
            bool b => b ? "True" : "False",
            _ => v.ToString()!,
        };
        bool unix = e.UnixSocketPath is not null;
        var p = new List<string> { unix ? "unix" : e.Ssl ? "ssl" : "tcp" };
        p.AddRange(unix ? new[] { $"path={e.UnixSocketPath}" } : new[] { $"host={e.Host}", $"port={e.Port}" });
        p.AddRange(new[]
        {
            $"db={e.Db ?? 0}", $"username={S(e.User)}", $"password={S(e.Password)}",
            $"socket_timeout={PythonRepr.FloatRepr(e.SocketTimeout)}", $"socket_connect_timeout={PythonRepr.FloatRepr(e.SocketConnectTimeout)}",
            $"client_name={S(e.ClientName)}", $"lib_name={e.LibraryName ?? "redis-py"}", $"protocol={e.Protocol ?? 3}",
        });
        if (e.Ssl)
            p.AddRange(new[]
            {
                $"cert_reqs={e.SslCertReqs}", $"check_hostname={S(e.SslCheckHostname)}", $"ca_certs={S(e.SslCaCerts)}",
                $"ca_data={S(e.SslCaData)}", $"ca_path={S(e.SslCaPath)}", $"certfile={S(e.SslCertFile)}", $"keyfile={S(e.SslKeyFile)}",
                $"ssl_password={S(e.SslPassword)}", $"min_version={S(e.SslMinVersion)}",
            });
        return "OK|" + string.Join(";", p);
    }

    /// <summary>
    ///     Every Redis URL is read as redis-py 8 reads it: the same connection class, host, port, path, database,
    ///     credentials, timeouts (5 s by default), names, protocol and TLS settings, or the same error with the same text
    ///     — the first-connect checks included.
    /// </summary>
    [TestMethod]
    [TestCategory("RequiresPython")]
    public void Redis_UrlsParseAsRedisPyParsesThem()
    {
        RequireModule("redis");
        string[] cs = RedisCases.Select(CSharpRedis).ToArray();
        PyExec(RedisTwin);
        string[] py = PythonOutcomes(RedisCases, "_sus_redis_url");
        for (int i = 0; i < RedisCases.Length; i++)
            cs[i].Should().Be(py[i], $"'{RedisCases[i]}' must parse as redis-py parses it");
    }

    /// <summary>
    ///     The documented divergences, held to their definition: redis-py accepts each of these URLs (it fails only at
    ///     the server or the socket, or never), and the port refuses it up front with the stated exception type.
    /// </summary>
    [TestMethod]
    [TestCategory("RequiresPython")]
    public void Redis_RefusedUpFront_AreAcceptedByRedisPy()
    {
        RequireModule("redis");
        PyExec(RedisTwin);
        string[] urls = RedisRefusedUpFront.Select(c => c.Url).ToArray();
        string[] py = PythonOutcomes(urls, "_sus_redis_url");
        for (int i = 0; i < urls.Length; i++)
        {
            py[i].Should().StartWith("OK|", $"redis-py constructs a connection for '{urls[i]}'");
            Action act = () => RedisUrl.Parse(urls[i]);
            act.Should().Throw<Exception>().Which.GetType().Should().Be(RedisRefusedUpFront[i].Kind, urls[i]);
        }
    }

    /// <summary>
    ///     A dictionary of the port's connect arguments, in the canonical forms the Python side prints: libpq's
    ///     <c>key=value</c> list sorted by key, or <c>json.dumps(..., sort_keys=True)</c>.
    /// </summary>
    private sealed class OrderedDictLike
    {
        /// <summary>The arguments by key.</summary>
        private readonly IDictionary _values;

        /// <summary>Wraps an argument dictionary.</summary>
        /// <param name="values">The arguments.</param>
        internal OrderedDictLike(IDictionary values) => _values = values;

        /// <summary>The <c>key=value</c> list sorted by key, joined with <c>;</c>.</summary>
        /// <returns>The canonical text.</returns>
        internal string Canonical()
            => string.Join(";", _values.Keys.Cast<string>().OrderBy(k => k, StringComparer.Ordinal).Select(k => $"{k}={_values[k]}"));

        /// <summary><c>json.dumps(..., sort_keys=True)</c> of the arguments, nested dictionaries included.</summary>
        /// <returns>The JSON text.</returns>
        internal string Json() => PyJson.Dumps(Sorted(_values));

        /// <summary>A key-sorted copy with Python-JSON-ready values (big integers as <see cref="long" />).</summary>
        /// <param name="values">A dictionary.</param>
        /// <returns>The sorted copy.</returns>
        private static SortedDictionary<string, object?> Sorted(IDictionary values)
        {
            var sorted = new SortedDictionary<string, object?>(StringComparer.Ordinal);
            foreach (DictionaryEntry entry in values)
                sorted[(string)entry.Key] = entry.Value switch
                {
                    BigInteger i => (long)i,
                    IDictionary nested => Sorted(nested),
                    var v => v,
                };
            return sorted;
        }
    }
}

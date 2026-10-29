using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Numerics;
using MySqlConnector;
using NumSharp;
using OptunaSharp.Collections;
using OptunaSharp.Internal;
using OptunaSharp.Storages.SqlAlchemy;

namespace OptunaSharp.Storages.EFCore;

/// <summary>
///     Turns a MySQL or MariaDB storage URL into the MySqlConnector connection string for the database Python Optuna
///     would reach: SQLAlchemy's <c>create_connect_args</c> for the URL's driver, then that driver's handling of the
///     keyword arguments (PyMySQL's <c>Connection.__init__</c>, MySQLdb's <c>connect</c>), expressed with MySqlConnector.
/// </summary>
/// <remarks>
///     <para><b>The two drivers Optuna documents</b> are reproduced argument by argument: <c>mysql://</c> and
///     <c>mariadb://</c> (MySQLdb, the <c>mysqlclient</c> C driver) and <c>+pymysql</c> (also used by the
///     <c>+aiomysql</c>/<c>+asyncmy</c> URLs before their asyncio failure). Unknown arguments raise the driver's own
///     <c>TypeError</c>; arguments a URL cannot type correctly fail the same way (<c>?port=</c> is a string to the
///     driver: PyMySQL raises <c>port should be of type int</c>); SSL follows each driver's rules — PyMySQL negotiates
///     TLS when the server offers it, verifies the certificate and host name once a CA is given, and treats any
///     non-empty <c>ssl_verify_cert</c>/<c>ssl_disabled</c> text as true, exactly as here.</para>
///     <para><b>MySQLdb's client-library defaults</b> apply to <c>mysql://</c>: host <c>localhost</c> (or none) means
///     the Unix-domain socket on Linux and macOS (<c>MYSQL_UNIX_PORT</c>, else <c>/var/run/mysqld/mysqld.sock</c>
///     when present, else <c>/tmp/mysql.sock</c>), <c>MYSQL_HOST</c> and <c>MYSQL_TCP_PORT</c> fill a missing host and
///     port, and a missing user is the login name (<c>ODBC</c> on Windows), as the MySQL C API does.</para>
///     <para><b>What MySqlConnector cannot express</b> raises <see cref="NotSupportedException" /> naming the argument
///     (<c>init_command</c>, <c>sql_mode</c>, <c>read_default_file</c>, <c>ssl_capath</c>, <c>ssl_cipher</c>, a character
///     set other than UTF-8 — MySqlConnector always talks <c>utf8mb4</c> —, …). The other MySQL drivers
///     (<c>+mysqlconnector</c>, <c>+mariadbconnector</c>, <c>+cymysql</c>) get their URL components and the common
///     arguments (<c>charset</c>, the connect timeout, <c>unix_socket</c>, the SSL files); anything else is refused
///     the same way.</para>
///     <para>Documented divergence: PyMySQL encodes a password as Latin-1, MySqlConnector as UTF-8, so a password
///     with characters outside ASCII reaches the server as different bytes (and one outside Latin-1 makes PyMySQL
///     raise). Python's connection pool and MySqlConnector's are configured separately and are not mapped.</para>
/// </remarks>
internal static class MySqlUrl
{
    /// <summary>
    ///     The MySqlConnector connection string for a MySQL or MariaDB URL.
    /// </summary>
    /// <param name="prepared">The URL after template filling, parsing and driver lookup.</param>
    /// <returns>The connection string builder for the database the URL names.</returns>
    /// <exception cref="ValueError">A URL port that is not an integer, a coercion failure, or a driver's own value check.</exception>
    /// <exception cref="TypeError">An argument the driver does not accept, or one a URL can only give as a string.</exception>
    /// <exception cref="NotImplementedException">PyMySQL's refusal of <c>compress</c> and <c>named_pipe</c>.</exception>
    /// <exception cref="NotSupportedException">An argument MySqlConnector cannot express.</exception>
    /// <exception cref="InvalidOperationException">An asyncio driver (<c>+aiomysql</c>, <c>+asyncmy</c>).</exception>
    internal static MySqlConnectionStringBuilder Build(PreparedStorageUrl prepared)
    {
        string driver = prepared.Driver.Driver;
        OrderedDict<string, object?> opts = ConnectArgs(prepared);
        if (prepared.Driver.IsAsync)
            throw new InvalidOperationException(SqlAlchemyRegistry.MissingGreenletMessage);
        return driver switch
        {
            "pymysql" => FromPyMySql(opts),
            "mysqlconnector" or "mariadbconnector" or "cymysql" => FromSecondary(opts, driver),
            _ => FromMySqlDb(opts),
        };
    }

    /// <summary>
    ///     SQLAlchemy's <c>create_connect_args</c> for the URL's MySQL driver: the keyword arguments Python passes to
    ///     the driver's <c>connect</c>, without the <c>FOUND_ROWS</c> client flag SQLAlchemy adds once the driver module
    ///     is imported (Pomelo's <c>UseAffectedRows=false</c> plays that part here).
    /// </summary>
    /// <param name="prepared">A URL of the <c>mysql</c> or <c>mariadb</c> dialect.</param>
    /// <returns>The arguments, in Python's order.</returns>
    /// <exception cref="ValueError">A coercion failure.</exception>
    /// <exception cref="TypeError">A repeated query key an <c>int</c> coercion cannot take.</exception>
    internal static OrderedDict<string, object?> ConnectArgs(PreparedStorageUrl prepared)
    {
        SqlAlchemyUrl url = prepared.Url;
        return prepared.Driver.Driver switch
        {
            "pymysql" => MySqlDbConnectArgs(url, new Dictionary<string, string> { ["username"] = "user" }),
            "aiomysql" or "asyncmy" => MySqlDbConnectArgs(url, new Dictionary<string, string> { ["username"] = "user", ["database"] = "db" }),
            "mysqlconnector" => MySqlConnectorConnectArgs(url),
            "mariadbconnector" => MariaDbConnectorConnectArgs(url),
            _ => MySqlDbConnectArgs(url, MySqlDbNames),
        };
    }

    /// <summary>MySQLdb's renames in <c>MySQLDialect_mysqldb.create_connect_args</c>.</summary>
    private static readonly Dictionary<string, string> MySqlDbNames = new()
    {
        ["database"] = "db", ["username"] = "user", ["password"] = "passwd",
    };

    // ------------------------------------------------------------------ SQLAlchemy create_connect_args

    /// <summary>
    ///     <c>MySQLDialect_mysqldb.create_connect_args</c> (shared by PyMySQL, cymysql, aiomysql and asyncmy with their
    ///     own renames): URL components, the query on top, type coercion, and the <c>ssl_*</c> keys gathered into an
    ///     <c>ssl</c> dictionary.
    /// </summary>
    /// <param name="url">The URL without plugin keys.</param>
    /// <param name="names">The renames of the URL components.</param>
    /// <returns>The keyword arguments for the driver's <c>connect</c>.</returns>
    /// <exception cref="ValueError">A coercion failure.</exception>
    /// <exception cref="TypeError">A repeated query key an <c>int</c> coercion cannot take.</exception>
    private static OrderedDict<string, object?> MySqlDbConnectArgs(SqlAlchemyUrl url, Dictionary<string, string> names)
    {
        OrderedDict<string, object?> opts = Translate(url, names);
        foreach (KeyValuePair<string, object> pair in url.Query)
            opts[pair.Key] = pair.Value;
        PyValues.CoerceKwType(opts, "compress", "bool");
        PyValues.CoerceKwType(opts, "connect_timeout", "int");
        PyValues.CoerceKwType(opts, "read_timeout", "int");
        PyValues.CoerceKwType(opts, "write_timeout", "int");
        PyValues.CoerceKwType(opts, "client_flag", "int");
        PyValues.CoerceKwType(opts, "local_infile", "bool");
        PyValues.CoerceKwType(opts, "use_unicode", "bool");
        PyValues.CoerceKwType(opts, "charset", "str");

        var ssl = new OrderedDict<string, object?>();
        foreach ((string key, string type) in new[]
                 {
                     ("ssl_ca", "str"), ("ssl_key", "str"), ("ssl_cert", "str"), ("ssl_capath", "str"), ("ssl_cipher", "str"),
                     ("ssl_check_hostname", "bool"),
                 })
        {
            if (!opts.TryGetValue(key, out object? value))
                continue;
            string inner = key.Substring(4);
            ssl[inner] = value;
            PyValues.CoerceKwType(ssl, inner, type);
            opts.Remove(key);
        }
        if (ssl.Count > 0)
            opts["ssl"] = ssl;
        return opts;
    }

    /// <summary><c>MySQLDialect_mysqlconnector.create_connect_args</c>.</summary>
    /// <param name="url">The URL without plugin keys.</param>
    /// <returns>The keyword arguments for <c>mysql.connector.connect</c>.</returns>
    /// <exception cref="ValueError">A coercion failure.</exception>
    /// <exception cref="TypeError">A repeated query key an <c>int</c> coercion cannot take.</exception>
    private static OrderedDict<string, object?> MySqlConnectorConnectArgs(SqlAlchemyUrl url)
    {
        OrderedDict<string, object?> opts = Translate(url, new Dictionary<string, string> { ["username"] = "user" });
        foreach (KeyValuePair<string, object> pair in url.Query)
            opts[pair.Key] = pair.Value;
        foreach (string key in new[] { "allow_local_infile", "autocommit", "buffered", "compress", "consume_results", "force_ipv6",
                     "get_warnings", "pool_reset_session", "raise_on_warnings", "raw", "ssl_verify_cert", "use_pure", "use_unicode" })
            PyValues.CoerceKwType(opts, key, "bool");
        foreach (string key in new[] { "client_flag", "connection_timeout", "connect_timeout", "pool_size" })
            PyValues.CoerceKwType(opts, key, "int");
        opts["buffered"] = true;
        return opts;
    }

    /// <summary><c>MySQLDialect_mariadbconnector.create_connect_args</c>.</summary>
    /// <param name="url">The URL without plugin keys.</param>
    /// <returns>The keyword arguments for <c>mariadb.connect</c>.</returns>
    /// <exception cref="ValueError">A coercion failure.</exception>
    /// <exception cref="TypeError">A repeated query key an <c>int</c> coercion cannot take.</exception>
    private static OrderedDict<string, object?> MariaDbConnectorConnectArgs(SqlAlchemyUrl url)
    {
        OrderedDict<string, object?> opts = Translate(url, new Dictionary<string, string>());
        foreach (KeyValuePair<string, object> pair in url.Query)
            opts[pair.Key] = pair.Value;
        foreach (string key in new[] { "connect_timeout", "read_timeout", "write_timeout", "client_flag", "port", "pool_size" })
            PyValues.CoerceKwType(opts, key, "int");
        foreach (string key in new[] { "local_infile", "ssl_verify_cert", "ssl", "pool_reset_connection", "compress" })
            PyValues.CoerceKwType(opts, key, "bool");
        return opts;
    }

    /// <summary><see cref="SqlAlchemyUrl.TranslateConnectArgs" /> as an object-valued dictionary.</summary>
    /// <param name="url">The URL.</param>
    /// <param name="names">SQLAlchemy's renames.</param>
    /// <returns>The translated arguments.</returns>
    private static OrderedDict<string, object?> Translate(SqlAlchemyUrl url, Dictionary<string, string> names)
    {
        var opts = new OrderedDict<string, object?>();
        foreach (KeyValuePair<string, object> pair in url.TranslateConnectArgs(names))
            opts[pair.Key] = pair.Value;
        return opts;
    }

    // ------------------------------------------------------------------ PyMySQL

    /// <summary>The keyword arguments of PyMySQL 2.2's <c>Connection.__init__</c>.</summary>
    private static readonly HashSet<string> PyMySqlArguments = new(StringComparer.Ordinal)
    {
        "user", "password", "host", "database", "unix_socket", "port", "charset", "collation", "sql_mode",
        "read_default_file", "conv", "use_unicode", "client_flag", "cursorclass", "init_command", "connect_timeout",
        "read_default_group", "autocommit", "local_infile", "max_allowed_packet", "defer_connect", "auth_plugin_map",
        "read_timeout", "write_timeout", "bind_address", "binary_prefix", "program_name", "server_public_key", "ssl",
        "ssl_ca", "ssl_cert", "ssl_disabled", "ssl_key", "ssl_key_password", "ssl_verify_cert", "ssl_verify_identity",
        "compress", "named_pipe", "passwd", "db",
    };

    /// <summary>
    ///     PyMySQL's <c>Connection.__init__(**opts)</c>, up to the point where it would open the socket, expressed with
    ///     MySqlConnector.
    /// </summary>
    /// <param name="opts">SQLAlchemy's arguments.</param>
    /// <returns>The connection string builder.</returns>
    /// <exception cref="TypeError">An argument PyMySQL does not accept, or a tuple where <c>int()</c> needs a number.</exception>
    /// <exception cref="NotImplementedException"><c>compress</c> or <c>named_pipe</c>.</exception>
    /// <exception cref="ValueError">
    ///     A string port, a timeout out of range, or <c>int()</c> of a non-numeric port behind an option file;
    ///     PyMySQL's messages. A <see cref="UnicodeEncodeError" /> for a password outside Latin-1.
    /// </exception>
    /// <exception cref="AttributeError">
    ///     PyMySQL's own failures on values a URL can only give as text: <c>?ssl=</c> (<c>'str' object has no
    ///     attribute 'get'</c>), an unknown <c>charset</c> (<c>'NoneType' object has no attribute 'encoding'</c>) and
    ///     <c>?conv=</c> (<c>'str' object has no attribute 'items'</c>).
    /// </exception>
    /// <exception cref="FileNotFoundException">The <c>ssl_ca</c> or <c>ssl_cert</c> file is missing (<c>[Errno 2] No such file or directory</c>).</exception>
    /// <exception cref="NotSupportedException">An argument MySqlConnector cannot express, refused once PyMySQL's own checks have passed.</exception>
    /// <remarks>
    ///     The checks run in <c>Connection.__init__</c>'s order. A refusal of what MySqlConnector cannot express waits
    ///     until they have all passed, so a URL Python rejects fails here with Python's error; only an option file
    ///     PyMySQL would actually read is refused at once, since its contents could change everything after it.
    /// </remarks>
    private static MySqlConnectionStringBuilder FromPyMySql(OrderedDict<string, object?> opts)
    {
        foreach (string key in opts.Keys)
        {
            if (!PyMySqlArguments.Contains(key))
                throw new TypeError($"Connection.__init__() got an unexpected keyword argument '{key}'");
        }
        object? Arg(string key) => opts.TryGetValue(key, out object? v) ? v : null;
        bool Truthy(string key) => PyValues.Truthy(Arg(key));
        NotSupportedException? refusal = null;
        void Refuse(NotSupportedException e) => refusal ??= e;

        if (Truthy("compress") || Truthy("named_pipe"))
            throw new NotImplementedException("compress and named_pipe arguments are not supported");
        foreach (string key in new[] { "cursorclass", "init_command", "sql_mode", "autocommit", "defer_connect",
                     "auth_plugin_map", "bind_address", "server_public_key", "collation" })
        {
            if (Truthy(key))
                Refuse(Unsupported("PyMySQL", key, Arg(key)));
        }

        var b = Defaults();
        object? database = Arg("database") ?? Arg("db");
        object? password = Truthy("password") ? Arg("password") : Arg("passwd") ?? Arg("password");
        object? port = Arg("port");

        // Option files: read_default_group alone means the platform's default file; a file configparser cannot open
        // is skipped, which leaves every argument as it was except that the port goes through int().
        object? optionFile = Arg("read_default_file");
        if (Truthy("read_default_group") && !PyValues.Truthy(optionFile))
            optionFile = OperatingSystem.IsWindows() ? "c:\\my.ini" : "/etc/my.cnf";
        if (PyValues.Truthy(optionFile))
        {
            string path = PyOsPath.ExpandUser(PyValues.Str(optionFile));
            if (CanOpen(path))
                throw Unsupported("PyMySQL", "read_default_file", path);
            port = PyValues.Int(port ?? BigInteger.Zero);
        }

        // ssl: SQLAlchemy's gathered dictionary, replaced by PyMySQL when any ssl_* flag is set; PREFERRED otherwise.
        // ssl_key_password never takes effect from a URL: PyMySQL puts it only into its replacement dictionary, which
        // then has no certificate (SQLAlchemy already moved ssl_cert into its own dictionary, which has no password),
        // so load_cert_chain never receives it. Accepted and ignored, as in Python.
        object? sslDict = Arg("ssl");
        if (!Truthy("ssl_disabled"))
        {
            if (Truthy("ssl_ca") || Truthy("ssl_cert") || Truthy("ssl_key") || Truthy("ssl_verify_cert") || Truthy("ssl_verify_identity"))
            {
                var replaced = new OrderedDict<string, object?>
                {
                    ["ca"] = Arg("ssl_ca"),
                    ["check_hostname"] = Truthy("ssl_verify_identity"),
                    ["verify_mode"] = Arg("ssl_verify_cert") ?? false,
                };
                if (Arg("ssl_cert") is not null)
                    replaced["cert"] = Arg("ssl_cert");
                if (Arg("ssl_key") is not null)
                    replaced["key"] = Arg("ssl_key");
                sslDict = replaced;
            }
            if (PyValues.Truthy(sslDict))
            {
                // _create_ssl_ctx(sslp) calls sslp.get("ca"): a ?ssl= value is text (or a tuple), not a dictionary.
                if (sslDict is not OrderedDict<string, object?> dictionary)
                    throw new AttributeError($"'{PyValues.TypeName(sslDict)}' object has no attribute 'get'");
                ApplyPyMySqlSsl(b, dictionary, Refuse);
            }
            else
            {
                b.SslMode = MySqlSslMode.Preferred;
            }
        }
        else
        {
            b.SslMode = MySqlSslMode.None;
        }

        object? host = Arg("host");
        b.Server = PyValues.Truthy(host) ? PyValues.Str(host) : "localhost";
        if (PyValues.Truthy(port))
        {
            if (port is not BigInteger number)
                throw new ValueError("port should be of type int");
            b.Port = CheckedPort(number);
        }
        b.UserID = PyValues.Truthy(Arg("user")) ? PyValues.Str(Arg("user")) : PyMySqlDefaultUser();
        if (PyValues.Truthy(password))
        {
            // PyMySQL encodes a text password as Latin-1, so one outside it fails here, before any connection.
            string text = PyValues.Str(password);
            if (password is string)
                PyLatin1.CheckEncodable(text);
            b.Password = text;
        }
        if (PyValues.Truthy(database))
            b.Database = PyValues.Str(database);
        if (Arg("unix_socket") is { } socket && PyValues.Truthy(socket))
        {
            b.Server = PyValues.Str(socket);
            b.ConnectionProtocol = MySqlConnectionProtocol.UnixSocket;
        }

        // PyMySQL's connect timeout defaults to 10 s and must lie in (0, 31536000].
        BigInteger connectTimeout = Arg("connect_timeout") is BigInteger t ? t : 10;
        if (connectTimeout <= 0 || connectTimeout > 31536000)
            throw new ValueError("connect_timeout should be >0 and <=31536000");
        b.ConnectionTimeout = (uint)connectTimeout;
        if (Arg("read_timeout") is BigInteger read)
        {
            if (read <= 0)
                throw new ValueError("read_timeout should be > 0");
            // A socket read deadline per query response; MySqlConnector bounds a command by its command timeout.
            b.DefaultCommandTimeout = (uint)BigInteger.Min(read, uint.MaxValue);
        }
        if (Arg("write_timeout") is BigInteger write && write <= 0)
            throw new ValueError("write_timeout should be > 0");

        // self.charset = charset or DEFAULT_CHARSET; charset_by_name(self.charset).encoding
        ApplyPyMySqlCharset(b, Arg("charset"), Refuse);
        if (Arg("local_infile") is { } infile)
            b.AllowLoadLocalInfile = PyValues.Truthy(infile);
        CheckClientFlag("PyMySQL", Arg("client_flag"), b, Refuse);
        // conv.items(): a URL can only give text (or a tuple), which has no items().
        if (Arg("conv") is { } conv)
            throw new AttributeError($"'{PyValues.TypeName(conv)}' object has no attribute 'items'");
        if (Arg("program_name") is { } program && PyValues.Truthy(program))
            b.ApplicationName = PyValues.Str(program);
        // use_unicode, binary_prefix and max_allowed_packet only change how Python encodes values; .NET strings and
        // MySqlConnector's parameters already behave as their defaults do.
        if (refusal is not null)
            throw refusal;
        return b;
    }

    /// <summary>
    ///     PyMySQL's <c>_create_ssl_ctx</c> as a MySqlConnector SSL mode: TLS is required; the certificate is verified
    ///     when a CA is given or <c>verify_mode</c> asks for it, and the host name only when a CA is given.
    /// </summary>
    /// <param name="b">The builder to configure.</param>
    /// <param name="ssl">PyMySQL's SSL dictionary.</param>
    /// <param name="refuse">Records a refusal to raise once PyMySQL's own checks have passed.</param>
    /// <exception cref="FileNotFoundException">
    ///     The CA file (<c>create_default_context(cafile=…)</c>), the certificate or its key
    ///     (<c>load_cert_chain</c>) is missing: <c>[Errno 2] No such file or directory</c>. Python loads them here,
    ///     before connecting; a file that exists but is not valid PEM fails in MySqlConnector at connect time instead
    ///     of in Python's <c>ssl</c> module.
    /// </exception>
    private static void ApplyPyMySqlSsl(MySqlConnectionStringBuilder b, OrderedDict<string, object?> ssl, Action<NotSupportedException> refuse)
    {
        object? Get(string key) => ssl.TryGetValue(key, out object? v) ? v : null;
        // hasnoca = ca is None and capath is None
        bool hasCa = Get("ca") is not null || Get("capath") is not null;
        if (Get("ca") is { } caFile)
            RequireFile(PyValues.Str(caFile));
        if (Get("capath") is not null)
            refuse(Unsupported("PyMySQL", "ssl_capath", Get("capath")));
        bool checkHostname = hasCa && (ssl.ContainsKey("check_hostname") ? PyValues.Truthy(Get("check_hostname")) : true);
        object? verify = Get("verify_mode");
        bool verifyCert = verify switch
        {
            null => hasCa,
            bool v => v,
            // Only text is lower-cased (Python's lower()); a tuple compares as it is and matches nothing.
            string text => PyStrings.Lower(text) switch
            {
                "none" or "0" or "false" or "no" => false,
                "optional" or "required" or "1" or "true" or "yes" => true,
                _ => hasCa,
            },
            _ => hasCa,
        };
        if (Get("ca") is { } ca)
            b.SslCa = PyValues.Str(ca);
        // ctx.load_cert_chain(cert, keyfile=key) runs only when a certificate is given; a key alone is ignored.
        if (ssl.ContainsKey("cert"))
        {
            string cert = PyValues.Str(Get("cert"));
            RequireFile(cert);
            b.SslCert = cert;
            if (Get("key") is { } key)
            {
                RequireFile(PyValues.Str(key));
                b.SslKey = PyValues.Str(key);
            }
        }
        if (Get("cipher") is not null)
            refuse(Unsupported("PyMySQL", "ssl_cipher", Get("cipher")));
        // ssl.create_default_context() trusts the system store when no CA file is given, as MySqlConnector does.
        b.SslMode = !verifyCert ? MySqlSslMode.Required : checkHostname ? MySqlSslMode.VerifyFull : MySqlSslMode.VerifyCA;
    }

    /// <summary>The <c>FileNotFoundError</c> Python's <c>ssl</c> module raises for a certificate file that is not there.</summary>
    /// <param name="path">The path, relative to the working directory as in Python.</param>
    /// <exception cref="FileNotFoundException">The file does not exist: <c>[Errno 2] No such file or directory</c>.</exception>
    private static void RequireFile(string path)
    {
        if (!File.Exists(path))
            throw new FileNotFoundException("[Errno 2] No such file or directory", path);
    }

    /// <summary>
    ///     Whether <c>configparser</c> would read an option file: <c>RawConfigParser.read</c> opens it and silently
    ///     skips it on any <c>OSError</c> (missing, a directory, no permission).
    /// </summary>
    /// <param name="path">The expanded path.</param>
    /// <returns><see langword="true" /> when the file opens for reading.</returns>
    private static bool CanOpen(string path)
    {
        try
        {
            using FileStream stream = File.OpenRead(path);
            return true;
        }
        catch (Exception e) when (e is IOException or UnauthorizedAccessException or ArgumentException or NotSupportedException)
        {
            return false;
        }
    }

    /// <summary>
    ///     PyMySQL's <c>user or DEFAULT_USER</c>, where <c>DEFAULT_USER</c> is <c>getpass.getuser()</c>: the first of
    ///     <c>LOGNAME</c>, <c>USER</c>, <c>LNAME</c> and <c>USERNAME</c> that is set and not empty, else the account name.
    /// </summary>
    /// <returns>The user name.</returns>
    /// <remarks>
    ///     CPython 3.12 on Windows has no account database to fall back on, so with none of the variables set PyMySQL's
    ///     user is <c>None</c> and its handshake fails; the account name is used here instead.
    /// </remarks>
    private static string PyMySqlDefaultUser()
    {
        foreach (string name in new[] { "LOGNAME", "USER", "LNAME", "USERNAME" })
        {
            if (Environment.GetEnvironmentVariable(name) is { Length: > 0 } user)
                return user;
        }
        return Environment.UserName;
    }

    /// <summary>The character sets PyMySQL 2.2 knows (<c>pymysql.charset._charsets</c>), by lower-cased name.</summary>
    private static readonly HashSet<string> PyMySqlCharsets = new(StringComparer.Ordinal)
    {
        "armscii8", "ascii", "big5", "binary", "cp1250", "cp1251", "cp1256", "cp1257", "cp850", "cp852", "cp866", "cp932",
        "dec8", "eucjpms", "euckr", "gb18030", "gb2312", "gbk", "geostd8", "greek", "hebrew", "hp8", "keybcs2", "koi8r",
        "koi8u", "latin1", "latin2", "latin5", "latin7", "macce", "macroman", "sjis", "swe7", "tis620", "ujis", "utf8mb3",
        "utf8mb4",
    };

    /// <summary>
    ///     PyMySQL's character set: <c>charset or "utf8mb4"</c>, looked up with <c>name.lower()</c> and <c>utf8</c>
    ///     read as <c>utf8mb4</c>. An unknown name fails as PyMySQL fails; a known one other than UTF-8 is refused,
    ///     because MySqlConnector always talks <c>utf8mb4</c>.
    /// </summary>
    /// <param name="b">The builder.</param>
    /// <param name="charset">The argument, or <see langword="null" />.</param>
    /// <param name="refuse">Records a refusal to raise once PyMySQL's own checks have passed.</param>
    /// <exception cref="AttributeError">A name PyMySQL does not know: <c>'NoneType' object has no attribute 'encoding'</c>.</exception>
    private static void ApplyPyMySqlCharset(MySqlConnectionStringBuilder b, object? charset, Action<NotSupportedException> refuse)
    {
        string name = PyValues.Truthy(charset) ? PyValues.Str(charset) : "utf8mb4";
        // Charsets.by_name: Python's lower(), so UTF8MB4 and Utf8 work as they do in Python.
        string lowered = PyStrings.Lower(name);
        if (lowered == "utf8")
            lowered = "utf8mb4";
        if (!PyMySqlCharsets.Contains(lowered))
            throw new AttributeError("'NoneType' object has no attribute 'encoding'");
        if (lowered is not ("utf8mb4" or "utf8mb3"))
            refuse(Unsupported("PyMySQL", "charset", charset));
        else if (PyValues.Truthy(charset))
            b.CharacterSet = "utf8mb4";
    }

    // ------------------------------------------------------------------ MySQLdb (mysqlclient)

    /// <summary>
    ///     <c>_mysql.connection.__init__</c>'s keyword list with each argument's <c>PyArg_ParseTupleAndKeywords</c>
    ///     format (<c>"|ssssisOiiisssiOsiiisss:connect"</c>): <c>s</c> text, <c>i</c> a C <c>int</c>, <c>O</c> any object.
    ///     The position (from 1) is what CPython's messages quote.
    /// </summary>
    private static readonly (string Name, char Format)[] MySqlDbKeywords =
    {
        ("host", 's'), ("user", 's'), ("password", 's'), ("database", 's'), ("port", 'i'), ("unix_socket", 's'), ("conv", 'O'),
        ("connect_timeout", 'i'), ("compress", 'i'), ("named_pipe", 'i'), ("init_command", 's'), ("read_default_file", 's'),
        ("read_default_group", 's'), ("client_flag", 'i'), ("ssl", 'O'), ("ssl_mode", 's'), ("local_infile", 'i'),
        ("read_timeout", 'i'), ("write_timeout", 'i'), ("charset", 's'), ("auth_plugin", 's'), ("server_public_key_path", 's'),
    };

    /// <summary>The arguments MySQLdb's Python <c>Connection.__init__</c> takes out before calling <c>_mysql.connect</c>.</summary>
    private static readonly string[] MySqlDbPythonArguments =
    {
        "cursorclass", "collation", "use_unicode", "sql_mode", "binary_prefix", "multi_statements", "autocommit",
    };

    /// <summary>
    ///     MySQLdb's <c>connect(**opts)</c> with the MySQL C API's defaults, expressed with MySqlConnector.
    /// </summary>
    /// <param name="opts">SQLAlchemy's arguments.</param>
    /// <returns>The connection string builder.</returns>
    /// <exception cref="AttributeError"><c>?conv=</c>: <c>Connection.__init__</c> calls <c>conv.items()</c> on the text.</exception>
    /// <exception cref="TypeError">
    ///     CPython's argument parsing: a tuple (a repeated query key) where text is expected
    ///     (<c>connect() argument 16 must be str, not tuple</c>), text where an integer is expected, or an unknown
    ///     argument (<c>'…' is an invalid keyword argument for connect()</c>).
    /// </exception>
    /// <exception cref="ValueError">A NUL in a text argument (<c>embedded null character</c>), or a port outside 0–65535.</exception>
    /// <exception cref="OverflowException">
    ///     An integer argument outside the C <c>int</c> range. The message is the platform's, as in CPython: C <c>long</c>
    ///     is 32-bit on Windows (<c>Python int too large to convert to C long</c>) and 64-bit elsewhere
    ///     (<c>signed integer is greater than maximum</c> until the value leaves the 64-bit range).
    /// </exception>
    /// <exception cref="FormatException">
    ///     mysqlclient's <c>NotSupportedError: Unknown ssl_mode specification</c>: the mode must be one of
    ///     <c>DISABLED PREFERRED REQUIRED VERIFY_CA VERIFY_IDENTITY</c>, compared case-sensitively (<c>strcmp</c>).
    /// </exception>
    /// <exception cref="NotSupportedException">An argument MySqlConnector cannot express, refused once MySQLdb's own checks have passed.</exception>
    private static MySqlConnectionStringBuilder FromMySqlDb(OrderedDict<string, object?> opts)
    {
        object? Arg(string key) => opts.TryGetValue(key, out object? v) ? v : null;
        NotSupportedException? refusal = null;
        void Refuse(NotSupportedException e) => refusal ??= e;

        // Connection.__init__ (MySQLdb/connections.py), in its order. conv.items() comes first, and a URL's conv is
        // text (or a tuple), which has no items().
        if (opts.TryGetValue("conv", out object? conv))
            throw new AttributeError($"'{PyValues.TypeName(conv)}' object has no attribute 'items'");
        // db and passwd are renamed (and so win over database= and password=); a renamed key keeps the position of an
        // existing target, as kwargs2["database"] = kwargs2.pop("db") does in a dict.
        var kwargs = new OrderedDict<string, object?>();
        foreach (KeyValuePair<string, object?> pair in opts)
            kwargs[pair.Key] = pair.Value;
        foreach ((string from, string to) in new[] { ("db", "database"), ("passwd", "password") })
        {
            if (kwargs.TryGetValue(from, out object? renamed))
            {
                kwargs.Remove(from);
                kwargs[to] = renamed;
            }
        }
        // kwargs2["conv"] = conv2: the default conversion mapping is always passed on. Only the key's presence and
        // position matter below (an O argument, which converts without error), so a marker stands in for the mapping.
        kwargs["conv"] = "<conversions>";
        foreach (string key in MySqlDbPythonArguments)
            kwargs.Remove(key);
        // client_flag |= MULTI_RESULTS, and MULTI_STATEMENTS unless multi_statements is falsy (a URL's "0" is truthy).
        BigInteger clientFlag = Arg("client_flag") switch
        {
            BigInteger value => value,
            bool value => value ? BigInteger.One : BigInteger.Zero,
            _ => BigInteger.Zero,
        };
        clientFlag |= 131072;
        if (!opts.ContainsKey("multi_statements") || PyValues.Truthy(Arg("multi_statements")))
            clientFlag |= 65536;
        kwargs["client_flag"] = clientFlag;

        // _mysql.connect's PyArg_ParseTupleAndKeywords: every given argument is converted in keyword-list order, and
        // only then are unknown keywords reported.
        for (int i = 0; i < MySqlDbKeywords.Length; i++)
        {
            (string name, char format) = MySqlDbKeywords[i];
            if (!kwargs.TryGetValue(name, out object? value))
                continue;
            if (format == 's')
            {
                if (value is not string text)
                    throw new TypeError($"connect() argument {i + 1} must be str, not {PyValues.TypeName(value)}");
                if (text.Contains('\0'))
                    throw new ValueError("embedded null character");
            }
            else if (format == 'i')
            {
                CheckCInt(value);
            }
        }
        foreach (string key in kwargs.Keys)
        {
            if (Array.FindIndex(MySqlDbKeywords, k => k.Name == key) < 0)
                throw new TypeError($"'{key}' is an invalid keyword argument for connect()");
        }
        MySqlSslMode? sslMode = null;
        if (Arg("ssl_mode") is string modeText)
        {
            // _get_ssl_mode_num: strcmp against the five names, so "required" is refused, as in mysqlclient.
            sslMode = modeText switch
            {
                "DISABLED" => MySqlSslMode.None,
                "PREFERRED" => MySqlSslMode.Preferred,
                "REQUIRED" => MySqlSslMode.Required,
                "VERIFY_CA" => MySqlSslMode.VerifyCA,
                "VERIFY_IDENTITY" => MySqlSslMode.VerifyFull,
                _ => throw new FormatException("Unknown ssl_mode specification"),
            };
        }

        foreach (string key in new[] { "named_pipe", "init_command", "read_default_file", "read_default_group",
                     "auth_plugin", "cursorclass", "collation", "sql_mode", "autocommit" })
        {
            if (PyValues.Truthy(Arg(key)))
                Refuse(Unsupported("MySQLdb", key, Arg(key)));
        }

        var b = Defaults();
        // Connection.__init__: db and passwd are renamed after the query is applied, so they win over ?database=/?password=.
        object? database = Arg("db") ?? Arg("database");
        object? password = Arg("passwd") ?? Arg("password");

        // The MySQL C API: MYSQL_HOST and MYSQL_TCP_PORT fill in a missing host and port, and "localhost" (or no
        // host) is the local Unix-domain socket outside Windows.
        string? host = PyValues.Truthy(Arg("host")) ? PyValues.Str(Arg("host")) : Environment.GetEnvironmentVariable("MYSQL_HOST");
        object? socket = Arg("unix_socket");
        if (PyValues.Truthy(socket))
        {
            b.Server = PyValues.Str(socket);
            b.ConnectionProtocol = MySqlConnectionProtocol.UnixSocket;
        }
        else if (!OperatingSystem.IsWindows() && (string.IsNullOrEmpty(host) || host == "localhost"))
        {
            b.Server = Environment.GetEnvironmentVariable("MYSQL_UNIX_PORT") is { Length: > 0 } envSocket ? envSocket
                : File.Exists("/var/run/mysqld/mysqld.sock") ? "/var/run/mysqld/mysqld.sock" : "/tmp/mysql.sock";
            b.ConnectionProtocol = MySqlConnectionProtocol.UnixSocket;
        }
        else
        {
            b.Server = string.IsNullOrEmpty(host) ? "localhost" : host;
        }
        if (Arg("port") is BigInteger port && !port.IsZero)
            b.Port = CheckedPort(port);
        else if (Environment.GetEnvironmentVariable("MYSQL_TCP_PORT") is { Length: > 0 } envPort
                 && uint.TryParse(envPort, NumberStyles.None, CultureInfo.InvariantCulture, out uint parsedPort))
            b.Port = parsedPort;
        b.UserID = PyValues.Truthy(Arg("user")) ? PyValues.Str(Arg("user")) : OperatingSystem.IsWindows() ? "ODBC" : Environment.UserName;
        if (PyValues.Truthy(password))
            b.Password = PyValues.Str(password);
        if (PyValues.Truthy(database))
            b.Database = PyValues.Str(database);

        if (Arg("connect_timeout") is BigInteger connect && connect > 0)
            b.ConnectionTimeout = (uint)BigInteger.Min(connect, uint.MaxValue);
        if (Arg("read_timeout") is BigInteger read && read > 0)
            b.DefaultCommandTimeout = (uint)BigInteger.Min(read, uint.MaxValue);
        if (Arg("compress") is { } compress)
            b.UseCompression = PyValues.Truthy(compress);
        if (Arg("local_infile") is { } infile)
            b.AllowLoadLocalInfile = PyValues.Truthy(infile);
        try
        {
            ApplyCharset(b, "MySQLdb", Arg("charset"));
        }
        catch (NotSupportedException e)
        {
            // set_character_set runs after connecting in Python; refusing it waits like every other refusal.
            Refuse(e);
        }
        CheckClientFlag("MySQLdb", Arg("client_flag"), b, Refuse);
        if (Arg("server_public_key_path") is { } keyPath && PyValues.Truthy(keyPath))
            b.ServerRsaPublicKeyFile = PyValues.Str(keyPath);

        // mysqlclient: an ssl dictionary turns TLS on (REQUIRED since 2.2.4); ssl_mode chooses explicitly.
        OrderedDict<string, object?>? ssl = Arg("ssl") as OrderedDict<string, object?>;
        if (ssl is not null)
        {
            object? Get(string key) => ssl.TryGetValue(key, out object? v) ? v : null;
            if (Get("capath") is not null)
                Refuse(Unsupported("MySQLdb", "ssl_capath", Get("capath")));
            if (Get("cipher") is not null)
                Refuse(Unsupported("MySQLdb", "ssl_cipher", Get("cipher")));
            if (Get("ca") is { } ca)
                b.SslCa = PyValues.Str(ca);
            if (Get("cert") is { } cert)
                b.SslCert = PyValues.Str(cert);
            if (Get("key") is { } key)
                b.SslKey = PyValues.Str(key);
            b.SslMode = Get("ca") is null ? MySqlSslMode.Required
                : ssl.ContainsKey("check_hostname") && PyValues.Truthy(Get("check_hostname")) ? MySqlSslMode.VerifyFull
                : MySqlSslMode.VerifyCA;
        }
        if (sslMode is { } mode)
            b.SslMode = mode;
        if (refusal is not null)
            throw refusal;
        return b;
    }

    /// <summary>
    ///     CPython's <c>i</c> argument format: <c>PyLong_AsLong</c>, then the C <c>int</c> range. C <c>long</c> is
    ///     32-bit on Windows and 64-bit elsewhere, which decides the message for a large value.
    /// </summary>
    /// <param name="value">The argument.</param>
    /// <exception cref="TypeError">Not an integer: <c>'str' object cannot be interpreted as an integer</c>.</exception>
    /// <exception cref="OverflowException">Outside the C <c>long</c> range (<c>Python int too large to convert to C long</c>), or outside the C <c>int</c> range (<c>signed integer is greater than maximum</c> / <c>less than minimum</c>).</exception>
    private static void CheckCInt(object? value)
    {
        if (value is bool)
            return;
        if (value is not BigInteger number)
            throw new TypeError($"'{PyValues.TypeName(value)}' object cannot be interpreted as an integer");
        BigInteger longMin = OperatingSystem.IsWindows() ? int.MinValue : long.MinValue;
        BigInteger longMax = OperatingSystem.IsWindows() ? int.MaxValue : long.MaxValue;
        if (number < longMin || number > longMax)
            throw new OverflowException("Python int too large to convert to C long");
        if (number > int.MaxValue)
            throw new OverflowException("signed integer is greater than maximum");
        if (number < int.MinValue)
            throw new OverflowException("signed integer is less than minimum");
    }

    // ------------------------------------------------------------------ mysql-connector, MariaDB Connector, cymysql

    /// <summary>
    ///     The common arguments of the other MySQL drivers, expressed with MySqlConnector: URL components,
    ///     <c>unix_socket</c>, <c>charset</c>, the connect timeout and the SSL files; anything else is refused.
    /// </summary>
    /// <param name="opts">SQLAlchemy's arguments for the driver.</param>
    /// <param name="driver">The SQLAlchemy driver name, for messages.</param>
    /// <returns>The connection string builder.</returns>
    /// <exception cref="NotSupportedException">An argument outside the common set.</exception>
    private static MySqlConnectionStringBuilder FromSecondary(OrderedDict<string, object?> opts, string driver)
    {
        var b = Defaults();
        b.Server = "localhost";
        foreach (KeyValuePair<string, object?> pair in opts)
        {
            string text = PyValues.Str(pair.Value);
            switch (pair.Key)
            {
                case "host":
                    b.Server = text;
                    break;
                case "port":
                    b.Port = pair.Value is BigInteger n ? CheckedPort(n) : CheckedPort(PyNumbers.ParseInt(text));
                    break;
                case "user":
                    b.UserID = text;
                    break;
                case "password" or "passwd":
                    b.Password = text;
                    break;
                case "database" or "db":
                    b.Database = text;
                    break;
                case "unix_socket":
                    b.Server = text;
                    b.ConnectionProtocol = MySqlConnectionProtocol.UnixSocket;
                    break;
                case "charset":
                    ApplyCharset(b, driver, pair.Value);
                    break;
                case "connect_timeout" or "connection_timeout":
                    b.ConnectionTimeout = (uint)BigInteger.Clamp(PyValues.Int(pair.Value), 0, uint.MaxValue);
                    break;
                case "ssl" when pair.Value is OrderedDict<string, object?> ssl:
                    foreach (KeyValuePair<string, object?> s in ssl)
                    {
                        switch (s.Key)
                        {
                            case "ca":
                                b.SslCa = PyValues.Str(s.Value);
                                break;
                            case "cert":
                                b.SslCert = PyValues.Str(s.Value);
                                break;
                            case "key":
                                b.SslKey = PyValues.Str(s.Value);
                                break;
                            default:
                                throw Unsupported(driver, "ssl_" + s.Key, s.Value);
                        }
                    }
                    b.SslMode = b.SslCa.Length > 0 ? MySqlSslMode.VerifyCA : MySqlSslMode.Required;
                    break;
                case "ssl_ca":
                    b.SslCa = text;
                    b.SslMode = MySqlSslMode.VerifyCA;
                    break;
                case "ssl_cert":
                    b.SslCert = text;
                    break;
                case "ssl_key":
                    b.SslKey = text;
                    break;
                case "buffered" or "use_unicode" or "use_pure":
                    // Python-side result handling with no effect on what the server sees.
                    break;
                default:
                    throw Unsupported(driver, pair.Key, pair.Value);
            }
        }
        return b;
    }

    // ------------------------------------------------------------------ shared

    /// <summary>The builder every MySQL URL starts from: Pomelo's flags, set again by the storage, and no pooling changes.</summary>
    /// <returns>A builder with the server still to be set.</returns>
    private static MySqlConnectionStringBuilder Defaults() => new() { Server = "localhost" };

    /// <summary>A port as MySqlConnector takes it, in 1–65535.</summary>
    /// <param name="port">The port.</param>
    /// <returns>The port.</returns>
    /// <exception cref="ValueError">The port is outside 0–65535; the MySQL client libraries refuse it too.</exception>
    private static uint CheckedPort(BigInteger port)
    {
        if (port < 0 || port > 65535)
            throw new ValueError($"port {port.ToString(CultureInfo.InvariantCulture)} is out of range 0-65535");
        return (uint)port;
    }

    /// <summary>
    ///     A <c>charset</c> argument: UTF-8 spellings are what MySqlConnector always uses (<c>utf8mb4</c>); any other
    ///     character set would change how text is stored, so it is refused.
    /// </summary>
    /// <param name="b">The builder.</param>
    /// <param name="driver">The driver name, for messages.</param>
    /// <param name="charset">The argument, or <see langword="null" />.</param>
    /// <exception cref="NotSupportedException">A character set other than <c>utf8mb4</c>, <c>utf8</c> or <c>utf8mb3</c>.</exception>
    private static void ApplyCharset(MySqlConnectionStringBuilder b, string driver, object? charset)
    {
        if (!PyValues.Truthy(charset))
            return;
        string name = PyValues.Str(charset);
        if (name is not ("utf8mb4" or "utf8" or "utf8mb3"))
            throw Unsupported(driver, "charset", charset);
        b.CharacterSet = "utf8mb4";
    }

    /// <summary>
    ///     The <c>CLIENT_*</c> capability bits a driver sets on its own or that only negotiate the protocol
    ///     MySqlConnector already speaks: <c>LONG_PASSWORD</c>, <c>FOUND_ROWS</c> (SQLAlchemy adds it; Pomelo's
    ///     <c>UseAffectedRows=false</c> provides it), <c>LONG_FLAG</c>, <c>CONNECT_WITH_DB</c>, <c>ODBC</c> (unused by
    ///     servers), <c>PROTOCOL_41</c>, <c>IGNORE_SIGPIPE</c> (client-side only), <c>TRANSACTIONS</c>, the reserved bit,
    ///     <c>SECURE_CONNECTION</c>, <c>MULTI_STATEMENTS</c> and <c>MULTI_RESULTS</c> (MySqlConnector always allows
    ///     both), <c>PS_MULTI_RESULTS</c>, <c>PLUGIN_AUTH</c>, <c>CONNECT_ATTRS</c> and
    ///     <c>PLUGIN_AUTH_LENENC_CLIENT_DATA</c>.
    /// </summary>
    private static readonly BigInteger HarmlessClientFlags =
        1 | 2 | 4 | 8 | 64 | 512 | 4096 | 8192 | 16384 | 32768 | 65536 | 131072 | 262144 | (1 << 19) | (1 << 20) | (1 << 21);

    /// <summary>
    ///     A <c>client_flag</c> argument: bits MySqlConnector already honors are accepted, <c>INTERACTIVE</c>
    ///     (1024, the server's <c>interactive_timeout</c>) becomes <see cref="MySqlConnectionStringBuilder.InteractiveSession" />,
    ///     and for MySQLdb, whose C library acts on them, <c>COMPRESS</c> (32) turns on compression and
    ///     <c>LOCAL_FILES</c> (128) allows <c>LOAD DATA LOCAL</c>. PyMySQL ignores <c>LOCAL_FILES</c> without
    ///     <c>local_infile</c> (it refuses the server's file request), so it is a no-op there.
    /// </summary>
    /// <param name="driver">The driver name: <c>PyMySQL</c> or <c>MySQLdb</c>.</param>
    /// <param name="flag">The argument, or <see langword="null" />.</param>
    /// <param name="b">The builder the mapped bits configure.</param>
    /// <param name="refuse">Records the refusal of any other bit (<c>NO_SCHEMA</c> and <c>IGNORE_SPACE</c> change how
    /// SQL parses, <c>SSL</c> and <c>COMPRESS</c> under PyMySQL break its handshake, the higher bits are unknown).</param>
    private static void CheckClientFlag(string driver, object? flag, MySqlConnectionStringBuilder b, Action<NotSupportedException> refuse)
    {
        BigInteger bits = flag switch
        {
            BigInteger value => value,
            bool value => value ? BigInteger.One : BigInteger.Zero,
            _ => BigInteger.Zero,
        };
        // BigInteger's & and ~ follow Python's infinite two's complement, so a negative flag shows its high bits here.
        BigInteger rest = bits & ~HarmlessClientFlags;
        if (!(rest & 1024).IsZero)
        {
            b.InteractiveSession = true;
            rest &= ~new BigInteger(1024);
        }
        if (!(rest & 128).IsZero)
        {
            if (driver == "MySQLdb")
                b.AllowLoadLocalInfile = true;
            rest &= ~new BigInteger(128);
        }
        if (driver == "MySQLdb" && !(rest & 32).IsZero)
        {
            b.UseCompression = true;
            rest &= ~new BigInteger(32);
        }
        if (!rest.IsZero)
            refuse(Unsupported(driver, "client_flag", flag));
    }

    /// <summary>The refusal for a driver argument MySqlConnector cannot express.</summary>
    /// <param name="driver">The Python driver whose argument it is.</param>
    /// <param name="key">The argument.</param>
    /// <param name="value">Its value.</param>
    /// <returns>The exception to throw.</returns>
    private static NotSupportedException Unsupported(string driver, string key, object? value)
        => new($"The {driver} connection argument {key}={PythonRepr.Repr(value is OrderedDict<string, object?> ? "{…}" : PyValues.Str(value))} " +
               "has no MySqlConnector counterpart, so OptunaSharp cannot connect the way Python would; remove it from the " +
               "storage URL, or configure MySqlConnector directly with EFCoreRdbOptions.ConfigureConnectionString.");
}

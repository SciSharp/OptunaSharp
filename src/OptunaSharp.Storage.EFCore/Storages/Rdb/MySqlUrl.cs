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
    /// <exception cref="TypeError">An argument PyMySQL does not accept.</exception>
    /// <exception cref="NotImplementedException"><c>compress</c> or <c>named_pipe</c>.</exception>
    /// <exception cref="ValueError">A string port, or a timeout out of range; PyMySQL's messages.</exception>
    /// <exception cref="NotSupportedException">An argument MySqlConnector cannot express.</exception>
    private static MySqlConnectionStringBuilder FromPyMySql(OrderedDict<string, object?> opts)
    {
        foreach (string key in opts.Keys)
        {
            if (!PyMySqlArguments.Contains(key))
                throw new TypeError($"Connection.__init__() got an unexpected keyword argument '{key}'");
        }
        object? Arg(string key) => opts.TryGetValue(key, out object? v) ? v : null;
        bool Truthy(string key) => PyValues.Truthy(Arg(key));

        if (Truthy("compress") || Truthy("named_pipe"))
            throw new NotImplementedException("compress and named_pipe arguments are not supported");
        foreach (string key in new[] { "read_default_file", "read_default_group", "conv", "cursorclass", "init_command",
                     "sql_mode", "autocommit", "defer_connect", "auth_plugin_map", "bind_address", "server_public_key",
                     "ssl_key_password", "collation" })
        {
            if (Truthy(key))
                throw Unsupported("PyMySQL", key, Arg(key));
        }

        var b = Defaults();
        object? database = Arg("database") ?? Arg("db");
        object? password = Truthy("password") ? Arg("password") : Arg("passwd") ?? Arg("password");

        // ssl: SQLAlchemy's gathered dictionary, replaced by PyMySQL when any ssl_* flag is set; PREFERRED otherwise.
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
                ApplyPyMySqlSsl(b, (OrderedDict<string, object?>)sslDict!);
            else
                b.SslMode = MySqlSslMode.Preferred;
        }
        else
        {
            b.SslMode = MySqlSslMode.None;
        }

        object? host = Arg("host");
        b.Server = PyValues.Truthy(host) ? PyValues.Str(host) : "localhost";
        object? port = Arg("port");
        if (PyValues.Truthy(port))
        {
            if (port is not BigInteger number)
                throw new ValueError("port should be of type int");
            b.Port = CheckedPort(number);
        }
        b.UserID = PyValues.Truthy(Arg("user")) ? PyValues.Str(Arg("user")) : Environment.UserName;
        if (PyValues.Truthy(password))
            b.Password = PyValues.Str(password);
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

        ApplyCharset(b, "PyMySQL", Arg("charset"));
        if (Arg("local_infile") is { } infile)
            b.AllowLoadLocalInfile = PyValues.Truthy(infile);
        CheckClientFlag("PyMySQL", Arg("client_flag"));
        if (Arg("program_name") is { } program && PyValues.Truthy(program))
            b.ApplicationName = PyValues.Str(program);
        // use_unicode, binary_prefix and max_allowed_packet only change how Python encodes values; .NET strings and
        // MySqlConnector's parameters already behave as their defaults do.
        return b;
    }

    /// <summary>
    ///     PyMySQL's <c>_create_ssl_ctx</c> as a MySqlConnector SSL mode: TLS is required; the certificate is verified
    ///     when a CA is given or <c>verify_mode</c> asks for it, and the host name only when a CA is given.
    /// </summary>
    /// <param name="b">The builder to configure.</param>
    /// <param name="ssl">PyMySQL's SSL dictionary.</param>
    /// <exception cref="NotSupportedException">A CA directory or a cipher list, which MySqlConnector cannot take.</exception>
    private static void ApplyPyMySqlSsl(MySqlConnectionStringBuilder b, OrderedDict<string, object?> ssl)
    {
        object? Get(string key) => ssl.TryGetValue(key, out object? v) ? v : null;
        if (Get("capath") is not null)
            throw Unsupported("PyMySQL", "ssl_capath", Get("capath"));
        if (Get("cipher") is not null)
            throw Unsupported("PyMySQL", "ssl_cipher", Get("cipher"));
        bool hasCa = Get("ca") is not null;
        bool checkHostname = hasCa && (ssl.ContainsKey("check_hostname") ? PyValues.Truthy(Get("check_hostname")) : true);
        object? verify = Get("verify_mode");
        bool verifyCert = verify switch
        {
            null => hasCa,
            bool v => v,
            _ => PyValues.Str(verify).ToLowerInvariant() switch
            {
                "none" or "0" or "false" or "no" => false,
                "optional" or "required" or "1" or "true" or "yes" => true,
                _ => hasCa,
            },
        };
        if (hasCa)
            b.SslCa = PyValues.Str(Get("ca"));
        // ctx.load_cert_chain(cert, keyfile=key) runs only when a certificate is given; a key alone is ignored.
        if (Get("cert") is { } cert)
        {
            b.SslCert = PyValues.Str(cert);
            if (Get("key") is { } key)
                b.SslKey = PyValues.Str(key);
        }
        // ssl.create_default_context() trusts the system store when no CA file is given, as MySqlConnector does.
        b.SslMode = !verifyCert ? MySqlSslMode.Required : checkHostname ? MySqlSslMode.VerifyFull : MySqlSslMode.VerifyCA;
    }

    // ------------------------------------------------------------------ MySQLdb (mysqlclient)

    /// <summary>The keyword arguments MySQLdb's <c>Connection.__init__</c> and <c>_mysql.connect</c> accept.</summary>
    private static readonly HashSet<string> MySqlDbArguments = new(StringComparer.Ordinal)
    {
        "host", "user", "password", "passwd", "database", "db", "port", "unix_socket", "conv", "connect_timeout",
        "compress", "named_pipe", "init_command", "read_default_file", "read_default_group", "client_flag", "ssl",
        "ssl_mode", "local_infile", "read_timeout", "write_timeout", "charset", "auth_plugin", "server_public_key_path",
        "cursorclass", "collation", "use_unicode", "sql_mode", "binary_prefix", "multi_statements", "autocommit",
    };

    /// <summary>The arguments <c>_mysql.connect</c> parses with the C <c>int</c> format.</summary>
    private static readonly HashSet<string> MySqlDbIntArguments = new(StringComparer.Ordinal)
    {
        "port", "connect_timeout", "compress", "named_pipe", "client_flag", "local_infile", "read_timeout", "write_timeout",
    };

    /// <summary>
    ///     MySQLdb's <c>connect(**opts)</c> with the MySQL C API's defaults, expressed with MySqlConnector.
    /// </summary>
    /// <param name="opts">SQLAlchemy's arguments.</param>
    /// <returns>The connection string builder.</returns>
    /// <exception cref="TypeError">An unknown argument, or a string where the C API takes an integer.</exception>
    /// <exception cref="ValueError">An unknown <c>ssl_mode</c>.</exception>
    /// <exception cref="NotSupportedException">An argument MySqlConnector cannot express.</exception>
    private static MySqlConnectionStringBuilder FromMySqlDb(OrderedDict<string, object?> opts)
    {
        foreach (KeyValuePair<string, object?> pair in opts)
        {
            if (!MySqlDbArguments.Contains(pair.Key))
                throw new TypeError($"'{pair.Key}' is an invalid keyword argument for connect()");
            if (MySqlDbIntArguments.Contains(pair.Key) && pair.Value is not (BigInteger or bool))
                throw new TypeError($"'{PyValues.TypeName(pair.Value)}' object cannot be interpreted as an integer");
        }
        object? Arg(string key) => opts.TryGetValue(key, out object? v) ? v : null;
        foreach (string key in new[] { "conv", "named_pipe", "init_command", "read_default_file", "read_default_group",
                     "auth_plugin", "cursorclass", "collation", "sql_mode", "autocommit" })
        {
            if (PyValues.Truthy(Arg(key)))
                throw Unsupported("MySQLdb", key, Arg(key));
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
        ApplyCharset(b, "MySQLdb", Arg("charset"));
        CheckClientFlag("MySQLdb", Arg("client_flag"));
        if (Arg("server_public_key_path") is { } keyPath && PyValues.Truthy(keyPath))
            b.ServerRsaPublicKeyFile = PyValues.Str(keyPath);

        // mysqlclient: an ssl dictionary turns TLS on (REQUIRED since 2.2.4); ssl_mode chooses explicitly.
        OrderedDict<string, object?>? ssl = Arg("ssl") as OrderedDict<string, object?>;
        if (ssl is not null)
        {
            object? Get(string key) => ssl.TryGetValue(key, out object? v) ? v : null;
            if (Get("capath") is not null)
                throw Unsupported("MySQLdb", "ssl_capath", Get("capath"));
            if (Get("cipher") is not null)
                throw Unsupported("MySQLdb", "ssl_cipher", Get("cipher"));
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
        if (Arg("ssl_mode") is { } mode)
        {
            b.SslMode = PyValues.Str(mode).ToUpperInvariant() switch
            {
                "DISABLED" => MySqlSslMode.None,
                "PREFERRED" => MySqlSslMode.Preferred,
                "REQUIRED" => MySqlSslMode.Required,
                "VERIFY_CA" => MySqlSslMode.VerifyCA,
                "VERIFY_IDENTITY" => MySqlSslMode.VerifyFull,
                _ => throw new ValueError($"Unknown ssl_mode specification: {PyValues.Str(mode)}"),
            };
        }
        return b;
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
    ///     A <c>client_flag</c> argument: only <c>FOUND_ROWS</c> (2), which SQLAlchemy adds itself and Pomelo's
    ///     <c>UseAffectedRows=false</c> already provides, has a MySqlConnector counterpart.
    /// </summary>
    /// <param name="driver">The driver name, for messages.</param>
    /// <param name="flag">The argument, or <see langword="null" />.</param>
    /// <exception cref="NotSupportedException">Any other protocol flag.</exception>
    private static void CheckClientFlag(string driver, object? flag)
    {
        if (flag is BigInteger f && (f & ~new BigInteger(2)) != 0)
            throw Unsupported(driver, "client_flag", flag);
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

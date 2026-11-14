using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Numerics;
using NumSharp;                 // ValueError, TypeError, OverflowError
using OptunaSharp.Internal;
using OptunaSharp.Storages.SqlAlchemy;

namespace OptunaSharp.Storages.Journal;

/// <summary>
///     A neutral, client-agnostic Redis connection target parsed from a <c>redis-py</c> URL. Each Redis journal backend
///     (StackExchange / NRedisStack / ServiceStack) builds its own client from this — so the one bit of URL parsing that
///     <c>redis-py</c>'s <c>from_url</c> gives Optuna for free is shared, not triplicated. This is shared <em>data</em>,
///     not an abstraction: no backend implements a common interface, they only agree on the parsed endpoint.
/// </summary>
/// <remarks>
///     The values are those redis-py 8's connection object would hold for the same URL, defaults included (a 5-second
///     command and connect timeout, port 6379 for <c>rediss://</c> too). <see cref="RedisUrl.Parse" /> has already
///     rejected everything redis-py would reject. A backend maps each value onto its client, and raises
///     <see cref="NotSupportedException" /> for one its client cannot express (see each backend's remarks).
/// </remarks>
public sealed record RedisEndpoint
{
    /// <summary>Host name (redis-py lower-cases it; <c>localhost</c> when the URL has none; empty when <see cref="UnixSocketPath" /> is set).</summary>
    public string Host { get; init; } = "localhost";

    /// <summary>Port (defaults to 6379, for <c>rediss://</c> as well; port 0 in the URL also means the default, as in redis-py).</summary>
    public int Port { get; init; } = 6379;

    /// <summary>TLS (the <c>rediss://</c> scheme).</summary>
    public bool Ssl { get; init; }

    /// <summary>Database index, if the URL specified one (<c>?db=</c>, else the path); <see langword="null" /> means database 0.</summary>
    public int? Db { get; init; }

    /// <summary>ACL user name, if supplied. redis-py authenticates as this user with an empty password when the URL has no password.</summary>
    public string? User { get; init; }

    /// <summary>Password, if supplied.</summary>
    public string? Password { get; init; }

    /// <summary>When set, a unix-domain-socket target (<see cref="Host" />/<see cref="Port" /> unused).</summary>
    public string? UnixSocketPath { get; init; }

    /// <summary>
    ///     <c>socket_timeout</c> in seconds: how long one command waits for the server. redis-py 8's default is 5 seconds,
    ///     the same as StackExchange.Redis' own; always finite and positive.
    /// </summary>
    public double SocketTimeout { get; init; } = 5;

    /// <summary><c>socket_connect_timeout</c> in seconds (redis-py 8's default is 5 seconds); always finite and positive.</summary>
    public double SocketConnectTimeout { get; init; } = 5;

    /// <summary><c>client_name</c>: the name redis-py sends with <c>CLIENT SETNAME</c> after connecting.</summary>
    public string? ClientName { get; init; }

    /// <summary><c>lib_name</c>: the library name redis-py sends with <c>CLIENT SETINFO LIB-NAME</c> (informational only).</summary>
    public string? LibraryName { get; init; }

    /// <summary>
    ///     <c>protocol</c>: 2 (RESP2) or 3 (RESP3) when the URL names one. <see langword="null" /> leaves each client's
    ///     own default: redis-py 8 then speaks RESP3, the .NET clients RESP2, and the journal's commands return the same
    ///     results under both.
    /// </summary>
    public int? Protocol { get; init; }

    /// <summary>
    ///     <c>ssl_cert_reqs</c> for <c>rediss://</c>: <c>none</c> (no certificate check), <c>optional</c> or <c>required</c>
    ///     (verify it; a server always presents one, so in a client the two behave alike). Defaults to <c>required</c>.
    /// </summary>
    public string SslCertReqs { get; init; } = "required";

    /// <summary>
    ///     Whether the server certificate must name <see cref="Host" />: <c>ssl_check_hostname</c>, which defaults to
    ///     <see langword="true" />, and is always <see langword="false" /> when <see cref="SslCertReqs" /> is <c>none</c>.
    /// </summary>
    public bool SslCheckHostname { get; init; } = true;

    /// <summary><c>ssl_ca_certs</c>: a PEM file of certificate authorities to trust besides the system's.</summary>
    public string? SslCaCerts { get; init; }

    /// <summary><c>ssl_ca_data</c>: PEM text of certificate authorities to trust besides the system's.</summary>
    public string? SslCaData { get; init; }

    /// <summary>
    ///     <c>ssl_ca_path</c>: an OpenSSL hashed certificate directory (files named <c>&lt;hash&gt;.&lt;n&gt;</c>, as
    ///     <c>c_rehash</c> writes them) of certificate authorities to trust besides the system's.
    /// </summary>
    public string? SslCaPath { get; init; }

    /// <summary><c>ssl_certfile</c>: the client certificate (PEM); it also holds the private key when <see cref="SslKeyFile" /> is not set.</summary>
    public string? SslCertFile { get; init; }

    /// <summary><c>ssl_keyfile</c>: the client certificate's private key (PEM).</summary>
    public string? SslKeyFile { get; init; }

    /// <summary><c>ssl_password</c>: the password of an encrypted <see cref="SslKeyFile" /> (ignored for an unencrypted key, as in Python).</summary>
    public string? SslPassword { get; init; }

    /// <summary>
    ///     <c>ssl_min_version</c> as Python's <c>ssl.TLSVersion</c> value: -2 (the minimum supported), -1 (the maximum
    ///     supported), 768 (SSL 3.0), 769–772 (TLS 1.0–1.3). <see langword="null" /> keeps the platform's default.
    /// </summary>
    public int? SslMinVersion { get; init; }

    /// <summary>
    ///     A timeout in whole milliseconds for the .NET clients: rounded up (Python rounds a socket timeout up as well)
    ///     and capped at <see cref="int.MaxValue" /> (about 24.8 days), the most a .NET socket or client takes.
    /// </summary>
    /// <param name="seconds">A finite, positive timeout in seconds.</param>
    /// <returns>The timeout in milliseconds, at least 1.</returns>
    internal static int ToMilliseconds(double seconds)
    {
        double milliseconds = Math.Ceiling(seconds * 1000);
        return milliseconds >= int.MaxValue ? int.MaxValue : Math.Max(1, (int)milliseconds);
    }
}

/// <summary>
///     Parses Redis connection URLs exactly as redis-py 8's <c>redis.connection.parse_url</c> (behind
///     <c>Redis.from_url</c>, which Optuna's <c>JournalRedisBackend</c> calls) does, and checks the connection options the
///     way redis-py's connection pool, its connection classes and its first connect do.
/// </summary>
/// <remarks>
///     <para><b>redis-py's rules, reproduced:</b> the scheme must be exactly <c>redis://</c>, <c>rediss://</c> or
///     <c>unix://</c> (case-sensitive); the user information is split at its <b>last</b> <c>@</c> and wins over
///     <c>?username=</c>/<c>?password=</c>; the host is lower-cased and percent-decoded; a port must be 0–65535 and
///     port 0 means the default; <c>?host=</c> and <c>?port=</c> are used when the URL itself has none; the database
///     comes from <c>?db=</c> or, failing that, the path with every <c>/</c> removed (<c>/1/2</c> is database 12);
///     query values are form-decoded (<c>+</c> is a space), the first of a repeated key wins, and a key without a value
///     is ignored; typed keys (<c>db</c>, <c>socket_timeout</c>, <c>protocol</c>, …) are converted with Python's
///     <c>int</c>/<c>float</c> and redis-py's <c>to_bool</c>, and a bad value raises redis-py's <c>Invalid value for
///     '…' in connection URL.</c>.</para>
///     <para><b>Errors:</b> what redis-py raises as <c>ValueError</c> or <c>TypeError</c> is a <see cref="ValueError" />
///     or <see cref="TypeError" /> with redis-py's message (an argument the connection class does not take, such as
///     <c>?foo=</c>, <c>?timeout=</c>, or an SSL argument on <c>redis://</c>, is the <c>TypeError</c>). What it raises as
///     its own <c>ConnectionError</c>, <c>RedisError</c> or <c>DataError</c> (a <c>protocol</c> other than 2 or 3, an
///     unknown <c>ssl_cert_reqs</c>) is a <see cref="ValueError" /> with the same message. redis-py raises the
///     connection-class errors on the first command, and the socket and server ones (a negative timeout, a TLS version
///     OpenSSL does not know, a client name the server refuses, a database index out of range) on the first connect;
///     Optuna's <c>JournalStorage</c> issues that command while it is constructed, and this parser raises them all up
///     front.</para>
///     <para><b>Accepted without effect</b>, because they only tune redis-py's own pool, retries, health checks or response
///     decoding: <c>max_connections</c>, <c>cache_factory</c>, <c>retry_on_timeout</c>, <c>retry_on_error</c>,
///     <c>health_check_interval</c>, <c>socket_keepalive</c>, <c>socket_read_size</c>, <c>encoding</c> (when it names
///     UTF-8, which the .NET clients always use), <c>encoding_errors</c>, <c>decode_responses</c>, <c>lib_version</c>,
///     <c>legacy_responses</c>, and <c>ssl_validate_ocsp</c> (from a URL it is text, which redis-py's <c>is True</c> test
///     never matches). <b>Refused</b> with <see cref="NotSupportedException" />: arguments that take a Python object
///     (<c>retry</c>, <c>credential_provider</c>, <c>event_dispatcher</c>, <c>cache</c>, …), another text encoding, and
///     TLS settings no .NET client offers (<c>ssl_ciphers</c>, <c>ssl_include_verify_flags</c>,
///     <c>ssl_exclude_verify_flags</c>, <c>ssl_validate_ocsp_stapled</c>).</para>
/// </remarks>
public static class RedisUrl
{
    /// <summary>redis-py 8.1's <c>URL_QUERY_ARGUMENT_PARSERS</c>: the query keys converted from text, and how.</summary>
    private static readonly Dictionary<string, string> Parsers = new(StringComparer.Ordinal)
    {
        ["db"] = "int", ["socket_timeout"] = "float", ["socket_connect_timeout"] = "float", ["socket_read_size"] = "int",
        ["socket_keepalive"] = "bool", ["retry_on_timeout"] = "bool", ["retry_on_error"] = "list", ["max_connections"] = "int",
        ["health_check_interval"] = "int", ["ssl_check_hostname"] = "bool", ["ssl_include_verify_flags"] = "verify_flags",
        ["ssl_exclude_verify_flags"] = "verify_flags", ["ssl_min_version"] = "int", ["timeout"] = "float", ["protocol"] = "int",
        ["legacy_responses"] = "bool",
    };

    /// <summary>The keyword arguments of redis-py 8.1's <c>AbstractConnection.__init__</c>, which every connection class takes.</summary>
    private static readonly HashSet<string> AbstractArguments = new(StringComparer.Ordinal)
    {
        "db", "password", "socket_timeout", "socket_connect_timeout", "retry_on_timeout", "retry_on_error", "encoding",
        "encoding_errors", "decode_responses", "parser_class", "socket_read_size", "health_check_interval", "client_name",
        "lib_name", "lib_version", "driver_info", "username", "retry", "redis_connect_func", "credential_provider", "protocol",
        "legacy_responses", "command_packer", "event_dispatcher", "maint_notifications_config", "maint_notifications_pool_handler",
        "maintenance_state", "maintenance_notification_hash", "orig_host_address", "orig_socket_timeout",
        "orig_socket_connect_timeout", "oss_cluster_maint_notifications_handler", "himport_registry",
    };

    /// <summary>The extra arguments of <c>Connection</c> (TCP, and the base of <c>SSLConnection</c>).</summary>
    private static readonly HashSet<string> TcpArguments = new(StringComparer.Ordinal)
    {
        "host", "port", "socket_keepalive", "socket_keepalive_options", "socket_type",
    };

    /// <summary>The extra arguments of <c>SSLConnection</c> (<c>rediss://</c>).</summary>
    private static readonly HashSet<string> SslArguments = new(StringComparer.Ordinal)
    {
        "ssl_keyfile", "ssl_certfile", "ssl_cert_reqs", "ssl_include_verify_flags", "ssl_exclude_verify_flags", "ssl_ca_certs",
        "ssl_ca_data", "ssl_check_hostname", "ssl_ca_path", "ssl_password", "ssl_validate_ocsp", "ssl_validate_ocsp_stapled",
        "ssl_ocsp_context", "ssl_ocsp_expected_cert", "ssl_min_version", "ssl_ciphers",
    };

    /// <summary>
    ///     Arguments <c>ConnectionPool.__init__</c> consumes itself (<c>cache</c> and <c>cache_config</c> it drops after
    ///     checking them), so the connection class never sees them.
    /// </summary>
    private static readonly HashSet<string> PoolArguments = new(StringComparer.Ordinal)
    {
        "connection_class", "max_connections", "cache_factory", "maint_notifications_config", "cache", "cache_config",
    };

    /// <summary>Arguments a backend maps onto its client (see <see cref="RedisEndpoint" />).</summary>
    private static readonly HashSet<string> MappedArguments = new(StringComparer.Ordinal)
    {
        "host", "port", "path", "db", "username", "password", "socket_timeout", "socket_connect_timeout", "client_name",
        "lib_name", "protocol", "ssl_cert_reqs", "ssl_check_hostname", "ssl_ca_certs", "ssl_ca_data", "ssl_ca_path",
        "ssl_certfile", "ssl_keyfile", "ssl_password", "ssl_min_version",
    };

    /// <summary>
    ///     Arguments that only tune redis-py's own pool, retries, health checks or decoding, or that redis-py never
    ///     consults for a URL's text value: accepted, without effect.
    /// </summary>
    private static readonly HashSet<string> NoOpArguments = new(StringComparer.Ordinal)
    {
        "max_connections", "cache_factory", "retry_on_timeout", "retry_on_error", "health_check_interval", "socket_keepalive",
        "socket_read_size", "encoding_errors", "decode_responses", "lib_version", "legacy_responses", "ssl_validate_ocsp",
        "ssl_ocsp_context", "ssl_ocsp_expected_cert",
    };

    /// <summary>The members of Python's <c>ssl.VerifyFlags</c>, the names <c>parse_ssl_verify_flags</c> accepts.</summary>
    private static readonly HashSet<string> VerifyFlags = new(StringComparer.Ordinal)
    {
        "VERIFY_DEFAULT", "VERIFY_CRL_CHECK_LEAF", "VERIFY_CRL_CHECK_CHAIN", "VERIFY_X509_STRICT", "VERIFY_X509_TRUSTED_FIRST",
        "VERIFY_ALLOW_PROXY_CERTS", "VERIFY_X509_PARTIAL_CHAIN",
    };

    /// <summary>The names Python's codec registry resolves to its UTF-8 codec, after its own name normalization.</summary>
    private static readonly HashSet<string> Utf8Codecs = new(StringComparer.Ordinal)
    {
        "utf_8", "u8", "utf", "utf8", "utf8_ucs2", "utf8_ucs4", "cp65001",
    };

    /// <summary>
    ///     Parses a redis-py URL and checks its connection options (see the type's remarks).
    /// </summary>
    /// <param name="url">The Redis URL.</param>
    /// <returns>The endpoint, with redis-py's defaults for everything the URL leaves out.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="url" /> is null.</exception>
    /// <exception cref="ValueError">
    ///     An unsupported scheme, a malformed port or bracketed host, an invalid typed query value, a negative
    ///     <c>max_connections</c>, an unknown <c>ssl_cert_reqs</c>, a non-integer or out-of-range <c>?port=</c>, a
    ///     <c>protocol</c> other than 2 or 3, a NaN or negative timeout, an unknown TLS version, both OCSP modes, a client
    ///     name with spaces or special characters, a database index outside 0–2147483647, or a user name or password
    ///     next to a <c>credential_provider</c>; redis-py's, Python's or the server's messages.
    /// </exception>
    /// <exception cref="TypeError">
    ///     An argument the connection class does not take, <c>self</c>, a text value where redis-py calls a class
    ///     (<c>connection_class</c> on <c>redis://</c>, <c>parser_class</c>) or needs a <c>DriverInfo</c>, or an
    ///     <c>ssl_keyfile</c> without an <c>ssl_certfile</c> (<c>certfile should be a valid filesystem path</c>).
    /// </exception>
    /// <exception cref="OverflowError">An infinite or astronomically large timeout, or a TLS version beyond a C long; Python's messages.</exception>
    /// <exception cref="NotSupportedException">An argument that takes a Python object, a non-UTF-8 encoding, or a TLS setting no .NET client offers.</exception>
    public static RedisEndpoint Parse(string url)
    {
        ArgumentNullException.ThrowIfNull(url);
        // parse_url: the scheme test is a case-sensitive prefix test, so " redis://h" and "REDIS://h" fail here.
        if (!(url.StartsWith("redis://", StringComparison.Ordinal) || url.StartsWith("rediss://", StringComparison.Ordinal)
              || url.StartsWith("unix://", StringComparison.Ordinal)))
            throw new ValueError("Redis URL must specify one of the following schemes (redis://, rediss://, unix://)");

        PySplitResult parts = PyUrllib.UrlSplit(url);
        var kwargs = new Kwargs();
        foreach (KeyValuePair<string, List<string>> pair in PyUrllib.ParseQs(parts.Query))
        {
            // parse_qs already decoded the value (redis-py does not unquote it again); the first of a repeated key wins.
            string value = pair.Value[0];
            if (!Parsers.TryGetValue(pair.Key, out string? parser))
            {
                kwargs.Set(pair.Key, value);
                continue;
            }
            try
            {
                kwargs.Set(pair.Key, parser switch
                {
                    "int" => PyNumbers.ParseInt(value),
                    "float" => PyNumbers.ParseFloat(value),
                    "bool" => ToBool(value),
                    // list(value) of a string is its characters.
                    "list" => value.Select(c => c.ToString()).ToArray(),
                    _ => (object)ParseVerifyFlags(value),
                });
            }
            catch (ValueError)
            {
                throw new ValueError($"Invalid value for '{pair.Key}' in connection URL.");
            }
        }
        // The user information wins over ?username= / ?password=, but only when it is not empty.
        if (!string.IsNullOrEmpty(parts.Username))
            kwargs.Set("username", PyUrllib.Unquote(parts.Username));
        if (!string.IsNullOrEmpty(parts.Password))
            kwargs.Set("password", PyUrllib.Unquote(parts.Password));

        bool unix = parts.Scheme == "unix";
        bool ssl = parts.Scheme == "rediss";
        if (unix)
        {
            if (parts.Path.Length > 0)
                kwargs.Set("path", PyUrllib.Unquote(parts.Path));
        }
        else
        {
            if (parts.Hostname is { Length: > 0 } hostname)
                kwargs.Set("host", PyUrllib.Unquote(hostname));
            // The port is read (and validated) only here: for unix:// a bad port is never looked at.
            if (parts.Port is int port and not 0)
                kwargs.Set("port", new BigInteger(port));
            // The path is the database only without ?db=, and only when the slash-less path is a Python int.
            if (parts.Path.Length > 0 && !kwargs.Contains("db")
                && PyNumbers.TryParseInt(PyUrllib.Unquote(parts.Path).Replace("/", "", StringComparison.Ordinal), out BigInteger pathDb))
                kwargs.Set("db", pathDb);
        }

        CheckPool(kwargs);
        CheckConnection(kwargs, unix, ssl);
        CheckSupported(kwargs, unix, ssl);
        return Build(kwargs, unix, ssl);
    }

    /// <summary>What <c>ConnectionPool(**kwargs)</c> rejects: its own binding, <c>max_connections</c>, and the Python-object pool arguments.</summary>
    /// <param name="kwargs">The parsed options.</param>
    /// <exception cref="TypeError">A <c>self</c> key, which collides with the pool's own <c>self</c> parameter.</exception>
    /// <exception cref="ValueError">A negative <c>max_connections</c>.</exception>
    /// <exception cref="NotSupportedException"><c>cache</c>, <c>cache_config</c> or <c>maint_notifications_config</c>, which take Python objects.</exception>
    private static void CheckPool(Kwargs kwargs)
    {
        if (kwargs.Contains("self"))
            throw new TypeError("ConnectionPool.__init__() got multiple values for argument 'self'");
        // max_connections or 100: zero means the default, a negative count is refused.
        if (kwargs.Get("max_connections") is BigInteger max && max < 0)
            throw new ValueError("\"max_connections\" must be a positive integer");
        foreach (string key in kwargs.Keys)
        {
            if (key is "cache" or "cache_config" or "maint_notifications_config")
                throw Unsupported(key, kwargs.Get(key));
        }
    }

    /// <summary>
    ///     What constructing the connection class rejects, in redis-py's order: <c>SSLConnection</c>'s
    ///     <c>ssl_cert_reqs</c>, <c>Connection</c>'s <c>int(port)</c>, <c>AbstractConnection</c>'s argument binding, then
    ///     its credential, <c>driver_info</c>, <c>protocol</c> and <c>parser_class</c> checks.
    /// </summary>
    /// <param name="kwargs">The parsed options; a <c>?port=</c> text is replaced by its integer.</param>
    /// <param name="unix">Whether the URL is <c>unix://</c>.</param>
    /// <param name="ssl">Whether the URL is <c>rediss://</c>.</param>
    /// <exception cref="TypeError">A <c>connection_class</c> on <c>redis://</c>, an argument the class does not take, a <c>driver_info</c> or a <c>parser_class</c>.</exception>
    /// <exception cref="ValueError">An unknown <c>ssl_cert_reqs</c>, a non-integer <c>?port=</c>, a protocol other than 2 or 3, or credentials next to a <c>credential_provider</c>.</exception>
    private static void CheckConnection(Kwargs kwargs, bool unix, bool ssl)
    {
        // parse_url replaces connection_class for rediss:// and unix://, so only redis:// calls the text.
        if (!unix && !ssl && kwargs.Contains("connection_class"))
            throw new TypeError("'str' object is not callable");
        if (ssl && kwargs.Get("ssl_cert_reqs") is string reqs && reqs is not ("none" or "optional" or "required"))
            throw new ValueError($"Invalid SSL Certificate Requirements Flag: {reqs}");
        // Connection.__init__ runs int(port) on a ?port= value before AbstractConnection sees any argument.
        if (!unix && kwargs.Get("port") is string portText)
            kwargs.Set("port", PyNumbers.ParseInt(portText));
        foreach (string key in kwargs.Keys)
        {
            if (PoolArguments.Contains(key))
                continue;
            bool accepted = AbstractArguments.Contains(key)
                            || (unix ? key == "path" : TcpArguments.Contains(key) || (ssl && SslArguments.Contains(key)));
            if (!accepted)
                throw new TypeError($"AbstractConnection.__init__() got an unexpected keyword argument '{key}'");
        }
        if (kwargs.Contains("credential_provider") && (Truthy(kwargs.Get("username")) || Truthy(kwargs.Get("password"))))
            throw new ValueError(
                "'username' and 'password' cannot be passed along with 'credential_provider'. Please provide only one of the " +
                "following arguments: \n1. 'password' and (optional) 'username'\n2. 'credential_provider'");
        if (kwargs.Contains("driver_info"))
            throw new TypeError("driver_info must be a DriverInfo instance or None");
        if (kwargs.Get("protocol") is BigInteger protocol && (protocol < 2 || protocol > 3))
            throw new ValueError("protocol must be either 2 or 3");
        // set_parser calls parser_class(socket_read_size=...), and text is not callable.
        if (kwargs.Contains("parser_class"))
            throw new TypeError("'str' object is not callable");
    }

    /// <summary>Refuses the accepted arguments no .NET client can honor (see the type's remarks).</summary>
    /// <param name="kwargs">The parsed options.</param>
    /// <param name="unix">Whether the URL is <c>unix://</c>.</param>
    /// <param name="ssl">Whether the URL is <c>rediss://</c>.</param>
    /// <exception cref="NotSupportedException">The first such argument, in URL order.</exception>
    private static void CheckSupported(Kwargs kwargs, bool unix, bool ssl)
    {
        foreach (string key in kwargs.Keys)
        {
            if (MappedArguments.Contains(key) || NoOpArguments.Contains(key))
                continue;
            // rediss:// and unix:// replace connection_class with their own class.
            if (key == "connection_class" && (unix || ssl))
                continue;
            if (key == "encoding" && IsUtf8Codec((string)kwargs.Get(key)!))
                continue;
            // The keepalive options are only read while socket_keepalive (True by default) is on.
            if (key == "socket_keepalive_options" && kwargs.Get("socket_keepalive") is false)
                continue;
            // With both OCSP modes, redis-py's connect fails on "not both" before it tries the stapled check; Build
            // raises that error.
            if (key == "ssl_validate_ocsp_stapled" && ssl && Truthy(kwargs.Get(key)) && Truthy(kwargs.Get("ssl_validate_ocsp")))
                continue;
            throw Unsupported(key, kwargs.Get(key));
        }
    }

    /// <summary>
    ///     Builds the endpoint, checking what redis-py's first connect checks, in its order: the port, the connect and
    ///     command timeouts, the TLS version, the OCSP modes, then the client name and database the server checks.
    /// </summary>
    /// <param name="kwargs">The checked options.</param>
    /// <param name="unix">Whether the URL is <c>unix://</c>.</param>
    /// <param name="ssl">Whether the URL is <c>rediss://</c>.</param>
    /// <returns>The endpoint.</returns>
    /// <exception cref="ValueError">A port outside 1–65535 from <c>?port=</c>, a NaN or negative timeout, an unknown TLS version, both OCSP modes, a client name the server refuses, or a database index outside 0–2147483647.</exception>
    /// <exception cref="OverflowError">A timeout of 2⁶³ nanoseconds or more, or a TLS version beyond a C long.</exception>
    /// <exception cref="NotSupportedException">A zero timeout, which makes a Python socket non-blocking.</exception>
    /// <exception cref="TypeError">
    ///     <c>rediss://</c> with an <c>ssl_keyfile</c> but no <c>ssl_certfile</c>: redis-py calls
    ///     <c>load_cert_chain(certfile=None, …)</c>, which the <c>ssl</c> module refuses (<c>certfile should be a valid
    ///     filesystem path</c>).
    /// </exception>
    private static RedisEndpoint Build(Kwargs kwargs, bool unix, bool ssl)
    {
        int port = 6379;
        if (!unix && kwargs.Get("port") is BigInteger p)
        {
            // Only ?port= can get here outside 1-65535 (the URL's own port is range-checked, and 0 means the default):
            // Python's getaddrinfo would wrap it modulo 65536 or fail, and a .NET client would take 0 as its default port.
            if (p < 1 || p > 65535)
                throw new ValueError($"port must be 1-65535, not {p.ToString(CultureInfo.InvariantCulture)}");
            port = (int)p;
        }
        double connectTimeout = Timeout("socket_connect_timeout", kwargs.Get("socket_connect_timeout"));
        double socketTimeout = Timeout("socket_timeout", kwargs.Get("socket_timeout"));

        // _wrap_socket_with_ssl loads the client certificate ("if self.certfile or self.keyfile") before it sets the
        // TLS version, and the ssl module refuses a key without a certificate file.
        if (ssl && kwargs.Get("ssl_certfile") is null && Truthy(kwargs.Get("ssl_keyfile")))
            throw new TypeError("certfile should be a valid filesystem path");

        int? minVersion = null;
        if (ssl && kwargs.Get("ssl_min_version") is BigInteger version)
        {
            if (version < int.MinValue || version > int.MaxValue)
                throw new OverflowError("Python int too large to convert to C long");
            int v = (int)version;
            // _ssl.c formats the rejected value with %x, i.e. as an unsigned 32-bit pattern.
            if (v is not (-2 or -1 or 768 or 769 or 770 or 771 or 772))
                throw new ValueError($"Unsupported TLS/SSL version 0x{unchecked((uint)v).ToString("x", CultureInfo.InvariantCulture)}");
            minVersion = v;
        }
        if (ssl && Truthy(kwargs.Get("ssl_validate_ocsp_stapled")) && Truthy(kwargs.Get("ssl_validate_ocsp")))
            throw new ValueError("Either an OCSP staple or pure OCSP connection must be validated - not both.");

        string? clientName = kwargs.Get("client_name") as string;
        // CLIENT SETNAME is sent for a non-empty name, and the server refuses any byte outside '!'..'~'; the .NET
        // clients would instead strip those characters, or send the name as it is, so it is refused up front.
        if (!string.IsNullOrEmpty(clientName) && clientName.Any(c => c < '!' || c > '~'))
            throw new ValueError("Client names cannot contain spaces, newlines or special characters.");

        int? db = null;
        if (kwargs.Get("db") is BigInteger d)
        {
            // SELECT runs for a non-zero index, and the server refuses a negative one; the .NET clients take -1 as
            // "the default database" and an int at most, so anything outside 0..int.MaxValue is refused up front.
            if (d < 0 || d > int.MaxValue)
                throw new ValueError("DB index is out of range");
            db = (int)d;
        }

        string certReqs = ssl && kwargs.Get("ssl_cert_reqs") is string r ? r : "required";
        return new RedisEndpoint
        {
            Host = unix ? "" : kwargs.Get("host") as string ?? "localhost",
            Port = port,
            Ssl = ssl,
            Db = db,
            User = kwargs.Get("username") as string,
            Password = kwargs.Get("password") as string,
            UnixSocketPath = unix ? kwargs.Get("path") as string ?? "" : null,
            SocketTimeout = socketTimeout,
            SocketConnectTimeout = connectTimeout,
            ClientName = string.IsNullOrEmpty(clientName) ? null : clientName,
            LibraryName = kwargs.Get("lib_name") as string,
            Protocol = kwargs.Get("protocol") is BigInteger protocol ? (int)protocol : null,
            SslCertReqs = certReqs,
            // SSLConnection turns the host name check off whenever certificates are not checked at all; a plain
            // connection keeps the record's default, which nothing reads.
            SslCheckHostname = !ssl || (certReqs != "none" && kwargs.Get("ssl_check_hostname") is not false),
            SslCaCerts = ssl ? kwargs.Get("ssl_ca_certs") as string : null,
            SslCaData = ssl ? kwargs.Get("ssl_ca_data") as string : null,
            SslCaPath = ssl ? kwargs.Get("ssl_ca_path") as string : null,
            SslCertFile = ssl ? kwargs.Get("ssl_certfile") as string : null,
            SslKeyFile = ssl ? kwargs.Get("ssl_keyfile") as string : null,
            SslPassword = ssl ? kwargs.Get("ssl_password") as string : null,
            SslMinVersion = minVersion,
        };
    }

    /// <summary>A socket timeout as Python's <c>socket.settimeout</c> takes it, defaulting to redis-py's 5 seconds.</summary>
    /// <param name="key">The option's name, for the message of a refused zero.</param>
    /// <param name="value">The parsed float, or <see langword="null" /> when the URL has none.</param>
    /// <returns>The timeout in seconds.</returns>
    /// <exception cref="ValueError">NaN or a negative value; Python's messages.</exception>
    /// <exception cref="OverflowError">2⁶³ nanoseconds or more (infinity included), which Python's time type cannot hold.</exception>
    /// <exception cref="NotSupportedException">Zero, which makes a Python socket non-blocking.</exception>
    private static double Timeout(string key, object? value)
    {
        if (value is not double seconds)
            return 5;
        if (double.IsNaN(seconds))
            throw new ValueError("Invalid value NaN (not a number)");
        // Python rounds a timeout away from zero to whole nanoseconds, then requires it to fit a signed 64-bit count.
        double nanoseconds = seconds >= 0 ? Math.Ceiling(seconds * 1e9) : Math.Floor(seconds * 1e9);
        if (!(nanoseconds >= -9223372036854775808.0 && nanoseconds < 9223372036854775808.0))
            throw new OverflowError("timestamp out of range for platform time_t");
        if (nanoseconds < 0)
            throw new ValueError("Timeout value out of range");
        if (nanoseconds == 0)
            throw new NotSupportedException(
                $"{key}=0 makes a redis-py socket non-blocking, so its first read fails; no .NET Redis client offers that mode. " +
                "Give a positive number of seconds.");
        return seconds;
    }

    /// <summary>redis-py's <c>to_bool</c> for a non-empty value: <c>0 F FALSE N NO</c> (any case) are false, everything else true.</summary>
    /// <param name="value">The query value.</param>
    /// <returns>The boolean.</returns>
    /// <remarks>
    ///     Python's <c>str.upper()</c> maps more characters than <see cref="string.ToUpperInvariant" /> (<c>ı</c>,
    ///     and full mappings such as <c>ß</c> → <c>SS</c>), but none of those can produce one of these five ASCII
    ///     words: the only non-ASCII characters whose upper case lands on their letters are <c>ſ</c> → <c>S</c>, which
    ///     both map, and ligatures such as <c>ﬂ</c> → <c>FL</c>, which no word contains. So the comparison is exact.
    /// </remarks>
    private static bool ToBool(string value) => value.ToUpperInvariant() is not ("0" or "F" or "FALSE" or "N" or "NO");

    /// <summary>redis-py's <c>parse_ssl_verify_flags</c>: <c>[A, B]</c> or <c>A,B</c>, each a member of <c>ssl.VerifyFlags</c>.</summary>
    /// <param name="value">The query value.</param>
    /// <returns>The flag names.</returns>
    /// <exception cref="ValueError">An unknown flag name.</exception>
    private static string[] ParseVerifyFlags(string value)
    {
        string[] flags = value.Replace("[", "", StringComparison.Ordinal).Replace("]", "", StringComparison.Ordinal)
            .Split(',').Select(PyStrings.Strip).ToArray();
        foreach (string flag in flags)
        {
            if (!VerifyFlags.Contains(flag))
                throw new ValueError($"Invalid ssl verify flag: {flag}");
        }
        return flags;
    }

    /// <summary>
    ///     Whether Python's <c>codecs.lookup(name)</c> finds its UTF-8 codec: the name is normalized as CPython does
    ///     (ASCII letters, digits and dots kept and lower-cased, every other run turned into one underscore), then looked
    ///     up among UTF-8's aliases.
    /// </summary>
    /// <param name="name">The <c>encoding</c> value.</param>
    /// <returns><see langword="true" /> for a UTF-8 name.</returns>
    private static bool IsUtf8Codec(string name)
    {
        var normalized = new System.Text.StringBuilder(name.Length);
        bool punctuation = false;
        foreach (char c in name)
        {
            if (char.IsAsciiLetterOrDigit(c) || c == '.')
            {
                if (punctuation && normalized.Length > 0)
                    normalized.Append('_');
                normalized.Append(char.ToLowerInvariant(c));
                punctuation = false;
            }
            else
            {
                punctuation = true;
            }
        }
        string text = normalized.ToString();
        // encodings.search_function tries the alias table as it is, then with dots as underscores.
        return Utf8Codecs.Contains(text) || Utf8Codecs.Contains(text.Replace('.', '_'));
    }

    /// <summary>Python truthiness of a parsed option: a non-empty string, <see langword="true" />, a non-zero number, a non-empty list.</summary>
    /// <param name="value">The option value, or <see langword="null" /> when absent.</param>
    /// <returns>Whether Python would treat the value as true.</returns>
    private static bool Truthy(object? value) => value switch
    {
        null => false,
        string s => s.Length > 0,
        bool b => b,
        BigInteger i => !i.IsZero,
        double d => d != 0,
        string[] list => list.Length > 0,
        _ => true,
    };

    /// <summary>The refusal of an argument that takes a Python object, or a setting no .NET Redis client offers.</summary>
    /// <param name="key">The argument.</param>
    /// <param name="value">Its parsed value.</param>
    /// <returns>The exception to throw.</returns>
    private static NotSupportedException Unsupported(string key, object? value)
    {
        string reason = key switch
        {
            "encoding" => "the .NET Redis clients always encode text as UTF-8",
            "ssl_ciphers" => "the .NET Redis clients cannot take an OpenSSL cipher list",
            "ssl_include_verify_flags" or "ssl_exclude_verify_flags" => "the .NET Redis clients cannot take OpenSSL verification flags",
            "ssl_validate_ocsp_stapled" => "the .NET Redis clients cannot require a stapled OCSP response",
            _ => "redis-py takes a Python object there, which a URL cannot express",
        };
        return new NotSupportedException(
            $"The redis-py connection argument {key}={PythonRepr.StringRepr(PyStr(value))} cannot be honored: {reason}. " +
            "Remove it from the URL, or configure the client through the backend's configure callback.");
    }

    /// <summary>Python's <c>str()</c> of a parsed option, for messages.</summary>
    /// <param name="value">The option value.</param>
    /// <returns>The text.</returns>
    private static string PyStr(object? value) => value switch
    {
        string[] list => PythonRepr.ListRepr(list),
        BigInteger i => i.ToString(CultureInfo.InvariantCulture),
        double d => PythonRepr.FloatRepr(d),
        bool b => b ? "True" : "False",
        _ => value?.ToString() ?? "None",
    };

    /// <summary>
    ///     The keyword arguments redis-py builds, as a Python <c>dict</c>: insertion-ordered, and assigning an existing key
    ///     keeps its original position. The order decides which bad argument Python reports first.
    /// </summary>
    private sealed class Kwargs
    {
        /// <summary>The values by key.</summary>
        private readonly Dictionary<string, object?> _values = new(StringComparer.Ordinal);

        /// <summary>The keys in first-insertion order.</summary>
        private readonly List<string> _order = new();

        /// <summary>The keys in Python's iteration order (first insertion), as a snapshot safe to iterate while assigning.</summary>
        internal string[] Keys => _order.ToArray();

        /// <summary>Sets a value; setting an existing key keeps its position, as a Python <c>dict</c> does.</summary>
        /// <param name="key">The argument name.</param>
        /// <param name="value">The parsed value.</param>
        internal void Set(string key, object? value)
        {
            if (!_values.ContainsKey(key))
                _order.Add(key);
            _values[key] = value;
        }

        /// <summary>Whether the argument is present.</summary>
        /// <param name="key">The argument name.</param>
        /// <returns><see langword="true" /> when present.</returns>
        internal bool Contains(string key) => _values.ContainsKey(key);

        /// <summary>The argument's value, or <see langword="null" /> when absent.</summary>
        /// <param name="key">The argument name.</param>
        /// <returns>The value.</returns>
        internal object? Get(string key) => _values.TryGetValue(key, out object? value) ? value : null;
    }
}

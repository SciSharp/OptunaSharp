using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Numerics;
using System.Text;
using System.Text.RegularExpressions;
using Npgsql;
using NumSharp;
using OptunaSharp.Collections;
using OptunaSharp.Internal;
using OptunaSharp.Storages.SqlAlchemy;

namespace OptunaSharp.Storages.EFCore;

/// <summary>
///     Turns a PostgreSQL storage URL into the Npgsql connection string for the database Python Optuna would reach:
///     SQLAlchemy's <c>create_connect_args</c> for the URL's driver, the driver's own processing of those arguments
///     (psycopg2's <c>make_dsn</c>, pg8000's keyword arguments), libpq's parsing and validation of the resulting
///     connection info, libpq's environment-variable defaults, and finally the libpq-to-Npgsql translation.
/// </summary>
/// <remarks>
///     <para><b>The standard.</b> With <c>postgresql://</c>, <c>+psycopg2</c>, <c>+psycopg2cffi</c> and <c>+psycopg</c>
///     every query argument is a libpq connection parameter
///     (<see href="https://www.postgresql.org/docs/current/libpq-connect.html#LIBPQ-PARAMKEYWORDS" />): its name must be
///     one libpq 18 knows (<c>invalid connection option "…"</c> otherwise), its value is checked the way libpq checks
///     it, and a parameter missing from the URL falls back to its <c>PG*</c> environment variable, as libpq does.
///     Multi-host URLs (<c>?host=h1:5432&amp;host=h2:5433</c> or <c>?host=h1,h2&amp;port=5432,5433</c>), Unix-domain
///     socket directories (<c>?host=/var/run/postgresql</c>) and the <c>?user=</c>/<c>?dbname=</c> overrides behave as
///     in Python.</para>
///     <para><b>What Npgsql cannot express.</b> A libpq parameter with no Npgsql counterpart (<c>hostaddr</c>,
///     <c>service</c>, <c>sslcrl</c>, <c>keepalives_count</c>, <c>tcp_user_timeout</c>, <c>requirepeer</c>,
///     <c>replication</c>, OAuth, …) raises <see cref="NotSupportedException" /> naming it, instead of being dropped:
///     dropping it would connect to a different server or with weaker security than Python. Parameters whose libpq
///     value Npgsql already honors by default are accepted as no-ops (<c>sslsni=1</c>, <c>gssencmode=disable</c>,
///     <c>sslcertmode=allow</c>, <c>min_protocol_version=3.0</c>, an <c>ssl_min_protocol_version</c> at or below
///     TLS 1.2, which .NET never goes below).</para>
///     <para><b>Defaults that differ by design</b> (documented divergences): with no host anywhere, libpq uses a
///     Unix-domain socket on Linux and macOS — <c>/var/run/postgresql</c> when that directory exists, else
///     <c>/tmp</c>, which covers the Debian and upstream builds — and <c>localhost</c> on Windows; Npgsql's own
///     15-second connection timeout applies when the URL sets no <c>connect_timeout</c> (libpq waits for the
///     operating system). TCP keepalives are on, libpq's default.</para>
/// </remarks>
internal static class PostgresUrl
{
    /// <summary>The libpq 18 connection keywords (<c>PQconndefaults()</c>) with their environment variables.</summary>
    private static readonly OrderedDict<string, string?> LibpqKeywords = BuildKeywords();

    /// <summary>Builds the keyword table, in libpq's order.</summary>
    /// <returns>The keywords mapped to their environment variable, or <see langword="null" /> when they have none.</returns>
    private static OrderedDict<string, string?> BuildKeywords()
    {
        var k = new OrderedDict<string, string?>();
        foreach ((string Key, string? Env) entry in new (string, string?)[]
                 {
                     ("service", "PGSERVICE"), ("user", "PGUSER"), ("password", "PGPASSWORD"), ("passfile", "PGPASSFILE"),
                     ("channel_binding", "PGCHANNELBINDING"), ("connect_timeout", "PGCONNECT_TIMEOUT"),
                     ("dbname", "PGDATABASE"), ("host", "PGHOST"), ("hostaddr", "PGHOSTADDR"), ("port", "PGPORT"),
                     ("client_encoding", "PGCLIENTENCODING"), ("options", "PGOPTIONS"), ("application_name", "PGAPPNAME"),
                     ("fallback_application_name", null), ("keepalives", null), ("keepalives_idle", null),
                     ("keepalives_interval", null), ("keepalives_count", null), ("tcp_user_timeout", null),
                     ("sslmode", "PGSSLMODE"), ("sslnegotiation", "PGSSLNEGOTIATION"), ("sslcompression", "PGSSLCOMPRESSION"),
                     ("sslcert", "PGSSLCERT"), ("sslkey", "PGSSLKEY"), ("sslcertmode", "PGSSLCERTMODE"), ("sslpassword", null),
                     ("sslrootcert", "PGSSLROOTCERT"), ("sslcrl", "PGSSLCRL"), ("sslcrldir", "PGSSLCRLDIR"), ("sslsni", "PGSSLSNI"),
                     ("requirepeer", "PGREQUIREPEER"), ("require_auth", "PGREQUIREAUTH"),
                     ("min_protocol_version", "PGMINPROTOCOLVERSION"), ("max_protocol_version", "PGMAXPROTOCOLVERSION"),
                     ("ssl_min_protocol_version", "PGSSLMINPROTOCOLVERSION"), ("ssl_max_protocol_version", "PGSSLMAXPROTOCOLVERSION"),
                     ("gssencmode", "PGGSSENCMODE"), ("krbsrvname", "PGKRBSRVNAME"), ("gsslib", "PGGSSLIB"),
                     ("gssdelegation", "PGGSSDELEGATION"), ("replication", null), ("target_session_attrs", "PGTARGETSESSIONATTRS"),
                     ("load_balance_hosts", "PGLOADBALANCEHOSTS"), ("scram_client_key", null), ("scram_server_key", null),
                     ("oauth_issuer", null), ("oauth_client_id", null), ("oauth_client_secret", null), ("oauth_scope", null),
                     ("sslkeylogfile", null),
                 })
            k[entry.Key] = entry.Env;
        return k;
    }

    /// <summary>
    ///     The Npgsql connection string for a PostgreSQL URL.
    /// </summary>
    /// <param name="prepared">The URL after template filling, parsing and driver lookup.</param>
    /// <returns>The connection string builder for the database the URL names.</returns>
    /// <exception cref="ArgumentException">SQLAlchemy's multi-host errors (<c>number of hosts and ports don't match</c>, …).</exception>
    /// <exception cref="ValueError">A URL port that is not an integer, or Python's unpacking error for malformed multi-host lists.</exception>
    /// <exception cref="TypeError">psycopg2's <c>database</c>/<c>dbname</c> clash, or pg8000 without a user.</exception>
    /// <exception cref="FormatException">libpq refuses the connection info or a value; libpq's message.</exception>
    /// <exception cref="NotSupportedException">A parameter Npgsql cannot express.</exception>
    /// <exception cref="InvalidOperationException">An asyncio driver (<c>+asyncpg</c>, <c>+psycopg_async</c>).</exception>
    internal static NpgsqlConnectionStringBuilder Build(PreparedStorageUrl prepared)
    {
        SqlAlchemyUrl url = prepared.Url;
        switch (prepared.Driver.Driver)
        {
            case "pg8000":
                return FromPg8000(Pg8000ConnectArgs(url));
            case "asyncpg":
                AsyncpgConnectArgs(url);
                throw new InvalidOperationException(SqlAlchemyRegistry.MissingGreenletMessage);
            default:
                return FromLibpq(LibpqParameters(prepared));
        }
    }

    /// <summary>
    ///     The libpq parameters a psycopg-family URL sets — what Python's <c>psycopg2.extensions.parse_dsn</c> returns
    ///     for the connection string psycopg2 builds — before environment defaults and value checks apply.
    /// </summary>
    /// <param name="prepared">A URL whose driver is <c>psycopg2</c>, <c>psycopg2cffi</c>, <c>psycopg</c> or <c>psycopg_async</c>.</param>
    /// <returns>The parameters, in connection-string order.</returns>
    /// <exception cref="ArgumentException">A SQLAlchemy multi-host error.</exception>
    /// <exception cref="ValueError">Python's unpacking error for malformed multi-host lists.</exception>
    /// <exception cref="TypeError">psycopg2's <c>database</c>/<c>dbname</c> clash.</exception>
    /// <exception cref="FormatException">libpq refuses the connection string.</exception>
    /// <exception cref="NotSupportedException">A driver-level argument.</exception>
    /// <exception cref="InvalidOperationException">The asyncio driver <c>psycopg_async</c>.</exception>
    internal static OrderedDict<string, string> LibpqParameters(PreparedStorageUrl prepared)
    {
        OrderedDict<string, object?> opts = PsycopgConnectArgs(prepared.Url);
        if (prepared.Driver.IsAsync)
            throw new InvalidOperationException(SqlAlchemyRegistry.MissingGreenletMessage);
        return ConnectionInfo(opts, psycopg3: prepared.Driver.Driver is "psycopg");
    }

    // ------------------------------------------------------------------ SQLAlchemy create_connect_args

    /// <summary><c>_PGDialect_common_psycopg.create_connect_args</c> (psycopg2, psycopg2cffi, psycopg).</summary>
    /// <param name="url">The URL without plugin keys.</param>
    /// <returns>The keyword arguments SQLAlchemy passes to the driver's <c>connect</c>.</returns>
    /// <exception cref="ArgumentException">A multi-host error.</exception>
    /// <exception cref="ValueError">Python's unpacking error for malformed multi-host lists.</exception>
    private static OrderedDict<string, object?> PsycopgConnectArgs(SqlAlchemyUrl url)
    {
        OrderedDict<string, object?> opts = Translate(url, new Dictionary<string, string> { ["username"] = "user", ["database"] = "dbname" });
        (string?[]? hosts, BigInteger?[]? ports) = SplitMultihost(url);
        if (opts.Count > 0 || url.Query.Count > 0)
        {
            foreach (KeyValuePair<string, object> pair in url.Query)
                opts[pair.Key] = pair.Value;
            if (hosts is { Length: > 0 })
            {
                opts["host"] = string.Join(",", hosts);
                string commaPorts = string.Join(",", ports!.Select(p => p is { IsZero: false } v ? v.ToString(CultureInfo.InvariantCulture) : ""));
                if (commaPorts.Length > 0)
                    opts["port"] = commaPorts;
            }
        }
        return opts;
    }

    /// <summary><c>PGDialect_pg8000.create_connect_args</c>.</summary>
    /// <param name="url">The URL without plugin keys.</param>
    /// <returns>The keyword arguments for <c>pg8000.connect</c>.</returns>
    private static OrderedDict<string, object?> Pg8000ConnectArgs(SqlAlchemyUrl url)
    {
        OrderedDict<string, object?> opts = Translate(url, new Dictionary<string, string> { ["username"] = "user" });
        foreach (KeyValuePair<string, object> pair in url.Query)
            opts[pair.Key] = pair.Value;
        return opts;
    }

    /// <summary>
    ///     <c>PGDialect_asyncpg.create_connect_args</c>, run only for its errors: the driver then fails as every
    ///     asyncio driver does under Optuna's synchronous engine.
    /// </summary>
    /// <param name="url">The URL without plugin keys.</param>
    /// <exception cref="ArgumentException">A multi-host error, including asyncpg's own two.</exception>
    /// <exception cref="ValueError">A port or <c>prepared_statement_cache_size</c> that is not an integer.</exception>
    private static void AsyncpgConnectArgs(SqlAlchemyUrl url)
    {
        OrderedDict<string, object?> opts = Translate(url, new Dictionary<string, string> { ["username"] = "user" });
        (string?[]? hosts, BigInteger?[]? ports) = SplitMultihost(url);
        foreach (KeyValuePair<string, object> pair in url.Query)
            opts[pair.Key] = pair.Value;
        if (hosts is { Length: > 1 })
        {
            if (hosts.Any(h => string.IsNullOrEmpty(h)))
                throw new ArgumentException("All hosts are required to be present for asyncpg multiple host URL");
            if (ports!.Any(p => p is null || p.Value.IsZero))
                throw new ArgumentException("All ports are required to be present for asyncpg multiple host URL");
        }
        else if (hosts is null)
        {
            PyValues.CoerceKwType(opts, "port", "int");
        }
        PyValues.CoerceKwType(opts, "prepared_statement_cache_size", "int");
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

    // The single-host "h:p" form SQLAlchemy recognizes (rfc1123 host characters and digits). Python's \d and .NET's
    // \d both match any Unicode decimal digit, and $ matches before a final newline in both.
    private static readonly Regex HostPortPattern = new(@"^([a-zA-Z0-9\-\.]*)(?:\:(\d*))?$", RegexOptions.CultureInvariant);

    /// <summary>
    ///     <c>PGDialect._split_multihost_from_url</c>: the hosts and ports of <c>?host=</c>/<c>?port=</c>, in either of
    ///     SQLAlchemy's two multi-host spellings.
    /// </summary>
    /// <param name="url">The URL.</param>
    /// <returns>The hosts and their ports (<see langword="null" /> entries for absent ports), or two nulls without <c>?host=</c>.</returns>
    /// <exception cref="ArgumentException">Mixed formats, non-integer ports, or a host/port count mismatch; SQLAlchemy's messages.</exception>
    /// <exception cref="ValueError">A repeated <c>host</c> whose tokens split into other than two parts everywhere (Python's unpacking error).</exception>
    private static (string?[]? Hosts, BigInteger?[]? Ports) SplitMultihost(SqlAlchemyUrl url)
    {
        string?[]? hosts = null;
        string?[]? portsText = null;
        bool integrated = false;
        if (url.Query.TryGetValue("host", out object? hostValue))
        {
            if (hostValue is string[] tokens)
            {
                integrated = true;
                // zip(*[token.split(":") if ":" in token else (token, None) for token in tokens]) transposes the
                // tokens and stops at the shortest; unpacking into (hosts, ports) needs exactly two rows.
                List<string?[]> rows = tokens.Select(t => t.Contains(':') ? t.Split(':').Cast<string?>().ToArray() : new[] { t, null }).ToList();
                int width = rows.Min(r => r.Length);
                if (width != 2)
                    throw new ValueError(width > 2 ? "too many values to unpack (expected 2)" : $"not enough values to unpack (expected 2, got {width})");
                hosts = rows.Select(r => r[0]).ToArray();
                portsText = rows.Select(r => r[1]).ToArray();
            }
            else if (hostValue is string single)
            {
                hosts = single.Split(',');
                if (!url.Query.ContainsKey("port") && hosts.Length == 1 && hosts[0]!.Contains(':'))
                {
                    Match m = HostPortPattern.Match(hosts[0]!);
                    if (m.Success)
                    {
                        integrated = true;
                        hosts = new[] { m.Groups[1].Value };
                        string p = m.Groups[2].Success ? m.Groups[2].Value : "";
                        portsText = new[] { p.Length > 0 ? p : null };
                    }
                }
            }
        }
        if (url.Query.TryGetValue("port", out object? portValue))
        {
            if (integrated)
                throw new ArgumentException(
                    "Can't mix 'multihost' formats together; use \"host=h1,h2,h3&port=p1,p2,p3\" or " +
                    "\"host=h1:p1&host=h2:p2&host=h3:p3\" separately");
            portsText = portValue is string[] list ? list : ((string)portValue).Split(',');
        }

        BigInteger?[]? ports = null;
        if (portsText is { Length: > 0 })
        {
            try
            {
                ports = portsText.Select(x => string.IsNullOrEmpty(x) ? (BigInteger?)null : PyNumbers.ParseInt(x)).ToArray();
            }
            catch (ValueError)
            {
                throw new ArgumentException($"Received non-integer port arguments: {PythonRepr.TupleRepr(portsText)}");
            }
        }
        if (ports is { Length: > 0 }
            && ((hosts is not { Length: > 0 } && ports.Length > 1)
                || (hosts is { Length: > 0 } && hosts.Length != ports.Length && (hosts.Length > 1 || ports.Length > 1))))
            throw new ArgumentException("number of hosts and ports don't match");
        if (hosts is not null && ports is null)
            ports = new BigInteger?[hosts.Length];
        return (hosts, ports);
    }

    // ------------------------------------------------------------------ psycopg2 / psycopg: connect → conninfo

    /// <summary>
    ///     <c>psycopg2.connect(**opts)</c> or <c>psycopg.connect(**opts)</c> up to libpq: driver-level arguments are
    ///     taken out, the rest become a libpq connection string exactly as the driver writes it, and libpq parses it.
    /// </summary>
    /// <param name="opts">SQLAlchemy's connection arguments.</param>
    /// <param name="psycopg3">Whether the driver is psycopg 3 (no <c>database</c> alias, its own driver-level arguments).</param>
    /// <returns>The libpq parameters the connection string sets, in order.</returns>
    /// <exception cref="TypeError">psycopg2's <c>database</c>/<c>dbname</c> clash.</exception>
    /// <exception cref="FormatException">libpq refuses the connection string; libpq's message after the driver's prefix.</exception>
    /// <exception cref="NotSupportedException">A driver-level argument (<c>connection_factory</c>, <c>autocommit</c>, …), which a URL cannot meaningfully carry.</exception>
    private static OrderedDict<string, string> ConnectionInfo(OrderedDict<string, object?> opts, bool psycopg3)
    {
        string prefix = psycopg3 ? "" : "invalid dsn: ";
        string[] driverLevel = psycopg3
            ? new[] { "autocommit", "prepare_threshold", "context", "row_factory", "cursor_factory" }
            : new[] { "connection_factory", "cursor_factory", "async", "async_" };
        foreach (string key in driverLevel)
        {
            if (opts.ContainsKey(key))
                throw new NotSupportedException(
                    $"The query argument '{key}' configures the {(psycopg3 ? "psycopg" : "psycopg2")} driver object itself, " +
                    "which has no counterpart in Npgsql; remove it from the storage URL.");
        }

        // connect(dsn=...) / connect(conninfo=...): a connection string given as a keyword is parsed and then
        // overridden by the other keywords, as make_dsn / make_conninfo do.
        string dsnKey = psycopg3 ? "conninfo" : "dsn";
        OrderedDict<string, string>? baseInfo = null;
        if (opts.TryGetValue(dsnKey, out object? dsnValue))
        {
            opts.Remove(dsnKey);
            string text = PyValues.Str(dsnValue);
            if (text.StartsWith("postgresql://", StringComparison.Ordinal) || text.StartsWith("postgres://", StringComparison.Ordinal))
                throw new NotSupportedException(
                    $"A libpq connection URI in the '{dsnKey}' query argument is not supported; write its parameters as query arguments instead.");
            baseInfo = text.Length > 0 ? ParseConnectionInfo(text, prefix) : new OrderedDict<string, string>();
        }

        if (!psycopg3 && opts.ContainsKey("database"))
        {
            if (opts.ContainsKey("dbname"))
                throw new TypeError("you can't specify both 'database' and 'dbname' arguments");
            object? database = opts["database"];
            opts.Remove("database");
            opts["dbname"] = database;
        }

        var parts = new List<KeyValuePair<string, string>>();
        if (baseInfo is not null)
        {
            foreach (KeyValuePair<string, string> pair in baseInfo)
            {
                if (!opts.ContainsKey(pair.Key))
                    parts.Add(pair);
            }
        }
        foreach (KeyValuePair<string, object?> pair in opts)
        {
            if (pair.Value is not null)
                parts.Add(new KeyValuePair<string, string>(pair.Key, PyValues.Str(pair.Value)));
        }
        if (parts.Count == 0)
            return new OrderedDict<string, string>();
        string conninfo = string.Join(" ", parts.Select(p => p.Key + "=" + ParamEscape(p.Value)));
        return ParseConnectionInfo(conninfo, prefix);
    }

    /// <summary>
    ///     psycopg2's <c>_param_escape</c>: an empty value is <c>''</c>; <c>'</c> and <c>\</c> are backslash-escaped;
    ///     a value containing whitespace is quoted.
    /// </summary>
    /// <param name="value">The value's <c>str()</c>.</param>
    /// <returns>The value as it appears in the connection string.</returns>
    private static string ParamEscape(string value)
    {
        if (value.Length == 0)
            return "''";
        var sb = new StringBuilder(value.Length + 2);
        foreach (char c in value)
        {
            if (c is '\'' or '\\')
                sb.Append('\\');
            sb.Append(c);
        }
        string escaped = sb.ToString();
        // Python's \s on a str is Unicode whitespace: Python's str.isspace set.
        return escaped.Any(c => char.IsWhiteSpace(c) || c is >= '\u001C' and <= '\u001F') ? "'" + escaped + "'" : escaped;
    }

    /// <summary>
    ///     libpq's <c>conninfo_parse</c>: <c>keyword = value</c> pairs separated by whitespace, values optionally
    ///     single-quoted, <c>\</c> escaping the next character, later keywords overriding earlier ones, and every
    ///     keyword checked against libpq's list (<c>requiressl</c> still maps to <c>sslmode</c>).
    /// </summary>
    /// <param name="conninfo">The connection string.</param>
    /// <param name="prefix">The driver's prefix for parse errors (<c>invalid dsn: </c> for psycopg2).</param>
    /// <returns>The parameters, in the order of their first appearance.</returns>
    /// <exception cref="FormatException">A missing <c>=</c>, an unterminated quote or an unknown keyword; libpq's message.</exception>
    internal static OrderedDict<string, string> ParseConnectionInfo(string conninfo, string prefix = "invalid dsn: ")
    {
        // libpq works on the UTF-8 bytes, with C isspace(); non-ASCII bytes are never spaces, so chars work too.
        static bool IsSpace(char c) => c is ' ' or '\t' or '\n' or '\v' or '\f' or '\r';
        var result = new OrderedDict<string, string>();
        int i = 0;
        while (i < conninfo.Length)
        {
            if (IsSpace(conninfo[i]))
            {
                i++;
                continue;
            }
            int nameStart = i;
            while (i < conninfo.Length && conninfo[i] != '=' && !IsSpace(conninfo[i]))
                i++;
            string name = conninfo.Substring(nameStart, i - nameStart);
            while (i < conninfo.Length && IsSpace(conninfo[i]))
                i++;
            if (i >= conninfo.Length || conninfo[i] != '=')
                throw new FormatException($"{prefix}missing \"=\" after \"{name}\" in connection info string");
            i++;
            while (i < conninfo.Length && IsSpace(conninfo[i]))
                i++;

            var value = new StringBuilder();
            if (i < conninfo.Length && conninfo[i] == '\'')
            {
                i++;
                while (true)
                {
                    if (i >= conninfo.Length)
                        throw new FormatException($"{prefix}unterminated quoted string in connection info string");
                    char c = conninfo[i];
                    if (c == '\\')
                    {
                        i++;
                        if (i < conninfo.Length)
                            value.Append(conninfo[i++]);
                        continue;
                    }
                    if (c == '\'')
                    {
                        i++;
                        break;
                    }
                    value.Append(c);
                    i++;
                }
            }
            else
            {
                while (i < conninfo.Length && !IsSpace(conninfo[i]))
                {
                    if (conninfo[i] == '\\')
                    {
                        i++;
                        if (i < conninfo.Length)
                            value.Append(conninfo[i++]);
                    }
                    else
                    {
                        value.Append(conninfo[i++]);
                    }
                }
            }

            // conninfo_storeval: the pre-7.4 spelling survives as an alias.
            string text = value.ToString();
            if (name == "requiressl")
            {
                name = "sslmode";
                text = text.StartsWith('1') ? "require" : "prefer";
            }
            if (!LibpqKeywords.ContainsKey(name))
                throw new FormatException($"{prefix}invalid connection option \"{name}\"");
            result[name] = text;
        }
        return result;
    }

    // ------------------------------------------------------------------ libpq → Npgsql

    /// <summary>
    ///     Applies libpq's semantics to parsed connection parameters — environment defaults, value validation, host
    ///     and port lists — and expresses the result with Npgsql.
    /// </summary>
    /// <param name="given">The parameters the connection string sets.</param>
    /// <returns>The Npgsql connection string builder.</returns>
    /// <exception cref="FormatException">A value libpq refuses; libpq's message.</exception>
    /// <exception cref="NotSupportedException">A parameter Npgsql cannot express.</exception>
    internal static NpgsqlConnectionStringBuilder FromLibpq(OrderedDict<string, string> given)
    {
        // conninfo_add_defaults: a parameter the connection string leaves out takes its PG* variable, if set.
        var p = new Dictionary<string, string>(StringComparer.Ordinal);
        foreach (KeyValuePair<string, string?> keyword in LibpqKeywords)
        {
            if (given.TryGetValue(keyword.Key, out string? value))
                p[keyword.Key] = value;
            else if (keyword.Value is not null && Environment.GetEnvironmentVariable(keyword.Value) is { } env)
                p[keyword.Key] = env;
        }
        string? Get(string key) => p.TryGetValue(key, out string? v) && v.Length > 0 ? v : null;

        foreach (string key in new[]
                 {
                     "service", "hostaddr", "keepalives_count", "tcp_user_timeout", "sslcrl", "sslcrldir", "requirepeer",
                     "require_auth", "gsslib", "replication", "scram_client_key", "scram_server_key", "oauth_issuer",
                     "oauth_client_id", "oauth_client_secret", "oauth_scope", "sslkeylogfile",
                 })
        {
            if (Get(key) is not null)
                throw Unsupported(key, p[key]);
        }

        var b = new NpgsqlConnectionStringBuilder();

        // pqConnectOptions2's checks, in its order.
        string? channelBinding = Get("channel_binding");
        if (channelBinding is not null)
            b.ChannelBinding = channelBinding switch
            {
                "disable" => ChannelBinding.Disable,
                "prefer" => ChannelBinding.Prefer,
                "require" => ChannelBinding.Require,
                _ => throw Invalid("channel_binding", channelBinding),
            };
        string sslmode = Get("sslmode") ?? "prefer";
        b.SslMode = sslmode switch
        {
            "disable" => SslMode.Disable,
            "allow" => SslMode.Allow,
            "prefer" => SslMode.Prefer,
            "require" => SslMode.Require,
            "verify-ca" => SslMode.VerifyCA,
            "verify-full" => SslMode.VerifyFull,
            _ => throw Invalid("sslmode", sslmode),
        };
        string? negotiation = Get("sslnegotiation");
        if (negotiation is not null)
        {
            if (negotiation is not ("postgres" or "direct"))
                throw Invalid("sslnegotiation", negotiation);
            if (negotiation == "direct" && sslmode is "disable" or "allow" or "prefer")
                throw new FormatException(
                    $"weak sslmode \"{sslmode}\" may not be used with sslnegotiation=direct (use \"require\", \"verify-ca\", or \"verify-full\")");
#if NET8_0
            // Npgsql 8 (the net8.0 build) predates PostgreSQL 17's direct TLS handshake; "postgres" is its only mode.
            if (negotiation == "direct")
                throw Unsupported("sslnegotiation", negotiation);
#else
            b.SslNegotiation = negotiation == "direct" ? SslNegotiation.Direct : SslNegotiation.Postgres;
#endif
        }
        string? rootCert = Get("sslrootcert");
        if (rootCert == "system")
        {
            // libpq 16+: the system CA store, and nothing weaker than verify-full.
            if (sslmode != "verify-full")
                throw new FormatException($"weak sslmode \"{sslmode}\" may not be used with sslrootcert=system (use \"verify-full\")");
        }
        else if (rootCert is not null)
        {
            b.RootCertificate = rootCert;
        }
        string? certMode = Get("sslcertmode");
        if (certMode is not null && certMode is not ("disable" or "allow" or "require"))
            throw Invalid("sslcertmode", certMode);
        if (certMode is "disable" or "require")
            throw Unsupported("sslcertmode", certMode);
        string? gss = Get("gssencmode");
        if (gss is not null && gss is not ("disable" or "prefer" or "require"))
            throw Invalid("gssencmode", gss);
        if (gss == "require")
            throw Unsupported("gssencmode", gss);
        string? tsa = Get("target_session_attrs");
        if (tsa is not null)
        {
            if (tsa is not ("any" or "read-write" or "read-only" or "primary" or "standby" or "prefer-standby"))
                throw Invalid("target_session_attrs", tsa);
            b.TargetSessionAttributes = tsa;
        }
        string? loadBalance = Get("load_balance_hosts");
        if (loadBalance is not null)
            b.LoadBalanceHosts = loadBalance switch
            {
                "disable" => false,
                "random" => true,
                _ => throw Invalid("load_balance_hosts", loadBalance),
            };
        CheckProtocol("min_protocol_version", Get("min_protocol_version"), allowed: "3.0");
        CheckProtocol("max_protocol_version", Get("max_protocol_version"), allowed: null);
        string? sslMin = Get("ssl_min_protocol_version");
        string? sslMax = Get("ssl_max_protocol_version");
        int TlsRank(string key, string? v) => v switch
        {
            null => -1,
            "TLSv1" => 0,
            "TLSv1.1" => 1,
            "TLSv1.2" => 2,
            "TLSv1.3" => 3,
            _ => throw new FormatException($"invalid \"{key}\" value: \"{v}\""),
        };
        int minRank = TlsRank("ssl_min_protocol_version", sslMin);
        int maxRank = TlsRank("ssl_max_protocol_version", sslMax);
        if (minRank >= 0 && maxRank >= 0 && minRank > maxRank)
            throw new FormatException("invalid SSL protocol version range");
        // .NET negotiates TLS 1.2 or later by OS policy, so a minimum up to 1.2 and a maximum of 1.3 already hold.
        if (minRank == 3)
            throw Unsupported("ssl_min_protocol_version", sslMin!);
        if (maxRank is >= 0 and < 3)
            throw Unsupported("ssl_max_protocol_version", sslMax!);

        (string Host, int Port)[] endpoints = Endpoints(Get("host"), p.TryGetValue("port", out string? portText) ? portText : null);
        if (endpoints.Length == 1)
        {
            b.Host = endpoints[0].Host;
            b.Port = endpoints[0].Port;
        }
        else
        {
            b.Host = string.Join(",", endpoints.Select(e => (e.Host.Contains(':') && !Path.IsPathRooted(e.Host) ? "[" + e.Host + "]" : e.Host)
                                                           + ":" + e.Port.ToString(CultureInfo.InvariantCulture)));
        }

        if (Get("dbname") is { } dbname)
            b.Database = dbname;
        if (Get("user") is { } user)
            b.Username = user;
        if (Get("password") is { } password)
            b.Password = password;
        if (Get("passfile") is { } passfile)
            b.Passfile = passfile;
        if (Get("options") is { } options)
            b.Options = options;
        string? application = Get("application_name") ?? Get("fallback_application_name");
        if (application is not null)
            b.ApplicationName = application;
        if (Get("client_encoding") is { } encoding
            && !encoding.Equals("UTF8", StringComparison.OrdinalIgnoreCase) && !encoding.Equals("UTF-8", StringComparison.OrdinalIgnoreCase)
            && !encoding.Equals("UNICODE", StringComparison.OrdinalIgnoreCase))
            throw Unsupported("client_encoding", encoding);
        if (Get("sslcert") is { } cert)
            b.SslCertificate = cert;
        if (Get("sslkey") is { } sslKey)
            b.SslKey = sslKey;
        if (Get("sslpassword") is { } keyPassword)
            b.SslPassword = keyPassword;
        if (Get("krbsrvname") is { } kerberos)
            b.KerberosServiceName = kerberos;
        if (Get("sslsni") is { } sni && !sni.StartsWith('1'))
            throw Unsupported("sslsni", sni);
        if (Get("gssdelegation") is { } delegation && delegation != "0")
            throw Unsupported("gssdelegation", delegation);
        // sslcompression is accepted and ignored: TLS compression is disabled in every modern TLS stack, libpq's too.

        if (Get("connect_timeout") is { } timeoutText)
        {
            int timeout = ParseIntParam(timeoutText, "connect_timeout");
            // libpq: zero or negative waits forever, and 1 is raised to 2. Npgsql's limit is 1024 s; longer client-side
            // limits never fire anyway because the operating system's TCP connect timeout is shorter.
            b.Timeout = timeout <= 0 ? 0 : timeout == 1 ? 2 : timeout > 1024 ? 0 : timeout;
        }
        b.TcpKeepAlive = Get("keepalives") is not { } keepalives || ParseIntParam(keepalives, "keepalives") != 0;
        if (Get("keepalives_idle") is { } idle)
            b.TcpKeepAliveTime = ParseIntParam(idle, "keepalives_idle");
        if (Get("keepalives_interval") is { } interval)
            b.TcpKeepAliveInterval = ParseIntParam(interval, "keepalives_interval");
        return b;
    }

    /// <summary>
    ///     libpq's host and port lists: comma-separated, one port for every host or one port for all, empty entries
    ///     meaning the defaults, each port an integer in 1–65535.
    /// </summary>
    /// <param name="host">The <c>host</c> parameter, or <see langword="null" />.</param>
    /// <param name="port">The <c>port</c> parameter, or <see langword="null" />.</param>
    /// <returns>The endpoints to try, in order.</returns>
    /// <exception cref="FormatException">A count mismatch or an invalid port; libpq's messages.</exception>
    private static (string Host, int Port)[] Endpoints(string? host, string? port)
    {
        string[] hosts = (host ?? "").Split(',');
        string[] ports = string.IsNullOrEmpty(port) ? new[] { "" } : port.Split(',');
        if (ports.Length != 1 && ports.Length != hosts.Length)
            throw new FormatException($"could not match {ports.Length} port numbers to {hosts.Length} hosts");
        var endpoints = new (string Host, int Port)[hosts.Length];
        for (int i = 0; i < hosts.Length; i++)
        {
            string text = ports.Length == 1 ? ports[0] : ports[i];
            int number = 5432;
            if (text.Length > 0)
            {
                number = ParseIntParam(text, "port");
                if (number is < 1 or > 65535)
                    throw new FormatException($"invalid port number: \"{text}\"");
            }
            endpoints[i] = (hosts[i].Length > 0 ? hosts[i] : DefaultHost(), number);
        }
        return endpoints;
    }

    /// <summary>
    ///     libpq's default host: <c>localhost</c> on Windows; elsewhere the Unix-domain socket directory —
    ///     <c>/var/run/postgresql</c> (Debian, Ubuntu, RHEL packages) when it exists, else <c>/tmp</c> (upstream builds).
    /// </summary>
    /// <returns>The host Npgsql should use.</returns>
    private static string DefaultHost()
        => OperatingSystem.IsWindows() ? "localhost" : Directory.Exists("/var/run/postgresql") ? "/var/run/postgresql" : "/tmp";

    /// <summary>
    ///     libpq's <c>parse_int_param</c>: <c>strtol</c> in base 10 (leading whitespace and a sign allowed), then only
    ///     trailing whitespace, and the value must fit a C <c>int</c>.
    /// </summary>
    /// <param name="value">The parameter's text.</param>
    /// <param name="context">The parameter's name, for the message.</param>
    /// <returns>The integer.</returns>
    /// <exception cref="FormatException">The text is not such an integer: <c>invalid integer value "…" for connection option "…"</c>.</exception>
    private static int ParseIntParam(string value, string context)
    {
        Match m = Regex.Match(value, @"\A[ \t\n\v\f\r]*([+-]?[0-9]+)[ \t\n\v\f\r]*\z", RegexOptions.CultureInvariant);
        if (!m.Success || !int.TryParse(m.Groups[1].Value, NumberStyles.AllowLeadingSign, CultureInfo.InvariantCulture, out int result))
            throw new FormatException($"invalid integer value \"{value}\" for connection option \"{context}\"");
        return result;
    }

    /// <summary>Validates <c>min_protocol_version</c>/<c>max_protocol_version</c> (libpq 18) and refuses a minimum above 3.0.</summary>
    /// <param name="key">The parameter name.</param>
    /// <param name="value">Its value, or <see langword="null" />.</param>
    /// <param name="allowed">The only minimum Npgsql meets, or <see langword="null" /> for a maximum (any valid one holds).</param>
    /// <exception cref="FormatException">Not one of <c>3.0</c>, <c>3.2</c>, <c>latest</c>.</exception>
    /// <exception cref="NotSupportedException">A minimum above protocol 3.0, which Npgsql does not speak.</exception>
    private static void CheckProtocol(string key, string? value, string? allowed)
    {
        if (value is null)
            return;
        if (value is not ("3.0" or "3.2" or "latest"))
            throw Invalid(key, value);
        if (allowed is not null && value != allowed)
            throw Unsupported(key, value);
    }

    /// <summary>libpq's message for an invalid enumerated value: <c>invalid key value: "value"</c>.</summary>
    /// <param name="key">The parameter.</param>
    /// <param name="value">The refused value.</param>
    /// <returns>The exception to throw.</returns>
    private static FormatException Invalid(string key, string value) => new($"invalid {key} value: \"{value}\"");

    /// <summary>The refusal for a libpq parameter Npgsql cannot express.</summary>
    /// <param name="key">The parameter.</param>
    /// <param name="value">Its value.</param>
    /// <returns>The exception to throw.</returns>
    private static NotSupportedException Unsupported(string key, string value)
        => new($"The libpq connection parameter {key}={PythonRepr.StringRepr(value)} has no Npgsql counterpart, so " +
               "OptunaSharp cannot connect the way libpq would; remove it from the storage URL (or the PG environment " +
               "variable that sets it), or configure Npgsql directly with EFCoreRdbOptions.ConfigureConnectionString.");

    // ------------------------------------------------------------------ pg8000 → Npgsql

    /// <summary>
    ///     <c>pg8000.connect(**opts)</c> expressed with Npgsql: the URL components, <c>application_name</c> and a
    ///     <c>unix_sock</c> path; pg8000 reads no environment variables and turns TCP keepalives on by default.
    /// </summary>
    /// <param name="opts">SQLAlchemy's pg8000 arguments.</param>
    /// <returns>The Npgsql connection string builder.</returns>
    /// <exception cref="TypeError">No user: pg8000's <c>connect()</c> requires one.</exception>
    /// <exception cref="NotSupportedException">Any other pg8000 argument; pg8000's keywords take Python objects a URL cannot express faithfully.</exception>
    private static NpgsqlConnectionStringBuilder FromPg8000(OrderedDict<string, object?> opts)
    {
        if (!opts.ContainsKey("user"))
            throw new TypeError("connect() missing 1 required positional argument: 'user'");
        var b = new NpgsqlConnectionStringBuilder { Host = "localhost", Port = 5432, TcpKeepAlive = true };
        foreach (KeyValuePair<string, object?> pair in opts)
        {
            string text = PyValues.Str(pair.Value);
            switch (pair.Key)
            {
                case "user":
                    b.Username = text;
                    break;
                case "password":
                    b.Password = text;
                    break;
                case "host":
                    b.Host = text;
                    break;
                case "port":
                    b.Port = pair.Value is BigInteger n ? (int)n : ParseIntParam(text, "port");
                    break;
                case "database":
                    b.Database = text;
                    break;
                case "application_name":
                    b.ApplicationName = text;
                    break;
                case "unix_sock":
                    // pg8000 takes the socket file; Npgsql takes its directory and derives ".s.PGSQL.<port>".
                    Match m = Regex.Match(text, @"\A(?<dir>.*)[/\\]\.s\.PGSQL\.(?<port>[0-9]+)\z", RegexOptions.CultureInvariant);
                    if (!m.Success)
                        throw new NotSupportedException(
                            $"The pg8000 unix_sock {PythonRepr.StringRepr(text)} is not a PostgreSQL socket file (<dir>/.s.PGSQL.<port>), which is the form Npgsql can open.");
                    b.Host = m.Groups["dir"].Value;
                    b.Port = int.Parse(m.Groups["port"].Value, CultureInfo.InvariantCulture);
                    break;
                default:
                    throw new NotSupportedException(
                        $"The pg8000 connection argument '{pair.Key}' has no OptunaSharp counterpart; use postgresql:// " +
                        "(libpq parameters) or EFCoreRdbOptions.ConfigureConnectionString instead.");
            }
        }
        return b;
    }
}

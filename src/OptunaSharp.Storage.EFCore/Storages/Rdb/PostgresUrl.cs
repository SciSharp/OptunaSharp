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
///     <para><b>Connection strings and service files.</b> A <c>?dsn=</c> (psycopg2) or <c>?conninfo=</c> (psycopg)
///     argument may be in libpq's keyword form or its URI form (<c>postgresql://…</c>), merged with the other
///     arguments as the driver merges them. <c>?service=name</c> or <c>PGSERVICE</c> reads the group from
///     <c>pg_service.conf</c> the way libpq does (see <see cref="LibpqService" />), filling in what the URL leaves
///     out, with libpq's errors for a missing or malformed file. <c>PGREQUIRESSL</c> and the
///     <c>sslrootcert=system</c> default of <c>sslmode=verify-full</c> apply too.</para>
///     <para><b>What Npgsql cannot express.</b> A libpq parameter with no Npgsql counterpart (<c>hostaddr</c>,
///     <c>require_auth</c>, <c>sslcrl</c>, <c>keepalives_count</c>, <c>tcp_user_timeout</c>, <c>requirepeer</c>,
///     <c>replication</c>, the SCRAM keys, OAuth, …) raises <see cref="NotSupportedException" /> naming it, instead of
///     being dropped: dropping it would connect to a different server or with weaker security than Python. libpq's
///     own validation runs first, so a value libpq refuses fails with libpq's message, not with the refusal.
///     Parameters whose libpq value Npgsql already honors by default are accepted as no-ops (<c>sslsni=1</c>,
///     <c>gssencmode=disable</c>, <c>sslcertmode=allow</c>, <c>min_protocol_version=3.0</c>, an
///     <c>ssl_min_protocol_version</c> at or below TLS 1.2, which .NET never goes below, and any UTF-8 spelling of
///     <c>client_encoding</c>).</para>
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
    /// <exception cref="ValueError">
    ///     A URL port that is not an integer, Python's unpacking error for malformed multi-host lists, psycopg2's
    ///     <c>embedded null character</c>, or a <see cref="UnicodeDecodeError" /> for a URI escape that is not UTF-8.
    /// </exception>
    /// <exception cref="TypeError">psycopg2's <c>database</c>/<c>dbname</c> clash, a repeated <c>?dsn=</c>, or pg8000 without a user.</exception>
    /// <exception cref="AttributeError">A repeated <c>?conninfo=</c> for psycopg 3.</exception>
    /// <exception cref="FormatException">libpq refuses the connection info, a value or the service file; libpq's message.</exception>
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
    /// <returns>The parameters, in libpq's option order.</returns>
    /// <exception cref="ArgumentException">A SQLAlchemy multi-host error.</exception>
    /// <exception cref="ValueError">
    ///     Python's unpacking error for malformed multi-host lists, psycopg2's <c>embedded null character</c>, or a
    ///     <see cref="UnicodeDecodeError" /> for a URI escape that is not UTF-8.
    /// </exception>
    /// <exception cref="TypeError">psycopg2's <c>database</c>/<c>dbname</c> clash, or a repeated <c>?dsn=</c>.</exception>
    /// <exception cref="AttributeError">A repeated <c>?conninfo=</c> for psycopg 3.</exception>
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
    /// <returns>The libpq parameters the connection string sets, in libpq's option order (as <c>parse_dsn</c> lists them).</returns>
    /// <exception cref="TypeError">
    ///     psycopg2's <c>database</c>/<c>dbname</c> clash, or a repeated <c>?dsn=</c> (a tuple, which psycopg2's
    ///     <c>parse_dsn</c> refuses).
    /// </exception>
    /// <exception cref="AttributeError">A repeated <c>?conninfo=</c> for psycopg 3, which calls <c>.encode()</c> on it.</exception>
    /// <exception cref="FormatException">libpq refuses the connection string; libpq's message after the driver's prefix.</exception>
    /// <exception cref="UnicodeDecodeError">A percent-escape in a connection URI decodes to bytes that are not UTF-8.</exception>
    /// <exception cref="ValueError">psycopg2 and a NUL anywhere in the connection string (<c>embedded null character</c>), once the part before it has parsed.</exception>
    /// <exception cref="NotSupportedException">A driver-level argument (<c>connection_factory</c>, <c>autocommit</c>, …), which a URL cannot meaningfully carry.</exception>
    /// <remarks>
    ///     <c>make_dsn</c> merges like a Python <c>dict.update</c>: a keyword the dsn already sets keeps the dsn's
    ///     position with the keyword's value, other keywords follow in order. The order matters because libpq reads the
    ///     joined string as a C string: with psycopg 3 a NUL in one value silently drops every parameter after it, and
    ///     with psycopg2 the part before the NUL is what <c>make_dsn</c> validates before <c>connect</c> refuses the NUL.
    /// </remarks>
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

        // connect(dsn=...) / connect(conninfo=...): a connection string given as a keyword, in libpq's keyword or URI
        // form, parsed and then overridden by the other keywords, as make_dsn / make_conninfo do.
        string dsnKey = psycopg3 ? "conninfo" : "dsn";
        string? dsn = null;
        if (opts.TryGetValue(dsnKey, out object? dsnValue))
        {
            opts.Remove(dsnKey);
            // A repeated key is a tuple: psycopg2's parse_dsn wants text, psycopg calls conninfo.encode().
            if (dsnValue is string[])
                throw psycopg3
                    ? new AttributeError("'tuple' object has no attribute 'encode'")
                    : new TypeError("Expected bytes or unicode string, got tuple instead");
            dsn = PyValues.Str(dsnValue);
        }

        // make_dsn verifies with parse_dsn, which reads a C string and so stops at a NUL; psycopg2's _connect then takes
        // the whole dsn with the "s" argument format, which refuses the NUL. psycopg hands libpq the bytes, which end there.
        OrderedDict<string, string> Verified(string text)
        {
            OrderedDict<string, string> parsed = ParseConnectionInfo(text, prefix);
            if (!psycopg3 && text.Contains('\0'))
                throw new ValueError("embedded null character");
            return parsed;
        }

        // Without other keywords the dsn is only verified and then used as it is: a URI reaches libpq as a URI.
        if (opts.Count == 0)
            return string.IsNullOrEmpty(dsn) ? new OrderedDict<string, string>() : Verified(dsn);

        if (!psycopg3 && opts.ContainsKey("database"))
        {
            if (opts.ContainsKey("dbname"))
                throw new TypeError("you can't specify both 'database' and 'dbname' arguments");
            object? database = opts["database"];
            opts.Remove("database");
            opts["dbname"] = database;
        }

        // tmp = parse_dsn(dsn); tmp.update(kwargs): OrderedDict assignment keeps an existing key's position, as a
        // Python dict does, and appends a new one.
        var merged = new OrderedDict<string, string>();
        if (!string.IsNullOrEmpty(dsn))
        {
            foreach (KeyValuePair<string, string> pair in ParseConnectionInfo(dsn, prefix))
                merged[pair.Key] = pair.Value;
        }
        foreach (KeyValuePair<string, object?> pair in opts)
        {
            if (pair.Value is not null)
                merged[pair.Key] = PyValues.Str(pair.Value);
        }
        if (merged.Count == 0)
            return new OrderedDict<string, string>();
        string conninfo = string.Join(" ", merged.Select(p => p.Key + "=" + ParamEscape(p.Value)));
        return Verified(conninfo);
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
    ///     libpq's <c>PQconninfoParse</c> as psycopg2's <c>parse_dsn</c> exposes it: the connection string is read as a
    ///     C string, in the URI grammar when it starts with <c>postgresql://</c> or <c>postgres://</c> and in the
    ///     <c>keyword=value</c> grammar otherwise.
    /// </summary>
    /// <param name="conninfo">The connection string.</param>
    /// <param name="prefix">The driver's prefix for parse errors (<c>invalid dsn: </c> for psycopg2).</param>
    /// <returns>The parameters the string sets, in libpq's option order, which is the order <c>parse_dsn</c> returns.</returns>
    /// <exception cref="FormatException">The string does not parse; libpq's message after the prefix.</exception>
    /// <exception cref="UnicodeDecodeError">A URI percent-escape decodes to bytes that are not UTF-8, which psycopg2 refuses as it builds the result.</exception>
    /// <remarks>
    ///     The string reaches libpq as a C string, so a NUL ends it here, as in <c>parse_dsn</c>. Footgun, as in
    ///     Python: with psycopg 3 a value such as <c>options=%00…</c> in the URL silently drops that value's remainder
    ///     and every parameter after it (psycopg2 refuses the NUL afterwards, see <c>ConnectionInfo</c>).
    /// </remarks>
    internal static OrderedDict<string, string> ParseConnectionInfo(string conninfo, string prefix = "invalid dsn: ")
    {
        int nul = conninfo.IndexOf('\0');
        if (nul >= 0)
            conninfo = conninfo.Substring(0, nul);
        // uri_prefix_length: the prefixes are case-sensitive; "POSTGRESQL://h" is a keyword string missing its "=".
        bool uri = conninfo.StartsWith("postgresql://", StringComparison.Ordinal) || conninfo.StartsWith("postgres://", StringComparison.Ordinal);
        var result = new OrderedDict<string, string>();
        if (uri)
        {
            Dictionary<string, byte[]> raw = ParseConnectionUri(conninfo, prefix);
            // psycopg2 decodes each value as UTF-8 while it builds the dict, in libpq's order, after libpq has parsed
            // the whole URI: so a syntax error anywhere wins over a bad percent-escape.
            foreach (string key in LibpqKeywords.Keys)
            {
                if (raw.TryGetValue(key, out byte[]? bytes))
                    result[key] = PyUtf8.DecodeStrict(bytes);
            }
            return result;
        }
        OrderedDict<string, string> found = ParseKeywordValue(conninfo, prefix);
        foreach (string key in LibpqKeywords.Keys)
        {
            if (found.TryGetValue(key, out string? value))
                result[key] = value;
        }
        return result;
    }

    /// <summary>
    ///     libpq's <c>conninfo_parse</c>: <c>keyword = value</c> pairs separated by whitespace, values optionally
    ///     single-quoted, <c>\</c> escaping the next character, later keywords overriding earlier ones, and every
    ///     keyword checked against libpq's list (<c>requiressl</c> still maps to <c>sslmode</c>).
    /// </summary>
    /// <param name="conninfo">The connection string, without a NUL.</param>
    /// <param name="prefix">The driver's prefix for parse errors (<c>invalid dsn: </c> for psycopg2).</param>
    /// <returns>The parameters, in the order of their first appearance.</returns>
    /// <exception cref="FormatException">A missing <c>=</c>, an unterminated quote or an unknown keyword; libpq's message.</exception>
    private static OrderedDict<string, string> ParseKeywordValue(string conninfo, string prefix)
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

    /// <summary>
    ///     libpq's <c>conninfo_uri_parse_options</c>:
    ///     <c>postgresql://[user[:password]@][host[:port][,host[:port]…]][/dbname][?key=value&amp;…]</c>, with IPv6 hosts
    ///     in brackets, percent-decoding, and the query keys checked against libpq's list (<c>ssl=true</c> is JDBC's
    ///     spelling of <c>sslmode=require</c>).
    /// </summary>
    /// <param name="uri">The URI, without a NUL.</param>
    /// <param name="prefix">The driver's prefix for parse errors.</param>
    /// <returns>The decoded values as bytes, which libpq never validates as UTF-8.</returns>
    /// <exception cref="FormatException">A malformed URI; libpq's message after the prefix.</exception>
    /// <remarks>
    ///     libpq works on the UTF-8 bytes in place, so this port does too: positions in messages are byte offsets, and
    ///     a percent-escape may produce any byte but NUL. The user stops at the first <c>:</c>, the password at the
    ///     first <c>@</c>; the look-ahead for <c>@</c> stops only at <c>/</c>, so an <c>@</c> in the query still marks
    ///     credentials, exactly as in libpq. Unlike urllib, a <c>#</c> is not special: it belongs to the database name.
    /// </remarks>
    private static Dictionary<string, byte[]> ParseConnectionUri(string uri, string prefix)
    {
        byte[] u = Encoding.UTF8.GetBytes(uri);
        var options = new Dictionary<string, byte[]>(StringComparer.Ordinal);
        int start = uri.StartsWith("postgresql://", StringComparison.Ordinal) ? 13 : 11;
        int p = start;
        while (p < u.Length && u[p] != '@' && u[p] != '/')
            p++;
        if (p < u.Length && u[p] == '@')
        {
            // user[:password]@ — the '@' found above is reached before the end, so these loops stay in bounds.
            int user = start;
            p = user;
            while (u[p] != ':' && u[p] != '@')
                p++;
            byte terminator = u[p];
            if (p > user)
                StoreUriComponent(options, "user", u, user, p, prefix);
            if (terminator == ':')
            {
                int password = p + 1;
                while (u[p] != '@')
                    p++;
                if (p > password)
                    StoreUriComponent(options, "password", u, password, p, prefix);
            }
            p++;
        }
        else
        {
            p = start;
        }

        // Every netloc[:port] pair, separated by commas, accumulated as libpq's host and port lists.
        var hosts = new List<byte>();
        var ports = new List<byte>();
        int prevchar;
        while (true)
        {
            int host;
            int hostEnd;
            if (p < u.Length && u[p] == '[')
            {
                host = ++p;
                while (p < u.Length && u[p] != ']')
                    p++;
                if (p >= u.Length)
                    throw new FormatException($"{prefix}end of string reached when looking for matching \"]\" in IPv6 host address in URI: \"{uri}\"");
                if (p == host)
                    throw new FormatException($"{prefix}IPv6 host address may not be empty in URI: \"{uri}\"");
                hostEnd = p++;
                if (p < u.Length && u[p] is not ((byte)':' or (byte)'/' or (byte)'?' or (byte)','))
                {
                    // %c prints the one byte; psycopg2 decodes the message with replacement, so a non-ASCII byte is U+FFFD.
                    char shown = u[p] < 0x80 ? (char)u[p] : '\uFFFD';
                    throw new FormatException(
                        $"{prefix}unexpected character \"{shown}\" at position {p + 1} in URI (expected \":\" or \"/\"): \"{uri}\"");
                }
            }
            else
            {
                host = p;
                while (p < u.Length && u[p] is not ((byte)':' or (byte)'/' or (byte)'?' or (byte)','))
                    p++;
                hostEnd = p;
            }
            hosts.AddRange(u.AsSpan(host, hostEnd - host).ToArray());
            prevchar = p < u.Length ? u[p] : 0;
            if (prevchar == ':')
            {
                int port = ++p;
                while (p < u.Length && u[p] is not ((byte)'/' or (byte)'?' or (byte)','))
                    p++;
                ports.AddRange(u.AsSpan(port, p - port).ToArray());
                prevchar = p < u.Length ? u[p] : 0;
            }
            if (prevchar != ',')
                break;
            p++;
            hosts.Add((byte)',');
            ports.Add((byte)',');
        }
        byte[] hostBytes = hosts.ToArray();
        byte[] portBytes = ports.ToArray();
        if (hostBytes.Length > 0)
            StoreUriComponent(options, "host", hostBytes, 0, hostBytes.Length, prefix);
        if (portBytes.Length > 0)
            StoreUriComponent(options, "port", portBytes, 0, portBytes.Length, prefix);

        if (prevchar != 0 && prevchar != '?')
        {
            // "/dbname": an empty name is not stored, so PGDATABASE (or the user name) still applies.
            int dbname = ++p;
            while (p < u.Length && u[p] != '?')
                p++;
            prevchar = p < u.Length ? u[p] : 0;
            if (p > dbname)
                StoreUriComponent(options, "dbname", u, dbname, p, prefix);
        }
        if (prevchar != 0)
            ParseUriParameters(u, p + 1, options, prefix);
        return options;
    }

    /// <summary>
    ///     <c>conninfo_uri_parse_params</c>: <c>key=value</c> pairs separated by <c>&amp;</c>, each part
    ///     percent-decoded, an unknown key refused (<c>invalid URI query parameter</c>) rather than ignored.
    /// </summary>
    /// <param name="u">The URI's bytes.</param>
    /// <param name="start">The index just past the <c>?</c>.</param>
    /// <param name="options">Where to store the values.</param>
    /// <param name="prefix">The driver's prefix for errors.</param>
    /// <exception cref="FormatException">A pair without <c>=</c> or with two, a bad escape, or an unknown key; libpq's messages.</exception>
    private static void ParseUriParameters(byte[] u, int start, Dictionary<string, byte[]> options, string prefix)
    {
        int i = start;
        while (i < u.Length)
        {
            int keyword = i;
            int keywordEnd = -1;
            int value = -1;
            int valueEnd;
            int q = i;
            while (true)
            {
                if (q < u.Length && u[q] == '=')
                {
                    // The first '=' was already cut off, so the message quotes the keyword alone.
                    if (value >= 0)
                        throw new FormatException($"{prefix}extra key/value separator \"=\" in URI query parameter: \"{Text(u, keyword, keywordEnd)}\"");
                    keywordEnd = q++;
                    value = q;
                }
                else if (q >= u.Length || u[q] == '&')
                {
                    int end = q;
                    if (q < u.Length)
                        q++;
                    if (value < 0)
                        throw new FormatException($"{prefix}missing key/value separator \"=\" in URI query parameter: \"{Text(u, keyword, end)}\"");
                    valueEnd = end;
                    break;
                }
                else
                {
                    q++;
                }
            }

            string key = Text(UriDecode(u, keyword, keywordEnd, prefix));
            byte[] decoded = UriDecode(u, value, valueEnd, prefix);
            // JDBC compatibility: ssl=true is sslmode=require (and nothing else about "ssl" is accepted).
            if (key == "ssl" && Text(decoded) == "true")
            {
                key = "sslmode";
                decoded = Encoding.ASCII.GetBytes("require");
            }
            // conninfo_storeval: requiressl maps to sslmode, then the key must be a libpq keyword.
            if (key == "requiressl")
            {
                key = "sslmode";
                decoded = Encoding.ASCII.GetBytes(decoded.Length > 0 && decoded[0] == '1' ? "require" : "prefer");
            }
            if (!LibpqKeywords.ContainsKey(key))
                throw new FormatException($"{prefix}invalid URI query parameter: \"{key}\"");
            options[key] = decoded;
            i = q;
        }
    }

    /// <summary>
    ///     <c>conninfo_storeval</c> for a URI component (user, password, host, port, dbname): percent-decoded as it is
    ///     stored, replacing any earlier value.
    /// </summary>
    /// <param name="options">Where to store it.</param>
    /// <param name="keyword">The libpq keyword.</param>
    /// <param name="bytes">The bytes holding the encoded component.</param>
    /// <param name="start">The component's first byte.</param>
    /// <param name="end">One past its last byte.</param>
    /// <param name="prefix">The driver's prefix for errors.</param>
    /// <exception cref="FormatException">A bad escape, <c>%00</c>, or an unescaped space inside the component.</exception>
    private static void StoreUriComponent(Dictionary<string, byte[]> options, string keyword, byte[] bytes, int start, int end, string prefix)
        => options[keyword] = UriDecode(bytes, start, end, prefix);

    /// <summary>
    ///     libpq's <c>conninfo_uri_decode</c>: leading and trailing spaces are dropped, <c>%XX</c> becomes its byte
    ///     (never NUL), and a space anywhere else ends the value, which is an error.
    /// </summary>
    /// <param name="u">The bytes.</param>
    /// <param name="start">The encoded value's first byte.</param>
    /// <param name="end">One past its last byte.</param>
    /// <param name="prefix">The driver's prefix for errors.</param>
    /// <returns>The decoded bytes.</returns>
    /// <exception cref="FormatException">
    ///     <c>invalid percent-encoded token</c>, <c>forbidden value %00 in percent-encoded value</c>, or
    ///     <c>unexpected spaces found in "…", use percent-encoded spaces (%20) instead</c>; each quotes the encoded value.
    /// </exception>
    private static byte[] UriDecode(byte[] u, int start, int end, string prefix)
    {
        var decoded = new List<byte>(end - start);
        int q = start;
        while (q < end && u[q] == ' ')
            q++;
        while (q < end && u[q] != ' ')
        {
            if (u[q] != '%')
            {
                decoded.Add(u[q++]);
                continue;
            }
            q++;
            // get_hexdigit(*q++) && get_hexdigit(*q++): the end of the value is not a hex digit.
            int hi = q < end ? HexDigit(u[q]) : -1;
            q++;
            int lo = hi >= 0 && q < end ? HexDigit(u[q]) : -1;
            q++;
            if (hi < 0 || lo < 0)
                throw new FormatException($"{prefix}invalid percent-encoded token: \"{Text(u, start, end)}\"");
            int c = (hi << 4) | lo;
            if (c == 0)
                throw new FormatException($"{prefix}forbidden value %00 in percent-encoded value: \"{Text(u, start, end)}\"");
            decoded.Add((byte)c);
        }
        while (q < end && u[q] == ' ')
            q++;
        if (q < end)
            throw new FormatException($"{prefix}unexpected spaces found in \"{Text(u, start, end)}\", use percent-encoded spaces (%20) instead");
        return decoded.ToArray();
    }

    /// <summary>libpq's <c>get_hexdigit</c>.</summary>
    /// <param name="c">The byte.</param>
    /// <returns>Its value, or -1 when it is not <c>0-9</c>, <c>a-f</c> or <c>A-F</c>.</returns>
    private static int HexDigit(byte c) => c switch
    {
        >= (byte)'0' and <= (byte)'9' => c - '0',
        >= (byte)'a' and <= (byte)'f' => c - 'a' + 10,
        >= (byte)'A' and <= (byte)'F' => c - 'A' + 10,
        _ => -1,
    };

    /// <summary>Bytes of the URI as text for a message or a key (they came from a .NET string, so they are UTF-8).</summary>
    /// <param name="u">The bytes.</param>
    /// <param name="start">The first byte.</param>
    /// <param name="end">One past the last byte.</param>
    /// <returns>The text, with U+FFFD for bytes a percent-escape made invalid.</returns>
    private static string Text(byte[] u, int start, int end) => Encoding.UTF8.GetString(u, start, end - start);

    /// <summary>Decoded bytes as text (see <see cref="Text(byte[], int, int)" />).</summary>
    /// <param name="u">The bytes.</param>
    /// <returns>The text.</returns>
    private static string Text(byte[] u) => Encoding.UTF8.GetString(u);

    // ------------------------------------------------------------------ libpq → Npgsql

    /// <summary>
    ///     Applies libpq's semantics to parsed connection parameters — a service-file group, environment defaults,
    ///     value validation, host and port lists — and expresses the result with Npgsql.
    /// </summary>
    /// <param name="given">The parameters the connection string sets.</param>
    /// <returns>The Npgsql connection string builder.</returns>
    /// <exception cref="FormatException">A service file or a value libpq refuses; libpq's message.</exception>
    /// <exception cref="NotSupportedException">A parameter Npgsql cannot express, refused only after libpq's own checks pass.</exception>
    /// <remarks>
    ///     The order is libpq's: <c>conninfo_add_defaults</c> (the service group, then the <c>PG*</c> variables, then
    ///     the compiled-in defaults), then <c>pqConnectOptions2</c>'s checks in the order libpq 18 runs them (measured
    ///     pairwise against psycopg2: the port count, <c>require_auth</c>, <c>channel_binding</c>, <c>sslmode</c>,
    ///     <c>sslnegotiation</c>, the TLS versions and their range, <c>sslcertmode</c>, <c>gssencmode</c>,
    ///     <c>target_session_attrs</c>, the SCRAM keys, <c>load_balance_hosts</c>, the protocol versions). A refusal of
    ///     what Npgsql cannot express waits until all of them have passed, so a URL Python rejects fails here with
    ///     Python's error rather than with OptunaSharp's.
    /// </remarks>
    internal static NpgsqlConnectionStringBuilder FromLibpq(OrderedDict<string, string> given)
    {
        var p = new Dictionary<string, string>(StringComparer.Ordinal);
        foreach (KeyValuePair<string, string> pair in given)
            p[pair.Key] = pair.Value;
        // conninfo_add_defaults: the service group fills what the string leaves out, then the PG* variables do.
        foreach (KeyValuePair<string, string> pair in LibpqService.Options(given, LibpqKeywords.Keys))
            p.TryAdd(pair.Key, pair.Value);
        bool sslmodeDefaulted = false;
        foreach (KeyValuePair<string, string?> keyword in LibpqKeywords)
        {
            if (p.ContainsKey(keyword.Key))
                continue;
            if (keyword.Value is not null && Environment.GetEnvironmentVariable(keyword.Value) is { } env)
                p[keyword.Key] = env;
            else if (keyword.Key == "sslmode")
            {
                // The pre-9.3 PGREQUIRESSL: a value starting with 1 means sslmode=require; PGSSLMODE wins over it.
                if (Environment.GetEnvironmentVariable("PGREQUIRESSL") is { } requireSsl && requireSsl.StartsWith('1'))
                    p["sslmode"] = "require";
                else
                    sslmodeDefaulted = true;
            }
        }
        // The compiled-in sslmode is "prefer", which libpq raises to verify-full for sslrootcert=system.
        if (sslmodeDefaulted)
            p["sslmode"] = p.TryGetValue("sslrootcert", out string? root) && root == "system" ? "verify-full" : "prefer";
        // Most parameters treat an empty value as unset; the enumerated ones validate it, so an explicit empty value
        // (PGSSLMODE exported empty, sslmode='' in a dsn) is refused as libpq refuses it. Raw keeps it; Get drops it.
        string? Get(string key) => p.TryGetValue(key, out string? v) && v.Length > 0 ? v : null;
        string? Raw(string key) => p.TryGetValue(key, out string? v) ? v : null;
        NotSupportedException? refusal = null;
        void Refuse(NotSupportedException e) => refusal ??= e;

        foreach (string key in new[]
                 {
                     "hostaddr", "keepalives_count", "tcp_user_timeout", "sslcrl", "sslcrldir", "requirepeer", "gsslib",
                     "replication", "oauth_issuer", "oauth_client_id", "oauth_client_secret", "oauth_scope", "sslkeylogfile",
                 })
        {
            if (Get(key) is not null)
                Refuse(Unsupported(key, p[key]));
        }

        var b = new NpgsqlConnectionStringBuilder();

        // pqConnectOptions2's checks, in its order. First the host and port lists must pair up.
        CheckPortCount(Get("host"), p.TryGetValue("port", out string? portText) ? portText : null);
        if (Get("require_auth") is { } requireAuth)
        {
            CheckRequireAuth(requireAuth);
            // Npgsql has no way to restrict which authentication requests it answers; dropping the restriction would
            // let the client accept weaker authentication than Python allows.
            Refuse(Unsupported("require_auth", requireAuth));
        }
        string? channelBinding = Raw("channel_binding");
        if (channelBinding is not null)
            b.ChannelBinding = channelBinding switch
            {
                "disable" => ChannelBinding.Disable,
                "prefer" => ChannelBinding.Prefer,
                "require" => ChannelBinding.Require,
                _ => throw Invalid("channel_binding", channelBinding),
            };
        string sslmode = Raw("sslmode")!;
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
        string? negotiation = Raw("sslnegotiation");
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
                Refuse(Unsupported("sslnegotiation", negotiation));
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
        // ssl_min_protocol_version has a compiled-in default, TLSv1.2, so a lone maximum below it is an invalid range.
        CheckTlsRange(p.TryGetValue("ssl_min_protocol_version", out string? sslMin) ? sslMin : "TLSv1.2",
            Raw("ssl_max_protocol_version"), Refuse);
        string? certMode = Raw("sslcertmode");
        if (certMode is not null && certMode is not ("disable" or "allow" or "require"))
            throw Invalid("sslcertmode", certMode);
        if (certMode is "disable" or "require")
            Refuse(Unsupported("sslcertmode", certMode));
        string? gss = Raw("gssencmode");
        if (gss is not null && gss is not ("disable" or "prefer" or "require"))
            throw Invalid("gssencmode", gss);
        if (gss == "require")
            Refuse(Unsupported("gssencmode", gss));
        string? tsa = Raw("target_session_attrs");
        if (tsa is not null)
        {
            if (tsa is not ("any" or "read-write" or "read-only" or "primary" or "standby" or "prefer-standby"))
                throw Invalid("target_session_attrs", tsa);
            b.TargetSessionAttributes = tsa;
        }
        foreach (string side in new[] { "client", "server" })
        {
            // Validated as libpq validates them, then refused: Npgsql cannot take precomputed SCRAM keys.
            if (Raw($"scram_{side}_key") is { } scramKey)
            {
                CheckScramKey(scramKey, side);
                Refuse(Unsupported($"scram_{side}_key", scramKey));
            }
        }
        string? loadBalance = Raw("load_balance_hosts");
        if (loadBalance is not null)
            b.LoadBalanceHosts = loadBalance switch
            {
                "disable" => false,
                "random" => true,
                _ => throw Invalid("load_balance_hosts", loadBalance),
            };
        CheckProtocolRange(Raw("min_protocol_version"), Raw("max_protocol_version"), Refuse);

        (string Host, int Port)[] endpoints = Endpoints(Get("host"), portText);
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
        // The server reads client_encoding through PostgreSQL's clean_encoding_name, so utf_8, Utf-8 and unicode all
        // name UTF-8 — the one encoding Npgsql speaks.
        if (Get("client_encoding") is { } encoding && CleanEncodingName(encoding) is not ("utf8" or "unicode"))
            Refuse(Unsupported("client_encoding", encoding));
        if (Get("sslcert") is { } cert)
            b.SslCertificate = cert;
        if (Get("sslkey") is { } sslKey)
            b.SslKey = sslKey;
        if (Get("sslpassword") is { } keyPassword)
            b.SslPassword = keyPassword;
        if (Get("krbsrvname") is { } kerberos)
            b.KerberosServiceName = kerberos;
        if (Get("sslsni") is { } sni && !sni.StartsWith('1'))
            Refuse(Unsupported("sslsni", sni));
        if (Get("gssdelegation") is { } delegation && delegation != "0")
            Refuse(Unsupported("gssdelegation", delegation));
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
        // Every libpq check has passed: only now does a parameter Npgsql cannot express stop the connection.
        if (refusal is not null)
            throw refusal;
        return b;
    }

    /// <summary>
    ///     PostgreSQL's <c>clean_encoding_name</c>, which the server applies to <c>client_encoding</c>: every
    ///     character that is not an ASCII letter or digit is dropped and ASCII letters are lower-cased.
    /// </summary>
    /// <param name="name">The encoding name.</param>
    /// <returns>The cleaned name, for example <c>utf8</c> for <c>UTF-8</c> or <c>utf_8</c>.</returns>
    private static string CleanEncodingName(string name)
    {
        var sb = new StringBuilder(name.Length);
        foreach (char c in name)
        {
            // isalnum((unsigned char) byte) in the C locale: ASCII only, so non-ASCII text (whose UTF-8 bytes are all
            // above 0x7F) vanishes entirely.
            if (c is >= 'a' and <= 'z' or >= '0' and <= '9')
                sb.Append(c);
            else if (c is >= 'A' and <= 'Z')
                sb.Append((char)(c + ('a' - 'A')));
        }
        return sb.ToString();
    }

    /// <summary>
    ///     The first check of <c>pqConnectOptions2</c>: one port for all hosts, or one port per host.
    /// </summary>
    /// <param name="host">The <c>host</c> parameter, or <see langword="null" />.</param>
    /// <param name="port">The <c>port</c> parameter, or <see langword="null" />.</param>
    /// <exception cref="FormatException"><c>could not match N port numbers to M hosts</c>.</exception>
    private static void CheckPortCount(string? host, string? port)
    {
        int hosts = (host ?? "").Split(',').Length;
        int ports = string.IsNullOrEmpty(port) ? 1 : port.Split(',').Length;
        if (ports != 1 && ports != hosts)
            throw new FormatException($"could not match {ports} port numbers to {hosts} hosts");
    }

    /// <summary>
    ///     libpq's parsing of <c>require_auth</c> (PostgreSQL 16+): a comma-separated list of <c>password</c>,
    ///     <c>md5</c>, <c>gss</c>, <c>sspi</c>, <c>scram-sha-256</c>, <c>oauth</c> and <c>none</c>, either all
    ///     negated with <c>!</c> or none of them, each at most once. Nothing is trimmed and names are case-sensitive.
    /// </summary>
    /// <param name="value">The non-empty parameter value.</param>
    /// <exception cref="FormatException">
    ///     libpq's messages: <c>invalid require_auth value: "…"</c>, the two mixing errors, and
    ///     <c>require_auth method "…" is specified more than once</c>.
    /// </exception>
    private static void CheckRequireAuth(string value)
    {
        // The request codes behind each method: gss and sspi share GSS_CONT, so "gss,sspi" is not a duplicate but
        // a second "gss" is (all of its bits are already set).
        const int Password = 1, Md5 = 2, Gss = 4, Sspi = 8, GssCont = 16;
        bool negated = false;
        bool authRequired = true;
        int allowed = 0;
        var mechanisms = new HashSet<string>(StringComparer.Ordinal);
        string[] parts = value.Split(',');
        for (int i = 0; i < parts.Length; i++)
        {
            string part = parts[i];
            string method = part;
            if (method.StartsWith('!'))
            {
                if (i == 0)
                {
                    // A negated list starts from everything allowed and subtracts.
                    authRequired = false;
                    allowed = Password | Md5 | Gss | Sspi | GssCont;
                    mechanisms.Add("scram-sha-256");
                    mechanisms.Add("oauth");
                }
                else if (!negated)
                {
                    throw new FormatException($"negative require_auth method \"{method}\" cannot be mixed with non-negative methods");
                }
                negated = true;
                method = method.Substring(1);
            }
            else if (negated)
            {
                throw new FormatException($"require_auth method \"{method}\" cannot be mixed with negative methods");
            }

            int bits = method switch
            {
                "password" => Password,
                "md5" => Md5,
                "gss" => Gss | GssCont,
                "sspi" => Sspi | GssCont,
                _ => 0,
            };
            bool duplicate;
            if (bits != 0)
            {
                duplicate = negated ? (allowed & bits) == 0 : (allowed & bits) == bits;
                allowed = negated ? allowed & ~bits : allowed | bits;
            }
            else if (method is "scram-sha-256" or "oauth")
            {
                // SASL mechanisms are tracked by name: removing an absent one or adding a present one repeats it.
                duplicate = negated ? !mechanisms.Remove(method) : !mechanisms.Add(method);
            }
            else if (method == "none")
            {
                // "none" allows connections without an authentication request; "!none" forbids them.
                duplicate = negated ? authRequired : !authRequired;
                authRequired = negated;
            }
            else
            {
                throw new FormatException($"invalid require_auth value: \"{method}\"");
            }
            if (duplicate)
                throw new FormatException($"require_auth method \"{part}\" is specified more than once");
        }
    }

    /// <summary>
    ///     libpq 18's check of <c>scram_client_key</c> / <c>scram_server_key</c>: strict base64 (PostgreSQL's
    ///     <c>pg_b64_decode</c>) of exactly 32 bytes.
    /// </summary>
    /// <param name="value">The parameter value, possibly empty.</param>
    /// <param name="side"><c>client</c> or <c>server</c>, for the message.</param>
    /// <exception cref="FormatException"><c>invalid SCRAM client key</c> or <c>invalid SCRAM client key length: N</c> (and the server variants).</exception>
    private static void CheckScramKey(string value, string side)
    {
        int length = DecodeBase64Length(value);
        if (length < 0)
            throw new FormatException($"invalid SCRAM {side} key");
        if (length != 32)
            throw new FormatException($"invalid SCRAM {side} key length: {length}");
    }

    /// <summary>
    ///     PostgreSQL's <c>pg_b64_decode</c>, reduced to the decoded length: whitespace and stray <c>=</c> are errors,
    ///     the input must end on a four-character boundary, and the output may not outgrow
    ///     <c>pg_b64_dec_len(len) = len * 3 / 4</c>.
    /// </summary>
    /// <param name="text">The base64 text.</param>
    /// <returns>The number of decoded bytes, or -1 for invalid input.</returns>
    private static int DecodeBase64Length(string text)
    {
        // The text reaches libpq as UTF-8 bytes; a byte outside ASCII is never a base64 symbol.
        byte[] src = Encoding.UTF8.GetBytes(text);
        int capacity = (src.Length * 3) >> 2;
        int produced = 0, pos = 0, end = 0;
        foreach (byte c in src)
        {
            if (c is (byte)' ' or (byte)'\t' or (byte)'\n' or (byte)'\r')
                return -1;
            if (c == '=')
            {
                // The first '=' fixes how many bytes the final group yields; it may only come third or fourth.
                if (end == 0)
                {
                    if (pos == 2)
                        end = 1;
                    else if (pos == 3)
                        end = 2;
                    else
                        return -1;
                }
            }
            else if (!(c is >= (byte)'A' and <= (byte)'Z' or >= (byte)'a' and <= (byte)'z' or >= (byte)'0' and <= (byte)'9' or (byte)'+' or (byte)'/'))
            {
                return -1;
            }
            if (++pos == 4)
            {
                int bytes = end == 0 ? 3 : end == 1 ? 1 : 2;
                if (produced + bytes > capacity)
                    return -1;
                produced += bytes;
                pos = 0;
            }
        }
        return pos != 0 ? -1 : produced;
    }

    /// <summary>
    ///     libpq's <c>sslVerifyProtocolVersion</c> and <c>sslVerifyProtocolRange</c>: each bound, when set and not
    ///     empty, is one of <c>TLSv1</c>, <c>TLSv1.1</c>, <c>TLSv1.2</c>, <c>TLSv1.3</c> in any ASCII case, and the
    ///     minimum is not above the maximum. Bounds .NET cannot enforce are refused afterwards.
    /// </summary>
    /// <param name="min">The minimum, with libpq's default <c>TLSv1.2</c> already applied when unset.</param>
    /// <param name="max">The maximum, or <see langword="null" />.</param>
    /// <param name="refuse">Records a refusal to raise once every libpq check has passed.</param>
    /// <exception cref="FormatException"><c>invalid "ssl_min_protocol_version" value: "…"</c> (or max), or <c>invalid SSL protocol version range</c>.</exception>
    private static void CheckTlsRange(string? min, string? max, Action<NotSupportedException> refuse)
    {
        static int Rank(string key, string? value)
        {
            if (string.IsNullOrEmpty(value))
                return -1;
            // pg_strcasecmp: ASCII letters fold, everything else compares as it is.
            for (int rank = 0; rank < TlsVersions.Length; rank++)
            {
                if (System.Text.Ascii.EqualsIgnoreCase(value, TlsVersions[rank]))
                    return rank;
            }
            throw new FormatException($"invalid \"{key}\" value: \"{value}\"");
        }
        int minRank = Rank("ssl_min_protocol_version", min);
        int maxRank = Rank("ssl_max_protocol_version", max);
        if (minRank >= 0 && maxRank >= 0 && minRank > 0 && (maxRank == 0 || minRank > maxRank))
            throw new FormatException("invalid SSL protocol version range");
        // .NET negotiates TLS 1.2 or later by OS policy, so a minimum up to 1.2 and a maximum of 1.3 already hold.
        if (minRank == 3)
            refuse(Unsupported("ssl_min_protocol_version", min!));
        if (maxRank is >= 0 and < 3)
            refuse(Unsupported("ssl_max_protocol_version", max!));
    }

    /// <summary>libpq's TLS protocol names, in rank order.</summary>
    private static readonly string[] TlsVersions = { "TLSv1", "TLSv1.1", "TLSv1.2", "TLSv1.3" };

    /// <summary>
    ///     libpq 18's <c>min_protocol_version</c> / <c>max_protocol_version</c>: each is exactly <c>3.0</c>,
    ///     <c>3.2</c> or <c>latest</c>; an unset maximum is 3.0, or the latest when the minimum is above 3.0; and the
    ///     minimum may not exceed the maximum. Npgsql speaks protocol 3.0 only, so a higher minimum is refused.
    /// </summary>
    /// <param name="min">The minimum, or <see langword="null" />.</param>
    /// <param name="max">The maximum, or <see langword="null" />.</param>
    /// <param name="refuse">Records a refusal to raise once every libpq check has passed.</param>
    /// <exception cref="FormatException"><c>invalid min_protocol_version value: "…"</c> (or max), or <c>"min_protocol_version" is greater than "max_protocol_version"</c>.</exception>
    private static void CheckProtocolRange(string? min, string? max, Action<NotSupportedException> refuse)
    {
        static int Version(string key, string value) => value switch
        {
            "3.0" => 30,
            "3.2" or "latest" => 32,
            _ => throw Invalid(key, value),
        };
        int minVersion = min is null ? 30 : Version("min_protocol_version", min);
        int maxVersion = max is null ? (minVersion > 30 ? 32 : 30) : Version("max_protocol_version", max);
        if (minVersion > maxVersion)
            throw new FormatException("\"min_protocol_version\" is greater than \"max_protocol_version\"");
        if (minVersion > 30)
            refuse(Unsupported("min_protocol_version", min!));
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

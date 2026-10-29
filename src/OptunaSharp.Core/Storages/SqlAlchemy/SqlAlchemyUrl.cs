using System;
using System.Collections.Generic;
using System.Globalization;
using System.Numerics;
using System.Text;
using System.Text.RegularExpressions;
using NumSharp;
using OptunaSharp.Collections;
using OptunaSharp.Internal;

namespace OptunaSharp.Storages.SqlAlchemy;

/// <summary>
///     Port of SQLAlchemy 2.0's <c>sqlalchemy.engine.URL</c> as <c>make_url(text)</c> builds it
///     (<c>sqlalchemy/engine/url.py: _parse_url</c>): the one grammar every Optuna RDB storage URL goes through.
///     The SQLite and EF Core storage packages, Core's URL resolver and the CLI all parse with this type, so a URL
///     means the same database in Python and in OptunaSharp.
/// </summary>
/// <remarks>
///     <para><b>The grammar, as SQLAlchemy has it.</b> <c>name://[user[:password]@][host|[ipv6]][:port][/database][?query]</c>,
///     matched from the start of the text and not required to consume all of it. Consequences that look odd but are
///     Python's:</para>
///     <list type="bullet">
///         <item>Only the user name and password are percent-decoded. The host and the database are taken as they
///         are, so <c>sqlite:///my%20db.sqlite</c> names a file containing <c>%20</c>.</item>
///         <item>The password stops at the first <c>@</c>: in <c>postgresql://u:p@x@h/db</c> the password is
///         <c>p</c> and the host is <c>x@h</c>. A literal <c>@</c> must be written <c>%40</c>.</item>
///         <item>An empty port (<c>host:/db</c>) or a non-numeric one is a Python <c>ValueError</c> from
///         <c>int()</c>; port <c>0</c> parses but is dropped when connection arguments are built.</item>
///         <item>The query is decoded with <c>parse_qsl</c>: <c>+</c> is a space, a key without a value is dropped,
///         and a repeated key becomes a tuple of its values. Because <c>.</c> does not match a line break, the query
///         ends at the first <c>\n</c> and the rest of the text is ignored; there is no <c>#fragment</c>.</item>
///     </list>
/// </remarks>
internal sealed class SqlAlchemyUrl
{
    // Everything after "name://", verbatim from _parse_url (re.X). Python's and .NET's backtracking engines agree on
    // this pattern: every class is ASCII or negated ASCII, and '.' excludes only '\n' in both.
    private static readonly Regex RestPattern = new(
        @"\A
          (?:
              (?<username>[^:/]*)
              (?::(?<password>[^@]*))?
          @)?
          (?:
              (?:
                  \[(?<ipv6host>[^/\?]+)\] |
                  (?<ipv4host>[^/:\?]+)
              )?
              (?::(?<port>[^/\?]*))?
          )?
          (?:/(?<database>[^\?]*))?
          (?:\?(?<query>.*))?",
        RegexOptions.IgnorePatternWhitespace | RegexOptions.CultureInvariant);

    /// <summary>The text before <c>://</c>: <c>dialect</c> or <c>dialect+driver</c>.</summary>
    internal string Drivername { get; }

    /// <summary>The percent-decoded user name, or <see langword="null" /> when the URL has no <c>@</c>.</summary>
    internal string? Username { get; }

    /// <summary>The percent-decoded password, or <see langword="null" /> when the user information has no <c>:</c>.</summary>
    internal string? Password { get; }

    /// <summary>The host as written (IPv6 without its brackets), or <see langword="null" /> when absent or empty.</summary>
    internal string? Host { get; }

    /// <summary>The port, unbounded like a Python <c>int</c>, or <see langword="null" /> when absent.</summary>
    internal BigInteger? Port { get; }

    /// <summary>The database (for SQLite the file path) as written, or <see langword="null" /> when the URL has no <c>/</c> after the host.</summary>
    internal string? Database { get; }

    /// <summary>
    ///     The query in order of first appearance: each value is a <see cref="string" />, or a
    ///     <see cref="string" /> array (a Python tuple) when the key was repeated.
    /// </summary>
    internal OrderedDict<string, object> Query { get; }

    /// <summary>Creates a URL from its components, as <c>URL.create</c> does.</summary>
    /// <param name="drivername">The dialect, optionally followed by <c>+driver</c>.</param>
    /// <param name="username">The decoded user name, or <see langword="null" />.</param>
    /// <param name="password">The decoded password, or <see langword="null" />.</param>
    /// <param name="host">The host, or <see langword="null" />.</param>
    /// <param name="port">The port, or <see langword="null" />.</param>
    /// <param name="database">The database, or <see langword="null" />.</param>
    /// <param name="query">The query, as described on <see cref="Query" />.</param>
    internal SqlAlchemyUrl(string drivername, string? username, string? password, string? host, BigInteger? port,
        string? database, OrderedDict<string, object> query)
    {
        Drivername = drivername;
        Username = username;
        Password = password;
        Host = host;
        Port = port;
        Database = database;
        Query = query;
    }

    /// <summary>
    ///     <c>sqlalchemy.engine.make_url(text)</c>.
    /// </summary>
    /// <param name="text">The URL, already passed through Optuna's template filling.</param>
    /// <returns>The parsed URL.</returns>
    /// <exception cref="ArgumentException">
    ///     The text does not start with <c>name://</c>; SQLAlchemy's <c>ArgumentError</c> message
    ///     <c>Could not parse SQLAlchemy URL from given URL string</c>.
    /// </exception>
    /// <exception cref="ValueError">The port is empty or not an integer (Python's <c>int()</c> message).</exception>
    internal static SqlAlchemyUrl Parse(string text)
    {
        // "(?P<name>[\w\+]+)://": Python's \w on a str is a letter, any Unicode number, or '_' — per code point,
        // so the scan walks runes (an astral letter is one character there, two UTF-16 units here).
        int nameEnd = 0;
        while (nameEnd < text.Length && Rune.TryGetRuneAt(text, nameEnd, out Rune rune)
               && (Rune.IsLetter(rune) || Rune.IsNumber(rune) || rune.Value is '_' or '+'))
            nameEnd += rune.Utf16SequenceLength;
        if (nameEnd == 0 || string.CompareOrdinal(text, nameEnd, "://", 0, 3) != 0)
            throw new ArgumentException("Could not parse SQLAlchemy URL from given URL string");

        Match m = RestPattern.Match(text.Substring(nameEnd + 3));
        // The pattern can always match the empty string, so this never fails; the check documents the contract.
        if (!m.Success)
            throw new ArgumentException("Could not parse SQLAlchemy URL from given URL string");

        var query = new OrderedDict<string, object>();
        if (m.Groups["query"].Success)
        {
            foreach (KeyValuePair<string, string> pair in PyUrllib.ParseQsl(m.Groups["query"].Value))
            {
                if (!query.TryGetValue(pair.Key, out object? existing))
                    query[pair.Key] = pair.Value;
                else if (existing is string single)
                    query[pair.Key] = new[] { single, pair.Value };
                else
                    query[pair.Key] = Append((string[])existing, pair.Value);
            }
        }

        string? username = m.Groups["username"].Success ? PyUrllib.Unquote(m.Groups["username"].Value) : null;
        string? password = m.Groups["password"].Success ? PyUrllib.Unquote(m.Groups["password"].Value) : null;
        // `ipv4host or ipv6host`: an empty match is falsy, so both empty means no host at all.
        string? host = m.Groups["ipv4host"].Success && m.Groups["ipv4host"].Length > 0 ? m.Groups["ipv4host"].Value
            : m.Groups["ipv6host"].Success && m.Groups["ipv6host"].Length > 0 ? m.Groups["ipv6host"].Value : null;
        // `if components["port"]: int(...)`, then URL.create's `int(port)`: a present port is always converted, so
        // an empty one fails exactly like a non-numeric one.
        BigInteger? port = m.Groups["port"].Success ? PyNumbers.ParseInt(m.Groups["port"].Value) : null;
        string? database = m.Groups["database"].Success ? m.Groups["database"].Value : null;
        return new SqlAlchemyUrl(text.Substring(0, nameEnd), username, password, host, port, database, query);
    }

    /// <summary>Copies an array and appends one value.</summary>
    /// <param name="values">The existing values.</param>
    /// <param name="value">The value to add.</param>
    /// <returns>The longer array.</returns>
    private static string[] Append(string[] values, string value)
    {
        var longer = new string[values.Length + 1];
        values.CopyTo(longer, 0);
        longer[^1] = value;
        return longer;
    }

    /// <summary><c>URL.get_backend_name()</c>: the dialect, the part of <see cref="Drivername" /> before the first <c>+</c>.</summary>
    /// <returns>The dialect name.</returns>
    internal string GetBackendName()
    {
        int plus = Drivername.IndexOf('+');
        return plus < 0 ? Drivername : Drivername.Substring(0, plus);
    }

    /// <summary>
    ///     <c>URL.difference_update_query(["plugin", "plugin_name"])</c>, which <c>create_engine</c> applies after
    ///     loading plugins: the same URL without those two query keys.
    /// </summary>
    /// <param name="names">The query keys to drop.</param>
    /// <returns>A URL whose query lacks the given keys.</returns>
    internal SqlAlchemyUrl WithoutQueryKeys(params string[] names)
    {
        var query = new OrderedDict<string, object>();
        foreach (KeyValuePair<string, object> pair in Query)
        {
            if (Array.IndexOf(names, pair.Key) < 0)
                query[pair.Key] = pair.Value;
        }
        return new SqlAlchemyUrl(Drivername, Username, Password, Host, Port, Database, query);
    }

    /// <summary>
    ///     <c>URL.translate_connect_args(**names)</c>: host, database, user name, password and port in that order, each
    ///     renamed through <paramref name="names" />, and each left out when it is falsy (empty text, port 0).
    /// </summary>
    /// <param name="names">Renames, keyed by SQLAlchemy's attribute names (<c>host</c>, <c>database</c>, <c>username</c>, <c>password</c>, <c>port</c>).</param>
    /// <returns>The connection arguments; the port is a <see cref="BigInteger" />, the others strings.</returns>
    internal OrderedDict<string, object> TranslateConnectArgs(IReadOnlyDictionary<string, string> names)
    {
        var translated = new OrderedDict<string, object>();
        void Add(string attribute, object? value)
        {
            bool truthy = value switch
            {
                null => false,
                string s => s.Length > 0,
                BigInteger b => !b.IsZero,
                _ => true,
            };
            if (truthy)
                translated[names.TryGetValue(attribute, out string? renamed) ? renamed : attribute] = value!;
        }
        Add("host", Host);
        Add("database", Database);
        Add("username", Username);
        Add("password", Password);
        Add("port", Port);
        return translated;
    }

    /// <summary>
    ///     <c>str(url)</c>, i.e. <c>render_as_string(hide_password=True)</c>: the URL rebuilt from its components with
    ///     the password replaced by <c>***</c>, the user name re-escaped, an IPv6 host bracketed and the query keys sorted.
    /// </summary>
    /// <returns>The rendering SQLAlchemy uses in error messages.</returns>
    public override string ToString()
    {
        var sb = new StringBuilder(Drivername).Append("://");
        if (Username is not null)
        {
            sb.Append(PyUrllib.Quote(Username, " +"));
            if (Password is not null)
                sb.Append(":***");
            sb.Append('@');
        }
        if (Host is not null)
            sb.Append(Host.Contains(':') ? "[" + Host + "]" : Host);
        if (Port is not null)
            sb.Append(':').Append(Port.Value.ToString(CultureInfo.InvariantCulture));
        if (Database is not null)
            sb.Append('/').Append(Database);
        if (Query.Count > 0)
        {
            var keys = new List<string>(Query.Keys);
            keys.Sort(PyStrings.CompareCodePoints);
            var parts = new List<string>();
            foreach (string key in keys)
            {
                foreach (string element in PyValues.ToList(Query[key]))
                    parts.Add(PyUrllib.QuotePlus(key) + "=" + PyUrllib.QuotePlus(element));
            }
            sb.Append('?').Append(string.Join("&", parts));
        }
        return sb.ToString();
    }
}

/// <summary>A SQLAlchemy dialect and the DBAPI driver module a URL selects.</summary>
/// <param name="Dialect">The dialect: <c>sqlite</c>, <c>postgresql</c>, <c>mysql</c>, <c>mariadb</c>, <c>oracle</c> or <c>mssql</c>.</param>
/// <param name="Driver">The driver module, with the default resolved (<c>postgresql</c> alone is <c>psycopg2</c>).</param>
/// <param name="IsAsync">Whether the driver is an asyncio driver, which Optuna's synchronous engine cannot use.</param>
internal sealed record SqlAlchemyDriver(string Dialect, string Driver, bool IsAsync);

/// <summary>
///     SQLAlchemy 2.0's dialect registry (<c>sqlalchemy.dialects.registry</c> with its <c>_auto_fn</c>) and the
///     plugin step of <c>create_engine</c>, reduced to the decision OptunaSharp needs: which dialect and driver a URL
///     selects, or which error SQLAlchemy raises instead.
/// </summary>
/// <remarks>
///     <para>The table lists the driver modules of SQLAlchemy 2.0.45's built-in dialect packages, plus <c>base</c>,
///     the default driver. SQLAlchemy 2.1 makes <c>psycopg</c> (version 3) the default PostgreSQL driver instead of
///     <c>psycopg2</c>; both are libpq drivers and read a URL identically.</para>
///     <para>Documented divergence: SQLAlchemy resolves a driver name by attribute lookup on the dialect package, so a
///     name that happens to be another member of it (<c>postgresql+json</c>, <c>mysql+types</c>) fails later with an
///     <c>AttributeError</c>. Here every name outside the table is the <c>NoSuchModuleError</c> an unknown name gets.</para>
/// </remarks>
internal static class SqlAlchemyRegistry
{
    /// <summary>The driver modules of each built-in dialect, the default first; <c>base</c> means the default.</summary>
    private static readonly Dictionary<string, (string Name, bool IsAsync)[]> Drivers = new(StringComparer.Ordinal)
    {
        ["sqlite"] = new[] { ("pysqlite", false), ("aiosqlite", true), ("pysqlcipher", false) },
        ["postgresql"] = new[]
        {
            ("psycopg2", false), ("psycopg", false), ("psycopg2cffi", false), ("pg8000", false), ("asyncpg", true),
            ("psycopg_async", true),
        },
        ["mysql"] = MySqlDrivers(),
        ["mariadb"] = MySqlDrivers(),
        ["oracle"] = new[] { ("cx_oracle", false), ("oracledb", false), ("oracledb_async", true) },
        ["mssql"] = new[] { ("pyodbc", false), ("pymssql", false), ("aioodbc", true) },
    };

    /// <summary>The MySQL driver modules; <c>mariadb://</c> loads the same ones through its own dialect class.</summary>
    /// <returns>The drivers, default first.</returns>
    private static (string Name, bool IsAsync)[] MySqlDrivers() => new[]
    {
        ("mysqldb", false), ("pymysql", false), ("mysqlconnector", false), ("mariadbconnector", false),
        ("cymysql", false), ("pyodbc", false), ("aiomysql", true), ("asyncmy", true),
    };

    /// <summary>
    ///     <c>create_engine</c>'s first steps: load every <c>?plugin=</c>, drop the plugin keys, then look the dialect
    ///     and driver up.
    /// </summary>
    /// <param name="url">The parsed URL.</param>
    /// <param name="withoutPlugins">The URL without its <c>plugin</c> and <c>plugin_name</c> query keys.</param>
    /// <returns>The dialect and driver.</returns>
    /// <exception cref="ArgumentException">
    ///     A plugin is named (no SQLAlchemy plugin exists in OptunaSharp), or the dialect or driver is unknown:
    ///     SQLAlchemy's <c>NoSuchModuleError</c> text <c>Can't load plugin: group:name</c>.
    /// </exception>
    /// <exception cref="ValueError">The driver name contains more than one <c>+</c> (Python's unpacking error).</exception>
    internal static SqlAlchemyDriver Resolve(SqlAlchemyUrl url, out SqlAlchemyUrl withoutPlugins)
    {
        if (url.Query.TryGetValue("plugin", out object? plugins))
            throw new ArgumentException($"Can't load plugin: sqlalchemy.plugins:{PyValues.ToList(plugins)[0]}");
        withoutPlugins = url.WithoutQueryKeys("plugin", "plugin_name");

        string name = url.Drivername.Replace('+', '.');
        string[] parts = name.Split('.');
        if (parts.Length > 2)
            throw new ValueError("too many values to unpack (expected 2)");
        string dialect = parts[0];
        string driver = parts.Length == 2 ? parts[1] : "base";
        if (Drivers.TryGetValue(dialect, out (string Name, bool IsAsync)[]? drivers))
        {
            if (driver == "base")
                return new SqlAlchemyDriver(dialect, drivers[0].Name, drivers[0].IsAsync);
            foreach ((string Name, bool IsAsync) known in drivers)
            {
                if (known.Name == driver)
                    return new SqlAlchemyDriver(dialect, known.Name, known.IsAsync);
            }
        }
        throw new ArgumentException($"Can't load plugin: sqlalchemy.dialects:{name}");
    }

    /// <summary>
    ///     SQLAlchemy's error when a synchronous engine connects through an asyncio driver
    ///     (<c>sqlalchemy.exc.MissingGreenlet</c>), which Optuna's <c>RDBStorage</c> raises while creating its tables.
    ///     The trailing "Background on this error" link is left out: it names the SQLAlchemy version.
    /// </summary>
    internal const string MissingGreenletMessage =
        "greenlet_spawn has not been called; can't call await_only() here. Was IO attempted in an unexpected place?";
}

/// <summary>
///     The value helpers SQLAlchemy's dialects use on connection arguments: <c>util.to_list</c>, <c>util.asbool</c>,
///     <c>util.coerce_kw_type</c>, and Python's <c>str()</c> / type names for the values a URL can produce.
/// </summary>
/// <remarks>
///     A URL produces these runtime types: <see cref="string" /> (a query value or component), <see cref="string" />
///     arrays (a repeated query key, a Python tuple), <see cref="BigInteger" /> (a port or a coerced <c>int</c>),
///     <see cref="double" /> (a coerced <c>float</c>), <see cref="bool" />, and <see langword="null" /> (<c>None</c>).
/// </remarks>
internal static class PyValues
{
    /// <summary><c>util.to_list(value)</c>: a tuple's elements, or a one-element list.</summary>
    /// <param name="value">A string or string array.</param>
    /// <returns>The elements.</returns>
    internal static IReadOnlyList<string> ToList(object value)
        => value is string[] many ? many : new[] { (string)value };

    /// <summary>Python's <c>str(value)</c> for the value types listed on the type.</summary>
    /// <param name="value">The value.</param>
    /// <returns>The text Python would produce.</returns>
    internal static string Str(object? value) => value switch
    {
        null => "None",
        string s => s,
        string[] tuple => PythonRepr.TupleRepr(tuple),
        bool b => b ? "True" : "False",
        BigInteger i => i.ToString(CultureInfo.InvariantCulture),
        double d => PythonRepr.FloatRepr(d),
        _ => value.ToString() ?? "None",
    };

    /// <summary>Python's type name of a value, for <c>TypeError</c> messages.</summary>
    /// <param name="value">The value.</param>
    /// <returns><c>str</c>, <c>tuple</c>, <c>bool</c>, <c>int</c>, <c>float</c>, <c>NoneType</c> or <c>dict</c>.</returns>
    internal static string TypeName(object? value) => value switch
    {
        null => "NoneType",
        string => "str",
        string[] => "tuple",
        bool => "bool",
        BigInteger => "int",
        double => "float",
        _ => "dict",
    };

    /// <summary>
    ///     SQLAlchemy's <c>util.asbool</c>: a string is stripped and lower-cased and must be one of
    ///     <c>true yes on y t 1</c> or <c>false no off n f 0</c>; any other value is Python truthiness.
    /// </summary>
    /// <param name="value">The value.</param>
    /// <returns>The boolean.</returns>
    /// <exception cref="ValueError">
    ///     A string outside both lists: <c>String is not true/false: '…'</c>, quoting the stripped and lower-cased
    ///     text as Python's <c>str.lower()</c> gives it (<c>İ</c> appears as <c>i̇</c>).
    /// </exception>
    internal static bool AsBool(object? value)
    {
        if (value is string s)
        {
            // Python's lower(), not ToLowerInvariant: the message quotes the lowered text, and the two differ on İ and Σ.
            string normalized = PyStrings.Lower(PyStrings.Strip(s));
            if (normalized is "true" or "yes" or "on" or "y" or "t" or "1")
                return true;
            if (normalized is "false" or "no" or "off" or "n" or "f" or "0")
                return false;
            throw new ValueError("String is not true/false: " + PythonRepr.StringRepr(normalized));
        }
        return Truthy(value);
    }

    /// <summary>Python truthiness of a value.</summary>
    /// <param name="value">The value.</param>
    /// <returns><see langword="false" /> for <c>None</c>, <c>False</c>, zero and empty text or tuples.</returns>
    internal static bool Truthy(object? value) => value switch
    {
        null => false,
        bool b => b,
        string s => s.Length > 0,
        string[] t => t.Length > 0,
        BigInteger i => !i.IsZero,
        double d => d != 0,
        _ => true,
    };

    /// <summary>Python <c>int(value)</c> for the value types listed on the type.</summary>
    /// <param name="value">The value.</param>
    /// <returns>The integer.</returns>
    /// <exception cref="ValueError">A string that is not an integer literal.</exception>
    /// <exception cref="TypeError">A tuple or <c>None</c>: CPython's <c>int() argument must be …</c> message.</exception>
    internal static BigInteger Int(object? value) => value switch
    {
        string s => PyNumbers.ParseInt(s),
        bool b => b ? BigInteger.One : BigInteger.Zero,
        BigInteger i => i,
        double d => double.IsFinite(d)
            ? new BigInteger(Math.Truncate(d))
            : throw (Exception)(double.IsNaN(d)
                ? new ValueError("cannot convert float NaN to integer")
                : new OverflowException("cannot convert float infinity to integer")),
        _ => throw new TypeError(
            $"int() argument must be a string, a bytes-like object or a real number, not '{TypeName(value)}'"),
    };

    /// <summary>Python <c>float(value)</c> for the value types listed on the type.</summary>
    /// <param name="value">The value.</param>
    /// <returns>The double.</returns>
    /// <exception cref="ValueError">A string that is not a float literal.</exception>
    /// <exception cref="TypeError">A tuple or <c>None</c>: CPython's <c>float() argument must be …</c> message.</exception>
    internal static double Float(object? value) => value switch
    {
        string s => PyNumbers.ParseFloat(s),
        bool b => b ? 1.0 : 0.0,
        BigInteger i => (double)i,
        double d => d,
        _ => throw new TypeError($"float() argument must be a string or a real number, not '{TypeName(value)}'"),
    };

    /// <summary>
    ///     SQLAlchemy's <c>util.coerce_kw_type(kw, key, type_, dest=dest)</c>: when <paramref name="key" /> is present,
    ///     not <c>None</c> and not already of the target type, its converted value is written to
    ///     <paramref name="dest" />. A value that already has the type is <b>not</b> copied to a separate
    ///     <paramref name="dest" /> — which is why a string <c>?isolation_level=</c> never reaches pysqlite.
    /// </summary>
    /// <param name="kw">The arguments to read.</param>
    /// <param name="key">The argument name.</param>
    /// <param name="type">The target Python type: <c>bool</c>, <c>int</c>, <c>float</c> or <c>str</c>.</param>
    /// <param name="dest">Where to write; <see langword="null" /> writes back to <paramref name="kw" />.</param>
    /// <exception cref="ValueError">The conversion fails (see <see cref="AsBool" />, <see cref="Int" />, <see cref="Float" />).</exception>
    /// <exception cref="TypeError">The value is a tuple that <c>int</c> or <c>float</c> cannot take.</exception>
    internal static void CoerceKwType(OrderedDict<string, object?> kw, string key, string type, OrderedDict<string, object?>? dest = null)
    {
        dest ??= kw;
        if (!kw.TryGetValue(key, out object? value) || value is null)
            return;
        bool alreadyTyped = type switch
        {
            // isinstance(True, int) holds in Python, so a bool is left alone by an int coercion.
            "bool" => value is bool,
            "int" => value is BigInteger or bool,
            "float" => value is double,
            _ => value is string,
        };
        if (alreadyTyped)
            return;
        dest[key] = type switch
        {
            "bool" => AsBool(value),
            "int" => Int(value),
            "float" => Float(value),
            _ => Str(value),
        };
    }
}

/// <summary>
///     Python <c>str</c> operations whose .NET counterparts differ: code-point ordering, <c>strip()</c> and
///     <c>lower()</c>.
/// </summary>
internal static class PyStrings
{
    /// <summary>
    ///     Python <c>str.lower()</c>: the full lower-case mapping, which differs from
    ///     <see cref="string.ToLowerInvariant" /> in two places. U+0130 (<c>İ</c>) becomes the two code points
    ///     <c>i̇</c> (U+0069 U+0307), where .NET leaves it unchanged; and U+03A3 (<c>Σ</c>) becomes the final sigma
    ///     <c>ς</c> at the end of a word (Unicode's Final_Sigma context), where .NET always gives <c>σ</c>.
    /// </summary>
    /// <param name="text">The text to lower-case; a lone surrogate passes through unchanged, as in Python.</param>
    /// <returns>The lower-cased text, possibly longer than <paramref name="text" />.</returns>
    /// <remarks>
    ///     Every other code point maps as .NET's invariant lower case maps it: the two tables were compared over all
    ///     of Unicode against CPython 3.12 and agree. The Final_Sigma context needs Unicode's Cased and
    ///     Case_Ignorable properties, which .NET does not expose; <see cref="IsCased" /> and
    ///     <see cref="IsCaseIgnorable" /> derive them from general categories plus Unicode's short lists, and
    ///     reproduce CPython 3.12's sets exactly for every character Unicode 15.0 assigns. A character assigned
    ///     later (Unicode 16.0, which .NET 10 knows and CPython 3.12 does not) can differ, as it can in
    ///     <see cref="PythonRepr.StringRepr" />.
    /// </remarks>
    internal static string Lower(string text)
    {
        // ASCII lower-cases the same everywhere; most URLs never leave it.
        bool ascii = true;
        foreach (char c in text)
        {
            if (c >= 0x80)
            {
                ascii = false;
                break;
            }
        }
        if (ascii)
            return text.ToLowerInvariant();

        var sb = new StringBuilder(text.Length);
        Span<char> buffer = stackalloc char[2];
        for (int i = 0; i < text.Length;)
        {
            int cp = CodePointAt(text, i, out int units);
            if (cp == 0x130)
                sb.Append("i\u0307");
            else if (cp == 0x3A3)
                sb.Append(IsFinalSigma(text, i, units) ? '\u03C2' : '\u03C3');
            else if (cp is >= 0xD800 and <= 0xDFFF)
                sb.Append((char)cp);
            else
                sb.Append(buffer[..Rune.ToLowerInvariant(new Rune(cp)).EncodeToUtf16(buffer)]);
            i += units;
        }
        return sb.ToString();
    }

    /// <summary>
    ///     The code point of the Python character at an index: a surrogate pair is one character, a lone surrogate is
    ///     a character of its own (its own value, not U+FFFD).
    /// </summary>
    /// <param name="text">The text.</param>
    /// <param name="index">The index of the character's first UTF-16 unit.</param>
    /// <param name="units">The number of UTF-16 units the character occupies, 1 or 2.</param>
    /// <returns>The code point.</returns>
    internal static int CodePointAt(string text, int index, out int units)
    {
        if (index + 1 < text.Length && char.IsSurrogatePair(text[index], text[index + 1]))
        {
            units = 2;
            return char.ConvertToUtf32(text[index], text[index + 1]);
        }
        units = 1;
        return text[index];
    }

    /// <summary>
    ///     CPython's <c>handle_capital_sigma</c>: the sigma at <paramref name="index" /> is final when, skipping
    ///     case-ignorable characters, a cased character precedes it and none follows it.
    /// </summary>
    /// <param name="text">The text.</param>
    /// <param name="index">The sigma's index.</param>
    /// <param name="units">The sigma's length in UTF-16 units (1).</param>
    /// <returns><see langword="true" /> when the sigma lower-cases to <c>ς</c>.</returns>
    private static bool IsFinalSigma(string text, int index, int units)
    {
        int before = -1;
        for (int j = index; j > 0;)
        {
            // Step back one Python character: a low surrogate preceded by its high surrogate is one code point.
            int start = j - 1;
            if (start > 0 && char.IsSurrogatePair(text[start - 1], text[start]))
                start--;
            int cp = CodePointAt(text, start, out _);
            j = start;
            if (!IsCaseIgnorable(cp))
            {
                before = cp;
                break;
            }
        }
        if (before < 0 || !IsCased(before))
            return false;
        for (int k = index + units; k < text.Length;)
        {
            int cp = CodePointAt(text, k, out int step);
            if (!IsCaseIgnorable(cp))
                return !IsCased(cp);
            k += step;
        }
        return true;
    }

    /// <summary>
    ///     Unicode's Case_Ignorable property: marks (Mn, Me), format characters (Cf), modifier letters and symbols
    ///     (Lm, Sk), and the Word_Break MidLetter, MidNumLet and Single_Quote characters.
    /// </summary>
    /// <param name="cp">The code point.</param>
    /// <returns>Whether the Final_Sigma scan skips it.</returns>
    private static bool IsCaseIgnorable(int cp)
        => CharUnicodeInfo.GetUnicodeCategory(cp) is UnicodeCategory.NonSpacingMark or UnicodeCategory.EnclosingMark
               or UnicodeCategory.Format or UnicodeCategory.ModifierLetter or UnicodeCategory.ModifierSymbol
           || cp is 0x27 or 0x2E or 0x3A or 0xB7 or 0x387 or 0x55F or 0x5F4 or 0x2018 or 0x2019 or 0x2024 or 0x2027
               or 0xFE13 or 0xFE52 or 0xFE55 or 0xFF07 or 0xFF0E or 0xFF1A;

    /// <summary>
    ///     Unicode's Cased property for a character that is not case-ignorable (the only kind the Final_Sigma scan asks
    ///     about): the cased letter categories plus the Other_Lowercase / Other_Uppercase characters outside them
    ///     (<c>ª º</c>, the Roman numerals, the circled and squared Latin letters).
    /// </summary>
    /// <param name="cp">The code point.</param>
    /// <returns>Whether it has case.</returns>
    private static bool IsCased(int cp)
        => CharUnicodeInfo.GetUnicodeCategory(cp) is UnicodeCategory.UppercaseLetter or UnicodeCategory.LowercaseLetter
               or UnicodeCategory.TitlecaseLetter
           || cp is 0xAA or 0xBA or (>= 0x2160 and <= 0x217F) or (>= 0x24B6 and <= 0x24E9)
               or (>= 0x1F130 and <= 0x1F149) or (>= 0x1F150 and <= 0x1F169) or (>= 0x1F170 and <= 0x1F189);

    /// <summary>
    ///     Python <c>str.strip()</c>: removes leading and trailing characters for which <c>str.isspace()</c> holds.
    ///     That set is .NET's <see cref="char.IsWhiteSpace(char)" /> plus U+001C–U+001F (the C0 information
    ///     separators, which Python counts as whitespace through their bidirectional class).
    /// </summary>
    /// <param name="text">The text to strip.</param>
    /// <returns>The text without surrounding whitespace.</returns>
    internal static string Strip(string text)
    {
        static bool IsSpace(char c) => char.IsWhiteSpace(c) || c is >= '\u001C' and <= '\u001F';
        int start = 0, end = text.Length;
        while (start < end && IsSpace(text[start]))
            start++;
        while (end > start && IsSpace(text[end - 1]))
            end--;
        return text.Substring(start, end - start);
    }

    /// <summary>
    ///     Compares two strings by Unicode code point. Ordinal UTF-16 comparison agrees except where a surrogate pair
    ///     (an astral character) meets a character in U+E000–U+FFFF, which UTF-16 orders the other way.
    /// </summary>
    /// <param name="a">The first string.</param>
    /// <param name="b">The second string.</param>
    /// <returns>A negative number, zero, or a positive number, as <c>a</c> sorts before, with, or after <c>b</c>.</returns>
    internal static int CompareCodePoints(string? a, string? b)
    {
        if (a is null || b is null)
            return string.CompareOrdinal(a, b);
        int i = 0, j = 0;
        while (i < a.Length && j < b.Length)
        {
            // CodePointAt, not Rune.DecodeFromUtf16: a lone surrogate keeps its own value (it would sort as U+FFFD).
            int ca = CodePointAt(a, i, out int la);
            int cb = CodePointAt(b, j, out int lb);
            if (ca != cb)
                return ca.CompareTo(cb);
            i += la;
            j += lb;
        }
        return (a.Length - i).CompareTo(b.Length - j);
    }
}

/// <summary>
///     Optuna's <c>RDBStorage._fill_storage_url_template</c>: every storage URL is passed through
///     <c>url.format(SCHEMA_VERSION=models.SCHEMA_VERSION)</c> before SQLAlchemy sees it.
/// </summary>
internal static class StorageUrlTemplate
{
    /// <summary><c>optuna.storages._rdb.models.SCHEMA_VERSION</c>, the value <c>{SCHEMA_VERSION}</c> expands to.</summary>
    internal const int SchemaVersion = 12;

    /// <summary>
    ///     Fills the template with Python's <c>str.format</c> rules (see <see cref="PyStrFormat" />): <c>{SCHEMA_VERSION}</c>
    ///     becomes <c>12</c>, <c>{{</c> and <c>}}</c> become single braces, any other brace is an error.
    /// </summary>
    /// <param name="template">The storage URL as the user wrote it.</param>
    /// <returns>The URL SQLAlchemy parses.</returns>
    /// <exception cref="ValueError">A single brace or a malformed field; CPython's message.</exception>
    /// <exception cref="KeyError">A field other than <c>SCHEMA_VERSION</c>, for example a password containing <c>{x}</c>.</exception>
    /// <exception cref="IndexError">A positional field such as <c>{}</c> or <c>{0}</c>.</exception>
    /// <exception cref="AttributeError">A field reads an attribute an <c>int</c> lacks.</exception>
    /// <exception cref="TypeError">A field indexes the integer.</exception>
    /// <exception cref="NotSupportedException">A field reads one of <c>int</c>'s methods.</exception>
    /// <exception cref="OverflowException">A format spec that CPython rejects with <c>OverflowError</c>.</exception>
    /// <exception cref="OutOfMemoryException">
    ///     A field padded past the longest .NET string (a width above about 2<sup>30</sup>), which CPython would build
    ///     when memory allows.
    /// </exception>
    /// <remarks>
    ///     Footgun: braces are special in every part of the URL. A password containing <c>{</c> or <c>}</c> must be
    ///     written <c>{{</c> / <c>}}</c> (or percent-encoded as <c>%7B</c> / <c>%7D</c>), exactly as with Python Optuna.
    /// </remarks>
    internal static string Fill(string template)
        => PyStrFormat.FormatKeyword(template, "SCHEMA_VERSION", SchemaVersion);
}

using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Numerics;
using NumSharp;
using OptunaSharp.Collections;
using OptunaSharp.Internal;
using OptunaSharp.Storages.SqlAlchemy;

namespace OptunaSharp.Storages;

/// <summary>
///     What a SQLite storage URL makes Python open: the arguments SQLAlchemy's pysqlite dialect hands to
///     <c>sqlite3.connect</c>, after <c>sqlite3.connect</c> itself has validated them.
/// </summary>
/// <param name="Filename">
///     The database: <c>:memory:</c>, an absolute file path (<c>os.path.abspath</c> of the URL's path), or — with
///     <c>?uri=true</c> — the URL's path as written followed by its non-pysqlite query arguments.
/// </param>
/// <param name="Uri">Whether SQLite interprets <paramref name="Filename" /> as a URI filename (<c>file:…</c>).</param>
/// <param name="TimeoutSeconds">The <c>?timeout=</c> value in seconds, or <see langword="null" /> when the URL has none.</param>
internal sealed record SqliteConnectArgs(string Filename, bool Uri, double? TimeoutSeconds)
{
    /// <summary>
    ///     The busy-handler timeout Python's <c>sqlite3</c> installs for <see cref="TimeoutSeconds" />:
    ///     <c>sqlite3_busy_timeout(db, (int)(timeout * 1000))</c>. Zero or negative turns the handler off, which is
    ///     also what x86-64 CPython ends up with for <c>inf</c> and <c>nan</c>.
    /// </summary>
    internal int? BusyTimeoutMs => TimeoutSeconds is double t ? PyNumbers.CTruncateToInt32(t * 1000.0) : null;

    /// <summary>
    ///     The Microsoft.Data.Sqlite command timeout that keeps its own busy-retry loop within the URL's timeout:
    ///     the timeout rounded up to whole seconds, and at least one second — zero would mean "retry forever"
    ///     there, the opposite of Python's "do not wait".
    /// </summary>
    internal int? CommandTimeoutSeconds => TimeoutSeconds is double t
        ? t is > 0 and < int.MaxValue ? Math.Max(1, (int)Math.Ceiling(t)) : 1
        : null;
}

/// <summary>
///     Port of the URL handling of Optuna's <c>RDBStorage</c> for SQLite: the template, SQLAlchemy's URL grammar,
///     and <c>SQLiteDialect_pysqlite.create_connect_args</c> (<c>sqlalchemy/dialects/sqlite/pysqlite.py</c>), with
///     the argument checks <c>sqlite3.connect</c> performs.
/// </summary>
/// <remarks>
///     <para><b>The query string.</b> pysqlite reads six query arguments and ignores every other one:
///     <c>timeout</c> (seconds; the busy timeout), <c>uri</c> (treat the path as a SQLite URI filename),
///     <c>isolation_level</c>, <c>detect_types</c>, <c>check_same_thread</c> and <c>cached_statements</c>. The
///     query is never part of the file name — except with <c>uri=true</c>, where the remaining arguments are
///     appended to the path as URI parameters (<c>sqlite:///file:x.db?mode=ro&amp;uri=true</c> opens
///     <c>file:x.db?mode=ro</c>). Only <c>timeout</c> and <c>uri</c> change what OptunaSharp does; the others
///     are type-checked like Python checks them and otherwise have no counterpart in Microsoft.Data.Sqlite.</para>
///     <para><b>A SQLAlchemy quirk kept on purpose.</b> pysqlite's <c>util.coerce_kw_type</c> forwards a value only
///     when it had to convert it, and a URL's <c>isolation_level</c> is already a string, so Python never passes
///     it to <c>sqlite3.connect</c>: <c>?isolation_level=IMMEDIATE</c> has no effect. Only a repeated key (a
///     tuple) gets forwarded, as its <c>str()</c>, which <c>sqlite3</c> then refuses.</para>
///     <para><b>Paths.</b> A file path is made absolute with <c>os.path.abspath</c> against the current working
///     directory when the storage is opened, as SQLAlchemy does — so <c>sqlite:////C:/x.db</c> means
///     <c>&lt;drive&gt;:\C:\x.db</c> on Windows and fails to open, exactly as in Python, and
///     <c>sqlite:///file:x.db</c> without <c>uri=true</c> is a file literally named <c>file:x.db</c>.</para>
/// </remarks>
internal static class SqliteUrl
{
    /// <summary>The query arguments pysqlite coerces, in its order, with their Python types.</summary>
    private static readonly (string Key, string Type)[] PysqliteArgs =
    {
        ("uri", "bool"), ("timeout", "float"), ("isolation_level", "str"), ("detect_types", "int"),
        ("check_same_thread", "bool"), ("cached_statements", "int"),
    };

    /// <summary>
    ///     Everything <c>RDBStorage(url)</c> does with a SQLite URL before the file is opened.
    /// </summary>
    /// <param name="url">The storage URL as written.</param>
    /// <param name="validateOptions">
    ///     The storage's option checks, run after the template is filled and before the URL is parsed — where
    ///     <c>RDBStorage.__init__</c> validates its heartbeat arguments.
    /// </param>
    /// <returns>The filled URL (Python's <c>RDBStorage.url</c>) and what to open.</returns>
    /// <exception cref="ValueError">A template error or malformed port; a URL of another dialect; an invalid query value.</exception>
    /// <exception cref="KeyError">The template names a field other than <c>SCHEMA_VERSION</c>.</exception>
    /// <exception cref="IndexError">The template has a positional field.</exception>
    /// <exception cref="AttributeError">The template reads an attribute an <c>int</c> lacks.</exception>
    /// <exception cref="TypeError">A template index; a repeated query key <c>int</c>/<c>float</c> cannot take; <c>?uri=true</c> without a path.</exception>
    /// <exception cref="ArgumentException">Not a SQLAlchemy URL; an unknown dialect, driver or plugin; or a SQLite URL with a host, user, password or port.</exception>
    /// <exception cref="NotSupportedException">A driver with no OptunaSharp counterpart (<c>sqlite+pysqlcipher</c>).</exception>
    /// <exception cref="OverflowError">A <c>detect_types</c> or <c>cached_statements</c> outside the C <c>int</c> range (Python's class, NumSharp's since 0.81.0).</exception>
    /// <exception cref="InvalidOperationException">An asyncio driver (<c>sqlite+aiosqlite</c>); SQLAlchemy's synchronous-engine error.</exception>
    internal static (string Filled, SqliteConnectArgs Args) Resolve(string url, Action? validateOptions = null)
    {
        PreparedStorageUrl prepared = StorageUrlResolver.Prepare(url, validateOptions);
        if (prepared.Driver.Dialect != "sqlite")
            throw new ValueError(
                $"'{prepared.Driver.Dialect}' URLs are served by the OptunaSharp.Storage.EFCore package, not " +
                "OptunaSharp.Storage.Sqlite. Use OptunaSharp.Storages.EFCore.RDBStorage from that package, or " +
                "StorageFunctions.GetStorage(url), which picks the package for the URL.");
        (string? filename, OrderedDict<string, object?> opts) = CreateConnectArgs(prepared.Url);
        // Python builds the engine (coercing the arguments above) before the first connection; an asyncio
        // driver fails on that connection, before sqlite3.connect ever checks its arguments.
        if (prepared.Driver.IsAsync)
            throw new InvalidOperationException(SqlAlchemyRegistry.MissingGreenletMessage);
        return (prepared.Filled, Connect(filename, opts));
    }

    /// <summary>
    ///     <c>SQLiteDialect_pysqlite.create_connect_args(url)</c>: refuses network components, coerces the six
    ///     pysqlite query arguments, and builds the filename.
    /// </summary>
    /// <param name="url">The parsed URL, without plugin keys.</param>
    /// <returns>
    ///     The filename (<see langword="null" /> for <c>sqlite://?uri=true</c>, which only <c>sqlite3.connect</c>
    ///     refuses) and the coerced arguments pysqlite passes to <c>sqlite3.connect</c>.
    /// </returns>
    /// <exception cref="ArgumentException">The URL has a user name, password, host or port (SQLAlchemy's <c>Invalid SQLite URL</c>).</exception>
    /// <exception cref="ValueError">An invalid boolean, float or integer query value.</exception>
    /// <exception cref="TypeError">A repeated <c>timeout</c>, <c>detect_types</c> or <c>cached_statements</c>; URI arguments on <c>sqlite://</c>.</exception>
    internal static (string? Filename, OrderedDict<string, object?> Opts) CreateConnectArgs(SqlAlchemyUrl url)
    {
        if (PyValues.Truthy(url.Username) || PyValues.Truthy(url.Password) || PyValues.Truthy(url.Host)
            || url.Port is { IsZero: false })
            throw new ArgumentException(
                $"Invalid SQLite URL: {url}\nValid SQLite URL forms are:\n sqlite:///:memory: (or, sqlite://)\n" +
                " sqlite:///relative/path/to/file.db\n sqlite:////absolute/path/to/file.db");

        var query = new OrderedDict<string, object?>();
        foreach (KeyValuePair<string, object> pair in url.Query)
            query[pair.Key] = pair.Value;
        var opts = new OrderedDict<string, object?>();
        foreach ((string key, string type) in PysqliteArgs)
            PyValues.CoerceKwType(query, key, type, opts);

        string? filename;
        bool uri = opts.TryGetValue("uri", out object? uriValue) && PyValues.Truthy(uriValue);
        if (uri)
        {
            // The URI filename is the path as written plus every query argument pysqlite did not consume,
            // sorted by key and NOT re-escaped — Python's "%s=%s" % (key, value), str() of a tuple included.
            var uriOpts = query.Where(pair => PysqliteArgs.All(a => a.Key != pair.Key)).ToList();
            filename = url.Database;
            if (uriOpts.Count > 0)
            {
                if (filename is null)
                    throw new TypeError("unsupported operand type(s) for +=: 'NoneType' and 'str'");
                filename += "?" + string.Join("&", uriOpts
                    .OrderBy(pair => pair.Key, Comparer<string>.Create(PyStrings.CompareCodePoints))
                    .Select(pair => pair.Key + "=" + PyValues.Str(pair.Value)));
            }
        }
        else
        {
            filename = string.IsNullOrEmpty(url.Database) ? ":memory:" : url.Database;
            if (filename != ":memory:")
                filename = PyOsPath.AbsPath(filename);
        }

        return (filename, opts);
    }

    /// <summary>
    ///     The argument checks of <c>sqlite3.connect(filename, **opts)</c> that run before SQLite opens anything.
    /// </summary>
    /// <param name="filename">The filename <see cref="CreateConnectArgs" /> built.</param>
    /// <param name="opts">The coerced pysqlite arguments.</param>
    /// <returns>What to open.</returns>
    /// <exception cref="TypeError">No filename (<c>sqlite://?uri=true</c>).</exception>
    /// <exception cref="ValueError">A NUL in the path, or an invalid repeated <c>isolation_level</c>.</exception>
    /// <exception cref="OverflowError">A <c>detect_types</c> or <c>cached_statements</c> outside the C <c>int</c> range (Python's class, NumSharp's since 0.81.0).</exception>
    internal static SqliteConnectArgs Connect(string? filename, OrderedDict<string, object?> opts)
    {
        // sqlite3.connect(database, timeout, detect_types, isolation_level, check_same_thread, factory,
        // cached_statements, uri): Argument Clinic converts in signature order, so the errors come in that order.
        if (filename is null)
            throw new TypeError("expected str, bytes or os.PathLike object, not NoneType");
        if (filename.Contains('\0'))
            throw new ValueError("embedded null character");
        CheckCInt(opts, "detect_types");
        // get_isolation_level compares with sqlite3_stricmp, which folds ASCII letters only and compares every other
        // byte as it is; Ascii.EqualsIgnoreCase is that comparison, OrdinalIgnoreCase folds non-ASCII letters too.
        if (opts.TryGetValue("isolation_level", out object? level) && level is string text
            && text is not "" && !System.Text.Ascii.EqualsIgnoreCase(text, "DEFERRED")
            && !System.Text.Ascii.EqualsIgnoreCase(text, "IMMEDIATE")
            && !System.Text.Ascii.EqualsIgnoreCase(text, "EXCLUSIVE"))
            throw new ValueError("isolation_level string must be '', 'DEFERRED', 'IMMEDIATE', or 'EXCLUSIVE'");
        CheckCInt(opts, "cached_statements");

        double? timeout = opts.TryGetValue("timeout", out object? t) ? (double)t! : null;
        bool uri = opts.TryGetValue("uri", out object? uriValue) && PyValues.Truthy(uriValue);
        return new SqliteConnectArgs(filename, uri, timeout);
    }

    /// <summary>Argument Clinic's C <c>int</c> conversion of an integer argument.</summary>
    /// <param name="opts">The coerced arguments.</param>
    /// <param name="key">The argument to check, if present.</param>
    /// <exception cref="OverflowError">The value is outside the 32-bit range: <c>Python int too large to convert to C int</c>.</exception>
    private static void CheckCInt(OrderedDict<string, object?> opts, string key)
    {
        if (opts.TryGetValue(key, out object? value) && value is BigInteger i && (i < int.MinValue || i > int.MaxValue))
            throw new OverflowError("Python int too large to convert to C int");
    }
}

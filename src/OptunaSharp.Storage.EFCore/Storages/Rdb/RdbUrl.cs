using System;
using System.Data.Common;
using NumSharp;
using OptunaSharp.Storages.SqlAlchemy;

namespace OptunaSharp.Storages.EFCore;

/// <summary>The RDB backend a URL selects. EF Core covers the two server databases Optuna supports.</summary>
public enum RdbProvider
{
    /// <summary>PostgreSQL, over Npgsql (<c>postgresql://</c>, <c>postgresql+psycopg2://</c>, …).</summary>
    PostgreSql,

    /// <summary>MySQL / MariaDB, over MySqlConnector + Pomelo (<c>mysql://</c>, <c>mysql+pymysql://</c>, <c>mariadb://</c>, …).</summary>
    MySql,
}

/// <summary>
///     Parses a SQLAlchemy storage URL — the same string Optuna's <c>optuna.storages.RDBStorage(url)</c> takes —
///     into a provider and a native ADO.NET connection string, with Python's semantics end to end: Optuna's
///     <c>{SCHEMA_VERSION}</c> template, SQLAlchemy's URL grammar and driver registry, the driver's
///     <c>create_connect_args</c>, and the driver's own handling of those arguments (libpq's for PostgreSQL, PyMySQL's
///     or MySQLdb's for MySQL). See <see cref="PostgresUrl" /> and <see cref="MySqlUrl" /> for the rules.
/// </summary>
/// <remarks>
///     SQLite URLs (<c>sqlite:///…</c>) are refused with a pointer to the <c>OptunaSharp.Storage.Sqlite</c> package,
///     which serves them. Native Npgsql or MySqlConnector keywords do not belong in the URL — Python's drivers would
///     reject them — and are set with <see cref="EFCoreRdbOptions.ConfigureConnectionString" /> instead.
/// </remarks>
internal static class RdbUrl
{
    /// <summary>
    ///     Parse <paramref name="url" /> into its provider and a connection-string builder (an
    ///     <see cref="Npgsql.NpgsqlConnectionStringBuilder" /> or <see cref="MySqlConnector.MySqlConnectionStringBuilder" />).
    /// </summary>
    /// <param name="url">The storage URL as written.</param>
    /// <param name="validateOptions">
    ///     The storage's option checks, run after the template is filled and before the URL is parsed — where
    ///     <c>RDBStorage.__init__</c> validates its heartbeat arguments. <see langword="null" /> runs nothing.
    /// </param>
    /// <returns>The selected relational provider and its native connection-string builder.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="url" /> is null.</exception>
    /// <exception cref="ArgumentException">Not a SQLAlchemy URL, an unknown dialect/driver/plugin, or a SQLAlchemy multi-host error.</exception>
    /// <exception cref="ValueError">A template or port error, a SQLite URL, or a driver value check; Python's messages.</exception>
    /// <exception cref="KeyError">The template names a field other than <c>SCHEMA_VERSION</c>.</exception>
    /// <exception cref="IndexError">The template has a positional field.</exception>
    /// <exception cref="AttributeError">The template reads an attribute <c>int</c> lacks.</exception>
    /// <exception cref="TypeError">An argument the driver refuses by type or name.</exception>
    /// <exception cref="FormatException">libpq refuses the PostgreSQL connection info or one of its values.</exception>
    /// <exception cref="NotImplementedException">PyMySQL's refusal of <c>compress</c> and <c>named_pipe</c>.</exception>
    /// <exception cref="NotSupportedException">A dialect/driver without an OptunaSharp counterpart, or an argument the .NET driver cannot express.</exception>
    /// <exception cref="InvalidOperationException">An asyncio driver.</exception>
    internal static (RdbProvider Provider, DbConnectionStringBuilder Builder) Parse(string url, Action? validateOptions = null)
    {
        ArgumentNullException.ThrowIfNull(url);
        PreparedStorageUrl prepared = StorageUrlResolver.Prepare(url, validateOptions);
        return prepared.Driver.Dialect switch
        {
            "postgresql" => (RdbProvider.PostgreSql, PostgresUrl.Build(prepared)),
            "mysql" or "mariadb" => (RdbProvider.MySql, MySqlUrl.Build(prepared)),
            _ => throw new ValueError(
                "SQLite URLs are served by the OptunaSharp.Storage.Sqlite package, not " +
                "OptunaSharp.Storage.EFCore. Use OptunaSharp.Storages.RDBStorage from that package, or " +
                "StorageFunctions.GetStorage(url), which picks the package for the URL."),
        };
    }

    /// <summary>
    ///     The filled form of a storage URL — Python's <c>RDBStorage.url</c> — for a URL <see cref="Parse" /> accepted.
    /// </summary>
    /// <param name="url">The storage URL as written.</param>
    /// <returns>The URL with <c>{SCHEMA_VERSION}</c> filled in.</returns>
    internal static string Filled(string url) => StorageUrlTemplate.Fill(url);
}

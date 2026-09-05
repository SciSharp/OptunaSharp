using System;
using System.Data.Common;
using MySqlConnector;
using Npgsql;
using NumSharp;

namespace OptunaSharp.Storages.EFCore;

/// <summary>The RDB backend a URL selects. EF Core covers the two server databases Optuna supports.</summary>
public enum RdbProvider
{
    /// <summary>PostgreSQL, over Npgsql (<c>postgresql://</c>, <c>postgresql+psycopg2://</c>, …).</summary>
    PostgreSql,

    /// <summary>MySQL / MariaDB, over MySqlConnector + Pomelo (<c>mysql://</c>, <c>mysql+pymysql://</c>, …).</summary>
    MySql,
}

/// <summary>
///     Parses a SQLAlchemy-style storage URL — the same string Optuna's
///     <c>optuna.storages.RDBStorage(url)</c> takes — into a provider and a native ADO.NET connection
///     string. Handles the two server dialects EF Core targets:
///     <c>postgresql[+driver]://user:pass@host:port/db</c> and <c>mysql[+driver]://…</c>.
///
///     <para>SQLite URLs (<c>sqlite:///…</c>) are intentionally rejected — they are served by the
///     separate <c>OptunaSharp.Storage.Sqlite</c> package.</para>
/// </summary>
internal static class RdbUrl
{
    /// <summary>
    ///     Parse <paramref name="url" /> into its provider and a connection-string builder (an
    ///     <see cref="NpgsqlConnectionStringBuilder" /> or <see cref="MySqlConnectionStringBuilder" />).
    /// </summary>
    internal static (RdbProvider Provider, DbConnectionStringBuilder Builder) Parse(string url)
    {
        ArgumentNullException.ThrowIfNull(url);

        int schemeEnd = url.IndexOf("://", StringComparison.Ordinal);
        if (schemeEnd < 0)
            throw new ValueError(
                $"Could not parse rfc1738 URL from string '{url}'. Expected a scheme like " +
                "'postgresql://user:pass@host:port/dbname' or 'mysql://…'.");

        string scheme = url.Substring(0, schemeEnd).ToLowerInvariant();
        string rest = url.Substring(schemeEnd + 3);
        string dialect = scheme.Split('+')[0];

        RdbProvider provider = dialect switch
        {
            "postgresql" or "postgres" => RdbProvider.PostgreSql,
            "mysql" or "mariadb" => RdbProvider.MySql,
            "sqlite" => throw new ValueError(
                "SQLite URLs are served by the OptunaSharp.Storage.Sqlite package, not " +
                "OptunaSharp.Storage.EFCore. Use OptunaSharp.Storages.RDBStorage from that package."),
            _ => throw new ValueError(
                $"Unsupported RDB dialect '{dialect}'. OptunaSharp.Storage.EFCore supports " +
                "'postgresql' and 'mysql'."),
        };

        // rest = [user[:password]@]host[:port][/dbname][?query]
        string query = "";
        int q = rest.IndexOf('?');
        if (q >= 0)
        {
            query = rest.Substring(q + 1);
            rest = rest.Substring(0, q);
        }

        string authority = rest;
        string database = "";
        int slash = rest.IndexOf('/');
        if (slash >= 0)
        {
            authority = rest.Substring(0, slash);
            database = Uri.UnescapeDataString(rest.Substring(slash + 1));
        }

        string? user = null;
        string? password = null;
        string hostPort = authority;
        int at = authority.LastIndexOf('@');
        if (at >= 0)
        {
            string userInfo = authority.Substring(0, at);
            hostPort = authority.Substring(at + 1);
            int colon = userInfo.IndexOf(':');
            if (colon >= 0)
            {
                user = Uri.UnescapeDataString(userInfo.Substring(0, colon));
                password = Uri.UnescapeDataString(userInfo.Substring(colon + 1));
            }
            else
            {
                user = Uri.UnescapeDataString(userInfo);
            }
        }

        string host = hostPort;
        int? port = null;
        int hostColon = hostPort.LastIndexOf(':');
        if (hostColon >= 0 && int.TryParse(hostPort.Substring(hostColon + 1), out int parsedPort))
        {
            host = hostPort.Substring(0, hostColon);
            port = parsedPort;
        }
        if (host.Length == 0)
            host = "localhost";

        return provider == RdbProvider.PostgreSql
            ? (provider, BuildNpgsql(host, port, user, password, database, query))
            : (provider, BuildMySql(host, port, user, password, database, query));
    }

    private static DbConnectionStringBuilder BuildNpgsql(
        string host, int? port, string? user, string? password, string database, string query)
    {
        var b = new NpgsqlConnectionStringBuilder { Host = host };
        if (port is not null)
            b.Port = port.Value;
        if (user is not null)
            b.Username = user;
        if (password is not null)
            b.Password = password;
        if (database.Length > 0)
            b.Database = database;
        ApplyQuery(b, query);
        return b;
    }

    private static DbConnectionStringBuilder BuildMySql(
        string host, int? port, string? user, string? password, string database, string query)
    {
        var b = new MySqlConnectionStringBuilder { Server = host };
        if (port is not null)
            b.Port = (uint)port.Value;
        if (user is not null)
            b.UserID = user;
        if (password is not null)
            b.Password = password;
        if (database.Length > 0)
            b.Database = database;
        ApplyQuery(b, query);
        return b;
    }

    // Pass query-string pairs (e.g. ?SSL Mode=Require) straight through as connection-string keywords.
    // Both builders derive from DbConnectionStringBuilder and validate keywords per provider.
    private static void ApplyQuery(DbConnectionStringBuilder builder, string query)
    {
        if (query.Length == 0)
            return;
        foreach (string pair in query.Split('&', StringSplitOptions.RemoveEmptyEntries))
        {
            int eq = pair.IndexOf('=');
            if (eq < 0)
                continue;
            string key = Uri.UnescapeDataString(pair.Substring(0, eq));
            string value = Uri.UnescapeDataString(pair.Substring(eq + 1));
            builder[key] = value;
        }
    }
}

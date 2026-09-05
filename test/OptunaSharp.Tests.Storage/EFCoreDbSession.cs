using System;
using MySqlConnector;
using Npgsql;
using OptunaSharp.Storages.EFCore;
using Testcontainers.MySql;
using Testcontainers.PostgreSql;

namespace OptunaSharp.Tests.Storage;

/// <summary>
///     Testcontainers-backed PostgreSQL and MySQL servers for the EF Core RDB conformance suite
///     (Phase 4.2). Each provider's container is started once, lazily, and reused. Mirrors
///     <see cref="RedisSession" />'s gating: a test calls <see cref="EnsureAvailable" /> first and
///     reports <see cref="Assert.Inconclusive(string)" /> when Docker is unavailable — so the suite is
///     safe without Docker — unless <c>OPTUNASHARP_EFCORE_REQUIRED</c> is truthy (then absent is a hard
///     failure, for CI once wired).
/// </summary>
internal static class EFCoreDbSession
{
    private sealed class Endpoint
    {
        public IAsyncDisposable? Container;
        public string ConnectionString = "";
        public string SqlUrl = "";
    }

    private static readonly Lazy<Endpoint?> _postgres = new(() => StartPostgres());
    private static readonly Lazy<Endpoint?> _mysql = new(() => StartMySql());

    private static Lazy<Endpoint?> Lazy(RdbProvider p) => p == RdbProvider.PostgreSql ? _postgres : _mysql;

    public static void EnsureAvailable(RdbProvider provider)
    {
        if (Lazy(provider).Value is not null)
            return;
        string reason =
            $"Docker is not available, so the {provider} testcontainer could not start. " +
            "Install/start Docker to run the EF Core RDB conformance suite.";
        if (Required())
            Assert.Fail(reason + " (OPTUNASHARP_EFCORE_REQUIRED is set, so this is a hard failure).");
        Assert.Inconclusive(reason);
    }

    /// <summary>A native ADO.NET connection string for the started container.</summary>
    public static string ConnectionString(RdbProvider provider) => Lazy(provider).Value!.ConnectionString;

    /// <summary>A SQLAlchemy-style URL (<c>postgresql://…</c> / <c>mysql://…</c>) for the container.</summary>
    public static string SqlUrl(RdbProvider provider) => Lazy(provider).Value!.SqlUrl;

    /// <summary>Build a storage against the container, over a fresh (dropped-and-recreated) schema.</summary>
    public static RDBStorage NewStorage(RdbProvider provider)
    {
        Reset(provider);
        return new RDBStorage(provider, ConnectionString(provider));
    }

    /// <summary>Drop every Optuna table (and, on PostgreSQL, the enum types) so each test starts clean.</summary>
    public static void Reset(RdbProvider provider)
    {
        string cs = ConnectionString(provider);
        if (provider == RdbProvider.PostgreSql)
        {
            using var conn = new NpgsqlConnection(cs);
            conn.Open();
            foreach (string t in Tables)
                Exec(conn, $"DROP TABLE IF EXISTS {t} CASCADE");
            foreach (string e in new[] { "studydirection", "trialstate", "trialvaluetype", "trialintermediatevaluetype" })
                Exec(conn, $"DROP TYPE IF EXISTS {e} CASCADE");
        }
        else
        {
            using var conn = new MySqlConnection(cs);
            conn.Open();
            Exec(conn, "SET FOREIGN_KEY_CHECKS = 0");
            foreach (string t in Tables)
                Exec(conn, $"DROP TABLE IF EXISTS {t}");
            Exec(conn, "SET FOREIGN_KEY_CHECKS = 1");
        }
    }

    internal static readonly string[] Tables =
    {
        "trial_heartbeats", "trial_intermediate_values", "trial_values", "trial_params",
        "trial_system_attributes", "trial_user_attributes", "trials",
        "study_system_attributes", "study_user_attributes", "study_directions",
        "alembic_version", "version_info", "studies",
    };

    private static void Exec(System.Data.Common.DbConnection conn, string sql)
    {
        using var cmd = conn.CreateCommand();
        cmd.CommandText = sql;
        cmd.ExecuteNonQuery();
    }

    private static bool Required()
    {
        string? v = Environment.GetEnvironmentVariable("OPTUNASHARP_EFCORE_REQUIRED");
        return v is "1" or "true" or "TRUE" or "yes";
    }

    private static Endpoint? StartPostgres()
    {
        try
        {
            var c = new PostgreSqlBuilder()
                .WithImage("postgres:16-alpine")
                .WithUsername("optuna").WithPassword("optuna").WithDatabase("optuna")
                .Build();
            c.StartAsync().GetAwaiter().GetResult();
            int port = c.GetMappedPublicPort(5432);
            return new Endpoint
            {
                Container = c,
                ConnectionString = c.GetConnectionString(),
                SqlUrl = $"postgresql://optuna:optuna@{c.Hostname}:{port}/optuna",
            };
        }
        catch
        {
            return null;
        }
    }

    private static Endpoint? StartMySql()
    {
        try
        {
            var c = new MySqlBuilder()
                .WithImage("mysql:8.0")
                .WithUsername("optuna").WithPassword("optuna").WithDatabase("optuna")
                .Build();
            c.StartAsync().GetAwaiter().GetResult();
            int port = c.GetMappedPublicPort(3306);
            return new Endpoint
            {
                Container = c,
                ConnectionString = c.GetConnectionString(),
                SqlUrl = $"mysql://optuna:optuna@{c.Hostname}:{port}/optuna",
            };
        }
        catch
        {
            return null;
        }
    }

    public static void DisposeAll()
    {
        foreach (var lazy in new[] { _postgres, _mysql })
            if (lazy.IsValueCreated && lazy.Value?.Container is { } c)
                c.DisposeAsync().AsTask().GetAwaiter().GetResult();
    }
}

/// <summary>Disposes the shared testcontainers at the end of the assembly's test run.</summary>
[TestClass]
public static class EFCoreDbSessionLifecycle
{
    [AssemblyCleanup]
    public static void Cleanup() => EFCoreDbSession.DisposeAll();
}

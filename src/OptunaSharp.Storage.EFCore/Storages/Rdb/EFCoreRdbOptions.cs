using System;
using System.Data.Common;
using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Infrastructure;
using Npgsql.EntityFrameworkCore.PostgreSQL.Infrastructure;

namespace OptunaSharp.Storages.EFCore;

/// <summary>
///     Configuration for the EF Core <see cref="RDBStorage" /> — the Optuna
///     (<c>optuna.storages.RDBStorage</c>) knobs plus a connection-string escape hatch. Every option
///     is opt-in; a <c>null</c> value leaves the provider default in place.
/// </summary>
public sealed class EFCoreRdbOptions
{
    // ------------------------------------------------------------------ RDBStorage (Optuna) knobs

    /// <summary>Skip the alembic schema-compatibility check (Python <c>skip_compatibility_check</c>).</summary>
    public bool SkipCompatibilityCheck { get; set; }

    /// <summary>Skip creating the tables — assume they already exist (Python <c>skip_table_creation</c>).</summary>
    public bool SkipTableCreation { get; set; }

    /// <summary>Heartbeat recording interval in seconds (experimental). Must be positive when set.</summary>
    public int? HeartbeatInterval { get; set; }

    /// <summary>
    ///     Grace period in seconds before a running trial is failed from its last heartbeat.
    ///     Defaults to <c>2 * HeartbeatInterval</c> when <c>null</c>. Must be positive when set.
    /// </summary>
    public int? GracePeriod { get; set; }

    /// <summary>Callback invoked after each heartbeat-stale trial is failed.</summary>
    public Action<Study, FrozenTrial>? HeartbeatStaleTrialCallback { get; set; }

    // ------------------------------------------------------------------ connection-string escape hatch

    /// <summary>
    ///     Final tweak of the connection-string builder before the data source is built — the escape
    ///     hatch for any provider connection-string keyword the URL doesn't carry (e.g.
    ///     <c>["SSL Mode"] = "Require"</c>, pooling, timeouts). The argument is an
    ///     <c>NpgsqlConnectionStringBuilder</c> or <c>MySqlConnectionStringBuilder</c> (both derive
    ///     from <see cref="DbConnectionStringBuilder" />). Applies only when OptunaSharp builds the
    ///     connection from a URL/native connection string; ignored when you inject your own data source.
    /// </summary>
    public Action<DbConnectionStringBuilder>? ConfigureConnectionString { get; set; }

    // ------------------------------------------------------------------ EF Core / provider surface

    /// <summary>
    ///     Configure the whole <see cref="DbContextOptionsBuilder" /> — the general EF Core surface,
    ///     applied to every context this storage creates. Use for logging
    ///     (<c>LogTo</c>/<c>UseLoggerFactory</c>), <c>EnableSensitiveDataLogging</c>, interceptors,
    ///     <c>UseQueryTrackingBehavior</c>, and the like. Runs after the provider is wired up.
    /// </summary>
    public Action<DbContextOptionsBuilder>? ConfigureDbContext { get; set; }

    /// <summary>
    ///     Configure the Npgsql (PostgreSQL) provider — the golden-standard Npgsql/EF surface:
    ///     <c>EnableRetryOnFailure()</c> (resilient connections; this storage wraps its transactions in
    ///     the execution strategy, so retry is safe), <c>CommandTimeout</c>, <c>MigrationsHistoryTable</c>,
    ///     etc. Ignored on MySQL. OptunaSharp's own enum mapping is applied first, then this.
    /// </summary>
    public Action<NpgsqlDbContextOptionsBuilder>? ConfigureNpgsql { get; set; }

    /// <summary>
    ///     Configure the Pomelo (MySQL) provider — <c>EnableRetryOnFailure()</c> (safe; see
    ///     <see cref="ConfigureNpgsql" />), <c>CommandTimeout</c>, <c>EnableStringComparisonTranslations</c>,
    ///     etc. Ignored on PostgreSQL.
    /// </summary>
    public Action<MySqlDbContextOptionsBuilder>? ConfigureMySql { get; set; }

    /// <summary>
    ///     The MySQL <see cref="ServerVersion" /> to target. When <c>null</c> (the default), OptunaSharp
    ///     opens a probe connection to auto-detect it (<c>ServerVersion.AutoDetect</c>); set it
    ///     explicitly (e.g. <c>new MySqlServerVersion(new Version(8, 0, 36))</c>) to skip that probe.
    ///     Ignored on PostgreSQL.
    /// </summary>
    public ServerVersion? MySqlServerVersion { get; set; }
}

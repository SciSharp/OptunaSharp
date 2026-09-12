using Microsoft.EntityFrameworkCore;

namespace OptunaSharp.Storages.EFCore;

/// <summary>
///     The EF Core <see cref="DbContext" /> over Optuna's RDB schema. One context is short-lived per
///     storage operation (EF Core contexts are not thread-safe); <see cref="RDBStorage" /> serializes
///     access and builds the options once.
///
///     <para>The schema is created by <see cref="RdbSchema" /> to match SQLAlchemy verbatim, so this
///     context only maps the entities for querying. The one provider-specific concern is the four enum
///     columns: on PostgreSQL they are native <c>ENUM</c> types handled by Npgsql's <c>MapEnum</c>
///     (configured on the options by <see cref="RDBStorage" />); on MySQL they are inline
///     <c>ENUM(...)</c> columns whose member name is stored via a string conversion configured here.</para>
/// </summary>
internal sealed class OptunaRdbContext : DbContext
{
    /// <summary>Creates an operation-scoped context with the configured relational provider.</summary>
    /// <param name="options">The storage and provider configuration applied before schema initialization.</param>
    public OptunaRdbContext(DbContextOptions<OptunaRdbContext> options) : base(options)
    {
    }

    /// <summary>The queryable and tracked set of <see cref="StudyEntity" /> records for this operation.</summary>
    public DbSet<StudyEntity> Studies => Set<StudyEntity>();
    /// <summary>The queryable and tracked set of <see cref="StudyDirectionEntity" /> records for this operation.</summary>
    public DbSet<StudyDirectionEntity> StudyDirections => Set<StudyDirectionEntity>();
    /// <summary>The queryable and tracked set of <see cref="StudyUserAttributeEntity" /> records for this operation.</summary>
    public DbSet<StudyUserAttributeEntity> StudyUserAttributes => Set<StudyUserAttributeEntity>();
    /// <summary>The queryable and tracked set of <see cref="StudySystemAttributeEntity" /> records for this operation.</summary>
    public DbSet<StudySystemAttributeEntity> StudySystemAttributes => Set<StudySystemAttributeEntity>();
    /// <summary>The queryable and tracked set of <see cref="TrialEntity" /> records for this operation.</summary>
    public DbSet<TrialEntity> Trials => Set<TrialEntity>();
    /// <summary>The queryable and tracked set of <see cref="TrialUserAttributeEntity" /> records for this operation.</summary>
    public DbSet<TrialUserAttributeEntity> TrialUserAttributes => Set<TrialUserAttributeEntity>();
    /// <summary>The queryable and tracked set of <see cref="TrialSystemAttributeEntity" /> records for this operation.</summary>
    public DbSet<TrialSystemAttributeEntity> TrialSystemAttributes => Set<TrialSystemAttributeEntity>();
    /// <summary>The queryable and tracked set of <see cref="TrialParamEntity" /> records for this operation.</summary>
    public DbSet<TrialParamEntity> TrialParams => Set<TrialParamEntity>();
    /// <summary>The queryable and tracked set of <see cref="TrialValueEntity" /> records for this operation.</summary>
    public DbSet<TrialValueEntity> TrialValues => Set<TrialValueEntity>();
    /// <summary>The queryable and tracked set of <see cref="TrialIntermediateValueEntity" /> records for this operation.</summary>
    public DbSet<TrialIntermediateValueEntity> TrialIntermediateValues => Set<TrialIntermediateValueEntity>();
    /// <summary>The queryable and tracked set of <see cref="TrialHeartbeatEntity" /> records for this operation.</summary>
    public DbSet<TrialHeartbeatEntity> TrialHeartbeats => Set<TrialHeartbeatEntity>();
    /// <summary>The queryable and tracked set of <see cref="VersionInfoEntity" /> records for this operation.</summary>
    public DbSet<VersionInfoEntity> VersionInfos => Set<VersionInfoEntity>();
    /// <summary>The queryable and tracked set of <see cref="AlembicVersionEntity" /> records for this operation.</summary>
    public DbSet<AlembicVersionEntity> AlembicVersions => Set<AlembicVersionEntity>();

    /// <inheritdoc />
    protected override void OnModelCreating(ModelBuilder b)
    {
        // On MySQL each enum column is an inline ENUM('NAME',...) whose *name* is stored, so convert
        // the CLR enum to its member-name string. On PostgreSQL the columns are native ENUM types that
        // Npgsql maps directly (RDBStorage registers them via MapEnum) — no conversion there, or the
        // text would be rejected as "column is of type studydirection but expression is of type text".
        if (Database.IsMySql())
        {
            b.Entity<StudyDirectionEntity>().Property(x => x.Direction).HasConversion<string>();
            b.Entity<TrialEntity>().Property(x => x.State).HasConversion<string>();
            b.Entity<TrialValueEntity>().Property(x => x.ValueType).HasConversion<string>();
            b.Entity<TrialIntermediateValueEntity>()
                .Property(x => x.IntermediateValueType).HasConversion<string>();
        }

        if (Database.IsNpgsql())
        {
            // Optuna stores naive UTC in TIMESTAMP WITHOUT TIME ZONE. Npgsql defaults a CLR DateTime to
            // `timestamp with time zone` (requiring Kind=Utc); pin the column type so a naive
            // (Kind=Unspecified) UTC value maps to the plain timestamp SQLAlchemy created.
            const string ts = "timestamp without time zone";
            b.Entity<TrialEntity>().Property(x => x.DatetimeStart).HasColumnType(ts);
            b.Entity<TrialEntity>().Property(x => x.DatetimeComplete).HasColumnType(ts);
            b.Entity<TrialHeartbeatEntity>().Property(x => x.Heartbeat).HasColumnType(ts);
        }
    }
}

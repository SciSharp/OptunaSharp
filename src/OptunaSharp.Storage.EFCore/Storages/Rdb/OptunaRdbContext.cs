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
    public OptunaRdbContext(DbContextOptions<OptunaRdbContext> options) : base(options)
    {
    }

    public DbSet<StudyEntity> Studies => Set<StudyEntity>();
    public DbSet<StudyDirectionEntity> StudyDirections => Set<StudyDirectionEntity>();
    public DbSet<StudyUserAttributeEntity> StudyUserAttributes => Set<StudyUserAttributeEntity>();
    public DbSet<StudySystemAttributeEntity> StudySystemAttributes => Set<StudySystemAttributeEntity>();
    public DbSet<TrialEntity> Trials => Set<TrialEntity>();
    public DbSet<TrialUserAttributeEntity> TrialUserAttributes => Set<TrialUserAttributeEntity>();
    public DbSet<TrialSystemAttributeEntity> TrialSystemAttributes => Set<TrialSystemAttributeEntity>();
    public DbSet<TrialParamEntity> TrialParams => Set<TrialParamEntity>();
    public DbSet<TrialValueEntity> TrialValues => Set<TrialValueEntity>();
    public DbSet<TrialIntermediateValueEntity> TrialIntermediateValues => Set<TrialIntermediateValueEntity>();
    public DbSet<TrialHeartbeatEntity> TrialHeartbeats => Set<TrialHeartbeatEntity>();
    public DbSet<VersionInfoEntity> VersionInfos => Set<VersionInfoEntity>();
    public DbSet<AlembicVersionEntity> AlembicVersions => Set<AlembicVersionEntity>();

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

using System;
using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;

namespace OptunaSharp.Storages.EFCore;

// POCO entities mapping Optuna's SQLAlchemy models (optuna/storages/_rdb/models.py) one-to-one.
// Column and table names match the schema SQLAlchemy generates (verified against a real Optuna
// PostgreSQL/MySQL database); the keys, unique indexes, enum mappings and datetime handling are
// configured in OptunaRdbContext.OnModelCreating. The schema itself is created by RdbSchema, not by
// EF Core migrations, so these types only drive query/CRUD generation.

/// <summary>An Optuna study row with its storage identifier and unique display name.</summary>
[Table("studies")]
internal sealed class StudyEntity
{
    /// <summary>The storage-wide study identifier linking this row to its study.</summary>
    [Key] [Column("study_id")] public int StudyId { get; set; }
    /// <summary>The unique, user-visible name of the study.</summary>
    [Column("study_name")] public string StudyName { get; set; } = "";
}

/// <summary>The direction of one zero-based objective within a study.</summary>
[Table("study_directions")]
internal sealed class StudyDirectionEntity
{
    /// <summary>The database-generated primary key for this row.</summary>
    [Key] [Column("study_direction_id")] public int StudyDirectionId { get; set; }
    /// <summary>The stored optimization direction for this objective.</summary>
    [Column("direction")] public DbStudyDirection Direction { get; set; }
    /// <summary>The storage-wide study identifier linking this row to its study.</summary>
    [Column("study_id")] public int StudyId { get; set; }
    /// <summary>The zero-based objective index within the study.</summary>
    [Column("objective")] public int Objective { get; set; }
}

/// <summary>A JSON-serialized user attribute belonging to a study.</summary>
[Table("study_user_attributes")]
internal sealed class StudyUserAttributeEntity
{
    /// <summary>The database-generated primary key for this row.</summary>
    [Key] [Column("study_user_attribute_id")] public int StudyUserAttributeId { get; set; }
    /// <summary>The storage-wide study identifier linking this row to its study.</summary>
    [Column("study_id")] public int? StudyId { get; set; }
    /// <summary>The case-sensitive attribute name.</summary>
    [Column("key")] public string? Key { get; set; }
    /// <summary>The attribute value serialized using Optuna-compatible JSON.</summary>
    [Column("value_json")] public string? ValueJson { get; set; }
}

/// <summary>A JSON-serialized system attribute belonging to a study.</summary>
[Table("study_system_attributes")]
internal sealed class StudySystemAttributeEntity
{
    /// <summary>The database-generated primary key for this row.</summary>
    [Key] [Column("study_system_attribute_id")] public int StudySystemAttributeId { get; set; }
    /// <summary>The storage-wide study identifier linking this row to its study.</summary>
    [Column("study_id")] public int? StudyId { get; set; }
    /// <summary>The case-sensitive attribute name.</summary>
    [Column("key")] public string? Key { get; set; }
    /// <summary>The attribute value serialized using Optuna-compatible JSON.</summary>
    [Column("value_json")] public string? ValueJson { get; set; }
}

/// <summary>The trial identity, lifecycle state, and timestamps stored separately from its child records.</summary>
[Table("trials")]
internal sealed class TrialEntity
{
    /// <summary>The storage-wide trial identifier linking this row to its trial.</summary>
    [Key] [Column("trial_id")] public int TrialId { get; set; }
    /// <summary>The zero-based trial number within its study.</summary>
    [Column("number")] public int? Number { get; set; }
    /// <summary>The storage-wide study identifier linking this row to its study.</summary>
    [Column("study_id")] public int? StudyId { get; set; }
    /// <summary>The persisted lifecycle state of the trial.</summary>
    [Column("state")] public DbTrialState State { get; set; }
    /// <summary>The trial start time stored as timezone-free UTC, or null before starting.</summary>
    [Column("datetime_start")] public DateTime? DatetimeStart { get; set; }
    /// <summary>The completion time stored as timezone-free UTC, or null while unfinished.</summary>
    [Column("datetime_complete")] public DateTime? DatetimeComplete { get; set; }
}

/// <summary>A JSON-serialized user attribute belonging to a trial.</summary>
[Table("trial_user_attributes")]
internal sealed class TrialUserAttributeEntity
{
    /// <summary>The database-generated primary key for this row.</summary>
    [Key] [Column("trial_user_attribute_id")] public int TrialUserAttributeId { get; set; }
    /// <summary>The storage-wide trial identifier linking this row to its trial.</summary>
    [Column("trial_id")] public int? TrialId { get; set; }
    /// <summary>The case-sensitive attribute name.</summary>
    [Column("key")] public string? Key { get; set; }
    /// <summary>The attribute value serialized using Optuna-compatible JSON.</summary>
    [Column("value_json")] public string? ValueJson { get; set; }
}

/// <summary>A JSON-serialized system attribute belonging to a trial.</summary>
[Table("trial_system_attributes")]
internal sealed class TrialSystemAttributeEntity
{
    /// <summary>The database-generated primary key for this row.</summary>
    [Key] [Column("trial_system_attribute_id")] public int TrialSystemAttributeId { get; set; }
    /// <summary>The storage-wide trial identifier linking this row to its trial.</summary>
    [Column("trial_id")] public int? TrialId { get; set; }
    /// <summary>The case-sensitive attribute name.</summary>
    [Column("key")] public string? Key { get; set; }
    /// <summary>The attribute value serialized using Optuna-compatible JSON.</summary>
    [Column("value_json")] public string? ValueJson { get; set; }
}

/// <summary>A trial parameter stored in its distribution's internal numeric representation.</summary>
[Table("trial_params")]
internal sealed class TrialParamEntity
{
    /// <summary>The database-generated primary key for this row.</summary>
    [Key] [Column("param_id")] public int ParamId { get; set; }
    /// <summary>The storage-wide trial identifier linking this row to its trial.</summary>
    [Column("trial_id")] public int? TrialId { get; set; }
    /// <summary>The case-sensitive suggested parameter name.</summary>
    [Column("param_name")] public string? ParamName { get; set; }
    /// <summary>The parameter value in its distribution's internal numeric representation.</summary>
    [Column("param_value")] public double? ParamValue { get; set; }
    /// <summary>The Optuna JSON representation of the parameter distribution.</summary>
    [Column("distribution_json")] public string? DistributionJson { get; set; }
}

/// <summary>One objective value represented by a numeric column and a nonfinite-value tag.</summary>
[Table("trial_values")]
internal sealed class TrialValueEntity
{
    /// <summary>The database-generated primary key for this row.</summary>
    [Key] [Column("trial_value_id")] public int TrialValueId { get; set; }
    /// <summary>The storage-wide trial identifier linking this row to its trial.</summary>
    [Column("trial_id")] public int TrialId { get; set; }
    /// <summary>The zero-based objective index within the study.</summary>
    [Column("objective")] public int Objective { get; set; }
    /// <summary>The finite numeric objective column; infinities are represented by the accompanying tag.</summary>
    [Column("value")] public double? Value { get; set; }
    /// <summary>The tag distinguishing finite objective values from positive and negative infinity.</summary>
    [Column("value_type")] public DbTrialValueType ValueType { get; set; }
}

/// <summary>One intermediate report represented by a step, numeric value, and nonfinite-value tag.</summary>
[Table("trial_intermediate_values")]
internal sealed class TrialIntermediateValueEntity
{
    /// <summary>The database-generated primary key for this row.</summary>
    [Key] [Column("trial_intermediate_value_id")] public int TrialIntermediateValueId { get; set; }
    /// <summary>The storage-wide trial identifier linking this row to its trial.</summary>
    [Column("trial_id")] public int TrialId { get; set; }
    /// <summary>The zero-based intermediate-report step.</summary>
    [Column("step")] public int Step { get; set; }
    /// <summary>The finite numeric report column; nonfinite values use the accompanying tag.</summary>
    [Column("intermediate_value")] public double? IntermediateValue { get; set; }
    /// <summary>The tag distinguishing finite intermediate values, infinities, and NaN.</summary>
    [Column("intermediate_value_type")] public DbTrialIntermediateValueType IntermediateValueType { get; set; }
}

/// <summary>The most recent heartbeat associated with a running trial.</summary>
[Table("trial_heartbeats")]
internal sealed class TrialHeartbeatEntity
{
    /// <summary>The database-generated primary key for this row.</summary>
    [Key] [Column("trial_heartbeat_id")] public int TrialHeartbeatId { get; set; }
    /// <summary>The storage-wide trial identifier linking this row to its trial.</summary>
    [Column("trial_id")] public int TrialId { get; set; }
    /// <summary>The latest heartbeat stored as timezone-free UTC.</summary>
    [Column("heartbeat")] public DateTime Heartbeat { get; set; }
}

/// <summary>The singleton informational schema-version and library-version record.</summary>
[Table("version_info")]
internal sealed class VersionInfoEntity
{
    /// <summary>The fixed singleton identifier used by the version-info row.</summary>
    [Key]
    [Column("version_info_id")]
    [DatabaseGenerated(DatabaseGeneratedOption.None)]
    public int VersionInfoId { get; set; }

    /// <summary>The legacy integer schema version retained for compatibility diagnostics.</summary>
    [Column("schema_version")] public int? SchemaVersion { get; set; }
    /// <summary>The library version that created the informational version record.</summary>
    [Column("library_version")] public string? LibraryVersion { get; set; }
}

/// <summary>The current Alembic migration revision recorded by the database.</summary>
[Table("alembic_version")]
internal sealed class AlembicVersionEntity
{
    /// <summary>The Alembic revision identifying the database's current schema.</summary>
    [Key] [Column("version_num")] public string VersionNum { get; set; } = "";
}

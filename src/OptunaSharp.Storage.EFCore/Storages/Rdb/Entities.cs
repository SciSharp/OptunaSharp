using System;
using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;

namespace OptunaSharp.Storages.EFCore;

// POCO entities mapping Optuna's SQLAlchemy models (optuna/storages/_rdb/models.py) one-to-one.
// Column and table names match the schema SQLAlchemy generates (verified against a real Optuna
// PostgreSQL/MySQL database); the keys, unique indexes, enum mappings and datetime handling are
// configured in OptunaRdbContext.OnModelCreating. The schema itself is created by RdbSchema, not by
// EF Core migrations, so these types only drive query/CRUD generation.

[Table("studies")]
internal sealed class StudyEntity
{
    [Key] [Column("study_id")] public int StudyId { get; set; }
    [Column("study_name")] public string StudyName { get; set; } = "";
}

[Table("study_directions")]
internal sealed class StudyDirectionEntity
{
    [Key] [Column("study_direction_id")] public int StudyDirectionId { get; set; }
    [Column("direction")] public DbStudyDirection Direction { get; set; }
    [Column("study_id")] public int StudyId { get; set; }
    [Column("objective")] public int Objective { get; set; }
}

[Table("study_user_attributes")]
internal sealed class StudyUserAttributeEntity
{
    [Key] [Column("study_user_attribute_id")] public int StudyUserAttributeId { get; set; }
    [Column("study_id")] public int? StudyId { get; set; }
    [Column("key")] public string? Key { get; set; }
    [Column("value_json")] public string? ValueJson { get; set; }
}

[Table("study_system_attributes")]
internal sealed class StudySystemAttributeEntity
{
    [Key] [Column("study_system_attribute_id")] public int StudySystemAttributeId { get; set; }
    [Column("study_id")] public int? StudyId { get; set; }
    [Column("key")] public string? Key { get; set; }
    [Column("value_json")] public string? ValueJson { get; set; }
}

[Table("trials")]
internal sealed class TrialEntity
{
    [Key] [Column("trial_id")] public int TrialId { get; set; }
    [Column("number")] public int? Number { get; set; }
    [Column("study_id")] public int? StudyId { get; set; }
    [Column("state")] public DbTrialState State { get; set; }
    [Column("datetime_start")] public DateTime? DatetimeStart { get; set; }
    [Column("datetime_complete")] public DateTime? DatetimeComplete { get; set; }
}

[Table("trial_user_attributes")]
internal sealed class TrialUserAttributeEntity
{
    [Key] [Column("trial_user_attribute_id")] public int TrialUserAttributeId { get; set; }
    [Column("trial_id")] public int? TrialId { get; set; }
    [Column("key")] public string? Key { get; set; }
    [Column("value_json")] public string? ValueJson { get; set; }
}

[Table("trial_system_attributes")]
internal sealed class TrialSystemAttributeEntity
{
    [Key] [Column("trial_system_attribute_id")] public int TrialSystemAttributeId { get; set; }
    [Column("trial_id")] public int? TrialId { get; set; }
    [Column("key")] public string? Key { get; set; }
    [Column("value_json")] public string? ValueJson { get; set; }
}

[Table("trial_params")]
internal sealed class TrialParamEntity
{
    [Key] [Column("param_id")] public int ParamId { get; set; }
    [Column("trial_id")] public int? TrialId { get; set; }
    [Column("param_name")] public string? ParamName { get; set; }
    [Column("param_value")] public double? ParamValue { get; set; }
    [Column("distribution_json")] public string? DistributionJson { get; set; }
}

[Table("trial_values")]
internal sealed class TrialValueEntity
{
    [Key] [Column("trial_value_id")] public int TrialValueId { get; set; }
    [Column("trial_id")] public int TrialId { get; set; }
    [Column("objective")] public int Objective { get; set; }
    [Column("value")] public double? Value { get; set; }
    [Column("value_type")] public DbTrialValueType ValueType { get; set; }
}

[Table("trial_intermediate_values")]
internal sealed class TrialIntermediateValueEntity
{
    [Key] [Column("trial_intermediate_value_id")] public int TrialIntermediateValueId { get; set; }
    [Column("trial_id")] public int TrialId { get; set; }
    [Column("step")] public int Step { get; set; }
    [Column("intermediate_value")] public double? IntermediateValue { get; set; }
    [Column("intermediate_value_type")] public DbTrialIntermediateValueType IntermediateValueType { get; set; }
}

[Table("trial_heartbeats")]
internal sealed class TrialHeartbeatEntity
{
    [Key] [Column("trial_heartbeat_id")] public int TrialHeartbeatId { get; set; }
    [Column("trial_id")] public int TrialId { get; set; }
    [Column("heartbeat")] public DateTime Heartbeat { get; set; }
}

[Table("version_info")]
internal sealed class VersionInfoEntity
{
    [Key]
    [Column("version_info_id")]
    [DatabaseGenerated(DatabaseGeneratedOption.None)]
    public int VersionInfoId { get; set; }

    [Column("schema_version")] public int? SchemaVersion { get; set; }
    [Column("library_version")] public string? LibraryVersion { get; set; }
}

[Table("alembic_version")]
internal sealed class AlembicVersionEntity
{
    [Key] [Column("version_num")] public string VersionNum { get; set; } = "";
}

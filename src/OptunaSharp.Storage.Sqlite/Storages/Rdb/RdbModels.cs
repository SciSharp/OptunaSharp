using System;
using System.Collections.Generic;
using System.Globalization;
using NumSharp;

namespace OptunaSharp.Storages;

/// <summary>
///     Schema constants, DDL, enum ↔ stored-string maps and the finite/±inf/NaN stored-value
///     representation. Port of <c>optuna/storages/_rdb/models.py</c> for the SQLite backend.
///
///     <para>Optuna's SQLite database is produced by SQLAlchemy's <c>metadata.create_all</c>. The
///     DDL below reproduces that schema exactly (verified against a database created by real Optuna
///     5.0.0.dev, and cross-checked against <c>rustuna_storage/src/sqlite3_schema.sql</c>). Every
///     statement uses <c>IF NOT EXISTS</c> so opening a database created by Python Optuna (or by an
///     earlier OptunaSharp run) is a no-op, which is what makes the same <c>.db</c> file usable from
///     both languages.</para>
/// </summary>
internal static class RdbModels
{
    /// <summary>Python <c>models.SCHEMA_VERSION</c> ("Don't modify this version number anymore.").</summary>
    internal const int SchemaVersion = 12;

    /// <summary>Python <c>models.MAX_INDEXED_STRING_LENGTH</c>.</summary>
    internal const int MaxIndexedStringLength = 512;

    /// <summary>Python <c>models.MAX_VERSION_LENGTH</c>.</summary>
    internal const int MaxVersionLength = 256;

    /// <summary>Python <c>models.NOT_FOUND_MSG</c>.</summary>
    internal const string NotFoundMsg = "Record does not exist.";

    /// <summary>
    ///     The alembic head revision the schema is stamped with — the leaf of the migration chain in
    ///     <c>optuna/storages/_rdb/alembic/versions</c> (nothing lists it as a <c>down_revision</c>).
    ///     Pinned so Python Optuna's compatibility check (which compares the stored alembic revision
    ///     to its own head) passes when it opens an OptunaSharp-created database.
    /// </summary>
    internal const string HeadVersion = "v3.2.0.a";

    /// <summary>
    ///     The library version string written into <c>version_info</c> (Python stores
    ///     <c>optuna.version.__version__</c>). This field is informational only — it is never
    ///     overwritten by a reader and appears solely in the incompatibility error message — so a
    ///     self-identifying OptunaSharp value is used.
    /// </summary>
    internal static readonly string LibraryVersion = "OptunaSharp " + OptunaSharpVersion.Version;

    /// <summary>
    ///     DDL for <c>version_info</c> — the schema-version bookkeeping table. Named so it is the single
    ///     source of truth for both <see cref="SchemaStatements" /> and the <c>InitVersionInfo</c> guard
    ///     that materializes it under <c>skip_table_creation</c> (mirroring Python's ORM insert).
    /// </summary>
    internal const string VersionInfoTableDdl =
        """
        CREATE TABLE IF NOT EXISTS version_info (
            version_info_id INTEGER NOT NULL,
            schema_version INTEGER,
            library_version VARCHAR(256),
            PRIMARY KEY (version_info_id),
            CHECK (version_info_id=1)
        )
        """;

    /// <summary>
    ///     DDL for <c>alembic_version</c> — the migration-revision marker table. Single source of truth
    ///     for both <see cref="SchemaStatements" /> and the <c>InitAlembic</c> guard that materializes it
    ///     (mirroring alembic's <c>stamp</c>, which creates the table when absent).
    /// </summary>
    internal const string AlembicVersionTableDdl =
        """
        CREATE TABLE IF NOT EXISTS alembic_version (
            version_num VARCHAR(32) NOT NULL,
            CONSTRAINT alembic_version_pkc PRIMARY KEY (version_num)
        )
        """;

    /// <summary>
    ///     The full schema, as the ordered list of statements SQLAlchemy's <c>create_all</c> emits
    ///     for the SQLite dialect. Executed once per connection with <c>IF NOT EXISTS</c> semantics.
    /// </summary>
    internal static readonly string[] SchemaStatements =
    {
        // studies
        """
        CREATE TABLE IF NOT EXISTS studies (
            study_id INTEGER NOT NULL,
            study_name VARCHAR(512) NOT NULL,
            PRIMARY KEY (study_id)
        )
        """,
        "CREATE UNIQUE INDEX IF NOT EXISTS ix_studies_study_name ON studies (study_name)",

        // version_info
        VersionInfoTableDdl,

        // study_directions
        """
        CREATE TABLE IF NOT EXISTS study_directions (
            study_direction_id INTEGER NOT NULL,
            direction VARCHAR(8) NOT NULL,
            study_id INTEGER NOT NULL,
            objective INTEGER NOT NULL,
            PRIMARY KEY (study_direction_id),
            UNIQUE (study_id, objective),
            FOREIGN KEY(study_id) REFERENCES studies (study_id)
        )
        """,

        // trials
        """
        CREATE TABLE IF NOT EXISTS trials (
            trial_id INTEGER NOT NULL,
            number INTEGER,
            study_id INTEGER,
            state VARCHAR(8) NOT NULL,
            datetime_start DATETIME,
            datetime_complete DATETIME,
            PRIMARY KEY (trial_id),
            FOREIGN KEY(study_id) REFERENCES studies (study_id)
        )
        """,
        "CREATE INDEX IF NOT EXISTS ix_trials_study_id ON trials (study_id)",

        // alembic_version
        AlembicVersionTableDdl,

        // trial_heartbeats
        """
        CREATE TABLE IF NOT EXISTS trial_heartbeats (
            trial_heartbeat_id INTEGER NOT NULL,
            trial_id INTEGER NOT NULL,
            heartbeat DATETIME NOT NULL,
            PRIMARY KEY (trial_heartbeat_id),
            UNIQUE (trial_id),
            FOREIGN KEY(trial_id) REFERENCES trials (trial_id)
        )
        """,

        // study_user_attributes
        """
        CREATE TABLE IF NOT EXISTS study_user_attributes (
            study_user_attribute_id INTEGER NOT NULL,
            study_id INTEGER,
            "key" VARCHAR(512),
            value_json TEXT,
            PRIMARY KEY (study_user_attribute_id),
            UNIQUE (study_id, "key"),
            FOREIGN KEY(study_id) REFERENCES studies (study_id)
        )
        """,

        // study_system_attributes
        """
        CREATE TABLE IF NOT EXISTS study_system_attributes (
            study_system_attribute_id INTEGER NOT NULL,
            study_id INTEGER,
            "key" VARCHAR(512),
            value_json TEXT,
            PRIMARY KEY (study_system_attribute_id),
            UNIQUE (study_id, "key"),
            FOREIGN KEY(study_id) REFERENCES studies (study_id)
        )
        """,

        // trial_user_attributes
        """
        CREATE TABLE IF NOT EXISTS trial_user_attributes (
            trial_user_attribute_id INTEGER NOT NULL,
            trial_id INTEGER,
            "key" VARCHAR(512),
            value_json TEXT,
            PRIMARY KEY (trial_user_attribute_id),
            UNIQUE (trial_id, "key"),
            FOREIGN KEY(trial_id) REFERENCES trials (trial_id)
        )
        """,

        // trial_system_attributes
        """
        CREATE TABLE IF NOT EXISTS trial_system_attributes (
            trial_system_attribute_id INTEGER NOT NULL,
            trial_id INTEGER,
            "key" VARCHAR(512),
            value_json TEXT,
            PRIMARY KEY (trial_system_attribute_id),
            UNIQUE (trial_id, "key"),
            FOREIGN KEY(trial_id) REFERENCES trials (trial_id)
        )
        """,

        // trial_params
        """
        CREATE TABLE IF NOT EXISTS trial_params (
            param_id INTEGER NOT NULL,
            trial_id INTEGER,
            param_name VARCHAR(512),
            param_value FLOAT,
            distribution_json TEXT,
            PRIMARY KEY (param_id),
            UNIQUE (trial_id, param_name),
            FOREIGN KEY(trial_id) REFERENCES trials (trial_id)
        )
        """,

        // trial_values
        """
        CREATE TABLE IF NOT EXISTS trial_values (
            trial_value_id INTEGER NOT NULL,
            trial_id INTEGER NOT NULL,
            objective INTEGER NOT NULL,
            value FLOAT,
            value_type VARCHAR(7) NOT NULL,
            PRIMARY KEY (trial_value_id),
            UNIQUE (trial_id, objective),
            FOREIGN KEY(trial_id) REFERENCES trials (trial_id)
        )
        """,

        // trial_intermediate_values
        """
        CREATE TABLE IF NOT EXISTS trial_intermediate_values (
            trial_intermediate_value_id INTEGER NOT NULL,
            trial_id INTEGER NOT NULL,
            step INTEGER NOT NULL,
            intermediate_value FLOAT,
            intermediate_value_type VARCHAR(7) NOT NULL,
            PRIMARY KEY (trial_intermediate_value_id),
            UNIQUE (trial_id, step),
            FOREIGN KEY(trial_id) REFERENCES trials (trial_id)
        )
        """,
    };

    // ------------------------------------------------------------------ enum ↔ stored string

    /// <summary>Map a <see cref="StudyDirection" /> to the string SQLAlchemy stores (the enum name).</summary>
    internal static string DirectionToDb(StudyDirection direction) => direction.PythonName();

    /// <summary>Inverse of <see cref="DirectionToDb" />.</summary>
    internal static StudyDirection DirectionFromDb(string value) => value switch
    {
        "MINIMIZE" => StudyDirection.Minimize,
        "MAXIMIZE" => StudyDirection.Maximize,
        "NOT_SET" => StudyDirection.NotSet,
        _ => throw new ValueError($"Unknown study direction: {value}"),
    };

    /// <summary>Map a <see cref="TrialState" /> to the string SQLAlchemy stores (the enum name).</summary>
    internal static string StateToDb(TrialState state) => state.PythonName();

    /// <summary>Inverse of <see cref="StateToDb" />.</summary>
    internal static TrialState StateFromDb(string value) => value switch
    {
        "RUNNING" => TrialState.Running,
        "COMPLETE" => TrialState.Complete,
        "PRUNED" => TrialState.Pruned,
        "FAIL" => TrialState.Fail,
        "WAITING" => TrialState.Waiting,
        _ => throw new ValueError($"Unknown trial state: {value}"),
    };

    // ------------------------------------------------------------------ trial_values stored repr

    /// <summary>
    ///     Python <c>TrialValueModel.value_to_stored_repr</c>: split a value into (stored column,
    ///     value_type tag). ±inf are stored as NULL with a tag so the column stays a plain FLOAT.
    /// </summary>
    internal static (double? Value, string Type) ValueToStoredRepr(double value)
    {
        if (double.IsPositiveInfinity(value))
            return (null, "INF_POS");
        if (double.IsNegativeInfinity(value))
            return (null, "INF_NEG");
        return (value, "FINITE");
    }

    /// <summary>Python <c>TrialValueModel.stored_repr_to_value</c>.</summary>
    internal static double StoredReprToValue(double? value, string type) => type switch
    {
        "INF_POS" => double.PositiveInfinity,
        "INF_NEG" => double.NegativeInfinity,
        "FINITE" => value!.Value,
        _ => throw new ValueError($"Unknown value type: {type}"),
    };

    /// <summary>
    ///     Python <c>TrialIntermediateValueModel.intermediate_value_to_stored_repr</c> — like
    ///     <see cref="ValueToStoredRepr" /> but also carries a NaN tag.
    /// </summary>
    internal static (double? Value, string Type) IntermediateValueToStoredRepr(double value)
    {
        if (double.IsNaN(value))
            return (null, "NAN");
        if (double.IsPositiveInfinity(value))
            return (null, "INF_POS");
        if (double.IsNegativeInfinity(value))
            return (null, "INF_NEG");
        return (value, "FINITE");
    }

    /// <summary>Python <c>TrialIntermediateValueModel.stored_repr_to_intermediate_value</c>.</summary>
    internal static double StoredReprToIntermediateValue(double? value, string type) => type switch
    {
        "NAN" => double.NaN,
        "INF_POS" => double.PositiveInfinity,
        "INF_NEG" => double.NegativeInfinity,
        "FINITE" => value!.Value,
        _ => throw new ValueError($"Unknown intermediate value type: {type}"),
    };

    // ------------------------------------------------------------------ datetime (naive UTC text)

    private const string DateTimeFormat = "yyyy-MM-dd HH:mm:ss.ffffff";

    private static readonly string[] DateTimeParseFormats =
    {
        "yyyy-MM-dd HH:mm:ss.ffffff",
        "yyyy-MM-dd HH:mm:ss.fff",
        "yyyy-MM-dd HH:mm:ss",
    };

    /// <summary>
    ///     Format a <see cref="DateTime" /> the way SQLAlchemy stores it for SQLite: naive UTC in
    ///     <c>yyyy-MM-dd HH:mm:ss.ffffff</c> (always six microsecond digits). A local/unspecified
    ///     value is converted to UTC first (Python's setter does <c>astimezone(utc)</c>).
    /// </summary>
    internal static string FormatDatetime(DateTime value)
    {
        DateTime utc = value.Kind switch
        {
            DateTimeKind.Utc => value,
            DateTimeKind.Local => value.ToUniversalTime(),
            _ => DateTime.SpecifyKind(value, DateTimeKind.Local).ToUniversalTime(),
        };
        return utc.ToString(DateTimeFormat, CultureInfo.InvariantCulture);
    }

    /// <summary>
    ///     Parse a stored naive-UTC datetime string back to a local <see cref="DateTime" /> — the
    ///     inverse of <see cref="FormatDatetime" /> (Python's getter converts the stored UTC back to
    ///     naive local time when constructing a <see cref="FrozenTrial" />).
    /// </summary>
    internal static DateTime? ParseDatetime(string? value)
    {
        if (string.IsNullOrEmpty(value))
            return null;
        var utc = DateTime.SpecifyKind(
            DateTime.ParseExact(value, DateTimeParseFormats, CultureInfo.InvariantCulture, DateTimeStyles.None),
            DateTimeKind.Utc);
        return utc.ToLocalTime();
    }
}

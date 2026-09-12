using System;
using System.Globalization;
using NumSharp;

namespace OptunaSharp.Storages.EFCore;

/// <summary>
///     Schema constants, the DB enum types, enum ↔ stored-name maps, the finite/±inf/NaN stored-value
///     representation and the naive-UTC datetime handling. Port of the parts of
///     <c>optuna/storages/_rdb/models.py</c> that are dialect-independent, for the EF Core backend.
///
///     <para>Unlike the SQLite backend (which hand-writes the DDL SQLAlchemy generates), the schema
///     itself is created by <see cref="RdbSchema" /> per provider — reproducing SQLAlchemy's
///     PostgreSQL/MySQL output verbatim (native <c>ENUM</c> types on PostgreSQL, inline
///     <c>ENUM(...)</c> on MySQL) — and mapped by <see cref="OptunaRdbContext" />. The constants and
///     helpers below are shared by both providers.</para>
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
    ///     <c>optuna/storages/_rdb/alembic/versions</c>. Pinned so Python Optuna's compatibility check
    ///     passes when it opens an OptunaSharp-created database. Identical to the SQLite backend's pin.
    /// </summary>
    internal const string HeadVersion = "v3.2.0.a";

    /// <summary>
    ///     The library version string written into <c>version_info</c> (Python stores
    ///     <c>optuna.version.__version__</c>). Informational only — never overwritten by a reader —
    ///     so a self-identifying OptunaSharp value is used.
    /// </summary>
    internal static readonly string LibraryVersion = "OptunaSharp " + OptunaSharpVersion.Version;

    /// <summary>The known alembic revisions, in migration order (Python's alembic history).</summary>
    internal static readonly string[] KnownVersions =
    {
        "v0.9.0.a", "v1.2.0.a", "v1.3.0.a", "v2.4.0.a", "v2.6.0.a",
        "v3.0.0.a", "v3.0.0.b", "v3.0.0.c", "v3.0.0.d", "v3.2.0.a",
    };

    // ------------------------------------------------------------------ enum ↔ stored string

    /// <summary>Map a <see cref="StudyDirection" /> to the DB enum (whose name SQLAlchemy stores).</summary>
    /// <param name="direction">The objective direction used for ordering or representation conversion.</param>
    /// <returns>The corresponding direction label used by the database schema.</returns>
    internal static DbStudyDirection DirectionToDb(StudyDirection direction) => direction switch
    {
        StudyDirection.Minimize => DbStudyDirection.MINIMIZE,
        StudyDirection.Maximize => DbStudyDirection.MAXIMIZE,
        _ => DbStudyDirection.NOT_SET,
    };

    /// <summary>Inverse of <see cref="DirectionToDb" />.</summary>
    /// <param name="value">The value to serialize, persist, or convert to the requested representation.</param>
    /// <returns>The matching OptunaSharp objective direction.</returns>
    /// <exception cref="NumSharp.ValueError">The stored direction label is unknown.</exception>
    internal static StudyDirection DirectionFromDb(DbStudyDirection value) => value switch
    {
        DbStudyDirection.MINIMIZE => StudyDirection.Minimize,
        DbStudyDirection.MAXIMIZE => StudyDirection.Maximize,
        DbStudyDirection.NOT_SET => StudyDirection.NotSet,
        _ => throw new ValueError($"Unknown study direction: {value}"),
    };

    /// <summary>Map a <see cref="TrialState" /> to the DB enum (whose name SQLAlchemy stores).</summary>
    /// <param name="state">The trial lifecycle state to convert or persist.</param>
    /// <returns>The corresponding lifecycle-state label used by the database schema.</returns>
    /// <exception cref="NumSharp.ValueError">The trial-state value has no supported database representation.</exception>
    internal static DbTrialState StateToDb(TrialState state) => state switch
    {
        TrialState.Running => DbTrialState.RUNNING,
        TrialState.Complete => DbTrialState.COMPLETE,
        TrialState.Pruned => DbTrialState.PRUNED,
        TrialState.Fail => DbTrialState.FAIL,
        TrialState.Waiting => DbTrialState.WAITING,
        _ => throw new ValueError($"Unknown trial state: {state}"),
    };

    /// <summary>Inverse of <see cref="StateToDb" />.</summary>
    /// <param name="value">The value to serialize, persist, or convert to the requested representation.</param>
    /// <returns>The matching OptunaSharp trial lifecycle state.</returns>
    /// <exception cref="NumSharp.ValueError">The stored trial-state label is unknown.</exception>
    internal static TrialState StateFromDb(DbTrialState value) => value switch
    {
        DbTrialState.RUNNING => TrialState.Running,
        DbTrialState.COMPLETE => TrialState.Complete,
        DbTrialState.PRUNED => TrialState.Pruned,
        DbTrialState.FAIL => TrialState.Fail,
        DbTrialState.WAITING => TrialState.Waiting,
        _ => throw new ValueError($"Unknown trial state: {value}"),
    };

    // ------------------------------------------------------------------ trial_values stored repr

    /// <summary>
    ///     Python <c>TrialValueModel.value_to_stored_repr</c>: split a value into (stored column,
    ///     value_type tag). ±inf are stored as NULL with a tag so the column stays a plain FLOAT.
    /// </summary>
    /// <param name="value">The value to serialize, persist, or convert to the requested representation.</param>
    /// <returns>The nullable numeric column and a tag preserving finite values or either infinity.</returns>
    internal static (double? Value, DbTrialValueType Type) ValueToStoredRepr(double value)
    {
        if (double.IsPositiveInfinity(value))
            return (null, DbTrialValueType.INF_POS);
        if (double.IsNegativeInfinity(value))
            return (null, DbTrialValueType.INF_NEG);
        return (value, DbTrialValueType.FINITE);
    }

    /// <summary>Python <c>TrialValueModel.stored_repr_to_value</c>.</summary>
    /// <param name="value">The value to serialize, persist, or convert to the requested representation.</param>
    /// <param name="type">The database tag distinguishing finite values, infinities, and, for intermediate reports, NaN.</param>
    /// <returns>The objective value reconstructed from its numeric column and type tag.</returns>
    /// <exception cref="NumSharp.ValueError">The stored objective-value tag is unknown.</exception>
    /// <exception cref="System.InvalidOperationException">A FINITE tag is paired with a null numeric value.</exception>
    internal static double StoredReprToValue(double? value, DbTrialValueType type) => type switch
    {
        DbTrialValueType.INF_POS => double.PositiveInfinity,
        DbTrialValueType.INF_NEG => double.NegativeInfinity,
        DbTrialValueType.FINITE => value!.Value,
        _ => throw new ValueError($"Unknown value type: {type}"),
    };

    /// <summary>
    ///     Python <c>TrialIntermediateValueModel.intermediate_value_to_stored_repr</c> — like
    ///     <see cref="ValueToStoredRepr" /> but also carries a NaN tag.
    /// </summary>
    /// <param name="value">The value to serialize, persist, or convert to the requested representation.</param>
    /// <returns>The nullable numeric column and a tag preserving finite values, either infinity, or NaN.</returns>
    internal static (double? Value, DbTrialIntermediateValueType Type) IntermediateValueToStoredRepr(double value)
    {
        if (double.IsNaN(value))
            return (null, DbTrialIntermediateValueType.NAN);
        if (double.IsPositiveInfinity(value))
            return (null, DbTrialIntermediateValueType.INF_POS);
        if (double.IsNegativeInfinity(value))
            return (null, DbTrialIntermediateValueType.INF_NEG);
        return (value, DbTrialIntermediateValueType.FINITE);
    }

    /// <summary>Python <c>TrialIntermediateValueModel.stored_repr_to_intermediate_value</c>.</summary>
    /// <param name="value">The value to serialize, persist, or convert to the requested representation.</param>
    /// <param name="type">The database tag distinguishing finite values, infinities, and, for intermediate reports, NaN.</param>
    /// <returns>The intermediate value reconstructed from its numeric column and type tag.</returns>
    /// <exception cref="NumSharp.ValueError">The stored intermediate-value tag is unknown.</exception>
    /// <exception cref="System.InvalidOperationException">A FINITE tag is paired with a null numeric value.</exception>
    internal static double StoredReprToIntermediateValue(double? value, DbTrialIntermediateValueType type) => type switch
    {
        DbTrialIntermediateValueType.NAN => double.NaN,
        DbTrialIntermediateValueType.INF_POS => double.PositiveInfinity,
        DbTrialIntermediateValueType.INF_NEG => double.NegativeInfinity,
        DbTrialIntermediateValueType.FINITE => value!.Value,
        _ => throw new ValueError($"Unknown intermediate value type: {type}"),
    };

    // ------------------------------------------------------------------ datetime (naive UTC)

    /// <summary>
    ///     Convert a <see cref="DateTime" /> to the naive-UTC value SQLAlchemy stores (Python's setter
    ///     does <c>value.astimezone(utc).replace(tzinfo=None)</c>). The result carries
    ///     <see cref="DateTimeKind.Unspecified" /> so Npgsql/Pomelo map it to
    ///     <c>TIMESTAMP WITHOUT TIME ZONE</c> / <c>DATETIME</c> rather than a tz-aware column.
    /// </summary>
    /// <param name="value">The value to serialize, persist, or convert to the requested representation.</param>
    /// <returns>A UTC timestamp with unspecified DateTime kind, matching Optuna's timezone-free database column.</returns>
    internal static DateTime ToDbUtc(DateTime value)
    {
        DateTime utc = value.Kind switch
        {
            DateTimeKind.Utc => value,
            DateTimeKind.Local => value.ToUniversalTime(),
            _ => DateTime.SpecifyKind(value, DateTimeKind.Local).ToUniversalTime(),
        };
        return DateTime.SpecifyKind(utc, DateTimeKind.Unspecified);
    }

    /// <summary>
    ///     Inverse of <see cref="ToDbUtc" />: interpret a stored naive-UTC value as UTC and return it
    ///     as naive local time — Python's getter does
    ///     <c>value.replace(tzinfo=utc).astimezone().replace(tzinfo=None)</c>.
    /// </summary>
    /// <param name="value">The value to serialize, persist, or convert to the requested representation.</param>
    /// <returns>The local timestamp corresponding to the stored UTC value, or null when absent.</returns>
    internal static DateTime? FromDbUtc(DateTime? value)
    {
        if (value is null)
            return null;
        var utc = DateTime.SpecifyKind(value.Value, DateTimeKind.Utc);
        return DateTime.SpecifyKind(utc.ToLocalTime(), DateTimeKind.Unspecified);
    }

    /// <summary>The invariant culture used for portable database-value formatting.</summary>
    internal static readonly CultureInfo Inv = CultureInfo.InvariantCulture;
}

/// <summary>
///     The <c>studydirection</c> PostgreSQL enum / <c>ENUM('NOT_SET','MINIMIZE','MAXIMIZE')</c> MySQL
///     column. Member names are the labels SQLAlchemy emits from <c>Enum(StudyDirection)</c> — kept
///     verbatim so Npgsql's null name-translator and Pomelo's string conversion both round-trip them.
/// </summary>
internal enum DbStudyDirection
{
    /// <summary>No optimization direction has been assigned.</summary>
    NOT_SET,
    /// <summary>Smaller objective values are preferred.</summary>
    MINIMIZE,
    /// <summary>Larger objective values are preferred.</summary>
    MAXIMIZE,
}

/// <summary>The <c>trialstate</c> enum — labels from SQLAlchemy's <c>Enum(TrialState)</c>.</summary>
internal enum DbTrialState
{
    /// <summary>The objective is currently being evaluated.</summary>
    RUNNING,
    /// <summary>The objective completed successfully.</summary>
    COMPLETE,
    /// <summary>The trial ended early following a pruning decision.</summary>
    PRUNED,
    /// <summary>The objective failed before successful completion.</summary>
    FAIL,
    /// <summary>The trial is queued for a future worker.</summary>
    WAITING,
}

/// <summary>The <c>trialvaluetype</c> enum — labels from <c>TrialValueModel.TrialValueType</c>.</summary>
internal enum DbTrialValueType
{
    /// <summary>The numeric value is stored in the accompanying value column.</summary>
    FINITE,
    /// <summary>The value is positive infinity; its numeric column is null.</summary>
    INF_POS,
    /// <summary>The value is negative infinity; its numeric column is null.</summary>
    INF_NEG,
}

/// <summary>
///     The <c>trialintermediatevaluetype</c> enum — labels from
///     <c>TrialIntermediateValueModel.TrialIntermediateValueType</c> (adds <c>NAN</c>).
/// </summary>
internal enum DbTrialIntermediateValueType
{
    /// <summary>The numeric value is stored in the accompanying value column.</summary>
    FINITE,
    /// <summary>The value is positive infinity; its numeric column is null.</summary>
    INF_POS,
    /// <summary>The value is negative infinity; its numeric column is null.</summary>
    INF_NEG,
    /// <summary>The intermediate value is NaN; its numeric column is null.</summary>
    NAN,
}

using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Text;
using NumSharp;
using OptunaSharp.Distributions;
using OptunaSharp.Internal;

namespace OptunaSharp.Storages;

/// <summary>
///     The alembic migration chain, ported for the SQLite backend — the revision list plus the pure
///     value transforms Optuna's migrations apply. Mirrors
///     <c>optuna/storages/_rdb/alembic/versions/*</c> and the <c>_VersionManager</c> revision helpers
///     in <c>optuna/storages/_rdb/storage.py</c>. The migration <em>logic</em> lives on
///     <see cref="RDBStorage" /> itself (the partial in this file) because it needs the storage's SQL
///     helpers, exactly as OptunaSharp already folded <c>_VersionManager</c>'s init/compat methods
///     onto <see cref="RDBStorage" />.
/// </summary>
internal static class RdbMigrations
{
    /// <summary>
    ///     The migration revisions in apply order (base → head). Each is the <c>revision</c> id of one
    ///     file in <c>optuna/storages/_rdb/alembic/versions</c>; the chain is linear (every file's
    ///     <c>down_revision</c> is the previous one). <c>upgrade</c> replays every revision after the
    ///     database's current one.
    /// </summary>
    internal static readonly string[] RevisionsBaseToHead =
    {
        "v0.9.0.a", "v1.2.0.a", "v1.3.0.a", "v2.4.0.a", "v2.6.0.a",
        "v3.0.0.a", "v3.0.0.b", "v3.0.0.c", "v3.0.0.d", "v3.2.0.a",
    };

    /// <summary>The base (oldest) revision — alembic's <c>script.get_base()</c>, "v0.9.0.a".</summary>
    internal static string BaseVersion => RevisionsBaseToHead[0];

    // Python `RDB_MAX_FLOAT` / `RDB_MIN_FLOAT` = np.finfo(np.float32).max/min — the sentinel old Optuna
    // clamped ±inf to before the value_type tag existed. `(double)float.MaxValue` is bit-identical to
    // NumPy's 3.4028234663852886e+38.
    private const double RdbMaxFloat = float.MaxValue;
    private const double RdbMinFloat = -(double)float.MaxValue;

    /// <summary>numpy <c>np.isclose(a, b)</c> with the default tolerances (<c>rtol=1e-5, atol=1e-8</c>).</summary>
    private static bool IsClose(double a, double b) => Math.Abs(a - b) <= 1e-8 + 1e-5 * Math.Abs(b);

    /// <summary>
    ///     Port of <c>v3.0.0.c: IntermediateValueModel.intermediate_value_to_stored_repr</c> composed
    ///     with the migration's clamp detection — maps a stored intermediate value (possibly a clamped
    ///     ±inf sentinel or NULL) to the (value, value_type) pair the new schema uses.
    /// </summary>
    internal static (double? Value, string Type) RetagIntermediateValue(double? raw)
    {
        double value;
        if (raw is null || double.IsNaN(raw.Value))
            value = double.NaN;
        else if (IsClose(raw.Value, RdbMaxFloat) || double.IsPositiveInfinity(raw.Value))
            value = double.PositiveInfinity;
        else if (IsClose(raw.Value, RdbMinFloat) || double.IsNegativeInfinity(raw.Value))
            value = double.NegativeInfinity;
        else
            value = raw.Value;

        if (double.IsNaN(value))
            return (null, "NAN");
        if (double.IsPositiveInfinity(value))
            return (null, "INF_POS");
        if (double.IsNegativeInfinity(value))
            return (null, "INF_NEG");
        return (value, "FINITE");
    }

    /// <summary>Port of <c>v3.0.0.d</c>'s clamp detection + <c>value_to_stored_repr</c> (no NaN case).</summary>
    internal static (double? Value, string Type) RetagValue(double raw)
    {
        double value;
        if (IsClose(raw, RdbMaxFloat) || double.IsPositiveInfinity(raw))
            value = double.PositiveInfinity;
        else if (IsClose(raw, RdbMinFloat) || double.IsNegativeInfinity(raw))
            value = double.NegativeInfinity;
        else
            value = raw;

        if (double.IsPositiveInfinity(value))
            return (null, "INF_POS");
        if (double.IsNegativeInfinity(value))
            return (null, "INF_NEG");
        return (value, "FINITE");
    }
}

public sealed partial class RDBStorage
{
    // ------------------------------------------------------------------ version manager surface

    /// <summary>
    ///     Return the alembic revision the storage is currently stamped at (Python
    ///     <c>RDBStorage.get_current_version</c> / <c>_VersionManager.get_current_version</c>).
    /// </summary>
    public string GetCurrentVersion()
    {
        lock (_lock)
            return GetCurrentVersionNoLock();
    }

    private string GetCurrentVersionNoLock()
        => ScalarString("SELECT version_num FROM alembic_version", null)
            ?? throw new RuntimeError("The alembic_version table is empty.");

    /// <summary>Return the latest (head) schema revision — Python <c>get_head_version</c>.</summary>
    public string GetHeadVersion() => RdbModels.HeadVersion;

    /// <summary>
    ///     Return every known schema revision, newest first — Python <c>get_all_versions</c>, which
    ///     lists alembic's <c>walk_revisions()</c> (head → base).
    /// </summary>
    public List<string> GetAllVersions() => Enumerable.Reverse(RdbMigrations.RevisionsBaseToHead).ToList();

    /// <summary>
    ///     Upgrade the storage schema to the head revision — Python <c>RDBStorage.upgrade</c> /
    ///     <c>_VersionManager.upgrade</c> (which runs <c>alembic upgrade head</c> then refreshes
    ///     <c>version_info</c>). Every migration after the database's current revision is replayed in
    ///     order, faithfully reproducing what <c>alembic upgrade head</c> does to the SQLite file, then
    ///     <c>version_info</c> is set to the current schema_version / library_version.
    ///
    ///     <para>Like alembic on SQLite this is non-transactional DDL: <c>alembic_version</c> is bumped
    ///     after each migration, so a re-run resumes from the last completed one. The storage should be
    ///     opened with <c>skipCompatibilityCheck: true, skipTableCreation: true</c> (as the
    ///     <c>storage upgrade</c> CLI does) so an out-of-date schema opens without raising and without
    ///     the head tables being pre-created underneath the migration.</para>
    /// </summary>
    public void Upgrade()
    {
        lock (_lock)
        {
            string current = GetCurrentVersionNoLock();
            int startIndex = Array.IndexOf(RdbMigrations.RevisionsBaseToHead, current);
            if (startIndex < 0)
                // A revision this build does not know (a database newer than us). alembic would fail
                // to find a path to head; matching that, we refuse rather than corrupt the schema.
                throw new RuntimeError(
                    $"Can't locate revision identified by '{current}'.");

            for (int i = startIndex + 1; i < RdbMigrations.RevisionsBaseToHead.Length; i++)
            {
                string revision = RdbMigrations.RevisionsBaseToHead[i];
                ApplyMigration(revision);
                Exec("UPDATE alembic_version SET version_num = $p0", null, revision);
            }

            Exec(
                "UPDATE version_info SET schema_version = $p0, library_version = $p1 WHERE version_info_id = 1",
                null, RdbModels.SchemaVersion, RdbModels.LibraryVersion);
        }
    }

    private void ApplyMigration(string revision)
    {
        switch (revision)
        {
            case "v1.2.0.a": UpgradeV120a(); break;
            case "v1.3.0.a": UpgradeV130a(); break;
            case "v2.4.0.a": UpgradeV240a(); break;
            case "v2.6.0.a": UpgradeV260a(); break;
            case "v3.0.0.a": UpgradeV300a(); break;
            case "v3.0.0.b": UpgradeV300b(); break;
            case "v3.0.0.c": UpgradeV300c(); break;
            case "v3.0.0.d": UpgradeV300d(); break;
            case "v3.2.0.a": UpgradeV320a(); break;
            default:
                throw new RuntimeError($"No migration is registered for revision '{revision}'.");
        }
    }

    // ------------------------------------------------------------------ per-revision migrations

    // v1.2.0.a — widen the trials.state enum with WAITING. SQLAlchemy's Enum on SQLite is a bare
    // VARCHAR(8) with no CHECK constraint (verified against a real Optuna 0.9.0 database), so the enum
    // change carries no schema-level effect on SQLite; alembic's batch recreate of `trials` is a
    // structural identity. Nothing to do beyond the version bump the driver applies.
    private void UpgradeV120a()
    {
    }

    // v1.3.0.a — add trials.number, backfilling it from each trial's `_number` system attribute, then
    // dropping those attributes.
    private void UpgradeV130a()
    {
        Exec("ALTER TABLE trials ADD COLUMN number INTEGER", null);

        var mapping = new List<(long TrialId, long Number)>();
        Query(
            "SELECT trial_id, value_json FROM trial_system_attributes WHERE \"key\" = '_number'",
            null,
            r => mapping.Add((r.GetInt64(0), Convert.ToInt64(PyJson.Loads(r.GetString(1)), CultureInfo.InvariantCulture))));
        foreach (var (trialId, number) in mapping)
            Exec("UPDATE trials SET number = $p0 WHERE trial_id = $p1", null, number, trialId);

        Exec("DELETE FROM trial_system_attributes WHERE \"key\" = '_number'", null);
    }

    // v2.4.0.a — the big restructure: introduce study_directions + trial_intermediate_values, move the
    // single `studies.direction` / `trials.value` / step-keyed `trial_values` into them, and reshape
    // trial_values into an objective-keyed table.
    private void UpgradeV240a()
    {
        if (!TableExists("study_directions"))
            Exec(
                """
                CREATE TABLE study_directions (
                    study_direction_id INTEGER NOT NULL,
                    direction VARCHAR(8) NOT NULL,
                    study_id INTEGER NOT NULL,
                    objective INTEGER NOT NULL,
                    PRIMARY KEY (study_direction_id),
                    UNIQUE (study_id, objective),
                    FOREIGN KEY(study_id) REFERENCES studies (study_id)
                )
                """, null);

        // The v2.4.0.a form of trial_intermediate_values: intermediate_value is NOT NULL (made nullable
        // in v3.0.0.b) and there is no intermediate_value_type column yet (added in v3.0.0.c).
        if (!TableExists("trial_intermediate_values"))
            Exec(
                """
                CREATE TABLE trial_intermediate_values (
                    trial_intermediate_value_id INTEGER NOT NULL,
                    trial_id INTEGER NOT NULL,
                    step INTEGER NOT NULL,
                    intermediate_value FLOAT NOT NULL,
                    PRIMARY KEY (trial_intermediate_value_id),
                    UNIQUE (trial_id, step),
                    FOREIGN KEY(trial_id) REFERENCES trials (trial_id)
                )
                """, null);

        // studies.direction -> a study_directions row per study (objective 0, single-objective era).
        var directions = new List<(long StudyId, string Direction)>();
        Query("SELECT study_id, direction FROM studies", null,
            r => directions.Add((r.GetInt64(0), r.GetString(1))));
        foreach (var (studyId, direction) in directions)
            Exec(
                "INSERT INTO study_directions (direction, study_id, objective) VALUES ($p0, $p1, 0)",
                null, direction, studyId);

        // The old step-keyed trial_values (== intermediate values) -> trial_intermediate_values.
        var intermediates = new List<(long TrialId, double? Value, long Step)>();
        Query("SELECT trial_id, value, step FROM trial_values", null,
            r => intermediates.Add((r.GetInt64(0), r.IsDBNull(1) ? null : r.GetDouble(1), r.GetInt64(2))));
        foreach (var (trialId, value, step) in intermediates)
            Exec(
                "INSERT INTO trial_intermediate_values (trial_id, step, intermediate_value) VALUES ($p0, $p1, $p2)",
                null, trialId, step, (object?)value);

        Exec("DELETE FROM trial_values", null);

        // Append objective + its unique constraint (the old (trial_id, step) unique survives here and is
        // dropped together with `step` below), then repopulate trial_values from trials.value.
        RebuildTable(
            "trial_values",
            columns =>
            {
                columns.Add(new RdbColumn("objective", "INTEGER", NotNull: true, Default: null, Pk: false));
                return columns;
            },
            addUnique: new[] { new[] { "trial_id", "objective" } });

        var trialValues = new List<(long TrialId, double? Value)>();
        Query("SELECT trial_id, value FROM trials", null,
            r => trialValues.Add((r.GetInt64(0), r.IsDBNull(1) ? null : r.GetDouble(1))));
        foreach (var (trialId, value) in trialValues)
            Exec(
                "INSERT INTO trial_values (trial_id, value, objective) VALUES ($p0, $p1, 0)",
                null, trialId, (object?)value);

        RebuildTable("studies", columns => columns.Where(c => c.Name != "direction").ToList());
        RebuildTable("trial_values", columns => columns.Where(c => c.Name != "step").ToList());
        RebuildTable("trials", columns => columns.Where(c => c.Name != "value").ToList());
    }

    // v2.6.0.a — widen the JSON columns from VARCHAR(2048) to TEXT.
    private void UpgradeV260a()
    {
        foreach (var (table, column) in new[]
        {
            ("study_user_attributes", "value_json"),
            ("study_system_attributes", "value_json"),
            ("trial_user_attributes", "value_json"),
            ("trial_system_attributes", "value_json"),
            ("trial_params", "distribution_json"),
        })
        {
            RebuildTable(table, columns => columns
                .Select(c => c.Name == column ? c with { Type = "TEXT" } : c)
                .ToList());
        }
    }

    // v3.0.0.a — rewrite every distribution_json from the deprecated distribution classes to the
    // unified {Int,Float}Distribution form (Python `migrate_new_distribution`).
    private void UpgradeV300a()
    {
        var updates = new List<(long ParamId, string Json)>();
        Query("SELECT param_id, distribution_json FROM trial_params", null,
            r =>
            {
                var distribution = Distribution.FromJson(r.GetString(1));
                var migrated = Distribution.ConvertOldDistributionToNewDistribution(distribution, suppressWarning: true);
                updates.Add((r.GetInt64(0), Distribution.ToJson(migrated)));
            });
        foreach (var (paramId, json) in updates)
            Exec("UPDATE trial_params SET distribution_json = $p0 WHERE param_id = $p1", null, json, paramId);
    }

    // v3.0.0.b — reject/clean invalid trial_values (value=None on a COMPLETE trial), then make
    // intermediate_value nullable. The Optuna float-precision alter_columns are no-ops on SQLite (all
    // real numbers are 8-byte); trial_values.value is toggled NOT NULL here only to be made nullable
    // again in v3.0.0.d, so it is left alone (the head nullability is identical either way).
    private void UpgradeV300b()
    {
        long invalid = ScalarLong(
            "SELECT COUNT(*) FROM trial_values tv JOIN trials t ON t.trial_id = tv.trial_id " +
            "WHERE t.state = 'COMPLETE' AND tv.value IS NULL",
            null);
        if (invalid != 0)
            throw new ValueError("Found invalid trial_values records (value=None and state='COMPLETE')");
        Exec("DELETE FROM trial_values WHERE value IS NULL", null);

        RebuildTable("trial_intermediate_values", columns => columns
            .Select(c => c.Name == "intermediate_value" ? c with { NotNull = false } : c)
            .ToList());
    }

    // v3.0.0.c — add trial_intermediate_values.intermediate_value_type (default FINITE, then the
    // default removed), and retag clamped/NaN intermediate values as ±inf / NaN.
    private void UpgradeV300c()
    {
        Exec(
            "ALTER TABLE trial_intermediate_values " +
            "ADD COLUMN intermediate_value_type VARCHAR(7) DEFAULT 'FINITE' NOT NULL",
            null);
        RebuildTable("trial_intermediate_values", columns => columns
            .Select(c => c.Name == "intermediate_value_type" ? c with { Default = null } : c)
            .ToList());

        var rows = new List<(long Id, double? Value)>();
        Query(
            "SELECT trial_intermediate_value_id, intermediate_value FROM trial_intermediate_values " +
            "WHERE intermediate_value > 1e16 OR intermediate_value < -1e16 OR intermediate_value IS NULL",
            null,
            r => rows.Add((r.GetInt64(0), r.IsDBNull(1) ? null : r.GetDouble(1))));
        foreach (var (id, value) in rows)
        {
            var (stored, type) = RdbMigrations.RetagIntermediateValue(value);
            Exec(
                "UPDATE trial_intermediate_values SET intermediate_value = $p0, intermediate_value_type = $p1 " +
                "WHERE trial_intermediate_value_id = $p2",
                null, (object?)stored, type, id);
        }
    }

    // v3.0.0.d — add trial_values.value_type (default FINITE, then removed) and make value nullable;
    // retag clamped values as ±inf.
    private void UpgradeV300d()
    {
        Exec("ALTER TABLE trial_values ADD COLUMN value_type VARCHAR(7) DEFAULT 'FINITE' NOT NULL", null);
        RebuildTable("trial_values", columns => columns
            .Select(c => c.Name switch
            {
                "value_type" => c with { Default = null },
                "value" => c with { NotNull = false },
                _ => c,
            })
            .ToList());

        var rows = new List<(long Id, double Value)>();
        Query(
            "SELECT trial_value_id, value FROM trial_values WHERE value > 1e16 OR value < -1e16",
            null,
            r => rows.Add((r.GetInt64(0), r.GetDouble(1))));
        foreach (var (id, value) in rows)
        {
            var (stored, type) = RdbMigrations.RetagValue(value);
            Exec(
                "UPDATE trial_values SET value = $p0, value_type = $p1 WHERE trial_value_id = $p2",
                null, (object?)stored, type, id);
        }
    }

    // v3.2.0.a — index trials.study_id. Named as alembic does (`op.f("trials_study_id_key")`), which
    // differs from the `ix_trials_study_id` a fresh create_all emits — matching the migration, not the
    // fresh schema, is the parity target when replaying alembic.
    private void UpgradeV320a()
    {
        Exec("CREATE INDEX trials_study_id_key ON trials (study_id)", null);
    }

    // ------------------------------------------------------------------ schema-reflection rebuild

    /// <summary>A reflected column of a table (a subset of <c>PRAGMA table_info</c>).</summary>
    private sealed record RdbColumn(string Name, string Type, bool NotNull, string? Default, bool Pk);

    private bool TableExists(string table)
        => ScalarLong("SELECT COUNT(*) FROM sqlite_master WHERE type = 'table' AND name = $p0", null, table) > 0;

    /// <summary>
    ///     Rebuild a table the way alembic's SQLite "batch" mode does: create a replacement table with
    ///     the transformed columns, copy the shared columns, drop the original, and rename. Unique
    ///     constraints, foreign keys and plain indexes are reflected and reproduced; any that reference
    ///     a removed column are dropped (as SQLite does when a column disappears), and
    ///     <paramref name="addUnique" /> adds new ones. Column order is preserved from the reflected
    ///     table, so the result matches what <c>alembic upgrade</c> leaves behind (which is
    ///     order-sensitive and differs from a fresh <c>create_all</c>).
    /// </summary>
    private void RebuildTable(
        string table,
        Func<List<RdbColumn>, List<RdbColumn>> transformColumns,
        IReadOnlyList<string[]>? addUnique = null)
    {
        List<RdbColumn> oldColumns = ReflectColumns(table);
        List<RdbColumn> newColumns = transformColumns(oldColumns.ToList());
        var newNames = new HashSet<string>(newColumns.Select(c => c.Name));

        var uniques = ReflectUniqueConstraints(table).Where(u => u.All(newNames.Contains)).ToList();
        if (addUnique is not null)
            uniques.AddRange(addUnique.Select(u => u.ToArray()));
        var indexes = ReflectPlainIndexes(table).Where(i => i.Columns.All(newNames.Contains)).ToList();
        var foreignKeys = ReflectForeignKeys(table).Where(f => newNames.Contains(f.Column)).ToList();

        string tmp = "_optunasharp_new_" + table;
        Exec($"DROP TABLE IF EXISTS \"{tmp}\"", null);

        var lines = new List<string>();
        string? primaryKey = null;
        foreach (var c in newColumns)
        {
            var def = new StringBuilder($"\"{c.Name}\" {c.Type}");
            if (c.NotNull)
                def.Append(" NOT NULL");
            if (c.Default is not null)
                def.Append(" DEFAULT ").Append(c.Default);
            lines.Add(def.ToString());
            if (c.Pk)
                primaryKey = c.Name;
        }
        if (primaryKey is not null)
            lines.Add($"PRIMARY KEY (\"{primaryKey}\")");
        foreach (var u in uniques)
            lines.Add($"UNIQUE ({QuoteList(u)})");
        foreach (var f in foreignKeys)
            lines.Add($"FOREIGN KEY(\"{f.Column}\") REFERENCES \"{f.RefTable}\" (\"{f.RefColumn}\")");

        Exec($"CREATE TABLE \"{tmp}\" (\n  {string.Join(",\n  ", lines)}\n)", null);

        string commonColumns = QuoteList(
            newColumns.Select(c => c.Name).Where(n => oldColumns.Any(o => o.Name == n)));
        Exec($"INSERT INTO \"{tmp}\" ({commonColumns}) SELECT {commonColumns} FROM \"{table}\"", null);
        Exec($"DROP TABLE \"{table}\"", null);
        Exec($"ALTER TABLE \"{tmp}\" RENAME TO \"{table}\"", null);

        foreach (var (name, unique, columns) in indexes)
            Exec($"CREATE {(unique ? "UNIQUE " : "")}INDEX \"{name}\" ON \"{table}\" ({QuoteList(columns)})", null);
    }

    private static string QuoteList(IEnumerable<string> names)
        => string.Join(", ", names.Select(n => $"\"{n}\""));

    private List<RdbColumn> ReflectColumns(string table)
    {
        var columns = new List<RdbColumn>();
        Query($"PRAGMA table_info(\"{table}\")", null, r => columns.Add(new RdbColumn(
            Name: r.GetString(1),
            Type: r.GetString(2),
            NotNull: r.GetInt64(3) != 0,
            Default: r.IsDBNull(4) ? null : Convert.ToString(r.GetValue(4), CultureInfo.InvariantCulture),
            Pk: r.GetInt64(5) != 0)));
        return columns;
    }

    private List<string[]> ReflectUniqueConstraints(string table)
    {
        var uniqueIndexNames = new List<string>();
        Query($"PRAGMA index_list(\"{table}\")", null, r =>
        {
            // 0=seq, 1=name, 2=unique, 3=origin ('c' created, 'u' unique-constraint, 'pk'), 4=partial.
            if (r.GetString(3) == "u")
                uniqueIndexNames.Add(r.GetString(1));
        });
        return uniqueIndexNames.Select(ReflectIndexColumns).ToList();
    }

    private List<(string Name, bool Unique, string[] Columns)> ReflectPlainIndexes(string table)
    {
        var plain = new List<(string Name, bool Unique)>();
        Query($"PRAGMA index_list(\"{table}\")", null, r =>
        {
            if (r.GetString(3) == "c")
                plain.Add((r.GetString(1), r.GetInt64(2) != 0));
        });
        return plain.Select(p => (p.Name, p.Unique, ReflectIndexColumns(p.Name))).ToList();
    }

    private string[] ReflectIndexColumns(string indexName)
    {
        var columns = new List<string>();
        // PRAGMA index_info rows arrive in seqno order: 0=seqno, 1=cid, 2=name.
        Query($"PRAGMA index_info(\"{indexName}\")", null, r => columns.Add(r.GetString(2)));
        return columns.ToArray();
    }

    private List<(string Column, string RefTable, string RefColumn)> ReflectForeignKeys(string table)
    {
        var raw = new List<(long Id, long Seq, string Column, string RefTable, string RefColumn)>();
        Query($"PRAGMA foreign_key_list(\"{table}\")", null, r => raw.Add((
            r.GetInt64(0),                                          // id
            r.GetInt64(1),                                          // seq
            r.GetString(3),                                         // from
            r.GetString(2),                                         // table
            r.IsDBNull(4) ? r.GetString(3) : r.GetString(4))));    // to (falls back to the from column)
        // Optuna foreign keys are single-column; group by id and keep the first (seq 0) column.
        return raw
            .GroupBy(f => f.Id)
            .OrderBy(g => g.Key)
            .Select(g =>
            {
                var f = g.OrderBy(x => x.Seq).First();
                return (f.Column, f.RefTable, f.RefColumn);
            })
            .ToList();
    }
}

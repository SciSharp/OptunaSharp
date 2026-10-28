using System.Collections.Generic;
using Microsoft.EntityFrameworkCore;

namespace OptunaSharp.Storages.EFCore;

/// <summary>
///     The provider-specific DDL that creates Optuna's RDB schema, reproducing what SQLAlchemy's
///     <c>metadata.create_all</c> emits for PostgreSQL and MySQL <em>verbatim</em> (verified against
///     the dialect compiler for Optuna 5.0.0.dev): native <c>ENUM</c> types + <c>SERIAL</c> +
///     <c>TIMESTAMP WITHOUT TIME ZONE</c> on PostgreSQL; inline <c>ENUM(...)</c> + <c>AUTO_INCREMENT</c>
///     + <c>DATETIME</c> on MySQL; <c>FLOAT(53)</c>, <c>VARCHAR(512/256)</c>, <c>TEXT</c> on both.
///
///     <para>Every statement is idempotent — <c>CREATE TABLE IF NOT EXISTS</c> (and, on PostgreSQL,
///     <c>CREATE TYPE</c> guarded by a <c>DO</c> block plus <c>CREATE INDEX IF NOT EXISTS</c>) — so
///     opening a database created by Python Optuna (or by an earlier OptunaSharp run) is a no-op, which
///     is what lets the same PostgreSQL/MySQL database be used from both languages. MySQL inlines the
///     two indexes into their <c>CREATE TABLE</c> (MySQL has no <c>CREATE INDEX IF NOT EXISTS</c>).</para>
/// </summary>
internal static class RdbSchema
{
    /// <summary>
    ///     Takes the transaction-scoped advisory lock under which PostgreSQL schema creation runs. The key is an arbitrary
    ///     constant, 0x6F7074756E612D73 (the ASCII of "optuna-s"), written out in decimal so the statement is a constant;
    ///     another application using the same key on the same database only makes the two wait for each other briefly.
    /// </summary>
    private const string PostgreSqlSchemaLockSql = "SELECT pg_advisory_xact_lock(8030046183313452403)";

    /// <summary>Execute the schema DDL for <paramref name="provider" /> against the context's database.</summary>
    /// <param name="ctx">The operation-scoped EF Core context; the caller owns its lifetime and transaction.</param>
    /// <param name="provider">The relational database dialect whose connection and SQL rules apply.</param>
    /// <remarks>
    ///     [Misaligned] Storages opening a fresh PostgreSQL database at once, as a fleet of workers starting together
    ///     does, used to race: each statement's "if not exists" check passed in several sessions, and all but one then
    ///     failed with <c>23505 duplicate key value violates unique constraint "pg_type_typname_nsp_index"</c>.
    ///     Optuna 5.0 fails the same way (15 of 16 storages opening one fresh database; on MySQL its
    ///     <c>create_all</c> also fails 15 of 16 with "Table 'studies' already exists"). On PostgreSQL the DDL now runs
    ///     in one transaction holding an advisory lock, and DDL is transactional there, so a later session waits, then
    ///     finds every type and table in place. MySQL needs nothing: its <c>CREATE TABLE IF NOT EXISTS</c> is atomic
    ///     under the server's metadata locks, and DDL commits implicitly, so a transaction could not hold a lock
    ///     across it anyway.
    /// </remarks>
    internal static void EnsureCreated(OptunaRdbContext ctx, RdbProvider provider)
    {
        var statements = provider == RdbProvider.PostgreSql ? PostgreSql : MySql;
        if (provider != RdbProvider.PostgreSql)
        {
            foreach (string sql in statements)
                ctx.Database.ExecuteSqlRaw(sql);
            return;
        }

        // Under the context's execution strategy, so a storage configured with EnableRetryOnFailure may start this
        // transaction (a retrying strategy refuses user transactions begun outside it) and retries it whole.
        ctx.Database.CreateExecutionStrategy().Execute(() =>
        {
            using var tx = ctx.Database.BeginTransaction();
            ctx.Database.ExecuteSqlRaw(PostgreSqlSchemaLockSql);
            foreach (string sql in statements)
                ctx.Database.ExecuteSqlRaw(sql);
            tx.Commit();
        });
    }

    // ------------------------------------------------------------------ PostgreSQL

    private static readonly IReadOnlyList<string> PostgreSql = new[]
    {
        // Native ENUM types (SQLAlchemy Enum(PyEnum) → a type named after the lowercased class). No
        // CREATE TYPE IF NOT EXISTS in PostgreSQL, so each is guarded by a catalog check.
        EnumType("studydirection", "'NOT_SET','MINIMIZE','MAXIMIZE'"),
        EnumType("trialstate", "'RUNNING','COMPLETE','PRUNED','FAIL','WAITING'"),
        EnumType("trialvaluetype", "'FINITE','INF_POS','INF_NEG'"),
        EnumType("trialintermediatevaluetype", "'FINITE','INF_POS','INF_NEG','NAN'"),

        """
        CREATE TABLE IF NOT EXISTS studies (
            study_id SERIAL NOT NULL,
            study_name VARCHAR(512) NOT NULL,
            PRIMARY KEY (study_id)
        )
        """,
        "CREATE UNIQUE INDEX IF NOT EXISTS ix_studies_study_name ON studies (study_name)",

        """
        CREATE TABLE IF NOT EXISTS version_info (
            version_info_id INTEGER NOT NULL,
            schema_version INTEGER,
            library_version VARCHAR(256),
            PRIMARY KEY (version_info_id),
            CHECK (version_info_id=1)
        )
        """,

        """
        CREATE TABLE IF NOT EXISTS alembic_version (
            version_num VARCHAR(32) NOT NULL,
            CONSTRAINT alembic_version_pkc PRIMARY KEY (version_num)
        )
        """,

        """
        CREATE TABLE IF NOT EXISTS study_directions (
            study_direction_id SERIAL NOT NULL,
            direction studydirection NOT NULL,
            study_id INTEGER NOT NULL,
            objective INTEGER NOT NULL,
            PRIMARY KEY (study_direction_id),
            UNIQUE (study_id, objective),
            FOREIGN KEY(study_id) REFERENCES studies (study_id)
        )
        """,

        """
        CREATE TABLE IF NOT EXISTS study_user_attributes (
            study_user_attribute_id SERIAL NOT NULL,
            study_id INTEGER,
            "key" VARCHAR(512),
            value_json TEXT,
            PRIMARY KEY (study_user_attribute_id),
            UNIQUE (study_id, "key"),
            FOREIGN KEY(study_id) REFERENCES studies (study_id)
        )
        """,

        """
        CREATE TABLE IF NOT EXISTS study_system_attributes (
            study_system_attribute_id SERIAL NOT NULL,
            study_id INTEGER,
            "key" VARCHAR(512),
            value_json TEXT,
            PRIMARY KEY (study_system_attribute_id),
            UNIQUE (study_id, "key"),
            FOREIGN KEY(study_id) REFERENCES studies (study_id)
        )
        """,

        """
        CREATE TABLE IF NOT EXISTS trials (
            trial_id SERIAL NOT NULL,
            number INTEGER,
            study_id INTEGER,
            state trialstate NOT NULL,
            datetime_start TIMESTAMP WITHOUT TIME ZONE,
            datetime_complete TIMESTAMP WITHOUT TIME ZONE,
            PRIMARY KEY (trial_id),
            FOREIGN KEY(study_id) REFERENCES studies (study_id)
        )
        """,
        "CREATE INDEX IF NOT EXISTS ix_trials_study_id ON trials (study_id)",

        """
        CREATE TABLE IF NOT EXISTS trial_user_attributes (
            trial_user_attribute_id SERIAL NOT NULL,
            trial_id INTEGER,
            "key" VARCHAR(512),
            value_json TEXT,
            PRIMARY KEY (trial_user_attribute_id),
            UNIQUE (trial_id, "key"),
            FOREIGN KEY(trial_id) REFERENCES trials (trial_id)
        )
        """,

        """
        CREATE TABLE IF NOT EXISTS trial_system_attributes (
            trial_system_attribute_id SERIAL NOT NULL,
            trial_id INTEGER,
            "key" VARCHAR(512),
            value_json TEXT,
            PRIMARY KEY (trial_system_attribute_id),
            UNIQUE (trial_id, "key"),
            FOREIGN KEY(trial_id) REFERENCES trials (trial_id)
        )
        """,

        """
        CREATE TABLE IF NOT EXISTS trial_params (
            param_id SERIAL NOT NULL,
            trial_id INTEGER,
            param_name VARCHAR(512),
            param_value FLOAT(53),
            distribution_json TEXT,
            PRIMARY KEY (param_id),
            UNIQUE (trial_id, param_name),
            FOREIGN KEY(trial_id) REFERENCES trials (trial_id)
        )
        """,

        """
        CREATE TABLE IF NOT EXISTS trial_values (
            trial_value_id SERIAL NOT NULL,
            trial_id INTEGER NOT NULL,
            objective INTEGER NOT NULL,
            value FLOAT(53),
            value_type trialvaluetype NOT NULL,
            PRIMARY KEY (trial_value_id),
            UNIQUE (trial_id, objective),
            FOREIGN KEY(trial_id) REFERENCES trials (trial_id)
        )
        """,

        """
        CREATE TABLE IF NOT EXISTS trial_intermediate_values (
            trial_intermediate_value_id SERIAL NOT NULL,
            trial_id INTEGER NOT NULL,
            step INTEGER NOT NULL,
            intermediate_value FLOAT(53),
            intermediate_value_type trialintermediatevaluetype NOT NULL,
            PRIMARY KEY (trial_intermediate_value_id),
            UNIQUE (trial_id, step),
            FOREIGN KEY(trial_id) REFERENCES trials (trial_id)
        )
        """,

        """
        CREATE TABLE IF NOT EXISTS trial_heartbeats (
            trial_heartbeat_id SERIAL NOT NULL,
            trial_id INTEGER NOT NULL,
            heartbeat TIMESTAMP WITHOUT TIME ZONE NOT NULL,
            PRIMARY KEY (trial_heartbeat_id),
            UNIQUE (trial_id),
            FOREIGN KEY(trial_id) REFERENCES trials (trial_id)
        )
        """,
    };

    /// <summary>Builds PostgreSQL DDL that creates an enum type only when it does not exist.</summary>
    /// <param name="name">The identifier used by the provider or schema operation.</param>
    /// <param name="labels">The SQL enum labels included in the type declaration.</param>
    /// <returns>The PostgreSQL enum-creation statement.</returns>
    private static string EnumType(string name, string labels) =>
        $"DO $$ BEGIN IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = '{name}') " +
        $"THEN CREATE TYPE {name} AS ENUM ({labels}); END IF; END $$;";

    // ------------------------------------------------------------------ MySQL

    private static readonly IReadOnlyList<string> MySql = new[]
    {
        // Indexes are inlined into CREATE TABLE (MySQL has no CREATE INDEX IF NOT EXISTS), which
        // produces the same named indexes SQLAlchemy creates separately (ix_studies_study_name,
        // ix_trials_study_id). Enum columns are inline ENUM('NAME',...) storing the member name.
        """
        CREATE TABLE IF NOT EXISTS studies (
            study_id INTEGER NOT NULL AUTO_INCREMENT,
            study_name VARCHAR(512) NOT NULL,
            PRIMARY KEY (study_id),
            UNIQUE KEY ix_studies_study_name (study_name)
        )
        """,

        """
        CREATE TABLE IF NOT EXISTS version_info (
            version_info_id INTEGER NOT NULL,
            schema_version INTEGER,
            library_version VARCHAR(256),
            PRIMARY KEY (version_info_id),
            CHECK (version_info_id=1)
        )
        """,

        """
        CREATE TABLE IF NOT EXISTS alembic_version (
            version_num VARCHAR(32) NOT NULL,
            CONSTRAINT alembic_version_pkc PRIMARY KEY (version_num)
        )
        """,

        """
        CREATE TABLE IF NOT EXISTS study_directions (
            study_direction_id INTEGER NOT NULL AUTO_INCREMENT,
            direction ENUM('NOT_SET','MINIMIZE','MAXIMIZE') NOT NULL,
            study_id INTEGER NOT NULL,
            objective INTEGER NOT NULL,
            PRIMARY KEY (study_direction_id),
            UNIQUE (study_id, objective),
            FOREIGN KEY(study_id) REFERENCES studies (study_id)
        )
        """,

        """
        CREATE TABLE IF NOT EXISTS study_user_attributes (
            study_user_attribute_id INTEGER NOT NULL AUTO_INCREMENT,
            study_id INTEGER,
            `key` VARCHAR(512),
            value_json TEXT,
            PRIMARY KEY (study_user_attribute_id),
            UNIQUE (study_id, `key`),
            FOREIGN KEY(study_id) REFERENCES studies (study_id)
        )
        """,

        """
        CREATE TABLE IF NOT EXISTS study_system_attributes (
            study_system_attribute_id INTEGER NOT NULL AUTO_INCREMENT,
            study_id INTEGER,
            `key` VARCHAR(512),
            value_json TEXT,
            PRIMARY KEY (study_system_attribute_id),
            UNIQUE (study_id, `key`),
            FOREIGN KEY(study_id) REFERENCES studies (study_id)
        )
        """,

        """
        CREATE TABLE IF NOT EXISTS trials (
            trial_id INTEGER NOT NULL AUTO_INCREMENT,
            number INTEGER,
            study_id INTEGER,
            state ENUM('RUNNING','COMPLETE','PRUNED','FAIL','WAITING') NOT NULL,
            datetime_start DATETIME,
            datetime_complete DATETIME,
            PRIMARY KEY (trial_id),
            KEY ix_trials_study_id (study_id),
            FOREIGN KEY(study_id) REFERENCES studies (study_id)
        )
        """,

        """
        CREATE TABLE IF NOT EXISTS trial_user_attributes (
            trial_user_attribute_id INTEGER NOT NULL AUTO_INCREMENT,
            trial_id INTEGER,
            `key` VARCHAR(512),
            value_json TEXT,
            PRIMARY KEY (trial_user_attribute_id),
            UNIQUE (trial_id, `key`),
            FOREIGN KEY(trial_id) REFERENCES trials (trial_id)
        )
        """,

        """
        CREATE TABLE IF NOT EXISTS trial_system_attributes (
            trial_system_attribute_id INTEGER NOT NULL AUTO_INCREMENT,
            trial_id INTEGER,
            `key` VARCHAR(512),
            value_json TEXT,
            PRIMARY KEY (trial_system_attribute_id),
            UNIQUE (trial_id, `key`),
            FOREIGN KEY(trial_id) REFERENCES trials (trial_id)
        )
        """,

        """
        CREATE TABLE IF NOT EXISTS trial_params (
            param_id INTEGER NOT NULL AUTO_INCREMENT,
            trial_id INTEGER,
            param_name VARCHAR(512),
            param_value FLOAT(53),
            distribution_json TEXT,
            PRIMARY KEY (param_id),
            UNIQUE (trial_id, param_name),
            FOREIGN KEY(trial_id) REFERENCES trials (trial_id)
        )
        """,

        """
        CREATE TABLE IF NOT EXISTS trial_values (
            trial_value_id INTEGER NOT NULL AUTO_INCREMENT,
            trial_id INTEGER NOT NULL,
            objective INTEGER NOT NULL,
            value FLOAT(53),
            value_type ENUM('FINITE','INF_POS','INF_NEG') NOT NULL,
            PRIMARY KEY (trial_value_id),
            UNIQUE (trial_id, objective),
            FOREIGN KEY(trial_id) REFERENCES trials (trial_id)
        )
        """,

        """
        CREATE TABLE IF NOT EXISTS trial_intermediate_values (
            trial_intermediate_value_id INTEGER NOT NULL AUTO_INCREMENT,
            trial_id INTEGER NOT NULL,
            step INTEGER NOT NULL,
            intermediate_value FLOAT(53),
            intermediate_value_type ENUM('FINITE','INF_POS','INF_NEG','NAN') NOT NULL,
            PRIMARY KEY (trial_intermediate_value_id),
            UNIQUE (trial_id, step),
            FOREIGN KEY(trial_id) REFERENCES trials (trial_id)
        )
        """,

        """
        CREATE TABLE IF NOT EXISTS trial_heartbeats (
            trial_heartbeat_id INTEGER NOT NULL AUTO_INCREMENT,
            trial_id INTEGER NOT NULL,
            heartbeat DATETIME NOT NULL,
            PRIMARY KEY (trial_heartbeat_id),
            UNIQUE (trial_id),
            FOREIGN KEY(trial_id) REFERENCES trials (trial_id)
        )
        """,
    };
}

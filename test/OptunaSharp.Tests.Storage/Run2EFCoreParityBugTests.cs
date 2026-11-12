using System;
using AwesomeAssertions;
using OptunaSharp;
using OptunaSharp.Storages.EFCore;

namespace OptunaSharp.Tests.Storage;

/// <summary>
///     Open parity bugs in the EF Core RDB storage (PostgreSQL, MySQL) found by the 2026-11-10 parity QA run
///     (<c>docs/qa/PARITY_QA_2026-11-10.md</c>). Each test fails today and passes once fixed, at which point its
///     <c>OpenBugs</c> category should be removed. The databases come from <see cref="EFCoreDbSession" />
///     (Testcontainers); without Docker the tests report Inconclusive, as the rest of the EF Core suite does.
/// </summary>
[TestClass]
public class Run2EFCoreParityBugTests
{
    /// <summary>The message of Optuna's <c>_create_scoped_session</c> for every non-integrity database error.</summary>
    private const string OptunaCommitMessage =
        "An exception is raised during the commit. This typically happens due to invalid data in the commit, " +
        "e.g. exceeding max length. ";

    /// <summary>
    ///     QA-74: Optuna's RDB storage runs every write in <c>_create_scoped_session</c>, which turns any SQLAlchemy error
    ///     other than an integrity error into <c>StorageInternalError</c> with <see cref="OptunaCommitMessage" />. A key
    ///     longer than the 512-character <c>key</c> column is such an error on PostgreSQL and MySQL. The port's EF Core
    ///     storage lets EF's <c>DbUpdateException</c> ("An error occurred while saving the entity changes. See the inner
    ///     exception for details.") escape, so code that catches <c>StorageInternalError</c> misses it.
    /// </summary>
    /// <remarks>
    ///     Measured with Optuna 5.0.0 / SQLAlchemy 2.0.45 (PyMySQL, psycopg2) against MySQL 8.0 and PostgreSQL 16 and the
    ///     port against the same servers: a study name, parameter name, study or trial user-attribute key of 600
    ///     characters, and on MySQL a 70,000-character attribute value (beyond <c>TEXT</c>'s 65,535 bytes), all raise
    ///     <c>StorageInternalError</c> in Optuna and <c>DbUpdateException</c> in the port (9 of 9 cases; PostgreSQL's
    ///     <c>TEXT</c> takes the long value on both sides). SQLite enforces no length, so both accept everything there. No
    ///     storage of the port raises <c>StorageInternalError</c> at all: the type exists but nothing throws it.
    ///     <b>Fixed 2026-11-12:</b> the EF Core storage's <c>Mutate</c> translates every non-integrity database error into
    ///     Optuna's <c>StorageInternalError</c>, wrapping the provider's error, outside the execution strategy.
    /// </remarks>
    /// <param name="provider">The database server to run against.</param>
    [TestMethod]
    [DataRow(RdbProvider.PostgreSql)]
    [DataRow(RdbProvider.MySql)]
    public void OverLongAttributeKey_RaisesStorageInternalErrorLikeOptuna(RdbProvider provider)
    {
        EFCoreDbSession.EnsureAvailable(provider);
        var study = Optuna.CreateStudy(storage: EFCoreDbSession.NewStorage(provider), studyName: "limits");

        Action setLongKey = () => study.SetUserAttr(new string('k', 600), 1L);

        setLongKey.Should().Throw<StorageInternalError>().WithMessage(OptunaCommitMessage);
    }

    /// <summary>
    ///     QA-71 on MySQL: Optuna reads a study's and a trial's attributes with no <c>ORDER BY</c>, and InnoDB answers
    ///     through the unique <c>(study_id, key)</c> / <c>(trial_id, key)</c> index, so they come back sorted by key, as on
    ///     SQLite; the EF Core storage orders by attribute id (insertion order).
    /// </summary>
    /// <remarks>
    ///     Measured against MySQL 8.0: study user attributes set <c>zz, b, m</c> read back <c>b, m, zz</c> in Optuna,
    ///     <c>zz, b, m</c> in the port; trial user attributes <c>k, a</c> as <c>a, k</c>. On PostgreSQL 16 Optuna's own order
    ///     is not stable (sorted on the first read of a study, insertion order after <c>load_study</c>), so there is no
    ///     single target and no PostgreSQL case here.
    /// </remarks>
    [TestMethod]
    [TestCategory("OpenBugs")]
    public void MySqlAttributeReads_ComeBackInKeyOrderLikeOptuna()
    {
        EFCoreDbSession.EnsureAvailable(RdbProvider.MySql);
        var study = Optuna.CreateStudy(storage: EFCoreDbSession.NewStorage(RdbProvider.MySql), studyName: "attrs");
        foreach (var key in new[] { "zz", "b", "m" })
            study.SetUserAttr(key, 1L);
        var trial = study.Ask();
        trial.SetUserAttr("k", 1L);
        trial.SetUserAttr("a", 2L);
        study.Tell(trial, 1.0);

        study.UserAttrs.Keys.Should().Equal(new[] { "b", "m", "zz" }, "InnoDB returns Optuna's unordered query in (study_id, key) order");
        study.Trials[0].UserAttrs.Keys.Should().Equal("a", "k");
    }
}

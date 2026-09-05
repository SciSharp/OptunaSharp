using System;
using AwesomeAssertions;
using MySqlConnector;
using Npgsql;
using NumSharp;
using OptunaSharp.Storages.EFCore;

namespace OptunaSharp.Tests.Storage;

/// <summary>
///     Unit tests for the SQLAlchemy-URL parser (<c>RdbUrl</c>) — no Docker needed. These pin the
///     provider selection and the URL → native-connection-string translation the EF Core
///     <c>RDBStorage</c> relies on.
/// </summary>
[TestClass]
public class EFCoreRdbUrlTests
{
    private static NpgsqlConnectionStringBuilder Pg(string url)
    {
        var (provider, builder) = RdbUrl.Parse(url);
        provider.Should().Be(RdbProvider.PostgreSql);
        return (NpgsqlConnectionStringBuilder)builder;
    }

    private static MySqlConnectionStringBuilder My(string url)
    {
        var (provider, builder) = RdbUrl.Parse(url);
        provider.Should().Be(RdbProvider.MySql);
        return (MySqlConnectionStringBuilder)builder;
    }

    [TestMethod]
    public void Postgres_FullUrl_ParsesEveryComponent()
    {
        var b = Pg("postgresql://alice:s3cret@db.example.com:5433/optuna");
        b.Host.Should().Be("db.example.com");
        b.Port.Should().Be(5433);
        b.Username.Should().Be("alice");
        b.Password.Should().Be("s3cret");
        b.Database.Should().Be("optuna");
    }

    [TestMethod]
    public void Postgres_DriverSuffix_IsAccepted()
    {
        Pg("postgresql+psycopg2://u:p@localhost/db").Database.Should().Be("db");
        Pg("postgres://u:p@localhost/db").Host.Should().Be("localhost");
    }

    [TestMethod]
    public void Postgres_NoPortNoPassword_LeavesDefaults()
    {
        var b = Pg("postgresql://someuser@localhost/optuna");
        b.Username.Should().Be("someuser");
        b.Password.Should().BeNull();
        b.Host.Should().Be("localhost");
    }

    [TestMethod]
    public void Mysql_FullUrl_ParsesEveryComponent()
    {
        var b = My("mysql://root:pw@127.0.0.1:3307/optuna");
        b.Server.Should().Be("127.0.0.1");
        b.Port.Should().Be(3307);
        b.UserID.Should().Be("root");
        b.Password.Should().Be("pw");
        b.Database.Should().Be("optuna");
    }

    [TestMethod]
    public void Mysql_DriverSuffix_IsAccepted()
    {
        My("mysql+pymysql://u:p@localhost/db").Database.Should().Be("db");
        My("mysql+mysqldb://u:p@localhost:3306/db").Port.Should().Be(3306);
    }

    [TestMethod]
    public void PercentEncoded_UserInfo_IsDecoded()
    {
        var b = Pg("postgresql://a%40b:p%3Aword@localhost/db");
        b.Username.Should().Be("a@b");
        b.Password.Should().Be("p:word");
    }

    [TestMethod]
    public void QueryParams_ArePassedThrough_AsConnectionStringKeywords()
    {
        var b = Pg("postgresql://u:p@localhost/db?SSL%20Mode=Require");
        b.SslMode.Should().Be(SslMode.Require);
    }

    [TestMethod]
    public void SqliteUrl_IsRejected_WithGuidanceToTheSqlitePackage()
    {
        Action act = () => RdbUrl.Parse("sqlite:///study.db");
        act.Should().Throw<ValueError>().WithMessage("*OptunaSharp.Storage.Sqlite*");
    }

    [TestMethod]
    public void UnsupportedDialect_Throws()
    {
        Action act = () => RdbUrl.Parse("oracle://u:p@localhost/db");
        act.Should().Throw<ValueError>().WithMessage("*Unsupported RDB dialect*");
    }

    [TestMethod]
    public void MissingScheme_Throws()
    {
        Action act = () => RdbUrl.Parse("just-a-path.db");
        act.Should().Throw<ValueError>();
    }

    [TestMethod]
    public void MariaDb_Dialect_MapsToMySql()
        => My("mariadb://u:p@localhost:3306/db").Database.Should().Be("db");

    [TestMethod]
    public void NoDatabase_LeavesDatabaseUnset()
    {
        var b = Pg("postgresql://u:p@localhost:5432");
        b.Host.Should().Be("localhost");
        b.Database.Should().BeNullOrEmpty();
    }

    [TestMethod]
    public void MultipleQueryParams_AllApplied()
    {
        var b = Pg("postgresql://u:p@localhost/db?SSL%20Mode=Require&Pooling=false");
        b.SslMode.Should().Be(SslMode.Require);
        b.Pooling.Should().BeFalse();
    }

    // ------------------------------------------------------------------ options validation (no Docker)

    [TestMethod]
    public void HeartbeatInterval_NonPositive_ThrowsBeforeConnecting()
    {
        Action act = () => new RDBStorage(
            "postgresql://u:p@localhost:5432/db", new EFCoreRdbOptions { HeartbeatInterval = 0 });
        act.Should().Throw<ValueError>().WithMessage("*heartbeat_interval*");
    }

    [TestMethod]
    public void GracePeriod_NonPositive_ThrowsBeforeConnecting()
    {
        Action act = () => new RDBStorage(
            "mysql://u:p@localhost:3306/db", new EFCoreRdbOptions { GracePeriod = -5 });
        act.Should().Throw<ValueError>().WithMessage("*grace_period*");
    }
}

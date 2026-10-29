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

    /// <summary>Driver suffixes SQLAlchemy registers for PostgreSQL (<c>+psycopg2</c>, <c>+pg8000</c>) select the PostgreSQL provider.</summary>
    [TestMethod]
    public void Postgres_DriverSuffix_IsAccepted()
    {
        Pg("postgresql+psycopg2://u:p@localhost/db").Database.Should().Be("db");
        Pg("postgresql+pg8000://u:p@localhost/db").Host.Should().Be("localhost");
    }

    /// <summary>
    ///     <c>postgres://</c> is refused with SQLAlchemy's <c>NoSuchModuleError</c> text: SQLAlchemy 1.4 dropped that
    ///     dialect name, so Python Optuna cannot open such a URL either.
    /// </summary>
    [TestMethod]
    public void Postgres_LegacyPostgresScheme_IsRefusedLikeSqlAlchemy()
    {
        Action act = () => RdbUrl.Parse("postgres://u:p@localhost/db");
        act.Should().Throw<ArgumentException>().WithMessage("Can't load plugin: sqlalchemy.dialects:postgres");
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

    /// <summary>
    ///     A query parameter is a libpq connection parameter, because psycopg2 hands the query to libpq:
    ///     <c>sslmode=require</c> reaches Npgsql as <see cref="SslMode.Require" />.
    /// </summary>
    [TestMethod]
    public void QueryParams_AreLibpqParameters()
    {
        var b = Pg("postgresql://u:p@localhost/db?sslmode=require");
        b.SslMode.Should().Be(SslMode.Require);
    }

    /// <summary>
    ///     Npgsql's own keywords are not libpq's, so a URL carrying them fails as psycopg2 fails: <c>make_dsn</c> writes
    ///     <c>SSL Mode=Require</c> and libpq stops at the space; <c>Pooling</c> is an unknown option. Both are libpq's
    ///     messages behind psycopg2's <c>invalid dsn:</c> prefix.
    /// </summary>
    [TestMethod]
    public void QueryParams_NativeNpgsqlKeyword_IsRefusedLikeLibpq()
    {
        Action spaced = () => RdbUrl.Parse("postgresql://u:p@localhost/db?SSL%20Mode=Require");
        spaced.Should().Throw<FormatException>().WithMessage("invalid dsn: missing \"=\" after \"SSL\" in connection info string");
        Action unknown = () => RdbUrl.Parse("postgresql://u:p@localhost/db?Pooling=false");
        unknown.Should().Throw<FormatException>().WithMessage("invalid dsn: invalid connection option \"Pooling\"");
    }

    /// <summary>
    ///     Values libpq checks when psycopg2 connects (an unknown <c>sslmode</c>, a non-integer <c>connect_timeout</c>)
    ///     fail with libpq's messages, up front: while the connection string is built.
    /// </summary>
    [TestMethod]
    public void QueryParams_LibpqValueChecks_RaiseLibpqMessages()
    {
        Action sslmode = () => RdbUrl.Parse("postgresql://u:p@localhost/db?sslmode=bogus");
        sslmode.Should().Throw<FormatException>().WithMessage("invalid sslmode value: \"bogus\"");
        Action timeout = () => RdbUrl.Parse("postgresql://u:p@localhost/db?connect_timeout=x");
        timeout.Should().Throw<FormatException>().WithMessage("invalid integer value \"x\" for connection option \"connect_timeout\"");
    }

    /// <summary>
    ///     libpq's <c>application_name</c>, <c>options</c> (form-decoded, so <c>%20</c> and <c>%3D</c> become a space and
    ///     <c>=</c>) and <c>target_session_attrs</c> reach the matching Npgsql settings.
    /// </summary>
    [TestMethod]
    public void QueryParams_ApplicationNameOptionsAndSessionAttributes_AreMapped()
    {
        var b = Pg("postgresql://u:p@localhost:5432/db?application_name=optuna&options=-c%20search_path%3Doptuna&target_session_attrs=primary");
        b.ApplicationName.Should().Be("optuna");
        b.Options.Should().Be("-c search_path=optuna");
        b.TargetSessionAttributes.Should().Be("primary");
        b.Port.Should().Be(5432);
    }

    /// <summary>
    ///     <c>connect_timeout</c> follows libpq: a number of seconds, zero (or less) waits forever, and 1 is raised to 2.
    /// </summary>
    [TestMethod]
    public void QueryParams_ConnectTimeout_FollowsLibpq()
    {
        Pg("postgresql://u:p@localhost/db?connect_timeout=7").Timeout.Should().Be(7);
        Pg("postgresql://u:p@localhost/db?connect_timeout=0").Timeout.Should().Be(0);
        Pg("postgresql://u:p@localhost/db?connect_timeout=1").Timeout.Should().Be(2);
    }

    /// <summary>
    ///     A repeated <c>?host=</c> is SQLAlchemy's multi-host form: it becomes libpq's <c>host=h1,h2 port=5433,</c>, and
    ///     the empty second port is libpq's default 5432.
    /// </summary>
    [TestMethod]
    public void QueryParams_RepeatedHost_IsALibpqHostList()
    {
        Pg("postgresql://u:p@/db?host=h1:5433&host=h2").Host.Should().Be("h1:5433,h2:5432");
    }

    [TestMethod]
    public void SqliteUrl_IsRejected_WithGuidanceToTheSqlitePackage()
    {
        Action act = () => RdbUrl.Parse("sqlite:///study.db");
        act.Should().Throw<ValueError>().WithMessage("*OptunaSharp.Storage.Sqlite*");
    }

    /// <summary>
    ///     A dialect SQLAlchemy knows but no OptunaSharp package serves (Oracle) raises Optuna's <c>ImportError</c> text,
    ///     as a <see cref="NotSupportedException" />.
    /// </summary>
    [TestMethod]
    public void UnsupportedDialect_Throws()
    {
        Action act = () => RdbUrl.Parse("oracle://u:p@localhost/db");
        act.Should().Throw<NotSupportedException>()
            .WithMessage("Failed to import DB access module for the specified storage URL. Please install appropriate one.*");
    }

    /// <summary>A string that is not a URL raises SQLAlchemy's <c>ArgumentError</c> text, as an <see cref="ArgumentException" />.</summary>
    [TestMethod]
    public void MissingScheme_Throws()
    {
        Action act = () => RdbUrl.Parse("just-a-path.db");
        act.Should().Throw<ArgumentException>().WithMessage("Could not parse SQLAlchemy URL from given URL string");
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

    /// <summary>Several libpq parameters in one query all apply.</summary>
    [TestMethod]
    public void MultipleQueryParams_AllApplied()
    {
        var b = Pg("postgresql://u:p@localhost/db?sslmode=require&connect_timeout=7");
        b.SslMode.Should().Be(SslMode.Require);
        b.Timeout.Should().Be(7);
    }

    // ------------------------------------------------------------------ MySQL drivers (PyMySQL, MySQLdb)

    /// <summary>
    ///     A <c>?port=</c> reaches PyMySQL as text (SQLAlchemy only types the URL's own port), and PyMySQL requires an
    ///     int: its <c>ValueError</c>.
    /// </summary>
    [TestMethod]
    public void Mysql_PyMySql_QueryPort_IsAStringToTheDriver()
    {
        Action act = () => RdbUrl.Parse("mysql+pymysql://u:p@h/db?port=3307");
        act.Should().Throw<ValueError>().WithMessage("port should be of type int");
    }

    /// <summary>
    ///     An argument the driver's <c>connect</c> does not take raises that driver's own <c>TypeError</c> text: PyMySQL's
    ///     <c>Connection.__init__</c> wording for <c>+pymysql</c>, MySQLdb's for <c>mysql://</c>.
    /// </summary>
    [TestMethod]
    public void Mysql_UnknownArgument_RaisesTheDriversTypeError()
    {
        Action pymysql = () => RdbUrl.Parse("mysql+pymysql://u:p@h/db?foo=1");
        pymysql.Should().Throw<TypeError>().WithMessage("Connection.__init__() got an unexpected keyword argument 'foo'");
        Action mysqldb = () => RdbUrl.Parse("mysql://u:p@h/db?foo=1");
        mysqldb.Should().Throw<TypeError>().WithMessage("'foo' is an invalid keyword argument for connect()");
    }

    /// <summary>PyMySQL's own refusals (<c>compress</c>, a zero <c>connect_timeout</c>) keep PyMySQL's exception kinds and messages.</summary>
    [TestMethod]
    public void Mysql_PyMySql_Refusals_UsePyMySqlsMessages()
    {
        Action compress = () => RdbUrl.Parse("mysql+pymysql://u:p@h/db?compress=true");
        compress.Should().Throw<NotImplementedException>().WithMessage("compress and named_pipe arguments are not supported");
        Action timeout = () => RdbUrl.Parse("mysql+pymysql://u:p@h/db?connect_timeout=0");
        timeout.Should().Throw<ValueError>().WithMessage("connect_timeout should be >0 and <=31536000");
    }

    /// <summary>
    ///     <c>ssl_ca</c> makes PyMySQL verify the server certificate and its host name, so MySqlConnector gets the CA
    ///     and <see cref="MySqlSslMode.VerifyFull" />.
    /// </summary>
    [TestMethod]
    public void Mysql_PyMySql_SslCa_VerifiesTheServer()
    {
        var b = My("mysql+pymysql://u:p@h/db?ssl_ca=/x/ca.pem");
        b.SslCa.Should().Be("/x/ca.pem");
        b.SslMode.Should().Be(MySqlSslMode.VerifyFull);
    }

    /// <summary>MySQLdb's integer <c>connect_timeout</c> and PyMySQL's <c>utf8mb4</c> character set are honored.</summary>
    [TestMethod]
    public void Mysql_ConnectTimeoutAndUtf8Charset_AreMapped()
    {
        My("mysql://u:p@h/db?connect_timeout=5").ConnectionTimeout.Should().Be(5u);
        My("mysql+pymysql://u:p@h:3307/db?charset=utf8mb4").Port.Should().Be(3307u);
    }

    /// <summary>
    ///     A character set other than UTF-8 is refused: MySqlConnector always talks <c>utf8mb4</c>, so it could not send
    ///     the bytes the Python driver would.
    /// </summary>
    [TestMethod]
    public void Mysql_NonUtf8Charset_IsRefused()
    {
        Action act = () => RdbUrl.Parse("mysql://u:p@h/db?charset=latin1");
        act.Should().Throw<NotSupportedException>().WithMessage("*charset*");
    }

    /// <summary>
    ///     An asyncio driver fails as a synchronous <c>create_engine</c> over it fails in Python: on SQLAlchemy's
    ///     greenlet check.
    /// </summary>
    [TestMethod]
    public void Mysql_AsyncDriver_FailsLikeSqlAlchemy()
    {
        Action act = () => RdbUrl.Parse("mysql+aiomysql://u:p@h/db");
        act.Should().Throw<InvalidOperationException>().WithMessage("greenlet_spawn has not been called*");
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

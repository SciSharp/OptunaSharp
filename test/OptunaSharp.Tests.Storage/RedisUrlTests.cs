using AwesomeAssertions;
using NumSharp;                          // ValueError
using OptunaSharp.Storages.Journal;      // RedisUrl / RedisEndpoint (shared package)

namespace OptunaSharp.Tests.Storage;

/// <summary>
///     Unit tests for the shared <c>redis://</c> → <see cref="RedisEndpoint" /> parser — the one bit of
///     bespoke glue (StackExchange.Redis / ServiceStack have no common URI parser; redis-py's
///     <c>from_url</c> gave Optuna this). All three backends build their client from this, so parsing
///     once here is what keeps their connection interpretation identical. No server required.
/// </summary>
[TestClass]
public class RedisUrlTests
{
    [TestMethod]
    public void PlainHostPort()
    {
        var e = RedisUrl.Parse("redis://localhost:6379");
        e.Host.Should().Be("localhost");
        e.Port.Should().Be(6379);
        e.Ssl.Should().BeFalse();
        e.Db.Should().BeNull();
        e.User.Should().BeNull();
        e.Password.Should().BeNull();
        e.UnixSocketPath.Should().BeNull();
    }

    [TestMethod]
    public void DefaultsPortWhenOmitted() => RedisUrl.Parse("redis://localhost").Port.Should().Be(6379);

    [TestMethod]
    public void PasswordOnly_NoUser_WithDbInPath()
    {
        var e = RedisUrl.Parse("redis://:secret@127.0.0.1:6380/3");
        e.Host.Should().Be("127.0.0.1");
        e.Port.Should().Be(6380);
        e.Password.Should().Be("secret");
        e.User.Should().BeNull();
        e.Db.Should().Be(3);
    }

    [TestMethod]
    public void UserAndPercentEncodedPassword()
    {
        var e = RedisUrl.Parse("redis://alice:p%40ss@redis.example.com:6379/1");
        e.User.Should().Be("alice");
        e.Password.Should().Be("p@ss");   // %40 → '@'
        e.Db.Should().Be(1);
    }

    [TestMethod]
    public void RedissEnablesTls()
    {
        var e = RedisUrl.Parse("rediss://user:pw@redis.example.com:6380");
        e.Ssl.Should().BeTrue();
        e.Host.Should().Be("redis.example.com");
        e.Port.Should().Be(6380);
    }

    [TestMethod]
    public void QueryDbOverridesPathDb() => RedisUrl.Parse("redis://host:6379/2?db=5").Db.Should().Be(5);

    [TestMethod]
    public void QueryUsernameAndPassword()
    {
        var e = RedisUrl.Parse("redis://host:6379?username=alice&password=secret");
        e.User.Should().Be("alice");
        e.Password.Should().Be("secret");
    }

    [TestMethod]
    public void UserOnly_NoPassword()
    {
        var e = RedisUrl.Parse("redis://alice@host:6379");
        e.User.Should().Be("alice");
        e.Password.Should().BeNull();
    }

    [TestMethod]
    public void UnixSocket()
    {
        var e = RedisUrl.Parse("unix:///var/run/redis/redis.sock?db=0&password=abc");
        e.UnixSocketPath.Should().Be("/var/run/redis/redis.sock");
        e.Db.Should().Be(0);
        e.Password.Should().Be("abc");
    }

    [TestMethod]
    public void NonNumericDbPath_IsIgnored() => RedisUrl.Parse("redis://host:6379/not-a-db").Db.Should().BeNull();

    [TestMethod]
    public void ValuelessQueryFlag_IsIgnored() => RedisUrl.Parse("redis://host:6379?verbose&db=2").Db.Should().Be(2);

    [TestMethod]
    public void UnknownScheme_ThrowsValueError()
    {
        var act = () => RedisUrl.Parse("http://localhost:6379");
        act.Should().Throw<ValueError>()
            .WithMessage("*Redis URL must specify one of the following schemes (redis://, rediss://, unix://)*");
    }

    [TestMethod]
    public void Malformed_ThrowsValueError()
    {
        var act = () => RedisUrl.Parse("not-a-url");
        act.Should().Throw<ValueError>().WithMessage("*Invalid Redis URL*");
    }
}

using System.Net;
using System.Net.Sockets;
using AwesomeAssertions;
using NumSharp;                          // ValueError
using OptunaSharp.Storages.Journal;      // RedisUrl (internal — reachable via the friend grant)
using StackExchange.Redis;

namespace OptunaSharp.Tests.Storage.Redis;

/// <summary>
///     Unit tests for the <c>redis://</c> → <see cref="ConfigurationOptions" /> adapter — the one
///     piece of bespoke glue in the Redis journal backend (StackExchange.Redis has no URI parser;
///     <c>redis-py</c>'s <c>Redis.from_url</c> gave Optuna this for free). No server required.
/// </summary>
[TestClass]
public class RedisUrlTests
{
    private static (string Host, int Port) Endpoint(ConfigurationOptions o)
    {
        o.EndPoints.Should().HaveCount(1);
        return o.EndPoints[0] switch
        {
            DnsEndPoint d => (d.Host, d.Port),
            IPEndPoint ip => (ip.Address.ToString(), ip.Port),
            var ep => (ep.ToString()!, -1),
        };
    }

    [TestMethod]
    public void PlainHostPort()
    {
        var o = RedisUrl.BuildConfiguration("redis://localhost:6379");
        Endpoint(o).Should().Be(("localhost", 6379));
        o.Ssl.Should().BeFalse();
        o.DefaultDatabase.Should().BeNull();
        o.User.Should().BeNull();
        o.Password.Should().BeNull();
        // redis-py connects lazily; construction must not fail if the server is momentarily down.
        o.AbortOnConnectFail.Should().BeFalse();
    }

    [TestMethod]
    public void DefaultsPortWhenOmitted()
    {
        Endpoint(RedisUrl.BuildConfiguration("redis://localhost")).Should().Be(("localhost", 6379));
    }

    [TestMethod]
    public void PasswordOnly_NoUser_WithDbInPath()
    {
        var o = RedisUrl.BuildConfiguration("redis://:secret@127.0.0.1:6380/3");
        Endpoint(o).Should().Be(("127.0.0.1", 6380));
        o.Password.Should().Be("secret");
        o.User.Should().BeNull();
        o.DefaultDatabase.Should().Be(3);
    }

    [TestMethod]
    public void UserAndPercentEncodedPassword()
    {
        var o = RedisUrl.BuildConfiguration("redis://alice:p%40ss@redis.example.com:6379/1");
        o.User.Should().Be("alice");
        o.Password.Should().Be("p@ss");   // %40 → '@', unescaped like redis-py.
        o.DefaultDatabase.Should().Be(1);
    }

    [TestMethod]
    public void RedissEnablesTls()
    {
        var o = RedisUrl.BuildConfiguration("rediss://user:pw@redis.example.com:6380");
        o.Ssl.Should().BeTrue();
        o.SslHost.Should().Be("redis.example.com");
        Endpoint(o).Should().Be(("redis.example.com", 6380));
    }

    [TestMethod]
    public void QueryDbOverridesPathDb()
    {
        var o = RedisUrl.BuildConfiguration("redis://host:6379/2?db=5");
        o.DefaultDatabase.Should().Be(5);   // ?db= wins over the /2 path (redis-py precedence).
    }

    [TestMethod]
    public void UnixSocket()
    {
        var o = RedisUrl.BuildConfiguration("unix:///var/run/redis/redis.sock?db=0&password=abc");
        o.EndPoints.Should().HaveCount(1);
        o.EndPoints[0].Should().BeOfType<UnixDomainSocketEndPoint>();
        o.EndPoints[0].ToString().Should().Be("/var/run/redis/redis.sock");
        o.DefaultDatabase.Should().Be(0);
        o.Password.Should().Be("abc");
    }

    [TestMethod]
    public void UnknownScheme_ThrowsValueError_VerbatimRedisPyMessage()
    {
        var act = () => RedisUrl.BuildConfiguration("http://localhost:6379");
        act.Should().Throw<ValueError>()
            .WithMessage("*Redis URL must specify one of the following schemes (redis://, rediss://, unix://)*");
    }

    [TestMethod]
    public void Malformed_ThrowsValueError()
    {
        var act = () => RedisUrl.BuildConfiguration("not-a-url");
        act.Should().Throw<ValueError>().WithMessage("*Invalid Redis URL*");
    }
}

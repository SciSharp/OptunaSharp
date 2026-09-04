using System.Net.Sockets;
using AwesomeAssertions;
using NumSharp;                          // ValueError
using OptunaSharp.Storages.Journal;
using IRedisClientsManager = ServiceStack.Redis.IRedisClientsManager;   // alias avoids the RedisEndpoint name clash

namespace OptunaSharp.Tests.Storage;

/// <summary>
///     Unit tests (no server) for each backend's internal <c>RedisEndpoint</c> → client-config mapper
///     — the ssl / user / password / db / unix-socket branches that a plain <c>redis://localhost</c>
///     integration URL never exercises. Friend access to the internal mappers.
/// </summary>
[TestClass]
public class BackendConfigTests
{
    private static RedisEndpoint Rich => new()
    {
        Host = "h", Port = 6380, Ssl = true, Db = 3, User = "u", Password = "p",
    };

    [TestMethod]
    public void StackExchange_ToConfiguration_MapsAllFields()
    {
        var o = StackExchangeJournalRedisBackend.ToConfiguration(Rich);
        o.Ssl.Should().BeTrue();
        o.SslHost.Should().Be("h");
        o.User.Should().Be("u");
        o.Password.Should().Be("p");
        o.DefaultDatabase.Should().Be(3);
        o.AbortOnConnectFail.Should().BeFalse();
    }

    [TestMethod]
    public void StackExchange_ToConfiguration_UnixSocket()
    {
        var o = StackExchangeJournalRedisBackend.ToConfiguration(new RedisEndpoint { UnixSocketPath = "/tmp/redis.sock" });
        o.EndPoints[0].Should().BeOfType<UnixDomainSocketEndPoint>();
    }

    [TestMethod]
    public void NRedisStack_ToConfiguration_MapsAllFields()
    {
        var o = NRedisStackJournalRedisBackend.ToConfiguration(Rich);
        o.Ssl.Should().BeTrue();
        o.SslHost.Should().Be("h");
        o.User.Should().Be("u");
        o.Password.Should().Be("p");
        o.DefaultDatabase.Should().Be(3);
    }

    [TestMethod]
    public void NRedisStack_ToConfiguration_UnixSocket()
    {
        var o = NRedisStackJournalRedisBackend.ToConfiguration(new RedisEndpoint { UnixSocketPath = "/tmp/redis.sock" });
        o.EndPoints[0].Should().BeOfType<UnixDomainSocketEndPoint>();
    }

    [TestMethod]
    public void ServiceStack_CreateManager_BuildsFromRichEndpoint()
    {
        // Constructing the pool is lazy (no server contacted); this exercises the ssl/db/user/password mapping.
        using IRedisClientsManager manager = ServiceStackJournalRedisBackend.CreateManager(Rich);
        manager.Should().NotBeNull();
    }

    [TestMethod]
    public void ServiceStack_CreateManager_UnixSocket_ThrowsValueError()
    {
        var act = () => ServiceStackJournalRedisBackend.CreateManager(new RedisEndpoint { UnixSocketPath = "/tmp/redis.sock" });
        act.Should().Throw<ValueError>().WithMessage("*unix domain sockets*");
    }
}

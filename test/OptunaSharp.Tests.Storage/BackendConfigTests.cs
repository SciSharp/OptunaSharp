using System;
using System.Net.Sockets;
using AwesomeAssertions;
using NumSharp;                          // ValueError
using OptunaSharp.Storages.Journal;
using StackExchange.Redis;               // ConfigurationOptions (no RedisEndpoint here, so no name clash)
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

    // --------------------------------------------------- native "URL + tweak" callback path (no server)
    //
    // The Action<...> overloads let a caller keep the redis:// URL yet reach the client's own options.
    // These exercise the mapping+callback directly (no connection is opened).

    [TestMethod]
    public void StackExchange_Configure_AppliesTweakOverUrlDerivedOptions()
    {
        ConfigurationOptions? seen = null;
        var o = StackExchangeJournalRedisBackend.Configure(Rich, opt => { seen = opt; opt.ClientName = "opt-app"; });
        seen.Should().BeSameAs(o);            // the caller edits the very options that will be used
        o.DefaultDatabase.Should().Be(3);     // URL-derived (from Rich)
        o.Ssl.Should().BeTrue();              // URL-derived
        o.ClientName.Should().Be("opt-app");  // the native tweak
    }

    [TestMethod]
    public void NRedisStack_Configure_AppliesTweakOverUrlDerivedOptions()
    {
        var o = NRedisStackJournalRedisBackend.Configure(Rich, opt => opt.ClientName = "opt-app");
        o.DefaultDatabase.Should().Be(3);
        o.ClientName.Should().Be("opt-app");
    }

    [TestMethod]
    public void Configure_NullCallback_ThrowsArgumentNull()
    {
        var act = () => StackExchangeJournalRedisBackend.Configure(Rich, null!);
        act.Should().Throw<ArgumentNullException>();
    }

    [TestMethod]
    public void ServiceStack_CreateManager_WithConfigure_PassesUrlDerivedEndpointToCallback()
    {
        // Pool construction is lazy (no server contacted); this proves the callback runs on the endpoint
        // mapped from the URL, and its edits are on the very endpoint the manager is built from.
        ServiceStack.Redis.RedisEndpoint? seen = null;
        using IRedisClientsManager manager =
            ServiceStackJournalRedisBackend.CreateManager(Rich, sep => { seen = sep; sep.Client = "cfg"; });
        seen.Should().NotBeNull();
        seen!.Host.Should().Be("h");    // URL-derived (from Rich)
        seen.Db.Should().Be(3);         // URL-derived
        seen.Ssl.Should().BeTrue();     // URL-derived
        seen.Client.Should().Be("cfg"); // the native tweak
        manager.Should().NotBeNull();
    }

    [TestMethod]
    public void ServiceStack_CreateManager_WithConfigure_UnixSocket_ThrowsValueError()
    {
        var act = () => ServiceStackJournalRedisBackend.CreateManager(
            new RedisEndpoint { UnixSocketPath = "/tmp/redis.sock" }, _ => { });
        act.Should().Throw<ValueError>().WithMessage("*unix domain sockets*");
    }

    [TestMethod]
    public void ServiceStack_CreateManager_WithConfigure_NullCallback_ThrowsArgumentNull()
    {
        var act = () => ServiceStackJournalRedisBackend.CreateManager(Rich, null!);
        act.Should().Throw<ArgumentNullException>();
    }
}

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

    // --------------------------------------------------- redis-py's URL options, mapped onto each client (no server)

    /// <summary>
    ///     StackExchange.Redis and NRedisStack get redis-py 8's 5-second command and connect timeouts explicitly, so the
    ///     connect timeout never follows a longer command timeout the way StackExchange.Redis' own default would.
    /// </summary>
    [TestMethod]
    public void StackExchangeAndNRedisStack_Timeouts_AreRedisPys()
    {
        foreach (ConfigurationOptions o in new[]
                 {
                     StackExchangeJournalRedisBackend.ToConfiguration(RedisUrl.Parse("redis://h")),
                     NRedisStackJournalRedisBackend.ToConfiguration(RedisUrl.Parse("redis://h")),
                 })
        {
            o.SyncTimeout.Should().Be(5000);
            o.AsyncTimeout.Should().Be(5000);
            o.ConnectTimeout.Should().Be(5000);
        }
        var tuned = StackExchangeJournalRedisBackend.ToConfiguration(RedisUrl.Parse("redis://h?socket_timeout=12&socket_connect_timeout=0.5"));
        tuned.SyncTimeout.Should().Be(12000);
        tuned.ConnectTimeout.Should().Be(500);
    }

    /// <summary>The URL's client name, library name and protocol reach StackExchange.Redis' and NRedisStack's options.</summary>
    [TestMethod]
    public void StackExchangeAndNRedisStack_NamesAndProtocol_AreMapped()
    {
        const string url = "redis://h?client_name=optuna&lib_name=lib&protocol=3";
        foreach (ConfigurationOptions o in new[]
                 {
                     StackExchangeJournalRedisBackend.ToConfiguration(RedisUrl.Parse(url)),
                     NRedisStackJournalRedisBackend.ToConfiguration(RedisUrl.Parse(url)),
                 })
        {
            o.ClientName.Should().Be("optuna");
            o.LibraryName.Should().Be("lib");
            o.Protocol.Should().Be(RedisProtocol.Resp3);
        }
        StackExchangeJournalRedisBackend.ToConfiguration(RedisUrl.Parse("redis://h?protocol=2")).Protocol.Should().Be(RedisProtocol.Resp2);
    }

    /// <summary>
    ///     A <c>rediss://</c> URL turns revocation checking off (Python's <c>ssl</c> never checks revocation, while
    ///     StackExchange.Redis does by default) and maps <c>ssl_min_version</c> onto the allowed TLS versions.
    /// </summary>
    [TestMethod]
    public void StackExchange_Tls_MatchesPythonsSslContext()
    {
        var o = StackExchangeJournalRedisBackend.ToConfiguration(RedisUrl.Parse("rediss://h:6380?ssl_min_version=771"));
        o.Ssl.Should().BeTrue();
        o.SslHost.Should().Be("h");
        o.CheckCertificateRevocation.Should().BeFalse();
        o.SslProtocols.Should().Be(System.Security.Authentication.SslProtocols.Tls12 | System.Security.Authentication.SslProtocols.Tls13);
        NRedisStackJournalRedisBackend.ToConfiguration(RedisUrl.Parse("rediss://h")).CheckCertificateRevocation.Should().BeFalse();
    }

    /// <summary>
    ///     Credentials made only of white space are refused: StackExchange.Redis would skip authenticating with them,
    ///     where redis-py sends them as they are.
    /// </summary>
    [TestMethod]
    public void StackExchangeAndNRedisStack_WhiteSpaceCredentials_AreRefused()
    {
        var blankUser = () => StackExchangeJournalRedisBackend.ToConfiguration(RedisUrl.Parse("redis://%20:pw@h"));
        blankUser.Should().Throw<NotSupportedException>();
        var blankPassword = () => NRedisStackJournalRedisBackend.ToConfiguration(RedisUrl.Parse("redis://:%20@h"));
        blankPassword.Should().Throw<NotSupportedException>();
        // A real user with a blank password is sent as redis-py sends it: AUTH user "  ".
        StackExchangeJournalRedisBackend.ToConfiguration(RedisUrl.Parse("redis://u:%20@h")).User.Should().Be("u");
    }

    /// <summary>
    ///     ServiceStack.Redis gets the URL's client name and redis-py's timeouts as the endpoint's connect, send and
    ///     receive timeouts, and its pool serves that very endpoint object.
    /// </summary>
    [TestMethod]
    public void ServiceStack_Endpoint_CarriesNameAndTimeouts()
    {
        ServiceStack.Redis.RedisEndpoint? seen = null;
        using IRedisClientsManager manager = ServiceStackJournalRedisBackend.CreateManager(
            RedisUrl.Parse("redis://:pw@h:6390/3?client_name=optuna&socket_timeout=2&socket_connect_timeout=1"), sep => seen = sep);
        seen!.Client.Should().Be("optuna");
        seen.SendTimeout.Should().Be(2000);
        seen.ReceiveTimeout.Should().Be(2000);
        seen.ConnectTimeout.Should().Be(1000);
        seen.Db.Should().Be(3);
        ((ServiceStack.Redis.RedisResolver)((ServiceStack.Redis.RedisManagerPool)manager).RedisResolver).PrimaryEndpoint
            .Should().BeSameAs(seen);
    }

    /// <summary>
    ///     What ServiceStack.Redis cannot honor per connection is refused: RESP3, a user name without a password,
    ///     certificate settings beyond the default check, and a minimum TLS version (which would turn revocation
    ///     checking on).
    /// </summary>
    /// <param name="url">The URL.</param>
    [TestMethod]
    [DataRow("redis://h?protocol=3")]
    [DataRow("redis://user@h")]
    [DataRow("rediss://h?ssl_cert_reqs=none")]
    [DataRow("rediss://h?ssl_check_hostname=false")]
    [DataRow("rediss://h?ssl_ca_certs=/x/ca.pem")]
    [DataRow("rediss://h?ssl_certfile=c")]
    [DataRow("rediss://h?ssl_min_version=771")]
    public void ServiceStack_UnhonorableSettings_AreRefused(string url)
    {
        var act = () => ServiceStackJournalRedisBackend.CreateManager(RedisUrl.Parse(url));
        act.Should().Throw<NotSupportedException>();
    }
}

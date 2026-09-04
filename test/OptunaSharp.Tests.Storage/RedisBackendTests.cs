using System.Linq;
using AwesomeAssertions;
using OptunaSharp.Collections;
using OptunaSharp.Storages.Journal;
using ServiceStack.Redis;
using StackExchange.Redis;

namespace OptunaSharp.Tests.Storage;

// The three concrete backend suites. Each inherits the full JournalBackendContractTests set (run against
// its client) and adds the one client-specific "bring your own connection" ctor test.

/// <summary>The full contract against the StackExchange.Redis backend + its shared-multiplexer ctor.</summary>
[TestClass]
[TestCategory("Integration")]
public class StackExchangeBackendTests : JournalBackendContractTests
{
    protected override string ExperimentalTypeName => "StackExchangeJournalRedisBackend";

    protected override BaseJournalBackend Create(string prefix, bool useCluster = false)
        => new StackExchangeJournalRedisBackend(RedisSession.Url, useCluster, prefix);

    [TestMethod]
    [TestCategory("Integration")]
    public void SharedMultiplexerCtor_RoundTrips()
    {
        RedisSession.EnsureReachable();
        using ConnectionMultiplexer mux = RedisSession.RawConnection();
        string prefix = RedisSession.UniquePrefix();
        var backend = new StackExchangeJournalRedisBackend(mux, useCluster: false, prefix: prefix);
        backend.AppendLogs(new[] { new OrderedDict<string, object?> { ["op"] = 0L, ["worker"] = "shared" } });
        backend.ReadLogs(0).Single()["worker"].Should().Be("shared");
    }

    [TestMethod]
    [TestCategory("Integration")]
    public void ConfigurationOptionsCtor_RoundTrips()
    {
        RedisSession.EnsureReachable();
        ConfigurationOptions options = StackExchangeJournalRedisBackend.ToConfiguration(RedisUrl.Parse(RedisSession.Url));
        options.ClientName = "optunasharp-options";
        string prefix = RedisSession.UniquePrefix();
        var backend = new StackExchangeJournalRedisBackend(options, useCluster: false, prefix: prefix);
        backend.AppendLogs(new[] { new OrderedDict<string, object?> { ["op"] = 0L, ["worker"] = "options" } });
        backend.ReadLogs(0).Single()["worker"].Should().Be("options");
    }

    [TestMethod]
    [TestCategory("Integration")]
    public void ConfigureCallbackCtor_RoundTrips()
    {
        RedisSession.EnsureReachable();
        string prefix = RedisSession.UniquePrefix();
        var backend = new StackExchangeJournalRedisBackend(
            RedisSession.Url, o => o.ClientName = "optunasharp-configure", useCluster: false, prefix: prefix);
        backend.AppendLogs(new[] { new OrderedDict<string, object?> { ["op"] = 0L, ["worker"] = "configure" } });
        backend.ReadLogs(0).Single()["worker"].Should().Be("configure");
    }
}

/// <summary>The full contract against the NRedisStack backend + its shared-multiplexer ctor.</summary>
[TestClass]
[TestCategory("Integration")]
public class NRedisStackBackendTests : JournalBackendContractTests
{
    protected override string ExperimentalTypeName => "NRedisStackJournalRedisBackend";

    protected override BaseJournalBackend Create(string prefix, bool useCluster = false)
        => new NRedisStackJournalRedisBackend(RedisSession.Url, useCluster, prefix);

    [TestMethod]
    [TestCategory("Integration")]
    public void SharedMultiplexerCtor_RoundTrips()
    {
        RedisSession.EnsureReachable();
        using ConnectionMultiplexer mux = RedisSession.RawConnection();
        string prefix = RedisSession.UniquePrefix();
        var backend = new NRedisStackJournalRedisBackend(mux, useCluster: false, prefix: prefix);
        backend.AppendLogs(new[] { new OrderedDict<string, object?> { ["op"] = 0L, ["worker"] = "shared" } });
        backend.ReadLogs(0).Single()["worker"].Should().Be("shared");
    }

    [TestMethod]
    [TestCategory("Integration")]
    public void ConfigurationOptionsCtor_RoundTrips()
    {
        RedisSession.EnsureReachable();
        ConfigurationOptions options = NRedisStackJournalRedisBackend.ToConfiguration(RedisUrl.Parse(RedisSession.Url));
        options.ClientName = "optunasharp-options";
        string prefix = RedisSession.UniquePrefix();
        var backend = new NRedisStackJournalRedisBackend(options, useCluster: false, prefix: prefix);
        backend.AppendLogs(new[] { new OrderedDict<string, object?> { ["op"] = 0L, ["worker"] = "options" } });
        backend.ReadLogs(0).Single()["worker"].Should().Be("options");
    }

    [TestMethod]
    [TestCategory("Integration")]
    public void ConfigureCallbackCtor_RoundTrips()
    {
        RedisSession.EnsureReachable();
        string prefix = RedisSession.UniquePrefix();
        var backend = new NRedisStackJournalRedisBackend(
            RedisSession.Url, o => o.ClientName = "optunasharp-configure", useCluster: false, prefix: prefix);
        backend.AppendLogs(new[] { new OrderedDict<string, object?> { ["op"] = 0L, ["worker"] = "configure" } });
        backend.ReadLogs(0).Single()["worker"].Should().Be("configure");
    }
}

/// <summary>The full contract against the ServiceStack.Redis backend + its shared-manager ctor.</summary>
[TestClass]
[TestCategory("Integration")]
public class ServiceStackBackendTests : JournalBackendContractTests
{
    protected override string ExperimentalTypeName => "ServiceStackJournalRedisBackend";

    protected override BaseJournalBackend Create(string prefix, bool useCluster = false)
        => new ServiceStackJournalRedisBackend(RedisSession.Url, useCluster, prefix);

    [TestMethod]
    [TestCategory("Integration")]
    public void SharedManagerCtor_RoundTrips()
    {
        RedisSession.EnsureReachable();
        using IRedisClientsManager manager = ServiceStackJournalRedisBackend.CreateManager(RedisUrl.Parse(RedisSession.Url));
        string prefix = RedisSession.UniquePrefix();
        var backend = new ServiceStackJournalRedisBackend(manager, useCluster: false, prefix: prefix);
        backend.AppendLogs(new[] { new OrderedDict<string, object?> { ["op"] = 0L, ["worker"] = "shared" } });
        backend.ReadLogs(0).Single()["worker"].Should().Be("shared");
    }

    [TestMethod]
    [TestCategory("Integration")]
    public void EndpointCtor_RoundTrips()
    {
        RedisSession.EnsureReachable();
        var ep = RedisUrl.Parse(RedisSession.Url);
        var sep = new ServiceStack.Redis.RedisEndpoint(ep.Host, ep.Port <= 0 ? 6379 : ep.Port)
        {
            Ssl = ep.Ssl, Db = ep.Db ?? 0, Password = ep.Password, Username = ep.User,
        };
        string prefix = RedisSession.UniquePrefix();
        var backend = new ServiceStackJournalRedisBackend(sep, useCluster: false, prefix: prefix);
        backend.AppendLogs(new[] { new OrderedDict<string, object?> { ["op"] = 0L, ["worker"] = "endpoint" } });
        backend.ReadLogs(0).Single()["worker"].Should().Be("endpoint");
    }

    [TestMethod]
    [TestCategory("Integration")]
    public void ConfigureCallbackCtor_RoundTrips()
    {
        RedisSession.EnsureReachable();
        string prefix = RedisSession.UniquePrefix();
        var backend = new ServiceStackJournalRedisBackend(
            RedisSession.Url, sep => sep.Client = "optunasharp-configure", useCluster: false, prefix: prefix);
        backend.AppendLogs(new[] { new OrderedDict<string, object?> { ["op"] = 0L, ["worker"] = "configure" } });
        backend.ReadLogs(0).Single()["worker"].Should().Be("configure");
    }
}

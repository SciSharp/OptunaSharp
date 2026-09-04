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
        var backend = new StackExchangeJournalRedisBackend(RedisSession.Url, mux, useCluster: false, prefix: prefix);
        backend.AppendLogs(new[] { new OrderedDict<string, object?> { ["op"] = 0L, ["worker"] = "shared" } });
        backend.ReadLogs(0).Single()["worker"].Should().Be("shared");
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
        var backend = new NRedisStackJournalRedisBackend(RedisSession.Url, mux, useCluster: false, prefix: prefix);
        backend.AppendLogs(new[] { new OrderedDict<string, object?> { ["op"] = 0L, ["worker"] = "shared" } });
        backend.ReadLogs(0).Single()["worker"].Should().Be("shared");
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
        var backend = new ServiceStackJournalRedisBackend(RedisSession.Url, manager, useCluster: false, prefix: prefix);
        backend.AppendLogs(new[] { new OrderedDict<string, object?> { ["op"] = 0L, ["worker"] = "shared" } });
        backend.ReadLogs(0).Single()["worker"].Should().Be("shared");
    }
}

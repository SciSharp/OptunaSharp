using System;
using System.Net;
using System.Net.Sockets;
using AwesomeAssertions;
using Microsoft.AspNetCore.Builder;
using Microsoft.Extensions.Hosting;
using OptunaSharp.Collections;
using OptunaSharp.Distributions;
using OptunaSharp.Samplers;
using OptunaSharp.Storages;
using OptunaSharp.Storages.Grpc;

namespace OptunaSharp.Tests.Interop;

/// <summary>
///     End-to-end gRPC round-trips over <b>localhost</b> with an OptunaSharp server AND OptunaSharp
///     client(s): <see cref="GrpcProxyServer.MakeServer" /> binds a real loopback TCP socket (Kestrel,
///     cleartext HTTP/2), and each <see cref="GrpcStorageProxy" /> opens a real HTTP/2 connection to it —
///     so every operation actually traverses the wire, not just an in-process call.
///
///     <para>These are pure C# (no Python), so they always run — the language-internal counterpart to the
///     cross-language gate in <c>GrpcStorageProxyInteropTests</c>. They cover a full study through the
///     proxy, two independent clients sharing one server (the distributed-worker pattern the proxy exists
///     for), and multi-objective + NaN/±inf values over the wire.</para>
/// </summary>
[TestClass]
public class GrpcLocalhostInteropTests
{
    private InMemoryStorage _backend = null!;
    private WebApplication _app = null!;
    private int _port;

    [TestInitialize]
    public void Init()
    {
        _port = FreePort();
        _backend = new InMemoryStorage();
        _app = GrpcProxyServer.MakeServer(_backend, "localhost", _port);
        _app.StartAsync().GetAwaiter().GetResult();
    }

    [TestCleanup]
    public void Cleanup()
    {
        if (_app is not null)
        {
            _app.StopAsync().GetAwaiter().GetResult();
            _app.DisposeAsync().AsTask().GetAwaiter().GetResult();
        }
    }

    private static int FreePort()
    {
        var listener = new TcpListener(IPAddress.Loopback, 0);
        listener.Start();
        int port = ((IPEndPoint)listener.LocalEndpoint).Port;
        listener.Stop();
        return port;
    }

    /// <summary>A fresh client (its own HTTP/2 connection) to the running localhost server.</summary>
    private GrpcStorageProxy Connect()
    {
        var proxy = new GrpcStorageProxy("localhost", _port);
        proxy.WaitServerReady(TimeSpan.FromSeconds(15));
        return proxy;
    }

    /// <summary>Mixed param types + an intermediate report incl. +inf.</summary>
    private static double MixedObjective(Trial t)
    {
        double x = t.SuggestFloat("x", -10, 10);
        long y = t.SuggestInt("y", 1, 100, log: true);
        _ = t.SuggestCategorical("z", new CategoricalChoice[] { "a", "b", (long)3, true, CategoricalChoice.None });
        t.Report(0.5, 0);
        t.Report(double.PositiveInfinity, 1);
        return x * x + y;
    }

    [TestMethod]
    public void Server_And_Client_FullStudy_Localhost()
    {
        using var client = Connect();

        var study = Optuna.CreateStudy(storage: client, studyName: "s", sampler: new RandomSampler(0));
        study.SetUserAttr("owner", "eli");
        study.SetUserAttr("run", 7L);
        study.Optimize(MixedObjective, nTrials: 12);

        // The client's over-the-wire view must equal the study read straight from the backing storage.
        var clientState = StudySnapshot.Capture(study);
        var backendState = StudySnapshot.Capture(Optuna.LoadStudy("s", _backend));
        StudySnapshot.AssertParity(backendState, clientState, DatetimeComparison.NullnessOnly);

        study.Trials.Count.Should().Be(12);
        study.BestTrial.IntermediateValues[1].Should().Be(double.PositiveInfinity);
        study.UserAttrs["owner"].Should().Be("eli");
    }

    [TestMethod]
    public void Server_And_TwoClients_ShareStudy_Localhost()
    {
        // Worker A creates and optimizes a study through its own connection.
        using var workerA = Connect();
        var studyA = Optuna.CreateStudy(storage: workerA, studyName: "shared", sampler: new RandomSampler(1));
        studyA.Optimize(MixedObjective, nTrials: 8);

        // Worker B, on a separate connection to the same server, sees the identical study.
        using var workerB = Connect();
        var studyB = Optuna.LoadStudy("shared", workerB);
        studyB.Trials.Count.Should().Be(8);
        studyB.BestValue.Should().Be(studyA.BestValue);
        StudySnapshot.AssertParity(
            StudySnapshot.Capture(studyA), StudySnapshot.Capture(studyB), DatetimeComparison.NullnessOnly);

        // Worker B adds more trials; Worker A (and the backing storage) observe them — one shared store.
        studyB.Optimize(MixedObjective, nTrials: 4);

        int sid = workerA.GetStudyIdFromName("shared");
        workerA.GetAllTrials(sid).Should().HaveCount(12); // A's proxy re-fetches B's additions over the wire
        _backend.GetAllTrials(sid).Should().HaveCount(12); // and they landed in the one backing storage
    }

    [TestMethod]
    public void Server_And_Client_MultiObjective_And_Nonfinite_Localhost()
    {
        using var client = Connect();

        // Single-objective study whose stored value is +inf, with NaN/±inf intermediates.
        var single = Optuna.CreateStudy(storage: client, studyName: "single", direction: StudyDirection.Minimize);
        single.AddTrial(FrozenTrial.CreateTrial(
            state: TrialState.Complete,
            value: double.PositiveInfinity,
            @params: new OrderedDict<string, object?> { ["x"] = 1.0 },
            distributions: new OrderedDict<string, BaseDistribution> { ["x"] = new FloatDistribution(-10, 10) },
            intermediateValues: new OrderedDict<long, double>
            {
                [0] = 0.5, [1] = double.PositiveInfinity, [2] = double.NaN, [3] = double.NegativeInfinity,
            }));
        single.AddTrial(FrozenTrial.CreateTrial(
            state: TrialState.Complete,
            value: 3.5,
            @params: new OrderedDict<string, object?> { ["x"] = -2.0 },
            distributions: new OrderedDict<string, BaseDistribution> { ["x"] = new FloatDistribution(-10, 10) }));

        // Multi-objective study.
        var mo = Optuna.CreateStudy(
            storage: client, studyName: "mo", directions: new[] { StudyDirection.Minimize, StudyDirection.Maximize });
        mo.AddTrial(FrozenTrial.CreateTrial(
            state: TrialState.Complete,
            values: new[] { 0.3, 10.0 },
            @params: new OrderedDict<string, object?> { ["a"] = 0.3, ["b"] = 5L },
            distributions: new OrderedDict<string, BaseDistribution>
            {
                ["a"] = new FloatDistribution(0, 1), ["b"] = new IntDistribution(1, 10),
            }));

        StudySnapshot.AssertParity(
            StudySnapshot.Capture(Optuna.LoadStudy("single", _backend)),
            StudySnapshot.Capture(single), DatetimeComparison.NullnessOnly);
        StudySnapshot.AssertParity(
            StudySnapshot.Capture(Optuna.LoadStudy("mo", _backend)),
            StudySnapshot.Capture(mo), DatetimeComparison.NullnessOnly);
    }
}

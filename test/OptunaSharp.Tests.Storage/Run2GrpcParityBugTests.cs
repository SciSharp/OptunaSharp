using System;
using System.Linq;
using System.Net;
using System.Net.Sockets;
using AwesomeAssertions;
using Microsoft.AspNetCore.Builder;
using OptunaSharp;
using OptunaSharp.Samplers;
using OptunaSharp.Storages;
using OptunaSharp.Storages.Grpc;

namespace OptunaSharp.Tests.Storage;

/// <summary>
///     Open parity bugs in the gRPC storage proxy found by the 2026-11-10 parity QA run
///     (<c>docs/qa/PARITY_QA_2026-11-10.md</c>). Each test fails today and passes once fixed, at which point its
///     <c>OpenBugs</c> category should be removed. Every test hosts the C# proxy server in-process over an
///     <see cref="InMemoryStorage" /> and talks to it through the C# <see cref="GrpcStorageProxy" />, so no Python and
///     no external service are needed.
/// </summary>
[TestClass]
public class Run2GrpcParityBugTests
{
    private WebApplication _app = null!;
    private GrpcStorageProxy _proxy = null!;

    /// <summary>Starts a proxy server over a fresh in-memory storage on a free loopback port and connects a client.</summary>
    [TestInitialize]
    public void Init()
    {
        int port;
        // Ask the OS for a free port, then release it for Kestrel; a race with another process is possible but rare.
        var listener = new TcpListener(IPAddress.Loopback, 0);
        listener.Start();
        port = ((IPEndPoint)listener.LocalEndpoint).Port;
        listener.Stop();
        _app = GrpcProxyServer.MakeServer(new InMemoryStorage(), "localhost", port);
        _app.StartAsync().GetAwaiter().GetResult();
        _proxy = new GrpcStorageProxy("localhost", port);
        _proxy.WaitServerReady(TimeSpan.FromSeconds(15));
    }

    /// <summary>Disposes the client, then stops and disposes the server.</summary>
    [TestCleanup]
    public void Cleanup()
    {
        _proxy?.Dispose();
        if (_app is not null)
        {
            _app.StopAsync().GetAwaiter().GetResult();
            _app.DisposeAsync().AsTask().GetAwaiter().GetResult();
        }
    }

    /// <summary>
    ///     QA-69: Optuna's <c>GrpcStorageProxy</c> rebuilds every <c>FrozenTrial</c> from protobuf maps
    ///     (<c>_from_proto_trial</c>: <c>{key: ... for key, value in trial.params.items()}</c>, the same for
    ///     <c>distributions</c>, <c>user_attributes</c>, <c>system_attributes</c>; the study attributes likewise), and
    ///     the Python protobuf runtime (upb) iterates a map in its hash-table order, not in insertion order. So through
    ///     Optuna's proxy <c>trial.params</c>, <c>trial.distributions</c> and the attribute dictionaries come back in a
    ///     fixed hash order whatever server sits behind it. The port's client keeps the wire order: insertion order
    ///     from a C# server, the Python server's own serialization order from a Python server.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///         Measured with Optuna 5.0.0, grpcio 1.78.0 and protobuf 6.33.5 (upb): nine floats suggested as
    ///         <c>u0, y1, depth2, x3, lr4, z5, a, b, c</c> read back as <c>depth2, u0, x3, b, y1, z5, lr4, c, a</c>
    ///         through a Python client against a Python or a C# server (the same in three processes with different
    ///         <c>PYTHONHASHSEED</c>s); user attributes <c>k1, a2, zz</c> read back as <c>a2, zz, k1</c>. The port's client
    ///         reads <c>u0, y1, ...</c> from a C# server and <c>a, c, lr4, z5, y1, depth2, x3, u0, b</c> from a Python
    ///         server.
    ///     </para>
    ///     <para>
    ///         Beyond the visible key order, two samplers depend on it: <c>QMCSampler</c> takes its relative search space
    ///         as <c>dict(first_trial.distributions)</c>, so the parameter-to-dimension assignment follows this order
    ///         (see <see cref="QmcThroughGrpc_AssignsDimensionsInOptunasProxyOrder" />), and <c>BruteForceSampler</c>
    ///         builds its tree from <c>trial.distributions.items()</c>. A random-program fuzz through the proxy changed
    ///         Optuna's own trajectory against the in-memory one in 41 of 65 QMC and 3 of 26 BruteForce programs; the
    ///         port's trajectories through its proxy equal its in-memory ones, so they differ from Optuna's there. Random,
    ///         TPE, CMA-ES, NSGA-II / III and Grid read sorted or per-parameter spaces and were unaffected.
    ///     </para>
    ///     <para>
    ///         Whether the port should reproduce upb's map order (an implementation artifact of Python protobuf) or keep
    ///         the order of every other storage is a decision for the owner (question Q6 in the journal).
    ///     </para>
    /// </remarks>
    [TestMethod]
    [TestCategory("OpenBugs")]
    public void TrialDictionaries_ReadBackInOptunasProxyOrder()
    {
        var study = Optuna.CreateStudy(storage: _proxy, studyName: "order");
        var trial = study.Ask();
        foreach (var name in new[] { "u0", "y1", "depth2", "x3", "lr4", "z5", "a", "b", "c" })
            trial.SuggestFloat(name, 0, 1);
        trial.SetUserAttr("k1", 1);
        trial.SetUserAttr("a2", 2);
        trial.SetUserAttr("zz", 3);
        study.Tell(trial, 0.0);

        var frozen = study.Trials[0];
        frozen.Params.Keys.Should().Equal(new[] { "depth2", "u0", "x3", "b", "y1", "z5", "lr4", "c", "a" },
            "Optuna's GrpcStorageProxy returns the parameters in protobuf (upb) map order");
        frozen.Distributions.Keys.Should().Equal(new[] { "depth2", "u0", "x3", "b", "y1", "z5", "lr4", "c", "a" });
        frozen.UserAttrs.Keys.Should().Equal(new[] { "a2", "zz", "k1" });
    }

    /// <summary>
    ///     QA-69, the sampler consequence: <c>QMCSampler(seed=0)</c> over <c>u0, y1, depth2</c> (each in [0, 1]) through
    ///     the proxy. Optuna's relative search space follows the first trial's <c>distributions</c>, which its proxy
    ///     returns as <c>depth2, u0, y1</c>, so the first Sobol dimension goes to <c>depth2</c>: trial 3 is
    ///     <c>u0 = 0.25, y1 = 0.25, depth2 = 0.75</c>, where the in-memory study (and the port, on either storage) gives
    ///     <c>u0 = 0.75, y1 = 0.25, depth2 = 0.25</c>.
    /// </summary>
    [TestMethod]
    [TestCategory("OpenBugs")]
    public void QmcThroughGrpc_AssignsDimensionsInOptunasProxyOrder()
    {
        var study = Optuna.CreateStudy(storage: _proxy, studyName: "qmc", sampler: new QMCSampler(seed: 0));
        study.Optimize(t => t.SuggestFloat("u0", 0, 1) + t.SuggestFloat("y1", 0, 1) + t.SuggestFloat("depth2", 0, 1),
            nTrials: 4);

        var third = study.Trials[3].Params;
        // Optuna 5.0.0 through GrpcStorageProxy (grpcio 1.78.0, protobuf 6.33.5 upb).
        ((double)third["u0"]!).Should().Be(0.25);
        ((double)third["y1"]!).Should().Be(0.25);
        ((double)third["depth2"]!).Should().Be(0.75);
    }
}

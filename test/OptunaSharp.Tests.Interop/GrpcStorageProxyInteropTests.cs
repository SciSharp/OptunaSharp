using System;
using System.Net;
using System.Net.Sockets;
using System.Threading;
using AwesomeAssertions;
using Microsoft.Extensions.Hosting;
using OptunaSharp.Collections;
using OptunaSharp.Distributions;
using OptunaSharp.Samplers;
using OptunaSharp.Storages;
using OptunaSharp.Storages.Grpc;

namespace OptunaSharp.Tests.Interop;

/// <summary>
///     The Phase 4.4 cross-language gate: <b>an OptunaSharp gRPC proxy and a Python Optuna gRPC proxy
///     server interoperate on the wire — in both directions.</b> Because both sides are generated from
///     Optuna's verbatim <c>api.proto</c>, a study created/optimized through one language's client, over
///     the other language's server, is read back identically:
///     <list type="number">
///       <item><description><see cref="CSharpServer_PythonClient_SameStorage" /> — OptunaSharp hosts
///         <see cref="GrpcProxyServer.MakeServer" />; Python's <c>GrpcStorageProxy</c> writes the study;
///         C# reads the backing storage back.</description></item>
///       <item><description><see cref="PythonServer_CSharpClient_SameStorage" /> — Python hosts
///         <c>run_grpc_proxy_server</c>; OptunaSharp's <see cref="GrpcStorageProxy" /> writes the study;
///         Python reads the backing storage back.</description></item>
///     </list>
///
///     <para>Values transit as IEEE-754 doubles and the attr/distribution strings as the same
///     <c>json.dumps</c>/<c>distribution_to_json</c> bytes, so the round-trip is exact (datetimes compared
///     for nullness only, like the SQLite gate). Self-skips (Inconclusive) when Python/optuna is absent or
///     when <c>grpcio</c> / the optuna gRPC proxy is not importable.</para>
/// </summary>
[TestClass]
public class GrpcStorageProxyInteropTests : OptunaInteropTestBase
{
    /// <summary>The C# objective — mixed param types, an intermediate report, an ±inf-friendly value.</summary>
    private static double MixedObjective(Trial t)
    {
        double x = t.SuggestFloat("x", -10, 10);
        long y = t.SuggestInt("y", 1, 100, log: true);
        _ = t.SuggestCategorical("z", new CategoricalChoice[] { "a", "b", (long)3, true, CategoricalChoice.None });
        t.Report(0.5, 0);
        t.Report(double.PositiveInfinity, 1);
        return x * x + y;
    }

    /// <summary>The identical objective on the Python side.</summary>
    private const string PyMixedObjective =
        "def _obj(t):\n" +
        "    x = t.suggest_float('x', -10, 10)\n" +
        "    y = t.suggest_int('y', 1, 100, log=True)\n" +
        "    z = t.suggest_categorical('z', ['a', 'b', 3, True, None])\n" +
        "    t.report(0.5, 0)\n" +
        "    t.report(float('inf'), 1)\n" +
        "    return x * x + y\n";

    /// <summary>Require a Python engine AND an importable <c>grpcio</c> + optuna gRPC proxy, else Inconclusive.</summary>
    private void EnsureGrpcOrInconclusive()
    {
        PythonSession.EnsureOrInconclusive();
        bool importable;
        using (Gil())
        {
            try
            {
                Scope.Exec("import grpc\nfrom optuna.storages import GrpcStorageProxy, run_grpc_proxy_server\n");
                importable = true;
            }
            catch (Exception)
            {
                importable = false;
            }
        }
        if (!importable)
            Assert.Inconclusive("Python 'grpcio' / optuna gRPC storage proxy not importable (pip install grpcio); gRPC interop skipped.");
    }

    /// <summary>An ephemeral free loopback TCP port.</summary>
    private static int FreePort()
    {
        var listener = new TcpListener(IPAddress.Loopback, 0);
        listener.Start();
        int port = ((IPEndPoint)listener.LocalEndpoint).Port;
        listener.Stop();
        return port;
    }

    /// <summary>Poll a real RPC until the (Python-hosted) server answers — robust against startup races.</summary>
    private static void WaitReady(GrpcStorageProxy proxy, int seconds = 20)
    {
        var deadline = DateTime.UtcNow.AddSeconds(seconds);
        while (true)
        {
            try
            {
                proxy.GetAllStudies();
                return;
            }
            catch when (DateTime.UtcNow < deadline)
            {
                Thread.Sleep(100);
            }
        }
    }

    /// <summary>
    ///     Review item P2-2, live, as pass 2B ran it: a raw Python gRPC stub sends <c>SetStudyUserAttribute</c> with
    ///     100 KB of nested brackets. Against the OptunaSharp server this used to end the server process with a stack
    ///     overflow ("Stack overflow.", every later call <c>UNAVAILABLE</c>; here the server runs in the test host).
    ///     The same stub script runs against an OptunaSharp server and a Python <c>run_grpc_proxy_server</c>, and both
    ///     must answer the same: <c>UNKNOWN</c> for the hostile request (Python's <c>json.loads</c> raises an uncaught
    ///     <c>RecursionError</c>), then normal service, with only the later, valid attribute stored.
    /// </summary>
    [TestMethod]
    [TestCategory("RequiresPython")]
    public void DeeplyNestedAttributeFromARawStub_FailsOnlyThatRequest_LikeOnAPythonServer()
    {
        EnsureGrpcOrInconclusive();

        int csPort = FreePort();
        var app = GrpcProxyServer.MakeServer(new InMemoryStorage(), "localhost", csPort);
        app.StartAsync().GetAwaiter().GetResult();
        int pyPort = FreePort();
        PyExec(
            "import threading\n" +
            "def _run_deep_server():\n" +
            $"    optuna.storages.run_grpc_proxy_server(optuna.storages.InMemoryStorage(), host='localhost', port={pyPort})\n" +
            "threading.Thread(target=_run_deep_server, daemon=True).start()\n");
        try
        {
            PyExec(
                "import grpc\n" +
                "from optuna.storages._grpc.auto_generated import api_pb2, api_pb2_grpc\n" +
                "def _oss_deep_attr(port):\n" +
                "    channel = grpc.insecure_channel(f'localhost:{port}')\n" +
                "    grpc.channel_ready_future(channel).result(timeout=20)\n" +
                "    stub = api_pb2_grpc.StorageServiceStub(channel)\n" +
                "    sid = stub.CreateNewStudy(api_pb2.CreateNewStudyRequest(\n" +
                "        directions=[api_pb2.MINIMIZE], study_name='deep')).study_id\n" +
                "    try:\n" +
                "        stub.SetStudyUserAttribute(api_pb2.SetStudyUserAttributeRequest(\n" +
                "            study_id=sid, key='k', value='[' * 50000 + ']' * 50000))\n" +
                "        hostile = 'OK'\n" +
                "    except grpc.RpcError as e:\n" +
                "        hostile = e.code().name\n" +
                "    stub.SetStudyUserAttribute(api_pb2.SetStudyUserAttributeRequest(study_id=sid, key='k2', value='[[1]]'))\n" +
                "    attrs = stub.GetStudyUserAttributes(api_pb2.GetStudyUserAttributesRequest(study_id=sid)).user_attributes\n" +
                "    channel.close()\n" +
                "    return hostile + '|' + repr(sorted(attrs.items()))\n");

            string csharpServer = PyStr($"_oss_deep_attr({csPort})");
            string pythonServer = PyStr($"_oss_deep_attr({pyPort})");

            csharpServer.Should().Be(pythonServer, "both servers answer the hostile request and the valid one alike");
            csharpServer.Should().Be("UNKNOWN|[('k2', '[[1]]')]");
        }
        finally
        {
            app.StopAsync().GetAwaiter().GetResult();
            app.DisposeAsync().AsTask().GetAwaiter().GetResult();
        }
    }

    [TestMethod]
    [TestCategory("RequiresPython")]
    [DataRow(0)]
    [DataRow(42)]
    public void CSharpServer_PythonClient_SameStorage(int seed)
    {
        EnsureGrpcOrInconclusive();

        int port = FreePort();
        var backend = new InMemoryStorage();
        var app = GrpcProxyServer.MakeServer(backend, "localhost", port);
        app.StartAsync().GetAwaiter().GetResult();
        try
        {
            // Python drives a full study through its GrpcStorageProxy against the OptunaSharp server.
            PyExec(
                $"proxy = optuna.storages.GrpcStorageProxy(host='localhost', port={port})\n" +
                "proxy.wait_server_ready(timeout=20)\n" +
                $"study = optuna.create_study(study_name='s', storage=proxy, sampler=optuna.samplers.RandomSampler({seed}))\n" +
                "study.set_user_attr('owner', 'eli')\n" +
                "study.set_user_attr('run', 7)\n" +
                PyMixedObjective +
                "study.optimize(_obj, n_trials=12)\n");
            var pyState = PyStudyState("study");

            // C# reads the SAME study straight from the storage the server fronts.
            var csStudy = Optuna.LoadStudy("s", backend);
            var csState = StudySnapshot.Capture(csStudy);
            StudySnapshot.AssertParity(pyState, csState, DatetimeComparison.NullnessOnly);
        }
        finally
        {
            app.StopAsync().GetAwaiter().GetResult();
            app.DisposeAsync().AsTask().GetAwaiter().GetResult();
        }
    }

    [TestMethod]
    [TestCategory("RequiresPython")]
    [DataRow(0)]
    [DataRow(42)]
    public void PythonServer_CSharpClient_SameStorage(int seed)
    {
        EnsureGrpcOrInconclusive();

        int port = FreePort();
        // Start a Python run_grpc_proxy_server (blocking) on a daemon thread; it dies with the process.
        PyExec(
            "import threading\n" +
            "_srv_storage = optuna.storages.InMemoryStorage()\n" +
            "def _run():\n" +
            $"    optuna.storages.run_grpc_proxy_server(_srv_storage, host='localhost', port={port})\n" +
            "_srv_thread = threading.Thread(target=_run, daemon=True)\n" +
            "_srv_thread.start()\n");

        using var proxy = new GrpcStorageProxy("localhost", port);
        WaitReady(proxy);

        // C# drives a full study through its GrpcStorageProxy against the Python server.
        var csStudy = Optuna.CreateStudy(storage: proxy, studyName: "s", sampler: new RandomSampler(seed));
        csStudy.SetUserAttr("owner", "eli");
        csStudy.SetUserAttr("run", 7L);
        csStudy.Optimize(MixedObjective, nTrials: 12);
        var csState = StudySnapshot.Capture(csStudy);

        // Python reads the SAME study straight from the storage its server fronts.
        PyExec("study = optuna.load_study(study_name='s', storage=_srv_storage)\n");
        var pyState = PyStudyState("study");
        StudySnapshot.AssertParity(pyState, csState, DatetimeComparison.NullnessOnly);
    }

    [TestMethod]
    [TestCategory("RequiresPython")]
    public void MultiObjective_And_Nonfinite_CSharpClient_To_PythonServer()
    {
        EnsureGrpcOrInconclusive();

        int port = FreePort();
        PyExec(
            "import threading\n" +
            "_srv_storage2 = optuna.storages.InMemoryStorage()\n" +
            "def _run2():\n" +
            $"    optuna.storages.run_grpc_proxy_server(_srv_storage2, host='localhost', port={port})\n" +
            "_srv_thread2 = threading.Thread(target=_run2, daemon=True)\n" +
            "_srv_thread2.start()\n");

        using var proxy = new GrpcStorageProxy("localhost", port);
        WaitReady(proxy);

        // A single-objective study whose stored value is +inf, with NaN/±inf intermediates.
        var single = Optuna.CreateStudy(storage: proxy, studyName: "single", direction: StudyDirection.Minimize);
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

        // A multi-objective study.
        var mo = Optuna.CreateStudy(
            storage: proxy, studyName: "mo",
            directions: new[] { StudyDirection.Minimize, StudyDirection.Maximize });
        mo.AddTrial(FrozenTrial.CreateTrial(
            state: TrialState.Complete,
            values: new[] { 0.3, 10.0 },
            @params: new OrderedDict<string, object?> { ["a"] = 0.3, ["b"] = 5L },
            distributions: new OrderedDict<string, BaseDistribution>
            {
                ["a"] = new FloatDistribution(0, 1), ["b"] = new IntDistribution(1, 10),
            }));

        var csSingle = StudySnapshot.Capture(single);
        var csMo = StudySnapshot.Capture(mo);

        PyExec("py_single = optuna.load_study(study_name='single', storage=_srv_storage2)\n");
        PyExec("py_mo = optuna.load_study(study_name='mo', storage=_srv_storage2)\n");
        StudySnapshot.AssertParity(PyStudyState("py_single"), csSingle, DatetimeComparison.NullnessOnly);
        StudySnapshot.AssertParity(PyStudyState("py_mo"), csMo, DatetimeComparison.NullnessOnly);
    }
}

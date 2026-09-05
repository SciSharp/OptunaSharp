using System;
using System.Collections.Generic;
using System.Net;
using System.Net.Sockets;
using AwesomeAssertions;
using Microsoft.AspNetCore.Builder;
using Microsoft.Extensions.Hosting;
using NumSharp;
using OptunaSharp;
using OptunaSharp.Collections;
using OptunaSharp.Distributions;
using OptunaSharp.Internal;
using OptunaSharp.Samplers;
using OptunaSharp.Storages;
using OptunaSharp.Storages.Grpc;

namespace OptunaSharp.Tests.Storage;

/// <summary>
///     Unit tests for the gRPC storage proxy (Phase 4.4). The whole client↔server round-trip runs
///     <b>in-process</b> — <see cref="GrpcProxyServer.MakeServer" /> hosts on Kestrel (cleartext HTTP/2)
///     on a loopback port and the <see cref="GrpcStorageProxy" /> connects to it — so these need no
///     external service and no Python, and run in the normal unit-test lane. (The cross-language wire
///     gate vs Python Optuna lives in the interop suite's <c>GrpcStorageProxyInteropTests</c>.)
/// </summary>
[TestClass]
public class GrpcStorageProxyTests
{
    private InMemoryStorage _backend = null!;
    private WebApplication _app = null!;
    private GrpcStorageProxy _proxy = null!;
    private int _port;

    [TestInitialize]
    public void Init()
    {
        _port = FreePort();
        _backend = new InMemoryStorage();
        _app = GrpcProxyServer.MakeServer(_backend, "localhost", _port);
        _app.StartAsync().GetAwaiter().GetResult();
        _proxy = new GrpcStorageProxy("localhost", _port);
        _proxy.WaitServerReady(TimeSpan.FromSeconds(15));
    }

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

    private static int FreePort()
    {
        var listener = new TcpListener(IPAddress.Loopback, 0);
        listener.Start();
        int port = ((IPEndPoint)listener.LocalEndpoint).Port;
        listener.Stop();
        return port;
    }

    [TestMethod]
    public void FullStudy_ThroughProxy_MatchesBackend()
    {
        var study = Optuna.CreateStudy(storage: _proxy, studyName: "s", sampler: new RandomSampler(0));
        study.SetUserAttr("owner", "eli");
        study.Optimize(t =>
        {
            double x = t.SuggestFloat("x", -10, 10);
            long y = t.SuggestInt("y", 1, 100, log: true);
            _ = t.SuggestCategorical("z", new CategoricalChoice[] { "a", "b", (long)3, true, CategoricalChoice.None });
            t.Report(0.5, 0);
            t.Report(double.PositiveInfinity, 1);
            return x * x + y;
        }, nTrials: 10);

        // The study straight from the backing storage the server fronts must match the proxy's view.
        var direct = Optuna.LoadStudy("s", _backend);
        study.BestValue.Should().Be(direct.BestValue);
        study.Trials.Count.Should().Be(10).And.Be(direct.Trials.Count);

        var bt = study.BestTrial;
        var dbt = direct.BestTrial;
        bt.Number.Should().Be(dbt.Number);
        ((double)bt.Params["x"]!).Should().Be((double)dbt.Params["x"]!);
        bt.IntermediateValues[0].Should().Be(0.5);
        bt.IntermediateValues[1].Should().Be(double.PositiveInfinity);
        study.UserAttrs["owner"].Should().Be("eli");
    }

    [TestMethod]
    public void ProtoConvert_TrialRoundTrip_Exact()
    {
        // Microsecond-aligned datetimes so the "%Y-%m-%d %H:%M:%S.%f" round-trip is bit-exact
        // (FrozenTrial equality compares datetimes; .NET ticks are 100 ns, the wire format is µs).
        var dt = new DateTime(2026, 9, 5, 12, 30, 15).AddTicks(1234560); // .123456 s
        var trial = new FrozenTrial(
            number: 4,
            state: TrialState.Complete,
            value: null,
            datetimeStart: dt,
            datetimeComplete: dt.AddSeconds(2),
            @params: new OrderedDict<string, object?> { ["x"] = 0.5, ["n"] = 3L, ["c"] = "b" },
            distributions: new OrderedDict<string, BaseDistribution>
            {
                ["x"] = new FloatDistribution(-1, 1),
                ["n"] = new IntDistribution(1, 10, log: true),
                ["c"] = new CategoricalDistribution(new CategoricalChoice[] { "a", "b", (long)3 }),
            },
            userAttrs: new OrderedDict<string, object?> { ["u"] = 7L },
            systemAttrs: new OrderedDict<string, object?> { ["s"] = "hi" },
            intermediateValues: new OrderedDict<int, double> { [0] = 0.1, [1] = double.PositiveInfinity },
            trialId: 11,
            values: new[] { 1.5 });

        var proto = GrpcProto.ToProtoTrial(trial);
        var back = GrpcProto.FromProtoTrial(proto);

        back.Should().Be(trial); // FrozenTrial.Equals is full-state (number/state/values/datetimes/params/dists/attrs/intermediates/id)
    }

    [TestMethod]
    public void ProtoConvert_EmptyValues_MapToNull()
    {
        var trial = new FrozenTrial(
            number: 0,
            state: TrialState.Running,
            value: null,
            datetimeStart: new DateTime(2026, 9, 5, 1, 2, 3),
            datetimeComplete: null,
            @params: new OrderedDict<string, object?>(),
            distributions: new OrderedDict<string, BaseDistribution>(),
            userAttrs: new OrderedDict<string, object?>(),
            systemAttrs: new OrderedDict<string, object?>(),
            intermediateValues: new OrderedDict<int, double>(),
            trialId: 1,
            values: null);

        var back = GrpcProto.FromProtoTrial(GrpcProto.ToProtoTrial(trial));
        back.Values.Should().BeNull();       // empty repeated<double> → None, like Optuna
        back.DatetimeComplete.Should().BeNull();
        back.State.Should().Be(TrialState.Running);
    }

    [TestMethod]
    public void ErrorTranslation_AcrossWire()
    {
        // NOT_FOUND -> KeyError
        ((Action)(() => _proxy.GetStudyNameFromId(999999))).Should().Throw<KeyError>();

        // ALREADY_EXISTS -> DuplicatedStudyError
        int sid = _proxy.CreateNewStudy(new[] { StudyDirection.Minimize }, "dup");
        ((Action)(() => _proxy.CreateNewStudy(new[] { StudyDirection.Minimize }, "dup")))
            .Should().Throw<DuplicatedStudyError>();

        // FAILED_PRECONDITION -> UpdateFinishedTrialError (write to a finished trial)
        int tid = _proxy.CreateNewTrial(sid);
        _proxy.SetTrialStateValues(tid, TrialState.Complete, new[] { 1.0 });
        ((Action)(() => _proxy.SetTrialParam(tid, "x", 0.0, new FloatDistribution(-1, 1))))
            .Should().Throw<UpdateFinishedTrialError>();

        // INVALID_ARGUMENT -> ValueError (incompatible distribution kind for the same param name)
        int tid2 = _proxy.CreateNewTrial(sid);
        _proxy.SetTrialParam(tid2, "p", 0.0, new FloatDistribution(-1, 1));
        ((Action)(() => _proxy.SetTrialParam(tid2, "p", 0.0, new IntDistribution(0, 10))))
            .Should().Throw<ValueError>();
    }

    [TestMethod]
    public void Cache_ReflectsNewAndAdvancingTrials()
    {
        int sid = _proxy.CreateNewStudy(new[] { StudyDirection.Minimize }, "c");

        int t0 = _proxy.CreateNewTrial(sid); // RUNNING
        var one = _proxy.GetAllTrials(sid);
        one.Should().HaveCount(1);
        one[0].State.Should().Be(TrialState.Running);

        // Completing t0 must be reflected on the next read (it is re-fetched via included_trial_ids).
        _proxy.SetTrialStateValues(t0, TrialState.Complete, new[] { 1.0 });
        var advanced = _proxy.GetAllTrials(sid);
        advanced.Should().HaveCount(1);
        advanced[0].State.Should().Be(TrialState.Complete);

        // A newly created trial appears via trial_id_greater_than.
        _proxy.CreateNewTrial(sid);
        _proxy.GetAllTrials(sid).Should().HaveCount(2);

        // A states filter is applied client-side over the cache.
        _proxy.GetAllTrials(sid, states: new[] { TrialState.Complete }).Should().HaveCount(1);
    }

    [TestMethod]
    public void Constructor_RaisesExperimentalWarning()
    {
        var raised = new List<(string Message, WarningCategory Category)>();
        void Handler(string m, WarningCategory c) => raised.Add((m, c));

        OptunaWarnings.WarningRaised += Handler;
        try
        {
            using var p = new GrpcStorageProxy("localhost", _port);
        }
        finally
        {
            OptunaWarnings.WarningRaised -= Handler;
        }

        raised.Should().ContainSingle(x =>
            x.Category == WarningCategory.ExperimentalWarning &&
            x.Message.Contains("GrpcStorageProxy is experimental (supported from v4.2.0)"));
    }

    [TestMethod]
    public void GetAllStudies_And_Attrs_ThroughProxy()
    {
        int sid = _proxy.CreateNewStudy(new[] { StudyDirection.Maximize }, "study-a");
        _proxy.SetStudyUserAttr(sid, "owner", "eli");
        _proxy.SetStudySystemAttr(sid, "n", 7L);

        var studies = _proxy.GetAllStudies();
        studies.Should().ContainSingle();
        studies[0].StudyName.Should().Be("study-a");
        studies[0].Direction.Should().Be(StudyDirection.Maximize);
        studies[0].UserAttrs["owner"].Should().Be("eli");

        _proxy.GetStudyUserAttrs(sid)["owner"].Should().Be("eli");
        _proxy.GetStudySystemAttrs(sid)["n"].Should().Be(7L);
        _proxy.GetStudyIdFromName("study-a").Should().Be(sid);
        _proxy.GetStudyDirections(sid).Should().Equal(StudyDirection.Maximize);
    }
}

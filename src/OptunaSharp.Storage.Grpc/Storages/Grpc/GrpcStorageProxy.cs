using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Grpc.Core;                    // ChannelCredentials, ConnectivityState, RpcException, StatusCode
using Grpc.Net.Client;              // GrpcChannel, GrpcChannelOptions
using NumSharp;                     // KeyError
using OptunaSharp.Collections;      // OrderedDict<,>
using OptunaSharp.Distributions;    // BaseDistribution, Distribution
using OptunaSharp.Internal;         // PyJson, OptunaWarnings, WarningCategory
using OptunaSharp.Storages;         // BaseStorage
using ProtoApi = global::OptunaSharp.Storages.Grpc.Protobuf;          // the generated protobuf types

namespace OptunaSharp.Storages.Grpc;

/// <summary>
///     gRPC client for <see cref="GrpcProxyServer.RunGrpcProxyServer" /> — the port of
///     <c>optuna/storages/_grpc/client.py: GrpcStorageProxy</c> (experimental since Optuna v4.2.0).
///
///     A <see cref="BaseStorage" /> whose every operation is a gRPC call to a proxy server that fronts a
///     real storage (typically an <c>RDBStorage</c>). Because it speaks Optuna's verbatim
///     <c>api.proto</c>, an OptunaSharp client and a Python <c>run_grpc_proxy_server</c> interoperate on
///     the wire — and vice versa.
///
///     <para>Like Optuna's, this proxy keeps a <b>client-side trial cache</b>
///     (<see cref="GrpcClientCache" />): <see cref="GetAllTrials" /> fetches only the trials that are new
///     or still unfinished on each call (<c>trial_id_greater_than</c> + <c>included_trial_ids</c>), so a
///     growing study is cheap to re-read.</para>
/// </summary>
/// <example>
///     <code>
///     using var storage = new GrpcStorageProxy(host: "localhost", port: 13000);
///     storage.WaitServerReady(TimeSpan.FromSeconds(10));
///     var study = Optuna.CreateStudy(storage: storage);
///     </code>
/// </example>
public sealed class GrpcStorageProxy : BaseStorage, IDisposable, IAsyncDisposable
{
    private readonly GrpcChannel _channel;
    private readonly bool _ownsChannel;
    private readonly ProtoApi.StorageService.StorageServiceClient _stub;
    private readonly GrpcClientCache _cache;

    // Four ways in, from "just give it host+port" (Optuna-identical) to "here is the channel I already
    // built". The richer overloads expose the whole Grpc.Net.Client surface (TLS credentials, a custom
    // HttpHandler, timeouts/retry via GrpcChannelOptions.ServiceConfig, max message sizes, compression,
    // a LoggerFactory, ...); the on-wire protobuf is fixed by api.proto regardless of how you connect,
    // so cross-language byte-compatibility holds however you configure the channel.

    /// <summary>Connect over insecure cleartext HTTP/2 (h2c) to <paramref name="host" />:<paramref name="port" /> — the Optuna-identical entry point.</summary>
    /// <param name="host">The hostname of the gRPC server.</param>
    /// <param name="port">The port of the gRPC server.</param>
    public GrpcStorageProxy(string host = "localhost", int port = 13000)
        : this(CreateInsecureChannel($"http://{host}:{port}", configure: null), ownsChannel: true)
    {
    }

    /// <summary>
    ///     Connect over insecure h2c to <paramref name="host" />:<paramref name="port" />, then hands you
    ///     the derived <see cref="GrpcChannelOptions" /> to adjust before the channel opens — the easy way
    ///     to keep the familiar host/port yet set Grpc.Net.Client knobs (interceptors via a custom
    ///     <c>HttpHandler</c>, <c>MaxSendMessageSize</c>, a retry <c>ServiceConfig</c>, a
    ///     <c>LoggerFactory</c>, ...).
    /// </summary>
    /// <param name="host">The hostname of the gRPC server.</param>
    /// <param name="port">The port of the gRPC server.</param>
    /// <param name="configureChannel">Invoked on the insecure-h2c <see cref="GrpcChannelOptions" /> just
    ///   before the channel is created; mutate it in place.</param>
    /// <exception cref="System.ArgumentNullException"><paramref name="configureChannel" /> is null.</exception>
    public GrpcStorageProxy(string host, int port, Action<GrpcChannelOptions> configureChannel)
        : this(CreateInsecureChannel($"http://{host}:{port}",
                configureChannel ?? throw new ArgumentNullException(nameof(configureChannel))),
               ownsChannel: true)
    {
    }

    /// <summary>
    ///     Connect to an arbitrary <paramref name="address" /> with fully-specified
    ///     <see cref="GrpcChannelOptions" /> — use an <c>https://</c> address with TLS
    ///     <see cref="GrpcChannelOptions.Credentials" /> for a secure channel, or any Grpc.Net.Client
    ///     configuration you like.
    /// </summary>
    /// <param name="address">The server address, e.g. <c>https://host:443</c> or <c>http://host:13000</c>.</param>
    /// <param name="channelOptions">The channel options; passed straight to <see cref="GrpcChannel.ForAddress(string, GrpcChannelOptions)" />.</param>
    /// <exception cref="System.ArgumentNullException"><paramref name="address" /> or <paramref name="channelOptions" /> is null.</exception>
    public GrpcStorageProxy(string address, GrpcChannelOptions channelOptions)
        : this(GrpcChannel.ForAddress(
                address ?? throw new ArgumentNullException(nameof(address)),
                channelOptions ?? throw new ArgumentNullException(nameof(channelOptions))),
               ownsChannel: true)
    {
    }

    /// <summary>
    ///     Bind to a <see cref="GrpcChannel" /> you already built and own — the full escape hatch
    ///     (connection pooling, TLS, interceptors, a channel shared across your app or managed by DI).
    ///     <b>You own the channel's lifetime</b> — the proxy never disposes a channel it did not create.
    /// </summary>
    /// <param name="channel">A configured <see cref="GrpcChannel" />.</param>
    /// <exception cref="System.ArgumentNullException"><paramref name="channel" /> is null.</exception>
    public GrpcStorageProxy(GrpcChannel channel)
        : this(channel ?? throw new ArgumentNullException(nameof(channel)), ownsChannel: false)
    {
    }

    /// <summary>Shared core: every public constructor funnels here.</summary>
    /// <param name="channel">The gRPC channel used for remote storage calls.</param>
    /// <param name="ownsChannel">Whether disposing this proxy should also dispose the channel.</param>
    private GrpcStorageProxy(GrpcChannel channel, bool ownsChannel)
    {
        // Optuna decorates GrpcStorageProxy with @experimental_class("4.2.0"), which raises an
        // ExperimentalWarning on construction. OptunaSharp targets the Optuna 5.x surface and treats
        // it as stable, so that warning is intentionally omitted (deliberate divergence).
        _channel = channel;
        _ownsChannel = ownsChannel;
        _stub = new ProtoApi.StorageService.StorageServiceClient(_channel);
        _cache = new GrpcClientCache(_stub);
    }

    /// <summary>
    ///     Optuna's <c>create_insecure_channel</c>: an HTTP/2 cleartext (h2c) channel with unlimited
    ///     receive. <paramref name="configure" />, when non-null, tweaks the derived options first.
    /// </summary>
    /// <param name="address">The HTTP or HTTPS address of the gRPC server.</param>
    /// <param name="configure">The callback that customizes native connection options before opening the connection.</param>
    /// <returns>A channel that the proxy will own and dispose.</returns>
    private static GrpcChannel CreateInsecureChannel(string address, Action<GrpcChannelOptions>? configure)
    {
        var options = new GrpcChannelOptions
        {
            Credentials = ChannelCredentials.Insecure,
            MaxReceiveMessageSize = null,   // Optuna: grpc.max_receive_message_length = -1 (unlimited)
        };
        configure?.Invoke(options);
        return GrpcChannel.ForAddress(address, options);
    }

    /// <summary>
    ///     Wait until the gRPC server is ready (Optuna's <c>wait_server_ready</c>).
    /// </summary>
    /// <param name="timeout">The maximum time to wait. If <c>null</c>, wait indefinitely.</param>
    /// <exception cref="TimeoutException">If the server is not ready within <paramref name="timeout" />.</exception>
    public void WaitServerReady(TimeSpan? timeout = null)
    {
        try
        {
            bool completed = _channel.ConnectAsync().Wait(timeout ?? Timeout.InfiniteTimeSpan);
            if (!completed || _channel.State != ConnectivityState.Ready)
                throw new TimeoutException("GRPC connection timeout");
        }
        catch (AggregateException e)
        {
            throw new TimeoutException("GRPC connection timeout", e.InnerException ?? e);
        }
    }

    /// <summary>Close the gRPC channel (Optuna's <c>close</c>) — unless it was injected, in which case the caller owns it.</summary>
    /// <exception cref="global::Grpc.Core.RpcException">The remote call fails with a status that is not translated into an Optuna storage exception.</exception>
    public void Close()
    {
        if (_ownsChannel)
            _channel.Dispose();
    }

    /// <summary>Releases resources owned by this storage instance; caller-owned connections or channels remain the caller's responsibility.</summary>
    public void Dispose() => Close();

    /// <summary>
    ///     The asynchronous form of <see cref="Dispose" />: closes the gRPC channel this proxy created; an injected
    ///     channel is left open. Lets the proxy be used with <c>await using</c> and disposed by DI scopes that are
    ///     disposed asynchronously.
    /// </summary>
    /// <returns>An already-completed task, or a faulted one when closing the channel failed.</returns>
    /// <remarks>
    ///     <see cref="GrpcChannel" /> has no asynchronous close (it is <see cref="IDisposable" /> only), so the channel
    ///     closes synchronously here, exactly as in <see cref="Dispose" />; this exists so every OptunaSharp storage that
    ///     owns a resource supports both <c>using</c> and <c>await using</c>. A failure is returned through the task
    ///     rather than thrown, as an awaiting caller expects. A second disposal, by either method, does nothing: a
    ///     disposed channel ignores it.
    /// </remarks>
    public ValueTask DisposeAsync()
    {
        try
        {
            Close();
            return default;
        }
        catch (Exception e)
        {
            return ValueTask.FromException(e);
        }
    }

    // ------------------------------------------------------------------ study manipulation

    /// <inheritdoc />
    /// <exception cref="DuplicatedStudyError">Another study already uses the requested name.</exception>
    /// <exception cref="global::Grpc.Core.RpcException">The remote call fails with a status that is not translated into an Optuna storage exception.</exception>
    public override int CreateNewStudy(IReadOnlyList<StudyDirection> directions, string? studyName = null)
    {
        studyName ??= DefaultStudyNamePrefix + Guid.NewGuid();
        var request = new ProtoApi.CreateNewStudyRequest { StudyName = studyName };
        request.Directions.AddRange(directions.Select(GrpcProto.ToProtoDirection));
        try
        {
            return (int)_stub.CreateNewStudy(request).StudyId;
        }
        catch (RpcException e) when (e.StatusCode == StatusCode.AlreadyExists)
        {
            throw new DuplicatedStudyError(e.Status.Detail);
        }
    }

    /// <inheritdoc />
    /// <exception cref="global::Grpc.Core.RpcException">The remote call fails with a status that is not translated into an Optuna storage exception.</exception>
    public override void DeleteStudy(int studyId)
    {
        try
        {
            _stub.DeleteStudy(new ProtoApi.DeleteStudyRequest { StudyId = studyId });
        }
        catch (RpcException e) when (e.StatusCode == StatusCode.NotFound)
        {
            throw new KeyError(e.Status.Detail);
        }
        _cache.DeleteStudyCache(studyId);
    }

    /// <inheritdoc />
    /// <exception cref="global::Grpc.Core.RpcException">The remote call fails with a status that is not translated into an Optuna storage exception.</exception>
    public override void SetStudyUserAttr(int studyId, string key, object? value)
    {
        var request = new ProtoApi.SetStudyUserAttributeRequest { StudyId = studyId, Key = key, Value = PyJson.Dumps(value) };
        try
        {
            _stub.SetStudyUserAttribute(request);
        }
        catch (RpcException e) when (e.StatusCode == StatusCode.NotFound)
        {
            throw new KeyError(e.Status.Detail);
        }
    }

    /// <inheritdoc />
    /// <exception cref="global::Grpc.Core.RpcException">The remote call fails with a status that is not translated into an Optuna storage exception.</exception>
    public override void SetStudySystemAttr(int studyId, string key, object? value)
    {
        var request = new ProtoApi.SetStudySystemAttributeRequest { StudyId = studyId, Key = key, Value = PyJson.Dumps(value) };
        try
        {
            _stub.SetStudySystemAttribute(request);
        }
        catch (RpcException e) when (e.StatusCode == StatusCode.NotFound)
        {
            throw new KeyError(e.Status.Detail);
        }
    }

    // ------------------------------------------------------------------ study access

    /// <inheritdoc />
    /// <exception cref="NumSharp.KeyError">No study has the requested name.</exception>
    /// <exception cref="global::Grpc.Core.RpcException">The remote call fails with a status that is not translated into an Optuna storage exception.</exception>
    public override int GetStudyIdFromName(string studyName)
    {
        try
        {
            return (int)_stub.GetStudyIdFromName(new ProtoApi.GetStudyIdFromNameRequest { StudyName = studyName }).StudyId;
        }
        catch (RpcException e) when (e.StatusCode == StatusCode.NotFound)
        {
            throw new KeyError(e.Status.Detail);
        }
    }

    /// <inheritdoc />
    /// <exception cref="NumSharp.KeyError">The study identifier does not exist.</exception>
    /// <exception cref="global::Grpc.Core.RpcException">The remote call fails with a status that is not translated into an Optuna storage exception.</exception>
    public override string GetStudyNameFromId(int studyId)
    {
        try
        {
            return _stub.GetStudyNameFromId(new ProtoApi.GetStudyNameFromIdRequest { StudyId = studyId }).StudyName;
        }
        catch (RpcException e) when (e.StatusCode == StatusCode.NotFound)
        {
            throw new KeyError(e.Status.Detail);
        }
    }

    /// <inheritdoc />
    /// <exception cref="global::Grpc.Core.RpcException">The remote call fails with a status that is not translated into an Optuna storage exception.</exception>
    public override List<StudyDirection> GetStudyDirections(int studyId)
    {
        try
        {
            var response = _stub.GetStudyDirections(new ProtoApi.GetStudyDirectionsRequest { StudyId = studyId });
            return response.Directions.Select(GrpcProto.FromProtoDirection).ToList();
        }
        catch (RpcException e) when (e.StatusCode == StatusCode.NotFound)
        {
            throw new KeyError(e.Status.Detail);
        }
    }

    /// <inheritdoc />
    /// <exception cref="global::Grpc.Core.RpcException">The remote call fails with a status that is not translated into an Optuna storage exception.</exception>
    public override OrderedDict<string, object?> GetStudyUserAttrs(int studyId)
    {
        try
        {
            var response = _stub.GetStudyUserAttributes(new ProtoApi.GetStudyUserAttributesRequest { StudyId = studyId });
            var result = new OrderedDict<string, object?>(response.UserAttributes.Count);
            foreach (var (key, value) in response.UserAttributes)
                result[key] = PyJson.Loads(value);
            return result;
        }
        catch (RpcException e) when (e.StatusCode == StatusCode.NotFound)
        {
            throw new KeyError(e.Status.Detail);
        }
    }

    /// <inheritdoc />
    /// <exception cref="global::Grpc.Core.RpcException">The remote call fails with a status that is not translated into an Optuna storage exception.</exception>
    public override OrderedDict<string, object?> GetStudySystemAttrs(int studyId)
    {
        try
        {
            var response = _stub.GetStudySystemAttributes(new ProtoApi.GetStudySystemAttributesRequest { StudyId = studyId });
            var result = new OrderedDict<string, object?>(response.SystemAttributes.Count);
            foreach (var (key, value) in response.SystemAttributes)
                result[key] = PyJson.Loads(value);
            return result;
        }
        catch (RpcException e) when (e.StatusCode == StatusCode.NotFound)
        {
            throw new KeyError(e.Status.Detail);
        }
    }

    /// <inheritdoc />
    /// <exception cref="global::Grpc.Core.RpcException">The remote call fails with a status that is not translated into an Optuna storage exception.</exception>
    public override List<FrozenStudy> GetAllStudies()
    {
        var response = _stub.GetAllStudies(new ProtoApi.GetAllStudiesRequest());
        var result = new List<FrozenStudy>(response.Studies.Count);
        foreach (var study in response.Studies)
        {
            var userAttrs = new OrderedDict<string, object?>(study.UserAttributes.Count);
            foreach (var (key, value) in study.UserAttributes)
                userAttrs[key] = PyJson.Loads(value);
            var systemAttrs = new OrderedDict<string, object?>(study.SystemAttributes.Count);
            foreach (var (key, value) in study.SystemAttributes)
                systemAttrs[key] = PyJson.Loads(value);
            result.Add(new FrozenStudy(
                studyName: study.StudyName,
                direction: null,
                userAttrs: userAttrs,
                systemAttrs: systemAttrs,
                studyId: (int)study.StudyId,
                directions: study.Directions.Select(GrpcProto.FromProtoDirection).ToList()));
        }
        return result;
    }

    // ------------------------------------------------------------------ trial manipulation

    /// <inheritdoc />
    /// <exception cref="global::Grpc.Core.RpcException">The remote call fails with a status that is not translated into an Optuna storage exception.</exception>
    public override int CreateNewTrial(int studyId, FrozenTrial? templateTrial = null)
    {
        var request = templateTrial is null
            ? new ProtoApi.CreateNewTrialRequest { StudyId = studyId, TemplateTrialIsNone = true }
            : new ProtoApi.CreateNewTrialRequest
            {
                StudyId = studyId,
                TemplateTrial = GrpcProto.ToProtoTrial(templateTrial),
                TemplateTrialIsNone = false,
            };
        try
        {
            return (int)_stub.CreateNewTrial(request).TrialId;
        }
        catch (RpcException e) when (e.StatusCode == StatusCode.NotFound)
        {
            throw new KeyError(e.Status.Detail);
        }
    }

    /// <inheritdoc />
    /// <exception cref="global::Grpc.Core.RpcException">The remote call fails with a status that is not translated into an Optuna storage exception.</exception>
    public override void SetTrialParam(int trialId, string paramName, double paramValueInternal, BaseDistribution distribution)
    {
        var request = new ProtoApi.SetTrialParameterRequest
        {
            TrialId = trialId,
            ParamName = paramName,
            ParamValueInternal = paramValueInternal,
            Distribution = Distribution.ToJson(distribution),
        };
        try
        {
            _stub.SetTrialParameter(request);
        }
        catch (RpcException e)
        {
            if (e.StatusCode == StatusCode.NotFound)
                throw new KeyError(e.Status.Detail);
            if (e.StatusCode == StatusCode.FailedPrecondition)
                throw new UpdateFinishedTrialError(e.Status.Detail);
            if (e.StatusCode == StatusCode.InvalidArgument)
                throw new ValueError(e.Status.Detail);
            throw;
        }
    }

    /// <inheritdoc />
    /// <exception cref="global::Grpc.Core.RpcException">The remote call fails with a status that is not translated into an Optuna storage exception.</exception>
    public override bool SetTrialStateValues(int trialId, TrialState state, IReadOnlyList<double>? values = null)
    {
        var request = new ProtoApi.SetTrialStateValuesRequest
        {
            TrialId = trialId,
            State = GrpcProto.ToProtoTrialState(state),
        };
        if (values is not null)
            request.Values.AddRange(values);
        try
        {
            return _stub.SetTrialStateValues(request).TrialUpdated;
        }
        catch (RpcException e)
        {
            if (e.StatusCode == StatusCode.NotFound)
                throw new KeyError(e.Status.Detail);
            if (e.StatusCode == StatusCode.FailedPrecondition)
                throw new UpdateFinishedTrialError(e.Status.Detail);
            throw;
        }
    }

    /// <inheritdoc />
    /// <exception cref="global::Grpc.Core.RpcException">The remote call fails with a status that is not translated into an Optuna storage exception.</exception>
    public override void SetTrialIntermediateValue(int trialId, long step, double intermediateValue)
    {
        // The wire field is int64 in Optuna's api.proto, so the full 64-bit step travels as is.
        var request = new ProtoApi.SetTrialIntermediateValueRequest
        {
            TrialId = trialId,
            Step = step,
            IntermediateValue = intermediateValue,
        };
        try
        {
            _stub.SetTrialIntermediateValue(request);
        }
        catch (RpcException e)
        {
            if (e.StatusCode == StatusCode.NotFound)
                throw new KeyError(e.Status.Detail);
            if (e.StatusCode == StatusCode.FailedPrecondition)
                throw new UpdateFinishedTrialError(e.Status.Detail);
            throw;
        }
    }

    /// <inheritdoc />
    /// <exception cref="global::Grpc.Core.RpcException">The remote call fails with a status that is not translated into an Optuna storage exception.</exception>
    public override void SetTrialUserAttr(int trialId, string key, object? value)
    {
        var request = new ProtoApi.SetTrialUserAttributeRequest { TrialId = trialId, Key = key, Value = PyJson.Dumps(value) };
        try
        {
            _stub.SetTrialUserAttribute(request);
        }
        catch (RpcException e)
        {
            if (e.StatusCode == StatusCode.NotFound)
                throw new KeyError(e.Status.Detail);
            if (e.StatusCode == StatusCode.FailedPrecondition)
                throw new UpdateFinishedTrialError(e.Status.Detail);
            throw;
        }
    }

    /// <inheritdoc />
    /// <exception cref="global::Grpc.Core.RpcException">The remote call fails with a status that is not translated into an Optuna storage exception.</exception>
    public override void SetTrialSystemAttr(int trialId, string key, object? value)
    {
        var request = new ProtoApi.SetTrialSystemAttributeRequest { TrialId = trialId, Key = key, Value = PyJson.Dumps(value) };
        try
        {
            _stub.SetTrialSystemAttribute(request);
        }
        catch (RpcException e)
        {
            if (e.StatusCode == StatusCode.NotFound)
                throw new KeyError(e.Status.Detail);
            if (e.StatusCode == StatusCode.FailedPrecondition)
                throw new UpdateFinishedTrialError(e.Status.Detail);
            throw;
        }
    }

    /// <inheritdoc />
    /// <exception cref="NumSharp.KeyError">The requested trial number does not exist in the study.</exception>
    /// <exception cref="global::Grpc.Core.RpcException">The remote call fails with a status that is not translated into an Optuna storage exception.</exception>
    public override int GetTrialIdFromStudyIdTrialNumber(int studyId, int trialNumber)
    {
        var request = new ProtoApi.GetTrialIdFromStudyIdTrialNumberRequest { StudyId = studyId, TrialNumber = trialNumber };
        try
        {
            return (int)_stub.GetTrialIdFromStudyIdTrialNumber(request).TrialId;
        }
        catch (RpcException e) when (e.StatusCode == StatusCode.NotFound)
        {
            throw new KeyError(e.Status.Detail);
        }
    }

    // ------------------------------------------------------------------ trial access

    /// <inheritdoc />
    /// <exception cref="global::Grpc.Core.RpcException">The remote call fails with a status that is not translated into an Optuna storage exception.</exception>
    public override FrozenTrial GetTrial(int trialId)
    {
        try
        {
            var response = _stub.GetTrial(new ProtoApi.GetTrialRequest { TrialId = trialId });
            return GrpcProto.FromProtoTrial(response.Trial);
        }
        catch (RpcException e) when (e.StatusCode == StatusCode.NotFound)
        {
            throw new KeyError(e.Status.Detail);
        }
    }

    /// <inheritdoc />
    /// <exception cref="global::Grpc.Core.RpcException">The remote call fails with a status that is not translated into an Optuna storage exception.</exception>
    public override List<FrozenTrial> GetAllTrials(
        int studyId, bool deepcopy = true, IReadOnlyCollection<TrialState>? states = null)
    {
        var trials = _cache.GetAllTrials(studyId, states);
        return deepcopy ? trials.Select(t => t.DeepCopy()).ToList() : trials;
    }
}

/// <summary>
///     The client-side trial cache — the port of <c>client.py: GrpcClientCache</c>. Incrementally reads
///     a study's trials from the server: on every refresh it asks only for trials newer than the last
///     finished one (<c>trial_id_greater_than</c>) plus the still-unfinished ones (whose state may have
///     advanced), keeping already-finished trials cached.
/// </summary>
internal sealed class GrpcClientCache
{
    private readonly Dictionary<int, GrpcClientCacheEntry> _studies = new();
    private readonly ProtoApi.StorageService.StorageServiceClient _grpcClient;
    private readonly object _lock = new();

    /// <summary>Creates an incremental trial cache bound to the generated storage client.</summary>
    /// <param name="grpcClient">The generated gRPC client used to fetch trial-cache updates.</param>
    internal GrpcClientCache(ProtoApi.StorageService.StorageServiceClient grpcClient) => _grpcClient = grpcClient;

    /// <summary>Drops locally cached trial state for a deleted study.</summary>
    /// <param name="studyId">The storage-wide study identifier, not its display name.</param>
    internal void DeleteStudyCache(int studyId)
    {
        lock (_lock)
            _studies.Remove(studyId);
    }

    /// <param name="studyId">The storage-wide study identifier, not its display name.</param>
    /// <param name="states">The allowed trial states, or null to include every state.</param>
    /// <returns>Cached trial snapshots filtered by the requested states after refreshing remote updates.</returns>
    /// <summary>Refreshes the remote trial cache and returns snapshots matching the requested states.</summary>
    internal List<FrozenTrial> GetAllTrials(int studyId, IReadOnlyCollection<TrialState>? states)
    {
        lock (_lock)
        {
            ReadTrialsFromRemoteStorage(studyId);
            var study = _studies[studyId];
            IEnumerable<FrozenTrial> trials = study.Trials.Values;
            if (states is not null)
                trials = trials.Where(t => states.Contains(t.State));
            return trials.OrderBy(t => t.Number).ToList();
        }
    }

    /// <summary>Fetches new and unfinished trials and merges them into the study cache.</summary>
    /// <param name="studyId">The storage-wide study identifier, not its display name.</param>
    private void ReadTrialsFromRemoteStorage(int studyId)
    {
        if (!_studies.ContainsKey(studyId))
            _studies[studyId] = new GrpcClientCacheEntry();
        var study = _studies[studyId];

        var request = new ProtoApi.GetTrialsRequest
        {
            StudyId = studyId,
            TrialIdGreaterThan = study.LastFinishedTrialId,
        };
        request.IncludedTrialIds.AddRange(study.UnfinishedTrialIds.Select(id => (long)id));

        ProtoApi.GetTrialsReply response;
        try
        {
            response = _grpcClient.GetTrials(request);
        }
        catch (RpcException e) when (e.StatusCode == StatusCode.NotFound)
        {
            _studies.Remove(studyId);
            throw new KeyError(e.Status.Detail);
        }
        if (response.Trials.Count == 0)
            return;

        foreach (var trialProto in response.Trials)
            AddTrialToCache(studyId, GrpcProto.FromProtoTrial(trialProto));
    }

    /// <summary>Updates a trial snapshot and the finished/unfinished incremental-cache cursors.</summary>
    /// <param name="studyId">The storage-wide study identifier, not its display name.</param>
    /// <param name="trial">The trial snapshot or database record being read or validated.</param>
    private void AddTrialToCache(int studyId, FrozenTrial trial)
    {
        var study = _studies[studyId];
        study.Trials[trial.Number] = trial;

        if (!trial.State.IsFinished())
        {
            study.UnfinishedTrialIds.Add(trial.TrialId);
            return;
        }

        study.LastFinishedTrialId = Math.Max(study.LastFinishedTrialId, trial.TrialId);
        study.UnfinishedTrialIds.Remove(trial.TrialId);
    }
}

/// <summary>Per-study cache entry (Optuna's <c>GrpcClientCacheEntry</c>). Trials are keyed by number.</summary>
internal sealed class GrpcClientCacheEntry
{
    /// <summary>Cached trial snapshots keyed by storage-wide trial identifier.</summary>
    internal Dictionary<int, FrozenTrial> Trials { get; } = new();
    /// <summary>Identifiers that must be fetched again because their trial state can still change.</summary>
    internal HashSet<int> UnfinishedTrialIds { get; } = new();
    /// <summary>The greatest finished trial identifier already seen by the incremental cache.</summary>
    internal int LastFinishedTrialId { get; set; } = -1;
}

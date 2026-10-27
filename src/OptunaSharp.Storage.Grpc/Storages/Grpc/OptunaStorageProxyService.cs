using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Grpc.Core;                    // RpcException, Status, StatusCode, ServerCallContext
using NumSharp;                     // KeyError, ValueError
using OptunaSharp.Distributions;    // Distribution
using OptunaSharp.Internal;         // PyJson
using OptunaSharp.Storages;         // BaseStorage
using ProtoApi = global::OptunaSharp.Storages.Grpc.Protobuf;          // the generated protobuf types

namespace OptunaSharp.Storages.Grpc;

/// <summary>
///     The server-side gRPC service — the port of
///     <c>optuna/storages/_grpc/servicer.py: OptunaStorageProxyService</c>. It wraps a real
///     <see cref="BaseStorage" /> (any backend: in-memory, journal, or the Tier X SQLite
///     <c>RDBStorage</c>) and answers every RPC by delegating to it, translating the storage's typed
///     exceptions into the same gRPC status codes Optuna uses so a Python or C# client re-raises the
///     matching typed exception:
///     <list type="bullet">
///       <item><description><see cref="DuplicatedStudyError" /> → <c>ALREADY_EXISTS</c></description></item>
///       <item><description><see cref="KeyError" /> → <c>NOT_FOUND</c></description></item>
///       <item><description><see cref="UpdateFinishedTrialError" /> → <c>FAILED_PRECONDITION</c></description></item>
///       <item><description><see cref="ValueError" /> → <c>INVALID_ARGUMENT</c></description></item>
///     </list>
///
///     <para>Registered on the server by <see cref="GrpcProxyServer.MakeServer" />; not meant to be
///     constructed directly.</para>
/// </summary>
public sealed class OptunaStorageProxyService : ProtoApi.StorageService.StorageServiceBase
{
    private readonly BaseStorage _backend;

    // Optuna's servicer also builds a `threading.Lock()` in __init__ but never takes it — the wrapped
    // BaseStorage is itself thread-safe (CachedStorage / RDBStorage hold their own locks), so no lock is
    // needed here either. Faithful to servicer.py, which relies on the backend's own thread-safety.

    /// <summary>Binds gRPC request handlers to the supplied Optuna storage backend.</summary>
    /// <param name="storage">The storage backend serving requests; its lifetime is controlled by its owner.</param>
    public OptunaStorageProxyService(BaseStorage storage) => _backend = storage;

    /// <summary>The <c>context.abort(code, details=str(e))</c> analog: an aborting gRPC fault.</summary>
    /// <param name="code">The gRPC status used to represent the storage failure.</param>
    /// <param name="e">The provider exception to classify or translate.</param>
    /// <returns>A gRPC exception carrying the chosen status and original message.</returns>
    private static RpcException Fault(StatusCode code, Exception e) => new(new Status(code, e.Message));

    // ------------------------------------------------------------------ study manipulation

    /// <summary>Creates a study through the backend and returns its storage identifier.</summary>
    /// <param name="request">The study name and ordered objective directions.</param>
    /// <param name="context">The server call context supplied by ASP.NET Core; backend operations execute synchronously.</param>
    /// <returns>A completed task containing the reply. The newly assigned study identifier.</returns>
    /// <exception cref="global::Grpc.Core.RpcException">The study name already exists; the reply status is AlreadyExists.</exception>
    public override Task<ProtoApi.CreateNewStudyReply> CreateNewStudy(ProtoApi.CreateNewStudyRequest request, ServerCallContext context)
    {
        var directions = request.Directions.Select(GrpcProto.FromProtoDirection).ToList();
        try
        {
            int studyId = _backend.CreateNewStudy(directions, request.StudyName);
            return Task.FromResult(new ProtoApi.CreateNewStudyReply { StudyId = studyId });
        }
        catch (DuplicatedStudyError e)
        {
            throw Fault(StatusCode.AlreadyExists, e);
        }
    }

    /// <summary>Deletes a study through the backend.</summary>
    /// <param name="request">The identifier of the study to delete.</param>
    /// <param name="context">The server call context supplied by ASP.NET Core; backend operations execute synchronously.</param>
    /// <returns>A completed task containing the reply. An empty success reply.</returns>
    /// <exception cref="global::Grpc.Core.RpcException">The study is absent; the reply status is NotFound.</exception>
    public override Task<ProtoApi.DeleteStudyReply> DeleteStudy(ProtoApi.DeleteStudyRequest request, ServerCallContext context)
    {
        try
        {
            _backend.DeleteStudy((int)request.StudyId);
            return Task.FromResult(new ProtoApi.DeleteStudyReply());
        }
        catch (KeyError e)
        {
            throw Fault(StatusCode.NotFound, e);
        }
    }

    /// <summary>Decodes and writes a study user attribute.</summary>
    /// <param name="request">The study identifier, attribute name, and JSON value.</param>
    /// <param name="context">The server call context supplied by ASP.NET Core; backend operations execute synchronously.</param>
    /// <returns>A completed task containing the reply. An empty success reply.</returns>
    /// <exception cref="global::Grpc.Core.RpcException">The study is absent; the reply status is NotFound.</exception>
    public override Task<ProtoApi.SetStudyUserAttributeReply> SetStudyUserAttribute(
        ProtoApi.SetStudyUserAttributeRequest request, ServerCallContext context)
    {
        try
        {
            _backend.SetStudyUserAttr((int)request.StudyId, request.Key, PyJson.Loads(request.Value));
            return Task.FromResult(new ProtoApi.SetStudyUserAttributeReply());
        }
        catch (KeyError e)
        {
            throw Fault(StatusCode.NotFound, e);
        }
    }

    /// <summary>Decodes and writes a study system attribute.</summary>
    /// <param name="request">The study identifier, attribute name, and JSON value.</param>
    /// <param name="context">The server call context supplied by ASP.NET Core; backend operations execute synchronously.</param>
    /// <returns>A completed task containing the reply. An empty success reply.</returns>
    /// <exception cref="global::Grpc.Core.RpcException">The study is absent; the reply status is NotFound.</exception>
    public override Task<ProtoApi.SetStudySystemAttributeReply> SetStudySystemAttribute(
        ProtoApi.SetStudySystemAttributeRequest request, ServerCallContext context)
    {
        try
        {
            _backend.SetStudySystemAttr((int)request.StudyId, request.Key, PyJson.Loads(request.Value));
            return Task.FromResult(new ProtoApi.SetStudySystemAttributeReply());
        }
        catch (KeyError e)
        {
            throw Fault(StatusCode.NotFound, e);
        }
    }

    // ------------------------------------------------------------------ study access

    /// <summary>Resolves a study name through the backend.</summary>
    /// <param name="request">The study name to resolve.</param>
    /// <param name="context">The server call context supplied by ASP.NET Core; backend operations execute synchronously.</param>
    /// <returns>A completed task containing the reply. The matching storage-wide study identifier.</returns>
    /// <exception cref="global::Grpc.Core.RpcException">The study name is absent; the reply status is NotFound.</exception>
    public override Task<ProtoApi.GetStudyIdFromNameReply> GetStudyIdFromName(
        ProtoApi.GetStudyIdFromNameRequest request, ServerCallContext context)
    {
        try
        {
            int studyId = _backend.GetStudyIdFromName(request.StudyName);
            return Task.FromResult(new ProtoApi.GetStudyIdFromNameReply { StudyId = studyId });
        }
        catch (KeyError e)
        {
            throw Fault(StatusCode.NotFound, e);
        }
    }

    /// <summary>Reads the display name of a stored study.</summary>
    /// <param name="request">The storage-wide study identifier.</param>
    /// <param name="context">The server call context supplied by ASP.NET Core; backend operations execute synchronously.</param>
    /// <returns>A completed task containing the reply. The stored study name.</returns>
    /// <exception cref="global::Grpc.Core.RpcException">The study is absent; the reply status is NotFound.</exception>
    public override Task<ProtoApi.GetStudyNameFromIdReply> GetStudyNameFromId(
        ProtoApi.GetStudyNameFromIdRequest request, ServerCallContext context)
    {
        try
        {
            string name = _backend.GetStudyNameFromId((int)request.StudyId);
            return Task.FromResult(new ProtoApi.GetStudyNameFromIdReply { StudyName = name });
        }
        catch (KeyError e)
        {
            throw Fault(StatusCode.NotFound, e);
        }
    }

    /// <summary>Reads the ordered objective directions of a study.</summary>
    /// <param name="request">The storage-wide study identifier.</param>
    /// <param name="context">The server call context supplied by ASP.NET Core; backend operations execute synchronously.</param>
    /// <returns>A completed task containing the reply. The directions encoded as protobuf enum values.</returns>
    /// <exception cref="global::Grpc.Core.RpcException">The study is absent; the reply status is NotFound.</exception>
    public override Task<ProtoApi.GetStudyDirectionsReply> GetStudyDirections(
        ProtoApi.GetStudyDirectionsRequest request, ServerCallContext context)
    {
        try
        {
            var directions = _backend.GetStudyDirections((int)request.StudyId);
            var reply = new ProtoApi.GetStudyDirectionsReply();
            reply.Directions.AddRange(directions.Select(GrpcProto.ToProtoDirection));
            return Task.FromResult(reply);
        }
        catch (KeyError e)
        {
            throw Fault(StatusCode.NotFound, e);
        }
    }

    /// <summary>Reads and JSON-encodes a study's user attributes.</summary>
    /// <param name="request">The storage-wide study identifier.</param>
    /// <param name="context">The server call context supplied by ASP.NET Core; backend operations execute synchronously.</param>
    /// <returns>A completed task containing the reply. The JSON-valued user-attribute mapping.</returns>
    /// <exception cref="global::Grpc.Core.RpcException">The study is absent; the reply status is NotFound.</exception>
    public override Task<ProtoApi.GetStudyUserAttributesReply> GetStudyUserAttributes(
        ProtoApi.GetStudyUserAttributesRequest request, ServerCallContext context)
    {
        try
        {
            var attributes = _backend.GetStudyUserAttrs((int)request.StudyId);
            var reply = new ProtoApi.GetStudyUserAttributesReply();
            foreach (var (key, value) in attributes)
                reply.UserAttributes[key] = PyJson.Dumps(value);
            return Task.FromResult(reply);
        }
        catch (KeyError e)
        {
            throw Fault(StatusCode.NotFound, e);
        }
    }

    /// <summary>Reads and JSON-encodes a study's system attributes.</summary>
    /// <param name="request">The storage-wide study identifier.</param>
    /// <param name="context">The server call context supplied by ASP.NET Core; backend operations execute synchronously.</param>
    /// <returns>A completed task containing the reply. The JSON-valued system-attribute mapping.</returns>
    /// <exception cref="global::Grpc.Core.RpcException">The study is absent; the reply status is NotFound.</exception>
    public override Task<ProtoApi.GetStudySystemAttributesReply> GetStudySystemAttributes(
        ProtoApi.GetStudySystemAttributesRequest request, ServerCallContext context)
    {
        try
        {
            var attributes = _backend.GetStudySystemAttrs((int)request.StudyId);
            var reply = new ProtoApi.GetStudySystemAttributesReply();
            foreach (var (key, value) in attributes)
                reply.SystemAttributes[key] = PyJson.Dumps(value);
            return Task.FromResult(reply);
        }
        catch (KeyError e)
        {
            throw Fault(StatusCode.NotFound, e);
        }
    }

    /// <summary>Reads study summaries and converts their metadata to the Optuna protobuf representation.</summary>
    /// <param name="request">The empty request selecting all studies.</param>
    /// <param name="context">The server call context supplied by ASP.NET Core; backend operations execute synchronously.</param>
    /// <returns>A completed task containing the reply. The collection of study summaries.</returns>
    public override Task<ProtoApi.GetAllStudiesReply> GetAllStudies(ProtoApi.GetAllStudiesRequest request, ServerCallContext context)
    {
        var studies = _backend.GetAllStudies();
        var reply = new ProtoApi.GetAllStudiesReply();
        foreach (var study in studies)
        {
            var protoStudy = new ProtoApi.Study
            {
                StudyId = study.StudyId,
                StudyName = study.StudyName,
            };
            protoStudy.Directions.AddRange(study.Directions.Select(GrpcProto.ToProtoDirection));
            foreach (var (key, value) in study.UserAttrs)
                protoStudy.UserAttributes[key] = PyJson.Dumps(value);
            foreach (var (key, value) in study.SystemAttrs)
                protoStudy.SystemAttributes[key] = PyJson.Dumps(value);
            reply.Studies.Add(protoStudy);
        }
        return Task.FromResult(reply);
    }

    // ------------------------------------------------------------------ trial manipulation

    /// <summary>Creates a trial from the optional template carried by the request.</summary>
    /// <param name="request">The owning study identifier and optional template trial.</param>
    /// <param name="context">The server call context supplied by ASP.NET Core; backend operations execute synchronously.</param>
    /// <returns>A completed task containing the reply. The newly assigned storage-wide trial identifier.</returns>
    /// <exception cref="global::Grpc.Core.RpcException">The study is absent; the reply status is NotFound.</exception>
    public override Task<ProtoApi.CreateNewTrialReply> CreateNewTrial(ProtoApi.CreateNewTrialRequest request, ServerCallContext context)
    {
        FrozenTrial? templateTrial = request.TemplateTrialIsNone ? null : GrpcProto.FromProtoTrial(request.TemplateTrial);
        try
        {
            int trialId = _backend.CreateNewTrial((int)request.StudyId, templateTrial);
            return Task.FromResult(new ProtoApi.CreateNewTrialReply { TrialId = trialId });
        }
        catch (KeyError e)
        {
            throw Fault(StatusCode.NotFound, e);
        }
    }

    /// <summary>Decodes a distribution and stores a trial parameter in internal numeric form.</summary>
    /// <param name="request">The trial identifier, parameter name, numeric value, and distribution JSON.</param>
    /// <param name="context">The server call context supplied by ASP.NET Core; backend operations execute synchronously.</param>
    /// <returns>A completed task containing the reply. An empty success reply.</returns>
    /// <exception cref="global::Grpc.Core.RpcException">The trial is missing, finished, or has an incompatible distribution; statuses are NotFound, FailedPrecondition, or InvalidArgument respectively.</exception>
    public override Task<ProtoApi.SetTrialParameterReply> SetTrialParameter(
        ProtoApi.SetTrialParameterRequest request, ServerCallContext context)
    {
        var distribution = Distribution.FromJson(request.Distribution);
        try
        {
            _backend.SetTrialParam((int)request.TrialId, request.ParamName, request.ParamValueInternal, distribution);
            return Task.FromResult(new ProtoApi.SetTrialParameterReply());
        }
        catch (KeyError e)
        {
            throw Fault(StatusCode.NotFound, e);
        }
        catch (UpdateFinishedTrialError e)
        {
            throw Fault(StatusCode.FailedPrecondition, e);
        }
        catch (ValueError e)
        {
            throw Fault(StatusCode.InvalidArgument, e);
        }
    }

    /// <summary>Resolves a study-local trial number to a storage-wide identifier.</summary>
    /// <param name="request">The study identifier and trial number within that study.</param>
    /// <param name="context">The server call context supplied by ASP.NET Core; backend operations execute synchronously.</param>
    /// <returns>A completed task containing the reply. The matching storage-wide trial identifier.</returns>
    /// <exception cref="global::Grpc.Core.RpcException">The requested trial is absent; the reply status is NotFound.</exception>
    public override Task<ProtoApi.GetTrialIdFromStudyIdTrialNumberReply> GetTrialIdFromStudyIdTrialNumber(
        ProtoApi.GetTrialIdFromStudyIdTrialNumberRequest request, ServerCallContext context)
    {
        try
        {
            int trialId = _backend.GetTrialIdFromStudyIdTrialNumber((int)request.StudyId, (int)request.TrialNumber);
            return Task.FromResult(new ProtoApi.GetTrialIdFromStudyIdTrialNumberReply { TrialId = trialId });
        }
        catch (KeyError e)
        {
            throw Fault(StatusCode.NotFound, e);
        }
    }

    /// <summary>Applies a trial lifecycle transition and its optional objective values.</summary>
    /// <param name="request">The trial identifier, requested lifecycle state, and objective values.</param>
    /// <param name="context">The server call context supplied by ASP.NET Core; backend operations execute synchronously.</param>
    /// <returns>A completed task containing the reply. Whether the backend accepted the requested transition.</returns>
    /// <exception cref="global::Grpc.Core.RpcException">The trial is missing or already finished; statuses are NotFound or FailedPrecondition respectively.</exception>
    public override Task<ProtoApi.SetTrialStateValuesReply> SetTrialStateValues(
        ProtoApi.SetTrialStateValuesRequest request, ServerCallContext context)
    {
        // proto3 can't distinguish an empty `repeated double` from None; Optuna maps empty → None.
        IReadOnlyList<double>? values = request.Values.Count > 0 ? request.Values.ToList() : null;
        try
        {
            bool trialUpdated = _backend.SetTrialStateValues(
                (int)request.TrialId, GrpcProto.FromProtoTrialState(request.State), values);
            return Task.FromResult(new ProtoApi.SetTrialStateValuesReply { TrialUpdated = trialUpdated });
        }
        catch (KeyError e)
        {
            throw Fault(StatusCode.NotFound, e);
        }
        catch (UpdateFinishedTrialError e)
        {
            throw Fault(StatusCode.FailedPrecondition, e);
        }
    }

    /// <summary>Writes a trial's intermediate report at the requested step.</summary>
    /// <param name="request">The trial identifier, report step, and intermediate value.</param>
    /// <param name="context">The server call context supplied by ASP.NET Core; backend operations execute synchronously.</param>
    /// <returns>A completed task containing the reply. An empty success reply.</returns>
    /// <exception cref="global::Grpc.Core.RpcException">The trial is missing or already finished; statuses are NotFound or FailedPrecondition respectively.</exception>
    public override Task<ProtoApi.SetTrialIntermediateValueReply> SetTrialIntermediateValue(
        ProtoApi.SetTrialIntermediateValueRequest request, ServerCallContext context)
    {
        try
        {
            // The int64 step reaches the backend unchanged (an (int) cast here used to wrap steps past Int32).
            _backend.SetTrialIntermediateValue((int)request.TrialId, request.Step, request.IntermediateValue);
            return Task.FromResult(new ProtoApi.SetTrialIntermediateValueReply());
        }
        catch (KeyError e)
        {
            throw Fault(StatusCode.NotFound, e);
        }
        catch (UpdateFinishedTrialError e)
        {
            throw Fault(StatusCode.FailedPrecondition, e);
        }
    }

    /// <summary>Decodes and writes a trial user attribute.</summary>
    /// <param name="request">The trial identifier, attribute name, and JSON value.</param>
    /// <param name="context">The server call context supplied by ASP.NET Core; backend operations execute synchronously.</param>
    /// <returns>A completed task containing the reply. An empty success reply.</returns>
    /// <exception cref="global::Grpc.Core.RpcException">The trial is missing or already finished; statuses are NotFound or FailedPrecondition respectively.</exception>
    public override Task<ProtoApi.SetTrialUserAttributeReply> SetTrialUserAttribute(
        ProtoApi.SetTrialUserAttributeRequest request, ServerCallContext context)
    {
        object? value = PyJson.Loads(request.Value);
        try
        {
            _backend.SetTrialUserAttr((int)request.TrialId, request.Key, value);
            return Task.FromResult(new ProtoApi.SetTrialUserAttributeReply());
        }
        catch (KeyError e)
        {
            throw Fault(StatusCode.NotFound, e);
        }
        catch (UpdateFinishedTrialError e)
        {
            throw Fault(StatusCode.FailedPrecondition, e);
        }
    }

    /// <summary>Decodes and writes a trial system attribute.</summary>
    /// <param name="request">The trial identifier, attribute name, and JSON value.</param>
    /// <param name="context">The server call context supplied by ASP.NET Core; backend operations execute synchronously.</param>
    /// <returns>A completed task containing the reply. An empty success reply.</returns>
    /// <exception cref="global::Grpc.Core.RpcException">The trial is missing or already finished; statuses are NotFound or FailedPrecondition respectively.</exception>
    public override Task<ProtoApi.SetTrialSystemAttributeReply> SetTrialSystemAttribute(
        ProtoApi.SetTrialSystemAttributeRequest request, ServerCallContext context)
    {
        object? value = PyJson.Loads(request.Value);
        try
        {
            _backend.SetTrialSystemAttr((int)request.TrialId, request.Key, value);
            return Task.FromResult(new ProtoApi.SetTrialSystemAttributeReply());
        }
        catch (KeyError e)
        {
            throw Fault(StatusCode.NotFound, e);
        }
        catch (UpdateFinishedTrialError e)
        {
            throw Fault(StatusCode.FailedPrecondition, e);
        }
    }

    // ------------------------------------------------------------------ trial access

    /// <summary>Reads a frozen trial and converts it to its wire representation.</summary>
    /// <param name="request">The storage-wide trial identifier.</param>
    /// <param name="context">The server call context supplied by ASP.NET Core; backend operations execute synchronously.</param>
    /// <returns>A completed task containing the reply. The complete protobuf trial snapshot.</returns>
    /// <exception cref="global::Grpc.Core.RpcException">The trial is absent; the reply status is NotFound.</exception>
    public override Task<ProtoApi.GetTrialReply> GetTrial(ProtoApi.GetTrialRequest request, ServerCallContext context)
    {
        try
        {
            var trial = _backend.GetTrial((int)request.TrialId);
            return Task.FromResult(new ProtoApi.GetTrialReply { Trial = GrpcProto.ToProtoTrial(trial) });
        }
        catch (KeyError e)
        {
            throw Fault(StatusCode.NotFound, e);
        }
    }

    /// <summary>Reads trial snapshots newer than the cursor or explicitly requested for refresh.</summary>
    /// <param name="request">The study identifier, exclusive trial-id cursor, and identifiers to refresh regardless of the cursor.</param>
    /// <param name="context">The server call context supplied by ASP.NET Core; backend operations execute synchronously.</param>
    /// <returns>A completed task containing the reply. The matching protobuf trial snapshots.</returns>
    /// <exception cref="global::Grpc.Core.RpcException">The study is absent; the reply status is NotFound.</exception>
    public override Task<ProtoApi.GetTrialsReply> GetTrials(ProtoApi.GetTrialsRequest request, ServerCallContext context)
    {
        var includedTrialIds = new HashSet<long>(request.IncludedTrialIds);
        long trialIdGreaterThan = request.TrialIdGreaterThan;

        List<FrozenTrial> trials;
        try
        {
            trials = _backend.GetAllTrials((int)request.StudyId, deepcopy: false);
        }
        catch (KeyError e)
        {
            throw Fault(StatusCode.NotFound, e);
        }

        var reply = new ProtoApi.GetTrialsReply();
        foreach (var t in trials)
        {
            if (t.TrialId > trialIdGreaterThan || includedTrialIds.Contains(t.TrialId))
                reply.Trials.Add(GrpcProto.ToProtoTrial(t));
        }
        return Task.FromResult(reply);
    }
}
